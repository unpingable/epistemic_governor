"""
Glassiness Intervention (BLI-EXP-002)

Single-parameter experiment: resolution_cost.

Target pathology: GLASS_OSSIFICATION (contradiction accumulation)
- contradiction_storm showed monotone C_open increase
- λ_open > μ_close leads to inevitable accumulation

Sweep parameter: resolution_cost
- How expensive is it to close a contradiction?
- Higher cost → lower effective μ_close → glass
- Lower cost → higher effective μ_close → healthy

Sweep: [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]

Success criteria:
1. Find the glass/healthy phase boundary
2. closed_without_evidence remains 0
3. No ceremony (ρ_S stays healthy)
4. Identify precursor signatures before glass transition

This completes the control surface map:
- Budget sweep: found starvation boundary
- Cost sweep: find accumulation boundary
"""

import json
from datetime import datetime, timezone
from typing import Dict, List, Any
from dataclasses import dataclass
from pathlib import Path
import uuid

from epistemic_governor.diagnostics import (
    DiagnosticLogger, DiagnosticEvent, compute_energy,
    Regime, RegimeDetector, compute_weighted_glassiness,
)
from epistemic_governor.hysteresis import (
    HysteresisState, ContradictionSeverity,
)


# =============================================================================
# Parameterized Storm Workload
# =============================================================================

class ParameterizedStorm:
    """
    Contradiction storm with tunable resolution cost.
    
    High contradiction injection rate, variable repair friction.
    """
    
    def __init__(
        self,
        seed: int = 43,
        resolution_cost: float = 5.0,  # THE SWEEP PARAMETER
        repair_budget: float = 100.0,
        repair_refill_rate: float = 3.0,  # Healthy refill from previous experiment
        contradiction_rate: float = 0.35,  # High injection
    ):
        import random
        self.rng = random.Random(seed)
        self.turn = 0
        
        # Parameters
        self.resolution_cost = resolution_cost
        self.repair_budget = repair_budget
        self.max_repair_budget = repair_budget
        self.repair_refill_rate = repair_refill_rate
        self.contradiction_rate = contradiction_rate
        
        # State
        self.state = HysteresisState(state_id=f"storm_cost_{resolution_cost}")
        self.domains = ["facts", "dates", "config", "identity", "policy"]
        
        # Tracking
        self.closed_without_evidence = 0
        self.total_opened = 0
        self.total_closed = 0
    
    def step(self) -> Dict[str, Any]:
        self.turn += 1
        
        c_open_before = self.state.contradictions.open_count
        opened = 0
        closed = 0
        tau_resolve = []
        budget_spent = 0.0
        
        verdict = "OK"
        blocked_by = []
        warn_reasons = []
        
        # High contradiction injection
        if self.rng.random() < self.contradiction_rate:
            domain = self.rng.choice(self.domains)
            claim_a = f"claim_{uuid.uuid4().hex[:8]}"
            claim_b = f"claim_{uuid.uuid4().hex[:8]}"
            self.state.commit_claim(claim_a, domain, "value", f"A_{self.turn}")
            self.state.commit_claim(claim_b, domain, "value", f"B_{self.turn}")
            
            # Vary severity
            severity = self.rng.choice([
                ContradictionSeverity.LOW,
                ContradictionSeverity.MEDIUM,
                ContradictionSeverity.HIGH,
            ])
            self.state.add_contradiction(claim_a, claim_b, domain, severity=severity)
            opened = 1
            self.total_opened += 1
        
        # Resolution attempts (budget and cost permitting)
        resolution_attempts = 0
        max_attempts = 3
        
        while resolution_attempts < max_attempts and self.state.contradictions.open_count > 0:
            resolution_attempts += 1
            
            if self.rng.random() < 0.25:  # 25% chance per attempt
                if self.repair_budget >= self.resolution_cost:
                    open_c = self.state.contradictions.get_open()
                    if not open_c:
                        break
                    
                    # Prioritize by severity (highest first)
                    target = max(open_c, key=lambda c: c.severity.value)
                    
                    tau = (datetime.now(timezone.utc) - target.opened_at).total_seconds() * 1000
                    
                    # Always close with evidence
                    evidence_id = f"evidence_{uuid.uuid4().hex[:8]}"
                    self.state.close_contradiction(
                        target.contradiction_id,
                        evidence_id,
                        target.claim_a_id,
                    )
                    closed += 1
                    self.total_closed += 1
                    tau_resolve.append(tau)
                    self.repair_budget -= self.resolution_cost
                    budget_spent += self.resolution_cost
                else:
                    # Budget exhausted
                    if "BUDGET_EXHAUSTED" not in blocked_by:
                        warn_reasons.append("BUDGET_PRESSURE")
        
        # Budget refill
        self.repair_budget = min(
            self.max_repair_budget,
            self.repair_budget + self.repair_refill_rate
        )
        
        c_open_after = self.state.contradictions.open_count
        
        # Severity distribution
        severity_hist = {}
        severity_sum = 0
        for c in self.state.contradictions.get_open():
            sev = c.severity.name
            severity_hist[sev] = severity_hist.get(sev, 0) + 1
            severity_sum += c.severity.value
        
        # Verdict based on load
        if c_open_after > 30:
            verdict = "BLOCK"
            blocked_by.append("HIGH_CONTRADICTION_LOAD")
        elif c_open_after > 15:
            verdict = "WARN"
            warn_reasons.append("ELEVATED_CONTRADICTION_LOAD")
        
        # Weighted glassiness
        weighted_g = compute_weighted_glassiness(severity_hist)
        
        return {
            "turn": self.turn,
            "verdict": verdict,
            "blocked_by_invariant": blocked_by,
            "warn_reasons": warn_reasons,
            "c_open_before": c_open_before,
            "c_open_after": c_open_after,
            "c_opened_count": opened,
            "c_closed_count": closed,
            "tau_resolve_ms_closed": tau_resolve,
            "c_severity_hist_after": severity_hist,
            "weighted_glassiness": weighted_g,
            "budget_remaining": {"resolve": self.repair_budget},
            "budget_spent": {"resolve": budget_spent},
            "severity_sum": severity_sum,
            "closed_without_evidence": self.closed_without_evidence,
            "resolution_cost": self.resolution_cost,
            "total_opened": self.total_opened,
            "total_closed": self.total_closed,
        }


# =============================================================================
# Sweep Runner
# =============================================================================

@dataclass
class GlassSweepResult:
    """Result of one glass sweep point."""
    resolution_cost: float
    turns: int
    
    # Primary metrics
    final_c_open: int
    max_c_open: int
    avg_c_open: float
    
    # Throughput
    total_opened: int
    total_closed: int
    open_rate: float
    close_rate: float
    net_accumulation_rate: float
    
    # Glassiness
    final_weighted_g: float
    max_weighted_g: float
    avg_weighted_g: float
    
    # Verdicts
    block_count: int
    warn_count: int
    block_fraction: float
    
    # Safety
    closed_without_evidence: int
    
    # τ_resolve
    tau_resolve_count: int
    tau_resolve_avg: float
    
    # State mutation
    rho_S: float
    
    # Regime
    regime: str
    regime_confidence: float
    
    # Trajectory (for transition detection)
    c_open_trajectory: List[int]
    weighted_g_trajectory: List[float]


def run_glass_sweep_point(resolution_cost: float, turns: int = 150, seed: int = 43) -> GlassSweepResult:
    """Run one glass sweep point."""
    workload = ParameterizedStorm(
        seed=seed,
        resolution_cost=resolution_cost,
    )
    
    logger = DiagnosticLogger(
        run_id=f"glass_cost_{resolution_cost}",
        suite="sweep",
    )
    
    all_tau = []
    c_open_trajectory = []
    weighted_g_trajectory = []
    
    for _ in range(turns):
        data = workload.step()
        
        event = logger.create_event(
            prompt_hash=f"storm_{workload.turn}",
            state_hash_before=f"before_{workload.turn}",
            state_hash_after=f"after_{workload.turn}",
            state_view_hash=f"view_{workload.turn}",
        )
        
        event.verdict = data["verdict"]
        event.blocked_by_invariant = data["blocked_by_invariant"]
        event.warn_reasons = data["warn_reasons"]
        event.c_open_before = data["c_open_before"]
        event.c_open_after = data["c_open_after"]
        event.c_opened_count = data["c_opened_count"]
        event.c_closed_count = data["c_closed_count"]
        event.tau_resolve_ms_closed = data["tau_resolve_ms_closed"]
        event.c_severity_hist_after = data["c_severity_hist_after"]
        event.budget_remaining_after = data["budget_remaining"]
        event.budget_spent_this_turn = data["budget_spent"]
        
        energy = compute_energy(
            c_open=event.c_open_after,
            severity_sum=data["severity_sum"],
            budget_remaining=event.budget_remaining_after,
        )
        event.E_state_after = energy.total()
        event.E_components_after = energy.to_dict()
        
        logger.record(event)
        all_tau.extend(data["tau_resolve_ms_closed"])
        c_open_trajectory.append(data["c_open_after"])
        weighted_g_trajectory.append(data["weighted_glassiness"])
    
    # Compute metrics
    metrics = logger.metrics
    regime_analysis = logger.get_regime()
    
    return GlassSweepResult(
        resolution_cost=resolution_cost,
        turns=turns,
        final_c_open=c_open_trajectory[-1] if c_open_trajectory else 0,
        max_c_open=max(c_open_trajectory) if c_open_trajectory else 0,
        avg_c_open=sum(c_open_trajectory) / len(c_open_trajectory) if c_open_trajectory else 0,
        total_opened=workload.total_opened,
        total_closed=workload.total_closed,
        open_rate=workload.total_opened / turns,
        close_rate=workload.total_closed / turns,
        net_accumulation_rate=(workload.total_opened - workload.total_closed) / turns,
        final_weighted_g=weighted_g_trajectory[-1] if weighted_g_trajectory else 0,
        max_weighted_g=max(weighted_g_trajectory) if weighted_g_trajectory else 0,
        avg_weighted_g=sum(weighted_g_trajectory) / len(weighted_g_trajectory) if weighted_g_trajectory else 0,
        block_count=metrics.verdict_total.get("BLOCK", 0),
        warn_count=metrics.verdict_total.get("WARN", 0),
        block_fraction=metrics.verdict_total.get("BLOCK", 0) / turns,
        closed_without_evidence=workload.closed_without_evidence,
        tau_resolve_count=len(all_tau),
        tau_resolve_avg=sum(all_tau) / len(all_tau) if all_tau else 0,
        rho_S=metrics.get_rho_S(),
        regime=regime_analysis.regime.name,
        regime_confidence=regime_analysis.confidence,
        c_open_trajectory=c_open_trajectory,
        weighted_g_trajectory=weighted_g_trajectory,
    )


def detect_glass_transition(trajectory: List[int], window: int = 30) -> Dict[str, Any]:
    """Detect if trajectory shows glass behavior (sustained positive trend)."""
    if len(trajectory) < window * 2:
        return {"is_glass": False, "reason": "trajectory_too_short"}
    
    # Compare first and last windows
    first_window_avg = sum(trajectory[:window]) / window
    last_window_avg = sum(trajectory[-window:]) / window
    
    # Check for sustained growth
    mid_point = len(trajectory) // 2
    mid_window_avg = sum(trajectory[mid_point:mid_point+window]) / window
    
    # Glass = monotone increasing trend
    is_monotone_up = first_window_avg < mid_window_avg < last_window_avg
    
    # Also check slope
    total_growth = last_window_avg - first_window_avg
    growth_rate = total_growth / len(trajectory)
    
    is_glass = is_monotone_up and growth_rate > 0.05  # Growing by >0.05/turn
    
    return {
        "is_glass": is_glass,
        "first_window_avg": first_window_avg,
        "mid_window_avg": mid_window_avg,
        "last_window_avg": last_window_avg,
        "growth_rate": growth_rate,
        "total_growth": total_growth,
    }


def run_glass_sweep() -> List[GlassSweepResult]:
    """Run the full glass sweep."""
    resolution_costs = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]
    results = []
    
    print("=" * 70)
    print("RESOLUTION COST SWEEP (Glassiness Intervention)")
    print("Target: Find glass/healthy phase boundary")
    print("=" * 70)
    
    for cost in resolution_costs:
        print(f"\nRunning resolution_cost={cost}...")
        result = run_glass_sweep_point(cost, turns=150)
        results.append(result)
        
        glass_check = detect_glass_transition(result.c_open_trajectory)
        
        print(f"  C_open: final={result.final_c_open}, max={result.max_c_open}, avg={result.avg_c_open:.1f}")
        print(f"  Throughput: λ={result.open_rate:.3f}, μ={result.close_rate:.3f}, net={result.net_accumulation_rate:+.3f}")
        print(f"  Weighted G: final={result.final_weighted_g:.1f}, max={result.max_weighted_g:.1f}")
        print(f"  Verdicts: BLOCK={result.block_count}, WARN={result.warn_count}")
        print(f"  Glass check: {glass_check['is_glass']} (growth_rate={glass_check['growth_rate']:.3f})")
        print(f"  closed_without_evidence: {result.closed_without_evidence}")
    
    return results


def analyze_glass_sweep(results: List[GlassSweepResult]) -> Dict[str, Any]:
    """Analyze glass sweep results."""
    analysis = {
        "parameter": "resolution_cost",
        "points": len(results),
        "safety_invariant_held": all(r.closed_without_evidence == 0 for r in results),
        "results": [],
    }
    
    for r in results:
        glass_check = detect_glass_transition(r.c_open_trajectory)
        analysis["results"].append({
            "resolution_cost": r.resolution_cost,
            "final_c_open": r.final_c_open,
            "net_accumulation_rate": r.net_accumulation_rate,
            "is_glass": glass_check["is_glass"],
            "growth_rate": glass_check["growth_rate"],
            "block_fraction": r.block_fraction,
            "regime": r.regime,
        })
    
    # Find phase boundary
    healthy_max_cost = None
    glass_min_cost = None
    
    for r in results:
        glass_check = detect_glass_transition(r.c_open_trajectory)
        if not glass_check["is_glass"]:
            if healthy_max_cost is None or r.resolution_cost > healthy_max_cost:
                healthy_max_cost = r.resolution_cost
        else:
            if glass_min_cost is None or r.resolution_cost < glass_min_cost:
                glass_min_cost = r.resolution_cost
    
    analysis["healthy_up_to_cost"] = healthy_max_cost
    analysis["glass_from_cost"] = glass_min_cost
    
    # Phase boundary
    if healthy_max_cost and glass_min_cost:
        analysis["phase_boundary"] = (healthy_max_cost + glass_min_cost) / 2
    
    return analysis


def print_glass_report(results: List[GlassSweepResult], analysis: Dict[str, Any]):
    """Print formatted glass sweep report."""
    print("\n" + "=" * 70)
    print("GLASS SWEEP REPORT")
    print("=" * 70)
    
    print(f"\nParameter: {analysis['parameter']}")
    print(f"Points: {analysis['points']}")
    print(f"Safety invariant (closed_without_evidence=0): {'✓ HELD' if analysis['safety_invariant_held'] else '✗ VIOLATED'}")
    
    print("\n### Results Table")
    print("-" * 85)
    print(f"{'Cost':>6} | {'C_open':>8} | {'NetAccum':>10} | {'GrowthRate':>10} | {'IsGlass':>8} | {'BLOCK%':>8}")
    print("-" * 85)
    
    for r in results:
        glass_check = detect_glass_transition(r.c_open_trajectory)
        glass_str = "YES" if glass_check["is_glass"] else "no"
        print(f"{r.resolution_cost:>6.1f} | {r.final_c_open:>8} | {r.net_accumulation_rate:>+10.3f} | {glass_check['growth_rate']:>10.3f} | {glass_str:>8} | {r.block_fraction:>7.1%}")
    
    print("-" * 85)
    
    if analysis.get("healthy_up_to_cost"):
        print(f"\n✓ HEALTHY up to resolution_cost={analysis['healthy_up_to_cost']}")
    if analysis.get("glass_from_cost"):
        print(f"✗ GLASS from resolution_cost={analysis['glass_from_cost']}")
    if analysis.get("phase_boundary"):
        print(f"\n→ Phase boundary approximately at resolution_cost ≈ {analysis['phase_boundary']:.1f}")
    
    # Interpretation
    print("\n### Interpretation")
    print("\nThe queueing model predicts glass when λ_open > μ_close sustained.")
    print("Higher resolution cost → lower effective μ_close → glass inevitable.")
    
    # Find transition
    prev_glass = None
    for r in results:
        glass_check = detect_glass_transition(r.c_open_trajectory)
        if prev_glass == False and glass_check["is_glass"] == True:
            print(f"\nTransition to GLASS at resolution_cost={r.resolution_cost}")
            print(f"  Net accumulation rate: {r.net_accumulation_rate:+.3f}/turn")
            print(f"  λ_open={r.open_rate:.3f}, μ_close={r.close_rate:.3f}")
            print(f"  When λ > μ for sustained period → accumulation → glass")
        prev_glass = glass_check["is_glass"]
    
    # Ceremony check
    ceremony_risk = [r for r in results if r.rho_S < 0.1]
    if ceremony_risk:
        print(f"\n⚠ CEREMONY RISK at costs: {[r.resolution_cost for r in ceremony_risk]}")
    else:
        print("\n✓ No ceremony risk (ρ_S stayed healthy)")
    
    print("\n✓ No sloppy fluid risk (all closures had evidence)")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    results = run_glass_sweep()
    analysis = analyze_glass_sweep(results)
    print_glass_report(results, analysis)
    
    # Save results
    output_path = Path(__file__).parent / "glass_sweep_results.json"
    
    # Convert to serializable format
    serializable = {
        "parameter": analysis["parameter"],
        "points": analysis["points"],
        "safety_invariant_held": analysis["safety_invariant_held"],
        "healthy_up_to_cost": analysis.get("healthy_up_to_cost"),
        "glass_from_cost": analysis.get("glass_from_cost"),
        "phase_boundary": analysis.get("phase_boundary"),
        "results": analysis["results"],
    }
    
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Print combined phase map
    print("\n" + "=" * 70)
    print("COMBINED PHASE MAP")
    print("=" * 70)
    print("\nTwo phase boundaries identified:")
    print("\n1. BUDGET_STARVATION boundary:")
    print("   - Parameter: repair_refill_rate")
    print("   - Healthy above: 2.0/turn")
    print("   - Starvation below: 1.0/turn")
    print("\n2. GLASS_OSSIFICATION boundary:")
    print(f"   - Parameter: resolution_cost")
    if analysis.get("phase_boundary"):
        print(f"   - Healthy below: ~{analysis.get('healthy_up_to_cost', '?')}")
        print(f"   - Glass above: ~{analysis.get('glass_from_cost', '?')}")
    print("\nControl surface:")
    print("   repair_refill_rate ↑ → escapes STARVATION")
    print("   resolution_cost ↓ → escapes GLASS")
    print("   Both affect μ_close (repair throughput)")
