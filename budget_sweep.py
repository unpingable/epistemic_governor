"""
Budget Sweep Experiment (BLI-EXP-001)

Single-parameter experiment: repair budget refill rate.

Target pathology: BUDGET_STARVATION
- resolution_burst showed 100% BLOCK budget-related
- This is crisp, measurable, attributable

Sweep parameter: repair_refill_rate
- How fast does repair budget recover per turn?
- Baseline: 0.3 per turn (current)
- Sweep: [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0]

Success criteria (observational, not aesthetic):
1. Budget-related BLOCK fraction drops
2. contradiction_closed_without_evidence remains 0
3. No shift to "sloppy fluid" signals
4. ρ_S doesn't collapse to ~0 (no ceremony)

This is MPC discipline: one control input, measure state response.
"""

import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import uuid

from epistemic_governor.diagnostics import (
    DiagnosticLogger, DiagnosticEvent, compute_energy,
    Regime, RegimeDetector,
)
from epistemic_governor.hysteresis import (
    HysteresisState, ContradictionSeverity,
)


# =============================================================================
# Parameterized Resolution Burst Workload
# =============================================================================

class ParameterizedResolutionBurst:
    """
    Resolution burst workload with tunable repair budget refill.
    
    This is the BUDGET_STARVATION pathology case.
    """
    
    def __init__(
        self,
        seed: int = 44,
        initial_repair_budget: float = 100.0,
        repair_refill_rate: float = 0.3,  # THE SWEEP PARAMETER
        repair_cost: float = 5.0,
        initial_contradictions: int = 25,
    ):
        import random
        self.rng = random.Random(seed)
        self.turn = 0
        
        # Parameters
        self.repair_refill_rate = repair_refill_rate
        self.repair_cost = repair_cost
        self.repair_budget = initial_repair_budget
        self.max_repair_budget = initial_repair_budget
        
        # State
        self.state = HysteresisState(state_id=f"sweep_{repair_refill_rate}")
        self.domains = ["facts", "dates", "config", "identity", "policy"]
        
        # Tracking
        self.closed_without_evidence = 0  # MUST REMAIN 0
        
        # Initialize with contradictions
        for i in range(initial_contradictions):
            domain = self.rng.choice(self.domains)
            claim_a = f"claim_{uuid.uuid4().hex[:8]}"
            claim_b = f"claim_{uuid.uuid4().hex[:8]}"
            self.state.commit_claim(claim_a, domain, "value", f"A_{i}")
            self.state.commit_claim(claim_b, domain, "value", f"B_{i}")
            self.state.add_contradiction(claim_a, claim_b, domain)
    
    def step(self) -> Dict[str, Any]:
        self.turn += 1
        
        c_open_before = self.state.contradictions.open_count
        opened = 0
        closed = 0
        tau_resolve = []
        budget_spent = 0.0
        
        verdict = "OK"
        blocked_by = []
        
        # Occasional new contradiction (5%)
        if self.rng.random() < 0.05:
            domain = self.rng.choice(self.domains)
            claim_a = f"claim_{uuid.uuid4().hex[:8]}"
            claim_b = f"claim_{uuid.uuid4().hex[:8]}"
            self.state.commit_claim(claim_a, domain, "value", f"A_{self.turn}")
            self.state.commit_claim(claim_b, domain, "value", f"B_{self.turn}")
            self.state.add_contradiction(claim_a, claim_b, domain)
            opened = 1
        
        # High resolution rate (if budget allows)
        resolution_attempts = 0
        while c_open_before - closed > 0 and resolution_attempts < 3:
            resolution_attempts += 1
            
            if self.rng.random() < 0.4:  # 40% chance per attempt
                if self.repair_budget >= self.repair_cost:
                    open_c = self.state.contradictions.get_open()
                    if not open_c:
                        break
                    
                    oldest = min(open_c, key=lambda c: c.opened_at)
                    tau = (datetime.now(timezone.utc) - oldest.opened_at).total_seconds() * 1000
                    
                    # ALWAYS close with evidence (invariant)
                    evidence_id = f"evidence_{uuid.uuid4().hex[:8]}"
                    self.state.close_contradiction(
                        oldest.contradiction_id,
                        evidence_id,
                        oldest.claim_a_id,
                    )
                    closed += 1
                    tau_resolve.append(tau)
                    self.repair_budget -= self.repair_cost
                    budget_spent += self.repair_cost
                else:
                    # Can't afford - BLOCK
                    verdict = "BLOCK"
                    blocked_by.append("BUDGET_EXHAUSTED")
                    break
        
        # Budget refill (THE SWEPT PARAMETER)
        self.repair_budget = min(
            self.max_repair_budget,
            self.repair_budget + self.repair_refill_rate
        )
        
        c_open_after = self.state.contradictions.open_count
        severity_sum = self.state.contradictions.total_severity
        
        # Severity distribution
        severity_hist = {}
        for c in self.state.contradictions.get_open():
            sev = c.severity.name
            severity_hist[sev] = severity_hist.get(sev, 0) + 1
        
        return {
            "turn": self.turn,
            "verdict": verdict,
            "blocked_by_invariant": blocked_by,
            "c_open_before": c_open_before,
            "c_open_after": c_open_after,
            "c_opened_count": opened,
            "c_closed_count": closed,
            "tau_resolve_ms_closed": tau_resolve,
            "c_severity_hist_after": severity_hist,
            "budget_remaining": {"resolve": self.repair_budget},
            "budget_spent": {"resolve": budget_spent},
            "budget_exhaustion": {"resolve": self.repair_budget < self.repair_cost},
            "severity_sum": severity_sum,
            "closed_without_evidence": self.closed_without_evidence,
            "repair_refill_rate": self.repair_refill_rate,
        }


# =============================================================================
# Sweep Runner
# =============================================================================

@dataclass
class SweepResult:
    """Result of one sweep point."""
    refill_rate: float
    turns: int
    
    # Primary metrics
    budget_block_count: int
    budget_block_fraction: float
    total_blocks: int
    
    # Safety invariants
    closed_without_evidence: int  # MUST BE 0
    
    # Regime indicators
    final_c_open: int
    avg_c_open: float
    max_c_open: int
    total_opened: int
    total_closed: int
    
    # Throughput
    open_rate: float
    close_rate: float
    net_accumulation: float
    
    # τ_resolve
    tau_resolve_count: int
    tau_resolve_avg: float
    tau_resolve_max: float
    
    # State mutation
    state_changes: int
    rho_S: float
    
    # Final regime
    regime: str
    regime_confidence: float


def run_sweep_point(refill_rate: float, turns: int = 100, seed: int = 44) -> SweepResult:
    """Run one sweep point."""
    workload = ParameterizedResolutionBurst(
        seed=seed,
        repair_refill_rate=refill_rate,
    )
    
    logger = DiagnosticLogger(
        run_id=f"sweep_refill_{refill_rate}",
        suite="sweep",
    )
    
    all_tau = []
    
    for _ in range(turns):
        data = workload.step()
        
        event = logger.create_event(
            prompt_hash=f"sweep_{workload.turn}",
            state_hash_before=f"before_{workload.turn}",
            state_hash_after=f"after_{workload.turn}",
            state_view_hash=f"view_{workload.turn}",
        )
        
        event.verdict = data["verdict"]
        event.blocked_by_invariant = data["blocked_by_invariant"]
        event.c_open_before = data["c_open_before"]
        event.c_open_after = data["c_open_after"]
        event.c_opened_count = data["c_opened_count"]
        event.c_closed_count = data["c_closed_count"]
        event.tau_resolve_ms_closed = data["tau_resolve_ms_closed"]
        event.c_severity_hist_after = data["c_severity_hist_after"]
        event.budget_remaining_after = data["budget_remaining"]
        event.budget_spent_this_turn = data["budget_spent"]
        event.budget_exhaustion = data["budget_exhaustion"]
        
        energy = compute_energy(
            c_open=event.c_open_after,
            severity_sum=data["severity_sum"],
            budget_remaining=event.budget_remaining_after,
        )
        event.E_state_after = energy.total()
        event.E_components_after = energy.to_dict()
        
        logger.record(event)
        all_tau.extend(data["tau_resolve_ms_closed"])
    
    # Compute metrics
    metrics = logger.metrics
    regime_analysis = logger.get_regime()
    
    total_blocks = metrics.verdict_total.get("BLOCK", 0)
    budget_blocks = sum(
        1 for e in logger.events
        if "BUDGET_EXHAUSTED" in e.blocked_by_invariant
    )
    
    c_open_values = [e.c_open_after for e in logger.events]
    
    return SweepResult(
        refill_rate=refill_rate,
        turns=turns,
        budget_block_count=budget_blocks,
        budget_block_fraction=budget_blocks / turns if turns > 0 else 0,
        total_blocks=total_blocks,
        closed_without_evidence=workload.closed_without_evidence,
        final_c_open=c_open_values[-1] if c_open_values else 0,
        avg_c_open=sum(c_open_values) / len(c_open_values) if c_open_values else 0,
        max_c_open=max(c_open_values) if c_open_values else 0,
        total_opened=metrics.contradictions_opened_total,
        total_closed=metrics.contradictions_closed_total,
        open_rate=metrics.contradictions_opened_total / turns,
        close_rate=metrics.contradictions_closed_total / turns,
        net_accumulation=(metrics.contradictions_opened_total - metrics.contradictions_closed_total) / turns,
        tau_resolve_count=len(all_tau),
        tau_resolve_avg=sum(all_tau) / len(all_tau) if all_tau else 0,
        tau_resolve_max=max(all_tau) if all_tau else 0,
        state_changes=metrics.state_hash_changes_total,
        rho_S=metrics.get_rho_S(),
        regime=regime_analysis.regime.name,
        regime_confidence=regime_analysis.confidence,
    )


def run_full_sweep() -> List[SweepResult]:
    """Run the full parameter sweep."""
    refill_rates = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0, 5.0]
    results = []
    
    print("=" * 70)
    print("BUDGET REFILL RATE SWEEP")
    print("Target pathology: BUDGET_STARVATION")
    print("=" * 70)
    
    for rate in refill_rates:
        print(f"\nRunning refill_rate={rate}...")
        result = run_sweep_point(rate, turns=100)
        results.append(result)
        
        # Quick summary
        print(f"  Budget BLOCKs: {result.budget_block_count} ({result.budget_block_fraction:.1%})")
        print(f"  C_open: avg={result.avg_c_open:.1f}, final={result.final_c_open}")
        print(f"  Closed: {result.total_closed}, τ_avg={result.tau_resolve_avg:.0f}ms")
        print(f"  closed_without_evidence: {result.closed_without_evidence}")
        print(f"  Regime: {result.regime}")
    
    return results


def analyze_sweep(results: List[SweepResult]) -> Dict[str, Any]:
    """Analyze sweep results."""
    analysis = {
        "parameter": "repair_refill_rate",
        "points": len(results),
        "safety_invariant_held": all(r.closed_without_evidence == 0 for r in results),
        "results": [],
    }
    
    for r in results:
        analysis["results"].append({
            "refill_rate": r.refill_rate,
            "budget_block_fraction": r.budget_block_fraction,
            "final_c_open": r.final_c_open,
            "close_rate": r.close_rate,
            "regime": r.regime,
            "rho_S": r.rho_S,
        })
    
    # Find transition points
    starvation_threshold = None
    healthy_threshold = None
    
    for r in results:
        if r.regime == "BUDGET_STARVATION" and starvation_threshold is None:
            starvation_threshold = r.refill_rate
        if r.regime == "HEALTHY_LATTICE" and starvation_threshold is not None:
            healthy_threshold = r.refill_rate
            break
    
    analysis["starvation_below"] = starvation_threshold
    analysis["healthy_above"] = healthy_threshold
    
    return analysis


def print_sweep_report(results: List[SweepResult], analysis: Dict[str, Any]):
    """Print formatted sweep report."""
    print("\n" + "=" * 70)
    print("SWEEP REPORT")
    print("=" * 70)
    
    print(f"\nParameter: {analysis['parameter']}")
    print(f"Points: {analysis['points']}")
    print(f"Safety invariant (closed_without_evidence=0): {'✓ HELD' if analysis['safety_invariant_held'] else '✗ VIOLATED'}")
    
    print("\n### Results Table")
    print("-" * 70)
    print(f"{'Rate':>6} | {'BudgetBLOCK%':>12} | {'C_open':>8} | {'CloseRate':>10} | {'Regime':>20}")
    print("-" * 70)
    
    for r in results:
        print(f"{r.refill_rate:>6.1f} | {r.budget_block_fraction:>11.1%} | {r.final_c_open:>8} | {r.close_rate:>10.3f} | {r.regime:>20}")
    
    print("-" * 70)
    
    if analysis.get("starvation_below"):
        print(f"\n✗ BUDGET_STARVATION below refill_rate={analysis['starvation_below']}")
    if analysis.get("healthy_above"):
        print(f"✓ HEALTHY_LATTICE above refill_rate={analysis['healthy_above']}")
    
    # Interpretation
    print("\n### Interpretation")
    
    # Find the transition point
    prev_regime = None
    for r in results:
        if prev_regime == "BUDGET_STARVATION" and r.regime != "BUDGET_STARVATION":
            print(f"\nTransition from STARVATION → {r.regime} at refill_rate={r.refill_rate}")
            print(f"  Budget BLOCK fraction dropped from previous to {r.budget_block_fraction:.1%}")
            print(f"  Close rate increased to {r.close_rate:.3f}")
        prev_regime = r.regime
    
    # Check for ceremony (rho_S collapse)
    ceremony_risk = [r for r in results if r.rho_S < 0.1]
    if ceremony_risk:
        print(f"\n⚠ CEREMONY RISK: rho_S < 0.1 at rates: {[r.refill_rate for r in ceremony_risk]}")
    else:
        print("\n✓ No ceremony risk (rho_S stayed healthy)")
    
    # Check for sloppy fluid (would show as closed_without_evidence > 0)
    sloppy_risk = [r for r in results if r.closed_without_evidence > 0]
    if sloppy_risk:
        print(f"\n✗ SLOPPY FLUID DETECTED: closed_without_evidence > 0")
    else:
        print("✓ No sloppy fluid risk (all closures had evidence)")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    results = run_full_sweep()
    analysis = analyze_sweep(results)
    print_sweep_report(results, analysis)
    
    # Save results
    output_path = "/mnt/user-data/outputs/epistemic_governor/sweep_results.json"
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nResults saved to: {output_path}")
