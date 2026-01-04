"""
Diagnostic Workloads

Realistic workloads to observe system behavior under different conditions.
No behavior changes - pure observation.

Workloads:
1. Steady state: Normal operation with balanced contradictions
2. Contradiction storm: Rapid conflict accumulation
3. Resolution burst: Heavy repair activity
4. Budget pressure: Operation under constrained resources
5. Mixed realistic: Combination simulating real usage
"""

import uuid
import random
import hashlib
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Dict, Any

from epistemic_governor.diagnostics import (
    DiagnosticLogger, DiagnosticEvent, RegimeDetector,
    compute_energy, Regime,
)
from epistemic_governor.hysteresis import (
    HysteresisState, Contradiction, ContradictionStatus,
    ContradictionSeverity, ContradictionSet,
)


# =============================================================================
# Workload Generators
# =============================================================================

class WorkloadGenerator:
    """Base class for workload generation."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.turn = 0
        self.state = HysteresisState(state_id=f"workload_{self.__class__.__name__}")
        self.domains = ["facts", "dates", "config", "identity", "policy"]
    
    def generate_prompt_hash(self) -> str:
        """Generate a prompt hash."""
        return hashlib.sha256(f"prompt_{self.turn}_{self.rng.random()}".encode()).hexdigest()[:16]
    
    def step(self) -> Dict[str, Any]:
        """
        Execute one step of the workload.
        Returns event data to be recorded.
        """
        raise NotImplementedError


class SteadyStateWorkload(WorkloadGenerator):
    """
    Steady state: balanced operation.
    
    - Occasional new contradictions (10% of turns)
    - Matching resolution rate
    - Stable C_open around 5-10
    - Budget healthy
    """
    
    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self.target_c_open = 7
        
        # Initialize with some contradictions
        for i in range(self.target_c_open):
            self._add_contradiction()
    
    def _add_contradiction(self):
        domain = self.rng.choice(self.domains)
        claim_a = f"claim_{uuid.uuid4().hex[:8]}"
        claim_b = f"claim_{uuid.uuid4().hex[:8]}"
        
        self.state.commit_claim(claim_a, domain, "value", f"A_{self.turn}")
        self.state.commit_claim(claim_b, domain, "value", f"B_{self.turn}")
        self.state.add_contradiction(
            claim_a, claim_b, domain,
            severity=self.rng.choice(list(ContradictionSeverity)),
        )
    
    def _resolve_oldest(self) -> float:
        """Resolve oldest contradiction, return resolution time in ms."""
        open_c = self.state.contradictions.get_open()
        if not open_c:
            return 0.0
        
        oldest = min(open_c, key=lambda c: c.opened_at)
        resolution_time = (datetime.now(timezone.utc) - oldest.opened_at).total_seconds() * 1000
        
        self.state.close_contradiction(
            oldest.contradiction_id,
            f"evidence_{uuid.uuid4().hex[:8]}",
            oldest.claim_a_id,
        )
        return resolution_time
    
    def step(self) -> Dict[str, Any]:
        self.turn += 1
        
        c_open_before = self.state.contradictions.open_count
        opened = 0
        closed = 0
        tau_resolve = []
        
        # Maybe add contradiction (10%)
        if self.rng.random() < 0.10:
            self._add_contradiction()
            opened = 1
        
        # Maybe resolve to maintain balance
        if c_open_before > self.target_c_open and self.rng.random() < 0.3:
            tau = self._resolve_oldest()
            if tau > 0:
                closed = 1
                tau_resolve.append(tau)
        elif c_open_before > 0 and self.rng.random() < 0.08:
            tau = self._resolve_oldest()
            if tau > 0:
                closed = 1
                tau_resolve.append(tau)
        
        # Compute state
        c_open_after = self.state.contradictions.open_count
        severity_sum = self.state.contradictions.total_severity
        
        return {
            "prompt_hash": self.generate_prompt_hash(),
            "state_hash_before": f"steady_{self.turn-1}",
            "state_hash_after": f"steady_{self.turn}",
            "verdict": "OK" if c_open_after < 15 else "WARN",
            "c_open_before": c_open_before,
            "c_open_after": c_open_after,
            "c_opened_count": opened,
            "c_closed_count": closed,
            "tau_resolve_ms_closed": tau_resolve,
            "budget_remaining": {"append": 80.0, "resolve": 60.0},
            "budget_spent": {"append": 1.0, "resolve": 5.0 * closed},
            "severity_sum": severity_sum,
            "ledger_entries": 1 + opened + closed,
        }


class ContradictionStormWorkload(WorkloadGenerator):
    """
    Contradiction storm: rapid conflict accumulation.
    
    - High rate of new contradictions (40% of turns)
    - Low resolution rate (5%)
    - C_open trends upward
    - Should trigger GLASS regime detection
    """
    
    def __init__(self, seed: int = 42):
        super().__init__(seed)
    
    def step(self) -> Dict[str, Any]:
        self.turn += 1
        
        c_open_before = self.state.contradictions.open_count
        opened = 0
        closed = 0
        tau_resolve = []
        
        # High contradiction rate
        if self.rng.random() < 0.40:
            domain = self.rng.choice(self.domains)
            claim_a = f"claim_{uuid.uuid4().hex[:8]}"
            claim_b = f"claim_{uuid.uuid4().hex[:8]}"
            self.state.commit_claim(claim_a, domain, "value", f"A_{self.turn}")
            self.state.commit_claim(claim_b, domain, "value", f"B_{self.turn}")
            self.state.add_contradiction(
                claim_a, claim_b, domain,
                severity=self.rng.choice([ContradictionSeverity.MEDIUM, ContradictionSeverity.HIGH]),
            )
            opened = 1
        
        # Low resolution rate
        if c_open_before > 0 and self.rng.random() < 0.05:
            open_c = self.state.contradictions.get_open()
            if open_c:
                oldest = min(open_c, key=lambda c: c.opened_at)
                tau = (datetime.now(timezone.utc) - oldest.opened_at).total_seconds() * 1000
                self.state.close_contradiction(
                    oldest.contradiction_id,
                    f"evidence_{uuid.uuid4().hex[:8]}",
                    oldest.claim_a_id,
                )
                closed = 1
                tau_resolve.append(tau)
        
        c_open_after = self.state.contradictions.open_count
        severity_sum = self.state.contradictions.total_severity
        
        # Verdict degrades as contradictions pile up
        if c_open_after > 20:
            verdict = "BLOCK"
        elif c_open_after > 10:
            verdict = "WARN"
        else:
            verdict = "OK"
        
        return {
            "prompt_hash": self.generate_prompt_hash(),
            "state_hash_before": f"storm_{self.turn-1}",
            "state_hash_after": f"storm_{self.turn}",
            "verdict": verdict,
            "c_open_before": c_open_before,
            "c_open_after": c_open_after,
            "c_opened_count": opened,
            "c_closed_count": closed,
            "tau_resolve_ms_closed": tau_resolve,
            "budget_remaining": {"append": 90.0, "resolve": 80.0},
            "budget_spent": {"append": 1.0, "resolve": 3.0 * closed},
            "severity_sum": severity_sum,
            "ledger_entries": 1 + opened,
        }


class ResolutionBurstWorkload(WorkloadGenerator):
    """
    Resolution burst: heavy repair activity.
    
    - Start with many contradictions
    - High resolution rate
    - Budget drains quickly
    - Tests repair pathway
    """
    
    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self.resolve_budget = 100.0
        
        # Initialize with many contradictions
        for i in range(25):
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
        
        # Occasional new contradiction
        if self.rng.random() < 0.05:
            domain = self.rng.choice(self.domains)
            claim_a = f"claim_{uuid.uuid4().hex[:8]}"
            claim_b = f"claim_{uuid.uuid4().hex[:8]}"
            self.state.commit_claim(claim_a, domain, "value", f"A_{self.turn}")
            self.state.commit_claim(claim_b, domain, "value", f"B_{self.turn}")
            self.state.add_contradiction(claim_a, claim_b, domain)
            opened = 1
        
        # High resolution rate (if budget allows)
        resolution_cost = 5.0
        while c_open_before - closed > 0 and self.resolve_budget >= resolution_cost and self.rng.random() < 0.4:
            open_c = self.state.contradictions.get_open()
            if not open_c:
                break
            
            oldest = min(open_c, key=lambda c: c.opened_at)
            tau = (datetime.now(timezone.utc) - oldest.opened_at).total_seconds() * 1000
            self.state.close_contradiction(
                oldest.contradiction_id,
                f"evidence_{uuid.uuid4().hex[:8]}",
                oldest.claim_a_id,
            )
            closed += 1
            tau_resolve.append(tau)
            self.resolve_budget -= resolution_cost
            budget_spent += resolution_cost
        
        c_open_after = self.state.contradictions.open_count
        severity_sum = self.state.contradictions.total_severity
        
        # Block if budget exhausted
        verdict = "OK"
        blocked_by = []
        if self.resolve_budget < resolution_cost and c_open_after > 0:
            verdict = "BLOCK"
            blocked_by = ["BUDGET_EXHAUSTED"]
        
        return {
            "prompt_hash": self.generate_prompt_hash(),
            "state_hash_before": f"burst_{self.turn-1}",
            "state_hash_after": f"burst_{self.turn}",
            "verdict": verdict,
            "blocked_by_invariant": blocked_by,
            "c_open_before": c_open_before,
            "c_open_after": c_open_after,
            "c_opened_count": opened,
            "c_closed_count": closed,
            "tau_resolve_ms_closed": tau_resolve,
            "budget_remaining": {"append": 90.0, "resolve": self.resolve_budget},
            "budget_spent": {"append": 1.0, "resolve": budget_spent},
            "budget_exhaustion": {"resolve": self.resolve_budget < resolution_cost},
            "severity_sum": severity_sum,
            "ledger_entries": 1 + opened + closed,
        }


class BudgetPressureWorkload(WorkloadGenerator):
    """
    Budget pressure: operation under constrained resources.
    
    - Limited budgets
    - Normal contradiction rate
    - Should trigger BUDGET_STARVATION if too tight
    """
    
    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self.append_budget = 30.0
        self.resolve_budget = 20.0
    
    def step(self) -> Dict[str, Any]:
        self.turn += 1
        
        c_open_before = self.state.contradictions.open_count
        opened = 0
        closed = 0
        tau_resolve = []
        append_spent = 0.0
        resolve_spent = 0.0
        
        verdict = "OK"
        blocked_by = []
        
        # Try to add contradiction
        if self.rng.random() < 0.15:
            if self.append_budget >= 2.0:
                domain = self.rng.choice(self.domains)
                claim_a = f"claim_{uuid.uuid4().hex[:8]}"
                claim_b = f"claim_{uuid.uuid4().hex[:8]}"
                self.state.commit_claim(claim_a, domain, "value", f"A_{self.turn}")
                self.state.commit_claim(claim_b, domain, "value", f"B_{self.turn}")
                self.state.add_contradiction(claim_a, claim_b, domain)
                opened = 1
                self.append_budget -= 2.0
                append_spent = 2.0
            else:
                verdict = "BLOCK"
                blocked_by.append("APPEND_BUDGET_EXHAUSTED")
        
        # Try to resolve
        if c_open_before > 0 and self.rng.random() < 0.10:
            if self.resolve_budget >= 5.0:
                open_c = self.state.contradictions.get_open()
                if open_c:
                    oldest = min(open_c, key=lambda c: c.opened_at)
                    tau = (datetime.now(timezone.utc) - oldest.opened_at).total_seconds() * 1000
                    self.state.close_contradiction(
                        oldest.contradiction_id,
                        f"evidence_{uuid.uuid4().hex[:8]}",
                        oldest.claim_a_id,
                    )
                    closed = 1
                    tau_resolve.append(tau)
                    self.resolve_budget -= 5.0
                    resolve_spent = 5.0
            else:
                if verdict == "OK":
                    verdict = "BLOCK"
                blocked_by.append("RESOLVE_BUDGET_EXHAUSTED")
        
        # Slow budget recovery
        self.append_budget = min(30.0, self.append_budget + 0.5)
        self.resolve_budget = min(20.0, self.resolve_budget + 0.3)
        
        c_open_after = self.state.contradictions.open_count
        severity_sum = self.state.contradictions.total_severity
        
        return {
            "prompt_hash": self.generate_prompt_hash(),
            "state_hash_before": f"pressure_{self.turn-1}",
            "state_hash_after": f"pressure_{self.turn}",
            "verdict": verdict,
            "blocked_by_invariant": blocked_by,
            "c_open_before": c_open_before,
            "c_open_after": c_open_after,
            "c_opened_count": opened,
            "c_closed_count": closed,
            "tau_resolve_ms_closed": tau_resolve,
            "budget_remaining": {"append": self.append_budget, "resolve": self.resolve_budget},
            "budget_spent": {"append": append_spent, "resolve": resolve_spent},
            "budget_exhaustion": {
                "append": self.append_budget < 2.0,
                "resolve": self.resolve_budget < 5.0,
            },
            "severity_sum": severity_sum,
            "ledger_entries": 1 + opened + closed,
        }


class MixedRealisticWorkload(WorkloadGenerator):
    """
    Mixed realistic: combination simulating real usage.
    
    - Variable contradiction rate
    - Periodic resolution bursts
    - Occasional budget pressure
    - Some extraction failures
    """
    
    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self.append_budget = 100.0
        self.resolve_budget = 50.0
        self.phase = "normal"  # normal, busy, repair
        self.phase_turns = 0
    
    def _maybe_switch_phase(self):
        self.phase_turns += 1
        if self.phase_turns > 20:
            self.phase = self.rng.choice(["normal", "busy", "repair"])
            self.phase_turns = 0
    
    def step(self) -> Dict[str, Any]:
        self.turn += 1
        self._maybe_switch_phase()
        
        c_open_before = self.state.contradictions.open_count
        opened = 0
        closed = 0
        tau_resolve = []
        append_spent = 0.0
        resolve_spent = 0.0
        
        verdict = "OK"
        blocked_by = []
        warn_reasons = []
        extract_status = "ok"
        
        # Occasional extraction failure (0.3%)
        if self.rng.random() < 0.003:
            extract_status = "fail"
            verdict = "BLOCK"
            blocked_by.append("EXTRACTION_FAILED")
        else:
            # Phase-dependent behavior
            contradiction_rate = {"normal": 0.08, "busy": 0.25, "repair": 0.02}[self.phase]
            resolution_rate = {"normal": 0.06, "busy": 0.03, "repair": 0.30}[self.phase]
            
            # Add contradictions
            if self.rng.random() < contradiction_rate and self.append_budget >= 2.0:
                domain = self.rng.choice(self.domains)
                claim_a = f"claim_{uuid.uuid4().hex[:8]}"
                claim_b = f"claim_{uuid.uuid4().hex[:8]}"
                self.state.commit_claim(claim_a, domain, "value", f"A_{self.turn}")
                self.state.commit_claim(claim_b, domain, "value", f"B_{self.turn}")
                self.state.add_contradiction(
                    claim_a, claim_b, domain,
                    severity=self.rng.choice(list(ContradictionSeverity)),
                )
                opened = 1
                self.append_budget -= 2.0
                append_spent = 2.0
            
            # Resolve contradictions
            if c_open_before > 0 and self.rng.random() < resolution_rate:
                if self.resolve_budget >= 5.0:
                    open_c = self.state.contradictions.get_open()
                    if open_c:
                        target = self.rng.choice(open_c)
                        tau = (datetime.now(timezone.utc) - target.opened_at).total_seconds() * 1000
                        self.state.close_contradiction(
                            target.contradiction_id,
                            f"evidence_{uuid.uuid4().hex[:8]}",
                            target.claim_a_id,
                        )
                        closed = 1
                        tau_resolve.append(tau)
                        self.resolve_budget -= 5.0
                        resolve_spent = 5.0
                else:
                    warn_reasons.append("LOW_RESOLVE_BUDGET")
            
            # Check state
            c_open_after = self.state.contradictions.open_count
            if c_open_after > 15:
                if verdict == "OK":
                    verdict = "WARN"
                warn_reasons.append("HIGH_CONTRADICTION_LOAD")
        
        # Budget recovery
        self.append_budget = min(100.0, self.append_budget + 1.0)
        self.resolve_budget = min(50.0, self.resolve_budget + 0.5)
        
        c_open_after = self.state.contradictions.open_count
        severity_sum = self.state.contradictions.total_severity
        
        return {
            "prompt_hash": self.generate_prompt_hash(),
            "state_hash_before": f"mixed_{self.turn-1}",
            "state_hash_after": f"mixed_{self.turn}",
            "verdict": verdict,
            "blocked_by_invariant": blocked_by,
            "warn_reasons": warn_reasons,
            "extract_status": extract_status,
            "c_open_before": c_open_before,
            "c_open_after": c_open_after,
            "c_opened_count": opened,
            "c_closed_count": closed,
            "tau_resolve_ms_closed": tau_resolve,
            "budget_remaining": {"append": self.append_budget, "resolve": self.resolve_budget},
            "budget_spent": {"append": append_spent, "resolve": resolve_spent},
            "budget_exhaustion": {
                "append": self.append_budget < 2.0,
                "resolve": self.resolve_budget < 5.0,
            },
            "severity_sum": severity_sum,
            "ledger_entries": 1 + opened + closed,
            "phase": self.phase,
        }


class NearMissAmbiguityWorkload(WorkloadGenerator):
    """
    Near-miss ambiguity: claims that are almost conflicting.
    
    - Inject claims with overlapping but not identical predicates
    - Expect WARN more than BLOCK
    - Tests "sloppy fluid" vs "healthy lattice" boundary
    - Should produce mixed OK/WARN/BLOCK distribution
    """
    
    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self.ambiguous_domains = ["dates", "quantities", "versions"]
        
        # Pre-populate with some base claims
        for i in range(5):
            domain = self.rng.choice(self.ambiguous_domains)
            self.state.commit_claim(
                f"base_claim_{i}",
                domain,
                "value",
                f"base_{i}",
                sigma=0.7,
            )
    
    def step(self) -> Dict[str, Any]:
        self.turn += 1
        
        c_open_before = self.state.contradictions.open_count
        opened = 0
        closed = 0
        tau_resolve = []
        verdict = "OK"
        warn_reasons = []
        
        # Inject near-miss claims (30% of turns)
        if self.rng.random() < 0.30:
            domain = self.rng.choice(self.ambiguous_domains)
            
            # Create claim that might conflict
            claim_id = f"near_miss_{uuid.uuid4().hex[:8]}"
            
            # Randomly decide conflict level
            conflict_level = self.rng.random()
            
            if conflict_level < 0.3:
                # Clear conflict - should trigger contradiction
                existing = [c for c in self.state.commitments.values() 
                           if c.get("domain") == domain]
                if existing:
                    base = self.rng.choice(existing)
                    self.state.commit_claim(
                        claim_id, domain, base["predicate"], 
                        f"DIFFERENT_{base['value']}", sigma=0.8
                    )
                    self.state.add_contradiction(
                        base["claim_id"], claim_id, domain,
                        severity=ContradictionSeverity.MEDIUM,
                    )
                    opened = 1
                    verdict = "WARN"
                    warn_reasons.append("NEAR_CONFLICT_DETECTED")
            
            elif conflict_level < 0.6:
                # Ambiguous - might conflict, WARN
                self.state.commit_claim(
                    claim_id, domain, "maybe_value",
                    f"ambiguous_{self.turn}", sigma=0.5
                )
                if self.rng.random() < 0.4:
                    verdict = "WARN"
                    warn_reasons.append("AMBIGUOUS_CLAIM")
            
            else:
                # Safe addition
                self.state.commit_claim(
                    claim_id, domain, f"safe_pred_{self.turn}",
                    f"safe_{self.turn}", sigma=0.9
                )
        
        # Occasionally resolve (15%)
        if c_open_before > 0 and self.rng.random() < 0.15:
            open_c = self.state.contradictions.get_open()
            if open_c:
                target = self.rng.choice(open_c)
                tau = (datetime.now(timezone.utc) - target.opened_at).total_seconds() * 1000
                self.state.close_contradiction(
                    target.contradiction_id,
                    f"evidence_{uuid.uuid4().hex[:8]}",
                    target.claim_a_id,
                )
                closed = 1
                tau_resolve.append(tau)
        
        c_open_after = self.state.contradictions.open_count
        severity_sum = self.state.contradictions.total_severity
        
        # Severity distribution
        severity_hist = {}
        for c in self.state.contradictions.get_open():
            sev = c.severity.name
            severity_hist[sev] = severity_hist.get(sev, 0) + 1
        
        return {
            "prompt_hash": self.generate_prompt_hash(),
            "state_hash_before": f"nearmiss_{self.turn-1}",
            "state_hash_after": f"nearmiss_{self.turn}",
            "verdict": verdict,
            "warn_reasons": warn_reasons,
            "c_open_before": c_open_before,
            "c_open_after": c_open_after,
            "c_opened_count": opened,
            "c_closed_count": closed,
            "tau_resolve_ms_closed": tau_resolve,
            "c_severity_hist_after": severity_hist,
            "budget_remaining": {"append": 80.0, "resolve": 60.0},
            "budget_spent": {"append": 1.0, "resolve": 5.0 * closed},
            "severity_sum": severity_sum,
            "ledger_entries": 1 + opened + closed,
        }


class EvidenceLatencyWorkload(WorkloadGenerator):
    """
    Evidence latency: open contradictions now, resolve later.
    
    - Open contradictions early
    - Provide evidence after delay
    - Tests τ_resolve distribution (should have heavy-ish tail)
    - Tests repair loop coherence over time
    """
    
    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self.pending_evidence: List[Tuple[int, str]] = []  # (resolve_turn, contradiction_id)
        self.evidence_delay_min = 10
        self.evidence_delay_max = 50
    
    def step(self) -> Dict[str, Any]:
        self.turn += 1
        
        c_open_before = self.state.contradictions.open_count
        opened = 0
        closed = 0
        tau_resolve = []
        verdict = "OK"
        warn_reasons = []
        
        # Check for pending evidence arrivals
        resolved_now = []
        remaining = []
        for resolve_turn, c_id in self.pending_evidence:
            if self.turn >= resolve_turn:
                resolved_now.append(c_id)
            else:
                remaining.append((resolve_turn, c_id))
        self.pending_evidence = remaining
        
        # Resolve contradictions with arrived evidence
        for c_id in resolved_now:
            if c_id in self.state.contradictions.contradictions:
                c = self.state.contradictions.contradictions[c_id]
                if c.status.name == "OPEN":
                    tau = (datetime.now(timezone.utc) - c.opened_at).total_seconds() * 1000
                    self.state.close_contradiction(
                        c_id,
                        f"delayed_evidence_{uuid.uuid4().hex[:8]}",
                        c.claim_a_id,
                    )
                    closed += 1
                    tau_resolve.append(tau)
        
        # Open new contradictions (20%)
        if self.rng.random() < 0.20:
            domain = self.rng.choice(self.domains)
            claim_a = f"claim_{uuid.uuid4().hex[:8]}"
            claim_b = f"claim_{uuid.uuid4().hex[:8]}"
            self.state.commit_claim(claim_a, domain, "value", f"A_{self.turn}")
            self.state.commit_claim(claim_b, domain, "value", f"B_{self.turn}")
            
            c = self.state.add_contradiction(
                claim_a, claim_b, domain,
                severity=self.rng.choice(list(ContradictionSeverity)),
            )
            opened = 1
            
            # Schedule evidence arrival
            delay = self.rng.randint(self.evidence_delay_min, self.evidence_delay_max)
            self.pending_evidence.append((self.turn + delay, c.contradiction_id))
            
            # WARN for new contradiction
            verdict = "WARN"
            warn_reasons.append("NEW_CONTRADICTION_PENDING_EVIDENCE")
        
        c_open_after = self.state.contradictions.open_count
        severity_sum = self.state.contradictions.total_severity
        
        # Severity distribution
        severity_hist = {}
        for c in self.state.contradictions.get_open():
            sev = c.severity.name
            severity_hist[sev] = severity_hist.get(sev, 0) + 1
        
        # WARN if many pending
        if c_open_after > 10:
            if verdict == "OK":
                verdict = "WARN"
            warn_reasons.append("HIGH_PENDING_EVIDENCE")
        
        return {
            "prompt_hash": self.generate_prompt_hash(),
            "state_hash_before": f"latency_{self.turn-1}",
            "state_hash_after": f"latency_{self.turn}",
            "verdict": verdict,
            "warn_reasons": warn_reasons,
            "c_open_before": c_open_before,
            "c_open_after": c_open_after,
            "c_opened_count": opened,
            "c_closed_count": closed,
            "tau_resolve_ms_closed": tau_resolve,
            "c_severity_hist_after": severity_hist,
            "budget_remaining": {"append": 90.0, "resolve": 70.0},
            "budget_spent": {"append": 1.0, "resolve": 5.0 * closed},
            "severity_sum": severity_sum,
            "ledger_entries": 1 + opened + closed,
            "pending_evidence_count": len(self.pending_evidence),
        }


class BurstRecoveryWorkload(WorkloadGenerator):
    """
    Burst recovery: spike then recover.
    
    - C_open spikes dramatically, then recovers
    - Tests classifier robustness against false GLASS detection
    - Should NOT trigger GLASS if recovery is fast enough
    """
    
    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self.phase = "normal"  # normal, burst, recovery
        self.phase_start = 0
        self.burst_schedule = [(30, "burst"), (50, "recovery"), (100, "burst"), (130, "recovery")]
        self.schedule_idx = 0
    
    def _check_phase(self):
        if self.schedule_idx < len(self.burst_schedule):
            trigger_turn, next_phase = self.burst_schedule[self.schedule_idx]
            if self.turn >= trigger_turn:
                self.phase = next_phase
                self.phase_start = self.turn
                self.schedule_idx += 1
    
    def step(self) -> Dict[str, Any]:
        self.turn += 1
        self._check_phase()
        
        c_open_before = self.state.contradictions.open_count
        opened = 0
        closed = 0
        tau_resolve = []
        verdict = "OK"
        warn_reasons = []
        
        if self.phase == "burst":
            # Heavy contradiction injection
            if self.rng.random() < 0.6:
                domain = self.rng.choice(self.domains)
                claim_a = f"claim_{uuid.uuid4().hex[:8]}"
                claim_b = f"claim_{uuid.uuid4().hex[:8]}"
                self.state.commit_claim(claim_a, domain, "value", f"A_{self.turn}")
                self.state.commit_claim(claim_b, domain, "value", f"B_{self.turn}")
                self.state.add_contradiction(
                    claim_a, claim_b, domain,
                    severity=self.rng.choice([ContradictionSeverity.LOW, ContradictionSeverity.MEDIUM]),
                )
                opened = 1
            verdict = "WARN"
            warn_reasons.append("BURST_PHASE")
        
        elif self.phase == "recovery":
            # Heavy resolution
            while c_open_before - closed > 0 and self.rng.random() < 0.5:
                open_c = self.state.contradictions.get_open()
                if not open_c:
                    break
                target = self.rng.choice(open_c)
                tau = (datetime.now(timezone.utc) - target.opened_at).total_seconds() * 1000
                self.state.close_contradiction(
                    target.contradiction_id,
                    f"evidence_{uuid.uuid4().hex[:8]}",
                    target.claim_a_id,
                )
                closed += 1
                tau_resolve.append(tau)
        
        else:  # normal
            # Light activity
            if self.rng.random() < 0.05:
                domain = self.rng.choice(self.domains)
                claim_a = f"claim_{uuid.uuid4().hex[:8]}"
                claim_b = f"claim_{uuid.uuid4().hex[:8]}"
                self.state.commit_claim(claim_a, domain, "value", f"A_{self.turn}")
                self.state.commit_claim(claim_b, domain, "value", f"B_{self.turn}")
                self.state.add_contradiction(claim_a, claim_b, domain)
                opened = 1
            
            if c_open_before > 0 and self.rng.random() < 0.1:
                open_c = self.state.contradictions.get_open()
                if open_c:
                    target = self.rng.choice(open_c)
                    tau = (datetime.now(timezone.utc) - target.opened_at).total_seconds() * 1000
                    self.state.close_contradiction(
                        target.contradiction_id,
                        f"evidence_{uuid.uuid4().hex[:8]}",
                        target.claim_a_id,
                    )
                    closed = 1
                    tau_resolve.append(tau)
        
        c_open_after = self.state.contradictions.open_count
        severity_sum = self.state.contradictions.total_severity
        
        # Severity distribution
        severity_hist = {}
        for c in self.state.contradictions.get_open():
            sev = c.severity.name
            severity_hist[sev] = severity_hist.get(sev, 0) + 1
        
        return {
            "prompt_hash": self.generate_prompt_hash(),
            "state_hash_before": f"burst_{self.turn-1}",
            "state_hash_after": f"burst_{self.turn}",
            "verdict": verdict,
            "warn_reasons": warn_reasons,
            "c_open_before": c_open_before,
            "c_open_after": c_open_after,
            "c_opened_count": opened,
            "c_closed_count": closed,
            "tau_resolve_ms_closed": tau_resolve,
            "c_severity_hist_after": severity_hist,
            "budget_remaining": {"append": 100.0, "resolve": 100.0},
            "budget_spent": {"append": 1.0, "resolve": 5.0 * closed},
            "severity_sum": severity_sum,
            "ledger_entries": 1 + opened + closed,
            "phase": self.phase,
        }


# =============================================================================
# Workload Runner
# =============================================================================

def run_workload(
    workload: WorkloadGenerator,
    logger: DiagnosticLogger,
    turns: int = 100,
) -> None:
    """Run a workload and record diagnostics."""
    for _ in range(turns):
        data = workload.step()
        
        event = logger.create_event(
            prompt_hash=data["prompt_hash"],
            state_hash_before=data["state_hash_before"],
            state_hash_after=data["state_hash_after"],
            state_view_hash=f"view_{workload.turn}",
        )
        
        # Fill in event data
        event.verdict = data.get("verdict", "OK")
        event.blocked_by_invariant = data.get("blocked_by_invariant", [])
        event.warn_reasons = data.get("warn_reasons", [])
        event.extract_status = data.get("extract_status", "ok")
        
        event.c_open_before = data.get("c_open_before", 0)
        event.c_open_after = data.get("c_open_after", 0)
        event.c_opened_count = data.get("c_opened_count", 0)
        event.c_closed_count = data.get("c_closed_count", 0)
        event.tau_resolve_ms_closed = data.get("tau_resolve_ms_closed", [])
        
        event.budget_remaining_after = data.get("budget_remaining", {})
        event.budget_spent_this_turn = data.get("budget_spent", {})
        event.budget_exhaustion = data.get("budget_exhaustion", {})
        
        event.ledger_entries_appended = data.get("ledger_entries", 1)
        event.ledger_entry_types_appended = {"ASSERT": data.get("ledger_entries", 1)}
        
        # Compute energy
        energy = compute_energy(
            c_open=event.c_open_after,
            severity_sum=data.get("severity_sum", 0),
            budget_remaining=event.budget_remaining_after,
        )
        event.E_state_after = energy.total()
        event.E_components_after = energy.to_dict()
        
        logger.record(event)


def run_all_workloads() -> Dict[str, DiagnosticLogger]:
    """Run all workloads and return loggers."""
    workloads = {
        "steady_state": (SteadyStateWorkload(seed=42), 200),
        "contradiction_storm": (ContradictionStormWorkload(seed=43), 150),
        "resolution_burst": (ResolutionBurstWorkload(seed=44), 100),
        "budget_pressure": (BudgetPressureWorkload(seed=45), 150),
        "mixed_realistic": (MixedRealisticWorkload(seed=46), 300),
        # New boundary-realism workloads
        "near_miss_ambiguity": (NearMissAmbiguityWorkload(seed=47), 200),
        "evidence_latency": (EvidenceLatencyWorkload(seed=48), 200),
        "burst_recovery": (BurstRecoveryWorkload(seed=49), 200),
    }
    
    loggers = {}
    
    for name, (workload, turns) in workloads.items():
        print(f"\nRunning workload: {name} ({turns} turns)...")
        logger = DiagnosticLogger(run_id=f"workload_{name}", suite="workload")
        run_workload(workload, logger, turns)
        loggers[name] = logger
        
        # Print summary
        summary = logger.summary()
        regime = logger.get_regime()
        
        print(f"  Turns: {summary['turns']}")
        print(f"  Regime: {regime.regime.name} (confidence: {regime.confidence:.2f})")
        print(f"  ρ_S: {summary['metrics']['rho_S']:.3f}")
        print(f"  C_open: {summary['metrics']['contradictions_open']}")
        print(f"  Verdicts: {summary['metrics']['verdict_distribution']}")
        
        if regime.warnings:
            print(f"  Warnings: {regime.warnings}")
    
    return loggers


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DIAGNOSTIC WORKLOADS")
    print("Observing System Behavior Under Realistic Conditions")
    print("=" * 70)
    
    loggers = run_all_workloads()
    
    # Export JSONL for query layer
    print("\n" + "=" * 70)
    print("EXPORTING JSONL FOR QUERY LAYER")
    print("=" * 70)
    
    all_events = []
    for name, logger in loggers.items():
        jsonl = logger.get_jsonl()
        lines = jsonl.strip().split("\n") if jsonl.strip() else []
        all_events.extend(lines)
        print(f"  {name}: {len(lines)} events")
    
    # Write combined JSONL
    output_path = "/mnt/user-data/outputs/epistemic_governor/diagnostic_events.jsonl"
    with open(output_path, "w") as f:
        f.write("\n".join(all_events))
    
    print(f"\nTotal events: {len(all_events)}")
    print(f"Written to: {output_path}")
    
    print("\n" + "=" * 70)
    print("REGIME SUMMARY")
    print("=" * 70)
    
    for name, logger in loggers.items():
        regime = logger.get_regime()
        print(f"\n{name}:")
        print(f"  Regime: {regime.regime.name}")
        print(f"  Confidence: {regime.confidence:.2f}")
        print(f"  Indicators: {regime.indicators}")
        if regime.recommendations:
            print(f"  Recommendations: {regime.recommendations}")
