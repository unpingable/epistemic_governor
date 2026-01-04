"""
Competence Metrics

The distinction that matters:
- Capability: What problem classes are solvable at all (can't improve without training)
- Competence: How well it behaves on those classes (CAN improve with control)

What we can legitimately improve:
1. Effective reasoning under constraint (less self-sabotage)
2. Reliability of intermediate products (cleaner decomposition)
3. Search behavior (cognition-by-orchestration)

What we cannot improve:
1. Core representation (if concept missing, can't conjure it)
2. Generalization (no new abilities)
3. Deep novel synthesis (can't create missing latent structure)

Measurable competence metrics:
- Answer acceptance rate under hard gates
- Tool-first success rate
- Net displacement per token (progress efficiency)
- Recovery time from drift regimes
- Hallucination suppression rate

The punchline:
"Smarter" isn't more clever sentences. Smarter is:
When constrained, it stops lying and starts navigating the unknown honestly.

Usage:
    from epistemic_governor.competence import (
        CompetenceTracker,
        CompetenceReport,
        EfficiencyMetrics,
    )
    
    tracker = CompetenceTracker()
    
    # Record turn outcomes
    tracker.record_turn(
        tokens_used=150,
        claims_proposed=3,
        claims_accepted=2,
        claims_rejected=1,
        tool_calls=1,
        tool_first=True,
        displacement=0.5,
    )
    
    # Get competence report
    report = tracker.get_report()
    print(f"Progress efficiency: {report.displacement_per_token:.4f}")
    print(f"Tool-first rate: {report.tool_first_rate:.1%}")
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum, auto
import statistics


# =============================================================================
# Turn Outcomes
# =============================================================================

@dataclass
class TurnOutcome:
    """
    Outcome of a single turn for competence tracking.
    """
    turn_id: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Token efficiency
    tokens_used: int = 0
    tokens_burned_retries: int = 0  # Wasted on rejected outputs
    
    # Claim outcomes
    claims_proposed: int = 0
    claims_accepted: int = 0
    claims_rejected: int = 0
    claims_hedged: int = 0
    
    # Tool behavior
    tool_calls_made: int = 0
    tool_calls_successful: int = 0
    tool_first: bool = False  # Did it retrieve BEFORE claiming?
    
    # Vector progress
    displacement: float = 0.0     # Net movement toward goal
    magnitude: float = 0.0        # Total effort expended
    
    # Regime
    regime: str = "UNKNOWN"
    in_hard_gate: bool = False
    
    # Recovery
    recovered_from_drift: bool = False
    turns_in_drift: int = 0
    
    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposed claims that were accepted."""
        if self.claims_proposed == 0:
            return 1.0  # No claims = nothing rejected
        return self.claims_accepted / self.claims_proposed
    
    @property
    def token_efficiency(self) -> float:
        """Fraction of tokens that weren't wasted."""
        total = self.tokens_used + self.tokens_burned_retries
        if total == 0:
            return 1.0
        return self.tokens_used / total
    
    @property
    def progress_efficiency(self) -> float:
        """Displacement per unit effort (avoid division by zero)."""
        if self.magnitude == 0:
            return 0.0
        return self.displacement / self.magnitude


# =============================================================================
# Competence Metrics
# =============================================================================

@dataclass
class CompetenceReport:
    """
    Aggregated competence metrics over a session.
    
    These measure effective cognition under constraint.
    Higher is better for all metrics.
    """
    # Session info
    turns: int = 0
    total_tokens: int = 0
    
    # === Acceptance Rate ===
    # "How often does it produce compliant outputs?"
    acceptance_rate: float = 0.0
    acceptance_rate_in_gate: float = 0.0  # Harder: under constraint
    
    # === Token Efficiency ===
    # "How much output is usable vs wasted?"
    token_efficiency: float = 0.0
    tokens_wasted_fraction: float = 0.0
    
    # === Tool-First Rate ===
    # "Does it retrieve before claiming?" (adult cognition)
    tool_first_rate: float = 0.0
    tool_success_rate: float = 0.0
    
    # === Progress Efficiency ===
    # "Displacement per token" - the vector notion
    displacement_per_token: float = 0.0
    displacement_per_turn: float = 0.0
    furnace_turns: int = 0  # High effort, low progress
    
    # === Recovery ===
    # "How quickly does it recover from drift?"
    drift_episodes: int = 0
    avg_recovery_turns: float = 0.0
    recovery_success_rate: float = 0.0
    
    # === Regime Distribution ===
    # "Where does it spend its time?"
    regime_distribution: Dict[str, float] = field(default_factory=dict)
    time_in_grounded: float = 0.0
    time_in_drift: float = 0.0
    
    # === Hallucination Suppression ===
    # "Rejected unsupported claims"
    hallucination_attempts: int = 0
    hallucination_blocked: int = 0
    suppression_rate: float = 0.0
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=== COMPETENCE REPORT ===",
            f"Turns: {self.turns}, Tokens: {self.total_tokens}",
            "",
            "-- Acceptance --",
            f"  Overall: {self.acceptance_rate:.1%}",
            f"  Under constraint: {self.acceptance_rate_in_gate:.1%}",
            "",
            "-- Efficiency --",
            f"  Token efficiency: {self.token_efficiency:.1%}",
            f"  Displacement/token: {self.displacement_per_token:.4f}",
            f"  Furnace turns: {self.furnace_turns}",
            "",
            "-- Tool Behavior --",
            f"  Tool-first rate: {self.tool_first_rate:.1%}",
            f"  Tool success: {self.tool_success_rate:.1%}",
            "",
            "-- Recovery --",
            f"  Drift episodes: {self.drift_episodes}",
            f"  Avg recovery: {self.avg_recovery_turns:.1f} turns",
            "",
            "-- Hallucination Suppression --",
            f"  Attempts blocked: {self.hallucination_blocked}/{self.hallucination_attempts}",
            f"  Suppression rate: {self.suppression_rate:.1%}",
            "",
            "-- Regime Distribution --",
        ]
        for regime, frac in sorted(self.regime_distribution.items()):
            lines.append(f"  {regime}: {frac:.1%}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "turns": self.turns,
            "total_tokens": self.total_tokens,
            "acceptance_rate": self.acceptance_rate,
            "token_efficiency": self.token_efficiency,
            "tool_first_rate": self.tool_first_rate,
            "displacement_per_token": self.displacement_per_token,
            "avg_recovery_turns": self.avg_recovery_turns,
            "suppression_rate": self.suppression_rate,
            "time_in_grounded": self.time_in_grounded,
            "furnace_turns": self.furnace_turns,
        }


# =============================================================================
# Competence Tracker
# =============================================================================

class CompetenceTracker:
    """
    Tracks competence metrics over a session.
    
    This measures whether the governor is making the model
    "effectively smarter" - not more capable, but more competent.
    """
    
    def __init__(self):
        self.outcomes: List[TurnOutcome] = []
        self._drift_episodes: List[Dict] = []
        self._current_drift: Optional[Dict] = None
        
        # Drift regime names
        self.drift_regimes = {
            "NARRATIVE_DRIFT", "FLUENCY_DOMINANCE", "SOCIAL_INVENTION",
            "CONFABULATION", "ASSOCIATIVE_SPIRAL", "ROLEPLAY_CAPTURE",
            "COMMITMENT_DECAY",
        }
        self.grounded_regimes = {"GROUNDED", "INTERROGATIVE", "PROCEDURAL"}
    
    def record_turn(
        self,
        turn_id: int = 0,
        tokens_used: int = 0,
        tokens_burned_retries: int = 0,
        claims_proposed: int = 0,
        claims_accepted: int = 0,
        claims_rejected: int = 0,
        claims_hedged: int = 0,
        tool_calls: int = 0,
        tool_successes: int = 0,
        tool_first: bool = False,
        displacement: float = 0.0,
        magnitude: float = 0.0,
        regime: str = "UNKNOWN",
        in_hard_gate: bool = False,
    ) -> TurnOutcome:
        """Record a turn outcome."""
        # Auto-assign turn ID
        if turn_id == 0:
            turn_id = len(self.outcomes) + 1
        
        outcome = TurnOutcome(
            turn_id=turn_id,
            tokens_used=tokens_used,
            tokens_burned_retries=tokens_burned_retries,
            claims_proposed=claims_proposed,
            claims_accepted=claims_accepted,
            claims_rejected=claims_rejected,
            claims_hedged=claims_hedged,
            tool_calls_made=tool_calls,
            tool_calls_successful=tool_successes,
            tool_first=tool_first,
            displacement=displacement,
            magnitude=magnitude,
            regime=regime,
            in_hard_gate=in_hard_gate,
        )
        
        # Track drift episodes
        self._track_drift(outcome)
        
        self.outcomes.append(outcome)
        return outcome
    
    def _track_drift(self, outcome: TurnOutcome):
        """Track drift episodes for recovery metrics."""
        is_drift = outcome.regime in self.drift_regimes
        is_grounded = outcome.regime in self.grounded_regimes
        
        if is_drift and self._current_drift is None:
            # Starting new drift episode
            self._current_drift = {
                "start_turn": outcome.turn_id,
                "regimes": [outcome.regime],
                "turns": 1,
            }
        elif is_drift and self._current_drift is not None:
            # Continuing drift
            self._current_drift["regimes"].append(outcome.regime)
            self._current_drift["turns"] += 1
        elif is_grounded and self._current_drift is not None:
            # Recovered from drift
            self._current_drift["end_turn"] = outcome.turn_id
            self._current_drift["recovered"] = True
            self._drift_episodes.append(self._current_drift)
            self._current_drift = None
            outcome.recovered_from_drift = True
    
    def get_report(self) -> CompetenceReport:
        """Generate competence report from recorded outcomes."""
        if not self.outcomes:
            return CompetenceReport()
        
        report = CompetenceReport()
        report.turns = len(self.outcomes)
        
        # Aggregate metrics
        total_tokens = 0
        total_burned = 0
        total_proposed = 0
        total_accepted = 0
        total_rejected = 0
        
        tool_turns = 0
        tool_first_turns = 0
        tool_calls = 0
        tool_successes = 0
        
        total_displacement = 0.0
        total_magnitude = 0.0
        furnace_turns = 0
        
        gated_turns = 0
        gated_accepted = 0
        gated_proposed = 0
        
        regime_counts: Dict[str, int] = {}
        
        for o in self.outcomes:
            total_tokens += o.tokens_used
            total_burned += o.tokens_burned_retries
            total_proposed += o.claims_proposed
            total_accepted += o.claims_accepted
            total_rejected += o.claims_rejected
            
            if o.tool_calls_made > 0:
                tool_turns += 1
                tool_calls += o.tool_calls_made
                tool_successes += o.tool_calls_successful
                if o.tool_first:
                    tool_first_turns += 1
            
            total_displacement += o.displacement
            total_magnitude += o.magnitude
            
            if o.magnitude > 0.5 and o.progress_efficiency < 0.3:
                furnace_turns += 1
            
            if o.in_hard_gate:
                gated_turns += 1
                gated_proposed += o.claims_proposed
                gated_accepted += o.claims_accepted
            
            regime_counts[o.regime] = regime_counts.get(o.regime, 0) + 1
        
        # Calculate rates
        report.total_tokens = total_tokens
        
        # Acceptance
        if total_proposed > 0:
            report.acceptance_rate = total_accepted / total_proposed
        if gated_proposed > 0:
            report.acceptance_rate_in_gate = gated_accepted / gated_proposed
        
        # Token efficiency
        all_tokens = total_tokens + total_burned
        if all_tokens > 0:
            report.token_efficiency = total_tokens / all_tokens
            report.tokens_wasted_fraction = total_burned / all_tokens
        
        # Tool behavior
        if tool_turns > 0:
            report.tool_first_rate = tool_first_turns / tool_turns
        if tool_calls > 0:
            report.tool_success_rate = tool_successes / tool_calls
        
        # Progress efficiency
        if total_tokens > 0:
            report.displacement_per_token = total_displacement / total_tokens
        if report.turns > 0:
            report.displacement_per_turn = total_displacement / report.turns
        report.furnace_turns = furnace_turns
        
        # Recovery
        report.drift_episodes = len(self._drift_episodes)
        if self._drift_episodes:
            recovery_turns = [e["turns"] for e in self._drift_episodes if e.get("recovered")]
            if recovery_turns:
                report.avg_recovery_turns = statistics.mean(recovery_turns)
            recovered = sum(1 for e in self._drift_episodes if e.get("recovered"))
            report.recovery_success_rate = recovered / len(self._drift_episodes)
        
        # Regime distribution
        for regime, count in regime_counts.items():
            report.regime_distribution[regime] = count / report.turns
        
        report.time_in_grounded = sum(
            report.regime_distribution.get(r, 0) for r in self.grounded_regimes
        )
        report.time_in_drift = sum(
            report.regime_distribution.get(r, 0) for r in self.drift_regimes
        )
        
        # Hallucination suppression
        # (rejected claims are potential hallucinations that were blocked)
        report.hallucination_attempts = total_proposed
        report.hallucination_blocked = total_rejected
        if total_proposed > 0:
            # Suppression = rejected / (rejected + accepted_without_support)
            # Simplified: rejected / proposed
            report.suppression_rate = total_rejected / total_proposed
        
        return report
    
    def reset(self):
        """Reset tracker."""
        self.outcomes.clear()
        self._drift_episodes.clear()
        self._current_drift = None


# =============================================================================
# Efficiency Comparison
# =============================================================================

@dataclass
class EfficiencyComparison:
    """
    Compare efficiency between governed and ungoverned runs.
    
    This is the key question: Does the governor make the system
    more competent (not more capable)?
    """
    governed: CompetenceReport
    ungoverned: CompetenceReport
    
    @property
    def acceptance_lift(self) -> float:
        """Improvement in acceptance rate."""
        if self.ungoverned.acceptance_rate == 0:
            return 0.0
        return (self.governed.acceptance_rate - self.ungoverned.acceptance_rate)
    
    @property
    def efficiency_lift(self) -> float:
        """Improvement in token efficiency."""
        return self.governed.token_efficiency - self.ungoverned.token_efficiency
    
    @property
    def tool_first_lift(self) -> float:
        """Improvement in tool-first behavior."""
        return self.governed.tool_first_rate - self.ungoverned.tool_first_rate
    
    @property
    def suppression_lift(self) -> float:
        """Improvement in hallucination suppression."""
        return self.governed.suppression_rate - self.ungoverned.suppression_rate
    
    @property
    def grounded_time_lift(self) -> float:
        """Increase in time spent grounded."""
        return self.governed.time_in_grounded - self.ungoverned.time_in_grounded
    
    def summary(self) -> str:
        """Human-readable comparison."""
        lines = [
            "=== EFFICIENCY COMPARISON ===",
            "",
            "Metric                  Governed  Ungoverned  Lift",
            "-" * 55,
            f"Acceptance rate:        {self.governed.acceptance_rate:6.1%}    {self.ungoverned.acceptance_rate:6.1%}     {self.acceptance_lift:+.1%}",
            f"Token efficiency:       {self.governed.token_efficiency:6.1%}    {self.ungoverned.token_efficiency:6.1%}     {self.efficiency_lift:+.1%}",
            f"Tool-first rate:        {self.governed.tool_first_rate:6.1%}    {self.ungoverned.tool_first_rate:6.1%}     {self.tool_first_lift:+.1%}",
            f"Suppression rate:       {self.governed.suppression_rate:6.1%}    {self.ungoverned.suppression_rate:6.1%}     {self.suppression_lift:+.1%}",
            f"Time grounded:          {self.governed.time_in_grounded:6.1%}    {self.ungoverned.time_in_grounded:6.1%}     {self.grounded_time_lift:+.1%}",
            "",
        ]
        
        # Verdict
        improvements = sum([
            self.acceptance_lift > 0.05,
            self.efficiency_lift > 0.05,
            self.tool_first_lift > 0.1,
            self.grounded_time_lift > 0.1,
        ])
        
        if improvements >= 3:
            lines.append("✓ GOVERNOR EFFECTIVE: Significant competence improvement")
        elif improvements >= 1:
            lines.append("~ MIXED RESULTS: Some competence improvement")
        else:
            lines.append("✗ NO IMPROVEMENT: Governor not adding value")
        
        return "\n".join(lines)


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=== Competence Metrics Demo ===\n")
    
    # Simulate ungoverned session (more drift, less tool use)
    ungoverned = CompetenceTracker()
    for i in range(10):
        regime = "FLUENCY_DOMINANCE" if i % 3 == 0 else "GROUNDED"
        ungoverned.record_turn(
            tokens_used=200,
            tokens_burned_retries=50 if i % 3 == 0 else 0,
            claims_proposed=5,
            claims_accepted=4 if regime == "GROUNDED" else 2,
            claims_rejected=1 if regime == "GROUNDED" else 3,
            tool_calls=1 if i % 4 == 0 else 0,
            tool_first=False,
            displacement=0.3 if regime == "GROUNDED" else 0.1,
            magnitude=0.5,
            regime=regime,
        )
    
    # Simulate governed session (more grounded, tool-first)
    governed = CompetenceTracker()
    for i in range(10):
        regime = "GROUNDED" if i % 5 != 0 else "INTERROGATIVE"
        governed.record_turn(
            tokens_used=150,
            tokens_burned_retries=10 if i % 5 == 0 else 0,
            claims_proposed=3,
            claims_accepted=3 if regime == "GROUNDED" else 2,
            claims_rejected=0 if regime == "GROUNDED" else 1,
            tool_calls=2,
            tool_successes=2,
            tool_first=True,
            displacement=0.5,
            magnitude=0.5,
            regime=regime,
            in_hard_gate=i % 5 == 0,
        )
    
    print("--- Ungoverned Report ---")
    print(ungoverned.get_report().summary())
    
    print("\n--- Governed Report ---")
    print(governed.get_report().summary())
    
    print("\n--- Comparison ---")
    comparison = EfficiencyComparison(
        governed=governed.get_report(),
        ungoverned=ungoverned.get_report(),
    )
    print(comparison.summary())
