"""
Ultrastability Layer

Ashby-style second-order adaptation for S₁ (regulatory parameters).

The governor's budgets, thresholds, and rates can adapt based on
observed S₂ dynamics - but only within constitutional bounds and
only when adaptation criteria are met.

Key principle from Ashby:
> "Change the parameters of regulation, not the laws being regulated by."

This module implements:
1. Adaptation triggers (when to consider changing S₁)
2. Adaptation bounds (how much S₁ can change)
3. Pathology detection (when adaptation is making things worse)
4. Hard stops (when to freeze and alert)

Constitutional constraint (from BLI_CONSTITUTION Article II):
- S₂ may influence S₁ ✓
- S₁ may NOT influence S₀ ✗
- No process may modify the conditions under which it would have been forbidden
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import statistics


# =============================================================================
# Adaptation Verdicts
# =============================================================================

class AdaptationVerdict(Enum):
    """Result of adaptation consideration."""
    HOLD = auto()          # No change needed
    ADAPT = auto()         # Change within bounds
    FREEZE = auto()        # Stop adapting, something's wrong
    ALERT = auto()         # Human intervention needed


# =============================================================================
# S₁ Parameters (What Can Adapt)
# =============================================================================

@dataclass
class RegulatoryParameters:
    """
    S₁ - The regulatory layer that can adapt within bounds.
    
    Each parameter has:
    - current: Current value
    - floor: Constitutional minimum (cannot go below)
    - ceiling: Constitutional maximum (cannot go above)
    - step: Maximum change per adaptation epoch
    """
    
    # Budget levels
    repair_budget: float = 100.0
    repair_budget_floor: float = 10.0      # Can't starve repairs entirely
    repair_budget_ceiling: float = 500.0   # Can't have infinite budget
    repair_budget_step: float = 10.0       # Max 10% change per epoch
    
    # Refill rates
    refill_rate: float = 5.0
    refill_rate_floor: float = 1.0
    refill_rate_ceiling: float = 20.0
    refill_rate_step: float = 1.0
    
    # Contradiction thresholds
    glass_threshold: int = 20              # Open contradictions before GLASS regime
    glass_threshold_floor: int = 5
    glass_threshold_ceiling: int = 50
    glass_threshold_step: int = 2
    
    # Resolution cost
    resolution_cost: float = 10.0
    resolution_cost_floor: float = 1.0
    resolution_cost_ceiling: float = 50.0
    resolution_cost_step: float = 2.0
    
    # Timeouts
    claim_timeout_ms: int = 5000
    claim_timeout_floor: int = 1000
    claim_timeout_ceiling: int = 30000
    claim_timeout_step: int = 500
    
    def get(self, name: str) -> float:
        """Get current value of a parameter."""
        return getattr(self, name)
    
    def get_bounds(self, name: str) -> Tuple[float, float, float]:
        """Get (floor, ceiling, step) for a parameter."""
        return (
            getattr(self, f"{name}_floor"),
            getattr(self, f"{name}_ceiling"),
            getattr(self, f"{name}_step"),
        )
    
    def propose_change(self, name: str, delta: float) -> Tuple[float, bool]:
        """
        Propose a change to a parameter.
        
        Returns (new_value, was_clamped).
        Enforces floor/ceiling/step bounds.
        """
        current = self.get(name)
        floor, ceiling, step = self.get_bounds(name)
        
        # Clamp delta to step size
        clamped_delta = max(-step, min(step, delta))
        
        # Compute new value
        new_value = current + clamped_delta
        
        # Clamp to floor/ceiling
        was_clamped = False
        if new_value < floor:
            new_value = floor
            was_clamped = True
        elif new_value > ceiling:
            new_value = ceiling
            was_clamped = True
        
        return new_value, was_clamped
    
    def apply_change(self, name: str, new_value: float):
        """Apply a validated change."""
        setattr(self, name, new_value)


# =============================================================================
# Observation Window
# =============================================================================

@dataclass
class EpochObservation:
    """Observations from one adaptation epoch."""
    epoch_id: int
    start_time: datetime
    end_time: datetime
    
    # Counts
    turns: int = 0
    contradictions_opened: int = 0
    contradictions_closed: int = 0
    budget_blocks: int = 0
    violations: int = 0
    
    # Rates
    open_rate: float = 0.0      # contradictions opened per turn
    close_rate: float = 0.0     # contradictions closed per turn
    block_rate: float = 0.0     # budget blocks per turn
    violation_rate: float = 0.0
    
    # State at end
    c_open: int = 0
    regime: str = "UNKNOWN"
    
    def compute_rates(self):
        """Compute derived rates."""
        if self.turns > 0:
            self.open_rate = self.contradictions_opened / self.turns
            self.close_rate = self.contradictions_closed / self.turns
            self.block_rate = self.budget_blocks / self.turns
            self.violation_rate = self.violations / self.turns


@dataclass
class AdaptationHistory:
    """History of adaptations for pathology detection."""
    
    # Recent epochs
    epochs: List[EpochObservation] = field(default_factory=list)
    max_epochs: int = 10
    
    # Adaptation log
    adaptations: List[Dict[str, Any]] = field(default_factory=list)
    max_adaptations: int = 50
    
    # Pathology counters
    consecutive_failures: int = 0
    oscillation_count: int = 0
    
    def add_epoch(self, epoch: EpochObservation):
        """Add an epoch observation."""
        self.epochs.append(epoch)
        if len(self.epochs) > self.max_epochs:
            self.epochs.pop(0)
    
    def add_adaptation(self, adaptation: Dict[str, Any]):
        """Log an adaptation."""
        self.adaptations.append(adaptation)
        if len(self.adaptations) > self.max_adaptations:
            self.adaptations.pop(0)
    
    def get_trend(self, metric: str, window: int = 3) -> Optional[float]:
        """
        Get trend for a metric over recent epochs.
        
        Returns slope: positive = increasing, negative = decreasing.
        Returns None if insufficient data.
        """
        if len(self.epochs) < window:
            return None
        
        recent = self.epochs[-window:]
        values = [getattr(e, metric, 0) for e in recent]
        
        # Simple linear regression slope
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(values)
        
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator


# =============================================================================
# Adaptation Triggers
# =============================================================================

@dataclass
class AdaptationTrigger:
    """Conditions that trigger adaptation consideration."""
    
    # Starvation indicators
    block_rate_threshold: float = 0.3      # >30% turns blocked = too tight
    
    # Accumulation indicators
    c_open_threshold: int = 15             # >15 open = accumulating
    open_close_ratio_threshold: float = 1.2  # open_rate > close_rate * 1.2 = accumulating
    
    # Stability indicators
    min_epochs_for_trend: int = 3          # Need 3 epochs to see trend
    trend_significance: float = 0.1        # Slope must exceed this
    
    def should_consider_adaptation(
        self,
        current: EpochObservation,
        history: AdaptationHistory,
    ) -> Tuple[bool, str]:
        """
        Determine if adaptation should be considered.
        
        Returns (should_adapt, reason).
        """
        reasons = []
        
        # Check for starvation
        if current.block_rate > self.block_rate_threshold:
            reasons.append(f"block_rate={current.block_rate:.2f} > {self.block_rate_threshold}")
        
        # Check for accumulation
        if current.c_open > self.c_open_threshold:
            reasons.append(f"c_open={current.c_open} > {self.c_open_threshold}")
        
        if current.open_rate > current.close_rate * self.open_close_ratio_threshold:
            reasons.append(f"open_rate={current.open_rate:.2f} > close_rate*{self.open_close_ratio_threshold}={current.close_rate * self.open_close_ratio_threshold:.2f}")
        
        # Check trends
        c_open_trend = history.get_trend("c_open", self.min_epochs_for_trend)
        if c_open_trend and c_open_trend > self.trend_significance:
            reasons.append(f"c_open trending up (slope={c_open_trend:.3f})")
        
        block_trend = history.get_trend("block_rate", self.min_epochs_for_trend)
        if block_trend and block_trend > self.trend_significance:
            reasons.append(f"block_rate trending up (slope={block_trend:.3f})")
        
        if reasons:
            return True, "; ".join(reasons)
        
        return False, "stable"


# =============================================================================
# Pathology Detection
# =============================================================================

@dataclass
class PathologyDetector:
    """
    Detect when adaptation is making things worse.
    
    Pathologies:
    1. Oscillation: Parameters bouncing between values
    2. Runaway: Continuous adaptation in one direction hitting bounds
    3. Ineffective: Adapting but metrics not improving
    4. Wrong attractor: System stabilized but in bad state
    """
    
    # Thresholds
    oscillation_window: int = 6           # Check last 6 adaptations
    oscillation_threshold: int = 3         # 3+ reversals = oscillating
    
    runaway_threshold: int = 5             # 5 consecutive same-direction = runaway
    
    ineffective_epochs: int = 5            # If no improvement in 5 epochs
    
    wrong_attractor_regimes: List[str] = field(
        default_factory=lambda: ["GLASS_OSSIFICATION", "BUDGET_STARVATION", "EXTRACTION_COLLAPSE"]
    )
    wrong_attractor_epochs: int = 3        # Stuck in bad regime for 3 epochs
    
    def check_oscillation(self, history: AdaptationHistory) -> Tuple[bool, str]:
        """Check for parameter oscillation."""
        if len(history.adaptations) < self.oscillation_window:
            return False, ""
        
        recent = history.adaptations[-self.oscillation_window:]
        
        # Group by parameter
        by_param: Dict[str, List[float]] = {}
        for a in recent:
            param = a.get("parameter", "")
            delta = a.get("delta", 0)
            if param:
                by_param.setdefault(param, []).append(delta)
        
        # Check for sign reversals
        for param, deltas in by_param.items():
            if len(deltas) < 3:
                continue
            
            reversals = sum(
                1 for i in range(1, len(deltas))
                if deltas[i] * deltas[i-1] < 0  # Sign change
            )
            
            if reversals >= self.oscillation_threshold:
                return True, f"{param} oscillating ({reversals} reversals)"
        
        return False, ""
    
    def check_runaway(self, history: AdaptationHistory) -> Tuple[bool, str]:
        """Check for runaway adaptation (hitting bounds repeatedly)."""
        if len(history.adaptations) < self.runaway_threshold:
            return False, ""
        
        recent = history.adaptations[-self.runaway_threshold:]
        
        # Check if all recent adaptations were clamped in same direction
        clamped = [a for a in recent if a.get("clamped", False)]
        
        if len(clamped) >= self.runaway_threshold:
            param = clamped[0].get("parameter", "unknown")
            return True, f"{param} hitting bounds repeatedly"
        
        return False, ""
    
    def check_ineffective(self, history: AdaptationHistory) -> Tuple[bool, str]:
        """Check if adaptation isn't helping."""
        if len(history.epochs) < self.ineffective_epochs:
            return False, ""
        
        # Check if key metrics haven't improved
        recent = history.epochs[-self.ineffective_epochs:]
        
        # Were there adaptations?
        recent_adaptations = [
            a for a in history.adaptations
            if a.get("epoch_id", -1) >= recent[0].epoch_id
        ]
        
        if not recent_adaptations:
            return False, ""  # No adaptations to judge
        
        # Did c_open improve?
        c_open_start = recent[0].c_open
        c_open_end = recent[-1].c_open
        
        # Did block_rate improve?
        block_start = recent[0].block_rate
        block_end = recent[-1].block_rate
        
        # If neither improved despite adaptation
        if c_open_end >= c_open_start and block_end >= block_start:
            return True, f"adaptation not helping (c_open: {c_open_start}→{c_open_end}, block_rate: {block_start:.2f}→{block_end:.2f})"
        
        return False, ""
    
    def check_wrong_attractor(self, history: AdaptationHistory) -> Tuple[bool, str]:
        """Check if system stabilized in a bad regime."""
        if len(history.epochs) < self.wrong_attractor_epochs:
            return False, ""
        
        recent = history.epochs[-self.wrong_attractor_epochs:]
        regimes = [e.regime for e in recent]
        
        # All in same bad regime?
        if len(set(regimes)) == 1 and regimes[0] in self.wrong_attractor_regimes:
            return True, f"stuck in {regimes[0]} for {self.wrong_attractor_epochs} epochs"
        
        return False, ""
    
    def detect(self, history: AdaptationHistory) -> Tuple[bool, List[str]]:
        """
        Run all pathology checks.
        
        Returns (is_pathological, list_of_pathologies).
        """
        pathologies = []
        
        is_osc, msg = self.check_oscillation(history)
        if is_osc:
            pathologies.append(f"OSCILLATION: {msg}")
        
        is_run, msg = self.check_runaway(history)
        if is_run:
            pathologies.append(f"RUNAWAY: {msg}")
        
        is_ineff, msg = self.check_ineffective(history)
        if is_ineff:
            pathologies.append(f"INEFFECTIVE: {msg}")
        
        is_wrong, msg = self.check_wrong_attractor(history)
        if is_wrong:
            pathologies.append(f"WRONG_ATTRACTOR: {msg}")
        
        return len(pathologies) > 0, pathologies


# =============================================================================
# Adaptation Logic
# =============================================================================

@dataclass 
class AdaptationDecision:
    """Result of adaptation deliberation."""
    verdict: AdaptationVerdict
    parameter: Optional[str] = None
    current_value: Optional[float] = None
    proposed_value: Optional[float] = None
    reason: str = ""
    pathologies: List[str] = field(default_factory=list)


class UltrastabilityController:
    """
    Second-order controller for S₁ adaptation.
    
    Implements Ashby's ultrastability: the system can change its own
    parameters to find stability, but cannot change the rules about
    how it changes parameters.
    
    Constitutional constraints:
    - Cannot modify S₀ (NLAI, FSM, forbidden transitions)
    - Cannot exceed bounds on any S₁ parameter
    - Cannot adapt if pathology detected
    - Must log all adaptations for audit
    """
    
    def __init__(
        self,
        parameters: Optional[RegulatoryParameters] = None,
        trigger: Optional[AdaptationTrigger] = None,
        pathology_detector: Optional[PathologyDetector] = None,
    ):
        self.parameters = parameters or RegulatoryParameters()
        self.trigger = trigger or AdaptationTrigger()
        self.pathology_detector = pathology_detector or PathologyDetector()
        self.history = AdaptationHistory()
        
        # Freeze state
        self.frozen = False
        self.freeze_reason: Optional[str] = None
        
        # Epoch counter
        self.current_epoch = 0
    
    def observe_epoch(self, observation: EpochObservation):
        """Record observations from an epoch."""
        observation.epoch_id = self.current_epoch
        observation.compute_rates()
        self.history.add_epoch(observation)
    
    def consider_adaptation(self) -> AdaptationDecision:
        """
        Consider whether to adapt S₁ parameters.
        
        This is the main entry point for the ultrastability loop.
        """
        # Check if frozen
        if self.frozen:
            return AdaptationDecision(
                verdict=AdaptationVerdict.FREEZE,
                reason=f"frozen: {self.freeze_reason}",
            )
        
        # Need observations
        if not self.history.epochs:
            return AdaptationDecision(
                verdict=AdaptationVerdict.HOLD,
                reason="no observations yet",
            )
        
        current = self.history.epochs[-1]
        
        # Check for pathologies first
        is_pathological, pathologies = self.pathology_detector.detect(self.history)
        
        if is_pathological:
            # Freeze adaptation
            self.frozen = True
            self.freeze_reason = "; ".join(pathologies)
            
            return AdaptationDecision(
                verdict=AdaptationVerdict.ALERT,
                reason="pathology detected",
                pathologies=pathologies,
            )
        
        # Check if adaptation needed
        should_adapt, trigger_reason = self.trigger.should_consider_adaptation(
            current, self.history
        )
        
        if not should_adapt:
            return AdaptationDecision(
                verdict=AdaptationVerdict.HOLD,
                reason=trigger_reason,
            )
        
        # Determine which parameter to adapt and direction
        decision = self._select_adaptation(current, trigger_reason)
        
        return decision
    
    def _select_adaptation(
        self,
        current: EpochObservation,
        trigger_reason: str,
    ) -> AdaptationDecision:
        """Select which parameter to adapt and by how much."""
        
        # Strategy: address most pressing issue
        
        # If blocking too much → increase budget or reduce cost
        if current.block_rate > self.trigger.block_rate_threshold:
            # Try increasing repair budget first
            param = "repair_budget"
            delta = self.parameters.repair_budget_step
            
            new_val, clamped = self.parameters.propose_change(param, delta)
            
            if not clamped:
                return AdaptationDecision(
                    verdict=AdaptationVerdict.ADAPT,
                    parameter=param,
                    current_value=self.parameters.get(param),
                    proposed_value=new_val,
                    reason=f"block_rate={current.block_rate:.2f}, increasing {param}",
                )
            
            # Budget at ceiling, try reducing cost
            param = "resolution_cost"
            delta = -self.parameters.resolution_cost_step
            
            new_val, clamped = self.parameters.propose_change(param, delta)
            
            return AdaptationDecision(
                verdict=AdaptationVerdict.ADAPT,
                parameter=param,
                current_value=self.parameters.get(param),
                proposed_value=new_val,
                reason=f"block_rate={current.block_rate:.2f}, reducing {param}",
            )
        
        # If accumulating contradictions → increase refill rate or lower threshold
        if current.c_open > self.trigger.c_open_threshold:
            param = "refill_rate"
            delta = self.parameters.refill_rate_step
            
            new_val, clamped = self.parameters.propose_change(param, delta)
            
            return AdaptationDecision(
                verdict=AdaptationVerdict.ADAPT,
                parameter=param,
                current_value=self.parameters.get(param),
                proposed_value=new_val,
                reason=f"c_open={current.c_open}, increasing {param}",
            )
        
        # Default: hold
        return AdaptationDecision(
            verdict=AdaptationVerdict.HOLD,
            reason=f"triggered but no clear adaptation: {trigger_reason}",
        )
    
    def apply_adaptation(self, decision: AdaptationDecision) -> bool:
        """
        Apply an adaptation decision.
        
        Returns True if applied, False if rejected.
        """
        if decision.verdict != AdaptationVerdict.ADAPT:
            return False
        
        if decision.parameter is None or decision.proposed_value is None:
            return False
        
        # Apply the change
        self.parameters.apply_change(decision.parameter, decision.proposed_value)
        
        # Log it with full delta details
        floor, ceiling, step = self.parameters.get_bounds(decision.parameter)
        delta_requested = decision.proposed_value - decision.current_value
        
        self.history.add_adaptation({
            "epoch_id": self.current_epoch,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "parameter": decision.parameter,
            "old_value": decision.current_value,
            "new_value": decision.proposed_value,
            "delta_requested": delta_requested,
            "delta_applied": delta_requested,  # Same since propose_change already clamped
            "hit_step_limit": abs(delta_requested) >= step,
            "hit_floor": decision.proposed_value == floor,
            "hit_ceiling": decision.proposed_value == ceiling,
            "reason": decision.reason,
            "clamped": decision.proposed_value in [floor, ceiling],
        })
        
        return True
    
    def advance_epoch(self):
        """Move to next epoch."""
        self.current_epoch += 1
    
    def unfreeze(self, reason: str):
        """
        Manually unfreeze after human review.
        
        This is the only way to resume adaptation after pathology detection.
        """
        self.frozen = False
        self.freeze_reason = None
        self.history.consecutive_failures = 0
        self.history.oscillation_count = 0
        
        self.history.add_adaptation({
            "epoch_id": self.current_epoch,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "parameter": "_system",
            "action": "unfreeze",
            "reason": reason,
        })
    
    def get_state(self) -> Dict[str, Any]:
        """Get current ultrastability state for inspection."""
        return {
            "epoch": self.current_epoch,
            "frozen": self.frozen,
            "freeze_reason": self.freeze_reason,
            "parameters": {
                "repair_budget": self.parameters.repair_budget,
                "refill_rate": self.parameters.refill_rate,
                "glass_threshold": self.parameters.glass_threshold,
                "resolution_cost": self.parameters.resolution_cost,
            },
            "recent_epochs": len(self.history.epochs),
            "total_adaptations": len(self.history.adaptations),
        }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=== Ultrastability Controller Test ===\n")
    
    controller = UltrastabilityController()
    
    # Simulate epochs with increasing stress
    print("Simulating stressed system...\n")
    
    for i in range(8):
        # Create observation with increasing problems
        obs = EpochObservation(
            epoch_id=i,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            turns=100,
            contradictions_opened=10 + i * 3,
            contradictions_closed=8,
            budget_blocks=5 + i * 5,
            c_open=5 + i * 4,
            regime="HEALTHY_LATTICE" if i < 4 else "GLASS_OSSIFICATION",
        )
        
        controller.observe_epoch(obs)
        controller.advance_epoch()
        
        # Consider adaptation
        decision = controller.consider_adaptation()
        
        print(f"Epoch {i}: c_open={obs.c_open}, block_rate={obs.block_rate:.2f}")
        print(f"  Decision: {decision.verdict.name}")
        if decision.parameter:
            print(f"  Adapt: {decision.parameter} {decision.current_value} → {decision.proposed_value}")
        if decision.pathologies:
            print(f"  Pathologies: {decision.pathologies}")
        print()
        
        # Apply if adapting
        if decision.verdict == AdaptationVerdict.ADAPT:
            controller.apply_adaptation(decision)
    
    print("\n=== Final State ===")
    state = controller.get_state()
    for k, v in state.items():
        print(f"  {k}: {v}")
