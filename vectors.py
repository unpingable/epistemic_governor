"""
Vector-Based Epistemic Control

The governor doesn't decide what's true — it shapes which directions 
of motion are cheap enough to exist.

State as a vector, not a score:
- position = current epistemic state relative to accreted core
- motion = how the last step moved that position

Axes (externally measured, not text-derived):
- support: validated commitments added
- coherence: contradictions resolved vs introduced
- entropy: branching / narrative spread (negative = good)
- effort: retries, token burn, constraint friction (negative = good)

Control primitives:
- Projection: Remove forbidden direction components
- Backpressure: State-dependent gain that shrinks step size
- Hysteresis: Enter pressure fast, exit slowly

The TCP analogy:
- outputs = requests
- hard gates = admission control
- retries = retransmits
- furnace = congestion collapse

Additive increase when stable, multiplicative decrease when hot.

Usage:
    from epistemic_governor.vectors import (
        StateVector,
        VectorController,
        BackpressurePolicy,
    )
    
    controller = VectorController()
    
    # Record a step
    step = controller.record_step(
        support_delta=1,      # Added a supported fact
        coherence_delta=0,    # No contradictions
        entropy_delta=-0.5,   # Narrowed uncertainty
        effort_delta=0.2,     # Some work required
    )
    
    # Get current pressure and policy
    pressure = controller.pressure
    policy = controller.get_policy()
    
    # Apply backpressure to proposed action
    allowed_step = controller.apply_backpressure(proposed_step)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum, auto
from datetime import datetime
import math


# =============================================================================
# State Vector
# =============================================================================

@dataclass
class StateVector:
    """
    A point in epistemic state space.
    
    ALL AXES: Positive = Good (no double negatives)
    
    - support: validated commitments added (more = better)
    - coherence: consistency with core (higher = better)
    - focus: inverse entropy, concentration (higher = better, less spread)
    - efficiency: inverse effort (higher = better, less work)
    
    This eliminates the "negative = good" confusion.
    """
    support: float = 0.0      # Validated commitments (positive = good)
    coherence: float = 0.0    # Consistency with core (positive = good)
    focus: float = 0.0        # Inverse entropy (positive = good)
    efficiency: float = 0.0   # Inverse effort (positive = good)
    
    def __add__(self, other: "StateVector") -> "StateVector":
        return StateVector(
            support=self.support + other.support,
            coherence=self.coherence + other.coherence,
            focus=self.focus + other.focus,
            efficiency=self.efficiency + other.efficiency,
        )
    
    def __sub__(self, other: "StateVector") -> "StateVector":
        return StateVector(
            support=self.support - other.support,
            coherence=self.coherence - other.coherence,
            focus=self.focus - other.focus,
            efficiency=self.efficiency - other.efficiency,
        )
    
    def __mul__(self, scalar: float) -> "StateVector":
        return StateVector(
            support=self.support * scalar,
            coherence=self.coherence * scalar,
            focus=self.focus * scalar,
            efficiency=self.efficiency * scalar,
        )
    
    def __rmul__(self, scalar: float) -> "StateVector":
        return self.__mul__(scalar)
    
    @property
    def magnitude(self) -> float:
        """L2 norm of the vector."""
        return math.sqrt(
            self.support ** 2 +
            self.coherence ** 2 +
            self.focus ** 2 +
            self.efficiency ** 2
        )
    
    @property
    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.support, self.coherence, self.focus, self.efficiency)
    
    @property
    def as_array(self) -> List[float]:
        return [self.support, self.coherence, self.focus, self.efficiency]
    
    def dot(self, other: "StateVector") -> float:
        """Dot product."""
        return (
            self.support * other.support +
            self.coherence * other.coherence +
            self.focus * other.focus +
            self.efficiency * other.efficiency
        )
    
    def project_onto(self, basis: "StateVector") -> "StateVector":
        """Project this vector onto a basis vector."""
        if basis.magnitude == 0:
            return StateVector()
        scalar = self.dot(basis) / (basis.magnitude ** 2)
        return basis * scalar
    
    def remove_component(self, basis: "StateVector") -> "StateVector":
        """Remove the component along a basis vector."""
        return self - self.project_onto(basis)
    
    def normalize(self) -> "StateVector":
        """Return unit vector in same direction."""
        mag = self.magnitude
        if mag == 0:
            return StateVector()
        return self * (1.0 / mag)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "support": self.support,
            "coherence": self.coherence,
            "focus": self.focus,
            "efficiency": self.efficiency,
            "magnitude": self.magnitude,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "StateVector":
        return cls(
            support=d.get("support", 0),
            coherence=d.get("coherence", 0),
            focus=d.get("focus", 0),
            efficiency=d.get("efficiency", 0),
        )


# =============================================================================
# Basis Vectors (Forbidden Directions)
# =============================================================================

class ForbiddenDirection(Enum):
    """Named basis vectors for forbidden motion directions."""
    
    # Unsupported claims: moving without adding support
    UNSUPPORTED = auto()
    
    # Unfocused: spreading without grounding (low focus)
    UNFOCUSED = auto()
    
    # Inefficient: lots of work, little progress
    INEFFICIENT = auto()
    
    # Incoherent: contradicting existing commitments
    INCOHERENT = auto()


# Basis vectors for forbidden directions
# These define the subspace to project OUT of
FORBIDDEN_BASES: Dict[ForbiddenDirection, StateVector] = {
    # Unsupported: negative focus (spreading), zero support
    ForbiddenDirection.UNSUPPORTED: StateVector(
        support=0, coherence=0, focus=-1, efficiency=0
    ),
    
    # Unfocused: negative focus only
    ForbiddenDirection.UNFOCUSED: StateVector(
        support=0, coherence=0, focus=-1, efficiency=0
    ),
    
    # Inefficient: negative efficiency
    ForbiddenDirection.INEFFICIENT: StateVector(
        support=0, coherence=0, focus=0, efficiency=-1
    ),
    
    # Incoherent: negative coherence
    ForbiddenDirection.INCOHERENT: StateVector(
        support=0, coherence=-1, focus=0, efficiency=0
    ),
}


# =============================================================================
# Backpressure Policy
# =============================================================================

class PressureLevel(Enum):
    """Discrete pressure levels for policy selection."""
    NORMAL = auto()      # Full action space
    ELEVATED = auto()    # Restricted actions
    HIGH = auto()        # Minimal actions
    CRITICAL = auto()    # Emergency mode
    SHUTDOWN = auto()    # Only defer


@dataclass
class BackpressurePolicy:
    """
    Policy derived from current pressure level.
    
    Maps pressure to concrete constraints:
    - max_tokens: Token budget
    - temperature: Sampling temperature
    - allowed_actions: Permitted output types
    - max_retries: Retry budget before escalation
    """
    level: PressureLevel
    pressure: float
    
    max_tokens: int = 2000
    temperature: float = 0.7
    allowed_actions: List[str] = field(default_factory=lambda: [
        "answer", "plan", "question", "tool", "defer"
    ])
    max_retries: int = 3
    
    # Vector constraints
    forbidden_directions: List[ForbiddenDirection] = field(default_factory=list)
    step_gain: float = 1.0  # Multiplier for proposed steps
    
    def describe(self) -> str:
        return (
            f"{self.level.name} (p={self.pressure:.2f}): "
            f"tokens={self.max_tokens}, temp={self.temperature:.1f}, "
            f"actions={self.allowed_actions}, gain={self.step_gain:.2f}"
        )


# =============================================================================
# Vector Controller
# =============================================================================

class VectorController:
    """
    Vector-based epistemic control.
    
    Tracks state as vectors, applies backpressure, and shapes
    which directions of motion are cheap enough to exist.
    
    Key operations:
    - record_step(): Log a state transition
    - get_policy(): Get current backpressure policy
    - apply_backpressure(): Modify proposed step based on pressure
    - project_step(): Remove forbidden direction components
    
    Pressure dynamics (TCP-like):
    - Violations increase pressure (superlinear - first violation matters most)
    - Stable turns decrease pressure (leak)
    - Hysteresis prevents chatter (enter fast, exit slow, minimum dwell)
    """
    
    def __init__(
        self,
        # Pressure dynamics
        alpha: float = 0.9,          # Pressure decay (leak rate)
        beta: float = 0.4,           # First violation impact
        beta2: float = 0.15,         # Additional violations impact
        gamma: float = 0.1,          # Pressure decrease per stable turn
        
        # Pressure thresholds (enter)
        elevated_threshold: float = 0.3,
        high_threshold: float = 0.5,
        critical_threshold: float = 0.7,
        shutdown_threshold: float = 0.9,
        
        # Hysteresis
        exit_margin: float = 0.15,    # Must be this much below threshold to exit
        min_dwell_turns: int = 2,     # Minimum turns before level change
        
        # Furnace detection
        furnace_threshold: float = 3.0,
        furnace_min_magnitude: float = 0.5,  # Don't flag low-activity as furnace
    ):
        # Pressure dynamics
        self.alpha = alpha
        self.beta = beta
        self.beta2 = beta2
        self.gamma = gamma
        
        # Thresholds
        self.elevated_threshold = elevated_threshold
        self.high_threshold = high_threshold
        self.critical_threshold = critical_threshold
        self.shutdown_threshold = shutdown_threshold
        
        # Hysteresis
        self.exit_margin = exit_margin
        self.min_dwell_turns = min_dwell_turns
        
        # Furnace
        self.furnace_threshold = furnace_threshold
        self.furnace_min_magnitude = furnace_min_magnitude
        
        # State
        self.pressure: float = 0.0
        self.position: StateVector = StateVector()  # Current position
        self.velocity: StateVector = StateVector()  # Recent motion
        
        # Hysteresis state
        self._current_level: PressureLevel = PressureLevel.NORMAL
        self._turns_at_level: int = 0
        
        # History
        self.steps: List[StateVector] = []
        self.pressure_history: List[float] = []
        self.level_history: List[PressureLevel] = []
        
        # Furnace detection
        self.total_magnitude: float = 0.0
        self.net_displacement: float = 0.0
        self._recent_violations: int = 0
        self._recent_retries: int = 0
    
    # =========================================================================
    # Step Recording
    # =========================================================================
    
    def record_step(
        self,
        support_delta: float = 0,
        coherence_delta: float = 0,
        focus_delta: float = 0,      # Positive = more focused, less entropy
        efficiency_delta: float = 0,  # Positive = more efficient, less effort
        violations: int = 0,
        retries: int = 0,
        stable: bool = False,
    ) -> StateVector:
        """
        Record a state transition.
        
        All deltas: POSITIVE = GOOD (no sign confusion)
        
        Args:
            support_delta: Change in supported commitments
            coherence_delta: Change in coherence
            focus_delta: Change in focus (positive = narrowed uncertainty)
            efficiency_delta: Change in efficiency (positive = less work)
            violations: Number of constraint violations this step
            retries: Number of retries this step
            stable: Whether this was a stable turn
            
        Returns:
            The recorded step vector
        """
        # Create step vector (all positive = good)
        step = StateVector(
            support=support_delta,
            coherence=coherence_delta,
            focus=focus_delta,
            efficiency=efficiency_delta,
        )
        
        # Update position
        self.position = self.position + step
        self.velocity = step
        
        # Track for furnace detection
        self.total_magnitude += step.magnitude
        self.net_displacement = self.position.magnitude
        self._recent_violations += violations
        self._recent_retries += retries
        
        # Update pressure with SUPERLINEAR violation response
        # First violation is a big deal, subsequent ones add less
        # p_{t+1} = α * p_t + β * 𝟙[V>0] + β₂ * (V-1)₊ - γ * stable
        violation_pressure = 0.0
        if violations > 0:
            violation_pressure = self.beta + self.beta2 * max(0, violations - 1)
        
        # Retries also add pressure (but less than violations)
        retry_pressure = retries * 0.1
        
        self.pressure = (
            self.alpha * self.pressure +
            violation_pressure +
            retry_pressure -
            (self.gamma if stable else 0)
        )
        self.pressure = max(0, min(1, self.pressure))  # Clamp to [0, 1]
        
        # Update level with hysteresis
        self._update_level()
        
        # Store history
        self.steps.append(step)
        self.pressure_history.append(self.pressure)
        self.level_history.append(self._current_level)
        
        # Limit history size
        if len(self.steps) > 100:
            self.steps = self.steps[-100:]
            self.pressure_history = self.pressure_history[-100:]
            self.level_history = self.level_history[-100:]
        
        return step
    
    def _update_level(self):
        """Update pressure level with hysteresis."""
        # Calculate what level the raw pressure would indicate
        if self.pressure >= self.shutdown_threshold:
            raw_level = PressureLevel.SHUTDOWN
        elif self.pressure >= self.critical_threshold:
            raw_level = PressureLevel.CRITICAL
        elif self.pressure >= self.high_threshold:
            raw_level = PressureLevel.HIGH
        elif self.pressure >= self.elevated_threshold:
            raw_level = PressureLevel.ELEVATED
        else:
            raw_level = PressureLevel.NORMAL
        
        # Hysteresis logic
        if raw_level.value > self._current_level.value:
            # Escalating: enter fast
            self._current_level = raw_level
            self._turns_at_level = 0
        elif raw_level.value < self._current_level.value:
            # De-escalating: exit slow
            # Need to be below threshold by margin AND have dwelt long enough
            current_threshold = self._get_threshold_for_level(self._current_level)
            exit_threshold = current_threshold - self.exit_margin
            
            if self.pressure < exit_threshold and self._turns_at_level >= self.min_dwell_turns:
                # OK to de-escalate
                self._current_level = raw_level
                self._turns_at_level = 0
            else:
                # Stay at current level
                self._turns_at_level += 1
        else:
            # Same level
            self._turns_at_level += 1
    
    def _get_threshold_for_level(self, level: PressureLevel) -> float:
        """Get the entry threshold for a level."""
        thresholds = {
            PressureLevel.NORMAL: 0,
            PressureLevel.ELEVATED: self.elevated_threshold,
            PressureLevel.HIGH: self.high_threshold,
            PressureLevel.CRITICAL: self.critical_threshold,
            PressureLevel.SHUTDOWN: self.shutdown_threshold,
        }
        return thresholds.get(level, 0)
    
    def record_from_thermal(
        self,
        thermal_state: Any,
        claims_added: int = 0,
        claims_rejected: int = 0,
        contradictions: int = 0,
    ) -> StateVector:
        """
        Record step from thermal state (for integration with existing code).
        """
        # Map thermal signals to vector components (all positive = good)
        support_delta = claims_added
        coherence_delta = -contradictions  # Contradictions reduce coherence
        
        # Focus: hedging reduces focus (spreading uncertainty)
        hedge_count = getattr(thermal_state, 'hedge_count', 0)
        focus_delta = -hedge_count * 0.2  # Hedging = less focused
        
        # Efficiency: retries and blocks reduce efficiency
        retry_count = getattr(thermal_state, 'retry_count', 0)
        block_count = getattr(thermal_state, 'block_count', 0)
        efficiency_delta = -(retry_count * 0.5 + block_count * 0.3)
        
        violations = claims_rejected + contradictions
        stable = violations == 0 and claims_added > 0
        
        return self.record_step(
            support_delta=support_delta,
            coherence_delta=coherence_delta,
            focus_delta=focus_delta,
            efficiency_delta=efficiency_delta,
            violations=violations,
            retries=retry_count,
            stable=stable,
        )
    
    # =========================================================================
    # Policy Selection
    # =========================================================================
    
    def get_level(self) -> PressureLevel:
        """Get current pressure level (with hysteresis applied)."""
        return self._current_level
    
    def get_policy(self) -> BackpressurePolicy:
        """Get current backpressure policy."""
        level = self.get_level()
        
        if level == PressureLevel.NORMAL:
            return BackpressurePolicy(
                level=level,
                pressure=self.pressure,
                max_tokens=2000,
                temperature=0.7,
                allowed_actions=["answer", "plan", "question", "tool", "defer"],
                max_retries=3,
                forbidden_directions=[],
                step_gain=1.0,
            )
        
        elif level == PressureLevel.ELEVATED:
            return BackpressurePolicy(
                level=level,
                pressure=self.pressure,
                max_tokens=1500,
                temperature=0.5,
                allowed_actions=["plan", "question", "tool", "defer"],
                max_retries=2,
                forbidden_directions=[ForbiddenDirection.UNFOCUSED],
                step_gain=0.8,
            )
        
        elif level == PressureLevel.HIGH:
            return BackpressurePolicy(
                level=level,
                pressure=self.pressure,
                max_tokens=1000,
                temperature=0.3,
                allowed_actions=["question", "tool", "defer"],
                max_retries=1,
                forbidden_directions=[
                    ForbiddenDirection.UNFOCUSED,
                    ForbiddenDirection.UNSUPPORTED,
                ],
                step_gain=0.5,
            )
        
        elif level == PressureLevel.CRITICAL:
            return BackpressurePolicy(
                level=level,
                pressure=self.pressure,
                max_tokens=500,
                temperature=0.1,
                allowed_actions=["question", "defer"],
                max_retries=0,
                forbidden_directions=[
                    ForbiddenDirection.UNFOCUSED,
                    ForbiddenDirection.UNSUPPORTED,
                    ForbiddenDirection.INEFFICIENT,
                ],
                step_gain=0.3,
            )
        
        else:  # SHUTDOWN
            return BackpressurePolicy(
                level=level,
                pressure=self.pressure,
                max_tokens=100,
                temperature=0.0,
                allowed_actions=["defer"],
                max_retries=0,
                forbidden_directions=list(ForbiddenDirection),
                step_gain=0.1,
            )
    
    # =========================================================================
    # Backpressure Application
    # =========================================================================
    
    def apply_backpressure(self, step: StateVector) -> StateVector:
        """
        Apply backpressure to a proposed step.
        
        1. Remove forbidden direction components
        2. Scale by step gain
        
        Returns the allowed step.
        """
        policy = self.get_policy()
        
        # Remove forbidden directions
        result = step
        for direction in policy.forbidden_directions:
            basis = FORBIDDEN_BASES.get(direction)
            if basis:
                result = result.remove_component(basis)
        
        # Apply gain (shrink step under pressure)
        result = result * policy.step_gain
        
        return result
    
    def project_step(
        self,
        step: StateVector,
        allowed_plane: List[str] = None,
    ) -> StateVector:
        """
        Project step onto allowed motion plane.
        
        Args:
            step: Proposed step
            allowed_plane: List of allowed axes (default: ["support", "coherence"])
            
        Returns:
            Step with only allowed components
        """
        allowed_plane = allowed_plane or ["support", "coherence"]
        
        return StateVector(
            support=step.support if "support" in allowed_plane else 0,
            coherence=step.coherence if "coherence" in allowed_plane else 0,
            focus=step.focus if "focus" in allowed_plane else 0,
            efficiency=step.efficiency if "efficiency" in allowed_plane else 0,
        )
    
    # =========================================================================
    # Furnace Detection
    # =========================================================================
    
    @property
    def furnace_ratio(self) -> float:
        """
        Ratio of total work to net displacement.
        
        High ratio = lots of work, little progress = furnace
        
        Uses epsilon to avoid division by zero and not flag low-activity.
        """
        epsilon = 0.1
        if self.net_displacement <= epsilon:
            return self.total_magnitude / epsilon if self.total_magnitude > 0 else 0
        return self.total_magnitude / self.net_displacement
    
    @property
    def is_furnace(self) -> bool:
        """
        Are we in furnace mode? (High work, low progress)
        
        Triggers only if:
        - Furnace ratio exceeds threshold AND
        - Total magnitude is significant AND
        - There have been violations or retries
        """
        return (
            self.furnace_ratio > self.furnace_threshold and
            self.total_magnitude > self.furnace_min_magnitude and
            (self._recent_violations > 0 or self._recent_retries > 0)
        )
    
    # =========================================================================
    # State Queries
    # =========================================================================
    
    def displacement_from_origin(self) -> float:
        """Distance from origin (accreted core)."""
        return self.position.magnitude
    
    def recent_velocity(self, n: int = 5) -> StateVector:
        """Average velocity over recent steps."""
        if not self.steps:
            return StateVector()
        recent = self.steps[-n:]
        total = StateVector()
        for s in recent:
            total = total + s
        return total * (1.0 / len(recent))
    
    def direction(self) -> str:
        """Which way are we moving? (Dominant axis)"""
        v = self.recent_velocity()
        components = {
            "support": v.support,
            "coherence": v.coherence,
            "focus": v.focus,
            "efficiency": v.efficiency,
        }
        
        # Find dominant positive and negative
        max_pos = max(components.items(), key=lambda x: x[1])
        max_neg = min(components.items(), key=lambda x: x[1])
        
        if max_pos[1] > abs(max_neg[1]):
            return f"+{max_pos[0]}"
        elif abs(max_neg[1]) > max_pos[1]:
            return f"-{max_neg[0]}"
        else:
            return "stationary"
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of current state."""
        policy = self.get_policy()
        return {
            "position": self.position.to_dict(),
            "velocity": self.velocity.to_dict(),
            "pressure": self.pressure,
            "level": policy.level.name,
            "turns_at_level": self._turns_at_level,
            "policy": policy.describe(),
            "furnace_ratio": self.furnace_ratio,
            "is_furnace": self.is_furnace,
            "direction": self.direction(),
            "steps": len(self.steps),
        }
    
    def reset(self):
        """Reset to initial state."""
        self.pressure = 0.0
        self.position = StateVector()
        self.velocity = StateVector()
        self._current_level = PressureLevel.NORMAL
        self._turns_at_level = 0
        self.steps.clear()
        self.pressure_history.clear()
        self.level_history.clear()
        self.total_magnitude = 0.0
        self.net_displacement = 0.0
        self._recent_violations = 0
        self._recent_retries = 0


# =============================================================================
# Integration Helper
# =============================================================================

def create_vector_controller_from_thermal(thermal_state: Any) -> VectorController:
    """
    Create a VectorController initialized from existing thermal state.
    
    Maps thermal signals to initial pressure.
    """
    controller = VectorController()
    
    # Initialize pressure from thermal instability
    if hasattr(thermal_state, 'instability'):
        controller.pressure = min(thermal_state.instability, 1.0)
    
    # Initialize from furnace ratio if available
    if hasattr(thermal_state, 'furnace_ratio'):
        fr = thermal_state.furnace_ratio
        if fr and fr != float('inf'):
            controller.total_magnitude = fr
            controller.net_displacement = 1.0
    
    return controller


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=== Vector-Based Epistemic Control Demo ===\n")
    
    controller = VectorController()
    
    # Simulate a conversation with varying quality
    # All deltas: POSITIVE = GOOD
    scenarios = [
        # Good turn: added support, focused, efficient
        {"support": 1, "coherence": 0, "focus": 0.2, "efficiency": 0.1, "violations": 0, "stable": True},
        
        # Another good turn
        {"support": 1, "coherence": 0.5, "focus": 0.3, "efficiency": 0.2, "violations": 0, "stable": True},
        
        # Bad turn: rejected claims, unfocused, inefficient
        {"support": 0, "coherence": -0.5, "focus": -0.5, "efficiency": -0.8, "violations": 2, "stable": False},
        
        # More bad turns (pressure builds)
        {"support": 0, "coherence": 0, "focus": -0.3, "efficiency": -0.6, "violations": 1, "stable": False},
        {"support": 0, "coherence": -0.2, "focus": -0.4, "efficiency": -0.7, "violations": 1, "stable": False},
        
        # Recovery attempt
        {"support": 0.5, "coherence": 0.2, "focus": 0.1, "efficiency": 0.3, "violations": 0, "stable": True},
        
        # More recovery (hysteresis: level should stay elevated)
        {"support": 1, "coherence": 0.3, "focus": 0.2, "efficiency": 0.5, "violations": 0, "stable": True},
        
        # Sustained recovery (now level should drop)
        {"support": 1, "coherence": 0.4, "focus": 0.3, "efficiency": 0.6, "violations": 0, "stable": True},
    ]
    
    print("--- Simulating conversation ---\n")
    
    for i, s in enumerate(scenarios):
        step = controller.record_step(
            support_delta=s["support"],
            coherence_delta=s["coherence"],
            focus_delta=s["focus"],
            efficiency_delta=s["efficiency"],
            violations=s["violations"],
            stable=s["stable"],
        )
        
        policy = controller.get_policy()
        
        print(f"Turn {i+1}:")
        print(f"  Step: support={s['support']}, violations={s['violations']}")
        print(f"  Pressure: {controller.pressure:.3f} → {policy.level.name} (dwell: {controller._turns_at_level})")
        print(f"  Policy: {policy.allowed_actions}")
        print(f"  Position magnitude: {controller.position.magnitude:.2f}")
        print()
    
    print("=== Final State ===")
    summary = controller.summary()
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for k2, v2 in v.items():
                print(f"  {k2}: {v2:.3f}" if isinstance(v2, float) else f"  {k2}: {v2}")
        else:
            print(f"{k}: {v}")
    
    print("\n=== Backpressure Demo ===")
    print("\nProposed step with negative focus (bad direction):")
    bad_step = StateVector(support=0, coherence=0, focus=-1, efficiency=-0.5)
    print(f"  Original: {bad_step.to_dict()}")
    
    allowed = controller.apply_backpressure(bad_step)
    print(f"  After backpressure: {allowed.to_dict()}")
    print(f"  Magnitude reduction: {bad_step.magnitude:.2f} → {allowed.magnitude:.2f}")
    
    print("\n=== Hysteresis Demo ===")
    print("\nFast escalation, slow de-escalation:")
    
    ctrl2 = VectorController(min_dwell_turns=3)
    
    # Escalate fast
    ctrl2.record_step(violations=2)
    print(f"  After 2 violations: {ctrl2._current_level.name} (p={ctrl2.pressure:.2f})")
    
    # Try to de-escalate (should be blocked by hysteresis)
    for i in range(4):
        ctrl2.record_step(support_delta=1, stable=True)
        print(f"  After stable turn {i+1}: {ctrl2._current_level.name} (p={ctrl2.pressure:.2f}, dwell={ctrl2._turns_at_level})")
