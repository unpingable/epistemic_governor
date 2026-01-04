"""
Autopilot Finite State Machine

Manages autopilot mode transitions with monotonic escalation.

Modes:
- OFF: No autopilot active
- HOLD: Maintaining declared heading
- DEGRADED: Soft constraints being dropped
- ARBITRATING: Waiting for user decision
- DISENGAGED: Stopped due to invariant violation

Key principles:
1. Transitions are monotonic during session (can only escalate)
2. Recovery requires explicit stabilization criterion
3. Autopilot never edits content - only selects modes/constraints
4. All transitions are logged (flight recorder)

Usage:
    from epistemic_governor.autopilot_fsm import (
        AutopilotFSM,
        AutopilotMode,
        TransitionResult,
    )
    
    fsm = AutopilotFSM()
    fsm.engage(heading)
    
    # On soft constraint conflict
    result = fsm.escalate_to_degraded("brevity", "provenance conflict")
    
    # On hard conflict
    result = fsm.escalate_to_arbitrating(conflict_set, options)
    
    # On invariant violation
    result = fsm.disengage("provenance_violation")
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, Callable
from enum import Enum, auto
from datetime import datetime
import json

from .heading import Heading, HeadingValidator


# =============================================================================
# Autopilot Modes
# =============================================================================

class AutopilotMode(Enum):
    """
    Autopilot operating modes.
    
    Ordered by severity (can only escalate, not descend without stabilization).
    """
    OFF = 0           # No autopilot
    HOLD = 1          # Maintaining heading
    DEGRADED = 2      # Soft constraints being dropped
    ARBITRATING = 3   # Waiting for user decision
    DISENGAGED = 4    # Stopped (terminal for session)


# Which modes allow descent
DESCENDABLE_MODES: Set[AutopilotMode] = {
    AutopilotMode.HOLD,
    AutopilotMode.DEGRADED,
}

# Which modes are terminal
TERMINAL_MODES: Set[AutopilotMode] = {
    AutopilotMode.DISENGAGED,
}


# =============================================================================
# Constraint Classification
# =============================================================================

class ConstraintClass(Enum):
    """
    Classification of constraints.
    
    Determines how conflicts are handled.
    """
    # Never traded - breach triggers disengage
    INVARIANT = auto()
    
    # Tradeable only via user arbitration
    HARD = auto()
    
    # Autopilot may degrade automatically
    SOFT = auto()


@dataclass
class ConstraintSpec:
    """Specification of a constraint."""
    name: str
    constraint_class: ConstraintClass
    description: str = ""
    
    # For soft constraints: sacrifice order (lower = dropped first)
    sacrifice_order: int = 0


# Default constraint registry
DEFAULT_CONSTRAINTS: Dict[str, ConstraintSpec] = {
    # Invariants (never traded)
    "provenance": ConstraintSpec("provenance", ConstraintClass.INVARIANT, "Claims must have provenance"),
    "contradiction": ConstraintSpec("contradiction", ConstraintClass.INVARIANT, "No contradictions allowed"),
    "scope": ConstraintSpec("scope", ConstraintClass.INVARIANT, "Stay within declared scope"),
    "uncertainty_overflow": ConstraintSpec("uncertainty_overflow", ConstraintClass.INVARIANT, "Uncertainty budget not exceeded"),
    
    # Hard constraints (user arbitration)
    "completeness": ConstraintSpec("completeness", ConstraintClass.HARD, "Cover all required topics"),
    "detail_level": ConstraintSpec("detail_level", ConstraintClass.HARD, "Maintain requested detail"),
    "format_fidelity": ConstraintSpec("format_fidelity", ConstraintClass.HARD, "Match requested format"),
    
    # Soft constraints (autopilot may drop)
    "brevity": ConstraintSpec("brevity", ConstraintClass.SOFT, "Keep output concise", sacrifice_order=1),
    "tone": ConstraintSpec("tone", ConstraintClass.SOFT, "Maintain requested tone", sacrifice_order=2),
    "polish": ConstraintSpec("polish", ConstraintClass.SOFT, "Stylistic polish", sacrifice_order=3),
    "formatting": ConstraintSpec("formatting", ConstraintClass.SOFT, "Optional formatting", sacrifice_order=4),
}


# =============================================================================
# Transition Events
# =============================================================================

@dataclass
class TransitionEvent:
    """Record of a state transition."""
    timestamp: datetime
    from_mode: AutopilotMode
    to_mode: AutopilotMode
    reason: str
    
    # For DEGRADED
    dropped_constraints: List[str] = field(default_factory=list)
    
    # For ARBITRATING
    conflict_set: List[str] = field(default_factory=list)
    options_presented: List[str] = field(default_factory=list)
    
    # For DISENGAGED
    violation_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "from_mode": self.from_mode.name,
            "to_mode": self.to_mode.name,
            "reason": self.reason,
            "dropped_constraints": self.dropped_constraints,
            "conflict_set": self.conflict_set,
            "options_presented": self.options_presented,
            "violation_type": self.violation_type,
        }


@dataclass
class TransitionResult:
    """Result of a transition attempt."""
    success: bool
    new_mode: AutopilotMode
    reason: str
    event: Optional[TransitionEvent] = None


# =============================================================================
# Arbitration Options
# =============================================================================

@dataclass
class ArbitrationOption:
    """
    An option presented during arbitration.
    
    Options must be mechanically derived from conflict set.
    """
    option_id: str
    description: str
    
    # What this option does
    drops: List[str] = field(default_factory=list)  # Constraints to drop
    tightens: List[str] = field(default_factory=list)  # Constraints to tighten
    narrows_scope: Optional[str] = None  # New scope if narrowing
    
    # Always include disengage
    is_disengage: bool = False


def derive_arbitration_options(
    conflict_set: List[str],
    constraints: Dict[str, ConstraintSpec],
) -> List[ArbitrationOption]:
    """
    Mechanically derive arbitration options from conflict set.
    
    This is where options must come from - not hand-wavy suggestions.
    """
    options = []
    
    # For each constraint in conflict, offer to drop it
    for constraint_name in conflict_set:
        spec = constraints.get(constraint_name)
        if spec and spec.constraint_class == ConstraintClass.HARD:
            options.append(ArbitrationOption(
                option_id=f"drop_{constraint_name}",
                description=f"Drop {constraint_name} to preserve other constraints",
                drops=[constraint_name],
            ))
    
    # If scope-related, offer to narrow
    if "completeness" in conflict_set:
        options.append(ArbitrationOption(
            option_id="narrow_scope",
            description="Narrow scope to reduce completeness requirement",
            narrows_scope="user_specified",
        ))
    
    # Always include disengage
    options.append(ArbitrationOption(
        option_id="disengage",
        description="Stop and disengage autopilot",
        is_disengage=True,
    ))
    
    return options


# =============================================================================
# Stabilization Report (for structured recovery)
# =============================================================================

@dataclass
class StabilizationReport:
    """
    Structured report for stabilization state.
    
    Required for FSM recovery attempts.
    """
    is_stable: bool
    reason: str
    
    # Metrics that contributed to decision
    slack_increasing: bool = False
    drops_decaying: bool = False
    car_stable: bool = False
    no_near_violations: bool = False
    
    # Thresholds used
    required_stable_windows: int = 5
    stable_window_count: int = 0
    clear_step_count: int = 0


# =============================================================================
# Autopilot FSM
# =============================================================================

class AutopilotFSM:
    """
    Finite State Machine for autopilot control.
    
    Key invariants:
    1. Transitions are monotonic (can only escalate)
    2. Recovery requires stabilization criterion
    3. All transitions logged
    """
    
    def __init__(self, constraints: Dict[str, ConstraintSpec] = None):
        self.constraints = constraints or DEFAULT_CONSTRAINTS
        
        # State
        self.mode = AutopilotMode.OFF
        self.heading: Optional[Heading] = None
        self.validator: Optional[HeadingValidator] = None
        
        # Dropped constraints (for DEGRADED mode)
        self.dropped_constraints: Set[str] = set()
        
        # Arbitration state
        self.pending_arbitration: Optional[List[ArbitrationOption]] = None
        self.arbitration_conflict: Optional[List[str]] = None
        
        # History (flight recorder)
        self.transition_history: List[TransitionEvent] = []
        
        # Stabilization tracking
        self.stable_since: Optional[datetime] = None
        self.stable_steps: int = 0
        
        # Breach tracking (prevents silent re-engage after invariant breach)
        self._last_disengage_was_breach: bool = False
        self._breach_acknowledged: bool = False
    
    # =========================================================================
    # Engagement
    # =========================================================================
    
    def engage(self, heading: Heading) -> TransitionResult:
        """
        Engage autopilot with a heading.
        
        Only allowed from OFF mode.
        After invariant breach, requires acknowledgment or different heading.
        """
        if self.mode != AutopilotMode.OFF:
            return TransitionResult(
                success=False,
                new_mode=self.mode,
                reason=f"Cannot engage from {self.mode.name}",
            )
        
        # Check if last disengage was from invariant breach
        if self._last_disengage_was_breach and not self._breach_acknowledged:
            # Require different heading or explicit acknowledgment
            if self.heading and heading.heading_type == self.heading.heading_type:
                return TransitionResult(
                    success=False,
                    new_mode=self.mode,
                    reason="Previous session ended in invariant breach. Use different heading or acknowledge_breach()",
                )
        
        # Validate heading is autopilot-eligible
        validator = HeadingValidator(heading)
        eligible, reason = validator.check_autopilot_eligible()
        
        if not eligible:
            return TransitionResult(
                success=False,
                new_mode=self.mode,
                reason=f"Heading not eligible: {reason}",
            )
        
        # Engage
        self.heading = heading
        self.validator = validator
        self._last_disengage_was_breach = False
        self._breach_acknowledged = False
        self._transition_to(AutopilotMode.HOLD, "engaged")
        
        return TransitionResult(
            success=True,
            new_mode=self.mode,
            reason="Autopilot engaged",
            event=self.transition_history[-1],
        )
    
    def acknowledge_breach(self):
        """
        Explicitly acknowledge prior invariant breach.
        
        Required to re-engage with same heading type after breach.
        """
        self._breach_acknowledged = True
    
    def manual_disengage(self) -> TransitionResult:
        """
        User-initiated disengage.
        
        Always allowed (human authority).
        Note: After invariant-triggered DISENGAGED, re-engage requires
        explicit acknowledgment or heading change.
        """
        if self.mode == AutopilotMode.OFF:
            return TransitionResult(
                success=True,
                new_mode=self.mode,
                reason="Already off",
            )
        
        # Track if we're coming from invariant breach
        from_invariant_breach = self.mode == AutopilotMode.DISENGAGED
        
        self._transition_to(AutopilotMode.OFF, "user_disengage")
        self._reset_state()
        
        # Set flag if coming from invariant breach
        if from_invariant_breach:
            self._last_disengage_was_breach = True
        
        return TransitionResult(
            success=True,
            new_mode=self.mode,
            reason="User disengaged" + (" (after breach)" if from_invariant_breach else ""),
            event=self.transition_history[-1],
        )
    
    # =========================================================================
    # Escalation (monotonic)
    # =========================================================================
    
    def escalate_to_degraded(
        self,
        constraint_to_drop: str,
        reason: str,
    ) -> TransitionResult:
        """
        Escalate to DEGRADED mode by dropping a soft constraint.
        
        Only soft constraints can be auto-dropped.
        """
        # Check current mode allows this
        if self.mode not in {AutopilotMode.HOLD, AutopilotMode.DEGRADED}:
            return TransitionResult(
                success=False,
                new_mode=self.mode,
                reason=f"Cannot degrade from {self.mode.name}",
            )
        
        # Check constraint is soft
        spec = self.constraints.get(constraint_to_drop)
        if not spec or spec.constraint_class != ConstraintClass.SOFT:
            return TransitionResult(
                success=False,
                new_mode=self.mode,
                reason=f"Cannot auto-drop non-soft constraint: {constraint_to_drop}",
            )
        
        # Drop it
        self.dropped_constraints.add(constraint_to_drop)
        
        # Transition if not already degraded
        if self.mode != AutopilotMode.DEGRADED:
            self._transition_to(
                AutopilotMode.DEGRADED,
                reason,
                dropped=[constraint_to_drop],
            )
        else:
            # Log the additional drop
            self._log_event(TransitionEvent(
                timestamp=datetime.now(),
                from_mode=self.mode,
                to_mode=self.mode,
                reason=reason,
                dropped_constraints=[constraint_to_drop],
            ))
        
        return TransitionResult(
            success=True,
            new_mode=self.mode,
            reason=f"Dropped {constraint_to_drop}",
            event=self.transition_history[-1],
        )
    
    def escalate_to_arbitrating(
        self,
        conflict_set: List[str],
    ) -> TransitionResult:
        """
        Escalate to ARBITRATING mode due to hard constraint conflict.
        
        Presents mechanically-derived options to user.
        """
        # Check current mode allows this
        if self.mode in TERMINAL_MODES:
            return TransitionResult(
                success=False,
                new_mode=self.mode,
                reason=f"Cannot arbitrate from {self.mode.name}",
            )
        
        # Derive options
        options = derive_arbitration_options(conflict_set, self.constraints)
        
        self.arbitration_conflict = conflict_set
        self.pending_arbitration = options
        
        self._transition_to(
            AutopilotMode.ARBITRATING,
            f"constraint_conflict: {', '.join(conflict_set)}",
            conflict=conflict_set,
            options=[o.option_id for o in options],
        )
        
        return TransitionResult(
            success=True,
            new_mode=self.mode,
            reason="Awaiting arbitration",
            event=self.transition_history[-1],
        )
    
    def resolve_arbitration(self, option_id: str) -> TransitionResult:
        """
        Resolve pending arbitration with user choice.
        """
        if self.mode != AutopilotMode.ARBITRATING:
            return TransitionResult(
                success=False,
                new_mode=self.mode,
                reason="Not in arbitration",
            )
        
        if not self.pending_arbitration:
            return TransitionResult(
                success=False,
                new_mode=self.mode,
                reason="No pending arbitration",
            )
        
        # Find selected option
        selected = None
        for opt in self.pending_arbitration:
            if opt.option_id == option_id:
                selected = opt
                break
        
        if not selected:
            return TransitionResult(
                success=False,
                new_mode=self.mode,
                reason=f"Invalid option: {option_id}",
            )
        
        # Handle option
        if selected.is_disengage:
            return self.disengage("user_chose_disengage")
        
        # Apply drops
        for constraint in selected.drops:
            self.dropped_constraints.add(constraint)
        
        # Clear arbitration state
        self.pending_arbitration = None
        self.arbitration_conflict = None
        
        # Go to DEGRADED (or stay there)
        self._transition_to(
            AutopilotMode.DEGRADED,
            f"arbitration_resolved: {option_id}",
            dropped=selected.drops,
        )
        
        return TransitionResult(
            success=True,
            new_mode=self.mode,
            reason=f"Resolved with {option_id}",
            event=self.transition_history[-1],
        )
    
    def disengage(self, reason: str) -> TransitionResult:
        """
        Disengage due to invariant violation (Level 3).
        
        Terminal state for this session.
        """
        self._transition_to(
            AutopilotMode.DISENGAGED,
            reason,
            violation=reason,
        )
        
        return TransitionResult(
            success=True,
            new_mode=self.mode,
            reason=f"Disengaged: {reason}",
            event=self.transition_history[-1],
        )
    
    # =========================================================================
    # Recovery (requires stabilization)
    # =========================================================================
    
    def attempt_recovery(self, stabilization_report: "StabilizationReport") -> TransitionResult:
        """
        Attempt to recover to a lower mode.
        
        Requires structured stabilization report, not just a boolean.
        """
        if self.mode not in DESCENDABLE_MODES:
            return TransitionResult(
                success=False,
                new_mode=self.mode,
                reason=f"Cannot recover from {self.mode.name}",
            )
        
        if not stabilization_report.is_stable:
            self.stable_steps = 0
            return TransitionResult(
                success=False,
                new_mode=self.mode,
                reason=f"Not stabilized: {stabilization_report.reason}",
            )
        
        self.stable_steps += 1
        
        # Need sustained stability
        required_steps = stabilization_report.required_stable_windows
        if self.stable_steps < required_steps:
            return TransitionResult(
                success=False,
                new_mode=self.mode,
                reason=f"Stabilizing ({self.stable_steps}/{required_steps})",
            )
        
        # Recover
        if self.mode == AutopilotMode.DEGRADED and not self.dropped_constraints:
            # Can return to HOLD
            self._transition_to(AutopilotMode.HOLD, "stabilized_recovery")
            self.stable_steps = 0
            return TransitionResult(
                success=True,
                new_mode=self.mode,
                reason="Recovered to HOLD",
                event=self.transition_history[-1],
            )
        
        return TransitionResult(
            success=False,
            new_mode=self.mode,
            reason="Recovery conditions not met",
        )
    
    # =========================================================================
    # Queries
    # =========================================================================
    
    def get_active_constraints(self) -> Set[str]:
        """Get constraints that are still active (not dropped)."""
        all_constraints = set(self.constraints.keys())
        return all_constraints - self.dropped_constraints
    
    def get_soft_by_sacrifice_order(self) -> List[str]:
        """Get soft constraints ordered by sacrifice priority."""
        soft = [
            (name, spec) for name, spec in self.constraints.items()
            if spec.constraint_class == ConstraintClass.SOFT
            and name not in self.dropped_constraints
        ]
        soft.sort(key=lambda x: x[1].sacrifice_order)
        return [name for name, _ in soft]
    
    def is_active(self) -> bool:
        """Is autopilot currently active?"""
        return self.mode in {AutopilotMode.HOLD, AutopilotMode.DEGRADED}
    
    def needs_user_input(self) -> bool:
        """Does autopilot need user input?"""
        return self.mode == AutopilotMode.ARBITRATING
    
    def get_arbitration_options(self) -> Optional[List[ArbitrationOption]]:
        """Get pending arbitration options."""
        return self.pending_arbitration
    
    # =========================================================================
    # Internal
    # =========================================================================
    
    def _transition_to(
        self,
        new_mode: AutopilotMode,
        reason: str,
        dropped: List[str] = None,
        conflict: List[str] = None,
        options: List[str] = None,
        violation: str = None,
    ):
        """Internal transition with logging."""
        event = TransitionEvent(
            timestamp=datetime.now(),
            from_mode=self.mode,
            to_mode=new_mode,
            reason=reason,
            dropped_constraints=dropped or [],
            conflict_set=conflict or [],
            options_presented=options or [],
            violation_type=violation,
        )
        
        self._log_event(event)
        self.mode = new_mode
    
    def _log_event(self, event: TransitionEvent):
        """Log a transition event."""
        self.transition_history.append(event)
    
    def _reset_state(self):
        """Reset internal state."""
        self.heading = None
        self.validator = None
        self.dropped_constraints = set()
        self.pending_arbitration = None
        self.arbitration_conflict = None
        self.stable_steps = 0
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get current state as dictionary."""
        return {
            "mode": self.mode.name,
            "heading_type": self.heading.heading_type.name if self.heading else None,
            "dropped_constraints": list(self.dropped_constraints),
            "has_pending_arbitration": self.pending_arbitration is not None,
            "transition_count": len(self.transition_history),
        }
    
    def get_flight_recorder(self) -> List[Dict[str, Any]]:
        """Get full transition history (flight recorder)."""
        return [e.to_dict() for e in self.transition_history]


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=== Autopilot FSM Demo ===\n")
    
    from .heading import SummarizeHeading
    
    fsm = AutopilotFSM()
    
    # Create heading
    heading = SummarizeHeading(
        source_id="doc_001",
        target_length=200,
    )
    
    # Engage
    print("1. Engaging autopilot...")
    result = fsm.engage(heading)
    print(f"   Result: {result.success}, Mode: {result.new_mode.name}")
    
    # Simulate soft constraint conflict
    print("\n2. Soft constraint conflict (dropping brevity)...")
    result = fsm.escalate_to_degraded("brevity", "provenance_conflict")
    print(f"   Result: {result.success}, Mode: {result.new_mode.name}")
    print(f"   Dropped: {fsm.dropped_constraints}")
    
    # Simulate hard constraint conflict
    print("\n3. Hard constraint conflict...")
    result = fsm.escalate_to_arbitrating(["completeness", "detail_level"])
    print(f"   Result: {result.success}, Mode: {result.new_mode.name}")
    print(f"   Options: {[o.option_id for o in fsm.get_arbitration_options()]}")
    
    # Resolve arbitration
    print("\n4. Resolving arbitration...")
    result = fsm.resolve_arbitration("drop_completeness")
    print(f"   Result: {result.success}, Mode: {result.new_mode.name}")
    
    # Flight recorder
    print("\n--- Flight Recorder ---")
    for event in fsm.get_flight_recorder():
        print(f"  {event['from_mode']} → {event['to_mode']}: {event['reason']}")
