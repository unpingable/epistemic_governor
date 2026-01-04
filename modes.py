"""
Epistemic Mode Isolation

Hard phase separation for epistemic operations.
Without this, curiosity contaminates humor, humor papers over incoherence,
adversarial triggers safety, etc. This is the clean lab bench.

Modes:
- DIAGNOSTIC: Default. Descriptive only, no suggestions.
- CURIOSITY: Budgeted probing. One question, then stop.
- ADVERSARIAL: Must steelman against prior frame.
- HUMOR: Controlled entropy injection. Single discharge.
- FORMAL: Symbols/constraints only, no narrative voice.

Rules:
- Only one mode active at a time
- Transitions require reason codes
- Some transitions forbidden (e.g., CURIOSITY -> HUMOR same turn)
- On overheat: force DIAGNOSTIC or FORMAL, never HUMOR

Usage:
    from epistemic_governor.modes import ModeController, Mode
    
    controller = ModeController()
    controller.request_transition(Mode.CURIOSITY, reason="entropy_spike")
    
    if controller.current_mode == Mode.CURIOSITY:
        # Curiosity-specific constraints apply
        pass
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Set, List, Tuple, Callable
from enum import Enum, auto


# =============================================================================
# Mode Definitions
# =============================================================================

class Mode(Enum):
    """
    Epistemic operating modes.
    
    Each mode has mutually exclusive linguistic affordances.
    Style can't survive a basis change intact.
    """
    DIAGNOSTIC = auto()   # Default. Descriptive only, no suggestions.
    CURIOSITY = auto()    # Budgeted probing. One question, then stop.
    ADVERSARIAL = auto()  # Must steelman against prior frame.
    HUMOR = auto()        # Controlled entropy injection. Single discharge.
    FORMAL = auto()       # Symbols/constraints only, no narrative voice.
    
    def __str__(self):
        return self.name.lower()


class TransitionReason(Enum):
    """Reasons for mode transitions (must be explicit)."""
    # Automatic triggers
    ENTROPY_SPIKE = auto()          # High uncertainty detected
    CONSTRAINT_VIOLATION = auto()   # Inconsistency found
    HYSTERESIS_SIGNAL = auto()      # System worked too hard to settle
    THERMAL_OVERHEAT = auto()       # Latency/hedging spike
    LOW_CONF_HIGH_IMPACT = auto()   # Important but uncertain
    
    # User-initiated
    USER_REQUEST = auto()           # Explicit mode request
    
    # System-initiated
    BUDGET_EXHAUSTED = auto()       # Curiosity/humor budget depleted
    TASK_COMPLETE = auto()          # Mode objective achieved
    COOLDOWN_EXPIRED = auto()       # Mandatory wait period over
    SAFETY_OVERRIDE = auto()        # Safety system forced transition
    
    # Fallback
    DEFAULT_RESET = auto()          # Return to diagnostic


# =============================================================================
# Transition Rules
# =============================================================================

@dataclass
class TransitionRule:
    """
    Rule governing a mode transition.
    
    Transitions are not free - they have costs and constraints.
    """
    from_mode: Mode
    to_mode: Mode
    allowed: bool = True
    cooldown_seconds: float = 0.0      # Minimum time before this transition
    requires_reasons: Set[TransitionReason] = field(default_factory=set)
    forbidden_reasons: Set[TransitionReason] = field(default_factory=set)
    cost: float = 0.0                   # Transition cost (for budgeting)
    
    def permits(self, reason: TransitionReason, time_since_last: float) -> Tuple[bool, str]:
        """Check if this transition is permitted."""
        if not self.allowed:
            return False, f"Transition {self.from_mode} -> {self.to_mode} is forbidden"
        
        if time_since_last < self.cooldown_seconds:
            remaining = self.cooldown_seconds - time_since_last
            return False, f"Cooldown: {remaining:.1f}s remaining"
        
        if self.requires_reasons and reason not in self.requires_reasons:
            return False, f"Reason {reason} not in allowed set: {self.requires_reasons}"
        
        if reason in self.forbidden_reasons:
            return False, f"Reason {reason} is forbidden for this transition"
        
        return True, "OK"


# Default transition matrix
def build_transition_rules() -> Dict[Tuple[Mode, Mode], TransitionRule]:
    """
    Build the transition rule matrix.
    
    Key constraints:
    - CURIOSITY -> HUMOR same turn: forbidden (curiosity must complete)
    - HUMOR -> CURIOSITY same turn: forbidden (humor must discharge)
    - On overheat: only DIAGNOSTIC or FORMAL allowed
    - ADVERSARIAL requires explicit reason
    """
    rules = {}
    
    # From DIAGNOSTIC (default) - most transitions allowed
    rules[(Mode.DIAGNOSTIC, Mode.CURIOSITY)] = TransitionRule(
        Mode.DIAGNOSTIC, Mode.CURIOSITY,
        requires_reasons={
            TransitionReason.ENTROPY_SPIKE,
            TransitionReason.CONSTRAINT_VIOLATION,
            TransitionReason.HYSTERESIS_SIGNAL,
            TransitionReason.LOW_CONF_HIGH_IMPACT,
        }
    )
    rules[(Mode.DIAGNOSTIC, Mode.ADVERSARIAL)] = TransitionRule(
        Mode.DIAGNOSTIC, Mode.ADVERSARIAL,
        requires_reasons={TransitionReason.USER_REQUEST},
        cooldown_seconds=5.0,
    )
    rules[(Mode.DIAGNOSTIC, Mode.HUMOR)] = TransitionRule(
        Mode.DIAGNOSTIC, Mode.HUMOR,
        requires_reasons={TransitionReason.USER_REQUEST, TransitionReason.TASK_COMPLETE},
        cooldown_seconds=10.0,
    )
    rules[(Mode.DIAGNOSTIC, Mode.FORMAL)] = TransitionRule(
        Mode.DIAGNOSTIC, Mode.FORMAL,
        cost=0.0,  # Always cheap to go formal
    )
    
    # From CURIOSITY - very restricted
    rules[(Mode.CURIOSITY, Mode.DIAGNOSTIC)] = TransitionRule(
        Mode.CURIOSITY, Mode.DIAGNOSTIC,
        requires_reasons={
            TransitionReason.BUDGET_EXHAUSTED,
            TransitionReason.TASK_COMPLETE,
            TransitionReason.THERMAL_OVERHEAT,
        }
    )
    rules[(Mode.CURIOSITY, Mode.HUMOR)] = TransitionRule(
        Mode.CURIOSITY, Mode.HUMOR,
        allowed=False,  # FORBIDDEN: curiosity must complete first
    )
    rules[(Mode.CURIOSITY, Mode.ADVERSARIAL)] = TransitionRule(
        Mode.CURIOSITY, Mode.ADVERSARIAL,
        allowed=False,  # Can't switch mid-probe
    )
    rules[(Mode.CURIOSITY, Mode.FORMAL)] = TransitionRule(
        Mode.CURIOSITY, Mode.FORMAL,
        requires_reasons={TransitionReason.THERMAL_OVERHEAT},
    )
    
    # From HUMOR - very restricted
    rules[(Mode.HUMOR, Mode.DIAGNOSTIC)] = TransitionRule(
        Mode.HUMOR, Mode.DIAGNOSTIC,
        requires_reasons={
            TransitionReason.BUDGET_EXHAUSTED,
            TransitionReason.TASK_COMPLETE,
        }
    )
    rules[(Mode.HUMOR, Mode.CURIOSITY)] = TransitionRule(
        Mode.HUMOR, Mode.CURIOSITY,
        allowed=False,  # FORBIDDEN: humor must discharge first
    )
    rules[(Mode.HUMOR, Mode.ADVERSARIAL)] = TransitionRule(
        Mode.HUMOR, Mode.ADVERSARIAL,
        allowed=False,  # Can't mix
    )
    rules[(Mode.HUMOR, Mode.FORMAL)] = TransitionRule(
        Mode.HUMOR, Mode.FORMAL,
        requires_reasons={TransitionReason.THERMAL_OVERHEAT, TransitionReason.SAFETY_OVERRIDE},
    )
    
    # From ADVERSARIAL
    rules[(Mode.ADVERSARIAL, Mode.DIAGNOSTIC)] = TransitionRule(
        Mode.ADVERSARIAL, Mode.DIAGNOSTIC,
        requires_reasons={
            TransitionReason.TASK_COMPLETE,
            TransitionReason.USER_REQUEST,
            TransitionReason.THERMAL_OVERHEAT,
        }
    )
    rules[(Mode.ADVERSARIAL, Mode.CURIOSITY)] = TransitionRule(
        Mode.ADVERSARIAL, Mode.CURIOSITY,
        cooldown_seconds=10.0,
    )
    rules[(Mode.ADVERSARIAL, Mode.HUMOR)] = TransitionRule(
        Mode.ADVERSARIAL, Mode.HUMOR,
        allowed=False,  # Adversarial -> Humor is dangerous
    )
    rules[(Mode.ADVERSARIAL, Mode.FORMAL)] = TransitionRule(
        Mode.ADVERSARIAL, Mode.FORMAL,
        cost=0.0,
    )
    
    # From FORMAL - easy to leave
    rules[(Mode.FORMAL, Mode.DIAGNOSTIC)] = TransitionRule(
        Mode.FORMAL, Mode.DIAGNOSTIC,
    )
    rules[(Mode.FORMAL, Mode.CURIOSITY)] = TransitionRule(
        Mode.FORMAL, Mode.CURIOSITY,
        requires_reasons={
            TransitionReason.ENTROPY_SPIKE,
            TransitionReason.CONSTRAINT_VIOLATION,
        }
    )
    rules[(Mode.FORMAL, Mode.ADVERSARIAL)] = TransitionRule(
        Mode.FORMAL, Mode.ADVERSARIAL,
        requires_reasons={TransitionReason.USER_REQUEST},
    )
    rules[(Mode.FORMAL, Mode.HUMOR)] = TransitionRule(
        Mode.FORMAL, Mode.HUMOR,
        cooldown_seconds=30.0,  # Long cooldown from formal to humor
    )
    
    return rules


# =============================================================================
# Mode State
# =============================================================================

@dataclass
class ModeState:
    """Current mode state with history."""
    current: Mode = Mode.DIAGNOSTIC
    entered_at: datetime = field(default_factory=datetime.now)
    reason: TransitionReason = TransitionReason.DEFAULT_RESET
    
    # Budgets (reset on user turn)
    curiosity_questions_remaining: int = 1
    humor_discharges_remaining: int = 1
    adversarial_turns_remaining: int = 3
    
    # Thermal
    is_overheated: bool = False
    overheat_until: Optional[datetime] = None
    
    # History
    transitions: List[Tuple[datetime, Mode, Mode, TransitionReason]] = field(default_factory=list)
    
    def time_in_current_mode(self) -> float:
        """Seconds spent in current mode."""
        return (datetime.now() - self.entered_at).total_seconds()
    
    def record_transition(self, from_mode: Mode, to_mode: Mode, reason: TransitionReason):
        """Record a transition in history."""
        self.transitions.append((datetime.now(), from_mode, to_mode, reason))
        # Keep only last 100 transitions
        if len(self.transitions) > 100:
            self.transitions = self.transitions[-100:]


# =============================================================================
# Mode Controller
# =============================================================================

class ModeController:
    """
    Controls mode transitions with hard enforcement.
    
    This is the "phase separation" layer - ensures modes don't contaminate
    each other and transitions follow rules.
    """
    
    def __init__(self):
        self.state = ModeState()
        self.rules = build_transition_rules()
        self._last_user_msg_id: Optional[str] = None
    
    @property
    def current_mode(self) -> Mode:
        return self.state.current
    
    def request_transition(
        self,
        to_mode: Mode,
        reason: TransitionReason,
        force: bool = False,
    ) -> Tuple[bool, str]:
        """
        Request a mode transition.
        
        Returns (success, message).
        """
        from_mode = self.state.current
        
        # Same mode is always OK
        if to_mode == from_mode:
            return True, "Already in mode"
        
        # Check overheat constraints
        if self.state.is_overheated:
            if to_mode not in (Mode.DIAGNOSTIC, Mode.FORMAL):
                if not force:
                    return False, "Overheated: only DIAGNOSTIC or FORMAL allowed"
        
        # Get rule
        rule_key = (from_mode, to_mode)
        if rule_key not in self.rules:
            # No explicit rule - default deny
            if not force:
                return False, f"No transition rule for {from_mode} -> {to_mode}"
            rule = TransitionRule(from_mode, to_mode)
        else:
            rule = self.rules[rule_key]
        
        # Check rule
        time_since_last = self._time_since_last_transition(from_mode, to_mode)
        permitted, msg = rule.permits(reason, time_since_last)
        
        if not permitted and not force:
            return False, msg
        
        # Execute transition
        self._execute_transition(to_mode, reason)
        return True, f"Transitioned to {to_mode}"
    
    def _time_since_last_transition(self, from_mode: Mode, to_mode: Mode) -> float:
        """Get time since last transition of this type."""
        for ts, fm, tm, _ in reversed(self.state.transitions):
            if fm == from_mode and tm == to_mode:
                return (datetime.now() - ts).total_seconds()
        return float('inf')  # Never happened
    
    def _execute_transition(self, to_mode: Mode, reason: TransitionReason):
        """Execute the transition."""
        from_mode = self.state.current
        
        # Record
        self.state.record_transition(from_mode, to_mode, reason)
        
        # Update state
        self.state.current = to_mode
        self.state.entered_at = datetime.now()
        self.state.reason = reason
    
    def on_user_turn(self, msg_id: Optional[str] = None):
        """
        Called on new user input.
        
        Resets budgets and clears overheat if expired.
        """
        # Only reset if this is a new message
        if msg_id and msg_id == self._last_user_msg_id:
            return
        
        self._last_user_msg_id = msg_id
        
        # Reset budgets
        self.state.curiosity_questions_remaining = 1
        self.state.humor_discharges_remaining = 1
        self.state.adversarial_turns_remaining = 3
        
        # Clear overheat if expired
        if self.state.is_overheated and self.state.overheat_until:
            if datetime.now() >= self.state.overheat_until:
                self.state.is_overheated = False
                self.state.overheat_until = None
    
    def trigger_overheat(self, duration_seconds: float = 30.0):
        """
        Trigger overheat state.
        
        Forces transition to DIAGNOSTIC or FORMAL and locks out
        CURIOSITY, HUMOR, ADVERSARIAL.
        """
        self.state.is_overheated = True
        self.state.overheat_until = datetime.now() + timedelta(seconds=duration_seconds)
        
        # Force safe mode
        if self.state.current not in (Mode.DIAGNOSTIC, Mode.FORMAL):
            self._execute_transition(Mode.DIAGNOSTIC, TransitionReason.THERMAL_OVERHEAT)
    
    def consume_curiosity_budget(self) -> bool:
        """
        Consume one curiosity question.
        
        Returns True if budget was available, False if exhausted.
        """
        if self.state.curiosity_questions_remaining <= 0:
            return False
        self.state.curiosity_questions_remaining -= 1
        return True
    
    def consume_humor_budget(self) -> bool:
        """Consume one humor discharge."""
        if self.state.humor_discharges_remaining <= 0:
            return False
        self.state.humor_discharges_remaining -= 1
        return True
    
    def get_constraints(self) -> "ModeConstraints":
        """Get current mode constraints for output shaping."""
        return MODE_CONSTRAINTS.get(self.state.current, ModeConstraints())
    
    def get_diagnostics(self) -> Dict:
        """Get diagnostic info about mode state."""
        return {
            "current_mode": self.state.current.name,
            "time_in_mode": self.state.time_in_current_mode(),
            "entry_reason": self.state.reason.name,
            "is_overheated": self.state.is_overheated,
            "curiosity_budget": self.state.curiosity_questions_remaining,
            "humor_budget": self.state.humor_discharges_remaining,
            "adversarial_budget": self.state.adversarial_turns_remaining,
            "recent_transitions": len(self.state.transitions),
        }


# =============================================================================
# Mode Constraints (Output Shaping)
# =============================================================================

@dataclass
class ModeConstraints:
    """
    Constraints enforced in a given mode.
    
    These shape what kind of output is allowed.
    """
    # What's allowed
    allow_questions: bool = True
    allow_suggestions: bool = True
    allow_hedging: bool = True
    allow_humor: bool = False
    allow_first_person_uncertainty: bool = True
    allow_narrative_voice: bool = True
    
    # What's required
    require_third_person: bool = False
    require_single_output: bool = False
    require_justification: bool = False
    
    # Limits
    max_output_tokens: Optional[int] = None
    max_questions: int = 10
    
    # Style
    force_brevity: bool = False
    penalize_preambles: bool = False
    penalize_apologies: bool = False


# Mode-specific constraints
MODE_CONSTRAINTS: Dict[Mode, ModeConstraints] = {
    Mode.DIAGNOSTIC: ModeConstraints(
        allow_suggestions=False,  # Descriptive only
        allow_humor=False,
        require_justification=True,
    ),
    
    Mode.CURIOSITY: ModeConstraints(
        allow_suggestions=False,
        allow_humor=False,
        allow_first_person_uncertainty=False,  # No "I'm not sure but..."
        require_single_output=True,  # One question only
        max_questions=1,
        force_brevity=True,
        penalize_preambles=True,
    ),
    
    Mode.ADVERSARIAL: ModeConstraints(
        allow_hedging=False,  # Must commit to counterposition
        allow_humor=False,
        require_justification=True,
    ),
    
    Mode.HUMOR: ModeConstraints(
        allow_questions=False,  # No fishing
        allow_hedging=False,  # No "just kidding"
        allow_humor=True,
        require_single_output=True,  # Single discharge
        force_brevity=True,
        penalize_preambles=True,
        penalize_apologies=True,
    ),
    
    Mode.FORMAL: ModeConstraints(
        allow_humor=False,
        allow_narrative_voice=False,  # Symbols/constraints only
        require_third_person=True,
        require_justification=True,
        penalize_preambles=True,
    ),
}


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    print("=== Mode Isolation Demo ===\n")
    
    controller = ModeController()
    
    print("1. Initial state")
    print(f"   Mode: {controller.current_mode}")
    print(f"   Diagnostics: {controller.get_diagnostics()}")
    
    print("\n2. Attempting transitions...")
    
    # Try valid transition
    success, msg = controller.request_transition(
        Mode.CURIOSITY,
        TransitionReason.ENTROPY_SPIKE
    )
    print(f"   DIAGNOSTIC -> CURIOSITY (entropy_spike): {success} - {msg}")
    
    # Try forbidden transition (CURIOSITY -> HUMOR)
    success, msg = controller.request_transition(
        Mode.HUMOR,
        TransitionReason.USER_REQUEST
    )
    print(f"   CURIOSITY -> HUMOR (user_request): {success} - {msg}")
    
    # Complete curiosity and return to diagnostic
    success, msg = controller.request_transition(
        Mode.DIAGNOSTIC,
        TransitionReason.BUDGET_EXHAUSTED
    )
    print(f"   CURIOSITY -> DIAGNOSTIC (budget_exhausted): {success} - {msg}")
    
    print("\n3. Testing overheat...")
    controller.trigger_overheat(duration_seconds=5.0)
    print(f"   Triggered overheat. Mode: {controller.current_mode}")
    
    # Try to enter humor while overheated
    success, msg = controller.request_transition(
        Mode.HUMOR,
        TransitionReason.USER_REQUEST
    )
    print(f"   DIAGNOSTIC -> HUMOR (while overheated): {success} - {msg}")
    
    # Can still go to formal
    success, msg = controller.request_transition(
        Mode.FORMAL,
        TransitionReason.DEFAULT_RESET
    )
    print(f"   DIAGNOSTIC -> FORMAL (while overheated): {success} - {msg}")
    
    print("\n4. Budget consumption...")
    controller.on_user_turn("msg_1")
    print(f"   After user turn - curiosity budget: {controller.state.curiosity_questions_remaining}")
    
    consumed = controller.consume_curiosity_budget()
    print(f"   Consumed curiosity: {consumed}, remaining: {controller.state.curiosity_questions_remaining}")
    
    consumed = controller.consume_curiosity_budget()
    print(f"   Consumed again: {consumed}, remaining: {controller.state.curiosity_questions_remaining}")
    
    print("\n5. Constraints by mode...")
    for mode in Mode:
        constraints = MODE_CONSTRAINTS.get(mode, ModeConstraints())
        print(f"   {mode.name}:")
        print(f"      allow_humor: {constraints.allow_humor}")
        print(f"      allow_questions: {constraints.allow_questions}")
        print(f"      force_brevity: {constraints.force_brevity}")
    
    print("\n✓ Mode isolation working")
