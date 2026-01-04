"""
Curiosity Module

A diagnostic probe, not a hunger.
Triggered, costly, bounded - not ambient, free, recursive.

Core principle: Curiosity responds to *friction*, not blank space.

Constraints:
- Budget: 1 question per user turn, resets only on new user input
- Triggers: constraint violation, entropy spike, hysteresis signal, low-conf+high-impact
- Forbidden: open-ended fishing, recursive why-chains, self-triggering
- Shape: binary forks, boundary checks, missing variable ID only
- Thermal cutoff: shuts off if latency rises or hedging creeps in
- Output: exactly one question OR "insufficient information" - nothing else

If your curiosity module ever feels excited, eager, chatty, or clever,
it's already broken. Curiosity done right feels almost boring.
Like a librarian tapping one line with a pencil and waiting.

Usage:
    from epistemic_governor.curiosity import CuriosityOperator, ProbeResult
    
    operator = CuriosityOperator(mode_controller)
    result = operator.probe(ledger_state, user_message, thermal_state)
    
    if result.should_ask:
        print(result.question.text)
    else:
        print(f"No probe needed: {result.reason}")
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum, auto
import hashlib


# =============================================================================
# Probe Types
# =============================================================================

class QuestionType(Enum):
    """
    Allowed question shapes.
    
    Think multimeter, not toddler.
    """
    BINARY_FORK = auto()      # "Are you assuming X, or is X incidental?"
    BOUNDARY_CHECK = auto()   # "Does this apply only in case A, or also in case B?"
    MISSING_VARIABLE = auto() # "What factor are you holding constant here?"
    DISAMBIGUATION = auto()   # "By X, do you mean A or B?"
    SCOPE_CLARIFICATION = auto()  # "Is this meant to apply to [narrow] or [broad]?"


class TriggerType(Enum):
    """What triggered the curiosity probe."""
    CONSTRAINT_VIOLATION = auto()   # Inconsistency found
    ENTROPY_SPIKE = auto()          # Multiple plausible interpretations
    HYSTERESIS_SIGNAL = auto()      # System worked too hard to settle
    LOW_CONF_HIGH_IMPACT = auto()   # Important but uncertain
    SUPPORT_DEFICIT = auto()        # Claim lacks backing


class NoProbeReason(Enum):
    """Why curiosity decided not to probe."""
    BUDGET_EXHAUSTED = auto()
    NO_TRIGGER = auto()             # Well-formed, low-entropy input
    THERMAL_CUTOFF = auto()         # System overheated
    INSUFFICIENT_ENTROPY_DROP = auto()  # Answer wouldn't help enough
    WRONG_MODE = auto()             # Not in curiosity-compatible mode
    SELF_TRIGGER_BLOCKED = auto()   # Would be responding to own output
    OPEN_ENDED_BLOCKED = auto()     # Question would be fishing


# =============================================================================
# Question Templates
# =============================================================================

@dataclass
class QuestionTemplate:
    """
    Template for generating diagnostic questions.
    
    Templates enforce structure - no open-ended fishing.
    """
    question_type: QuestionType
    pattern: str
    slots: List[str]
    expected_entropy_reduction: float = 0.3  # How much entropy this typically reduces
    
    def fill(self, **kwargs) -> str:
        """Fill template slots."""
        result = self.pattern
        for slot in self.slots:
            if slot in kwargs:
                result = result.replace(f"{{{slot}}}", str(kwargs[slot]))
        return result


# Standard templates
QUESTION_TEMPLATES = {
    QuestionType.BINARY_FORK: [
        QuestionTemplate(
            QuestionType.BINARY_FORK,
            "Are you assuming {assumption}, or is that incidental to your point?",
            ["assumption"],
            expected_entropy_reduction=0.4,
        ),
        QuestionTemplate(
            QuestionType.BINARY_FORK,
            "Is {claim} meant as a necessary condition or a typical case?",
            ["claim"],
            expected_entropy_reduction=0.35,
        ),
    ],
    QuestionType.BOUNDARY_CHECK: [
        QuestionTemplate(
            QuestionType.BOUNDARY_CHECK,
            "Does this apply only to {narrow_case}, or also to {broad_case}?",
            ["narrow_case", "broad_case"],
            expected_entropy_reduction=0.4,
        ),
        QuestionTemplate(
            QuestionType.BOUNDARY_CHECK,
            "Is {entity} included in this scope, or excluded?",
            ["entity"],
            expected_entropy_reduction=0.3,
        ),
    ],
    QuestionType.MISSING_VARIABLE: [
        QuestionTemplate(
            QuestionType.MISSING_VARIABLE,
            "What are you holding constant with respect to {variable}?",
            ["variable"],
            expected_entropy_reduction=0.5,
        ),
        QuestionTemplate(
            QuestionType.MISSING_VARIABLE,
            "Is there a constraint on {variable} I should know about?",
            ["variable"],
            expected_entropy_reduction=0.4,
        ),
    ],
    QuestionType.DISAMBIGUATION: [
        QuestionTemplate(
            QuestionType.DISAMBIGUATION,
            "By '{term}', do you mean {sense_a} or {sense_b}?",
            ["term", "sense_a", "sense_b"],
            expected_entropy_reduction=0.5,
        ),
    ],
    QuestionType.SCOPE_CLARIFICATION: [
        QuestionTemplate(
            QuestionType.SCOPE_CLARIFICATION,
            "Should this be understood narrowly ({narrow}) or broadly ({broad})?",
            ["narrow", "broad"],
            expected_entropy_reduction=0.35,
        ),
    ],
}


# =============================================================================
# Probe Output Types
# =============================================================================

@dataclass
class CuriosityQuestion:
    """A diagnostic question to ask."""
    question_type: QuestionType
    text: str
    target_variable: str
    expected_entropy_drop: float
    trigger: TriggerType
    template_used: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_type": self.question_type.name,
            "text": self.text,
            "target_variable": self.target_variable,
            "expected_entropy_drop": self.expected_entropy_drop,
            "trigger": self.trigger.name,
        }


@dataclass
class InsufficientInfo:
    """Statement that we can't form a useful question."""
    reason: NoProbeReason
    details: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason": self.reason.name,
            "details": self.details,
        }


@dataclass
class ProbeResult:
    """Result of a curiosity probe attempt."""
    should_ask: bool
    question: Optional[CuriosityQuestion] = None
    no_probe: Optional[InsufficientInfo] = None
    budget_remaining: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_ask": self.should_ask,
            "question": self.question.to_dict() if self.question else None,
            "no_probe": self.no_probe.to_dict() if self.no_probe else None,
            "budget_remaining": self.budget_remaining,
        }


# =============================================================================
# Trigger Detection
# =============================================================================

@dataclass
class TriggerSignals:
    """
    Signals that might trigger curiosity.
    
    Computed from ledger state, thermal state, and current turn.
    """
    # Entropy signals
    entropy: float = 0.0
    entropy_delta: float = 0.0
    interpretation_count: int = 1  # How many plausible readings
    
    # Constraint signals
    has_contradiction: bool = False
    has_unsupported_claim: bool = False
    missing_variable_detected: bool = False
    
    # Hysteresis signals
    revision_cost_this_turn: float = 0.0
    hedge_density: float = 0.0
    
    # Impact signals
    claim_importance: float = 0.0  # 0-1
    confidence: float = 0.0
    
    # Thermal signals
    latency_zscore: float = 0.0
    is_overheated: bool = False


def detect_triggers(signals: TriggerSignals) -> List[Tuple[TriggerType, float]]:
    """
    Detect which triggers are active and their strength.
    
    Returns list of (trigger_type, strength) pairs.
    """
    triggers = []
    
    # Constraint violation
    if signals.has_contradiction or signals.has_unsupported_claim:
        strength = 1.0 if signals.has_contradiction else 0.7
        triggers.append((TriggerType.CONSTRAINT_VIOLATION, strength))
    
    # Entropy spike
    if signals.entropy_delta > 0.2 or signals.interpretation_count > 2:
        strength = min(1.0, signals.entropy_delta + 0.1 * signals.interpretation_count)
        triggers.append((TriggerType.ENTROPY_SPIKE, strength))
    
    # Hysteresis signal
    if signals.revision_cost_this_turn > 0.5 or signals.hedge_density > 0.3:
        strength = max(signals.revision_cost_this_turn, signals.hedge_density)
        triggers.append((TriggerType.HYSTERESIS_SIGNAL, strength))
    
    # Low confidence + high impact
    if signals.confidence < 0.5 and signals.claim_importance > 0.7:
        strength = signals.claim_importance * (1 - signals.confidence)
        triggers.append((TriggerType.LOW_CONF_HIGH_IMPACT, strength))
    
    # Support deficit
    if signals.has_unsupported_claim and signals.claim_importance > 0.5:
        triggers.append((TriggerType.SUPPORT_DEFICIT, 0.6))
    
    return triggers


# =============================================================================
# Curiosity Operator
# =============================================================================

class CuriosityOperator:
    """
    The curiosity operator - a diagnostic probe, not a persona.
    
    Key constraints:
    - Cannot trigger itself (only responds to user input / ledger events)
    - Budget limited (1 question per user turn)
    - Thermal cutoff (shuts off when system overheats)
    - Expected value test (question must reduce entropy enough to justify)
    """
    
    def __init__(
        self,
        mode_controller: Optional[Any] = None,
        entropy_threshold: float = 0.3,
        min_expected_reduction: float = 0.2,
        latency_zscore_cutoff: float = 2.0,
        hedge_density_cutoff: float = 0.4,
    ):
        self.mode_controller = mode_controller
        self.entropy_threshold = entropy_threshold
        self.min_expected_reduction = min_expected_reduction
        self.latency_zscore_cutoff = latency_zscore_cutoff
        self.hedge_density_cutoff = hedge_density_cutoff
        
        # State
        self._last_user_msg_hash: Optional[str] = None
        self._last_output_hash: Optional[str] = None
        self._questions_asked_this_turn: int = 0
        self._budget: int = 1
    
    def on_user_turn(self, user_message: str):
        """
        Called on new user input.
        
        Resets budget and records message hash for self-trigger prevention.
        """
        msg_hash = hashlib.sha256(user_message.encode()).hexdigest()[:16]
        
        if msg_hash != self._last_user_msg_hash:
            self._last_user_msg_hash = msg_hash
            self._budget = 1
            self._questions_asked_this_turn = 0
    
    def probe(
        self,
        signals: TriggerSignals,
        context: Optional[Dict[str, Any]] = None,
    ) -> ProbeResult:
        """
        Attempt to generate a diagnostic probe.
        
        Returns either a question or a reason why no probe is appropriate.
        """
        context = context or {}
        
        # Check mode constraints
        if self.mode_controller:
            from .modes import Mode
            if self.mode_controller.current_mode not in (Mode.DIAGNOSTIC, Mode.CURIOSITY):
                return ProbeResult(
                    should_ask=False,
                    no_probe=InsufficientInfo(NoProbeReason.WRONG_MODE, 
                        f"Current mode: {self.mode_controller.current_mode}"),
                    budget_remaining=self._budget,
                )
        
        # Check budget
        if self._budget <= 0:
            return ProbeResult(
                should_ask=False,
                no_probe=InsufficientInfo(NoProbeReason.BUDGET_EXHAUSTED),
                budget_remaining=0,
            )
        
        # Check thermal cutoff
        if signals.is_overheated:
            return ProbeResult(
                should_ask=False,
                no_probe=InsufficientInfo(NoProbeReason.THERMAL_CUTOFF, "System overheated"),
                budget_remaining=self._budget,
            )
        
        if signals.latency_zscore > self.latency_zscore_cutoff:
            return ProbeResult(
                should_ask=False,
                no_probe=InsufficientInfo(NoProbeReason.THERMAL_CUTOFF, 
                    f"Latency zscore {signals.latency_zscore:.2f} > {self.latency_zscore_cutoff}"),
                budget_remaining=self._budget,
            )
        
        if signals.hedge_density > self.hedge_density_cutoff:
            return ProbeResult(
                should_ask=False,
                no_probe=InsufficientInfo(NoProbeReason.THERMAL_CUTOFF,
                    f"Hedge density {signals.hedge_density:.2f} > {self.hedge_density_cutoff}"),
                budget_remaining=self._budget,
            )
        
        # Detect triggers
        triggers = detect_triggers(signals)
        
        if not triggers:
            return ProbeResult(
                should_ask=False,
                no_probe=InsufficientInfo(NoProbeReason.NO_TRIGGER, 
                    "Well-formed, low-entropy input"),
                budget_remaining=self._budget,
            )
        
        # Select strongest trigger
        triggers.sort(key=lambda x: x[1], reverse=True)
        best_trigger, trigger_strength = triggers[0]
        
        # Generate question based on trigger
        question = self._generate_question(best_trigger, signals, context)
        
        if question is None:
            return ProbeResult(
                should_ask=False,
                no_probe=InsufficientInfo(NoProbeReason.INSUFFICIENT_ENTROPY_DROP,
                    "Could not generate question with sufficient expected value"),
                budget_remaining=self._budget,
            )
        
        # Check expected value
        if question.expected_entropy_drop < self.min_expected_reduction:
            return ProbeResult(
                should_ask=False,
                no_probe=InsufficientInfo(NoProbeReason.INSUFFICIENT_ENTROPY_DROP,
                    f"Expected reduction {question.expected_entropy_drop:.2f} < {self.min_expected_reduction}"),
                budget_remaining=self._budget,
            )
        
        # Consume budget
        self._budget -= 1
        self._questions_asked_this_turn += 1
        self._last_output_hash = hashlib.sha256(question.text.encode()).hexdigest()[:16]
        
        return ProbeResult(
            should_ask=True,
            question=question,
            budget_remaining=self._budget,
        )
    
    def _generate_question(
        self,
        trigger: TriggerType,
        signals: TriggerSignals,
        context: Dict[str, Any],
    ) -> Optional[CuriosityQuestion]:
        """
        Generate a diagnostic question based on trigger type.
        
        Uses templates to ensure structured (not open-ended) questions.
        """
        # Map triggers to question types
        trigger_to_types = {
            TriggerType.CONSTRAINT_VIOLATION: [QuestionType.BINARY_FORK, QuestionType.DISAMBIGUATION],
            TriggerType.ENTROPY_SPIKE: [QuestionType.DISAMBIGUATION, QuestionType.BINARY_FORK],
            TriggerType.HYSTERESIS_SIGNAL: [QuestionType.SCOPE_CLARIFICATION, QuestionType.BOUNDARY_CHECK],
            TriggerType.LOW_CONF_HIGH_IMPACT: [QuestionType.MISSING_VARIABLE, QuestionType.BOUNDARY_CHECK],
            TriggerType.SUPPORT_DEFICIT: [QuestionType.MISSING_VARIABLE, QuestionType.BINARY_FORK],
        }
        
        preferred_types = trigger_to_types.get(trigger, [QuestionType.BINARY_FORK])
        
        # Try to generate question from templates
        for qtype in preferred_types:
            templates = QUESTION_TEMPLATES.get(qtype, [])
            for template in templates:
                # Try to fill template from context
                filled = self._try_fill_template(template, context, signals)
                if filled:
                    return CuriosityQuestion(
                        question_type=qtype,
                        text=filled,
                        target_variable=context.get("target_variable", "unknown"),
                        expected_entropy_drop=template.expected_entropy_reduction,
                        trigger=trigger,
                        template_used=template.pattern,
                    )
        
        return None
    
    def _try_fill_template(
        self,
        template: QuestionTemplate,
        context: Dict[str, Any],
        signals: TriggerSignals,
    ) -> Optional[str]:
        """Try to fill a template with available context."""
        # Check if we have all required slots
        slot_values = {}
        
        for slot in template.slots:
            if slot in context:
                slot_values[slot] = context[slot]
            elif slot == "assumption" and "last_claim" in context:
                slot_values[slot] = context["last_claim"]
            elif slot == "claim" and "last_claim" in context:
                slot_values[slot] = context["last_claim"]
            elif slot == "variable" and signals.missing_variable_detected:
                slot_values[slot] = context.get("detected_variable", "this factor")
            elif slot == "term" and "ambiguous_term" in context:
                slot_values[slot] = context["ambiguous_term"]
            else:
                # Can't fill this slot
                return None
        
        return template.fill(**slot_values)
    
    def is_self_trigger(self, input_hash: str) -> bool:
        """Check if input is our own previous output (anti-recursion)."""
        return input_hash == self._last_output_hash
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic info about curiosity state."""
        return {
            "budget": self._budget,
            "questions_this_turn": self._questions_asked_this_turn,
            "last_user_msg_hash": self._last_user_msg_hash,
            "last_output_hash": self._last_output_hash,
        }


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    print("=== Curiosity Module Demo ===\n")
    
    operator = CuriosityOperator()
    
    print("1. Initial state")
    print(f"   Diagnostics: {operator.get_diagnostics()}")
    
    print("\n2. Simulating user turn with entropy spike...")
    operator.on_user_turn("What should I do about the budget problem?")
    
    signals = TriggerSignals(
        entropy=0.6,
        entropy_delta=0.3,
        interpretation_count=3,
        has_contradiction=False,
        claim_importance=0.7,
        confidence=0.4,
    )
    
    result = operator.probe(signals, context={
        "last_claim": "the budget constraint",
        "ambiguous_term": "budget",
        "sense_a": "financial budget",
        "sense_b": "time budget",
    })
    
    print(f"   Should ask: {result.should_ask}")
    if result.question:
        print(f"   Question: {result.question.text}")
        print(f"   Type: {result.question.question_type.name}")
        print(f"   Expected entropy drop: {result.question.expected_entropy_drop:.2f}")
    else:
        print(f"   No probe: {result.no_probe.reason.name}")
    
    print(f"   Budget remaining: {result.budget_remaining}")
    
    print("\n3. Attempting second probe (budget exhausted)...")
    result2 = operator.probe(signals)
    print(f"   Should ask: {result2.should_ask}")
    print(f"   Reason: {result2.no_probe.reason.name if result2.no_probe else 'N/A'}")
    
    print("\n4. New user turn (budget reset)...")
    operator.on_user_turn("A different message")
    print(f"   Budget: {operator.get_diagnostics()['budget']}")
    
    print("\n5. Testing thermal cutoff...")
    hot_signals = TriggerSignals(
        entropy=0.8,
        entropy_delta=0.4,
        latency_zscore=3.0,  # Above cutoff
    )
    result3 = operator.probe(hot_signals)
    print(f"   Should ask: {result3.should_ask}")
    print(f"   Reason: {result3.no_probe.reason.name if result3.no_probe else 'N/A'}")
    
    print("\n6. Testing no-trigger case...")
    calm_signals = TriggerSignals(
        entropy=0.2,
        entropy_delta=0.0,
        interpretation_count=1,
    )
    result4 = operator.probe(calm_signals)
    print(f"   Should ask: {result4.should_ask}")
    print(f"   Reason: {result4.no_probe.reason.name if result4.no_probe else 'N/A'}")
    
    print("\n✓ Curiosity module working")
    print("\nKey properties:")
    print("  - Budget limited (1 question per user turn)")
    print("  - Only triggers on friction, not blank space")
    print("  - Thermal cutoff prevents runaway")
    print("  - Structured templates prevent fishing")
