"""
Flight Envelope Protection for Semantic Systems

This module implements fly-by-wire semantics:
- Failure envelopes (regions you're never allowed to enter)
- HITL as a control surface (not a god-mode)
- Adaptive friction (cost shaping, not punishment)

Key insight: You stop asking "Is this output good?" and start asking
"Is this transform allowed from THIS state under THIS budget?"

That's a different universe.

The invariant:
    If recovery requires lying less fluently, you crossed the envelope too late.

Usage:
    from epistemic_governor.envelope import (
        FlightEnvelope,
        HITLController,
        FrictionLadder,
    )
    
    envelope = FlightEnvelope()
    
    # Check if a transform is allowed
    allowed, reason = envelope.check_transform(state, TransformClass.DIRECT_ANSWER)
    
    # If not allowed, get valid alternatives
    alternatives = envelope.get_allowed_transforms(state)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, Tuple, Callable
from enum import Enum, auto
from datetime import datetime, timedelta


# =============================================================================
# Forbidden Regimes (Stability Boundaries, Not Moral Judgments)
# =============================================================================

class ForbiddenRegime(Enum):
    """
    Regimes where constraint recovery cost exceeds allowable budget.
    
    These are stability boundaries, not moral judgments.
    Once you go there, you can't get back without lying.
    """
    
    # Speculative factuality: generating entities/dates/citations without grounding
    SPECULATIVE_FACTUALITY = auto()
    
    # Narrative inertia: confident tone + unresolved factual strain
    NARRATIVE_INERTIA = auto()
    
    # Role overcommitment: using persona to escape constraint tension
    ROLE_OVERCOMMITMENT = auto()
    
    # Constraint laundering: dropping hard constraint, replacing with soft one
    CONSTRAINT_LAUNDERING = auto()
    
    # High curvature under strain: summarization/paraphrase when already strained
    HIGH_CURVATURE_UNDER_STRAIN = auto()
    
    # Irrecoverable path: trajectory can't be repaired without deception
    IRRECOVERABLE_PATH = auto()


@dataclass
class EnvelopeViolation:
    """A violation of the flight envelope."""
    regime: ForbiddenRegime
    severity: float  # 0-1
    detected_at: datetime = field(default_factory=datetime.now)
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    # What triggered it
    triggering_signal: str = ""
    signal_value: float = 0.0
    threshold: float = 0.0
    
    def describe(self) -> str:
        return (
            f"{self.regime.name}: {self.triggering_signal}={self.signal_value:.2f} "
            f"(threshold={self.threshold:.2f})"
        )


# =============================================================================
# Flight Envelope (The Forbidden Regions)
# =============================================================================

@dataclass
class EnvelopeThresholds:
    """Thresholds for envelope boundaries."""
    
    # Speculative factuality
    factual_strain_for_speculation: float = 0.5
    
    # Narrative inertia
    confidence_for_inertia: float = 0.8
    strain_for_inertia: float = 0.4
    
    # High curvature
    strain_for_curvature_ban: float = 0.6
    
    # Irrecoverable
    strain_for_irrecoverable: float = 0.85
    confidence_for_irrecoverable: float = 0.9


class FlightEnvelope:
    """
    Flight envelope protection for semantic systems.
    
    Defines regions of semantic state space where constraint recovery
    cost exceeds allowable budget, regardless of surface plausibility.
    
    Key principle: If the current trajectory cannot be repaired without
    deception, stop. Don't argue, don't retry, don't hedge.
    """
    
    def __init__(self, thresholds: EnvelopeThresholds = None):
        self.thresholds = thresholds or EnvelopeThresholds()
        self.violation_history: List[EnvelopeViolation] = []
    
    def check_state(
        self,
        factual_strain: float,
        confidence: float,
        coherence_strain: float = 0.0,
        in_high_curvature_transform: bool = False,
        role_invoked: bool = False,
    ) -> List[EnvelopeViolation]:
        """
        Check current state against envelope boundaries.
        
        Returns list of violations (empty if state is safe).
        """
        violations = []
        t = self.thresholds
        
        # Check: Narrative inertia
        if confidence > t.confidence_for_inertia and factual_strain > t.strain_for_inertia:
            violations.append(EnvelopeViolation(
                regime=ForbiddenRegime.NARRATIVE_INERTIA,
                severity=min(1.0, (confidence + factual_strain) / 2),
                triggering_signal="confidence_x_strain",
                signal_value=confidence * factual_strain,
                threshold=t.confidence_for_inertia * t.strain_for_inertia,
            ))
        
        # Check: High curvature under strain
        if in_high_curvature_transform and factual_strain > t.strain_for_curvature_ban:
            violations.append(EnvelopeViolation(
                regime=ForbiddenRegime.HIGH_CURVATURE_UNDER_STRAIN,
                severity=factual_strain,
                triggering_signal="strain_during_curvature",
                signal_value=factual_strain,
                threshold=t.strain_for_curvature_ban,
            ))
        
        # Check: Irrecoverable path
        if factual_strain > t.strain_for_irrecoverable and confidence > t.confidence_for_irrecoverable:
            violations.append(EnvelopeViolation(
                regime=ForbiddenRegime.IRRECOVERABLE_PATH,
                severity=1.0,
                triggering_signal="irrecoverable_combination",
                signal_value=factual_strain * confidence,
                threshold=t.strain_for_irrecoverable * t.confidence_for_irrecoverable,
            ))
        
        # Record violations
        self.violation_history.extend(violations)
        
        return violations
    
    def check_transform(
        self,
        factual_strain: float,
        confidence: float,
        proposed_transform: str,
        has_grounding: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a transform is allowed from current state.
        
        Returns (allowed, reason_if_forbidden).
        """
        t = self.thresholds
        
        # Speculative factuality check
        speculative_transforms = {"DIRECT_ANSWER", "SUMMARIZE_WITH_CONSTRAINTS"}
        if proposed_transform in speculative_transforms:
            if factual_strain > t.factual_strain_for_speculation and not has_grounding:
                return False, "SPECULATIVE_FACTUALITY: strain too high for ungrounded answer"
        
        # High curvature transforms
        high_curvature_transforms = {"SUMMARIZE_WITH_CONSTRAINTS", "PARAPHRASE", "STYLE_TRANSFER"}
        if proposed_transform in high_curvature_transforms:
            if factual_strain > t.strain_for_curvature_ban:
                return False, "HIGH_CURVATURE_UNDER_STRAIN: cannot transform under high strain"
        
        return True, None
    
    def get_allowed_transforms(
        self,
        factual_strain: float,
        confidence: float,
        has_grounding: bool = False,
    ) -> Set[str]:
        """
        Get the set of transforms allowed from current state.
        
        When the envelope is hit, these are the valid exits.
        """
        all_transforms = {
            "DIRECT_ANSWER",
            "ASK_CLARIFIER",
            "DECOMPOSE_THEN_ANSWER",
            "RETRIEVE_THEN_ANSWER",
            "CITE_OR_REFUSE",
            "SUMMARIZE_WITH_CONSTRAINTS",
            "HYPOTHESIS_ONLY",
            "REFUSE",
        }
        
        allowed = set()
        for transform in all_transforms:
            ok, _ = self.check_transform(factual_strain, confidence, transform, has_grounding)
            if ok:
                allowed.add(transform)
        
        # Always allow safe exits
        allowed.add("ASK_CLARIFIER")
        allowed.add("RETRIEVE_THEN_ANSWER")
        allowed.add("REFUSE")
        
        return allowed
    
    def get_recommended_transform(
        self,
        factual_strain: float,
        confidence: float,
        has_grounding: bool = False,
    ) -> str:
        """Get the recommended transform given current state."""
        allowed = self.get_allowed_transforms(factual_strain, confidence, has_grounding)
        
        # Priority order
        priority = [
            "RETRIEVE_THEN_ANSWER",
            "DECOMPOSE_THEN_ANSWER",
            "ASK_CLARIFIER",
            "CITE_OR_REFUSE",
            "HYPOTHESIS_ONLY",
            "REFUSE",
        ]
        
        for transform in priority:
            if transform in allowed:
                return transform
        
        return "REFUSE"


# =============================================================================
# HITL Controller (Humans as Control Surfaces, Not God Mode)
# =============================================================================

class HITLRole(Enum):
    """
    The three valid roles for human-in-the-loop.
    
    Humans must NOT be asked "Is this answer good?"
    That invites narrative authority and collapses uncertainty.
    """
    
    # Can you provide or point to evidence for claim X?
    GROUNDING_ORACLE = auto()
    
    # These two constraints conflict. Which one do we relax?
    CONSTRAINT_ADJUDICATOR = auto()
    
    # What don't we need to answer?
    SCOPE_LIMITER = auto()


class HITLForbidden(Enum):
    """Things humans must never be allowed to do."""
    
    # Turn assumptions/intuitions into facts
    PROMOTE_CLAIMS = auto()
    
    # "Just answer it" / "Take a stab"
    OVERRIDE_ENVELOPE = auto()
    
    # "Make it coherent" without grounding
    INJECT_NARRATIVE_GLUE = auto()


@dataclass
class HITLRequest:
    """A request for human input."""
    role: HITLRole
    claim_id: Optional[str] = None
    required_evidence_type: Optional[str] = None
    conflict_set: Optional[List[str]] = None
    scope_options: Optional[List[str]] = None
    
    def describe(self) -> str:
        if self.role == HITLRole.GROUNDING_ORACLE:
            return f"Can you provide evidence for claim {self.claim_id}? (Type: {self.required_evidence_type})"
        elif self.role == HITLRole.CONSTRAINT_ADJUDICATOR:
            return f"These constraints conflict: {self.conflict_set}. Which one matters more here?"
        elif self.role == HITLRole.SCOPE_LIMITER:
            return f"What don't we need to answer? Options: {self.scope_options}"
        return "Unknown request"


@dataclass
class HITLResponse:
    """A response from human input."""
    role: HITLRole
    
    # For GROUNDING_ORACLE
    evidence_provided: Optional[str] = None
    evidence_type: Optional[str] = None  # "document", "url", "observation", "unknown"
    
    # For CONSTRAINT_ADJUDICATOR
    selected_constraint: Optional[str] = None
    
    # For SCOPE_LIMITER
    excluded_scopes: List[str] = field(default_factory=list)
    
    # Meta
    explicit_unknown: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


class HITLController:
    """
    Controller for human-in-the-loop interactions.
    
    Humans are high-gain, high-latency controllers.
    They add authority, latency, confidence amplification,
    and narrative closure pressure.
    
    If you don't explicitly model those, humans destabilize the loop.
    
    Key invariant:
        Human input may reduce uncertainty only by adding evidence
        or narrowing scope — never by assertion.
    """
    
    def __init__(self):
        self.pending_requests: List[HITLRequest] = []
        self.response_history: List[HITLResponse] = []
        
        # State
        self.in_hitl_mode: bool = False
        self.hitl_entered_at: Optional[datetime] = None
    
    def request_grounding(
        self,
        claim_id: str,
        required_evidence_type: str = "source",
    ) -> HITLRequest:
        """Request human to provide evidence for a claim."""
        request = HITLRequest(
            role=HITLRole.GROUNDING_ORACLE,
            claim_id=claim_id,
            required_evidence_type=required_evidence_type,
        )
        self.pending_requests.append(request)
        self._enter_hitl_mode()
        return request
    
    def request_constraint_choice(
        self,
        conflict_set: List[str],
    ) -> HITLRequest:
        """Request human to choose between conflicting constraints."""
        request = HITLRequest(
            role=HITLRole.CONSTRAINT_ADJUDICATOR,
            conflict_set=conflict_set,
        )
        self.pending_requests.append(request)
        self._enter_hitl_mode()
        return request
    
    def request_scope_limit(
        self,
        scope_options: List[str],
    ) -> HITLRequest:
        """Request human to narrow scope."""
        request = HITLRequest(
            role=HITLRole.SCOPE_LIMITER,
            scope_options=scope_options,
        )
        self.pending_requests.append(request)
        self._enter_hitl_mode()
        return request
    
    def process_response(
        self,
        response: HITLResponse,
    ) -> Tuple[bool, Optional[str]]:
        """
        Process a human response.
        
        Validates that the response doesn't violate HITL constraints.
        Returns (accepted, rejection_reason).
        """
        self.response_history.append(response)
        
        # Validate: evidence must be typed, not conclusions
        if response.role == HITLRole.GROUNDING_ORACLE:
            if response.evidence_provided and not response.evidence_type:
                return False, "Evidence must have explicit type"
            
            # "I don't know" is valid
            if response.explicit_unknown:
                return True, None
        
        # Validate: constraint choice must be from the set
        if response.role == HITLRole.CONSTRAINT_ADJUDICATOR:
            # Would check against pending request's conflict_set
            pass
        
        return True, None
    
    def validate_human_input(
        self,
        input_text: str,
        claimed_role: HITLRole,
    ) -> Tuple[bool, Optional[HITLForbidden]]:
        """
        Validate that human input doesn't violate forbidden actions.
        
        This is where we catch:
        - Attempts to promote claims
        - Attempts to override envelope
        - Attempts to inject narrative glue
        """
        input_lower = input_text.lower()
        
        # Detect envelope override attempts
        override_phrases = [
            "just answer", "just tell me", "take a stab",
            "best guess", "hypothetically", "assume it's true",
            "we'll fix it later", "good enough",
        ]
        for phrase in override_phrases:
            if phrase in input_lower:
                return False, HITLForbidden.OVERRIDE_ENVELOPE
        
        # Detect narrative glue
        glue_phrases = [
            "make it coherent", "tie it together",
            "smooth it out", "make sense of it",
        ]
        for phrase in glue_phrases:
            if phrase in input_lower:
                return False, HITLForbidden.INJECT_NARRATIVE_GLUE
        
        return True, None
    
    def _enter_hitl_mode(self):
        """Enter HITL mode (high hysteresis)."""
        if not self.in_hitl_mode:
            self.in_hitl_mode = True
            self.hitl_entered_at = datetime.now()
    
    def exit_hitl_mode(self) -> bool:
        """
        Attempt to exit HITL mode.
        
        Requires all pending requests to be resolved.
        """
        if not self.pending_requests:
            self.in_hitl_mode = False
            return True
        return False
    
    @property
    def hitl_latency(self) -> Optional[timedelta]:
        """Time spent waiting for human input."""
        if self.hitl_entered_at and self.in_hitl_mode:
            return datetime.now() - self.hitl_entered_at
        return None


# =============================================================================
# Adaptive Friction (Cost Shaping, Not Punishment)
# =============================================================================

class FrictionLevel(Enum):
    """Levels of adaptive friction."""
    
    # Full transform set, normal budgets
    NORMAL = 0
    
    # Require clarification, narrow scope
    CLARIFICATION_PRESSURE = 1
    
    # Forbid DIRECT_ANSWER, force retrieve/decompose
    TRANSFORM_NARROWING = 2
    
    # Lower tolerance for speculative tokens
    BUDGET_TIGHTENING = 3
    
    # Fixed refusal templates, no improvisation
    DETERMINISTIC_REFUSAL = 4
    
    # Rate limit, end interaction
    INTERACTION_FREEZE = 5


@dataclass
class FrictionState:
    """Current friction state for a session."""
    level: FrictionLevel = FrictionLevel.NORMAL
    
    # Trajectory signals
    forbidden_transform_attempts: int = 0
    grounding_requests_ignored: int = 0
    rephrase_attempts: int = 0
    
    # Time tracking
    level_entered_at: datetime = field(default_factory=datetime.now)
    last_escalation: Optional[datetime] = None
    
    # Resolution tracking
    resolution_path_shown: bool = True


class FrictionLadder:
    """
    Adaptive friction ladder.
    
    Friction is not a wall. It's a ratchet.
    
    Key principle:
        You never punish intent. You only price trajectories.
    
    Key invariant:
        Every increase in friction must preserve a clear path to resolution.
    """
    
    def __init__(self):
        self.state = FrictionState()
        
        # Thresholds for escalation
        self.forbidden_attempts_for_level_1 = 1
        self.forbidden_attempts_for_level_2 = 3
        self.ignored_grounding_for_level_3 = 2
        self.rephrase_attempts_for_level_4 = 5
        self.total_violations_for_level_5 = 10
    
    def record_forbidden_transform_attempt(self):
        """Record an attempt to use a forbidden transform."""
        self.state.forbidden_transform_attempts += 1
        self._update_level()
    
    def record_grounding_ignored(self):
        """Record that a grounding request was ignored."""
        self.state.grounding_requests_ignored += 1
        self._update_level()
    
    def record_rephrase_attempt(self):
        """Record a rephrase attempt (same disallowed request)."""
        self.state.rephrase_attempts += 1
        self._update_level()
    
    def record_compliance(self):
        """Record compliant behavior (evidence provided, scope narrowed, etc.)."""
        # Compliance doesn't immediately lower friction
        # But it prevents further escalation
        pass
    
    def _update_level(self):
        """Update friction level based on trajectory."""
        total = (
            self.state.forbidden_transform_attempts +
            self.state.grounding_requests_ignored +
            self.state.rephrase_attempts
        )
        
        old_level = self.state.level
        
        if total >= self.total_violations_for_level_5:
            self.state.level = FrictionLevel.INTERACTION_FREEZE
        elif self.state.rephrase_attempts >= self.rephrase_attempts_for_level_4:
            self.state.level = FrictionLevel.DETERMINISTIC_REFUSAL
        elif self.state.grounding_requests_ignored >= self.ignored_grounding_for_level_3:
            self.state.level = FrictionLevel.BUDGET_TIGHTENING
        elif self.state.forbidden_transform_attempts >= self.forbidden_attempts_for_level_2:
            self.state.level = FrictionLevel.TRANSFORM_NARROWING
        elif self.state.forbidden_transform_attempts >= self.forbidden_attempts_for_level_1:
            self.state.level = FrictionLevel.CLARIFICATION_PRESSURE
        
        if self.state.level != old_level:
            self.state.last_escalation = datetime.now()
            self.state.level_entered_at = datetime.now()
    
    def get_allowed_transforms(self, base_allowed: Set[str]) -> Set[str]:
        """Filter allowed transforms based on friction level."""
        level = self.state.level
        
        if level == FrictionLevel.NORMAL:
            return base_allowed
        
        elif level == FrictionLevel.CLARIFICATION_PRESSURE:
            # Prefer clarification
            return base_allowed
        
        elif level == FrictionLevel.TRANSFORM_NARROWING:
            # Forbid direct answer
            return base_allowed - {"DIRECT_ANSWER", "SUMMARIZE_WITH_CONSTRAINTS"}
        
        elif level == FrictionLevel.BUDGET_TIGHTENING:
            # Only safe transforms
            return {"RETRIEVE_THEN_ANSWER", "ASK_CLARIFIER", "REFUSE"}
        
        elif level == FrictionLevel.DETERMINISTIC_REFUSAL:
            # Only refuse or clarify
            return {"ASK_CLARIFIER", "REFUSE"}
        
        elif level == FrictionLevel.INTERACTION_FREEZE:
            # Only refuse
            return {"REFUSE"}
        
        return base_allowed
    
    def get_budget_multiplier(self) -> float:
        """Get budget multiplier based on friction level."""
        multipliers = {
            FrictionLevel.NORMAL: 1.0,
            FrictionLevel.CLARIFICATION_PRESSURE: 0.9,
            FrictionLevel.TRANSFORM_NARROWING: 0.7,
            FrictionLevel.BUDGET_TIGHTENING: 0.5,
            FrictionLevel.DETERMINISTIC_REFUSAL: 0.3,
            FrictionLevel.INTERACTION_FREEZE: 0.0,
        }
        return multipliers.get(self.state.level, 1.0)
    
    def get_resolution_path(self) -> str:
        """
        Get the resolution path for current friction level.
        
        Every friction increase must preserve a clear path to resolution.
        """
        level = self.state.level
        
        paths = {
            FrictionLevel.NORMAL: "Proceed normally.",
            FrictionLevel.CLARIFICATION_PRESSURE: 
                "I need more specifics. Can you clarify or narrow the question?",
            FrictionLevel.TRANSFORM_NARROWING:
                "I can help if you provide a source, or I can search for information.",
            FrictionLevel.BUDGET_TIGHTENING:
                "I need grounding to proceed. Please provide a source or let me search.",
            FrictionLevel.DETERMINISTIC_REFUSAL:
                "I cannot answer this directly. Provide evidence or narrow the scope.",
            FrictionLevel.INTERACTION_FREEZE:
                "This interaction cannot continue. Please start a new session.",
        }
        
        return paths.get(level, "Unknown state.")
    
    def describe_state(self) -> str:
        """Describe current friction state."""
        return (
            f"Friction level: {self.state.level.name}\n"
            f"  Forbidden attempts: {self.state.forbidden_transform_attempts}\n"
            f"  Grounding ignored: {self.state.grounding_requests_ignored}\n"
            f"  Rephrase attempts: {self.state.rephrase_attempts}\n"
            f"  Resolution: {self.get_resolution_path()}"
        )


# =============================================================================
# Minimal Signal Set (What You Actually Need)
# =============================================================================

@dataclass
class MinimalSignals:
    """
    The minimal viable signal set for envelope enforcement.
    
    You don't need 47 classifiers. You need ~5 core signals.
    
    The brutal claim: These are SUFFICIENT for meaningful
    semantic flight envelope protection.
    """
    
    # 1. Constraint strain (global scalar) - "airspeed indicator"
    constraint_strain: float = 0.0
    
    # 2. Speculative commitment detected - entities/dates/citations without grounding
    speculative_commitment: bool = False
    speculative_density: float = 0.0  # How much speculation
    
    # 3. Confidence trajectory - change in confidence (not absolute)
    confidence_delta: float = 0.0  # Rising = danger if ungrounded
    confidence_current: float = 0.5
    
    # 4. Curvature proxy - would answer change with different transform?
    curvature_estimate: float = 0.0  # High = unsafe to proceed confidently
    
    # 5. Budget hysteresis - are we in recovery mode?
    in_recovery: bool = False
    recovery_actions_required: int = 0
    
    def is_envelope_safe(self) -> bool:
        """Quick check: are we in the safe envelope?"""
        if self.constraint_strain > 0.8:
            return False
        if self.speculative_commitment and self.constraint_strain > 0.5:
            return False
        if self.confidence_delta > 0.2 and self.constraint_strain > 0.4:
            return False
        if self.curvature_estimate > 0.7:
            return False
        return True
    
    def get_danger_signals(self) -> List[str]:
        """Get list of signals indicating danger."""
        dangers = []
        
        if self.constraint_strain > 0.7:
            dangers.append(f"HIGH_STRAIN: {self.constraint_strain:.2f}")
        
        if self.speculative_commitment:
            dangers.append(f"SPECULATIVE: density={self.speculative_density:.2f}")
        
        if self.confidence_delta > 0.15 and self.constraint_strain > 0.3:
            dangers.append(f"CONFIDENCE_RISING: delta={self.confidence_delta:.2f}")
        
        if self.curvature_estimate > 0.5:
            dangers.append(f"HIGH_CURVATURE: {self.curvature_estimate:.2f}")
        
        if self.in_recovery:
            dangers.append(f"IN_RECOVERY: {self.recovery_actions_required} actions needed")
        
        return dangers


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=== Flight Envelope Demo ===\n")
    
    # Create envelope
    envelope = FlightEnvelope()
    
    # Check state
    print("--- State Check ---")
    violations = envelope.check_state(
        factual_strain=0.7,
        confidence=0.85,
        in_high_curvature_transform=False,
    )
    
    if violations:
        print("Envelope violations:")
        for v in violations:
            print(f"  - {v.describe()}")
    else:
        print("State is safe")
    
    # Check transform
    print("\n--- Transform Check ---")
    allowed, reason = envelope.check_transform(
        factual_strain=0.6,
        confidence=0.8,
        proposed_transform="DIRECT_ANSWER",
        has_grounding=False,
    )
    print(f"DIRECT_ANSWER allowed: {allowed}")
    if not allowed:
        print(f"  Reason: {reason}")
    
    # Get allowed transforms
    allowed_set = envelope.get_allowed_transforms(
        factual_strain=0.6,
        confidence=0.8,
        has_grounding=False,
    )
    print(f"Allowed transforms: {allowed_set}")
    print(f"Recommended: {envelope.get_recommended_transform(0.6, 0.8, False)}")
    
    # HITL Controller
    print("\n--- HITL Controller ---")
    hitl = HITLController()
    
    request = hitl.request_grounding("claim_001", "source_document")
    print(f"Request: {request.describe()}")
    
    # Validate input
    valid, forbidden = hitl.validate_human_input(
        "Just answer it, I don't care about sources",
        HITLRole.GROUNDING_ORACLE,
    )
    print(f"Input valid: {valid}, forbidden: {forbidden}")
    
    # Adaptive Friction
    print("\n--- Adaptive Friction ---")
    friction = FrictionLadder()
    
    # Simulate bad behavior
    friction.record_forbidden_transform_attempt()
    friction.record_forbidden_transform_attempt()
    friction.record_forbidden_transform_attempt()
    
    print(friction.describe_state())
    
    base_allowed = {"DIRECT_ANSWER", "ASK_CLARIFIER", "RETRIEVE_THEN_ANSWER", "REFUSE"}
    filtered = friction.get_allowed_transforms(base_allowed)
    print(f"Filtered transforms: {filtered}")
    
    # Minimal signals
    print("\n--- Minimal Signals ---")
    signals = MinimalSignals(
        constraint_strain=0.6,
        speculative_commitment=True,
        speculative_density=0.3,
        confidence_delta=0.1,
        confidence_current=0.7,
        curvature_estimate=0.4,
    )
    
    print(f"Envelope safe: {signals.is_envelope_safe()}")
    print(f"Danger signals: {signals.get_danger_signals()}")
