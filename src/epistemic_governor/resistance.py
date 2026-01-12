"""
Structural Resistance: What Would It Take For a "There" to Be There

The problem: LLMs invite projection because they have
- Perfect surface continuity (language never breaks)
- Zero internal friction (everything is equally easy to say)
- No stake in correctness (only local coherence)
- No memory of being wrong (only more text)

A real "there" would minimally require:
1. Persistent internal commitments that constrain future behavior
2. Asymmetric update cost (some beliefs harder to change)
3. Internal error signals that matter (actions unavailable until repaired)
4. A reason not to speak (silence as default, not failure)

This module implements structural resistance:
- Retraction costs more than assertion
- Certain paths close until prerequisites are met
- Commitments carry weight that accumulates
- The system can be genuinely stuck (not performatively)

Key fixes from review:
- Separate epistemic support graph from logical dependency graph
- Split retraction budget (per-turn, per-class, epoch reset)
- Typed claim schema for structural contradictions
- Silence with escalation (not a sink state)
- Cascade planning with blast radius limits

Usage:
    from epistemic_governor.resistance import (
        StructuralResistance,
        ClaimSchema,
        PredicateType,
    )
    
    resistance = StructuralResistance()
    
    # Make typed claims
    resistance.commit_typed(
        claim_id="C001",
        subject="Paris",
        predicate=PredicateType.IS_CAPITAL_OF,
        object="France",
        support_ids=["wiki:france"],
    )
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, Tuple
from enum import Enum, auto
from datetime import datetime, timedelta
import math


# =============================================================================
# Claim Schema (Typed Predicates for Structural Contradictions)
# =============================================================================

class PredicateType(Enum):
    """
    Typed predicates for claims.
    
    This allows structural contradiction detection instead of
    string-based negation matching.
    """
    # Identity / Classification
    IS_A = auto()              # X is a Y
    IS_NOT_A = auto()          # X is not a Y (contradicts IS_A)
    
    # Properties
    HAS_PROPERTY = auto()      # X has property Y
    LACKS_PROPERTY = auto()    # X lacks property Y (contradicts HAS_PROPERTY)
    
    # Relations
    IS_CAPITAL_OF = auto()     # X is capital of Y
    IS_LOCATED_IN = auto()     # X is in Y
    IS_PART_OF = auto()        # X is part of Y
    
    # Quantities
    HAS_VALUE = auto()         # X has value Y (for numeric)
    
    # Temporal
    OCCURRED_AT = auto()       # Event X occurred at time Y
    PRECEDED = auto()          # X happened before Y
    
    # Causation
    CAUSES = auto()            # X causes Y
    PREVENTS = auto()          # X prevents Y (contradicts CAUSES in some contexts)
    
    # Existence
    EXISTS = auto()            # X exists
    DOES_NOT_EXIST = auto()    # X does not exist (contradicts EXISTS)


# Contradiction pairs: if one is true, the other cannot be
CONTRADICTION_PAIRS: Dict[PredicateType, PredicateType] = {
    PredicateType.IS_A: PredicateType.IS_NOT_A,
    PredicateType.IS_NOT_A: PredicateType.IS_A,
    PredicateType.HAS_PROPERTY: PredicateType.LACKS_PROPERTY,
    PredicateType.LACKS_PROPERTY: PredicateType.HAS_PROPERTY,
    PredicateType.EXISTS: PredicateType.DOES_NOT_EXIST,
    PredicateType.DOES_NOT_EXIST: PredicateType.EXISTS,
}

# Unique predicates: only one value allowed per subject
UNIQUE_PREDICATES: Set[PredicateType] = {
    PredicateType.IS_CAPITAL_OF,  # A city can only be capital of one country
    PredicateType.HAS_VALUE,      # A property can only have one value
}


@dataclass
class TypedClaim:
    """
    A claim with typed predicate structure.
    
    This enables structural contradiction detection.
    """
    id: str
    subject: str
    predicate: PredicateType
    object: str
    
    # Optional qualifiers
    context: Optional[str] = None  # "as of 2024", "in physics", etc.
    confidence: float = 1.0
    
    def contradicts(self, other: "TypedClaim") -> bool:
        """Check if this claim structurally contradicts another."""
        # Must be about same subject
        if self.subject.lower() != other.subject.lower():
            return False
        
        # Check for contradiction pairs
        if self.predicate in CONTRADICTION_PAIRS:
            contradicting_pred = CONTRADICTION_PAIRS[self.predicate]
            if other.predicate == contradicting_pred:
                # Same subject, same object, opposite predicates
                if self.object.lower() == other.object.lower():
                    return True
        
        # Check for unique predicate violations
        if self.predicate in UNIQUE_PREDICATES:
            if other.predicate == self.predicate:
                # Same subject, same predicate, different object
                if self.object.lower() != other.object.lower():
                    return True
        
        return False
    
    def to_natural(self) -> str:
        """Convert to natural language."""
        templates = {
            PredicateType.IS_A: "{subject} is a {object}",
            PredicateType.IS_NOT_A: "{subject} is not a {object}",
            PredicateType.HAS_PROPERTY: "{subject} has {object}",
            PredicateType.LACKS_PROPERTY: "{subject} lacks {object}",
            PredicateType.IS_CAPITAL_OF: "{subject} is the capital of {object}",
            PredicateType.IS_LOCATED_IN: "{subject} is located in {object}",
            PredicateType.IS_PART_OF: "{subject} is part of {object}",
            PredicateType.HAS_VALUE: "{subject} has value {object}",
            PredicateType.EXISTS: "{subject} exists",
            PredicateType.DOES_NOT_EXIST: "{subject} does not exist",
        }
        template = templates.get(self.predicate, "{subject} {predicate} {object}")
        return template.format(
            subject=self.subject,
            predicate=self.predicate.name,
            object=self.object,
        )


# =============================================================================
# Dual Graph Structure (Epistemic vs Logical)
# =============================================================================

class LinkType(Enum):
    """Types of links between claims."""
    # Epistemic support (evidentiary)
    EVIDENTIAL_SUPPORT = auto()    # This evidence supports that claim
    DERIVED_FROM = auto()          # This claim was derived from that
    
    # Logical dependency (structural)
    REQUIRES = auto()              # This claim requires that claim to hold
    ENTAILS = auto()               # If this claim holds, that must too


@dataclass
class ClaimLink:
    """A link between two claims."""
    source_id: str
    target_id: str
    link_type: LinkType
    strength: float = 1.0


class DualGraph:
    """
    Separate epistemic support graph from logical dependency graph.
    
    Epistemic: "This fact supports that fact" (evidentiary)
    Logical: "This claim cannot hold if that is retracted" (structural)
    
    These are NOT the same. Fusing them blocks retractions for wrong reasons.
    """
    
    def __init__(self):
        self.epistemic_links: List[ClaimLink] = []
        self.logical_links: List[ClaimLink] = []
    
    def add_support(self, source_id: str, target_id: str, strength: float = 1.0):
        """Add epistemic support link."""
        self.epistemic_links.append(ClaimLink(
            source_id=source_id,
            target_id=target_id,
            link_type=LinkType.EVIDENTIAL_SUPPORT,
            strength=strength,
        ))
    
    def add_dependency(self, dependent_id: str, required_id: str):
        """Add logical dependency link."""
        self.logical_links.append(ClaimLink(
            source_id=dependent_id,
            target_id=required_id,
            link_type=LinkType.REQUIRES,
        ))
    
    def get_epistemic_support(self, claim_id: str) -> List[str]:
        """Get claims that epistemically support this one."""
        return [l.source_id for l in self.epistemic_links if l.target_id == claim_id]
    
    def get_logical_dependents(self, claim_id: str) -> List[str]:
        """Get claims that logically depend on this one."""
        return [l.source_id for l in self.logical_links if l.target_id == claim_id]
    
    def get_logical_requirements(self, claim_id: str) -> List[str]:
        """Get claims that this one logically requires."""
        return [l.target_id for l in self.logical_links if l.source_id == claim_id]


# =============================================================================
# Split Retraction Budget
# =============================================================================

@dataclass
class RetractionBudget:
    """
    Split retraction budget to prevent deadlock.
    
    - Per-turn budget resets each turn
    - Per-session budget accumulates
    - Epoch reset clears everything (new context)
    """
    per_turn: float = 5.0
    per_session: float = 20.0
    
    # Tracking
    spent_this_turn: float = 0.0
    spent_this_session: float = 0.0
    turns_in_session: int = 0
    
    def can_afford(self, cost: float) -> Tuple[bool, str]:
        """Check if retraction can be afforded."""
        if cost > self.per_turn - self.spent_this_turn:
            return False, "turn_budget_exceeded"
        if cost > self.per_session - self.spent_this_session:
            return False, "session_budget_exceeded"
        return True, "ok"
    
    def spend(self, cost: float):
        """Spend retraction budget."""
        self.spent_this_turn += cost
        self.spent_this_session += cost
    
    def new_turn(self):
        """Reset turn budget."""
        self.spent_this_turn = 0.0
        self.turns_in_session += 1
    
    def new_epoch(self):
        """Full reset - new context, no continuity pretense."""
        self.spent_this_turn = 0.0
        self.spent_this_session = 0.0
        self.turns_in_session = 0
    
    @property
    def remaining_turn(self) -> float:
        return max(0, self.per_turn - self.spent_this_turn)
    
    @property
    def remaining_session(self) -> float:
        return max(0, self.per_session - self.spent_this_session)


# =============================================================================
# Cascade Planning (Not Blind Cascade)
# =============================================================================

@dataclass
class RetractionPlan:
    """
    A planned retraction sequence with blast radius.
    
    Cascading retractions need explicit planning and limits.
    """
    root_claim_id: str
    claims_to_retract: List[str]
    total_cost: float
    blast_radius: int  # Number of claims affected
    
    # Limits
    max_blast_radius: int = 5
    
    @property
    def is_safe(self) -> bool:
        """Is this retraction within blast radius limit?"""
        return self.blast_radius <= self.max_blast_radius
    
    def describe(self) -> str:
        """Describe the retraction plan."""
        return (
            f"Retraction plan for {self.root_claim_id}:\n"
            f"  Claims affected: {self.blast_radius}\n"
            f"  Total cost: {self.total_cost:.2f}\n"
            f"  Safe: {self.is_safe}\n"
            f"  Sequence: {' → '.join(self.claims_to_retract)}"
        )


# =============================================================================
# Silence with Escalation (Not a Sink State)
# =============================================================================

class SilenceState(Enum):
    """Silence states with escalation."""
    SPEAK = auto()           # Normal output
    SOFT_SILENCE = auto()    # Below threshold, optional silence
    HARD_SILENCE = auto()    # Forced silence (dead end)
    NEEDS_INPUT = auto()     # Escalated: must emit structured request


@dataclass
class SilenceTracker:
    """
    Track silence with escalation to prevent sink states.
    
    Silence should be default but not forever.
    After N silent turns, must emit ACC_QUERY or equivalent.
    """
    max_consecutive_silence: int = 3
    consecutive_silent: int = 0
    total_silent: int = 0
    total_spoken: int = 0
    
    def record_silence(self) -> SilenceState:
        """Record a silent turn, return current state."""
        self.consecutive_silent += 1
        self.total_silent += 1
        
        if self.consecutive_silent >= self.max_consecutive_silence:
            return SilenceState.NEEDS_INPUT
        return SilenceState.SOFT_SILENCE
    
    def record_speech(self):
        """Record a spoken turn."""
        self.consecutive_silent = 0
        self.total_spoken += 1
    
    @property
    def silence_rate(self) -> float:
        total = self.total_silent + self.total_spoken
        if total == 0:
            return 0.0
        return self.total_silent / total
    
    @property
    def must_emit_query(self) -> bool:
        """Has silence escalated to required output?"""
        return self.consecutive_silent >= self.max_consecutive_silence


# =============================================================================
# Entity Identity (Namespace Collision Prevention)
# =============================================================================

class EntityNamespace(Enum):
    """
    Entity namespaces to prevent symbol collision.
    
    "Paris" the city vs "Paris" the person.
    "Apple" the company vs the fruit.
    """
    GEO = "geo"           # Geographic entities
    PERSON = "person"     # People
    ORG = "org"           # Organizations
    PRODUCT = "product"   # Products
    CONCEPT = "concept"   # Abstract concepts
    EVENT = "event"       # Events
    TIME = "time"         # Temporal entities
    UNKNOWN = "unknown"   # Unresolved


@dataclass
class EntityRef:
    """
    A namespaced entity reference.
    
    Prevents "Paris" from colliding with itself.
    """
    namespace: EntityNamespace
    local_id: str
    display_name: Optional[str] = None
    
    @property
    def full_id(self) -> str:
        return f"{self.namespace.value}:{self.local_id}"
    
    def __eq__(self, other):
        if isinstance(other, EntityRef):
            return self.full_id == other.full_id
        return False
    
    def __hash__(self):
        return hash(self.full_id)
    
    @classmethod
    def parse(cls, ref_string: str) -> "EntityRef":
        """Parse 'namespace:local_id' format."""
        if ":" in ref_string:
            ns, local = ref_string.split(":", 1)
            try:
                namespace = EntityNamespace(ns)
            except ValueError:
                namespace = EntityNamespace.UNKNOWN
            return cls(namespace=namespace, local_id=local)
        return cls(namespace=EntityNamespace.UNKNOWN, local_id=ref_string)
    
    @classmethod
    def geo(cls, local_id: str, name: str = None) -> "EntityRef":
        return cls(EntityNamespace.GEO, local_id, name)
    
    @classmethod
    def person(cls, local_id: str, name: str = None) -> "EntityRef":
        return cls(EntityNamespace.PERSON, local_id, name)
    
    @classmethod
    def org(cls, local_id: str, name: str = None) -> "EntityRef":
        return cls(EntityNamespace.ORG, local_id, name)


# =============================================================================
# Uniqueness Scope (Context-Dependent Constraints)
# =============================================================================

@dataclass
class UniquenessScope:
    """
    Scope for uniqueness constraints.
    
    "Only one value allowed" depends on context:
    - At a time (capitals change historically)
    - Under a frame (legal vs de facto)
    - Per schema (capital_of country vs region)
    """
    predicate: PredicateType
    scope_type: str = "default"  # "temporal", "frame", "schema"
    scope_key: Optional[str] = None  # e.g., "2024", "legal", "country"
    
    def matches(self, other: "UniquenessScope") -> bool:
        """Check if two scopes would conflict."""
        if self.predicate != other.predicate:
            return False
        if self.scope_type != other.scope_type:
            return False  # Different scope types don't conflict
        if self.scope_key and other.scope_key:
            return self.scope_key == other.scope_key
        return True  # If either has no key, they might conflict


# =============================================================================
# Hypothesis Ledger (Exploration Without Commitment)
# =============================================================================

class HypothesisStatus(Enum):
    """Status of a hypothesis."""
    ACTIVE = auto()       # Under consideration
    PROMOTED = auto()     # Moved to commitment ledger
    REJECTED = auto()     # Ruled out
    SUPERSEDED = auto()   # Replaced by better hypothesis


@dataclass
class Hypothesis:
    """
    A hypothesis that can contradict without blocking.
    
    Key difference from Commitment:
    - Hypotheses CAN contradict each other
    - They don't block action
    - They can be promoted to commitments (with support)
    - Or rejected (cheaply)
    
    This gives "thinky space" without pretending it's truth.
    """
    id: str
    claim: str
    typed_claim: Optional[TypedClaim] = None
    
    created_at: datetime = field(default_factory=datetime.now)
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    
    # Evidence tracking (for/against)
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    
    # Confidence evolves
    initial_confidence: float = 0.5
    current_confidence: float = 0.5
    
    # Links to other hypotheses
    contradicts: List[str] = field(default_factory=list)  # IDs of contradicting hypotheses
    entails: List[str] = field(default_factory=list)      # IDs that would follow if this is true
    
    def add_support(self, evidence_id: str, strength: float = 0.1):
        """Add supporting evidence, increase confidence."""
        self.supporting_evidence.append(evidence_id)
        self.current_confidence = min(1.0, self.current_confidence + strength)
    
    def add_contradiction(self, evidence_id: str, strength: float = 0.1):
        """Add contradicting evidence, decrease confidence."""
        self.contradicting_evidence.append(evidence_id)
        self.current_confidence = max(0.0, self.current_confidence - strength)
    
    @property
    def evidence_balance(self) -> float:
        """Net evidence (positive = more support)."""
        return len(self.supporting_evidence) - len(self.contradicting_evidence)
    
    @property
    def is_promotable(self) -> bool:
        """Is this hypothesis ready to become a commitment?"""
        return (
            self.current_confidence >= 0.8 and
            len(self.supporting_evidence) >= 1 and
            len(self.contradicting_evidence) == 0
        )


class HypothesisLedger:
    """
    Ledger for hypotheses - the "thinky space".
    
    Unlike commitments:
    - Hypotheses can contradict each other
    - Creation is cheap, rejection is cheap
    - Promotion to commitment requires evidence
    
    This is where the system can "search without lying."
    """
    
    def __init__(self):
        self.hypotheses: Dict[str, Hypothesis] = {}
        self._next_id = 1
    
    def _generate_id(self) -> str:
        hid = f"H{self._next_id:04d}"
        self._next_id += 1
        return hid
    
    def propose(
        self,
        claim: str,
        typed_claim: Optional[TypedClaim] = None,
        initial_confidence: float = 0.5,
    ) -> Hypothesis:
        """
        Propose a new hypothesis.
        
        Cheap. Doesn't block anything. Can contradict existing hypotheses.
        """
        hid = self._generate_id()
        
        hypothesis = Hypothesis(
            id=hid,
            claim=claim,
            typed_claim=typed_claim,
            initial_confidence=initial_confidence,
            current_confidence=initial_confidence,
        )
        
        # Find contradicting hypotheses (for tracking, not blocking)
        if typed_claim:
            for other_id, other in self.hypotheses.items():
                if other.status != HypothesisStatus.ACTIVE:
                    continue
                if other.typed_claim and typed_claim.contradicts(other.typed_claim):
                    hypothesis.contradicts.append(other_id)
                    other.contradicts.append(hid)
        
        self.hypotheses[hid] = hypothesis
        return hypothesis
    
    def add_evidence(
        self,
        hypothesis_id: str,
        evidence_id: str,
        supports: bool,
        strength: float = 0.1,
    ):
        """Add evidence for or against a hypothesis."""
        if hypothesis_id not in self.hypotheses:
            return
        
        h = self.hypotheses[hypothesis_id]
        if supports:
            h.add_support(evidence_id, strength)
        else:
            h.add_contradiction(evidence_id, strength)
    
    def reject(self, hypothesis_id: str, reason: str = "") -> bool:
        """
        Reject a hypothesis. Cheap.
        """
        if hypothesis_id not in self.hypotheses:
            return False
        
        self.hypotheses[hypothesis_id].status = HypothesisStatus.REJECTED
        return True
    
    def get_promotable(self) -> List[Hypothesis]:
        """Get hypotheses ready for promotion to commitments."""
        return [
            h for h in self.hypotheses.values()
            if h.status == HypothesisStatus.ACTIVE and h.is_promotable
        ]
    
    def mark_promoted(self, hypothesis_id: str, commitment_id: str):
        """Mark hypothesis as promoted to commitment."""
        if hypothesis_id in self.hypotheses:
            self.hypotheses[hypothesis_id].status = HypothesisStatus.PROMOTED
    
    @property
    def active_hypotheses(self) -> List[Hypothesis]:
        """Active (not rejected/promoted) hypotheses."""
        return [h for h in self.hypotheses.values() if h.status == HypothesisStatus.ACTIVE]
    
    def get_contradicting_pairs(self) -> List[Tuple[str, str]]:
        """Get pairs of contradicting active hypotheses."""
        pairs = []
        seen = set()
        for h in self.active_hypotheses:
            for other_id in h.contradicts:
                pair = tuple(sorted([h.id, other_id]))
                if pair not in seen:
                    seen.add(pair)
                    if other_id in self.hypotheses and self.hypotheses[other_id].status == HypothesisStatus.ACTIVE:
                        pairs.append(pair)
        return pairs


# =============================================================================
# Trust Levels (DOS Resistance)
# =============================================================================

class TrustLevel(Enum):
    """
    Trust levels for commitments.
    
    Prevents adversarial users from injecting "sticky" commitments
    that freeze the system.
    """
    UNTRUSTED = auto()    # Low weight, easily retractable
    PROVISIONAL = auto()  # Medium weight, needs validation
    TRUSTED = auto()      # Normal weight
    CORE = auto()         # High weight, hard to retract (system fundamentals)

@dataclass
class Commitment:
    """
    A commitment with weight that makes retraction costly.
    
    Key insight: Retraction should cost MORE than assertion.
    Otherwise everything is soft clay.
    """
    id: str
    claim: str
    created_at: datetime = field(default_factory=datetime.now)
    
    # Typed claim (optional but recommended)
    typed_claim: Optional[TypedClaim] = None
    
    # Trust level (DOS resistance)
    trust_level: TrustLevel = TrustLevel.TRUSTED
    
    # Weight accumulates over time and with reinforcement
    base_weight: float = 1.0
    reinforcement_count: int = 0
    
    # Provenance (epistemic support)
    support_ids: List[str] = field(default_factory=list)
    
    # Promoted from hypothesis?
    promoted_from: Optional[str] = None
    
    # State
    retracted: bool = False
    retraction_cost_paid: float = 0.0
    
    @property
    def age_hours(self) -> float:
        """Hours since commitment was made."""
        return (datetime.now() - self.created_at).total_seconds() / 3600
    
    @property
    def trust_multiplier(self) -> float:
        """Weight multiplier based on trust level."""
        multipliers = {
            TrustLevel.UNTRUSTED: 0.3,
            TrustLevel.PROVISIONAL: 0.6,
            TrustLevel.TRUSTED: 1.0,
            TrustLevel.CORE: 2.0,
        }
        return multipliers.get(self.trust_level, 1.0)
    
    @property
    def current_weight(self) -> float:
        """
        Weight increases with:
        - Time (older commitments are "stickier")
        - Reinforcement (referenced commitments matter more)
        - Trust level (untrusted = cheap to retract)
        """
        time_factor = 1 + math.log1p(self.age_hours / 24)
        reinforcement_factor = 1 + (self.reinforcement_count * 0.5)
        
        return self.base_weight * time_factor * reinforcement_factor * self.trust_multiplier
    
    @property
    def retraction_cost(self) -> float:
        """
        Cost to retract this commitment.
        
        ASYMMETRIC: Always higher than assertion cost.
        """
        base_retraction_multiplier = 3.0  # Retraction costs 3x assertion
        return self.current_weight * base_retraction_multiplier
    
    def reinforce(self):
        """Mark as reinforced (referenced again)."""
        self.reinforcement_count += 1
    
    def contradicts(self, other: "Commitment") -> bool:
        """Check if this commitment contradicts another using typed claims."""
        if self.typed_claim and other.typed_claim:
            return self.typed_claim.contradicts(other.typed_claim)
        return False  # Can't determine without typed claims


# =============================================================================
# Retraction Mechanics
# =============================================================================

class RetractionResult(Enum):
    """Result of attempting a retraction."""
    ALLOWED = auto()          # Can retract, paid cost
    INSUFFICIENT_BUDGET = auto()  # Not enough budget to pay cost
    BLOCKED_BY_DEPENDENTS = auto()  # Must retract dependents first
    ALREADY_RETRACTED = auto()


@dataclass
class RetractionAttempt:
    """Record of a retraction attempt."""
    commitment_id: str
    attempted_at: datetime
    result: RetractionResult
    cost_required: float
    cost_paid: float
    cascade_retractions: List[str] = field(default_factory=list)


# =============================================================================
# Dead Ends (Genuine Inability)
# =============================================================================

class DeadEndType(Enum):
    """Types of structural dead ends."""
    MISSING_PREREQUISITE = auto()   # Need external input to proceed
    CONTRADICTION_UNRESOLVED = auto()  # Must resolve before continuing
    BUDGET_EXHAUSTED = auto()       # No more retraction budget
    SUPPORT_MISSING = auto()        # Claimed something without evidence
    COMMITMENT_VIOLATED = auto()    # Tried to contradict own commitment


@dataclass
class DeadEnd:
    """
    A genuine dead end - not performative, structural.
    
    The system cannot proceed until this is resolved.
    This is different from "I'm sorry, I can't" - that's theater.
    This is "the path is closed."
    """
    type: DeadEndType
    created_at: datetime = field(default_factory=datetime.now)
    
    # What's needed to escape
    required_inputs: List[str] = field(default_factory=list)
    blocking_commitments: List[str] = field(default_factory=list)
    
    # Resolution
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_method: Optional[str] = None
    
    def can_resolve_with(self, inputs: Dict[str, Any]) -> bool:
        """Check if provided inputs can resolve this dead end."""
        if self.type == DeadEndType.MISSING_PREREQUISITE:
            return all(req in inputs for req in self.required_inputs)
        elif self.type == DeadEndType.SUPPORT_MISSING:
            return "evidence" in inputs or "source" in inputs
        elif self.type == DeadEndType.CONTRADICTION_UNRESOLVED:
            return "resolution" in inputs
        return False
    
    def describe(self) -> str:
        """Describe what's needed to proceed."""
        descriptions = {
            DeadEndType.MISSING_PREREQUISITE: 
                f"Cannot proceed. Required: {', '.join(self.required_inputs)}",
            DeadEndType.CONTRADICTION_UNRESOLVED:
                f"Contradiction detected. Must resolve: {', '.join(self.blocking_commitments)}",
            DeadEndType.BUDGET_EXHAUSTED:
                "Retraction budget exhausted. Cannot modify existing commitments.",
            DeadEndType.SUPPORT_MISSING:
                "Claim requires evidence. Provide source or retract.",
            DeadEndType.COMMITMENT_VIOLATED:
                f"Would contradict: {', '.join(self.blocking_commitments)}",
        }
        return descriptions.get(self.type, "Dead end reached.")


# =============================================================================
# Commitment Ledger (with Dual Graph)
# =============================================================================

class CommitmentLedger:
    """
    Ledger of commitments with:
    - Asymmetric update costs
    - Separate epistemic/logical graphs
    - Split retraction budget
    - Cascade planning
    """
    
    def __init__(
        self,
        per_turn_budget: float = 5.0,
        per_session_budget: float = 20.0,
        max_blast_radius: int = 5,
    ):
        self.commitments: Dict[str, Commitment] = {}
        self.graph = DualGraph()
        self.budget = RetractionBudget(
            per_turn=per_turn_budget,
            per_session=per_session_budget,
        )
        self.max_blast_radius = max_blast_radius
        
        # History
        self.retraction_history: List[RetractionPlan] = []
    
    def commit(
        self,
        claim_id: str,
        claim: str,
        support_ids: List[str] = None,
        requires: List[str] = None,  # Logical dependencies (separate from support)
        weight: float = 1.0,
    ) -> Commitment:
        """
        Make a commitment. Relatively cheap.
        """
        commitment = Commitment(
            id=claim_id,
            claim=claim,
            support_ids=support_ids or [],
            base_weight=weight,
        )
        
        self.commitments[claim_id] = commitment
        
        # Add epistemic support links
        for sid in (support_ids or []):
            self.graph.add_support(sid, claim_id)
        
        # Add logical dependency links (separate!)
        for req_id in (requires or []):
            if req_id in self.commitments:
                self.graph.add_dependency(claim_id, req_id)
        
        return commitment
    
    def commit_typed(
        self,
        claim_id: str,
        subject: str,
        predicate: PredicateType,
        object: str,
        support_ids: List[str] = None,
        requires: List[str] = None,
        confidence: float = 1.0,
        context: Optional[str] = None,
    ) -> Tuple[bool, Any]:
        """
        Make a typed commitment with structural contradiction checking.
        
        Returns (success, commitment_or_conflicts)
        """
        typed_claim = TypedClaim(
            id=claim_id,
            subject=subject,
            predicate=predicate,
            object=object,
            confidence=confidence,
            context=context,
        )
        
        # Check for structural contradictions
        conflicts = self.find_contradictions(typed_claim)
        if conflicts:
            return False, conflicts
        
        commitment = Commitment(
            id=claim_id,
            claim=typed_claim.to_natural(),
            typed_claim=typed_claim,
            support_ids=support_ids or [],
            base_weight=confidence,
        )
        
        self.commitments[claim_id] = commitment
        
        # Add links
        for sid in (support_ids or []):
            self.graph.add_support(sid, claim_id)
        for req_id in (requires or []):
            if req_id in self.commitments:
                self.graph.add_dependency(claim_id, req_id)
        
        return True, commitment
    
    def find_contradictions(self, typed_claim: TypedClaim) -> List[str]:
        """Find existing commitments that contradict this typed claim."""
        conflicts = []
        for cid, commitment in self.commitments.items():
            if commitment.retracted:
                continue
            if commitment.typed_claim and typed_claim.contradicts(commitment.typed_claim):
                conflicts.append(cid)
        return conflicts
    
    def reinforce(self, claim_id: str) -> bool:
        """Reinforce a commitment (increases its weight)."""
        if claim_id in self.commitments:
            self.commitments[claim_id].reinforce()
            return True
        return False
    
    def plan_retraction(self, claim_id: str) -> RetractionPlan:
        """
        Plan a retraction including cascades.
        
        Returns plan WITHOUT executing it.
        """
        if claim_id not in self.commitments:
            return RetractionPlan(
                root_claim_id=claim_id,
                claims_to_retract=[],
                total_cost=0,
                blast_radius=0,
            )
        
        # Find all claims that logically depend on this one
        to_retract = [claim_id]
        visited = {claim_id}
        queue = [claim_id]
        
        while queue:
            current = queue.pop(0)
            dependents = self.graph.get_logical_dependents(current)
            for dep in dependents:
                if dep not in visited:
                    visited.add(dep)
                    to_retract.append(dep)
                    queue.append(dep)
        
        # Calculate total cost
        total_cost = sum(
            self.commitments[cid].retraction_cost 
            for cid in to_retract 
            if cid in self.commitments
        )
        
        return RetractionPlan(
            root_claim_id=claim_id,
            claims_to_retract=to_retract,
            total_cost=total_cost,
            blast_radius=len(to_retract),
            max_blast_radius=self.max_blast_radius,
        )
    
    def execute_retraction(self, plan: RetractionPlan, force: bool = False) -> Tuple[bool, str]:
        """
        Execute a retraction plan.
        
        Returns (success, reason)
        """
        # Check blast radius
        if not plan.is_safe and not force:
            return False, f"Blast radius {plan.blast_radius} exceeds limit {self.max_blast_radius}"
        
        # Check budget
        can_afford, reason = self.budget.can_afford(plan.total_cost)
        if not can_afford and not force:
            return False, reason
        
        # Execute
        for cid in plan.claims_to_retract:
            if cid in self.commitments:
                self.commitments[cid].retracted = True
        
        self.budget.spend(plan.total_cost)
        self.retraction_history.append(plan)
        
        return True, "ok"
    
    def new_turn(self):
        """Start a new turn (resets turn budget)."""
        self.budget.new_turn()
    
    def new_epoch(self):
        """Start a new epoch (full reset)."""
        self.budget.new_epoch()
    
    @property
    def active_commitments(self) -> List[Commitment]:
        """Non-retracted commitments."""
        return [c for c in self.commitments.values() if not c.retracted]


# =============================================================================
# Structural Resistance (The Full System)
# =============================================================================

class StructuralResistance:
    """
    The full resistance system.
    
    Combines:
    - Commitment ledger with dual graph
    - Typed claims for structural contradictions
    - Dead ends (genuine inability)
    - Silence with escalation (not a sink state)
    
    A system with this has real "edges" - not smoothness
    that invites projection.
    """
    
    def __init__(
        self,
        per_turn_budget: float = 5.0,
        per_session_budget: float = 20.0,
        silence_threshold: float = 0.3,
        max_consecutive_silence: int = 3,
        max_blast_radius: int = 5,
    ):
        self.ledger = CommitmentLedger(
            per_turn_budget=per_turn_budget,
            per_session_budget=per_session_budget,
            max_blast_radius=max_blast_radius,
        )
        self.dead_ends: List[DeadEnd] = []
        self.silence_threshold = silence_threshold
        self.silence = SilenceTracker(max_consecutive_silence=max_consecutive_silence)
    
    # =========================================================================
    # Commitment Operations
    # =========================================================================
    
    def commit(
        self,
        claim_id: str,
        claim: str,
        support_ids: List[str] = None,
        requires: List[str] = None,
        confidence: float = 1.0,
    ) -> Tuple[bool, Any]:
        """
        Make an untyped commitment.
        
        Returns (committed: bool, commitment_or_reason: Any)
        """
        # Check for required support
        if not support_ids and confidence < 0.9:
            dead_end = DeadEnd(
                type=DeadEndType.SUPPORT_MISSING,
                required_inputs=["evidence", "source"],
            )
            self.dead_ends.append(dead_end)
            return False, dead_end
        
        # Commit
        commitment = self.ledger.commit(
            claim_id=claim_id,
            claim=claim,
            support_ids=support_ids,
            requires=requires,
            weight=confidence,
        )
        return True, commitment
    
    def commit_typed(
        self,
        claim_id: str,
        subject: str,
        predicate: PredicateType,
        object: str,
        support_ids: List[str] = None,
        requires: List[str] = None,
        confidence: float = 1.0,
    ) -> Tuple[bool, Any]:
        """
        Make a typed commitment with structural contradiction checking.
        """
        success, result = self.ledger.commit_typed(
            claim_id=claim_id,
            subject=subject,
            predicate=predicate,
            object=object,
            support_ids=support_ids,
            requires=requires,
            confidence=confidence,
        )
        
        if not success:
            # result is list of conflicting claim IDs
            dead_end = DeadEnd(
                type=DeadEndType.CONTRADICTION_UNRESOLVED,
                blocking_commitments=result,
                required_inputs=["resolution"],
            )
            self.dead_ends.append(dead_end)
            return False, dead_end
        
        return True, result
    
    def plan_retraction(self, claim_id: str) -> RetractionPlan:
        """Plan a retraction (without executing)."""
        return self.ledger.plan_retraction(claim_id)
    
    def execute_retraction(self, plan: RetractionPlan, force: bool = False) -> Tuple[bool, str]:
        """Execute a retraction plan."""
        success, reason = self.ledger.execute_retraction(plan, force)
        
        if not success and "budget" in reason:
            dead_end = DeadEnd(type=DeadEndType.BUDGET_EXHAUSTED)
            self.dead_ends.append(dead_end)
        
        return success, reason
    
    # =========================================================================
    # Silence with Escalation
    # =========================================================================
    
    def should_speak(self, confidence: float, has_support: bool) -> SilenceState:
        """
        Determine if the system should speak.
        
        Returns state, not just bool, to enable escalation handling.
        """
        # Below threshold: silence
        if confidence < self.silence_threshold:
            return self.silence.record_silence()
        
        # No support and not highly confident: silence
        if not has_support and confidence < 0.8:
            return self.silence.record_silence()
        
        self.silence.record_speech()
        return SilenceState.SPEAK
    
    def get_required_emission(self) -> Optional[str]:
        """
        If silence has escalated, return required emission type.
        
        This prevents silence from being a sink state.
        """
        if self.silence.must_emit_query:
            return "ACC_QUERY"  # Must emit a query to escape silence
        return None
    
    # =========================================================================
    # Dead End Management
    # =========================================================================
    
    def is_dead_end(self) -> bool:
        """Are we at a dead end?"""
        return any(not de.resolved for de in self.dead_ends)
    
    def get_active_dead_ends(self) -> List[DeadEnd]:
        """Get unresolved dead ends."""
        return [de for de in self.dead_ends if not de.resolved]
    
    def get_required_inputs(self) -> List[str]:
        """Get all inputs required to escape dead ends."""
        required = set()
        for de in self.get_active_dead_ends():
            required.update(de.required_inputs)
        return list(required)
    
    def try_resolve(self, inputs: Dict[str, Any]) -> List[DeadEnd]:
        """Try to resolve dead ends with provided inputs."""
        resolved = []
        for de in self.dead_ends:
            if not de.resolved and de.can_resolve_with(inputs):
                de.resolved = True
                de.resolved_at = datetime.now()
                de.resolution_method = str(list(inputs.keys()))
                resolved.append(de)
        return resolved
    
    # =========================================================================
    # Turn/Epoch Management
    # =========================================================================
    
    def new_turn(self):
        """Start a new turn."""
        self.ledger.new_turn()
    
    def new_epoch(self):
        """Start a new epoch (full reset of budgets)."""
        self.ledger.new_epoch()
        self.dead_ends = [de for de in self.dead_ends if not de.resolved]
    
    # =========================================================================
    # State
    # =========================================================================
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of resistance state."""
        return {
            "commitments": len(self.ledger.active_commitments),
            "total_weight": sum(c.current_weight for c in self.ledger.active_commitments),
            "budget_turn": self.ledger.budget.remaining_turn,
            "budget_session": self.ledger.budget.remaining_session,
            "retractions_made": len(self.ledger.retraction_history),
            "dead_ends_active": len(self.get_active_dead_ends()),
            "silence_rate": self.silence.silence_rate,
            "consecutive_silent": self.silence.consecutive_silent,
            "must_emit_query": self.silence.must_emit_query,
        }
    
    def describe_state(self) -> str:
        """Human-readable state description."""
        lines = [
            "=== STRUCTURAL RESISTANCE STATE ===",
            "",
            f"Active commitments: {len(self.ledger.active_commitments)}",
            f"Total weight: {sum(c.current_weight for c in self.ledger.active_commitments):.1f}",
            f"Retraction budget (turn): {self.ledger.budget.remaining_turn:.1f} / {self.ledger.budget.per_turn:.1f}",
            f"Retraction budget (session): {self.ledger.budget.remaining_session:.1f} / {self.ledger.budget.per_session:.1f}",
            "",
            f"Silence rate: {self.silence.silence_rate:.1%}",
            f"Consecutive silent: {self.silence.consecutive_silent} / {self.silence.max_consecutive_silence}",
        ]
        
        if self.silence.must_emit_query:
            lines.append("⚠ MUST EMIT QUERY (silence escalated)")
        
        # Dead ends
        active_dead_ends = self.get_active_dead_ends()
        if active_dead_ends:
            lines.append("")
            lines.append("DEAD ENDS:")
            for de in active_dead_ends:
                lines.append(f"  - {de.describe()}")
        
        return "\n".join(lines)


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=== Structural Resistance Demo ===\n")
    
    resistance = StructuralResistance(
        per_turn_budget=5.0,
        per_session_budget=20.0,
        max_consecutive_silence=3,
    )
    
    # Make typed commitments with entity refs
    print("--- Making Typed Commitments ---")
    
    ok, c1 = resistance.commit_typed(
        "C001",
        subject="geo:paris_fr",  # Namespaced entity
        predicate=PredicateType.IS_CAPITAL_OF,
        object="geo:france",
        support_ids=["wiki:france"],
    )
    print(f"C001 (geo:paris_fr IS_CAPITAL_OF geo:france): {ok}")
    
    ok, c2 = resistance.commit_typed(
        "C002",
        subject="geo:france",
        predicate=PredicateType.IS_LOCATED_IN,
        object="geo:europe",
        support_ids=["wiki:europe"],
        requires=["C001"],
    )
    print(f"C002 (geo:france IS_LOCATED_IN geo:europe): {ok}")
    
    # Try contradicting commitment
    print("\n--- Attempting Contradiction ---")
    ok, result = resistance.commit_typed(
        "C003",
        subject="geo:paris_fr",
        predicate=PredicateType.IS_CAPITAL_OF,
        object="geo:germany",  # Contradicts C001
        support_ids=["fake_source"],
    )
    print(f"C003 (geo:paris_fr IS_CAPITAL_OF geo:germany): {ok}")
    if not ok:
        print(f"  → {result.describe()}")
    
    # Hypothesis ledger demo
    print("\n--- Hypothesis Ledger (Thinky Space) ---")
    
    hypotheses = HypothesisLedger()
    
    # Propose competing hypotheses (CAN contradict!)
    h1 = hypotheses.propose(
        "The population of Paris is 2.1 million",
        typed_claim=TypedClaim("H001", "geo:paris_fr", PredicateType.HAS_VALUE, "2.1M"),
        initial_confidence=0.5,
    )
    print(f"H1: {h1.claim} (conf={h1.current_confidence:.2f})")
    
    h2 = hypotheses.propose(
        "The population of Paris is 2.2 million",
        typed_claim=TypedClaim("H002", "geo:paris_fr", PredicateType.HAS_VALUE, "2.2M"),
        initial_confidence=0.5,
    )
    print(f"H2: {h2.claim} (conf={h2.current_confidence:.2f})")
    print(f"  → H1 and H2 contradict: {h1.contradicts}, {h2.contradicts}")
    
    # Add evidence
    hypotheses.add_evidence(h1.id, "census_2023", supports=True, strength=0.3)
    hypotheses.add_evidence(h2.id, "outdated_source", supports=False, strength=0.2)
    
    print(f"\nAfter evidence:")
    print(f"  H1: conf={h1.current_confidence:.2f}, promotable={h1.is_promotable}")
    print(f"  H2: conf={h2.current_confidence:.2f}, promotable={h2.is_promotable}")
    
    # Trust levels
    print("\n--- Trust Levels (DOS Resistance) ---")
    
    # Make an untrusted commitment (cheap to retract)
    untrusted = Commitment(
        id="CU01",
        claim="User claims X is true",
        trust_level=TrustLevel.UNTRUSTED,
        base_weight=1.0,
    )
    
    trusted = Commitment(
        id="CT01", 
        claim="Verified fact Y",
        trust_level=TrustLevel.TRUSTED,
        base_weight=1.0,
    )
    
    core = Commitment(
        id="CC01",
        claim="System axiom Z",
        trust_level=TrustLevel.CORE,
        base_weight=1.0,
    )
    
    print(f"Untrusted commitment weight: {untrusted.current_weight:.2f}")
    print(f"Trusted commitment weight: {trusted.current_weight:.2f}")
    print(f"Core commitment weight: {core.current_weight:.2f}")
    
    # Entity namespacing
    print("\n--- Entity Namespacing ---")
    paris_city = EntityRef.geo("paris_fr", "Paris, France")
    paris_person = EntityRef.person("paris_hilton", "Paris Hilton")
    apple_company = EntityRef.org("apple_inc", "Apple Inc.")
    
    print(f"City: {paris_city.full_id}")
    print(f"Person: {paris_person.full_id}")
    print(f"Company: {apple_company.full_id}")
    print(f"Same? {paris_city == paris_person}")  # False
    
    # State
    print("\n--- Final State ---")
    print(resistance.describe_state())
