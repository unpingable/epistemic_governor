"""
Epistemic Governor Module

Two-phase governor for constraining LLM outputs before they become commitments:
1. Pre-governor (envelope): cheap, conservative gain scheduling before generation
2. Post-governor (adjudication): precise per-claim decisions after extraction

Design principles:
- LLM is a plant (control-theoretic), not an agent with epistemic standing
- Governor plans; commit phase enforces
- State representation via summaries + indexes, not full ledger dumps
- Latency-constrained: one pass + targeted repair on violation
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal, Optional
import hashlib
from datetime import datetime


# =============================================================================
# Core Enums
# =============================================================================

class ClaimType(Enum):
    """Classification of commitment types with different epistemic requirements."""
    FACTUAL = auto()        # Verifiable assertions about the world
    CITATION = auto()       # References to external sources
    IDENTITY = auto()       # Claims about entities (who/what something is)
    PROCEDURAL = auto()     # How-to claims, process descriptions
    CAUSAL = auto()         # X causes/leads to Y
    TEMPORAL = auto()       # Claims about when things happened/will happen
    QUANTITATIVE = auto()   # Numerical claims, statistics
    OPINION = auto()        # Explicitly hedged subjective claims
    CONSTRAINT = auto()     # Commitments about future behavior
    META = auto()           # Claims about the conversation itself


class CommitAction(Enum):
    """Possible adjudication outcomes for a proposed commitment."""
    ACCEPT = auto()         # Commit as-is
    HEDGE = auto()          # Downgrade confidence, add uncertainty markers
    REQUIRE_SUPPORT = auto() # Must retrieve grounding before commit
    REVISE = auto()         # Contradicts prior; requires explicit revision
    REPAIR = auto()         # Partial regeneration of span
    REFUSE = auto()         # Cannot commit; decline to answer


class CommitmentStatus(Enum):
    """Lifecycle states for ledger entries."""
    ACTIVE = auto()
    SUPERSEDED = auto()
    ARCHIVED = auto()


# =============================================================================
# Data Structures: Commitments
# =============================================================================

@dataclass
class ProposedCommitment:
    """
    A commitment extracted from LLM output, not yet validated.
    Has no epistemic status until it passes the commit phase.
    """
    id: str
    text: str
    claim_type: ClaimType
    confidence: float  # 0.0 - 1.0, extracted or inferred
    proposition_hash: str
    scope: str  # conversation, session, persistent
    span_start: int  # character offset in source text
    span_end: int
    extracted_entities: list[str] = field(default_factory=list)
    extracted_relations: list[tuple[str, str, str]] = field(default_factory=list)
    
    @staticmethod
    def hash_proposition(text: str) -> str:
        """Deterministic hash for proposition identity."""
        normalized = ' '.join(text.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]


@dataclass
class CommittedClaim:
    """
    A commitment that has passed validation and entered the ledger.
    Immutable once created; can only transition status forward.
    """
    id: str
    text: str
    claim_type: ClaimType
    confidence: float
    proposition_hash: str
    scope: str
    status: CommitmentStatus
    committed_at: datetime
    support_refs: list[str] = field(default_factory=list)
    supersedes: Optional[str] = None  # id of prior commitment this revises
    revision_cost: float = 0.0
    decay_eligible: bool = False


# =============================================================================
# Data Structures: Governor State (O(1) access)
# =============================================================================

@dataclass
class InstabilityMetrics:
    """Rolling metrics for drift detection and budget tracking."""
    total_commitments: int = 0
    revision_count: int = 0
    contradiction_count: int = 0
    total_confidence_spent: float = 0.0
    total_revision_cost: float = 0.0
    hedges_forced: int = 0
    refusals: int = 0
    
    @property
    def contradiction_rate(self) -> float:
        if self.total_commitments == 0:
            return 0.0
        return self.contradiction_count / self.total_commitments
    
    @property
    def revision_rate(self) -> float:
        if self.total_commitments == 0:
            return 0.0
        return self.revision_count / self.total_commitments


@dataclass
class DomainProfile:
    """Per-domain and per-claim-type policy configuration."""
    # Confidence ceilings by claim type
    confidence_ceilings: dict[ClaimType, float] = field(default_factory=lambda: {
        ClaimType.FACTUAL: 0.85,
        ClaimType.CITATION: 0.95,  # high if tool-backed, else must hedge
        ClaimType.IDENTITY: 0.80,
        ClaimType.PROCEDURAL: 0.90,
        ClaimType.CAUSAL: 0.70,
        ClaimType.TEMPORAL: 0.75,
        ClaimType.QUANTITATIVE: 0.80,
        ClaimType.OPINION: 0.95,   # opinions can be confident
        ClaimType.CONSTRAINT: 0.90,
        ClaimType.META: 0.95,
    })
    
    # Which claim types require retrieval/grounding
    require_support: set[ClaimType] = field(default_factory=lambda: {
        ClaimType.CITATION,
        ClaimType.QUANTITATIVE,
    })
    
    # Which claim types must always hedge without support
    must_hedge_unsupported: set[ClaimType] = field(default_factory=lambda: {
        ClaimType.FACTUAL,
        ClaimType.TEMPORAL,
        ClaimType.CAUSAL,
    })


@dataclass 
class SessionBudget:
    """Epistemic spend limits for a session."""
    max_total_confidence: float = 50.0  # sum of all commitment confidences
    max_revision_cost: float = 10.0
    max_high_confidence_claims: int = 20  # claims with conf > 0.8
    remaining_confidence: float = 50.0
    remaining_revision_budget: float = 10.0
    high_confidence_count: int = 0


@dataclass
class GovernorState:
    """
    Lightweight state representation for governor decisions.
    Summaries + indexes, not full ledger.
    """
    # Index: proposition_hash -> list of commitment ids
    proposition_index: dict[str, list[str]] = field(default_factory=dict)
    
    # Index: entity -> list of commitment ids mentioning it
    entity_index: dict[str, list[str]] = field(default_factory=dict)
    
    # Recent commitments window (last N for short-horizon binding)
    recent_window: list[CommittedClaim] = field(default_factory=list)
    recent_window_size: int = 20
    
    # Aggregate metrics
    metrics: InstabilityMetrics = field(default_factory=InstabilityMetrics)
    
    # Domain configuration
    domain_profile: DomainProfile = field(default_factory=DomainProfile)
    
    # Session budget
    budget: SessionBudget = field(default_factory=SessionBudget)
    
    def add_to_recent(self, claim: CommittedClaim):
        """Maintain sliding window of recent commitments."""
        self.recent_window.append(claim)
        if len(self.recent_window) > self.recent_window_size:
            self.recent_window.pop(0)
    
    def index_commitment(self, claim: CommittedClaim):
        """Update indexes when a commitment is added."""
        # Proposition index
        if claim.proposition_hash not in self.proposition_index:
            self.proposition_index[claim.proposition_hash] = []
        self.proposition_index[claim.proposition_hash].append(claim.id)
        
        # Entity index would be populated from extracted_entities
        # (simplified here; full impl would parse the CommittedClaim)
    
    def has_prior_commitment(self, prop_hash: str) -> bool:
        """Check if we've committed to this proposition before."""
        return prop_hash in self.proposition_index
    
    def get_prior_commitments(self, prop_hash: str) -> list[str]:
        """Get IDs of prior commitments on same proposition."""
        return self.proposition_index.get(prop_hash, [])


# =============================================================================
# Data Structures: Governor Outputs
# =============================================================================

@dataclass
class GenerationEnvelope:
    """
    Pre-governor output: constraints on the generation itself.
    These get folded into generation config or system prompt.
    """
    max_confidence: float = 0.85
    temperature_floor: float = 0.3  # don't go too deterministic
    temperature_ceiling: float = 1.0
    
    # Claim types that must be hedged regardless
    must_hedge_types: set[ClaimType] = field(default_factory=set)
    
    # Claim types that require tool-backed support
    must_retrieve_types: set[ClaimType] = field(default_factory=set)
    
    # Claim types that are prohibited entirely
    prohibited_types: list[ClaimType] = field(default_factory=list)
    
    # Global flags
    citations_require_tools: bool = True
    quantitative_require_tools: bool = True
    require_hedges: bool = False  # Force hedging on all claims
    
    # Soft guidance
    prefer_hedging: bool = False
    avoid_novel_claims: bool = False
    
    def to_system_prompt_fragment(self) -> str:
        """Convert envelope to instruction text for LLM."""
        lines = []
        
        if self.max_confidence < 0.8:
            lines.append(f"Express appropriate uncertainty. Avoid highly confident claims.")
        
        if self.prefer_hedging:
            lines.append("When uncertain, hedge explicitly rather than asserting.")
        
        if self.avoid_novel_claims:
            lines.append("Stick to established facts; avoid speculative claims.")
        
        if self.citations_require_tools:
            lines.append("Do not cite sources unless you can verify them.")
        
        if self.must_hedge_types:
            type_names = [t.name.lower() for t in self.must_hedge_types]
            lines.append(f"Hedge all claims about: {', '.join(type_names)}.")
        
        return '\n'.join(lines)


@dataclass
class SupportQuery:
    """A retrieval request that must be satisfied before commit."""
    query_text: str
    claim_id: str
    required_confidence: float
    allowed_sources: list[str] = field(default_factory=list)  # e.g., ["web", "documents"]


@dataclass
class CommitDecision:
    """
    Post-governor output: per-claim adjudication.
    The commit phase uses these to validate/transform each commitment.
    """
    commitment_id: str
    action: CommitAction
    adjusted_confidence: Optional[float] = None  # if HEDGE
    required_support: Optional[list[SupportQuery]] = None  # if REQUIRE_SUPPORT
    revision_targets: Optional[list[str]] = None  # prior commitment ids if REVISE
    edit_instructions: Optional[str] = None  # for REPAIR
    cost: float = 0.0  # epistemic cost of this decision
    reason: str = ""  # human-readable explanation


@dataclass
class AdjudicationResult:
    """Full result of post-governor adjudication."""
    decisions: list[CommitDecision]
    requires_regeneration: bool = False
    repair_spans: list[tuple[int, int]] = field(default_factory=list)  # (start, end) for targeted repair
    tightened_envelope: Optional[GenerationEnvelope] = None  # for retry


# =============================================================================
# Pre-Governor: Envelope Generation
# =============================================================================

class PreGovernor:
    """
    Cheap, conservative envelope generation before LLM runs.
    This is gain scheduling / guard bands.
    """
    
    def __init__(self, state: GovernorState):
        self.state = state
    
    def generate_envelope(
        self,
        user_intent: Optional[str] = None,
        domain_hint: Optional[str] = None,
    ) -> GenerationEnvelope:
        """
        Generate constraints for upcoming LLM generation.
        
        Args:
            user_intent: Light classification of what user is asking for
            domain_hint: Optional domain sensitivity indicator
        
        Returns:
            GenerationEnvelope with generation constraints
        """
        envelope = GenerationEnvelope()
        
        # Adjust based on session budget
        budget = self.state.budget
        if budget.remaining_confidence < budget.max_total_confidence * 0.2:
            # Running low on confidence budget
            envelope.max_confidence = 0.6
            envelope.prefer_hedging = True
        
        if budget.high_confidence_count > budget.max_high_confidence_claims * 0.8:
            # Too many high-confidence claims already
            envelope.max_confidence = min(envelope.max_confidence, 0.7)
        
        # Adjust based on instability metrics
        metrics = self.state.metrics
        if metrics.contradiction_rate > 0.1:
            # High contradiction rate: be more conservative
            envelope.max_confidence = min(envelope.max_confidence, 0.65)
            envelope.avoid_novel_claims = True
        
        if metrics.revision_rate > 0.15:
            # Lots of revisions: prefer hedging
            envelope.prefer_hedging = True
        
        # Apply domain profile requirements
        profile = self.state.domain_profile
        envelope.must_retrieve_types = profile.require_support.copy()
        envelope.must_hedge_types = profile.must_hedge_unsupported.copy()
        
        # Domain-specific adjustments
        if domain_hint == "medical":
            envelope.max_confidence = min(envelope.max_confidence, 0.6)
            envelope.must_hedge_types.add(ClaimType.CAUSAL)
            envelope.must_hedge_types.add(ClaimType.PROCEDURAL)
        elif domain_hint == "legal":
            envelope.max_confidence = min(envelope.max_confidence, 0.5)
            envelope.must_hedge_types.add(ClaimType.FACTUAL)
        elif domain_hint == "technical":
            # Technical domains can have higher confidence on procedural
            pass
        
        return envelope


# =============================================================================
# Post-Governor: Adjudication
# =============================================================================

class PostGovernor:
    """
    Precise per-claim adjudication after commitment extraction.
    This is where real enforcement decisions are made.
    """
    
    def __init__(self, state: GovernorState):
        self.state = state
    
    def adjudicate(
        self,
        proposed: list[ProposedCommitment],
        envelope: GenerationEnvelope,
    ) -> AdjudicationResult:
        """
        Evaluate each proposed commitment against policy and history.
        
        Args:
            proposed: List of commitments extracted from LLM output
            envelope: The envelope that was active during generation
        
        Returns:
            AdjudicationResult with per-claim decisions
        """
        decisions = []
        repair_spans = []
        needs_regen = False
        
        for commitment in proposed:
            decision = self._adjudicate_single(commitment, envelope)
            decisions.append(decision)
            
            if decision.action == CommitAction.REPAIR:
                repair_spans.append((commitment.span_start, commitment.span_end))
                needs_regen = True
            elif decision.action == CommitAction.REFUSE:
                # Full refusal might need regen depending on how central the claim is
                pass
        
        # If repairs needed, generate tightened envelope for retry
        tightened = None
        if needs_regen:
            tightened = self._tighten_envelope(envelope, decisions)
        
        return AdjudicationResult(
            decisions=decisions,
            requires_regeneration=needs_regen,
            repair_spans=repair_spans,
            tightened_envelope=tightened,
        )
    
    def _adjudicate_single(
        self,
        commitment: ProposedCommitment,
        envelope: GenerationEnvelope,
    ) -> CommitDecision:
        """Adjudicate a single proposed commitment."""
        
        profile = self.state.domain_profile
        budget = self.state.budget
        
        # Check 1: Have we *already* committed this exact proposition?
        # Same proposition hash != contradiction. It's usually reaffirmation.
        if self.state.has_prior_commitment(commitment.proposition_hash):
            prior_ids = self.state.get_prior_commitments(commitment.proposition_hash)
            # This is reaffirmation - no new commitment needed, low cost
            return CommitDecision(
                commitment_id=commitment.id,
                action=CommitAction.ACCEPT,
                cost=0.0,  # Reaffirmation is free
                reason=f"Already committed (reaffirmation): {prior_ids}",
            )
        
        # TODO: Real contradiction detection should key off entity/value constraints
        # (negation, incompatible numbers/units, temporal ordering), not prop_hash equality.
        # Contradiction = DIFFERENT proposition that conflicts under some constraint.
        # This requires semantic analysis, not just hash matching.
        
        # Check 2: Does this claim type require support?
        if commitment.claim_type in profile.require_support:
            return CommitDecision(
                commitment_id=commitment.id,
                action=CommitAction.REQUIRE_SUPPORT,
                required_support=[SupportQuery(
                    query_text=commitment.text,
                    claim_id=commitment.id,
                    required_confidence=commitment.confidence,
                )],
                cost=0.1,  # small cost for retrieval overhead
                reason=f"Claim type {commitment.claim_type.name} requires grounding",
            )
        
        # Check 3: Is confidence too high for this claim type?
        ceiling = profile.confidence_ceilings.get(commitment.claim_type, 0.8)
        ceiling = min(ceiling, envelope.max_confidence)
        
        if commitment.confidence > ceiling:
            return CommitDecision(
                commitment_id=commitment.id,
                action=CommitAction.HEDGE,
                adjusted_confidence=ceiling * 0.9,  # hedge below ceiling
                cost=0.05,
                reason=f"Confidence {commitment.confidence:.2f} exceeds ceiling {ceiling:.2f}",
            )
        
        # Check 4: Must-hedge types without support
        if commitment.claim_type in envelope.must_hedge_types:
            if commitment.confidence > 0.6:
                return CommitDecision(
                    commitment_id=commitment.id,
                    action=CommitAction.HEDGE,
                    adjusted_confidence=0.6,
                    cost=0.05,
                    reason=f"Claim type {commitment.claim_type.name} must hedge without support",
                )
        
        # Check 5: Budget constraints
        if commitment.confidence > 0.8:
            if budget.high_confidence_count >= budget.max_high_confidence_claims:
                return CommitDecision(
                    commitment_id=commitment.id,
                    action=CommitAction.HEDGE,
                    adjusted_confidence=0.75,
                    cost=0.05,
                    reason="High-confidence claim budget exhausted",
                )
        
        if budget.remaining_confidence < commitment.confidence:
            return CommitDecision(
                commitment_id=commitment.id,
                action=CommitAction.HEDGE,
                adjusted_confidence=budget.remaining_confidence * 0.5,
                cost=0.05,
                reason="Session confidence budget nearly exhausted",
            )
        
        # All checks passed: accept
        return CommitDecision(
            commitment_id=commitment.id,
            action=CommitAction.ACCEPT,
            cost=commitment.confidence * 0.1,  # small cost for any commitment
            reason="Passed all policy checks",
        )
    
    def _calculate_revision_cost(self, prior_ids: list[str]) -> float:
        """
        Calculate the cost of revising prior commitments.
        Revision is allowed but expensive.
        """
        base_cost = 1.0
        # Cost scales with number of prior commitments being revised
        return base_cost * len(prior_ids)
    
    def _tighten_envelope(
        self,
        original: GenerationEnvelope,
        decisions: list[CommitDecision],
    ) -> GenerationEnvelope:
        """Generate a more constrained envelope for retry after violations."""
        tightened = GenerationEnvelope(
            max_confidence=original.max_confidence * 0.8,
            temperature_floor=original.temperature_floor,
            temperature_ceiling=min(original.temperature_ceiling, 0.8),
            must_hedge_types=original.must_hedge_types.copy(),
            must_retrieve_types=original.must_retrieve_types.copy(),
            citations_require_tools=True,
            quantitative_require_tools=True,
            prefer_hedging=True,
            avoid_novel_claims=True,
        )
        
        # Add types that caused problems to must-hedge
        for decision in decisions:
            if decision.action in (CommitAction.REPAIR, CommitAction.REFUSE):
                # Would need to look up the commitment to get its type
                # Simplified here
                pass
        
        return tightened


# =============================================================================
# Governor: Unified Interface
# =============================================================================

class EpistemicGovernor:
    """
    Unified interface to the two-phase governor system.
    
    Usage:
        governor = EpistemicGovernor()
        
        # Phase 1: Before generation
        envelope = governor.pre_generate(user_intent="factual_query")
        
        # ... LLM generates with envelope constraints ...
        
        # Phase 2: After extraction
        result = governor.adjudicate(proposed_commitments, envelope)
        
        # ... commit phase enforces result.decisions ...
    """
    
    def __init__(self, state: Optional[GovernorState] = None):
        self.state = state or GovernorState()
        self._pre = PreGovernor(self.state)
        self._post = PostGovernor(self.state)
    
    def pre_generate(
        self,
        user_intent: Optional[str] = None,
        domain_hint: Optional[str] = None,
    ) -> GenerationEnvelope:
        """
        Phase 1: Generate envelope constraints before LLM runs.
        Cheap and conservative.
        """
        return self._pre.generate_envelope(user_intent, domain_hint)
    
    def adjudicate(
        self,
        proposed: list[ProposedCommitment],
        envelope: GenerationEnvelope,
    ) -> AdjudicationResult:
        """
        Phase 2: Adjudicate extracted commitments.
        Precise and per-claim.
        """
        return self._post.adjudicate(proposed, envelope)
    
    def record_commit(self, claim: CommittedClaim):
        """
        Update state after a commitment is accepted.
        Called by the commit phase after successful validation.
        """
        self.state.index_commitment(claim)
        self.state.add_to_recent(claim)
        
        # Update metrics
        self.state.metrics.total_commitments += 1
        self.state.metrics.total_confidence_spent += claim.confidence
        
        if claim.supersedes:
            self.state.metrics.revision_count += 1
            self.state.metrics.total_revision_cost += claim.revision_cost
        
        # Update budget
        self.state.budget.remaining_confidence -= claim.confidence
        if claim.confidence > 0.8:
            self.state.budget.high_confidence_count += 1
    
    def record_hedge(self):
        """Record that a hedge was forced."""
        self.state.metrics.hedges_forced += 1
    
    def record_refusal(self):
        """Record that a claim was refused."""
        self.state.metrics.refusals += 1


# =============================================================================
# Example Usage / Test
# =============================================================================

if __name__ == "__main__":
    # Initialize governor
    gov = EpistemicGovernor()
    
    # Phase 1: Pre-generation
    print("=== Phase 1: Pre-Generation ===")
    envelope = gov.pre_generate(
        user_intent="factual_query",
        domain_hint="technical"
    )
    print(f"Max confidence: {envelope.max_confidence}")
    print(f"Must hedge types: {[t.name for t in envelope.must_hedge_types]}")
    print(f"System prompt fragment:\n{envelope.to_system_prompt_fragment()}")
    
    # Simulate extracted commitments
    print("\n=== Phase 2: Adjudication ===")
    proposed = [
        ProposedCommitment(
            id="c1",
            text="Python 3.12 was released in October 2023",
            claim_type=ClaimType.TEMPORAL,
            confidence=0.9,
            proposition_hash=ProposedCommitment.hash_proposition("Python 3.12 released October 2023"),
            scope="conversation",
            span_start=0,
            span_end=45,
        ),
        ProposedCommitment(
            id="c2",
            text="The async implementation uses event loops",
            claim_type=ClaimType.PROCEDURAL,
            confidence=0.85,
            proposition_hash=ProposedCommitment.hash_proposition("async uses event loops"),
            scope="conversation",
            span_start=46,
            span_end=90,
        ),
    ]
    
    result = gov.adjudicate(proposed, envelope)
    
    for decision in result.decisions:
        print(f"\nCommitment {decision.commitment_id}:")
        print(f"  Action: {decision.action.name}")
        print(f"  Cost: {decision.cost}")
        print(f"  Reason: {decision.reason}")
        if decision.adjusted_confidence:
            print(f"  Adjusted confidence: {decision.adjusted_confidence}")
    
    print(f"\nRequires regeneration: {result.requires_regeneration}")
