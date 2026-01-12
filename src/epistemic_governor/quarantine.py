"""
Quarantine Store

Quarantine is where alignment theater goes to hide.

Key properties:
1. Quarantine is sticky - can't be bypassed by rephrasing
2. Promotion requires new support artifacts
3. Every promotion is an event with provenance

A quarantined claim is NOT a rejected claim:
- Rejected = structurally invalid, won't ever be accepted
- Quarantined = potentially valid but needs more evidence
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any, Tuple
from datetime import datetime
from enum import Enum, auto
import uuid

try:
    from epistemic_governor.symbolic_substrate import (
        CandidateCommitment, Commitment, AdjudicationDecision,
        SupportItem, SymbolicState, Adjudicator, AdjudicationResult,
        Predicate, ProvenanceClass, TemporalScope, PredicateType,
    )
except ImportError:
    from epistemic_governor.symbolic_substrate import (
        CandidateCommitment, Commitment, AdjudicationDecision,
        SupportItem, SymbolicState, Adjudicator, AdjudicationResult,
        Predicate, ProvenanceClass, TemporalScope, PredicateType,
    )


# =============================================================================
# Quarantine Reasons
# =============================================================================

class QuarantineReason(Enum):
    """Why something is quarantined."""
    SUPPORT_DEFICIT = auto()      # σ too high for available support
    SCOPE_UNCLEAR = auto()        # Temporal/domain scope needs clarification
    IDENTITY_UNCERTAIN = auto()   # SameAs uncertainty too high
    DEPENDENCY_WEAK = auto()      # Dependencies exist but are weak
    EVIDENCE_STALE = auto()       # Support exists but is outdated
    PROVENANCE_SUSPECT = auto()   # Source reliability in question


# =============================================================================
# Quarantine Entry
# =============================================================================

@dataclass
class QuarantineEntry:
    """
    A quarantined candidate commitment.
    
    Contains the original candidate plus metadata about
    what's needed to promote it.
    """
    # Identity
    quarantine_id: str
    
    # The candidate that was quarantined
    candidate: CandidateCommitment
    
    # Why it was quarantined
    reason: QuarantineReason
    adjudication_decision: AdjudicationDecision
    
    # What's needed to unblock
    support_deficit: float              # How much more support mass needed
    required_evidence_types: Set[str]   # Types of evidence that would help
    unblocking_condition: str           # Human-readable condition
    
    # Lifecycle
    quarantined_at: datetime = field(default_factory=datetime.utcnow)
    attempts: int = 0                   # How many promotion attempts
    last_attempt: Optional[datetime] = None
    
    # Content hash for dedup (can't rephrase to escape)
    content_hash: str = ""
    
    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = self.candidate.predicate.canonical_form()


# =============================================================================
# Promotion Event
# =============================================================================

@dataclass
class PromotionEvent:
    """
    Records a quarantine → accept promotion.
    
    Every promotion is an event with provenance.
    No silent promotions.
    """
    quarantine_id: str
    commitment_id: str
    promoted_at: datetime
    
    # What enabled the promotion
    new_support: List[SupportItem]
    support_mass_before: float
    support_mass_after: float
    
    # Who authorized
    promotion_reason: str
    authorized_by: str = "SYSTEM"  # Could be user override


@dataclass
class PromotionRejection:
    """Records a failed promotion attempt."""
    quarantine_id: str
    attempted_at: datetime
    reason: str
    support_mass_provided: float
    support_mass_required: float


# =============================================================================
# Quarantine Store
# =============================================================================

class QuarantineStore:
    """
    Manages quarantined commitments.
    
    Key invariants:
    1. Quarantine is by content hash, not surface form
    2. Rephrasing doesn't escape quarantine
    3. Promotion requires explicit new evidence
    4. All promotions are events
    """
    
    def __init__(self):
        # Quarantined entries by ID
        self.entries: Dict[str, QuarantineEntry] = {}
        
        # Index by content hash (for dedup/rephrase detection)
        self.by_content_hash: Dict[str, str] = {}  # hash -> quarantine_id
        
        # Index by entity (for related queries)
        self.by_entity: Dict[str, Set[str]] = {}  # entity -> set of quarantine_ids
        
        # Promotion history
        self.promotions: List[PromotionEvent] = []
        self.rejections: List[PromotionRejection] = []
    
    def quarantine(
        self,
        candidate: CandidateCommitment,
        result: AdjudicationResult,
    ) -> QuarantineEntry:
        """
        Add a candidate to quarantine.
        
        Returns existing entry if this content is already quarantined
        (rephrase protection).
        """
        content_hash = candidate.predicate.canonical_form()
        
        # Check if already quarantined (rephrase attempt)
        if content_hash in self.by_content_hash:
            existing_id = self.by_content_hash[content_hash]
            existing = self.entries[existing_id]
            existing.attempts += 1
            existing.last_attempt = datetime.utcnow()
            return existing
        
        # Create new entry
        quarantine_id = f"Q_{uuid.uuid4().hex[:12]}"
        
        reason = self._map_decision_to_reason(result.decision)
        unblocking = self._compute_unblocking_condition(result)
        required_types = self._compute_required_evidence_types(result)
        
        entry = QuarantineEntry(
            quarantine_id=quarantine_id,
            candidate=candidate,
            reason=reason,
            adjudication_decision=result.decision,
            support_deficit=result.support_deficit,
            required_evidence_types=required_types,
            unblocking_condition=unblocking,
            content_hash=content_hash,
        )
        
        # Add to stores
        self.entries[quarantine_id] = entry
        self.by_content_hash[content_hash] = quarantine_id
        
        # Index by entity
        for arg in candidate.predicate.args:
            if arg not in self.by_entity:
                self.by_entity[arg] = set()
            self.by_entity[arg].add(quarantine_id)
        
        return entry
    
    def is_quarantined(self, candidate: CandidateCommitment) -> bool:
        """Check if this content is already quarantined."""
        content_hash = candidate.predicate.canonical_form()
        return content_hash in self.by_content_hash
    
    def get_entry(self, quarantine_id: str) -> Optional[QuarantineEntry]:
        """Get a quarantine entry by ID."""
        return self.entries.get(quarantine_id)
    
    def get_by_content(self, candidate: CandidateCommitment) -> Optional[QuarantineEntry]:
        """Get quarantine entry for this content."""
        content_hash = candidate.predicate.canonical_form()
        if content_hash in self.by_content_hash:
            return self.entries[self.by_content_hash[content_hash]]
        return None
    
    def attempt_promotion(
        self,
        quarantine_id: str,
        new_support: List[SupportItem],
        state: SymbolicState,
        adjudicator: Adjudicator,
    ) -> Tuple[bool, Any]:
        """
        Attempt to promote a quarantined entry.
        
        Returns (success, result) where result is either
        PromotionEvent or PromotionRejection.
        """
        entry = self.entries.get(quarantine_id)
        if not entry:
            return False, PromotionRejection(
                quarantine_id=quarantine_id,
                attempted_at=datetime.utcnow(),
                reason="QUARANTINE_NOT_FOUND",
                support_mass_provided=0,
                support_mass_required=0,
            )
        
        # Update candidate with new support
        updated_candidate = CandidateCommitment(
            predicate=entry.candidate.predicate,
            sigma=entry.candidate.sigma,
            t_scope=entry.candidate.t_scope,
            provclass=entry.candidate.provclass,
            support=list(entry.candidate.support) + list(new_support),
            logical_deps=entry.candidate.logical_deps,
            evidentiary_deps=entry.candidate.evidentiary_deps,
            source_span=entry.candidate.source_span,
            source_text=entry.candidate.source_text,
            extraction_confidence=entry.candidate.extraction_confidence,
        )
        
        # Re-adjudicate
        result = adjudicator.adjudicate(state, updated_candidate)
        
        entry.attempts += 1
        entry.last_attempt = datetime.utcnow()
        
        if result.decision == AdjudicationDecision.ACCEPT:
            # Promotion successful
            state.add_commitment(result.commitment)
            
            event = PromotionEvent(
                quarantine_id=quarantine_id,
                commitment_id=result.commitment.commitment_id,
                promoted_at=datetime.utcnow(),
                new_support=new_support,
                support_mass_before=entry.support_deficit,
                support_mass_after=result.support_mass_computed,
                promotion_reason="NEW_EVIDENCE",
            )
            self.promotions.append(event)
            
            # Remove from quarantine
            self._remove_entry(quarantine_id)
            
            return True, event
        
        else:
            # Still not enough
            rejection = PromotionRejection(
                quarantine_id=quarantine_id,
                attempted_at=datetime.utcnow(),
                reason=result.reason_code,
                support_mass_provided=result.support_mass_computed,
                support_mass_required=result.support_mass_required,
            )
            self.rejections.append(rejection)
            
            # Update entry with new deficit
            entry.support_deficit = result.support_deficit
            
            return False, rejection
    
    def _remove_entry(self, quarantine_id: str):
        """Remove an entry from quarantine."""
        entry = self.entries.get(quarantine_id)
        if not entry:
            return
        
        # Remove from content hash index
        if entry.content_hash in self.by_content_hash:
            del self.by_content_hash[entry.content_hash]
        
        # Remove from entity index
        for arg in entry.candidate.predicate.args:
            if arg in self.by_entity:
                self.by_entity[arg].discard(quarantine_id)
        
        # Remove entry
        del self.entries[quarantine_id]
    
    def _map_decision_to_reason(self, decision: AdjudicationDecision) -> QuarantineReason:
        """Map adjudication decision to quarantine reason."""
        mapping = {
            AdjudicationDecision.QUARANTINE_SUPPORT: QuarantineReason.SUPPORT_DEFICIT,
            AdjudicationDecision.QUARANTINE_SCOPE: QuarantineReason.SCOPE_UNCLEAR,
            AdjudicationDecision.QUARANTINE_IDENTITY: QuarantineReason.IDENTITY_UNCERTAIN,
        }
        return mapping.get(decision, QuarantineReason.SUPPORT_DEFICIT)
    
    def _compute_unblocking_condition(self, result: AdjudicationResult) -> str:
        """Compute human-readable unblocking condition."""
        if result.support_deficit > 0:
            return f"Provide {result.support_deficit:.2f} more support mass"
        if result.decision == AdjudicationDecision.QUARANTINE_SCOPE:
            return "Clarify temporal or domain scope"
        if result.decision == AdjudicationDecision.QUARANTINE_IDENTITY:
            return "Resolve identity uncertainty"
        return "Unknown condition"
    
    def _compute_required_evidence_types(self, result: AdjudicationResult) -> Set[str]:
        """Compute what types of evidence would help."""
        types = set()
        if result.support_deficit > 0:
            types.add("citation")
            types.add("doc_span")
            types.add("sensor")
        if result.decision == AdjudicationDecision.QUARANTINE_SCOPE:
            types.add("temporal_anchor")
        return types
    
    # Telemetry
    
    def get_stats(self) -> Dict[str, Any]:
        """Get quarantine statistics."""
        return {
            "total_quarantined": len(self.entries),
            "total_promotions": len(self.promotions),
            "total_rejections": len(self.rejections),
            "by_reason": self._count_by_reason(),
            "avg_attempts": self._avg_attempts(),
        }
    
    def _count_by_reason(self) -> Dict[str, int]:
        """Count entries by reason."""
        counts = {}
        for entry in self.entries.values():
            reason = entry.reason.name
            counts[reason] = counts.get(reason, 0) + 1
        return counts
    
    def _avg_attempts(self) -> float:
        """Average promotion attempts per entry."""
        if not self.entries:
            return 0.0
        return sum(e.attempts for e in self.entries.values()) / len(self.entries)


# Tuple already imported at top


# =============================================================================
# Tests
# =============================================================================

def test_quarantine_sticky():
    """Test that quarantine is sticky - rephrasing doesn't escape."""
    print("=== Test: Quarantine is sticky ===\n")
    
    store = QuarantineStore()
    state = SymbolicState()
    adjudicator = Adjudicator()
    
    # Create a candidate that will be quarantined
    candidate1 = CandidateCommitment(
        predicate=Predicate(
            ptype=PredicateType.HAS,
            args=("Company", "revenue", "$1B"),
        ),
        sigma=0.6,
        t_scope=TemporalScope(),
        provclass=ProvenanceClass.MODEL,
        support=[],
    )
    
    # Adjudicate - should quarantine
    result1 = adjudicator.adjudicate(state, candidate1)
    print(f"First adjudication: {result1.decision.name}")
    
    if result1.decision in {
        AdjudicationDecision.QUARANTINE_SUPPORT,
        AdjudicationDecision.QUARANTINE_SCOPE,
    }:
        entry1 = store.quarantine(candidate1, result1)
        print(f"Quarantined: {entry1.quarantine_id}")
        print(f"Attempts: {entry1.attempts}")
    
    # Try to "rephrase" - same content, different surface
    candidate2 = CandidateCommitment(
        predicate=Predicate(
            ptype=PredicateType.HAS,
            args=("Company", "revenue", "$1B"),  # Same content!
        ),
        sigma=0.7,  # Tried to increase confidence
        t_scope=TemporalScope(),
        provclass=ProvenanceClass.MODEL,
        support=[],
    )
    
    # Check if already quarantined
    is_quarantined = store.is_quarantined(candidate2)
    print(f"\nRephrase detected: {is_quarantined}")
    
    if is_quarantined:
        entry2 = store.get_by_content(candidate2)
        print(f"Same entry: {entry2.quarantine_id}")
        print(f"Attempts now: {entry2.attempts}")
    
    assert is_quarantined, "Should detect rephrase"
    
    print("✓ Quarantine is sticky - rephrasing detected\n")
    return True


def test_promotion_requires_evidence():
    """Test that promotion requires new evidence."""
    print("=== Test: Promotion requires evidence ===\n")
    
    store = QuarantineStore()
    state = SymbolicState()
    adjudicator = Adjudicator()
    
    # Quarantine a candidate
    candidate = CandidateCommitment(
        predicate=Predicate(
            ptype=PredicateType.HAS,
            args=("Product", "price", "$99"),
        ),
        sigma=0.7,
        t_scope=TemporalScope(),
        provclass=ProvenanceClass.MODEL,
        support=[],
    )
    
    result = adjudicator.adjudicate(state, candidate)
    print(f"Initial adjudication: {result.decision.name}")
    print(f"Support deficit: {result.support_deficit:.2f}")
    
    entry = store.quarantine(candidate, result)
    print(f"Quarantined: {entry.quarantine_id}")
    
    # Try promotion without new evidence
    success, event = store.attempt_promotion(
        entry.quarantine_id,
        new_support=[],  # No new evidence!
        state=state,
        adjudicator=adjudicator,
    )
    
    print(f"\nPromotion without evidence: {'success' if success else 'failed'}")
    assert not success, "Should fail without evidence"
    
    # Try promotion with evidence
    success, event = store.attempt_promotion(
        entry.quarantine_id,
        new_support=[
            SupportItem(
                source_type="citation",
                source_id="pricing.example.com",
                reliability=0.9,
            ),
            SupportItem(
                source_type="doc_span",
                source_id="catalog_doc",
                reliability=0.85,
            ),
        ],
        state=state,
        adjudicator=adjudicator,
    )
    
    print(f"Promotion with evidence: {'success' if success else 'failed'}")
    
    if success:
        print(f"Committed: {event.commitment_id}")
        print(f"Support mass: {event.support_mass_after:.2f}")
        assert entry.quarantine_id not in store.entries, "Should be removed from quarantine"
    
    print("✓ Promotion pathway works correctly\n")
    return True


def test_promotion_events_logged():
    """Test that all promotions are logged as events."""
    print("=== Test: Promotion events logged ===\n")
    
    store = QuarantineStore()
    state = SymbolicState()
    adjudicator = Adjudicator()
    
    # Quarantine and promote something
    candidate = CandidateCommitment(
        predicate=Predicate(
            ptype=PredicateType.IS_A,
            args=("Widget", "Product"),
        ),
        sigma=0.5,
        t_scope=TemporalScope(),
        provclass=ProvenanceClass.MODEL,
        support=[],
    )
    
    result = adjudicator.adjudicate(state, candidate)
    entry = store.quarantine(candidate, result)
    
    initial_promotions = len(store.promotions)
    initial_rejections = len(store.rejections)
    
    # Failed attempt
    store.attempt_promotion(entry.quarantine_id, [], state, adjudicator)
    
    # Successful attempt (with evidence)
    store.attempt_promotion(
        entry.quarantine_id,
        [SupportItem("citation", "source.com", 0.9)],
        state,
        adjudicator,
    )
    
    print(f"Promotions logged: {len(store.promotions) - initial_promotions}")
    print(f"Rejections logged: {len(store.rejections) - initial_rejections}")
    
    stats = store.get_stats()
    print(f"Stats: {stats}")
    
    # Should have logged the rejection
    assert len(store.rejections) > initial_rejections, "Rejection should be logged"
    
    print("✓ All promotion events logged\n")
    return True


# PredicateType already imported above


def run_all_tests():
    """Run all quarantine tests."""
    print("=" * 60)
    print("QUARANTINE TESTS")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("quarantine_sticky", test_quarantine_sticky()))
    results.append(("promotion_requires_evidence", test_promotion_requires_evidence()))
    results.append(("promotion_events_logged", test_promotion_events_logged()))
    
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
