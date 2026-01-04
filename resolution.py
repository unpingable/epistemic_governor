"""
Resolution Events

Every state change must go through explicit events.
No silent resolution. No direct mutation outside commit.

Resolution primitives:
- WITHDRAW: Remove a commitment (with reason and provenance)
- SUPERSEDE: Replace old commitment with new one
- NARROW_SCOPE: Reduce temporal/domain scope of a commitment
- LOWER_SIGMA: Reduce confidence (with reason)

These are the ONLY ways to modify committed state.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum, auto
import uuid

try:
    from .symbolic_substrate import (
        SymbolicState, Commitment, CandidateCommitment,
        TemporalScope, AdjudicationDecision, Adjudicator,
    )
except ImportError:
    from symbolic_substrate import (
        SymbolicState, Commitment, CandidateCommitment,
        TemporalScope, AdjudicationDecision, Adjudicator,
    )


# =============================================================================
# Resolution Event Types
# =============================================================================

class ResolutionType(Enum):
    """Types of resolution events."""
    WITHDRAW = auto()      # Remove commitment entirely
    SUPERSEDE = auto()     # Replace with new commitment
    NARROW_SCOPE = auto()  # Reduce temporal/domain scope
    LOWER_SIGMA = auto()   # Reduce confidence level
    SPLIT_CONTEXT = auto() # Fork into multiple context-specific claims


class ResolutionProvenance(Enum):
    """What triggered the resolution."""
    NEW_EVIDENCE = auto()       # Better evidence arrived
    CONTRADICTION = auto()      # Conflict with another commitment
    USER_CORRECTION = auto()    # User explicitly corrected
    TEMPORAL_DECAY = auto()     # Time-based decay
    SCOPE_REFINEMENT = auto()   # Scope became clearer
    ERROR_CORRECTION = auto()   # Internal error detected


# =============================================================================
# Resolution Events (Immutable Records)
# =============================================================================

@dataclass(frozen=True)
class WithdrawEvent:
    """
    Withdraw a commitment from state.
    
    The commitment is marked as retracted, not deleted.
    History is preserved.
    """
    event_id: str
    commitment_id: str
    reason: str
    provenance: ResolutionProvenance
    timestamp: datetime
    authorized_by: str = "SYSTEM"
    
    @classmethod
    def create(
        cls,
        commitment_id: str,
        reason: str,
        provenance: ResolutionProvenance,
        authorized_by: str = "SYSTEM",
    ) -> "WithdrawEvent":
        return cls(
            event_id=f"W_{uuid.uuid4().hex[:12]}",
            commitment_id=commitment_id,
            reason=reason,
            provenance=provenance,
            timestamp=datetime.utcnow(),
            authorized_by=authorized_by,
        )


@dataclass(frozen=True)
class SupersedeEvent:
    """
    Supersede one commitment with another.
    
    The old commitment is marked as superseded (not deleted).
    The new commitment references what it replaced.
    """
    event_id: str
    old_commitment_id: str
    new_commitment_id: str
    reason: str
    provenance: ResolutionProvenance
    timestamp: datetime
    evidence_delta: float = 0.0  # How much more evidence the new one has
    
    @classmethod
    def create(
        cls,
        old_id: str,
        new_id: str,
        reason: str,
        provenance: ResolutionProvenance,
        evidence_delta: float = 0.0,
    ) -> "SupersedeEvent":
        return cls(
            event_id=f"S_{uuid.uuid4().hex[:12]}",
            old_commitment_id=old_id,
            new_commitment_id=new_id,
            reason=reason,
            provenance=provenance,
            timestamp=datetime.utcnow(),
            evidence_delta=evidence_delta,
        )


@dataclass(frozen=True)
class NarrowScopeEvent:
    """
    Narrow the temporal or domain scope of a commitment.
    
    Example: "X is true" becomes "X was true in 2022"
    """
    event_id: str
    commitment_id: str
    old_scope: TemporalScope
    new_scope: TemporalScope
    reason: str
    provenance: ResolutionProvenance
    timestamp: datetime
    
    @classmethod
    def create(
        cls,
        commitment_id: str,
        old_scope: TemporalScope,
        new_scope: TemporalScope,
        reason: str,
        provenance: ResolutionProvenance,
    ) -> "NarrowScopeEvent":
        return cls(
            event_id=f"N_{uuid.uuid4().hex[:12]}",
            commitment_id=commitment_id,
            old_scope=old_scope,
            new_scope=new_scope,
            reason=reason,
            provenance=provenance,
            timestamp=datetime.utcnow(),
        )


@dataclass(frozen=True)
class LowerSigmaEvent:
    """
    Lower the confidence level of a commitment.
    
    σ can only go down through explicit events.
    Increasing σ requires new evidence (which would be a supersede).
    """
    event_id: str
    commitment_id: str
    old_sigma: float
    new_sigma: float
    reason: str
    provenance: ResolutionProvenance
    timestamp: datetime
    
    @classmethod
    def create(
        cls,
        commitment_id: str,
        old_sigma: float,
        new_sigma: float,
        reason: str,
        provenance: ResolutionProvenance,
    ) -> "LowerSigmaEvent":
        assert new_sigma < old_sigma, "Can only lower sigma, not raise"
        return cls(
            event_id=f"L_{uuid.uuid4().hex[:12]}",
            commitment_id=commitment_id,
            old_sigma=old_sigma,
            new_sigma=new_sigma,
            reason=reason,
            provenance=provenance,
            timestamp=datetime.utcnow(),
        )


# =============================================================================
# Resolution Manager
# =============================================================================

class ResolutionManager:
    """
    Manages state changes through explicit resolution events.
    
    ALL state modifications go through this.
    No direct mutation of SymbolicState outside of event handlers.
    """
    
    def __init__(self, state: SymbolicState, adjudicator: Adjudicator = None):
        self.state = state
        self.adjudicator = adjudicator or Adjudicator()
        
        # Event log (append-only)
        self.events: List[Any] = []
    
    def withdraw(
        self,
        commitment_id: str,
        reason: str,
        provenance: ResolutionProvenance,
        authorized_by: str = "SYSTEM",
    ) -> WithdrawEvent:
        """
        Withdraw a commitment from state.
        
        The commitment is marked as retracted, not deleted.
        """
        commitment = self.state.commitments.get(commitment_id)
        if not commitment:
            raise ValueError(f"Commitment {commitment_id} not found")
        
        if commitment.status != "active":
            raise ValueError(f"Commitment {commitment_id} is already {commitment.status}")
        
        # Create event
        event = WithdrawEvent.create(
            commitment_id=commitment_id,
            reason=reason,
            provenance=provenance,
            authorized_by=authorized_by,
        )
        
        # Apply to state
        commitment.status = "retracted"
        commitment.retraction_reason = reason
        
        # Update sigma budget
        self.state.total_sigma_allocated -= commitment.sigma
        
        # Log event
        self.events.append(event)
        
        return event
    
    def supersede(
        self,
        old_commitment_id: str,
        new_candidate: CandidateCommitment,
        reason: str,
        provenance: ResolutionProvenance,
    ) -> SupersedeEvent:
        """
        Supersede one commitment with another.
        
        The new candidate must pass adjudication.
        """
        old_commitment = self.state.commitments.get(old_commitment_id)
        if not old_commitment:
            raise ValueError(f"Commitment {old_commitment_id} not found")
        
        if old_commitment.status != "active":
            raise ValueError(f"Commitment {old_commitment_id} is already {old_commitment.status}")
        
        # Temporarily remove old commitment for adjudication
        # (so it doesn't trigger contradiction with itself)
        old_commitment.status = "superseded"
        self.state.total_sigma_allocated -= old_commitment.sigma
        
        # Adjudicate new candidate
        result = self.adjudicator.adjudicate(self.state, new_candidate)
        
        if result.decision != AdjudicationDecision.ACCEPT:
            # Rollback
            old_commitment.status = "active"
            self.state.total_sigma_allocated += old_commitment.sigma
            raise ValueError(
                f"New commitment not accepted: {result.decision.name} - {result.reason_code}"
            )
        
        # Commit new
        new_commitment = result.commitment
        new_commitment.supersedes = old_commitment_id
        self.state.add_commitment(new_commitment)
        
        # Create event
        evidence_delta = result.support_mass_computed - old_commitment.support_mass
        event = SupersedeEvent.create(
            old_id=old_commitment_id,
            new_id=new_commitment.commitment_id,
            reason=reason,
            provenance=provenance,
            evidence_delta=evidence_delta,
        )
        
        # Update old commitment
        old_commitment.superseded_by = new_commitment.commitment_id
        
        # Log event
        self.events.append(event)
        
        return event
    
    def narrow_scope(
        self,
        commitment_id: str,
        new_scope: TemporalScope,
        reason: str,
        provenance: ResolutionProvenance,
    ) -> NarrowScopeEvent:
        """
        Narrow the scope of a commitment.
        
        Scope can only be narrowed, not expanded (expansion would need new evidence).
        """
        commitment = self.state.commitments.get(commitment_id)
        if not commitment:
            raise ValueError(f"Commitment {commitment_id} not found")
        
        old_scope = commitment.t_scope
        
        # Validate that new scope is actually narrower
        # (simplified check - could be more sophisticated)
        if new_scope.start is None and old_scope.start is not None:
            raise ValueError("Cannot expand scope start")
        if new_scope.end is None and old_scope.end is not None:
            raise ValueError("Cannot expand scope end")
        
        # Create event
        event = NarrowScopeEvent.create(
            commitment_id=commitment_id,
            old_scope=old_scope,
            new_scope=new_scope,
            reason=reason,
            provenance=provenance,
        )
        
        # Apply (need to create new commitment object since frozen)
        # For now, we'll update in place (commitment is not frozen)
        commitment.t_scope = new_scope
        
        # Log event
        self.events.append(event)
        
        return event
    
    def lower_sigma(
        self,
        commitment_id: str,
        new_sigma: float,
        reason: str,
        provenance: ResolutionProvenance,
    ) -> LowerSigmaEvent:
        """
        Lower the confidence of a commitment.
        
        σ can only go down. To raise σ, must supersede with new evidence.
        """
        commitment = self.state.commitments.get(commitment_id)
        if not commitment:
            raise ValueError(f"Commitment {commitment_id} not found")
        
        old_sigma = commitment.sigma
        
        if new_sigma >= old_sigma:
            raise ValueError(f"Can only lower sigma: {old_sigma} -> {new_sigma}")
        
        # Create event
        event = LowerSigmaEvent.create(
            commitment_id=commitment_id,
            old_sigma=old_sigma,
            new_sigma=new_sigma,
            reason=reason,
            provenance=provenance,
        )
        
        # Apply
        sigma_delta = old_sigma - new_sigma
        commitment.sigma = new_sigma
        self.state.total_sigma_allocated -= sigma_delta
        
        # Log event
        self.events.append(event)
        
        return event
    
    def resolve_contradiction(
        self,
        existing_id: str,
        new_candidate: CandidateCommitment,
        winner: str,  # "existing" or "new"
        reason: str,
    ) -> Any:
        """
        Resolve a contradiction between existing commitment and new candidate.
        
        This is a compound operation that results in either:
        - Withdraw of existing + accept of new
        - Reject of new (keeping existing)
        - Supersede (if new has better evidence)
        """
        if winner == "new":
            return self.supersede(
                old_commitment_id=existing_id,
                new_candidate=new_candidate,
                reason=reason,
                provenance=ResolutionProvenance.NEW_EVIDENCE,
            )
        elif winner == "existing":
            # Just return the rejection - new candidate stays rejected
            return self.adjudicator.adjudicate(self.state, new_candidate)
        else:
            raise ValueError(f"Unknown winner: {winner}")
    
    def get_event_log(self) -> List[Dict[str, Any]]:
        """Get event log as serializable list."""
        log = []
        for event in self.events:
            log.append({
                "type": type(event).__name__,
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "commitment_id": getattr(event, "commitment_id", None),
                "reason": event.reason,
                "provenance": event.provenance.name,
            })
        return log


# =============================================================================
# Tests
# =============================================================================

def test_withdraw():
    """Test commitment withdrawal."""
    print("=== Test: Withdraw ===\n")
    
    from symbolic_substrate import Predicate, PredicateType, ProvenanceClass
    
    state = SymbolicState()
    adjudicator = Adjudicator(config={"support_deficit_tolerance": 20.0})
    manager = ResolutionManager(state, adjudicator)
    
    # Add a commitment
    candidate = CandidateCommitment(
        predicate=Predicate(PredicateType.HAS, ("Entity", "prop", "val")),
        sigma=0.5,
        t_scope=TemporalScope(),
        provclass=ProvenanceClass.MODEL,
        support=[],
    )
    result = adjudicator.adjudicate(state, candidate)
    if result.decision == AdjudicationDecision.ACCEPT:
        state.add_commitment(result.commitment)
        commitment_id = result.commitment.commitment_id
    
    print(f"Added commitment: {commitment_id}")
    print(f"Sigma allocated: {state.total_sigma_allocated}")
    
    # Withdraw it
    event = manager.withdraw(
        commitment_id=commitment_id,
        reason="Test withdrawal",
        provenance=ResolutionProvenance.ERROR_CORRECTION,
    )
    
    print(f"\nWithdrawn: {event.event_id}")
    print(f"Commitment status: {state.commitments[commitment_id].status}")
    print(f"Sigma allocated: {state.total_sigma_allocated}")
    
    assert state.commitments[commitment_id].status == "retracted"
    assert state.total_sigma_allocated == 0
    
    print("✓ Withdraw working\n")
    return True


def test_supersede():
    """Test commitment supersession."""
    print("=== Test: Supersede ===\n")
    
    from symbolic_substrate import Predicate, PredicateType, ProvenanceClass, SupportItem
    
    state = SymbolicState()
    # Relaxed config for testing
    adjudicator = Adjudicator(config={
        "support_deficit_tolerance": 20.0,
        "sigma_hard_gate": 0.99,
    })
    manager = ResolutionManager(state, adjudicator)
    
    # Add initial commitment with bounded scope
    candidate1 = CandidateCommitment(
        predicate=Predicate(PredicateType.HAS, ("Python", "version", "3.11")),
        sigma=0.5,
        t_scope=TemporalScope(start=datetime(2022, 1, 1), granularity="interval"),
        provclass=ProvenanceClass.MODEL,
        support=[],
    )
    result = adjudicator.adjudicate(state, candidate1)
    
    if result.decision != AdjudicationDecision.ACCEPT:
        print(f"Initial commit failed: {result.decision.name} - {result.reason_code}")
        # Force add for test
        from symbolic_substrate import Commitment
        commitment = Commitment(
            commitment_id=f"φ_test_{uuid.uuid4().hex[:8]}",
            predicate=candidate1.predicate,
            sigma=candidate1.sigma,
            t_scope=candidate1.t_scope,
            provclass=candidate1.provclass,
            support=[],
            logical_deps=[],
            evidentiary_deps=[],
            status="active",
        )
        state.add_commitment(commitment)
        old_id = commitment.commitment_id
    else:
        state.add_commitment(result.commitment)
        old_id = result.commitment.commitment_id
    
    print(f"Initial commitment: {old_id}")
    
    # Supersede with better evidence
    candidate2 = CandidateCommitment(
        predicate=Predicate(PredicateType.HAS, ("Python", "version", "3.12")),
        sigma=0.6,
        t_scope=TemporalScope(start=datetime(2023, 1, 1), granularity="interval"),
        provclass=ProvenanceClass.CITED,
        support=[SupportItem("citation", "python.org", 0.9)],
    )
    
    event = manager.supersede(
        old_commitment_id=old_id,
        new_candidate=candidate2,
        reason="New evidence from python.org",
        provenance=ResolutionProvenance.NEW_EVIDENCE,
    )
    
    print(f"\nSupersede event: {event.event_id}")
    print(f"Old status: {state.commitments[old_id].status}")
    print(f"New commitment: {event.new_commitment_id}")
    print(f"Evidence delta: {event.evidence_delta:.3f}")
    
    assert state.commitments[old_id].status == "superseded"
    assert state.commitments[old_id].superseded_by == event.new_commitment_id
    
    print("✓ Supersede working\n")
    return True


def test_lower_sigma():
    """Test sigma lowering."""
    print("=== Test: Lower Sigma ===\n")
    
    from symbolic_substrate import Predicate, PredicateType, ProvenanceClass, Commitment
    
    state = SymbolicState()
    adjudicator = Adjudicator(config={
        "support_deficit_tolerance": 20.0,
        "sigma_hard_gate": 0.99,
    })
    manager = ResolutionManager(state, adjudicator)
    
    # Add commitment with bounded scope
    candidate = CandidateCommitment(
        predicate=Predicate(PredicateType.IS_A, ("Widget", "Product")),
        sigma=0.8,
        t_scope=TemporalScope(start=datetime(2023, 1, 1), granularity="interval"),
        provclass=ProvenanceClass.MODEL,
        support=[],
    )
    result = adjudicator.adjudicate(state, candidate)
    
    if result.decision == AdjudicationDecision.ACCEPT and result.commitment:
        state.add_commitment(result.commitment)
        cid = result.commitment.commitment_id
    else:
        # Force add for test
        commitment = Commitment(
            commitment_id=f"φ_test_{uuid.uuid4().hex[:8]}",
            predicate=candidate.predicate,
            sigma=candidate.sigma,
            t_scope=candidate.t_scope,
            provclass=candidate.provclass,
            support=[],
            logical_deps=[],
            evidentiary_deps=[],
            status="active",
        )
        state.add_commitment(commitment)
        cid = commitment.commitment_id
    
    print(f"Initial sigma: {state.commitments[cid].sigma}")
    print(f"Sigma allocated: {state.total_sigma_allocated}")
    
    # Lower it
    event = manager.lower_sigma(
        commitment_id=cid,
        new_sigma=0.3,
        reason="Evidence became uncertain",
        provenance=ResolutionProvenance.TEMPORAL_DECAY,
    )
    
    print(f"\nAfter lowering:")
    print(f"New sigma: {state.commitments[cid].sigma}")
    print(f"Sigma allocated: {state.total_sigma_allocated}")
    
    assert state.commitments[cid].sigma == 0.3
    
    # Try to raise - should fail
    try:
        manager.lower_sigma(cid, 0.9, "try to raise", ResolutionProvenance.NEW_EVIDENCE)
        assert False, "Should have raised"
    except ValueError as e:
        print(f"\nCorrectly rejected raise attempt: {e}")
    
    print("✓ Lower sigma working\n")
    return True


def test_event_log():
    """Test that all events are logged."""
    print("=== Test: Event Log ===\n")
    
    from symbolic_substrate import Predicate, PredicateType, ProvenanceClass, Commitment
    
    state = SymbolicState()
    adjudicator = Adjudicator(config={
        "support_deficit_tolerance": 20.0,
        "sigma_hard_gate": 0.99,
    })
    manager = ResolutionManager(state, adjudicator)
    
    # Add commitment with bounded scope
    candidate = CandidateCommitment(
        predicate=Predicate(PredicateType.IS_A, ("Item", "Thing")),
        sigma=0.5,
        t_scope=TemporalScope(start=datetime(2023, 1, 1), granularity="interval"),
        provclass=ProvenanceClass.MODEL,
        support=[],
    )
    result = adjudicator.adjudicate(state, candidate)
    
    if result.decision == AdjudicationDecision.ACCEPT and result.commitment:
        state.add_commitment(result.commitment)
        cid = result.commitment.commitment_id
    else:
        commitment = Commitment(
            commitment_id=f"φ_test_{uuid.uuid4().hex[:8]}",
            predicate=candidate.predicate,
            sigma=candidate.sigma,
            t_scope=candidate.t_scope,
            provclass=candidate.provclass,
            support=[],
            logical_deps=[],
            evidentiary_deps=[],
            status="active",
        )
        state.add_commitment(commitment)
        cid = commitment.commitment_id
    
    manager.lower_sigma(cid, 0.3, "decay", ResolutionProvenance.TEMPORAL_DECAY)
    manager.withdraw(cid, "cleanup", ResolutionProvenance.ERROR_CORRECTION)
    
    log = manager.get_event_log()
    
    print(f"Event log ({len(log)} events):")
    for entry in log:
        print(f"  {entry['type']}: {entry['event_id']}")
    
    assert len(log) == 2
    assert log[0]["type"] == "LowerSigmaEvent"
    assert log[1]["type"] == "WithdrawEvent"
    
    print("✓ Event log working\n")
    return True


def run_all_tests():
    """Run all resolution tests."""
    print("=" * 60)
    print("RESOLUTION EVENT TESTS")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("withdraw", test_withdraw()))
    results.append(("supersede", test_supersede()))
    results.append(("lower_sigma", test_lower_sigma()))
    results.append(("event_log", test_event_log()))
    
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
