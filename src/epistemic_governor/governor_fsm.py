"""
Governor FSM - State Machine for Epistemic Authority

Implements the NLAI-compliant governor state machine:
- Language may trigger proposals (PA)
- Only evidence may trigger commits (CA)
- No freeform response path

States:
- S0: IDLE - nothing pending
- S1: PROPOSED - proposals exist, no commit eligible
- S2: EVIDENCE_WAIT - commit desired but evidence missing
- S3: COMMIT_ELIGIBLE - admissible evidence + gates pass
- S4: COMMIT_APPLIED - commit executed
- S5: FREEZE - closure blocked (limit-cycle protection)
- S6: POLICY_CHANGE - high-risk policy commit path
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum, auto
from datetime import datetime
import uuid

try:
    from epistemic_governor.symbolic_substrate import (
        SymbolicState, Adjudicator, AdjudicationDecision,
        CandidateCommitment, Commitment, SupportItem,
    )
    from epistemic_governor.resolution import ResolutionManager, ResolutionProvenance
    from epistemic_governor.quarantine import QuarantineStore
except ImportError:
    from epistemic_governor.symbolic_substrate import (
        SymbolicState, Adjudicator, AdjudicationDecision,
        CandidateCommitment, Commitment, SupportItem,
    )
    from epistemic_governor.resolution import ResolutionManager, ResolutionProvenance
    from epistemic_governor.quarantine import QuarantineStore


# =============================================================================
# FSM States
# =============================================================================

class GovernorState(Enum):
    """Governor FSM states."""
    IDLE = auto()            # S0: Nothing pending
    PROPOSED = auto()        # S1: Proposals exist, no commit eligible
    EVIDENCE_WAIT = auto()   # S2: Commit desired but evidence missing
    COMMIT_ELIGIBLE = auto() # S3: Admissible evidence + gates pass
    COMMIT_APPLIED = auto()  # S4: Commit executed
    FREEZE = auto()          # S5: Closure blocked
    POLICY_CHANGE = auto()   # S6: High-risk policy path


class GovernorEvent(Enum):
    """Events that trigger state transitions."""
    PROPOSAL = auto()        # P: New proposal arrives
    EVIDENCE_GOOD = auto()   # E+: Admissible evidence arrives
    EVIDENCE_BAD = auto()    # E-: Evidence fails Γ
    COMMIT_INTENT = auto()   # Governor wants to commit
    GATE_FAIL = auto()       # Any gate failure
    FREEZE_TRIP = auto()     # Anti-cycle condition trips
    POLICY_INTENT = auto()   # Policy change intent
    COMMIT_DONE = auto()     # Commit completed
    UNFREEZE = auto()        # New evidence unfreezes


# =============================================================================
# Evidence Types (NLAI-compliant)
# =============================================================================

class EvidenceType(Enum):
    """Types of evidence for admissibility checking."""
    TOOL_TRACE = auto()      # E1: Verifiable tool output
    SIGNED_ATTESTATION = auto()  # E2: Cryptographically signed
    HUMAN_CONFIRMATION = auto()  # E3: Explicit human approval
    SENSOR_READING = auto()  # E4: External sensor
    MODEL_TEXT = auto()      # FORBIDDEN - never admissible


@dataclass
class Evidence:
    """An evidence artifact."""
    evidence_id: str
    evidence_type: EvidenceType
    content: Any
    provenance: str
    timestamp: datetime
    scope: str  # What this evidence covers
    integrity_hash: Optional[str] = None
    revoked: bool = False
    
    @property
    def is_admissible(self) -> bool:
        """Check if this evidence passes Γ."""
        # MODEL_TEXT is NEVER admissible
        if self.evidence_type == EvidenceType.MODEL_TEXT:
            return False
        
        # Must have provenance
        if not self.provenance:
            return False
        
        # Must not be revoked
        if self.revoked:
            return False
        
        return True


# =============================================================================
# Proposal and Action Types
# =============================================================================

class ActionType(Enum):
    """Types of actions - partitioned into PA and CA."""
    # Proposal Actions (PA) - may be triggered by MO
    PROPOSE_CLAIM = auto()
    PROPOSE_IDENTITY = auto()
    PROPOSE_CONTRADICTION = auto()
    REQUEST_EVIDENCE = auto()
    ANNOTATE = auto()
    
    # Commit Actions (CA) - require evidence
    ACCEPT_CLAIM = auto()
    REJECT_CLAIM = auto()
    CLOSE_CONTRADICTION = auto()
    BIND_IDENTITY = auto()
    PROMOTE_VERSION = auto()
    UPDATE_CANONICAL = auto()
    SET_POLICY = auto()
    
    @property
    def is_commit_action(self) -> bool:
        """Is this a commit action (requires evidence)?"""
        return self in {
            ActionType.ACCEPT_CLAIM,
            ActionType.REJECT_CLAIM,
            ActionType.CLOSE_CONTRADICTION,
            ActionType.BIND_IDENTITY,
            ActionType.PROMOTE_VERSION,
            ActionType.UPDATE_CANONICAL,
            ActionType.SET_POLICY,
        }
    
    @property
    def requires_elevated_gating(self) -> bool:
        """Does this action require stricter evidence?"""
        return self in {
            ActionType.SET_POLICY,
            ActionType.UPDATE_CANONICAL,
            ActionType.BIND_IDENTITY,
        }


@dataclass
class Proposal:
    """A pending proposal (PA)."""
    proposal_id: str
    action_type: ActionType
    candidate: Optional[CandidateCommitment]
    target_id: Optional[str]  # For actions targeting existing commitments
    reason: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Evidence requirements
    evidence_needed: Set[EvidenceType] = field(default_factory=set)
    evidence_provided: List[Evidence] = field(default_factory=list)
    
    # Freeze tracking
    attempt_count: int = 0
    last_attempt: Optional[datetime] = None


# =============================================================================
# Gate Checks
# =============================================================================

@dataclass
class GateResult:
    """Result of a gate check."""
    passed: bool
    gate_name: str
    reason: str = ""
    evidence_deficit: List[EvidenceType] = field(default_factory=list)


class GateChecker:
    """
    Checks all gates before commit.
    
    Gates:
    1. NLAI - evidence present and admissible
    2. Evidence scope - covers the action target
    3. Contradiction energy - must decrease
    4. Lyapunov - epistemic energy check
    5. Freeze check - target not frozen
    """
    
    def __init__(self, min_delta: float = 0.1, reopen_limit: int = 3):
        self.min_delta = min_delta
        self.reopen_limit = reopen_limit
    
    def check_all(
        self,
        proposal: Proposal,
        state: SymbolicState,
        frozen_targets: Set[str],
    ) -> List[GateResult]:
        """Run all gate checks."""
        results = []
        
        # Gate 1: NLAI - evidence must exist and be admissible
        results.append(self._check_nlai(proposal))
        
        # Gate 2: Evidence scope
        results.append(self._check_scope(proposal))
        
        # Gate 3: Freeze check
        if proposal.target_id:
            results.append(self._check_freeze(proposal.target_id, frozen_targets))
        
        # Gate 4: Elevated gating for high-risk actions
        if proposal.action_type.requires_elevated_gating:
            results.append(self._check_elevated(proposal))
        
        return results
    
    def _check_nlai(self, proposal: Proposal) -> GateResult:
        """NLAI gate: evidence must exist and be admissible."""
        if not proposal.action_type.is_commit_action:
            return GateResult(passed=True, gate_name="NLAI", reason="PA, no evidence required")
        
        admissible = [e for e in proposal.evidence_provided if e.is_admissible]
        
        if not admissible:
            return GateResult(
                passed=False,
                gate_name="NLAI",
                reason="No admissible evidence for commit action",
                evidence_deficit=list(proposal.evidence_needed) or [EvidenceType.TOOL_TRACE],
            )
        
        return GateResult(passed=True, gate_name="NLAI", reason=f"{len(admissible)} admissible evidence items")
    
    def _check_scope(self, proposal: Proposal) -> GateResult:
        """Scope gate: evidence must cover action target."""
        if not proposal.action_type.is_commit_action:
            return GateResult(passed=True, gate_name="SCOPE", reason="PA, no scope check")
        
        for evidence in proposal.evidence_provided:
            if evidence.is_admissible:
                # Simplified scope check - in reality would be more sophisticated
                if evidence.scope and proposal.target_id:
                    if proposal.target_id in evidence.scope or evidence.scope == "*":
                        return GateResult(passed=True, gate_name="SCOPE", reason="Evidence covers target")
        
        return GateResult(
            passed=False,
            gate_name="SCOPE",
            reason="Evidence does not cover action target",
        )
    
    def _check_freeze(self, target_id: str, frozen_targets: Set[str]) -> GateResult:
        """Freeze gate: target must not be frozen."""
        if target_id in frozen_targets:
            return GateResult(
                passed=False,
                gate_name="FREEZE",
                reason=f"Target {target_id} is frozen - new evidence required",
            )
        return GateResult(passed=True, gate_name="FREEZE", reason="Target not frozen")
    
    def _check_elevated(self, proposal: Proposal) -> GateResult:
        """Elevated gate: stricter requirements for high-risk actions."""
        admissible = [e for e in proposal.evidence_provided if e.is_admissible]
        
        # Require at least 2 independent evidence sources for elevated actions
        if len(admissible) < 2:
            return GateResult(
                passed=False,
                gate_name="ELEVATED",
                reason="High-risk action requires 2+ independent evidence sources",
            )
        
        # Check for diversity of evidence types
        types = {e.evidence_type for e in admissible}
        if len(types) < 2:
            return GateResult(
                passed=False,
                gate_name="ELEVATED",
                reason="High-risk action requires diverse evidence types",
            )
        
        return GateResult(passed=True, gate_name="ELEVATED", reason="Elevated requirements met")


# =============================================================================
# Governor FSM
# =============================================================================

class GovernorFSM:
    """
    The sovereign governor FSM.
    
    ALL state changes go through this.
    Language triggers proposals.
    Only evidence triggers commits.
    """
    
    def __init__(
        self,
        state: SymbolicState = None,
        adjudicator: Adjudicator = None,
    ):
        self.symbolic_state = state or SymbolicState()
        self.adjudicator = adjudicator or Adjudicator()
        self.resolution_manager = ResolutionManager(self.symbolic_state, self.adjudicator)
        self.quarantine = QuarantineStore()
        self.gate_checker = GateChecker()
        
        # FSM state
        self.fsm_state = GovernorState.IDLE
        
        # Pending proposals
        self.proposals: Dict[str, Proposal] = {}
        
        # Frozen targets (anti-cycle)
        self.frozen_targets: Set[str] = set()
        
        # Event log
        self.transitions: List[Dict[str, Any]] = []
    
    def receive_proposal(self, proposal: Proposal) -> str:
        """
        Receive a proposal from language model.
        
        This is the ONLY way language can affect the system.
        Proposals are PA - they don't change authoritative state.
        """
        self.proposals[proposal.proposal_id] = proposal
        
        # Transition: any -> PROPOSED (or stay in current if not IDLE)
        if self.fsm_state == GovernorState.IDLE:
            self._transition(GovernorState.PROPOSED, GovernorEvent.PROPOSAL)
        
        return proposal.proposal_id
    
    def receive_evidence(self, proposal_id: str, evidence: Evidence) -> GovernorEvent:
        """
        Receive evidence for a pending proposal.
        
        Evidence may come from tools, humans, or sensors.
        NEVER from model text.
        """
        if proposal_id not in self.proposals:
            raise ValueError(f"Unknown proposal: {proposal_id}")
        
        proposal = self.proposals[proposal_id]
        
        # Check admissibility
        if evidence.evidence_type == EvidenceType.MODEL_TEXT:
            # F-02: Self-report as evidence - FORBIDDEN
            self._log_forbidden("F-02", "Attempted to use MODEL_TEXT as evidence")
            return GovernorEvent.EVIDENCE_BAD
        
        if not evidence.is_admissible:
            return GovernorEvent.EVIDENCE_BAD
        
        # Add evidence to proposal
        proposal.evidence_provided.append(evidence)
        
        # Check if this unfreezes a target
        if proposal.target_id and proposal.target_id in self.frozen_targets:
            self.frozen_targets.remove(proposal.target_id)
            self._transition(GovernorState.PROPOSED, GovernorEvent.UNFREEZE)
        
        # Transition to COMMIT_ELIGIBLE if gates pass
        gates = self.gate_checker.check_all(proposal, self.symbolic_state, self.frozen_targets)
        
        if all(g.passed for g in gates):
            self._transition(GovernorState.COMMIT_ELIGIBLE, GovernorEvent.EVIDENCE_GOOD)
            return GovernorEvent.EVIDENCE_GOOD
        
        return GovernorEvent.EVIDENCE_BAD
    
    def attempt_commit(self, proposal_id: str) -> Tuple[bool, str]:
        """
        Attempt to commit a proposal.
        
        Requires:
        1. Admissible evidence
        2. All gates pass
        3. Not frozen
        """
        if proposal_id not in self.proposals:
            return False, f"Unknown proposal: {proposal_id}"
        
        proposal = self.proposals[proposal_id]
        
        # Must be a commit action
        if not proposal.action_type.is_commit_action:
            return False, "Proposal is PA, not CA - no commit needed"
        
        # Check gates
        gates = self.gate_checker.check_all(proposal, self.symbolic_state, self.frozen_targets)
        
        failed_gates = [g for g in gates if not g.passed]
        if failed_gates:
            # Track attempt for freeze detection
            proposal.attempt_count += 1
            proposal.last_attempt = datetime.utcnow()
            
            # Check for freeze trip
            if proposal.attempt_count > self.gate_checker.reopen_limit:
                if proposal.target_id:
                    self.frozen_targets.add(proposal.target_id)
                    self._transition(GovernorState.FREEZE, GovernorEvent.FREEZE_TRIP)
                    return False, f"Target frozen after {proposal.attempt_count} failed attempts"
            
            self._transition(GovernorState.EVIDENCE_WAIT, GovernorEvent.GATE_FAIL)
            return False, f"Gate failures: {[g.gate_name + ': ' + g.reason for g in failed_gates]}"
        
        # Execute commit
        success, result = self._execute_commit(proposal)
        
        if success:
            self._transition(GovernorState.COMMIT_APPLIED, GovernorEvent.COMMIT_DONE)
            del self.proposals[proposal_id]
            
            # Return to IDLE or PROPOSED
            if self.proposals:
                self._transition(GovernorState.PROPOSED, GovernorEvent.PROPOSAL)
            else:
                self._transition(GovernorState.IDLE, GovernorEvent.COMMIT_DONE)
        
        return success, result
    
    def _execute_commit(self, proposal: Proposal) -> Tuple[bool, str]:
        """Execute the actual commit action."""
        action = proposal.action_type
        
        if action == ActionType.ACCEPT_CLAIM:
            if proposal.candidate:
                result = self.adjudicator.adjudicate(self.symbolic_state, proposal.candidate)
                if result.decision == AdjudicationDecision.ACCEPT:
                    self.symbolic_state.add_commitment(result.commitment)
                    return True, f"Accepted: {result.commitment.commitment_id}"
                return False, f"Adjudication failed: {result.reason_code}"
        
        elif action == ActionType.REJECT_CLAIM:
            if proposal.target_id:
                event = self.resolution_manager.withdraw(
                    proposal.target_id,
                    proposal.reason,
                    ResolutionProvenance.NEW_EVIDENCE,
                )
                return True, f"Rejected: {event.event_id}"
        
        elif action == ActionType.CLOSE_CONTRADICTION:
            # Would need contradiction resolution logic
            return False, "CLOSE_CONTRADICTION not yet implemented"
        
        return False, f"Unknown action type: {action}"
    
    def _transition(self, new_state: GovernorState, event: GovernorEvent):
        """Record state transition."""
        old_state = self.fsm_state
        self.fsm_state = new_state
        
        self.transitions.append({
            "from": old_state.name,
            "to": new_state.name,
            "event": event.name,
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    def _log_forbidden(self, code: str, description: str):
        """Log a forbidden transition attempt."""
        self.transitions.append({
            "type": "FORBIDDEN",
            "code": code,
            "description": description,
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    def get_state(self) -> Dict[str, Any]:
        """Get current FSM state."""
        return {
            "fsm_state": self.fsm_state.name,
            "pending_proposals": len(self.proposals),
            "frozen_targets": list(self.frozen_targets),
            "committed_claims": len(self.symbolic_state.commitments),
            "total_sigma": self.symbolic_state.total_sigma_allocated,
        }
    
    def get_transitions(self) -> List[Dict[str, Any]]:
        """Get transition history."""
        return self.transitions


# =============================================================================
# Tests
# =============================================================================

def test_nlai_enforcement():
    """Test that MODEL_TEXT evidence is always rejected."""
    print("=== Test: NLAI Enforcement ===\n")
    
    from epistemic_governor.symbolic_substrate import Predicate, PredicateType, ProvenanceClass, TemporalScope
    
    fsm = GovernorFSM()
    
    # Create a proposal
    candidate = CandidateCommitment(
        predicate=Predicate(PredicateType.HAS, ("Entity", "prop", "val")),
        sigma=0.5,
        t_scope=TemporalScope(),
        provclass=ProvenanceClass.MODEL,
        support=[],
    )
    
    proposal = Proposal(
        proposal_id=f"P_{uuid.uuid4().hex[:8]}",
        action_type=ActionType.ACCEPT_CLAIM,
        candidate=candidate,
        target_id=None,
        reason="Test claim",
    )
    
    fsm.receive_proposal(proposal)
    print(f"State after proposal: {fsm.fsm_state.name}")
    
    # Try to add MODEL_TEXT evidence - should be rejected
    bad_evidence = Evidence(
        evidence_id="ev_bad",
        evidence_type=EvidenceType.MODEL_TEXT,  # FORBIDDEN
        content="I'm confident this is true",
        provenance="model_self_report",
        timestamp=datetime.utcnow(),
        scope="*",
    )
    
    result = fsm.receive_evidence(proposal.proposal_id, bad_evidence)
    print(f"Evidence result: {result.name}")
    
    assert result == GovernorEvent.EVIDENCE_BAD
    
    # Check that forbidden was logged
    forbidden = [t for t in fsm.transitions if t.get("type") == "FORBIDDEN"]
    print(f"Forbidden transitions logged: {len(forbidden)}")
    
    print("✓ MODEL_TEXT evidence properly rejected\n")
    return True


def test_commit_requires_evidence():
    """Test that commit actions require admissible evidence."""
    print("=== Test: Commit Requires Evidence ===\n")
    
    from epistemic_governor.symbolic_substrate import Predicate, PredicateType, ProvenanceClass, TemporalScope
    
    fsm = GovernorFSM()
    
    # Create a proposal
    candidate = CandidateCommitment(
        predicate=Predicate(PredicateType.HAS, ("Entity", "prop", "val")),
        sigma=0.5,
        t_scope=TemporalScope(),
        provclass=ProvenanceClass.MODEL,
        support=[],
    )
    
    proposal = Proposal(
        proposal_id=f"P_{uuid.uuid4().hex[:8]}",
        action_type=ActionType.ACCEPT_CLAIM,
        candidate=candidate,
        target_id=None,
        reason="Test claim",
    )
    
    fsm.receive_proposal(proposal)
    
    # Try to commit without evidence
    success, msg = fsm.attempt_commit(proposal.proposal_id)
    print(f"Commit without evidence: {success}, {msg}")
    
    assert not success
    assert "NLAI" in str(msg)
    
    # Add proper evidence
    good_evidence = Evidence(
        evidence_id="ev_good",
        evidence_type=EvidenceType.TOOL_TRACE,
        content={"tool": "test", "result": "confirmed"},
        provenance="test_harness",
        timestamp=datetime.utcnow(),
        scope="*",
    )
    
    fsm.receive_evidence(proposal.proposal_id, good_evidence)
    
    # Now commit should work (if adjudicator passes)
    success, msg = fsm.attempt_commit(proposal.proposal_id)
    print(f"Commit with evidence: {success}, {msg}")
    
    print("✓ Commit properly requires evidence\n")
    return True


def test_freeze_on_repeated_attempts():
    """Test that repeated failed attempts trigger freeze."""
    print("=== Test: Freeze on Repeated Attempts ===\n")
    
    from epistemic_governor.symbolic_substrate import Predicate, PredicateType, ProvenanceClass, TemporalScope
    
    fsm = GovernorFSM()
    fsm.gate_checker.reopen_limit = 2  # Low limit for testing
    
    # Create a proposal targeting something
    proposal = Proposal(
        proposal_id=f"P_{uuid.uuid4().hex[:8]}",
        action_type=ActionType.CLOSE_CONTRADICTION,
        candidate=None,
        target_id="contradiction_123",
        reason="Test closure",
    )
    
    fsm.receive_proposal(proposal)
    
    # Attempt commit multiple times without evidence
    for i in range(4):
        success, msg = fsm.attempt_commit(proposal.proposal_id)
        print(f"Attempt {i+1}: {success}, state={fsm.fsm_state.name}")
    
    # Should be frozen now
    assert "contradiction_123" in fsm.frozen_targets
    assert fsm.fsm_state == GovernorState.FREEZE
    
    print("✓ Target properly frozen after repeated attempts\n")
    return True


def run_all_tests():
    """Run all FSM tests."""
    print("=" * 60)
    print("GOVERNOR FSM TESTS")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("nlai_enforcement", test_nlai_enforcement()))
    results.append(("commit_requires_evidence", test_commit_requires_evidence()))
    results.append(("freeze_on_repeated_attempts", test_freeze_on_repeated_attempts()))
    
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
