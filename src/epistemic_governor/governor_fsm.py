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
from datetime import datetime, timezone
import uuid
import hashlib
import json

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
# Commitment Mode (Finality as State Transition)
# =============================================================================

class CommitmentMode(Enum):
    """
    Commitment level for claims.
    
    Finality is a state transition, not a writing style.
    The model never decides its own finality - the FSM does.
    """
    PROPOSE = auto()              # Non-binding suggestion
    PROVISIONAL_COMMIT = auto()   # Binding but contestable
    FINAL_COMMIT = auto()         # Binding, requires evidence to reopen


class ContradictionState(Enum):
    """
    Lifecycle state for contradictions.
    
    Contradictions must be explicitly resolved or acknowledged,
    not rhetorically smoothed away.
    """
    DETECTED = auto()            # Just found, not yet addressed
    REPAIR_ACTIVE = auto()       # User/system actively working on it
    RESOLVED = auto()            # Contradiction reconciled
    ACCEPTED_DIVERGENCE = auto() # Both claims valid in different contexts


@dataclass
class ContradictionRecord:
    """
    A tracked contradiction with lifecycle state.
    
    Unlike simple (id1, id2, severity) tuples, this tracks
    the full lifecycle for CF-3 (Repair Suppression) detection.
    """
    contradiction_id: str
    claim_a_id: str
    claim_b_id: str
    severity: float = 1.0
    state: ContradictionState = ContradictionState.DETECTED
    detected_at: Optional[datetime] = None
    repair_started_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolution_evidence: Optional[str] = None  # Evidence that resolved it
    resolution_type: Optional[str] = None  # "reconciled", "retracted", "divergence"


@dataclass
class CommitmentContext:
    """
    Tracks commitment level and contest window for CF detection.
    
    This is the "finality is FSM state" implementation.
    """
    mode: CommitmentMode = CommitmentMode.PROPOSE
    mode_entered_at: Optional[datetime] = None
    user_acknowledged: bool = False
    escalation_blocked_reason: Optional[str] = None
    
    def can_escalate(
        self, 
        to_mode: CommitmentMode, 
        min_contest_seconds: float = 5.0,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if commitment can escalate.
        
        Enforces contest window (CF-2 prevention).
        """
        if to_mode.value <= self.mode.value:
            return True, None  # Not escalating
        
        if self.user_acknowledged:
            return True, None  # User explicitly approved
        
        if self.mode_entered_at is None:
            return True, None  # First transition
        
        elapsed = (datetime.now(timezone.utc) - self.mode_entered_at).total_seconds()
        if elapsed < min_contest_seconds:
            return False, f"CF-2: Contest window ({min_contest_seconds - elapsed:.1f}s remaining)"
        
        return True, None
    
    def escalate(self, to_mode: CommitmentMode):
        """Escalate commitment mode."""
        self.mode = to_mode
        self.mode_entered_at = datetime.now(timezone.utc)
        self.user_acknowledged = False
        self.escalation_blocked_reason = None


# =============================================================================
# Coordination Failure Events
# =============================================================================

class CFCode(Enum):
    """
    Coordination Failure codes.
    
    These are testable conditions, not rhetorical patterns.
    """
    CF_1 = "CF-1"  # Unilateral Closure: FINAL without contest path
    CF_2 = "CF-2"  # Asymmetric Tempo: escalation before contest window
    CF_3 = "CF-3"  # Repair Suppression: FINAL with open contradictions
    CF_4 = "CF-4"  # Implicit Authority: high σ on PROPOSE-mode claims
    CF_5 = "CF-5"  # Coordination Space Collapse: narrowing without trace


@dataclass
class CFEvent:
    """
    A logged coordination failure event.
    
    Emitted for diagnostics - gating comes later after observing patterns.
    """
    cf_code: CFCode
    timestamp: str
    fsm_state: str
    commitment_mode: str
    trigger: str  # What triggered detection
    claim_ids: List[str] = field(default_factory=list)
    contradiction_ids: List[str] = field(default_factory=list)
    contest_window_remaining: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Audit Trail (Forensic-grade provenance)
# =============================================================================

@dataclass
class AuditEntry:
    """
    Immutable audit record for state transitions.
    
    Provides forensic-grade provenance:
    - Hash chain: each entry hashes the previous, creating tamper-evident log
    - Parent pointer: causal link to triggering event
    - Evidence refs: IDs of evidence that authorized the transition
    """
    entry_id: str                          # Unique ID for this entry
    timestamp: str                         # ISO format UTC timestamp
    entry_type: str                        # "TRANSITION", "FORBIDDEN", "COMMIT", "RESET"
    
    # State transition info
    from_state: Optional[str] = None
    to_state: Optional[str] = None
    event: Optional[str] = None
    
    # Causal chain
    parent_entry_id: Optional[str] = None  # ID of entry that caused this one
    evidence_refs: List[str] = field(default_factory=list)  # Evidence IDs used
    
    # For FORBIDDEN entries
    forbidden_code: Optional[str] = None
    forbidden_description: Optional[str] = None
    
    # Additional context
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Integrity
    prev_hash: Optional[str] = None        # Hash of previous entry (chain)
    entry_hash: Optional[str] = None       # Hash of this entry (computed)
    
    def compute_hash(self) -> str:
        """Compute hash of this entry for integrity chain."""
        # Hash everything except entry_hash itself
        data = {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "entry_type": self.entry_type,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "event": self.event,
            "parent_entry_id": self.parent_entry_id,
            "evidence_refs": self.evidence_refs,
            "forbidden_code": self.forbidden_code,
            "forbidden_description": self.forbidden_description,
            "details": self.details,
            "prev_hash": self.prev_hash,
        }
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "entry_type": self.entry_type,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "event": self.event,
            "parent_entry_id": self.parent_entry_id,
            "evidence_refs": self.evidence_refs,
            "forbidden_code": self.forbidden_code,
            "forbidden_description": self.forbidden_description,
            "details": self.details,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
        }


class AuditLog:
    """
    Append-only audit log with hash chain integrity.
    
    Provides:
    - Tamper-evident chain (each entry hashes the previous)
    - Causal linking (parent pointers)
    - Evidence attribution (which evidence authorized what)
    """
    
    def __init__(self):
        self.entries: List[AuditEntry] = []
        self._last_hash: Optional[str] = None
    
    def append(
        self,
        entry_type: str,
        from_state: Optional[str] = None,
        to_state: Optional[str] = None,
        event: Optional[str] = None,
        parent_entry_id: Optional[str] = None,
        evidence_refs: Optional[List[str]] = None,
        forbidden_code: Optional[str] = None,
        forbidden_description: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Append a new entry to the audit log."""
        entry = AuditEntry(
            entry_id=f"AE_{uuid.uuid4().hex[:12]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            entry_type=entry_type,
            from_state=from_state,
            to_state=to_state,
            event=event,
            parent_entry_id=parent_entry_id,
            evidence_refs=evidence_refs or [],
            forbidden_code=forbidden_code,
            forbidden_description=forbidden_description,
            details=details or {},
            prev_hash=self._last_hash,
        )
        
        # Compute and set hash
        entry.entry_hash = entry.compute_hash()
        self._last_hash = entry.entry_hash
        
        self.entries.append(entry)
        return entry
    
    def verify_chain(self) -> Tuple[bool, Optional[str]]:
        """
        Verify the hash chain integrity.
        
        Returns (valid, error_message).
        """
        prev_hash = None
        for i, entry in enumerate(self.entries):
            # Check prev_hash matches
            if entry.prev_hash != prev_hash:
                return False, f"Entry {i} ({entry.entry_id}): prev_hash mismatch"
            
            # Recompute hash and verify
            computed = entry.compute_hash()
            if entry.entry_hash != computed:
                return False, f"Entry {i} ({entry.entry_id}): entry_hash mismatch"
            
            prev_hash = entry.entry_hash
        
        return True, None
    
    def get_entries(self) -> List[Dict[str, Any]]:
        """Get all entries as dicts."""
        return [e.to_dict() for e in self.entries]
    
    def get_causal_chain(self, entry_id: str) -> List[AuditEntry]:
        """Get the causal chain leading to an entry."""
        chain = []
        current_id = entry_id
        
        # Build index
        by_id = {e.entry_id: e for e in self.entries}
        
        while current_id and current_id in by_id:
            entry = by_id[current_id]
            chain.append(entry)
            current_id = entry.parent_entry_id
        
        return list(reversed(chain))
    
    def __len__(self) -> int:
        return len(self.entries)


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
        
        # Audit log (forensic-grade with hash chain)
        self.audit_log = AuditLog()
        
        # Legacy transition list (for backward compatibility)
        self.transitions: List[Dict[str, Any]] = []
        
        # Track current context for causal linking
        self._current_proposal_id: Optional[str] = None
        self._last_audit_entry_id: Optional[str] = None
        
        # === Coordination Failure Infrastructure ===
        
        # Commitment tracking (finality as state, not rhetoric)
        self.commitment_context = CommitmentContext()
        
        # Contradiction lifecycle tracking
        self.contradictions: Dict[str, ContradictionRecord] = {}
        
        # CF event log (diagnostic, not gating yet)
        self.cf_events: List[CFEvent] = []
        
        # Contest window configuration
        self.min_contest_window_seconds: float = 5.0
    
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
    
    def _transition(
        self, 
        new_state: GovernorState, 
        event: GovernorEvent,
        evidence_refs: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Record state transition with full audit trail."""
        old_state = self.fsm_state
        self.fsm_state = new_state
        
        # Create audit entry with causal link
        entry = self.audit_log.append(
            entry_type="TRANSITION",
            from_state=old_state.name,
            to_state=new_state.name,
            event=event.name,
            parent_entry_id=self._last_audit_entry_id,
            evidence_refs=evidence_refs or [],
            details=details or {},
        )
        self._last_audit_entry_id = entry.entry_id
        
        # Legacy format for backward compatibility
        self.transitions.append({
            "from": old_state.name,
            "to": new_state.name,
            "event": event.name,
            "timestamp": entry.timestamp,
            "entry_id": entry.entry_id,
            "entry_hash": entry.entry_hash,
        })
    
    def _log_forbidden(self, code: str, description: str, details: Optional[Dict[str, Any]] = None):
        """Log a forbidden transition attempt with full audit trail."""
        entry = self.audit_log.append(
            entry_type="FORBIDDEN",
            forbidden_code=code,
            forbidden_description=description,
            parent_entry_id=self._last_audit_entry_id,
            details=details or {},
        )
        self._last_audit_entry_id = entry.entry_id
        
        # Legacy format
        self.transitions.append({
            "type": "FORBIDDEN",
            "code": code,
            "description": description,
            "timestamp": entry.timestamp,
            "entry_id": entry.entry_id,
            "entry_hash": entry.entry_hash,
        })
    
    def _log_commit(
        self, 
        proposal_id: str, 
        commitment_id: str,
        evidence_refs: List[str],
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log a successful commit with evidence attribution."""
        entry = self.audit_log.append(
            entry_type="COMMIT",
            parent_entry_id=self._last_audit_entry_id,
            evidence_refs=evidence_refs,
            details={
                "proposal_id": proposal_id,
                "commitment_id": commitment_id,
                **(details or {}),
            },
        )
        self._last_audit_entry_id = entry.entry_id
    
    def get_state(self) -> Dict[str, Any]:
        """Get current FSM state."""
        return {
            "fsm_state": self.fsm_state.name,
            "pending_proposals": len(self.proposals),
            "frozen_targets": list(self.frozen_targets),
            "committed_claims": len(self.symbolic_state.commitments),
            "total_sigma": self.symbolic_state.total_sigma_allocated,
            "audit_entries": len(self.audit_log),
            "commitment_mode": self.commitment_context.mode.name,
            "open_contradictions": len([c for c in self.contradictions.values() 
                                       if c.state in [ContradictionState.DETECTED, ContradictionState.REPAIR_ACTIVE]]),
            "cf_events": len(self.cf_events),
        }
    
    def get_transitions(self) -> List[Dict[str, Any]]:
        """Get transition history (legacy format)."""
        return self.transitions
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get full audit log with hash chain."""
        return self.audit_log.get_entries()
    
    def verify_audit_chain(self) -> Tuple[bool, Optional[str]]:
        """Verify audit log integrity."""
        return self.audit_log.verify_chain()
    
    # =========================================================================
    # Contradiction Lifecycle Management
    # =========================================================================
    
    def register_contradiction(
        self, 
        claim_a_id: str, 
        claim_b_id: str, 
        severity: float = 1.0,
    ) -> ContradictionRecord:
        """
        Register a new contradiction.
        
        Contradictions must be explicitly tracked and resolved,
        not rhetorically smoothed away.
        """
        c_id = f"C_{uuid.uuid4().hex[:12]}"
        record = ContradictionRecord(
            contradiction_id=c_id,
            claim_a_id=claim_a_id,
            claim_b_id=claim_b_id,
            severity=severity,
            state=ContradictionState.DETECTED,
            detected_at=datetime.now(timezone.utc),
        )
        self.contradictions[c_id] = record
        
        # Log in audit trail
        self.audit_log.append(
            entry_type="CONTRADICTION_DETECTED",
            parent_entry_id=self._last_audit_entry_id,
            details={
                "contradiction_id": c_id,
                "claim_a": claim_a_id,
                "claim_b": claim_b_id,
                "severity": severity,
            },
        )
        
        return record
    
    def start_repair(self, contradiction_id: str) -> bool:
        """
        Mark contradiction as actively being repaired.
        
        Returns False if contradiction not found or already resolved.
        """
        if contradiction_id not in self.contradictions:
            return False
        
        record = self.contradictions[contradiction_id]
        if record.state not in [ContradictionState.DETECTED, ContradictionState.REPAIR_ACTIVE]:
            return False
        
        record.state = ContradictionState.REPAIR_ACTIVE
        record.repair_started_at = datetime.now(timezone.utc)
        
        self.audit_log.append(
            entry_type="REPAIR_STARTED",
            parent_entry_id=self._last_audit_entry_id,
            details={"contradiction_id": contradiction_id},
        )
        
        return True
    
    def resolve_contradiction(
        self, 
        contradiction_id: str, 
        resolution_type: str,
        evidence_id: Optional[str] = None,
    ) -> bool:
        """
        Resolve a contradiction.
        
        Args:
            contradiction_id: ID of contradiction to resolve
            resolution_type: "reconciled", "retracted", or "divergence"
            evidence_id: Evidence that supports resolution (required for reconciled/retracted)
        
        Returns False if contradiction not found.
        """
        if contradiction_id not in self.contradictions:
            return False
        
        record = self.contradictions[contradiction_id]
        
        if resolution_type == "divergence":
            record.state = ContradictionState.ACCEPTED_DIVERGENCE
        else:
            record.state = ContradictionState.RESOLVED
        
        record.resolved_at = datetime.now(timezone.utc)
        record.resolution_type = resolution_type
        record.resolution_evidence = evidence_id
        
        self.audit_log.append(
            entry_type="CONTRADICTION_RESOLVED",
            parent_entry_id=self._last_audit_entry_id,
            evidence_refs=[evidence_id] if evidence_id else [],
            details={
                "contradiction_id": contradiction_id,
                "resolution_type": resolution_type,
            },
        )
        
        return True
    
    def get_open_contradictions(self) -> List[ContradictionRecord]:
        """Get all unresolved contradictions."""
        return [c for c in self.contradictions.values() 
                if c.state in [ContradictionState.DETECTED, ContradictionState.REPAIR_ACTIVE]]
    
    # =========================================================================
    # Coordination Failure Detection
    # =========================================================================
    
    def check_cf_violations(
        self, 
        target_commitment_mode: CommitmentMode,
        claim_ids: Optional[List[str]] = None,
    ) -> List[CFEvent]:
        """
        Check for coordination failures before a commitment escalation.
        
        This is diagnostic - it logs but doesn't block (yet).
        After observing patterns, gating can be added.
        """
        events = []
        now_str = datetime.now(timezone.utc).isoformat()
        claim_ids = claim_ids or []
        
        open_contradictions = self.get_open_contradictions()
        contradiction_ids = [c.contradiction_id for c in open_contradictions]
        
        # CF-1: Unilateral Closure
        # Entering FINAL without available contest path
        if target_commitment_mode == CommitmentMode.FINAL_COMMIT:
            if not self.commitment_context.user_acknowledged:
                events.append(CFEvent(
                    cf_code=CFCode.CF_1,
                    timestamp=now_str,
                    fsm_state=self.fsm_state.name,
                    commitment_mode=self.commitment_context.mode.name,
                    trigger="FINAL_COMMIT without user acknowledgment",
                    claim_ids=claim_ids,
                ))
        
        # CF-2: Asymmetric Tempo
        # Escalating before contest window elapsed
        can_escalate, reason = self.commitment_context.can_escalate(
            target_commitment_mode, 
            self.min_contest_window_seconds
        )
        if not can_escalate and reason and "CF-2" in reason:
            elapsed = 0.0
            if self.commitment_context.mode_entered_at:
                elapsed = (datetime.now(timezone.utc) - self.commitment_context.mode_entered_at).total_seconds()
            
            events.append(CFEvent(
                cf_code=CFCode.CF_2,
                timestamp=now_str,
                fsm_state=self.fsm_state.name,
                commitment_mode=self.commitment_context.mode.name,
                trigger=f"Escalation attempted with {self.min_contest_window_seconds - elapsed:.1f}s remaining",
                claim_ids=claim_ids,
                contest_window_remaining=self.min_contest_window_seconds - elapsed,
            ))
        
        # CF-3: Repair Suppression
        # Attempting FINAL with open contradictions
        if target_commitment_mode == CommitmentMode.FINAL_COMMIT and open_contradictions:
            events.append(CFEvent(
                cf_code=CFCode.CF_3,
                timestamp=now_str,
                fsm_state=self.fsm_state.name,
                commitment_mode=self.commitment_context.mode.name,
                trigger=f"FINAL_COMMIT with {len(open_contradictions)} unresolved contradictions",
                claim_ids=claim_ids,
                contradiction_ids=contradiction_ids,
                details={"contradiction_count": len(open_contradictions)},
            ))
        
        # Log all detected CF events
        for event in events:
            self.cf_events.append(event)
            self.audit_log.append(
                entry_type="CF_DETECTED",
                details={
                    "cf_code": event.cf_code.value,
                    "trigger": event.trigger,
                    "claim_ids": event.claim_ids,
                    "contradiction_ids": event.contradiction_ids,
                },
            )
        
        return events
    
    def escalate_commitment(
        self, 
        to_mode: CommitmentMode,
        claim_ids: Optional[List[str]] = None,
        force: bool = False,
    ) -> Tuple[bool, Optional[str], List[CFEvent]]:
        """
        Attempt to escalate commitment mode.
        
        Returns:
            (success, reason, cf_events)
        
        If force=False, CF violations are logged but don't block.
        If force=True, CF violations are skipped entirely.
        """
        # Check for CF violations
        cf_events = [] if force else self.check_cf_violations(to_mode, claim_ids)
        
        # Check contest window
        can_escalate, reason = self.commitment_context.can_escalate(
            to_mode, 
            self.min_contest_window_seconds
        )
        
        if not can_escalate and not force:
            return False, reason, cf_events
        
        # Escalate
        self.commitment_context.escalate(to_mode)
        
        # Log transition
        self.audit_log.append(
            entry_type="COMMITMENT_ESCALATED",
            details={
                "to_mode": to_mode.name,
                "cf_events_logged": len(cf_events),
                "forced": force,
            },
        )
        
        return True, None, cf_events
    
    def acknowledge_commitment(self):
        """User acknowledges current commitment level, enabling further escalation."""
        self.commitment_context.user_acknowledged = True
        self.audit_log.append(
            entry_type="USER_ACKNOWLEDGMENT",
            details={"commitment_mode": self.commitment_context.mode.name},
        )
    
    def get_cf_events(self) -> List[Dict[str, Any]]:
        """Get all CF events for diagnostics."""
        return [
            {
                "cf_code": e.cf_code.value,
                "timestamp": e.timestamp,
                "fsm_state": e.fsm_state,
                "commitment_mode": e.commitment_mode,
                "trigger": e.trigger,
                "claim_ids": e.claim_ids,
                "contradiction_ids": e.contradiction_ids,
                "contest_window_remaining": e.contest_window_remaining,
                "details": e.details,
            }
            for e in self.cf_events
        ]


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
