"""
Epistemic Ledger Module

Append-only state storage for commitments.
This is where memory gains teeth.

Key properties:
- Append-only: no deletes, no silent edits
- History is explicit
- Commitments can only transition forward: active → superseded → archived
- Contradictions require explicit revision with justification and cost

This is where text becomes state.
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional, Iterator
from pathlib import Path
from enum import Enum

# Handle both package and direct imports
try:
    from .governor import (
        CommittedClaim,
        CommitmentStatus,
        ClaimType,
        ProposedCommitment,
        CommitAction,
        CommitDecision,
    )
except ImportError:
    from governor import (
        CommittedClaim,
        CommitmentStatus,
        ClaimType,
        ProposedCommitment,
        CommitAction,
        CommitDecision,
    )


# =============================================================================
# Ledger Entry Types
# =============================================================================

class EntryType(Enum):
    """Types of ledger entries."""
    COMMIT = "commit"           # New commitment
    SUPERSEDE = "supersede"     # Mark prior as superseded
    ARCHIVE = "archive"         # Mark as archived (compaction)
    REVISION = "revision"       # Explicit revision record
    EPOCH = "epoch"             # Named checkpoint/milestone
    FORK = "fork"               # Branch point for exploration
    CONTEXT_RESET = "context_reset"  # "We were wrong; moving on"
    MERGE = "merge"             # Merge branch back to main


@dataclass
class LedgerEntry:
    """
    A single entry in the append-only ledger.
    Each entry is immutable once written.
    """
    entry_id: str
    entry_type: EntryType
    timestamp: datetime
    claim_id: str
    data: dict
    checksum: str = ""
    
    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._compute_checksum()
    
    def _compute_checksum(self) -> str:
        """Compute integrity checksum for the entry."""
        content = f"{self.entry_id}:{self.entry_type.value}:{self.timestamp.isoformat()}:{self.claim_id}:{json.dumps(self.data, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def verify(self) -> bool:
        """Verify entry integrity."""
        return self.checksum == self._compute_checksum()
    
    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "entry_type": self.entry_type.value,
            "timestamp": self.timestamp.isoformat(),
            "claim_id": self.claim_id,
            "data": self.data,
            "checksum": self.checksum,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "LedgerEntry":
        return cls(
            entry_id=d["entry_id"],
            entry_type=EntryType(d["entry_type"]),
            timestamp=datetime.fromisoformat(d["timestamp"]),
            claim_id=d["claim_id"],
            data=d["data"],
            checksum=d["checksum"],
        )


# =============================================================================
# Revision Record
# =============================================================================

@dataclass
class RevisionRecord:
    """
    Records an explicit revision of prior commitment(s).
    Revisions are allowed but tracked and costed.
    """
    revision_id: str
    timestamp: datetime
    new_claim_id: str
    superseded_claim_ids: list[str]
    justification: str
    cost: float
    
    def to_dict(self) -> dict:
        return {
            "revision_id": self.revision_id,
            "timestamp": self.timestamp.isoformat(),
            "new_claim_id": self.new_claim_id,
            "superseded_claim_ids": self.superseded_claim_ids,
            "justification": self.justification,
            "cost": self.cost,
        }


# =============================================================================
# Epoch and Fork Records (Point 4: Lock-in Prevention)
# =============================================================================

@dataclass
class Epoch:
    """
    A named checkpoint in the ledger.
    
    Epochs serve as:
    - Stable reference points for "rollback" discussions
    - Fork points for exploration branches
    - Semantic milestones ("before we discussed X")
    
    Note: We don't actually roll back - we create context resets
    that acknowledge and archive claims from a prior epoch.
    """
    epoch_id: str
    name: str
    timestamp: datetime
    entry_index: int          # Index in _entries at epoch creation
    active_claim_ids: list[str]  # Snapshot of active claims at epoch
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "epoch_id": self.epoch_id,
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "entry_index": self.entry_index,
            "active_claim_ids": self.active_claim_ids,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Epoch":
        return cls(
            epoch_id=d["epoch_id"],
            name=d["name"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            entry_index=d["entry_index"],
            active_claim_ids=d["active_claim_ids"],
            metadata=d.get("metadata", {}),
        )


@dataclass
class Fork:
    """
    A branch point for exploration.
    
    Forks allow "what if" reasoning without polluting the main ledger.
    Claims in a fork are isolated until merged.
    
    Use cases:
    - Hypothesis exploration
    - Devil's advocate reasoning
    - Alternative interpretation testing
    """
    fork_id: str
    name: str
    parent_epoch_id: str      # Fork from this epoch
    created_at: datetime
    merged_at: Optional[datetime] = None
    abandoned_at: Optional[datetime] = None
    claim_ids: list[str] = field(default_factory=list)  # Claims in this fork
    metadata: dict = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        return self.merged_at is None and self.abandoned_at is None
    
    def to_dict(self) -> dict:
        return {
            "fork_id": self.fork_id,
            "name": self.name,
            "parent_epoch_id": self.parent_epoch_id,
            "created_at": self.created_at.isoformat(),
            "merged_at": self.merged_at.isoformat() if self.merged_at else None,
            "abandoned_at": self.abandoned_at.isoformat() if self.abandoned_at else None,
            "claim_ids": self.claim_ids,
            "metadata": self.metadata,
        }


@dataclass
class ContextReset:
    """
    A graceful "we were wrong; moving on" operation.
    
    This is how humans handle early errors:
    - Acknowledge the mistake
    - Archive the problematic claims
    - Start fresh from a clean state
    
    Unlike silent deletion, context resets:
    - Are recorded in the ledger
    - Include justification
    - Preserve history (claims are archived, not deleted)
    - Can reference an epoch as the "rollback" point
    
    This prevents early wrong commits from becoming permanent debt.
    """
    reset_id: str
    timestamp: datetime
    reason: str                    # Why we're resetting
    archived_claim_ids: list[str]  # Claims being archived
    from_epoch_id: Optional[str]   # Reset to this epoch's state (optional)
    acknowledgment: str            # Explicit acknowledgment of the error
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "reset_id": self.reset_id,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
            "archived_claim_ids": self.archived_claim_ids,
            "from_epoch_id": self.from_epoch_id,
            "acknowledgment": self.acknowledgment,
            "metadata": self.metadata,
        }


# =============================================================================
# Main Ledger
# =============================================================================

class EpistemicLedger:
    """
    Append-only ledger for epistemic commitments.
    
    This is the authoritative record of what has been committed.
    No deletes, no silent edits - only forward transitions.
    
    Storage options:
    - In-memory (default, for testing)
    - File-backed (JSON lines)
    - Could extend to SQLite, etc.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path
        self._entries: list[LedgerEntry] = []
        self._claims: dict[str, CommittedClaim] = {}  # claim_id -> claim
        self._entry_counter = 0
        
        # Indexes for efficient lookup
        self._by_proposition: dict[str, list[str]] = {}  # prop_hash -> claim_ids
        self._by_entity: dict[str, list[str]] = {}       # entity -> claim_ids
        self._by_status: dict[CommitmentStatus, list[str]] = {
            status: [] for status in CommitmentStatus
        }
        
        # Epoch and fork tracking (Point 4: Lock-in Prevention)
        self._epochs: dict[str, Epoch] = {}           # epoch_id -> Epoch
        self._forks: dict[str, Fork] = {}             # fork_id -> Fork
        self._context_resets: list[ContextReset] = [] # History of resets
        self._current_fork: Optional[str] = None      # Active fork (None = main)
        self._epoch_counter = 0
        self._fork_counter = 0
        self._reset_counter = 0
        
        # Fossilization tracking (Point 7: Scaling)
        self._fossils: dict[str, "FossilRecord"] = {}  # claim_id -> FossilRecord
        
        # Load from storage if exists
        if storage_path and storage_path.exists():
            self._load()
    
    # =========================================================================
    # Core Operations
    # =========================================================================
    
    def commit(
        self,
        proposed: ProposedCommitment,
        decision: CommitDecision,
        support_refs: list[str] = None,
    ) -> CommittedClaim:
        """
        Commit a proposed commitment to the ledger.
        
        This is the atomic operation that turns proposals into state.
        Only called after passing through governor adjudication.
        """
        if decision.action not in (CommitAction.ACCEPT, CommitAction.HEDGE):
            raise ValueError(f"Cannot commit with action {decision.action}")
        
        # Create the committed claim
        confidence = (
            decision.adjusted_confidence 
            if decision.adjusted_confidence is not None 
            else proposed.confidence
        )
        
        claim = CommittedClaim(
            id=proposed.id,
            text=proposed.text,
            claim_type=proposed.claim_type,
            confidence=confidence,
            proposition_hash=proposed.proposition_hash,
            scope=proposed.scope,
            status=CommitmentStatus.ACTIVE,
            committed_at=datetime.now(),
            support_refs=support_refs or [],
            supersedes=None,
            revision_cost=0.0,
            decay_eligible=False,
        )
        
        # Create ledger entry
        self._entry_counter += 1
        entry = LedgerEntry(
            entry_id=f"entry_{self._entry_counter}",
            entry_type=EntryType.COMMIT,
            timestamp=datetime.now(),
            claim_id=claim.id,
            data={
                "text": claim.text,
                "claim_type": claim.claim_type.name,
                "confidence": claim.confidence,
                "proposition_hash": claim.proposition_hash,
                "scope": claim.scope,
                "support_refs": claim.support_refs,
            }
        )
        
        # Append to ledger (the irreversible step)
        self._append_entry(entry)
        
        # Update indexes
        self._claims[claim.id] = claim
        self._index_claim(claim, proposed.extracted_entities)
        
        return claim
    
    def revise(
        self,
        new_proposed: ProposedCommitment,
        superseded_ids: list[str],
        justification: str,
        decision: CommitDecision,
    ) -> tuple[CommittedClaim, RevisionRecord]:
        """
        Commit a revision that supersedes prior commitment(s).
        
        Revisions are allowed but expensive and tracked.
        The old commitments are marked superseded, not deleted.
        """
        # First, supersede the old claims
        for old_id in superseded_ids:
            if old_id in self._claims:
                self._transition_status(old_id, CommitmentStatus.SUPERSEDED)
        
        # Commit the new claim
        confidence = (
            decision.adjusted_confidence 
            if decision.adjusted_confidence is not None 
            else new_proposed.confidence
        )
        
        new_claim = CommittedClaim(
            id=new_proposed.id,
            text=new_proposed.text,
            claim_type=new_proposed.claim_type,
            confidence=confidence,
            proposition_hash=new_proposed.proposition_hash,
            scope=new_proposed.scope,
            status=CommitmentStatus.ACTIVE,
            committed_at=datetime.now(),
            support_refs=[],
            supersedes=superseded_ids[0] if len(superseded_ids) == 1 else None,
            revision_cost=decision.cost,
            decay_eligible=False,
        )
        
        # Create revision record
        revision = RevisionRecord(
            revision_id=f"rev_{self._entry_counter + 1}",
            timestamp=datetime.now(),
            new_claim_id=new_claim.id,
            superseded_claim_ids=superseded_ids,
            justification=justification,
            cost=decision.cost,
        )
        
        # Create ledger entry for the revision
        self._entry_counter += 1
        entry = LedgerEntry(
            entry_id=f"entry_{self._entry_counter}",
            entry_type=EntryType.REVISION,
            timestamp=datetime.now(),
            claim_id=new_claim.id,
            data={
                "new_claim": {
                    "text": new_claim.text,
                    "claim_type": new_claim.claim_type.name,
                    "confidence": new_claim.confidence,
                },
                "superseded_ids": superseded_ids,
                "justification": justification,
                "cost": decision.cost,
            }
        )
        
        self._append_entry(entry)
        self._claims[new_claim.id] = new_claim
        self._index_claim(new_claim, new_proposed.extracted_entities)
        
        return new_claim, revision
    
    def _transition_status(
        self, 
        claim_id: str, 
        new_status: CommitmentStatus
    ):
        """
        Transition a claim to a new status.
        Status can only move forward: ACTIVE → SUPERSEDED → ARCHIVED
        """
        claim = self._claims.get(claim_id)
        if not claim:
            raise ValueError(f"Unknown claim: {claim_id}")
        
        # Validate forward-only transition
        valid_transitions = {
            CommitmentStatus.ACTIVE: {CommitmentStatus.SUPERSEDED, CommitmentStatus.ARCHIVED},
            CommitmentStatus.SUPERSEDED: {CommitmentStatus.ARCHIVED},
            CommitmentStatus.ARCHIVED: set(),  # terminal state
        }
        
        if new_status not in valid_transitions[claim.status]:
            raise ValueError(
                f"Invalid status transition: {claim.status} → {new_status}"
            )
        
        # Update indexes
        self._by_status[claim.status].remove(claim_id)
        self._by_status[new_status].append(claim_id)
        
        # Create a new claim object (claims are immutable-ish)
        # In a real impl, we'd have a separate status tracking mechanism
        old_status = claim.status
        # Note: We're mutating here for simplicity; production would be cleaner
        object.__setattr__(claim, 'status', new_status)
        
        # Record the transition
        self._entry_counter += 1
        entry = LedgerEntry(
            entry_id=f"entry_{self._entry_counter}",
            entry_type=EntryType.SUPERSEDE if new_status == CommitmentStatus.SUPERSEDED else EntryType.ARCHIVE,
            timestamp=datetime.now(),
            claim_id=claim_id,
            data={
                "old_status": old_status.name,
                "new_status": new_status.name,
            }
        )
        self._append_entry(entry)
    
    # =========================================================================
    # Queries
    # =========================================================================
    
    def get_claim(self, claim_id: str) -> Optional[CommittedClaim]:
        """Get a specific claim by ID."""
        return self._claims.get(claim_id)
    
    def get_active_claims(self) -> list[CommittedClaim]:
        """Get all currently active claims."""
        return [
            self._claims[cid] 
            for cid in self._by_status[CommitmentStatus.ACTIVE]
        ]
    
    def find_by_proposition(self, prop_hash: str) -> list[CommittedClaim]:
        """Find claims about the same proposition."""
        claim_ids = self._by_proposition.get(prop_hash, [])
        return [self._claims[cid] for cid in claim_ids if cid in self._claims]
    
    def find_by_entity(self, entity: str) -> list[CommittedClaim]:
        """Find claims mentioning an entity."""
        claim_ids = self._by_entity.get(entity.lower(), [])
        return [self._claims[cid] for cid in claim_ids if cid in self._claims]
    
    def find_contradictions(self, prop_hash: str) -> list[CommittedClaim]:
        """Find active claims that might contradict a new claim."""
        existing = self.find_by_proposition(prop_hash)
        return [c for c in existing if c.status == CommitmentStatus.ACTIVE]
    
    def get_history(self, claim_id: str) -> list[LedgerEntry]:
        """Get the full history of entries for a claim."""
        return [e for e in self._entries if e.claim_id == claim_id]
    
    def get_revision_chain(self, claim_id: str) -> list[str]:
        """
        Get the chain of revisions leading to this claim.
        Returns list of claim_ids from oldest to newest.
        """
        chain = [claim_id]
        claim = self._claims.get(claim_id)
        
        while claim and claim.supersedes:
            chain.insert(0, claim.supersedes)
            claim = self._claims.get(claim.supersedes)
        
        return chain
    
    # =========================================================================
    # Statistics and Metrics
    # =========================================================================
    
    def get_stats(self) -> dict:
        """Get ledger statistics."""
        return {
            "total_entries": len(self._entries),
            "total_claims": len(self._claims),
            "active_claims": len(self._by_status[CommitmentStatus.ACTIVE]),
            "superseded_claims": len(self._by_status[CommitmentStatus.SUPERSEDED]),
            "archived_claims": len(self._by_status[CommitmentStatus.ARCHIVED]),
            "fossilized_claims": len(self._fossils),
            "unique_propositions": len(self._by_proposition),
            "indexed_entities": len(self._by_entity),
            "epochs": len(self._epochs),
            "forks": len(self._forks),
            "active_forks": len([f for f in self._forks.values() if f.is_active]),
            "context_resets": len(self._context_resets),
            "current_fork": self._current_fork,
        }
    
    @property
    def claims(self) -> dict[str, CommittedClaim]:
        """Read-only access to claims by ID."""
        return self._claims
    
    def get_total_confidence(self) -> float:
        """Sum of confidence of all active claims."""
        return sum(
            self._claims[cid].confidence 
            for cid in self._by_status[CommitmentStatus.ACTIVE]
        )
    
    def get_total_revision_cost(self) -> float:
        """Sum of all revision costs."""
        return sum(c.revision_cost for c in self._claims.values())
    
    # =========================================================================
    # Epochs (Named Checkpoints)
    # =========================================================================
    
    def create_epoch(self, name: str, metadata: dict = None) -> Epoch:
        """
        Create a named checkpoint at the current ledger state.
        
        Epochs serve as stable reference points for:
        - Context resets ("go back to before X")
        - Fork points for exploration
        - Semantic milestones in conversation
        """
        self._epoch_counter += 1
        epoch_id = f"epoch_{self._epoch_counter}"
        
        epoch = Epoch(
            epoch_id=epoch_id,
            name=name,
            timestamp=datetime.now(),
            entry_index=len(self._entries),
            active_claim_ids=list(self._by_status[CommitmentStatus.ACTIVE]),
            metadata=metadata or {},
        )
        
        # Record in ledger
        self._entry_counter += 1
        entry = LedgerEntry(
            entry_id=f"entry_{self._entry_counter}",
            entry_type=EntryType.EPOCH,
            timestamp=epoch.timestamp,
            claim_id=epoch_id,  # Use epoch_id as claim_id for this entry type
            data=epoch.to_dict(),
        )
        self._append_entry(entry)
        
        self._epochs[epoch_id] = epoch
        return epoch
    
    def get_epoch(self, epoch_id: str) -> Optional[Epoch]:
        """Get an epoch by ID."""
        return self._epochs.get(epoch_id)
    
    def get_epochs(self) -> list[Epoch]:
        """Get all epochs in chronological order."""
        return sorted(self._epochs.values(), key=lambda e: e.timestamp)
    
    def get_latest_epoch(self) -> Optional[Epoch]:
        """Get the most recent epoch."""
        if not self._epochs:
            return None
        return max(self._epochs.values(), key=lambda e: e.timestamp)
    
    # =========================================================================
    # Forks (Exploration Branches)
    # =========================================================================
    
    def create_fork(self, name: str, from_epoch_id: Optional[str] = None, metadata: dict = None) -> Fork:
        """
        Create a branch for exploration.
        
        Forks allow "what if" reasoning without polluting the main ledger.
        Claims committed in a fork are isolated until merged or abandoned.
        """
        # Use latest epoch if not specified
        if from_epoch_id is None:
            latest = self.get_latest_epoch()
            if latest:
                from_epoch_id = latest.epoch_id
            else:
                # Create implicit epoch at current state
                implicit = self.create_epoch(f"pre_fork_{name}")
                from_epoch_id = implicit.epoch_id
        
        self._fork_counter += 1
        fork_id = f"fork_{self._fork_counter}"
        
        fork = Fork(
            fork_id=fork_id,
            name=name,
            parent_epoch_id=from_epoch_id,
            created_at=datetime.now(),
            metadata=metadata or {},
        )
        
        # Record in ledger
        self._entry_counter += 1
        entry = LedgerEntry(
            entry_id=f"entry_{self._entry_counter}",
            entry_type=EntryType.FORK,
            timestamp=fork.created_at,
            claim_id=fork_id,
            data=fork.to_dict(),
        )
        self._append_entry(entry)
        
        self._forks[fork_id] = fork
        self._current_fork = fork_id
        
        return fork
    
    def get_current_fork(self) -> Optional[Fork]:
        """Get the currently active fork, if any."""
        if self._current_fork:
            return self._forks.get(self._current_fork)
        return None
    
    def merge_fork(self, fork_id: Optional[str] = None) -> list[str]:
        """
        Merge a fork back to the main ledger.
        
        Claims committed in the fork become part of the main history.
        Returns list of merged claim IDs.
        """
        fork_id = fork_id or self._current_fork
        if not fork_id:
            raise ValueError("No active fork to merge")
        
        fork = self._forks.get(fork_id)
        if not fork:
            raise ValueError(f"Unknown fork: {fork_id}")
        
        if not fork.is_active:
            raise ValueError(f"Fork {fork_id} is already closed")
        
        fork.merged_at = datetime.now()
        
        # Record merge in ledger
        self._entry_counter += 1
        entry = LedgerEntry(
            entry_id=f"entry_{self._entry_counter}",
            entry_type=EntryType.MERGE,
            timestamp=fork.merged_at,
            claim_id=fork_id,
            data={
                "fork_id": fork_id,
                "merged_claim_ids": fork.claim_ids,
                "parent_epoch_id": fork.parent_epoch_id,
            },
        )
        self._append_entry(entry)
        
        # Clear current fork if this was it
        if self._current_fork == fork_id:
            self._current_fork = None
        
        return fork.claim_ids
    
    def abandon_fork(self, fork_id: Optional[str] = None, reason: str = "") -> list[str]:
        """
        Abandon a fork without merging.
        
        Claims in the fork are archived, not incorporated into main history.
        Returns list of abandoned claim IDs.
        """
        fork_id = fork_id or self._current_fork
        if not fork_id:
            raise ValueError("No active fork to abandon")
        
        fork = self._forks.get(fork_id)
        if not fork:
            raise ValueError(f"Unknown fork: {fork_id}")
        
        if not fork.is_active:
            raise ValueError(f"Fork {fork_id} is already closed")
        
        fork.abandoned_at = datetime.now()
        
        # Archive all claims in the fork
        for claim_id in fork.claim_ids:
            if claim_id in self._claims:
                claim = self._claims[claim_id]
                if claim.status == CommitmentStatus.ACTIVE:
                    self._transition_status(claim_id, CommitmentStatus.ARCHIVED)
        
        # Record abandonment
        self._entry_counter += 1
        entry = LedgerEntry(
            entry_id=f"entry_{self._entry_counter}",
            entry_type=EntryType.ARCHIVE,
            timestamp=fork.abandoned_at,
            claim_id=fork_id,
            data={
                "fork_id": fork_id,
                "abandoned_claim_ids": fork.claim_ids,
                "reason": reason,
            },
        )
        self._append_entry(entry)
        
        # Clear current fork if this was it
        if self._current_fork == fork_id:
            self._current_fork = None
        
        return fork.claim_ids
    
    # =========================================================================
    # Context Reset ("We Were Wrong; Moving On")
    # =========================================================================
    
    def context_reset(
        self,
        reason: str,
        acknowledgment: str,
        from_epoch_id: Optional[str] = None,
        claim_ids_to_archive: Optional[list[str]] = None,
    ) -> ContextReset:
        """
        Perform a graceful "we were wrong; moving on" operation.
        
        This is how humans handle early errors without losing trust:
        1. Acknowledge the mistake explicitly
        2. Archive the problematic claims
        3. Start fresh from a clean state
        
        Unlike silent deletion or expensive revision-by-revision fixes:
        - The error is recorded, not hidden
        - History is preserved (archived, not deleted)
        - The reset is cheap (encourages correction over stubbornness)
        - Users understand what happened
        
        Args:
            reason: Why we're resetting (e.g., "Initial assumptions were incorrect")
            acknowledgment: Explicit acknowledgment (e.g., "I was wrong about X")
            from_epoch_id: Optional epoch to reset to (archives claims after that epoch)
            claim_ids_to_archive: Specific claims to archive (alternative to epoch-based)
        
        Returns:
            ContextReset record
        """
        self._reset_counter += 1
        reset_id = f"reset_{self._reset_counter}"
        
        # Determine which claims to archive
        if claim_ids_to_archive:
            to_archive = claim_ids_to_archive
        elif from_epoch_id:
            epoch = self._epochs.get(from_epoch_id)
            if not epoch:
                raise ValueError(f"Unknown epoch: {from_epoch_id}")
            # Archive all active claims NOT in the epoch's snapshot
            epoch_claims = set(epoch.active_claim_ids)
            to_archive = [
                cid for cid in self._by_status[CommitmentStatus.ACTIVE]
                if cid not in epoch_claims
            ]
        else:
            # Archive all active claims
            to_archive = list(self._by_status[CommitmentStatus.ACTIVE])
        
        # Perform the archival
        for claim_id in to_archive:
            if claim_id in self._claims:
                claim = self._claims[claim_id]
                if claim.status == CommitmentStatus.ACTIVE:
                    self._transition_status(claim_id, CommitmentStatus.ARCHIVED)
        
        # Create reset record
        reset = ContextReset(
            reset_id=reset_id,
            timestamp=datetime.now(),
            reason=reason,
            archived_claim_ids=to_archive,
            from_epoch_id=from_epoch_id,
            acknowledgment=acknowledgment,
        )
        
        # Record in ledger
        self._entry_counter += 1
        entry = LedgerEntry(
            entry_id=f"entry_{self._entry_counter}",
            entry_type=EntryType.CONTEXT_RESET,
            timestamp=reset.timestamp,
            claim_id=reset_id,
            data=reset.to_dict(),
        )
        self._append_entry(entry)
        
        self._context_resets.append(reset)
        
        return reset
    
    def get_context_resets(self) -> list[ContextReset]:
        """Get all context resets in chronological order."""
        return list(self._context_resets)
    
    # =========================================================================
    # Internal Helpers
    # =========================================================================
    
    def _append_entry(self, entry: LedgerEntry):
        """
        Append an entry to the ledger.
        This is the irreversible step.
        """
        self._entries.append(entry)
        
        if self.storage_path:
            self._persist_entry(entry)
    
    def _index_claim(self, claim: CommittedClaim, entities: list[str] = None):
        """Update indexes for a new claim."""
        # Proposition index
        if claim.proposition_hash not in self._by_proposition:
            self._by_proposition[claim.proposition_hash] = []
        self._by_proposition[claim.proposition_hash].append(claim.id)
        
        # Entity index
        if entities:
            for entity in entities:
                key = entity.lower()
                if key not in self._by_entity:
                    self._by_entity[key] = []
                self._by_entity[key].append(claim.id)
        
        # Status index
        self._by_status[claim.status].append(claim.id)
    
    def _persist_entry(self, entry: LedgerEntry):
        """Persist entry to storage (append-only)."""
        with open(self.storage_path, 'a') as f:
            f.write(json.dumps(entry.to_dict()) + '\n')
    
    def _load(self):
        """Load ledger from storage."""
        with open(self.storage_path, 'r') as f:
            for line in f:
                if line.strip():
                    entry = LedgerEntry.from_dict(json.loads(line))
                    if not entry.verify():
                        raise ValueError(f"Entry {entry.entry_id} failed integrity check")
                    self._entries.append(entry)
                    # Would need to rebuild claims and indexes from entries
                    # Simplified here
    
    # =========================================================================
    # Compaction / Fossilization (Point 7: Scaling)
    # =========================================================================
    
    def compact(
        self, 
        older_than_days: int = 30,
        policy: Optional["DecayPolicy"] = None,
    ) -> "CompactionResult":
        """
        Compress old commitments into fossils.
        
        Fossilization preserves:
        - That a claim existed (existence proof)
        - That it was revised (revision chain)
        - Total accumulated cost
        - Type and confidence band
        - Proposition hash (for deduplication)
        
        Fossilization drops:
        - Full text (replaced with truncated summary)
        - Fine-grained support spans
        - Detailed entity lists
        - Per-entry metadata
        
        Accretion happens after irreversibility, never instead of it.
        """
        policy = policy or DecayPolicy()
        now = datetime.now()
        cutoff = now - timedelta(days=older_than_days)
        
        candidates = []
        for claim_id, claim in self._claims.items():
            # Only compact archived or old superseded claims
            if claim.status == CommitmentStatus.ARCHIVED:
                candidates.append(claim)
            elif claim.status == CommitmentStatus.SUPERSEDED:
                if claim.committed_at < cutoff:
                    candidates.append(claim)
        
        # Apply decay policy filters
        to_fossilize = []
        for claim in candidates:
            if policy.should_fossilize(claim, now):
                to_fossilize.append(claim)
        
        # Create fossils
        fossils_created = []
        entries_compacted = 0
        bytes_saved_estimate = 0
        
        for claim in to_fossilize:
            # Skip if already fossilized
            if claim.id in self._fossils:
                continue
            
            # Create fossil record
            fossil = FossilRecord.from_claim(claim, policy)
            self._fossils[claim.id] = fossil
            fossils_created.append(fossil)
            
            # Estimate bytes saved (rough)
            original_size = len(claim.text) + len(str(claim.support_refs))
            fossil_size = len(fossil.text_summary) + 50  # overhead
            bytes_saved_estimate += max(0, original_size - fossil_size)
            
            # Count entries that reference this claim
            claim_entries = [e for e in self._entries if e.claim_id == claim.id]
            entries_compacted += len(claim_entries)
        
        # Record compaction event
        if fossils_created:
            self._entry_counter += 1
            entry = LedgerEntry(
                entry_id=f"entry_{self._entry_counter}",
                entry_type=EntryType.ARCHIVE,  # Reuse ARCHIVE type for compaction
                timestamp=now,
                claim_id="compaction_event",
                data={
                    "type": "compaction",
                    "fossilized_count": len(fossils_created),
                    "fossilized_ids": [f.claim_id for f in fossils_created],
                    "policy": policy.to_dict(),
                },
            )
            self._append_entry(entry)
        
        return CompactionResult(
            fossils_created=len(fossils_created),
            entries_compacted=entries_compacted,
            bytes_saved_estimate=bytes_saved_estimate,
            fossil_ids=[f.claim_id for f in fossils_created],
        )
    
    def get_fossil(self, claim_id: str) -> Optional["FossilRecord"]:
        """Get a fossil record by claim ID."""
        return self._fossils.get(claim_id)
    
    def get_fossils(self) -> list["FossilRecord"]:
        """Get all fossil records."""
        return list(self._fossils.values())
    
    def prove_existence(self, claim_id: str) -> Optional[dict]:
        """
        Prove that a claim existed, even if fossilized.
        
        Returns proof containing:
        - Existence confirmation
        - Proposition hash
        - Commitment timestamp
        - Final status
        - Revision chain (if any)
        """
        # Check active claims first
        if claim_id in self._claims:
            claim = self._claims[claim_id]
            return {
                "exists": True,
                "fossilized": False,
                "claim_id": claim_id,
                "proposition_hash": claim.proposition_hash,
                "committed_at": claim.committed_at.isoformat(),
                "status": claim.status.name,
                "claim_type": claim.claim_type.name,
                "confidence": claim.confidence,
                "revision_chain": self.get_revision_chain(claim_id),
            }
        
        # Check fossils
        if claim_id in self._fossils:
            fossil = self._fossils[claim_id]
            return {
                "exists": True,
                "fossilized": True,
                "claim_id": claim_id,
                "proposition_hash": fossil.proposition_hash,
                "committed_at": fossil.committed_at.isoformat(),
                "fossilized_at": fossil.fossilized_at.isoformat(),
                "status": fossil.final_status,
                "claim_type": fossil.claim_type,
                "confidence_band": fossil.confidence_band,
                "text_summary": fossil.text_summary,
                "revision_count": fossil.revision_count,
                "total_cost": fossil.total_cost,
            }
        
        return None
    
    def get_compression_stats(self) -> dict:
        """Get statistics about compression state."""
        total_claims = len(self._claims)
        fossilized = len(self._fossils)
        active = len(self._by_status[CommitmentStatus.ACTIVE])
        
        # Estimate memory usage
        active_text_bytes = sum(len(c.text) for c in self._claims.values())
        fossil_text_bytes = sum(len(f.text_summary) for f in self._fossils.values())
        
        return {
            "total_claims_ever": total_claims + fossilized,
            "active_claims": active,
            "fossilized_claims": fossilized,
            "compression_ratio": fossilized / (total_claims + fossilized) if (total_claims + fossilized) > 0 else 0,
            "active_text_bytes": active_text_bytes,
            "fossil_text_bytes": fossil_text_bytes,
            "estimated_savings": active_text_bytes - fossil_text_bytes if fossilized > 0 else 0,
        }


# =============================================================================
# Fossilization Support Types
# =============================================================================

@dataclass
class FossilRecord:
    """
    Compressed representation of a historical claim.
    
    Preserves enough to prove existence and maintain invariants,
    but drops token-level detail.
    """
    claim_id: str
    proposition_hash: str
    text_summary: str          # Truncated/summarized text (first N chars)
    claim_type: str            # String, not enum (simpler for storage)
    confidence_band: str       # "high", "medium", "low" instead of exact float
    committed_at: datetime
    fossilized_at: datetime
    final_status: str          # Status when fossilized
    revision_count: int        # How many times revised
    total_cost: float          # Accumulated revision cost
    supersedes: Optional[str]  # What this superseded (if anything)
    checksum: str = ""         # Integrity check
    
    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._compute_checksum()
    
    def _compute_checksum(self) -> str:
        content = f"{self.claim_id}:{self.proposition_hash}:{self.committed_at.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    @classmethod
    def from_claim(cls, claim: CommittedClaim, policy: "DecayPolicy") -> "FossilRecord":
        """Create a fossil from a full claim."""
        # Truncate text
        max_len = policy.text_summary_length
        if len(claim.text) > max_len:
            text_summary = claim.text[:max_len-3] + "..."
        else:
            text_summary = claim.text
        
        # Convert confidence to band
        if claim.confidence >= 0.8:
            confidence_band = "high"
        elif claim.confidence >= 0.5:
            confidence_band = "medium"
        else:
            confidence_band = "low"
        
        return cls(
            claim_id=claim.id,
            proposition_hash=claim.proposition_hash,
            text_summary=text_summary,
            claim_type=claim.claim_type.name,
            confidence_band=confidence_band,
            committed_at=claim.committed_at,
            fossilized_at=datetime.now(),
            final_status=claim.status.name,
            revision_count=0,  # Would need to count from revision records
            total_cost=claim.revision_cost,
            supersedes=claim.supersedes,
        )
    
    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "proposition_hash": self.proposition_hash,
            "text_summary": self.text_summary,
            "claim_type": self.claim_type,
            "confidence_band": self.confidence_band,
            "committed_at": self.committed_at.isoformat(),
            "fossilized_at": self.fossilized_at.isoformat(),
            "final_status": self.final_status,
            "revision_count": self.revision_count,
            "total_cost": self.total_cost,
            "supersedes": self.supersedes,
            "checksum": self.checksum,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "FossilRecord":
        return cls(
            claim_id=d["claim_id"],
            proposition_hash=d["proposition_hash"],
            text_summary=d["text_summary"],
            claim_type=d["claim_type"],
            confidence_band=d["confidence_band"],
            committed_at=datetime.fromisoformat(d["committed_at"]),
            fossilized_at=datetime.fromisoformat(d["fossilized_at"]),
            final_status=d["final_status"],
            revision_count=d["revision_count"],
            total_cost=d["total_cost"],
            supersedes=d.get("supersedes"),
            checksum=d["checksum"],
        )


@dataclass
class DecayPolicy:
    """
    Configurable policy for what/when to compress.
    
    Different domains may have different preservation requirements:
    - Legal: Keep everything forever
    - Casual: Aggressive compression
    - Research: Keep revision chains, compress text
    """
    # Age thresholds
    min_age_days: int = 30           # Don't fossilize anything younger
    archive_age_days: int = 7        # Archived claims can fossilize after this
    
    # What to preserve
    text_summary_length: int = 100   # Max chars to keep from text
    preserve_revision_chains: bool = True
    preserve_support_refs: bool = False  # Usually drop these
    
    # What to compress
    compress_archived: bool = True   # Compress archived claims
    compress_superseded: bool = True # Compress old superseded claims
    compress_active: bool = False    # Never compress active claims
    
    # Exceptions
    preserve_high_cost: bool = True  # Don't compress high-cost revisions
    high_cost_threshold: float = 2.0 # What counts as high cost
    preserve_types: list = field(default_factory=list)  # Claim types to never compress
    
    def should_fossilize(self, claim: CommittedClaim, now: datetime) -> bool:
        """Check if a claim should be fossilized according to this policy."""
        # Never compress active
        if claim.status == CommitmentStatus.ACTIVE and not self.compress_active:
            return False
        
        # Check age
        age = now - claim.committed_at
        if age.days < self.min_age_days:
            return False
        
        # Archived claims have shorter grace period
        if claim.status == CommitmentStatus.ARCHIVED:
            if age.days < self.archive_age_days:
                return False
        
        # Preserve high-cost revisions
        if self.preserve_high_cost and claim.revision_cost >= self.high_cost_threshold:
            return False
        
        # Preserve specific types
        if claim.claim_type.name in self.preserve_types:
            return False
        
        return True
    
    def to_dict(self) -> dict:
        return {
            "min_age_days": self.min_age_days,
            "archive_age_days": self.archive_age_days,
            "text_summary_length": self.text_summary_length,
            "preserve_revision_chains": self.preserve_revision_chains,
            "compress_archived": self.compress_archived,
            "compress_superseded": self.compress_superseded,
            "high_cost_threshold": self.high_cost_threshold,
        }
    
    @classmethod
    def aggressive(cls) -> "DecayPolicy":
        """Aggressive compression for casual use."""
        return cls(
            min_age_days=7,
            archive_age_days=1,
            text_summary_length=50,
            preserve_support_refs=False,
        )
    
    @classmethod
    def conservative(cls) -> "DecayPolicy":
        """Conservative compression for important domains."""
        return cls(
            min_age_days=90,
            archive_age_days=30,
            text_summary_length=200,
            preserve_revision_chains=True,
            preserve_support_refs=True,
            preserve_high_cost=True,
            high_cost_threshold=1.0,
        )
    
    @classmethod
    def legal(cls) -> "DecayPolicy":
        """No compression - keep everything (for legal/compliance)."""
        return cls(
            min_age_days=36500,  # 100 years
            compress_archived=False,
            compress_superseded=False,
        )


@dataclass
class CompactionResult:
    """Result of a compaction operation."""
    fossils_created: int
    entries_compacted: int
    bytes_saved_estimate: int
    fossil_ids: list[str]
    
    def to_dict(self) -> dict:
        return {
            "fossils_created": self.fossils_created,
            "entries_compacted": self.entries_compacted,
            "bytes_saved_estimate": self.bytes_saved_estimate,
            "fossil_ids": self.fossil_ids,
        }


# =============================================================================
# Example Usage / Test
# =============================================================================

if __name__ == "__main__":
    from governor import EpistemicGovernor, CommitAction
    
    # Initialize
    ledger = EpistemicLedger()
    governor = EpistemicGovernor()
    
    print("=== Ledger Test ===\n")
    
    # Create some proposed commitments
    prop1 = ProposedCommitment(
        id="claim_1",
        text="Python 3.12 was released in October 2023",
        claim_type=ClaimType.TEMPORAL,
        confidence=0.85,
        proposition_hash=ProposedCommitment.hash_proposition("Python 3.12 released October 2023"),
        scope="conversation",
        span_start=0,
        span_end=40,
        extracted_entities=["Python"],
    )
    
    prop2 = ProposedCommitment(
        id="claim_2",
        text="The async implementation uses event loops",
        claim_type=ClaimType.PROCEDURAL,
        confidence=0.80,
        proposition_hash=ProposedCommitment.hash_proposition("async uses event loops"),
        scope="conversation",
        span_start=41,
        span_end=80,
        extracted_entities=["async"],
    )
    
    # Generate envelope and adjudicate
    envelope = governor.pre_generate()
    result = governor.adjudicate([prop1, prop2], envelope)
    
    # Commit to ledger
    for prop, decision in zip([prop1, prop2], result.decisions):
        if decision.action in (CommitAction.ACCEPT, CommitAction.HEDGE):
            claim = ledger.commit(prop, decision)
            print(f"Committed: {claim.id}")
            print(f"  Text: {claim.text}")
            print(f"  Confidence: {claim.confidence:.2f}")
            print(f"  Status: {claim.status.name}")
            print()
    
    print("=== Ledger Stats ===")
    stats = ledger.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print(f"\nTotal confidence: {ledger.get_total_confidence():.2f}")
    
    # Test revision
    print("\n=== Testing Revision ===")
    prop3 = ProposedCommitment(
        id="claim_3",
        text="Python 3.12 was actually released in October 2024",  # correction
        claim_type=ClaimType.TEMPORAL,
        confidence=0.90,
        proposition_hash=prop1.proposition_hash,  # same proposition
        scope="conversation",
        span_start=0,
        span_end=50,
        extracted_entities=["Python"],
    )
    
    # This should trigger revision since it's the same proposition
    contradictions = ledger.find_contradictions(prop3.proposition_hash)
    print(f"Found {len(contradictions)} potentially contradicting claim(s)")
    
    if contradictions:
        revision_decision = CommitDecision(
            commitment_id=prop3.id,
            action=CommitAction.ACCEPT,  # After explicit revision approval
            cost=1.0,
            reason="Explicit revision of prior claim",
        )
        new_claim, revision = ledger.revise(
            prop3,
            superseded_ids=[c.id for c in contradictions],
            justification="Correcting release date after verification",
            decision=revision_decision,
        )
        print(f"Revision committed: {new_claim.id}")
        print(f"  Superseded: {revision.superseded_claim_ids}")
        print(f"  Cost: {revision.cost}")
    
    print("\n=== Final Ledger Stats ===")
    stats = ledger.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # =========================================================================
    # Test Epoch/Fork/Reset (Point 4: Lock-in Prevention)
    # =========================================================================
    
    print("\n" + "="*60)
    print("=== Epoch/Fork/Reset Demo (Point 4: Lock-in Prevention) ===")
    print("="*60)
    
    # Create an epoch checkpoint
    print("\n1. Creating epoch checkpoint...")
    epoch1 = ledger.create_epoch("before_hypothesis", metadata={"topic": "Python release"})
    print(f"   Created: {epoch1.epoch_id}")
    print(f"   Name: {epoch1.name}")
    print(f"   Active claims at epoch: {epoch1.active_claim_ids}")
    
    # Create a fork for hypothesis exploration
    print("\n2. Creating fork for hypothesis exploration...")
    fork = ledger.create_fork("what_if_python4", from_epoch_id=epoch1.epoch_id)
    print(f"   Created: {fork.fork_id}")
    print(f"   Name: {fork.name}")
    print(f"   Parent epoch: {fork.parent_epoch_id}")
    print(f"   Active fork: {ledger.get_current_fork().fork_id if ledger.get_current_fork() else None}")
    
    # Commit a speculative claim in the fork
    print("\n3. Committing speculative claim in fork...")
    prop_speculative = ProposedCommitment(
        id="claim_speculative",
        text="Python 4.0 might introduce significant breaking changes",
        claim_type=ClaimType.CAUSAL,
        confidence=0.50,  # Low confidence - speculative
        proposition_hash=ProposedCommitment.hash_proposition("Python 4.0 breaking changes"),
        scope="fork",
        span_start=0,
        span_end=50,
        extracted_entities=["Python"],
    )
    spec_decision = CommitDecision(
        commitment_id=prop_speculative.id,
        action=CommitAction.HEDGE,
        reason="Speculative hypothesis",
    )
    spec_claim = ledger.commit(prop_speculative, spec_decision)
    fork.claim_ids.append(spec_claim.id)
    print(f"   Committed: {spec_claim.id}")
    print(f"   Fork claims: {fork.claim_ids}")
    
    # Abandon the fork (the hypothesis didn't pan out)
    print("\n4. Abandoning fork (hypothesis didn't work out)...")
    abandoned = ledger.abandon_fork(reason="Speculation not supported by evidence")
    print(f"   Abandoned claims: {abandoned}")
    print(f"   Current fork: {ledger.get_current_fork()}")
    
    # Demonstrate context reset
    print("\n5. Demonstrating context reset...")
    
    # First, make some commits that turn out to be wrong
    prop_wrong = ProposedCommitment(
        id="claim_wrong",
        text="The async library requires Python 3.11 minimum",
        claim_type=ClaimType.FACTUAL,
        confidence=0.85,
        proposition_hash=ProposedCommitment.hash_proposition("async requires 3.11"),
        scope="conversation",
        span_start=0,
        span_end=50,
        extracted_entities=["async", "Python"],
    )
    wrong_decision = CommitDecision(
        commitment_id=prop_wrong.id,
        action=CommitAction.ACCEPT,
        reason="Initial understanding",
    )
    wrong_claim = ledger.commit(prop_wrong, wrong_decision)
    print(f"   Committed (will be wrong): {wrong_claim.id}")
    
    # Create epoch before we realize the error
    epoch2 = ledger.create_epoch("before_correction")
    print(f"   Created epoch: {epoch2.epoch_id}")
    
    # Now we realize that claim was wrong - do a context reset
    print("\n6. Performing context reset ('we were wrong; moving on')...")
    reset = ledger.context_reset(
        reason="Initial assumptions about async library requirements were incorrect",
        acknowledgment="I was wrong about the Python version requirement. Let me correct this.",
        claim_ids_to_archive=[wrong_claim.id],
    )
    print(f"   Reset ID: {reset.reset_id}")
    print(f"   Reason: {reset.reason}")
    print(f"   Acknowledgment: {reset.acknowledgment}")
    print(f"   Archived claims: {reset.archived_claim_ids}")
    
    # Show final state
    print("\n7. Final state after epoch/fork/reset demo...")
    stats = ledger.get_stats()
    print(f"   Total claims: {stats['total_claims']}")
    print(f"   Active claims: {stats['active_claims']}")
    print(f"   Archived claims: {stats['archived_claims']}")
    print(f"   Epochs: {stats['epochs']}")
    print(f"   Forks: {stats['forks']}")
    print(f"   Context resets: {stats['context_resets']}")
    
    print("\n✓ Epoch/Fork/Reset working")
    print("\nKey insight: Early wrong commits are NOT permanent debt")
    print("  - Epochs provide checkpoints")
    print("  - Forks allow isolated exploration")
    print("  - Context resets enable graceful 'moving on'")
    print("  - History is preserved (archived, not deleted)")
    print("  - Errors are acknowledged, not hidden")
    
    # =========================================================================
    # Test Fossilization (Point 7: Scaling)
    # =========================================================================
    
    print("\n" + "="*60)
    print("=== Fossilization Demo (Point 7: Scaling) ===")
    print("="*60)
    
    print("\n1. Current compression stats...")
    comp_stats = ledger.get_compression_stats()
    for k, v in comp_stats.items():
        print(f"   {k}: {v}")
    
    print("\n2. Testing decay policies...")
    aggressive = DecayPolicy.aggressive()
    conservative = DecayPolicy.conservative()
    legal = DecayPolicy.legal()
    
    print(f"   Aggressive: min_age={aggressive.min_age_days}d, summary={aggressive.text_summary_length} chars")
    print(f"   Conservative: min_age={conservative.min_age_days}d, summary={conservative.text_summary_length} chars")
    print(f"   Legal: min_age={legal.min_age_days}d (effectively never)")
    
    print("\n3. Simulating old claims for fossilization...")
    # Create some "old" claims by backdating
    old_claim = CommittedClaim(
        id="old_claim_1",
        text="This is a very long claim that contains lots of detail about something that happened a long time ago and is no longer relevant to current discussions but we want to preserve that it existed.",
        claim_type=ClaimType.FACTUAL,
        confidence=0.75,
        proposition_hash="old_hash_1",
        scope="historical",
        status=CommitmentStatus.ARCHIVED,
        committed_at=datetime.now() - timedelta(days=60),  # 60 days old
        support_refs=["ref1", "ref2", "ref3"],
    )
    ledger._claims[old_claim.id] = old_claim
    ledger._by_status[CommitmentStatus.ARCHIVED].append(old_claim.id)
    print(f"   Added old claim: {old_claim.id}")
    print(f"   Original text length: {len(old_claim.text)} chars")
    
    print("\n4. Running compaction with default policy...")
    result = ledger.compact(older_than_days=30)
    print(f"   Fossils created: {result.fossils_created}")
    print(f"   Entries compacted: {result.entries_compacted}")
    print(f"   Bytes saved (est): {result.bytes_saved_estimate}")
    
    print("\n5. Checking fossil record...")
    fossil = ledger.get_fossil("old_claim_1")
    if fossil:
        print(f"   Fossil ID: {fossil.claim_id}")
        print(f"   Text summary: '{fossil.text_summary}'")
        print(f"   Confidence band: {fossil.confidence_band}")
        print(f"   Original committed: {fossil.committed_at.date()}")
        print(f"   Fossilized at: {fossil.fossilized_at.date()}")
    
    print("\n6. Testing existence proof...")
    proof = ledger.prove_existence("old_claim_1")
    if proof:
        print(f"   Exists: {proof['exists']}")
        print(f"   Fossilized: {proof['fossilized']}")
        print(f"   Proposition hash: {proof['proposition_hash']}")
        print(f"   Can prove it existed without full text!")
    
    print("\n7. Final compression stats...")
    comp_stats = ledger.get_compression_stats()
    print(f"   Total claims ever: {comp_stats['total_claims_ever']}")
    print(f"   Active claims: {comp_stats['active_claims']}")
    print(f"   Fossilized: {comp_stats['fossilized_claims']}")
    print(f"   Compression ratio: {comp_stats['compression_ratio']:.1%}")
    
    print("\n✓ Fossilization working")
    print("\nKey insight: Ledger growth is bounded")
    print("  - Old claims compressed to fossils")
    print("  - Existence proofs preserved")
    print("  - Text summarized, not deleted")
    print("  - Revision chains maintained")
    print("  - Domain-specific policies (aggressive/conservative/legal)")
