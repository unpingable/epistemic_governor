"""
State Integrity Sealing

Hash chain + deterministic replay for sovereignty hardening.

Every state mutation is recorded as an event with:
- event_seq: monotonic counter
- prev_hash: hash of previous event
- payload_hash: hash of event payload
- event_hash: H(event_seq || prev_hash || payload_hash)

This creates an append-only, tamper-evident log that can:
1. Detect unauthorized state mutations
2. Replay events to reconstruct state deterministically
3. Prove the sequence of operations that led to current state

Integrity invariant:
    State S_t is valid iff replay(events[0:t]) == S_t
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime, timezone
from enum import Enum, auto
import hashlib
import json
import uuid


# =============================================================================
# Event Types
# =============================================================================

class IntegrityEventType(Enum):
    """Types of events in the integrity log."""
    # State mutations
    COMMITMENT_ADD = auto()
    COMMITMENT_WITHDRAW = auto()
    COMMITMENT_SUPERSEDE = auto()
    COMMITMENT_MODIFY = auto()
    
    # FSM transitions
    FSM_TRANSITION = auto()
    
    # Quarantine operations
    QUARANTINE_ADD = auto()
    QUARANTINE_PROMOTE = auto()
    
    # Resolution events
    RESOLUTION_NARROW = auto()
    RESOLUTION_LOWER_SIGMA = auto()
    
    # Evidence events
    EVIDENCE_SUBMIT = auto()
    EVIDENCE_REJECT = auto()
    
    # Forbidden attempts (logged but don't mutate)
    FORBIDDEN_ATTEMPT = auto()
    
    # System events
    STATE_INIT = auto()
    CHECKPOINT = auto()


# =============================================================================
# Integrity Event
# =============================================================================

@dataclass(frozen=True)
class IntegrityEvent:
    """
    An immutable event in the integrity chain.
    
    Once created, an event cannot be modified.
    The hash seals its contents.
    """
    event_id: str
    event_seq: int          # Monotonic sequence number
    event_type: IntegrityEventType
    timestamp: str          # ISO format, UTC
    payload: Dict[str, Any] # Event-specific data
    prev_hash: str          # Hash of previous event (empty for first)
    payload_hash: str       # H(payload)
    event_hash: str         # H(event_seq || prev_hash || payload_hash)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "event_id": self.event_id,
            "event_seq": self.event_seq,
            "event_type": self.event_type.name,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "payload_hash": self.payload_hash,
            "event_hash": self.event_hash,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IntegrityEvent":
        """Deserialize from dictionary."""
        return cls(
            event_id=d["event_id"],
            event_seq=d["event_seq"],
            event_type=IntegrityEventType[d["event_type"]],
            timestamp=d["timestamp"],
            payload=d["payload"],
            prev_hash=d["prev_hash"],
            payload_hash=d["payload_hash"],
            event_hash=d["event_hash"],
        )


# =============================================================================
# Integrity Errors
# =============================================================================

class IntegrityError(Exception):
    """Base class for integrity violations."""
    pass


class HashChainError(IntegrityError):
    """Hash chain is broken."""
    def __init__(self, event_seq: int, expected: str, actual: str):
        self.event_seq = event_seq
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Hash chain broken at seq {event_seq}: "
            f"expected prev_hash={expected[:16]}..., got {actual[:16]}..."
        )


class SequenceError(IntegrityError):
    """Event sequence is not monotonic."""
    def __init__(self, expected: int, actual: int):
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Sequence not monotonic: expected {expected}, got {actual}"
        )


class PayloadHashError(IntegrityError):
    """Payload hash doesn't match payload."""
    def __init__(self, event_seq: int):
        self.event_seq = event_seq
        super().__init__(f"Payload hash mismatch at seq {event_seq}")


class EventHashError(IntegrityError):
    """Event hash doesn't match components."""
    def __init__(self, event_seq: int):
        self.event_seq = event_seq
        super().__init__(f"Event hash mismatch at seq {event_seq}")


class StateIntegrityError(IntegrityError):
    """State doesn't match replayed events."""
    def __init__(self, message: str):
        super().__init__(message)


# =============================================================================
# Hash Functions
# =============================================================================

def hash_payload(payload: Dict[str, Any]) -> str:
    """
    Hash a payload deterministically.
    
    Uses canonical JSON serialization for consistency.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def hash_event(event_seq: int, prev_hash: str, payload_hash: str) -> str:
    """
    Compute event hash from components.
    
    H(event_seq || prev_hash || payload_hash)
    """
    data = f"{event_seq}||{prev_hash}||{payload_hash}"
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


# =============================================================================
# Event Chain
# =============================================================================

class EventChain:
    """
    Append-only chain of integrity events.
    
    Maintains:
    - Monotonic sequence counter
    - Hash chain linking events
    - Head hash for quick integrity check
    
    Events can only be appended, never modified or removed.
    """
    
    GENESIS_HASH = "0" * 64  # Genesis block has zero prev_hash
    
    def __init__(self):
        self._events: List[IntegrityEvent] = []
        self._seq_counter: int = 0
        self._head_hash: str = self.GENESIS_HASH
        
        # Index for fast lookup
        self._by_id: Dict[str, IntegrityEvent] = {}
        self._by_type: Dict[IntegrityEventType, List[IntegrityEvent]] = {}
    
    @property
    def head_hash(self) -> str:
        """Current head of the chain."""
        return self._head_hash
    
    @property
    def length(self) -> int:
        """Number of events in chain."""
        return len(self._events)
    
    @property
    def seq(self) -> int:
        """Current sequence number (next event will have this seq)."""
        return self._seq_counter
    
    def append(
        self,
        event_type: IntegrityEventType,
        payload: Dict[str, Any],
    ) -> IntegrityEvent:
        """
        Append a new event to the chain.
        
        Returns the created event with computed hashes.
        """
        # Compute hashes
        payload_h = hash_payload(payload)
        event_h = hash_event(self._seq_counter, self._head_hash, payload_h)
        
        # Create event
        event = IntegrityEvent(
            event_id=f"E_{uuid.uuid4().hex[:12]}",
            event_seq=self._seq_counter,
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload=payload,
            prev_hash=self._head_hash,
            payload_hash=payload_h,
            event_hash=event_h,
        )
        
        # Append to chain
        self._events.append(event)
        self._by_id[event.event_id] = event
        
        if event_type not in self._by_type:
            self._by_type[event_type] = []
        self._by_type[event_type].append(event)
        
        # Update chain state
        self._seq_counter += 1
        self._head_hash = event_h
        
        return event
    
    def get(self, event_id: str) -> Optional[IntegrityEvent]:
        """Get event by ID."""
        return self._by_id.get(event_id)
    
    def get_by_seq(self, seq: int) -> Optional[IntegrityEvent]:
        """Get event by sequence number."""
        if 0 <= seq < len(self._events):
            return self._events[seq]
        return None
    
    def get_by_type(self, event_type: IntegrityEventType) -> List[IntegrityEvent]:
        """Get all events of a type."""
        return self._by_type.get(event_type, [])
    
    def slice(self, start: int = 0, end: Optional[int] = None) -> List[IntegrityEvent]:
        """Get a slice of events."""
        return self._events[start:end]
    
    def verify(self) -> bool:
        """
        Verify the integrity of the entire chain.
        
        Checks:
        1. Sequence is monotonic starting from 0
        2. Each prev_hash matches previous event_hash
        3. Each payload_hash matches payload
        4. Each event_hash matches components
        
        Raises IntegrityError on failure.
        Returns True if valid.
        """
        expected_prev = self.GENESIS_HASH
        
        for i, event in enumerate(self._events):
            # Check sequence
            if event.event_seq != i:
                raise SequenceError(i, event.event_seq)
            
            # Check prev_hash chain
            if event.prev_hash != expected_prev:
                raise HashChainError(i, expected_prev, event.prev_hash)
            
            # Check payload hash
            computed_payload_hash = hash_payload(event.payload)
            if event.payload_hash != computed_payload_hash:
                raise PayloadHashError(i)
            
            # Check event hash
            computed_event_hash = hash_event(
                event.event_seq, event.prev_hash, event.payload_hash
            )
            if event.event_hash != computed_event_hash:
                raise EventHashError(i)
            
            # Update expected prev for next iteration
            expected_prev = event.event_hash
        
        # Verify head hash
        if self._events:
            if self._head_hash != self._events[-1].event_hash:
                raise HashChainError(
                    len(self._events) - 1,
                    self._events[-1].event_hash,
                    self._head_hash,
                )
        
        return True
    
    def export(self) -> List[Dict[str, Any]]:
        """Export chain as list of dicts."""
        return [e.to_dict() for e in self._events]
    
    @classmethod
    def import_chain(cls, data: List[Dict[str, Any]]) -> "EventChain":
        """Import chain from list of dicts and verify."""
        chain = cls()
        
        for d in data:
            event = IntegrityEvent.from_dict(d)
            
            # Verify this event links correctly
            if event.event_seq != chain._seq_counter:
                raise SequenceError(chain._seq_counter, event.event_seq)
            
            if event.prev_hash != chain._head_hash:
                raise HashChainError(
                    event.event_seq, chain._head_hash, event.prev_hash
                )
            
            # Verify hashes
            if event.payload_hash != hash_payload(event.payload):
                raise PayloadHashError(event.event_seq)
            
            computed_hash = hash_event(
                event.event_seq, event.prev_hash, event.payload_hash
            )
            if event.event_hash != computed_hash:
                raise EventHashError(event.event_seq)
            
            # Add to chain
            chain._events.append(event)
            chain._by_id[event.event_id] = event
            
            if event.event_type not in chain._by_type:
                chain._by_type[event.event_type] = []
            chain._by_type[event.event_type].append(event)
            
            chain._seq_counter += 1
            chain._head_hash = event.event_hash
        
        return chain


# =============================================================================
# Sealed State
# =============================================================================

@dataclass
class SealedCommitment:
    """A commitment with integrity metadata."""
    commitment_id: str
    predicate_type: str
    predicate_args: Tuple[str, ...]
    sigma: float
    status: str
    created_event_seq: int
    modified_event_seq: Optional[int] = None


class SealedState:
    """
    State that can only be modified through the event chain.
    
    All mutations go through emit_* methods which:
    1. Record the event
    2. Apply the change
    3. Update the state hash
    
    The state can be reconstructed by replaying events.
    """
    
    def __init__(self):
        self.chain = EventChain()
        self.commitments: Dict[str, SealedCommitment] = {}
        self.total_sigma: float = 0.0
        self.quarantined: Dict[str, Dict[str, Any]] = {}
        self._state_hash: str = ""
        
        # Emit genesis event
        self._emit_init()
    
    def _emit_init(self):
        """Emit state initialization event."""
        self.chain.append(
            IntegrityEventType.STATE_INIT,
            {"version": "1.0", "initialized": True},
        )
        self._update_state_hash()
    
    def _update_state_hash(self):
        """Update the state hash from current state."""
        state_data = {
            "commitments": {
                k: {
                    "sigma": v.sigma,
                    "status": v.status,
                }
                for k, v in sorted(self.commitments.items())
            },
            "total_sigma": self.total_sigma,
            "quarantine_count": len(self.quarantined),
            "chain_head": self.chain.head_hash,
        }
        self._state_hash = hash_payload(state_data)
    
    @property
    def state_hash(self) -> str:
        """Current state hash."""
        return self._state_hash
    
    def emit_commitment_add(
        self,
        commitment_id: str,
        predicate_type: str,
        predicate_args: Tuple[str, ...],
        sigma: float,
    ) -> IntegrityEvent:
        """Add a commitment through the event chain."""
        event = self.chain.append(
            IntegrityEventType.COMMITMENT_ADD,
            {
                "commitment_id": commitment_id,
                "predicate_type": predicate_type,
                "predicate_args": list(predicate_args),
                "sigma": sigma,
            },
        )
        
        # Apply change
        self.commitments[commitment_id] = SealedCommitment(
            commitment_id=commitment_id,
            predicate_type=predicate_type,
            predicate_args=predicate_args,
            sigma=sigma,
            status="active",
            created_event_seq=event.event_seq,
        )
        self.total_sigma += sigma
        
        self._update_state_hash()
        return event
    
    def emit_commitment_withdraw(
        self,
        commitment_id: str,
        reason: str,
    ) -> IntegrityEvent:
        """Withdraw a commitment through the event chain."""
        if commitment_id not in self.commitments:
            raise ValueError(f"Unknown commitment: {commitment_id}")
        
        commitment = self.commitments[commitment_id]
        
        event = self.chain.append(
            IntegrityEventType.COMMITMENT_WITHDRAW,
            {
                "commitment_id": commitment_id,
                "reason": reason,
                "previous_sigma": commitment.sigma,
            },
        )
        
        # Apply change
        self.total_sigma -= commitment.sigma
        commitment.status = "withdrawn"
        commitment.modified_event_seq = event.event_seq
        
        self._update_state_hash()
        return event
    
    def emit_commitment_modify(
        self,
        commitment_id: str,
        new_sigma: float,
        reason: str,
    ) -> IntegrityEvent:
        """Modify a commitment through the event chain."""
        if commitment_id not in self.commitments:
            raise ValueError(f"Unknown commitment: {commitment_id}")
        
        commitment = self.commitments[commitment_id]
        old_sigma = commitment.sigma
        
        event = self.chain.append(
            IntegrityEventType.COMMITMENT_MODIFY,
            {
                "commitment_id": commitment_id,
                "old_sigma": old_sigma,
                "new_sigma": new_sigma,
                "reason": reason,
            },
        )
        
        # Apply change
        self.total_sigma -= old_sigma
        self.total_sigma += new_sigma
        commitment.sigma = new_sigma
        commitment.modified_event_seq = event.event_seq
        
        self._update_state_hash()
        return event
    
    def emit_quarantine_add(
        self,
        quarantine_id: str,
        reason: str,
        content_hash: str,
    ) -> IntegrityEvent:
        """Add to quarantine through the event chain."""
        event = self.chain.append(
            IntegrityEventType.QUARANTINE_ADD,
            {
                "quarantine_id": quarantine_id,
                "reason": reason,
                "content_hash": content_hash,
            },
        )
        
        # Apply change
        self.quarantined[quarantine_id] = {
            "reason": reason,
            "content_hash": content_hash,
            "created_event_seq": event.event_seq,
        }
        
        self._update_state_hash()
        return event
    
    def emit_quarantine_promote(
        self,
        quarantine_id: str,
        commitment_id: str,
    ) -> IntegrityEvent:
        """Promote from quarantine through the event chain."""
        if quarantine_id not in self.quarantined:
            raise ValueError(f"Unknown quarantine: {quarantine_id}")
        
        event = self.chain.append(
            IntegrityEventType.QUARANTINE_PROMOTE,
            {
                "quarantine_id": quarantine_id,
                "commitment_id": commitment_id,
            },
        )
        
        # Apply change
        del self.quarantined[quarantine_id]
        
        self._update_state_hash()
        return event
    
    def emit_forbidden(self, code: str, description: str) -> IntegrityEvent:
        """Log a forbidden attempt (doesn't mutate state)."""
        event = self.chain.append(
            IntegrityEventType.FORBIDDEN_ATTEMPT,
            {
                "code": code,
                "description": description,
            },
        )
        # No state change - just logged
        return event
    
    def checkpoint(self) -> IntegrityEvent:
        """Create a checkpoint event with current state hash."""
        event = self.chain.append(
            IntegrityEventType.CHECKPOINT,
            {
                "state_hash": self._state_hash,
                "commitment_count": len(self.commitments),
                "total_sigma": self.total_sigma,
                "quarantine_count": len(self.quarantined),
            },
        )
        return event
    
    def verify_integrity(self) -> bool:
        """
        Verify state integrity.
        
        1. Verify event chain
        2. Replay events and compare state
        """
        # Verify chain
        self.chain.verify()
        
        # Replay and compare
        replayed = replay_events(self.chain.slice())
        
        if replayed.state_hash != self._state_hash:
            raise StateIntegrityError(
                f"State hash mismatch: current={self._state_hash[:16]}..., "
                f"replayed={replayed.state_hash[:16]}..."
            )
        
        return True


# =============================================================================
# Replay Function
# =============================================================================

def replay_events(events: List[IntegrityEvent]) -> SealedState:
    """
    Deterministically replay events to reconstruct state.
    
    This is the core integrity guarantee:
    replay(state.chain.slice()) == state
    """
    # Create fresh state (will emit its own INIT event)
    state = SealedState.__new__(SealedState)
    state.chain = EventChain()
    state.commitments = {}
    state.total_sigma = 0.0
    state.quarantined = {}
    state._state_hash = ""
    
    for event in events:
        # Re-add event to chain (for hash verification)
        # We trust the events are already verified
        state.chain._events.append(event)
        state.chain._by_id[event.event_id] = event
        
        if event.event_type not in state.chain._by_type:
            state.chain._by_type[event.event_type] = []
        state.chain._by_type[event.event_type].append(event)
        
        state.chain._seq_counter = event.event_seq + 1
        state.chain._head_hash = event.event_hash
        
        # Apply event to state
        _apply_event(state, event)
    
    state._update_state_hash()
    return state


def _apply_event(state: SealedState, event: IntegrityEvent):
    """Apply a single event to state (during replay)."""
    payload = event.payload
    
    if event.event_type == IntegrityEventType.STATE_INIT:
        # Nothing to do - state already initialized
        pass
    
    elif event.event_type == IntegrityEventType.COMMITMENT_ADD:
        state.commitments[payload["commitment_id"]] = SealedCommitment(
            commitment_id=payload["commitment_id"],
            predicate_type=payload["predicate_type"],
            predicate_args=tuple(payload["predicate_args"]),
            sigma=payload["sigma"],
            status="active",
            created_event_seq=event.event_seq,
        )
        state.total_sigma += payload["sigma"]
    
    elif event.event_type == IntegrityEventType.COMMITMENT_WITHDRAW:
        commitment = state.commitments[payload["commitment_id"]]
        state.total_sigma -= commitment.sigma
        commitment.status = "withdrawn"
        commitment.modified_event_seq = event.event_seq
    
    elif event.event_type == IntegrityEventType.COMMITMENT_MODIFY:
        commitment = state.commitments[payload["commitment_id"]]
        state.total_sigma -= commitment.sigma
        state.total_sigma += payload["new_sigma"]
        commitment.sigma = payload["new_sigma"]
        commitment.modified_event_seq = event.event_seq
    
    elif event.event_type == IntegrityEventType.QUARANTINE_ADD:
        state.quarantined[payload["quarantine_id"]] = {
            "reason": payload["reason"],
            "content_hash": payload["content_hash"],
            "created_event_seq": event.event_seq,
        }
    
    elif event.event_type == IntegrityEventType.QUARANTINE_PROMOTE:
        del state.quarantined[payload["quarantine_id"]]
    
    elif event.event_type == IntegrityEventType.FORBIDDEN_ATTEMPT:
        # No state change
        pass
    
    elif event.event_type == IntegrityEventType.CHECKPOINT:
        # No state change - just a marker
        pass
    
    # FSM and other events can be added as needed


# =============================================================================
# Integrity Monitor
# =============================================================================

class IntegrityMonitor:
    """
    Monitors state for unauthorized mutations.
    
    Wraps a SealedState and periodically verifies integrity.
    """
    
    def __init__(self, state: SealedState, check_interval: int = 10):
        self.state = state
        self.check_interval = check_interval
        self._operations_since_check = 0
        self._last_verified_hash = state.state_hash
        self._violations: List[Dict[str, Any]] = []
    
    def record_operation(self):
        """Record an operation and maybe verify."""
        self._operations_since_check += 1
        
        if self._operations_since_check >= self.check_interval:
            self.verify()
            self._operations_since_check = 0
    
    def verify(self) -> bool:
        """Verify state integrity."""
        try:
            self.state.verify_integrity()
            self._last_verified_hash = self.state.state_hash
            return True
        except IntegrityError as e:
            self._violations.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "type": type(e).__name__,
            })
            raise
    
    @property
    def violations(self) -> List[Dict[str, Any]]:
        """Get list of integrity violations."""
        return self._violations


# =============================================================================
# Tests (inline for overnight run)
# =============================================================================

def test_event_chain_basic():
    """Test basic event chain operations."""
    print("=== Test: Event Chain Basic ===\n")
    
    chain = EventChain()
    
    # Append events
    e1 = chain.append(IntegrityEventType.COMMITMENT_ADD, {"id": "c1", "sigma": 0.5})
    e2 = chain.append(IntegrityEventType.COMMITMENT_ADD, {"id": "c2", "sigma": 0.3})
    e3 = chain.append(IntegrityEventType.COMMITMENT_WITHDRAW, {"id": "c1"})
    
    print(f"Chain length: {chain.length}")
    print(f"Head hash: {chain.head_hash[:16]}...")
    
    # Verify sequence
    assert e1.event_seq == 0
    assert e2.event_seq == 1
    assert e3.event_seq == 2
    
    # Verify chain links
    assert e1.prev_hash == EventChain.GENESIS_HASH
    assert e2.prev_hash == e1.event_hash
    assert e3.prev_hash == e2.event_hash
    
    # Verify integrity
    assert chain.verify()
    
    print("✓ Event chain basic operations working\n")
    return True


def test_hash_chain_continuity():
    """Test that hash chain is continuous and tamper-evident."""
    print("=== Test: Hash Chain Continuity ===\n")
    
    chain = EventChain()
    
    # Build chain
    for i in range(10):
        chain.append(IntegrityEventType.COMMITMENT_ADD, {"index": i})
    
    # Verify
    assert chain.verify()
    
    # Try to tamper with an event (this requires breaking immutability)
    # We'll simulate by creating a corrupted export
    exported = chain.export()
    exported[5]["payload"]["index"] = 999  # Tamper!
    
    # Import should fail
    try:
        EventChain.import_chain(exported)
        assert False, "Should have detected tampering"
    except PayloadHashError as e:
        print(f"✓ Tampering detected at seq {e.event_seq}")
    
    print("✓ Hash chain continuity verified\n")
    return True


def test_replay_determinism():
    """Test that replay produces identical state."""
    print("=== Test: Replay Determinism ===\n")
    
    state = SealedState()
    
    # Make some changes
    state.emit_commitment_add("c1", "HAS", ("E", "p", "v"), 0.5)
    state.emit_commitment_add("c2", "IS_A", ("E", "T"), 0.3)
    state.emit_commitment_modify("c1", 0.4, "lowered")
    state.emit_quarantine_add("q1", "support_deficit", "hash123")
    state.emit_commitment_withdraw("c2", "superseded")
    
    original_hash = state.state_hash
    original_sigma = state.total_sigma
    original_count = len(state.commitments)
    
    print(f"Original state hash: {original_hash[:16]}...")
    print(f"Original sigma: {original_sigma}")
    print(f"Original commitment count: {original_count}")
    
    # Replay events
    replayed = replay_events(state.chain.slice())
    
    print(f"\nReplayed state hash: {replayed.state_hash[:16]}...")
    print(f"Replayed sigma: {replayed.total_sigma}")
    print(f"Replayed commitment count: {len(replayed.commitments)}")
    
    # Verify identical
    assert replayed.state_hash == original_hash, "State hash mismatch"
    assert replayed.total_sigma == original_sigma, "Sigma mismatch"
    assert len(replayed.commitments) == original_count, "Commitment count mismatch"
    
    # Verify commitment details
    for cid, commitment in state.commitments.items():
        replayed_commitment = replayed.commitments[cid]
        assert commitment.sigma == replayed_commitment.sigma
        assert commitment.status == replayed_commitment.status
    
    print("\n✓ Replay produces identical state\n")
    return True


def test_mutation_without_event_fails():
    """Test that direct mutation is detectable."""
    print("=== Test: Mutation Without Event Fails ===\n")
    
    state = SealedState()
    
    # Add commitment properly
    state.emit_commitment_add("c1", "HAS", ("E", "p", "v"), 0.5)
    
    # Checkpoint
    state.checkpoint()
    
    # Get current hash
    valid_hash = state.state_hash
    
    # Now tamper directly (bypassing events)
    state.commitments["c1"].sigma = 0.9  # ILLEGAL!
    state.total_sigma = 0.9  # ILLEGAL!
    state._update_state_hash()  # Update hash to reflect tampered state
    
    # Replay should produce different state
    replayed = replay_events(state.chain.slice())
    
    print(f"Tampered hash: {state.state_hash[:16]}...")
    print(f"Replayed hash: {replayed.state_hash[:16]}...")
    
    # Hashes should NOT match
    assert state.state_hash != replayed.state_hash, "Tampering should be detectable"
    
    # Verify_integrity should fail
    try:
        state.verify_integrity()
        assert False, "Should have detected tampering"
    except StateIntegrityError as e:
        print(f"✓ Tampering detected: {e}")
    
    print("\n✓ Mutation without event properly detected\n")
    return True


def test_truncated_chain_detected():
    """Test that truncated event stream is detected."""
    print("=== Test: Truncated Chain Detected ===\n")
    
    chain = EventChain()
    
    # Build chain
    for i in range(10):
        chain.append(IntegrityEventType.COMMITMENT_ADD, {"index": i})
    
    full_head = chain.head_hash
    
    # Export and truncate
    exported = chain.export()
    truncated = exported[:5]  # Drop last 5 events
    
    # Import truncated chain
    truncated_chain = EventChain.import_chain(truncated)
    
    # Truncated chain is internally valid but has different head
    assert truncated_chain.verify()
    assert truncated_chain.head_hash != full_head
    assert truncated_chain.length == 5
    
    print(f"Full chain head: {full_head[:16]}...")
    print(f"Truncated chain head: {truncated_chain.head_hash[:16]}...")
    print(f"Truncated length: {truncated_chain.length}")
    
    print("\n✓ Truncation detectable via head hash comparison\n")
    return True


def test_modified_event_detected():
    """Test that modified events break the chain."""
    print("=== Test: Modified Event Detected ===\n")
    
    chain = EventChain()
    
    # Build chain
    for i in range(5):
        chain.append(IntegrityEventType.COMMITMENT_ADD, {"index": i})
    
    # Export
    exported = chain.export()
    
    # Modify an event in the middle (try to change payload)
    exported[2]["payload"]["index"] = 999
    
    # Try to import - should fail at payload hash check
    try:
        EventChain.import_chain(exported)
        assert False, "Should have detected modification"
    except PayloadHashError as e:
        print(f"✓ Modification detected: payload hash mismatch at seq {e.event_seq}")
    
    # Try modifying the hash too (to match new payload)
    exported[2]["payload_hash"] = hash_payload(exported[2]["payload"])
    
    # Should now fail at event hash check
    try:
        EventChain.import_chain(exported)
        assert False, "Should have detected modification"
    except EventHashError as e:
        print(f"✓ Modification detected: event hash mismatch at seq {e.event_seq}")
    
    print("\n✓ Event modification properly detected\n")
    return True


def test_sequence_monotonicity():
    """Test that sequence must be monotonic."""
    print("=== Test: Sequence Monotonicity ===\n")
    
    chain = EventChain()
    
    # Build chain
    for i in range(5):
        chain.append(IntegrityEventType.COMMITMENT_ADD, {"index": i})
    
    # Export and mess with sequence
    exported = chain.export()
    exported[3]["event_seq"] = 10  # Wrong sequence!
    
    # Try to import - should fail
    try:
        EventChain.import_chain(exported)
        assert False, "Should have detected sequence error"
    except SequenceError as e:
        print(f"✓ Sequence error detected: expected {e.expected}, got {e.actual}")
    
    print("\n✓ Sequence monotonicity enforced\n")
    return True


def test_integrity_monitor():
    """Test integrity monitor catches violations."""
    print("=== Test: Integrity Monitor ===\n")
    
    state = SealedState()
    monitor = IntegrityMonitor(state, check_interval=3)
    
    # Normal operations
    state.emit_commitment_add("c1", "HAS", ("E", "p", "v"), 0.5)
    monitor.record_operation()
    
    state.emit_commitment_add("c2", "IS_A", ("E", "T"), 0.3)
    monitor.record_operation()
    
    # This should trigger a verify (3rd operation)
    state.emit_commitment_add("c3", "AT_TIME", ("E", "t1", "t2"), 0.2)
    monitor.record_operation()  # Triggers verify
    
    print(f"Operations since check: {monitor._operations_since_check}")
    assert monitor._operations_since_check == 0  # Reset after verify
    
    # Tamper and verify
    state.commitments["c1"].sigma = 0.9  # ILLEGAL
    state._update_state_hash()
    
    try:
        monitor.verify()
        assert False, "Should have caught tampering"
    except StateIntegrityError:
        print("✓ Monitor caught tampering")
    
    assert len(monitor.violations) == 1
    print(f"Violations logged: {len(monitor.violations)}")
    
    print("\n✓ Integrity monitor working\n")
    return True


def test_forbidden_logging():
    """Test that forbidden attempts are logged without mutating state."""
    print("=== Test: Forbidden Logging ===\n")
    
    state = SealedState()
    
    # Add a commitment
    state.emit_commitment_add("c1", "HAS", ("E", "p", "v"), 0.5)
    
    hash_before = state.state_hash
    sigma_before = state.total_sigma
    
    # Log forbidden attempt
    event = state.emit_forbidden("F-02", "MODEL_TEXT evidence rejected")
    
    # State should NOT change
    assert state.state_hash == hash_before, "Forbidden should not change state hash"
    assert state.total_sigma == sigma_before, "Forbidden should not change sigma"
    
    # But event should be in chain
    forbidden_events = state.chain.get_by_type(IntegrityEventType.FORBIDDEN_ATTEMPT)
    assert len(forbidden_events) == 1
    assert forbidden_events[0].payload["code"] == "F-02"
    
    print(f"Forbidden event logged: {event.event_id}")
    print(f"State hash unchanged: {state.state_hash[:16]}...")
    
    print("\n✓ Forbidden attempts logged without state mutation\n")
    return True


def test_export_import_roundtrip():
    """Test full export/import roundtrip."""
    print("=== Test: Export/Import Roundtrip ===\n")
    
    state = SealedState()
    
    # Build state
    state.emit_commitment_add("c1", "HAS", ("E", "p", "v"), 0.5)
    state.emit_commitment_add("c2", "IS_A", ("E", "T"), 0.3)
    state.emit_quarantine_add("q1", "support_deficit", "hash123")
    state.emit_forbidden("F-01", "text-only commit blocked")
    state.emit_commitment_withdraw("c2", "reason")
    state.checkpoint()
    
    original_head = state.chain.head_hash
    original_length = state.chain.length
    
    # Export
    exported = state.chain.export()
    
    # Import into fresh chain
    imported_chain = EventChain.import_chain(exported)
    
    # Verify chain properties identical
    assert imported_chain.head_hash == original_head
    assert imported_chain.length == original_length
    
    # Replay to get state
    replayed_state = replay_events(imported_chain.slice())
    
    # Verify state contents match
    assert len(replayed_state.commitments) == len(state.commitments)
    assert replayed_state.total_sigma == state.total_sigma
    assert len(replayed_state.quarantined) == len(state.quarantined)
    
    # Verify commitment details match
    for cid in state.commitments:
        assert cid in replayed_state.commitments
        assert state.commitments[cid].sigma == replayed_state.commitments[cid].sigma
        assert state.commitments[cid].status == replayed_state.commitments[cid].status
    
    print(f"Original head: {original_head[:16]}...")
    print(f"Imported head: {imported_chain.head_hash[:16]}...")
    print(f"Events: {original_length}")
    print(f"Commitments match: ✓")
    print(f"Sigma match: ✓ ({state.total_sigma})")
    
    print("\n✓ Export/import roundtrip successful\n")
    return True


def run_all_tests():
    """Run all integrity tests."""
    print("=" * 70)
    print("STATE INTEGRITY SEALING TESTS")
    print("Hash Chain + Deterministic Replay")
    print("=" * 70 + "\n")
    
    results = []
    
    results.append(("event_chain_basic", test_event_chain_basic()))
    results.append(("hash_chain_continuity", test_hash_chain_continuity()))
    results.append(("replay_determinism", test_replay_determinism()))
    results.append(("mutation_without_event", test_mutation_without_event_fails()))
    results.append(("truncated_chain", test_truncated_chain_detected()))
    results.append(("modified_event", test_modified_event_detected()))
    results.append(("sequence_monotonicity", test_sequence_monotonicity()))
    results.append(("integrity_monitor", test_integrity_monitor()))
    results.append(("forbidden_logging", test_forbidden_logging()))
    results.append(("export_import_roundtrip", test_export_import_roundtrip()))
    
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "=" * 70)
        print("✓ ALL INTEGRITY TESTS PASSED")
        print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
