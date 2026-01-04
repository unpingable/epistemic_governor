"""
Concurrency Safety for the Epistemic Governor

Race condition risks in a controller with retries, pressure, hysteresis:
- Pressure/thermal state updates from concurrent workers → lost updates
- Accretion store writes without transactions → corrupted provenance
- Retry loops with shared counters → double-increments
- Tool results arriving out of order → stale data attached to wrong turn
- Classification history appended non-atomically → shuffled entries

Solution: Event sourcing with atomic turns.

Key principles:
1. Turn is the unit of atomicity - everything keyed by turn_id
2. Single writer for global state - one loop owns commits
3. Immutable snapshots - envelope in → envelope out
4. Event log first, state derived - append-only, deterministic replay

Usage:
    from epistemic_governor.concurrency import (
        TurnEvent,
        EventLog,
        AtomicTurn,
        SafeController,
    )
    
    log = EventLog()
    controller = SafeController(log)
    
    with controller.begin_turn() as turn:
        turn.record_violation("claim rejected")
        turn.record_tool_result("R:search_001", result)
        turn.set_classification(Regime.CONFABULATION)
        # All changes committed atomically when context exits
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum, auto
from datetime import datetime
from threading import Lock, RLock
from contextlib import contextmanager
import json
import copy


# =============================================================================
# Events (Immutable Records)
# =============================================================================

class EventType(Enum):
    """Types of events in the log."""
    TURN_START = auto()
    TURN_END = auto()
    
    # State changes
    CLAIM_PROPOSED = auto()
    CLAIM_COMMITTED = auto()
    CLAIM_REJECTED = auto()
    CLAIM_HEDGED = auto()
    
    # Violations
    VIOLATION = auto()
    CONTRADICTION = auto()
    
    # Tool interactions
    TOOL_CALLED = auto()
    TOOL_RESULT = auto()
    
    # Controller signals
    PRESSURE_UPDATE = auto()
    LEVEL_CHANGE = auto()
    REGIME_CHANGE = auto()
    
    # Accretion
    FACT_ACCRETED = auto()
    FACT_SUPERSEDED = auto()
    FACT_DECAYED = auto()
    
    # Retry/recovery
    RETRY = auto()
    FUSE_BLOWN = auto()
    GATE_ENTERED = auto()
    GATE_EXITED = auto()


@dataclass(frozen=True)  # Immutable
class TurnEvent:
    """
    An immutable event record.
    
    Frozen dataclass ensures events can't be modified after creation.
    """
    turn_id: int
    sequence: int                    # Order within turn
    timestamp: datetime
    event_type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "sequence": self.sequence,
            "timestamp": self.timestamp.isoformat(),
            "type": self.event_type.name,
            "data": self.data,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TurnEvent":
        return cls(
            turn_id=d["turn_id"],
            sequence=d["sequence"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            event_type=EventType[d["type"]],
            data=d.get("data", {}),
        )


# =============================================================================
# Event Log (Append-Only)
# =============================================================================

class EventLog:
    """
    Append-only event log.
    
    All state changes go through here first.
    State is derived from the log, not the other way around.
    
    Thread-safe: uses lock for appends.
    """
    
    def __init__(self):
        self._events: List[TurnEvent] = []
        self._lock = Lock()
        self._turn_counter = 0
    
    def append(self, event: TurnEvent):
        """Append an event (thread-safe)."""
        with self._lock:
            self._events.append(event)
    
    def append_batch(self, events: List[TurnEvent]):
        """Append multiple events atomically."""
        with self._lock:
            self._events.extend(events)
    
    def get_turn_events(self, turn_id: int) -> List[TurnEvent]:
        """Get all events for a turn."""
        with self._lock:
            return [e for e in self._events if e.turn_id == turn_id]
    
    def get_recent(self, n: int = 100) -> List[TurnEvent]:
        """Get recent events."""
        with self._lock:
            return list(self._events[-n:])
    
    def next_turn_id(self) -> int:
        """Get next turn ID (thread-safe)."""
        with self._lock:
            self._turn_counter += 1
            return self._turn_counter
    
    @property
    def current_turn(self) -> int:
        """Current turn ID."""
        with self._lock:
            return self._turn_counter
    
    def replay(self, handler: Callable[[TurnEvent], None]):
        """Replay all events through a handler."""
        with self._lock:
            events = list(self._events)
        for event in events:
            handler(event)
    
    def to_json(self) -> str:
        """Serialize log to JSON."""
        with self._lock:
            return json.dumps([e.to_dict() for e in self._events], indent=2)
    
    @classmethod
    def from_json(cls, data: str) -> "EventLog":
        """Deserialize log from JSON."""
        log = cls()
        events = json.loads(data)
        log._events = [TurnEvent.from_dict(e) for e in events]
        if log._events:
            log._turn_counter = max(e.turn_id for e in log._events)
        return log


# =============================================================================
# Atomic Turn (Transaction)
# =============================================================================

class AtomicTurn:
    """
    A turn transaction that collects events and commits atomically.
    
    All state changes within a turn are buffered until commit.
    If an error occurs, changes can be rolled back.
    
    Usage:
        with controller.begin_turn() as turn:
            turn.record_claim_committed(...)
            turn.record_violation(...)
            # Auto-commits on exit, or rolls back on exception
    """
    
    def __init__(self, turn_id: int, log: EventLog):
        self.turn_id = turn_id
        self.log = log
        self._events: List[TurnEvent] = []
        self._sequence = 0
        self._committed = False
        self._rolled_back = False
        
        # Record turn start
        self._add_event(EventType.TURN_START, {})
    
    def _add_event(self, event_type: EventType, data: Dict[str, Any]):
        """Add event to buffer (not yet committed)."""
        if self._committed or self._rolled_back:
            raise RuntimeError("Turn already finalized")
        
        event = TurnEvent(
            turn_id=self.turn_id,
            sequence=self._sequence,
            timestamp=datetime.now(),
            event_type=event_type,
            data=data,
        )
        self._events.append(event)
        self._sequence += 1
    
    # === Event Recording Methods ===
    
    def record_claim_proposed(self, claim_id: str, text: str, confidence: float):
        self._add_event(EventType.CLAIM_PROPOSED, {
            "claim_id": claim_id,
            "text": text,
            "confidence": confidence,
        })
    
    def record_claim_committed(self, claim_id: str, final_confidence: float):
        self._add_event(EventType.CLAIM_COMMITTED, {
            "claim_id": claim_id,
            "confidence": final_confidence,
        })
    
    def record_claim_rejected(self, claim_id: str, reason: str):
        self._add_event(EventType.CLAIM_REJECTED, {
            "claim_id": claim_id,
            "reason": reason,
        })
    
    def record_claim_hedged(self, claim_id: str, hedge_text: str):
        self._add_event(EventType.CLAIM_HEDGED, {
            "claim_id": claim_id,
            "hedge_text": hedge_text,
        })
    
    def record_violation(self, reason: str, severity: float = 1.0):
        self._add_event(EventType.VIOLATION, {
            "reason": reason,
            "severity": severity,
        })
    
    def record_contradiction(self, claim_id: str, conflicts_with: List[str]):
        self._add_event(EventType.CONTRADICTION, {
            "claim_id": claim_id,
            "conflicts_with": conflicts_with,
        })
    
    def record_tool_called(self, tool_name: str, request_id: str, params: Dict):
        self._add_event(EventType.TOOL_CALLED, {
            "tool": tool_name,
            "request_id": request_id,
            "params": params,
        })
    
    def record_tool_result(self, request_id: str, result: Any, success: bool = True):
        self._add_event(EventType.TOOL_RESULT, {
            "request_id": request_id,
            "result": result,
            "success": success,
        })
    
    def record_pressure_update(self, old_pressure: float, new_pressure: float):
        self._add_event(EventType.PRESSURE_UPDATE, {
            "old": old_pressure,
            "new": new_pressure,
            "delta": new_pressure - old_pressure,
        })
    
    def record_level_change(self, old_level: str, new_level: str):
        self._add_event(EventType.LEVEL_CHANGE, {
            "old": old_level,
            "new": new_level,
        })
    
    def record_regime_change(self, old_regime: str, new_regime: str):
        self._add_event(EventType.REGIME_CHANGE, {
            "old": old_regime,
            "new": new_regime,
        })
    
    def record_fact_accreted(self, fact_id: str, claim: str, support_ids: List[str]):
        self._add_event(EventType.FACT_ACCRETED, {
            "fact_id": fact_id,
            "claim": claim,
            "support_ids": support_ids,
        })
    
    def record_retry(self, attempt: int, reason: str):
        self._add_event(EventType.RETRY, {
            "attempt": attempt,
            "reason": reason,
        })
    
    def record_fuse_blown(self, attempts: int, final_reason: str):
        self._add_event(EventType.FUSE_BLOWN, {
            "attempts": attempts,
            "reason": final_reason,
        })
    
    def record_gate_entered(self, gate_type: str, trigger: str):
        self._add_event(EventType.GATE_ENTERED, {
            "gate_type": gate_type,
            "trigger": trigger,
        })
    
    def record_gate_exited(self, gate_type: str, stable_turns: int):
        self._add_event(EventType.GATE_EXITED, {
            "gate_type": gate_type,
            "stable_turns": stable_turns,
        })
    
    # === Transaction Control ===
    
    def commit(self):
        """Commit all buffered events to the log."""
        if self._committed or self._rolled_back:
            return
        
        self._add_event(EventType.TURN_END, {
            "event_count": len(self._events),
        })
        
        self.log.append_batch(self._events)
        self._committed = True
    
    def rollback(self):
        """Discard all buffered events."""
        self._events.clear()
        self._rolled_back = True
    
    @property
    def events(self) -> List[TurnEvent]:
        """Get buffered events (for inspection)."""
        return list(self._events)


# =============================================================================
# Safe Controller Wrapper
# =============================================================================

class SafeController:
    """
    Thread-safe wrapper for the vector controller.
    
    Ensures:
    - All state updates go through event log
    - Turns are atomic
    - State can be reconstructed from events
    - No lost updates from concurrent access
    """
    
    def __init__(self, log: Optional[EventLog] = None):
        self.log = log or EventLog()
        self._lock = RLock()  # Reentrant for nested calls
        
        # Derived state (reconstructed from log)
        self._pressure: float = 0.0
        self._level: str = "NORMAL"
        self._regime: str = "GROUNDED"
        self._violations_this_session: int = 0
        self._commits_this_session: int = 0
        
        # Active turn tracking
        self._active_turn: Optional[AtomicTurn] = None
    
    @contextmanager
    def begin_turn(self):
        """
        Begin an atomic turn.
        
        Usage:
            with controller.begin_turn() as turn:
                turn.record_claim_committed(...)
                # Auto-commits on clean exit
                # Rolls back on exception
        """
        with self._lock:
            if self._active_turn is not None:
                raise RuntimeError("Turn already in progress")
            
            turn_id = self.log.next_turn_id()
            turn = AtomicTurn(turn_id, self.log)
            self._active_turn = turn
        
        try:
            yield turn
            turn.commit()
            self._apply_turn_events(turn)
        except Exception:
            turn.rollback()
            raise
        finally:
            with self._lock:
                self._active_turn = None
    
    def _apply_turn_events(self, turn: AtomicTurn):
        """Apply turn events to derived state."""
        with self._lock:
            for event in turn.events:
                if event.event_type == EventType.VIOLATION:
                    self._violations_this_session += 1
                elif event.event_type == EventType.CLAIM_COMMITTED:
                    self._commits_this_session += 1
                elif event.event_type == EventType.PRESSURE_UPDATE:
                    self._pressure = event.data.get("new", self._pressure)
                elif event.event_type == EventType.LEVEL_CHANGE:
                    self._level = event.data.get("new", self._level)
                elif event.event_type == EventType.REGIME_CHANGE:
                    self._regime = event.data.get("new", self._regime)
    
    def reconstruct_state(self):
        """Reconstruct all state from event log (for recovery)."""
        with self._lock:
            self._pressure = 0.0
            self._level = "NORMAL"
            self._regime = "GROUNDED"
            self._violations_this_session = 0
            self._commits_this_session = 0
            
            def handler(event: TurnEvent):
                if event.event_type == EventType.VIOLATION:
                    self._violations_this_session += 1
                elif event.event_type == EventType.CLAIM_COMMITTED:
                    self._commits_this_session += 1
                elif event.event_type == EventType.PRESSURE_UPDATE:
                    self._pressure = event.data.get("new", self._pressure)
                elif event.event_type == EventType.LEVEL_CHANGE:
                    self._level = event.data.get("new", self._level)
                elif event.event_type == EventType.REGIME_CHANGE:
                    self._regime = event.data.get("new", self._regime)
            
            self.log.replay(handler)
    
    @property
    def pressure(self) -> float:
        with self._lock:
            return self._pressure
    
    @property
    def level(self) -> str:
        with self._lock:
            return self._level
    
    @property
    def regime(self) -> str:
        with self._lock:
            return self._regime
    
    @property
    def current_turn(self) -> int:
        return self.log.current_turn
    
    def summary(self) -> Dict[str, Any]:
        """Get current state summary."""
        with self._lock:
            return {
                "current_turn": self.log.current_turn,
                "pressure": self._pressure,
                "level": self._level,
                "regime": self._regime,
                "violations": self._violations_this_session,
                "commits": self._commits_this_session,
                "total_events": len(self.log._events),
            }


# =============================================================================
# Immutable Snapshot
# =============================================================================

@dataclass(frozen=True)
class ControllerSnapshot:
    """
    Immutable snapshot of controller state.
    
    Use for passing state between components without
    risk of mutation.
    """
    turn_id: int
    timestamp: datetime
    pressure: float
    level: str
    regime: str
    position: Dict[str, float]
    velocity: Dict[str, float]
    
    @classmethod
    def from_controller(cls, controller: Any, turn_id: int) -> "ControllerSnapshot":
        """Create snapshot from controller state."""
        return cls(
            turn_id=turn_id,
            timestamp=datetime.now(),
            pressure=getattr(controller, 'pressure', 0.0),
            level=getattr(controller, '_current_level', 'NORMAL'),
            regime="UNKNOWN",  # Would come from regime detector
            position=getattr(controller, 'position', {}).to_dict() if hasattr(controller, 'position') else {},
            velocity=getattr(controller, 'velocity', {}).to_dict() if hasattr(controller, 'velocity') else {},
        )


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=== Concurrency Safety Demo ===\n")
    
    log = EventLog()
    controller = SafeController(log)
    
    print("--- Turn 1: Normal operation ---")
    with controller.begin_turn() as turn:
        turn.record_claim_proposed("C001", "Paris is the capital of France", 0.9)
        turn.record_claim_committed("C001", 0.85)
        turn.record_pressure_update(0.0, 0.05)
    
    print(f"After turn 1: {controller.summary()}")
    
    print("\n--- Turn 2: Violations ---")
    with controller.begin_turn() as turn:
        turn.record_claim_proposed("C002", "The moon is made of cheese", 0.95)
        turn.record_violation("Unsupported claim", severity=1.0)
        turn.record_claim_rejected("C002", "No evidence")
        turn.record_pressure_update(0.05, 0.45)
        turn.record_level_change("NORMAL", "ELEVATED")
    
    print(f"After turn 2: {controller.summary()}")
    
    print("\n--- Turn 3: Rollback on error ---")
    try:
        with controller.begin_turn() as turn:
            turn.record_claim_proposed("C003", "Test claim", 0.5)
            raise ValueError("Simulated error")
    except ValueError:
        print("Turn 3 rolled back due to error")
    
    print(f"After rollback: {controller.summary()}")
    
    print("\n--- Event Log ---")
    events = log.get_recent(10)
    for e in events:
        print(f"  Turn {e.turn_id}.{e.sequence}: {e.event_type.name} {e.data}")
    
    print("\n--- Reconstruct State ---")
    controller2 = SafeController(log)
    controller2.reconstruct_state()
    print(f"Reconstructed: {controller2.summary()}")
