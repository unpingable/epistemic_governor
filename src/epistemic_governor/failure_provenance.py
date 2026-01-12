"""
Failure Provenance

Structured failure classification and causal chain tracking.

"Prevent blame laundering via vague 'system error'."

This module implements:
1. Failure taxonomy by subsystem
2. Epistemic vs control failure distinction
3. Causal chain construction
4. Structured failure events

Every failure should be attributable to a specific subsystem and cause.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set


# =============================================================================
# Failure Taxonomy
# =============================================================================

class FailureSubsystem(Enum):
    """Which subsystem failed."""
    INPUT = auto()          # Extraction, parsing
    VALIDATION = auto()     # Claim validation
    CONTRADICTION = auto()  # Contradiction detection/resolution
    BUDGET = auto()         # Budget exhaustion
    TEMPORAL = auto()       # Timing violations
    AUTHORITY = auto()      # NLAI violations
    BOUNDARY = auto()       # Jurisdiction/interface violations
    VARIETY = auto()        # Load shedding
    ADAPTATION = auto()     # Ultrastability failures
    STORAGE = auto()        # Ledger/persistence
    UNKNOWN = auto()        # Should never happen


class FailureType(Enum):
    """Type of failure."""
    # Epistemic failures (about knowledge)
    EXTRACTION_FAILED = auto()      # Couldn't extract claims
    CONTRADICTION_UNRESOLVED = auto()  # Contradiction remains open
    EVIDENCE_INSUFFICIENT = auto()  # Not enough evidence
    EVIDENCE_STALE = auto()         # Evidence expired
    CLAIM_EXPIRED = auto()          # Claim TTL exceeded
    
    # Control failures (about mechanism)
    BUDGET_EXHAUSTED = auto()       # No budget remaining
    LAG_EXCEEDED = auto()           # Processing too slow
    CLOCK_DRIFT = auto()            # Temporal incoherence
    VARIETY_OVERFLOW = auto()       # Too many claims
    VARIETY_EXPLOIT = auto()        # Gaming detected
    
    # Authority failures (about permissions)
    NLAI_VIOLATION = auto()         # Language tried to commit
    FORBIDDEN_TRANSITION = auto()   # Invalid FSM transition
    JURISDICTION_ESCAPE = auto()    # Cross-jurisdiction violation
    BOUNDARY_BREACH = auto()        # Interface contract violated
    
    # Adaptation failures
    PATHOLOGY_DETECTED = auto()     # Ultrastability freeze
    OSCILLATION = auto()            # Parameter oscillation
    RUNAWAY = auto()                # Hitting bounds repeatedly
    WRONG_ATTRACTOR = auto()        # Stuck in bad regime
    
    # Storage failures
    LEDGER_CORRUPT = auto()         # Hash chain broken
    WRITE_FAILED = auto()           # Persistence failed
    
    # Meta failures
    UNKNOWN_ERROR = auto()          # Catch-all (should be rare)


class FailureSeverity(Enum):
    """How bad is it."""
    INFO = auto()       # Logged but not a problem
    WARNING = auto()    # Degraded but functional
    ERROR = auto()      # Operation failed
    CRITICAL = auto()   # System compromised
    FATAL = auto()      # Must halt


# =============================================================================
# Failure Event
# =============================================================================

@dataclass
class FailureCause:
    """One link in the causal chain."""
    cause_id: str
    subsystem: FailureSubsystem
    failure_type: FailureType
    description: str
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cause_id": self.cause_id,
            "subsystem": self.subsystem.name,
            "failure_type": self.failure_type.name,
            "description": self.description,
            "context": self.context,
        }


@dataclass
class FailureEvent:
    """
    A structured failure event with full provenance.
    
    Every failure should be traceable to:
    - Which subsystem failed
    - What type of failure
    - Why it happened (causal chain)
    - What was the immediate trigger
    - What was the root cause
    """
    event_id: str
    timestamp: datetime
    turn_id: int
    
    # Classification
    subsystem: FailureSubsystem
    failure_type: FailureType
    severity: FailureSeverity
    
    # Description
    summary: str
    details: Optional[str] = None
    
    # Causal chain (from immediate to root)
    causal_chain: List[FailureCause] = field(default_factory=list)
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Resolution
    resolved: bool = False
    resolution_action: Optional[str] = None
    
    @property
    def immediate_cause(self) -> Optional[FailureCause]:
        """Get the immediate cause (first in chain)."""
        return self.causal_chain[0] if self.causal_chain else None
    
    @property
    def root_cause(self) -> Optional[FailureCause]:
        """Get the root cause (last in chain)."""
        return self.causal_chain[-1] if self.causal_chain else None
    
    def add_cause(self, cause: FailureCause):
        """Add a cause to the chain."""
        self.causal_chain.append(cause)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "turn_id": self.turn_id,
            "subsystem": self.subsystem.name,
            "failure_type": self.failure_type.name,
            "severity": self.severity.name,
            "summary": self.summary,
            "details": self.details,
            "causal_chain": [c.to_dict() for c in self.causal_chain],
            "context": self.context,
            "resolved": self.resolved,
            "resolution_action": self.resolution_action,
        }


# =============================================================================
# Failure Registry
# =============================================================================

class FailureRegistry:
    """
    Collects and indexes failure events.
    
    Prevents blame laundering by requiring structured classification.
    """
    
    def __init__(self):
        self.events: List[FailureEvent] = []
        self.by_subsystem: Dict[FailureSubsystem, List[FailureEvent]] = {}
        self.by_type: Dict[FailureType, List[FailureEvent]] = {}
        self.by_severity: Dict[FailureSeverity, List[FailureEvent]] = {}
        
        # Counters
        self.total_failures: int = 0
        self.unresolved_count: int = 0
    
    def record(self, event: FailureEvent):
        """Record a failure event."""
        self.events.append(event)
        self.total_failures += 1
        
        if not event.resolved:
            self.unresolved_count += 1
        
        # Index
        self.by_subsystem.setdefault(event.subsystem, []).append(event)
        self.by_type.setdefault(event.failure_type, []).append(event)
        self.by_severity.setdefault(event.severity, []).append(event)
    
    def resolve(self, event_id: str, resolution_action: str):
        """Mark a failure as resolved."""
        for event in self.events:
            if event.event_id == event_id and not event.resolved:
                event.resolved = True
                event.resolution_action = resolution_action
                self.unresolved_count -= 1
                return True
        return False
    
    def get_unresolved(self) -> List[FailureEvent]:
        """Get all unresolved failures."""
        return [e for e in self.events if not e.resolved]
    
    def get_by_subsystem(self, subsystem: FailureSubsystem) -> List[FailureEvent]:
        """Get failures for a subsystem."""
        return self.by_subsystem.get(subsystem, [])
    
    def get_critical(self) -> List[FailureEvent]:
        """Get CRITICAL and FATAL failures."""
        critical = self.by_severity.get(FailureSeverity.CRITICAL, [])
        fatal = self.by_severity.get(FailureSeverity.FATAL, [])
        return critical + fatal
    
    def get_stats(self) -> Dict[str, Any]:
        """Get failure statistics."""
        by_subsystem = {s.name: len(v) for s, v in self.by_subsystem.items()}
        by_severity = {s.name: len(v) for s, v in self.by_severity.items()}
        
        return {
            "total_failures": self.total_failures,
            "unresolved": self.unresolved_count,
            "by_subsystem": by_subsystem,
            "by_severity": by_severity,
        }


# =============================================================================
# Failure Builder (Convenience)
# =============================================================================

class FailureBuilder:
    """
    Builder for constructing failure events with proper classification.
    
    Forces structured failure creation.
    """
    
    def __init__(self, turn_id: int = 0):
        self.turn_id = turn_id
        self._counter = 0
    
    def _next_id(self) -> str:
        self._counter += 1
        return f"F-{self.turn_id}-{self._counter}"
    
    def extraction_failed(
        self,
        reason: str,
        raw_input: Optional[str] = None,
    ) -> FailureEvent:
        """Create an extraction failure."""
        event = FailureEvent(
            event_id=self._next_id(),
            timestamp=datetime.now(timezone.utc),
            turn_id=self.turn_id,
            subsystem=FailureSubsystem.INPUT,
            failure_type=FailureType.EXTRACTION_FAILED,
            severity=FailureSeverity.WARNING,
            summary=f"Extraction failed: {reason}",
            context={"raw_input_length": len(raw_input) if raw_input else 0},
        )
        event.add_cause(FailureCause(
            cause_id="C1",
            subsystem=FailureSubsystem.INPUT,
            failure_type=FailureType.EXTRACTION_FAILED,
            description=reason,
        ))
        return event
    
    def budget_exhausted(
        self,
        budget_type: str,
        requested: float,
        remaining: float,
    ) -> FailureEvent:
        """Create a budget exhaustion failure."""
        event = FailureEvent(
            event_id=self._next_id(),
            timestamp=datetime.now(timezone.utc),
            turn_id=self.turn_id,
            subsystem=FailureSubsystem.BUDGET,
            failure_type=FailureType.BUDGET_EXHAUSTED,
            severity=FailureSeverity.ERROR,
            summary=f"Budget exhausted: {budget_type}",
            context={
                "budget_type": budget_type,
                "requested": requested,
                "remaining": remaining,
            },
        )
        event.add_cause(FailureCause(
            cause_id="C1",
            subsystem=FailureSubsystem.BUDGET,
            failure_type=FailureType.BUDGET_EXHAUSTED,
            description=f"Requested {requested}, only {remaining} remaining",
            context={"deficit": requested - remaining},
        ))
        return event
    
    def nlai_violation(
        self,
        attempted_action: str,
        source: str = "language",
    ) -> FailureEvent:
        """Create an NLAI violation failure."""
        event = FailureEvent(
            event_id=self._next_id(),
            timestamp=datetime.now(timezone.utc),
            turn_id=self.turn_id,
            subsystem=FailureSubsystem.AUTHORITY,
            failure_type=FailureType.NLAI_VIOLATION,
            severity=FailureSeverity.CRITICAL,
            summary=f"NLAI violation: {source} attempted {attempted_action}",
            context={"source": source, "attempted_action": attempted_action},
        )
        event.add_cause(FailureCause(
            cause_id="C1",
            subsystem=FailureSubsystem.AUTHORITY,
            failure_type=FailureType.NLAI_VIOLATION,
            description=f"Language attempted to {attempted_action} without evidence",
        ))
        return event
    
    def temporal_violation(
        self,
        violation_type: FailureType,
        details: str,
    ) -> FailureEvent:
        """Create a temporal violation failure."""
        event = FailureEvent(
            event_id=self._next_id(),
            timestamp=datetime.now(timezone.utc),
            turn_id=self.turn_id,
            subsystem=FailureSubsystem.TEMPORAL,
            failure_type=violation_type,
            severity=FailureSeverity.ERROR,
            summary=f"Temporal violation: {violation_type.name}",
            details=details,
        )
        event.add_cause(FailureCause(
            cause_id="C1",
            subsystem=FailureSubsystem.TEMPORAL,
            failure_type=violation_type,
            description=details,
        ))
        return event
    
    def pathology_detected(
        self,
        pathology_type: str,
        details: str,
    ) -> FailureEvent:
        """Create an adaptation pathology failure."""
        # Map pathology type to failure type
        type_map = {
            "oscillation": FailureType.OSCILLATION,
            "runaway": FailureType.RUNAWAY,
            "wrong_attractor": FailureType.WRONG_ATTRACTOR,
        }
        failure_type = type_map.get(pathology_type.lower(), FailureType.PATHOLOGY_DETECTED)
        
        event = FailureEvent(
            event_id=self._next_id(),
            timestamp=datetime.now(timezone.utc),
            turn_id=self.turn_id,
            subsystem=FailureSubsystem.ADAPTATION,
            failure_type=failure_type,
            severity=FailureSeverity.CRITICAL,
            summary=f"Adaptation pathology: {pathology_type}",
            details=details,
        )
        event.add_cause(FailureCause(
            cause_id="C1",
            subsystem=FailureSubsystem.ADAPTATION,
            failure_type=failure_type,
            description=details,
        ))
        return event
    
    def with_root_cause(
        self,
        event: FailureEvent,
        root_subsystem: FailureSubsystem,
        root_type: FailureType,
        root_description: str,
    ) -> FailureEvent:
        """Add a root cause to an existing failure."""
        event.add_cause(FailureCause(
            cause_id=f"C{len(event.causal_chain) + 1}",
            subsystem=root_subsystem,
            failure_type=root_type,
            description=root_description,
        ))
        return event


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=== Failure Provenance Test ===\n")
    
    registry = FailureRegistry()
    builder = FailureBuilder(turn_id=1)
    
    # Create various failures
    f1 = builder.extraction_failed("Invalid JSON in model output", "raw...")
    registry.record(f1)
    print(f"Recorded: {f1.summary}")
    
    f2 = builder.budget_exhausted("repair", requested=15.0, remaining=3.0)
    registry.record(f2)
    print(f"Recorded: {f2.summary}")
    
    f3 = builder.nlai_violation("commit claim", source="model_text")
    registry.record(f3)
    print(f"Recorded: {f3.summary}")
    
    # Add root cause
    f3 = builder.with_root_cause(
        f3,
        root_subsystem=FailureSubsystem.INPUT,
        root_type=FailureType.EVIDENCE_INSUFFICIENT,
        root_description="No external evidence provided with claim",
    )
    
    f4 = builder.pathology_detected("oscillation", "repair_budget bouncing between 50 and 60")
    registry.record(f4)
    print(f"Recorded: {f4.summary}")
    
    # Stats
    print("\n=== Statistics ===")
    import json
    print(json.dumps(registry.get_stats(), indent=2))
    
    # Critical failures
    print("\n=== Critical Failures ===")
    for f in registry.get_critical():
        print(f"  {f.event_id}: {f.summary}")
        if f.root_cause:
            print(f"    Root cause: {f.root_cause.description}")
    
    # Full event
    print("\n=== Full Event (f3) ===")
    print(json.dumps(f3.to_dict(), indent=2, default=str))
