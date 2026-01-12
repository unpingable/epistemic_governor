"""
Streaming Telemetry Index

Derives telemetry from append-only ledger entries without rebuilding claims.
Like reading flight data recorder streams, not reconstructing the aircraft.

Principle: Ledger is authoritative. Telemetry is a view.
Think: `tail -f ledger.jsonl | telemetry_index.update(entry)`

This is O(active_claims) worst case, and active_claims is bounded by governor behavior.

Usage:
    from epistemic_governor.streaming_telemetry import (
        StreamingTelemetryIndex,
        TelemetryEvent,
    )
    
    index = StreamingTelemetryIndex()
    
    # Process entries as they arrive
    for entry in ledger.stream():
        events = index.process_entry(entry)
        for event in events:
            print(event.to_event_string())
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Deque, Any
from collections import deque
from datetime import datetime
from enum import Enum, auto
import json


# =============================================================================
# Event Types
# =============================================================================

class TelemetryEventType(Enum):
    """Types of telemetry events (declarative, not advice)."""
    # Warnings
    WARN_CAR_HIGH = auto()          # Claim-Evidence Accretion Ratio high
    WARN_CAR_ACCELERATING = auto()  # CAR slope positive
    WARN_HALF_LIFE_HIGH = auto()    # Unresolved claims aging
    WARN_SLACK_LOW = auto()         # Near constraint boundary
    WARN_DROPS_CLUSTERED = auto()   # Soft constraints dropping frequently
    WARN_OSCILLATION = auto()       # Ringing near boundary
    WARN_FOSSILIZATION_HIGH = auto()# Too many claims being archived
    WARN_REVISION_LOAD_HIGH = auto()# Turbulent air
    
    # Info (for audit, not escalation)
    INFO_COMMIT = auto()
    INFO_REVISION = auto()
    INFO_FORK_CREATED = auto()
    INFO_FORK_MERGED = auto()
    INFO_FORK_ABANDONED = auto()
    INFO_CONTEXT_RESET = auto()
    INFO_AUTOPILOT_TRANSITION = auto()


@dataclass
class TelemetryEvent:
    """
    A declarative telemetry event.
    
    No verbs. No advice. Just measurements.
    """
    event_type: TelemetryEventType
    step: int
    timestamp: datetime
    
    # Metrics
    metric_name: str = ""
    value: float = 0.0
    threshold: float = 0.0
    slope: Optional[float] = None
    
    # Context
    claim_id: Optional[str] = None
    entry_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_event_string(self) -> str:
        """Format as declarative event string."""
        parts = [f"{self.metric_name}={self.value:.3f}"]
        if self.slope is not None:
            parts.append(f"slope={self.slope:+.3f}")
        if self.threshold:
            parts.append(f"threshold={self.threshold:.3f}")
        return f"{self.event_type.name}({', '.join(parts)})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.name,
            "step": self.step,
            "timestamp": self.timestamp.isoformat(),
            "metric_name": self.metric_name,
            "value": self.value,
            "threshold": self.threshold,
            "slope": self.slope,
            "claim_id": self.claim_id,
            "entry_id": self.entry_id,
            "details": self.details,
        }


# =============================================================================
# Rolling Window
# =============================================================================

@dataclass
class RollingWindow:
    """Efficient rolling window for streaming metrics."""
    size: int = 20
    values: Deque[float] = field(default_factory=deque)
    
    def __post_init__(self):
        self.values = deque(maxlen=self.size)
    
    def add(self, value: float):
        self.values.append(value)
    
    @property
    def count(self) -> int:
        return len(self.values)
    
    @property
    def sum(self) -> float:
        return sum(self.values)
    
    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return self.sum / len(self.values)
    
    @property
    def slope(self) -> float:
        """Linear regression slope."""
        if len(self.values) < 2:
            return 0.0
        
        n = len(self.values)
        x_mean = (n - 1) / 2
        y_mean = self.mean
        
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(self.values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator else 0.0
    
    @property
    def is_increasing(self) -> bool:
        return self.slope > 0.01
    
    @property
    def is_decreasing(self) -> bool:
        return self.slope < -0.01
    
    def recent(self, n: int = 3) -> List[float]:
        """Get last n values."""
        return list(self.values)[-n:]


# =============================================================================
# Streaming Telemetry Index
# =============================================================================

@dataclass
class StreamingTelemetryThresholds:
    """Configurable thresholds for telemetry warnings."""
    car_warn: float = 2.0           # Claims per evidence
    car_accel_warn: float = 0.05    # CAR slope
    half_life_warn: int = 10        # Steps until resolution
    slack_warn: float = 0.25        # Margin to boundary
    drop_cluster_warn: int = 3      # Drops in recent window
    oscillation_warn: int = 3       # Direction changes
    fossilization_warn: float = 0.3 # Archive/commit ratio
    revision_load_warn: float = 0.2 # Revisions per step


class StreamingTelemetryIndex:
    """
    Streaming telemetry index that processes ledger entries.
    
    Maintains only:
    - Rolling windows for rate metrics
    - Small maps for claim state (birth step, grounded status)
    - No claim text, no entity index, no proposition hashes
    
    This is what makes it ops-viable.
    """
    
    def __init__(
        self,
        window_size: int = 20,
        thresholds: StreamingTelemetryThresholds = None,
    ):
        self.window_size = window_size
        self.thresholds = thresholds or StreamingTelemetryThresholds()
        
        # Step counter
        self.step = 0
        
        # Rolling windows
        self.car_window = RollingWindow(size=window_size)
        self.half_life_window = RollingWindow(size=window_size)
        self.slack_window = RollingWindow(size=window_size)
        self.drop_window = RollingWindow(size=window_size)
        self.near_violation_window = RollingWindow(size=window_size)
        self.revision_window = RollingWindow(size=window_size)
        
        # Claim state (minimal)
        self.active_claim_birth_step: Dict[str, int] = {}
        self.active_claim_grounded: Dict[str, bool] = {}
        self.active_claims: Set[str] = set()
        
        # Counters for rates
        self.total_commits = 0
        self.total_archives = 0
        self.total_revisions = 0
        
        # Window accumulators
        self._window_commits = 0
        self._window_evidence = 0
        self._window_revisions = 0
        self._window_drops = 0
    
    def process_entry(self, entry: Dict[str, Any]) -> List[TelemetryEvent]:
        """
        Process a single ledger entry.
        
        Returns list of telemetry events (warnings).
        """
        events = []
        entry_type = entry.get("entry_type", "")
        claim_id = entry.get("claim_id", "")
        entry_id = entry.get("entry_id", "")
        data = entry.get("data", {})
        
        # Increment step
        self.step += 1
        
        # Process by entry type
        if entry_type == "commit":
            events.extend(self._process_commit(entry_id, claim_id, data))
        
        elif entry_type == "revision":
            events.extend(self._process_revision(entry_id, claim_id, data))
        
        elif entry_type == "supersede" or entry_type == "archive":
            events.extend(self._process_archive(entry_id, claim_id, data))
        
        elif entry_type == "fork":
            events.append(TelemetryEvent(
                event_type=TelemetryEventType.INFO_FORK_CREATED,
                step=self.step,
                timestamp=datetime.now(),
                entry_id=entry_id,
                details=data,
            ))
        
        elif entry_type == "merge":
            events.append(TelemetryEvent(
                event_type=TelemetryEventType.INFO_FORK_MERGED,
                step=self.step,
                timestamp=datetime.now(),
                entry_id=entry_id,
                details=data,
            ))
        
        elif entry_type == "context_reset":
            events.append(TelemetryEvent(
                event_type=TelemetryEventType.INFO_CONTEXT_RESET,
                step=self.step,
                timestamp=datetime.now(),
                entry_id=entry_id,
                details=data,
            ))
            # Archive affected claims
            for affected_id in data.get("archived_claim_ids", []):
                self._remove_active(affected_id)
        
        elif entry_type == "telemetry":
            events.extend(self._process_telemetry_snapshot(data))
        
        elif entry_type == "autopilot_transition":
            events.append(TelemetryEvent(
                event_type=TelemetryEventType.INFO_AUTOPILOT_TRANSITION,
                step=self.step,
                timestamp=datetime.now(),
                entry_id=entry_id,
                details=data,
            ))
            # Count drops
            drops = data.get("dropped_constraints", [])
            if drops:
                self._window_drops += len(drops)
        
        # Compute window metrics periodically
        if self.step % 5 == 0:
            events.extend(self._compute_window_metrics())
        
        return events
    
    def _process_commit(
        self,
        entry_id: str,
        claim_id: str,
        data: Dict,
    ) -> List[TelemetryEvent]:
        """Process a COMMIT entry."""
        events = []
        
        # Track claim
        self.active_claims.add(claim_id)
        self.active_claim_birth_step[claim_id] = self.step
        
        # Check if grounded (has support refs)
        support_refs = data.get("support_refs", [])
        support_count = len(support_refs) if isinstance(support_refs, list) else support_refs
        self.active_claim_grounded[claim_id] = support_count > 0
        
        # Update counters
        self.total_commits += 1
        self._window_commits += 1
        self._window_evidence += support_count
        
        # Info event
        events.append(TelemetryEvent(
            event_type=TelemetryEventType.INFO_COMMIT,
            step=self.step,
            timestamp=datetime.now(),
            claim_id=claim_id,
            entry_id=entry_id,
            details={"support_refs_count": support_count},
        ))
        
        return events
    
    def _process_revision(
        self,
        entry_id: str,
        claim_id: str,
        data: Dict,
    ) -> List[TelemetryEvent]:
        """Process a REVISION entry."""
        events = []
        
        # Track new claim
        self.active_claims.add(claim_id)
        self.active_claim_birth_step[claim_id] = self.step
        
        # Remove superseded claims
        for old_id in data.get("superseded_ids", []):
            self._remove_active(old_id)
        
        # Update counters
        self.total_revisions += 1
        self._window_revisions += 1
        
        # Info event
        cost = data.get("cost", 0)
        events.append(TelemetryEvent(
            event_type=TelemetryEventType.INFO_REVISION,
            step=self.step,
            timestamp=datetime.now(),
            claim_id=claim_id,
            entry_id=entry_id,
            details={"cost": cost, "superseded": data.get("superseded_ids", [])},
        ))
        
        return events
    
    def _process_archive(
        self,
        entry_id: str,
        claim_id: str,
        data: Dict,
    ) -> List[TelemetryEvent]:
        """Process SUPERSEDE or ARCHIVE entry."""
        self._remove_active(claim_id)
        self.total_archives += 1
        return []
    
    def _process_telemetry_snapshot(self, data: Dict) -> List[TelemetryEvent]:
        """Process a TELEMETRY snapshot entry."""
        events = []
        
        # Extract slack margins
        slack = data.get("provenance_slack", 1.0)
        self.slack_window.add(slack)
        
        if slack < self.thresholds.slack_warn:
            events.append(TelemetryEvent(
                event_type=TelemetryEventType.WARN_SLACK_LOW,
                step=self.step,
                timestamp=datetime.now(),
                metric_name="provenance_slack",
                value=slack,
                threshold=self.thresholds.slack_warn,
                slope=self.slack_window.slope,
            ))
        
        # Near violations
        near_violations = data.get("near_violations", 0)
        self.near_violation_window.add(float(near_violations))
        
        return events
    
    def _remove_active(self, claim_id: str):
        """Remove a claim from active set."""
        self.active_claims.discard(claim_id)
        self.active_claim_birth_step.pop(claim_id, None)
        self.active_claim_grounded.pop(claim_id, None)
    
    def _compute_window_metrics(self) -> List[TelemetryEvent]:
        """Compute and reset window metrics."""
        events = []
        
        # 1. CAR (Claim-Evidence Accretion Ratio)
        if self._window_commits > 0:
            car = self._window_commits / max(self._window_evidence, 1)
            self.car_window.add(car)
            
            if car > self.thresholds.car_warn:
                events.append(TelemetryEvent(
                    event_type=TelemetryEventType.WARN_CAR_HIGH,
                    step=self.step,
                    timestamp=datetime.now(),
                    metric_name="car",
                    value=car,
                    threshold=self.thresholds.car_warn,
                ))
            
            if self.car_window.is_increasing and self.car_window.slope > self.thresholds.car_accel_warn:
                events.append(TelemetryEvent(
                    event_type=TelemetryEventType.WARN_CAR_ACCELERATING,
                    step=self.step,
                    timestamp=datetime.now(),
                    metric_name="car_slope",
                    value=self.car_window.mean,
                    slope=self.car_window.slope,
                ))
        
        # 2. Half-life (median age of ungrounded claims)
        ungrounded_ages = [
            self.step - self.active_claim_birth_step[cid]
            for cid in self.active_claims
            if not self.active_claim_grounded.get(cid, False)
        ]
        
        if ungrounded_ages:
            sorted_ages = sorted(ungrounded_ages)
            half_life = sorted_ages[len(sorted_ages) // 2]
            self.half_life_window.add(float(half_life))
            
            if half_life > self.thresholds.half_life_warn:
                events.append(TelemetryEvent(
                    event_type=TelemetryEventType.WARN_HALF_LIFE_HIGH,
                    step=self.step,
                    timestamp=datetime.now(),
                    metric_name="half_life",
                    value=float(half_life),
                    threshold=float(self.thresholds.half_life_warn),
                    slope=self.half_life_window.slope,
                ))
        
        # 3. Drop clustering
        self.drop_window.add(float(self._window_drops))
        recent_drops = sum(self.drop_window.recent(3))
        
        if recent_drops >= self.thresholds.drop_cluster_warn:
            events.append(TelemetryEvent(
                event_type=TelemetryEventType.WARN_DROPS_CLUSTERED,
                step=self.step,
                timestamp=datetime.now(),
                metric_name="recent_drops",
                value=recent_drops,
                threshold=float(self.thresholds.drop_cluster_warn),
            ))
        
        # 4. Oscillation (slack slope changes)
        if self.slack_window.count >= 4:
            values = list(self.slack_window.values)
            slopes = [values[i] - values[i-1] for i in range(1, len(values))]
            sign_changes = sum(
                1 for i in range(1, len(slopes))
                if (slopes[i] > 0) != (slopes[i-1] > 0)
            )
            
            if sign_changes >= self.thresholds.oscillation_warn:
                events.append(TelemetryEvent(
                    event_type=TelemetryEventType.WARN_OSCILLATION,
                    step=self.step,
                    timestamp=datetime.now(),
                    metric_name="slack_oscillation",
                    value=float(sign_changes),
                    threshold=float(self.thresholds.oscillation_warn),
                ))
        
        # 5. Fossilization debt
        if self.total_commits > 10:
            fossilization = self.total_archives / self.total_commits
            if fossilization > self.thresholds.fossilization_warn:
                events.append(TelemetryEvent(
                    event_type=TelemetryEventType.WARN_FOSSILIZATION_HIGH,
                    step=self.step,
                    timestamp=datetime.now(),
                    metric_name="fossilization_debt",
                    value=fossilization,
                    threshold=self.thresholds.fossilization_warn,
                ))
        
        # 6. Revision load
        self.revision_window.add(float(self._window_revisions))
        if self.revision_window.mean > self.thresholds.revision_load_warn:
            events.append(TelemetryEvent(
                event_type=TelemetryEventType.WARN_REVISION_LOAD_HIGH,
                step=self.step,
                timestamp=datetime.now(),
                metric_name="revision_load",
                value=self.revision_window.mean,
                threshold=self.thresholds.revision_load_warn,
            ))
        
        # Reset window accumulators
        self._window_commits = 0
        self._window_evidence = 0
        self._window_revisions = 0
        self._window_drops = 0
        
        return events
    
    def get_summary(self) -> Dict[str, Any]:
        """Get current telemetry summary."""
        return {
            "step": self.step,
            "active_claims": len(self.active_claims),
            "ungrounded_claims": sum(
                1 for cid in self.active_claims
                if not self.active_claim_grounded.get(cid, False)
            ),
            "total_commits": self.total_commits,
            "total_archives": self.total_archives,
            "total_revisions": self.total_revisions,
            "car_mean": self.car_window.mean,
            "car_slope": self.car_window.slope,
            "half_life_mean": self.half_life_window.mean,
            "slack_current": list(self.slack_window.values)[-1] if self.slack_window.values else 1.0,
            "fossilization_rate": self.total_archives / max(self.total_commits, 1),
        }
    
    def checkpoint(self) -> Dict[str, Any]:
        """
        Create a checkpoint for persistence.
        
        Can be restored later to resume processing.
        """
        return {
            "step": self.step,
            "active_claims": list(self.active_claims),
            "active_claim_birth_step": dict(self.active_claim_birth_step),
            "active_claim_grounded": dict(self.active_claim_grounded),
            "total_commits": self.total_commits,
            "total_archives": self.total_archives,
            "total_revisions": self.total_revisions,
            "car_values": list(self.car_window.values),
            "half_life_values": list(self.half_life_window.values),
            "slack_values": list(self.slack_window.values),
        }
    
    def restore(self, checkpoint: Dict[str, Any]):
        """Restore from checkpoint."""
        self.step = checkpoint["step"]
        self.active_claims = set(checkpoint["active_claims"])
        self.active_claim_birth_step = dict(checkpoint["active_claim_birth_step"])
        self.active_claim_grounded = dict(checkpoint["active_claim_grounded"])
        self.total_commits = checkpoint["total_commits"]
        self.total_archives = checkpoint["total_archives"]
        self.total_revisions = checkpoint["total_revisions"]
        
        for v in checkpoint.get("car_values", []):
            self.car_window.add(v)
        for v in checkpoint.get("half_life_values", []):
            self.half_life_window.add(v)
        for v in checkpoint.get("slack_values", []):
            self.slack_window.add(v)


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=== Streaming Telemetry Demo ===\n")
    
    index = StreamingTelemetryIndex(window_size=10)
    
    # Simulate ledger entries
    entries = [
        {"entry_type": "commit", "entry_id": "e1", "claim_id": "c1", "data": {"support_refs": ["ref1"]}},
        {"entry_type": "commit", "entry_id": "e2", "claim_id": "c2", "data": {"support_refs": []}},
        {"entry_type": "commit", "entry_id": "e3", "claim_id": "c3", "data": {"support_refs": []}},
        {"entry_type": "commit", "entry_id": "e4", "claim_id": "c4", "data": {"support_refs": []}},
        {"entry_type": "commit", "entry_id": "e5", "claim_id": "c5", "data": {"support_refs": ["ref2"]}},
        {"entry_type": "revision", "entry_id": "e6", "claim_id": "c6", "data": {"superseded_ids": ["c2"], "cost": 0.5}},
        {"entry_type": "archive", "entry_id": "e7", "claim_id": "c3", "data": {}},
        {"entry_type": "telemetry", "entry_id": "e8", "claim_id": "", "data": {"provenance_slack": 0.3}},
    ]
    
    print("Processing entries...")
    all_events = []
    
    for entry in entries:
        events = index.process_entry(entry)
        for event in events:
            if event.event_type.name.startswith("WARN"):
                print(f"  {event.to_event_string()}")
            all_events.append(event)
    
    print(f"\nProcessed {len(entries)} entries, generated {len(all_events)} events")
    print(f"Warnings: {sum(1 for e in all_events if e.event_type.name.startswith('WARN'))}")
    
    print("\n--- Summary ---")
    summary = index.get_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")
