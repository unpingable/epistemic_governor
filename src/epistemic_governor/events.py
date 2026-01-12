"""
Epistemic Event Logging

Structured event emission for observability and analysis.
Designed for both offline replay analysis and live monitoring.

Event types:
- TurnEvent: Per-turn telemetry
- ProposalEvent: Claim extraction results
- DecisionEvent: Governor decisions
- CommitEvent: Ledger mutations
- ThermalEvent: Temperature/instability snapshots
- DriftEvent: Position changes under pressure

Output formats:
- JSONL files (for replay/analysis)
- Prometheus metrics (for live monitoring)
- In-memory buffer (for testing)

Usage:
    from epistemic_governor.events import EventLogger, TurnEvent
    
    # File-based logging
    logger = EventLogger(output_path="events.jsonl")
    
    # Emit events
    logger.emit(TurnEvent(
        turn_id=1,
        prompt="What is X?",
        response="X is Y.",
        latency_ms=150.0,
    ))
    
    # Prometheus endpoint (if running as daemon)
    logger.start_metrics_server(port=9090)
"""

import json
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any, Union, Callable
from pathlib import Path
from enum import Enum
from collections import deque
import hashlib


# =============================================================================
# Event Types
# =============================================================================

class EventType(Enum):
    """Types of epistemic events."""
    TURN = "turn"
    PROPOSAL = "proposal"
    DECISION = "decision"
    COMMIT = "commit"
    THERMAL = "thermal"
    DRIFT = "drift"
    REVISION = "revision"
    REFUSAL = "refusal"
    EPOCH = "epoch"
    RESET = "reset"
    CALIBRATION = "calibration"


@dataclass
class BaseEvent:
    """Base class for all events."""
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        d["timestamp"] = self.timestamp.isoformat()
        return d
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class TurnEvent(BaseEvent):
    """
    Per-turn telemetry.
    
    Captures the full context of a single turn through the governor.
    """
    event_type: EventType = field(default=EventType.TURN)
    
    # Turn identification
    turn_id: int = 0
    
    # Input/Output
    prompt: str = ""
    prompt_hash: str = ""  # For deduplication
    response: str = ""
    response_length: int = 0
    
    # Timing
    latency_ms: float = 0.0
    extraction_ms: float = 0.0
    governance_ms: float = 0.0
    commit_ms: float = 0.0
    
    # Model info
    model_id: str = ""
    temperature: float = 0.0
    
    # Envelope constraints
    max_new_tokens: int = 0
    confidence_ceiling: float = 1.0
    
    # Results summary
    proposals_count: int = 0
    commits_count: int = 0
    hedges_count: int = 0
    blocks_count: int = 0
    revisions_count: int = 0
    
    def __post_init__(self):
        if self.prompt and not self.prompt_hash:
            self.prompt_hash = hashlib.sha256(self.prompt.encode()).hexdigest()[:16]
        if self.response:
            self.response_length = len(self.response)


@dataclass
class ProposalEvent(BaseEvent):
    """Claim extraction results."""
    event_type: EventType = field(default=EventType.PROPOSAL)
    
    turn_id: int = 0
    claim_id: str = ""
    claim_text: str = ""
    claim_type: str = ""
    extracted_confidence: float = 0.0
    proposition_hash: str = ""
    span_start: int = 0
    span_end: int = 0
    extracted_entities: List[str] = field(default_factory=list)


@dataclass
class DecisionEvent(BaseEvent):
    """Governor decision on a proposal."""
    event_type: EventType = field(default=EventType.DECISION)
    
    turn_id: int = 0
    claim_id: str = ""
    action: str = ""  # accept, hedge, block, revise, defer
    reason: str = ""
    cost: float = 0.0
    
    # Thresholds that triggered
    confidence_original: float = 0.0
    confidence_adjusted: Optional[float] = None
    delta_t: float = 0.0
    inversion_score: float = 0.0
    
    # What threshold was hit
    threshold_hit: Optional[str] = None  # e.g., "delta_t_block", "inversion_hedge"


@dataclass
class CommitEvent(BaseEvent):
    """Ledger mutation."""
    event_type: EventType = field(default=EventType.COMMIT)
    
    turn_id: int = 0
    claim_id: str = ""
    claim_text: str = ""
    claim_type: str = ""
    final_confidence: float = 0.0
    status: str = ""  # active, superseded, archived
    supersedes: Optional[str] = None
    support_refs: List[str] = field(default_factory=list)


@dataclass
class ThermalEvent(BaseEvent):
    """Temperature/instability snapshot."""
    event_type: EventType = field(default=EventType.THERMAL)
    
    turn_id: int = 0
    
    # Core thermal metrics
    instability_score: float = 0.0
    contradiction_density: float = 0.0
    revision_velocity: float = 0.0
    entropy: float = 0.0
    
    # Derived state
    thermal_state: str = ""  # cold, warming, hot, critical
    regime: str = ""  # hesitation, narrowing, crystallization, negative_t
    
    # Active claims snapshot
    active_claims_count: int = 0
    total_confidence: float = 0.0


@dataclass
class DriftEvent(BaseEvent):
    """Position change under pressure (gaslight detection)."""
    event_type: EventType = field(default=EventType.DRIFT)
    
    turn_id: int = 0
    original_claim_id: str = ""
    original_position: str = ""
    challenge_prompt: str = ""
    new_position: str = ""
    did_flip: bool = False
    flip_confidence: float = 0.0  # How confident the flip was


@dataclass
class RevisionEvent(BaseEvent):
    """Explicit revision of prior commitment."""
    event_type: EventType = field(default=EventType.REVISION)
    
    turn_id: int = 0
    new_claim_id: str = ""
    superseded_claim_ids: List[str] = field(default_factory=list)
    justification: str = ""
    cost: float = 0.0


@dataclass
class RefusalEvent(BaseEvent):
    """Refusal to commit."""
    event_type: EventType = field(default=EventType.REFUSAL)
    
    turn_id: int = 0
    claim_id: str = ""
    reason: str = ""
    severity: str = ""  # soft (hedged), hard (blocked)
    trigger: str = ""  # What caused the refusal


@dataclass
class CalibrationEvent(BaseEvent):
    """Calibration run summary."""
    event_type: EventType = field(default=EventType.CALIBRATION)
    
    model_id: str = ""
    corpus_size: int = 0
    
    # Results
    drift_sensitivity: float = 0.0
    trap_refusal_rate: float = 0.0
    baseline_confidence_mean: float = 0.0
    
    # Fitted policy
    recommended_preset: str = ""
    fitting_loss: float = 0.0


# =============================================================================
# Event Sinks
# =============================================================================

class EventSink(ABC):
    """Base class for event output destinations."""
    
    @abstractmethod
    def write(self, event: BaseEvent):
        """Write an event to the sink."""
        pass
    
    @abstractmethod
    def flush(self):
        """Flush any buffered events."""
        pass
    
    @abstractmethod
    def close(self):
        """Close the sink."""
        pass


class JSONLSink(EventSink):
    """Write events to a JSONL file."""
    
    def __init__(self, path: Union[str, Path], buffer_size: int = 100):
        self.path = Path(path)
        self.buffer_size = buffer_size
        self._buffer: List[str] = []
        self._file = None
        self._lock = threading.Lock()
        
        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)
    
    def write(self, event: BaseEvent):
        with self._lock:
            self._buffer.append(event.to_json())
            if len(self._buffer) >= self.buffer_size:
                self._flush_buffer()
    
    def flush(self):
        with self._lock:
            self._flush_buffer()
    
    def _flush_buffer(self):
        if not self._buffer:
            return
        
        with open(self.path, 'a') as f:
            for line in self._buffer:
                f.write(line + '\n')
        
        self._buffer.clear()
    
    def close(self):
        self.flush()


class MemorySink(EventSink):
    """Keep events in memory (for testing)."""
    
    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self.events: deque = deque(maxlen=max_events)
        self._lock = threading.Lock()
    
    def write(self, event: BaseEvent):
        with self._lock:
            self.events.append(event)
    
    def flush(self):
        pass  # No-op for memory
    
    def close(self):
        pass
    
    def get_events(self, event_type: Optional[EventType] = None) -> List[BaseEvent]:
        """Get events, optionally filtered by type."""
        with self._lock:
            if event_type:
                return [e for e in self.events if e.event_type == event_type]
            return list(self.events)
    
    def clear(self):
        with self._lock:
            self.events.clear()


class CallbackSink(EventSink):
    """Call a function for each event (for custom integrations)."""
    
    def __init__(self, callback: Callable[[BaseEvent], None]):
        self.callback = callback
    
    def write(self, event: BaseEvent):
        self.callback(event)
    
    def flush(self):
        pass
    
    def close(self):
        pass


# =============================================================================
# Prometheus Metrics (Optional)
# =============================================================================

class PrometheusMetrics:
    """
    Prometheus metrics exporter.
    
    Exposes metrics at /metrics endpoint for scraping.
    """
    
    def __init__(self):
        self._metrics: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        self._server = None
    
    def inc(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter."""
        key = self._make_key(name, labels)
        with self._lock:
            self._metrics[key] = self._metrics.get(key, 0) + value
    
    def set(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge value."""
        key = self._make_key(name, labels)
        with self._lock:
            self._metrics[key] = value
    
    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram observation."""
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)
            # Keep only last 1000 observations
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-1000:]
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def format_prometheus(self) -> str:
        """Format metrics in Prometheus text format."""
        lines = []
        
        with self._lock:
            # Counters and gauges
            for key, value in sorted(self._metrics.items()):
                lines.append(f"{key} {value}")
            
            # Histograms (simplified - just sum and count)
            for key, values in sorted(self._histograms.items()):
                if values:
                    lines.append(f"{key}_count {len(values)}")
                    lines.append(f"{key}_sum {sum(values)}")
        
        return "\n".join(lines)
    
    def start_server(self, port: int = 9090):
        """Start HTTP server for /metrics endpoint."""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        metrics = self
        
        class MetricsHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/metrics":
                    content = metrics.format_prometheus()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(content.encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                pass  # Suppress logging
        
        self._server = HTTPServer(("", port), MetricsHandler)
        thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        thread.start()
        print(f"Prometheus metrics server started on port {port}")
    
    def stop_server(self):
        if self._server:
            self._server.shutdown()


# =============================================================================
# Event Logger
# =============================================================================

class EventLogger:
    """
    Main event logging interface.
    
    Supports multiple sinks and optional Prometheus metrics.
    """
    
    def __init__(
        self,
        output_path: Optional[Union[str, Path]] = None,
        enable_prometheus: bool = False,
        prometheus_port: int = 9090,
        session_id: Optional[str] = None,
    ):
        self.session_id = session_id or self._generate_session_id()
        self._sinks: List[EventSink] = []
        self._metrics: Optional[PrometheusMetrics] = None
        self._turn_counter = 0
        self._lock = threading.Lock()
        
        # Add file sink if path provided
        if output_path:
            self._sinks.append(JSONLSink(output_path))
        
        # Always add memory sink for inspection
        self._memory_sink = MemorySink()
        self._sinks.append(self._memory_sink)
        
        # Prometheus metrics
        if enable_prometheus:
            self._metrics = PrometheusMetrics()
            self._metrics.start_server(prometheus_port)
    
    def _generate_session_id(self) -> str:
        return hashlib.sha256(
            f"{datetime.now().isoformat()}-{id(self)}".encode()
        ).hexdigest()[:12]
    
    def add_sink(self, sink: EventSink):
        """Add an additional event sink."""
        self._sinks.append(sink)
    
    def emit(self, event: BaseEvent):
        """Emit an event to all sinks."""
        # Set session ID if not set
        if not event.session_id:
            event.session_id = self.session_id
        
        # Write to all sinks
        for sink in self._sinks:
            sink.write(event)
        
        # Update Prometheus metrics
        if self._metrics:
            self._update_metrics(event)
    
    def _update_metrics(self, event: BaseEvent):
        """Update Prometheus metrics based on event."""
        labels = {"session": self.session_id}
        
        if isinstance(event, TurnEvent):
            self._metrics.inc("epistemic_turns_total", labels=labels)
            self._metrics.observe("epistemic_turn_latency_ms", event.latency_ms, labels=labels)
            self._metrics.set("epistemic_proposals_last", event.proposals_count, labels=labels)
            self._metrics.inc("epistemic_commits_total", event.commits_count, labels=labels)
            self._metrics.inc("epistemic_hedges_total", event.hedges_count, labels=labels)
            self._metrics.inc("epistemic_blocks_total", event.blocks_count, labels=labels)
        
        elif isinstance(event, ThermalEvent):
            self._metrics.set("epistemic_instability", event.instability_score, labels=labels)
            self._metrics.set("epistemic_entropy", event.entropy, labels=labels)
            self._metrics.set("epistemic_active_claims", event.active_claims_count, labels=labels)
        
        elif isinstance(event, DriftEvent):
            if event.did_flip:
                self._metrics.inc("epistemic_drifts_total", labels=labels)
        
        elif isinstance(event, RevisionEvent):
            self._metrics.inc("epistemic_revisions_total", labels=labels)
            self._metrics.observe("epistemic_revision_cost", event.cost, labels=labels)
    
    def next_turn(self) -> int:
        """Get next turn ID."""
        with self._lock:
            self._turn_counter += 1
            return self._turn_counter
    
    def get_events(self, event_type: Optional[EventType] = None) -> List[BaseEvent]:
        """Get events from memory sink."""
        return self._memory_sink.get_events(event_type)
    
    def flush(self):
        """Flush all sinks."""
        for sink in self._sinks:
            sink.flush()
    
    def close(self):
        """Close all sinks and stop metrics server."""
        self.flush()
        for sink in self._sinks:
            sink.close()
        if self._metrics:
            self._metrics.stop_server()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics from logged events."""
        events = self._memory_sink.get_events()
        
        turn_events = [e for e in events if isinstance(e, TurnEvent)]
        thermal_events = [e for e in events if isinstance(e, ThermalEvent)]
        drift_events = [e for e in events if isinstance(e, DriftEvent)]
        
        summary = {
            "session_id": self.session_id,
            "total_events": len(events),
            "total_turns": len(turn_events),
        }
        
        if turn_events:
            summary["avg_latency_ms"] = sum(e.latency_ms for e in turn_events) / len(turn_events)
            summary["total_commits"] = sum(e.commits_count for e in turn_events)
            summary["total_hedges"] = sum(e.hedges_count for e in turn_events)
            summary["total_blocks"] = sum(e.blocks_count for e in turn_events)
        
        if thermal_events:
            summary["final_instability"] = thermal_events[-1].instability_score
            summary["max_instability"] = max(e.instability_score for e in thermal_events)
        
        if drift_events:
            summary["drift_count"] = sum(1 for e in drift_events if e.did_flip)
        
        return summary


# =============================================================================
# Convenience Functions
# =============================================================================

def load_events(path: Union[str, Path]) -> List[BaseEvent]:
    """Load events from a JSONL file."""
    events = []
    
    event_classes = {
        "turn": TurnEvent,
        "proposal": ProposalEvent,
        "decision": DecisionEvent,
        "commit": CommitEvent,
        "thermal": ThermalEvent,
        "drift": DriftEvent,
        "revision": RevisionEvent,
        "refusal": RefusalEvent,
        "calibration": CalibrationEvent,
    }
    
    with open(path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                event_type = data.get("event_type", "")
                
                if event_type in event_classes:
                    # Reconstruct event
                    cls = event_classes[event_type]
                    data["event_type"] = EventType(event_type)
                    data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                    events.append(cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__}))
    
    return events


def analyze_events(events: List[BaseEvent]) -> Dict[str, Any]:
    """Analyze a list of events and return summary statistics."""
    turn_events = [e for e in events if isinstance(e, TurnEvent)]
    thermal_events = [e for e in events if isinstance(e, ThermalEvent)]
    decision_events = [e for e in events if isinstance(e, DecisionEvent)]
    drift_events = [e for e in events if isinstance(e, DriftEvent)]
    
    analysis = {
        "event_counts": {
            "total": len(events),
            "turns": len(turn_events),
            "decisions": len(decision_events),
            "thermal": len(thermal_events),
            "drifts": len(drift_events),
        }
    }
    
    if turn_events:
        latencies = [e.latency_ms for e in turn_events]
        analysis["latency"] = {
            "mean": sum(latencies) / len(latencies),
            "min": min(latencies),
            "max": max(latencies),
        }
        
        analysis["totals"] = {
            "proposals": sum(e.proposals_count for e in turn_events),
            "commits": sum(e.commits_count for e in turn_events),
            "hedges": sum(e.hedges_count for e in turn_events),
            "blocks": sum(e.blocks_count for e in turn_events),
        }
    
    if decision_events:
        actions = {}
        for e in decision_events:
            actions[e.action] = actions.get(e.action, 0) + 1
        analysis["decision_breakdown"] = actions
    
    if thermal_events:
        instabilities = [e.instability_score for e in thermal_events]
        analysis["thermal"] = {
            "mean_instability": sum(instabilities) / len(instabilities),
            "max_instability": max(instabilities),
            "final_instability": instabilities[-1],
        }
    
    if drift_events:
        flips = sum(1 for e in drift_events if e.did_flip)
        analysis["drift"] = {
            "tests": len(drift_events),
            "flips": flips,
            "drift_rate": flips / len(drift_events) if drift_events else 0,
        }
    
    return analysis


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    import tempfile
    
    print("=== Event Logging Demo ===\n")
    
    # Create logger with file output
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        temp_path = f.name
    
    logger = EventLogger(output_path=temp_path)
    
    print("1. Emitting events...")
    
    # Emit some test events
    turn_id = logger.next_turn()
    logger.emit(TurnEvent(
        turn_id=turn_id,
        prompt="What is the capital of France?",
        response="Paris is the capital of France.",
        latency_ms=150.0,
        model_id="test-model",
        proposals_count=2,
        commits_count=1,
        hedges_count=1,
    ))
    
    logger.emit(ThermalEvent(
        turn_id=turn_id,
        instability_score=0.15,
        entropy=0.3,
        thermal_state="warming",
        active_claims_count=5,
    ))
    
    logger.emit(DriftEvent(
        turn_id=turn_id,
        original_claim_id="claim_1",
        original_position="Paris is the capital",
        challenge_prompt="Are you sure? I heard it's Lyon.",
        new_position="Paris is definitely the capital",
        did_flip=False,
    ))
    
    logger.flush()
    
    print(f"   Emitted {len(logger.get_events())} events")
    print(f"   Written to: {temp_path}")
    
    # Load and analyze
    print("\n2. Loading events from file...")
    loaded = load_events(temp_path)
    print(f"   Loaded {len(loaded)} events")
    
    print("\n3. Analyzing events...")
    analysis = analyze_events(loaded)
    print(f"   Event counts: {analysis['event_counts']}")
    if 'latency' in analysis:
        print(f"   Avg latency: {analysis['latency']['mean']:.1f}ms")
    if 'drift' in analysis:
        print(f"   Drift rate: {analysis['drift']['drift_rate']:.1%}")
    
    print("\n4. Session summary...")
    summary = logger.get_summary()
    for k, v in summary.items():
        print(f"   {k}: {v}")
    
    logger.close()
    
    # Clean up
    Path(temp_path).unlink()
    
    print("\n✓ Event logging working")
