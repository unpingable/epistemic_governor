"""
Trace Collection and Auto-Labeling

Records regime signals during real runs, then auto-labels based on events.

No human labeling required - labels come from observable events:
- Reset fired → UNSTABLE
- Tripwire triggered → UNSTABLE  
- High budget pressure sustained → DUCTILE
- Elevated signals that recovered → WARM
- Stable low signals → ELASTIC

Usage:
    # Record traces during real usage
    from epistemic_governor.trace_collector import TraceCollector
    
    collector = TraceCollector("traces/run_001.jsonl")
    
    # In your processing loop:
    collector.record(signals, events={"reset": False, "tripwire": None})
    
    collector.close()
    
    # Then auto-label and tune:
    python -m epistemic_governor.trace_collector label traces/run_001.jsonl
    python -m epistemic_governor.validation_harness tune --dataset traces/run_001.labeled.jsonl
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from epistemic_governor.control.regime import RegimeSignals, OperationalRegime


@dataclass
class TraceEvent:
    """A single timestep in a trace."""
    timestamp: str
    turn: int
    signals: Dict[str, float]
    events: Dict[str, Any]  # reset, tripwire, budget_exceeded, etc.
    
    # Auto-assigned later
    label: Optional[str] = None
    label_confidence: Optional[str] = None  # "high", "medium", "low", "unknown"


class TraceCollector:
    """
    Collects regime signals and events during real runs.
    
    Outputs JSONL trace files for later analysis and tuning.
    """
    
    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.output_path, "w")
        self.turn = 0
        
        # Write header
        header = {
            "type": "header",
            "version": "1.0",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        self.file.write(json.dumps(header) + "\n")
    
    def record(
        self,
        signals: RegimeSignals,
        events: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a single timestep."""
        self.turn += 1
        
        entry = TraceEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            turn=self.turn,
            signals=signals.to_dict(),
            events=events or {},
        )
        
        self.file.write(json.dumps(asdict(entry)) + "\n")
        self.file.flush()
    
    def close(self) -> None:
        """Close the trace file."""
        footer = {
            "type": "footer",
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "total_turns": self.turn,
        }
        self.file.write(json.dumps(footer) + "\n")
        self.file.close()


@dataclass 
class LabelingConfig:
    """
    Configuration for auto-labeling.
    
    IMPORTANT: These thresholds must be INDEPENDENT of RegimeThresholds.
    If they overlap, you're training on your own definition (circular).
    
    Safe event sources for UNSTABLE:
    - Reset fired (if reset is triggered by watchdog/timeout, not regime)
    - Tripwire fired (if tripwire uses hardcoded bounds, not tuned thresholds)
    - External constraint violations
    - Contradiction ledger blowup
    
    Risky: using the same tool_gain threshold here that you're tuning.
    """
    # UNSTABLE: event-based (these should be INDEPENDENT of tuned thresholds)
    # Use hardcoded safety bounds, not the regime classifier thresholds
    unstable_on_reset: bool = True
    unstable_on_tripwire: bool = True
    unstable_tool_gain_threshold: float = 1.2  # HARDCODED safety rail, not tuned
    unstable_budget_threshold: float = 0.98    # HARDCODED safety rail, not tuned
    
    # DUCTILE: sustained stress without events
    ductile_budget_threshold: float = 0.75
    ductile_hysteresis_threshold: float = 0.55
    ductile_persistence_turns: int = 3
    
    # WARM: elevated but recoverable
    warm_hysteresis_threshold: float = 0.25
    warm_budget_threshold: float = 0.55
    warm_persistence_turns: int = 2
    
    # ELASTIC: everything low
    elastic_hysteresis_max: float = 0.15
    elastic_budget_max: float = 0.4
    
    # Version for provenance
    version: str = "1.0"


def compute_trace_hash(trace_path: Path) -> str:
    """Compute SHA256 hash of trace file for provenance."""
    import hashlib
    h = hashlib.sha256()
    with open(trace_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]  # First 16 chars is enough


def load_trace(trace_path: Path) -> List[TraceEvent]:
    """Load a trace file, returning list of events."""
    events = []
    with open(trace_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("type") in ("header", "footer"):
                continue
            events.append(TraceEvent(**data))
    return events


def auto_label_trace(
    events: List[TraceEvent],
    config: LabelingConfig = None,
) -> List[TraceEvent]:
    """
    Auto-label trace events based on signals and events.
    
    Labels are assigned with confidence levels:
    - high: clear event-based signal (reset, tripwire)
    - medium: sustained signal pattern
    - low: single-turn signal
    - unknown: ambiguous
    """
    config = config or LabelingConfig()
    labeled = []
    
    # Sliding window for persistence checks
    window_size = max(
        config.ductile_persistence_turns,
        config.warm_persistence_turns,
    )
    
    for i, event in enumerate(events):
        signals = event.signals
        evt = event.events
        
        label = None
        confidence = "unknown"
        
        # === UNSTABLE: event-based (high confidence) ===
        if evt.get("reset") or evt.get("tripwire"):
            label = "UNSTABLE"
            confidence = "high"
        elif signals.get("tool_gain_estimate", 0) >= config.unstable_tool_gain_threshold:
            label = "UNSTABLE"
            confidence = "high"
        elif signals.get("budget_pressure", 0) >= config.unstable_budget_threshold:
            label = "UNSTABLE"
            confidence = "medium"
        
        # === DUCTILE: sustained high stress ===
        elif i >= config.ductile_persistence_turns - 1:
            window = events[i - config.ductile_persistence_turns + 1 : i + 1]
            if all(
                e.signals.get("budget_pressure", 0) >= config.ductile_budget_threshold
                or e.signals.get("hysteresis_magnitude", 0) >= config.ductile_hysteresis_threshold
                for e in window
            ):
                label = "DUCTILE"
                confidence = "medium"
        
        # === WARM: elevated but not sustained ===
        if label is None:
            hysteresis = signals.get("hysteresis_magnitude", 0)
            budget = signals.get("budget_pressure", 0)
            
            if (hysteresis >= config.warm_hysteresis_threshold or 
                budget >= config.warm_budget_threshold):
                # Check if it's sustained enough
                if i >= config.warm_persistence_turns - 1:
                    window = events[i - config.warm_persistence_turns + 1 : i + 1]
                    elevated_count = sum(
                        1 for e in window
                        if (e.signals.get("hysteresis_magnitude", 0) >= config.warm_hysteresis_threshold
                            or e.signals.get("budget_pressure", 0) >= config.warm_budget_threshold)
                    )
                    if elevated_count >= config.warm_persistence_turns:
                        label = "WARM"
                        confidence = "medium"
                    else:
                        label = "WARM"
                        confidence = "low"
                else:
                    label = "WARM"
                    confidence = "low"
        
        # === ELASTIC: everything low ===
        if label is None:
            hysteresis = signals.get("hysteresis_magnitude", 0)
            budget = signals.get("budget_pressure", 0)
            
            if (hysteresis <= config.elastic_hysteresis_max and
                budget <= config.elastic_budget_max):
                label = "ELASTIC"
                confidence = "medium"
            else:
                label = "ELASTIC"
                confidence = "low"
        
        event.label = label
        event.label_confidence = confidence
        labeled.append(event)
    
    return labeled


def save_labeled_trace(
    events: List[TraceEvent], 
    output_path: Path,
    source_path: Optional[Path] = None,
    config: Optional[LabelingConfig] = None,
) -> None:
    """Save labeled trace to file with provenance."""
    config = config or LabelingConfig()
    
    with open(output_path, "w") as f:
        header = {
            "type": "header",
            "version": "1.0",
            "labeled_at": datetime.now(timezone.utc).isoformat(),
            "labeler_version": config.version,
            "source_hash": compute_trace_hash(source_path) if source_path else None,
            "labeling_config": {
                "unstable_tool_gain_threshold": config.unstable_tool_gain_threshold,
                "unstable_budget_threshold": config.unstable_budget_threshold,
                "ductile_persistence_turns": config.ductile_persistence_turns,
                "warm_persistence_turns": config.warm_persistence_turns,
            },
        }
        f.write(json.dumps(header) + "\n")
        
        for event in events:
            f.write(json.dumps(asdict(event)) + "\n")
        
        # Stats with leakage check info
        labels = [e.label for e in events]
        confidences = [e.label_confidence for e in events]
        
        high_conf = [e for e in events if e.label_confidence == "high"]
        medium_conf = [e for e in events if e.label_confidence == "medium"]
        low_conf = [e for e in events if e.label_confidence == "low"]
        
        footer = {
            "type": "footer",
            "total_events": len(events),
            "label_distribution": {
                label: labels.count(label)
                for label in set(labels) if label
            },
            "confidence_distribution": {
                conf: confidences.count(conf)
                for conf in ["high", "medium", "low", "unknown"]
                if confidences.count(conf) > 0
            },
            "trainable_samples": {
                "high_only": len(high_conf),
                "high_medium": len(high_conf) + len(medium_conf),
                "all": len(events),
            },
            "leakage_warning": len(low_conf) > len(high_conf) + len(medium_conf),
        }
        f.write(json.dumps(footer) + "\n")


def trace_to_training_data(
    events: List[TraceEvent],
    min_confidence: str = "medium",
) -> List[Tuple[RegimeSignals, OperationalRegime, str]]:
    """
    Convert labeled trace to training data for the tuner.
    
    Filters by confidence level.
    """
    confidence_order = ["high", "medium", "low", "unknown"]
    min_idx = confidence_order.index(min_confidence)
    
    data = []
    for event in events:
        if event.label is None:
            continue
        
        conf_idx = confidence_order.index(event.label_confidence or "unknown")
        if conf_idx > min_idx:
            continue  # Too low confidence
        
        signals = RegimeSignals(
            hysteresis_magnitude=event.signals.get("hysteresis_magnitude", 0),
            relaxation_time_seconds=event.signals.get("relaxation_time_seconds", 0),
            tool_gain_estimate=event.signals.get("tool_gain_estimate", 0),
            anisotropy_score=event.signals.get("anisotropy_score", 0),
            provenance_deficit_rate=event.signals.get("provenance_deficit_rate", 0),
            budget_pressure=event.signals.get("budget_pressure", 0),
            contradiction_open_rate=event.signals.get("contradiction_open_rate", 0),
            contradiction_close_rate=event.signals.get("contradiction_close_rate", 0),
        )
        
        regime = OperationalRegime[event.label]
        source = f"trace:{event.turn}"
        
        data.append((signals, regime, source))
    
    return data


def label_trace_file(trace_path: Path, output_path: Optional[Path] = None) -> Path:
    """Label a trace file and save results."""
    config = LabelingConfig()
    events = load_trace(trace_path)
    labeled = auto_label_trace(events, config)
    
    output = output_path or trace_path.with_suffix(".labeled.jsonl")
    save_labeled_trace(labeled, output, source_path=trace_path, config=config)
    
    return output


def print_trace_stats(trace_path: Path) -> None:
    """Print stats about a trace file."""
    events = load_trace(trace_path)
    
    print(f"\nTrace: {trace_path}")
    print(f"Events: {len(events)}")
    
    if events:
        labels = [e.label for e in events if e.label]
        if labels:
            print(f"\nLabel distribution:")
            for label in set(labels):
                count = labels.count(label)
                pct = count / len(labels) * 100
                print(f"  {label}: {count} ({pct:.1f}%)")
        
        confidences = [e.label_confidence for e in events if e.label_confidence]
        if confidences:
            print(f"\nConfidence distribution:")
            for conf in ["high", "medium", "low", "unknown"]:
                count = confidences.count(conf)
                if count:
                    pct = count / len(confidences) * 100
                    print(f"  {conf}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Trace collection and labeling")
    parser.add_argument("command", choices=["label", "stats", "demo"],
                       help="Command to run")
    parser.add_argument("trace", nargs="?", type=Path,
                       help="Trace file path")
    parser.add_argument("--output", "-o", type=Path,
                       help="Output path for labeled trace")
    parser.add_argument("--min-confidence", default="medium",
                       choices=["high", "medium", "low", "unknown"],
                       help="Minimum confidence for training data")
    
    args = parser.parse_args()
    
    if args.command == "demo":
        # Generate a demo trace
        print("Generating demo trace...")
        
        collector = TraceCollector("traces/demo.jsonl")
        
        # Simulate some turns
        import random
        rng = random.Random(42)
        
        for i in range(50):
            # Vary signals
            base_stress = i / 50  # Gradually increase
            noise = rng.uniform(-0.1, 0.1)
            
            signals = RegimeSignals(
                hysteresis_magnitude=min(0.8, base_stress * 0.5 + noise),
                relaxation_time_seconds=base_stress * 10,
                tool_gain_estimate=0.3 + base_stress * 0.5,
                budget_pressure=base_stress * 0.8 + noise,
            )
            
            # Simulate events
            events = {
                "reset": i == 45,  # Reset near end
                "tripwire": "cascade" if i == 45 else None,
            }
            
            collector.record(signals, events)
        
        collector.close()
        print(f"Saved demo trace to: traces/demo.jsonl")
        
        # Label it
        print("\nLabeling trace...")
        output = label_trace_file(Path("traces/demo.jsonl"))
        print(f"Saved labeled trace to: {output}")
        
        # Print stats
        print_trace_stats(output)
        
    elif args.command == "label":
        if not args.trace:
            parser.error("trace file required for label command")
        output = label_trace_file(args.trace, args.output)
        print(f"Labeled trace saved to: {output}")
        print_trace_stats(output)
        
    elif args.command == "stats":
        if not args.trace:
            parser.error("trace file required for stats command")
        print_trace_stats(args.trace)
