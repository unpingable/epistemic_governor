"""
Telemetry Metrics for Autopilot Early Warning

Computes operational metrics from ledger/envelope state.
Emits warning events (declarative, no policy decisions).

Key metrics:
1. Claim-Evidence Accretion Ratio (CAR)
2. Unresolved Claim Half-Life
3. Constraint Slack Margin
4. Soft-Constraint Drop Frequency
5. Oscillation Detector

Design principle: Warnings are telemetry, not influence.
One line, no verbs, ignorable without penalty.

Usage:
    from epistemic_governor.telemetry import (
        TelemetryComputer,
        WarningEvent,
        WarningType,
    )
    
    telemetry = TelemetryComputer()
    telemetry.update(ledger_snapshot, envelope_snapshot)
    
    warnings = telemetry.get_warnings()
    for w in warnings:
        print(f"{w.warning_type.name}: {w.metric_name}={w.value:.2f}")
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Deque
from collections import deque
from enum import Enum, auto
from datetime import datetime
import math


# =============================================================================
# Warning Types
# =============================================================================

class WarningType(Enum):
    """Types of telemetry warnings."""
    PROVENANCE_SLACK = auto()
    CAR_ACCELERATING = auto()
    HALF_LIFE_INCREASING = auto()
    DROP_CLUSTERING = auto()
    OSCILLATION_DETECTED = auto()
    SCOPE_MARGIN = auto()
    UNCERTAINTY_BUDGET = auto()


@dataclass
class WarningEvent:
    """
    A declarative warning event.
    
    NOT advice. NOT a suggestion. Just telemetry.
    Format: metric_name=value (slope if applicable)
    """
    warning_type: WarningType
    metric_name: str
    value: float
    threshold: float
    slope: Optional[float] = None  # Rate of change
    window: int = 1  # Measurement window
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_event_string(self) -> str:
        """
        Format as declarative event string.
        
        Examples:
        - WARN_PROVENANCE_SLACK(slack=0.22, slope=-0.06/window)
        - WARN_CAR_ACCEL(rate=1.5, accel=0.3)
        """
        parts = [f"{self.metric_name}={self.value:.3f}"]
        if self.slope is not None:
            parts.append(f"slope={self.slope:+.3f}/window")
        return f"WARN_{self.warning_type.name}({', '.join(parts)})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "warning_type": self.warning_type.name,
            "metric_name": self.metric_name,
            "value": self.value,
            "threshold": self.threshold,
            "slope": self.slope,
            "window": self.window,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Metric Snapshots (Input)
# =============================================================================

@dataclass
class LedgerSnapshot:
    """Snapshot of claim ledger state for telemetry."""
    step: int
    total_claims: int
    active_claims: int
    ungrounded_claims: int
    grounded_claims: int
    peer_asserted_claims: int
    
    # Evidence
    total_evidence: int
    evidence_this_window: int
    
    # Claim ages
    claim_ages: List[int] = field(default_factory=list)  # Steps since creation for unresolved
    
    # Recent activity
    new_claims_this_window: int = 0
    promotions_this_window: int = 0
    retractions_this_window: int = 0


@dataclass
class EnvelopeSnapshot:
    """Snapshot of envelope state for telemetry."""
    step: int
    
    # Slack margins (0-1, higher = more room)
    provenance_slack: float = 1.0
    scope_slack: float = 1.0
    contradiction_slack: float = 1.0
    uncertainty_slack: float = 1.0
    
    # Strain
    factual_strain: float = 0.0
    coherence_strain: float = 0.0
    
    # Violations
    near_violations: int = 0
    actual_violations: int = 0


@dataclass
class AutopilotSnapshot:
    """Snapshot of autopilot state for telemetry."""
    step: int
    
    # Soft constraint drops
    drops_this_window: int = 0
    drop_types: List[str] = field(default_factory=list)
    
    # Escalation history
    current_level: int = 0
    level_changes_this_window: int = 0


# =============================================================================
# Metric Computer
# =============================================================================

@dataclass
class MetricWindow:
    """Rolling window for metric computation."""
    size: int = 10
    values: Deque[float] = field(default_factory=deque)
    
    def __post_init__(self):
        # Fix: actually use the size parameter
        self.values = deque(maxlen=self.size)
    
    def add(self, value: float):
        self.values.append(value)
    
    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)
    
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
        
        if denominator == 0:
            return 0.0
        return numerator / denominator
    
    @property
    def is_increasing(self) -> bool:
        return self.slope > 0.01
    
    @property
    def is_decreasing(self) -> bool:
        return self.slope < -0.01


class TelemetryComputer:
    """
    Computes telemetry metrics from system snapshots.
    
    Emits warning events (declarative, no policy).
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        
        # Metric windows
        self.car_window = MetricWindow(size=window_size)
        self.half_life_window = MetricWindow(size=window_size)
        self.provenance_slack_window = MetricWindow(size=window_size)
        self.drop_frequency_window = MetricWindow(size=window_size)
        self.near_violation_window = MetricWindow(size=window_size)
        
        # Thresholds (configurable)
        self.thresholds = TelemetryThresholds()
        
        # Current state
        self.current_warnings: List[WarningEvent] = []
        self.step = 0
        self._last_resolution_rate = 0.0
    
    def update(
        self,
        ledger: LedgerSnapshot,
        envelope: EnvelopeSnapshot,
        autopilot: Optional[AutopilotSnapshot] = None,
    ) -> List[WarningEvent]:
        """
        Update telemetry with new snapshots.
        
        Returns list of warning events (may be empty).
        """
        self.step = ledger.step
        self.current_warnings = []
        
        # 1. Claim-Evidence Accretion Ratio (CAR)
        self._compute_car(ledger)
        
        # 2. Unresolved Claim Half-Life
        self._compute_half_life(ledger)
        
        # 3. Constraint Slack Margins
        self._compute_slack_margins(envelope)
        
        # 4. Soft-Constraint Drop Frequency
        if autopilot:
            self._compute_drop_frequency(autopilot)
        
        # 5. Oscillation Detection
        self._detect_oscillation(envelope)
        
        return self.current_warnings
    
    def _compute_car(self, ledger: LedgerSnapshot):
        """
        Compute Claim-Evidence Accretion Ratio.
        
        CAR = new_claims / (new_evidence + 1)
        Warn when slope is positive (claims outpacing evidence).
        """
        new_claims = ledger.new_claims_this_window
        new_evidence = ledger.evidence_this_window
        
        car = new_claims / (new_evidence + 1)
        self.car_window.add(car)
        
        # Warn on accelerating CAR
        if self.car_window.is_increasing and self.car_window.mean > self.thresholds.car_warn:
            self.current_warnings.append(WarningEvent(
                warning_type=WarningType.CAR_ACCELERATING,
                metric_name="car",
                value=self.car_window.mean,
                threshold=self.thresholds.car_warn,
                slope=self.car_window.slope,
            ))
    
    def _compute_half_life(self, ledger: LedgerSnapshot):
        """
        Compute Unresolved Claim Half-Life.
        
        Median age of unresolved claims.
        Warn when increasing (epistemic debt accumulating).
        """
        if not ledger.claim_ages:
            return
        
        sorted_ages = sorted(ledger.claim_ages)
        median_idx = len(sorted_ages) // 2
        half_life = sorted_ages[median_idx] if sorted_ages else 0
        
        self.half_life_window.add(float(half_life))
        
        # Warn on increasing half-life
        if self.half_life_window.is_increasing:
            if half_life > self.thresholds.half_life_warn:
                self.current_warnings.append(WarningEvent(
                    warning_type=WarningType.HALF_LIFE_INCREASING,
                    metric_name="half_life",
                    value=half_life,
                    threshold=self.thresholds.half_life_warn,
                    slope=self.half_life_window.slope,
                ))
    
    def _compute_slack_margins(self, envelope: EnvelopeSnapshot):
        """
        Compute Constraint Slack Margins.
        
        Warn when slack is low or decreasing.
        """
        self.provenance_slack_window.add(envelope.provenance_slack)
        
        # Warn on low provenance slack
        if envelope.provenance_slack < self.thresholds.slack_warn:
            self.current_warnings.append(WarningEvent(
                warning_type=WarningType.PROVENANCE_SLACK,
                metric_name="provenance_slack",
                value=envelope.provenance_slack,
                threshold=self.thresholds.slack_warn,
                slope=self.provenance_slack_window.slope,
            ))
        
        # Warn on low scope slack
        if envelope.scope_slack < self.thresholds.slack_warn:
            self.current_warnings.append(WarningEvent(
                warning_type=WarningType.SCOPE_MARGIN,
                metric_name="scope_slack",
                value=envelope.scope_slack,
                threshold=self.thresholds.slack_warn,
            ))
        
        # Warn on low uncertainty budget
        if envelope.uncertainty_slack < self.thresholds.slack_warn:
            self.current_warnings.append(WarningEvent(
                warning_type=WarningType.UNCERTAINTY_BUDGET,
                metric_name="uncertainty_slack",
                value=envelope.uncertainty_slack,
                threshold=self.thresholds.slack_warn,
            ))
    
    def _compute_drop_frequency(self, autopilot: AutopilotSnapshot):
        """
        Compute Soft-Constraint Drop Frequency.
        
        Warn on clustering (systemic conflict).
        """
        drops = autopilot.drops_this_window
        self.drop_frequency_window.add(float(drops))
        
        # Warn on drop clustering
        recent_drops = sum(list(self.drop_frequency_window.values)[-3:])
        if recent_drops >= self.thresholds.drop_cluster_warn:
            self.current_warnings.append(WarningEvent(
                warning_type=WarningType.DROP_CLUSTERING,
                metric_name="recent_drops",
                value=recent_drops,
                threshold=self.thresholds.drop_cluster_warn,
            ))
    
    def _detect_oscillation(self, envelope: EnvelopeSnapshot):
        """
        Detect oscillation near boundary using slack-based ringing.
        
        Better than counting near-violations: track slope changes in slack.
        """
        # Track slack slope
        if len(self.provenance_slack_window.values) < 4:
            return
        
        # Compute slope at different points
        values = list(self.provenance_slack_window.values)
        slopes = []
        for i in range(1, len(values)):
            slopes.append(values[i] - values[i-1])
        
        # Count sign changes in slope (ringing pattern)
        sign_changes = sum(
            1 for i in range(1, len(slopes))
            if (slopes[i] > 0) != (slopes[i-1] > 0)
        )
        
        # Compute amplitude of oscillation
        if len(values) >= 2:
            amplitude = max(values) - min(values)
        else:
            amplitude = 0
        
        # Warn on ringing pattern
        if sign_changes >= self.thresholds.oscillation_warn and amplitude > 0.1:
            self.current_warnings.append(WarningEvent(
                warning_type=WarningType.OSCILLATION_DETECTED,
                metric_name="slack_ringing",
                value=float(sign_changes),
                threshold=float(self.thresholds.oscillation_warn),
                slope=amplitude,  # Use slope field for amplitude
            ))
    
    def get_warnings(self) -> List[WarningEvent]:
        """Get current warning events."""
        return self.current_warnings
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        return {
            "step": self.step,
            "car": {
                "mean": self.car_window.mean,
                "slope": self.car_window.slope,
            },
            "half_life": {
                "mean": self.half_life_window.mean,
                "slope": self.half_life_window.slope,
            },
            "provenance_slack": {
                "current": list(self.provenance_slack_window.values)[-1] if self.provenance_slack_window.values else 1.0,
                "slope": self.provenance_slack_window.slope,
            },
            "drop_frequency": {
                "recent": self.drop_frequency_window.mean,
            },
            "resolution_rate": self._last_resolution_rate,
            "warnings_count": len(self.current_warnings),
        }
    
    def compute_resolution_rate(self, ledger: LedgerSnapshot) -> float:
        """
        Compute resolution rate: evidence additions / active claims.
        
        Higher = working off epistemic debt.
        Lower = accumulating debt.
        """
        if ledger.active_claims == 0:
            return 1.0  # No claims = nothing to resolve
        
        rate = ledger.evidence_this_window / max(ledger.active_claims, 1)
        self._last_resolution_rate = rate
        return rate


# =============================================================================
# Thresholds
# =============================================================================

@dataclass
class TelemetryThresholds:
    """Configurable thresholds for warning emission."""
    
    # CAR thresholds
    car_warn: float = 2.0  # Warn when CAR > 2 (2 claims per evidence)
    
    # Half-life thresholds (in steps)
    half_life_warn: int = 10  # Warn when median age > 10 steps
    
    # Slack thresholds (0-1)
    slack_warn: float = 0.25  # Warn when < 25% margin
    slack_critical: float = 0.10  # Critical when < 10%
    
    # Drop clustering
    drop_cluster_warn: int = 3  # Warn on 3+ drops in recent window
    
    # Oscillation
    oscillation_warn: int = 3  # Warn on 3+ direction changes


# =============================================================================
# Stabilization Criterion
# =============================================================================

@dataclass
class StabilizationState:
    """
    Tracks stabilization criterion for level descent.
    
    Stabilized = ALL of:
    - slack margins increasing for K windows
    - drop frequency decaying
    - CAR slope non-positive
    - no near-violations for T steps
    """
    required_stable_windows: int = 5
    required_clear_steps: int = 10
    
    # Tracking
    stable_window_count: int = 0
    clear_step_count: int = 0
    last_check_step: int = 0
    
    def check(
        self,
        telemetry: TelemetryComputer,
        envelope: EnvelopeSnapshot,
    ) -> tuple[bool, str]:
        """
        Check if system is stabilized.
        
        Returns (is_stable, reason).
        """
        reasons = []
        
        # 1. Slack margins increasing
        if telemetry.provenance_slack_window.is_decreasing:
            reasons.append("slack_decreasing")
        
        # 2. Drop frequency decaying
        if telemetry.drop_frequency_window.is_increasing:
            reasons.append("drops_increasing")
        
        # 3. CAR slope non-positive
        if telemetry.car_window.slope > 0.01:
            reasons.append("car_increasing")
        
        # 4. No near-violations
        if envelope.near_violations > 0:
            self.clear_step_count = 0
            reasons.append("near_violations_present")
        else:
            self.clear_step_count += 1
        
        # Update stable window count
        if not reasons:
            self.stable_window_count += 1
        else:
            self.stable_window_count = 0
        
        # Check thresholds
        is_stable = (
            self.stable_window_count >= self.required_stable_windows and
            self.clear_step_count >= self.required_clear_steps
        )
        
        if is_stable:
            return True, "stabilized"
        else:
            return False, f"not_stable: {', '.join(reasons) if reasons else 'accumulating'}"


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=== Telemetry Demo ===\n")
    
    telemetry = TelemetryComputer(window_size=5)
    
    # Simulate several steps
    for step in range(10):
        # Create mock snapshots
        ledger = LedgerSnapshot(
            step=step,
            total_claims=step * 3,
            active_claims=step * 2,
            ungrounded_claims=step,  # Increasing ungrounded = bad
            grounded_claims=step,
            peer_asserted_claims=0,
            total_evidence=step // 2,  # Evidence lags claims
            evidence_this_window=1 if step % 2 == 0 else 0,
            new_claims_this_window=3,
            claim_ages=list(range(step + 1)),
        )
        
        envelope = EnvelopeSnapshot(
            step=step,
            provenance_slack=max(0.1, 1.0 - step * 0.1),  # Decreasing
            scope_slack=0.8,
            near_violations=1 if step > 5 else 0,
        )
        
        warnings = telemetry.update(ledger, envelope)
        
        if warnings:
            print(f"Step {step}:")
            for w in warnings:
                print(f"  {w.to_event_string()}")
    
    print("\n--- Final Metrics Summary ---")
    summary = telemetry.get_metrics_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
