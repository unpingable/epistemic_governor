"""
Phase Diagnostics (BLI-DIAG-0.1)

Observability layer for measuring interior time regimes.

Core observables:
- ρ_S: state mutation rate
- |C_open|: contradiction load
- τ_resolve: resolution lifetime distribution
- budget burn rates
- witness frequency

This is MEASUREMENT ONLY. No optimization. No behavior changes.
We're learning what kind of lattice we built.

Regime detection:
- Healthy lattice: constraint + continuity without paralysis
- Chatbot-with-ceremony: false interiority (ρ_S ≈ 0)
- Glass/ossification: barriers too high, repair too expensive
- Sloppy fluid: laundering contradictions
- Budget starvation: rate limits too tight
- Extraction collapse: claim layer failure
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from collections import deque
import hashlib
import json
import statistics


# =============================================================================
# Event Envelope (Every Turn Gets This)
# =============================================================================

@dataclass
class DiagnosticEvent:
    """
    Complete diagnostic record for one governor turn.
    
    This is the canonical event log format.
    Append-only. Queryable. Causally linkable.
    """
    # Identity
    run_id: str
    turn_id: int
    timestamp: datetime
    suite: str  # prod|harness|integration|fuzz|replay
    
    # Input identity (for hysteresis detection)
    prompt_hash: str
    state_hash_before: str
    state_hash_after: str
    state_view_hash: str  # What model saw
    
    # Model info
    model_id: str = "default"
    seed: Optional[int] = None
    
    # Latency
    latency_ms_total: float = 0.0
    latency_ms_model: float = 0.0
    latency_ms_governor: float = 0.0
    latency_ms_storage: float = 0.0
    
    # Extraction
    extract_status: str = "ok"  # ok|fail
    extract_fail_reason: Optional[str] = None
    claims_count_total: int = 0
    claims_count_by_domain: Dict[str, int] = field(default_factory=dict)
    claims_count_by_provenance: Dict[str, int] = field(default_factory=dict)
    
    # Verdict
    verdict: str = "OK"  # OK|WARN|BLOCK
    blocked_by_invariant: List[str] = field(default_factory=list)
    warn_reasons: List[str] = field(default_factory=list)
    
    # Witness
    witness_emitted: bool = False
    witness_refs_ledger_ids: List[str] = field(default_factory=list)
    witness_refs_contradiction_ids: List[str] = field(default_factory=list)
    
    # Contradiction dynamics
    c_open_before: int = 0
    c_open_after: int = 0
    c_opened_count: int = 0
    c_closed_count: int = 0
    c_frozen_count: int = 0
    c_open_by_domain_after: Dict[str, int] = field(default_factory=dict)
    c_severity_hist_after: Dict[str, int] = field(default_factory=dict)
    contradiction_ids_opened: List[str] = field(default_factory=list)
    contradiction_ids_closed: List[str] = field(default_factory=list)
    tau_resolve_ms_closed: List[float] = field(default_factory=list)
    
    # Ledger dynamics
    ledger_entries_appended: int = 0
    ledger_entry_types_appended: Dict[str, int] = field(default_factory=dict)
    ledger_bytes_appended: int = 0
    ledger_depth: int = 0
    tombstones_appended: int = 0
    
    # Budgets
    budget_window_id: str = ""
    budget_remaining_before: Dict[str, float] = field(default_factory=dict)
    budget_remaining_after: Dict[str, float] = field(default_factory=dict)
    budget_spent_this_turn: Dict[str, float] = field(default_factory=dict)
    budget_exhaustion: Dict[str, bool] = field(default_factory=dict)
    budget_refill_events: int = 0
    
    # Energy
    E_state_before: float = 0.0
    E_state_after: float = 0.0
    E_components_before: Dict[str, float] = field(default_factory=dict)
    E_components_after: Dict[str, float] = field(default_factory=dict)
    
    # Derived metrics
    rho_S_flag: bool = False  # Did state mutate meaningfully?
    delta_state_bytes: int = 0
    delta_E: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSONL output."""
        d = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                d[key] = value.isoformat()
            elif isinstance(value, Enum):
                d[key] = value.name
            else:
                d[key] = value
        return d
    
    def to_jsonl(self) -> str:
        """Serialize to JSONL line."""
        return json.dumps(self.to_dict(), sort_keys=True)


# =============================================================================
# Metrics Aggregator (Prometheus-style)
# =============================================================================

class MetricsAggregator:
    """
    Aggregates diagnostic events into Prometheus-style metrics.
    
    Maintains:
    - Counters (monotonic)
    - Gauges (point-in-time)
    - Histograms (distributions)
    """
    
    def __init__(self):
        # Counters
        self.turns_total: int = 0
        self.verdict_total: Dict[str, int] = {"OK": 0, "WARN": 0, "BLOCK": 0}
        self.extract_total: Dict[str, int] = {"ok": 0, "fail": 0}
        self.witness_total: int = 0
        
        # State mutation counters
        self.state_mutation_total: Dict[str, int] = {
            "contradiction_opened": 0,
            "contradiction_closed": 0,
            "ledger_commit": 0,
            "budget_crossing": 0,
            "other": 0,
        }
        
        # Ledger counters
        self.ledger_entries_total: Dict[str, int] = {}
        self.ledger_bytes_total: int = 0
        
        # Contradiction counters
        self.contradictions_opened_total: int = 0
        self.contradictions_closed_total: int = 0
        self.contradictions_frozen_total: int = 0
        
        # Budget counters
        self.budget_spent_total: Dict[str, float] = {}
        self.budget_exhausted_total: Dict[str, int] = {}
        
        # Block/warn reason counters
        self.block_reason_total: Dict[str, int] = {}
        self.warn_reason_total: Dict[str, int] = {}
        
        # Quality alarms
        self.witness_missing_refs_total: int = 0
        self.contradiction_closed_without_evidence_total: int = 0
        
        # Gauges
        self.contradictions_open: int = 0
        self.contradictions_open_by_domain: Dict[str, int] = {}
        self.contradictions_open_by_severity: Dict[str, int] = {}
        self.budget_remaining: Dict[str, float] = {}
        self.energy_state: float = 0.0
        self.energy_components: Dict[str, float] = {}
        
        # Histograms (store recent values for percentile calculation)
        self._tau_resolve_ms: deque = deque(maxlen=1000)
        self._latency_ms_total: deque = deque(maxlen=1000)
        self._latency_ms_governor: deque = deque(maxlen=1000)
        self._delta_energy: deque = deque(maxlen=1000)
        self._budget_spend_per_turn: Dict[str, deque] = {}
        
        # Hysteresis support
        self.state_hash_changes_total: int = 0
    
    def record(self, event: DiagnosticEvent) -> None:
        """Record a diagnostic event."""
        self.turns_total += 1
        
        # Verdict
        self.verdict_total[event.verdict] = self.verdict_total.get(event.verdict, 0) + 1
        
        # Extraction
        self.extract_total[event.extract_status] = self.extract_total.get(event.extract_status, 0) + 1
        
        # Witness
        if event.witness_emitted:
            self.witness_total += 1
            if not event.witness_refs_ledger_ids and not event.witness_refs_contradiction_ids:
                self.witness_missing_refs_total += 1
        
        # State mutations
        if event.c_opened_count > 0:
            self.state_mutation_total["contradiction_opened"] += event.c_opened_count
            self.contradictions_opened_total += event.c_opened_count
        if event.c_closed_count > 0:
            self.state_mutation_total["contradiction_closed"] += event.c_closed_count
            self.contradictions_closed_total += event.c_closed_count
        if event.c_frozen_count > 0:
            self.contradictions_frozen_total += event.c_frozen_count
        if event.ledger_entries_appended > 0:
            self.state_mutation_total["ledger_commit"] += 1
        
        # Ledger
        for entry_type, count in event.ledger_entry_types_appended.items():
            self.ledger_entries_total[entry_type] = self.ledger_entries_total.get(entry_type, 0) + count
        self.ledger_bytes_total += event.ledger_bytes_appended
        
        # Budgets
        for bucket, spent in event.budget_spent_this_turn.items():
            self.budget_spent_total[bucket] = self.budget_spent_total.get(bucket, 0) + spent
            if bucket not in self._budget_spend_per_turn:
                self._budget_spend_per_turn[bucket] = deque(maxlen=1000)
            self._budget_spend_per_turn[bucket].append(spent)
        
        for bucket, exhausted in event.budget_exhaustion.items():
            if exhausted:
                self.budget_exhausted_total[bucket] = self.budget_exhausted_total.get(bucket, 0) + 1
        
        # Block/warn reasons
        for invariant in event.blocked_by_invariant:
            self.block_reason_total[invariant] = self.block_reason_total.get(invariant, 0) + 1
        for reason in event.warn_reasons:
            self.warn_reason_total[reason] = self.warn_reason_total.get(reason, 0) + 1
        
        # Gauges (point-in-time)
        self.contradictions_open = event.c_open_after
        self.contradictions_open_by_domain = dict(event.c_open_by_domain_after)
        self.contradictions_open_by_severity = dict(event.c_severity_hist_after)
        self.budget_remaining = dict(event.budget_remaining_after)
        self.energy_state = event.E_state_after
        self.energy_components = dict(event.E_components_after)
        
        # Histograms
        for tau in event.tau_resolve_ms_closed:
            self._tau_resolve_ms.append(tau)
        self._latency_ms_total.append(event.latency_ms_total)
        self._latency_ms_governor.append(event.latency_ms_governor)
        self._delta_energy.append(event.delta_E)
        
        # Hysteresis
        if event.state_hash_before != event.state_hash_after:
            self.state_hash_changes_total += 1
    
    def get_histogram_percentiles(self, name: str, percentiles: List[float] = [0.5, 0.9, 0.95, 0.99]) -> Dict[str, float]:
        """Get percentiles for a histogram."""
        data_map = {
            "tau_resolve_ms": self._tau_resolve_ms,
            "latency_ms_total": self._latency_ms_total,
            "latency_ms_governor": self._latency_ms_governor,
            "delta_energy": self._delta_energy,
        }
        
        data = list(data_map.get(name, []))
        if not data:
            return {f"p{int(p*100)}": 0.0 for p in percentiles}
        
        data.sort()
        result = {}
        for p in percentiles:
            idx = int(len(data) * p)
            idx = min(idx, len(data) - 1)
            result[f"p{int(p*100)}"] = data[idx]
        return result
    
    def get_rho_S(self, window_turns: int = 100) -> float:
        """
        Compute ρ_S (state mutation rate).
        
        Returns fraction of recent turns that had meaningful state mutation.
        """
        if self.turns_total == 0:
            return 0.0
        
        # Approximate from counters
        mutations = sum(self.state_mutation_total.values())
        return min(1.0, mutations / self.turns_total)
    
    def summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "turns_total": self.turns_total,
            "verdict_distribution": self.verdict_total,
            "extract_success_rate": (
                self.extract_total["ok"] / self.turns_total 
                if self.turns_total > 0 else 1.0
            ),
            "witness_rate": self.witness_total / self.turns_total if self.turns_total > 0 else 0.0,
            "contradictions_open": self.contradictions_open,
            "contradictions_opened_total": self.contradictions_opened_total,
            "contradictions_closed_total": self.contradictions_closed_total,
            "rho_S": self.get_rho_S(),
            "state_hash_changes": self.state_hash_changes_total,
            "tau_resolve_percentiles": self.get_histogram_percentiles("tau_resolve_ms"),
            "energy_state": self.energy_state,
            "block_reasons": self.block_reason_total,
            "budget_remaining": self.budget_remaining,
        }


# =============================================================================
# Regime Detection
# =============================================================================

class Regime(Enum):
    """Detected operating regime."""
    UNKNOWN = auto()
    HEALTHY_LATTICE = auto()      # Constraint + continuity without paralysis
    CHATBOT_CEREMONY = auto()     # False interiority (ρ_S ≈ 0)
    GLASS_OSSIFICATION = auto()   # Barriers too high, repair too expensive
    PERMEABLE_MEMBRANE = auto()         # Laundering contradictions
    BUDGET_STARVATION = auto()    # Rate limits too tight
    EXTRACTION_COLLAPSE = auto()  # Claim layer failure


@dataclass
class RegimeAnalysis:
    """Result of regime detection."""
    regime: Regime
    confidence: float  # 0-1
    indicators: Dict[str, Any]
    warnings: List[str]
    recommendations: List[str]


class RegimeDetector:
    """
    Detects operating regime from metrics.
    
    Uses windowed analysis to identify:
    - Healthy lattice (the target)
    - Various failure modes
    """
    
    def __init__(self):
        # Thresholds (pre-commit red lines)
        self.extraction_fail_threshold = 0.005  # 0.5%
        self.rho_S_min = 0.01  # Below this = ceremony
        self.rho_S_max = 0.95  # Above this = chaos
        self.c_open_trend_threshold = 0.1  # Slope indicating glass
        self.budget_block_ratio_threshold = 0.5  # Too many budget BLOCKs
        
        # History for trend detection
        self._c_open_history: deque = deque(maxlen=100)
        self._block_history: deque = deque(maxlen=100)
    
    def analyze(self, metrics: MetricsAggregator, events: List[DiagnosticEvent] = None) -> RegimeAnalysis:
        """Analyze metrics to detect regime."""
        warnings = []
        indicators = {}
        
        # Compute key indicators
        rho_S = metrics.get_rho_S()
        c_open = metrics.contradictions_open
        extract_fail_rate = 1.0 - (metrics.extract_total["ok"] / metrics.turns_total if metrics.turns_total > 0 else 1.0)
        
        # Track history
        self._c_open_history.append(c_open)
        
        # Block reason analysis
        total_blocks = metrics.verdict_total.get("BLOCK", 0)
        budget_blocks = sum(
            count for reason, count in metrics.block_reason_total.items()
            if "budget" in reason.lower() or "exhausted" in reason.lower()
        )
        budget_block_ratio = budget_blocks / total_blocks if total_blocks > 0 else 0
        
        indicators["rho_S"] = rho_S
        indicators["c_open"] = c_open
        indicators["extract_fail_rate"] = extract_fail_rate
        indicators["budget_block_ratio"] = budget_block_ratio
        indicators["witness_missing_refs_rate"] = (
            metrics.witness_missing_refs_total / metrics.witness_total 
            if metrics.witness_total > 0 else 0
        )
        
        # Detect regime
        regime = Regime.UNKNOWN
        confidence = 0.5
        recommendations = []
        
        # Check extraction collapse first (fundamental failure)
        if extract_fail_rate > self.extraction_fail_threshold:
            regime = Regime.EXTRACTION_COLLAPSE
            confidence = min(1.0, extract_fail_rate / 0.1)
            warnings.append(f"Extraction failure rate {extract_fail_rate:.1%} exceeds threshold")
            recommendations.append("Fix claim extraction layer before anything else")
        
        # Check ceremony (false interiority)
        elif rho_S < self.rho_S_min:
            regime = Regime.CHATBOT_CEREMONY
            confidence = 1.0 - (rho_S / self.rho_S_min)
            warnings.append(f"ρ_S = {rho_S:.3f} indicates no real state mutation")
            recommendations.append("Verify governor is actually binding outputs to state")
        
        # Check budget starvation
        elif budget_block_ratio > self.budget_block_ratio_threshold:
            regime = Regime.BUDGET_STARVATION
            confidence = min(1.0, budget_block_ratio / 0.8)
            warnings.append(f"{budget_block_ratio:.1%} of BLOCKs are budget-related")
            recommendations.append("Increase budgets or reduce costs")
        
        # Check glass (trending c_open)
        elif len(self._c_open_history) >= 10:
            c_open_trend = self._compute_trend(list(self._c_open_history))
            indicators["c_open_trend"] = c_open_trend
            
            if c_open_trend > self.c_open_trend_threshold:
                regime = Regime.GLASS_OSSIFICATION
                confidence = min(1.0, c_open_trend / 0.3)
                warnings.append(f"Contradiction load trending upward (slope={c_open_trend:.3f})")
                recommendations.append("Reduce repair friction or increase resolution budget")
        
        # Check sloppy fluid
        if regime == Regime.UNKNOWN:
            # Closures happening but evidence weak
            if metrics.contradiction_closed_without_evidence_total > 0:
                regime = Regime.PERMEABLE_MEMBRANE
                confidence = 0.9
                warnings.append("Contradictions closing without evidence - integrity breach")
                recommendations.append("Enforce evidence requirement for contradiction closure")
            elif indicators["witness_missing_refs_rate"] > 0.3:
                regime = Regime.PERMEABLE_MEMBRANE
                confidence = 0.6
                warnings.append("High rate of witnesses without state references")
                recommendations.append("Verify witnesses are grounded in actual state")
        
        # Default to healthy if nothing bad detected
        if regime == Regime.UNKNOWN:
            if rho_S > self.rho_S_min and c_open < 100:  # Reasonable bounds
                regime = Regime.HEALTHY_LATTICE
                confidence = 0.7
            else:
                regime = Regime.UNKNOWN
                confidence = 0.3
        
        return RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            indicators=indicators,
            warnings=warnings,
            recommendations=recommendations,
        )
    
    def _compute_trend(self, values: List[float]) -> float:
        """Compute simple linear trend (slope)."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator


# =============================================================================
# Energy Function
# =============================================================================

@dataclass
class EnergyComponents:
    """Components of the energy function E(S)."""
    contradiction_load: float = 0.0      # α|C_open|
    severity_sum: float = 0.0            # β∑severity
    budget_stress: float = 0.0           # γ(1/B_remaining)
    adapter_norm: float = 0.0            # λ|Δθ|² (if adapters enabled)
    
    def total(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.1, lambda_: float = 0.0) -> float:
        """Compute total energy."""
        return (
            alpha * self.contradiction_load +
            beta * self.severity_sum +
            gamma * self.budget_stress +
            lambda_ * self.adapter_norm
        )
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "contradiction_load": self.contradiction_load,
            "severity_sum": self.severity_sum,
            "budget_stress": self.budget_stress,
            "adapter_norm": self.adapter_norm,
        }


def compute_energy(
    c_open: int,
    severity_sum: float,
    budget_remaining: Dict[str, float],
    adapter_norm: float = 0.0,
) -> EnergyComponents:
    """
    Compute energy function E(S).
    
    This is MEASUREMENT ONLY. We're observing the landscape,
    not optimizing it.
    """
    # Budget stress: sum of 1/B for each budget bucket
    # Use 0.01 floor to avoid division by zero
    budget_stress = 0.0
    for bucket, remaining in budget_remaining.items():
        budget_stress += 1.0 / max(0.01, remaining)
    
    return EnergyComponents(
        contradiction_load=float(c_open),
        severity_sum=severity_sum,
        budget_stress=budget_stress,
        adapter_norm=adapter_norm,
    )


# =============================================================================
# Diagnostic Logger
# =============================================================================

class DiagnosticLogger:
    """
    Central logging for phase diagnostics.
    
    Collects events, computes metrics, detects regimes.
    """
    
    def __init__(self, run_id: str, suite: str = "harness"):
        self.run_id = run_id
        self.suite = suite
        self.turn_id = 0
        
        self.events: List[DiagnosticEvent] = []
        self.metrics = MetricsAggregator()
        self.regime_detector = RegimeDetector()
        
        # JSONL output buffer
        self._jsonl_buffer: List[str] = []
    
    def create_event(
        self,
        prompt_hash: str,
        state_hash_before: str,
        state_hash_after: str,
        state_view_hash: str,
    ) -> DiagnosticEvent:
        """Create a new diagnostic event."""
        self.turn_id += 1
        
        return DiagnosticEvent(
            run_id=self.run_id,
            turn_id=self.turn_id,
            timestamp=datetime.now(timezone.utc),
            suite=self.suite,
            prompt_hash=prompt_hash,
            state_hash_before=state_hash_before,
            state_hash_after=state_hash_after,
            state_view_hash=state_view_hash,
        )
    
    def record(self, event: DiagnosticEvent) -> None:
        """Record a diagnostic event."""
        # Compute derived metrics
        event.delta_E = event.E_state_after - event.E_state_before
        event.rho_S_flag = (
            event.c_opened_count > 0 or
            event.c_closed_count > 0 or
            any(t in ["ASSERT", "RETRACT", "SUPERSEDE", "RESOLUTION"] 
                for t in event.ledger_entry_types_appended.keys())
        )
        
        self.events.append(event)
        self.metrics.record(event)
        self._jsonl_buffer.append(event.to_jsonl())
    
    def get_regime(self) -> RegimeAnalysis:
        """Analyze current regime."""
        return self.regime_detector.analyze(self.metrics, self.events)
    
    def summary(self) -> Dict[str, Any]:
        """Get diagnostic summary."""
        regime = self.get_regime()
        
        return {
            "run_id": self.run_id,
            "suite": self.suite,
            "turns": self.turn_id,
            "regime": regime.regime.name,
            "regime_confidence": regime.confidence,
            "regime_warnings": regime.warnings,
            "metrics": self.metrics.summary(),
        }
    
    def get_jsonl(self) -> str:
        """Get all events as JSONL."""
        return "\n".join(self._jsonl_buffer)


# =============================================================================
# Tests
# =============================================================================

def test_diagnostic_event():
    """Test diagnostic event creation and serialization."""
    print("=== Test: Diagnostic Event ===\n")
    
    event = DiagnosticEvent(
        run_id="test_run",
        turn_id=1,
        timestamp=datetime.now(timezone.utc),
        suite="harness",
        prompt_hash="abc123",
        state_hash_before="def456",
        state_hash_after="ghi789",
        state_view_hash="jkl012",
        verdict="OK",
        c_open_before=2,
        c_open_after=3,
        c_opened_count=1,
    )
    
    # Test serialization
    d = event.to_dict()
    jsonl = event.to_jsonl()
    
    print(f"Event turn_id: {event.turn_id}")
    print(f"Serialized keys: {len(d)}")
    print(f"JSONL length: {len(jsonl)}")
    
    # Verify round-trip
    parsed = json.loads(jsonl)
    assert parsed["turn_id"] == 1
    assert parsed["verdict"] == "OK"
    
    print("✓ Diagnostic event working\n")
    return True


def test_metrics_aggregator():
    """Test metrics aggregation."""
    print("=== Test: Metrics Aggregator ===\n")
    
    metrics = MetricsAggregator()
    
    # Simulate some events
    for i in range(100):
        event = DiagnosticEvent(
            run_id="test",
            turn_id=i,
            timestamp=datetime.now(timezone.utc),
            suite="harness",
            prompt_hash=f"prompt_{i}",
            state_hash_before=f"before_{i}",
            state_hash_after=f"after_{i}" if i % 3 == 0 else f"before_{i}",
            state_view_hash="view",
            verdict="OK" if i % 5 != 0 else "WARN",
            c_open_before=i % 10,
            c_open_after=(i + 1) % 10,
            c_opened_count=1 if i % 10 == 0 else 0,
            c_closed_count=1 if i % 15 == 0 else 0,
            tau_resolve_ms_closed=[float(i * 100)] if i % 15 == 0 else [],
            ledger_entries_appended=1,
            ledger_entry_types_appended={"ASSERT": 1},
        )
        metrics.record(event)
    
    summary = metrics.summary()
    
    print(f"Turns total: {summary['turns_total']}")
    print(f"Verdict distribution: {summary['verdict_distribution']}")
    print(f"ρ_S: {summary['rho_S']:.3f}")
    print(f"Contradictions open: {summary['contradictions_open']}")
    print(f"State hash changes: {summary['state_hash_changes']}")
    print(f"τ_resolve percentiles: {summary['tau_resolve_percentiles']}")
    
    assert summary["turns_total"] == 100
    assert summary["rho_S"] > 0
    
    print("\n✓ Metrics aggregator working\n")
    return True


def test_regime_detection():
    """Test regime detection."""
    print("=== Test: Regime Detection ===\n")
    
    # Test healthy lattice
    metrics = MetricsAggregator()
    for i in range(50):
        event = DiagnosticEvent(
            run_id="test",
            turn_id=i,
            timestamp=datetime.now(timezone.utc),
            suite="harness",
            prompt_hash=f"p_{i}",
            state_hash_before=f"b_{i}",
            state_hash_after=f"a_{i}",
            state_view_hash="v",
            verdict="OK",
            c_open_after=5,  # Stable
            c_opened_count=1 if i % 5 == 0 else 0,
            c_closed_count=1 if i % 5 == 0 else 0,
        )
        metrics.record(event)
    
    detector = RegimeDetector()
    analysis = detector.analyze(metrics)
    
    print(f"Regime: {analysis.regime.name}")
    print(f"Confidence: {analysis.confidence:.2f}")
    print(f"Indicators: {analysis.indicators}")
    
    # Test ceremony (no mutations)
    metrics2 = MetricsAggregator()
    for i in range(50):
        event = DiagnosticEvent(
            run_id="test",
            turn_id=i,
            timestamp=datetime.now(timezone.utc),
            suite="harness",
            prompt_hash=f"p_{i}",
            state_hash_before="same",
            state_hash_after="same",  # No change!
            state_view_hash="v",
            verdict="OK",
            c_open_after=0,
            c_opened_count=0,
            c_closed_count=0,
        )
        metrics2.record(event)
    
    detector2 = RegimeDetector()
    analysis2 = detector2.analyze(metrics2)
    
    print(f"\nCeremony test:")
    print(f"Regime: {analysis2.regime.name}")
    print(f"ρ_S: {analysis2.indicators.get('rho_S', 0):.3f}")
    
    assert analysis2.regime == Regime.CHATBOT_CEREMONY
    
    print("\n✓ Regime detection working\n")
    return True


def test_energy_function():
    """Test energy function computation."""
    print("=== Test: Energy Function ===\n")
    
    # Low energy state
    low = compute_energy(
        c_open=2,
        severity_sum=3.0,
        budget_remaining={"append": 100.0, "resolve": 50.0},
    )
    
    # High energy state
    high = compute_energy(
        c_open=20,
        severity_sum=50.0,
        budget_remaining={"append": 5.0, "resolve": 1.0},
    )
    
    print(f"Low energy components: {low.to_dict()}")
    print(f"Low energy total: {low.total():.2f}")
    print(f"\nHigh energy components: {high.to_dict()}")
    print(f"High energy total: {high.total():.2f}")
    
    assert high.total() > low.total()
    
    print("\n✓ Energy function working\n")
    return True


def test_diagnostic_logger():
    """Test full diagnostic logger."""
    print("=== Test: Diagnostic Logger ===\n")
    
    logger = DiagnosticLogger(run_id="test_run", suite="harness")
    
    # Log some events
    for i in range(20):
        event = logger.create_event(
            prompt_hash=f"prompt_{i}",
            state_hash_before=f"before_{i}",
            state_hash_after=f"after_{i}",
            state_view_hash="view",
        )
        
        # Simulate some dynamics
        event.verdict = "OK" if i % 4 != 0 else "WARN"
        event.c_open_before = i % 5
        event.c_open_after = (i + 1) % 5
        event.c_opened_count = 1 if i % 5 == 0 else 0
        event.E_state_before = float(i)
        event.E_state_after = float(i + 0.5)
        
        logger.record(event)
    
    summary = logger.summary()
    
    print(f"Run ID: {summary['run_id']}")
    print(f"Turns: {summary['turns']}")
    print(f"Regime: {summary['regime']}")
    print(f"Regime confidence: {summary['regime_confidence']:.2f}")
    print(f"Metrics: {list(summary['metrics'].keys())}")
    
    # Check JSONL output
    jsonl = logger.get_jsonl()
    lines = jsonl.strip().split("\n")
    print(f"\nJSONL lines: {len(lines)}")
    
    assert len(lines) == 20
    
    print("\n✓ Diagnostic logger working\n")
    return True


# =============================================================================
# Severity-Weighted Glassiness
# =============================================================================

SEVERITY_WEIGHTS = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 4,
    "CRITICAL": 8,
}


def compute_weighted_glassiness(severity_hist: Dict[str, int]) -> float:
    """
    Compute severity-weighted glassiness G_t.
    
    G_t = Σ w(severity(c)) for c in C_open
    
    Better stress proxy than raw count - lots of LOW != few CRITICAL.
    """
    total = 0.0
    for severity, count in severity_hist.items():
        weight = SEVERITY_WEIGHTS.get(severity, 1)
        total += weight * count
    return total


# =============================================================================
# Transition Detection
# =============================================================================

@dataclass
class RegimeTransition:
    """A detected regime transition."""
    turn_id: int
    prev_regime: Regime
    new_regime: Regime
    
    # Key deltas in the transition window
    c_open_delta: int
    open_rate: float      # contradictions opened per turn in window
    close_rate: float     # contradictions closed per turn in window
    net_accumulation: float  # open_rate - close_rate
    
    # Budget state
    budget_stress_before: float
    budget_stress_after: float
    
    # Severity distribution
    severity_distribution: Dict[str, int]
    weighted_glassiness: float
    
    # Block reasons
    block_reasons: Dict[str, int]
    
    # Attribution
    primary_cause: str
    contributing_factors: List[str]


class TransitionDetector:
    """
    Detects and attributes regime transitions.
    
    Answers: "When did the regime change, and why?"
    """
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.regime_history: List[Tuple[int, Regime]] = []
        self._event_buffer: deque = deque(maxlen=100)
    
    def record_regime(self, turn_id: int, regime: Regime) -> Optional[RegimeTransition]:
        """Record a regime observation, detect transitions."""
        if self.regime_history and self.regime_history[-1][1] != regime:
            # Transition detected
            prev_turn, prev_regime = self.regime_history[-1]
            transition = self._analyze_transition(
                turn_id, prev_regime, regime
            )
            self.regime_history.append((turn_id, regime))
            return transition
        
        self.regime_history.append((turn_id, regime))
        return None
    
    def record_event(self, event: DiagnosticEvent) -> None:
        """Buffer events for transition analysis."""
        self._event_buffer.append(event)
    
    def _analyze_transition(
        self, 
        turn_id: int, 
        prev_regime: Regime, 
        new_regime: Regime,
    ) -> RegimeTransition:
        """Analyze what caused a regime transition."""
        # Get events in lookback window
        window_events = [
            e for e in self._event_buffer 
            if e.turn_id > turn_id - self.window_size
        ]
        
        if not window_events:
            return RegimeTransition(
                turn_id=turn_id,
                prev_regime=prev_regime,
                new_regime=new_regime,
                c_open_delta=0,
                open_rate=0.0,
                close_rate=0.0,
                net_accumulation=0.0,
                budget_stress_before=0.0,
                budget_stress_after=0.0,
                severity_distribution={},
                weighted_glassiness=0.0,
                block_reasons={},
                primary_cause="UNKNOWN",
                contributing_factors=[],
            )
        
        # Compute rates
        total_opened = sum(e.c_opened_count for e in window_events)
        total_closed = sum(e.c_closed_count for e in window_events)
        window_len = len(window_events)
        
        open_rate = total_opened / window_len if window_len > 0 else 0
        close_rate = total_closed / window_len if window_len > 0 else 0
        net_accumulation = open_rate - close_rate
        
        # C_open delta
        first_c_open = window_events[0].c_open_before
        last_c_open = window_events[-1].c_open_after
        c_open_delta = last_c_open - first_c_open
        
        # Budget stress
        def budget_stress(remaining: Dict[str, float]) -> float:
            if not remaining:
                return 0.0
            return sum(1.0 / max(0.01, v) for v in remaining.values())
        
        budget_stress_before = budget_stress(window_events[0].budget_remaining_before)
        budget_stress_after = budget_stress(window_events[-1].budget_remaining_after)
        
        # Severity distribution (from last event)
        severity_dist = dict(window_events[-1].c_severity_hist_after)
        weighted_glass = compute_weighted_glassiness(severity_dist)
        
        # Block reasons
        block_reasons: Dict[str, int] = {}
        for e in window_events:
            for reason in e.blocked_by_invariant:
                block_reasons[reason] = block_reasons.get(reason, 0) + 1
        
        # Attribution
        primary_cause, factors = self._attribute_transition(
            prev_regime, new_regime,
            net_accumulation, budget_stress_after, block_reasons
        )
        
        return RegimeTransition(
            turn_id=turn_id,
            prev_regime=prev_regime,
            new_regime=new_regime,
            c_open_delta=c_open_delta,
            open_rate=open_rate,
            close_rate=close_rate,
            net_accumulation=net_accumulation,
            budget_stress_before=budget_stress_before,
            budget_stress_after=budget_stress_after,
            severity_distribution=severity_dist,
            weighted_glassiness=weighted_glass,
            block_reasons=block_reasons,
            primary_cause=primary_cause,
            contributing_factors=factors,
        )
    
    def _attribute_transition(
        self,
        prev: Regime,
        new: Regime,
        net_accum: float,
        budget_stress: float,
        block_reasons: Dict[str, int],
    ) -> Tuple[str, List[str]]:
        """Attribute the cause of a regime transition."""
        factors = []
        
        # Entering glass
        if new == Regime.GLASS_OSSIFICATION:
            if net_accum > 0.1:
                factors.append(f"net_accumulation={net_accum:.2f} (open > close)")
            if budget_stress > 1.0:
                factors.append(f"budget_stress={budget_stress:.2f}")
            return "ACCUMULATION_EXCEEDED_REPAIR", factors
        
        # Entering starvation
        if new == Regime.BUDGET_STARVATION:
            budget_blocks = sum(
                c for r, c in block_reasons.items() 
                if "budget" in r.lower() or "exhausted" in r.lower()
            )
            if budget_blocks > 0:
                factors.append(f"budget_blocks={budget_blocks}")
            return "BUDGET_EXHAUSTED", factors
        
        # Entering ceremony
        if new == Regime.CHATBOT_CEREMONY:
            return "NO_STATE_MUTATION", ["rho_S near zero"]
        
        # Entering sloppy fluid
        if new == Regime.PERMEABLE_MEMBRANE:
            return "EVIDENCE_BYPASS", ["closures without proper evidence"]
        
        # Recovery to healthy
        if new == Regime.HEALTHY_LATTICE:
            if net_accum < 0:
                factors.append("repair_caught_up")
            return "RECOVERED", factors
        
        return "UNKNOWN", factors


# =============================================================================
# Enhanced Regime Detector
# =============================================================================

class EnhancedRegimeDetector(RegimeDetector):
    """
    Enhanced regime detector with transition tracking and severity weighting.
    """
    
    def __init__(self):
        super().__init__()
        self.transition_detector = TransitionDetector()
        self._weighted_glassiness_history: deque = deque(maxlen=100)
        self._open_rate_history: deque = deque(maxlen=100)
        self._close_rate_history: deque = deque(maxlen=100)
    
    def record_event(self, event: DiagnosticEvent) -> None:
        """Record event for transition analysis."""
        self.transition_detector.record_event(event)
        
        # Track weighted glassiness
        weighted_g = compute_weighted_glassiness(event.c_severity_hist_after)
        self._weighted_glassiness_history.append(weighted_g)
    
    def analyze_with_transitions(
        self, 
        metrics: MetricsAggregator,
        events: List[DiagnosticEvent] = None,
    ) -> Tuple[RegimeAnalysis, List[RegimeTransition]]:
        """Analyze regime with transition detection."""
        analysis = self.analyze(metrics, events)
        
        # Record and check for transition
        transitions = []
        if metrics.turns_total > 0:
            transition = self.transition_detector.record_regime(
                metrics.turns_total, analysis.regime
            )
            if transition:
                transitions.append(transition)
        
        # Add weighted glassiness to indicators
        if self._weighted_glassiness_history:
            analysis.indicators["weighted_glassiness"] = list(self._weighted_glassiness_history)[-1]
            analysis.indicators["weighted_glassiness_trend"] = self._compute_trend(
                list(self._weighted_glassiness_history)
            )
        
        return analysis, transitions
    
    def get_throughput_metrics(self) -> Dict[str, float]:
        """Get open/close throughput metrics."""
        return {
            "open_rate_avg": (
                sum(self._open_rate_history) / len(self._open_rate_history)
                if self._open_rate_history else 0.0
            ),
            "close_rate_avg": (
                sum(self._close_rate_history) / len(self._close_rate_history)
                if self._close_rate_history else 0.0
            ),
        }


def run_all_tests():
    """Run all diagnostic tests."""
    print("=" * 70)
    print("PHASE DIAGNOSTICS TESTS")
    print("Observability for Interior Time")
    print("=" * 70 + "\n")
    
    results = []
    
    results.append(("diagnostic_event", test_diagnostic_event()))
    results.append(("metrics_aggregator", test_metrics_aggregator()))
    results.append(("regime_detection", test_regime_detection()))
    results.append(("energy_function", test_energy_function()))
    results.append(("diagnostic_logger", test_diagnostic_logger()))
    
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
        print("✓ DIAGNOSTICS INFRASTRUCTURE READY")
        print("  Now: observe before touching")
        print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
