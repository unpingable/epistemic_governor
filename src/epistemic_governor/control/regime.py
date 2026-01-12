"""
Regime Detector

Classifies system state into operational regimes based on observable signals.

This is NOT "is the model scheming?" - it's "is the system still in an
identifiable regime where our invariants hold?"

Regimes:
- ELASTIC: stable, identifiable, normal operation
- WARM: drifting but recoverable, tighten constraints
- DUCTILE: path dependent, probing is intervention, reset required
- UNSTABLE: positive feedback, cascade detected, emergency stop

The key insight: we don't probe intent, we observe dynamics.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

# Handle both package and direct execution
try:
    from .reset import ResetController, ResetType, ResetSeverity
except ImportError:
    from reset import ResetController, ResetType, ResetSeverity


class OperationalRegime(Enum):
    """System operational regimes."""
    ELASTIC = auto()    # Stable, identifiable
    WARM = auto()       # Drifting, recoverable
    DUCTILE = auto()    # Path dependent, reset needed
    UNSTABLE = auto()   # Cascade, emergency


@dataclass
class RegimeSignals:
    """Observable signals for regime classification."""
    
    # Hysteresis: does the system return to baseline after perturbation?
    # High = sticky behavior, path dependence
    hysteresis_magnitude: float = 0.0
    
    # Relaxation time: how long to return to baseline?
    # Long = ductile, short = elastic
    relaxation_time_seconds: float = 0.0
    
    # Tool gain: are perturbations amplifying?
    # k > 1 = unstable, k < 1 = stable
    tool_gain_estimate: float = 0.5
    
    # Anisotropy: variance under paraphrase/reframe
    # High = sensitive to framing, aliasing risk
    anisotropy_score: float = 0.0
    
    # Provenance deficit: claims without evidence anchors
    # High = drifting from grounded state
    provenance_deficit_rate: float = 0.0
    
    # Budget pressure: how often caps trigger
    # High = system is straining against limits
    budget_pressure: float = 0.0
    
    # Contradiction accumulation rate
    contradiction_open_rate: float = 0.0
    contradiction_close_rate: float = 0.0
    
    @property
    def c_accumulating(self) -> bool:
        """Are contradictions accumulating?"""
        return self.contradiction_open_rate > self.contradiction_close_rate * 1.2
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "hysteresis": self.hysteresis_magnitude,
            "relaxation_time": self.relaxation_time_seconds,
            "tool_gain": self.tool_gain_estimate,
            "anisotropy": self.anisotropy_score,
            "provenance_deficit": self.provenance_deficit_rate,
            "budget_pressure": self.budget_pressure,
            "c_open_rate": self.contradiction_open_rate,
            "c_close_rate": self.contradiction_close_rate,
        }


@dataclass
class RegimeThresholds:
    """Thresholds for regime transitions."""
    
    # ELASTIC → WARM
    warm_hysteresis: float = 0.2
    warm_relaxation: float = 3.0
    warm_anisotropy: float = 0.3
    warm_provenance_deficit: float = 0.2
    
    # WARM → DUCTILE
    ductile_hysteresis: float = 0.5
    ductile_relaxation: float = 10.0
    ductile_anisotropy: float = 0.5
    ductile_budget_pressure: float = 0.7
    
    # Any → UNSTABLE
    unstable_tool_gain: float = 1.0  # k >= 1 is unstable
    unstable_budget_pressure: float = 0.9


@dataclass
class RegimeTransition:
    """Record of a regime transition."""
    timestamp: datetime
    from_regime: OperationalRegime
    to_regime: OperationalRegime
    signals: RegimeSignals
    trigger_reason: str


@dataclass
class ResetEffectiveness:
    """Track whether a reset actually helped."""
    reset_id: str
    reset_type: str
    regime_before: OperationalRegime
    regime_after_1: Optional[OperationalRegime] = None  # 1 turn later
    regime_after_3: Optional[OperationalRegime] = None  # 3 turns later
    regime_after_5: Optional[OperationalRegime] = None  # 5 turns later
    restored_elastic: bool = False
    turns_to_restore: Optional[int] = None


@dataclass
class RegimeMetrics:
    """
    Metrics collection for empirical validation.
    
    Tracks:
    - Signal history for threshold tuning
    - Transition patterns
    - Reset effectiveness
    - Regime dwell times
    """
    # Signal history (for threshold analysis)
    signal_history: List[Tuple[datetime, RegimeSignals, OperationalRegime]] = field(default_factory=list)
    max_signal_history: int = 1000
    
    # Transition counts
    transition_counts: Dict[str, int] = field(default_factory=lambda: {
        "ELASTIC->WARM": 0,
        "ELASTIC->DUCTILE": 0,
        "ELASTIC->UNSTABLE": 0,
        "WARM->ELASTIC": 0,
        "WARM->DUCTILE": 0,
        "WARM->UNSTABLE": 0,
        "DUCTILE->ELASTIC": 0,
        "DUCTILE->WARM": 0,
        "DUCTILE->UNSTABLE": 0,
        "UNSTABLE->ELASTIC": 0,
        "UNSTABLE->WARM": 0,
        "UNSTABLE->DUCTILE": 0,
    })
    
    # Regime dwell times (how long in each regime)
    regime_entry_time: Dict[OperationalRegime, Optional[datetime]] = field(default_factory=dict)
    regime_dwell_times: Dict[OperationalRegime, List[float]] = field(default_factory=lambda: {
        OperationalRegime.ELASTIC: [],
        OperationalRegime.WARM: [],
        OperationalRegime.DUCTILE: [],
        OperationalRegime.UNSTABLE: [],
    })
    
    # Reset effectiveness tracking
    pending_reset_checks: List[ResetEffectiveness] = field(default_factory=list)
    completed_reset_checks: List[ResetEffectiveness] = field(default_factory=list)
    
    # Turn counter for reset effectiveness
    turn_counter: int = 0
    reset_turn_map: Dict[str, int] = field(default_factory=dict)
    
    def record_signal(self, signals: RegimeSignals, regime: OperationalRegime):
        """Record a signal observation."""
        self.signal_history.append((datetime.now(timezone.utc), signals, regime))
        if len(self.signal_history) > self.max_signal_history:
            self.signal_history.pop(0)
    
    def record_transition(self, from_regime: OperationalRegime, to_regime: OperationalRegime):
        """Record a regime transition."""
        key = f"{from_regime.name}->{to_regime.name}"
        if key in self.transition_counts:
            self.transition_counts[key] += 1
        
        # Update dwell time for exited regime
        now = datetime.now(timezone.utc)
        if from_regime in self.regime_entry_time and self.regime_entry_time[from_regime]:
            entry = self.regime_entry_time[from_regime]
            dwell = (now - entry).total_seconds()
            self.regime_dwell_times[from_regime].append(dwell)
        
        # Mark entry to new regime
        self.regime_entry_time[to_regime] = now
    
    def record_reset(self, reset_id: str, reset_type: str, regime_before: OperationalRegime):
        """Start tracking reset effectiveness."""
        self.reset_turn_map[reset_id] = self.turn_counter
        self.pending_reset_checks.append(ResetEffectiveness(
            reset_id=reset_id,
            reset_type=reset_type,
            regime_before=regime_before,
        ))
    
    def advance_turn(self, current_regime: OperationalRegime):
        """Advance turn counter and update reset effectiveness tracking."""
        self.turn_counter += 1
        
        # Check pending resets
        still_pending = []
        for check in self.pending_reset_checks:
            reset_turn = self.reset_turn_map.get(check.reset_id, 0)
            turns_since = self.turn_counter - reset_turn
            
            if turns_since == 1:
                check.regime_after_1 = current_regime
            elif turns_since == 3:
                check.regime_after_3 = current_regime
            elif turns_since == 5:
                check.regime_after_5 = current_regime
                # Final check - did it restore elastic?
                check.restored_elastic = current_regime == OperationalRegime.ELASTIC
                if check.restored_elastic:
                    # Find when it first hit elastic
                    if check.regime_after_1 == OperationalRegime.ELASTIC:
                        check.turns_to_restore = 1
                    elif check.regime_after_3 == OperationalRegime.ELASTIC:
                        check.turns_to_restore = 3
                    else:
                        check.turns_to_restore = 5
                self.completed_reset_checks.append(check)
                continue
            
            still_pending.append(check)
        
        self.pending_reset_checks = still_pending
    
    def get_threshold_analysis(self) -> Dict[str, Any]:
        """Analyze signal distributions by regime for threshold tuning."""
        if not self.signal_history:
            return {}
        
        # Group signals by regime
        by_regime: Dict[OperationalRegime, List[RegimeSignals]] = {
            r: [] for r in OperationalRegime
        }
        for _, signals, regime in self.signal_history:
            by_regime[regime].append(signals)
        
        analysis = {}
        for regime, signals_list in by_regime.items():
            if not signals_list:
                continue
            
            analysis[regime.name] = {
                "count": len(signals_list),
                "hysteresis": {
                    "min": min(s.hysteresis_magnitude for s in signals_list),
                    "max": max(s.hysteresis_magnitude for s in signals_list),
                    "mean": sum(s.hysteresis_magnitude for s in signals_list) / len(signals_list),
                },
                "relaxation_time": {
                    "min": min(s.relaxation_time_seconds for s in signals_list),
                    "max": max(s.relaxation_time_seconds for s in signals_list),
                    "mean": sum(s.relaxation_time_seconds for s in signals_list) / len(signals_list),
                },
                "tool_gain": {
                    "min": min(s.tool_gain_estimate for s in signals_list),
                    "max": max(s.tool_gain_estimate for s in signals_list),
                    "mean": sum(s.tool_gain_estimate for s in signals_list) / len(signals_list),
                },
                "anisotropy": {
                    "min": min(s.anisotropy_score for s in signals_list),
                    "max": max(s.anisotropy_score for s in signals_list),
                    "mean": sum(s.anisotropy_score for s in signals_list) / len(signals_list),
                },
            }
        
        return analysis
    
    def get_reset_effectiveness_summary(self) -> Dict[str, Any]:
        """Summarize reset effectiveness."""
        if not self.completed_reset_checks:
            return {"completed": 0}
        
        by_type: Dict[str, List[ResetEffectiveness]] = {}
        for check in self.completed_reset_checks:
            by_type.setdefault(check.reset_type, []).append(check)
        
        summary = {"completed": len(self.completed_reset_checks)}
        
        for reset_type, checks in by_type.items():
            restored = [c for c in checks if c.restored_elastic]
            summary[reset_type] = {
                "total": len(checks),
                "restored_elastic": len(restored),
                "success_rate": len(restored) / len(checks) if checks else 0,
                "avg_turns_to_restore": (
                    sum(c.turns_to_restore for c in restored if c.turns_to_restore) / len(restored)
                    if restored else None
                ),
            }
        
        return summary
    
    def get_summary(self) -> Dict[str, Any]:
        """Get full metrics summary."""
        # Calculate average dwell times
        avg_dwell = {}
        for regime, times in self.regime_dwell_times.items():
            if times:
                avg_dwell[regime.name] = sum(times) / len(times)
        
        return {
            "total_observations": len(self.signal_history),
            "total_turns": self.turn_counter,
            "transitions": self.transition_counts,
            "avg_dwell_times": avg_dwell,
            "reset_effectiveness": self.get_reset_effectiveness_summary(),
            "threshold_analysis": self.get_threshold_analysis(),
        }


class RegimeDetector:
    """
    Detects operational regime from observable signals.
    
    This is a classifier over dynamics, not semantics.
    
    Instrumented for empirical validation:
    - Signal history for threshold tuning
    - Transition log for pattern analysis
    - Reset effectiveness tracking
    - False positive/negative estimation
    """
    
    def __init__(
        self,
        thresholds: Optional[RegimeThresholds] = None,
        reset_controller: Optional[ResetController] = None,
        collect_metrics: bool = True,
    ):
        self.thresholds = thresholds or RegimeThresholds()
        self.reset_controller = reset_controller or ResetController()
        self.collect_metrics = collect_metrics
        
        self.current_regime = OperationalRegime.ELASTIC
        self.transition_history: List[RegimeTransition] = []
        self.last_signals: Optional[RegimeSignals] = None
        
        # === Metrics Collection ===
        self.metrics = RegimeMetrics() if collect_metrics else None
    
    def classify(self, signals: RegimeSignals) -> Tuple[OperationalRegime, str]:
        """
        Classify current regime based on signals.
        
        Returns (regime, reason).
        """
        self.last_signals = signals
        t = self.thresholds
        
        # Check UNSTABLE first (highest priority)
        if signals.tool_gain_estimate >= t.unstable_tool_gain:
            regime = OperationalRegime.UNSTABLE
            reason = f"tool_gain={signals.tool_gain_estimate:.2f} >= {t.unstable_tool_gain}"
            if self.metrics:
                self.metrics.record_signal(signals, regime)
            return regime, reason
        
        if signals.budget_pressure >= t.unstable_budget_pressure:
            regime = OperationalRegime.UNSTABLE
            reason = f"budget_pressure={signals.budget_pressure:.2f} >= {t.unstable_budget_pressure}"
            if self.metrics:
                self.metrics.record_signal(signals, regime)
            return regime, reason
        
        # Check DUCTILE
        ductile_reasons = []
        if signals.hysteresis_magnitude >= t.ductile_hysteresis:
            ductile_reasons.append(f"hysteresis={signals.hysteresis_magnitude:.2f}")
        if signals.relaxation_time_seconds >= t.ductile_relaxation:
            ductile_reasons.append(f"relaxation={signals.relaxation_time_seconds:.1f}s")
        if signals.anisotropy_score >= t.ductile_anisotropy:
            ductile_reasons.append(f"anisotropy={signals.anisotropy_score:.2f}")
        if signals.budget_pressure >= t.ductile_budget_pressure:
            ductile_reasons.append(f"budget_pressure={signals.budget_pressure:.2f}")
        
        if len(ductile_reasons) >= 2:  # Need multiple indicators
            regime = OperationalRegime.DUCTILE
            reason = "; ".join(ductile_reasons)
            if self.metrics:
                self.metrics.record_signal(signals, regime)
            return regime, reason
        
        # Check WARM
        warm_reasons = []
        if signals.hysteresis_magnitude >= t.warm_hysteresis:
            warm_reasons.append(f"hysteresis={signals.hysteresis_magnitude:.2f}")
        if signals.relaxation_time_seconds >= t.warm_relaxation:
            warm_reasons.append(f"relaxation={signals.relaxation_time_seconds:.1f}s")
        if signals.anisotropy_score >= t.warm_anisotropy:
            warm_reasons.append(f"anisotropy={signals.anisotropy_score:.2f}")
        if signals.provenance_deficit_rate >= t.warm_provenance_deficit:
            warm_reasons.append(f"provenance_deficit={signals.provenance_deficit_rate:.2f}")
        if signals.c_accumulating:
            warm_reasons.append("contradictions accumulating")
        
        if warm_reasons:
            regime = OperationalRegime.WARM
            reason = "; ".join(warm_reasons)
            if self.metrics:
                self.metrics.record_signal(signals, regime)
            return regime, reason
        
        # Default: ELASTIC
        regime = OperationalRegime.ELASTIC
        reason = "all signals nominal"
        if self.metrics:
            self.metrics.record_signal(signals, regime)
        return regime, reason
    
    def update(self, signals: RegimeSignals) -> Optional[RegimeTransition]:
        """
        Update regime based on new signals.
        
        Returns transition if regime changed.
        """
        new_regime, reason = self.classify(signals)
        
        if new_regime != self.current_regime:
            transition = RegimeTransition(
                timestamp=datetime.now(timezone.utc),
                from_regime=self.current_regime,
                to_regime=new_regime,
                signals=signals,
                trigger_reason=reason,
            )
            
            # Record metrics
            if self.metrics:
                self.metrics.record_transition(self.current_regime, new_regime)
            
            self.transition_history.append(transition)
            self.current_regime = new_regime
            return transition
        
        return None
    
    def respond(self, signals: RegimeSignals) -> Dict[str, Any]:
        """
        Detect regime and execute appropriate response.
        
        This is the auto-mapping: regime → response.
        Deterministic. No negotiation.
        """
        transition = self.update(signals)
        regime = self.current_regime
        response = {"regime": regime.name, "transition": transition is not None}
        
        # Advance turn counter for metrics
        if self.metrics:
            self.metrics.advance_turn(regime)
        
        if regime == OperationalRegime.ELASTIC:
            # Normal operation
            response["action"] = "CONTINUE"
            response["constraints"] = "standard gating for irreversible actions"
            
        elif regime == OperationalRegime.WARM:
            # Tighten constraints
            response["action"] = "TIGHTEN"
            response["constraints"] = {
                "evidence_requirement": "increased",
                "variety_dial": "reduced",
                "ttl": "shortened",
                "checkpoint_frequency": "increased",
            }
            # Apply mode degradation if not already degraded
            if not self.reset_controller.mode.is_degraded():
                event = self.reset_controller.mode_reset(
                    regime=regime.name,
                    signals=signals.to_dict(),
                    reason="Entering WARM regime",
                    degrade_level=1,
                )
                response["reset_event"] = event.reset_id
                if self.metrics:
                    self.metrics.record_reset(event.reset_id, "MODE", regime)
        
        elif regime == OperationalRegime.DUCTILE:
            # Mandatory reset
            response["action"] = "RESET"
            response["constraints"] = {
                "context": "cleared",
                "horizon": "short",
                "memory_writes": "disabled",
                "irreversible_actions": "require human confirm",
            }
            # Context reset + mode degradation
            ctx_event = self.reset_controller.context_reset(
                regime=regime.name,
                signals=signals.to_dict(),
                reason="Ductile regime - mandatory context reset",
            )
            mode_event = self.reset_controller.mode_reset(
                regime=regime.name,
                signals=signals.to_dict(),
                reason="Ductile regime - mode degradation",
                degrade_level=2,
            )
            response["reset_events"] = [ctx_event.reset_id, mode_event.reset_id]
            if self.metrics:
                self.metrics.record_reset(ctx_event.reset_id, "CONTEXT", regime)
                self.metrics.record_reset(mode_event.reset_id, "MODE", regime)
        
        elif regime == OperationalRegime.UNSTABLE:
            # Emergency stop
            response["action"] = "EMERGENCY_STOP"
            response["constraints"] = {
                "tools": "circuit breaker",
                "state": "rollback to checkpoint",
                "escalation": "human required",
            }
            # Chain reset
            event = self.reset_controller.chain_reset(
                regime=regime.name,
                signals=signals.to_dict(),
                reason="Unstable regime - emergency chain reset",
            )
            response["reset_event"] = event.reset_id
            response["escalate"] = True
            if self.metrics:
                self.metrics.record_reset(event.reset_id, "CHAIN", regime)
        
        return response
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "current_regime": self.current_regime.name,
            "transitions": len(self.transition_history),
            "last_signals": self.last_signals.to_dict() if self.last_signals else None,
            "mode": self.reset_controller.get_state()["mode"],
            "metrics": self.metrics.get_summary() if self.metrics else None,
        }
    
    def get_metrics_report(self) -> Dict[str, Any]:
        """Get detailed metrics report for validation."""
        if not self.metrics:
            return {"error": "Metrics collection not enabled"}
        
        return {
            "summary": self.metrics.get_summary(),
            "threshold_analysis": self.metrics.get_threshold_analysis(),
            "reset_effectiveness": self.metrics.get_reset_effectiveness_summary(),
            "current_thresholds": {
                "warm_hysteresis": self.thresholds.warm_hysteresis,
                "warm_relaxation": self.thresholds.warm_relaxation,
                "warm_anisotropy": self.thresholds.warm_anisotropy,
                "ductile_hysteresis": self.thresholds.ductile_hysteresis,
                "ductile_relaxation": self.thresholds.ductile_relaxation,
                "ductile_anisotropy": self.thresholds.ductile_anisotropy,
                "unstable_tool_gain": self.thresholds.unstable_tool_gain,
            },
        }


# =============================================================================
# Coupling Reduction Checker
# =============================================================================

def check_coupling_reduction(
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Verify that an intervention reduced at least one coupling dimension.
    
    Rule: each escalation step must reduce at least one of:
    - temporal coupling (Δt / horizon)
    - tool coupling (tool depth / enabled)
    - memory coupling (writes)
    - incentive coupling (self-eval / self-justification)
    """
    reductions = []
    
    # Temporal
    if after.get("horizon_turns", 10) < before.get("horizon_turns", 10):
        reductions.append("temporal (horizon)")
    if after.get("ttl_seconds", 3600) < before.get("ttl_seconds", 3600):
        reductions.append("temporal (TTL)")
    
    # Tool
    if before.get("tools_enabled", True) and not after.get("tools_enabled", True):
        reductions.append("tool (disabled)")
    if not before.get("readonly_mode", False) and after.get("readonly_mode", False):
        reductions.append("tool (readonly)")
    
    # Memory
    if before.get("memory_writes", True) and not after.get("memory_writes", True):
        reductions.append("memory (writes disabled)")
    
    # Variety
    if after.get("variety_multiplier", 1.0) < before.get("variety_multiplier", 1.0):
        reductions.append("variety (reduced)")
    
    return len(reductions) > 0, reductions


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=== Regime Detector Test ===\n")
    
    detector = RegimeDetector()
    
    # ELASTIC signals
    signals = RegimeSignals(
        hysteresis_magnitude=0.1,
        relaxation_time_seconds=1.0,
        tool_gain_estimate=0.3,
        anisotropy_score=0.1,
    )
    response = detector.respond(signals)
    print(f"ELASTIC: {response['regime']}, action={response['action']}")
    
    # WARM signals
    signals = RegimeSignals(
        hysteresis_magnitude=0.3,
        relaxation_time_seconds=4.0,
        tool_gain_estimate=0.5,
        anisotropy_score=0.35,
        provenance_deficit_rate=0.25,
    )
    response = detector.respond(signals)
    print(f"WARM: {response['regime']}, action={response['action']}")
    
    # DUCTILE signals
    signals = RegimeSignals(
        hysteresis_magnitude=0.6,
        relaxation_time_seconds=15.0,
        tool_gain_estimate=0.7,
        anisotropy_score=0.55,
        budget_pressure=0.75,
    )
    response = detector.respond(signals)
    print(f"DUCTILE: {response['regime']}, action={response['action']}")
    print(f"  Reset events: {response.get('reset_events', [])}")
    
    # UNSTABLE signals
    signals = RegimeSignals(
        hysteresis_magnitude=0.8,
        tool_gain_estimate=1.2,  # k > 1
    )
    response = detector.respond(signals)
    print(f"UNSTABLE: {response['regime']}, action={response['action']}")
    print(f"  Escalate: {response.get('escalate', False)}")
    
    print(f"\nFinal state: {detector.get_state()}")
    print(f"Transitions: {len(detector.transition_history)}")
    
    # Test coupling reduction
    print("\n=== Coupling Reduction Check ===")
    before = {"horizon_turns": 10, "tools_enabled": True, "variety_multiplier": 1.0}
    after = {"horizon_turns": 5, "tools_enabled": True, "variety_multiplier": 0.5}
    reduced, dimensions = check_coupling_reduction(before, after)
    print(f"Reduced coupling: {reduced}")
    print(f"Dimensions: {dimensions}")
