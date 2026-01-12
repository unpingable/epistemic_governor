"""
Control - S₁ Regulatory Controllers

This package contains the adaptive controllers that tune system behavior
within constitutional bounds.

Contents:
- ultrastability: Ashby-style second-order adaptation
- variety: Requisite variety management (load shedding)
- temporal: TTL, lag budgets, clock coherence
- provenance: Failure taxonomy and causal chains
- reset: Typed state contraction operations
- regime: Operational regime detection and response
- boil: Named regime presets with dwell time (kettle pattern)
"""

from epistemic_governor.control.ultrastability import (
    UltrastabilityController,
    RegulatoryParameters,
    AdaptationTrigger,
    PathologyDetector,
    AdaptationVerdict,
    AdaptationDecision,
)

from epistemic_governor.control.variety import (
    VarietyController,
    VarietyBounds,
    VarietyVerdict,
)

from epistemic_governor.control.temporal import (
    TemporalController,
    TemporalBounds,
    TemporalVerdict,
    TimestampedItem,
    check_turn_temporal,
)

from epistemic_governor.control.provenance import (
    FailureRegistry,
    FailureBuilder,
    FailureEvent,
    FailureCause,
    FailureSubsystem,
    FailureType,
    FailureSeverity,
)

from epistemic_governor.control.reset import (
    ResetController,
    ResetType,
    ResetSeverity,
    ResetTarget,
    ResetEvent,
    ModeState,
)

from epistemic_governor.control.regime import (
    RegimeDetector,
    OperationalRegime,
    RegimeSignals,
    RegimeThresholds,
    RegimeTransition,
    check_coupling_reduction,
)

from epistemic_governor.control.boil import (
    BoilController,
    BoilPreset,
    ControlMode,
    PRESETS,
)

__all__ = [
    # Ultrastability
    "UltrastabilityController",
    "RegulatoryParameters",
    "AdaptationTrigger",
    "PathologyDetector",
    "AdaptationVerdict",
    "AdaptationDecision",
    # Variety
    "VarietyController",
    "VarietyBounds",
    "VarietyVerdict",
    # Temporal
    "TemporalController",
    "TemporalBounds",
    "TemporalVerdict",
    "TimestampedItem",
    "check_turn_temporal",
    # Provenance
    "FailureRegistry",
    "FailureBuilder",
    "FailureEvent",
    "FailureCause",
    "FailureSubsystem",
    "FailureType",
    "FailureSeverity",
    # Reset
    "ResetController",
    "ResetType",
    "ResetSeverity",
    "ResetTarget",
    "ResetEvent",
    "ModeState",
    # Regime
    "RegimeDetector",
    "OperationalRegime",
    "RegimeSignals",
    "RegimeThresholds",
    "RegimeTransition",
    "check_coupling_reduction",
    # Boil
    "BoilController",
    "BoilPreset",
    "ControlMode",
    "PRESETS",
]
