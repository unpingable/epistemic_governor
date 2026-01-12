"""
Observability - Telemetry and Diagnostics

This package contains the observability layer for the governor.

Contents:
- otel: OpenTelemetry projection and emission
- diagnostics: DiagnosticEvent and regime detection (imported from existing)
"""

from epistemic_governor.observability.otel import (
    project_to_otel,
    decision_step_span,
    emit_decision_step,
    compute_enforcement_verdict,
    classify_action_kind,
)

__all__ = [
    "project_to_otel",
    "decision_step_span",
    "emit_decision_step",
    "compute_enforcement_verdict",
    "classify_action_kind",
]
