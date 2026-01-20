"""
Observability - Telemetry and Diagnostics

This package contains the observability layer for the governor.

Contents:
- otel: OpenTelemetry projection and emission
- trace_tui: Terminal UI for trace visualization ("Cyberpunk Console")
- diagnostics: DiagnosticEvent and regime detection (imported from existing)
"""

from epistemic_governor.observability.otel import (
    project_to_otel,
    decision_step_span,
    emit_decision_step,
    compute_enforcement_verdict,
    classify_action_kind,
)

# TUI imports (optional - requires rich)
try:
    from epistemic_governor.observability.trace_tui import (
        run_tui,
        generate_demo_trace,
        print_trace_stats,
    )
    TUI_AVAILABLE = True
except ImportError:
    TUI_AVAILABLE = False
    run_tui = None
    generate_demo_trace = None
    print_trace_stats = None

__all__ = [
    "project_to_otel",
    "decision_step_span",
    "emit_decision_step",
    "compute_enforcement_verdict",
    "classify_action_kind",
    "run_tui",
    "generate_demo_trace",
    "print_trace_stats",
    "TUI_AVAILABLE",
]
