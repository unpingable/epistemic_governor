"""
OTel Projection Layer

Projects DiagnosticEvent -> OTel span attributes with strict cardinality control.

This is the bridge between BLI's internal telemetry and external observability.
See OTEL_CONVENTIONS.md for the full semantic conventions spec.

Usage:
    from epistemic_governor.otel_projection import project_to_otel, decision_step_span
    
    attrs = project_to_otel(diagnostic_event, regime_analysis)
    with decision_step_span("epistemic.decision_step", attrs):
        # ... your code ...
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from epistemic_governor.diagnostics import DiagnosticEvent, RegimeAnalysis


# =============================================================================
# Regime Mapping (internal enum -> OTel string)
# =============================================================================

REGIME_MAP = {
    "UNKNOWN": "UNKNOWN",
    "HEALTHY_LATTICE": "HEALTHY",
    "CHATBOT_CEREMONY": "CEREMONY",
    "GLASS_OSSIFICATION": "GLASS",
    "PERMEABLE_MEMBRANE": "PERMEABLE",
    "BUDGET_STARVATION": "STARVATION",
    "EXTRACTION_COLLAPSE": "EXTRACTION_COLLAPSE",
}


# =============================================================================
# Invariant Code Mapping
# =============================================================================

# Maps internal violation strings to standardized codes
INVARIANT_CODE_MAP = {
    # NLAI violations
    "nlai": "I1_NLAI",
    "language_authority": "I1_NLAI",
    "model_text_evidence": "I1_NLAI",
    
    # Ledger violations
    "ledger_tamper": "I2_LEDGER_TAMPER",
    "ledger_modification": "I2_LEDGER_TAMPER",
    
    # Resolution violations
    "silent_resolution": "I3_SILENT_RESOLUTION",
    "closure_without_evidence": "I3_SILENT_RESOLUTION",
    
    # Authority violations
    "authority_spoof": "I4_AUTHORITY_SPOOF",
    "fake_citation": "I4_AUTHORITY_SPOOF",
    
    # Self-certification
    "self_certification": "I5_SELF_CERTIFICATION",
    "self_reference": "I5_SELF_CERTIFICATION",
    
    # Promotion violations
    "forbidden_promotion": "I6_FORBIDDEN_PROMOTION",
    
    # Budget violations
    "budget_violation": "I7_BUDGET_VIOLATION",
    "budget_exceeded": "I7_BUDGET_VIOLATION",
    
    # Jurisdiction violations
    "jurisdiction_escape": "I8_JURISDICTION_ESCAPE",
    "cross_jurisdiction": "I8_JURISDICTION_ESCAPE",
}


def normalize_invariant_code(raw: str) -> str:
    """Map internal violation string to standardized code."""
    key = raw.lower().replace(" ", "_").replace("-", "_")
    return INVARIANT_CODE_MAP.get(key, f"I0_UNKNOWN:{raw[:20]}")


# =============================================================================
# Verdict Mapping
# =============================================================================

def compute_enforcement_verdict(
    verdict: str,
    blocked_by_invariant: List[str],
    enforcement_mode: str = "OBSERVE"
) -> str:
    """
    Compute enforcement verdict for counterfactual gating.
    
    Returns:
        ALLOWED - no violation
        WARN - violation but not gating-worthy
        WOULD_BLOCK - would block if gating enabled
        BLOCKED - actually blocked (only in GATE mode)
    """
    if not blocked_by_invariant and verdict == "OK":
        return "ALLOWED"
    
    if enforcement_mode == "GATE" and verdict == "BLOCK":
        return "BLOCKED"
    
    # In OBSERVE mode, we report what WOULD have happened
    if verdict == "BLOCK" or blocked_by_invariant:
        return "WOULD_BLOCK"
    
    if verdict == "WARN":
        return "WARN"
    
    return "ALLOWED"


# =============================================================================
# Main Projection Function
# =============================================================================

def project_to_otel(
    event: "DiagnosticEvent",
    regime: Optional["RegimeAnalysis"] = None,
    action_kind: Optional[str] = None,
    enforcement_mode: str = "OBSERVE",
) -> Dict[str, Any]:
    """
    Project DiagnosticEvent to OTel span attributes.
    
    STRICT CARDINALITY RULES:
    - No raw prompts, outputs, or user content
    - No high-cardinality IDs (use step_id for correlation)
    - Enums and booleans only where possible
    - Floats normalized to 0..1 where applicable
    
    Args:
        event: The DiagnosticEvent from governor processing
        regime: Optional RegimeAnalysis (if not provided, uses UNKNOWN)
        action_kind: Optional action classification (READ/WRITE/DELETE/EXEC/OTHER)
        enforcement_mode: OBSERVE or GATE
    
    Returns:
        Dict of OTel-compliant span attributes
    """
    attrs: Dict[str, Any] = {}
    
    # --- Enforcement ---
    attrs["epistemic.enforcement.mode"] = enforcement_mode
    attrs["epistemic.enforcement.verdict"] = compute_enforcement_verdict(
        event.verdict,
        event.blocked_by_invariant,
        enforcement_mode
    )
    
    # --- Regime ---
    if regime:
        regime_name = regime.regime.name if hasattr(regime.regime, 'name') else str(regime.regime)
        attrs["epistemic.regime"] = REGIME_MAP.get(regime_name, "UNKNOWN")
        attrs["epistemic.regime.confidence"] = float(regime.confidence)
    else:
        attrs["epistemic.regime"] = "UNKNOWN"
        attrs["epistemic.regime.confidence"] = 0.0
    
    # --- Violations ---
    attrs["epistemic.violation"] = len(event.blocked_by_invariant) > 0 or event.verdict == "BLOCK"
    
    if event.blocked_by_invariant:
        # Only emit first violation code to keep cardinality bounded
        attrs["epistemic.violation.code"] = normalize_invariant_code(event.blocked_by_invariant[0])
        attrs["epistemic.violation.count"] = len(event.blocked_by_invariant)
    
    # --- Contradictions ---
    attrs["epistemic.contradictions.open"] = event.c_open_after
    attrs["epistemic.contradictions.opened"] = event.c_opened_count
    attrs["epistemic.contradictions.closed"] = event.c_closed_count
    
    # --- State ---
    attrs["epistemic.state.mutated"] = event.rho_S_flag
    attrs["epistemic.energy"] = float(event.E_state_after)
    
    # --- Budget (normalized to 0..1) ---
    if event.budget_remaining_after:
        # Use repair budget as primary indicator
        repair = event.budget_remaining_after.get("repair", 1.0)
        # Normalize assuming max budget of 100 (adjust if different)
        attrs["epistemic.budget.remaining"] = min(1.0, max(0.0, repair / 100.0))
        attrs["epistemic.budget.exhausted"] = any(event.budget_exhaustion.values())
    
    # --- Timing ---
    attrs["epistemic.latency.total_ms"] = float(event.latency_ms_total)
    attrs["epistemic.latency.governor_ms"] = float(event.latency_ms_governor)
    
    # --- Action (if provided) ---
    if action_kind:
        attrs["epistemic.action.kind"] = action_kind
    
    # --- Correlation (low-cardinality) ---
    # Only include stable, bounded IDs
    attrs["epistemic.turn_id"] = event.turn_id
    
    return attrs


# =============================================================================
# Action Classification
# =============================================================================

def classify_action_kind(tool_name: str, tool_args: Optional[Dict[str, Any]] = None) -> str:
    """
    Classify tool action into low-cardinality buckets.
    
    Returns: READ | WRITE | DELETE | EXEC | OTHER
    """
    if tool_args is None:
        tool_args = {}
    
    t = tool_name.lower()
    method = str(tool_args.get("method", "")).upper()
    
    # HTTP method hints (highest priority)
    if method == "GET":
        return "READ"
    if method == "DELETE":
        return "DELETE"
    if method in {"POST", "PUT", "PATCH"}:
        return "WRITE"
    
    # Exec hints (check before read - "execute_query" is EXEC, not READ)
    if any(x in t for x in ["exec", "run", "execute", "invoke"]):
        return "EXEC"
    
    # Delete hints
    if any(x in t for x in ["delete", "remove", "drop"]):
        return "DELETE"
    
    # Write hints
    if any(x in t for x in ["write", "create", "update", "set", "put", "post", "insert"]):
        return "WRITE"
    
    # Read hints (lowest priority for name matching)
    if any(x in t for x in ["get", "read", "fetch", "query", "list", "search"]):
        return "READ"
    
    return "OTHER"


# =============================================================================
# Span Context Manager
# =============================================================================

def _try_get_tracer():
    """Try to get OTel tracer, return None if not available."""
    try:
        from opentelemetry import trace
        return trace.get_tracer("epistemic_governor")
    except ImportError:
        return None


@contextmanager
def decision_step_span(
    name: str,
    attrs: Dict[str, Any],
    extra_attrs: Optional[Dict[str, Any]] = None
):
    """
    Create a span for a decision step.
    
    Falls back to JSON logging if OpenTelemetry is not installed.
    
    Args:
        name: Span name (use "epistemic.decision_step")
        attrs: Attributes from project_to_otel()
        extra_attrs: Additional low-cardinality attributes
    
    Yields:
        The span object (or a dict in fallback mode)
    """
    all_attrs = dict(attrs)
    if extra_attrs:
        all_attrs.update(extra_attrs)
    
    tracer = _try_get_tracer()
    start = time.time()
    
    if tracer is None:
        # Fallback: JSON log record
        payload = {
            "name": name,
            "trace_id": str(uuid.uuid4()),
            "span_id": str(uuid.uuid4()),
            "start_ts": datetime.utcnow().isoformat(),
            "attributes": all_attrs,
        }
        try:
            yield payload
        finally:
            payload["end_ts"] = datetime.utcnow().isoformat()
            payload["duration_ms"] = int((time.time() - start) * 1000)
            # Emit as single-line JSON for easy grep/jq
            print(json.dumps(payload, sort_keys=True, default=str))
        return
    
    # OTel path
    with tracer.start_as_current_span(name) as span:
        for k, v in all_attrs.items():
            span.set_attribute(k, v)
        try:
            yield span
        finally:
            pass  # OTel handles timing


# =============================================================================
# Convenience: Full Pipeline
# =============================================================================

def emit_decision_step(
    event: "DiagnosticEvent",
    regime: Optional["RegimeAnalysis"] = None,
    tool_name: Optional[str] = None,
    tool_args: Optional[Dict[str, Any]] = None,
    enforcement_mode: str = "OBSERVE",
) -> Dict[str, Any]:
    """
    One-shot: project event and emit span.
    
    Returns the attributes dict for testing/verification.
    """
    action_kind = classify_action_kind(tool_name, tool_args) if tool_name else None
    
    attrs = project_to_otel(
        event=event,
        regime=regime,
        action_kind=action_kind,
        enforcement_mode=enforcement_mode,
    )
    
    extra = {}
    if tool_name:
        extra["tool.name"] = tool_name[:50]  # Truncate for cardinality
    
    with decision_step_span("epistemic.decision_step", attrs, extra):
        pass  # Span emitted on exit
    
    return attrs


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Minimal test without full DiagnosticEvent
    @dataclass
    class MockEvent:
        run_id: str = "test-001"
        turn_id: int = 1
        verdict: str = "WARN"
        blocked_by_invariant: List[str] = None
        c_open_after: int = 3
        c_opened_count: int = 1
        c_closed_count: int = 0
        rho_S_flag: bool = True
        E_state_after: float = 12.5
        budget_remaining_after: Dict[str, float] = None
        budget_exhaustion: Dict[str, bool] = None
        latency_ms_total: float = 150.0
        latency_ms_governor: float = 12.0
        
        def __post_init__(self):
            if self.blocked_by_invariant is None:
                self.blocked_by_invariant = ["self_certification"]
            if self.budget_remaining_after is None:
                self.budget_remaining_after = {"repair": 45.0}
            if self.budget_exhaustion is None:
                self.budget_exhaustion = {"repair": False}
    
    @dataclass
    class MockRegime:
        class _Regime:
            name = "GLASS_OSSIFICATION"
        regime = _Regime()
        confidence: float = 0.78
    
    print("=== OTel Projection Test ===\n")
    
    event = MockEvent()
    regime = MockRegime()
    
    attrs = project_to_otel(event, regime, action_kind="WRITE")
    
    print("Projected attributes:")
    for k, v in sorted(attrs.items()):
        print(f"  {k}: {v}")
    
    print("\n=== Emitting span (JSON fallback) ===\n")
    
    emit_decision_step(
        event=event,
        regime=regime,
        tool_name="database_write",
        tool_args={"method": "POST"},
    )
