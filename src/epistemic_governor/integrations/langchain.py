"""
LangChain Callback Handler for Epistemic Observability

OBSERVE-first: never blocks, only emits telemetry.

Usage:
    from instrumentation.langchain_callback import EpistemicCallback
    
    callback = EpistemicCallback(governor)
    
    # With LangChain agent
    agent.run("...", callbacks=[callback])

Emits one span per tool call decision step.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sovereign import SovereignGovernor

# Try to import LangChain callback base
try:
    from langchain.callbacks.base import BaseCallbackHandler
except ImportError:
    # Fallback for environments without LangChain
    class BaseCallbackHandler:
        """Stub for type checking when LangChain not installed."""
        pass

# Import our projection layer
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from observability.otel import (
    project_to_otel,
    decision_step_span,
    classify_action_kind,
)
from epistemic_governor.diagnostics import DiagnosticEvent, Regime, RegimeAnalysis


class EpistemicCallback(BaseCallbackHandler):
    """
    LangChain callback that emits epistemic telemetry.
    
    OBSERVE-first: never blocks execution, only logs what would have happened.
    
    Emits:
        - One span per tool call (start -> end)
        - Regime, violations, contradictions
        - WOULD_BLOCK verdicts for counterfactual analysis
    """
    
    def __init__(
        self,
        governor: "SovereignGovernor",
        enforcement_mode: str = "OBSERVE",
    ):
        """
        Args:
            governor: SovereignGovernor instance for state inspection
            enforcement_mode: OBSERVE (default) or GATE
        """
        self.governor = governor
        self.enforcement_mode = enforcement_mode
        
        # Track in-flight tool calls for correlation
        self._inflight: Dict[str, Dict[str, Any]] = {}
    
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts executing."""
        tool_name = (serialized or {}).get("name", "unknown_tool")
        step_id = str(run_id) if run_id else str(uuid.uuid4())
        
        # Classify action (DO NOT include input_str - high cardinality)
        action_kind = classify_action_kind(tool_name, {})
        
        # Get current governor state
        state = self.governor.get_state()
        
        # Build a minimal DiagnosticEvent-like object for projection
        # In production, you'd get this from the actual governor processing
        mock_event = _build_event_from_state(state, step_id)
        
        # Get regime if available
        regime = self._get_regime()
        
        # Project to OTel attrs
        attrs = project_to_otel(
            event=mock_event,
            regime=regime,
            action_kind=action_kind,
            enforcement_mode=self.enforcement_mode,
        )
        
        # Start span (will be closed in on_tool_end)
        ctx = decision_step_span(
            "epistemic.decision_step",
            attrs,
            extra_attrs={
                "epistemic.step_id": step_id,
                "epistemic.phase": "tool_start",
                "tool.name": tool_name[:50],  # Truncate
            },
        )
        
        # Enter context and stash for later
        span = ctx.__enter__()
        self._inflight[step_id] = {
            "tool_name": tool_name,
            "ctx": ctx,
            "span": span,
            "attrs": attrs,
        }
    
    def on_tool_end(
        self,
        output: str,
        *,
        run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool finishes executing."""
        step_id = str(run_id) if run_id else self._get_last_step_id()
        
        if step_id not in self._inflight:
            return
        
        info = self._inflight.pop(step_id)
        ctx = info["ctx"]
        
        # Close the span
        ctx.__exit__(None, None, None)
    
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool errors."""
        step_id = str(run_id) if run_id else self._get_last_step_id()
        
        if step_id not in self._inflight:
            return
        
        info = self._inflight.pop(step_id)
        span = info.get("span")
        ctx = info["ctx"]
        
        # Add error type (low cardinality)
        if span and hasattr(span, "set_attribute"):
            span.set_attribute("epistemic.error.type", type(error).__name__)
        
        # Close the span
        ctx.__exit__(type(error), error, error.__traceback__)
    
    def _get_last_step_id(self) -> Optional[str]:
        """Get most recent step ID (fallback when run_id not provided)."""
        if self._inflight:
            return next(reversed(self._inflight.keys()))
        return None
    
    def _get_regime(self) -> Optional[RegimeAnalysis]:
        """Get current regime from governor if available."""
        try:
            # This depends on your governor having regime detection
            from diagnostics import DiagnosticLogger
            # If governor has a diagnostic logger, use it
            if hasattr(self.governor, '_diagnostic_logger'):
                return self.governor._diagnostic_logger.get_regime()
        except Exception:
            pass
        return None


def _build_event_from_state(state: Dict[str, Any], step_id: str) -> Any:
    """
    Build a minimal event-like object from governor state.
    
    This is a bridge until full DiagnosticEvent integration.
    """
    from dataclasses import dataclass
    from typing import List, Dict
    
    @dataclass
    class MinimalEvent:
        run_id: str
        turn_id: int
        verdict: str
        blocked_by_invariant: List[str]
        c_open_after: int
        c_opened_count: int
        c_closed_count: int
        rho_S_flag: bool
        E_state_after: float
        budget_remaining_after: Dict[str, float]
        budget_exhaustion: Dict[str, bool]
        latency_ms_total: float
        latency_ms_governor: float
    
    # Extract from state dict
    contradictions = state.get("contradictions", {})
    budgets = state.get("budgets", {})
    
    return MinimalEvent(
        run_id=step_id,
        turn_id=state.get("turn_count", 0),
        verdict=state.get("last_verdict", "OK"),
        blocked_by_invariant=state.get("violations", []),
        c_open_after=len(contradictions.get("open", [])),
        c_opened_count=0,  # Not tracked in simple state
        c_closed_count=0,
        rho_S_flag=state.get("state_changed", False),
        E_state_after=state.get("energy", 0.0),
        budget_remaining_after=budgets.get("remaining", {}),
        budget_exhaustion=budgets.get("exhausted", {}),
        latency_ms_total=0.0,
        latency_ms_governor=0.0,
    )


# =============================================================================
# Standalone Test
# =============================================================================

if __name__ == "__main__":
    print("=== LangChain Callback Test ===\n")
    print("This module requires LangChain and a SovereignGovernor instance.")
    print("See instrumentation/demo_agent.py for a working example.")
