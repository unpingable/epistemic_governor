"""
Integrations - External Adapters

This package contains adapters for external systems.
These are intentionally "dumb" - they translate between external
protocols and the governor's internal interface.

IMPORTANT: Integrations do not grant authority. They are I/O adapters only.

Contents:
- langchain: LangChain callback handler
- demo: Demo agent for testing
- mcp: MCP server adapter (stub)
"""

from epistemic_governor.integrations.langchain import EpistemicCallback
from epistemic_governor.integrations.demo import run_demo

__all__ = [
    "EpistemicCallback",
    "run_demo",
]
