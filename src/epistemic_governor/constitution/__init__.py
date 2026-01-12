"""
Constitution - S₀ Immutable Law

This package contains the unchangeable rules of the governor.
Nothing here may be modified at runtime.

Contents:
- contracts: Interface boundary definitions
- forbidden: Inadmissible actions and transitions
- nlai: Non-Linguistic Authority Invariant enforcement
"""

from epistemic_governor.constitution.contracts import (
    BoundaryGate,
    InputContract,
    OutputContract,
    ControlContract,
    CrossingVerdict,
    is_inadmissible,
    INADMISSIBLE_ACTIONS,
)

__all__ = [
    "BoundaryGate",
    "InputContract",
    "OutputContract", 
    "ControlContract",
    "CrossingVerdict",
    "is_inadmissible",
    "INADMISSIBLE_ACTIONS",
]
