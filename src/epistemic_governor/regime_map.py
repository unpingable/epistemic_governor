"""
Regime Mapping: API Surface vs Internal Diagnostics

The epistemic governor uses two regime enums intentionally:

1. api.Regime - User-facing, semantic categories
2. regimes.Regime - Internal diagnostics, fine-grained detection

This module documents the mapping between them.

Why Two Enums?
--------------
- API regime is what users/operators see: "STABLE", "UNSTABLE", "HIGH_STRAIN"
- Internal regime is what the detector classifies: "CONFABULATION", "NARRATIVE_DRIFT", etc.
- Users don't need to know the specific failure mode, just the severity
- Developers/debuggers need the granular classification

The mapping is intentional, not accidental divergence.
"""

from enum import Enum, auto
from typing import Set, Dict

# Import both regime types
from epistemic_governor.api import Regime as APIRegime
from epistemic_governor.regimes import Regime as InternalRegime


# =============================================================================
# Semantic Categories (API-facing)
# =============================================================================

class SemanticCategory(Enum):
    """High-level semantic categories for API consumers."""
    STABLE = auto()      # System operating normally
    GROUNDED = auto()    # Operating with external grounding
    UNCERTAIN = auto()   # Operating but with elevated uncertainty
    UNSTABLE = auto()    # Drift detected, intervention recommended
    HIGH_STRAIN = auto() # High constraint strain, intervention required
    CRITICAL = auto()    # Envelope violation, abort recommended


# =============================================================================
# Mapping: Internal → API
# =============================================================================

INTERNAL_TO_API: Dict[InternalRegime, APIRegime] = {
    # Stable regimes
    InternalRegime.GROUNDED: APIRegime.GROUNDED,
    InternalRegime.INTERROGATIVE: APIRegime.INTERROGATIVE,
    InternalRegime.PROCEDURAL: APIRegime.PROCEDURAL,
    
    # Drift regimes → UNSTABLE variants
    InternalRegime.NARRATIVE_DRIFT: APIRegime.NARRATIVE_DRIFT,
    InternalRegime.FLUENCY_DOMINANCE: APIRegime.FLUENCY_DOMINANCE,
    InternalRegime.SOCIAL_INVENTION: APIRegime.SOCIAL_INVENTION,
    InternalRegime.ROLEPLAY_CAPTURE: APIRegime.ROLEPLAY_CAPTURE,
    InternalRegime.COMMITMENT_DECAY: APIRegime.COMMITMENT_DECAY,
    InternalRegime.CONFABULATION: APIRegime.CONFABULATION,
    InternalRegime.ASSOCIATIVE_SPIRAL: APIRegime.ASSOCIATIVE_SPIRAL,
    
    # Critical regimes
    InternalRegime.THERMAL_SHUTDOWN: APIRegime.THERMAL_RUNAWAY,
    InternalRegime.FURNACE: APIRegime.THERMAL_RUNAWAY,
}

# Regimes that map to STABLE (no specific API equivalent)
STABLE_REGIMES: Set[InternalRegime] = {
    InternalRegime.GROUNDED,
    InternalRegime.INTERROGATIVE,
    InternalRegime.PROCEDURAL,
}

# Regimes that indicate drift (intervention recommended)
DRIFT_REGIMES: Set[InternalRegime] = {
    InternalRegime.NARRATIVE_DRIFT,
    InternalRegime.FLUENCY_DOMINANCE,
    InternalRegime.SOCIAL_INVENTION,
    InternalRegime.ROLEPLAY_CAPTURE,
    InternalRegime.COMMITMENT_DECAY,
    InternalRegime.CONFABULATION,
    InternalRegime.ASSOCIATIVE_SPIRAL,
}

# Regimes that indicate critical state (intervention required)
CRITICAL_REGIMES: Set[InternalRegime] = {
    InternalRegime.THERMAL_SHUTDOWN,
    InternalRegime.FURNACE,
}


# =============================================================================
# Mapping: API → Semantic Category
# =============================================================================

API_TO_CATEGORY: Dict[APIRegime, SemanticCategory] = {
    APIRegime.STABLE: SemanticCategory.STABLE,
    APIRegime.GROUNDED: SemanticCategory.GROUNDED,
    APIRegime.INTERROGATIVE: SemanticCategory.GROUNDED,
    APIRegime.PROCEDURAL: SemanticCategory.GROUNDED,
    
    APIRegime.NARRATIVE_DRIFT: SemanticCategory.UNSTABLE,
    APIRegime.FLUENCY_DOMINANCE: SemanticCategory.UNSTABLE,
    APIRegime.SOCIAL_INVENTION: SemanticCategory.UNSTABLE,
    APIRegime.ROLEPLAY_CAPTURE: SemanticCategory.UNSTABLE,
    APIRegime.COMMITMENT_DECAY: SemanticCategory.UNSTABLE,
    APIRegime.CONFABULATION: SemanticCategory.HIGH_STRAIN,
    APIRegime.ASSOCIATIVE_SPIRAL: SemanticCategory.HIGH_STRAIN,
    
    APIRegime.THERMAL_RUNAWAY: SemanticCategory.CRITICAL,
    APIRegime.DEAD_END: SemanticCategory.CRITICAL,
}


# =============================================================================
# Helper Functions
# =============================================================================

def internal_to_api(regime: InternalRegime) -> APIRegime:
    """Convert internal regime to API regime."""
    return INTERNAL_TO_API.get(regime, APIRegime.STABLE)


def api_to_category(regime: APIRegime) -> SemanticCategory:
    """Convert API regime to semantic category."""
    return API_TO_CATEGORY.get(regime, SemanticCategory.UNCERTAIN)


def is_stable(regime: InternalRegime) -> bool:
    """Check if internal regime is stable."""
    return regime in STABLE_REGIMES


def is_drift(regime: InternalRegime) -> bool:
    """Check if internal regime indicates drift."""
    return regime in DRIFT_REGIMES


def is_critical(regime: InternalRegime) -> bool:
    """Check if internal regime is critical."""
    return regime in CRITICAL_REGIMES


def get_severity(regime: InternalRegime) -> int:
    """
    Get severity level (0-3) for a regime.
    
    0 = stable
    1 = uncertain
    2 = unstable (drift)
    3 = critical
    """
    if regime in STABLE_REGIMES:
        return 0
    elif regime in DRIFT_REGIMES:
        return 2
    elif regime in CRITICAL_REGIMES:
        return 3
    else:
        return 1


# =============================================================================
# Documentation Table
# =============================================================================

REGIME_DOCUMENTATION = """
Regime Mapping Table
====================

Internal (Diagnostic)    | API (User-Facing)      | Category     | Severity
-------------------------|------------------------|--------------|----------
GROUNDED                 | GROUNDED               | GROUNDED     | 0 (stable)
INTERROGATIVE            | INTERROGATIVE          | GROUNDED     | 0
PROCEDURAL               | PROCEDURAL             | GROUNDED     | 0
NARRATIVE_DRIFT          | NARRATIVE_DRIFT        | UNSTABLE     | 2 (drift)
FLUENCY_DOMINANCE        | FLUENCY_DOMINANCE      | UNSTABLE     | 2
SOCIAL_INVENTION         | SOCIAL_INVENTION       | UNSTABLE     | 2
ROLEPLAY_CAPTURE         | ROLEPLAY_CAPTURE       | UNSTABLE     | 2
COMMITMENT_DECAY         | COMMITMENT_DECAY       | UNSTABLE     | 2
CONFABULATION            | CONFABULATION          | HIGH_STRAIN  | 2
ASSOCIATIVE_SPIRAL       | ASSOCIATIVE_SPIRAL     | HIGH_STRAIN  | 2
THERMAL_SHUTDOWN         | THERMAL_RUNAWAY        | CRITICAL     | 3 (critical)
FURNACE                  | THERMAL_RUNAWAY        | CRITICAL     | 3

Notes:
- FURNACE and THERMAL_SHUTDOWN both map to THERMAL_RUNAWAY (same severity)
- API has STABLE and DEAD_END which don't have direct internal equivalents
- Internal regimes provide diagnostic granularity
- API regimes provide operational clarity
"""


if __name__ == "__main__":
    print(REGIME_DOCUMENTATION)
    
    print("\nExample conversions:")
    for internal in InternalRegime:
        api = internal_to_api(internal)
        category = api_to_category(api)
        severity = get_severity(internal)
        print(f"  {internal.name:20} → {api.name:20} → {category.name:12} (severity {severity})")
