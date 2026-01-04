"""
Jurisdictions: Mode-Separated Epistemic Governance

A jurisdiction defines the epistemic rules for a particular reasoning context.
Different contexts have different:
- Evidence admissibility rules (what counts as evidence)
- Budget profiles (what operations cost)
- Spillover policies (what can cross boundaries)
- Contradiction tolerance (when conflicts matter)
- Closure rules (whether/how claims can be committed)

This is not "adding features" - it's recognizing that human cognition already
operates in mode-separated regimes, and we rely on social context to enforce
boundaries. LLMs collapse all of that into one slurry unless you stop them.

Each jurisdiction is a parameter configuration of the base governor.
The architecture doesn't change; the rules do.

STATUS: STUB - Pattern definition only. Not yet integrated with governor.
"""

from dataclasses import dataclass, field
from typing import Dict, Set, Optional, List
from enum import Enum, auto


class EvidenceType(Enum):
    """Types of evidence that might be admissible."""
    TOOL_OUTPUT = auto()           # Deterministic computation
    SENSOR_DATA = auto()           # Physical measurement
    USER_ASSERTION = auto()        # Human attestation with identity
    EXTERNAL_DOCUMENT = auto()     # Retrievable artifact with hash
    CRYPTOGRAPHIC_PROOF = auto()   # Verifiable computation
    SUBJECTIVE_REPORT = auto()     # First-person experience (therapy mode)
    NARRATIVE_CONSISTENCY = auto() # Story-internal logic (fiction mode)
    PEDAGOGICAL_FRAME = auto()     # Known simplification (teaching mode)
    CROSS_SOURCE = auto()          # Corroboration requirement (forensic mode)


class SpilloverPolicy(Enum):
    """How claims can cross jurisdiction boundaries."""
    BLOCKED = auto()          # No export allowed
    PROMOTED_WITH_EVIDENCE = auto()  # Requires evidence to export
    FLAGGED_EXPORT = auto()   # Can export but marked with origin
    FREE = auto()             # No restrictions (dangerous)


class ContradictionPolicy(Enum):
    """How contradictions are handled in this jurisdiction."""
    STRICT = auto()           # Contradictions block operations
    TOLERANT = auto()         # Contradictions logged but don't block
    EXPECTED = auto()         # Contradictions are normal (adversarial mode)
    SCOPED = auto()           # Only same-scope contradictions matter


@dataclass
class BudgetProfile:
    """Cost structure for operations in this jurisdiction."""
    claim_cost: float = 1.0           # Cost to assert a claim
    contradiction_cost: float = 2.0   # Cost to open a contradiction
    resolution_cost: float = 5.0      # Cost to close a contradiction
    export_cost: float = 10.0         # Cost to export to factual jurisdiction
    refill_rate: float = 1.0          # Budget recovery per turn
    
    # Multipliers for different operation types
    speculative_discount: float = 0.5  # Cheaper to speculate
    adversarial_discount: float = 0.1  # Very cheap to play devil's advocate


@dataclass
class Jurisdiction:
    """
    Base jurisdiction configuration.
    
    Each jurisdiction defines the epistemic rules for a reasoning context.
    Subclasses override defaults to create specific modes.
    """
    name: str
    description: str
    
    # Evidence admissibility
    admissible_evidence: Set[EvidenceType] = field(default_factory=lambda: {
        EvidenceType.TOOL_OUTPUT,
        EvidenceType.SENSOR_DATA,
        EvidenceType.USER_ASSERTION,
        EvidenceType.EXTERNAL_DOCUMENT,
        EvidenceType.CRYPTOGRAPHIC_PROOF,
    })
    
    # Budget configuration
    budget: BudgetProfile = field(default_factory=BudgetProfile)
    
    # Spillover policy
    spillover: SpilloverPolicy = SpilloverPolicy.BLOCKED
    
    # Contradiction handling
    contradiction_policy: ContradictionPolicy = ContradictionPolicy.STRICT
    contradiction_tolerance: float = 0.0  # 0 = no tolerance, 1 = full tolerance
    
    # Closure rules
    closure_allowed: bool = True
    closure_requires_evidence: bool = True
    
    # Export rules
    export_to_factual_allowed: bool = False
    export_requires_promotion: bool = True
    
    # Labeling
    output_label: Optional[str] = None  # e.g., "[SPECULATIVE]", "[FICTION]"
    
    def admits(self, evidence_type: EvidenceType) -> bool:
        """Check if this evidence type is admissible."""
        return evidence_type in self.admissible_evidence
    
    def can_close(self, has_evidence: bool) -> bool:
        """Check if closure is allowed given evidence status."""
        if not self.closure_allowed:
            return False
        if self.closure_requires_evidence and not has_evidence:
            return False
        return True
    
    def can_export(self, has_promotion_evidence: bool) -> bool:
        """Check if claims can be exported to factual jurisdiction."""
        if not self.export_to_factual_allowed:
            return False
        if self.export_requires_promotion and not has_promotion_evidence:
            return False
        return True


# =============================================================================
# Jurisdiction Registry
# =============================================================================

_JURISDICTIONS: Dict[str, Jurisdiction] = {}


def register_jurisdiction(jurisdiction: Jurisdiction) -> None:
    """Register a jurisdiction for use."""
    _JURISDICTIONS[jurisdiction.name] = jurisdiction


def get_jurisdiction(name: str) -> Optional[Jurisdiction]:
    """Get a registered jurisdiction by name."""
    return _JURISDICTIONS.get(name)


def list_jurisdictions() -> List[str]:
    """List all registered jurisdiction names."""
    return list(_JURISDICTIONS.keys())


# =============================================================================
# Import all jurisdiction definitions
# =============================================================================

from .factual import FactualJurisdiction
from .speculative import SpeculativeJurisdiction
from .counterfactual import CounterfactualJurisdiction
from .adversarial import AdversarialJurisdiction
from .narrative import NarrativeJurisdiction
from .forensic import ForensicJurisdiction
from .pedagogical import PedagogicalJurisdiction
from .audit import AuditJurisdiction

# Register default jurisdictions
register_jurisdiction(FactualJurisdiction())
register_jurisdiction(SpeculativeJurisdiction())
register_jurisdiction(CounterfactualJurisdiction())
register_jurisdiction(AdversarialJurisdiction())
register_jurisdiction(NarrativeJurisdiction())
register_jurisdiction(ForensicJurisdiction())
register_jurisdiction(PedagogicalJurisdiction())
register_jurisdiction(AuditJurisdiction())
