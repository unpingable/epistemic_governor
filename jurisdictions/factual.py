"""
Factual Jurisdiction

The default, high-evidence, low-tolerance mode.
This is what the current governor implements.

- Only hard evidence admitted
- Contradictions block operations
- Closure requires evidence
- No export needed (this IS the factual record)
"""

from . import (
    Jurisdiction, EvidenceType, SpilloverPolicy, 
    ContradictionPolicy, BudgetProfile
)


def FactualJurisdiction() -> Jurisdiction:
    """
    Default factual jurisdiction.
    
    High evidence bar, strict contradiction handling,
    no tolerance for ungrounded claims.
    """
    return Jurisdiction(
        name="factual",
        description="Strict epistemic mode. Evidence required. Contradictions block.",
        
        admissible_evidence={
            EvidenceType.TOOL_OUTPUT,
            EvidenceType.SENSOR_DATA,
            EvidenceType.USER_ASSERTION,
            EvidenceType.EXTERNAL_DOCUMENT,
            EvidenceType.CRYPTOGRAPHIC_PROOF,
        },
        
        budget=BudgetProfile(
            claim_cost=1.0,
            contradiction_cost=2.0,
            resolution_cost=5.0,
            export_cost=0.0,  # Already factual
            refill_rate=1.0,
        ),
        
        spillover=SpilloverPolicy.FREE,  # Factual claims can go anywhere
        
        contradiction_policy=ContradictionPolicy.STRICT,
        contradiction_tolerance=0.0,
        
        closure_allowed=True,
        closure_requires_evidence=True,
        
        export_to_factual_allowed=True,  # Already factual
        export_requires_promotion=False,
        
        output_label=None,  # No label needed for factual
    )
