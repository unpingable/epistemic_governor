"""
Pedagogical Jurisdiction

Simplification without corruption.

- Incomplete models allowed
- Known falsehoods must be flagged as approximations
- Contradictions allowed only if pedagogically necessary
- Explicit "lie-to-children" tags

This would fix so much educational misuse.
"""

from . import (
    Jurisdiction, EvidenceType, SpilloverPolicy,
    ContradictionPolicy, BudgetProfile
)


def PedagogicalJurisdiction() -> Jurisdiction:
    """
    Teaching/pedagogy mode.
    
    Deliberate simplifications permitted but flagged.
    Approximations marked. Lie-to-children acknowledged.
    """
    return Jurisdiction(
        name="pedagogical",
        description="Teaching mode. Simplifications flagged. Approximations explicit.",
        
        admissible_evidence={
            EvidenceType.TOOL_OUTPUT,
            EvidenceType.EXTERNAL_DOCUMENT,
            EvidenceType.PEDAGOGICAL_FRAME,  # KEY: Simplification is evidence
            EvidenceType.USER_ASSERTION,
        },
        
        budget=BudgetProfile(
            claim_cost=0.5,           # Teaching should be easy
            contradiction_cost=2.0,   # But contradictions matter
            resolution_cost=3.0,      # Can resolve with better explanation
            export_cost=15.0,         # Expensive - must de-simplify
            refill_rate=2.0,
        ),
        
        spillover=SpilloverPolicy.FLAGGED_EXPORT,  # Must mark as simplified
        
        contradiction_policy=ContradictionPolicy.TOLERANT,
        contradiction_tolerance=0.5,  # Some pedagogical contradictions OK
        
        closure_allowed=True,
        closure_requires_evidence=False,  # Can close with pedagogical frame
        
        export_to_factual_allowed=True,
        export_requires_promotion=True,  # Must remove simplifications
        
        output_label="[PEDAGOGICAL - SIMPLIFIED]",
    )
