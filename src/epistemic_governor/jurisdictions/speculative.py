"""
Speculative Jurisdiction

"What if..." without commitment.

- Claims are explicitly provisional
- Contradictions are allowed but tagged as speculative
- No closure allowed
- Leakage into factual domains blocked unless promoted with evidence

This is the mode academics think they're in, but systems never are.
"""

from . import (
    Jurisdiction, EvidenceType, SpilloverPolicy,
    ContradictionPolicy, BudgetProfile
)


def SpeculativeJurisdiction() -> Jurisdiction:
    """
    Speculative/hypothesis mode.
    
    Provisional claims, no closure, explicit labeling.
    Export to factual requires evidence promotion.
    """
    return Jurisdiction(
        name="speculative",
        description="Provisional claims. No closure. Exploration mode.",
        
        admissible_evidence={
            EvidenceType.TOOL_OUTPUT,
            EvidenceType.USER_ASSERTION,
            # Lower bar - even weak evidence can inform speculation
        },
        
        budget=BudgetProfile(
            claim_cost=0.5,           # Cheap to speculate
            contradiction_cost=0.5,   # Contradictions are fine
            resolution_cost=0.0,      # Can't resolve anyway
            export_cost=10.0,         # Expensive to promote
            refill_rate=2.0,          # Fast recovery for exploration
            speculative_discount=0.5,
        ),
        
        spillover=SpilloverPolicy.PROMOTED_WITH_EVIDENCE,
        
        contradiction_policy=ContradictionPolicy.TOLERANT,
        contradiction_tolerance=0.8,  # High tolerance
        
        closure_allowed=False,  # KEY: Cannot commit in speculation mode
        closure_requires_evidence=True,
        
        export_to_factual_allowed=True,
        export_requires_promotion=True,  # Must have evidence to promote
        
        output_label="[SPECULATIVE]",
    )
