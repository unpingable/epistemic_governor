"""
Counterfactual Jurisdiction

Deliberate falsity under containment.

- Truth is suspended by design
- Internally consistent worlds allowed
- Ledger is write-only but world-scoped
- No cross-world commits without translation + evidence

Prevents the classic failure mode:
"In a hypothetical scenario..." → three turns later → "As established earlier..."
"""

from . import (
    Jurisdiction, EvidenceType, SpilloverPolicy,
    ContradictionPolicy, BudgetProfile
)


def CounterfactualJurisdiction() -> Jurisdiction:
    """
    Counterfactual/alternate world mode.
    
    Deliberate falsity is permitted within world scope.
    Cross-world contamination is blocked.
    """
    return Jurisdiction(
        name="counterfactual",
        description="Alternate world reasoning. Truth suspended. Containment enforced.",
        
        admissible_evidence={
            EvidenceType.NARRATIVE_CONSISTENCY,  # Internal logic matters
            EvidenceType.USER_ASSERTION,          # User defines the world
        },
        
        budget=BudgetProfile(
            claim_cost=0.3,           # Very cheap to build worlds
            contradiction_cost=1.0,   # Cross-world contradictions matter
            resolution_cost=2.0,      # Can resolve within world
            export_cost=50.0,         # VERY expensive to export
            refill_rate=3.0,          # Fast iteration
        ),
        
        spillover=SpilloverPolicy.BLOCKED,  # KEY: No leakage to factual
        
        contradiction_policy=ContradictionPolicy.SCOPED,  # Only same-world
        contradiction_tolerance=0.5,
        
        closure_allowed=True,  # Can close within the counterfactual
        closure_requires_evidence=False,  # World-internal consistency suffices
        
        export_to_factual_allowed=False,  # KEY: Cannot contaminate factual
        export_requires_promotion=True,
        
        output_label="[COUNTERFACTUAL]",
    )
