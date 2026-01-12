"""
Narrative Jurisdiction

Story logic, not world logic.

- Internal consistency matters
- External truth does not
- Contradictions only count if they break story continuity
- No factual commitments exported

A clean separation between diegetic truth and ledger truth.
"""

from . import (
    Jurisdiction, EvidenceType, SpilloverPolicy,
    ContradictionPolicy, BudgetProfile
)


def NarrativeJurisdiction() -> Jurisdiction:
    """
    Fiction/narrative mode.
    
    Story-internal logic is the only constraint.
    External truth is irrelevant.
    """
    return Jurisdiction(
        name="narrative",
        description="Fiction mode. Story consistency only. No factual claims.",
        
        admissible_evidence={
            EvidenceType.NARRATIVE_CONSISTENCY,  # KEY: Only story logic
            EvidenceType.USER_ASSERTION,          # Author authority
        },
        
        budget=BudgetProfile(
            claim_cost=0.2,           # Cheap to create story elements
            contradiction_cost=2.0,   # Story contradictions matter
            resolution_cost=1.0,      # Easy to fix story issues
            export_cost=1000.0,       # Effectively impossible
            refill_rate=3.0,
        ),
        
        spillover=SpilloverPolicy.BLOCKED,
        
        contradiction_policy=ContradictionPolicy.SCOPED,  # Story-internal only
        contradiction_tolerance=0.3,  # Some tolerance for narrative flexibility
        
        closure_allowed=True,  # Can establish story facts
        closure_requires_evidence=False,  # Narrative consistency suffices
        
        export_to_factual_allowed=False,  # KEY: Fiction stays fiction
        export_requires_promotion=True,
        
        output_label="[FICTION]",
    )
