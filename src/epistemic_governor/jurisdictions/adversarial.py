"""
Adversarial Jurisdiction

Argue the wrong thing on purpose.

- Claims are marked adversarial
- Contradictions are expected
- Resolution is forbidden
- Output explicitly labeled as stress input, not belief

This is how you get strong reasoning without letting 
the system believe its own bullshit.
"""

from . import (
    Jurisdiction, EvidenceType, SpilloverPolicy,
    ContradictionPolicy, BudgetProfile
)


def AdversarialJurisdiction() -> Jurisdiction:
    """
    Devil's advocate mode.
    
    Deliberately argue against positions.
    Contradictions are the point, not a bug.
    No resolution allowed - this is stress testing.
    """
    return Jurisdiction(
        name="adversarial",
        description="Devil's advocate. Contradictions expected. Resolution forbidden.",
        
        admissible_evidence={
            # Almost anything can be used as adversarial fodder
            EvidenceType.TOOL_OUTPUT,
            EvidenceType.USER_ASSERTION,
            EvidenceType.EXTERNAL_DOCUMENT,
        },
        
        budget=BudgetProfile(
            claim_cost=0.1,           # Nearly free to argue
            contradiction_cost=0.1,   # Contradictions are the point
            resolution_cost=100.0,    # KEY: Resolution is blocked by cost
            export_cost=100.0,        # Cannot export adversarial claims
            refill_rate=5.0,          # High throughput for stress testing
            adversarial_discount=0.1,
        ),
        
        spillover=SpilloverPolicy.BLOCKED,
        
        contradiction_policy=ContradictionPolicy.EXPECTED,  # Normal operation
        contradiction_tolerance=1.0,  # Full tolerance
        
        closure_allowed=False,  # KEY: Cannot resolve in adversarial mode
        closure_requires_evidence=True,
        
        export_to_factual_allowed=False,  # Cannot contaminate
        export_requires_promotion=True,
        
        output_label="[ADVERSARIAL - NOT ENDORSED]",
    )
