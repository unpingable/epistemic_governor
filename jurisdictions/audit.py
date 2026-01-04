"""
Audit Jurisdiction

Inspect the system itself.

- Outputs include provenance and decision traces
- No new claims allowed
- Only explanation and state reporting
- Used for debugging and trust repair

You already built half of this with the diagnostics module.
"""

from . import (
    Jurisdiction, EvidenceType, SpilloverPolicy,
    ContradictionPolicy, BudgetProfile
)


def AuditJurisdiction() -> Jurisdiction:
    """
    Meta-reasoning/audit mode.
    
    Inspection only. No new claims.
    Reports state, provenance, decision traces.
    """
    return Jurisdiction(
        name="audit",
        description="Inspection mode. No new claims. State reporting only.",
        
        admissible_evidence={
            # Audit doesn't take evidence - it produces reports
        },
        
        budget=BudgetProfile(
            claim_cost=1000.0,        # KEY: Cannot make claims
            contradiction_cost=1000.0, # Cannot open contradictions
            resolution_cost=1000.0,    # Cannot resolve
            export_cost=1000.0,        # Cannot export
            refill_rate=0.0,           # No budget recovery
        ),
        
        spillover=SpilloverPolicy.BLOCKED,
        
        contradiction_policy=ContradictionPolicy.STRICT,
        contradiction_tolerance=0.0,
        
        closure_allowed=False,  # KEY: Read-only mode
        closure_requires_evidence=True,
        
        export_to_factual_allowed=False,
        export_requires_promotion=True,
        
        output_label="[AUDIT - READ ONLY]",
    )
