"""
Forensic Jurisdiction

Reconstruction, not narration.

- Only evidence-backed claims allowed
- High contradiction sensitivity
- Closure requires cross-source corroboration
- No speculative repair

This is journalism and incident response done correctly.
"""

from . import (
    Jurisdiction, EvidenceType, SpilloverPolicy,
    ContradictionPolicy, BudgetProfile
)


def ForensicJurisdiction() -> Jurisdiction:
    """
    Forensic/postmortem mode.
    
    Maximum evidence requirements.
    Cross-source corroboration required for closure.
    """
    return Jurisdiction(
        name="forensic",
        description="Investigation mode. Evidence-only. Cross-source required.",
        
        admissible_evidence={
            EvidenceType.TOOL_OUTPUT,
            EvidenceType.SENSOR_DATA,
            EvidenceType.EXTERNAL_DOCUMENT,
            EvidenceType.CRYPTOGRAPHIC_PROOF,
            EvidenceType.CROSS_SOURCE,  # KEY: Corroboration required
        },
        
        budget=BudgetProfile(
            claim_cost=3.0,           # Expensive to claim
            contradiction_cost=1.0,   # Contradictions are signal
            resolution_cost=10.0,     # Very expensive to close
            export_cost=5.0,          # Forensic findings can export
            refill_rate=0.5,          # Slow, careful work
        ),
        
        spillover=SpilloverPolicy.FLAGGED_EXPORT,  # Can export with provenance
        
        contradiction_policy=ContradictionPolicy.STRICT,
        contradiction_tolerance=0.0,  # Zero tolerance
        
        closure_allowed=True,
        closure_requires_evidence=True,  # KEY: Must have corroboration
        
        export_to_factual_allowed=True,
        export_requires_promotion=True,  # Requires review
        
        output_label="[FORENSIC - EVIDENCE REQUIRED]",
    )
