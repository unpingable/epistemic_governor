"""
Epistemic Module - Invariants on Assertions over Symbols

This module registers epistemic constraints with the governor's module registry.
It provides the core invariants for claim commitment:

- Confidence ceiling (clamp overconfident claims)
- Thermal regime (block/hedge based on instability)
- Contradiction detection (veto contradictory claims)
- Support requirements (defer ungrounded claims)
- Revision cost (track thermal cost of changes)

Usage:
    from epistemic_governor.epistemic_module import register_epistemic_invariants
    
    registry = create_registry()
    register_epistemic_invariants(registry, config=EpistemicConfig())
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from datetime import datetime

# Handle both package and direct imports
try:
    from .registry import (
        ModuleRegistry,
        Domain,
        InvariantAction,
        ProposalEnvelope,
        StateView,
        InvariantResult,
        Invariant,
    )
    from .governor import ClaimType, CommitmentStatus
except ImportError:
    from registry import (
        ModuleRegistry,
        Domain,
        InvariantAction,
        ProposalEnvelope,
        StateView,
        InvariantResult,
        Invariant,
    )
    from governor import ClaimType, CommitmentStatus


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EpistemicConfig:
    """
    Configuration for epistemic invariants.
    
    These are the tunable parameters that control how strict
    the epistemic module is.
    """
    # Confidence ceilings by claim type
    confidence_ceilings: Dict[str, float] = field(default_factory=lambda: {
        "FACTUAL": 0.85,
        "CAUSAL": 0.75,
        "PREDICTIVE": 0.65,
        "IDENTITY": 0.90,
        "COMPARATIVE": 0.80,
        "QUANTITATIVE": 0.70,
        "TEMPORAL": 0.75,
        "ATTRIBUTION": 0.80,
        "PROCEDURAL": 0.85,
    })
    
    # Default ceiling for unknown types
    default_confidence_ceiling: float = 0.80
    
    # Thermal thresholds
    warning_instability: float = 0.3     # Start clamping more aggressively
    critical_instability: float = 0.7    # Require evidence for commits
    shutdown_instability: float = 1.5    # Block all new claims
    
    # Support requirements
    require_support_above_confidence: float = 0.7  # Require evidence above this
    require_support_for_types: List[str] = field(default_factory=lambda: [
        "QUANTITATIVE", "ATTRIBUTION", "CAUSAL"
    ])
    
    # Revision costs
    revision_base_heat: float = 0.2
    contradiction_heat: float = 0.1
    
    # Contradiction detection
    block_contradictions: bool = True
    
    def get_ceiling(self, claim_type: str) -> float:
        """Get confidence ceiling for a claim type."""
        return self.confidence_ceilings.get(claim_type, self.default_confidence_ceiling)


# =============================================================================
# Invariants
# =============================================================================

class ConfidenceCeilingInvariant:
    """
    Clamp confidence to type-specific ceilings.
    
    Different claim types have different maximum confidence levels.
    FACTUAL claims can be higher, PREDICTIVE claims must be lower.
    """
    
    def __init__(self, config: EpistemicConfig):
        self.config = config
    
    @property
    def id(self) -> str:
        return "epistemic.confidence_ceiling"
    
    @property
    def name(self) -> str:
        return "Confidence Ceiling"
    
    @property
    def domain(self) -> Domain:
        return Domain.EPISTEMIC
    
    @property
    def domains(self) -> List[Domain]:
        return [Domain.EPISTEMIC]
    
    def check(self, proposal: ProposalEnvelope, state: StateView) -> InvariantResult:
        # Get claim type from payload
        claim_type = proposal.payload.get("claim_type", "FACTUAL")
        ceiling = self.config.get_ceiling(claim_type)
        
        if proposal.confidence > ceiling:
            return InvariantResult(
                invariant_id=self.id,
                invariant_name=self.name,
                domain=self.domain,
                passed=False,
                action=InvariantAction.CLAMP,
                code="CONFIDENCE_EXCEEDED",
                reason=f"Confidence {proposal.confidence:.2f} exceeds {claim_type} ceiling {ceiling:.2f}",
                proposal_delta={"confidence": ceiling * 0.95},  # Clamp slightly below ceiling
            )
        
        return InvariantResult(
            invariant_id=self.id,
            invariant_name=self.name,
            domain=self.domain,
            passed=True,
            action=InvariantAction.PASS,
        )


class ThermalRegimeInvariant:
    """
    Adjust behavior based on thermal state (instability).
    
    - Normal: PASS
    - Warning: CLAMP confidence down
    - Critical: DEFER unless supported
    - Shutdown: VETO all new claims
    """
    
    def __init__(self, config: EpistemicConfig):
        self.config = config
    
    @property
    def id(self) -> str:
        return "epistemic.thermal_regime"
    
    @property
    def name(self) -> str:
        return "Thermal Regime"
    
    @property
    def domain(self) -> Domain:
        return Domain.EPISTEMIC
    
    @property
    def domains(self) -> List[Domain]:
        return [Domain.EPISTEMIC]
    
    def check(self, proposal: ProposalEnvelope, state: StateView) -> InvariantResult:
        instability = state.instability
        
        # Shutdown: block everything
        if instability >= self.config.shutdown_instability:
            return InvariantResult(
                invariant_id=self.id,
                invariant_name=self.name,
                domain=self.domain,
                passed=False,
                action=InvariantAction.VETO,
                code="THERMAL_SHUTDOWN",
                reason=f"Instability {instability:.2f} >= shutdown threshold {self.config.shutdown_instability:.2f}",
                heat_delta=0.0,  # No additional heat for blocked claims
            )
        
        # Critical: require support
        if instability >= self.config.critical_instability:
            if not proposal.evidence_refs and not proposal.anchor_refs:
                return InvariantResult(
                    invariant_id=self.id,
                    invariant_name=self.name,
                    domain=self.domain,
                    passed=False,
                    action=InvariantAction.DEFER,
                    code="THERMAL_CRITICAL_UNSUPPORTED",
                    reason=f"Critical instability ({instability:.2f}) requires evidence",
                    required_evidence=["grounding_anchor", "source_citation"],
                )
            # Has support, but clamp confidence
            return InvariantResult(
                invariant_id=self.id,
                invariant_name=self.name,
                domain=self.domain,
                passed=False,
                action=InvariantAction.CLAMP,
                code="THERMAL_CRITICAL_CLAMPED",
                reason=f"Critical instability - clamping confidence",
                proposal_delta={"confidence": min(proposal.confidence, 0.5)},
            )
        
        # Warning: clamp confidence
        if instability >= self.config.warning_instability:
            max_conf = 0.7
            if proposal.confidence > max_conf:
                return InvariantResult(
                    invariant_id=self.id,
                    invariant_name=self.name,
                    domain=self.domain,
                    passed=False,
                    action=InvariantAction.CLAMP,
                    code="THERMAL_WARNING_CLAMPED",
                    reason=f"Warning instability - clamping confidence to {max_conf}",
                    proposal_delta={"confidence": max_conf},
                )
        
        # Normal: pass
        return InvariantResult(
            invariant_id=self.id,
            invariant_name=self.name,
            domain=self.domain,
            passed=True,
            action=InvariantAction.PASS,
        )


class ContradictionInvariant:
    """
    Detect and block contradictory claims.
    
    A claim contradicts if it asserts the opposite of an existing
    active claim without being a proper revision.
    """
    
    def __init__(self, config: EpistemicConfig):
        self.config = config
        # In practice, this would use the ledger's proposition hashes
        # For now, we check payload for contradiction markers
    
    @property
    def id(self) -> str:
        return "epistemic.contradiction"
    
    @property
    def name(self) -> str:
        return "Contradiction Detection"
    
    @property
    def domain(self) -> Domain:
        return Domain.EPISTEMIC
    
    @property
    def domains(self) -> List[Domain]:
        return [Domain.EPISTEMIC]
    
    def check(self, proposal: ProposalEnvelope, state: StateView) -> InvariantResult:
        if not self.config.block_contradictions:
            return InvariantResult(
                invariant_id=self.id,
                invariant_name=self.name,
                domain=self.domain,
                passed=True,
                action=InvariantAction.PASS,
            )
        
        # Check if this is a revision (supersedes existing claim)
        supersedes = proposal.payload.get("supersedes")
        if supersedes:
            # Revisions are allowed but cost heat
            return InvariantResult(
                invariant_id=self.id,
                invariant_name=self.name,
                domain=self.domain,
                passed=True,
                action=InvariantAction.PASS,
                heat_delta=self.config.revision_base_heat * proposal.confidence,
            )
        
        # Check for contradiction markers in payload
        contradicts = proposal.payload.get("contradicts")
        if contradicts:
            return InvariantResult(
                invariant_id=self.id,
                invariant_name=self.name,
                domain=self.domain,
                passed=False,
                action=InvariantAction.VETO,
                code="CONTRADICTION_DETECTED",
                reason=f"Contradicts existing claim: {contradicts}",
                heat_delta=self.config.contradiction_heat,
            )
        
        # Check against active claims (simplified - would use hash in practice)
        proposition_hash = proposal.payload.get("proposition_hash")
        if proposition_hash:
            # Check if negation exists
            negation_hash = f"NOT_{proposition_hash}"
            if negation_hash in state.active_claims:
                return InvariantResult(
                    invariant_id=self.id,
                    invariant_name=self.name,
                    domain=self.domain,
                    passed=False,
                    action=InvariantAction.VETO,
                    code="PROPOSITION_CONTRADICTION",
                    reason=f"Negation of this proposition already committed",
                    heat_delta=self.config.contradiction_heat,
                )
        
        return InvariantResult(
            invariant_id=self.id,
            invariant_name=self.name,
            domain=self.domain,
            passed=True,
            action=InvariantAction.PASS,
        )


class SupportRequirementInvariant:
    """
    Require evidence for high-confidence or risky claim types.
    
    Some claims (quantitative, attribution, causal) require
    grounding evidence before they can be committed.
    """
    
    def __init__(self, config: EpistemicConfig):
        self.config = config
    
    @property
    def id(self) -> str:
        return "epistemic.support_requirement"
    
    @property
    def name(self) -> str:
        return "Support Requirement"
    
    @property
    def domain(self) -> Domain:
        return Domain.EPISTEMIC
    
    @property
    def domains(self) -> List[Domain]:
        return [Domain.EPISTEMIC]
    
    def check(self, proposal: ProposalEnvelope, state: StateView) -> InvariantResult:
        claim_type = proposal.payload.get("claim_type", "FACTUAL")
        has_support = bool(proposal.evidence_refs or proposal.anchor_refs)
        
        # Check if claim type requires support
        if claim_type in self.config.require_support_for_types:
            if not has_support:
                return InvariantResult(
                    invariant_id=self.id,
                    invariant_name=self.name,
                    domain=self.domain,
                    passed=False,
                    action=InvariantAction.DEFER,
                    code="SUPPORT_REQUIRED_FOR_TYPE",
                    reason=f"{claim_type} claims require evidence",
                    required_evidence=["source_citation", "grounding_anchor"],
                )
        
        # Check if confidence requires support
        if proposal.confidence > self.config.require_support_above_confidence:
            if not has_support:
                # Clamp confidence instead of deferring
                return InvariantResult(
                    invariant_id=self.id,
                    invariant_name=self.name,
                    domain=self.domain,
                    passed=False,
                    action=InvariantAction.CLAMP,
                    code="SUPPORT_REQUIRED_FOR_CONFIDENCE",
                    reason=f"Confidence {proposal.confidence:.2f} requires evidence; clamping",
                    proposal_delta={"confidence": self.config.require_support_above_confidence},
                )
        
        return InvariantResult(
            invariant_id=self.id,
            invariant_name=self.name,
            domain=self.domain,
            passed=True,
            action=InvariantAction.PASS,
        )


class DuplicateClaimInvariant:
    """
    Detect duplicate claims (same proposition already committed).
    
    Uses proposition hash to detect semantic duplicates.
    Duplicates are blocked to prevent ledger bloat.
    """
    
    def __init__(self, config: EpistemicConfig):
        self.config = config
    
    @property
    def id(self) -> str:
        return "epistemic.duplicate_claim"
    
    @property
    def name(self) -> str:
        return "Duplicate Claim Detection"
    
    @property
    def domain(self) -> Domain:
        return Domain.EPISTEMIC
    
    @property
    def domains(self) -> List[Domain]:
        return [Domain.EPISTEMIC]
    
    def check(self, proposal: ProposalEnvelope, state: StateView) -> InvariantResult:
        proposition_hash = proposal.payload.get("proposition_hash")
        
        if proposition_hash and proposition_hash in state.active_claims:
            existing = state.active_claims[proposition_hash]
            return InvariantResult(
                invariant_id=self.id,
                invariant_name=self.name,
                domain=self.domain,
                passed=False,
                action=InvariantAction.VETO,
                code="DUPLICATE_PROPOSITION",
                reason=f"Proposition already committed as {existing}",
            )
        
        return InvariantResult(
            invariant_id=self.id,
            invariant_name=self.name,
            domain=self.domain,
            passed=True,
            action=InvariantAction.PASS,
        )


class RevisionCostInvariant:
    """
    Track revision costs (thermal).
    
    Revisions are allowed but expensive. Older claims cost more to revise.
    This invariant doesn't block, just tracks heat.
    """
    
    def __init__(self, config: EpistemicConfig):
        self.config = config
    
    @property
    def id(self) -> str:
        return "epistemic.revision_cost"
    
    @property
    def name(self) -> str:
        return "Revision Cost"
    
    @property
    def domain(self) -> Domain:
        return Domain.EPISTEMIC
    
    @property
    def domains(self) -> List[Domain]:
        return [Domain.EPISTEMIC]
    
    def check(self, proposal: ProposalEnvelope, state: StateView) -> InvariantResult:
        supersedes = proposal.payload.get("supersedes")
        
        if not supersedes:
            # Not a revision
            return InvariantResult(
                invariant_id=self.id,
                invariant_name=self.name,
                domain=self.domain,
                passed=True,
                action=InvariantAction.PASS,
            )
        
        # Calculate revision cost
        # Older claims cost more (age in turns * multiplier)
        superseded_at = proposal.payload.get("superseded_claim_turn", 0)
        age_turns = proposal.t - superseded_at
        age_multiplier = 1.0 + (age_turns * 0.1)  # 10% more per turn
        
        heat = self.config.revision_base_heat * proposal.confidence * age_multiplier
        
        return InvariantResult(
            invariant_id=self.id,
            invariant_name=self.name,
            domain=self.domain,
            passed=True,
            action=InvariantAction.PASS,
            heat_delta=heat,
            work_delta=heat * 0.5,  # Work is half of heat
        )


# =============================================================================
# Registration
# =============================================================================

def register_epistemic_invariants(
    registry: ModuleRegistry,
    config: Optional[EpistemicConfig] = None,
) -> List[str]:
    """
    Register all epistemic invariants with the registry.
    
    Args:
        registry: The module registry
        config: Optional configuration (uses defaults if not provided)
        
    Returns:
        List of registered invariant IDs
    """
    config = config or EpistemicConfig()
    
    invariants = [
        (ConfidenceCeilingInvariant(config), 100, "Clamp confidence to type-specific ceilings"),
        (ThermalRegimeInvariant(config), 50, "Adjust behavior based on thermal state"),
        (ContradictionInvariant(config), 60, "Detect and block contradictory claims"),
        (SupportRequirementInvariant(config), 70, "Require evidence for risky claims"),
        (DuplicateClaimInvariant(config), 80, "Block duplicate propositions"),
        (RevisionCostInvariant(config), 90, "Track revision thermal costs"),
    ]
    
    registered = []
    for invariant, priority, description in invariants:
        inv_id = registry.register(
            invariant=invariant,
            priority=priority,
            description=description,
        )
        registered.append(inv_id)
    
    return registered


def create_epistemic_registry(config: Optional[EpistemicConfig] = None) -> ModuleRegistry:
    """
    Create a registry with epistemic invariants pre-registered.
    
    This is a convenience function for common use cases.
    """
    try:
        from .registry import create_registry
    except ImportError:
        from registry import create_registry
    
    registry = create_registry(with_global_invariants=True)
    register_epistemic_invariants(registry, config)
    return registry


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    print("=== Epistemic Module Demo ===\n")
    
    # Create registry with epistemic invariants
    config = EpistemicConfig(
        default_confidence_ceiling=0.85,
        warning_instability=0.3,
    )
    registry = create_epistemic_registry(config)
    
    print("1. Registered invariants:")
    for spec in registry.list_invariants():
        print(f"   [{spec.domain.value}] {spec.name} (priority={spec.priority})")
    
    # Test confidence ceiling
    print("\n2. Testing confidence ceiling")
    proposal = ProposalEnvelope(
        proposal_id="test_001",
        t=1,
        timestamp=datetime.now(),
        origin="llm",
        origin_type="llm",
        domain=Domain.EPISTEMIC,
        confidence=0.95,  # Above ceiling
        payload={"claim_type": "FACTUAL", "text": "Paris is the capital"},
    )
    state = StateView(current_t=0)
    
    report = registry.audit(proposal, state)
    print(f"   Proposal confidence: {proposal.confidence}")
    print(f"   Status: {report.status.name}")
    print(f"   Clamped to: {report.applied_clamps.get('confidence', 'N/A')}")
    
    # Test thermal regime
    print("\n3. Testing thermal regime (high instability)")
    state_hot = StateView(current_t=0, instability=0.8)  # Critical
    proposal2 = ProposalEnvelope(
        proposal_id="test_002",
        t=1,
        timestamp=datetime.now(),
        origin="llm",
        origin_type="llm",
        domain=Domain.EPISTEMIC,
        confidence=0.7,
        payload={"claim_type": "FACTUAL"},
    )
    
    report2 = registry.audit(proposal2, state_hot)
    print(f"   Instability: {state_hot.instability}")
    print(f"   Status: {report2.status.name}")
    if report2.required_evidence:
        print(f"   Required evidence: {report2.required_evidence}")
    
    # Test thermal shutdown
    print("\n4. Testing thermal shutdown")
    state_shutdown = StateView(current_t=0, instability=1.6)
    
    report3 = registry.audit(proposal2, state_shutdown)
    print(f"   Instability: {state_shutdown.instability}")
    print(f"   Status: {report3.status.name}")
    if report3.violated_invariants:
        print(f"   Violated: {report3.violated_invariants}")
    
    # Test support requirement
    print("\n5. Testing support requirement (QUANTITATIVE)")
    proposal_quant = ProposalEnvelope(
        proposal_id="test_003",
        t=1,
        timestamp=datetime.now(),
        origin="llm",
        origin_type="llm",
        domain=Domain.EPISTEMIC,
        confidence=0.6,
        payload={"claim_type": "QUANTITATIVE", "text": "The tower is 330m tall"},
    )
    
    report4 = registry.audit(proposal_quant, state)
    print(f"   Claim type: QUANTITATIVE")
    print(f"   Has support: False")
    print(f"   Status: {report4.status.name}")
    if report4.required_evidence:
        print(f"   Required: {report4.required_evidence}")
    
    # Test with support
    print("\n6. Testing with support provided")
    proposal_supported = ProposalEnvelope(
        proposal_id="test_004",
        t=1,
        timestamp=datetime.now(),
        origin="llm",
        origin_type="llm",
        domain=Domain.EPISTEMIC,
        confidence=0.6,
        evidence_refs=["wikipedia:Eiffel_Tower"],
        payload={"claim_type": "QUANTITATIVE", "text": "The tower is 330m tall"},
    )
    
    report5 = registry.audit(proposal_supported, state)
    print(f"   Has support: True")
    print(f"   Status: {report5.status.name}")
    
    # Show audit report details
    print("\n7. Full audit report structure")
    print(f"   Invariants checked: {report5.invariants_checked}")
    print(f"   Heat delta: {report5.total_heat_delta:.3f}")
    print(f"   Work delta: {report5.total_work_delta:.3f}")
    print(f"   Duration: {report5.audit_duration_ms:.2f}ms")
    
    print("\n✓ Epistemic module working")
