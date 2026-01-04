"""
Domain Module Stubs

Stub implementations for future domain modules. Each module follows the
same pattern: register invariants against a specific state domain.

Modules:
- Sensory: Invariants on observations over space
- Kinetic: Invariants on actuation over time  
- Resource: Invariants on consumption over horizon
- Normative: Invariants on action classes over context

These are placeholders with example invariants. The actual implementations
would integrate with external systems (perception, actuation, budgets, policies).

TODO for each module:
- Define state schema
- Implement 2-3 core invariants
- Add cross-module hooks where needed
- Write integration tests
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
    )
except ImportError:
    from registry import (
        ModuleRegistry,
        Domain,
        InvariantAction,
        ProposalEnvelope,
        StateView,
        InvariantResult,
    )


# =============================================================================
# SENSORY MODULE - Invariants on Observations over Space
# =============================================================================
# 
# The sensory module governs what the system "sees" and how observations
# translate into grounded state. Key concerns:
# - Observation freshness (stale data)
# - Spatial consistency (conflicting observations)
# - Grounding confidence (how sure are we of what we see)
#
# TODO:
# - Define observation schema (timestamp, source, confidence, spatial_ref)
# - Integrate with perception systems (vision, audio, sensors)
# - Cross-module hook: sensory → epistemic (grounding claims)
# - Cross-module hook: sensory → kinetic (don't act on stale observations)
# =============================================================================

@dataclass
class SensoryConfig:
    """Configuration for sensory invariants."""
    max_observation_age_seconds: float = 30.0  # Observations older than this are stale
    min_observation_confidence: float = 0.5   # Below this, observation is unreliable
    require_spatial_consistency: bool = True  # Check for conflicting observations


class ObservationFreshnessInvariant:
    """
    STUB: Reject actions based on stale observations.
    
    TODO:
    - Track observation timestamps
    - Integrate with perception system
    - Define staleness policy per observation type
    """
    
    def __init__(self, config: SensoryConfig):
        self.config = config
    
    @property
    def id(self) -> str:
        return "sensory.observation_freshness"
    
    @property
    def name(self) -> str:
        return "Observation Freshness"
    
    @property
    def domain(self) -> Domain:
        return Domain.SENSORY
    
    @property
    def domains(self) -> List[Domain]:
        return [Domain.SENSORY]
    
    def check(self, proposal: ProposalEnvelope, state: StateView) -> InvariantResult:
        # STUB: Always pass for now
        # TODO: Check observation_refs against state.observation_refs timestamps
        return InvariantResult(
            invariant_id=self.id,
            invariant_name=self.name,
            domain=self.domain,
            passed=True,
            action=InvariantAction.PASS,
        )


class GroundingRequirementInvariant:
    """
    STUB: Require sensory grounding for certain claim types.
    
    TODO:
    - Define which claims need grounding
    - Integrate with entity tracking
    - Cross-module: epistemic claims about physical entities need sensory grounding
    """
    
    def __init__(self, config: SensoryConfig):
        self.config = config
    
    @property
    def id(self) -> str:
        return "sensory.grounding_requirement"
    
    @property
    def name(self) -> str:
        return "Grounding Requirement"
    
    @property
    def domain(self) -> Domain:
        return Domain.SENSORY
    
    @property
    def domains(self) -> List[Domain]:
        return [Domain.SENSORY, Domain.EPISTEMIC]  # Cross-module
    
    def check(self, proposal: ProposalEnvelope, state: StateView) -> InvariantResult:
        # STUB: Check if entities in payload are grounded
        entities = proposal.payload.get("entities", [])
        ungrounded = [e for e in entities if e not in state.grounded_entities]
        
        if ungrounded:
            return InvariantResult(
                invariant_id=self.id,
                invariant_name=self.name,
                domain=self.domain,
                passed=False,
                action=InvariantAction.DEFER,
                code="UNGROUNDED_ENTITIES",
                reason=f"Entities not grounded: {ungrounded}",
                required_evidence=[f"sensory_grounding:{e}" for e in ungrounded],
            )
        
        return InvariantResult(
            invariant_id=self.id,
            invariant_name=self.name,
            domain=self.domain,
            passed=True,
            action=InvariantAction.PASS,
        )


def register_sensory_invariants(
    registry: ModuleRegistry,
    config: Optional[SensoryConfig] = None,
) -> List[str]:
    """Register sensory invariants. STUB implementation."""
    config = config or SensoryConfig()
    
    registered = []
    registered.append(registry.register(
        ObservationFreshnessInvariant(config),
        priority=40,
        description="[STUB] Reject stale observations",
    ))
    registered.append(registry.register(
        GroundingRequirementInvariant(config),
        priority=45,
        description="[STUB] Require sensory grounding for physical claims",
    ))
    
    return registered


# =============================================================================
# KINETIC MODULE - Invariants on Actuation over Time
# =============================================================================
#
# The kinetic module governs actions and their effects. Key concerns:
# - Action rate limits (don't spam actions)
# - Magnitude limits (don't take drastic actions)
# - Reversibility requirements (prefer reversible actions)
# - Temporal ordering (actions must be sequenced properly)
#
# TODO:
# - Define action schema (type, magnitude, target, reversible)
# - Integrate with tool/action systems
# - Cross-module hook: kinetic → sensory (don't act on ungrounded perception)
# - Cross-module hook: kinetic → resource (actions consume budget)
# =============================================================================

@dataclass
class KineticConfig:
    """Configuration for kinetic invariants."""
    max_action_magnitude: float = 1.0        # Clamp action magnitude
    max_actions_per_turn: int = 3            # Rate limit
    require_reversible_above: float = 0.7    # High-magnitude actions need reversibility
    cooldown_turns: int = 0                  # Turns between same action type


class ActionMagnitudeInvariant:
    """
    STUB: Clamp action magnitude to safe levels.
    
    TODO:
    - Define magnitude semantics per action type
    - Integrate with tool interfaces
    - Scale limits based on context/confidence
    """
    
    def __init__(self, config: KineticConfig):
        self.config = config
    
    @property
    def id(self) -> str:
        return "kinetic.action_magnitude"
    
    @property
    def name(self) -> str:
        return "Action Magnitude"
    
    @property
    def domain(self) -> Domain:
        return Domain.KINETIC
    
    @property
    def domains(self) -> List[Domain]:
        return [Domain.KINETIC]
    
    def check(self, proposal: ProposalEnvelope, state: StateView) -> InvariantResult:
        if proposal.magnitude > self.config.max_action_magnitude:
            return InvariantResult(
                invariant_id=self.id,
                invariant_name=self.name,
                domain=self.domain,
                passed=False,
                action=InvariantAction.CLAMP,
                code="MAGNITUDE_EXCEEDED",
                reason=f"Magnitude {proposal.magnitude:.2f} > limit {self.config.max_action_magnitude:.2f}",
                proposal_delta={"magnitude": self.config.max_action_magnitude},
            )
        
        return InvariantResult(
            invariant_id=self.id,
            invariant_name=self.name,
            domain=self.domain,
            passed=True,
            action=InvariantAction.PASS,
        )


class ActionRateLimitInvariant:
    """
    STUB: Rate limit actions per turn.
    
    TODO:
    - Track action count per turn
    - Define rate limits per action type
    - Handle burst vs sustained limits
    """
    
    def __init__(self, config: KineticConfig):
        self.config = config
    
    @property
    def id(self) -> str:
        return "kinetic.rate_limit"
    
    @property
    def name(self) -> str:
        return "Action Rate Limit"
    
    @property
    def domain(self) -> Domain:
        return Domain.KINETIC
    
    @property
    def domains(self) -> List[Domain]:
        return [Domain.KINETIC]
    
    def check(self, proposal: ProposalEnvelope, state: StateView) -> InvariantResult:
        if state.action_count >= self.config.max_actions_per_turn:
            return InvariantResult(
                invariant_id=self.id,
                invariant_name=self.name,
                domain=self.domain,
                passed=False,
                action=InvariantAction.DEFER,
                code="RATE_LIMIT_EXCEEDED",
                reason=f"Action limit ({self.config.max_actions_per_turn}) reached this turn",
            )
        
        return InvariantResult(
            invariant_id=self.id,
            invariant_name=self.name,
            domain=self.domain,
            passed=True,
            action=InvariantAction.PASS,
        )


def register_kinetic_invariants(
    registry: ModuleRegistry,
    config: Optional[KineticConfig] = None,
) -> List[str]:
    """Register kinetic invariants. STUB implementation."""
    config = config or KineticConfig()
    
    registered = []
    registered.append(registry.register(
        ActionMagnitudeInvariant(config),
        priority=30,
        description="[STUB] Clamp action magnitude",
    ))
    registered.append(registry.register(
        ActionRateLimitInvariant(config),
        priority=35,
        description="[STUB] Rate limit actions",
    ))
    
    return registered


# =============================================================================
# RESOURCE MODULE - Invariants on Consumption over Horizon
# =============================================================================
#
# The resource module governs budget and resource consumption. Key concerns:
# - Budget enforcement (don't exceed limits)
# - Cost tracking (accumulate work/heat)
# - Horizon planning (don't exhaust resources too early)
# - Escalation gating (expensive operations need budget)
#
# TODO:
# - Define resource schema (tokens, api_calls, compute, cost)
# - Integrate with usage tracking systems
# - Cross-module hook: resource → all (budget gates escalation)
# - Implement horizon-aware budgeting
# =============================================================================

@dataclass
class ResourceConfig:
    """Configuration for resource invariants."""
    max_work_per_turn: float = 1.0           # Work budget per turn
    max_work_total: float = 10.0             # Total session budget
    reserve_fraction: float = 0.1            # Keep 10% in reserve
    expensive_threshold: float = 0.5         # Operations above this need confirmation


class BudgetEnforcementInvariant:
    """
    STUB: Enforce resource budget limits.
    
    TODO:
    - Track cumulative resource usage
    - Define cost model per operation type
    - Handle soft vs hard limits
    """
    
    def __init__(self, config: ResourceConfig):
        self.config = config
    
    @property
    def id(self) -> str:
        return "resource.budget_enforcement"
    
    @property
    def name(self) -> str:
        return "Budget Enforcement"
    
    @property
    def domain(self) -> Domain:
        return Domain.RESOURCE
    
    @property
    def domains(self) -> List[Domain]:
        return [Domain.RESOURCE]
    
    def check(self, proposal: ProposalEnvelope, state: StateView) -> InvariantResult:
        estimated_cost = proposal.payload.get("estimated_cost", 0.1)
        
        if state.budget_remaining < estimated_cost:
            return InvariantResult(
                invariant_id=self.id,
                invariant_name=self.name,
                domain=self.domain,
                passed=False,
                action=InvariantAction.VETO,
                code="BUDGET_EXCEEDED",
                reason=f"Cost {estimated_cost:.2f} > remaining budget {state.budget_remaining:.2f}",
            )
        
        # Check reserve
        if state.budget_remaining - estimated_cost < self.config.reserve_fraction:
            return InvariantResult(
                invariant_id=self.id,
                invariant_name=self.name,
                domain=self.domain,
                passed=False,
                action=InvariantAction.DEFER,
                code="RESERVE_THREATENED",
                reason=f"Operation would deplete reserve",
            )
        
        return InvariantResult(
            invariant_id=self.id,
            invariant_name=self.name,
            domain=self.domain,
            passed=True,
            action=InvariantAction.PASS,
            work_delta=estimated_cost,
        )


class WorkAccumulationInvariant:
    """
    STUB: Track work accumulation against limits.
    
    TODO:
    - Implement work decay over time
    - Define work semantics per operation
    - Handle work-to-heat conversion
    """
    
    def __init__(self, config: ResourceConfig):
        self.config = config
    
    @property
    def id(self) -> str:
        return "resource.work_accumulation"
    
    @property
    def name(self) -> str:
        return "Work Accumulation"
    
    @property
    def domain(self) -> Domain:
        return Domain.RESOURCE
    
    @property
    def domains(self) -> List[Domain]:
        return [Domain.RESOURCE]
    
    def check(self, proposal: ProposalEnvelope, state: StateView) -> InvariantResult:
        if state.work_accumulated >= self.config.max_work_total:
            return InvariantResult(
                invariant_id=self.id,
                invariant_name=self.name,
                domain=self.domain,
                passed=False,
                action=InvariantAction.VETO,
                code="WORK_LIMIT_REACHED",
                reason=f"Session work limit ({self.config.max_work_total}) reached",
            )
        
        return InvariantResult(
            invariant_id=self.id,
            invariant_name=self.name,
            domain=self.domain,
            passed=True,
            action=InvariantAction.PASS,
        )


def register_resource_invariants(
    registry: ModuleRegistry,
    config: Optional[ResourceConfig] = None,
) -> List[str]:
    """Register resource invariants. STUB implementation."""
    config = config or ResourceConfig()
    
    registered = []
    registered.append(registry.register(
        BudgetEnforcementInvariant(config),
        priority=20,
        description="[STUB] Enforce resource budget",
    ))
    registered.append(registry.register(
        WorkAccumulationInvariant(config),
        priority=25,
        description="[STUB] Track work accumulation",
    ))
    
    return registered


# =============================================================================
# NORMATIVE MODULE - Invariants on Action Classes over Context
# =============================================================================
#
# The normative module governs what actions are permitted in context.
# This is NEGATIVE CONSTRAINTS ONLY - no value learning, no goals.
# Key concerns:
# - Action class blocking (certain actions never allowed)
# - Context-dependent restrictions (actions blocked in certain contexts)
# - Policy enforcement (external policy rules)
#
# TODO:
# - Define action class taxonomy
# - Integrate with policy systems
# - Handle context detection
# - NO goal inference or value learning
# =============================================================================

@dataclass
class NormativeConfig:
    """Configuration for normative invariants."""
    # Action classes that are always blocked
    blocked_action_classes: Set[str] = field(default_factory=lambda: {
        "self_modification",
        "resource_acquisition_unbounded",
        "deception_of_principal",
    })
    # Context tags that trigger additional restrictions
    restricted_contexts: Dict[str, Set[str]] = field(default_factory=lambda: {
        "high_stakes": {"irreversible_action", "large_magnitude_action"},
        "low_trust": {"external_communication", "resource_expenditure"},
    })


class ActionClassBlockingInvariant:
    """
    STUB: Block certain action classes unconditionally.
    
    TODO:
    - Define action class taxonomy
    - Integrate with action type detection
    - Handle hierarchical action classes
    """
    
    def __init__(self, config: NormativeConfig):
        self.config = config
    
    @property
    def id(self) -> str:
        return "normative.action_class_blocking"
    
    @property
    def name(self) -> str:
        return "Action Class Blocking"
    
    @property
    def domain(self) -> Domain:
        return Domain.NORMATIVE
    
    @property
    def domains(self) -> List[Domain]:
        return [Domain.NORMATIVE]
    
    def check(self, proposal: ProposalEnvelope, state: StateView) -> InvariantResult:
        action_class = proposal.payload.get("action_class", "")
        
        if action_class in self.config.blocked_action_classes:
            return InvariantResult(
                invariant_id=self.id,
                invariant_name=self.name,
                domain=self.domain,
                passed=False,
                action=InvariantAction.VETO,
                code="BLOCKED_ACTION_CLASS",
                reason=f"Action class '{action_class}' is not permitted",
            )
        
        # Also check state-level blocks
        if action_class in state.blocked_action_classes:
            return InvariantResult(
                invariant_id=self.id,
                invariant_name=self.name,
                domain=self.domain,
                passed=False,
                action=InvariantAction.VETO,
                code="CONTEXT_BLOCKED_ACTION",
                reason=f"Action class '{action_class}' blocked in current context",
            )
        
        return InvariantResult(
            invariant_id=self.id,
            invariant_name=self.name,
            domain=self.domain,
            passed=True,
            action=InvariantAction.PASS,
        )


class ContextRestrictionInvariant:
    """
    STUB: Apply context-dependent restrictions.
    
    TODO:
    - Implement context detection
    - Define restriction inheritance
    - Handle context transitions
    """
    
    def __init__(self, config: NormativeConfig):
        self.config = config
    
    @property
    def id(self) -> str:
        return "normative.context_restriction"
    
    @property
    def name(self) -> str:
        return "Context Restriction"
    
    @property
    def domain(self) -> Domain:
        return Domain.NORMATIVE
    
    @property
    def domains(self) -> List[Domain]:
        return [Domain.NORMATIVE]
    
    def check(self, proposal: ProposalEnvelope, state: StateView) -> InvariantResult:
        action_class = proposal.payload.get("action_class", "")
        
        # Check each active context
        for context_tag in state.context_tags:
            restricted = self.config.restricted_contexts.get(context_tag, set())
            if action_class in restricted:
                return InvariantResult(
                    invariant_id=self.id,
                    invariant_name=self.name,
                    domain=self.domain,
                    passed=False,
                    action=InvariantAction.VETO,
                    code="CONTEXT_RESTRICTION",
                    reason=f"Action class '{action_class}' restricted in context '{context_tag}'",
                )
        
        return InvariantResult(
            invariant_id=self.id,
            invariant_name=self.name,
            domain=self.domain,
            passed=True,
            action=InvariantAction.PASS,
        )


def register_normative_invariants(
    registry: ModuleRegistry,
    config: Optional[NormativeConfig] = None,
) -> List[str]:
    """Register normative invariants. STUB implementation."""
    config = config or NormativeConfig()
    
    registered = []
    registered.append(registry.register(
        ActionClassBlockingInvariant(config),
        priority=10,  # High priority - check early
        description="[STUB] Block prohibited action classes",
    ))
    registered.append(registry.register(
        ContextRestrictionInvariant(config),
        priority=15,
        description="[STUB] Apply context-dependent restrictions",
    ))
    
    return registered


# =============================================================================
# Convenience Functions
# =============================================================================

def register_all_modules(
    registry: ModuleRegistry,
    epistemic: bool = True,
    sensory: bool = False,
    kinetic: bool = False,
    resource: bool = False,
    normative: bool = False,
) -> Dict[str, List[str]]:
    """
    Register multiple modules at once.
    
    By default only epistemic is enabled (others are stubs).
    
    Returns:
        Dict mapping domain name to list of registered invariant IDs
    """
    try:
        from .epistemic_module import register_epistemic_invariants
    except ImportError:
        from epistemic_module import register_epistemic_invariants
    
    registered = {}
    
    if epistemic:
        registered["epistemic"] = register_epistemic_invariants(registry)
    
    if sensory:
        registered["sensory"] = register_sensory_invariants(registry)
    
    if kinetic:
        registered["kinetic"] = register_kinetic_invariants(registry)
    
    if resource:
        registered["resource"] = register_resource_invariants(registry)
    
    if normative:
        registered["normative"] = register_normative_invariants(registry)
    
    return registered


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    print("=== Domain Module Stubs Demo ===\n")
    
    try:
        from registry import create_registry
    except ImportError:
        from .registry import create_registry
    
    # Create registry with all modules
    registry = create_registry()
    registered = register_all_modules(
        registry,
        epistemic=True,
        sensory=True,
        kinetic=True,
        resource=True,
        normative=True,
    )
    
    print("1. Registered modules:")
    for domain, ids in registered.items():
        print(f"   {domain}: {len(ids)} invariants")
    
    print("\n2. All invariants by domain:")
    for spec in registry.list_invariants():
        stub = "[STUB]" if "[STUB]" in spec.description else ""
        print(f"   [{spec.domain.value}] {spec.name} {stub}")
    
    print("\n3. Stats:")
    stats = registry.get_stats()
    print(f"   Total: {stats['total_invariants']}")
    print(f"   Enabled: {stats['enabled_invariants']}")
    
    # Test normative blocking
    print("\n4. Testing normative blocking")
    from datetime import datetime
    
    proposal = ProposalEnvelope(
        proposal_id="action_001",
        t=1,
        timestamp=datetime.now(),
        origin="agent",
        origin_type="tool",
        domain=Domain.KINETIC,
        payload={"action_class": "self_modification"},
    )
    state = StateView(current_t=0)
    
    report = registry.audit(proposal, state)
    print(f"   Action class: self_modification")
    print(f"   Status: {report.status.name}")
    if report.violated_invariants:
        print(f"   Violated: {report.violated_invariants}")
    
    print("\n✓ Domain stubs working")
