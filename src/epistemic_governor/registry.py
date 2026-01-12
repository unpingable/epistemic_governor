"""
Module Registry - The ABI for Constraint Drivers

This is the kernel's module loading surface. All domain-specific constraints
register here and are evaluated uniformly during the audit phase.

Core invariant (load-bearing):
    No proposal may be committed unless it passes all registered module
    invariants at the current time index.

Design principles:
- Deterministic ordering (same inputs → same decision)
- Explicit multi-action semantics (PASS/CLAMP/DEFER/VETO)
- Aggregation by default (full violated-set for diagnostics)
- Time as ABI, not module (monotonic, append-only, irreversible)
- Cross-module invariants are first-class

Module registration:
    registry = ModuleRegistry()
    registry.register(
        invariant=confidence_ceiling,
        domain=Domain.EPISTEMIC,
        priority=100,
    )

Audit flow:
    report = registry.audit(proposal, state)
    if report.status == AuditStatus.ACCEPT:
        ledger.commit(proposal)
    elif report.status == AuditStatus.CLAMP:
        ledger.commit(report.clamped_proposal)
    else:
        # DEFER or VETO - do not commit
        handle_rejection(report)
"""

from dataclasses import dataclass, field
from typing import (
    Optional, List, Dict, Any, Callable, Set, 
    Union, TypeVar, Generic, Protocol, Tuple
)
from datetime import datetime
from enum import Enum, auto
from abc import ABC, abstractmethod
import hashlib


# =============================================================================
# Core Enums
# =============================================================================

class Domain(Enum):
    """
    Module domains.
    
    Each domain owns a slice of state and registers invariants against it.
    GLOBAL and CROSS are special: they apply across all domains.
    """
    # Meta-domains (not modules, part of ABI)
    GLOBAL = "global"       # Time, ordering, irreversibility
    CROSS = "cross"         # Cross-module invariants
    
    # Domain modules
    EPISTEMIC = "epistemic"   # Assertions over symbols
    SENSORY = "sensory"       # Observations over space
    KINETIC = "kinetic"       # Actuation over time
    RESOURCE = "resource"     # Consumption over horizon
    NORMATIVE = "normative"   # Action classes over context


class InvariantAction(Enum):
    """
    Actions an invariant can take on a proposal.
    
    Composition rules:
    - VETO dominates everything
    - DEFER dominates PASS/CLAMP
    - Multiple CLAMPs: last-by-priority wins per field
    """
    PASS = auto()    # No changes needed
    CLAMP = auto()   # Modify proposal and continue (or re-audit)
    DEFER = auto()   # Insufficient evidence; do not commit
    VETO = auto()    # Hard reject; do not commit


class AuditStatus(Enum):
    """Overall audit result."""
    ACCEPT = auto()      # All invariants passed (possibly with clamps)
    CLAMPED = auto()     # Accepted with modifications
    DEFERRED = auto()    # Waiting for evidence
    REJECTED = auto()    # Hard veto


class AuditMode(Enum):
    """How to run the audit."""
    AGGREGATE = auto()   # Run all invariants, collect full report (default)
    FIRST_FAIL = auto()  # Stop at first VETO/DEFER (fast path)


# =============================================================================
# Proposal Protocol
# =============================================================================

@dataclass
class ProposalEnvelope:
    """
    Every proposal carries this envelope. Time is part of the ABI.
    
    This is the "system call" interface - what modules see when
    they evaluate a proposal.
    """
    # Identity
    proposal_id: str
    
    # Temporal (ABI-level, not module-level)
    t: int                          # Monotonic time index (turn number)
    timestamp: datetime             # Wall clock (for logging)
    
    # Origin
    origin: str                     # Proposer ID (model, tool, user)
    origin_type: str                # "llm", "tool", "user", "system"
    
    # Evidence pointers
    evidence_refs: List[str] = field(default_factory=list)
    anchor_refs: List[str] = field(default_factory=list)
    
    # The actual proposal (domain-specific)
    domain: Domain = Domain.EPISTEMIC
    payload: Dict[str, Any] = field(default_factory=dict)
    
    # Mutable fields that clamps can modify
    confidence: float = 0.0
    magnitude: float = 1.0          # For kinetic: action magnitude
    ttl: Optional[int] = None       # Time-to-live in turns
    
    def copy(self) -> "ProposalEnvelope":
        """Create a copy for clamping."""
        return ProposalEnvelope(
            proposal_id=self.proposal_id,
            t=self.t,
            timestamp=self.timestamp,
            origin=self.origin,
            origin_type=self.origin_type,
            evidence_refs=list(self.evidence_refs),
            anchor_refs=list(self.anchor_refs),
            domain=self.domain,
            payload=dict(self.payload),
            confidence=self.confidence,
            magnitude=self.magnitude,
            ttl=self.ttl,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "t": self.t,
            "timestamp": self.timestamp.isoformat(),
            "origin": self.origin,
            "origin_type": self.origin_type,
            "domain": self.domain.value,
            "confidence": self.confidence,
            "magnitude": self.magnitude,
            "ttl": self.ttl,
            "evidence_refs": self.evidence_refs,
            "payload": self.payload,
        }


@dataclass
class StateView:
    """
    Read-only view of system state for invariant evaluation.
    
    Modules cannot mutate state directly - they can only
    evaluate proposals against it.
    """
    # Temporal
    current_t: int
    
    # Ledger state (epistemic)
    active_claims: Dict[str, Any] = field(default_factory=dict)
    claim_count: int = 0
    
    # Thermal state
    instability: float = 0.0
    revision_count: int = 0
    
    # Resource state
    budget_remaining: float = 1.0
    work_accumulated: float = 0.0
    
    # Sensory state (grounding)
    grounded_entities: Set[str] = field(default_factory=set)
    observation_refs: List[str] = field(default_factory=list)
    
    # Kinetic state (actuation history)
    pending_actions: List[str] = field(default_factory=list)
    action_count: int = 0
    
    # Normative state (context)
    context_tags: Set[str] = field(default_factory=set)
    blocked_action_classes: Set[str] = field(default_factory=set)


# =============================================================================
# Invariant Result
# =============================================================================

@dataclass
class InvariantResult:
    """
    Result of evaluating a single invariant.
    
    This is what an invariant returns after checking a proposal.
    """
    # Identity
    invariant_id: str
    invariant_name: str
    domain: Domain
    
    # Outcome
    passed: bool
    action: InvariantAction
    
    # Diagnostics (machine-readable)
    code: str = ""              # e.g., "CONFIDENCE_EXCEEDED", "UNGROUNDED_ENTITY"
    reason: str = ""            # Human-readable explanation
    
    # For CLAMP action: what to modify
    # Keys are field names on ProposalEnvelope, values are new values
    proposal_delta: Dict[str, Any] = field(default_factory=dict)
    
    # For DEFER action: what evidence is needed
    required_evidence: List[str] = field(default_factory=list)
    
    # Metrics
    heat_delta: float = 0.0     # Thermal cost of this result
    work_delta: float = 0.0     # Work/budget cost
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "invariant_id": self.invariant_id,
            "invariant_name": self.invariant_name,
            "domain": self.domain.value,
            "passed": self.passed,
            "action": self.action.name,
            "code": self.code,
            "reason": self.reason,
            "proposal_delta": self.proposal_delta,
            "required_evidence": self.required_evidence,
            "heat_delta": self.heat_delta,
            "work_delta": self.work_delta,
        }


# =============================================================================
# Invariant Protocol
# =============================================================================

class Invariant(Protocol):
    """
    Protocol for invariant implementations.
    
    An invariant is a constraint that must pass for a proposal to commit.
    Invariants have no authority over output beyond veto/clamp/defer.
    """
    
    @property
    def id(self) -> str:
        """Unique identifier for this invariant."""
        ...
    
    @property
    def name(self) -> str:
        """Human-readable name."""
        ...
    
    @property
    def domain(self) -> Domain:
        """Which domain this invariant belongs to."""
        ...
    
    @property
    def domains(self) -> List[Domain]:
        """For cross-domain invariants, list all relevant domains."""
        ...
    
    def check(self, proposal: ProposalEnvelope, state: StateView) -> InvariantResult:
        """
        Evaluate the invariant against a proposal.
        
        This is the core method. Must be deterministic and side-effect free.
        """
        ...


@dataclass
class InvariantSpec:
    """
    Specification for a registered invariant.
    
    Wraps an invariant implementation with registration metadata.
    """
    invariant: Invariant
    priority: int = 100           # Lower = earlier evaluation
    enabled: bool = True
    
    # For cross-domain invariants
    domains: List[Domain] = field(default_factory=list)
    
    # Metadata
    description: str = ""
    registered_at: datetime = field(default_factory=datetime.now)
    
    @property
    def id(self) -> str:
        return self.invariant.id
    
    @property
    def name(self) -> str:
        return self.invariant.name
    
    @property
    def domain(self) -> Domain:
        return self.invariant.domain


# =============================================================================
# Audit Report
# =============================================================================

@dataclass
class AuditReport:
    """
    Complete result of auditing a proposal against all invariants.
    
    This is the "return value" of the audit phase.
    """
    # Identity
    proposal_id: str
    audit_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Overall status
    status: AuditStatus = AuditStatus.ACCEPT
    
    # The proposal (possibly clamped)
    original_proposal: Optional[ProposalEnvelope] = None
    clamped_proposal: Optional[ProposalEnvelope] = None
    
    # Per-invariant results
    results: List[InvariantResult] = field(default_factory=list)
    
    # Aggregated diagnostics
    violated_invariants: List[str] = field(default_factory=list)
    applied_clamps: Dict[str, Any] = field(default_factory=dict)  # field -> value
    required_evidence: List[str] = field(default_factory=list)
    
    # Aggregated costs
    total_heat_delta: float = 0.0
    total_work_delta: float = 0.0
    
    # Audit metadata
    invariants_checked: int = 0
    audit_mode: AuditMode = AuditMode.AGGREGATE
    audit_duration_ms: float = 0.0
    
    @property
    def accepted(self) -> bool:
        return self.status in (AuditStatus.ACCEPT, AuditStatus.CLAMPED)
    
    @property
    def final_proposal(self) -> Optional[ProposalEnvelope]:
        """The proposal to commit (clamped if modified, original otherwise)."""
        if self.status == AuditStatus.CLAMPED:
            return self.clamped_proposal
        elif self.status == AuditStatus.ACCEPT:
            return self.original_proposal
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "audit_id": self.audit_id,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.name,
            "violated_invariants": self.violated_invariants,
            "applied_clamps": self.applied_clamps,
            "required_evidence": self.required_evidence,
            "total_heat_delta": self.total_heat_delta,
            "total_work_delta": self.total_work_delta,
            "invariants_checked": self.invariants_checked,
            "audit_mode": self.audit_mode.name,
            "audit_duration_ms": self.audit_duration_ms,
            "results": [r.to_dict() for r in self.results],
        }


# =============================================================================
# Module Registry
# =============================================================================

class ModuleRegistry:
    """
    Central registry for module invariants.
    
    This is the kernel's module loading surface. Invariants register here
    and are evaluated uniformly during the audit phase.
    
    Ordering:
    1. GLOBAL invariants (time/irreversibility) - always first
    2. Domain modules by priority
    3. Within domain, invariants by priority
    
    Same inputs → same decision. No registration-order dependence.
    """
    
    def __init__(self):
        self._invariants: Dict[str, InvariantSpec] = {}
        self._by_domain: Dict[Domain, List[str]] = {d: [] for d in Domain}
        self._audit_counter = 0
    
    def register(
        self,
        invariant: Invariant,
        priority: int = 100,
        domains: Optional[List[Domain]] = None,
        description: str = "",
    ) -> str:
        """
        Register an invariant.
        
        Args:
            invariant: The invariant implementation
            priority: Evaluation order (lower = earlier)
            domains: For cross-domain, list all relevant domains
            description: Human-readable description
            
        Returns:
            Invariant ID
        """
        spec = InvariantSpec(
            invariant=invariant,
            priority=priority,
            domains=domains or [invariant.domain],
            description=description or invariant.name,
        )
        
        self._invariants[spec.id] = spec
        
        # Index by domain
        for domain in spec.domains:
            if spec.id not in self._by_domain[domain]:
                self._by_domain[domain].append(spec.id)
        
        return spec.id
    
    def unregister(self, invariant_id: str) -> bool:
        """Remove an invariant from the registry."""
        if invariant_id not in self._invariants:
            return False
        
        spec = self._invariants.pop(invariant_id)
        for domain in spec.domains:
            if invariant_id in self._by_domain[domain]:
                self._by_domain[domain].remove(invariant_id)
        
        return True
    
    def enable(self, invariant_id: str) -> bool:
        """Enable a registered invariant."""
        if invariant_id in self._invariants:
            self._invariants[invariant_id].enabled = True
            return True
        return False
    
    def disable(self, invariant_id: str) -> bool:
        """Disable a registered invariant (still registered, not evaluated)."""
        if invariant_id in self._invariants:
            self._invariants[invariant_id].enabled = False
            return True
        return False
    
    def get_invariant(self, invariant_id: str) -> Optional[InvariantSpec]:
        """Get an invariant spec by ID."""
        return self._invariants.get(invariant_id)
    
    def list_invariants(
        self,
        domain: Optional[Domain] = None,
        enabled_only: bool = True,
    ) -> List[InvariantSpec]:
        """List registered invariants."""
        if domain:
            ids = self._by_domain.get(domain, [])
            specs = [self._invariants[i] for i in ids if i in self._invariants]
        else:
            specs = list(self._invariants.values())
        
        if enabled_only:
            specs = [s for s in specs if s.enabled]
        
        return sorted(specs, key=lambda s: (s.domain.value, s.priority))
    
    def _get_evaluation_order(self) -> List[InvariantSpec]:
        """
        Get invariants in deterministic evaluation order.
        
        Order:
        1. GLOBAL invariants by priority
        2. CROSS invariants by priority
        3. Other domains alphabetically, then by priority
        """
        specs = [s for s in self._invariants.values() if s.enabled]
        
        def sort_key(spec: InvariantSpec) -> Tuple[int, str, int]:
            # Domain order: GLOBAL=0, CROSS=1, others=2
            if spec.domain == Domain.GLOBAL:
                domain_order = 0
            elif spec.domain == Domain.CROSS:
                domain_order = 1
            else:
                domain_order = 2
            
            return (domain_order, spec.domain.value, spec.priority)
        
        return sorted(specs, key=sort_key)
    
    def audit(
        self,
        proposal: ProposalEnvelope,
        state: StateView,
        mode: AuditMode = AuditMode.AGGREGATE,
        max_clamp_iterations: int = 3,
    ) -> AuditReport:
        """
        Audit a proposal against all registered invariants.
        
        The core invariant:
            No proposal may be committed unless it passes all registered
            module invariants at the current time index.
        
        Args:
            proposal: The proposal to audit
            state: Current system state
            mode: AGGREGATE (full report) or FIRST_FAIL (fast path)
            max_clamp_iterations: Max times to re-audit after clamps
            
        Returns:
            AuditReport with status and diagnostics
        """
        import time
        start_time = time.time()
        
        self._audit_counter += 1
        audit_id = f"audit_{self._audit_counter}_{proposal.proposal_id[:8]}"
        
        report = AuditReport(
            proposal_id=proposal.proposal_id,
            audit_id=audit_id,
            original_proposal=proposal,
            audit_mode=mode,
        )
        
        # Get evaluation order
        specs = self._get_evaluation_order()
        
        # Current proposal (may be modified by clamps)
        current_proposal = proposal.copy()
        clamp_iteration = 0
        
        while clamp_iteration <= max_clamp_iterations:
            results = []
            has_veto = False
            has_defer = False
            has_clamp = False
            clamps_this_round: Dict[str, Any] = {}
            
            for spec in specs:
                # Check invariant
                result = spec.invariant.check(current_proposal, state)
                results.append(result)
                report.invariants_checked += 1
                
                # Accumulate costs
                report.total_heat_delta += result.heat_delta
                report.total_work_delta += result.work_delta
                
                # Handle action
                if result.action == InvariantAction.VETO:
                    has_veto = True
                    report.violated_invariants.append(result.invariant_id)
                    if mode == AuditMode.FIRST_FAIL:
                        break
                
                elif result.action == InvariantAction.DEFER:
                    has_defer = True
                    report.violated_invariants.append(result.invariant_id)
                    report.required_evidence.extend(result.required_evidence)
                    if mode == AuditMode.FIRST_FAIL:
                        break
                
                elif result.action == InvariantAction.CLAMP:
                    has_clamp = True
                    # Collect clamps (later clamps override earlier for same field)
                    for field, value in result.proposal_delta.items():
                        clamps_this_round[field] = value
            
            report.results.extend(results)
            
            # Determine status based on composition rules
            if has_veto:
                report.status = AuditStatus.REJECTED
                break
            
            if has_defer:
                report.status = AuditStatus.DEFERRED
                break
            
            if has_clamp and clamp_iteration < max_clamp_iterations:
                # Apply clamps and re-audit
                for field, value in clamps_this_round.items():
                    if hasattr(current_proposal, field):
                        setattr(current_proposal, field, value)
                        report.applied_clamps[field] = value
                
                clamp_iteration += 1
                continue
            
            # All passed (possibly with clamps)
            if report.applied_clamps:
                report.status = AuditStatus.CLAMPED
                report.clamped_proposal = current_proposal
            else:
                report.status = AuditStatus.ACCEPT
            
            break
        
        report.audit_duration_ms = (time.time() - start_time) * 1000
        return report
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        by_domain = {}
        for domain, ids in self._by_domain.items():
            enabled = sum(1 for i in ids if i in self._invariants and self._invariants[i].enabled)
            by_domain[domain.value] = {"total": len(ids), "enabled": enabled}
        
        return {
            "total_invariants": len(self._invariants),
            "enabled_invariants": sum(1 for s in self._invariants.values() if s.enabled),
            "by_domain": by_domain,
            "audit_count": self._audit_counter,
        }


# =============================================================================
# Built-in Global Invariants (Time ABI)
# =============================================================================

class MonotonicTimeInvariant:
    """
    GLOBAL invariant: time must be monotonically increasing.
    
    This is part of the ABI, not a module.
    """
    
    def __init__(self):
        self._last_t = -1
    
    @property
    def id(self) -> str:
        return "global.monotonic_time"
    
    @property
    def name(self) -> str:
        return "Monotonic Time"
    
    @property
    def domain(self) -> Domain:
        return Domain.GLOBAL
    
    @property
    def domains(self) -> List[Domain]:
        return [Domain.GLOBAL]
    
    def check(self, proposal: ProposalEnvelope, state: StateView) -> InvariantResult:
        if proposal.t < state.current_t:
            return InvariantResult(
                invariant_id=self.id,
                invariant_name=self.name,
                domain=self.domain,
                passed=False,
                action=InvariantAction.VETO,
                code="TIME_REGRESSION",
                reason=f"Proposal t={proposal.t} < current t={state.current_t}",
                heat_delta=0.1,
            )
        
        return InvariantResult(
            invariant_id=self.id,
            invariant_name=self.name,
            domain=self.domain,
            passed=True,
            action=InvariantAction.PASS,
        )


class ProposalIdInvariant:
    """
    GLOBAL invariant: proposals must have unique IDs.
    
    Note: This tracks IDs seen during audits within a single audit cycle.
    The actual deduplication should happen at commit time in the ledger.
    This invariant prevents re-auditing the same proposal in a loop.
    """
    
    def __init__(self):
        self._committed_ids: Set[str] = set()
    
    @property
    def id(self) -> str:
        return "global.unique_proposal_id"
    
    @property
    def name(self) -> str:
        return "Unique Proposal ID"
    
    @property
    def domain(self) -> Domain:
        return Domain.GLOBAL
    
    @property
    def domains(self) -> List[Domain]:
        return [Domain.GLOBAL]
    
    def check(self, proposal: ProposalEnvelope, state: StateView) -> InvariantResult:
        # Only check against actually committed proposals
        if proposal.proposal_id in self._committed_ids:
            return InvariantResult(
                invariant_id=self.id,
                invariant_name=self.name,
                domain=self.domain,
                passed=False,
                action=InvariantAction.VETO,
                code="DUPLICATE_PROPOSAL",
                reason=f"Proposal ID already committed: {proposal.proposal_id}",
            )
        
        return InvariantResult(
            invariant_id=self.id,
            invariant_name=self.name,
            domain=self.domain,
            passed=True,
            action=InvariantAction.PASS,
        )
    
    def mark_committed(self, proposal_id: str):
        """Mark a proposal as committed (call after successful commit)."""
        self._committed_ids.add(proposal_id)
    
    def reset(self):
        """Reset committed IDs (for testing)."""
        self._committed_ids.clear()


# =============================================================================
# Factory Functions
# =============================================================================

def create_registry(with_global_invariants: bool = True) -> ModuleRegistry:
    """
    Create a module registry.
    
    Args:
        with_global_invariants: Register built-in global invariants (recommended)
        
    Returns:
        Configured ModuleRegistry
    """
    registry = ModuleRegistry()
    
    if with_global_invariants:
        # Time ABI
        registry.register(
            MonotonicTimeInvariant(),
            priority=0,
            description="Enforce monotonic time ordering",
        )
        registry.register(
            ProposalIdInvariant(),
            priority=1,
            description="Enforce unique proposal IDs",
        )
    
    return registry


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    print("=== Module Registry Demo ===\n")
    
    # Create registry with global invariants
    registry = create_registry()
    print(f"1. Created registry")
    stats = registry.get_stats()
    print(f"   Invariants: {stats['total_invariants']}")
    print(f"   By domain: {stats['by_domain']}")
    
    # Create a simple epistemic invariant
    class ConfidenceCeilingInvariant:
        def __init__(self, ceiling: float = 0.95):
            self.ceiling = ceiling
        
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
            if proposal.confidence > self.ceiling:
                return InvariantResult(
                    invariant_id=self.id,
                    invariant_name=self.name,
                    domain=self.domain,
                    passed=False,
                    action=InvariantAction.CLAMP,
                    code="CONFIDENCE_EXCEEDED",
                    reason=f"Confidence {proposal.confidence:.2f} > ceiling {self.ceiling:.2f}",
                    proposal_delta={"confidence": self.ceiling},
                )
            
            return InvariantResult(
                invariant_id=self.id,
                invariant_name=self.name,
                domain=self.domain,
                passed=True,
                action=InvariantAction.PASS,
            )
    
    # Register epistemic invariant
    print("\n2. Registering epistemic invariant")
    registry.register(
        ConfidenceCeilingInvariant(ceiling=0.85),
        priority=100,
        description="Clamp confidence to 85%",
    )
    
    stats = registry.get_stats()
    print(f"   Total invariants: {stats['total_invariants']}")
    
    # Create a proposal
    print("\n3. Creating proposal with high confidence")
    proposal = ProposalEnvelope(
        proposal_id="prop_001",
        t=1,
        timestamp=datetime.now(),
        origin="llm",
        origin_type="llm",
        domain=Domain.EPISTEMIC,
        confidence=0.92,  # Above ceiling
        payload={"text": "Paris is the capital of France"},
    )
    print(f"   Proposal: {proposal.proposal_id}")
    print(f"   Confidence: {proposal.confidence}")
    
    # Create state
    state = StateView(current_t=0)
    
    # Audit
    print("\n4. Running audit")
    report = registry.audit(proposal, state)
    print(f"   Status: {report.status.name}")
    print(f"   Invariants checked: {report.invariants_checked}")
    print(f"   Applied clamps: {report.applied_clamps}")
    
    if report.clamped_proposal:
        print(f"   Clamped confidence: {report.clamped_proposal.confidence}")
    
    # Simulate commit
    if report.accepted:
        print("\n5. Simulating commit")
        # Mark proposal as committed
        id_invariant = registry.get_invariant("global.unique_proposal_id")
        if id_invariant:
            id_invariant.invariant.mark_committed(proposal.proposal_id)
        print(f"   Marked {proposal.proposal_id} as committed")
    
    # Try to re-audit same proposal (should be rejected as duplicate)
    print("\n6. Testing duplicate detection")
    report2 = registry.audit(proposal, state)
    print(f"   Status: {report2.status.name}")
    if report2.violated_invariants:
        print(f"   Violated: {report2.violated_invariants}")
    
    # Test time regression
    print("\n7. Testing time regression")
    proposal_old = ProposalEnvelope(
        proposal_id="prop_002",
        t=0,  # Same as current_t, but we'll set state.current_t = 1
        timestamp=datetime.now(),
        origin="llm",
        origin_type="llm",
        domain=Domain.EPISTEMIC,
        confidence=0.5,
    )
    state_future = StateView(current_t=2)  # State is at t=2
    report3 = registry.audit(proposal_old, state_future)
    print(f"   Proposal t={proposal_old.t}, State t={state_future.current_t}")
    print(f"   Status: {report3.status.name}")
    if report3.violated_invariants:
        print(f"   Violated: {report3.violated_invariants}")
    
    # List invariants
    print("\n8. Listing invariants in evaluation order")
    for spec in registry.list_invariants():
        print(f"   [{spec.domain.value}] {spec.name} (priority={spec.priority})")
    
    print("\n✓ Module registry working")
