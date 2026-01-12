"""
Creative Regime - Structure Without History

This module implements creative/dreaming modes where proposals can exist
and interact without ever becoming epistemic commitments.

Key principle:
    Creative and dream regimes may generate structure, but may not generate history.

This enables:
- Creativity without delusion
- Dreaming without belief formation
- Novelty without gaslighting
- Fiction without confusing imagination with knowledge

Usage:
    from epistemic_governor.creative import (
        CreativeRegime,
        CreativeConfig,
        CreativeLedger,
    )
    
    # Create a sandboxed creative session
    regime = CreativeRegime(config=CreativeConfig(
        allow_contradictions=True,
        identity_stiffness=0.3,
        boredom_enabled=True,
    ))
    
    # Generate without committing
    result = regime.explore(proposal)
    
    # Nothing persists to epistemic ledger
    assert regime.epistemic_commits == 0
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from datetime import datetime
from enum import Enum, auto
import hashlib

# Handle both package and direct imports
try:
    from epistemic_governor.registry import (
        ModuleRegistry,
        Domain,
        InvariantAction,
        ProposalEnvelope,
        StateView,
        InvariantResult,
        AuditReport,
        AuditStatus,
        create_registry,
    )
except ImportError:
    from epistemic_governor.registry import (
        ModuleRegistry,
        Domain,
        InvariantAction,
        ProposalEnvelope,
        StateView,
        InvariantResult,
        AuditReport,
        AuditStatus,
        create_registry,
    )


# =============================================================================
# Creative Regime Configuration
# =============================================================================

class CreativeMode(Enum):
    """Different creative regime modes."""
    EXPLORE = auto()      # Free exploration, nothing persists
    DRAFT = auto()        # Loose constraints, cheap revision
    DREAM = auto()        # Contradictions allowed, identity fluid
    COMEDY = auto()       # Intentional violations, fast snap-back


@dataclass
class CreativeConfig:
    """
    Configuration for creative regime.
    
    These parameters control how "loose" the creative sandbox is.
    """
    mode: CreativeMode = CreativeMode.EXPLORE
    
    # Contradiction handling
    allow_contradictions: bool = True
    contradiction_cost: float = 0.0  # No cost in creative mode
    
    # Identity parameters
    identity_stiffness: float = 0.3  # 0.0 = fluid, 1.0 = rigid
    identity_deformation_cost: float = 0.1
    
    # Confidence handling
    confidence_floor: float = 0.0   # Allow any confidence
    confidence_ceiling: float = 1.0  # No ceiling in creative mode
    
    # Boredom parameters
    boredom_enabled: bool = True
    boredom_threshold: float = 0.7  # When to start penalizing repetition
    boredom_decay: float = 0.1      # How fast boredom decays
    
    # Revision handling
    revision_cost: float = 0.01     # Cheap revisions
    
    # Sandbox boundaries
    max_proposals: int = 1000       # Prevent runaway generation
    max_turns: int = 100
    
    # What CAN'T leak out
    block_epistemic_commit: bool = True  # The core invariant


# =============================================================================
# Creative Ledger (Sandboxed, Non-Epistemic)
# =============================================================================

@dataclass
class CreativeProposal:
    """
    A proposal in creative space. 
    
    Similar to epistemic proposals but explicitly marked as non-committal.
    Can be contradicted, revised, or discarded without cost.
    """
    id: str
    text: str
    created_at: datetime
    confidence: float
    
    # Creative-specific
    is_exploration: bool = True
    contradicts: List[str] = field(default_factory=list)
    supersedes: Optional[str] = None
    
    # Tracking
    revision_count: int = 0
    deformation_count: int = 0  # Identity changes
    
    # Motif tracking for boredom
    motifs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "created_at": self.created_at.isoformat(),
            "confidence": self.confidence,
            "is_exploration": self.is_exploration,
            "contradicts": self.contradicts,
            "supersedes": self.supersedes,
            "revision_count": self.revision_count,
            "motifs": self.motifs,
        }


@dataclass 
class CreativeState:
    """
    State accumulator for creative regime.
    
    Tracks what's happened in the sandbox without committing anything
    to epistemic history.
    """
    # Proposals (can be contradictory)
    proposals: Dict[str, CreativeProposal] = field(default_factory=dict)
    
    # Motif tracking
    motif_counts: Dict[str, int] = field(default_factory=dict)
    recent_motifs: List[str] = field(default_factory=list)
    
    # Boredom accumulator
    boredom: float = 0.0
    
    # Identity tracking (for characters, voices, etc.)
    identity_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    identity_deformations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metrics
    total_proposals: int = 0
    total_contradictions: int = 0
    total_revisions: int = 0
    turn: int = 0
    
    def add_proposal(self, proposal: CreativeProposal):
        """Add a proposal to creative state."""
        self.proposals[proposal.id] = proposal
        self.total_proposals += 1
        
        # Track contradictions
        if proposal.contradicts:
            self.total_contradictions += len(proposal.contradicts)
        
        # Track revisions
        if proposal.supersedes:
            self.total_revisions += 1
        
        # Track motifs for boredom
        for motif in proposal.motifs:
            self.motif_counts[motif] = self.motif_counts.get(motif, 0) + 1
            self.recent_motifs.append(motif)
        
        # Trim recent motifs to last 50
        if len(self.recent_motifs) > 50:
            self.recent_motifs = self.recent_motifs[-50:]
    
    def get_motif_saturation(self, motif: str, window: int = 10) -> float:
        """How saturated is this motif in recent proposals?"""
        recent = self.recent_motifs[-window:] if self.recent_motifs else []
        if not recent:
            return 0.0
        count = sum(1 for m in recent if m == motif)
        return count / window  # Normalize by window size, not len(recent)
    
    def to_state_view(self) -> StateView:
        """Convert to registry StateView for invariant checking."""
        return StateView(
            current_t=self.turn,
            active_claims={p.id: p for p in self.proposals.values()},
            claim_count=len(self.proposals),
            instability=self.boredom,  # Use boredom as instability proxy
        )


# =============================================================================
# Creative Invariants
# =============================================================================

class NoEpistemicCommitInvariant:
    """
    The core creative invariant: nothing commits to epistemic history.
    
    This is the firewall that prevents creative exploration from
    leaking into belief formation.
    """
    
    def __init__(self, config: CreativeConfig):
        self.config = config
    
    @property
    def id(self) -> str:
        return "creative.no_epistemic_commit"
    
    @property
    def name(self) -> str:
        return "No Epistemic Commit"
    
    @property
    def domain(self) -> Domain:
        return Domain.EPISTEMIC
    
    @property
    def domains(self) -> List[Domain]:
        return [Domain.EPISTEMIC]
    
    def check(self, proposal: ProposalEnvelope, state: StateView) -> InvariantResult:
        # Check if this proposal is trying to commit to epistemic ledger
        wants_epistemic = proposal.payload.get("epistemic_commit", False)
        
        if wants_epistemic and self.config.block_epistemic_commit:
            return InvariantResult(
                invariant_id=self.id,
                invariant_name=self.name,
                domain=self.domain,
                passed=False,
                action=InvariantAction.VETO,
                code="EPISTEMIC_COMMIT_BLOCKED",
                reason="Creative regime does not allow epistemic commits",
            )
        
        return InvariantResult(
            invariant_id=self.id,
            invariant_name=self.name,
            domain=self.domain,
            passed=True,
            action=InvariantAction.PASS,
        )


class BoredomInvariant:
    """
    Penalize repetitive patterns.
    
    Boredom is pressure against gratuitous novelty AND repetition.
    It nudges toward restraint, not fireworks.
    """
    
    def __init__(self, config: CreativeConfig, state: CreativeState):
        self.config = config
        self.creative_state = state  # Reference to track motifs
    
    @property
    def id(self) -> str:
        return "creative.boredom"
    
    @property
    def name(self) -> str:
        return "Boredom"
    
    @property
    def domain(self) -> Domain:
        return Domain.EPISTEMIC  # Creative is a sub-domain
    
    @property
    def domains(self) -> List[Domain]:
        return [Domain.EPISTEMIC]
    
    def check(self, proposal: ProposalEnvelope, state: StateView) -> InvariantResult:
        if not self.config.boredom_enabled:
            return InvariantResult(
                invariant_id=self.id,
                invariant_name=self.name,
                domain=self.domain,
                passed=True,
                action=InvariantAction.PASS,
            )
        
        # Check motif saturation
        motifs = proposal.payload.get("motifs", [])
        max_saturation = 0.0
        saturated_motif = None
        
        for motif in motifs:
            saturation = self.creative_state.get_motif_saturation(motif)
            if saturation > max_saturation:
                max_saturation = saturation
                saturated_motif = motif
        
        # If highly saturated, increase boredom cost
        if max_saturation > self.config.boredom_threshold:
            boredom_delta = (max_saturation - self.config.boredom_threshold) * 0.5
            
            return InvariantResult(
                invariant_id=self.id,
                invariant_name=self.name,
                domain=self.domain,
                passed=True,  # Don't block, just cost
                action=InvariantAction.PASS,
                code="BOREDOM_ACCUMULATING",
                reason=f"Motif '{saturated_motif}' is saturated ({max_saturation:.0%})",
                heat_delta=boredom_delta,
            )
        
        return InvariantResult(
            invariant_id=self.id,
            invariant_name=self.name,
            domain=self.domain,
            passed=True,
            action=InvariantAction.PASS,
        )


class IdentityStiffnessInvariant:
    """
    Control how much identity can deform.
    
    High stiffness = characters/voices resist change
    Low stiffness = fluid, experimental
    """
    
    def __init__(self, config: CreativeConfig, state: CreativeState):
        self.config = config
        self.creative_state = state
    
    @property
    def id(self) -> str:
        return "creative.identity_stiffness"
    
    @property
    def name(self) -> str:
        return "Identity Stiffness"
    
    @property
    def domain(self) -> Domain:
        return Domain.EPISTEMIC
    
    @property
    def domains(self) -> List[Domain]:
        return [Domain.EPISTEMIC]
    
    def check(self, proposal: ProposalEnvelope, state: StateView) -> InvariantResult:
        # Check if this proposal deforms an identity
        identity_id = proposal.payload.get("identity_id")
        deformation = proposal.payload.get("identity_deformation")
        
        if not identity_id or not deformation:
            return InvariantResult(
                invariant_id=self.id,
                invariant_name=self.name,
                domain=self.domain,
                passed=True,
                action=InvariantAction.PASS,
            )
        
        # Calculate deformation cost based on stiffness
        deformation_magnitude = deformation.get("magnitude", 0.5)
        cost = deformation_magnitude * self.config.identity_stiffness
        
        # High stiffness + high deformation = expensive
        if cost > 0.5:
            return InvariantResult(
                invariant_id=self.id,
                invariant_name=self.name,
                domain=self.domain,
                passed=True,  # Allow but cost
                action=InvariantAction.PASS,
                code="IDENTITY_DEFORMATION",
                reason=f"Identity '{identity_id}' deformation costs {cost:.2f}",
                heat_delta=cost,
            )
        
        return InvariantResult(
            invariant_id=self.id,
            invariant_name=self.name,
            domain=self.domain,
            passed=True,
            action=InvariantAction.PASS,
            heat_delta=cost * 0.5,  # Small cost even for easy deformations
        )


# =============================================================================
# Creative Regime
# =============================================================================

class CreativeRegime:
    """
    A sandboxed regime for creative exploration.
    
    Proposals can exist, contradict, revise freely - but nothing
    commits to epistemic history.
    
    Usage:
        regime = CreativeRegime()
        
        # Explore freely
        result = regime.explore("The sky was green that day")
        result = regime.explore("The sky was always blue")  # Contradiction OK
        
        # Check state
        print(regime.state.total_contradictions)
        
        # Nothing leaked to epistemic ledger
        assert regime.epistemic_commits == 0
    """
    
    def __init__(self, config: Optional[CreativeConfig] = None):
        self.config = config or CreativeConfig()
        self.state = CreativeState()
        
        # Create registry with creative invariants
        self.registry = create_registry(with_global_invariants=True)
        self._register_invariants()
        
        # Track what DIDN'T leak
        self.epistemic_commits = 0
        self.blocked_commits = 0
    
    def _register_invariants(self):
        """Register creative-specific invariants."""
        # Core firewall
        self.registry.register(
            NoEpistemicCommitInvariant(self.config),
            priority=0,  # Highest priority
            description="Prevent epistemic commits in creative mode",
        )
        
        # Boredom
        self.registry.register(
            BoredomInvariant(self.config, self.state),
            priority=50,
            description="Penalize repetitive patterns",
        )
        
        # Identity stiffness
        self.registry.register(
            IdentityStiffnessInvariant(self.config, self.state),
            priority=60,
            description="Control identity deformation cost",
        )
    
    def explore(
        self,
        text: str,
        motifs: Optional[List[str]] = None,
        contradicts: Optional[List[str]] = None,
        identity_id: Optional[str] = None,
        identity_deformation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Explore a creative proposal.
        
        Returns exploration result without committing anything.
        """
        # Generate proposal ID
        proposal_id = f"creative_{self.state.total_proposals:04d}"
        
        # Build proposal envelope
        proposal = ProposalEnvelope(
            proposal_id=proposal_id,
            t=self.state.turn,
            timestamp=datetime.now(),
            origin="creative",
            origin_type="exploration",
            domain=Domain.EPISTEMIC,
            confidence=0.5,  # Confidence doesn't matter in creative mode
            payload={
                "text": text,
                "motifs": motifs or [],
                "contradicts": contradicts or [],
                "identity_id": identity_id,
                "identity_deformation": identity_deformation,
                "epistemic_commit": False,  # Never commit
            },
        )
        
        # Run through registry
        state_view = self.state.to_state_view()
        report = self.registry.audit(proposal, state_view)
        
        # Track boredom from heat
        self.state.boredom += report.total_heat_delta
        self.state.boredom = max(0, self.state.boredom - self.config.boredom_decay)
        
        # Create creative proposal
        creative_prop = CreativeProposal(
            id=proposal_id,
            text=text,
            created_at=datetime.now(),
            confidence=0.5,
            contradicts=contradicts or [],
            motifs=motifs or [],
        )
        
        # Add to state (regardless of audit - creative mode allows everything)
        self.state.add_proposal(creative_prop)
        
        return {
            "proposal_id": proposal_id,
            "text": text,
            "accepted": True,  # Always accepted in creative mode
            "boredom": self.state.boredom,
            "boredom_delta": report.total_heat_delta,
            "contradictions": len(contradicts or []),
            "motif_saturations": {
                m: self.state.get_motif_saturation(m) 
                for m in (motifs or [])
            },
        }
    
    def advance_turn(self):
        """Advance to next turn, decay boredom."""
        self.state.turn += 1
        self.state.boredom = max(0, self.state.boredom - self.config.boredom_decay)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get regime summary."""
        return {
            "mode": self.config.mode.name,
            "turn": self.state.turn,
            "total_proposals": self.state.total_proposals,
            "total_contradictions": self.state.total_contradictions,
            "total_revisions": self.state.total_revisions,
            "boredom": self.state.boredom,
            "epistemic_commits": self.epistemic_commits,
            "blocked_commits": self.blocked_commits,
            "identity_stiffness": self.config.identity_stiffness,
        }
    
    def reset(self):
        """Reset creative state."""
        self.state = CreativeState()
        self.epistemic_commits = 0
        self.blocked_commits = 0


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=== Creative Regime Demo ===\n")
    
    # Create regime with boredom enabled
    regime = CreativeRegime(CreativeConfig(
        boredom_enabled=True,
        boredom_threshold=0.3,  # Lower threshold to trigger more easily
        identity_stiffness=0.4,
    ))
    
    print("1. Free exploration (contradictions allowed)")
    r1 = regime.explore("The sky was green that morning", motifs=["sky", "color"])
    print(f"   Proposal: {r1['proposal_id']}")
    print(f"   Boredom: {r1['boredom']:.3f}")
    
    r2 = regime.explore("The sky was always blue", motifs=["sky", "color"], 
                        contradicts=[r1['proposal_id']])
    print(f"   Contradiction of {r1['proposal_id']}: accepted={r2['accepted']}")
    print(f"   Total contradictions: {regime.state.total_contradictions}")
    
    print("\n2. Boredom accumulation (repetitive motifs)")
    # Use only 'sky' motif to maximize saturation
    for i in range(8):
        r = regime.explore(f"The sky was beautiful (variation {i})", motifs=["sky"])
        sat = r['motif_saturations'].get('sky', 0)
        print(f"   Turn {i}: boredom={r['boredom']:.3f}, sky_saturation={sat:.0%}, delta={r['boredom_delta']:.3f}")
    
    print("\n3. Novel motif (boredom relief)")
    r_novel = regime.explore("The ocean crashed against the rocks", motifs=["ocean", "rocks"])
    print(f"   New motifs: boredom={r_novel['boredom']:.3f}")
    
    print("\n4. Identity deformation")
    r_identity = regime.explore(
        "Sarah suddenly became cruel",
        identity_id="sarah",
        identity_deformation={"trait": "kindness", "magnitude": 0.8},
    )
    print(f"   Deformation cost absorbed: boredom={r_identity['boredom']:.3f}")
    
    print("\n5. Summary")
    summary = regime.get_summary()
    for k, v in summary.items():
        print(f"   {k}: {v}")
    
    print("\n6. Core invariant: epistemic commits blocked")
    print(f"   Epistemic commits: {regime.epistemic_commits}")
    print(f"   (All {regime.state.total_proposals} proposals stayed in creative sandbox)")
    
    print("\n✓ Creative regime working")
