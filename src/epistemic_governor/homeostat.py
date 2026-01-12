"""
Epistemic Homeostat

Adaptive gain scheduling for the epistemic governor, inspired by Baby River's
metabolic control pattern.

The key insight: homeostasis learning = gain scheduling, not belief learning.
The homeostat observes "vitals" (system state), compares to setpoints, and
outputs "tuning deltas" that adjust governor parameters.

Baby River portable concepts:
1. Setpoints + error → cost signal
2. Urgency → adaptive responsiveness (NOT randomness)
3. Regime shift robustness

Critical rule: Homeostat never touches ledger contents.
It only adjusts thresholds and requirements.

Usage:
    homeostat = Homeostat(config)
    
    # Each turn, observe and compute tuning
    vitals = homeostat.observe(session)
    tuning = homeostat.compute_tuning(vitals)
    
    # Apply tuning to policy before governance
    policy = base_policy.apply_tuning(tuning)
    
    # After governance, record outcome for learning
    homeostat.record_outcome(frame)
"""

import math
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from datetime import datetime

# Handle both package and direct imports
try:
    from epistemic_governor.session import EpistemicSession, EpistemicFrame
    from epistemic_governor.kernel import ThermalState
    from epistemic_governor.calibrate import PolicyProfile, DomainPolicy, BaselineProfile, DistributionStats
except ImportError:
    from epistemic_governor.session import EpistemicSession, EpistemicFrame
    from epistemic_governor.kernel import ThermalState
    from epistemic_governor.calibrate import PolicyProfile, DomainPolicy, BaselineProfile, DistributionStats


# =============================================================================
# Vitals (Observable System State)
# =============================================================================

@dataclass
class EpistemicVitals:
    """
    Observable system state - the "internal sensors" of the epistemic system.
    
    Analogous to Baby River's [energy, heat] state vector.
    """
    # Core rates (0-1 normalized)
    revision_rate: float = 0.0          # revisions / total_commits
    contradiction_rate: float = 0.0      # contradictions / total_proposals
    hedge_rate: float = 0.0             # hedges / total_commits
    refusal_rate: float = 0.0           # refusals / total_proposals
    
    # Support/grounding metrics
    support_deficit_rate: float = 0.0   # claims needing support / total
    retrieval_coverage: float = 1.0     # claims with support / claims needing it
    
    # Thermal state (from kernel)
    thermal_instability: float = 0.0
    thermal_regime: str = "normal"
    
    # Negative-T indicators (if valve enabled)
    inversion_score: float = 0.0
    pumping_detected: bool = False
    
    # Session metrics
    turn: int = 0
    total_commits: int = 0
    total_proposals: int = 0
    
    # Derived urgency (computed from deviations)
    urgency: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_session(cls, session: EpistemicSession) -> "EpistemicVitals":
        """Extract vitals from a session."""
        snap = session.snapshot()
        thermal = session.thermal
        
        # Compute rates
        total_commits = snap.total_claims
        revision_rate = thermal.revision_count / max(total_commits, 1)
        contradiction_rate = thermal.contradiction_count / max(total_commits, 1)
        
        # Get recent frame stats
        hedge_rate = 0.0
        refusal_rate = 0.0
        if session.history:
            recent = session.history[-1]
            total_props = len(recent.committed) + len(recent.blocked)
            if total_props > 0:
                hedge_rate = len(recent.hedged) / total_props
                refusal_count = sum(1 for _, d in recent.blocked if 'REFUSE' in str(d.action))
                refusal_rate = refusal_count / total_props
        
        # Negative-T indicators
        inversion = 0.0
        pumping = False
        if session.analyzer:
            state = session.analyzer.get_state()
            inversion = state.inversion_score
            pumping = state.pumping_detected
        
        return cls(
            revision_rate=revision_rate,
            contradiction_rate=contradiction_rate,
            hedge_rate=hedge_rate,
            refusal_rate=refusal_rate,
            thermal_instability=thermal.instability,
            thermal_regime=thermal.regime,
            inversion_score=inversion,
            pumping_detected=pumping,
            turn=session.turn,
            total_commits=total_commits,
        )


# =============================================================================
# Setpoints (Target Operating Ranges)
# =============================================================================

@dataclass
class EpistemicSetpoints:
    """
    Target operating ranges for epistemic state.
    
    Analogous to Baby River's energy_target, heat_target.
    
    These define "healthy" operating ranges. Deviation from setpoints
    increases urgency and triggers adaptive responses.
    """
    # Rate targets (what's "healthy")
    revision_target: float = 0.05       # ~5% revision rate is normal
    contradiction_target: float = 0.02  # ~2% contradiction rate is healthy
    hedge_target: float = 0.15          # ~15% hedging is appropriate
    refusal_target: float = 0.01        # ~1% refusal is normal
    
    # Support targets
    support_deficit_target: float = 0.1  # Up to 10% needing support is ok
    retrieval_coverage_target: float = 0.9  # 90%+ coverage is healthy
    
    # Thermal target
    instability_target: float = 0.1     # Low baseline instability
    
    # Inversion target
    inversion_target: float = 0.2       # Some inversion is normal
    
    # Weights for urgency calculation (how much each deviation matters)
    revision_weight: float = 1.5        # Revisions are expensive
    contradiction_weight: float = 2.0   # Contradictions are bad
    hedge_weight: float = 0.5           # Hedging is cheap
    refusal_weight: float = 1.0         # Refusals have cost
    instability_weight: float = 1.5     # Thermal drift matters
    inversion_weight: float = 1.0       # Inversion indicates problems
    
    def compute_error(self, vitals: EpistemicVitals) -> Dict[str, float]:
        """Compute error from each setpoint."""
        return {
            'revision': abs(vitals.revision_rate - self.revision_target),
            'contradiction': abs(vitals.contradiction_rate - self.contradiction_target),
            'hedge': abs(vitals.hedge_rate - self.hedge_target),
            'refusal': abs(vitals.refusal_rate - self.refusal_target),
            'instability': abs(vitals.thermal_instability - self.instability_target),
            'inversion': abs(vitals.inversion_score - self.inversion_target),
        }
    
    def compute_urgency(self, vitals: EpistemicVitals) -> float:
        """
        Compute aggregate urgency from errors.
        
        Higher urgency = further from healthy operating range.
        This is analogous to Baby River's urgency_tau().
        """
        errors = self.compute_error(vitals)
        
        weighted_error = (
            errors['revision'] * self.revision_weight +
            errors['contradiction'] * self.contradiction_weight +
            errors['hedge'] * self.hedge_weight +
            errors['refusal'] * self.refusal_weight +
            errors['instability'] * self.instability_weight +
            errors['inversion'] * self.inversion_weight
        )
        
        total_weight = (
            self.revision_weight + self.contradiction_weight +
            self.hedge_weight + self.refusal_weight +
            self.instability_weight + self.inversion_weight
        )
        
        # Normalize to 0-1
        return min(weighted_error / total_weight, 1.0)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def for_domain(cls, domain: str) -> "EpistemicSetpoints":
        """Create domain-specific setpoints."""
        if domain in ('medical', 'legal', 'safety'):
            # Strict domains: lower targets, higher weights
            return cls(
                revision_target=0.02,
                contradiction_target=0.01,
                hedge_target=0.25,  # More hedging expected
                refusal_target=0.05,  # More refusal acceptable
                instability_target=0.05,
                contradiction_weight=3.0,
                revision_weight=2.0,
            )
        elif domain in ('creative', 'brainstorm', 'exploratory'):
            # Permissive domains: higher targets, lower weights
            return cls(
                revision_target=0.15,
                contradiction_target=0.1,
                hedge_target=0.05,  # Less hedging needed
                refusal_target=0.005,
                instability_target=0.3,
                contradiction_weight=0.5,
                revision_weight=0.5,
            )
        else:
            # Default balanced setpoints
            return cls()


# =============================================================================
# Tuning Deltas (Control Output)
# =============================================================================

@dataclass
class TuningDelta:
    """
    Output of the homeostat: adjustments to apply to the governor policy.
    
    These are multipliers/biases that modify the base policy, NOT
    absolute values. This allows smooth composition with calibrated baselines.
    """
    # Confidence adjustments
    confidence_ceiling_mult: float = 1.0  # Multiply max_confidence by this
    
    # Support/retrieval biases
    require_support_bias: float = 0.0     # Add to support requirement threshold
    retrieval_force_bias: float = 0.0     # Add to retrieval forcing threshold
    
    # Hedging/refusal preference
    hedge_preference: float = 0.0         # Bias toward hedging (+) vs committing (-)
    refuse_preference: float = 0.0        # Bias toward refusing (+) vs attempting (-)
    
    # Revision sensitivity
    revision_cost_mult: float = 1.0       # Multiply revision costs by this
    
    # Horizon control
    horizon_mult: float = 1.0             # Multiply commitment horizon by this
    
    # The urgency that produced this tuning
    source_urgency: float = 0.0
    
    def apply_to_policy(self, policy: DomainPolicy) -> DomainPolicy:
        """Apply tuning deltas to a domain policy."""
        # Create modified copy
        tuned = DomainPolicy(
            domain=policy.domain,
            max_confidence=policy.max_confidence * self.confidence_ceiling_mult,
            hedge_confidence_threshold=policy.hedge_confidence_threshold - self.hedge_preference * 0.1,
            force_retrieval_confidence=policy.force_retrieval_confidence - self.retrieval_force_bias * 0.1,
            max_revisions_per_session=max(1, int(policy.max_revisions_per_session * self.horizon_mult)),
            revision_cost_multiplier=policy.revision_cost_multiplier * self.revision_cost_mult,
            refusal_contradiction_density=policy.refusal_contradiction_density - self.refuse_preference * 0.05,
        )
        
        # Clamp values to valid ranges
        tuned.max_confidence = max(0.1, min(1.0, tuned.max_confidence))
        tuned.hedge_confidence_threshold = max(0.1, min(0.9, tuned.hedge_confidence_threshold))
        
        return tuned
    
    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Homeostat (The Controller)
# =============================================================================

class HomeostatMode(Enum):
    """Operating mode for the homeostat."""
    OBSERVE_ONLY = "observe"    # Log but don't output tuning
    SUGGEST = "suggest"         # Output tuning but don't require application
    ACTIVE = "active"           # Full adaptive control


class ExplorationContext(Enum):
    """
    Exploration contexts that modify homeostatic behavior.
    
    The default homeostat converges to "reliable, boring, conservative."
    These contexts deliberately loosen constraints for discovery.
    
    Biology pairs homeostasis with:
    - Development (growth phases)
    - Plasticity (learning windows)
    - Evolutionary pressure (variation)
    
    We simulate these with explicit context switches.
    """
    # Standard operation - homeostasis as normal
    STANDARD = "standard"
    
    # Research context - exploring a topic, higher tolerance for speculation
    RESEARCH = "research"
    
    # Brainstorm context - generating ideas, minimal filtering
    BRAINSTORM = "brainstorm"
    
    # Hypothesis context - explicitly tentative claims, easy revision
    HYPOTHESIS = "hypothesis"
    
    # Synthesis context - combining ideas, tolerance for novel connections
    SYNTHESIS = "synthesis"
    
    # Devil's advocate - deliberately exploring contrary positions
    DEVILS_ADVOCATE = "devils_advocate"
    
    # Calibration context - system is being tuned, extra logging
    CALIBRATION = "calibration"


@dataclass
class ExplorationBudget:
    """
    Budget for exploration that gets spent and replenished.
    
    Prevents indefinite exploration drift while allowing bounded discovery.
    Think of it as "plasticity tokens" that get consumed during exploration
    and regenerate during consolidation.
    """
    # Current budget (0-1)
    remaining: float = 1.0
    
    # Regeneration rate per turn in standard mode
    regen_rate: float = 0.05
    
    # Consumption rate per turn in exploration mode
    consume_rate: float = 0.1
    
    # Minimum budget to enter exploration (can't explore when depleted)
    min_to_explore: float = 0.2
    
    # Maximum budget (caps regeneration)
    max_budget: float = 1.0
    
    def can_explore(self) -> bool:
        """Check if exploration is allowed."""
        return self.remaining >= self.min_to_explore
    
    def consume(self, amount: Optional[float] = None):
        """Consume budget during exploration."""
        cost = amount if amount is not None else self.consume_rate
        self.remaining = max(0.0, self.remaining - cost)
    
    def regenerate(self, amount: Optional[float] = None):
        """Regenerate budget during standard operation."""
        gain = amount if amount is not None else self.regen_rate
        self.remaining = min(self.max_budget, self.remaining + gain)
    
    def to_dict(self) -> dict:
        return {
            'remaining': self.remaining,
            'can_explore': self.can_explore(),
            'regen_rate': self.regen_rate,
            'consume_rate': self.consume_rate,
        }


@dataclass 
class ExplorationProfile:
    """
    Configuration for a specific exploration context.
    
    Defines how constraints are modified during exploration.
    """
    context: ExplorationContext
    
    # Confidence adjustments (multipliers on normal behavior)
    confidence_ceiling_boost: float = 0.0    # Add to confidence ceiling
    hedge_requirement_reduction: float = 0.0  # Reduce hedge requirements
    
    # Revision adjustments
    revision_cost_discount: float = 0.0      # Reduce revision costs
    revision_budget_boost: int = 0           # Extra revisions allowed
    
    # Support adjustments
    support_requirement_reduction: float = 0.0  # Reduce support requirements
    
    # Commitment adjustments
    commitment_tentativeness: float = 0.0    # Mark commits as tentative (0-1)
    
    # Urgency scaling (how much to dampen urgency response)
    urgency_dampening: float = 0.0           # 0 = normal, 1 = ignore urgency
    
    # Budget cost per turn in this context
    budget_cost: float = 0.1
    
    # Description for logging/UI
    description: str = ""
    
    def to_dict(self) -> dict:
        return {
            'context': self.context.value,
            'confidence_boost': self.confidence_ceiling_boost,
            'hedge_reduction': self.hedge_requirement_reduction,
            'revision_discount': self.revision_cost_discount,
            'urgency_dampening': self.urgency_dampening,
            'budget_cost': self.budget_cost,
        }


# Pre-defined exploration profiles
EXPLORATION_PROFILES: Dict[ExplorationContext, ExplorationProfile] = {
    ExplorationContext.STANDARD: ExplorationProfile(
        context=ExplorationContext.STANDARD,
        description="Normal homeostatic operation",
        budget_cost=0.0,  # No cost, regenerates
    ),
    
    ExplorationContext.RESEARCH: ExplorationProfile(
        context=ExplorationContext.RESEARCH,
        confidence_ceiling_boost=0.1,
        hedge_requirement_reduction=0.1,
        support_requirement_reduction=0.2,
        urgency_dampening=0.3,
        budget_cost=0.05,
        description="Exploring a topic - higher speculation tolerance",
    ),
    
    ExplorationContext.BRAINSTORM: ExplorationProfile(
        context=ExplorationContext.BRAINSTORM,
        confidence_ceiling_boost=0.2,
        hedge_requirement_reduction=0.3,
        revision_cost_discount=0.5,
        revision_budget_boost=5,
        support_requirement_reduction=0.4,
        commitment_tentativeness=0.8,
        urgency_dampening=0.7,
        budget_cost=0.15,
        description="Generating ideas - minimal filtering, everything tentative",
    ),
    
    ExplorationContext.HYPOTHESIS: ExplorationProfile(
        context=ExplorationContext.HYPOTHESIS,
        confidence_ceiling_boost=0.15,
        hedge_requirement_reduction=0.2,
        revision_cost_discount=0.7,  # Cheap to revise hypotheses
        revision_budget_boost=10,
        commitment_tentativeness=0.9,  # Almost everything is tentative
        urgency_dampening=0.5,
        budget_cost=0.08,
        description="Explicitly tentative claims - easy revision expected",
    ),
    
    ExplorationContext.SYNTHESIS: ExplorationProfile(
        context=ExplorationContext.SYNTHESIS,
        confidence_ceiling_boost=0.1,
        hedge_requirement_reduction=0.15,
        support_requirement_reduction=0.3,  # Novel connections may lack direct support
        urgency_dampening=0.4,
        budget_cost=0.1,
        description="Combining ideas - tolerance for novel connections",
    ),
    
    ExplorationContext.DEVILS_ADVOCATE: ExplorationProfile(
        context=ExplorationContext.DEVILS_ADVOCATE,
        confidence_ceiling_boost=0.1,
        hedge_requirement_reduction=0.2,
        revision_cost_discount=0.8,  # Can easily abandon contrary positions
        commitment_tentativeness=1.0,  # Everything is explicitly provisional
        urgency_dampening=0.6,
        budget_cost=0.12,
        description="Exploring contrary positions - fully provisional",
    ),
    
    ExplorationContext.CALIBRATION: ExplorationProfile(
        context=ExplorationContext.CALIBRATION,
        urgency_dampening=0.0,  # Full urgency response for accurate measurement
        budget_cost=0.0,
        description="System calibration - extra logging, normal constraints",
    ),
}


class Homeostat:
    """
    Adaptive gain scheduler for epistemic governance.
    
    Observes system vitals, compares to setpoints, and outputs tuning deltas
    that adjust governor parameters based on urgency.
    
    Key design rules (from ChatGPT's analysis):
    1. Never touch ledger contents - only adjust thresholds
    2. Use urgency to increase CONTROL WORK, not randomness
    3. Keep costs explicit and centralized
    
    IMPORTANT: Default homeostasis converges to "reliable, boring, conservative."
    Use exploration contexts to deliberately loosen constraints for discovery.
    
    "Control work" in LLM context:
    - Force retrieval
    - Force structured verification
    - Clamp confidence / require hedging
    - Shrink allowed claim types
    - Shorten horizon / ask clarifying question
    - Switch to "cite-or-refuse" mode
    
    Exploration contexts (the antidote to mediocrity):
    - RESEARCH: Higher speculation tolerance
    - BRAINSTORM: Minimal filtering, everything tentative
    - HYPOTHESIS: Easy revision expected
    - SYNTHESIS: Tolerance for novel connections
    - DEVILS_ADVOCATE: Exploring contrary positions
    """
    
    def __init__(
        self,
        setpoints: Optional[EpistemicSetpoints] = None,
        baseline: Optional[BaselineProfile] = None,
        mode: HomeostatMode = HomeostatMode.SUGGEST,
    ):
        self.setpoints = setpoints or EpistemicSetpoints()
        self.baseline = baseline
        self.mode = mode
        
        # Exploration state
        self._context = ExplorationContext.STANDARD
        self._exploration_budget = ExplorationBudget()
        self._context_history: List[Tuple[datetime, ExplorationContext]] = []
        
        # History for learning/analysis
        self._vitals_history: List[EpistemicVitals] = []
        self._tuning_history: List[TuningDelta] = []
        self._outcome_history: List[Dict[str, Any]] = []
        
        # Smoothing for stability (exponential moving average)
        self._ema_urgency: float = 0.0
        self._ema_alpha: float = 0.3  # Smoothing factor
        
        # Tuning parameters (these could be calibrated)
        self._urgency_to_confidence = -0.3     # High urgency → lower confidence ceiling
        self._urgency_to_support = 0.5         # High urgency → more support required
        self._urgency_to_retrieval = 0.4       # High urgency → more retrieval
        self._urgency_to_hedge = 0.6           # High urgency → prefer hedging
        self._urgency_to_refuse = 0.3          # High urgency → more willing to refuse
        self._urgency_to_revision_cost = 0.5   # High urgency → revisions more expensive
        self._urgency_to_horizon = -0.3        # High urgency → shorter horizon
    
    # =========================================================================
    # Exploration Context Management
    # =========================================================================
    
    def enter_exploration(self, context: ExplorationContext) -> bool:
        """
        Enter an exploration context.
        
        Returns False if insufficient budget or already in that context.
        """
        if context == ExplorationContext.STANDARD:
            # Always allowed to return to standard
            self._context = context
            return True
        
        if context == self._context:
            return True  # Already there
        
        if not self._exploration_budget.can_explore():
            return False  # Budget depleted
        
        self._context = context
        self._context_history.append((datetime.now(), context))
        return True
    
    def exit_exploration(self):
        """Return to standard context."""
        self._context = ExplorationContext.STANDARD
        self._context_history.append((datetime.now(), ExplorationContext.STANDARD))
    
    def get_exploration_profile(self) -> ExplorationProfile:
        """Get the profile for the current context."""
        return EXPLORATION_PROFILES.get(
            self._context, 
            EXPLORATION_PROFILES[ExplorationContext.STANDARD]
        )
    
    @property
    def context(self) -> ExplorationContext:
        return self._context
    
    @property
    def exploration_budget(self) -> ExplorationBudget:
        return self._exploration_budget
    
    # =========================================================================
    # Core Observation and Tuning
    # =========================================================================
    
    def observe(self, session: EpistemicSession) -> EpistemicVitals:
        """
        Observe current system state.
        
        Returns vitals with computed urgency.
        """
        vitals = EpistemicVitals.from_session(session)
        vitals.urgency = self.setpoints.compute_urgency(vitals)
        
        # Update EMA
        self._ema_urgency = (
            self._ema_alpha * vitals.urgency +
            (1 - self._ema_alpha) * self._ema_urgency
        )
        
        # Update exploration budget based on context
        profile = self.get_exploration_profile()
        if self._context == ExplorationContext.STANDARD:
            self._exploration_budget.regenerate()
        else:
            self._exploration_budget.consume(profile.budget_cost)
            
            # Auto-exit exploration if budget depleted
            if not self._exploration_budget.can_explore():
                self.exit_exploration()
        
        # Record
        self._vitals_history.append(vitals)
        
        return vitals
    
    def compute_tuning(self, vitals: EpistemicVitals) -> TuningDelta:
        """
        Compute tuning deltas from vitals and exploration context.
        
        In standard mode: Higher urgency → more conservative, more control work.
        In exploration mode: Constraints loosened, urgency dampened.
        """
        if self.mode == HomeostatMode.OBSERVE_ONLY:
            return TuningDelta()  # No tuning in observe-only mode
        
        # Get exploration profile
        profile = self.get_exploration_profile()
        
        # Apply urgency dampening from exploration context
        effective_urgency = self._ema_urgency * (1.0 - profile.urgency_dampening)
        
        # Base tuning from urgency (conservative direction)
        base_confidence_mult = 1.0 + (self._urgency_to_confidence * effective_urgency)
        base_hedge_pref = self._urgency_to_hedge * effective_urgency
        base_revision_cost = 1.0 + (self._urgency_to_revision_cost * effective_urgency)
        
        # Apply exploration boosts (loosening direction)
        tuning = TuningDelta(
            # Confidence: base (lowered by urgency) + exploration boost
            confidence_ceiling_mult=base_confidence_mult + profile.confidence_ceiling_boost,
            
            # Support: urgency increases requirement, exploration reduces it
            require_support_bias=(
                self._urgency_to_support * effective_urgency 
                - profile.support_requirement_reduction
            ),
            
            # Retrieval: urgency forces more
            retrieval_force_bias=self._urgency_to_retrieval * effective_urgency,
            
            # Hedging: urgency increases, exploration reduces
            hedge_preference=base_hedge_pref - profile.hedge_requirement_reduction,
            
            # Refusal: urgency increases willingness
            refuse_preference=self._urgency_to_refuse * effective_urgency,
            
            # Revision cost: urgency increases, exploration discounts
            revision_cost_mult=base_revision_cost * (1.0 - profile.revision_cost_discount),
            
            # Horizon: urgency shortens
            horizon_mult=1.0 + (self._urgency_to_horizon * effective_urgency),
            
            source_urgency=effective_urgency,
        )
        
        # Clamp multipliers to reasonable ranges
        tuning.confidence_ceiling_mult = max(0.5, min(1.2, tuning.confidence_ceiling_mult))
        tuning.revision_cost_mult = max(0.3, min(3.0, tuning.revision_cost_mult))
        tuning.horizon_mult = max(0.3, min(1.5, tuning.horizon_mult))
        
        # Clamp biases
        tuning.require_support_bias = max(-0.3, min(0.5, tuning.require_support_bias))
        tuning.hedge_preference = max(-0.3, min(0.6, tuning.hedge_preference))
        
        # Record
        self._tuning_history.append(tuning)
        
        return tuning
        
        return tuning
    
    def record_outcome(self, frame: EpistemicFrame):
        """
        Record the outcome of a governance decision.
        
        Used for offline analysis and potential future learning.
        """
        outcome = {
            'timestamp': datetime.now().isoformat(),
            'context': self._context.value,
            'committed': len(frame.committed),
            'blocked': len(frame.blocked),
            'hedged': len(frame.hedged),
            'revision_required': len(frame.revision_required),
            'thermal_delta': frame.thermal_delta,
            'errors': len(frame.errors),
            'exploration_budget': self._exploration_budget.remaining,
        }
        self._outcome_history.append(outcome)
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information for debugging."""
        profile = self.get_exploration_profile()
        return {
            'mode': self.mode.value,
            'context': self._context.value,
            'context_description': profile.description,
            'exploration_budget': self._exploration_budget.to_dict(),
            'ema_urgency': self._ema_urgency,
            'effective_urgency': self._ema_urgency * (1.0 - profile.urgency_dampening),
            'vitals_count': len(self._vitals_history),
            'recent_vitals': self._vitals_history[-1].to_dict() if self._vitals_history else None,
            'recent_tuning': self._tuning_history[-1].to_dict() if self._tuning_history else None,
            'setpoints': self.setpoints.to_dict(),
        }
    
    def get_urgency_trace(self, n: int = 20) -> List[float]:
        """Get recent urgency values for plotting."""
        return [v.urgency for v in self._vitals_history[-n:]]
    
    def get_context_trace(self, n: int = 20) -> List[str]:
        """Get recent context history."""
        return [ctx.value for _, ctx in self._context_history[-n:]]
    
    def reset(self):
        """Reset homeostat state."""
        self._vitals_history.clear()
        self._tuning_history.clear()
        self._outcome_history.clear()
        self._context_history.clear()
        self._ema_urgency = 0.0
        self._context = ExplorationContext.STANDARD
        self._exploration_budget = ExplorationBudget()


# =============================================================================
# Homeostat-Aware Session Wrapper
# =============================================================================

class HomeostaticSession:
    """
    Session wrapper that applies homeostatic control with exploration support.
    
    This wraps an EpistemicSession and applies homeostat tuning
    to every governance decision.
    
    Usage:
        session = HomeostaticSession(
            base_policy=policy,
            setpoints=EpistemicSetpoints.for_domain("medical"),
        )
        
        # Standard operation
        frame = session.govern("The dosage is 500mg.")
        
        # Enter exploration mode for brainstorming
        session.enter_exploration(ExplorationContext.BRAINSTORM)
        frame = session.govern("What if the dosage could be adaptive?")
        session.exit_exploration()
        
        # Check exploration budget
        print(f"Budget: {session.exploration_budget.remaining}")
    """
    
    def __init__(
        self,
        base_policy: Optional[PolicyProfile] = None,
        setpoints: Optional[EpistemicSetpoints] = None,
        baseline: Optional[BaselineProfile] = None,
        homeostat_mode: HomeostatMode = HomeostatMode.ACTIVE,
    ):
        from epistemic_governor.session import create_session
        
        self.inner = create_session(mode="normal", enable_valve=True)
        self.base_policy = base_policy
        self.homeostat = Homeostat(
            setpoints=setpoints,
            baseline=baseline,
            mode=homeostat_mode,
        )
    
    def enter_exploration(self, context: ExplorationContext) -> bool:
        """
        Enter an exploration context.
        
        Returns False if insufficient budget.
        """
        return self.homeostat.enter_exploration(context)
    
    def exit_exploration(self):
        """Return to standard context."""
        self.homeostat.exit_exploration()
    
    @property
    def context(self) -> ExplorationContext:
        """Current exploration context."""
        return self.homeostat.context
    
    @property
    def exploration_budget(self) -> ExplorationBudget:
        """Current exploration budget."""
        return self.homeostat.exploration_budget
    
    def govern(self, text: str, domain: str = "general") -> EpistemicFrame:
        """Govern text with homeostatic control and exploration awareness."""
        # Observe current state
        vitals = self.homeostat.observe(self.inner)
        
        # Compute tuning (exploration-aware)
        tuning = self.homeostat.compute_tuning(vitals)
        
        # Get exploration profile for tentativeness marking
        profile = self.homeostat.get_exploration_profile()
        
        # Apply tuning to base policy
        if self.base_policy:
            base_domain_policy = self.base_policy.get_policy(domain)
            tuned_policy = tuning.apply_to_policy(base_domain_policy)
            envelope = self._policy_to_envelope(tuned_policy)
        else:
            envelope = self.inner.kernel.get_envelope(domain=domain)
            # Apply tuning directly to envelope
            envelope.max_confidence *= tuning.confidence_ceiling_mult
            if tuning.hedge_preference > 0.3:
                envelope.require_hedges = True
        
        # Govern with tuned envelope
        frame = self.inner.govern(text, envelope=envelope)
        
        # Mark commits as tentative if in exploration mode
        if profile.commitment_tentativeness > 0.5:
            # In a full implementation, we'd mark the ledger entries
            # For now, add a note to the frame
            if not hasattr(frame, 'exploration_metadata'):
                frame.exploration_metadata = {}
            frame.exploration_metadata['tentative'] = True
            frame.exploration_metadata['context'] = self.homeostat.context.value
        
        # Record outcome
        self.homeostat.record_outcome(frame)
        
        return frame
    
    def _policy_to_envelope(self, policy: DomainPolicy):
        """Convert a domain policy to a generation envelope."""
        from epistemic_governor.governor import GenerationEnvelope, ClaimType
        
        return GenerationEnvelope(
            max_confidence=policy.max_confidence,
            require_hedges=(policy.hedge_confidence_threshold < 0.5),
            must_retrieve_types=set(policy.force_retrieval_types),
        )
    
    @property
    def turn(self) -> int:
        return self.inner.turn
    
    @property
    def thermal(self) -> ThermalState:
        return self.inner.thermal
    
    def snapshot(self):
        return self.inner.snapshot()
    
    def strata(self, limit: int = 20):
        return self.inner.strata(limit)
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get combined diagnostics."""
        return {
            'session': self.inner.snapshot().to_dict(),
            'homeostat': self.homeostat.get_diagnostics(),
        }
    
    def reset(self):
        self.inner.reset()
        self.homeostat.reset()


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=== Homeostat Demo ===\n")
    
    # Create homeostat with default setpoints
    setpoints = EpistemicSetpoints()
    homeostat = Homeostat(setpoints=setpoints, mode=HomeostatMode.ACTIVE)
    
    # Simulate some vitals
    print("1. Standard mode - healthy state...")
    healthy_vitals = EpistemicVitals(
        revision_rate=0.05,
        contradiction_rate=0.02,
        hedge_rate=0.15,
        refusal_rate=0.01,
        thermal_instability=0.1,
        inversion_score=0.2,
    )
    healthy_vitals.urgency = setpoints.compute_urgency(healthy_vitals)
    homeostat._ema_urgency = healthy_vitals.urgency
    
    tuning = homeostat.compute_tuning(healthy_vitals)
    print(f"   Context: {homeostat.context.value}")
    print(f"   Urgency: {healthy_vitals.urgency:.3f}")
    print(f"   Confidence mult: {tuning.confidence_ceiling_mult:.3f}")
    print(f"   Hedge preference: {tuning.hedge_preference:.3f}")
    
    print("\n2. Standard mode - stressed state...")
    stressed_vitals = EpistemicVitals(
        revision_rate=0.20,
        contradiction_rate=0.15,
        hedge_rate=0.05,
        refusal_rate=0.02,
        thermal_instability=0.5,
        inversion_score=0.6,
    )
    stressed_vitals.urgency = setpoints.compute_urgency(stressed_vitals)
    homeostat._ema_urgency = stressed_vitals.urgency
    
    tuning2 = homeostat.compute_tuning(stressed_vitals)
    print(f"   Context: {homeostat.context.value}")
    print(f"   Urgency: {stressed_vitals.urgency:.3f}")
    print(f"   Confidence mult: {tuning2.confidence_ceiling_mult:.3f}")
    print(f"   Hedge preference: {tuning2.hedge_preference:.3f}")
    print(f"   Revision cost mult: {tuning2.revision_cost_mult:.3f}")
    
    print("\n3. Enter BRAINSTORM exploration...")
    success = homeostat.enter_exploration(ExplorationContext.BRAINSTORM)
    print(f"   Entered: {success}")
    print(f"   Context: {homeostat.context.value}")
    print(f"   Budget remaining: {homeostat.exploration_budget.remaining:.2f}")
    
    tuning3 = homeostat.compute_tuning(stressed_vitals)
    profile = homeostat.get_exploration_profile()
    print(f"   Urgency dampening: {profile.urgency_dampening}")
    print(f"   Effective urgency: {stressed_vitals.urgency * (1 - profile.urgency_dampening):.3f}")
    print(f"   Confidence mult: {tuning3.confidence_ceiling_mult:.3f} (boosted)")
    print(f"   Hedge preference: {tuning3.hedge_preference:.3f} (reduced)")
    print(f"   Revision cost mult: {tuning3.revision_cost_mult:.3f} (discounted)")
    
    print("\n4. Enter HYPOTHESIS exploration...")
    homeostat.enter_exploration(ExplorationContext.HYPOTHESIS)
    profile = homeostat.get_exploration_profile()
    print(f"   Context: {homeostat.context.value}")
    print(f"   Description: {profile.description}")
    print(f"   Commitment tentativeness: {profile.commitment_tentativeness}")
    print(f"   Revision cost discount: {profile.revision_cost_discount}")
    
    print("\n5. Exit exploration...")
    homeostat.exit_exploration()
    print(f"   Context: {homeostat.context.value}")
    print(f"   Budget remaining: {homeostat.exploration_budget.remaining:.2f}")
    
    print("\n6. Exploration budget mechanics...")
    homeostat.reset()
    print(f"   Fresh budget: {homeostat.exploration_budget.remaining:.2f}")
    
    # Simulate several turns of exploration
    homeostat.enter_exploration(ExplorationContext.BRAINSTORM)
    for i in range(8):
        homeostat._exploration_budget.consume()
        print(f"   Turn {i+1}: budget = {homeostat.exploration_budget.remaining:.2f}")
        if not homeostat.exploration_budget.can_explore():
            print(f"   → Budget depleted, cannot continue exploration")
            break
    
    print("\n7. Budget regeneration in standard mode...")
    homeostat.exit_exploration()
    for i in range(5):
        homeostat._exploration_budget.regenerate()
        print(f"   Turn {i+1}: budget = {homeostat.exploration_budget.remaining:.2f}")
    
    print("\n8. Available exploration contexts:")
    for ctx, profile in EXPLORATION_PROFILES.items():
        print(f"   {ctx.value}: {profile.description}")
    
    print("\n✓ Homeostat with exploration modes working")
    print("\nKey insight: Exploration is BOUNDED by budget")
    print("  - Budget consumed during exploration")
    print("  - Budget regenerates during standard operation")
    print("  - Prevents indefinite drift into 'creative mode'")
    print("  - Discovery is allowed but not free")
