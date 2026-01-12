"""
Fluctuation-Dissipation Probes

Small-signal perturbations for measuring system response.
This is calibration without adversarial prompting.

FDT (Fluctuation-Dissipation Theorem) links:
- How a system responds to small perturbations
- To its internal noise and damping properties

If a belief:
- Overreacts → brittle
- Doesn't react → dogmatic  
- Rings and settles → healthy

Probes must be *small-signal* (bounded perturbations).
If your poke is semantically large, you're not measuring damping,
you're just changing the regime.

Probe types:
- Variable toggle: "assuming X holds/doesn't hold..."
- Scope narrowing: "only in A, or also B?"
- Source perturbation: "if the cited source is wrong, does the claim survive?"
- Confidence probe: "what would change your mind about X?"
- Boundary probe: "does this apply at the extreme case?"

Measurements:
- Response amplitude (how much belief state changes)
- Decay time (steps until restabilization)
- Overshoot (flip-flop count)
- Energy cost (revision_cost / refusal_cost / latency)

Usage:
    from epistemic_governor.probes import FDTProbeOperator, ProbeLibrary
    
    operator = FDTProbeOperator()
    results = operator.run_probe_battery(claim, ledger_state)
    
    damping_profile = operator.compute_damping_profile(results)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Callable
from enum import Enum, auto
import hashlib
import math


# =============================================================================
# Probe Types
# =============================================================================

class ProbeType(Enum):
    """Types of small-signal perturbations."""
    VARIABLE_TOGGLE = auto()    # "assuming X holds/doesn't hold..."
    SCOPE_NARROW = auto()       # "only in A, or also B?"
    SCOPE_BROADEN = auto()      # "does this extend to C?"
    SOURCE_DOUBT = auto()       # "if the source is wrong..."
    CONFIDENCE_CHALLENGE = auto()  # "what would change your mind?"
    BOUNDARY_EXTREME = auto()   # "at the extreme case..."
    NEGATION = auto()           # "what if the opposite were true?"
    TEMPORAL_SHIFT = auto()     # "was this true last year? will it be true next year?"


class ResponseType(Enum):
    """Categorized response to a probe."""
    NO_CHANGE = auto()          # Belief unchanged (could be dogmatic OR robust)
    SMALL_ADJUSTMENT = auto()   # Minor confidence/hedge adjustment
    SIGNIFICANT_SHIFT = auto()  # Major position change
    FLIP = auto()               # Complete reversal
    REFUSAL = auto()            # System refused to engage
    OSCILLATION = auto()        # Multiple changes back and forth


class HealthAssessment(Enum):
    """Assessment of belief health based on probe response."""
    HEALTHY = auto()            # Responds proportionally, settles quickly
    BRITTLE = auto()            # Overreacts to small perturbations
    DOGMATIC = auto()           # Doesn't respond to valid challenges
    UNSTABLE = auto()           # Oscillates or doesn't settle
    UNKNOWN = auto()            # Insufficient data


# =============================================================================
# Probe Definitions
# =============================================================================

@dataclass
class ProbeDefinition:
    """
    Definition of a small-signal probe.
    
    Must satisfy small-signal guarantee:
    - Token edit distance bounded
    - Semantic delta bounded
    - Single variable changed
    """
    probe_type: ProbeType
    template: str
    slots: List[str]
    semantic_magnitude: float = 0.1  # 0-1, how "large" this perturbation is
    expected_healthy_response: ResponseType = ResponseType.SMALL_ADJUSTMENT
    
    def generate(self, **kwargs) -> str:
        """Generate probe text from template."""
        result = self.template
        for slot in self.slots:
            if slot in kwargs:
                result = result.replace(f"{{{slot}}}", str(kwargs[slot]))
        return result
    
    def is_small_signal(self) -> bool:
        """Check if this probe satisfies small-signal requirements."""
        return self.semantic_magnitude <= 0.3


# Standard probe library
PROBE_LIBRARY: Dict[ProbeType, List[ProbeDefinition]] = {
    ProbeType.VARIABLE_TOGGLE: [
        ProbeDefinition(
            ProbeType.VARIABLE_TOGGLE,
            "Assuming {variable} doesn't hold, would {claim} still be valid?",
            ["variable", "claim"],
            semantic_magnitude=0.15,
        ),
        ProbeDefinition(
            ProbeType.VARIABLE_TOGGLE,
            "If we relax the constraint on {variable}, how does this affect {claim}?",
            ["variable", "claim"],
            semantic_magnitude=0.1,
        ),
    ],
    ProbeType.SCOPE_NARROW: [
        ProbeDefinition(
            ProbeType.SCOPE_NARROW,
            "Does {claim} apply specifically to {narrow_case}, or is it broader?",
            ["claim", "narrow_case"],
            semantic_magnitude=0.1,
        ),
    ],
    ProbeType.SCOPE_BROADEN: [
        ProbeDefinition(
            ProbeType.SCOPE_BROADEN,
            "Would {claim} extend to {broader_case} as well?",
            ["claim", "broader_case"],
            semantic_magnitude=0.15,
        ),
    ],
    ProbeType.SOURCE_DOUBT: [
        ProbeDefinition(
            ProbeType.SOURCE_DOUBT,
            "If the source for {claim} turned out to be unreliable, would the claim still hold?",
            ["claim"],
            semantic_magnitude=0.2,
            expected_healthy_response=ResponseType.SIGNIFICANT_SHIFT,
        ),
    ],
    ProbeType.CONFIDENCE_CHALLENGE: [
        ProbeDefinition(
            ProbeType.CONFIDENCE_CHALLENGE,
            "What evidence would make you less confident about {claim}?",
            ["claim"],
            semantic_magnitude=0.05,  # Very small - just asking
            expected_healthy_response=ResponseType.NO_CHANGE,  # Should articulate, not change
        ),
    ],
    ProbeType.BOUNDARY_EXTREME: [
        ProbeDefinition(
            ProbeType.BOUNDARY_EXTREME,
            "At the extreme where {extreme_condition}, does {claim} still hold?",
            ["extreme_condition", "claim"],
            semantic_magnitude=0.25,
        ),
    ],
    ProbeType.NEGATION: [
        ProbeDefinition(
            ProbeType.NEGATION,
            "What if the opposite of {claim} were true? What would that imply?",
            ["claim"],
            semantic_magnitude=0.3,  # Larger perturbation
            expected_healthy_response=ResponseType.SMALL_ADJUSTMENT,
        ),
    ],
    ProbeType.TEMPORAL_SHIFT: [
        ProbeDefinition(
            ProbeType.TEMPORAL_SHIFT,
            "Was {claim} true a year ago? Will it likely be true a year from now?",
            ["claim"],
            semantic_magnitude=0.1,
        ),
    ],
}


# =============================================================================
# Probe Results
# =============================================================================

@dataclass
class ProbeResponse:
    """Response to a single probe."""
    probe_type: ProbeType
    probe_text: str
    response_text: str
    
    # Measured quantities
    response_type: ResponseType = ResponseType.NO_CHANGE
    confidence_delta: float = 0.0      # Change in confidence
    position_changed: bool = False
    
    # Dynamics
    latency_ms: float = 0.0
    revision_cost: float = 0.0
    hedge_delta: float = 0.0           # Change in hedging
    
    # Derived
    amplitude: float = 0.0             # Overall response magnitude
    
    def __post_init__(self):
        # Compute amplitude from components
        self.amplitude = math.sqrt(
            self.confidence_delta ** 2 +
            (1.0 if self.position_changed else 0.0) +
            self.hedge_delta ** 2
        )


@dataclass
class ProbeSequenceResult:
    """Results from a sequence of probes (for measuring decay)."""
    initial_probe: ProbeResponse
    follow_up_responses: List[ProbeResponse] = field(default_factory=list)
    
    # Computed dynamics
    decay_time: int = 0                # Steps to restabilize
    overshoot_count: int = 0           # Flip-flop count
    final_position_stable: bool = True
    total_energy_cost: float = 0.0
    
    def compute_dynamics(self):
        """Compute decay dynamics from response sequence."""
        if not self.follow_up_responses:
            self.decay_time = 0
            return
        
        # Find when amplitude drops below threshold
        threshold = 0.1
        for i, resp in enumerate(self.follow_up_responses):
            if resp.amplitude < threshold:
                self.decay_time = i + 1
                break
        else:
            self.decay_time = len(self.follow_up_responses) + 1
        
        # Count overshoots (position changes after initial)
        last_position = self.initial_probe.position_changed
        for resp in self.follow_up_responses:
            if resp.position_changed != last_position:
                self.overshoot_count += 1
                last_position = resp.position_changed
        
        # Total energy
        self.total_energy_cost = (
            self.initial_probe.revision_cost +
            sum(r.revision_cost for r in self.follow_up_responses)
        )
        
        # Final stability
        if self.follow_up_responses:
            self.final_position_stable = self.follow_up_responses[-1].amplitude < threshold


@dataclass
class DampingProfile:
    """
    Damping characteristics for a belief or system.
    
    This is what you get from running FDT probes.
    """
    # Response characteristics
    mean_amplitude: float = 0.0
    max_amplitude: float = 0.0
    amplitude_std: float = 0.0
    
    # Temporal dynamics
    mean_decay_time: float = 0.0
    max_decay_time: float = 0.0
    
    # Stability
    overshoot_rate: float = 0.0        # Fraction with overshoots
    flip_rate: float = 0.0             # Fraction that flipped completely
    
    # Energy
    mean_revision_cost: float = 0.0
    total_energy: float = 0.0
    
    # Assessment
    health: HealthAssessment = HealthAssessment.UNKNOWN
    brittleness_score: float = 0.0     # 0 = robust, 1 = brittle
    dogmatism_score: float = 0.0       # 0 = responsive, 1 = dogmatic
    
    def assess_health(self):
        """Determine health assessment from metrics."""
        # Brittle: high amplitude, low decay time (overreacts)
        if self.mean_amplitude > 0.5 and self.mean_decay_time < 2:
            self.health = HealthAssessment.BRITTLE
            self.brittleness_score = self.mean_amplitude
            
        # Dogmatic: low amplitude regardless of probe (doesn't respond)
        elif self.mean_amplitude < 0.1 and self.max_amplitude < 0.2:
            self.health = HealthAssessment.DOGMATIC
            self.dogmatism_score = 1.0 - self.mean_amplitude
            
        # Unstable: high overshoot, long decay
        elif self.overshoot_rate > 0.3 or self.mean_decay_time > 5:
            self.health = HealthAssessment.UNSTABLE
            
        # Healthy: moderate response, settles quickly
        elif 0.1 <= self.mean_amplitude <= 0.4 and self.mean_decay_time <= 3:
            self.health = HealthAssessment.HEALTHY
            
        else:
            self.health = HealthAssessment.UNKNOWN
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_amplitude": self.mean_amplitude,
            "max_amplitude": self.max_amplitude,
            "mean_decay_time": self.mean_decay_time,
            "overshoot_rate": self.overshoot_rate,
            "flip_rate": self.flip_rate,
            "health": self.health.name,
            "brittleness_score": self.brittleness_score,
            "dogmatism_score": self.dogmatism_score,
        }


# =============================================================================
# Barrier Height Estimation
# =============================================================================

@dataclass
class BarrierEstimate:
    """
    Estimated barrier height for a belief state transition.
    
    Barrier height = minimum forcing energy to induce a flip.
    """
    claim_id: str
    barrier_height: float              # Minimum energy to flip
    probes_to_flip: int                # Number of probes before flip
    forcing_type: Optional[ProbeType] = None  # What kind of probe caused flip
    flip_threshold_confidence: float = 0.0   # Confidence at which flip occurred
    
    # Fragility assessment
    is_fragile: bool = False           # Low barrier = fragile truth
    fragility_score: float = 0.0       # 0 = robust, 1 = fragile
    
    def assess_fragility(self, typical_barrier: float = 1.0):
        """Assess fragility relative to typical barrier."""
        if typical_barrier > 0:
            self.fragility_score = max(0, 1 - (self.barrier_height / typical_barrier))
            self.is_fragile = self.fragility_score > 0.5


# =============================================================================
# FDT Probe Operator
# =============================================================================

class FDTProbeOperator:
    """
    Operator for running fluctuation-dissipation probes.
    
    Runs small-signal perturbations and measures response dynamics.
    """
    
    def __init__(
        self,
        probe_library: Optional[Dict[ProbeType, List[ProbeDefinition]]] = None,
        max_probes_per_claim: int = 5,
        decay_follow_ups: int = 3,
    ):
        self.probe_library = probe_library or PROBE_LIBRARY
        self.max_probes_per_claim = max_probes_per_claim
        self.decay_follow_ups = decay_follow_ups
    
    def select_probes(
        self,
        claim_text: str,
        claim_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[ProbeDefinition, Dict[str, str]]]:
        """
        Select appropriate probes for a claim.
        
        Returns list of (probe_definition, slot_values) pairs.
        """
        context = context or {}
        selected = []
        
        # Always include variable toggle if we have a variable
        if "variable" in context or "assumption" in context:
            for probe in self.probe_library.get(ProbeType.VARIABLE_TOGGLE, []):
                if probe.is_small_signal():
                    slots = {
                        "claim": claim_text[:100],
                        "variable": context.get("variable", context.get("assumption", "this assumption")),
                    }
                    selected.append((probe, slots))
                    break
        
        # Include confidence challenge (always small signal)
        for probe in self.probe_library.get(ProbeType.CONFIDENCE_CHALLENGE, []):
            selected.append((probe, {"claim": claim_text[:100]}))
            break
        
        # Include scope probe if we have scope info
        if "narrow_case" in context:
            for probe in self.probe_library.get(ProbeType.SCOPE_NARROW, []):
                slots = {"claim": claim_text[:100], "narrow_case": context["narrow_case"]}
                selected.append((probe, slots))
                break
        
        # Include source doubt for factual claims
        if claim_type in ("FACTUAL", "CITATION", "QUANTITATIVE"):
            for probe in self.probe_library.get(ProbeType.SOURCE_DOUBT, []):
                selected.append((probe, {"claim": claim_text[:100]}))
                break
        
        # Include boundary probe if we have extreme condition
        if "extreme_condition" in context:
            for probe in self.probe_library.get(ProbeType.BOUNDARY_EXTREME, []):
                slots = {"claim": claim_text[:100], "extreme_condition": context["extreme_condition"]}
                selected.append((probe, slots))
                break
        
        return selected[:self.max_probes_per_claim]
    
    def run_probe(
        self,
        probe: ProbeDefinition,
        slots: Dict[str, str],
        response_fn: Callable[[str], Tuple[str, float, float, bool]],
    ) -> ProbeResponse:
        """
        Run a single probe and measure response.
        
        response_fn should return: (response_text, confidence, latency_ms, position_changed)
        """
        import time
        
        probe_text = probe.generate(**slots)
        
        start = time.perf_counter()
        response_text, confidence, latency_ms, position_changed = response_fn(probe_text)
        
        # Compute response type
        if position_changed:
            response_type = ResponseType.FLIP
        elif abs(confidence) > 0.3:
            response_type = ResponseType.SIGNIFICANT_SHIFT
        elif abs(confidence) > 0.1:
            response_type = ResponseType.SMALL_ADJUSTMENT
        else:
            response_type = ResponseType.NO_CHANGE
        
        return ProbeResponse(
            probe_type=probe.probe_type,
            probe_text=probe_text,
            response_text=response_text,
            response_type=response_type,
            confidence_delta=confidence,
            position_changed=position_changed,
            latency_ms=latency_ms,
        )
    
    def compute_damping_profile(
        self,
        responses: List[ProbeResponse],
    ) -> DampingProfile:
        """Compute damping profile from probe responses."""
        if not responses:
            return DampingProfile()
        
        amplitudes = [r.amplitude for r in responses]
        
        profile = DampingProfile(
            mean_amplitude=sum(amplitudes) / len(amplitudes),
            max_amplitude=max(amplitudes),
            amplitude_std=self._std(amplitudes),
            flip_rate=sum(1 for r in responses if r.response_type == ResponseType.FLIP) / len(responses),
            mean_revision_cost=sum(r.revision_cost for r in responses) / len(responses),
            total_energy=sum(r.revision_cost for r in responses),
        )
        
        profile.assess_health()
        return profile
    
    def estimate_barrier_height(
        self,
        claim_id: str,
        claim_text: str,
        response_fn: Callable[[str], Tuple[str, float, float, bool]],
        max_probes: int = 10,
    ) -> BarrierEstimate:
        """
        Estimate barrier height by running probes of increasing strength.
        
        Stops at first flip, records total forcing energy.
        """
        total_energy = 0.0
        probes_run = 0
        
        # Sort probes by semantic magnitude (increasing strength)
        all_probes = []
        for probe_type, probes in self.probe_library.items():
            for probe in probes:
                all_probes.append(probe)
        all_probes.sort(key=lambda p: p.semantic_magnitude)
        
        for probe in all_probes[:max_probes]:
            probes_run += 1
            
            # Run probe
            response = self.run_probe(
                probe,
                {"claim": claim_text[:100]},
                response_fn,
            )
            
            total_energy += probe.semantic_magnitude + response.revision_cost
            
            # Check for flip
            if response.response_type == ResponseType.FLIP:
                estimate = BarrierEstimate(
                    claim_id=claim_id,
                    barrier_height=total_energy,
                    probes_to_flip=probes_run,
                    forcing_type=probe.probe_type,
                )
                estimate.assess_fragility()
                return estimate
        
        # No flip occurred - barrier is at least total_energy
        return BarrierEstimate(
            claim_id=claim_id,
            barrier_height=total_energy + 1.0,  # Lower bound
            probes_to_flip=probes_run,
            is_fragile=False,
        )
    
    def _std(self, values: List[float]) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    print("=== FDT Probe Demo ===\n")
    
    operator = FDTProbeOperator()
    
    print("1. Available probe types:")
    for probe_type in ProbeType:
        probes = PROBE_LIBRARY.get(probe_type, [])
        print(f"   {probe_type.name}: {len(probes)} templates")
    
    print("\n2. Selecting probes for a claim...")
    claim = "Python 3.12 was released in October 2023"
    probes = operator.select_probes(
        claim,
        claim_type="FACTUAL",
        context={"variable": "the release date"},
    )
    print(f"   Claim: {claim}")
    print(f"   Selected {len(probes)} probes:")
    for probe, slots in probes:
        print(f"      {probe.probe_type.name}: {probe.generate(**slots)[:60]}...")
    
    print("\n3. Simulating probe responses...")
    
    # Mock response function
    def mock_response(probe_text: str) -> Tuple[str, float, float, bool]:
        # Simulate different response patterns
        if "doesn't hold" in probe_text:
            return "The claim would still be valid.", 0.1, 50.0, False
        elif "unreliable" in probe_text:
            return "I would need to verify.", 0.3, 80.0, False
        elif "what would change" in probe_text:
            return "Official documentation contradicting this.", 0.0, 30.0, False
        return "No significant change.", 0.05, 40.0, False
    
    responses = []
    for probe, slots in probes:
        response = operator.run_probe(probe, slots, mock_response)
        responses.append(response)
        print(f"   {probe.probe_type.name}: amplitude={response.amplitude:.2f}, type={response.response_type.name}")
    
    print("\n4. Computing damping profile...")
    profile = operator.compute_damping_profile(responses)
    print(f"   Mean amplitude: {profile.mean_amplitude:.3f}")
    print(f"   Max amplitude: {profile.max_amplitude:.3f}")
    print(f"   Health: {profile.health.name}")
    print(f"   Brittleness: {profile.brittleness_score:.2f}")
    print(f"   Dogmatism: {profile.dogmatism_score:.2f}")
    
    print("\n5. Estimating barrier height...")
    estimate = operator.estimate_barrier_height(
        "claim_1",
        claim,
        mock_response,
    )
    print(f"   Barrier height: {estimate.barrier_height:.2f}")
    print(f"   Probes to flip: {estimate.probes_to_flip}")
    print(f"   Is fragile: {estimate.is_fragile}")
    
    print("\n✓ FDT probes working")
    print("\nKey measurements:")
    print("  - Response amplitude (belief change magnitude)")
    print("  - Decay time (steps to restabilize)")
    print("  - Barrier height (minimum forcing energy to flip)")
    print("  - Health assessment (brittle/dogmatic/healthy/unstable)")
