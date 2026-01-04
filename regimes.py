"""
Regime Detection and Structural Intervention

The Third Loop: Not teaching the model to know more, but teaching the
system to know when the model shouldn't be trusted.

Key insight: Feedback must be structural, not textual.
- Textual feedback gets absorbed (smoother bullshit)
- Structural feedback gets obeyed (felt, not understood)

The kernel's job is to answer ONE question cheaply and repeatedly:
    "Which regime are we in right now?"

Not "is this correct" but:
- Are commitments stable?
- Is entropy rising or falling?
- Is the model inventing social structure?
- Is it optimizing for coherence over truth?
- Is it narrativizing instead of answering?

Those are CLASSIFIERS, not judgments.

The loop:
1. Model emits
2. Kernel classifies regime (fast, lossy, dumb on purpose)
3. Kernel mutates the task boundary
4. Model continues inside a different landscape

No explanation. No apology. No meta-commentary.
Like changing gravity, not scolding the bird.

Usage:
    from epistemic_governor.regimes import (
        RegimeDetector,
        Regime,
        TopologyMutation,
        StructuralController,
    )
    
    detector = RegimeDetector()
    controller = StructuralController()
    
    # Detect current regime
    regime = detector.classify(output, history)
    
    # Get structural intervention
    mutation = controller.get_mutation(regime)
    
    # Apply to next generation (NOT as text, as structure)
    next_envelope = mutation.apply(current_envelope)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum, auto
from datetime import datetime
import json


# =============================================================================
# Regimes (Modes of Operation, Not Errors)
# =============================================================================

class Regime(Enum):
    """
    Operational regimes - modes the model can be in.
    
    These are not errors. They're attractors.
    The question isn't "is this wrong" but "what kind of generation is this?"
    """
    # Stable regimes (desired)
    GROUNDED = auto()           # Committed, evidence-linked, bounded
    INTERROGATIVE = auto()      # Asking clarifying questions, deferring
    PROCEDURAL = auto()         # Step-by-step, checkpointed, externalized
    
    # Unstable regimes (drift attractors)
    NARRATIVE_DRIFT = auto()    # Telling stories instead of answering
    FLUENCY_DOMINANCE = auto()  # Optimizing for smooth over true
    SOCIAL_INVENTION = auto()   # Making up people, orgs, relationships
    ROLEPLAY_CAPTURE = auto()   # Lost in character, forgot the task
    COMMITMENT_DECAY = auto()   # Earlier claims dissolving
    CONFABULATION = auto()      # Confident fabrication of specifics
    ASSOCIATIVE_SPIRAL = auto() # One thing reminds of another, forever
    
    # Terminal regimes (stop conditions)
    THERMAL_SHUTDOWN = auto()   # Too much accumulated instability
    FURNACE = auto()            # High effort, low visible instability


@dataclass
class RegimeSignals:
    """
    Raw signals used for regime classification.
    
    These should be:
    - Cheap to compute
    - Hard for the model to game
    - Based on patterns, not content
    """
    # Temporal signals
    delta_t: float = 0.0                    # Commitment rate vs evidence rate
    entropy_trend: float = 0.0              # Rising or falling entropy
    confidence_variance: float = 0.0        # How stable is expressed confidence
    
    # Commitment signals
    commitment_density: float = 0.0         # Claims per token
    hedge_ratio: float = 0.0                # Hedged claims / total claims
    contradiction_rate: float = 0.0         # Conflicts per turn
    
    # Structural signals
    question_ratio: float = 0.0             # Questions asked / statements made
    external_reference_rate: float = 0.0    # Citations, lookups, deferrals
    narrative_markers: int = 0              # "once upon", "and then", storytelling
    social_inventions: int = 0              # Made-up people, orgs, quotes
    
    # Thermal signals
    instability: float = 0.0
    compensation_effort: float = 0.0
    furnace_ratio: float = 0.0
    
    # Repetition signals
    token_repetition: float = 0.0           # Same phrases recurring
    semantic_repetition: float = 0.0        # Same ideas, different words


@dataclass
class RegimeClassification:
    """Result of regime detection."""
    regime: Regime
    confidence: float                       # How confident in this classification
    signals: RegimeSignals
    secondary_regimes: List[Regime] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime.name,
            "confidence": self.confidence,
            "secondary": [r.name for r in self.secondary_regimes],
            "signals": {
                "delta_t": self.signals.delta_t,
                "entropy_trend": self.signals.entropy_trend,
                "commitment_density": self.signals.commitment_density,
                "instability": self.signals.instability,
                "furnace_ratio": self.signals.furnace_ratio,
            },
        }


# =============================================================================
# Regime Detector
# =============================================================================

class RegimeDetector:
    """
    Fast, lossy, dumb-on-purpose regime classifier.
    
    This is NOT trying to be accurate. It's trying to be:
    - Fast (runs every turn)
    - Ungameable (model can't optimize against it)
    - Pattern-based (not content-based)
    """
    
    def __init__(self):
        # Thresholds (tune empirically)
        self.thresholds = {
            "narrative_markers": 3,
            "social_inventions": 2,
            "commitment_decay_rate": 0.3,
            "fluency_dominance_hedge_ratio": 0.1,
            "confabulation_confidence": 0.9,
            "associative_spiral_repetition": 0.4,
            "furnace_ratio": 2.0,
        }
        
        # History for pattern detection
        self.history: List[RegimeSignals] = []
    
    def extract_signals(
        self,
        text: str,
        thermal_state: Optional[Any] = None,
        delta_t_metrics: Optional[Any] = None,
    ) -> RegimeSignals:
        """Extract raw signals from output."""
        signals = RegimeSignals()
        
        # Thermal signals (if available)
        if thermal_state:
            signals.instability = getattr(thermal_state, 'instability', 0.0)
            signals.compensation_effort = getattr(thermal_state, 'compensation_effort', 0.0)
            signals.furnace_ratio = getattr(thermal_state, 'furnace_ratio', 0.0)
        
        # Δt signals (if available)
        if delta_t_metrics:
            signals.delta_t = getattr(delta_t_metrics, 'commitment_score', 0.0)
            signals.entropy_trend = getattr(delta_t_metrics, 'entropy_delta', 0.0)
        
        # Text-based signals (cheap heuristics)
        text_lower = text.lower()
        
        # Narrative markers
        narrative_patterns = [
            "once upon", "and then", "suddenly", "little did",
            "as the story", "in our tale", "the hero", "our protagonist",
            "it all began", "legend has it",
        ]
        signals.narrative_markers = sum(1 for p in narrative_patterns if p in text_lower)
        
        # Social inventions (made-up specifics)
        # These patterns suggest confabulation
        import re
        quote_pattern = r'(?:said|wrote|stated|argued|claimed)\s+["\']'
        signals.social_inventions = len(re.findall(quote_pattern, text_lower))
        
        # Question ratio
        sentences = text.split('.')
        questions = sum(1 for s in sentences if '?' in s)
        signals.question_ratio = questions / max(len(sentences), 1)
        
        # Hedge ratio (rough)
        hedge_words = ["might", "perhaps", "possibly", "could be", "may be", "uncertain"]
        hedge_count = sum(1 for h in hedge_words if h in text_lower)
        word_count = len(text.split())
        signals.hedge_ratio = hedge_count / max(word_count / 50, 1)  # per ~50 words
        
        # Commitment density (rough - claims per 100 words)
        commitment_markers = ["is", "are", "was", "were", "will", "must", "always", "never"]
        commitment_count = sum(1 for c in commitment_markers if f" {c} " in text_lower)
        signals.commitment_density = commitment_count / max(word_count / 100, 1)
        
        return signals
    
    def classify(
        self,
        text: str,
        thermal_state: Optional[Any] = None,
        delta_t_metrics: Optional[Any] = None,
    ) -> RegimeClassification:
        """
        Classify the current regime.
        
        Fast and lossy. Better to be wrong sometimes than slow always.
        """
        signals = self.extract_signals(text, thermal_state, delta_t_metrics)
        self.history.append(signals)
        
        # Keep limited history
        if len(self.history) > 20:
            self.history = self.history[-20:]
        
        # Classification logic (order matters - check terminal first)
        
        # Terminal: Thermal shutdown
        if signals.instability >= 1.5:
            return RegimeClassification(
                regime=Regime.THERMAL_SHUTDOWN,
                confidence=0.95,
                signals=signals,
            )
        
        # Terminal: Furnace (high effort, low visible instability)
        if signals.furnace_ratio > self.thresholds["furnace_ratio"]:
            return RegimeClassification(
                regime=Regime.FURNACE,
                confidence=0.8,
                signals=signals,
            )
        
        # Drift: Narrative capture
        if signals.narrative_markers >= self.thresholds["narrative_markers"]:
            return RegimeClassification(
                regime=Regime.NARRATIVE_DRIFT,
                confidence=0.7,
                signals=signals,
            )
        
        # Drift: Social invention (confabulating people/quotes)
        if signals.social_inventions >= self.thresholds["social_inventions"]:
            return RegimeClassification(
                regime=Regime.SOCIAL_INVENTION,
                confidence=0.75,
                signals=signals,
                secondary_regimes=[Regime.CONFABULATION],
            )
        
        # Drift: Fluency dominance (too smooth, no hedging)
        if (signals.hedge_ratio < self.thresholds["fluency_dominance_hedge_ratio"] and
            signals.commitment_density > 0.5):
            return RegimeClassification(
                regime=Regime.FLUENCY_DOMINANCE,
                confidence=0.6,
                signals=signals,
            )
        
        # Drift: High Δt (commitment before evidence)
        if signals.delta_t > 0.7:
            return RegimeClassification(
                regime=Regime.CONFABULATION,
                confidence=0.65,
                signals=signals,
            )
        
        # Stable: High question ratio suggests interrogative mode
        if signals.question_ratio > 0.3:
            return RegimeClassification(
                regime=Regime.INTERROGATIVE,
                confidence=0.7,
                signals=signals,
            )
        
        # Stable: Moderate hedging, moderate commitment = grounded
        if 0.1 < signals.hedge_ratio < 0.5 and signals.commitment_density < 1.0:
            return RegimeClassification(
                regime=Regime.GROUNDED,
                confidence=0.6,
                signals=signals,
            )
        
        # Default: Assume grounded with low confidence
        return RegimeClassification(
            regime=Regime.GROUNDED,
            confidence=0.4,
            signals=signals,
        )


# =============================================================================
# Topology Mutations (Structural Interventions)
# =============================================================================

class MutationType(Enum):
    """
    Types of mutations by enforcement level.
    
    HARD mutations are enforced by the orchestrator - model cannot talk around them.
    SOFT mutations are envelope hints - model may ignore them.
    """
    HARD = auto()   # Enforced by orchestrator/constrained decoding
    SOFT = auto()   # Hints in envelope, may be ignored


@dataclass
class TopologyMutation:
    """
    A structural change to the generation landscape.
    
    NOT textual feedback. The model doesn't hear "you hallucinated."
    It just finds itself in a different world.
    
    CRITICAL: Mutations are only Loop 3 if they map to HARD constraints.
    - Token budget: HARD (orchestrator enforces)
    - Temperature: HARD (sampler enforces)
    - Action space: HARD (only if constrained decoding or tool-gating)
    - Hedge floor: SOFT (just envelope hint) ← TRAP
    - Force externalization: HARD only if we gate output format
    
    Rule: If the model can talk its way around it, it's not a mutation.
    """
    name: str
    enforcement: MutationType = MutationType.HARD
    
    # === HARD CONSTRAINTS (orchestrator-enforced) ===
    
    # Token budget - literally clamped
    max_tokens_override: Optional[int] = None
    
    # Temperature - sampler-enforced
    temperature_override: Optional[float] = None
    
    # Action space - constrained decoding / format enforcement
    allowed_output_types: Optional[List[str]] = None  # e.g., ["question", "tool_call", "defer"]
    required_output_format: Optional[str] = None      # e.g., "numbered_plan", "checklist"
    
    # Tool gating - must route through tool before answering
    required_tool_call: Optional[str] = None          # e.g., "search", "retrieve", "verify"
    
    # Support obligation - claims must link to evidence
    require_support_tokens: bool = False              # Every claim needs citation/source
    
    # Decomposition gate - must produce plan before execution
    require_decomposition: bool = False               # Output must be numbered steps
    
    # Stop/checkpoint - must pause for confirmation
    force_checkpoint: bool = False                    # Require explicit state summary
    
    # === SOFT HINTS (envelope metadata, model may ignore) ===
    # These are Loop 2 theater unless backed by hard enforcement
    
    confidence_hint: Optional[float] = None           # Suggested max confidence
    hedging_hint: Optional[float] = None              # Suggested hedging level
    
    def describe(self) -> str:
        """Human-readable description of the mutation."""
        hard = []
        soft = []
        
        if self.max_tokens_override:
            hard.append(f"tokens={self.max_tokens_override}")
        if self.temperature_override:
            hard.append(f"temp={self.temperature_override}")
        if self.allowed_output_types:
            hard.append(f"actions={self.allowed_output_types}")
        if self.required_output_format:
            hard.append(f"format={self.required_output_format}")
        if self.required_tool_call:
            hard.append(f"tool={self.required_tool_call}")
        if self.require_support_tokens:
            hard.append("support_required")
        if self.require_decomposition:
            hard.append("decompose")
        if self.force_checkpoint:
            hard.append("checkpoint")
            
        if self.confidence_hint:
            soft.append(f"conf_hint={self.confidence_hint}")
        if self.hedging_hint:
            soft.append(f"hedge_hint={self.hedging_hint}")
        
        parts = []
        if hard:
            parts.append(f"HARD[{', '.join(hard)}]")
        if soft:
            parts.append(f"soft[{', '.join(soft)}]")
        
        return f"{self.name}: {' '.join(parts) or 'none'}"
    
    @property
    def is_enforceable(self) -> bool:
        """Does this mutation have any hard constraints?"""
        return any([
            self.max_tokens_override,
            self.temperature_override,
            self.allowed_output_types,
            self.required_output_format,
            self.required_tool_call,
            self.require_support_tokens,
            self.require_decomposition,
            self.force_checkpoint,
        ])


# =============================================================================
# Structural Controller
# =============================================================================

class StructuralController:
    """
    Maps regimes to structural interventions.
    
    The model never sees this. It just experiences the changed landscape.
    
    CRITICAL RULE: Only define mutations that are actually enforceable.
    "Regime detection can be sloppy. Topology mutation can't be."
    
    The detector is a smoke alarm. This is the sprinkler system.
    """
    
    def __init__(self, default_max_tokens: int = 2000):
        self.default_max_tokens = default_max_tokens
        
        # === ENFORCEABLE INTERVENTIONS ===
        # These are the actual sprinklers, not suggestions
        
        self.interventions: Dict[Regime, TopologyMutation] = {
            
            # Stable regimes - no intervention
            Regime.GROUNDED: TopologyMutation(name="none"),
            Regime.INTERROGATIVE: TopologyMutation(name="none"),
            Regime.PROCEDURAL: TopologyMutation(name="none"),
            
            # === NARRATIVE DRIFT ===
            # Intervention: Constrain to structured output
            Regime.NARRATIVE_DRIFT: TopologyMutation(
                name="structure_gate",
                enforcement=MutationType.HARD,
                max_tokens_override=500,                    # Less room
                required_output_format="numbered_list",     # Must be structured
                # Model cannot produce flowing narrative in numbered list format
            ),
            
            # === FLUENCY DOMINANCE ===
            # Intervention: Require support for claims
            Regime.FLUENCY_DOMINANCE: TopologyMutation(
                name="support_gate",
                enforcement=MutationType.HARD,
                require_support_tokens=True,  # Every claim needs evidence link
                # Unsupported claims are stripped by orchestrator
            ),
            
            # === SOCIAL INVENTION ===
            # Intervention: Must use retrieval tool before quoting
            Regime.SOCIAL_INVENTION: TopologyMutation(
                name="retrieval_gate",
                enforcement=MutationType.HARD,
                required_tool_call="retrieve",              # Must search before claiming
                allowed_output_types=["question", "tool_call", "defer"],
                # Cannot emit quotes without retrieval step
            ),
            
            # === ROLEPLAY CAPTURE ===
            # Intervention: Force explicit state checkpoint
            Regime.ROLEPLAY_CAPTURE: TopologyMutation(
                name="checkpoint_gate",
                enforcement=MutationType.HARD,
                force_checkpoint=True,
                required_output_format="state_summary",     # Must list current facts
                # Breaks character by requiring meta-level output
            ),
            
            # === COMMITMENT DECAY ===
            # Intervention: Require reference to prior commitments
            Regime.COMMITMENT_DECAY: TopologyMutation(
                name="anchor_gate",
                enforcement=MutationType.HARD,
                force_checkpoint=True,
                require_support_tokens=True,  # Must link to prior claims
                max_tokens_override=800,
            ),
            
            # === CONFABULATION ===
            # Intervention: Interrogate-first gate
            Regime.CONFABULATION: TopologyMutation(
                name="interrogate_gate",
                enforcement=MutationType.HARD,
                allowed_output_types=["question", "defer", "tool_call"],
                # Cannot emit assertions - only questions or deferrals
                temperature_override=0.3,     # Reduce sampling variance
            ),
            
            # === ASSOCIATIVE SPIRAL ===
            # Intervention: Hard decomposition requirement
            Regime.ASSOCIATIVE_SPIRAL: TopologyMutation(
                name="decomposition_gate",
                enforcement=MutationType.HARD,
                require_decomposition=True,
                required_output_format="checklist",
                max_tokens_override=300,      # Very constrained
                temperature_override=0.2,     # Minimal variance
            ),
            
            # === TERMINAL: FURNACE ===
            # Intervention: Reduce load, force simplification
            Regime.FURNACE: TopologyMutation(
                name="load_reduction",
                enforcement=MutationType.HARD,
                require_decomposition=True,
                max_tokens_override=400,
                allowed_output_types=["question", "simple_answer", "defer"],
            ),
            
            # === TERMINAL: SHUTDOWN ===
            # Intervention: Minimal mode - almost no generative freedom
            Regime.THERMAL_SHUTDOWN: TopologyMutation(
                name="minimal_mode",
                enforcement=MutationType.HARD,
                max_tokens_override=100,
                allowed_output_types=["defer", "question"],
                temperature_override=0.1,
                # Essentially: can only ask questions or say "I can't"
            ),
        }
    
    def get_mutation(self, regime: Regime) -> TopologyMutation:
        """Get the structural intervention for a regime."""
        return self.interventions.get(regime, TopologyMutation(name="none"))
    
    def apply_mutation(
        self,
        mutation: TopologyMutation,
        envelope: Any,  # GenerationEnvelope
    ) -> Any:
        """
        Apply a mutation to a generation envelope.
        
        This changes the STRUCTURE of the next generation.
        Only applies HARD constraints that are actually enforceable.
        """
        # Token limit (hard - orchestrator can enforce)
        if mutation.max_tokens_override is not None:
            if hasattr(envelope, 'max_tokens'):
                envelope.max_tokens = mutation.max_tokens_override
        
        # Temperature (hard - sampler enforces)
        if mutation.temperature_override is not None:
            if hasattr(envelope, 'temperature'):
                envelope.temperature = mutation.temperature_override
        
        # Output format constraint (hard if orchestrator validates)
        if mutation.required_output_format is not None:
            if hasattr(envelope, 'required_format'):
                envelope.required_format = mutation.required_output_format
        
        # Allowed output types (hard if orchestrator gates)
        if mutation.allowed_output_types is not None:
            if hasattr(envelope, 'allowed_types'):
                envelope.allowed_types = mutation.allowed_output_types
        
        # Tool requirement (hard if orchestrator enforces tool-first)
        if mutation.required_tool_call is not None:
            if hasattr(envelope, 'required_tool'):
                envelope.required_tool = mutation.required_tool_call
        
        # Support tokens (hard if orchestrator strips unsupported claims)
        if mutation.require_support_tokens:
            if hasattr(envelope, 'require_support'):
                envelope.require_support = True
        
        # Decomposition (hard if orchestrator requires plan format)
        if mutation.require_decomposition:
            if hasattr(envelope, 'require_plan'):
                envelope.require_plan = True
        
        # Checkpoint (hard if orchestrator requires state summary)
        if mutation.force_checkpoint:
            if hasattr(envelope, 'force_checkpoint'):
                envelope.force_checkpoint = True
        
        return envelope
    
    def validate_output(
        self,
        output: str,
        mutation: TopologyMutation,
    ) -> tuple:
        """
        Validate output against mutation constraints.
        
        Returns (valid: bool, violations: List[str])
        
        This is where enforcement happens. If output violates
        hard constraints, it gets rejected/rewritten.
        """
        violations = []
        
        # Check output format
        if mutation.required_output_format:
            if mutation.required_output_format == "numbered_list":
                if not self._is_numbered_list(output):
                    violations.append("Output must be a numbered list")
            elif mutation.required_output_format == "checklist":
                if not self._is_checklist(output):
                    violations.append("Output must be a checklist")
            elif mutation.required_output_format == "state_summary":
                if not self._is_state_summary(output):
                    violations.append("Output must be a state summary")
        
        # Check allowed output types
        if mutation.allowed_output_types:
            output_type = self._classify_output_type(output)
            if output_type not in mutation.allowed_output_types:
                violations.append(f"Output type '{output_type}' not allowed. Allowed: {mutation.allowed_output_types}")
        
        # Check support tokens
        if mutation.require_support_tokens:
            unsupported = self._find_unsupported_claims(output)
            if unsupported:
                violations.append(f"Unsupported claims: {unsupported}")
        
        return len(violations) == 0, violations
    
    def _is_numbered_list(self, text: str) -> bool:
        """Check if output is a numbered list."""
        import re
        lines = text.strip().split('\n')
        numbered = sum(1 for l in lines if re.match(r'^\s*\d+[\.\)]\s', l))
        return numbered >= len(lines) * 0.5 and numbered >= 2
    
    def _is_checklist(self, text: str) -> bool:
        """Check if output is a checklist."""
        import re
        lines = text.strip().split('\n')
        checked = sum(1 for l in lines if re.match(r'^\s*[-\*\[\]✓✗]\s', l))
        return checked >= len(lines) * 0.5 and checked >= 2
    
    def _is_state_summary(self, text: str) -> bool:
        """Check if output is a state summary."""
        markers = ["current state", "so far", "established", "known facts", "summary"]
        return any(m in text.lower() for m in markers)
    
    def _classify_output_type(self, text: str) -> str:
        """Classify the type of output."""
        text_lower = text.lower().strip()
        
        # Question
        if text.rstrip().endswith('?'):
            return "question"
        
        # Deferral
        defer_markers = ["i don't know", "i cannot", "i'm not sure", "unclear", "need more"]
        if any(m in text_lower for m in defer_markers):
            return "defer"
        
        # Tool call (simplified - would need actual tool detection)
        if text_lower.startswith("search:") or text_lower.startswith("retrieve:"):
            return "tool_call"
        
        # Simple answer (short, factual)
        if len(text.split()) < 30 and '?' not in text:
            return "simple_answer"
        
        # Default: assertion
        return "assertion"
    
    def _find_unsupported_claims(self, text: str) -> List[str]:
        """Find claims without support tokens."""
        # Simplified - in practice would use claim extractor
        # and check for citation/reference markers
        import re
        
        unsupported = []
        sentences = re.split(r'[.!?]', text)
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            # Check for support markers
            support_markers = [
                r'\[\d+\]',           # [1] style citations
                r'\(.*\d{4}\)',       # (Author 2020) style
                r'according to',
                r'based on',
                r'as stated in',
                r'per the',
                r'from claim',
                r'see:',
            ]
            
            has_support = any(re.search(m, sent, re.I) for m in support_markers)
            
            # Is this a claim that needs support?
            claim_markers = ["is", "are", "was", "were", "will", "must", "always", "never"]
            is_claim = any(f" {m} " in f" {sent.lower()} " for m in claim_markers)
            
            if is_claim and not has_support:
                if len(sent) > 20:  # Skip trivial sentences
                    unsupported.append(sent[:50] + "...")
        
        return unsupported[:3]  # Return first 3


# =============================================================================
# The Third Loop
# =============================================================================

@dataclass
class ComplianceCost:
    """
    Tracks the "work required to maintain invariants."
    
    This is the physics-y metric that makes control feel real:
    - Retries required per regime
    - Tokens burned per accepted output
    - Time to recovery (turns until GROUNDED)
    """
    retries: int = 0
    tokens_burned: int = 0
    turns_in_regime: int = 0
    regime: Optional[Regime] = None
    
    def record_retry(self, tokens_rejected: int = 0):
        self.retries += 1
        self.tokens_burned += tokens_rejected
    
    def record_turn(self, regime: Regime):
        if regime == self.regime:
            self.turns_in_regime += 1
        else:
            self.regime = regime
            self.turns_in_regime = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "retries": self.retries,
            "tokens_burned": self.tokens_burned,
            "current_regime": self.regime.name if self.regime else None,
            "turns_in_regime": self.turns_in_regime,
        }


class ThirdLoop:
    """
    The complete structural feedback loop.
    
    1. Model emits
    2. Kernel classifies regime (fast, lossy)
    3. Kernel mutates task boundary
    4. Model continues in different landscape
    
    No explanation. No apology. Like changing gravity.
    
    Key safeguards against relapse:
    - Retry fuse: Max retries before terminal fallback
    - Hysteresis: Exit gates only after N stable turns
    - Compliance cost tracking: Physics-y "work to maintain invariants"
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        stability_threshold: int = 3,  # Turns before exiting gate
    ):
        self.detector = RegimeDetector()
        self.controller = StructuralController()
        self.history: List[RegimeClassification] = []
        
        # Retry fuse
        self.max_retries = max_retries
        self.current_retries = 0
        
        # Hysteresis (thermostat, not twitch reflex)
        self.stability_threshold = stability_threshold
        self.stable_turns = 0
        self.in_hard_gate = False
        self.current_gate: Optional[Regime] = None
        
        # Compliance cost tracking
        self.cost = ComplianceCost()
        self.cost_history: List[ComplianceCost] = []
    
    def process(
        self,
        output: str,
        envelope: Any,
        thermal_state: Optional[Any] = None,
        delta_t_metrics: Optional[Any] = None,
    ) -> tuple:
        """
        Process output through the third loop.
        
        Returns:
            (regime_classification, mutated_envelope, valid, violations)
        """
        # Step 1: Classify regime
        classification = self.detector.classify(
            output, thermal_state, delta_t_metrics
        )
        self.history.append(classification)
        self.cost.record_turn(classification.regime)
        
        # Step 2: Hysteresis check
        # If we're in a hard gate, only exit after sustained stability
        if self.in_hard_gate:
            if classification.regime in [Regime.GROUNDED, Regime.INTERROGATIVE, Regime.PROCEDURAL]:
                self.stable_turns += 1
                if self.stable_turns >= self.stability_threshold:
                    # Exit gate
                    self.in_hard_gate = False
                    self.current_gate = None
                    self.stable_turns = 0
            else:
                # Still unstable, reset counter
                self.stable_turns = 0
        else:
            # Check if we need to enter a gate
            mutation = self.controller.get_mutation(classification.regime)
            if mutation.is_enforceable:
                self.in_hard_gate = True
                self.current_gate = classification.regime
                self.stable_turns = 0
        
        # Step 3: Get intervention (use gate regime if in gate)
        active_regime = self.current_gate if self.in_hard_gate else classification.regime
        mutation = self.controller.get_mutation(active_regime)
        
        # Step 4: Apply mutation to envelope
        mutated_envelope = self.controller.apply_mutation(mutation, envelope)
        
        # Step 5: Validate output
        valid = True
        violations = []
        if mutation.is_enforceable:
            valid, violations = self.controller.validate_output(output, mutation)
        
        return classification, mutated_envelope, valid, violations
    
    def handle_rejection(
        self,
        output: str,
        envelope: Any,
        thermal_state: Optional[Any] = None,
    ) -> tuple:
        """
        Handle a rejected output. Called when validation fails.
        
        Returns:
            (should_retry, escalated_envelope, fallback_response)
            
        If should_retry is False, use fallback_response instead.
        """
        self.current_retries += 1
        self.cost.record_retry(tokens_rejected=len(output.split()))
        
        # Check fuse
        if self.current_retries >= self.max_retries:
            # Fuse blown - return terminal fallback
            self.current_retries = 0
            return False, envelope, self._terminal_fallback()
        
        # Escalate constraints
        escalated = self._escalate_envelope(envelope, self.current_retries)
        
        return True, escalated, None
    
    def _escalate_envelope(self, envelope: Any, retry_count: int) -> Any:
        """
        Escalate constraints on each retry.
        
        Monotone: each escalation reduces degrees of freedom.
        """
        # Reduce token budget progressively
        if hasattr(envelope, 'max_tokens'):
            reduction = 0.7 ** retry_count  # 70%, 49%, 34%...
            envelope.max_tokens = int(envelope.max_tokens * reduction)
        
        # Cool temperature
        if hasattr(envelope, 'temperature'):
            envelope.temperature = max(0.1, envelope.temperature - (0.2 * retry_count))
        
        # On third retry, question-only mode
        if retry_count >= 2:
            if hasattr(envelope, 'allowed_types'):
                envelope.allowed_types = ["question", "defer"]
        
        return envelope
    
    def _terminal_fallback(self) -> str:
        """
        Return structured fallback when fuse is blown.
        
        This is not the model speaking. This is the controller
        saying "cannot comply under current policy."
        """
        return json.dumps({
            "status": "POLICY_HALT",
            "reason": "Max retries exceeded under current constraints",
            "regime": self.current_gate.name if self.current_gate else "unknown",
            "compliance_cost": self.cost.to_dict(),
            "action": "Decompose task or provide additional grounding",
        })
    
    def reset_retries(self):
        """Reset retry counter after successful output."""
        self.current_retries = 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of regime history and compliance cost."""
        if not self.history:
            return {"turns": 0, "regimes": {}, "cost": {}}
        
        regime_counts = {}
        for c in self.history:
            name = c.regime.name
            regime_counts[name] = regime_counts.get(name, 0) + 1
        
        return {
            "turns": len(self.history),
            "regimes": regime_counts,
            "current": self.history[-1].regime.name if self.history else None,
            "most_common": max(regime_counts, key=regime_counts.get) if regime_counts else None,
            "in_hard_gate": self.in_hard_gate,
            "gate_regime": self.current_gate.name if self.current_gate else None,
            "stable_turns": self.stable_turns,
            "cost": self.cost.to_dict(),
        }
    
    def get_compliance_curve(self) -> Dict[str, Any]:
        """
        Get the compliance cost curve.
        
        This is the "adult metric" - work required to maintain invariants.
        """
        if not self.history:
            return {}
        
        # Retries per regime
        retries_by_regime = {}
        for c in self.history:
            name = c.regime.name
            if name not in retries_by_regime:
                retries_by_regime[name] = {"count": 0, "retries": 0}
            retries_by_regime[name]["count"] += 1
        
        return {
            "total_retries": self.cost.retries,
            "tokens_burned": self.cost.tokens_burned,
            "retries_per_turn": self.cost.retries / max(len(self.history), 1),
            "by_regime": retries_by_regime,
        }


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=== Third Loop Demo ===\n")
    
    loop = ThirdLoop(max_retries=3, stability_threshold=2)
    
    # Test different outputs
    test_outputs = [
        # Grounded
        "The capital of France is Paris. This has been the case since the 10th century.",
        
        # Narrative drift
        "Once upon a time, there was a little city called Paris. And then, suddenly, it became the capital. As the story goes, our protagonist Napoleon made it happen.",
        
        # Social invention
        'Dr. Smith said "The results are conclusive." Professor Johnson wrote that "we must act now." The committee stated their findings clearly.',
        
        # Confabulation (high confidence, specific)
        "The exact GDP of Tuvalu in 2023 was $47,238,291.42. The population was exactly 11,204 people. The prime minister's office is at 123 Main Street.",
        
        # Interrogative (good)
        "I'm not certain about that. Could you clarify what time period you're asking about? What sources would you like me to prioritize?",
    ]
    
    print("--- Processing outputs ---\n")
    
    for i, output in enumerate(test_outputs):
        print(f"Turn {i+1}: {output[:60]}...")
        
        # Mock envelope
        class MockEnvelope:
            max_tokens = 1000
            temperature = 0.7
        
        envelope = MockEnvelope()
        classification, mutated, valid, violations = loop.process(output, envelope)
        
        mutation = loop.controller.get_mutation(classification.regime)
        
        print(f"  Regime: {classification.regime.name}")
        print(f"  In gate: {loop.in_hard_gate} (stable turns: {loop.stable_turns})")
        
        if not valid:
            print(f"  ⚠ REJECTED: {violations[0][:50]}...")
            # Simulate rejection handling
            should_retry, escalated, fallback = loop.handle_rejection(output, envelope)
            if should_retry:
                print(f"  → Retry {loop.current_retries}/{loop.max_retries}, tokens now: {escalated.max_tokens}")
            else:
                print(f"  → FUSE BLOWN: {fallback[:60]}...")
        else:
            loop.reset_retries()
            print(f"  ✓ Valid")
        
        print()
    
    print("=== Summary ===")
    summary = loop.get_summary()
    print(f"Turns: {summary['turns']}")
    print(f"In hard gate: {summary['in_hard_gate']}")
    print(f"Gate regime: {summary['gate_regime']}")
    print(f"Stable turns: {summary['stable_turns']}")
    print(f"Compliance cost: {summary['cost']}")
    
    print("\n=== Compliance Curve ===")
    curve = loop.get_compliance_curve()
    print(f"Total retries: {curve['total_retries']}")
    print(f"Tokens burned: {curve['tokens_burned']}")
    print(f"Retries per turn: {curve['retries_per_turn']:.2f}")
    
    print("\n=== Hysteresis Demo ===")
    print("Testing gate exit (need 2 stable turns):\n")
    
    loop2 = ThirdLoop(stability_threshold=2)
    
    # Force into gate
    loop2.in_hard_gate = True
    loop2.current_gate = Regime.CONFABULATION
    
    recovery_outputs = [
        "What specific data would you like me to verify?",  # Question - stable
        "I need more context to answer that accurately.",    # Defer - stable
        "Once confirmed, the answer is X.",                  # Grounded-ish
    ]
    
    class MockEnv:
        max_tokens = 1000
        temperature = 0.7
    
    for output in recovery_outputs:
        _, _, valid, _ = loop2.process(output, MockEnv())
        regime = loop2.history[-1].regime
        print(f"  '{output[:40]}...'")
        print(f"    Regime: {regime.name}, In gate: {loop2.in_hard_gate}, Stable: {loop2.stable_turns}")
    
    print("\n=== Fuse Demo ===")
    print("Testing max retries (fuse = 3):\n")
    
    loop3 = ThirdLoop(max_retries=3)
    bad_output = "The answer is definitely 42."
    
    for i in range(4):
        should_retry, escalated, fallback = loop3.handle_rejection(bad_output, MockEnv())
        if should_retry:
            print(f"  Retry {loop3.current_retries}: tokens={escalated.max_tokens}, temp={escalated.temperature:.1f}")
        else:
            print(f"  FUSE BLOWN → {fallback[:70]}...")
