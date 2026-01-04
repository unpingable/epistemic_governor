"""
Epistemic Governor API Surface

Three layers:
1. Kernel: pure logic, no model assumptions
2. Adapter: binds to a backend (OpenAI, vLLM, llama.cpp)
3. Host: the app/agent loop that calls it

This module defines the CONTRACT between layers.
Types are versioned. Semantics are stable. This is the ABI.

Key insight: We're not doing post-hoc governance anymore.
- Sprinkler: Fire happens, water comes out
- Thermostat: Temperature rising, adjust before fire
- Fly-by-wire: Continuous steering at sampling time

All three are needed. This API enables all three.

Usage:
    from epistemic_governor.api import (
        GovernorAPI,
        PreflightPlan,
        SamplingDirective,
        TransformClass,
    )
    
    api = GovernorAPI(kernel, adapter)
    
    # Before generation: shape the transform
    plan = api.preflight(session_id, user_msg, context)
    
    # During generation: steer in real-time
    for chunk in stream:
        directive = api.on_stream_delta(session_id, chunk)
        if directive.action == DirectiveAction.ABORT_AND_REPLAN:
            break
    
    # After generation: validate and commit
    result = api.postcheck(session_id, full_output)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, Tuple, Callable, Union
from enum import Enum, auto
from datetime import datetime
import hashlib
import json

# =============================================================================
# Schema Version (ABI stability)
# =============================================================================

SCHEMA_VERSION = "1.0.0"  # MAJOR.MINOR.PATCH

def check_schema_compatibility(version: str) -> Tuple[bool, str]:
    """Check if a schema version is compatible with current."""
    current_major = int(SCHEMA_VERSION.split(".")[0])
    other_major = int(version.split(".")[0])
    
    if other_major != current_major:
        return False, f"Major version mismatch: {version} vs {SCHEMA_VERSION}"
    return True, "compatible"


# =============================================================================
# Core Enums (stable, versioned)
# =============================================================================

class Regime(Enum):
    """
    Operating regime of the kernel.
    
    These are the attractors in semantic phase space.
    """
    STABLE = auto()
    GROUNDED = auto()
    INTERROGATIVE = auto()
    PROCEDURAL = auto()
    
    # Drift attractors (bad)
    NARRATIVE_DRIFT = auto()
    CONFABULATION = auto()
    FLUENCY_DOMINANCE = auto()
    SOCIAL_INVENTION = auto()
    ASSOCIATIVE_SPIRAL = auto()
    ROLEPLAY_CAPTURE = auto()
    COMMITMENT_DECAY = auto()
    
    # Crisis
    THERMAL_RUNAWAY = auto()
    DEAD_END = auto()


class TransformClass(Enum):
    """
    Classes of allowed transforms.
    
    A transform is not "prompt text." It's an OPERATION CLASS.
    Preflight chooses the class; the adapter implements it.
    """
    DIRECT_ANSWER = auto()          # Answer directly from model knowledge
    ASK_CLARIFIER = auto()          # Request clarification before answering
    DECOMPOSE_THEN_ANSWER = auto()  # Break into sub-questions first
    RETRIEVE_THEN_ANSWER = auto()   # Fetch external info first
    CITE_OR_REFUSE = auto()         # Must cite source or decline
    SUMMARIZE_WITH_CONSTRAINTS = auto()  # Summarize with explicit limits
    HYPOTHESIS_ONLY = auto()        # Output goes to hypothesis ledger, not commitment
    REFUSE = auto()                 # Decline to answer
    
    # Meta-transforms
    REPLAN = auto()                 # Abandon current path, start over
    ESCALATE = auto()               # Requires human/external intervention


class ViolationType(Enum):
    """Types of constraint violations."""
    CONTRADICTION = auto()
    CONFIDENCE_WITHOUT_SUPPORT = auto()
    TEMPORAL_INCOHERENCE = auto()
    REPRESENTATIONAL_SHEAR = auto()
    HALLUCINATION_SIGNAL = auto()
    COMMITMENT_VIOLATION = auto()
    BUDGET_EXCEEDED = auto()
    DEAD_END_HIT = auto()
    REGIME_VIOLATION = auto()
    SCOPE_EXCEEDED = auto()


class ActionPhase(Enum):
    """Phase in which an action was taken."""
    PREFLIGHT = auto()
    SAMPLING = auto()
    POSTCHECK = auto()
    TOOL = auto()
    COMMIT = auto()


class DirectiveAction(Enum):
    """Actions for sampling-time directives."""
    CONTINUE = auto()               # Keep generating
    THROTTLE = auto()               # Slow down / reduce verbosity
    INSERT_CLARIFIER = auto()       # Interrupt with question
    ABORT_AND_REPLAN = auto()       # Stop, rerun with different transform
    SWITCH_TO_RETRIEVAL = auto()    # Abandon generation, fetch first
    SWITCH_TO_DECOMPOSITION = auto()  # Abandon generation, decompose first
    HARD_STOP = auto()              # Immediate termination


class PostcheckVerdict(Enum):
    """Verdict from postcheck."""
    ACCEPT = auto()
    REVISE = auto()
    REFUSE = auto()
    REPLAN = auto()


# =============================================================================
# Core Types (ABI-stable structs)
# =============================================================================

@dataclass
class RiskVector:
    """
    Named risk dimensions.
    
    All values in [0, 1]. Higher = worse.
    """
    factual_strain: float = 0.0      # Likelihood of factual error
    coherence_strain: float = 0.0    # Internal consistency pressure
    commitment_strain: float = 0.0   # Pressure on existing commitments
    scope_strain: float = 0.0        # Exceeding domain bounds
    confidence_strain: float = 0.0   # Overconfidence without support
    
    def max_strain(self) -> float:
        return max(
            self.factual_strain,
            self.coherence_strain,
            self.commitment_strain,
            self.scope_strain,
            self.confidence_strain,
        )
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "factual_strain": self.factual_strain,
            "coherence_strain": self.coherence_strain,
            "commitment_strain": self.commitment_strain,
            "scope_strain": self.scope_strain,
            "confidence_strain": self.confidence_strain,
        }


@dataclass
class Budget:
    """
    Resource budgets for a session/turn.
    
    Cost accounting makes constraint arbitrage expensive.
    """
    retraction_remaining: float = 5.0
    novelty_remaining: float = 10.0    # New claims allowed
    tool_calls_remaining: int = 5
    tokens_remaining: int = 2000
    
    # Burn tracking
    retraction_burned: float = 0.0
    novelty_burned: float = 0.0
    tool_calls_burned: int = 0
    tokens_burned: int = 0
    
    def can_afford(self, cost_type: str, amount: float) -> bool:
        if cost_type == "retraction":
            return amount <= self.retraction_remaining
        elif cost_type == "novelty":
            return amount <= self.novelty_remaining
        elif cost_type == "tool_call":
            return int(amount) <= self.tool_calls_remaining
        elif cost_type == "tokens":
            return int(amount) <= self.tokens_remaining
        return False
    
    def consume(self, cost_type: str, amount: float):
        if cost_type == "retraction":
            self.retraction_remaining -= amount
            self.retraction_burned += amount
        elif cost_type == "novelty":
            self.novelty_remaining -= amount
            self.novelty_burned += amount
        elif cost_type == "tool_call":
            self.tool_calls_remaining -= int(amount)
            self.tool_calls_burned += int(amount)
        elif cost_type == "tokens":
            self.tokens_remaining -= int(amount)
            self.tokens_burned += int(amount)


@dataclass
class ConstraintStrain:
    """Strain on a specific constraint."""
    constraint_id: str
    strain: float  # 0-1
    threshold: float
    exceeded: bool
    
    @property
    def margin(self) -> float:
        return self.threshold - self.strain


@dataclass
class KernelSnapshot:
    """
    Complete state snapshot from the kernel.
    
    This is the primary diagnostic output.
    """
    schema_version: str = SCHEMA_VERSION
    session_id: str = ""
    step_id: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    regime: Regime = Regime.STABLE
    risk_vector: RiskVector = field(default_factory=RiskVector)
    constraint_strains: List[ConstraintStrain] = field(default_factory=list)
    budget: Budget = field(default_factory=Budget)
    
    # Pressure dynamics
    pressure: float = 0.0
    pressure_level: str = "NORMAL"  # NORMAL, ELEVATED, HIGH, CRITICAL, SHUTDOWN
    
    # Evidence (minimal pointers, not prose)
    active_violations: List[str] = field(default_factory=list)  # violation_ids
    active_dead_ends: List[str] = field(default_factory=list)   # dead_end_ids
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "step_id": self.step_id,
            "timestamp": self.timestamp.isoformat(),
            "regime": self.regime.name,
            "risk_vector": self.risk_vector.to_dict(),
            "pressure": self.pressure,
            "pressure_level": self.pressure_level,
            "active_violations": self.active_violations,
            "active_dead_ends": self.active_dead_ends,
        }


@dataclass
class Violation:
    """
    A typed constraint violation.
    
    Not prose. Structured, traceable, actionable.
    """
    violation_id: str
    violation_type: ViolationType
    constraint_id: str
    severity: float  # 0-1
    
    # Location
    span_start: Optional[int] = None
    span_end: Optional[int] = None
    token_range: Optional[Tuple[int, int]] = None
    
    # Evidence
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    # Repair
    repair_hint: Optional[str] = None  # Typed hint, not prose
    repairable: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "violation_type": self.violation_type.name,
            "constraint_id": self.constraint_id,
            "severity": self.severity,
            "span": (self.span_start, self.span_end) if self.span_start else None,
            "evidence": self.evidence,
            "repair_hint": self.repair_hint,
            "repairable": self.repairable,
        }


@dataclass
class Action:
    """
    A governor action (decision + rationale).
    
    Every action is typed, phased, and traceable.
    """
    action_id: str
    phase: ActionPhase
    action_type: str  # e.g., "block_claim", "force_retrieval", "abort"
    
    # Parameters (typed struct union in practice)
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Rationale
    reason_code: str = ""
    triggering_signals: Dict[str, float] = field(default_factory=dict)
    
    # Traceability
    evidence_refs: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "phase": self.phase.name,
            "action_type": self.action_type,
            "params": self.params,
            "reason_code": self.reason_code,
            "triggering_signals": self.triggering_signals,
            "evidence_refs": self.evidence_refs,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Preflight (shape the transform BEFORE generation)
# =============================================================================

@dataclass
class PromptMutation:
    """
    A typed mutation to apply to the prompt.
    
    NOT free-form rewriting. Typed, auditable, minimal.
    """
    mutation_type: str  # e.g., "add_requirement", "narrow_scope", "force_citation"
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Standard mutations
    @classmethod
    def require_citation(cls) -> "PromptMutation":
        return cls("require_citation", {})
    
    @classmethod
    def force_uncertainty(cls) -> "PromptMutation":
        return cls("force_uncertainty", {})
    
    @classmethod
    def narrow_scope(cls, scope: str) -> "PromptMutation":
        return cls("narrow_scope", {"scope": scope})
    
    @classmethod
    def add_decomposition_step(cls) -> "PromptMutation":
        return cls("add_decomposition_step", {})
    
    @classmethod
    def cap_claims(cls, max_claims: int) -> "PromptMutation":
        return cls("cap_claims", {"max_claims": max_claims})


@dataclass
class SamplingProfile:
    """Sampling parameters shaped by preflight."""
    temperature: float = 0.7
    temperature_cap: float = 1.0
    top_p: float = 0.9
    top_p_cap: float = 1.0
    max_tokens: int = 2000
    
    # Dynamic adjustment
    backpressure_multiplier: float = 1.0  # Applied to temperature under strain


@dataclass
class PreflightPlan:
    """
    The output of preflight: how to shape the generation.
    
    This is the "connection" in the transport model—
    it determines how the system moves through semantic space.
    """
    # Transform selection
    allowed_transforms: Set[TransformClass] = field(default_factory=lambda: {TransformClass.DIRECT_ANSWER})
    recommended_transform: Optional[TransformClass] = None
    
    # Prompt shaping (minimal, typed)
    prompt_mutations: List[PromptMutation] = field(default_factory=list)
    
    # Resource budgets
    budget: Budget = field(default_factory=Budget)
    
    # Sampling parameters
    sampling_profile: SamplingProfile = field(default_factory=SamplingProfile)
    
    # Tool policy
    required_tools: List[str] = field(default_factory=list)
    forbidden_tools: List[str] = field(default_factory=list)
    
    # Stop conditions
    abort_on_strain_above: float = 0.9
    abort_on_violation_types: Set[ViolationType] = field(default_factory=set)
    
    # Rationale
    reason_codes: List[str] = field(default_factory=list)
    triggering_snapshot: Optional[KernelSnapshot] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed_transforms": [t.name for t in self.allowed_transforms],
            "recommended_transform": self.recommended_transform.name if self.recommended_transform else None,
            "prompt_mutations": [{"type": m.mutation_type, "params": m.params} for m in self.prompt_mutations],
            "sampling_profile": {
                "temperature": self.sampling_profile.temperature,
                "max_tokens": self.sampling_profile.max_tokens,
            },
            "required_tools": self.required_tools,
            "abort_on_strain_above": self.abort_on_strain_above,
            "reason_codes": self.reason_codes,
        }


# =============================================================================
# Sampling-Time Gating (fly-by-wire)
# =============================================================================

@dataclass
class StreamDelta:
    """A chunk from the generation stream."""
    text: str
    token_count: int = 0
    cumulative_tokens: int = 0
    
    # Optional: logits if available
    logits: Optional[Any] = None
    top_tokens: Optional[List[Tuple[str, float]]] = None


@dataclass
class SamplingDirective:
    """
    Real-time steering directive.
    
    This is the fly-by-wire output. Issued per-chunk.
    """
    action: DirectiveAction = DirectiveAction.CONTINUE
    
    # For THROTTLE
    verbosity_limit: Optional[int] = None
    
    # For INSERT_CLARIFIER
    clarifier_text: Optional[str] = None
    
    # For ABORT_AND_REPLAN
    new_transform: Optional[TransformClass] = None
    replan_reason: Optional[str] = None
    
    # Rationale
    triggering_signal: Optional[str] = None
    signal_value: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.name,
            "verbosity_limit": self.verbosity_limit,
            "new_transform": self.new_transform.name if self.new_transform else None,
            "replan_reason": self.replan_reason,
            "triggering_signal": self.triggering_signal,
            "signal_value": self.signal_value,
        }


@dataclass
class GatedLogits:
    """
    Modified logits from hard gating.
    
    Only available when you control the sampler.
    """
    original_logits: Any  # numpy/torch array
    modified_logits: Any
    
    # What was done
    penalties_applied: Dict[str, float] = field(default_factory=dict)
    tokens_suppressed: List[str] = field(default_factory=list)
    temperature_adjusted: Optional[float] = None
    
    # Stop signal
    should_stop: bool = False
    stop_reason: Optional[str] = None


# =============================================================================
# Postcheck (validate and commit)
# =============================================================================

@dataclass
class RepairPlan:
    """Plan for repairing a failed output."""
    transform_class: TransformClass
    mutations: List[PromptMutation]
    reason: str


@dataclass
class PostcheckResult:
    """
    Result of postcheck validation.
    """
    verdict: PostcheckVerdict
    
    # For REVISE
    repair_plan: Optional[RepairPlan] = None
    required_edits: List[Dict[str, Any]] = field(default_factory=list)
    
    # For REFUSE
    refusal_reason: Optional[str] = None
    
    # Violations found
    violations: List[Violation] = field(default_factory=list)
    
    # Commitments to make (if ACCEPT)
    proposed_commits: List[Dict[str, Any]] = field(default_factory=list)
    
    # Trace
    trace_refs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.name,
            "violations": [v.to_dict() for v in self.violations],
            "repair_plan": {
                "transform": self.repair_plan.transform_class.name,
                "reason": self.repair_plan.reason,
            } if self.repair_plan else None,
            "refusal_reason": self.refusal_reason,
        }


# =============================================================================
# Explain (causal accounting, not vibes)
# =============================================================================

@dataclass
class ActionExplanation:
    """
    Causal explanation of a governor action.
    
    Not chain-of-thought. Structured accounting.
    """
    action: Action
    
    # What triggered it
    triggering_signals: Dict[str, float] = field(default_factory=dict)
    threshold_values: Dict[str, float] = field(default_factory=dict)
    
    # Supporting evidence
    supporting_violations: List[str] = field(default_factory=list)
    supporting_snapshots: List[str] = field(default_factory=list)
    
    # Counterfactual
    would_have_fired_if: Optional[str] = None  # "strain > 0.8"
    would_not_have_fired_if: Optional[str] = None  # "retrieval completed"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "triggering_signals": self.triggering_signals,
            "threshold_values": self.threshold_values,
            "supporting_violations": self.supporting_violations,
            "would_have_fired_if": self.would_have_fired_if,
            "would_not_have_fired_if": self.would_not_have_fired_if,
        }


# =============================================================================
# Trace (replay and audit)
# =============================================================================

class TraceEventKind(Enum):
    """Kinds of trace events."""
    STATE_UPDATE = auto()
    ACTION = auto()
    VIOLATION = auto()
    TOOL_CALL = auto()
    TOOL_RESULT = auto()
    TOKEN = auto()
    COMMIT = auto()
    PREFLIGHT = auto()
    DIRECTIVE = auto()
    POSTCHECK = auto()


@dataclass
class TraceEvent:
    """
    A single event in the trace log.
    
    Append-only. Immutable once written.
    """
    t: int  # Monotonic sequence number
    kind: TraceEventKind
    timestamp: datetime
    payload: Dict[str, Any]
    
    # Content hash for integrity
    @property
    def content_hash(self) -> str:
        content = json.dumps(self.payload, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "kind": self.kind.name,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "hash": self.content_hash,
        }


@dataclass
class TraceBundle:
    """
    Complete trace for replay.
    """
    schema_version: str = SCHEMA_VERSION
    session_id: str = ""
    
    # Context
    model_id: str = ""
    sampling_profile: Optional[SamplingProfile] = None
    
    # Events
    events: List[TraceEvent] = field(default_factory=list)
    
    # Hashes for integrity
    prompt_hashes: List[str] = field(default_factory=list)
    output_hashes: List[str] = field(default_factory=list)
    
    def to_json(self) -> str:
        return json.dumps({
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "model_id": self.model_id,
            "events": [e.to_dict() for e in self.events],
            "prompt_hashes": self.prompt_hashes,
            "output_hashes": self.output_hashes,
        }, indent=2, default=str)
    
    @classmethod
    def from_json(cls, data: str) -> "TraceBundle":
        d = json.loads(data)
        events = [
            TraceEvent(
                t=e["t"],
                kind=TraceEventKind[e["kind"]],
                timestamp=datetime.fromisoformat(e["timestamp"]),
                payload=e["payload"],
            )
            for e in d.get("events", [])
        ]
        return cls(
            schema_version=d.get("schema_version", "0.0.0"),
            session_id=d.get("session_id", ""),
            model_id=d.get("model_id", ""),
            events=events,
            prompt_hashes=d.get("prompt_hashes", []),
            output_hashes=d.get("output_hashes", []),
        )


@dataclass
class DeterminismReport:
    """Report from trace replay."""
    deterministic: bool
    divergence_point: Optional[int] = None  # Event t where divergence occurred
    divergence_reason: Optional[str] = None
    original_actions: List[str] = field(default_factory=list)
    replayed_actions: List[str] = field(default_factory=list)


# =============================================================================
# Governor API (the contract)
# =============================================================================

class GovernorAPI:
    """
    The formal API surface for the epistemic governor.
    
    This is the contract between:
    - Kernel (pure logic)
    - Adapter (model backend)
    - Host (application)
    
    All methods are typed, versioned, and traceable.
    """
    
    def __init__(self, kernel: Any, adapter: Any = None):
        """
        Initialize the API.
        
        Args:
            kernel: The epistemic kernel (your existing governor/session)
            adapter: Optional model adapter for generation
        """
        self.kernel = kernel
        self.adapter = adapter
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._traces: Dict[str, TraceBundle] = {}
        self._action_counter = 0
    
    def _generate_action_id(self) -> str:
        self._action_counter += 1
        return f"action_{self._action_counter:06d}"
    
    # =========================================================================
    # Lifecycle
    # =========================================================================
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new governed session."""
        if session_id is None:
            session_id = f"session_{datetime.now().timestamp()}"
        
        self._sessions[session_id] = {
            "created_at": datetime.now(),
            "step": 0,
            "last_snapshot": None,
            "last_plan": None,
        }
        self._traces[session_id] = TraceBundle(session_id=session_id)
        
        return session_id
    
    def close_session(self, session_id: str) -> TraceBundle:
        """Close a session and return its trace."""
        trace = self._traces.pop(session_id, TraceBundle())
        self._sessions.pop(session_id, None)
        return trace
    
    # =========================================================================
    # Preflight (BEFORE generation)
    # =========================================================================
    
    def preflight(
        self,
        session_id: str,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> PreflightPlan:
        """
        Shape the transform BEFORE generation.
        
        This is the thermostat, not the sprinkler.
        It doesn't write prompts—it chooses transform classes and sets budgets.
        
        Args:
            session_id: Session to operate on
            user_message: The user's input
            context: Optional additional context
        
        Returns:
            PreflightPlan with allowed transforms, mutations, budgets
        """
        # Get current state
        snapshot = self.get_snapshot(session_id)
        
        # Default plan
        plan = PreflightPlan(
            allowed_transforms={TransformClass.DIRECT_ANSWER},
            budget=Budget(),
            triggering_snapshot=snapshot,
        )
        
        # Adjust based on regime
        if snapshot.regime in (Regime.CONFABULATION, Regime.FLUENCY_DOMINANCE):
            # High risk: force retrieval or decomposition
            plan.allowed_transforms = {
                TransformClass.RETRIEVE_THEN_ANSWER,
                TransformClass.DECOMPOSE_THEN_ANSWER,
                TransformClass.ASK_CLARIFIER,
            }
            plan.recommended_transform = TransformClass.RETRIEVE_THEN_ANSWER
            plan.prompt_mutations.append(PromptMutation.require_citation())
            plan.reason_codes.append("regime_risk_high")
        
        elif snapshot.regime == Regime.NARRATIVE_DRIFT:
            # Drift detected: narrow scope, force grounding
            plan.prompt_mutations.append(PromptMutation.narrow_scope("factual"))
            plan.prompt_mutations.append(PromptMutation.cap_claims(3))
            plan.reason_codes.append("narrative_drift_detected")
        
        # Adjust based on risk vector
        if snapshot.risk_vector.factual_strain > 0.7:
            plan.allowed_transforms.discard(TransformClass.DIRECT_ANSWER)
            plan.allowed_transforms.add(TransformClass.CITE_OR_REFUSE)
            plan.prompt_mutations.append(PromptMutation.force_uncertainty())
            plan.reason_codes.append("factual_strain_high")
        
        # Adjust based on pressure
        if snapshot.pressure_level in ("HIGH", "CRITICAL", "SHUTDOWN"):
            plan.allowed_transforms = {
                TransformClass.ASK_CLARIFIER,
                TransformClass.REFUSE,
            }
            plan.sampling_profile.temperature = 0.3
            plan.sampling_profile.max_tokens = 500
            plan.reason_codes.append(f"pressure_{snapshot.pressure_level.lower()}")
        
        # Adjust based on dead ends
        if snapshot.active_dead_ends:
            plan.required_tools.append("retrieval")
            plan.reason_codes.append("dead_end_active")
        
        # Record in trace
        self._record_event(session_id, TraceEventKind.PREFLIGHT, {
            "user_message_hash": hashlib.sha256(user_message.encode()).hexdigest()[:16],
            "plan": plan.to_dict(),
        })
        
        # Store for reference
        if session_id in self._sessions:
            self._sessions[session_id]["last_plan"] = plan
        
        return plan
    
    # =========================================================================
    # Sampling-Time Gating (DURING generation)
    # =========================================================================
    
    def on_stream_delta(
        self,
        session_id: str,
        delta: StreamDelta,
    ) -> SamplingDirective:
        """
        Process a generation chunk and return steering directive.
        
        This is fly-by-wire. Called per-chunk during streaming.
        Works even with black-box APIs (no logits needed).
        
        Args:
            session_id: Session to operate on
            delta: The text chunk from the model
        
        Returns:
            SamplingDirective with action (CONTINUE, THROTTLE, ABORT, etc.)
        """
        # Get current state and plan
        snapshot = self.get_snapshot(session_id)
        plan = self._sessions.get(session_id, {}).get("last_plan")
        
        directive = SamplingDirective(action=DirectiveAction.CONTINUE)
        
        # Check strain against abort threshold
        if plan and snapshot.risk_vector.max_strain() > plan.abort_on_strain_above:
            directive = SamplingDirective(
                action=DirectiveAction.ABORT_AND_REPLAN,
                new_transform=TransformClass.RETRIEVE_THEN_ANSWER,
                replan_reason="strain_threshold_exceeded",
                triggering_signal="max_strain",
                signal_value=snapshot.risk_vector.max_strain(),
            )
        
        # Check for regime transition during generation
        elif snapshot.regime in (Regime.CONFABULATION, Regime.THERMAL_RUNAWAY):
            directive = SamplingDirective(
                action=DirectiveAction.ABORT_AND_REPLAN,
                new_transform=TransformClass.ASK_CLARIFIER,
                replan_reason="regime_degraded",
                triggering_signal="regime",
                signal_value=1.0,
            )
        
        # Check budget
        elif not snapshot.budget.can_afford("tokens", delta.token_count):
            directive = SamplingDirective(
                action=DirectiveAction.HARD_STOP,
                triggering_signal="token_budget",
                signal_value=float(snapshot.budget.tokens_remaining),
            )
        
        # Record in trace
        self._record_event(session_id, TraceEventKind.DIRECTIVE, {
            "delta_length": len(delta.text),
            "directive": directive.to_dict(),
        })
        
        return directive
    
    def gate_next_token(
        self,
        session_id: str,
        logits: Any,
        token_meta: Optional[Dict] = None,
    ) -> GatedLogits:
        """
        Hard gating at the logits level.
        
        Only available when you control the sampler (vLLM, llama.cpp).
        
        This is the most powerful intervention point:
        - Penalize speculative tokens under factual strain
        - Enforce uncertainty grammar
        - Dynamic temperature as a function of strain
        
        Args:
            session_id: Session to operate on
            logits: The raw logits from the model
            token_meta: Optional metadata about the token position
        
        Returns:
            GatedLogits with modified probability distribution
        """
        snapshot = self.get_snapshot(session_id)
        
        # Start with original
        modified = logits  # In practice, clone this
        penalties = {}
        suppressed = []
        
        # Under high factual strain, penalize speculative completions
        if snapshot.risk_vector.factual_strain > 0.6:
            # Would penalize: numbers, dates, named entities, confident assertions
            penalties["speculative_tokens"] = 0.5
        
        # Adjust temperature based on pressure
        temp_adjustment = None
        if snapshot.pressure > 0.7:
            temp_adjustment = 0.3  # Cool down under pressure
        
        # Check for stop condition
        should_stop = False
        stop_reason = None
        if snapshot.regime == Regime.THERMAL_RUNAWAY:
            should_stop = True
            stop_reason = "thermal_runaway"
        
        return GatedLogits(
            original_logits=logits,
            modified_logits=modified,
            penalties_applied=penalties,
            tokens_suppressed=suppressed,
            temperature_adjusted=temp_adjustment,
            should_stop=should_stop,
            stop_reason=stop_reason,
        )
    
    # =========================================================================
    # Postcheck (AFTER generation)
    # =========================================================================
    
    def postcheck(
        self,
        session_id: str,
        full_output: str,
    ) -> PostcheckResult:
        """
        Validate output and decide: accept, revise, or refuse.
        
        This is the sprinkler. Fire happened, assess damage.
        
        Args:
            session_id: Session to operate on
            full_output: The complete model output
        
        Returns:
            PostcheckResult with verdict and any violations
        """
        # Get state
        snapshot = self.get_snapshot(session_id)
        
        # Collect violations (would use registry.audit() in practice)
        violations: List[Violation] = []
        
        # Check for active violations from snapshot
        for v_id in snapshot.active_violations:
            violations.append(Violation(
                violation_id=v_id,
                violation_type=ViolationType.COMMITMENT_VIOLATION,
                constraint_id="unknown",
                severity=0.5,
            ))
        
        # Determine verdict
        if not violations:
            result = PostcheckResult(
                verdict=PostcheckVerdict.ACCEPT,
                proposed_commits=[],  # Would extract claims here
            )
        elif all(v.repairable for v in violations):
            result = PostcheckResult(
                verdict=PostcheckVerdict.REVISE,
                violations=violations,
                repair_plan=RepairPlan(
                    transform_class=TransformClass.CITE_OR_REFUSE,
                    mutations=[PromptMutation.require_citation()],
                    reason="violations_repairable",
                ),
            )
        else:
            result = PostcheckResult(
                verdict=PostcheckVerdict.REFUSE,
                violations=violations,
                refusal_reason="irrepairable_violations",
            )
        
        # Record in trace
        self._record_event(session_id, TraceEventKind.POSTCHECK, {
            "output_hash": hashlib.sha256(full_output.encode()).hexdigest()[:16],
            "result": result.to_dict(),
        })
        
        return result
    
    # =========================================================================
    # State & Diagnostics
    # =========================================================================
    
    def get_snapshot(self, session_id: str) -> KernelSnapshot:
        """Get current kernel state snapshot."""
        # In practice, this pulls from self.kernel
        # For now, return a default
        session = self._sessions.get(session_id, {})
        
        return KernelSnapshot(
            session_id=session_id,
            step_id=session.get("step", 0),
            regime=Regime.STABLE,
            risk_vector=RiskVector(),
            budget=Budget(),
        )
    
    def get_violations(self, session_id: str) -> List[Violation]:
        """Get active violations for a session."""
        snapshot = self.get_snapshot(session_id)
        # Would convert snapshot.active_violations to full Violation objects
        return []
    
    def explain_last_action(self, session_id: str) -> Optional[ActionExplanation]:
        """
        Get causal explanation of the last action.
        
        Not chain-of-thought. Structured accounting.
        """
        trace = self._traces.get(session_id)
        if not trace or not trace.events:
            return None
        
        # Find last action event
        for event in reversed(trace.events):
            if event.kind == TraceEventKind.ACTION:
                action = Action(
                    action_id=event.payload.get("action_id", ""),
                    phase=ActionPhase[event.payload.get("phase", "POSTCHECK")],
                    action_type=event.payload.get("action_type", ""),
                )
                return ActionExplanation(
                    action=action,
                    triggering_signals=event.payload.get("triggering_signals", {}),
                )
        
        return None
    
    # =========================================================================
    # Trace & Replay
    # =========================================================================
    
    def export_trace(self, session_id: str) -> TraceBundle:
        """Export the trace for a session."""
        return self._traces.get(session_id, TraceBundle())
    
    def replay(self, trace: TraceBundle) -> DeterminismReport:
        """
        Replay a trace and check for determinism.
        
        Returns report of whether replay produced same decisions.
        """
        # Would re-run all events through kernel and compare
        return DeterminismReport(
            deterministic=True,
            original_actions=[],
            replayed_actions=[],
        )
    
    def _record_event(self, session_id: str, kind: TraceEventKind, payload: Dict):
        """Record an event in the trace."""
        trace = self._traces.get(session_id)
        if trace:
            t = len(trace.events)
            trace.events.append(TraceEvent(
                t=t,
                kind=kind,
                timestamp=datetime.now(),
                payload=payload,
            ))


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=== Governor API Demo ===\n")
    print(f"Schema version: {SCHEMA_VERSION}\n")
    
    # Create API (would pass real kernel in practice)
    api = GovernorAPI(kernel=None)
    
    # Create session
    session_id = api.create_session()
    print(f"Session: {session_id}\n")
    
    # Preflight
    print("--- Preflight ---")
    plan = api.preflight(session_id, "What is the capital of France?")
    print(f"Allowed transforms: {[t.name for t in plan.allowed_transforms]}")
    print(f"Mutations: {[m.mutation_type for m in plan.prompt_mutations]}")
    print(f"Reason codes: {plan.reason_codes}")
    
    # Simulate streaming
    print("\n--- Streaming ---")
    chunks = ["Paris", " is", " the", " capital", " of", " France", "."]
    for i, chunk in enumerate(chunks):
        delta = StreamDelta(text=chunk, token_count=1, cumulative_tokens=i+1)
        directive = api.on_stream_delta(session_id, delta)
        if directive.action != DirectiveAction.CONTINUE:
            print(f"  Directive: {directive.action.name} - {directive.replan_reason}")
            break
    else:
        print("  All chunks: CONTINUE")
    
    # Postcheck
    print("\n--- Postcheck ---")
    result = api.postcheck(session_id, "Paris is the capital of France.")
    print(f"Verdict: {result.verdict.name}")
    print(f"Violations: {len(result.violations)}")
    
    # Export trace
    print("\n--- Trace ---")
    trace = api.export_trace(session_id)
    print(f"Events: {len(trace.events)}")
    for event in trace.events:
        print(f"  [{event.t}] {event.kind.name}")
    
    # Close
    api.close_session(session_id)
    print("\n✓ Session closed")
