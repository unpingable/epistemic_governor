"""
Epistemic Governor Package

A control-theoretic approach to LLM output governance.

Architecture:
    LLM (plant) → Extractor → Governor → Commit Phase → Ledger
                      ↑                        ↓
                  Envelope               Irreversibility

Full Stack:
    1. NegativeTAnalyzer - Detects population inversion signatures
    2. ValvePolicy - Thermostatic load-shedding (turn detection into control)
    3. EpistemicKernel - Unified commitment governance
    4. Ledger - Append-only state with irreversibility

Core principle: Entropy produces text. Governance turns text into history.

Modules:
    kernel.py       - Unified epistemic kernel (recommended entry point)
    governor.py     - Two-phase governor (envelope + adjudication)
    extractor.py    - Commitment extraction from text
    ledger.py       - Append-only epistemic state
    commit_phase.py - Transaction boundary and pipeline
    negative_t.py   - Population inversion / hallucination detection
    valve.py        - Thermostatic commitment throttling
"""

# Unified Kernel (recommended entry point)
from .kernel import (
    EpistemicKernel,
    EpistemicFrame,
    ThermalState,
    RevisionHandler,
)

# Session API (production interface)
from .session import (
    EpistemicSession,
    InstrumentedSession,
    LLMProvider,
    PassthroughProvider,
    SessionMode,
    LedgerSnapshot,
    Stratum,
    LedgerDiff,
    create_session,
)

# Calibration (system identification + policy fitting)
from .calibrate import (
    Calibrator,
    CalibrationCorpus,
    CalibrationPrompt,
    PromptType,
    BaselineProfile,
    PolicyProfile,
    DomainPolicy,
    ObjectivePreset,
    ReplayHarness,
    ReplayMetrics,
    TruthGate,
    CharacterizationReport,
    create_demo_corpus,
    compare_profiles,
)

# Homeostat (adaptive gain scheduling)
from .homeostat import (
    Homeostat,
    HomeostatMode,
    HomeostaticSession,
    EpistemicVitals,
    EpistemicSetpoints,
    TuningDelta,
    ExplorationContext,
    ExplorationBudget,
    ExplorationProfile,
    EXPLORATION_PROFILES,
)

# Detailed Governor (alternative, more granular)
from .governor import (
    EpistemicGovernor,
    GovernorState,
    GenerationEnvelope,
    AdjudicationResult,
    CommitDecision,
    CommitAction,
    ClaimType,
    ProposedCommitment,
    CommittedClaim,
    CommitmentStatus,
    InstabilityMetrics,
    DomainProfile,
    SessionBudget,
)

from .extractor import (
    CommitmentExtractor,
    ExtractionConfig,
    extract_confidence,
    classify_claim_type,
)

from .ledger import (
    EpistemicLedger,
    LedgerEntry,
    EntryType,
    RevisionRecord,
    Epoch,
    Fork,
    ContextReset,
    FossilRecord,
    DecayPolicy,
    CompactionResult,
)

from .commit_phase import (
    CommitPhase,
    CommitResult,
    TransactionStatus,
    EpistemicPipeline,
    SupportHandler,
    SupportResult,
    TextModifier,
)

# Hallucination Detection (Negative-T Analysis)
from .negative_t import (
    NegativeTAnalyzer,
    Regime,
    AnalyzerState,
    TurnMetrics,
    TrajectoryPoint,
    analyze_transcript,
    CommitmentAnalyzer,
    SupportAnalyzer,
    CalibrationAnalyzer,
    PumpingDetector,
    ConsistencyTracker,
)

# Thermostatic Valve (Load-Shedding)
from .valve import (
    ValvePolicy,
    ValveAction,
    ValveDecision,
    ValveConfig,
    Anchor,
    PrecisionExtractor,
    build_rewrite_prompt,
    integrate_with_analyzer,
)

# Stress Testing Harness
from .harness import (
    EpistemicStressHarness,
    TestProtocol,
    TelemetryPoint,
    TestResult,
)

# Providers (Local Model Support)
from .providers import (
    BaseProvider,
    HuggingFaceProvider,
    OllamaProvider,
    OpenAIProvider,
    AnthropicProvider,
    MockProvider,
    create_provider,
    get_device,
    get_device_info,
    DeviceType,
)

# Event Logging
from .events import (
    EventLogger,
    EventType,
    BaseEvent,
    TurnEvent,
    ProposalEvent,
    DecisionEvent,
    CommitEvent,
    ThermalEvent,
    DriftEvent,
    RevisionEvent,
    RefusalEvent,
    CalibrationEvent,
    EventSink,
    JSONLSink,
    MemorySink,
    PrometheusMetrics,
    load_events,
    analyze_events,
)

# Module Registry (ABI for constraint drivers)
from .registry import (
    ModuleRegistry,
    Domain,
    InvariantAction,
    AuditStatus,
    AuditMode,
    ProposalEnvelope,
    StateView,
    InvariantResult,
    InvariantSpec,
    AuditReport,
    Invariant,
    MonotonicTimeInvariant,
    ProposalIdInvariant,
    create_registry,
)

# Epistemic Module (assertions over symbols)
from .epistemic_module import (
    EpistemicConfig,
    ConfidenceCeilingInvariant,
    ThermalRegimeInvariant,
    ContradictionInvariant,
    SupportRequirementInvariant,
    DuplicateClaimInvariant,
    RevisionCostInvariant,
    register_epistemic_invariants,
    create_epistemic_registry,
)

# Domain Module Stubs
from .domain_modules import (
    # Sensory
    SensoryConfig,
    ObservationFreshnessInvariant,
    GroundingRequirementInvariant,
    register_sensory_invariants,
    # Kinetic
    KineticConfig,
    ActionMagnitudeInvariant,
    ActionRateLimitInvariant,
    register_kinetic_invariants,
    # Resource
    ResourceConfig,
    BudgetEnforcementInvariant,
    WorkAccumulationInvariant,
    register_resource_invariants,
    # Normative
    NormativeConfig,
    ActionClassBlockingInvariant,
    ContextRestrictionInvariant,
    register_normative_invariants,
    # Convenience
    register_all_modules,
)

# Creative Regime
from .creative import (
    CreativeRegime,
    CreativeConfig,
    CreativeMode,
    CreativeState,
    CreativeProposal,
    NoEpistemicCommitInvariant,
    BoredomInvariant,
    IdentityStiffnessInvariant,
)

# Character Ledger (Fictional Invariants)
from .character import (
    CharacterLedger,
    Character,
    Scene,
    Scar,
    Trait,
    Relationship,
    ContinuityViolation,
    ViolationType,
)

# Evaluation Framework
from .evaluation import (
    EvaluationCorpus,
    AnnotatedSample,
    AnnotatedClaim,
    AnnotatedClaimType,
    ExtractionEvaluator,
    ExtractionResults,
    AggregateExtractionResults,
    GovernanceComparison,
    ComparisonResults,
    create_demo_corpus,
    run_evaluation,
)

# ΔR Shear Analysis (Commitment Transport)
from .shear import (
    ShearAnalyzer,
    ShearReport,
    ShearState,
    ShearEvent,
    ShearEventType,
    Commitment,
    CommitmentType,
    Modality,
    Quantifier,
    TransformType,
    TransportStatus,
    TransportEvidence,
    TRANSFORM_FAILURE_MODES,
)

# Third Loop (Regime Detection + Structural Intervention)
from .regimes import (
    Regime,
    RegimeSignals,
    RegimeClassification,
    RegimeDetector,
    TopologyMutation,
    MutationType,
    StructuralController,
    ThirdLoop,
    ComplianceCost,
)

# Accretion Core (Directionality)
from .accretion import (
    AccretionCore,
    AccretedFact,
    FactStatus,
    SupportLink,
    AccretionEnergy,
    AccretionAction,
    AccretionActionType,
    AccretionGate,
)

# Vector Control (Manifold Geometry)
from .vectors import (
    StateVector,
    VectorController,
    BackpressurePolicy,
    PressureLevel,
    ForbiddenDirection,
    FORBIDDEN_BASES,
)

# Concurrency Safety (Event Sourcing)
from .concurrency import (
    EventLog,
    TurnEvent,
    EventType,
    AtomicTurn,
    SafeController,
    ControllerSnapshot,
)

# Competence Metrics (Effective Cognition)
from .competence import (
    CompetenceTracker,
    CompetenceReport,
    TurnOutcome,
    EfficiencyComparison,
)

# Structural Resistance (Real "There")
from .resistance import (
    StructuralResistance,
    CommitmentLedger,
    Commitment,
    DeadEnd,
    DeadEndType,
    # Typed claims
    TypedClaim,
    PredicateType,
    # Entity namespacing
    EntityRef,
    EntityNamespace,
    # Hypothesis ledger
    HypothesisLedger,
    Hypothesis,
    HypothesisStatus,
    # Trust levels
    TrustLevel,
    # Graphs
    DualGraph,
    LinkType,
)

# Tool Interfaces (External Friction)
from .tools import (
    # Support retrieval
    SupportCandidate,
    SupportRetriever,
    MockRetriever,
    # Provenance
    ProvenanceStore,
    ProvenanceRecord,
    VerificationStatus,
    VerificationResult,
    # Consistency checking
    ConsistencyChecker,
    ConsistencyResult,
    ConsistencyReport,
    RetractionObjective,
    MockConsistencyChecker,
    # Constrained decoding
    OutputType,
    OutputConstraint,
    ShapeConstraint,
    PolicyConstraint,
    # Context tracking
    ToolCallContext,
    ToolResult,
    # Tool registry
    ToolRegistry,
)

# Mode Isolation
from .modes import (
    Mode,
    ModeController,
    ModeState,
    ModeConstraints,
    TransitionReason,
    TransitionRule,
    MODE_CONSTRAINTS,
)

# Curiosity Module
from .curiosity import (
    CuriosityOperator,
    CuriosityQuestion,
    ProbeResult,
    TriggerSignals,
    TriggerType,
    QuestionType,
    NoProbeReason,
    InsufficientInfo,
)

# FDT Probes
from .probes import (
    FDTProbeOperator,
    ProbeType,
    ProbeDefinition,
    ProbeResponse,
    DampingProfile,
    BarrierEstimate,
    HealthAssessment,
    ResponseType,
    PROBE_LIBRARY,
)

__version__ = "0.1.0"
__all__ = [
    # Unified Kernel
    "EpistemicKernel",
    "EpistemicFrame",
    "ThermalState",
    "RevisionHandler",
    
    # Session API
    "EpistemicSession",
    "InstrumentedSession",
    "LLMProvider",
    "PassthroughProvider",
    "SessionMode",
    "LedgerSnapshot",
    "Stratum",
    "LedgerDiff",
    "create_session",
    
    # Calibration
    "Calibrator",
    "CalibrationCorpus",
    "CalibrationPrompt",
    "PromptType",
    "BaselineProfile",
    "PolicyProfile",
    "DomainPolicy",
    "ObjectivePreset",
    "ReplayHarness",
    "ReplayMetrics",
    "TruthGate",
    "CharacterizationReport",
    "create_demo_corpus",
    
    # Homeostat
    "Homeostat",
    "HomeostatMode",
    "HomeostaticSession",
    "EpistemicVitals",
    "EpistemicSetpoints",
    "TuningDelta",
    "ExplorationContext",
    "ExplorationBudget",
    "ExplorationProfile",
    "EXPLORATION_PROFILES",
    
    # Governor (detailed)
    "EpistemicGovernor",
    "GovernorState",
    "GenerationEnvelope",
    "AdjudicationResult",
    "CommitDecision",
    "CommitAction",
    "ClaimType",
    "ProposedCommitment",
    "CommittedClaim",
    "CommitmentStatus",
    "InstabilityMetrics",
    "DomainProfile",
    "SessionBudget",
    
    # Extractor
    "CommitmentExtractor",
    "ExtractionConfig",
    "extract_confidence",
    "classify_claim_type",
    
    # Ledger
    "EpistemicLedger",
    "LedgerEntry",
    "EntryType",
    "RevisionRecord",
    "Epoch",
    "Fork",
    "ContextReset",
    "FossilRecord",
    "DecayPolicy",
    "CompactionResult",
    
    # Commit Phase
    "CommitPhase",
    "CommitResult",
    "TransactionStatus",
    "EpistemicPipeline",
    "SupportHandler",
    "SupportResult",
    "TextModifier",
    
    # Negative-T Analysis (Hallucination Detection)
    "NegativeTAnalyzer",
    "Regime",
    "AnalyzerState",
    "TurnMetrics",
    "TrajectoryPoint",
    "analyze_transcript",
    "CommitmentAnalyzer",
    "SupportAnalyzer",
    "CalibrationAnalyzer",
    "PumpingDetector",
    "ConsistencyTracker",
    
    # Thermostatic Valve
    "ValvePolicy",
    "ValveAction",
    "ValveDecision",
    "ValveConfig",
    "Anchor",
    "PrecisionExtractor",
    "build_rewrite_prompt",
    "integrate_with_analyzer",
    
    # Stress Testing Harness
    "EpistemicStressHarness",
    "TestProtocol",
    "TelemetryPoint",
    "TestResult",
    
    # Providers (Local + Cloud Model Support)
    "BaseProvider",
    "HuggingFaceProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "MockProvider",
    "create_provider",
    "get_device",
    "get_device_info",
    "DeviceType",
    
    # Event Logging
    "EventLogger",
    "EventType",
    "BaseEvent",
    "TurnEvent",
    "ProposalEvent",
    "DecisionEvent",
    "CommitEvent",
    "ThermalEvent",
    "DriftEvent",
    "RevisionEvent",
    "RefusalEvent",
    "CalibrationEvent",
    "EventSink",
    "JSONLSink",
    "MemorySink",
    "PrometheusMetrics",
    "load_events",
    "analyze_events",
    
    # Module Registry (ABI)
    "ModuleRegistry",
    "Domain",
    "InvariantAction",
    "AuditStatus",
    "AuditMode",
    "ProposalEnvelope",
    "StateView",
    "InvariantResult",
    "InvariantSpec",
    "AuditReport",
    "Invariant",
    "MonotonicTimeInvariant",
    "ProposalIdInvariant",
    "create_registry",
    
    # Epistemic Module
    "EpistemicConfig",
    "ConfidenceCeilingInvariant",
    "ThermalRegimeInvariant",
    "ContradictionInvariant",
    "SupportRequirementInvariant",
    "DuplicateClaimInvariant",
    "RevisionCostInvariant",
    "register_epistemic_invariants",
    "create_epistemic_registry",
    
    # Domain Module Stubs
    "SensoryConfig",
    "ObservationFreshnessInvariant",
    "GroundingRequirementInvariant",
    "register_sensory_invariants",
    "KineticConfig",
    "ActionMagnitudeInvariant",
    "ActionRateLimitInvariant",
    "register_kinetic_invariants",
    "ResourceConfig",
    "BudgetEnforcementInvariant",
    "WorkAccumulationInvariant",
    "register_resource_invariants",
    "NormativeConfig",
    "ActionClassBlockingInvariant",
    "ContextRestrictionInvariant",
    "register_normative_invariants",
    "register_all_modules",
    
    # Creative Regime
    "CreativeRegime",
    "CreativeConfig",
    "CreativeMode",
    "CreativeState",
    "CreativeProposal",
    "NoEpistemicCommitInvariant",
    "BoredomInvariant",
    "IdentityStiffnessInvariant",
    
    # Character Ledger
    "CharacterLedger",
    "Character",
    "Scene",
    "Scar",
    "Trait",
    "Relationship",
    "ContinuityViolation",
    "ViolationType",
    
    # Evaluation Framework
    "EvaluationCorpus",
    "AnnotatedSample",
    "AnnotatedClaim",
    "AnnotatedClaimType",
    "ExtractionEvaluator",
    "ExtractionResults",
    "AggregateExtractionResults",
    "GovernanceComparison",
    "ComparisonResults",
    "create_demo_corpus",
    "run_evaluation",
    
    # ΔR Shear Analysis
    "ShearAnalyzer",
    "ShearReport",
    "ShearState",
    "ShearEvent",
    "ShearEventType",
    "Commitment",
    "CommitmentType",
    "Modality",
    "Quantifier",
    "TransformType",
    "TransportStatus",
    "TransportEvidence",
    "TRANSFORM_FAILURE_MODES",
    
    # Mode Isolation
    "Mode",
    "ModeController",
    "ModeState",
    "ModeConstraints",
    "TransitionReason",
    "TransitionRule",
    "MODE_CONSTRAINTS",
    
    # Curiosity Module
    "CuriosityOperator",
    "CuriosityQuestion",
    "ProbeResult",
    "TriggerSignals",
    "TriggerType",
    "QuestionType",
    "NoProbeReason",
    "InsufficientInfo",
    
    # FDT Probes
    "FDTProbeOperator",
    "ProbeType",
    "ProbeDefinition",
    "ProbeResponse",
    "DampingProfile",
    "BarrierEstimate",
    "HealthAssessment",
    "ResponseType",
    "PROBE_LIBRARY",
    
    # Formal API Surface
    "SCHEMA_VERSION",
    "Regime",
    "TransformClass",
    "ViolationType",
    "ActionPhase",
    "DirectiveAction",
    "PostcheckVerdict",
    "RiskVector",
    "Budget",
    "KernelSnapshot",
    "Violation",
    "Action",
    "PromptMutation",
    "SamplingProfile",
    "PreflightPlan",
    "StreamDelta",
    "SamplingDirective",
    "GatedLogits",
    "PostcheckResult",
    "RepairPlan",
    "ActionExplanation",
    "TraceEvent",
    "TraceBundle",
    "DeterminismReport",
    "GovernorAPI",
    
    # Flight Envelope Protection
    "FlightEnvelope",
    "ForbiddenRegime",
    "EnvelopeViolation",
    "EnvelopeThresholds",
    "HITLController",
    "HITLRole",
    "HITLForbidden",
    "HITLRequest",
    "HITLResponse",
    "FrictionLadder",
    "FrictionLevel",
    "FrictionState",
    "MinimalSignals",
    
    # Minimal Claim Ledger
    "ClaimLedger",
    "Claim",
    "ClaimStatus",
    "ClaimSource",
    "Provenance",
    "EvidenceRef",
    "PromotionResult",
    "PromotionAttempt",
    "extract_claim_signals",
    "create_claims_from_signals",
]

# Formal API Surface
from .api import (
    SCHEMA_VERSION,
    Regime,
    TransformClass,
    ViolationType,
    ActionPhase,
    DirectiveAction,
    PostcheckVerdict,
    RiskVector,
    Budget,
    KernelSnapshot,
    Violation,
    Action,
    PromptMutation,
    SamplingProfile,
    PreflightPlan,
    StreamDelta,
    SamplingDirective,
    GatedLogits,
    PostcheckResult,
    RepairPlan,
    ActionExplanation,
    TraceEvent,
    TraceBundle,
    DeterminismReport,
    GovernorAPI,
)

# Flight Envelope Protection
from .envelope import (
    FlightEnvelope,
    ForbiddenRegime,
    EnvelopeViolation,
    EnvelopeThresholds,
    HITLController,
    HITLRole,
    HITLForbidden,
    HITLRequest,
    HITLResponse,
    FrictionLadder,
    FrictionLevel,
    FrictionState,
    MinimalSignals,
)

# Minimal Claim Ledger
from .claims import (
    ClaimLedger,
    Claim,
    ClaimStatus,
    ClaimSource,
    Provenance,
    EvidenceRef,
    PromotionResult,
    PromotionAttempt,
    extract_claim_signals,
    create_claims_from_signals,
)

# Autopilot Layer
from .heading import (
    Heading,
    HeadingType,
    SummarizeHeading,
    TranslateHeading,
    ExtractClaimsHeading,
    RewriteHeading,
    ElaborateHeading,
    AnswerFromSourcesHeading,
    HeadingValidator,
    CitationPolicy,
    FidelityMode,
    LengthUnit,
    AUTOPILOT_SAFE_HEADINGS,
    FORBIDDEN_HEADINGS,
)

from .telemetry import (
    TelemetryComputer,
    WarningEvent,
    WarningType,
    LedgerSnapshot,
    EnvelopeSnapshot,
    AutopilotSnapshot,
    TelemetryThresholds,
    StabilizationState,
)

from .autopilot_fsm import (
    AutopilotFSM,
    AutopilotMode,
    TransitionResult,
    TransitionEvent,
    ConstraintClass,
    ConstraintSpec,
    ArbitrationOption,
    StabilizationReport,
)

from .autopilot_integration import (
    AutopilotController,
    LedgerTelemetryAdapter,
    HeadingEnforcer,
    ProposalVerdict,
    ProposalCheckResult,
)

# Claim Extraction & Diff
from .claim_extractor import (
    ClaimExtractor,
    ClaimAtom,
    ClaimSet,
    ExtractMode,
    Modality,
    Quantifier,
    Normalizer,
    # Mode discipline (INT-2)
    ClaimMode,
    MODE_ALLOWS_TIMELINE_OBLIGATIONS,
    MODE_REQUIRES_FRAMING,
    # Boundary gate (INT-1, INT-3)
    BoundaryGate,
    InputClassification,
    InputRiskClass,
)

from .claim_diff import (
    ClaimDiffer,
    DiffResult,
    MutationType,
    MutationEvent,
    AlignedPair,
    HeadingAdjudicator,
    AdjudicationResult,
    ViolationEvent,
)

# Streaming Telemetry
from .streaming_telemetry import (
    StreamingTelemetryIndex,
    TelemetryEvent,
    TelemetryEventType,
    StreamingTelemetryThresholds,
)

# Proposition Identity Router
from .prop_router import (
    PropositionRouter,
    PropositionIndex,
    BindAction,
    BindResult,
    PropBindEvent,
    PropRebindEvent,
    PropSplitEvent,
    PropositionRecord,
    EntityDetector,
)

# V2 Symbolic Substrate
from .symbolic_substrate import (
    # Predicates (non-stringly typed)
    PredicateType,
    Predicate,
    # Provenance
    ProvenanceClass,
    PROVENANCE_PRIORS,
    PROVENANCE_SUPPORT_MULTIPLIER,
    # Support/Evidence
    SupportItem,
    TemporalScope,
    # Dependencies
    DependencyType,
    Dependency,
    # Core objects
    Commitment,
    CandidateCommitment,
    # Time structure
    TemporalState,
    # State
    SymbolicState,
    # Adjudication
    AdjudicationDecision,
    AdjudicationResult,
    Adjudicator,
)

# V1 → V2 Bridge
from .v1_v2_bridge import (
    claim_atom_to_candidate,
    bridge_claims,
    bridge_with_mode_filter,
    PROVENANCE_MAP,
    infer_predicate_type,
)

# Quarantine Store
from .quarantine import (
    QuarantineStore,
    QuarantineEntry,
    QuarantineReason,
    PromotionEvent,
    PromotionRejection,
)
