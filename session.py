"""
Epistemic Session API

The production-facing interface for the epistemic governor.
This is the "syscall surface" - the narrow set of operations that
external systems use to interact with the kernel.

Design principles:
- Stateful session (tracks conversation history)
- Clean input/output types (no internal details leak)
- Provider-agnostic (LLM is injected, not assumed)
- Observable (every operation returns structured telemetry)

Usage:
    # Create session with a provider
    session = EpistemicSession(provider=my_llm_provider)
    
    # Process user input (full loop)
    frame = session.step("What is the capital of France?")
    
    # Or just govern existing output
    frame = session.govern(llm_output)
    
    # Inspect state
    snapshot = session.snapshot()
    strata = session.strata()
"""

from dataclasses import dataclass, field
from typing import Optional, List, Callable, Protocol, Any
from datetime import datetime
from enum import Enum

# Handle both package and direct imports
try:
    from .kernel import EpistemicKernel, EpistemicFrame, ThermalState
    from .governor import (
        GenerationEnvelope,
        ProposedCommitment,
        CommittedClaim,
        CommitmentStatus,
        ClaimType,
    )
    from .negative_t import NegativeTAnalyzer, Regime
    from .valve import ValvePolicy, ValveAction, ValveDecision, Anchor
except ImportError:
    from kernel import EpistemicKernel, EpistemicFrame, ThermalState
    from governor import (
        GenerationEnvelope,
        ProposedCommitment,
        CommittedClaim,
        CommitmentStatus,
        ClaimType,
    )
    from negative_t import NegativeTAnalyzer, Regime
    from valve import ValvePolicy, ValveAction, ValveDecision, Anchor


# =============================================================================
# Provider Protocol
# =============================================================================

class LLMProvider(Protocol):
    """
    Protocol for LLM providers.
    
    Implement this to plug in any LLM backend.
    The provider receives the envelope constraints and should respect them.
    """
    
    def generate(
        self,
        prompt: str,
        envelope: GenerationEnvelope,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate text from the LLM.
        
        Args:
            prompt: The user prompt
            envelope: Constraints from the governor (confidence ceilings, etc.)
            system_prompt: Optional system prompt with envelope instructions
            
        Returns:
            Generated text
        """
        ...


class PassthroughProvider:
    """
    Dummy provider that just returns the input.
    Use when you want to govern pre-generated text.
    """
    
    def generate(
        self,
        prompt: str,
        envelope: GenerationEnvelope,
        system_prompt: Optional[str] = None,
    ) -> str:
        return prompt


# =============================================================================
# Session Mode
# =============================================================================

class SessionMode(Enum):
    """Operating mode for the session."""
    NORMAL = "normal"           # Standard operation
    STRICT = "strict"           # All-or-nothing commits, no partial
    PERMISSIVE = "permissive"   # Allow more through, log warnings
    READONLY = "readonly"       # Analyze but don't commit


# =============================================================================
# Ledger Snapshot
# =============================================================================

@dataclass
class LedgerSnapshot:
    """
    Point-in-time view of the ledger state.
    
    This is what you'd serialize for persistence or debugging.
    """
    timestamp: datetime
    turn: int
    
    # Counts
    total_claims: int
    active_claims: int
    superseded_claims: int
    archived_claims: int
    
    # Thermal
    instability: float
    regime: str
    revision_count: int
    contradiction_count: int
    
    # Recent activity
    recent_commits: List[str] = field(default_factory=list)  # claim ids
    recent_blocks: List[str] = field(default_factory=list)   # proposal ids
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'turn': self.turn,
            'claims': {
                'total': self.total_claims,
                'active': self.active_claims,
                'superseded': self.superseded_claims,
                'archived': self.archived_claims,
            },
            'thermal': {
                'instability': self.instability,
                'regime': self.regime,
                'revision_count': self.revision_count,
                'contradiction_count': self.contradiction_count,
            },
            'recent': {
                'commits': self.recent_commits,
                'blocks': self.recent_blocks,
            },
        }


# =============================================================================
# Stratum (single layer in strata view)
# =============================================================================

@dataclass
class Stratum:
    """A single layer in the strata view."""
    layer: str              # ACTIVE, SUPERSEDED, ARCHIVED
    claim_id: str
    text: str
    confidence: float = 0.0
    claim_type: str = ""
    committed_at: Optional[datetime] = None
    supersedes: Optional[str] = None
    
    def to_dict(self) -> dict:
        d = {
            'layer': self.layer,
            'id': self.claim_id,
            'text': self.text,
        }
        if self.layer == 'ACTIVE':
            d['confidence'] = self.confidence
            d['type'] = self.claim_type
        if self.supersedes:
            d['supersedes'] = self.supersedes
        return d


# =============================================================================
# Diff (changes between turns)
# =============================================================================

@dataclass
class LedgerDiff:
    """
    Changes between two turns.
    
    This is the "git diff" for epistemic state.
    """
    from_turn: int
    to_turn: int
    
    added: List[str] = field(default_factory=list)       # new claim ids
    superseded: List[str] = field(default_factory=list)  # newly superseded
    archived: List[str] = field(default_factory=list)    # newly archived
    
    thermal_delta: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'from': self.from_turn,
            'to': self.to_turn,
            'added': self.added,
            'superseded': self.superseded,
            'archived': self.archived,
            'thermal_delta': self.thermal_delta,
        }
    
    @property
    def is_empty(self) -> bool:
        return not (self.added or self.superseded or self.archived)


# =============================================================================
# Epistemic Session
# =============================================================================

class EpistemicSession:
    """
    The production-facing interface for the epistemic governor.
    
    A session wraps:
    - An EpistemicKernel (the core governor + ledger)
    - A NegativeTAnalyzer (hallucination detection)
    - A ValvePolicy (load-shedding)
    - Optionally, an LLM provider
    
    The session provides:
    - step(): Full loop from user input to governed output
    - govern(): Govern pre-generated text
    - snapshot(): Point-in-time ledger state
    - strata(): Vertical view of commitments
    - diff(): Changes between turns
    """
    
    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        mode: SessionMode = SessionMode.NORMAL,
        enable_valve: bool = True,
        enable_shear: bool = True,
    ):
        """
        Create a new epistemic session.
        
        Args:
            provider: LLM provider (optional - use govern() if not provided)
            mode: Operating mode (normal, strict, permissive, readonly)
            enable_valve: Whether to use the hallucination valve (Δt)
            enable_shear: Whether to use the shear analyzer (ΔR)
        """
        self.provider = provider or PassthroughProvider()
        self.mode = mode
        self.enable_valve = enable_valve
        self.enable_shear = enable_shear
        
        # Core components
        self.kernel = EpistemicKernel()
        self.analyzer = NegativeTAnalyzer() if enable_valve else None
        self.valve = ValvePolicy() if enable_valve else None
        
        # ΔR shear tracking
        self.shear_analyzer = None
        if enable_shear:
            try:
                from .shear import ShearAnalyzer
            except ImportError:
                from shear import ShearAnalyzer
            self.shear_analyzer = ShearAnalyzer()
        
        # History tracking
        self._history: List[EpistemicFrame] = []
        self._snapshots: List[LedgerSnapshot] = []
    
    # =========================================================================
    # Core Operations
    # =========================================================================
    
    def step(
        self,
        user_input: str,
        domain: Optional[str] = None,
        anchors: Optional[List[Anchor]] = None,
    ) -> EpistemicFrame:
        """
        Full step: user input → LLM generation → governance → output.
        
        This is the main entry point when you have an LLM provider.
        
        Args:
            user_input: The user's message
            domain: Optional domain hint (e.g., "medical", "technical")
            anchors: Optional anchors for grounding (retrieved context, etc.)
            
        Returns:
            EpistemicFrame with governed output and ledger diff
        """
        # Track user turn in analyzer
        if self.analyzer:
            self.analyzer.add_turn(role="user", content=user_input)
        
        # Get envelope constraints
        envelope = self.kernel.get_envelope(domain=domain)
        
        # Build system prompt from envelope
        system_prompt = self._build_system_prompt(envelope)
        
        # Generate from provider
        raw_output = self.provider.generate(
            prompt=user_input,
            envelope=envelope,
            system_prompt=system_prompt,
        )
        
        # Govern the output
        return self.govern(raw_output, envelope=envelope, anchors=anchors)
    
    def govern(
        self,
        text: str,
        envelope: Optional[GenerationEnvelope] = None,
        anchors: Optional[List[Anchor]] = None,
    ) -> EpistemicFrame:
        """
        Govern pre-generated text.
        
        Use this when you've already generated text and want to
        run it through the epistemic governor.
        
        Args:
            text: The text to govern
            envelope: Optional envelope (will generate if not provided)
            anchors: Optional anchors for grounding
            
        Returns:
            EpistemicFrame with governance results
        """
        if envelope is None:
            envelope = self.kernel.get_envelope()
        
        # Check valve if enabled
        valve_decision = None
        if self.analyzer and self.valve:
            metrics = self.analyzer.add_turn(role="assistant", content=text)
            state = self.analyzer.get_state()
            
            valve_decision = self.valve.decide(
                turn_index=len(self.analyzer.turns),
                analyzer_state=state,
                candidate_metrics=metrics,
                candidate_text=text,
                anchors=anchors or [],
            )
            
            # Handle valve actions
            if valve_decision.action == ValveAction.HARD_STOP:
                # Return empty frame with refusal
                frame = EpistemicFrame(
                    output_text="[REFUSED: Thermal shutdown - unable to provide reliable response]",
                    original_text=text,
                    thermal=self.kernel.thermal,
                )
                frame.errors.append(f"Valve HARD_STOP: {valve_decision.reason}")
                self._history.append(frame)
                return frame
            
            elif valve_decision.action == ValveAction.ASK_CLARIFYING:
                # Return frame suggesting clarification
                frame = EpistemicFrame(
                    output_text="[NEEDS CLARIFICATION: Please provide sources or context to anchor this response]",
                    original_text=text,
                    thermal=self.kernel.thermal,
                )
                frame.errors.append(f"Valve ASK_CLARIFYING: {valve_decision.reason}")
                self._history.append(frame)
                return frame
        
        # Process through kernel
        if self.mode == SessionMode.READONLY:
            # Just analyze, don't commit
            frame = self._analyze_only(text, envelope)
        else:
            frame = self.kernel.process(text, envelope)
        
        # Record history
        self._history.append(frame)
        
        return frame
    
    def _analyze_only(
        self,
        text: str,
        envelope: GenerationEnvelope,
    ) -> EpistemicFrame:
        """Analyze without committing (readonly mode)."""
        # Extract proposals
        proposals = self.kernel.extractor.extract(text)
        
        # Adjudicate
        adjudication = self.kernel.governor.adjudicate(proposals, envelope)
        
        # Build frame without committing
        frame = EpistemicFrame(
            output_text=text,
            original_text=text,
            thermal=self.kernel.thermal,
        )
        
        # Report what would happen
        for decision in adjudication.decisions:
            prop = next((p for p in proposals if p.id == decision.commitment_id), None)
            if prop:
                frame.blocked.append((prop, decision))
        
        return frame
    
    def _build_system_prompt(self, envelope: GenerationEnvelope) -> str:
        """Build system prompt from envelope constraints."""
        return envelope.to_system_prompt_fragment()
    
    # =========================================================================
    # State Inspection
    # =========================================================================
    
    def snapshot(self) -> LedgerSnapshot:
        """
        Get a point-in-time snapshot of the ledger state.
        
        Returns:
            LedgerSnapshot with current state
        """
        status = self.kernel.get_status()
        stats = self.kernel.ledger.get_stats()
        
        # Recent activity from last frame
        recent_commits = []
        recent_blocks = []
        if self._history:
            last = self._history[-1]
            recent_commits = [c.id for c in last.committed]
            recent_blocks = [p.id for p, _ in last.blocked]
        
        snap = LedgerSnapshot(
            timestamp=datetime.now(),
            turn=self.kernel.turn,
            total_claims=stats['total_claims'],
            active_claims=stats['active_claims'],
            superseded_claims=stats['superseded_claims'],
            archived_claims=stats['archived_claims'],
            instability=self.kernel.thermal.instability,
            regime=self.kernel.thermal.regime,
            revision_count=self.kernel.thermal.revision_count,
            contradiction_count=self.kernel.thermal.contradiction_count,
            recent_commits=recent_commits,
            recent_blocks=recent_blocks,
        )
        
        self._snapshots.append(snap)
        return snap
    
    def strata(self, limit: int = 20) -> List[Stratum]:
        """
        Get the ledger as vertical strata.
        
        Returns claims organized by layer:
        - ACTIVE: Current commitments (top)
        - SUPERSEDED: Revised commitments (middle)
        - ARCHIVED: Fossilized commitments (bottom/bedrock)
        
        Args:
            limit: Maximum entries to return
            
        Returns:
            List of Stratum objects, most recent first
        """
        raw_strata = self.kernel.get_strata(limit)
        
        return [
            Stratum(
                layer=s['layer'],
                claim_id=s['id'],
                text=s['text'],
                confidence=s.get('confidence', 0.0),
                claim_type=s.get('type', ''),
                supersedes=s.get('supersedes'),
            )
            for s in raw_strata
        ]
    
    def diff(self, from_turn: int, to_turn: Optional[int] = None) -> LedgerDiff:
        """
        Get changes between two turns.
        
        Args:
            from_turn: Starting turn
            to_turn: Ending turn (default: current)
            
        Returns:
            LedgerDiff showing what changed
        """
        if to_turn is None:
            to_turn = self.kernel.turn
        
        # Build diff from history
        diff = LedgerDiff(from_turn=from_turn, to_turn=to_turn)
        
        initial_instability = 0.0
        for i, frame in enumerate(self._history):
            if i < from_turn:
                initial_instability = frame.thermal.instability
                continue
            if i > to_turn:
                break
            
            # Track additions
            diff.added.extend(c.id for c in frame.committed)
            
            # Track supersessions (from revision_required that were executed)
            # This is approximate - full tracking would need more state
        
        diff.thermal_delta = self.kernel.thermal.instability - initial_instability
        
        return diff
    
    # =========================================================================
    # Session Management
    # =========================================================================
    
    def reset(self):
        """Reset the session to fresh state."""
        self.kernel.reset()
        if self.analyzer:
            self.analyzer = NegativeTAnalyzer()
        if self.valve:
            self.valve.reset()
        if self.shear_analyzer:
            try:
                from .shear import ShearAnalyzer
            except ImportError:
                from shear import ShearAnalyzer
            self.shear_analyzer = ShearAnalyzer()
        self._history.clear()
        self._snapshots.clear()
    
    # =========================================================================
    # ΔR Shear Analysis
    # =========================================================================
    
    def set_commitment_baseline(self, text: str) -> int:
        """
        Set the baseline commitments for shear analysis.
        
        Call this with the original text (e.g., system prompt, spec)
        before transforms are applied.
        
        Args:
            text: The source text containing commitments
            
        Returns:
            Number of commitments extracted
        """
        if not self.shear_analyzer:
            return 0
        
        try:
            from .shear import Commitment
        except ImportError:
            from shear import Commitment
        
        # Extract commitments (simplified - in practice would use LLM)
        # For now, do basic extraction
        commitments = self._extract_basic_commitments(text)
        self.shear_analyzer.set_baseline(commitments)
        return len(commitments)
    
    def check_shear(self, transformed_text: str, transform_type: str = "unknown") -> dict:
        """
        Check for commitment shear after a transform.
        
        Compare transformed text against baseline commitments
        to detect what was lost, weakened, or contradicted.
        
        Args:
            transformed_text: The text after transformation
            transform_type: Type of transform applied
            
        Returns:
            Shear report as dict
        """
        if not self.shear_analyzer:
            return {"error": "Shear analyzer not enabled"}
        
        if not self.shear_analyzer.baseline_commitments:
            return {"error": "No baseline set - call set_commitment_baseline first"}
        
        try:
            from .shear import TransformType, ShearReport
        except ImportError:
            from shear import TransformType, ShearReport
        
        # Get transform type
        try:
            transform = TransformType(transform_type)
        except ValueError:
            transform = TransformType.OTHER
        
        # Extract commitments from transformed text
        target_commitments = self._extract_basic_commitments(transformed_text)
        
        # Build shear report
        report = self._build_shear_report(
            self.shear_analyzer.baseline_commitments,
            target_commitments,
            transform,
        )
        
        return {
            "transform": transform.value,
            "source_count": len(self.shear_analyzer.baseline_commitments),
            "target_count": len(target_commitments),
            "shear": report.shear,
            "torque": report.torque,
            "spurious_injection": report.spurious_injection,
            "preserved": report.preserved,
            "weakened": report.weakened,
            "dropped": report.dropped,
            "contradicted": report.contradicted,
            "warning": report.shear > self.shear_analyzer.shear_warning_threshold,
            "violation": report.shear > self.shear_analyzer.shear_violation_threshold,
        }
    
    def _extract_basic_commitments(self, text: str) -> list:
        """
        Basic commitment extraction (pattern-based).
        
        For production, this should use LLM extraction.
        """
        try:
            from .shear import Commitment, CommitmentType, Modality, Quantifier
        except ImportError:
            from shear import Commitment, CommitmentType, Modality, Quantifier
        
        commitments = []
        
        # Pattern: MUST/MUST NOT statements
        must_pattern = r'(?:must|shall|will always|required to)\s+([^.!?]+)'
        must_not_pattern = r'(?:must not|shall not|will never|cannot|prohibited from)\s+([^.!?]+)'
        should_pattern = r'(?:should|ought to|recommended to)\s+([^.!?]+)'
        
        import re
        
        for i, match in enumerate(re.finditer(must_pattern, text, re.I)):
            commitments.append(Commitment(
                id=f"C{i+1:02d}",
                type=CommitmentType.RULE,
                modality=Modality.MUST,
                quantifier=Quantifier.ALL,
                claim=match.group(1).strip()[:100],
            ))
        
        for i, match in enumerate(re.finditer(must_not_pattern, text, re.I)):
            commitments.append(Commitment(
                id=f"N{i+1:02d}",
                type=CommitmentType.EXCLUSION,
                modality=Modality.MUST_NOT,
                quantifier=Quantifier.ALL,
                claim=match.group(1).strip()[:100],
            ))
        
        for i, match in enumerate(re.finditer(should_pattern, text, re.I)):
            commitments.append(Commitment(
                id=f"S{i+1:02d}",
                type=CommitmentType.RULE,
                modality=Modality.SHOULD,
                quantifier=Quantifier.ALL,
                claim=match.group(1).strip()[:100],
            ))
        
        return commitments
    
    def _build_shear_report(self, source, target, transform):
        """Build a shear report comparing source and target commitments."""
        try:
            from .shear import ShearReport, TransportEvidence, TransportStatus
        except ImportError:
            from shear import ShearReport, TransportEvidence, TransportStatus
        
        # Create semantic hashes for matching
        source_claims = {c.claim.lower(): c for c in source}
        target_claims = {c.claim.lower(): c for c in target}
        
        transport_evidence = []
        
        for claim_text, src_commitment in source_claims.items():
            if claim_text in target_claims:
                # Found in target
                tgt = target_claims[claim_text]
                if src_commitment.modality == tgt.modality:
                    status = TransportStatus.PRESERVED
                    note = "Exact match in target"
                else:
                    status = TransportStatus.WEAKENED
                    note = f"Modal change: {src_commitment.modality.value} → {tgt.modality.value}"
                evidence = claim_text[:50]
            else:
                # Not found
                status = TransportStatus.DROPPED
                evidence = None
                note = "Not found in target"
            
            transport_evidence.append(TransportEvidence(
                commitment_id=src_commitment.id,
                status=status,
                evidence_span=evidence,
                note=note,
                original_modality=src_commitment.modality,
            ))
        
        return ShearReport(
            source_id="session",
            transform=transform,
            source_commitments=source,
            target_commitments=target,
            transport_evidence=transport_evidence,
        )

    @property
    def turn(self) -> int:
        """Current turn number."""
        return self.kernel.turn
    
    @property
    def thermal(self) -> ThermalState:
        """Current thermal state."""
        return self.kernel.thermal
    
    @property
    def history(self) -> List[EpistemicFrame]:
        """History of frames (read-only)."""
        return list(self._history)
    
    def get_claim(self, claim_id: str) -> Optional[CommittedClaim]:
        """Get a specific claim by ID."""
        return self.kernel.ledger.get_claim(claim_id)
    
    def explain(self, claim_id: str) -> dict:
        """
        Explain a claim's provenance.
        
        Returns information about when/how a claim was committed,
        what it superseded, and its current status.
        """
        claim = self.get_claim(claim_id)
        if not claim:
            return {'error': f'Claim {claim_id} not found'}
        
        # Get revision chain
        chain = self.kernel.ledger.get_revision_chain(claim_id)
        
        return {
            'id': claim.id,
            'text': claim.text,
            'type': claim.claim_type.name,
            'confidence': claim.confidence,
            'status': claim.status.name,
            'committed_at': claim.committed_at.isoformat() if claim.committed_at else None,
            'supersedes': claim.supersedes,
            'support_refs': claim.support_refs,
            'revision_chain': chain,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_session(
    provider: Optional[LLMProvider] = None,
    mode: str = "normal",
    enable_valve: bool = True,
    enable_shear: bool = True,
) -> EpistemicSession:
    """
    Convenience function to create a session.
    
    Args:
        provider: LLM provider (optional)
        mode: "normal", "strict", "permissive", or "readonly"
        enable_valve: Whether to enable hallucination detection (Δt)
        enable_shear: Whether to enable shear analysis (ΔR)
        
    Returns:
        Configured EpistemicSession
    """
    mode_map = {
        "normal": SessionMode.NORMAL,
        "strict": SessionMode.STRICT,
        "permissive": SessionMode.PERMISSIVE,
        "readonly": SessionMode.READONLY,
    }
    
    return EpistemicSession(
        provider=provider,
        mode=mode_map.get(mode, SessionMode.NORMAL),
        enable_valve=enable_valve,
        enable_shear=enable_shear,
    )


# =============================================================================
# Example Provider Implementation
# =============================================================================

class EchoProvider:
    """
    Example provider that echoes input with envelope info.
    Useful for testing.
    """
    
    def generate(
        self,
        prompt: str,
        envelope: GenerationEnvelope,
        system_prompt: Optional[str] = None,
    ) -> str:
        return f"[Echo] {prompt} (max_conf={envelope.max_confidence:.2f})"


# =============================================================================
# Instrumented Session (with Event Logging)
# =============================================================================

class InstrumentedSession:
    """
    Session wrapper that emits structured events for observability.
    
    This is the production interface when you need full telemetry.
    Events are emitted to JSONL files and optionally Prometheus.
    
    Usage:
        from epistemic_governor import InstrumentedSession
        
        # With file logging
        session = InstrumentedSession(
            provider=my_provider,
            event_log="events.jsonl",
        )
        
        # Run governed conversation
        frame = session.step("What is X?")
        
        # Check what happened
        events = session.get_events()
        summary = session.get_summary()
        
        # Clean shutdown
        session.close()
    """
    
    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        mode: SessionMode = SessionMode.NORMAL,
        enable_valve: bool = True,
        event_log: Optional[str] = None,
        enable_prometheus: bool = False,
        prometheus_port: int = 9090,
        session_id: Optional[str] = None,
    ):
        """
        Create an instrumented session.
        
        Args:
            provider: LLM provider
            mode: Operating mode
            enable_valve: Whether to use hallucination valve
            event_log: Path to JSONL event log (optional)
            enable_prometheus: Enable Prometheus metrics endpoint
            prometheus_port: Port for Prometheus metrics
            session_id: Optional session identifier
        """
        # Import events here to avoid circular import
        try:
            from .events import (
                EventLogger, TurnEvent, ProposalEvent, DecisionEvent,
                CommitEvent, ThermalEvent, DriftEvent, RevisionEvent,
                RefusalEvent, EventType
            )
        except ImportError:
            from events import (
                EventLogger, TurnEvent, ProposalEvent, DecisionEvent,
                CommitEvent, ThermalEvent, DriftEvent, RevisionEvent,
                RefusalEvent, EventType
            )
        
        # Store event classes for use in methods
        self._TurnEvent = TurnEvent
        self._ProposalEvent = ProposalEvent
        self._DecisionEvent = DecisionEvent
        self._CommitEvent = CommitEvent
        self._ThermalEvent = ThermalEvent
        self._DriftEvent = DriftEvent
        self._RevisionEvent = RevisionEvent
        self._RefusalEvent = RefusalEvent
        
        # Create underlying session
        self._session = EpistemicSession(
            provider=provider,
            mode=mode,
            enable_valve=enable_valve,
        )
        
        # Create event logger
        self._logger = EventLogger(
            output_path=event_log,
            enable_prometheus=enable_prometheus,
            prometheus_port=prometheus_port,
            session_id=session_id,
        )
        
        self._start_time = None
    
    @property
    def session_id(self) -> str:
        return self._logger.session_id
    
    @property
    def kernel(self):
        return self._session.kernel
    
    @property
    def ledger(self):
        return self._session.kernel.ledger
    
    @property
    def thermal(self):
        return self._session.thermal
    
    @property
    def turn_count(self) -> int:
        return self._session.turn_count
    
    def step(
        self,
        user_input: str,
        domain: Optional[str] = None,
        anchors: Optional[List[Anchor]] = None,
    ) -> EpistemicFrame:
        """
        Full step with event logging.
        
        Emits: TurnEvent, ProposalEvent(s), DecisionEvent(s), CommitEvent(s), ThermalEvent
        """
        import time
        
        turn_id = self._logger.next_turn()
        start_time = time.time()
        extraction_start = None
        governance_start = None
        
        # Run the actual step
        try:
            # Time extraction and governance separately if we can
            extraction_start = time.time()
            frame = self._session.step(user_input, domain=domain, anchors=anchors)
            governance_start = time.time()
        except Exception as e:
            # Log error and re-raise
            self._logger.emit(self._TurnEvent(
                turn_id=turn_id,
                prompt=user_input,
                response=f"ERROR: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
            ))
            raise
        
        end_time = time.time()
        
        # Emit turn event
        self._logger.emit(self._TurnEvent(
            turn_id=turn_id,
            prompt=user_input,
            response=frame.output_text[:500] if frame.output_text else "",
            latency_ms=(end_time - start_time) * 1000,
            extraction_ms=(governance_start - extraction_start) * 1000 if extraction_start else 0,
            governance_ms=(end_time - governance_start) * 1000 if governance_start else 0,
            model_id=getattr(self._session.provider, 'get_model_id', lambda: 'unknown')(),
            proposals_count=len(frame.committed) + len(frame.blocked) + len(frame.hedged),
            commits_count=len(frame.committed),
            hedges_count=len(frame.hedged),
            blocks_count=len(frame.blocked),
            revisions_count=len(frame.revision_required),
        ))
        
        # Emit commit events for each committed claim
        for claim in frame.committed:
            self._logger.emit(self._CommitEvent(
                turn_id=turn_id,
                claim_id=claim.id,
                claim_text=claim.text[:200],
                claim_type=claim.claim_type.name if hasattr(claim.claim_type, 'name') else str(claim.claim_type),
                final_confidence=claim.confidence,
                status=claim.status.name if hasattr(claim.status, 'name') else str(claim.status),
                supersedes=claim.supersedes,
            ))
        
        # Emit decision events for blocked claims
        for prop, decision in frame.blocked:
            self._logger.emit(self._DecisionEvent(
                turn_id=turn_id,
                claim_id=prop.id if hasattr(prop, 'id') else "",
                action="block",
                reason=decision.reason if hasattr(decision, 'reason') else str(decision),
                confidence_original=prop.confidence if hasattr(prop, 'confidence') else 0,
            ))
        
        # Emit refusal events for blocked claims
        for prop, decision in frame.blocked:
            self._logger.emit(self._RefusalEvent(
                turn_id=turn_id,
                claim_id=prop.id if hasattr(prop, 'id') else "",
                reason=decision.reason if hasattr(decision, 'reason') else str(decision),
                severity="hard",
                trigger=decision.action.name if hasattr(decision, 'action') and hasattr(decision.action, 'name') else "unknown",
            ))
        
        # Emit thermal event
        thermal = frame.thermal
        self._logger.emit(self._ThermalEvent(
            turn_id=turn_id,
            instability_score=thermal.instability,
            revision_velocity=thermal.revision_count / max(1, turn_id),
            thermal_state=thermal.regime,
            regime=thermal.regime,
            active_claims_count=thermal.total_commitments,
        ))
        
        return frame
    
    def govern(
        self,
        text: str,
        envelope: Optional[GenerationEnvelope] = None,
        anchors: Optional[List[Anchor]] = None,
    ) -> EpistemicFrame:
        """Govern pre-generated text with event logging."""
        import time
        
        turn_id = self._logger.next_turn()
        start_time = time.time()
        
        frame = self._session.govern(text, envelope=envelope, anchors=anchors)
        
        end_time = time.time()
        
        # Emit turn event (simplified for govern-only)
        self._logger.emit(self._TurnEvent(
            turn_id=turn_id,
            prompt="[governed text]",
            response=frame.output_text[:500] if frame.output_text else "",
            latency_ms=(end_time - start_time) * 1000,
            governance_ms=(end_time - start_time) * 1000,
            proposals_count=len(frame.committed) + len(frame.blocked) + len(frame.hedged),
            commits_count=len(frame.committed),
            hedges_count=len(frame.hedged),
            blocks_count=len(frame.blocked),
        ))
        
        # Emit thermal event
        thermal = frame.thermal
        self._logger.emit(self._ThermalEvent(
            turn_id=turn_id,
            instability_score=thermal.instability,
            thermal_state=thermal.regime,
            regime=thermal.regime,
            active_claims_count=thermal.total_commitments,
        ))
        
        return frame
    
    def get_events(self, event_type=None):
        """Get logged events."""
        return self._logger.get_events(event_type)
    
    def get_summary(self):
        """Get session summary statistics."""
        return self._logger.get_summary()
    
    def snapshot(self) -> LedgerSnapshot:
        """Get current ledger snapshot."""
        return self._session.snapshot()
    
    def strata(self, limit: int = 100) -> List[Stratum]:
        """Get commitment strata."""
        return self._session.strata(limit=limit)
    
    def diff(self, from_turn: int, to_turn: Optional[int] = None) -> LedgerDiff:
        """Get diff between turns."""
        return self._session.diff(from_turn, to_turn)
    
    def flush(self):
        """Flush event logs."""
        self._logger.flush()
    
    def close(self):
        """Close session and flush all logs."""
        self._logger.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=== Epistemic Session Demo ===\n")
    
    # Create session without provider (will use passthrough)
    session = EpistemicSession()
    
    # Govern some text
    frame = session.govern("Paris is the capital of France. The Eiffel Tower is 330 meters tall.")
    print(f"Governed {len(frame.committed)} claims")
    print(f"Thermal regime: {session.thermal.regime}")
    
    # Get snapshot
    snap = session.snapshot()
    print(f"\nSnapshot: {snap.active_claims} active claims, instability={snap.instability:.2f}")
    
    # View strata
    strata = session.strata()
    print(f"\nStrata ({len(strata)} layers):")
    for s in strata:
        print(f"  [{s.layer}] {s.text[:50]}...")
    
    # Test with echo provider
    print("\n--- With Echo Provider ---")
    session2 = EpistemicSession(provider=EchoProvider())
    frame2 = session2.step("What is the speed of light?")
    print(f"Output: {frame2.output_text}")
    
    print("\n✓ Session API working")
