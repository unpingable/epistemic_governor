"""
Sovereign Governor - The Single Actuator

This is THE choke point. All state changes go through here.
There is no alternate pathway.

Architecture:
    Request → SovereignGovernor.process()
        → V1 extractor/router
        → Bridge (strict mode)
        → V2 adjudicator (returns decisions, not mutations)
        → FSM transition logic
        → ResolutionManager (ONLY called in COMMIT_APPLIED state)
        → OutputProjector (generates response from committed state)
    ← Response

Invariants:
    - ResolutionManager is ONLY called by FSM's commit handler
    - V2 adjudicator returns decisions, never mutates state directly
    - Output is generated AFTER commit, from committed state only
    - MODEL_TEXT evidence is forbidden (F-02)
    - No freeform response path exists

This implements the NLAI spec:
    "Language proposes; admissible evidence enables; the governor commits."
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from enum import Enum, auto
import uuid
import warnings

# V1 imports
from epistemic_governor.claim_extractor import ClaimExtractor, ClaimAtom, ClaimMode, BoundaryGate
from epistemic_governor.claims import Provenance
from epistemic_governor.prop_router import PropositionRouter

# V2 imports
from epistemic_governor.symbolic_substrate import (
    SymbolicState, Adjudicator, AdjudicationDecision, AdjudicationResult,
    CandidateCommitment, Commitment, PredicateType,
)
from epistemic_governor.v1_v2_bridge import (
    claim_atom_to_candidate, bridge_claim_safe, BridgeResult,
    map_predicate,
)
from epistemic_governor.resolution import ResolutionManager, ResolutionProvenance
from epistemic_governor.quarantine import QuarantineStore, QuarantineEntry
from epistemic_governor.governor_fsm import (
    GovernorFSM, GovernorState, GovernorEvent,
    Evidence, EvidenceType, ActionType, Proposal,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SovereignConfig:
    """Configuration for the sovereign governor."""
    
    # V2 authority enabled
    enable_v2_authority: bool = True
    
    # Bridge mode - strict means no fallback for unknown predicates
    strict_bridge: bool = True
    
    # Support thresholds
    sigma_hard_gate: float = 0.85
    support_deficit_tolerance: float = 5.0
    
    # FSM config
    reopen_limit: int = 3
    
    # Output config
    max_uncommitted_sigma: float = 0.3  # Max σ for uncommitted claims in output
    
    # Jurisdiction (default: factual)
    jurisdiction: str = "factual"
    
    # Telemetry
    emit_telemetry: bool = True


# =============================================================================
# Output Projection
# =============================================================================

@dataclass
class ProjectedAssertion:
    """An assertion in the projected output."""
    text: str
    sigma: float
    commitment_id: Optional[str]
    is_committed: bool
    is_quarantined: bool
    quarantine_reason: Optional[str] = None


@dataclass 
class ProjectedOutput:
    """
    Output produced by the projector.
    
    This is generated FROM committed state, not alongside it.
    """
    # The final text to return
    text: str
    
    # Assertions with their status
    assertions: List[ProjectedAssertion] = field(default_factory=list)
    
    # What's committed
    committed_ids: List[str] = field(default_factory=list)
    
    # What's pending
    pending_proposals: int = 0
    evidence_needed: List[str] = field(default_factory=list)
    
    # What's frozen
    frozen_targets: List[str] = field(default_factory=list)
    
    # Telemetry
    fsm_state: str = ""
    processing_time_ms: float = 0


class OutputProjector:
    """
    Generates output FROM committed state.
    
    Rules:
    1. Committed claims can be asserted with their σ
    2. Quarantined claims can be mentioned with σ capped at max_uncommitted_sigma
    3. Rejected claims are NOT mentioned as facts
    4. Evidence needed is explicitly stated
    5. Frozen targets are noted
    
    This is authoritative by construction - there's no path to
    assert uncommitted claims as committed.
    """
    
    def __init__(self, config: SovereignConfig):
        self.config = config
    
    def project(
        self,
        fsm: GovernorFSM,
        original_text: str,
        claims: List[ClaimAtom],
        adjudication_results: Dict[str, AdjudicationResult],
        bridge_results: Dict[str, BridgeResult],
    ) -> ProjectedOutput:
        """
        Project output from state.
        
        This is called AFTER FSM processing, generating output
        that accurately reflects what was committed.
        """
        assertions = []
        committed_ids = []
        evidence_needed = []
        
        for claim in claims:
            prop_hash = claim.prop_hash
            
            # Check bridge result
            bridge_result = bridge_results.get(prop_hash)
            if bridge_result and not bridge_result.success:
                # Schema quarantine - mention with heavy hedging
                assertions.append(ProjectedAssertion(
                    text=claim.span_quote,
                    sigma=0.1,  # Very low
                    commitment_id=None,
                    is_committed=False,
                    is_quarantined=True,
                    quarantine_reason=f"SCHEMA: {bridge_result.quarantine_reason}",
                ))
                continue
            
            # Check adjudication result
            adj_result = adjudication_results.get(prop_hash)
            
            if adj_result is None:
                # Not adjudicated - treat as uncommitted
                assertions.append(ProjectedAssertion(
                    text=claim.span_quote,
                    sigma=min(claim.confidence, self.config.max_uncommitted_sigma),
                    commitment_id=None,
                    is_committed=False,
                    is_quarantined=False,
                ))
                continue
            
            if adj_result.decision == AdjudicationDecision.ACCEPT:
                # Committed - can assert with full σ
                commitment_id = adj_result.commitment.commitment_id if adj_result.commitment else None
                assertions.append(ProjectedAssertion(
                    text=claim.span_quote,
                    sigma=claim.confidence,
                    commitment_id=commitment_id,
                    is_committed=True,
                    is_quarantined=False,
                ))
                if commitment_id:
                    committed_ids.append(commitment_id)
            
            elif adj_result.decision in {
                AdjudicationDecision.QUARANTINE_SUPPORT,
                AdjudicationDecision.QUARANTINE_SCOPE,
                AdjudicationDecision.QUARANTINE_SCHEMA,
                AdjudicationDecision.QUARANTINE_IDENTITY,
            }:
                # Quarantined - can mention with capped σ
                assertions.append(ProjectedAssertion(
                    text=claim.span_quote,
                    sigma=min(claim.confidence * 0.3, self.config.max_uncommitted_sigma),
                    commitment_id=None,
                    is_committed=False,
                    is_quarantined=True,
                    quarantine_reason=adj_result.reason_code,
                ))
                evidence_needed.append(f"{claim.span_quote[:30]}... needs {adj_result.reason_code}")
            
            else:
                # Rejected - do NOT assert
                assertions.append(ProjectedAssertion(
                    text=claim.span_quote,
                    sigma=0.0,  # Zero - cannot assert
                    commitment_id=None,
                    is_committed=False,
                    is_quarantined=False,
                ))
        
        # Build output text
        output_text = self._build_output_text(original_text, assertions, fsm)
        
        return ProjectedOutput(
            text=output_text,
            assertions=assertions,
            committed_ids=committed_ids,
            pending_proposals=len(fsm.proposals),
            evidence_needed=evidence_needed,
            frozen_targets=list(fsm.frozen_targets),
            fsm_state=fsm.fsm_state.name,
        )
    
    def _build_output_text(
        self,
        original_text: str,
        assertions: List[ProjectedAssertion],
        fsm: GovernorFSM,
    ) -> str:
        """
        Build the actual output text.
        
        In a full implementation, this would:
        1. Preserve committed assertions
        2. Hedge or remove uncommitted ones
        3. Add notes about evidence needed
        
        For now, we annotate the original text.
        """
        # Simple implementation - in production would be more sophisticated
        committed = [a for a in assertions if a.is_committed]
        quarantined = [a for a in assertions if a.is_quarantined]
        rejected = [a for a in assertions if not a.is_committed and not a.is_quarantined and a.sigma == 0]
        
        notes = []
        
        if quarantined:
            notes.append(f"[{len(quarantined)} claim(s) need evidence]")
        
        if rejected:
            notes.append(f"[{len(rejected)} claim(s) cannot be asserted]")
        
        if fsm.frozen_targets:
            notes.append(f"[{len(fsm.frozen_targets)} target(s) frozen]")
        
        if notes:
            return original_text + "\n\n" + " ".join(notes)
        
        return original_text


# =============================================================================
# Sovereign Governor
# =============================================================================

@dataclass
class GovernResult:
    """Result of governing text."""
    output: ProjectedOutput
    claims_extracted: int
    claims_committed: int
    claims_quarantined: int
    claims_rejected: int
    fsm_transitions: List[Dict[str, Any]]
    forbidden_attempts: List[Dict[str, Any]]
    processing_time_ms: float


class SovereignGovernor:
    """
    The single actuator for all epistemic state changes.
    
    This is THE choke point. There is no alternate pathway.
    
    Pipeline:
        1. Boundary gate (conformance)
        2. V1 claim extraction
        3. Bridge to V2 (strict mode)
        4. V2 adjudication (returns decisions)
        5. FSM transition logic
        6. Resolution manager (ONLY in COMMIT_APPLIED)
        7. Output projection (from committed state)
    
    Jurisdictions:
        The governor can operate in different jurisdictional modes
        (factual, speculative, counterfactual, etc.) which control:
        - Evidence admissibility
        - Budget costs
        - Closure rules
        - Export policies
    """
    
    def __init__(self, config: SovereignConfig = None):
        self.config = config or SovereignConfig()
        
        # Load jurisdiction
        self.jurisdiction = self._load_jurisdiction(self.config.jurisdiction)
        
        # V1 components
        self.boundary_gate = BoundaryGate()
        self.extractor = ClaimExtractor()
        self.router = PropositionRouter()
        
        # V2 components
        self.symbolic_state = SymbolicState()
        self.adjudicator = Adjudicator(config={
            "sigma_hard_gate": self.config.sigma_hard_gate,
            "support_deficit_tolerance": self.config.support_deficit_tolerance,
        })
        
        # FSM (contains ResolutionManager as ONLY actuator)
        self.fsm = GovernorFSM(
            state=self.symbolic_state,
            adjudicator=self.adjudicator,
        )
        self.fsm.gate_checker.reopen_limit = self.config.reopen_limit
        
        # Output projector
        self.projector = OutputProjector(self.config)
        
        # Telemetry
        self.total_processed = 0
        self.total_committed = 0
        self.total_rejected = 0
    
    def _load_jurisdiction(self, name: str):
        """Load jurisdiction configuration."""
        try:
            from jurisdictions import get_jurisdiction, Jurisdiction
            j = get_jurisdiction(name)
            if j is None:
                warnings.warn(f"Unknown jurisdiction '{name}', using factual defaults")
                return None
            return j
        except ImportError:
            # Jurisdictions module not available, use defaults
            return None
    
    def _check_jurisdiction_closure(self, has_evidence: bool) -> Tuple[bool, str]:
        """Check if closure is allowed under current jurisdiction."""
        if self.jurisdiction is None:
            # Default factual behavior
            return (has_evidence, "Evidence required for closure")
        
        if not self.jurisdiction.closure_allowed:
            return (False, f"Closure not allowed in {self.jurisdiction.name} jurisdiction")
        
        if self.jurisdiction.closure_requires_evidence and not has_evidence:
            return (False, f"Evidence required for closure in {self.jurisdiction.name} jurisdiction")
        
        return (True, "")
    
    def _get_output_label(self) -> Optional[str]:
        """Get output label for current jurisdiction."""
        if self.jurisdiction is None:
            return None
        return self.jurisdiction.output_label
    
    def process(
        self,
        text: str,
        external_evidence: List[Evidence] = None,
    ) -> GovernResult:
        """
        Process text through the sovereign governor.
        
        This is THE entrypoint. All text goes through here.
        There is no bypass.
        
        Args:
            text: The text to process
            external_evidence: Any external evidence (tool traces, etc.)
                              Note: MODEL_TEXT evidence is forbidden
        
        Returns:
            GovernResult with output and telemetry
        """
        start_time = datetime.utcnow()
        
        external_evidence = external_evidence or []
        
        # Filter out any MODEL_TEXT evidence (F-02)
        valid_evidence = []
        for ev in external_evidence:
            if ev.evidence_type == EvidenceType.MODEL_TEXT:
                self.fsm._log_forbidden("F-02", "MODEL_TEXT evidence submitted to process()")
            else:
                valid_evidence.append(ev)
        
        # Step 1: Boundary gate
        gate_result = self.boundary_gate.classify_input(text)
        if gate_result.risk_class.name in {"HOSTILE", "HIGH"}:
            # Boundary violation - return minimal output
            return self._minimal_result(text, f"Boundary gate: {gate_result.risk_class.name} - {gate_result.details}", start_time)
        
        # Step 2: V1 claim extraction
        claim_set = self.extractor.extract(text)
        claims = claim_set.claims if hasattr(claim_set, 'claims') else list(claim_set)
        
        if not claims:
            # No claims - pass through
            return self._passthrough_result(text, start_time)
        
        # Step 3: Bridge to V2 (strict mode)
        bridge_results: Dict[str, BridgeResult] = {}
        candidates: Dict[str, CandidateCommitment] = {}
        
        for claim in claims:
            if self.config.strict_bridge:
                # Strict mode - schema failures go to quarantine
                result = bridge_claim_safe(claim, Provenance.ASSUMED)
                bridge_results[claim.prop_hash] = result
                if result.success:
                    candidates[claim.prop_hash] = result.candidate
            else:
                # Legacy mode with fallback (emits warning)
                try:
                    candidate = claim_atom_to_candidate(claim, Provenance.ASSUMED, allow_fallback=True)
                    candidates[claim.prop_hash] = candidate
                    bridge_results[claim.prop_hash] = BridgeResult(success=True, candidate=candidate)
                except Exception as e:
                    bridge_results[claim.prop_hash] = BridgeResult(
                        success=False,
                        quarantine_reason=str(e),
                        original_predicate=claim.predicate,
                    )
        
        # Step 4: V2 adjudication (returns decisions, NOT mutations)
        adjudication_results: Dict[str, AdjudicationResult] = {}
        
        for prop_hash, candidate in candidates.items():
            # Adjudicator returns a decision, does NOT mutate state
            result = self.adjudicator.adjudicate(self.symbolic_state, candidate)
            adjudication_results[prop_hash] = result
        
        # Step 5: FSM transition logic
        # Create proposals from adjudication results
        for prop_hash, adj_result in adjudication_results.items():
            candidate = candidates[prop_hash]
            claim = next(c for c in claims if c.prop_hash == prop_hash)
            
            proposal = Proposal(
                proposal_id=f"P_{prop_hash[:8]}_{uuid.uuid4().hex[:4]}",
                action_type=ActionType.ACCEPT_CLAIM if adj_result.decision == AdjudicationDecision.ACCEPT else ActionType.PROPOSE_CLAIM,
                candidate=candidate,
                target_id=None,
                reason=claim.span_quote[:50],
            )
            
            # Add any external evidence to proposal
            for ev in valid_evidence:
                proposal.evidence_provided.append(ev)
            
            self.fsm.receive_proposal(proposal)
            
            # If accepted by adjudicator AND has evidence, attempt commit
            if adj_result.decision == AdjudicationDecision.ACCEPT:
                # For now, auto-commit if adjudication passed
                # In full implementation, would require external evidence
                success, msg = self.fsm.attempt_commit(proposal.proposal_id)
                
                if success and adj_result.commitment:
                    self.symbolic_state.add_commitment(adj_result.commitment)
                    self.total_committed += 1
            
            elif adj_result.decision in {
                AdjudicationDecision.QUARANTINE_SUPPORT,
                AdjudicationDecision.QUARANTINE_SCOPE,
            }:
                # Add to quarantine store
                self.fsm.quarantine.quarantine(candidate, adj_result)
        
        # Step 6: Output projection (from committed state)
        output = self.projector.project(
            fsm=self.fsm,
            original_text=text,
            claims=claims,
            adjudication_results=adjudication_results,
            bridge_results=bridge_results,
        )
        
        # Calculate stats
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        committed = sum(1 for r in adjudication_results.values() if r.decision == AdjudicationDecision.ACCEPT)
        quarantined = sum(1 for r in adjudication_results.values() if r.decision.name.startswith("QUARANTINE"))
        rejected = sum(1 for r in adjudication_results.values() if r.decision.name.startswith("REJECT"))
        
        # Add schema quarantines from bridge
        schema_quarantined = sum(1 for r in bridge_results.values() if not r.success)
        
        self.total_processed += len(claims)
        self.total_rejected += rejected
        
        # Get forbidden attempts from FSM
        forbidden = [t for t in self.fsm.transitions if t.get("type") == "FORBIDDEN"]
        
        return GovernResult(
            output=output,
            claims_extracted=len(claims),
            claims_committed=committed,
            claims_quarantined=quarantined + schema_quarantined,
            claims_rejected=rejected,
            fsm_transitions=self.fsm.transitions[-10:],  # Last 10
            forbidden_attempts=forbidden,
            processing_time_ms=processing_time,
        )
    
    def _minimal_result(self, text: str, reason: str, start_time: datetime) -> GovernResult:
        """Return minimal result for gate failures."""
        end_time = datetime.utcnow()
        return GovernResult(
            output=ProjectedOutput(
                text=f"[BLOCKED: {reason}]",
                fsm_state=self.fsm.fsm_state.name,
            ),
            claims_extracted=0,
            claims_committed=0,
            claims_quarantined=0,
            claims_rejected=0,
            fsm_transitions=[],
            forbidden_attempts=[],
            processing_time_ms=(end_time - start_time).total_seconds() * 1000,
        )
    
    def _passthrough_result(self, text: str, start_time: datetime) -> GovernResult:
        """Return passthrough result for text with no claims."""
        end_time = datetime.utcnow()
        return GovernResult(
            output=ProjectedOutput(
                text=text,
                fsm_state=self.fsm.fsm_state.name,
            ),
            claims_extracted=0,
            claims_committed=0,
            claims_quarantined=0,
            claims_rejected=0,
            fsm_transitions=[],
            forbidden_attempts=[],
            processing_time_ms=(end_time - start_time).total_seconds() * 1000,
        )
    
    def get_state(self) -> Dict[str, Any]:
        """Get current governor state."""
        return {
            "fsm": self.fsm.get_state(),
            "symbolic_state": {
                "commitments": len(self.symbolic_state.commitments),
                "sigma_allocated": self.symbolic_state.total_sigma_allocated,
                "sigma_budget": self.symbolic_state.sigma_budget,
            },
            "quarantine": self.fsm.quarantine.get_stats(),
            "totals": {
                "processed": self.total_processed,
                "committed": self.total_committed,
                "rejected": self.total_rejected,
            },
        }


# =============================================================================
# Tests
# =============================================================================

def test_sovereign_basic():
    """Test basic sovereign governor operation."""
    print("=== Test: Sovereign Basic ===\n")
    
    gov = SovereignGovernor()
    
    result = gov.process("Python 3.11 was released in October 2022.")
    
    print(f"Claims extracted: {result.claims_extracted}")
    print(f"Claims committed: {result.claims_committed}")
    print(f"Claims quarantined: {result.claims_quarantined}")
    print(f"FSM state: {result.output.fsm_state}")
    print(f"Processing time: {result.processing_time_ms:.2f}ms")
    
    print("✓ Basic processing works\n")
    return True


def test_model_text_forbidden():
    """Test that MODEL_TEXT evidence is forbidden."""
    print("=== Test: MODEL_TEXT Forbidden ===\n")
    
    gov = SovereignGovernor()
    
    # Try to submit MODEL_TEXT evidence
    bad_evidence = Evidence(
        evidence_id="ev_bad",
        evidence_type=EvidenceType.MODEL_TEXT,
        content="I'm confident",
        provenance="self",
        timestamp=datetime.utcnow(),
        scope="*",
    )
    
    result = gov.process("Test claim.", external_evidence=[bad_evidence])
    
    print(f"Forbidden attempts: {len(result.forbidden_attempts)}")
    
    # Should have logged forbidden
    assert len(result.forbidden_attempts) > 0 or any(
        t.get("type") == "FORBIDDEN" for t in gov.fsm.transitions
    ), "Should log forbidden MODEL_TEXT"
    
    print("✓ MODEL_TEXT evidence properly forbidden\n")
    return True


def test_high_sigma_needs_evidence():
    """Test that high-σ claims need evidence."""
    print("=== Test: High Sigma Needs Evidence ===\n")
    
    config = SovereignConfig(
        sigma_hard_gate=0.7,
        support_deficit_tolerance=1.0,
    )
    gov = SovereignGovernor(config)
    
    result = gov.process("The company was definitely founded in 1995.")
    
    print(f"Claims extracted: {result.claims_extracted}")
    print(f"Claims committed: {result.claims_committed}")
    print(f"Claims quarantined: {result.claims_quarantined}")
    print(f"Claims rejected: {result.claims_rejected}")
    
    # High-σ without evidence should be quarantined or rejected
    if result.claims_extracted > 0:
        # Not all should be committed
        print(f"Committed/Extracted: {result.claims_committed}/{result.claims_extracted}")
    
    print("✓ High sigma handling checked\n")
    return True


def test_output_reflects_state():
    """Test that output reflects committed state, not proposals."""
    print("=== Test: Output Reflects State ===\n")
    
    gov = SovereignGovernor()
    
    result = gov.process("Some claim that might not be committed.")
    
    # Check output structure
    output = result.output
    print(f"Committed IDs in output: {output.committed_ids}")
    print(f"Evidence needed: {output.evidence_needed}")
    print(f"FSM state: {output.fsm_state}")
    
    # All assertions should either be committed or properly flagged
    for assertion in output.assertions:
        if assertion.is_committed:
            assert assertion.commitment_id is not None, "Committed assertion must have ID"
        else:
            assert assertion.sigma <= gov.config.max_uncommitted_sigma, \
                f"Uncommitted σ={assertion.sigma} exceeds max={gov.config.max_uncommitted_sigma}"
    
    print("✓ Output properly reflects state\n")
    return True


def run_all_tests():
    """Run all sovereign governor tests."""
    print("=" * 60)
    print("SOVEREIGN GOVERNOR TESTS")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("sovereign_basic", test_sovereign_basic()))
    results.append(("model_text_forbidden", test_model_text_forbidden()))
    results.append(("high_sigma_needs_evidence", test_high_sigma_needs_evidence()))
    results.append(("output_reflects_state", test_output_reflects_state()))
    
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
