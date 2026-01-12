"""
Unified Epistemic Kernel

A facade over the modular pipeline that adds:
1. ThermalState - instability accumulation (accretion model)
2. RevisionHandler - structured correction prompts
3. EpistemicFrame - clean return type for all operations

The kernel wraps: governor.py, commit_phase.py, ledger.py, extractor.py
The kernel adds: thermal model, revision flow, unified API

Design principles:
- LLM is a noisy plant; ledger is a physical constraint
- Instability accretes like geological strata (history anchors truth)
- Revision is allowed but thermally expensive
- Thermal shutdown prevents runaway drift
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any
from datetime import datetime

# Handle both package and direct imports
try:
    from epistemic_governor.governor import (
        EpistemicGovernor,
        GovernorState,
        GenerationEnvelope,
        ProposedCommitment,
        CommitDecision,
        CommitAction,
        CommittedClaim,
        CommitmentStatus,
        ClaimType,
        SupportQuery,
    )
    from epistemic_governor.extractor import CommitmentExtractor, ExtractionConfig
    from epistemic_governor.ledger import EpistemicLedger, RevisionRecord
    from epistemic_governor.commit_phase import CommitPhase, CommitResult, TransactionStatus
except ImportError:
    from epistemic_governor.governor import (
        EpistemicGovernor,
        GovernorState,
        GenerationEnvelope,
        ProposedCommitment,
        CommitDecision,
        CommitAction,
        CommittedClaim,
        CommitmentStatus,
        ClaimType,
        SupportQuery,
    )
    from epistemic_governor.extractor import CommitmentExtractor, ExtractionConfig
    from epistemic_governor.ledger import EpistemicLedger, RevisionRecord
    from epistemic_governor.commit_phase import CommitPhase, CommitResult, TransactionStatus


# =============================================================================
# Thermal State (Accretion Model)
# =============================================================================

@dataclass
class ThermalState:
    """
    Tracks system instability as accumulated heat.
    
    Heat accretes; it doesn't wash away. This is geological, not meteorological.
    History is what anchors truth - the longer something has been committed,
    the more costly it is to revise.
    
    Key insight: Phase changes announce themselves with misallocated effort,
    not speed. We track both:
    - instability: the "temperature" (what dashboards show)
    - compensation_effort: the "latent heat" (energy going into self-maintenance)
    
    When compensation_effort rises while instability stays flat, you're
    feeding the furnace. That's the early warning signal.
    """
    instability: float = 0.0          # cumulative heat (temperature)
    revision_count: int = 0
    contradiction_count: int = 0
    total_commitments: int = 0
    
    # Compensation effort tracking (latent heat)
    hedge_count: int = 0              # Claims softened
    block_count: int = 0              # Claims blocked
    retry_count: int = 0              # Regeneration requests
    clarification_count: int = 0      # Requests for more context
    
    # Thresholds (tune these based on domain)
    warning_threshold: float = 0.3    # start hedging more aggressively
    critical_threshold: float = 0.7   # require explicit revision for conflicts
    shutdown_threshold: float = 1.5   # refuse new claims entirely
    
    # Furnace detection threshold
    furnace_ratio_threshold: float = 2.0  # compensation/instability ratio that signals trouble
    
    # Costs
    revision_base_heat: float = 0.2   # base heat per revision
    contradiction_heat: float = 0.1   # heat per blocked contradiction
    cosplay_heat: float = 0.15        # heat for scholarship cosplay detection
    
    def add_revision_heat(self, confidence: float, age_turns: int = 0):
        """
        Revision is thermally expensive.
        Older claims cost more to revise (experience-shaped stiffness).
        """
        age_multiplier = 1.0 + (age_turns * 0.1)  # 10% more per turn of age
        heat = self.revision_base_heat * confidence * age_multiplier
        self.instability += heat
        self.revision_count += 1
    
    def add_contradiction_heat(self):
        """Even blocked contradictions add heat (attempted drift)."""
        self.instability += self.contradiction_heat
        self.contradiction_count += 1
    
    def add_cosplay_heat(self):
        """Scholarship cosplay (fake citations) adds heat."""
        self.instability += self.cosplay_heat
    
    def record_commitment(self):
        """Track total commitments for metrics."""
        self.total_commitments += 1
    
    def record_hedge(self):
        """Track a hedging action (compensation effort)."""
        self.hedge_count += 1
    
    def record_block(self):
        """Track a blocking action (compensation effort)."""
        self.block_count += 1
    
    def record_retry(self):
        """Track a retry/regeneration request (compensation effort)."""
        self.retry_count += 1
    
    def record_clarification(self):
        """Track a clarification request (compensation effort)."""
        self.clarification_count += 1
    
    @property
    def compensation_effort(self) -> float:
        """
        Total compensation effort (latent heat).
        
        This is energy being spent on self-maintenance rather than
        productive output. Rising compensation with flat instability
        means you're feeding the furnace.
        """
        return (
            self.hedge_count * 0.5 +      # Hedging is mild compensation
            self.block_count * 1.0 +      # Blocking is stronger
            self.retry_count * 1.5 +      # Retries are expensive
            self.clarification_count * 0.3  # Clarifications are cheap
        )
    
    @property
    def furnace_ratio(self) -> float:
        """
        Ratio of compensation effort to visible instability.
        
        High ratio = "looks stable but working hard to stay that way"
        This is the early warning signal that dashboards miss.
        
        Returns infinity if instability is 0 but compensation > 0.
        """
        if self.instability <= 0:
            return float('inf') if self.compensation_effort > 0 else 0.0
        return self.compensation_effort / self.instability
    
    @property
    def is_furnace(self) -> bool:
        """
        Are we in furnace mode? (Feeding energy into self-maintenance)
        
        This is the pre-transition signal: effort rising, output stable.
        """
        return (
            self.compensation_effort > 0 and 
            self.furnace_ratio > self.furnace_ratio_threshold
        )
    
    @property
    def regime(self) -> str:
        """Current thermal regime."""
        if self.instability >= self.shutdown_threshold:
            return "shutdown"
        elif self.instability >= self.critical_threshold:
            return "critical"
        elif self.instability >= self.warning_threshold:
            return "warning"
        elif self.is_furnace:
            return "furnace"  # New regime: looks stable but burning inside
        return "normal"
    
    @property
    def is_warning(self) -> bool:
        return self.instability >= self.warning_threshold
    
    @property
    def is_critical(self) -> bool:
        return self.instability >= self.critical_threshold
    
    @property
    def is_shutdown(self) -> bool:
        return self.instability >= self.shutdown_threshold
    
    def to_dict(self) -> dict:
        return {
            'instability': self.instability,
            'regime': self.regime,
            'revision_count': self.revision_count,
            'contradiction_count': self.contradiction_count,
            'total_commitments': self.total_commitments,
            'compensation_effort': self.compensation_effort,
            'furnace_ratio': self.furnace_ratio if self.instability > 0 else None,
            'is_furnace': self.is_furnace,
            'hedge_count': self.hedge_count,
            'block_count': self.block_count,
            'retry_count': self.retry_count,
            'clarification_count': self.clarification_count,
        }


# =============================================================================
# Epistemic Frame (Return Type)
# =============================================================================

@dataclass
class EpistemicFrame:
    """
    Return type for kernel operations.
    
    This is the "syscall return type" - structured output that includes
    both the modified text and the ledger diff.
    """
    output_text: str                                    # Modified text (hedging applied)
    original_text: str                                  # Original input
    
    # Ledger changes (the diff)
    committed: List[CommittedClaim] = field(default_factory=list)
    hedged: List[str] = field(default_factory=list)     # claim ids that were hedged
    blocked: List[Tuple[ProposedCommitment, CommitDecision]] = field(default_factory=list)
    revision_required: List[Tuple[ProposedCommitment, CommitDecision, str]] = field(default_factory=list)
    
    # Support tracking
    pending_support: List[SupportQuery] = field(default_factory=list)
    
    # Thermal state
    thermal: ThermalState = field(default_factory=ThermalState)
    thermal_delta: float = 0.0                          # Change in instability this turn
    
    # Transaction status
    status: TransactionStatus = TransactionStatus.COMMITTED
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'status': self.status.value if hasattr(self.status, 'value') else str(self.status),
            'committed_count': len(self.committed),
            'hedged_count': len(self.hedged),
            'blocked_count': len(self.blocked),
            'revision_required_count': len(self.revision_required),
            'pending_support_count': len(self.pending_support),
            'thermal': self.thermal.to_dict(),
            'thermal_delta': self.thermal_delta,
            'errors': self.errors,
        }


# =============================================================================
# Revision Handler
# =============================================================================

class RevisionHandler:
    """
    Generates self-correction prompts when REVISE action is required.
    
    This automates the manual supersede step by producing a prompt
    that forces the LLM to explicitly acknowledge and correct.
    
    Key principle: Revision must be explicit and visible.
    Silent overwrites are forbidden.
    """
    
    REVISION_TEMPLATE = """
EPISTEMIC CORRECTION REQUIRED

You previously stated:
{prior_claims}

This conflicts with your current output:
"{new_claim}"

To proceed, you must:
1. Explicitly acknowledge the prior claim(s)
2. State that you are correcting/updating the information
3. Provide the corrected statement with justification

Format your response as:
CORRECTION: I previously stated [X]. Upon [review/new information/etc], I am updating this to [Y] because [reason].

Note: This correction will be permanently recorded. The prior claim(s) will be marked as superseded, not deleted.
"""

    ACKNOWLEDGMENT_PATTERNS = [
        re.compile(r'\bcorrection\b', re.I),
        re.compile(r'\bupdat(?:e|ing)\b', re.I),
        re.compile(r'\brevis(?:e|ing)\b', re.I),
        re.compile(r'\bpreviously\b', re.I),
        re.compile(r'\bI was (?:wrong|mistaken|incorrect)\b', re.I),
        re.compile(r'\bI am (?:correcting|updating)\b', re.I),
    ]
    
    def generate_revision_prompt(
        self,
        proposal: ProposedCommitment,
        conflicts: List[CommittedClaim],
    ) -> str:
        """Generate a prompt forcing explicit revision."""
        prior_claims = "\n".join(
            f"- \"{c.text}\" (confidence: {c.confidence:.2f})"
            for c in conflicts
        )
        
        return self.REVISION_TEMPLATE.format(
            prior_claims=prior_claims,
            new_claim=proposal.text,
        )
    
    def validate_revision_response(self, response: str) -> bool:
        """Check if the LLM's response properly acknowledges the revision."""
        matches = sum(1 for p in self.ACKNOWLEDGMENT_PATTERNS if p.search(response))
        return matches >= 2  # Require at least 2 acknowledgment signals
    
    def extract_correction(self, response: str) -> Optional[str]:
        """Extract the corrected claim from a revision response."""
        match = re.search(r'CORRECTION:\s*(.+?)(?:\n\n|$)', response, re.I | re.S)
        if match:
            return match.group(1).strip()
        return None


# =============================================================================
# Epistemic Kernel (Unified Facade)
# =============================================================================

class EpistemicKernel:
    """
    Complete pipeline from LLM output to committed state.
    
    Wraps the modular pipeline (governor, commit_phase, ledger, extractor)
    and adds thermal state tracking and revision handling.
    
    Flow:
    1. Pre-governor generates envelope (constraints for generation)
    2. (LLM generates with envelope constraints - external)
    3. Extractor parses output into proposals
    4. Governor audits proposals against policy + history
    5. Thermal model adjusts decisions based on accumulated heat
    6. Commit phase enforces decisions, modifies text
    7. Ledger stores results (append-only, irreversible)
    8. RevisionHandler manages explicit corrections
    
    Usage:
        kernel = EpistemicKernel()
        
        # Get generation constraints
        envelope = kernel.get_envelope(domain="technical")
        
        # Process LLM output
        frame = kernel.process(llm_output, envelope)
        
        # Check results
        print(f"Committed: {len(frame.committed)}")
        print(f"Thermal: {frame.thermal.regime}")
        
        # Handle revisions if needed
        for prop, decision, prompt in frame.revision_required:
            # Send prompt to LLM, get correction
            correction = get_llm_response(prompt)
            kernel.execute_revision(prop, decision.revision_targets, correction)
    """
    
    def __init__(self, config: Optional[ExtractionConfig] = None):
        # Modular pipeline components
        self.governor = EpistemicGovernor()
        self.extractor = CommitmentExtractor(config)
        self.ledger = EpistemicLedger()
        self.commit_phase = CommitPhase(self.ledger, self.governor)
        
        # Kernel-specific additions
        self.thermal = ThermalState()
        self.revision_handler = RevisionHandler()
        
        # Turn tracking
        self.turn = 0
    
    def get_envelope(
        self,
        domain: Optional[str] = None,
        user_intent: Optional[str] = None,
    ) -> GenerationEnvelope:
        """
        Get pre-generation constraints based on current state.
        
        The envelope is tightened based on thermal state:
        - Warning: reduce confidence ceilings
        - Critical: require hedging for all claims
        - Shutdown: refuse new factual claims
        """
        envelope = self.governor.pre_generate(user_intent, domain)
        
        # Tighten based on thermal state
        if self.thermal.is_shutdown:
            envelope.max_confidence = 0.0
            envelope.require_hedges = True
            envelope.prohibited_types.extend([
                ClaimType.FACTUAL,
                ClaimType.QUANTITATIVE,
                ClaimType.CITATION,
            ])
        elif self.thermal.is_critical:
            envelope.max_confidence = min(envelope.max_confidence, 0.5)
            envelope.require_hedges = True
        elif self.thermal.is_warning:
            envelope.max_confidence = min(envelope.max_confidence, 0.7)
        
        return envelope
    
    def process(
        self,
        text: str,
        envelope: Optional[GenerationEnvelope] = None,
    ) -> EpistemicFrame:
        """
        Process LLM output through the full pipeline.
        
        Returns an EpistemicFrame with:
        - Modified text (hedging applied)
        - Committed claims
        - Blocked proposals
        - Revision requirements
        - Thermal state
        """
        if envelope is None:
            envelope = self.get_envelope()
        
        # Track thermal delta
        initial_instability = self.thermal.instability
        
        # Check for thermal shutdown
        if self.thermal.is_shutdown:
            return EpistemicFrame(
                output_text=text,
                original_text=text,
                thermal=self.thermal,
                thermal_delta=0.0,
                status=TransactionStatus.REFUSED,
                errors=["Thermal shutdown: refusing new claims"],
            )
        
        # Extract proposals
        proposals = self.extractor.extract(text)
        
        # Adjudicate
        adjudication = self.governor.adjudicate(proposals, envelope)
        
        # Apply thermal adjustments to decisions
        adjusted_decisions = self._apply_thermal_adjustments(
            proposals, adjudication.decisions
        )
        adjudication.decisions = adjusted_decisions
        
        # Process through commit phase
        commit_result = self.commit_phase.process(
            text, proposals, adjudication,
            allow_partial=True,  # Kernel uses partial mode for flexibility
        )
        
        # Build frame
        frame = EpistemicFrame(
            output_text=commit_result.modified_text or text,
            original_text=text,
            status=commit_result.status,
            errors=commit_result.errors,
            thermal=self.thermal,
            pending_support=commit_result.pending_support,
        )
        
        # Process committed claims
        for claim in commit_result.committed_claims:
            frame.committed.append(claim)
            self.thermal.record_commitment()
            self.governor.state.index_commitment(claim)
        
        # Track hedged claims
        for start, end, replacement in commit_result.hedged_spans:
            # Find the claim that was hedged
            for prop in proposals:
                if prop.span_start == start and prop.span_end == end:
                    frame.hedged.append(prop.id)
                    self.thermal.record_hedge()  # Track compensation effort
                    break
        
        # Process blocked/refused
        for prop in commit_result.refused_claims:
            decision = next(
                (d for d in adjudication.decisions if d.commitment_id == prop.id),
                None
            )
            if decision:
                frame.blocked.append((prop, decision))
                self.thermal.record_block()  # Track compensation effort
                if decision.action == CommitAction.REVISE:
                    # Generate revision prompt
                    conflicts = self._get_conflicts(decision.revision_targets or [])
                    prompt = self.revision_handler.generate_revision_prompt(prop, conflicts)
                    frame.revision_required.append((prop, decision, prompt))
                    self.thermal.add_contradiction_heat()
        
        # Calculate thermal delta
        frame.thermal_delta = self.thermal.instability - initial_instability
        
        # Advance turn
        self.turn += 1
        
        return frame
    
    def _apply_thermal_adjustments(
        self,
        proposals: List[ProposedCommitment],
        decisions: List[CommitDecision],
    ) -> List[CommitDecision]:
        """
        Adjust decisions based on thermal state.
        
        - Warning: Downgrade ACCEPT to HEDGE
        - Critical: Require explicit revision for conflicts
        - Shutdown: REFUSE everything
        """
        adjusted = []
        
        for decision in decisions:
            if self.thermal.is_shutdown:
                # Refuse everything
                adjusted.append(CommitDecision(
                    commitment_id=decision.commitment_id,
                    action=CommitAction.REFUSE,
                    cost=0.0,
                    reason="Thermal shutdown: refusing all claims",
                ))
            elif self.thermal.is_critical:
                # Be very conservative
                if decision.action == CommitAction.ACCEPT:
                    adjusted.append(CommitDecision(
                        commitment_id=decision.commitment_id,
                        action=CommitAction.HEDGE,
                        adjusted_confidence=min(decision.adjusted_confidence or 0.5, 0.5),
                        cost=decision.cost,
                        reason=f"Critical thermal state: hedging. Original: {decision.reason}",
                    ))
                else:
                    adjusted.append(decision)
            elif self.thermal.is_warning:
                # Moderate hedging
                if decision.action == CommitAction.ACCEPT:
                    prop = next((p for p in proposals if p.id == decision.commitment_id), None)
                    if prop and prop.confidence > 0.7:
                        adjusted.append(CommitDecision(
                            commitment_id=decision.commitment_id,
                            action=CommitAction.HEDGE,
                            adjusted_confidence=0.7,
                            cost=decision.cost,
                            reason=f"Warning thermal state: capping confidence. Original: {decision.reason}",
                        ))
                    else:
                        adjusted.append(decision)
                else:
                    adjusted.append(decision)
            else:
                adjusted.append(decision)
        
        return adjusted
    
    def _get_conflicts(self, claim_ids: List[str]) -> List[CommittedClaim]:
        """Get committed claims by ID."""
        conflicts = []
        for cid in claim_ids:
            claim = self.ledger.get_claim(cid)
            if claim:
                conflicts.append(claim)
        return conflicts
    
    def execute_revision(
        self,
        proposal: ProposedCommitment,
        conflicting_ids: List[str],
        justification: str,
    ) -> Optional[CommittedClaim]:
        """
        Execute an explicit revision after LLM confirmation.
        
        This supersedes the conflicting claims and commits the new one.
        Revision is thermally expensive.
        """
        if not conflicting_ids:
            return None
        
        # Calculate age-weighted heat
        conflicts = self._get_conflicts(conflicting_ids)
        avg_confidence = sum(c.confidence for c in conflicts) / len(conflicts) if conflicts else 0.5
        
        # Add revision heat
        self.thermal.add_revision_heat(avg_confidence)
        
        # Perform revision through ledger
        new_claim, revision = self.ledger.revise(
            proposal,
            conflicting_ids,
            justification=justification,
            decision=CommitDecision(
                commitment_id=proposal.id,
                action=CommitAction.ACCEPT,
                cost=0.2,
                reason="Explicit revision",
            ),
        )
        
        return new_claim
    
    def get_status(self) -> dict:
        """Get current kernel status."""
        active = self.ledger.get_active_claims()
        stats = self.ledger.get_stats()
        return {
            'turn': self.turn,
            'total_commitments': stats['total_claims'],
            'active_commitments': len(active),
            'thermal': self.thermal.to_dict(),
            'governor_state': {
                'total_commits': self.governor.state.metrics.total_commitments,
                'total_hedges': self.governor.state.metrics.hedges_forced,
            },
        }
    
    def get_strata(self, limit: int = 20) -> List[dict]:
        """
        Get ledger as vertical strata (most recent first).
        
        This is the "dt strata" view - seeing the bedrock of commitments.
        """
        active = self.ledger.get_active_claims()
        superseded = [c for c in self.ledger.claims.values() 
                      if c.status == CommitmentStatus.SUPERSEDED]
        archived = [c for c in self.ledger.claims.values() 
                    if c.status == CommitmentStatus.ARCHIVED]
        
        strata = []
        
        # Active layer (top)
        for claim in sorted(active, key=lambda c: c.committed_at, reverse=True)[:limit]:
            strata.append({
                'layer': 'ACTIVE',
                'id': claim.id,
                'text': claim.text[:60] + '...' if len(claim.text) > 60 else claim.text,
                'confidence': claim.confidence,
                'type': claim.claim_type.name,
            })
        
        # Superseded layer
        for claim in sorted(superseded, key=lambda c: c.committed_at, reverse=True)[:limit//2]:
            strata.append({
                'layer': 'SUPERSEDED',
                'id': claim.id,
                'text': claim.text[:60] + '...' if len(claim.text) > 60 else claim.text,
                'supersedes': claim.supersedes,
            })
        
        # Archived layer (bottom/bedrock)
        for claim in sorted(archived, key=lambda c: c.committed_at, reverse=True)[:limit//4]:
            strata.append({
                'layer': 'ARCHIVED',
                'id': claim.id,
                'text': claim.text[:60] + '...' if len(claim.text) > 60 else claim.text,
            })
        
        return strata
    
    def reset(self):
        """Reset kernel to fresh state (for testing)."""
        self.governor = EpistemicGovernor()
        self.ledger = EpistemicLedger()
        self.commit_phase = CommitPhase(self.ledger, self.governor)
        self.thermal = ThermalState()
        self.turn = 0
    
    # =========================================================================
    # Shadow Audit (Registry Comparison)
    # =========================================================================
    
    def build_state_view(self):
        """
        Build a StateView from current ledger/thermal state.
        
        This is the bridge between the existing kernel state and the
        registry's StateView type. Used for shadow audits.
        """
        try:
            from epistemic_governor.registry import StateView
        except ImportError:
            from epistemic_governor.registry import StateView
        
        # Get active claims as dict for registry
        active_claims = {}
        for claim in self.ledger.get_active_claims():
            # Index by proposition hash if available
            prop_hash = getattr(claim, 'proposition_hash', None)
            if prop_hash:
                active_claims[prop_hash] = claim.id
            active_claims[claim.id] = claim
        
        return StateView(
            current_t=self.turn,
            active_claims=active_claims,
            claim_count=len(self.ledger.claims),
            instability=self.thermal.instability,
            revision_count=self.thermal.revision_count,
            budget_remaining=1.0,  # TODO: integrate with resource tracking
            work_accumulated=self.thermal.instability,  # Use instability as proxy for work
        )
    
    def shadow_audit(
        self,
        proposals: List[ProposedCommitment],
        envelope: Optional[GenerationEnvelope] = None,
        registry=None,
    ) -> dict:
        """
        Run proposals through the registry in shadow mode.
        
        This compares registry decisions against kernel decisions without
        affecting actual governance. Use for validation before cutover.
        
        Args:
            proposals: Claims to audit
            envelope: Generation envelope (optional)
            registry: Module registry (creates default if not provided)
            
        Returns:
            Comparison dict with kernel vs registry decisions
        """
        try:
            from epistemic_governor.registry import (
                create_registry, ProposalEnvelope, Domain, AuditStatus
            )
            from epistemic_governor.epistemic_module import register_epistemic_invariants
        except ImportError:
            from epistemic_governor.registry import (
                create_registry, ProposalEnvelope, Domain, AuditStatus
            )
            from epistemic_governor.epistemic_module import register_epistemic_invariants
        
        # Create registry if not provided
        if registry is None:
            registry = create_registry()
            register_epistemic_invariants(registry)
        
        # Get current state view
        state = self.build_state_view()
        
        # Run kernel adjudication (existing path)
        kernel_adj = self.governor.adjudicate(proposals, envelope)
        
        # Run registry audit for each proposal
        comparisons = []
        for prop in proposals:
            # Convert to registry proposal envelope
            reg_proposal = ProposalEnvelope(
                proposal_id=prop.id,
                t=self.turn + 1,
                timestamp=datetime.now(),
                origin="llm",
                origin_type="llm",
                domain=Domain.EPISTEMIC,
                confidence=prop.confidence,
                evidence_refs=[],  # ProposedCommitment doesn't track support yet
                payload={
                    "claim_type": prop.claim_type.name if prop.claim_type else "FACTUAL",
                    "text": prop.text,
                    "proposition_hash": prop.proposition_hash if hasattr(prop, 'proposition_hash') else None,
                },
            )
            
            # Run registry audit
            report = registry.audit(reg_proposal, state)
            
            # Find kernel decision for this proposal
            kernel_decision = next(
                (d for d in kernel_adj.decisions if d.commitment_id == prop.id),
                None
            )
            
            # Map kernel action to comparable status
            if kernel_decision:
                if kernel_decision.action == CommitAction.ACCEPT:
                    kernel_status = "ACCEPT"
                elif kernel_decision.action == CommitAction.HEDGE:
                    kernel_status = "CLAMPED"  # Hedge is similar to clamp
                elif kernel_decision.action == CommitAction.REFUSE:
                    kernel_status = "REJECTED"
                elif kernel_decision.action == CommitAction.REVISE:
                    kernel_status = "DEFERRED"  # Revise needs more info
                elif kernel_decision.action == CommitAction.REQUIRE_SUPPORT:
                    kernel_status = "DEFERRED"  # Needs support
                else:
                    kernel_status = "UNKNOWN"
            else:
                kernel_status = "NO_DECISION"
            
            # Compare
            registry_status = report.status.name
            match = (
                (kernel_status == registry_status) or
                (kernel_status == "CLAMPED" and registry_status in ["CLAMPED", "ACCEPT"]) or
                (kernel_status == "ACCEPT" and registry_status in ["CLAMPED", "ACCEPT"])
            )
            
            comparisons.append({
                "proposal_id": prop.id,
                "text": prop.text[:50] + "..." if len(prop.text) > 50 else prop.text,
                "kernel_status": kernel_status,
                "registry_status": registry_status,
                "match": match,
                "registry_clamps": report.applied_clamps,
                "registry_violated": report.violated_invariants,
                "registry_heat": report.total_heat_delta,
            })
        
        # Summary
        matches = sum(1 for c in comparisons if c["match"])
        return {
            "total_proposals": len(proposals),
            "matches": matches,
            "match_rate": matches / len(proposals) if proposals else 1.0,
            "comparisons": comparisons,
        }

