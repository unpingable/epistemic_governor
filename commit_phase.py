"""
Commit Phase Module

This is the critical primitive - the transaction boundary.

Function:
- Validate proposed commitments against policy, history, and support requirements
- Apply governor decisions to transform or reject commitments
- Atomically commit valid claims to the ledger

Properties:
- Atomic: all commitments succeed or none do (per batch)
- Pre-exposure: text is modified before user sees it
- State-aware: decisions depend on history

Analogy: Database transaction + constraints + WAL semantics

Nothing becomes true unless it passes here.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
from datetime import datetime
from enum import Enum, auto

# Handle both package and direct imports
try:
    from .governor import (
        EpistemicGovernor,
        GovernorState,
        GenerationEnvelope,
        AdjudicationResult,
        CommitDecision,
        CommitAction,
        ProposedCommitment,
        CommittedClaim,
        SupportQuery,
    )
    from .extractor import CommitmentExtractor
    from .ledger import EpistemicLedger
except ImportError:
    from governor import (
        EpistemicGovernor,
        GovernorState,
        GenerationEnvelope,
        AdjudicationResult,
        CommitDecision,
        CommitAction,
        ProposedCommitment,
        CommittedClaim,
        SupportQuery,
    )
    from extractor import CommitmentExtractor
    from ledger import EpistemicLedger


# =============================================================================
# Transaction Status
# =============================================================================

class TransactionStatus(Enum):
    """Status of a commit transaction."""
    PENDING = auto()
    VALIDATING = auto()
    AWAITING_SUPPORT = auto()
    AWAITING_REPAIR = auto()
    COMMITTED = auto()
    ROLLED_BACK = auto()
    REFUSED = auto()


# =============================================================================
# Transaction Result
# =============================================================================

@dataclass
class CommitResult:
    """Result of a commit transaction."""
    status: TransactionStatus
    committed_claims: list[CommittedClaim] = field(default_factory=list)
    hedged_spans: list[tuple[int, int, str]] = field(default_factory=list)  # (start, end, replacement)
    refused_claims: list[ProposedCommitment] = field(default_factory=list)
    pending_support: list[SupportQuery] = field(default_factory=list)
    repair_spans: list[tuple[int, int]] = field(default_factory=list)
    modified_text: Optional[str] = None
    original_text: str = ""
    total_cost: float = 0.0
    errors: list[str] = field(default_factory=list)


# =============================================================================
# Text Modifier
# =============================================================================

class TextModifier:
    """
    Applies hedging and other modifications to text before exposure.
    """
    
    # Hedging templates by confidence level
    HEDGE_TEMPLATES = {
        0.7: "likely {text}",
        0.6: "probably {text}",
        0.5: "possibly {text}",
        0.4: "it seems that {text}",
        0.3: "it's unclear, but {text}",
    }
    
    # Confidence markers to inject
    CONFIDENCE_MARKERS = {
        0.9: "",  # high confidence, no marker needed
        0.8: "",
        0.7: "I believe ",
        0.6: "I think ",
        0.5: "It seems ",
        0.4: "It's possible that ",
        0.3: "I'm uncertain, but ",
    }
    
    @classmethod
    def apply_hedge(
        cls, 
        text: str, 
        original_confidence: float,
        target_confidence: float,
    ) -> str:
        """
        Apply hedging to reduce expressed confidence.
        """
        if target_confidence >= original_confidence:
            return text
        
        # Find appropriate confidence marker
        marker = ""
        for threshold, template in sorted(cls.CONFIDENCE_MARKERS.items(), reverse=True):
            if target_confidence <= threshold:
                marker = template
        
        # Don't double-hedge if text already has uncertainty markers
        uncertainty_markers = [
            "i think", "i believe", "probably", "possibly", 
            "it seems", "it's possible", "might", "may"
        ]
        text_lower = text.lower()
        if any(m in text_lower for m in uncertainty_markers):
            return text
        
        # Apply marker
        if marker:
            # Lowercase first letter if adding prefix
            if text and text[0].isupper():
                text = text[0].lower() + text[1:]
            return marker + text
        
        return text
    
    @classmethod
    def apply_all_modifications(
        cls,
        original_text: str,
        decisions: list[CommitDecision],
        proposals: list[ProposedCommitment],
    ) -> tuple[str, list[tuple[int, int, str]]]:
        """
        Apply all modifications to the original text.
        Returns modified text and list of changes made.
        """
        # Build map of proposal id -> proposal
        proposal_map = {p.id: p for p in proposals}
        
        # Collect modifications (sorted by position, reverse order for safe replacement)
        modifications = []
        for decision in decisions:
            if decision.action == CommitAction.HEDGE:
                prop = proposal_map.get(decision.commitment_id)
                if prop and decision.adjusted_confidence:
                    hedged = cls.apply_hedge(
                        prop.text,
                        prop.confidence,
                        decision.adjusted_confidence,
                    )
                    if hedged != prop.text:
                        modifications.append((prop.span_start, prop.span_end, hedged))
        
        # Sort by start position, descending (so we can replace without offset issues)
        modifications.sort(key=lambda x: x[0], reverse=True)
        
        # Apply modifications
        modified_text = original_text
        for start, end, replacement in modifications:
            modified_text = modified_text[:start] + replacement + modified_text[end:]
        
        return modified_text, modifications


# =============================================================================
# Support Handler
# =============================================================================

@dataclass
class SupportResult:
    """Result of a support/retrieval operation."""
    query: SupportQuery
    satisfied: bool
    evidence: Optional[str] = None
    source: Optional[str] = None
    confidence_boost: float = 0.0


class SupportHandler:
    """
    Handles support/retrieval requirements.
    In a real implementation, this would integrate with:
    - Web search
    - Document retrieval
    - Tool calls
    - Cached knowledge
    """
    
    def __init__(self, retrieval_fn: Optional[Callable] = None):
        self.retrieval_fn = retrieval_fn
    
    def satisfy(self, query: SupportQuery) -> SupportResult:
        """
        Attempt to satisfy a support requirement.
        Returns result indicating whether support was found.
        """
        if self.retrieval_fn:
            # Use provided retrieval function
            evidence, source = self.retrieval_fn(query.query_text)
            if evidence:
                return SupportResult(
                    query=query,
                    satisfied=True,
                    evidence=evidence,
                    source=source,
                    confidence_boost=0.1,
                )
        
        # Default: support not available
        return SupportResult(
            query=query,
            satisfied=False,
        )
    
    def satisfy_batch(self, queries: list[SupportQuery]) -> list[SupportResult]:
        """Satisfy multiple support queries."""
        return [self.satisfy(q) for q in queries]


# =============================================================================
# Main Commit Phase
# =============================================================================

class CommitPhase:
    """
    The transaction boundary for epistemic commitments.
    
    This is where governance decisions are enforced:
    - ACCEPT: commit as-is
    - HEDGE: modify text, reduce confidence, commit
    - REQUIRE_SUPPORT: attempt retrieval, then commit or hedge
    - REVISE: handle contradiction, commit with revision cost
    - REPAIR: mark for regeneration
    - REFUSE: reject entirely
    
    Nothing becomes true unless it passes through here.
    """
    
    def __init__(
        self,
        ledger: EpistemicLedger,
        governor: EpistemicGovernor,
        support_handler: Optional[SupportHandler] = None,
    ):
        self.ledger = ledger
        self.governor = governor
        self.support_handler = support_handler or SupportHandler()
        self.text_modifier = TextModifier()
    
    def process(
        self,
        original_text: str,
        proposals: list[ProposedCommitment],
        adjudication: AdjudicationResult,
        allow_partial: bool = False,  # Atomic by default; partial is explicit degraded mode
    ) -> CommitResult:
        """
        Process a batch of proposed commitments through the commit phase.
        
        Args:
            original_text: The original LLM output
            proposals: Extracted proposed commitments
            adjudication: Governor adjudication result
            allow_partial: If False (default), all-or-nothing; if True, commit what we can
        
        Returns:
            CommitResult with committed claims and modifications
        """
        result = CommitResult(
            status=TransactionStatus.VALIDATING,
            original_text=original_text,
        )
        
        # Build decision map
        decisions = {d.commitment_id: d for d in adjudication.decisions}
        
        # Track support refs for grounded claims
        support_refs_by_prop: dict[str, list[str]] = {}
        
        # Process each proposal
        to_commit = []
        to_hedge = []
        to_support = []
        to_revise = []
        to_refuse = []
        to_repair = []
        
        for prop in proposals:
            decision = decisions.get(prop.id)
            if not decision:
                continue
            
            if decision.action == CommitAction.ACCEPT:
                to_commit.append((prop, decision))
            elif decision.action == CommitAction.HEDGE:
                to_hedge.append((prop, decision))
            elif decision.action == CommitAction.REQUIRE_SUPPORT:
                to_support.append((prop, decision))
            elif decision.action == CommitAction.REVISE:
                to_revise.append((prop, decision))
            elif decision.action == CommitAction.REPAIR:
                to_repair.append((prop, decision))
            elif decision.action == CommitAction.REFUSE:
                to_refuse.append((prop, decision))
        
        # Handle refused claims
        for prop, decision in to_refuse:
            result.refused_claims.append(prop)
            result.total_cost += decision.cost
            self.governor.record_refusal()
        
        # Handle repair requests
        if to_repair:
            result.status = TransactionStatus.AWAITING_REPAIR
            for prop, decision in to_repair:
                result.repair_spans.append((prop.span_start, prop.span_end))
            # Don't commit anything if repairs needed and not allowing partial
            if not allow_partial:
                result.status = TransactionStatus.ROLLED_BACK
                return result
        
        # Handle support requirements
        if to_support:
            pending_queries = []
            for prop, decision in to_support:
                if decision.required_support:
                    for query in decision.required_support:
                        support_result = self.support_handler.satisfy(query)
                        if support_result.satisfied:
                            # Support found - can commit with boost
                            # Attach evidence to the commitment
                            if support_result.source or support_result.evidence:
                                if prop.id not in support_refs_by_prop:
                                    support_refs_by_prop[prop.id] = []
                                ref = f"{support_result.source or 'unknown'}: {support_result.evidence or ''}".strip()
                                support_refs_by_prop[prop.id].append(ref)
                            to_commit.append((prop, decision))
                        else:
                            # Support not found - hedge instead
                            hedge_decision = CommitDecision(
                                commitment_id=prop.id,
                                action=CommitAction.HEDGE,
                                adjusted_confidence=prop.confidence * 0.6,
                                cost=decision.cost,
                                reason="Support not found; hedging",
                            )
                            to_hedge.append((prop, hedge_decision))
                            # Track pending support for the result
                            result.pending_support.append(query)
        
        # Handle revisions
        for prop, decision in to_revise:
            if decision.revision_targets:
                # Check revision budget
                if decision.cost > self.governor.state.budget.remaining_revision_budget:
                    # Can't afford revision - refuse
                    result.refused_claims.append(prop)
                    result.errors.append(f"Revision cost {decision.cost} exceeds budget")
                    continue
                
                # Perform revision
                new_claim, revision = self.ledger.revise(
                    prop,
                    decision.revision_targets,
                    justification=decision.reason,
                    decision=decision,
                )
                result.committed_claims.append(new_claim)
                result.total_cost += decision.cost
                self.governor.record_commit(new_claim)
        
        # Handle hedges (apply text modifications)
        for prop, decision in to_hedge:
            # The actual hedging will be applied to text below
            # Commit with reduced confidence
            if decision.adjusted_confidence is not None:
                # Create modified decision for commit
                commit_decision = CommitDecision(
                    commitment_id=prop.id,
                    action=CommitAction.ACCEPT,
                    adjusted_confidence=decision.adjusted_confidence,
                    cost=decision.cost,
                    reason=decision.reason,
                )
                to_commit.append((prop, commit_decision))
                self.governor.record_hedge()
        
        # Commit accepted claims (dedupe by proposal id; last write wins)
        commit_map: dict[str, tuple[ProposedCommitment, CommitDecision]] = {}
        for prop, decision in to_commit:
            commit_map[prop.id] = (prop, decision)
        
        for prop, decision in commit_map.values():
            try:
                # Pass support refs if we have them
                refs = support_refs_by_prop.get(prop.id)
                claim = self.ledger.commit(prop, decision, support_refs=refs)
                result.committed_claims.append(claim)
                result.total_cost += decision.cost
                self.governor.record_commit(claim)
            except Exception as e:
                result.errors.append(f"Failed to commit {prop.id}: {str(e)}")
                if not allow_partial:
                    result.status = TransactionStatus.ROLLED_BACK
                    return result
        
        # Apply text modifications
        result.modified_text, result.hedged_spans = self.text_modifier.apply_all_modifications(
            original_text,
            [d for _, d in to_hedge],
            proposals,
        )
        
        # Determine final status
        if result.repair_spans and not allow_partial:
            result.status = TransactionStatus.AWAITING_REPAIR
        elif result.errors and not allow_partial:
            result.status = TransactionStatus.ROLLED_BACK
        elif result.committed_claims:
            result.status = TransactionStatus.COMMITTED
        elif result.refused_claims:
            result.status = TransactionStatus.REFUSED
        else:
            result.status = TransactionStatus.COMMITTED  # empty but valid
        
        return result


# =============================================================================
# Full Pipeline
# =============================================================================

class EpistemicPipeline:
    """
    Complete pipeline from LLM output to committed state.
    
    Flow:
    1. Pre-governor generates envelope
    2. (LLM generates text with envelope constraints)
    3. Extractor parses text into proposals
    4. Post-governor adjudicates proposals
    5. Commit phase enforces decisions
    6. Ledger stores results
    
    "2.5" mode: generate once → validate → targeted repair on violations
    """
    
    def __init__(
        self,
        ledger: Optional[EpistemicLedger] = None,
        retrieval_fn: Optional[Callable] = None,
    ):
        self.ledger = ledger or EpistemicLedger()
        self.governor = EpistemicGovernor()
        self.extractor = CommitmentExtractor()
        self.support_handler = SupportHandler(retrieval_fn)
        self.commit_phase = CommitPhase(
            self.ledger,
            self.governor,
            self.support_handler,
        )
    
    def get_envelope(
        self,
        user_intent: Optional[str] = None,
        domain_hint: Optional[str] = None,
    ) -> GenerationEnvelope:
        """
        Phase 1: Get generation envelope (pre-governor).
        Call this before LLM generation.
        """
        return self.governor.pre_generate(user_intent, domain_hint)
    
    def process_output(
        self,
        llm_output: str,
        envelope: GenerationEnvelope,
        scope: str = "conversation",
    ) -> CommitResult:
        """
        Phases 2-5: Process LLM output through the full pipeline.
        
        Args:
            llm_output: Raw text from LLM
            envelope: The envelope that was active during generation
            scope: Scope for commitments
        
        Returns:
            CommitResult with all processing outcomes
        """
        # Extract proposals
        proposals = self.extractor.extract(llm_output, scope)
        
        # Adjudicate
        adjudication = self.governor.adjudicate(proposals, envelope)
        
        # Commit
        result = self.commit_phase.process(
            llm_output,
            proposals,
            adjudication,
            allow_partial=True,
        )
        
        return result
    
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return {
            "ledger": self.ledger.get_stats(),
            "governor_metrics": {
                "total_commitments": self.governor.state.metrics.total_commitments,
                "contradiction_rate": self.governor.state.metrics.contradiction_rate,
                "revision_rate": self.governor.state.metrics.revision_rate,
                "hedges_forced": self.governor.state.metrics.hedges_forced,
                "refusals": self.governor.state.metrics.refusals,
            },
            "budget": {
                "remaining_confidence": self.governor.state.budget.remaining_confidence,
                "remaining_revision_budget": self.governor.state.budget.remaining_revision_budget,
                "high_confidence_count": self.governor.state.budget.high_confidence_count,
            },
        }


# =============================================================================
# Example Usage / Test
# =============================================================================

if __name__ == "__main__":
    print("=== Epistemic Pipeline Test ===\n")
    
    # Initialize pipeline
    pipeline = EpistemicPipeline()
    
    # Phase 1: Get envelope
    envelope = pipeline.get_envelope(
        user_intent="factual_query",
        domain_hint="technical",
    )
    print(f"Envelope max_confidence: {envelope.max_confidence}")
    print(f"Must hedge types: {[t.name for t in envelope.must_hedge_types]}")
    
    # Simulate LLM output
    llm_output = """
    Python 3.12 was released in October 2023 with several important improvements.
    The new version is approximately 5% faster than Python 3.11 on average.
    I think this makes it worth upgrading for performance-critical applications.
    According to the release notes, the key features include improved error messages.
    You should definitely upgrade if you're still on Python 3.10.
    """
    
    print(f"\n=== Processing LLM Output ===")
    print(f"Original length: {len(llm_output)} chars\n")
    
    # Process through pipeline
    result = pipeline.process_output(llm_output, envelope)
    
    print(f"Status: {result.status.name}")
    print(f"Total cost: {result.total_cost:.3f}")
    print(f"Committed claims: {len(result.committed_claims)}")
    print(f"Refused claims: {len(result.refused_claims)}")
    print(f"Hedged spans: {len(result.hedged_spans)}")
    
    if result.errors:
        print(f"Errors: {result.errors}")
    
    print("\n=== Committed Claims ===")
    for claim in result.committed_claims:
        print(f"\n[{claim.id}] {claim.claim_type.name}")
        print(f"  Text: {claim.text[:60]}...")
        print(f"  Confidence: {claim.confidence:.2f}")
        print(f"  Status: {claim.status.name}")
    
    if result.modified_text and result.modified_text != result.original_text:
        print("\n=== Text Modifications ===")
        print("Original:")
        print(result.original_text[:200])
        print("\nModified:")
        print(result.modified_text[:200])
    
    print("\n=== Pipeline Stats ===")
    stats = pipeline.get_stats()
    print(f"Ledger: {stats['ledger']}")
    print(f"Governor: {stats['governor_metrics']}")
    print(f"Budget: {stats['budget']}")
