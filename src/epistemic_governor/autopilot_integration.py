"""
Autopilot Integration with Epistemic Ledger

This module bridges the autopilot layer with the real EpistemicLedger.

Key principles (from ChatGPT review):
1. Autopilot never calls ledger.commit() directly
2. Autopilot emits ProposedCommitment batches + constraints
3. Governor adjudicates, ledger records
4. Forks are the sanctioned exploration mechanism

Integration points:
- LedgerTelemetryAdapter: Extracts telemetry from real ledger
- HeadingEnforcer: Validates proposed commits against heading
- AutopilotController: Coordinates FSM + telemetry + enforcement

Usage:
    from epistemic_governor.autopilot_integration import (
        AutopilotController,
        LedgerTelemetryAdapter,
    )
    
    controller = AutopilotController(ledger, envelope)
    controller.engage(heading)
    
    # On proposed commit
    result = controller.check_proposal(proposed)
    # → ALLOW | FORK_REQUIRED | ARBITRATE | REJECT
"""

from dataclasses import dataclass, field
from typing import Optional, List, Set, Dict, Any, Tuple
from datetime import datetime
from enum import Enum, auto

# Core modules
from epistemic_governor.ledger import EpistemicLedger, EntryType, LedgerEntry, Fork
from epistemic_governor.governor import ProposedCommitment, CommitmentStatus
from epistemic_governor.envelope import FlightEnvelope

# Autopilot modules
from epistemic_governor.heading import (
    Heading, HeadingValidator, HeadingType,
    AUTOPILOT_SAFE_HEADINGS, FORBIDDEN_HEADINGS,
)
from epistemic_governor.telemetry import (
    TelemetryComputer, WarningEvent,
    LedgerSnapshot, EnvelopeSnapshot, AutopilotSnapshot,
    TelemetryThresholds, StabilizationState,
)
from epistemic_governor.autopilot_fsm import (
    AutopilotFSM, AutopilotMode, TransitionResult,
    ConstraintClass, ArbitrationOption,
)


# =============================================================================
# Proposal Check Results
# =============================================================================

class ProposalVerdict(Enum):
    """Result of checking a proposal against heading constraints."""
    ALLOW = auto()           # Proposal conforms to heading
    FORK_REQUIRED = auto()   # Proposal requires fork for exploration
    ARBITRATE = auto()       # User arbitration needed
    REJECT = auto()          # Proposal violates invariants


@dataclass
class ProposalCheckResult:
    """Result of checking a proposal."""
    verdict: ProposalVerdict
    reason: str
    
    # If FORK_REQUIRED
    suggested_fork_name: Optional[str] = None
    
    # If ARBITRATE
    conflict_set: Optional[List[str]] = None
    options: Optional[List[ArbitrationOption]] = None
    
    # Metrics
    is_new_proposition: bool = False
    support_refs_count: int = 0
    within_scope: bool = True


# =============================================================================
# Ledger Telemetry Adapter
# =============================================================================

class LedgerTelemetryAdapter:
    """
    Extracts telemetry metrics directly from EpistemicLedger.
    
    Uses actual ledger entries, not placeholders.
    """
    
    def __init__(self, ledger: EpistemicLedger, window_size: int = 10):
        self.ledger = ledger
        self.window_size = window_size
        self._last_entry_count = 0
        
        # Cached metrics
        self._commits_this_window = 0
        self._revisions_this_window = 0
        self._evidence_refs_this_window = 0
    
    def get_snapshot(self) -> LedgerSnapshot:
        """
        Build LedgerSnapshot from real ledger state.
        """
        # Get entries since last check
        new_entries = self.ledger._entries[self._last_entry_count:]
        self._last_entry_count = len(self.ledger._entries)
        
        # Count activity in window
        commits = sum(1 for e in new_entries if e.entry_type == EntryType.COMMIT)
        revisions = sum(1 for e in new_entries if e.entry_type == EntryType.REVISION)
        
        # Count evidence refs in new commits
        evidence_count = 0
        for e in new_entries:
            if e.entry_type == EntryType.COMMIT:
                refs = e.data.get("support_refs", [])
                evidence_count += len(refs)
        
        # Get claim states
        active_claims = self.ledger.get_active_claims()
        
        # Calculate claim ages (steps since commit for unresolved)
        now = datetime.now()
        claim_ages = []
        ungrounded_count = 0
        
        for claim in active_claims:
            age_seconds = (now - claim.committed_at).total_seconds()
            age_steps = int(age_seconds / 60)  # Rough: 1 step = 1 minute
            
            if not claim.support_refs:
                ungrounded_count += 1
                claim_ages.append(age_steps)
        
        return LedgerSnapshot(
            step=len(self.ledger._entries),
            total_claims=len(self.ledger._claims),
            active_claims=len(active_claims),
            ungrounded_claims=ungrounded_count,
            grounded_claims=len(active_claims) - ungrounded_count,
            peer_asserted_claims=0,  # Would need claim source tracking
            total_evidence=sum(len(c.support_refs) for c in active_claims),
            evidence_this_window=evidence_count,
            claim_ages=claim_ages,
            new_claims_this_window=commits,
            promotions_this_window=0,  # Not tracked in this ledger
            retractions_this_window=revisions,
        )
    
    def get_fossilization_debt(self) -> float:
        """
        Compute fossilization debt: fossils / commits ratio.
        
        High FD = "we're spraying claims and later compressing the mess."
        """
        total_commits = sum(
            1 for e in self.ledger._entries 
            if e.entry_type == EntryType.COMMIT
        )
        total_archives = sum(
            1 for e in self.ledger._entries 
            if e.entry_type == EntryType.ARCHIVE
        )
        
        if total_commits == 0:
            return 0.0
        return total_archives / total_commits
    
    def get_revision_load(self) -> float:
        """
        Compute revision load from recent entries.
        
        How turbulent is the air?
        """
        recent = self.ledger._entries[-self.window_size:]
        revisions = sum(1 for e in recent if e.entry_type == EntryType.REVISION)
        return revisions / max(len(recent), 1)
    
    def get_resolution_rate(self) -> float:
        """
        Compute resolution rate: how fast are we working off epistemic debt?
        
        resolution_rate = evidence_added / active_claims
        """
        active = self.ledger.get_active_claims()
        if not active:
            return 1.0  # No claims = nothing to resolve
        
        # Count claims that gained evidence recently
        recent = self.ledger._entries[-self.window_size:]
        claims_with_new_evidence = set()
        
        for e in recent:
            if e.entry_type == EntryType.COMMIT:
                if e.data.get("support_refs"):
                    claims_with_new_evidence.add(e.claim_id)
        
        return len(claims_with_new_evidence) / len(active)


# =============================================================================
# Heading Enforcer
# =============================================================================

class HeadingEnforcer:
    """
    Enforces heading constraints on proposed commits.
    
    Key checks:
    1. Is this a new proposition_hash?
    2. Does it have required support_refs?
    3. Is it within allowed scope?
    4. Does the heading allow this transform?
    """
    
    def __init__(self, heading: Heading, ledger: EpistemicLedger):
        self.heading = heading
        self.ledger = ledger
        self.validator = HeadingValidator(heading)
        
        # Track new propositions in this session
        self._new_propositions: Set[str] = set()
    
    def check_proposal(self, proposed: ProposedCommitment) -> ProposalCheckResult:
        """
        Check if a proposed commit conforms to heading constraints.
        """
        # 1. Check if this is a new proposition
        is_new = self._is_new_proposition(proposed.proposition_hash)
        
        # 2. Check support refs
        support_count = len(proposed.support_refs) if hasattr(proposed, 'support_refs') else 0
        
        # 3. Check scope
        within_scope = self._check_scope(proposed.scope)
        
        # 4. Check new claim budget
        if is_new:
            self._new_propositions.add(proposed.proposition_hash)
            
            if len(self._new_propositions) > self.heading.max_new_claims:
                # Over budget - need to fork or arbitrate
                if self.heading.heading_type in FORBIDDEN_HEADINGS:
                    return ProposalCheckResult(
                        verdict=ProposalVerdict.REJECT,
                        reason="Heading forbids new claims",
                        is_new_proposition=True,
                    )
                elif self.heading.requires_scope_bounds:
                    # Bounded expansion heading - fork for exploration
                    return ProposalCheckResult(
                        verdict=ProposalVerdict.FORK_REQUIRED,
                        reason="New claim budget exceeded",
                        suggested_fork_name=f"exploration_{proposed.proposition_hash[:8]}",
                        is_new_proposition=True,
                    )
                else:
                    return ProposalCheckResult(
                        verdict=ProposalVerdict.ARBITRATE,
                        reason="New claim budget exceeded",
                        conflict_set=["new_claim_budget", "completeness"],
                        is_new_proposition=True,
                    )
        
        # 5. Check support requirements for bounded headings
        if is_new and self.heading.requires_scope_bounds:
            if hasattr(self.heading, 'require_sources_for_new'):
                if self.heading.require_sources_for_new and support_count == 0:
                    return ProposalCheckResult(
                        verdict=ProposalVerdict.REJECT,
                        reason="New claims require support refs",
                        is_new_proposition=True,
                        support_refs_count=support_count,
                    )
        
        # 6. Check scope fence
        if not within_scope:
            return ProposalCheckResult(
                verdict=ProposalVerdict.REJECT,
                reason="Proposal outside allowed scope",
                within_scope=False,
            )
        
        return ProposalCheckResult(
            verdict=ProposalVerdict.ALLOW,
            reason="Proposal conforms to heading",
            is_new_proposition=is_new,
            support_refs_count=support_count,
            within_scope=within_scope,
        )
    
    def _is_new_proposition(self, prop_hash: str) -> bool:
        """Check if proposition_hash is new (not in active claims)."""
        # Check ledger index
        existing = self.ledger._by_proposition.get(prop_hash, [])
        
        # Check if any are still active
        for claim_id in existing:
            claim = self.ledger._claims.get(claim_id)
            if claim and claim.status == CommitmentStatus.ACTIVE:
                return False
        
        # Also check if we've seen it in this session
        return prop_hash not in self._new_propositions
    
    def _check_scope(self, scope: str) -> bool:
        """Check if scope is within heading's allowed scopes."""
        if not self.heading.allowed_source_ids:
            return True  # No restrictions
        
        # Check if scope matches any allowed source
        return scope in self.heading.allowed_source_ids


# =============================================================================
# Autopilot Controller
# =============================================================================

class AutopilotController:
    """
    Main autopilot controller.
    
    Coordinates:
    - FSM state management
    - Telemetry monitoring
    - Heading enforcement
    - Fork management for exploration
    
    Key principle: Autopilot never commits directly.
    It only determines what the governor is allowed to accept.
    """
    
    def __init__(
        self,
        ledger: EpistemicLedger,
        envelope: FlightEnvelope,
        thresholds: TelemetryThresholds = None,
    ):
        self.ledger = ledger
        self.envelope = envelope
        
        # Components
        self.fsm = AutopilotFSM()
        self.telemetry_adapter = LedgerTelemetryAdapter(ledger)
        self.telemetry = TelemetryComputer()
        self.stabilization = StabilizationState()
        
        # Current heading enforcement
        self.enforcer: Optional[HeadingEnforcer] = None
        
        # Thresholds
        if thresholds:
            self.telemetry.thresholds = thresholds
        
        # Warning history (for "ignorable without penalty")
        self._warnings_emitted: List[WarningEvent] = []
    
    # =========================================================================
    # Engagement
    # =========================================================================
    
    def engage(self, heading: Heading) -> TransitionResult:
        """
        Engage autopilot with a heading.
        """
        result = self.fsm.engage(heading)
        
        if result.success:
            self.enforcer = HeadingEnforcer(heading, self.ledger)
        
        return result
    
    def disengage(self) -> TransitionResult:
        """
        User-initiated disengage.
        """
        result = self.fsm.manual_disengage()
        self.enforcer = None
        return result
    
    # =========================================================================
    # Proposal Checking
    # =========================================================================
    
    def check_proposal(self, proposed: ProposedCommitment) -> ProposalCheckResult:
        """
        Check if a proposed commit is allowed under current heading.
        
        This is called BEFORE governor adjudication.
        """
        if not self.fsm.is_active():
            # Autopilot not active - allow everything
            return ProposalCheckResult(
                verdict=ProposalVerdict.ALLOW,
                reason="Autopilot not active",
            )
        
        if not self.enforcer:
            return ProposalCheckResult(
                verdict=ProposalVerdict.ALLOW,
                reason="No heading enforcer",
            )
        
        return self.enforcer.check_proposal(proposed)
    
    def request_fork(self, name: str, reason: str) -> Optional[Fork]:
        """
        Request a fork for exploration.
        
        This is the sanctioned mechanism for epistemic expansion
        when autopilot is active.
        """
        if not self.fsm.is_active():
            return None
        
        fork = self.ledger.create_fork(name, reason)
        return fork
    
    # =========================================================================
    # Telemetry Tick
    # =========================================================================
    
    def tick(self) -> List[WarningEvent]:
        """
        Update telemetry and check for mode transitions.
        
        Called periodically (e.g., after each turn).
        Returns any warning events (declarative, not coercive).
        """
        if not self.fsm.is_active():
            return []
        
        # Get snapshots
        ledger_snap = self.telemetry_adapter.get_snapshot()
        envelope_snap = self._get_envelope_snapshot()
        autopilot_snap = self._get_autopilot_snapshot()
        
        # Update telemetry
        warnings = self.telemetry.update(ledger_snap, envelope_snap, autopilot_snap)
        
        # Record warnings (for audit)
        self._warnings_emitted.extend(warnings)
        
        # Check for mode transitions based on telemetry
        self._check_escalation(ledger_snap, envelope_snap)
        
        # Check for recovery
        self._check_recovery(envelope_snap)
        
        return warnings
    
    def _get_envelope_snapshot(self) -> EnvelopeSnapshot:
        """Build envelope snapshot from current state."""
        # Get constraint slack from envelope
        # This is approximate - would need envelope API updates
        return EnvelopeSnapshot(
            step=len(self.ledger._entries),
            provenance_slack=0.8,  # Would compute from envelope
            scope_slack=0.9,
            contradiction_slack=1.0,
            uncertainty_slack=0.7,
            near_violations=0,
            actual_violations=0,
        )
    
    def _get_autopilot_snapshot(self) -> AutopilotSnapshot:
        """Build autopilot snapshot."""
        return AutopilotSnapshot(
            step=len(self.ledger._entries),
            drops_this_window=len(self.fsm.dropped_constraints),
            drop_types=list(self.fsm.dropped_constraints),
            current_level=self.fsm.mode.value,
            level_changes_this_window=0,
        )
    
    def _check_escalation(
        self,
        ledger: LedgerSnapshot,
        envelope: EnvelopeSnapshot,
    ):
        """Check if we need to escalate mode."""
        # Check for soft constraint conflicts
        metrics = self.telemetry.get_metrics_summary()
        
        # High CAR + low slack → degrade brevity first
        if metrics["car"]["mean"] > 2.0 and envelope.provenance_slack < 0.3:
            soft_to_drop = self.fsm.get_soft_by_sacrifice_order()
            if soft_to_drop:
                self.fsm.escalate_to_degraded(
                    soft_to_drop[0],
                    "high_car_low_slack",
                )
        
        # Critical slack → arbitrate
        if envelope.provenance_slack < 0.1:
            self.fsm.escalate_to_arbitrating(
                ["provenance", "completeness"]
            )
    
    def _check_recovery(self, envelope: EnvelopeSnapshot):
        """Check if we can recover to a lower mode."""
        is_stable, reason = self.stabilization.check(self.telemetry, envelope)
        self.fsm.attempt_recovery(is_stable)
    
    # =========================================================================
    # Queries
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get current autopilot status."""
        return {
            "mode": self.fsm.mode.name,
            "heading": self.fsm.heading.heading_type.name if self.fsm.heading else None,
            "dropped_constraints": list(self.fsm.dropped_constraints),
            "warnings_count": len(self._warnings_emitted),
            "needs_arbitration": self.fsm.needs_user_input(),
            "metrics": self.telemetry.get_metrics_summary(),
        }
    
    def get_arbitration_options(self) -> Optional[List[ArbitrationOption]]:
        """Get pending arbitration options."""
        return self.fsm.get_arbitration_options()
    
    def resolve_arbitration(self, option_id: str) -> TransitionResult:
        """Resolve pending arbitration."""
        return self.fsm.resolve_arbitration(option_id)
    
    def get_flight_recorder(self) -> List[Dict[str, Any]]:
        """Get full event history."""
        return self.fsm.get_flight_recorder()


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=== Autopilot Integration Demo ===\n")
    
    # This would require setting up the full ledger infrastructure
    print("Integration module loaded successfully.")
    print("Use AutopilotController with real EpistemicLedger.")
