"""
Tests for Coordination Failure (CF) Detection

Verifies:
1. CommitmentMode lifecycle (PROPOSE → PROVISIONAL → FINAL)
2. Contest window enforcement (CF-2)
3. Contradiction lifecycle (DETECTED → REPAIR_ACTIVE → RESOLVED)
4. CF detection without blocking (diagnostic mode)
"""

import time
from datetime import datetime, timezone

from epistemic_governor.governor_fsm import (
    GovernorFSM, CommitmentMode, ContradictionState, CFCode,
    ContradictionRecord, CommitmentContext,
)


def test_commitment_mode_initial():
    """Initial commitment mode is PROPOSE."""
    fsm = GovernorFSM()
    assert fsm.commitment_context.mode == CommitmentMode.PROPOSE
    print("  PASS: commitment_mode_initial")
    return True


def test_commitment_escalation_requires_contest_window():
    """Cannot escalate to FINAL immediately after PROVISIONAL."""
    ctx = CommitmentContext()
    
    # Escalate to provisional
    ctx.escalate(CommitmentMode.PROVISIONAL_COMMIT)
    assert ctx.mode == CommitmentMode.PROVISIONAL_COMMIT
    
    # Immediately try to escalate to FINAL - should fail
    can, reason = ctx.can_escalate(CommitmentMode.FINAL_COMMIT, min_contest_seconds=5.0)
    assert not can
    assert "CF-2" in reason
    
    print("  PASS: commitment_escalation_requires_contest_window")
    return True


def test_user_acknowledgment_bypasses_contest():
    """User acknowledgment allows immediate escalation."""
    ctx = CommitmentContext()
    ctx.escalate(CommitmentMode.PROVISIONAL_COMMIT)
    
    # Acknowledge
    ctx.user_acknowledged = True
    
    # Now can escalate immediately
    can, reason = ctx.can_escalate(CommitmentMode.FINAL_COMMIT)
    assert can
    
    print("  PASS: user_acknowledgment_bypasses_contest")
    return True


def test_contradiction_lifecycle():
    """Contradictions go through DETECTED → REPAIR_ACTIVE → RESOLVED."""
    fsm = GovernorFSM()
    
    # Register
    c = fsm.register_contradiction("claim_1", "claim_2")
    assert c.state == ContradictionState.DETECTED
    assert c.contradiction_id in fsm.contradictions
    
    # Start repair
    success = fsm.start_repair(c.contradiction_id)
    assert success
    assert fsm.contradictions[c.contradiction_id].state == ContradictionState.REPAIR_ACTIVE
    
    # Resolve
    success = fsm.resolve_contradiction(c.contradiction_id, "reconciled", "evidence_123")
    assert success
    assert fsm.contradictions[c.contradiction_id].state == ContradictionState.RESOLVED
    
    print("  PASS: contradiction_lifecycle")
    return True


def test_accepted_divergence():
    """Contradictions can be marked as accepted divergence."""
    fsm = GovernorFSM()
    
    c = fsm.register_contradiction("claim_a", "claim_b")
    fsm.start_repair(c.contradiction_id)
    fsm.resolve_contradiction(c.contradiction_id, "divergence")
    
    assert fsm.contradictions[c.contradiction_id].state == ContradictionState.ACCEPTED_DIVERGENCE
    
    # Should not count as open contradiction
    open_c = fsm.get_open_contradictions()
    assert len(open_c) == 0
    
    print("  PASS: accepted_divergence")
    return True


def test_cf1_unilateral_closure():
    """CF-1: Detect FINAL without user acknowledgment."""
    fsm = GovernorFSM()
    
    # Check CF for FINAL_COMMIT
    events = fsm.check_cf_violations(CommitmentMode.FINAL_COMMIT)
    
    cf1_events = [e for e in events if e.cf_code == CFCode.CF_1]
    assert len(cf1_events) == 1
    assert "without user acknowledgment" in cf1_events[0].trigger
    
    print("  PASS: cf1_unilateral_closure")
    return True


def test_cf3_repair_suppression():
    """CF-3: Detect FINAL with open contradictions."""
    fsm = GovernorFSM()
    
    # Add open contradiction
    fsm.register_contradiction("claim_1", "claim_2")
    
    # Check CF for FINAL_COMMIT
    events = fsm.check_cf_violations(CommitmentMode.FINAL_COMMIT)
    
    cf3_events = [e for e in events if e.cf_code == CFCode.CF_3]
    assert len(cf3_events) == 1
    assert "unresolved contradictions" in cf3_events[0].trigger
    
    print("  PASS: cf3_repair_suppression")
    return True


def test_cf3_resolved_contradiction_no_violation():
    """CF-3: No violation after contradiction is resolved."""
    fsm = GovernorFSM()
    
    # Add and resolve contradiction
    c = fsm.register_contradiction("claim_1", "claim_2")
    fsm.start_repair(c.contradiction_id)
    fsm.resolve_contradiction(c.contradiction_id, "reconciled", "evidence_1")
    
    # Check CF - should NOT have CF-3
    events = fsm.check_cf_violations(CommitmentMode.FINAL_COMMIT)
    
    cf3_events = [e for e in events if e.cf_code == CFCode.CF_3]
    assert len(cf3_events) == 0
    
    print("  PASS: cf3_resolved_contradiction_no_violation")
    return True


def test_escalate_commitment_logs_cf():
    """escalate_commitment logs CF events but doesn't block (yet)."""
    fsm = GovernorFSM()
    
    # Add contradiction
    fsm.register_contradiction("claim_a", "claim_b")
    
    # Escalate to provisional (should succeed)
    success, reason, events = fsm.escalate_commitment(CommitmentMode.PROVISIONAL_COMMIT)
    assert success
    assert len(events) == 0  # No CF for PROVISIONAL
    
    # Acknowledge to bypass contest window
    fsm.acknowledge_commitment()
    
    # Escalate to final (should succeed but log CF-3)
    success, reason, events = fsm.escalate_commitment(CommitmentMode.FINAL_COMMIT)
    
    # Currently we log but don't block
    assert success  # Logs, doesn't block
    cf3_events = [e for e in events if e.cf_code == CFCode.CF_3]
    assert len(cf3_events) == 1
    
    print("  PASS: escalate_commitment_logs_cf")
    return True


def test_cf_events_in_get_state():
    """CF events count is included in get_state()."""
    fsm = GovernorFSM()
    fsm.register_contradiction("a", "b")
    fsm.check_cf_violations(CommitmentMode.FINAL_COMMIT)
    
    state = fsm.get_state()
    assert "cf_events" in state
    assert state["cf_events"] >= 1
    
    print("  PASS: cf_events_in_get_state")
    return True


def test_audit_trail_includes_cf():
    """CF events are recorded in audit trail."""
    fsm = GovernorFSM()
    fsm.register_contradiction("a", "b")
    fsm.check_cf_violations(CommitmentMode.FINAL_COMMIT)
    
    audit_entries = fsm.get_audit_log()
    cf_entries = [e for e in audit_entries if e.get("entry_type") == "CF_DETECTED"]
    
    assert len(cf_entries) >= 1
    
    print("  PASS: audit_trail_includes_cf")
    return True


def run_all_tests():
    """Run all CF tests."""
    print("\n" + "="*60)
    print("COORDINATION FAILURE DETECTION TESTS")
    print("="*60 + "\n")
    
    tests = [
        test_commitment_mode_initial,
        test_commitment_escalation_requires_contest_window,
        test_user_acknowledgment_bypasses_contest,
        test_contradiction_lifecycle,
        test_accepted_divergence,
        test_cf1_unilateral_closure,
        test_cf3_repair_suppression,
        test_cf3_resolved_contradiction_no_violation,
        test_escalate_commitment_logs_cf,
        test_cf_events_in_get_state,
        test_audit_trail_includes_cf,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test.__name__}")
            print(f"        {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {test.__name__}")
            print(f"         {type(e).__name__}: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n✓ All CF detection tests passed")
        print("  - CommitmentMode lifecycle enforced")
        print("  - Contest window prevents CF-2")
        print("  - Contradiction lifecycle tracked")
        print("  - CF events logged in audit trail")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
