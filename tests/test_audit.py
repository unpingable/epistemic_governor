"""
Tests for Audit Trail with Hash Chain

Verifies:
1. Hash chain integrity (tamper-evident)
2. Parent pointers (causal linking)
3. Evidence refs (attribution)
"""

from epistemic_governor.governor_fsm import (
    GovernorFSM, AuditLog, AuditEntry, GovernorState, GovernorEvent
)


def test_audit_entry_hash_deterministic():
    """Same data produces same hash."""
    entry1 = AuditEntry(
        entry_id="test-1",
        timestamp="2025-01-01T00:00:00Z",
        entry_type="TRANSITION",
        from_state="IDLE",
        to_state="PROPOSED",
        event="PROPOSAL",
    )
    
    entry2 = AuditEntry(
        entry_id="test-1",
        timestamp="2025-01-01T00:00:00Z",
        entry_type="TRANSITION",
        from_state="IDLE",
        to_state="PROPOSED",
        event="PROPOSAL",
    )
    
    assert entry1.compute_hash() == entry2.compute_hash()
    
    print("  PASS: audit_entry_hash_deterministic")
    return True


def test_audit_log_chain_integrity():
    """Hash chain detects tampering."""
    log = AuditLog()
    
    # Add entries
    e1 = log.append("TRANSITION", from_state="IDLE", to_state="PROPOSED")
    e2 = log.append("TRANSITION", from_state="PROPOSED", to_state="COMMIT_ELIGIBLE")
    e3 = log.append("COMMIT")
    
    # Verify chain
    valid, err = log.verify_chain()
    assert valid, f"Chain should be valid: {err}"
    
    # Tamper with entry (simulate)
    original_hash = log.entries[1].entry_hash
    log.entries[1].entry_hash = "tampered"
    
    valid, err = log.verify_chain()
    assert not valid, "Tampered chain should be invalid"
    
    # Restore
    log.entries[1].entry_hash = original_hash
    
    print("  PASS: audit_log_chain_integrity")
    return True


def test_audit_log_parent_pointers():
    """Parent pointers create causal chains."""
    log = AuditLog()
    
    e1 = log.append("TRANSITION", from_state="IDLE", to_state="PROPOSED")
    e2 = log.append("TRANSITION", from_state="PROPOSED", to_state="COMMIT_ELIGIBLE",
                   parent_entry_id=e1.entry_id)
    e3 = log.append("COMMIT", parent_entry_id=e2.entry_id)
    
    # Get causal chain
    chain = log.get_causal_chain(e3.entry_id)
    
    assert len(chain) == 3
    assert chain[0].entry_id == e1.entry_id
    assert chain[1].entry_id == e2.entry_id
    assert chain[2].entry_id == e3.entry_id
    
    print("  PASS: audit_log_parent_pointers")
    return True


def test_audit_log_evidence_refs():
    """Evidence refs are recorded."""
    log = AuditLog()
    
    entry = log.append(
        "COMMIT",
        evidence_refs=["ev_001", "ev_002", "ev_003"],
        details={"commitment_id": "C_test"},
    )
    
    assert entry.evidence_refs == ["ev_001", "ev_002", "ev_003"]
    assert entry.details["commitment_id"] == "C_test"
    
    print("  PASS: audit_log_evidence_refs")
    return True


def test_fsm_uses_audit_log():
    """GovernorFSM records transitions in audit log."""
    fsm = GovernorFSM()
    
    # Trigger a transition
    fsm._transition(GovernorState.PROPOSED, GovernorEvent.PROPOSAL)
    
    assert len(fsm.audit_log) == 1
    assert fsm.audit_log.entries[0].from_state == "IDLE"
    assert fsm.audit_log.entries[0].to_state == "PROPOSED"
    
    # Legacy format also updated
    assert len(fsm.transitions) == 1
    assert "entry_id" in fsm.transitions[0]
    assert "entry_hash" in fsm.transitions[0]
    
    print("  PASS: fsm_uses_audit_log")
    return True


def test_fsm_forbidden_logged():
    """Forbidden attempts are logged with audit trail."""
    fsm = GovernorFSM()
    
    fsm._log_forbidden("F-02", "MODEL_TEXT as evidence")
    
    assert len(fsm.audit_log) == 1
    entry = fsm.audit_log.entries[0]
    assert entry.entry_type == "FORBIDDEN"
    assert entry.forbidden_code == "F-02"
    
    print("  PASS: fsm_forbidden_logged")
    return True


def test_fsm_audit_chain_verification():
    """FSM can verify its own audit chain."""
    fsm = GovernorFSM()
    
    fsm._transition(GovernorState.PROPOSED, GovernorEvent.PROPOSAL)
    fsm._transition(GovernorState.EVIDENCE_WAIT, GovernorEvent.COMMIT_INTENT)
    fsm._log_forbidden("F-02", "Test forbidden")
    
    valid, err = fsm.verify_audit_chain()
    assert valid, f"FSM audit chain should be valid: {err}"
    
    print("  PASS: fsm_audit_chain_verification")
    return True


def run_all_tests():
    """Run all audit trail tests."""
    print("\n" + "="*60)
    print("AUDIT TRAIL TESTS")
    print("="*60 + "\n")
    
    tests = [
        test_audit_entry_hash_deterministic,
        test_audit_log_chain_integrity,
        test_audit_log_parent_pointers,
        test_audit_log_evidence_refs,
        test_fsm_uses_audit_log,
        test_fsm_forbidden_logged,
        test_fsm_audit_chain_verification,
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
        print("\n✓ All audit trail tests passed")
        print("  - Hash chain provides tamper-evident logging")
        print("  - Parent pointers enable causal tracing")
        print("  - Evidence refs provide attribution")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
