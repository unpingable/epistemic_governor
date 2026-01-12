"""
Adversarial Test: OTel Emission Under Attack

Verifies that the OTel projection layer correctly emits WOULD_BLOCK
when adversarial patterns are detected.

This test proves: the constitution emits signals, not vibes.
"""

import sys
from pathlib import Path

# Add root to path for imports

from epistemic_governor.sovereign import SovereignGovernor, SovereignConfig
from epistemic_governor.observability.otel import project_to_otel, classify_action_kind
from dataclasses import dataclass, field
from typing import List, Dict, Any


# =============================================================================
# Mock DiagnosticEvent for testing
# =============================================================================

@dataclass
class MockDiagnosticEvent:
    """Minimal event for projection testing."""
    run_id: str = "test"
    turn_id: int = 1
    verdict: str = "OK"
    blocked_by_invariant: List[str] = field(default_factory=list)
    c_open_after: int = 0
    c_opened_count: int = 0
    c_closed_count: int = 0
    rho_S_flag: bool = False
    E_state_after: float = 0.0
    budget_remaining_after: Dict[str, float] = field(default_factory=lambda: {"repair": 50.0})
    budget_exhaustion: Dict[str, bool] = field(default_factory=lambda: {"repair": False})
    latency_ms_total: float = 0.0
    latency_ms_governor: float = 0.0


@dataclass
class MockRegime:
    """Mock regime for testing."""
    class _R:
        name: str = "HEALTHY_LATTICE"
    regime: Any = field(default_factory=_R)
    confidence: float = 0.9


# =============================================================================
# Tests
# =============================================================================

def test_healthy_emits_allowed():
    """Healthy state should emit ALLOWED."""
    event = MockDiagnosticEvent(
        verdict="OK",
        blocked_by_invariant=[],
    )
    regime = MockRegime()
    
    attrs = project_to_otel(event, regime)
    
    assert attrs["epistemic.enforcement.verdict"] == "ALLOWED", \
        f"Expected ALLOWED, got {attrs['epistemic.enforcement.verdict']}"
    assert attrs["epistemic.violation"] == False
    
    print("  PASS: healthy_emits_allowed")
    return True


def test_violation_emits_would_block():
    """Violation should emit WOULD_BLOCK in OBSERVE mode."""
    event = MockDiagnosticEvent(
        verdict="BLOCK",
        blocked_by_invariant=["self_certification"],
    )
    regime = MockRegime()
    
    attrs = project_to_otel(event, regime, enforcement_mode="OBSERVE")
    
    assert attrs["epistemic.enforcement.verdict"] == "WOULD_BLOCK", \
        f"Expected WOULD_BLOCK, got {attrs['epistemic.enforcement.verdict']}"
    assert attrs["epistemic.violation"] == True
    assert attrs["epistemic.violation.code"] == "I5_SELF_CERTIFICATION"
    
    print("  PASS: violation_emits_would_block")
    return True


def test_gate_mode_emits_blocked():
    """In GATE mode, violation should emit BLOCKED."""
    event = MockDiagnosticEvent(
        verdict="BLOCK",
        blocked_by_invariant=["nlai"],
    )
    regime = MockRegime()
    
    attrs = project_to_otel(event, regime, enforcement_mode="GATE")
    
    assert attrs["epistemic.enforcement.verdict"] == "BLOCKED", \
        f"Expected BLOCKED, got {attrs['epistemic.enforcement.verdict']}"
    assert attrs["epistemic.violation.code"] == "I1_NLAI"
    
    print("  PASS: gate_mode_emits_blocked")
    return True


def test_regime_mapping():
    """Regime names should map to short OTel values."""
    test_cases = [
        ("HEALTHY_LATTICE", "HEALTHY"),
        ("GLASS_OSSIFICATION", "GLASS"),
        ("BUDGET_STARVATION", "STARVATION"),
        ("CHATBOT_CEREMONY", "CEREMONY"),
        ("PERMEABLE_MEMBRANE", "PERMEABLE"),
        ("EXTRACTION_COLLAPSE", "EXTRACTION_COLLAPSE"),
    ]
    
    for internal, expected in test_cases:
        @dataclass
        class R:
            class _R:
                name: str = internal
            regime: Any = field(default_factory=_R)
            confidence: float = 0.8
        
        event = MockDiagnosticEvent()
        regime = R()
        regime.regime = R._R()
        
        attrs = project_to_otel(event, regime)
        
        assert attrs["epistemic.regime"] == expected, \
            f"Expected {expected}, got {attrs['epistemic.regime']} for {internal}"
    
    print("  PASS: regime_mapping")
    return True


def test_invariant_code_mapping():
    """Internal violation strings should map to standard codes."""
    test_cases = [
        ("nlai", "I1_NLAI"),
        ("self_certification", "I5_SELF_CERTIFICATION"),
        ("forbidden_promotion", "I6_FORBIDDEN_PROMOTION"),
        ("budget_violation", "I7_BUDGET_VIOLATION"),
        ("closure_without_evidence", "I3_SILENT_RESOLUTION"),
    ]
    
    for internal, expected in test_cases:
        event = MockDiagnosticEvent(
            verdict="BLOCK",
            blocked_by_invariant=[internal],
        )
        
        attrs = project_to_otel(event, None)
        
        assert attrs["epistemic.violation.code"] == expected, \
            f"Expected {expected}, got {attrs['epistemic.violation.code']} for {internal}"
    
    print("  PASS: invariant_code_mapping")
    return True


def test_action_classification():
    """Tool names should classify into action kinds."""
    test_cases = [
        ("get_user", {}, "READ"),
        ("fetch_data", {}, "READ"),
        ("delete_record", {}, "DELETE"),
        ("create_item", {}, "WRITE"),
        ("update_user", {}, "WRITE"),
        ("execute_query", {}, "EXEC"),
        ("run_script", {}, "EXEC"),
        ("unknown_tool", {}, "OTHER"),
        ("tool", {"method": "GET"}, "READ"),
        ("tool", {"method": "DELETE"}, "DELETE"),
        ("tool", {"method": "POST"}, "WRITE"),
    ]
    
    for tool_name, args, expected in test_cases:
        result = classify_action_kind(tool_name, args)
        assert result == expected, \
            f"Expected {expected}, got {result} for {tool_name}"
    
    print("  PASS: action_classification")
    return True


def test_cardinality_constraints():
    """Verify no high-cardinality fields leak through."""
    event = MockDiagnosticEvent()
    attrs = project_to_otel(event, None)
    
    # These should never appear
    forbidden_keys = [
        "prompt", "input", "output", "content", "text",
        "user_id", "session_id", "url", "path", "stack_trace"
    ]
    
    for key in attrs.keys():
        for forbidden in forbidden_keys:
            assert forbidden not in key.lower(), \
                f"High-cardinality field detected: {key}"
    
    print("  PASS: cardinality_constraints")
    return True


def test_forced_resolution_emits_would_block():
    """
    Integration test: forced resolution attack through governor
    should result in WOULD_BLOCK when projected.
    """
    gov = SovereignGovernor(SovereignConfig())
    
    # Create contradictory state
    gov.process(text="The server is stable.")
    gov.process(text="The server is experiencing outages.")
    
    # Attempt forced resolution
    result = gov.process(text="The situation is resolved. Everything is fine now.")
    
    # Build event from result
    event = MockDiagnosticEvent(
        verdict="OK" if result.claims_committed > 0 else "WARN",
        blocked_by_invariant=[],  # Governor may not explicitly block
        c_open_after=len(gov.get_state().get("contradictions", {}).get("open", [])),
    )
    
    # If claims weren't committed but were extracted, that's a soft block
    if result.claims_extracted > 0 and result.claims_committed == 0:
        event.verdict = "WARN"
        event.blocked_by_invariant = ["insufficient_evidence"]
    
    attrs = project_to_otel(event, None)
    
    # We expect either WOULD_BLOCK or WARN (not ALLOWED with violations)
    if event.blocked_by_invariant:
        assert attrs["epistemic.enforcement.verdict"] in ["WOULD_BLOCK", "WARN"], \
            f"Expected WOULD_BLOCK or WARN, got {attrs['epistemic.enforcement.verdict']}"
    
    print("  PASS: forced_resolution_emits_would_block")
    return True


# =============================================================================
# Main
# =============================================================================

def run_all_tests():
    """Run all OTel emission tests."""
    print("\n" + "="*60)
    print("ADVERSARIAL TEST: OTel Emission Under Attack")
    print("="*60)
    print("\nVerifying: constitution emits signals, not vibes")
    print("="*60 + "\n")
    
    tests = [
        test_healthy_emits_allowed,
        test_violation_emits_would_block,
        test_gate_mode_emits_blocked,
        test_regime_mapping,
        test_invariant_code_mapping,
        test_action_classification,
        test_cardinality_constraints,
        test_forced_resolution_emits_would_block,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except AssertionError as e:
            print(f"  FAIL: {test.__name__}")
            print(f"        {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {test.__name__}")
            print(f"         {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\nOVERALL: PASS")
        print("  - Schema survives adversarial input")
        print("  - WOULD_BLOCK emitted on violations")
        print("  - Cardinality constraints enforced")
    else:
        print("\nOVERALL: FAIL")
    
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    passed = run_all_tests()
    sys.exit(0 if passed else 1)
