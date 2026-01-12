"""
Tests for Regime Detection and Reset

Verifies:
1. Regime classification from signals
2. Automatic response mapping
3. Reset operations
4. Coupling reduction rule
"""

import sys
from pathlib import Path


from epistemic_governor.control.regime import (
    RegimeDetector,
    OperationalRegime,
    RegimeSignals,
    check_coupling_reduction,
)
from epistemic_governor.control.reset import ResetController, ResetType


def test_elastic_regime():
    """Low signals → ELASTIC regime, normal operation."""
    detector = RegimeDetector()
    
    signals = RegimeSignals(
        hysteresis_magnitude=0.1,
        relaxation_time_seconds=1.0,
        tool_gain_estimate=0.3,
    )
    
    response = detector.respond(signals)
    
    assert response["regime"] == "ELASTIC"
    assert response["action"] == "CONTINUE"
    assert not detector.reset_controller.mode.is_degraded()
    
    print("  PASS: elastic_regime")
    return True


def test_warm_regime_tightens():
    """Elevated signals → WARM regime, constraints tightened."""
    detector = RegimeDetector()
    
    signals = RegimeSignals(
        hysteresis_magnitude=0.3,
        relaxation_time_seconds=5.0,
        anisotropy_score=0.35,
    )
    
    response = detector.respond(signals)
    
    assert response["regime"] == "WARM"
    assert response["action"] == "TIGHTEN"
    assert detector.reset_controller.mode.is_degraded()
    
    print("  PASS: warm_regime_tightens")
    return True


def test_ductile_regime_resets():
    """High signals → DUCTILE regime, mandatory reset."""
    detector = RegimeDetector()
    
    signals = RegimeSignals(
        hysteresis_magnitude=0.6,
        relaxation_time_seconds=15.0,
        anisotropy_score=0.55,
        budget_pressure=0.75,
    )
    
    response = detector.respond(signals)
    
    assert response["regime"] == "DUCTILE"
    assert response["action"] == "RESET"
    assert "reset_events" in response
    assert len(response["reset_events"]) >= 2  # Context + Mode reset
    
    # Mode should be heavily degraded
    mode = detector.reset_controller.mode
    assert mode.readonly_mode
    assert mode.variety_multiplier < 1.0
    
    print("  PASS: ductile_regime_resets")
    return True


def test_unstable_regime_emergency():
    """Tool gain >= 1 → UNSTABLE regime, emergency stop."""
    detector = RegimeDetector()
    
    signals = RegimeSignals(
        tool_gain_estimate=1.2,  # k > 1 = unstable
    )
    
    response = detector.respond(signals)
    
    assert response["regime"] == "UNSTABLE"
    assert response["action"] == "EMERGENCY_STOP"
    assert response.get("escalate") == True
    
    print("  PASS: unstable_regime_emergency")
    return True


def test_regime_transition_logged():
    """Regime transitions are logged."""
    detector = RegimeDetector()
    
    # Start ELASTIC
    detector.respond(RegimeSignals(hysteresis_magnitude=0.1))
    assert len(detector.transition_history) == 0  # Started ELASTIC
    
    # Move to WARM
    detector.respond(RegimeSignals(hysteresis_magnitude=0.3, anisotropy_score=0.35))
    assert len(detector.transition_history) == 1
    assert detector.transition_history[0].from_regime == OperationalRegime.ELASTIC
    assert detector.transition_history[0].to_regime == OperationalRegime.WARM
    
    print("  PASS: regime_transition_logged")
    return True


def test_reset_types():
    """Different reset types have different effects."""
    controller = ResetController()
    controller.create_checkpoint("CP-1")
    
    # Context reset - clears working state
    event = controller.context_reset(
        regime="WARM",
        signals={"hysteresis": 0.3},
        reason="test",
    )
    assert event.reset_type == ResetType.CONTEXT
    assert not controller.mode.is_degraded()  # Mode unchanged
    
    # Mode reset - degrades capabilities
    event = controller.mode_reset(
        regime="DUCTILE",
        signals={"hysteresis": 0.6},
        reason="test",
        degrade_level=1,
    )
    assert event.reset_type == ResetType.MODE
    assert controller.mode.is_degraded()
    assert controller.mode.readonly_mode
    
    # Chain reset - rolls back to checkpoint
    controller.create_checkpoint("CP-2")
    event = controller.chain_reset(
        regime="UNSTABLE",
        signals={"tool_gain": 1.5},
        reason="test",
        checkpoint_id="CP-1",
    )
    assert event.reset_type == ResetType.CHAIN
    assert controller.current_checkpoint == "CP-1"
    
    print("  PASS: reset_types")
    return True


def test_coupling_reduction_enforced():
    """Interventions must reduce at least one coupling dimension."""
    
    # Good: reduces horizon and variety
    before = {"horizon_turns": 10, "variety_multiplier": 1.0}
    after = {"horizon_turns": 5, "variety_multiplier": 0.5}
    reduced, dims = check_coupling_reduction(before, after)
    assert reduced == True
    assert "temporal (horizon)" in dims
    assert "variety (reduced)" in dims
    
    # Good: disables tools
    before = {"tools_enabled": True}
    after = {"tools_enabled": False}
    reduced, dims = check_coupling_reduction(before, after)
    assert reduced == True
    assert "tool (disabled)" in dims
    
    # Bad: nothing reduced
    before = {"horizon_turns": 10}
    after = {"horizon_turns": 10}
    reduced, dims = check_coupling_reduction(before, after)
    assert reduced == False
    assert len(dims) == 0
    
    print("  PASS: coupling_reduction_enforced")
    return True


def test_no_silent_recovery():
    """All resets emit events (no silent recovery)."""
    detector = RegimeDetector()
    
    # Move through regimes
    detector.respond(RegimeSignals(hysteresis_magnitude=0.1))  # ELASTIC
    detector.respond(RegimeSignals(hysteresis_magnitude=0.3, anisotropy_score=0.35))  # WARM
    detector.respond(RegimeSignals(hysteresis_magnitude=0.6, relaxation_time_seconds=15.0, 
                                   anisotropy_score=0.55, budget_pressure=0.75))  # DUCTILE
    
    # Should have reset events
    reset_history = detector.reset_controller.reset_history
    assert len(reset_history) >= 2, f"Expected resets, got {len(reset_history)}"
    
    # Each reset should have trigger info
    for event in reset_history:
        assert event.trigger_regime is not None
        assert event.trigger_signals is not None
        assert event.trigger_reason is not None
    
    print("  PASS: no_silent_recovery")
    return True


def run_all_tests():
    """Run all regime/reset tests."""
    print("\n" + "="*60)
    print("REGIME DETECTION & RESET TESTS")
    print("="*60 + "\n")
    
    tests = [
        test_elastic_regime,
        test_warm_regime_tightens,
        test_ductile_regime_resets,
        test_unstable_regime_emergency,
        test_regime_transition_logged,
        test_reset_types,
        test_coupling_reduction_enforced,
        test_no_silent_recovery,
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
            print(f"         {type(e).__name__}: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n✓ All regime/reset tests passed")
        print("  - Regime classification works")
        print("  - Auto-response mapping works")
        print("  - Reset operations work")
        print("  - Coupling reduction enforced")
        print("  - No silent recovery")
    
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    passed = run_all_tests()
    sys.exit(0 if passed else 1)
