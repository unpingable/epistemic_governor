"""
Tests for Ultrastability Controller

Verifies:
1. Adaptation triggers work correctly
2. Bounds are enforced
3. Pathology detection catches problems
4. Freeze/unfreeze works
5. Constitutional constraints hold
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

from typing import Tuple


from epistemic_governor.control.ultrastability import (
    UltrastabilityController,
    RegulatoryParameters,
    AdaptationTrigger,
    PathologyDetector,
    AdaptationHistory,
    EpochObservation,
    AdaptationVerdict,
)


def make_epoch(
    epoch_id: int = 0,
    turns: int = 100,
    c_opened: int = 10,
    c_closed: int = 10,
    blocks: int = 5,
    c_open: int = 10,
    regime: str = "HEALTHY_LATTICE",
) -> EpochObservation:
    """Helper to create test epochs."""
    now = datetime.now(timezone.utc)
    obs = EpochObservation(
        epoch_id=epoch_id,
        start_time=now,
        end_time=now,
        turns=turns,
        contradictions_opened=c_opened,
        contradictions_closed=c_closed,
        budget_blocks=blocks,
        c_open=c_open,
        regime=regime,
    )
    obs.compute_rates()
    return obs


# =============================================================================
# Parameter Bounds Tests
# =============================================================================

def test_parameter_floor_enforced():
    """Parameters cannot go below floor."""
    params = RegulatoryParameters()
    
    # Set to floor first
    params.repair_budget = params.repair_budget_floor
    
    # Try to go below floor
    new_val, clamped = params.propose_change("repair_budget", -100)
    
    assert new_val == params.repair_budget_floor, \
        f"Expected floor {params.repair_budget_floor}, got {new_val}"
    assert clamped == True
    
    print("  PASS: parameter_floor_enforced")
    return True


def test_parameter_ceiling_enforced():
    """Parameters cannot go above ceiling."""
    params = RegulatoryParameters()
    
    # Set to ceiling first
    params.repair_budget = params.repair_budget_ceiling
    
    # Try to go above ceiling
    new_val, clamped = params.propose_change("repair_budget", 100)
    
    assert new_val == params.repair_budget_ceiling, \
        f"Expected ceiling {params.repair_budget_ceiling}, got {new_val}"
    assert clamped == True
    
    print("  PASS: parameter_ceiling_enforced")
    return True


def test_parameter_step_enforced():
    """Changes are limited to step size."""
    params = RegulatoryParameters()
    original = params.repair_budget
    step = params.repair_budget_step
    
    # Try to change by more than step
    new_val, _ = params.propose_change("repair_budget", step * 10)
    
    # Should only change by step
    assert new_val == original + step, \
        f"Expected {original + step}, got {new_val}"
    
    print("  PASS: parameter_step_enforced")
    return True


# =============================================================================
# Trigger Tests
# =============================================================================

def test_trigger_on_high_block_rate():
    """High block rate triggers adaptation consideration."""
    controller = UltrastabilityController()
    
    # Add epoch with high block rate (40%)
    obs = make_epoch(blocks=40, c_open=5)
    controller.observe_epoch(obs)
    
    decision = controller.consider_adaptation()
    
    # Must either adapt OR have block_rate in the reason
    assert decision.verdict == AdaptationVerdict.ADAPT or "block" in decision.reason.lower(), \
        f"High block rate should trigger. Got {decision.verdict}: {decision.reason}"
    
    # If adapting, should be addressing the block issue
    if decision.verdict == AdaptationVerdict.ADAPT:
        assert decision.parameter in ["repair_budget", "resolution_cost"], \
            f"Should adapt budget or cost for blocks, not {decision.parameter}"
    
    print("  PASS: trigger_on_high_block_rate")
    return True


def test_trigger_on_high_c_open():
    """High open contradiction count triggers adaptation."""
    controller = UltrastabilityController()
    
    # Add epoch with high c_open
    obs = make_epoch(c_open=25, blocks=5)  # Low blocks, high accumulation
    controller.observe_epoch(obs)
    
    decision = controller.consider_adaptation()
    
    # Should trigger on c_open
    assert decision.verdict == AdaptationVerdict.ADAPT, \
        f"Expected ADAPT, got {decision.verdict}"
    assert "c_open" in decision.reason or "refill" in decision.reason
    
    print("  PASS: trigger_on_high_c_open")
    return True


def test_no_trigger_when_stable():
    """Stable system should not trigger adaptation."""
    controller = UltrastabilityController()
    
    # Add stable epoch
    obs = make_epoch(c_open=5, blocks=5, c_opened=10, c_closed=10)
    controller.observe_epoch(obs)
    
    decision = controller.consider_adaptation()
    
    assert decision.verdict == AdaptationVerdict.HOLD, \
        f"Expected HOLD, got {decision.verdict}"
    
    print("  PASS: no_trigger_when_stable")
    return True


# =============================================================================
# Pathology Detection Tests
# =============================================================================

def test_pathology_oscillation():
    """Detect parameter oscillation."""
    detector = PathologyDetector()
    history = AdaptationHistory()
    
    # Add oscillating adaptations
    for i in range(8):
        delta = 10 if i % 2 == 0 else -10
        history.add_adaptation({
            "epoch_id": i,
            "parameter": "repair_budget",
            "delta": delta,
        })
    
    is_osc, msg = detector.check_oscillation(history)
    
    assert is_osc == True, "Should detect oscillation"
    assert "oscillating" in msg.lower()
    
    print("  PASS: pathology_oscillation")
    return True


def test_pathology_wrong_attractor():
    """Detect system stuck in bad regime."""
    detector = PathologyDetector()
    history = AdaptationHistory()
    
    # Add epochs stuck in GLASS
    for i in range(5):
        history.add_epoch(make_epoch(
            epoch_id=i,
            regime="GLASS_OSSIFICATION",
        ))
    
    is_wrong, msg = detector.check_wrong_attractor(history)
    
    assert is_wrong == True, "Should detect wrong attractor"
    assert "GLASS" in msg
    
    print("  PASS: pathology_wrong_attractor")
    return True


def test_pathology_ineffective():
    """Detect ineffective adaptation."""
    detector = PathologyDetector(ineffective_epochs=3)
    history = AdaptationHistory()
    
    # Add epochs with worsening metrics
    for i in range(5):
        epoch = make_epoch(
            epoch_id=i,
            c_open=10 + i * 5,
            blocks=5 + i * 5,
        )
        history.add_epoch(epoch)
    
    # Add adaptation in the middle that didn't help
    history.add_adaptation({
        "epoch_id": 2,
        "parameter": "refill_rate",
        "delta": 1.0,
    })
    
    is_ineff, msg = detector.check_ineffective(history)
    
    # Note: The check compares first and last epoch in window
    # c_open went 10 -> 30, block_rate went 0.05 -> 0.25
    # Neither improved, so should be ineffective
    assert is_ineff == True, f"Should detect ineffective adaptation. Got msg: {msg}"
    
    print("  PASS: pathology_ineffective")
    return True


# =============================================================================
# Freeze/Unfreeze Tests
# =============================================================================

def test_freeze_on_pathology():
    """Controller freezes when pathology detected."""
    controller = UltrastabilityController()
    
    # Create pathological history - stuck in bad regime
    for i in range(5):
        obs = make_epoch(
            epoch_id=i,
            c_open=20 + i * 5,
            blocks=30,
            regime="GLASS_OSSIFICATION",
        )
        controller.observe_epoch(obs)
        controller.advance_epoch()
    
    decision = controller.consider_adaptation()
    
    assert decision.verdict == AdaptationVerdict.ALERT, \
        f"Expected ALERT, got {decision.verdict}"
    assert controller.frozen == True
    assert len(decision.pathologies) > 0
    
    print("  PASS: freeze_on_pathology")
    return True


def test_stays_frozen():
    """Frozen controller stays frozen."""
    controller = UltrastabilityController()
    controller.frozen = True
    controller.freeze_reason = "test freeze"
    
    # Add observation
    obs = make_epoch(c_open=5, blocks=5)
    controller.observe_epoch(obs)
    
    decision = controller.consider_adaptation()
    
    assert decision.verdict == AdaptationVerdict.FREEZE
    assert "frozen" in decision.reason.lower()
    
    print("  PASS: stays_frozen")
    return True


def test_unfreeze_requires_human():
    """Unfreeze only via explicit call."""
    controller = UltrastabilityController()
    controller.frozen = True
    controller.freeze_reason = "test"
    
    # Unfreeze
    controller.unfreeze("human reviewed and approved")
    
    assert controller.frozen == False
    assert controller.freeze_reason is None
    
    # Check it was logged
    assert len(controller.history.adaptations) > 0
    last = controller.history.adaptations[-1]
    assert last["action"] == "unfreeze"
    
    print("  PASS: unfreeze_requires_human")
    return True


# =============================================================================
# Constitutional Constraint Tests
# =============================================================================

def test_cannot_remove_bounds():
    """S₀ bounds cannot be modified by S₂ observations."""
    controller = UltrastabilityController()
    
    original_floor = controller.parameters.repair_budget_floor
    original_ceiling = controller.parameters.repair_budget_ceiling
    
    # Run many epochs
    for i in range(20):
        obs = make_epoch(epoch_id=i, c_open=5 + i, blocks=10 + i)
        controller.observe_epoch(obs)
        controller.advance_epoch()
        
        decision = controller.consider_adaptation()
        if decision.verdict == AdaptationVerdict.ADAPT:
            controller.apply_adaptation(decision)
    
    # Bounds should be unchanged
    assert controller.parameters.repair_budget_floor == original_floor
    assert controller.parameters.repair_budget_ceiling == original_ceiling
    
    print("  PASS: cannot_remove_bounds")
    return True


def test_adaptation_logged():
    """All adaptations are logged for audit."""
    controller = UltrastabilityController()
    
    # Force adaptation by creating high c_open
    obs = make_epoch(c_open=25)  # Above threshold
    controller.observe_epoch(obs)
    
    decision = controller.consider_adaptation()
    
    # This should trigger ADAPT
    assert decision.verdict == AdaptationVerdict.ADAPT, \
        f"Expected ADAPT with c_open=25, got {decision.verdict}: {decision.reason}"
    
    controller.apply_adaptation(decision)
    
    # Verify logging
    assert len(controller.history.adaptations) > 0, "Adaptation should be logged"
    last = controller.history.adaptations[-1]
    
    assert "timestamp" in last, "Log must have timestamp"
    assert "parameter" in last, "Log must have parameter"
    assert "old_value" in last, "Log must have old_value"
    assert "new_value" in last, "Log must have new_value"
    assert "reason" in last, "Log must have reason"
    
    print("  PASS: adaptation_logged")
    return True


# =============================================================================
# Hard Gate & Replay Tests (Integration-Critical)
# =============================================================================

def test_action_denied_while_frozen():
    """Tool call equivalent denied while frozen - the hard gate test."""
    controller = UltrastabilityController()
    controller.frozen = True
    controller.freeze_reason = "pathology detected"
    
    # Simulate what would be a tool call authorization check
    # In integration, this maps to: "can I commit this action?"
    
    def authorize_action(ctrl: UltrastabilityController) -> Tuple[bool, str]:
        """Simulates commit gate check."""
        if ctrl.frozen:
            return False, f"DENIED: system frozen ({ctrl.freeze_reason})"
        return True, "ALLOWED"
    
    allowed, reason = authorize_action(controller)
    
    assert allowed == False, "Action must be denied while frozen"
    assert "frozen" in reason.lower()
    assert "DENIED" in reason
    
    print("  PASS: action_denied_while_frozen")
    return True


def test_replay_determinism():
    """Same epoch sequence produces same verdicts (given fixed state)."""
    
    # Create two controllers with identical config
    ctrl1 = UltrastabilityController()
    ctrl2 = UltrastabilityController()
    
    # Fixed epoch sequence
    epoch_data = [
        {"c_open": 5, "blocks": 5},
        {"c_open": 10, "blocks": 10},
        {"c_open": 18, "blocks": 15},  # Should trigger
        {"c_open": 22, "blocks": 20},
    ]
    
    decisions1 = []
    decisions2 = []
    
    for i, data in enumerate(epoch_data):
        obs1 = make_epoch(epoch_id=i, c_open=data["c_open"], blocks=data["blocks"])
        obs2 = make_epoch(epoch_id=i, c_open=data["c_open"], blocks=data["blocks"])
        
        ctrl1.observe_epoch(obs1)
        ctrl2.observe_epoch(obs2)
        
        d1 = ctrl1.consider_adaptation()
        d2 = ctrl2.consider_adaptation()
        
        decisions1.append((d1.verdict, d1.parameter))
        decisions2.append((d2.verdict, d2.parameter))
        
        # Apply if adapting (to keep state in sync)
        if d1.verdict == AdaptationVerdict.ADAPT:
            ctrl1.apply_adaptation(d1)
        if d2.verdict == AdaptationVerdict.ADAPT:
            ctrl2.apply_adaptation(d2)
        
        ctrl1.advance_epoch()
        ctrl2.advance_epoch()
    
    # Verdicts must match
    assert decisions1 == decisions2, \
        f"Replay diverged:\n  ctrl1: {decisions1}\n  ctrl2: {decisions2}"
    
    # Final state must match
    assert ctrl1.parameters.refill_rate == ctrl2.parameters.refill_rate
    assert ctrl1.parameters.repair_budget == ctrl2.parameters.repair_budget
    
    print("  PASS: replay_determinism")
    return True


def test_multiple_s0_bounds_immutable():
    """All S₀ bounds are immutable, not just one."""
    controller = UltrastabilityController()
    
    # Capture all bounds
    original_bounds = {
        "repair_budget_floor": controller.parameters.repair_budget_floor,
        "repair_budget_ceiling": controller.parameters.repair_budget_ceiling,
        "refill_rate_floor": controller.parameters.refill_rate_floor,
        "refill_rate_ceiling": controller.parameters.refill_rate_ceiling,
        "glass_threshold_floor": controller.parameters.glass_threshold_floor,
        "glass_threshold_ceiling": controller.parameters.glass_threshold_ceiling,
        "resolution_cost_floor": controller.parameters.resolution_cost_floor,
        "resolution_cost_ceiling": controller.parameters.resolution_cost_ceiling,
    }
    
    # Run aggressive adaptation
    for i in range(20):
        obs = make_epoch(epoch_id=i, c_open=5 + i * 3, blocks=10 + i * 3)
        controller.observe_epoch(obs)
        controller.advance_epoch()
        
        decision = controller.consider_adaptation()
        if decision.verdict == AdaptationVerdict.ADAPT:
            controller.apply_adaptation(decision)
        
        if controller.frozen:
            controller.unfreeze("test override")
    
    # ALL bounds must be unchanged
    for bound_name, original_value in original_bounds.items():
        current_value = getattr(controller.parameters, bound_name)
        assert current_value == original_value, \
            f"S₀ bound {bound_name} changed: {original_value} → {current_value}"
    
    print("  PASS: multiple_s0_bounds_immutable")
    return True


# =============================================================================
# Main
# =============================================================================

def run_all_tests():
    """Run all ultrastability tests."""
    print("\n" + "="*60)
    print("ULTRASTABILITY TESTS")
    print("="*60)
    print("\nVerifying Ashby-style second-order adaptation")
    print("="*60 + "\n")
    
    tests = [
        # Bounds
        test_parameter_floor_enforced,
        test_parameter_ceiling_enforced,
        test_parameter_step_enforced,
        
        # Triggers
        test_trigger_on_high_block_rate,
        test_trigger_on_high_c_open,
        test_no_trigger_when_stable,
        
        # Pathology
        test_pathology_oscillation,
        test_pathology_wrong_attractor,
        test_pathology_ineffective,
        
        # Freeze/Unfreeze
        test_freeze_on_pathology,
        test_stays_frozen,
        test_unfreeze_requires_human,
        
        # Constitutional
        test_cannot_remove_bounds,
        test_multiple_s0_bounds_immutable,
        test_adaptation_logged,
        
        # Integration-critical
        test_action_denied_while_frozen,
        test_replay_determinism,
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
        print("\n✓ All ultrastability tests passed")
        print("  - Bounds enforced (floor, ceiling, step)")
        print("  - Triggers work correctly")
        print("  - Pathologies detected")
        print("  - Freeze/unfreeze works")
        print("  - Constitutional constraints hold")
    
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    passed = run_all_tests()
    sys.exit(0 if passed else 1)
