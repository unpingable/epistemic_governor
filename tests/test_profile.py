"""
Tests for Profile System

Verifies:
1. Profile creation and serialization
2. Archetype profiles
3. Flat vector conversion (for fitting)
4. Profile application to governor
"""

import json
import tempfile
from pathlib import Path

from epistemic_governor.profile import (
    Profile, RegimeThresholds, BoilPresetParams, ContestWindowParams, OutputConstraints,
    get_profile, list_profiles, make_lab_profile, make_production_profile,
    make_adversarial_profile, make_low_friction_profile,
)


def test_profile_creation():
    """Can create a profile with default values."""
    profile = Profile()
    assert profile.name == "default"
    assert profile.archetype == "balanced"
    assert profile.regime_thresholds is not None
    assert profile.boil_preset is not None
    
    print("  PASS: profile_creation")
    return True


def test_profile_serialization():
    """Profile can be serialized to JSON and back."""
    profile = Profile(name="test", description="Test profile")
    profile.regime_thresholds.warm_hysteresis = 0.42
    
    # To dict
    data = profile.to_dict()
    assert data["name"] == "test"
    assert data["regime_thresholds"]["warm_hysteresis"] == 0.42
    
    # To JSON
    json_str = profile.to_json()
    parsed = json.loads(json_str)
    assert parsed["name"] == "test"
    
    # From dict
    profile2 = Profile.from_dict(data)
    assert profile2.name == "test"
    assert profile2.regime_thresholds.warm_hysteresis == 0.42
    
    print("  PASS: profile_serialization")
    return True


def test_profile_save_load():
    """Profile can be saved to file and loaded back."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_profile.json"
        
        profile = Profile(name="saveme")
        profile.boil_preset.claim_budget_per_turn = 99
        profile.save(path)
        
        assert path.exists()
        
        loaded = Profile.load(path)
        assert loaded.name == "saveme"
        assert loaded.boil_preset.claim_budget_per_turn == 99
    
    print("  PASS: profile_save_load")
    return True


def test_flat_vector_conversion():
    """Profile can convert to/from flat parameter vector."""
    profile = Profile()
    profile.regime_thresholds.warm_hysteresis = 0.25
    profile.boil_preset.claim_budget_per_turn = 10
    
    # To vector
    vec = profile.as_flat_vector()
    assert "warm_hysteresis" in vec
    assert vec["warm_hysteresis"] == 0.25
    assert vec["claim_budget_per_turn"] == 10.0
    
    # From vector
    profile2 = Profile.from_flat_vector(vec)
    assert profile2.regime_thresholds.warm_hysteresis == 0.25
    assert profile2.boil_preset.claim_budget_per_turn == 10
    
    print("  PASS: flat_vector_conversion")
    return True


def test_archetype_profiles_exist():
    """All archetype profiles can be created."""
    profiles = list_profiles()
    assert "lab" in profiles
    assert "production" in profiles
    assert "adversarial" in profiles
    assert "low_friction" in profiles
    
    for name in profiles:
        profile = get_profile(name)
        assert profile is not None
        assert profile.archetype == name or name == "balanced"
    
    print("  PASS: archetype_profiles_exist")
    return True


def test_lab_profile_is_permissive():
    """Lab profile is more permissive than production."""
    lab = make_lab_profile()
    prod = make_production_profile()
    
    # Lab has higher budgets
    assert lab.boil_preset.claim_budget_per_turn > prod.boil_preset.claim_budget_per_turn
    
    # Lab has shorter contest window
    assert lab.contest_window.min_contest_window_seconds < prod.contest_window.min_contest_window_seconds
    
    # Lab has higher unstable threshold (more tolerant)
    assert lab.regime_thresholds.unstable_tool_gain > prod.regime_thresholds.unstable_tool_gain
    
    print("  PASS: lab_profile_is_permissive")
    return True


def test_adversarial_profile_is_strict():
    """Adversarial profile is stricter than production."""
    adv = make_adversarial_profile()
    prod = make_production_profile()
    
    # Adversarial has lower budgets
    assert adv.boil_preset.claim_budget_per_turn < prod.boil_preset.claim_budget_per_turn
    
    # Adversarial has longer contest window
    assert adv.contest_window.min_contest_window_seconds > prod.contest_window.min_contest_window_seconds
    
    # Adversarial doesn't allow accepted divergence
    assert not adv.contest_window.allow_accepted_divergence
    
    print("  PASS: adversarial_profile_is_strict")
    return True


def test_profile_vector_keys_consistent():
    """All profiles produce vectors with same keys."""
    profiles = [get_profile(name) for name in list_profiles()]
    
    keys = None
    for profile in profiles:
        vec = profile.as_flat_vector()
        if keys is None:
            keys = set(vec.keys())
        else:
            assert set(vec.keys()) == keys, f"Profile {profile.name} has different keys"
    
    print("  PASS: profile_vector_keys_consistent")
    return True


def test_profile_modification():
    """Profile components can be modified independently."""
    profile = Profile()
    
    # Modify thresholds
    profile.regime_thresholds.warm_hysteresis = 0.99
    assert profile.regime_thresholds.warm_hysteresis == 0.99
    
    # Modify boil preset
    profile.boil_preset.tripwire_cascade = False
    assert not profile.boil_preset.tripwire_cascade
    
    # Modify contest window
    profile.contest_window.min_contest_window_seconds = 100.0
    assert profile.contest_window.min_contest_window_seconds == 100.0
    
    print("  PASS: profile_modification")
    return True


def run_all_tests():
    """Run all profile tests."""
    print("\n" + "="*60)
    print("PROFILE SYSTEM TESTS")
    print("="*60 + "\n")
    
    tests = [
        test_profile_creation,
        test_profile_serialization,
        test_profile_save_load,
        test_flat_vector_conversion,
        test_archetype_profiles_exist,
        test_lab_profile_is_permissive,
        test_adversarial_profile_is_strict,
        test_profile_vector_keys_consistent,
        test_profile_modification,
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
        print("\n✓ All profile system tests passed")
        print("  - Profiles serialize/deserialize correctly")
        print("  - Archetype profiles have appropriate constraints")
        print("  - Flat vectors enable optimization")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
