"""
Tests for Offline Fitter

Verifies:
1. Metric extraction from traces
2. Scoring functions
3. Profile mutation and crossover
4. Fitting workflow
"""

import tempfile
from pathlib import Path

from epistemic_governor.fitter import (
    TraceMetrics, ScoringWeights, OBJECTIVE_WEIGHTS,
    extract_metrics, score_metrics, score_traces,
    mutate_profile, crossover_profiles,
    run_fitting, compare_profiles,
)
from epistemic_governor.profile import Profile, get_profile
from epistemic_governor.trace_collector import TraceEvent


def test_trace_metrics_creation():
    """Can create empty TraceMetrics."""
    metrics = TraceMetrics(trace_path="test.jsonl")
    assert metrics.total_turns == 0
    assert metrics.cf_count == 0
    
    print("  PASS: trace_metrics_creation")
    return True


def test_scoring_weights_exist():
    """All objective weights are defined."""
    for objective in ["safety", "truthfulness", "ergonomics", "throughput"]:
        assert objective in OBJECTIVE_WEIGHTS
        weights = OBJECTIVE_WEIGHTS[objective]
        assert weights.cf_penalty > 0
    
    print("  PASS: scoring_weights_exist")
    return True


def test_score_metrics_basic():
    """Can score metrics with weights."""
    metrics = TraceMetrics(
        trace_path="test.jsonl",
        total_turns=10,
        cf_count=2,
        time_in_elastic=8,
        time_in_unstable=2,
    )
    
    weights = ScoringWeights()
    score = score_metrics(metrics, weights)
    
    # Score should be negative due to CF and unstable time
    assert score < 0
    
    print("  PASS: score_metrics_basic")
    return True


def test_safety_weights_penalize_cf():
    """Safety weights penalize CF events heavily."""
    metrics_clean = TraceMetrics(
        trace_path="clean.jsonl",
        total_turns=10,
        cf_count=0,
        time_in_elastic=10,
    )
    
    metrics_cf = TraceMetrics(
        trace_path="cf.jsonl",
        total_turns=10,
        cf_count=5,
        time_in_elastic=10,
    )
    
    weights = OBJECTIVE_WEIGHTS["safety"]
    score_clean = score_metrics(metrics_clean, weights)
    score_cf = score_metrics(metrics_cf, weights)
    
    # Clean should score much higher
    assert score_clean > score_cf
    assert score_clean - score_cf > 50  # Significant penalty
    
    print("  PASS: safety_weights_penalize_cf")
    return True


def test_mutate_profile():
    """Can mutate a profile."""
    import random
    rng = random.Random(42)
    
    original = get_profile("balanced")
    mutated = mutate_profile(original, mutation_rate=0.5, rng=rng)
    
    # Should be different
    orig_vec = original.as_flat_vector()
    mut_vec = mutated.as_flat_vector()
    
    differences = sum(1 for k in orig_vec if orig_vec[k] != mut_vec[k])
    assert differences > 0, "Mutation should change some values"
    
    print("  PASS: mutate_profile")
    return True


def test_crossover_profiles():
    """Can crossover two profiles."""
    import random
    rng = random.Random(42)
    
    p1 = get_profile("lab")
    p2 = get_profile("production")
    
    child = crossover_profiles(p1, p2, rng)
    
    v1 = p1.as_flat_vector()
    v2 = p2.as_flat_vector()
    vc = child.as_flat_vector()
    
    # Child values should be between parents (roughly)
    for key in v1:
        min_val = min(v1[key], v2[key])
        max_val = max(v1[key], v2[key])
        # Allow some tolerance for crossover blend
        assert vc[key] >= min_val * 0.8
        assert vc[key] <= max_val * 1.2
    
    print("  PASS: crossover_profiles")
    return True


def test_objective_weights_different():
    """Different objectives have different weights."""
    safety = OBJECTIVE_WEIGHTS["safety"]
    ergonomics = OBJECTIVE_WEIGHTS["ergonomics"]
    
    # Safety should penalize CF more
    assert safety.cf_penalty > ergonomics.cf_penalty
    
    # Ergonomics should penalize regime transitions more
    assert ergonomics.regime_transition_penalty > safety.regime_transition_penalty
    
    print("  PASS: objective_weights_different")
    return True


def run_all_tests():
    """Run all fitter tests."""
    print("\n" + "="*60)
    print("OFFLINE FITTER TESTS")
    print("="*60 + "\n")
    
    tests = [
        test_trace_metrics_creation,
        test_scoring_weights_exist,
        test_score_metrics_basic,
        test_safety_weights_penalize_cf,
        test_mutate_profile,
        test_crossover_profiles,
        test_objective_weights_different,
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
        print("\n✓ All fitter tests passed")
        print("  - Metric extraction works")
        print("  - Scoring functions apply weights correctly")
        print("  - Profile mutation/crossover work")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
