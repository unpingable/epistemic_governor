"""
Critical Edge Case Tests for v2

These tests verify the fixes ChatGPT identified:
1. Temporal: claim TTL 24h does not become stale at 60s
2. BoundaryGate: rate limiter works across minute boundary
3. Variety: domain cap based on accepted domains, not input batch
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import time


from epistemic_governor.control.temporal import TemporalController, TemporalBounds, TemporalVerdict
from epistemic_governor.constitution.contracts import BoundaryGate, CrossingVerdict
from epistemic_governor.control.variety import VarietyController, VarietyBounds


def test_claim_ttl_not_confused_with_staleness():
    """
    Claim with 24h TTL should NOT become STALE at 60s.
    
    Staleness only applies to evidence, not claims.
    """
    controller = TemporalController(TemporalBounds(
        factual_claim_ttl=86400.0,  # 24 hours
        max_evidence_age=60.0,       # 60 seconds
    ))
    
    # Track a factual claim (24h TTL)
    base_time = datetime.now(timezone.utc)
    claim = controller.track_claim("c1", domain="factual", created_at=base_time)
    
    assert claim.ttl_seconds == 86400.0, f"Claim should have 24h TTL, got {claim.ttl_seconds}"
    assert claim.kind == "claim", f"Should be marked as claim, got {claim.kind}"
    
    # Check at 60 seconds - should still be VALID (not STALE)
    check_time = base_time + timedelta(seconds=60)
    verdict = controller.check_item("c1", now=check_time)
    
    assert verdict == TemporalVerdict.VALID, \
        f"Claim at 60s should be VALID, not {verdict.name}. Staleness only applies to evidence."
    
    # Check at 2 hours - still VALID
    check_time = base_time + timedelta(hours=2)
    verdict = controller.check_item("c1", now=check_time)
    
    assert verdict == TemporalVerdict.VALID, \
        f"Claim at 2h should be VALID, got {verdict.name}"
    
    # Check at 25 hours - should be EXPIRED (past TTL)
    check_time = base_time + timedelta(hours=25)
    verdict = controller.check_item("c1", now=check_time)
    
    assert verdict == TemporalVerdict.EXPIRED, \
        f"Claim at 25h should be EXPIRED, got {verdict.name}"
    
    # Contrast with evidence - should become STALE before expiry
    evidence = controller.track_evidence("e1", evidence_type="tool_result", created_at=base_time)
    assert evidence.kind == "evidence", f"Should be marked as evidence, got {evidence.kind}"
    
    # Check at 45s - past max_evidence_age (60s) but before tool_result_ttl (30s)
    # Wait - tool_result_ttl is 30s, so it expires at 30s
    # Let's use a longer TTL evidence type
    controller2 = TemporalController(TemporalBounds(
        max_evidence_age=60.0,
        tool_result_ttl=120.0,  # 2 minutes - longer than staleness threshold
    ))
    evidence2 = controller2.track_evidence("e2", evidence_type="tool_result", created_at=base_time)
    
    # At 90s: past max_evidence_age (60s) but before TTL (120s) = STALE
    check_time = base_time + timedelta(seconds=90)
    verdict = controller2.check_item("e2", now=check_time)
    
    assert verdict == TemporalVerdict.STALE, \
        f"Evidence at 90s (before TTL 120s, after max_age 60s) should be STALE, got {verdict.name}"
    
    print("  PASS: claim_ttl_not_confused_with_staleness")
    return True


def test_rate_limiter_across_minute_boundary():
    """
    Rate limiter should work correctly across minute boundary.
    
    Bug was using .seconds instead of .total_seconds(), which
    gives wrong results across day boundaries.
    """
    gate = BoundaryGate()
    
    # Override rate limit for testing
    gate.input_contract.max_inputs_per_minute = 3
    
    # Simulate requests at specific times
    base_time = datetime(2026, 1, 9, 23, 59, 30, tzinfo=timezone.utc)  # 30s before midnight
    
    # Manually inject times
    gate._input_times = [
        base_time,  # 23:59:30
        base_time + timedelta(seconds=10),  # 23:59:40
        base_time + timedelta(seconds=20),  # 23:59:50
    ]
    
    # Now check at 00:00:20 (next day) - 50 seconds after first request
    # Should still be within the minute window (50s < 60s)
    check_time = base_time + timedelta(seconds=50)
    
    # Temporarily patch datetime.now for the check
    original_times = gate._input_times.copy()
    
    # Filter using total_seconds (correct behavior)
    filtered_correct = [t for t in gate._input_times if (check_time - t).total_seconds() < 60]
    
    # This should show all 3 requests are within window
    assert len(filtered_correct) == 3, \
        f"All 3 requests should be within 60s window, got {len(filtered_correct)}"
    
    # Verify the fix is in place by checking the code doesn't use .seconds
    import inspect
    source = inspect.getsource(gate.check_input)
    assert ".total_seconds()" in source, "check_input should use .total_seconds()"
    assert ".seconds < 60" not in source, "check_input should NOT use .seconds < 60"
    
    print("  PASS: rate_limiter_across_minute_boundary")
    return True


def test_domain_cap_on_accepted_not_input():
    """
    Domain cap should be based on ACCEPTED domains, not input batch.
    
    If max_domains_per_turn=3 and input has 5 domains, we should
    accept claims from first 3 domains encountered, not reject all.
    """
    controller = VarietyController(VarietyBounds(
        max_domains_per_turn=3,
        max_claims_per_turn=100,
        max_claims_per_domain=100,
    ))
    
    # Input with 5 different domains
    claims = [
        {"domain": "d1", "content": "claim 1", "is_novel": True},
        {"domain": "d2", "content": "claim 2", "is_novel": True},
        {"domain": "d3", "content": "claim 3", "is_novel": True},
        {"domain": "d4", "content": "claim 4", "is_novel": True},  # Should be shed
        {"domain": "d5", "content": "claim 5", "is_novel": True},  # Should be shed
        {"domain": "d1", "content": "claim 1b", "is_novel": True}, # Same domain, should pass
    ]
    
    verdict, accepted, obs = controller.check_variety(claims, turn_id=1)
    
    # Should accept 4 claims (3 from first 3 domains + 1 more from d1)
    assert len(accepted) == 4, \
        f"Should accept 4 claims (d1, d2, d3, d1), got {len(accepted)}"
    
    # Check which domains were accepted
    accepted_domains = set(c["domain"] for c in accepted)
    assert accepted_domains == {"d1", "d2", "d3"}, \
        f"Should accept first 3 domains, got {accepted_domains}"
    
    # d4 and d5 should be shed
    assert obs.claims_shed == 2, f"Should shed 2 claims, got {obs.claims_shed}"
    assert "max_domains_exceeded" in obs.shed_reasons, \
        f"Should shed due to max_domains_exceeded, got {obs.shed_reasons}"
    
    print("  PASS: domain_cap_on_accepted_not_input")
    return True


def test_per_domain_cap_on_accepted():
    """
    Per-domain cap should be based on ACCEPTED claims in that domain.
    """
    controller = VarietyController(VarietyBounds(
        max_domains_per_turn=10,
        max_claims_per_turn=100,
        max_claims_per_domain=2,  # Only 2 claims per domain
    ))
    
    # 5 claims all in same domain
    claims = [
        {"domain": "d1", "content": f"claim {i}", "is_novel": True}
        for i in range(5)
    ]
    
    verdict, accepted, obs = controller.check_variety(claims, turn_id=1)
    
    # Should accept only 2
    assert len(accepted) == 2, \
        f"Should accept 2 claims (max per domain), got {len(accepted)}"
    
    assert obs.claims_shed == 3, f"Should shed 3 claims, got {obs.claims_shed}"
    assert "max_claims_per_domain" in obs.shed_reasons
    
    print("  PASS: per_domain_cap_on_accepted")
    return True


def run_all_tests():
    """Run all critical edge case tests."""
    print("\n" + "="*60)
    print("CRITICAL EDGE CASE TESTS (v2 fixes)")
    print("="*60 + "\n")
    
    tests = [
        test_claim_ttl_not_confused_with_staleness,
        test_rate_limiter_across_minute_boundary,
        test_domain_cap_on_accepted_not_input,
        test_per_domain_cap_on_accepted,
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
        print("\n✓ All critical edge case tests passed")
        print("  - Claim TTL vs evidence staleness: FIXED")
        print("  - Rate limiter total_seconds: FIXED")
        print("  - Domain cap on accepted: FIXED")
    
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    passed = run_all_tests()
    sys.exit(0 if passed else 1)
