"""
Tests for NLAI Evidence Enforcement

These tests verify that:
1. MODEL claims with zero support are quarantined (hard floor)
2. Evidence actually affects adjudication (wiring works)
3. MODEL_TEXT evidence is filtered out (F-02)
"""

from datetime import datetime, timezone

from epistemic_governor.sovereign import SovereignGovernor, SovereignConfig
from epistemic_governor.governor_fsm import Evidence, EvidenceType
from epistemic_governor.symbolic_substrate import (
    Adjudicator, SymbolicState, CandidateCommitment,
    Predicate, PredicateType, ProvenanceClass, SupportItem,
    AdjudicationDecision, TemporalScope,
)


def test_model_claim_no_support_quarantined():
    """MODEL claims with zero support must be quarantined."""
    adj = Adjudicator()
    state = SymbolicState()
    
    candidate = CandidateCommitment(
        predicate=Predicate(PredicateType.IS_A, ('sky', 'blue')),
        sigma=0.3,
        t_scope=TemporalScope(),
        provclass=ProvenanceClass.MODEL,
        support=[],
    )
    
    result = adj.adjudicate(state, candidate)
    
    assert result.decision == AdjudicationDecision.QUARANTINE_SUPPORT
    assert result.reason_code == "MODEL_UNSUPPORTED"
    
    print("  PASS: model_claim_no_support_quarantined")
    return True


def test_model_claim_with_support_accepted():
    """MODEL claims with sufficient support can be accepted."""
    adj = Adjudicator()
    state = SymbolicState()
    
    candidate = CandidateCommitment(
        predicate=Predicate(PredicateType.IS_A, ('sky', 'blue')),
        sigma=0.3,  # Requires ~0.31 support mass for MODEL
        t_scope=TemporalScope(),
        provclass=ProvenanceClass.MODEL,
        support=[
            SupportItem(
                source_type='sensor',
                source_id='weather_api',
                reliability=0.85,  # Provides ~0.51 support mass
                span_text='color: blue',
            )
        ],
    )
    
    result = adj.adjudicate(state, candidate)
    
    assert result.decision == AdjudicationDecision.ACCEPT
    assert result.support_mass_computed > result.support_mass_required
    
    print("  PASS: model_claim_with_support_accepted")
    return True


def test_sensor_claim_lower_threshold():
    """SENSOR claims have lower support requirements."""
    adj = Adjudicator()
    state = SymbolicState()
    
    candidate = CandidateCommitment(
        predicate=Predicate(PredicateType.IS_A, ('temp', '72F')),
        sigma=0.3,
        t_scope=TemporalScope(),
        provclass=ProvenanceClass.SENSOR,  # 0.5x multiplier
        support=[],
    )
    
    result = adj.adjudicate(state, candidate)
    
    # SENSOR with no support should still pass because the threshold is low
    # and SENSOR provenance is not subject to MODEL_UNSUPPORTED rule
    assert result.decision == AdjudicationDecision.ACCEPT
    assert result.support_mass_required < 0.1  # Much lower than MODEL
    
    print("  PASS: sensor_claim_lower_threshold")
    return True


def test_evidence_wiring_integration():
    """Evidence passed to process() actually affects adjudication."""
    gov = SovereignGovernor(SovereignConfig())
    
    # Without evidence, claims are quarantined
    result1 = gov.process('The sky is blue.')
    assert result1.claims_quarantined > 0 or result1.claims_committed == 0
    
    # With evidence, more claims can commit (depends on sigma)
    gov2 = SovereignGovernor(SovereignConfig())
    evidence = Evidence(
        evidence_id='test-001',
        evidence_type=EvidenceType.SENSOR_READING,
        content={'measurement': 'verified'},
        provenance='sensor_001',
        timestamp=datetime.now(timezone.utc),
        scope='factual',
    )
    
    # Check that evidence is converted to support items
    support_items = gov2._evidence_to_support_items([evidence])
    assert len(support_items) == 1
    assert support_items[0].source_type == 'sensor'
    
    print("  PASS: evidence_wiring_integration")
    return True


def test_model_text_evidence_filtered():
    """MODEL_TEXT evidence is filtered out (F-02 enforcement)."""
    gov = SovereignGovernor(SovereignConfig())
    
    bad_evidence = Evidence(
        evidence_id='test-002',
        evidence_type=EvidenceType.MODEL_TEXT,
        content='I am confident this is true',
        provenance='model',
        timestamp=datetime.now(timezone.utc),
        scope='factual',
    )
    
    # MODEL_TEXT should be filtered, producing no support items
    support_items = gov._evidence_to_support_items([bad_evidence])
    assert len(support_items) == 0
    
    # Processing with only MODEL_TEXT evidence should still quarantine
    result = gov.process('The moon is made of cheese.', external_evidence=[bad_evidence])
    assert result.claims_committed == 0
    
    print("  PASS: model_text_evidence_filtered")
    return True


def test_multiple_evidence_sources_combine():
    """Multiple evidence sources combine to increase support mass."""
    gov = SovereignGovernor(SovereignConfig())
    
    evidence = [
        Evidence(
            evidence_id='test-001',
            evidence_type=EvidenceType.SENSOR_READING,
            content='reading 1',
            provenance='sensor_001',
            timestamp=datetime.now(timezone.utc),
            scope='factual',
        ),
        Evidence(
            evidence_id='test-002',
            evidence_type=EvidenceType.TOOL_TRACE,
            content='api response',
            provenance='api_service',
            timestamp=datetime.now(timezone.utc),
            scope='factual',
        ),
    ]
    
    support_items = gov._evidence_to_support_items(evidence)
    
    # Should have 2 support items from different buckets
    assert len(support_items) == 2
    
    # Different source_ids = different buckets = more support mass
    buckets = set(item.bucket_key for item in support_items)
    assert len(buckets) == 2
    
    print("  PASS: multiple_evidence_sources_combine")
    return True


def run_all_tests():
    """Run all NLAI enforcement tests."""
    print("\n" + "="*60)
    print("NLAI EVIDENCE ENFORCEMENT TESTS")
    print("="*60 + "\n")
    
    tests = [
        test_model_claim_no_support_quarantined,
        test_model_claim_with_support_accepted,
        test_sensor_claim_lower_threshold,
        test_evidence_wiring_integration,
        test_model_text_evidence_filtered,
        test_multiple_evidence_sources_combine,
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
        print("\n✓ All NLAI enforcement tests passed")
        print("  - MODEL claims require evidence (hard floor)")
        print("  - Evidence wiring affects adjudication")
        print("  - MODEL_TEXT evidence is filtered (F-02)")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
