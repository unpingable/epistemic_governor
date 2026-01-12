#!/usr/bin/env python3
"""
Quick test runner for the Epistemic Governor package.

Run with:
    cd epistemic_governor
    python -m epistemic_governor.test_all

Or from parent directory:
    PYTHONPATH=. python epistemic_governor/test_all.py
"""

import sys
import traceback
from datetime import datetime


def run_tests():
    """Run all package tests."""
    print("=" * 60)
    print("EPISTEMIC GOVERNOR - TEST SUITE")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)
    print()
    
    tests_passed = 0
    tests_failed = 0
    failures = []
    
    # Test 1: All imports
    print("1. Testing imports...")
    try:
        from epistemic_governor import (
            # Core
            StateVector, VectorController,
            RegimeDetector,
            StructuralResistance,
            # Envelope
            FlightEnvelope, FrictionLadder, HITLController,
            # API
            GovernorAPI, SCHEMA_VERSION,
            # Claims
            ClaimLedger, Provenance, EvidenceRef,
            # Tools
            ToolRegistry, ProvenanceStore,
        )
        print("   ✓ All imports successful")
        tests_passed += 1
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        tests_failed += 1
        failures.append(("imports", str(e)))
    
    # Test 2: Claim Ledger - The Money Rule
    print()
    print("2. Testing Claim Ledger (The Money Rule)...")
    try:
        from epistemic_governor import ClaimLedger, Provenance, EvidenceRef
        
        ledger = ClaimLedger()
        
        # Create assumed claim
        claim = ledger.new_assumed_claim("Test claim", confidence=0.3)
        assert claim.confidence == 0.3, "Initial confidence wrong"
        
        # Cannot increase without evidence
        ledger.update_confidence(claim.claim_id, +0.5)
        assert claim.confidence == 0.3, "Confidence increased without evidence!"
        
        # Cannot promote without evidence
        result = ledger.promote(claim.claim_id, Provenance.RETRIEVED)
        assert result.name == "NEEDS_EVIDENCE", "Should need evidence"
        
        # Attach evidence
        ledger.attach_evidence(claim.claim_id, EvidenceRef.from_tool_trace("test", "test"))
        
        # Now can promote
        result = ledger.promote(claim.claim_id, Provenance.RETRIEVED)
        assert result.name == "SUCCESS", "Should succeed with evidence"
        
        # Now confidence can increase
        ledger.update_confidence(claim.claim_id, +0.5)
        assert claim.confidence == 0.8, "Confidence should increase with evidence"
        
        print("   ✓ Money Rule enforced correctly")
        tests_passed += 1
    except Exception as e:
        print(f"   ✗ Money Rule test failed: {e}")
        tests_failed += 1
        failures.append(("money_rule", traceback.format_exc()))
    
    # Test 3: Peer Claim Resonance Prevention
    print()
    print("3. Testing Peer Claim Resonance Prevention...")
    try:
        from epistemic_governor import ClaimLedger
        
        ledger = ClaimLedger()
        
        # Peer claims are capped
        peer = ledger.new_peer_claim("Peer assertion", "agent_b", confidence=0.99)
        assert peer.confidence <= 0.3, "Peer confidence not capped!"
        
        # Cannot increase without evidence
        old = peer.confidence
        ledger.update_confidence(peer.claim_id, +0.5)
        assert peer.confidence == old, "Peer confidence increased without evidence!"
        
        print("   ✓ Resonance prevention working")
        tests_passed += 1
    except Exception as e:
        print(f"   ✗ Resonance prevention failed: {e}")
        tests_failed += 1
        failures.append(("resonance", traceback.format_exc()))
    
    # Test 4: Flight Envelope
    print()
    print("4. Testing Flight Envelope...")
    try:
        from epistemic_governor import FlightEnvelope
        
        envelope = FlightEnvelope()
        
        # High strain + high confidence = violation
        violations = envelope.check_state(factual_strain=0.7, confidence=0.85)
        assert len(violations) > 0, "Should detect violations"
        
        # Check transform blocking
        allowed, reason = envelope.check_transform(
            factual_strain=0.6,
            confidence=0.8,
            proposed_transform="DIRECT_ANSWER",
            has_grounding=False,
        )
        assert not allowed, "Should block speculative answer"
        
        print("   ✓ Flight envelope working")
        tests_passed += 1
    except Exception as e:
        print(f"   ✗ Flight envelope failed: {e}")
        tests_failed += 1
        failures.append(("envelope", traceback.format_exc()))
    
    # Test 5: Adaptive Friction
    print()
    print("5. Testing Adaptive Friction...")
    try:
        from epistemic_governor import FrictionLadder, FrictionLevel
        
        friction = FrictionLadder()
        assert friction.state.level == FrictionLevel.NORMAL
        
        # Escalate
        friction.record_forbidden_transform_attempt()
        friction.record_forbidden_transform_attempt()
        friction.record_forbidden_transform_attempt()
        
        assert friction.state.level != FrictionLevel.NORMAL, "Should escalate"
        
        # Resolution path should exist
        path = friction.get_resolution_path()
        assert len(path) > 0, "Should have resolution path"
        
        print("   ✓ Adaptive friction working")
        tests_passed += 1
    except Exception as e:
        print(f"   ✗ Adaptive friction failed: {e}")
        tests_failed += 1
        failures.append(("friction", traceback.format_exc()))
    
    # Test 6: Provenance Store
    print()
    print("6. Testing Provenance Store...")
    try:
        from epistemic_governor import ProvenanceStore
        from tools import SupportCandidate, VerificationStatus
        
        store = ProvenanceStore()
        
        candidate = SupportCandidate(
            source_uri="test:source",
            span_start=0,
            span_end=10,
            quote="Test content",
            score=0.9,
            source_type="document",
        )
        
        # Ingest
        h = store.ingest(candidate)
        assert len(h) > 0, "Should return hash"
        
        # Verify
        result = store.verify(h, current_content="Test content")
        assert result.status == VerificationStatus.VERIFIED, "Should verify"
        
        # Mismatch detection
        result = store.verify(h, current_content="Different content")
        assert result.status == VerificationStatus.MISMATCH, "Should detect mismatch"
        
        print("   ✓ Provenance store working")
        tests_passed += 1
    except Exception as e:
        print(f"   ✗ Provenance store failed: {e}")
        tests_failed += 1
        failures.append(("provenance", traceback.format_exc()))
    
    # Test 7: API
    print()
    print("7. Testing Governor API...")
    try:
        from epistemic_governor import GovernorAPI, SCHEMA_VERSION
        
        api = GovernorAPI(kernel=None)
        
        session_id = api.create_session()
        assert session_id is not None
        
        plan = api.preflight(session_id, "Test query")
        assert plan is not None
        
        trace = api.export_trace(session_id)
        assert len(trace.events) > 0, "Should have trace events"
        
        api.close_session(session_id)
        
        print(f"   ✓ API working (schema {SCHEMA_VERSION})")
        tests_passed += 1
    except Exception as e:
        print(f"   ✗ API failed: {e}")
        tests_failed += 1
        failures.append(("api", traceback.format_exc()))
    
    # Summary
    print()
    print("=" * 60)
    print(f"RESULTS: {tests_passed} passed, {tests_failed} failed")
    print("=" * 60)
    
    if failures:
        print()
        print("FAILURES:")
        for name, err in failures:
            print(f"\n--- {name} ---")
            print(err[:500])
    
    return tests_failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
