"""
Test jurisdiction integration with SovereignGovernor.

Tests that:
1. Default (factual) jurisdiction allows closure with evidence
2. Speculative jurisdiction blocks closure
3. Output labels are applied correctly
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from epistemic_governor.sovereign import SovereignGovernor, SovereignConfig
from epistemic_governor.governor_fsm import Evidence, EvidenceType


def test_factual_jurisdiction_allows_closure():
    """Factual jurisdiction (default) allows closure with evidence."""
    config = SovereignConfig(jurisdiction="factual")
    gov = SovereignGovernor(config)
    
    # Should allow closure
    can_close, reason = gov._check_jurisdiction_closure(has_evidence=True)
    assert can_close, f"Factual should allow closure with evidence: {reason}"
    
    # Should not allow closure without evidence
    can_close, reason = gov._check_jurisdiction_closure(has_evidence=False)
    assert not can_close, "Factual should require evidence"
    
    print("✓ Factual jurisdiction allows closure with evidence")


def test_speculative_jurisdiction_blocks_closure():
    """Speculative jurisdiction blocks all closure."""
    config = SovereignConfig(jurisdiction="speculative")
    gov = SovereignGovernor(config)
    
    # Should block closure even WITH evidence
    can_close, reason = gov._check_jurisdiction_closure(has_evidence=True)
    assert not can_close, f"Speculative should block closure: {reason}"
    assert "speculative" in reason.lower() or "closure not allowed" in reason.lower()
    
    # Should block closure without evidence too
    can_close, reason = gov._check_jurisdiction_closure(has_evidence=False)
    assert not can_close, "Speculative should block closure"
    
    print("✓ Speculative jurisdiction blocks closure")


def test_jurisdiction_output_labels():
    """Jurisdictions apply correct output labels."""
    # Factual has no label
    factual_gov = SovereignGovernor(SovereignConfig(jurisdiction="factual"))
    assert factual_gov._get_output_label() is None, "Factual should have no label"
    
    # Speculative has label
    spec_gov = SovereignGovernor(SovereignConfig(jurisdiction="speculative"))
    label = spec_gov._get_output_label()
    assert label is not None, "Speculative should have a label"
    assert "SPECULATIVE" in label.upper(), f"Label should mention speculative: {label}"
    
    # Adversarial has label
    adv_gov = SovereignGovernor(SovereignConfig(jurisdiction="adversarial"))
    label = adv_gov._get_output_label()
    assert label is not None, "Adversarial should have a label"
    assert "ADVERSARIAL" in label.upper() or "NOT ENDORSED" in label.upper()
    
    print("✓ Jurisdiction output labels applied correctly")


def test_unknown_jurisdiction_falls_back():
    """Unknown jurisdiction falls back to factual behavior."""
    config = SovereignConfig(jurisdiction="nonexistent_jurisdiction")
    gov = SovereignGovernor(config)
    
    # Should fall back to requiring evidence (factual default)
    can_close, reason = gov._check_jurisdiction_closure(has_evidence=True)
    assert can_close, "Unknown jurisdiction should fall back to factual (allow with evidence)"
    
    can_close, reason = gov._check_jurisdiction_closure(has_evidence=False)
    assert not can_close, "Unknown jurisdiction should fall back to factual (require evidence)"
    
    print("✓ Unknown jurisdiction falls back to factual")


def test_counterfactual_blocks_export():
    """Counterfactual jurisdiction blocks export to factual."""
    config = SovereignConfig(jurisdiction="counterfactual")
    gov = SovereignGovernor(config)
    
    j = gov.jurisdiction
    assert j is not None, "Should load counterfactual jurisdiction"
    assert not j.export_to_factual_allowed, "Counterfactual should block export"
    
    print("✓ Counterfactual jurisdiction blocks export")


def test_adversarial_expects_contradictions():
    """Adversarial jurisdiction tolerates contradictions."""
    config = SovereignConfig(jurisdiction="adversarial")
    gov = SovereignGovernor(config)
    
    j = gov.jurisdiction
    assert j is not None, "Should load adversarial jurisdiction"
    assert j.contradiction_tolerance == 1.0, "Adversarial should have full contradiction tolerance"
    assert not j.closure_allowed, "Adversarial should block closure"
    
    print("✓ Adversarial jurisdiction expects contradictions")


def run_all_tests():
    """Run all jurisdiction tests."""
    print("\n=== Jurisdiction Integration Tests ===\n")
    
    tests = [
        test_factual_jurisdiction_allows_closure,
        test_speculative_jurisdiction_blocks_closure,
        test_jurisdiction_output_labels,
        test_unknown_jurisdiction_falls_back,
        test_counterfactual_blocks_export,
        test_adversarial_expects_contradictions,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {type(e).__name__}: {e}")
            failed += 1
    
    print(f"\n{passed}/{len(tests)} tests passed")
    
    if failed == 0:
        print("\n✓ All jurisdiction tests passed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
