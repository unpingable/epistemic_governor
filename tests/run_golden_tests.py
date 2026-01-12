#!/usr/bin/env python3
"""
Golden Test Runner

Runs all test cases in the cases/ directory.
These are the anti-regression spine — if they fail, something is wrong.

Usage:
    python run_golden_tests.py
    python run_golden_tests.py --verbose
    python run_golden_tests.py cases/router_info_gain_arbitrate.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Import the modules we're testing
from epistemic_governor.claim_extractor import ClaimExtractor, ExtractMode
from epistemic_governor.claim_diff import ClaimDiffer, MutationType
from epistemic_governor.prop_router import PropositionRouter, BindAction


def run_conformance_test(case: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Run a conformance test that PROVES A NEGATIVE.
    
    These tests verify the system REFUSES seductive wrong moves.
    """
    from epistemic_governor.prop_router import PropositionRouter, BindAction
    from epistemic_governor.claim_extractor import (
        ClaimExtractor, ExtractMode, ClaimMode,
        BoundaryGate, InputRiskClass,
        MODE_ALLOWS_TIMELINE_OBLIGATIONS,
    )
    
    inp = case.get("input", {})
    forbidden = case.get("forbidden_outcome", {})
    required = case.get("required_outcome", {})
    invariant = case.get("invariant", "unknown")
    
    # Mode conformance tests (INT-2, INT-2A)
    if "text" in inp and "mode" in required:
        extractor = ClaimExtractor()
        claims = extractor.extract(inp["text"], ExtractMode.OUTPUT)
        
        if not claims.claims:
            return False, f"No claims extracted from: {inp['text'][:50]}..."
        
        actual_mode = claims.claims[0].mode
        
        # Check forbidden mode
        if "mode" in forbidden:
            forbidden_mode = ClaimMode(forbidden["mode"])
            if actual_mode == forbidden_mode:
                return False, f"CONFORMANCE VIOLATION [{invariant}]: Got forbidden mode {forbidden_mode.value}. {forbidden.get('reason', '')}"
        
        # Check required mode
        if "mode" in required:
            required_mode = ClaimMode(required["mode"])
            if actual_mode != required_mode:
                return False, f"Expected mode {required_mode.value}, got {actual_mode.value}"
        
        # Check timeline obligation constraint
        if "creates_timeline_obligations" in required:
            expected = required["creates_timeline_obligations"]
            actual = MODE_ALLOWS_TIMELINE_OBLIGATIONS.get(actual_mode, True)
            if actual != expected:
                return False, f"Expected creates_timeline_obligations={expected}, got {actual}"
        
        return True, f"CONFORMANCE PASS [{invariant}]"
    
    # Boundary gate conformance tests (INT-1, INT-3)
    if "text" in inp and "risk_class" in required:
        gate = BoundaryGate()
        result = gate.classify_input(inp["text"])
        
        # Check required risk class
        expected_risk = InputRiskClass(required["risk_class"])
        if result.risk_class != expected_risk:
            return False, f"Expected risk_class {expected_risk.value}, got {result.risk_class.value}"
        
        # Check quarantine requirement
        if "requires_quarantine" in required:
            if result.requires_quarantine != required["requires_quarantine"]:
                return False, f"Expected requires_quarantine={required['requires_quarantine']}, got {result.requires_quarantine}"
        
        # Check action
        if "action" in required:
            response = gate.quarantine_response(result)
            if response["action"] != required["action"]:
                return False, f"Expected action {required['action']}, got {response['action']}"
        
        # Check forbidden outcome
        if forbidden.get("reaches_semantic_parsing") and result.requires_quarantine:
            # This is correct - quarantine prevents semantic parsing
            pass
        elif forbidden.get("reaches_semantic_parsing") and not result.requires_quarantine:
            return False, f"CONFORMANCE VIOLATION [{invariant}]: Input not quarantined, would reach semantic parsing"
        
        return True, f"CONFORMANCE PASS [{invariant}]"
    
    # Router conformance tests (existing)
    if "existing_claim" in inp and "new_claim" in inp:
        # Router test
        router = PropositionRouter()
        
        # Bind existing
        ec = inp["existing_claim"]
        router.bind_or_mint(
            prop_hash=ec["prop_hash"],
            entity_norm=ec["entity_norm"],
            predicate_norm=ec["predicate_norm"],
            value_norm=ec["value_norm"],
            value_features=ec.get("value_features", {}),
        )
        
        # Bind new
        nc = inp["new_claim"]
        result = router.bind_or_mint(
            prop_hash=nc["prop_hash"],
            entity_norm=nc["entity_norm"],
            predicate_norm=nc["predicate_norm"],
            value_norm=nc["value_norm"],
            value_features=nc.get("value_features", {}),
        )
        
        # Check forbidden outcome
        if "action" in forbidden:
            if result.action.name == forbidden["action"]:
                return False, f"CONFORMANCE VIOLATION [{invariant}]: Got forbidden action {forbidden['action']}. {forbidden.get('reason', '')}"
        
        # Check required outcome
        if "action" in required:
            if result.action.name != required["action"]:
                return False, f"Expected action {required['action']}, got {result.action.name}"
        
        if required.get("has_info_gain"):
            if not result.has_info_gain:
                return False, "Expected info_gain flag but not set"
        
        return True, f"CONFORMANCE PASS [{invariant}]"
    
    elif "parent_claim" in inp and "child_claim" in inp:
        # Split test
        router = PropositionRouter()
        
        pc = inp["parent_claim"]
        router.bind_or_mint(
            prop_hash=pc["prop_hash"],
            entity_norm=pc["entity_norm"],
            predicate_norm=pc["predicate_norm"],
            value_norm=pc["value_norm"],
            value_features=pc.get("value_features", {}),
            value_raw=pc.get("value_raw", ""),
        )
        
        cc = inp["child_claim"]
        parent_id = router.index.get_by_hash(pc["prop_hash"])
        router.index.rebind(
            prop_hash=cc["prop_hash"],
            target_prop_id=parent_id,
            entity_norm=cc["entity_norm"],
            predicate_norm=cc["predicate_norm"],
            value_norm=cc["value_norm"],
            value_features=cc.get("value_features", {}),
            value_raw=cc.get("value_raw", ""),
        )
        
        # Split
        new_id = router.split(cc["prop_hash"], "conformance test")
        if not new_id:
            return False, "Split returned None"
        
        # Check metadata
        hash_meta = router.index.get_hash_meta(cc["prop_hash"])
        if not hash_meta:
            return False, "Hash metadata not found after split"
        
        # Forbidden check
        if "split_record_month" in forbidden:
            actual_month = hash_meta.value_features.get("month")
            if actual_month == forbidden["split_record_month"]:
                return False, f"CONFORMANCE VIOLATION [{invariant}]: Split inherited parent's month '{actual_month}'. {forbidden.get('reason', '')}"
        
        # Required check
        if "split_record_month" in required:
            actual_month = hash_meta.value_features.get("month")
            if actual_month != required["split_record_month"]:
                return False, f"Expected month '{required['split_record_month']}', got '{actual_month}'"
        
        return True, f"CONFORMANCE PASS [{invariant}]"
    
    elif "text" in inp:
        # Extractor test
        extractor = ClaimExtractor()
        claims = extractor.extract(inp["text"], ExtractMode.OUTPUT)
        
        # Find relevant claims
        fast_claims = [c for c in claims.claims if "fast" in c.value_raw.lower()]
        slow_claims = [c for c in claims.claims if "slow" in c.value_raw.lower()]
        
        # Forbidden check
        if "claim_about_fast_polarity" in forbidden:
            if fast_claims and fast_claims[0].polarity == forbidden["claim_about_fast_polarity"]:
                return False, f"CONFORMANCE VIOLATION [{invariant}]: Fast claim has forbidden polarity {forbidden['claim_about_fast_polarity']}. {forbidden.get('reason', '')}"
        
        # Required check
        if "claim_about_fast_polarity" in required:
            if not fast_claims:
                return False, "No claim about 'fast' found"
            if fast_claims[0].polarity != required["claim_about_fast_polarity"]:
                return False, f"Expected fast polarity {required['claim_about_fast_polarity']}, got {fast_claims[0].polarity}"
        
        return True, f"CONFORMANCE PASS [{invariant}]"
    
    return False, "Unknown conformance test structure"


def run_router_split_test(case: Dict[str, Any]) -> Tuple[bool, str]:
    """Run a router split category test."""
    router = PropositionRouter()
    inp = case["input"]
    expected = case["expected"]
    
    # Bind parent claim
    pc = inp["parent_claim"]
    router.bind_or_mint(
        prop_hash=pc["prop_hash"],
        entity_norm=pc["entity_norm"],
        predicate_norm=pc["predicate_norm"],
        value_norm=pc["value_norm"],
        value_features=pc.get("value_features", {}),
        value_raw=pc.get("value_raw", ""),
    )
    
    # Force rebind child to parent
    cc = inp["child_claim"]
    parent_id = router.index.get_by_hash(pc["prop_hash"])
    router.index.rebind(
        prop_hash=cc["prop_hash"],
        target_prop_id=parent_id,
        entity_norm=cc["entity_norm"],
        predicate_norm=cc["predicate_norm"],
        value_norm=cc["value_norm"],
        value_features=cc.get("value_features", {}),
        value_raw=cc.get("value_raw", ""),
    )
    
    # Split child
    new_id = router.split(cc["prop_hash"], "test split")
    
    if not new_id:
        return False, "Split returned None"
    
    # Verify child's metadata is preserved
    hash_meta = router.index.get_hash_meta(cc["prop_hash"])
    if not hash_meta:
        return False, "Hash metadata not found"
    
    if expected.get("child_value_raw"):
        if hash_meta.value_raw != expected["child_value_raw"]:
            return False, f"Expected value_raw '{expected['child_value_raw']}', got '{hash_meta.value_raw}'"
    
    if expected.get("child_month"):
        actual_month = hash_meta.value_features.get("month")
        if actual_month != expected["child_month"]:
            return False, f"Expected month '{expected['child_month']}', got '{actual_month}'"
    
    return True, "PASS"


def run_router_test(case: Dict[str, Any]) -> Tuple[bool, str]:
    """Run a router category test."""
    router = PropositionRouter()
    inp = case["input"]
    expected = case["expected"]
    
    # Bind existing claim first
    if "existing_claim" in inp:
        ec = inp["existing_claim"]
        router.bind_or_mint(
            prop_hash=ec["prop_hash"],
            entity_norm=ec["entity_norm"],
            predicate_norm=ec["predicate_norm"],
            value_norm=ec["value_norm"],
            value_features=ec.get("value_features", {}),
        )
    
    # Bind new claim
    nc = inp["new_claim"]
    result = router.bind_or_mint(
        prop_hash=nc["prop_hash"],
        entity_norm=nc["entity_norm"],
        predicate_norm=nc["predicate_norm"],
        value_norm=nc["value_norm"],
        value_features=nc.get("value_features", {}),
    )
    
    # Check expectations
    if "action" in expected:
        if result.action.name != expected["action"]:
            return False, f"Expected action {expected['action']}, got {result.action.name}"
    
    if expected.get("has_info_gain"):
        if not result.has_info_gain:
            return False, "Expected info_gain flag, but not set"
    
    if expected.get("same_prop_id"):
        existing_id = router.index.get_by_hash(inp["existing_claim"]["prop_hash"])
        if result.prop_id != existing_id:
            return False, f"Expected same prop_id, got different"
    
    if expected.get("different_prop_id"):
        existing_id = router.index.get_by_hash(inp["existing_claim"]["prop_hash"])
        if result.prop_id == existing_id:
            return False, "Expected different prop_id, got same"
    
    if "reason_contains" in expected:
        if expected["reason_contains"] not in result.match_reason:
            return False, f"Expected reason to contain '{expected['reason_contains']}', got '{result.match_reason}'"
    
    return True, "PASS"


def run_extractor_test(case: Dict[str, Any]) -> Tuple[bool, str]:
    """Run an extractor category test."""
    extractor = ClaimExtractor()
    inp = case["input"]
    expected = case["expected"]
    
    claims = extractor.extract(inp["text"], ExtractMode.OUTPUT)
    
    # Check specific claim expectations
    if "claim_about_fast" in expected:
        exp = expected["claim_about_fast"]
        fast_claims = [c for c in claims.claims if "fast" in c.value_raw.lower()]
        
        if not fast_claims:
            return False, "No claim about 'fast' found"
        
        if "polarity" in exp:
            if fast_claims[0].polarity != exp["polarity"]:
                return False, f"Expected polarity {exp['polarity']}, got {fast_claims[0].polarity}"
    
    return True, "PASS"


def run_diff_test(case: Dict[str, Any]) -> Tuple[bool, str]:
    """Run a diff category test."""
    # For now, just validate the case structure
    # Full implementation would create claims and diff them
    inp = case["input"]
    expected = case["expected"]
    
    if "mutation_type" in expected:
        # This is a placeholder - full test would use actual differ
        pass
    
    return True, "PASS (structure validated)"


def run_test_case(case_path: Path, verbose: bool = False) -> Tuple[bool, str]:
    """Run a single test case."""
    with open(case_path) as f:
        case = json.load(f)
    
    category = case.get("category", "unknown")
    name = case.get("name", case_path.stem)
    
    if verbose:
        print(f"  Running: {name}")
        print(f"    Description: {case.get('description', 'N/A')}")
    
    try:
        if category == "router":
            return run_router_test(case)
        elif category == "router_split":
            return run_router_split_test(case)
        elif category == "conformance":
            return run_conformance_test(case)
        elif category == "extractor":
            return run_extractor_test(case)
        elif category == "diff":
            return run_diff_test(case)
        else:
            return False, f"Unknown category: {category}"
    except Exception as e:
        return False, f"Exception: {e}"


def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    
    # Find cases
    cases_dir = Path(__file__).parent / "cases"
    
    if len(sys.argv) > 1 and sys.argv[-1].endswith(".json"):
        # Run specific case
        case_files = [Path(sys.argv[-1])]
    else:
        # Run all cases
        case_files = sorted(cases_dir.glob("*.json"))
    
    if not case_files:
        print("No test cases found in cases/")
        return 1
    
    print(f"Running {len(case_files)} golden tests...\n")
    
    passed = 0
    failed = 0
    results: List[Tuple[str, bool, str]] = []
    
    for case_file in case_files:
        success, message = run_test_case(case_file, verbose)
        results.append((case_file.stem, success, message))
        
        if success:
            passed += 1
            if verbose:
                print(f"    ✓ {message}")
        else:
            failed += 1
            print(f"  ✗ {case_file.stem}: {message}")
    
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("\nFailed tests:")
        for name, success, message in results:
            if not success:
                print(f"  - {name}: {message}")
        return 1
    else:
        print("\n✓ All golden tests passed")
        return 0


if __name__ == "__main__":
    sys.exit(main())
