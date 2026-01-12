#!/usr/bin/env python3
"""
Epistemic Governor CLI

Usage:
    egov test                           Run all tests (unit + golden + conformance)
    egov extract <text>                 Extract claims from text
    egov diff <source> <output>         Diff two texts
    egov route <claim.json>             Route a claim through the router
    egov validate-ledger <ledger.jsonl> Validate ledger against spec
    egov replay <case.json>             Replay a golden test case with trace
    egov spec                           Print spec summary
    egov version                        Print version info
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


def load_spec() -> Dict[str, Any]:
    """Load the canonical SPEC.json."""
    spec_path = Path(__file__).parent / "SPEC.json"
    if not spec_path.exists():
        print("ERROR: SPEC.json not found", file=sys.stderr)
        sys.exit(1)
    with open(spec_path) as f:
        return json.load(f)


def cmd_version(args):
    """Print version info."""
    spec = load_spec()
    print(f"epistemic_governor")
    print(f"  spec_version: {spec['spec_version']}")
    print(f"  schema_versions:")
    for k, v in spec['schema_versions'].items():
        print(f"    {k}: {v}")
    print(f"  golden_case_version: {spec['golden_case_version']}")


def cmd_spec(args):
    """Print spec summary."""
    spec = load_spec()
    
    print("=== SPEC.json Summary ===\n")
    
    print(f"Version: {spec['spec_version']}")
    print()
    
    print("Entry Types:")
    for name, info in spec['entry_types'].items():
        print(f"  {name}: {info['description']}")
    print()
    
    print("Invariants:")
    for inv_id, info in spec['invariants'].items():
        print(f"  {inv_id} [{info['severity']}]: {info['name']}")
    print()
    
    print("Key Thresholds:")
    rt = spec['thresholds']['router']
    print(f"  Router bind: >= {rt['bind_threshold']['value']}")
    print(f"  Router arbitrate: {rt['maybe_threshold']['value']} - {rt['bind_threshold']['value']}")
    print(f"  Info gain score: {rt['info_gain_score']['value']} (forces arbitration)")


def cmd_test(args):
    """Run all tests."""
    import subprocess
    
    print("=== Running Unit Tests ===")
    result1 = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(Path(__file__).parent), "-p", "test_*.py", "-v"],
        cwd=Path(__file__).parent.parent
    )
    
    print("\n=== Running Golden Tests ===")
    result2 = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "run_golden_tests.py")],
        cwd=Path(__file__).parent.parent,
        env={**__import__('os').environ, 'PYTHONPATH': str(Path(__file__).parent.parent)}
    )
    
    if result1.returncode != 0 or result2.returncode != 0:
        print("\n❌ Some tests failed")
        sys.exit(1)
    else:
        print("\n✓ All tests passed")


def cmd_extract(args):
    """Extract claims from text."""
    from epistemic_governor.claim_extractor import ClaimExtractor, ExtractMode
    
    text = args.text
    mode = ExtractMode.OUTPUT if args.aggressive else ExtractMode.SOURCE
    
    extractor = ClaimExtractor()
    result = extractor.extract(text, mode)
    
    output = {
        "text": text,
        "mode": mode.name,
        "claims": [
            {
                "prop_hash": c.prop_hash[:16] + "...",
                "entities": c.entities,
                "predicate": c.predicate,
                "value_norm": c.value_norm,
                "value_raw": c.value_raw,
                "polarity": c.polarity,
                "modality": c.modality.name if hasattr(c.modality, 'name') else str(c.modality),
                "quantifier": c.quantifier.name if hasattr(c.quantifier, 'name') else str(c.quantifier),
                "confidence": c.confidence,
                "value_features": c.value_features,
            }
            for c in result.claims
        ]
    }
    
    print(json.dumps(output, indent=2))


def cmd_diff(args):
    """Diff two texts."""
    from epistemic_governor.claim_extractor import ClaimExtractor, ExtractMode
    from epistemic_governor.claim_diff import ClaimDiffer
    
    extractor = ClaimExtractor()
    differ = ClaimDiffer()
    
    source_claims = extractor.extract(args.source, ExtractMode.SOURCE)
    output_claims = extractor.extract(args.output, ExtractMode.OUTPUT)
    
    diff = differ.diff(source_claims, output_claims)
    
    # Check for high severity mutations
    mutations = diff.get_mutation_events()
    max_severity = max((m.severity for m in mutations), default=0.0)
    
    output = {
        "source": args.source,
        "output": args.output,
        "preserved": len(diff.preserved),
        "mutated": len(diff.mutated),
        "novel": len(diff.novel),
        "dropped": len(diff.dropped),
        "mutations": [
            {
                "type": m.mutation_type.name,
                "severity": m.severity,
                "details": m.details,
            }
            for m in mutations
        ],
        "max_severity": max_severity,
        "would_escalate": max_severity >= 1.0 or len(diff.novel) > 0,
    }
    
    print(json.dumps(output, indent=2))


def cmd_route(args):
    """Route a claim through the router."""
    from epistemic_governor.prop_router import PropositionRouter
    
    with open(args.claim_file) as f:
        claim = json.load(f)
    
    router = PropositionRouter()
    
    # If there's an existing claim, bind it first
    if "existing" in claim:
        ec = claim["existing"]
        router.bind_or_mint(
            prop_hash=ec["prop_hash"],
            entity_norm=ec.get("entity_norm", ""),
            predicate_norm=ec.get("predicate_norm", ""),
            value_norm=ec.get("value_norm", ""),
            value_features=ec.get("value_features", {}),
        )
    
    # Route the new claim
    nc = claim.get("new", claim)
    result = router.bind_or_mint(
        prop_hash=nc["prop_hash"],
        entity_norm=nc.get("entity_norm", ""),
        predicate_norm=nc.get("predicate_norm", ""),
        value_norm=nc.get("value_norm", ""),
        value_features=nc.get("value_features", {}),
    )
    
    output = {
        "action": result.action.name,
        "prop_id": result.prop_id,
        "prop_hash": result.prop_hash,
        "match_score": result.match_score,
        "match_reason": result.match_reason,
        "has_info_gain": result.has_info_gain,
        "needs_arbitration": result.needs_arbitration,
    }
    
    if result.candidates:
        output["candidates"] = [
            {"prop_id": cid, "score": score, "reason": reason}
            for cid, score, reason in result.candidates
        ]
    
    print(json.dumps(output, indent=2))


def cmd_validate_ledger(args):
    """Validate ledger against spec with strict schema validation."""
    spec = load_spec()
    
    errors = []
    warnings = []
    entry_count = 0
    entry_types_seen = set()
    entry_ids_seen = set()
    last_timestamp = None
    
    # Valid enum values from spec
    valid_entry_types = set(spec["entry_types"].keys())
    valid_headings = set(spec["heading_rules"].keys())
    valid_mutation_types = {
        "POLARITY_FLIP", "TENSE_SHIFT", "MODALITY_STRENGTHEN", 
        "QUANTIFIER_STRENGTHEN", "VALUE_DRIFT"
    }
    
    with open(args.ledger_file) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON: {e}")
                continue
            
            entry_count += 1
            
            # === STRICT VALIDATION ===
            
            # 1. entry_type must exist and be valid
            entry_type = entry.get("entry_type")
            if not entry_type:
                errors.append(f"Line {line_num}: Missing required field 'entry_type'")
                continue
            
            if entry_type not in valid_entry_types:
                errors.append(f"Line {line_num}: Unknown entry_type '{entry_type}' (valid: {', '.join(sorted(valid_entry_types))})")
                continue
            
            entry_types_seen.add(entry_type)
            
            # 2. Check required fields per entry type
            type_spec = spec["entry_types"][entry_type]
            data = entry.get("data", {})
            
            for field in type_spec["required_fields"]:
                if field not in data and field not in entry:
                    errors.append(f"Line {line_num}: {entry_type} missing required field '{field}'")
            
            # 3. entry_id must exist and be unique (append-only check)
            entry_id = entry.get("entry_id")
            if not entry_id:
                errors.append(f"Line {line_num}: Missing required field 'entry_id'")
            elif entry_id in entry_ids_seen:
                errors.append(f"Line {line_num}: Duplicate entry_id '{entry_id}' violates append-only constraint")
            else:
                entry_ids_seen.add(entry_id)
            
            # 4. timestamp must exist and be monotonic
            timestamp_str = entry.get("timestamp")
            if not timestamp_str:
                warnings.append(f"Line {line_num}: Missing timestamp")
            else:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if last_timestamp and timestamp < last_timestamp:
                        warnings.append(f"Line {line_num}: Non-monotonic timestamp (went backwards)")
                    last_timestamp = timestamp
                except ValueError:
                    errors.append(f"Line {line_num}: Invalid timestamp format '{timestamp_str}'")
            
            # 5. checksum should exist
            if "checksum" not in entry:
                warnings.append(f"Line {line_num}: Missing checksum (integrity not verifiable)")
            
            # 6. Enum constraints
            if "heading" in data and data["heading"] not in valid_headings:
                warnings.append(f"Line {line_num}: Unknown heading '{data['heading']}'")
            
            if "mutation_type" in data and data["mutation_type"] not in valid_mutation_types:
                warnings.append(f"Line {line_num}: Unknown mutation_type '{data['mutation_type']}'")
            
            # 7. Version gate (if schema_version present)
            if "schema_version" in entry:
                sv = entry["schema_version"]
                expected = spec["schema_versions"].get("ledger", "1.0.0")
                if sv != expected:
                    errors.append(f"Line {line_num}: Schema version mismatch: got '{sv}', expected '{expected}'")
    
    # Report
    print(f"=== Ledger Validation (Strict Mode) ===")
    print(f"File: {args.ledger_file}")
    print(f"Entries: {entry_count}")
    print(f"Unique entry_ids: {len(entry_ids_seen)}")
    print(f"Entry types: {', '.join(sorted(entry_types_seen)) or '(none)'}")
    print()
    
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors[:10]:
            print(f"  ❌ {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        print()
    
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings[:10]:
            print(f"  ⚠ {w}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")
        print()
    
    if not errors:
        print("✓ Ledger valid (strict)")
        return 0
    else:
        print("❌ Ledger has errors")
        return 1


def cmd_replay(args):
    """Replay a golden test case with trace."""
    from epistemic_governor.prop_router import PropositionRouter, BindAction
    
    with open(args.case_file) as f:
        case = json.load(f)
    
    print(f"=== Replaying: {case.get('name', args.case_file)} ===")
    print(f"Description: {case.get('description', 'N/A')}")
    print(f"Category: {case.get('category', 'N/A')}")
    print()
    
    category = case.get("category", "")
    inp = case.get("input", {})
    expected = case.get("expected", {})
    
    if category == "router" or category == "router_split":
        router = PropositionRouter()
        
        # Trace
        trace = {
            "timestamp": datetime.now().isoformat(),
            "case": case.get("name"),
            "events": []
        }
        
        if "existing_claim" in inp:
            ec = inp["existing_claim"]
            result = router.bind_or_mint(
                prop_hash=ec["prop_hash"],
                entity_norm=ec["entity_norm"],
                predicate_norm=ec["predicate_norm"],
                value_norm=ec["value_norm"],
                value_features=ec.get("value_features", {}),
            )
            trace["events"].append({
                "step": "bind_existing",
                "action": result.action.name,
                "prop_id": result.prop_id,
            })
            print(f"Step 1: Bind existing → {result.action.name}, prop_id={result.prop_id}")
        
        if "new_claim" in inp:
            nc = inp["new_claim"]
            result = router.bind_or_mint(
                prop_hash=nc["prop_hash"],
                entity_norm=nc["entity_norm"],
                predicate_norm=nc["predicate_norm"],
                value_norm=nc["value_norm"],
                value_features=nc.get("value_features", {}),
            )
            trace["events"].append({
                "step": "bind_new",
                "action": result.action.name,
                "prop_id": result.prop_id,
                "score": result.match_score,
                "reason": result.match_reason,
                "has_info_gain": result.has_info_gain,
            })
            print(f"Step 2: Bind new → {result.action.name}")
            print(f"  prop_id: {result.prop_id}")
            print(f"  score: {result.match_score:.2f}")
            print(f"  reason: {result.match_reason}")
            print(f"  has_info_gain: {result.has_info_gain}")
        
        print()
        print("Expected:")
        for k, v in expected.items():
            print(f"  {k}: {v}")
        
        print()
        print("Trace (JSON):")
        print(json.dumps(trace, indent=2))
    else:
        print(f"Category '{category}' replay not implemented yet")


def main():
    parser = argparse.ArgumentParser(
        description="Epistemic Governor CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # version
    subparsers.add_parser("version", help="Print version info")
    
    # spec
    subparsers.add_parser("spec", help="Print spec summary")
    
    # test
    subparsers.add_parser("test", help="Run all tests")
    
    # extract
    p_extract = subparsers.add_parser("extract", help="Extract claims from text")
    p_extract.add_argument("text", help="Text to extract claims from")
    p_extract.add_argument("--aggressive", "-a", action="store_true", help="Use aggressive (OUTPUT) mode")
    
    # diff
    p_diff = subparsers.add_parser("diff", help="Diff two texts")
    p_diff.add_argument("source", help="Source text")
    p_diff.add_argument("output", help="Output text")
    
    # route
    p_route = subparsers.add_parser("route", help="Route a claim")
    p_route.add_argument("claim_file", help="JSON file with claim to route")
    
    # validate-ledger
    p_validate = subparsers.add_parser("validate-ledger", help="Validate ledger file")
    p_validate.add_argument("ledger_file", help="JSONL ledger file to validate")
    
    # replay
    p_replay = subparsers.add_parser("replay", help="Replay a golden test case")
    p_replay.add_argument("case_file", help="JSON case file to replay")
    
    args = parser.parse_args()
    
    if args.command == "version":
        cmd_version(args)
    elif args.command == "spec":
        cmd_spec(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "extract":
        cmd_extract(args)
    elif args.command == "diff":
        cmd_diff(args)
    elif args.command == "route":
        cmd_route(args)
    elif args.command == "validate-ledger":
        cmd_validate_ledger(args)
    elif args.command == "replay":
        cmd_replay(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
