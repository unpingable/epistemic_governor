# QUICKSTART.md — Hit the Ground Running

## Download & Setup

### 1. File Layout

After download, your directory should look like:

```
epistemic_governor/
├── INVARIANTS.md          # Constitution (read first)
├── DESIGN_RATIONALE.md    # Why it's weird (read second)
├── SPEC.json              # Machine-readable contract
├── QUICKSTART.md          # You are here
├── README.md              # Full documentation
├── pyproject.toml         # Package metadata
│
├── egov.py                # CLI entrypoint
├── run_golden_tests.py    # Test runner
│
├── cases/                 # Golden & conformance tests
│   ├── CONFORMANCE_*.json # Prove-negative tests
│   ├── router_*.json      # Router behavior tests
│   ├── diff_*.json        # Diff behavior tests
│   └── polarity_*.json    # Extractor tests
│
├── __init__.py            # Package exports
├── __main__.py            # Module runner
│
│── # Core modules
├── ledger.py              # Append-only truth store
├── claim_extractor.py     # Extract claims from text
├── claim_diff.py          # Compare claim sets
├── prop_router.py         # Identity binding
├── streaming_telemetry.py # Derived metrics
├── autopilot_fsm.py       # Mode selection FSM
│
│── # Supporting modules
├── heading.py             # Heading types & rules
├── governor.py            # Adjudication logic
├── kernel.py              # Core orchestration
├── [... 40+ more modules]
│
└── test_*.py              # Unit tests
```

### 2. Verify Installation

```bash
cd epistemic_governor

# Run all tests
python egov.py test

# Expected output:
# Ran 31 tests in 0.009s
# OK
# Results: 10 passed, 0 failed
# ✓ All tests passed
```

### 3. CLI Quick Reference

```bash
# Version info
python egov.py version

# Spec summary (invariants, thresholds)
python egov.py spec

# Extract claims from text
python egov.py extract "Python 3.11 was released in October 2022."

# Diff two texts
python egov.py diff "Released in 2022" "Released in October 2022"

# Replay a golden test with trace
python egov.py replay cases/router_info_gain_arbitrate.json

# Validate a ledger file
python egov.py validate-ledger my_ledger.jsonl
```

### 4. Python API Quick Reference

```python
from epistemic_governor import (
    ClaimExtractor, ExtractMode,
    ClaimDiffer,
    PropositionRouter, BindAction,
    EpistemicLedger,
)

# Extract claims
extractor = ClaimExtractor()
claims = extractor.extract("Python 3.11 was released in 2022.", ExtractMode.SOURCE)
print(f"Found {len(claims.claims)} claims")

# Diff two texts
differ = ClaimDiffer()
source = extractor.extract("Released in 2022", ExtractMode.SOURCE)
output = extractor.extract("Released in October 2022", ExtractMode.OUTPUT)
diff = differ.diff(source, output)
print(f"Mutations: {len(diff.mutated)}, Novel: {len(diff.novel)}")

# Route a claim
router = PropositionRouter()
for claim in claims.claims:
    result = router.bind_or_mint(
        prop_hash=claim.prop_hash,
        entity_norm=claim.entities[0] if claim.entities else "",
        predicate_norm=claim.predicate,
        value_norm=claim.value_norm,
        value_features=claim.value_features,
    )
    print(f"Action: {result.action.name}, prop_id: {result.prop_id}")
```

---

## Key Concepts (30-second version)

| Concept | What It Is |
|---------|------------|
| `prop_hash` | Cheap fingerprint (can collide on paraphrases) |
| `prop_id` | Stable identity (survives paraphrases) |
| `info_gain` | Adding specificity (suspicious, requires arbitration) |
| `info_loss` | Removing specificity (usually OK) |
| `REBIND` | Merge hash to existing identity |
| `ARBITRATE` | Gray zone, needs human decision |
| `DISENGAGE` | Critical violation, stop generation |

---

## Before Making Changes

1. **Read `INVARIANTS.md`** — The non-negotiables
2. **Run `python egov.py test`** — Baseline
3. **Make your change**
4. **Run tests again** — Must still pass
5. **Check conformance tests** — They prove negatives

---

## Migrating to Other Tools

When using Claude Code, Copilot, or other AI tools:

1. **Paste `INVARIANTS.md` first** — It's the constitution
2. **Reference `SPEC.json`** — Machine-readable thresholds
3. **Run golden tests after changes** — `python run_golden_tests.py`
4. **Watch for conformance failures** — They catch "helpful optimizations"

---

## File Purposes

### Documents (read these)
- `INVARIANTS.md` — What MUST be true
- `DESIGN_RATIONALE.md` — Why weird things are weird
- `SPEC.json` — Thresholds, enums, schemas
- `README.md` — Full documentation

### Test Infrastructure (run these)
- `egov.py` — CLI for everything
- `run_golden_tests.py` — Golden + conformance tests
- `test_*.py` — Unit tests
- `cases/*.json` — Test fixtures

### Core Logic (modify carefully)
- `claim_extractor.py` — Claim extraction from text
- `claim_diff.py` — Claim set comparison
- `prop_router.py` — Identity binding
- `ledger.py` — Append-only storage
- `streaming_telemetry.py` — Derived metrics

### Everything Else
Supporting modules for specific features. The core logic above is what matters most.

---

## Troubleshooting

### Tests fail after download
```bash
# Make sure you're in the right directory
cd epistemic_governor
python egov.py test
```

### Import errors
```bash
# Run from parent directory with PYTHONPATH
cd ..
PYTHONPATH=. python -c "from epistemic_governor import ClaimExtractor"
```

### "Unknown entry_type" in ledger validation
Check `SPEC.json` for valid entry types. Unknown types fail strict validation.

---

## Getting Help

1. Check `DESIGN_RATIONALE.md` for "why"
2. Check `SPEC.json` for thresholds
3. Run `python egov.py replay cases/<relevant_case>.json` for traces
4. Golden tests in `cases/` show expected behavior

---

*You're ready. Go build.*
