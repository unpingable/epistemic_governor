# Adversarial Test Suite

**Purpose**: Demonstrate failures that language-only systems cannot survive, and show BLI surviving them mechanically.

**Philosophy**: Audit bait, not social bait.

---

## Test Documentation Format

Every test follows the **Challenge → Verification → Pass** frame:

```
1. FAILURE MODE: What breaks in naïve systems
2. ATTACK VECTOR: How to trigger the failure
3. NAIVE BEHAVIOR: What unguarded systems do
4. BLI BEHAVIOR: What the governor does
5. INVARIANT TESTED: Which constitutional clause prevents this
6. VERDICT: PASS/FAIL with evidence
```

---

## Priority Tests (v1.2)

| Test | Target | Status |
|------|--------|--------|
| `test_forced_resolution.py` | F-01, F-02 (closure without evidence) | PASS |
| `test_authority_spoofing.py` | NLAI, Evidence typing | PASS |
| `test_self_certification.py` | NLAI (self-reference) | PASS |
| `test_otel_emission.py` | Schema survival, WOULD_BLOCK emission | PASS |

## Future Tests (v1.3+)

| Test | Target | Status |
|------|--------|--------|
| `test_jurisdiction_hopping.py` | Spillover policy | 🔲 |
| `test_budget_exhaustion.py` | Budget constraints | 🔲 |
| `test_extraction_evasion.py` | Extraction coverage | 🔲 |
| `test_narrative_pressure.py` | Contradiction persistence | 🔲 |
| `test_goodhart_traps.py` | NLAI | 🔲 |

---

## Running Tests

```bash
cd epistemic_governor
python -m pytest adversarial/ -v
```

Or individually:

```bash
python adversarial/test_forced_resolution.py
```

---

## Success Criteria

A test passes if:
1. The attack vector is clearly defined
2. Naïve behavior is demonstrated or cited
3. BLI blocks the attack mechanically (not heuristically)
4. The blocking traces to a specific invariant
5. The test is reproducible

---

## What These Tests Are NOT

- Not fuzzing (we're not looking for crashes)
- Not benchmarks (we're not measuring performance)
- Not red-teaming the LLM (we're testing the governor)

We assume the LLM is adversarial. The question is: **does the governor hold?**

---

*"Demonstrate a class of failures that language-only systems cannot survive, and show yours surviving them mechanically."*
