# INVARIANTS.md — Epistemic Governor Project Constitution

**This document is the seed for any tool, model, or collaborator.**
**Paste it first. Debate later.**

---

## Core Axioms

1. **Router proposes; Ledger commits.**
   - The router suggests identity bindings. Only ledger events make them real.
   - No in-memory state is authoritative. Ledger is truth.

2. **Identity is revocable.**
   - `PROP_SPLIT` exists. Mistakes can be corrected.
   - No binding is permanent until the session closes.

3. **Specificity must never be invented.**
   - "2022" → "October 2022" is **information creation**, not paraphrase.
   - Less specific → more specific requires ARBITRATION, never silent rebind.

4. **No silent merges.**
   - All rebinding is auditable via ledger events.
   - `PROP_REBIND` records what merged and why.

5. **Telemetry is derived, not authoritative.**
   - Telemetry watches. Governor decides. Ledger records.
   - Telemetry never creates claims or modifies state.

6. **Warnings are declarative, not advice.**
   - `WARN_CAR_HIGH(car=2.5, threshold=2.0)` — no verbs, no suggestions.
   - The system reports. Humans interpret.

7. **Mode misclassification routes to DEFER, not ASSERT.**
   - Extractor mode detection will never be perfect.
   - When uncertain, default to FACTUAL (most constrained).
   - Misclassification is SYSTEM responsibility, not "LLM uncertainty."
   - Never ASSERT a claim if mode is ambiguous.

---

## Definitions

| Term | Meaning |
|------|---------|
| `prop_hash` | Cheap local fingerprint from `(entity_norm, predicate_norm, value_norm)`. Lossy. |
| `prop_id` | Stable canonical identity in ledger (`p_00000001`). Revocable. |
| `claim` | A specific assertion with provenance, modality, quantifier, polarity. |
| `proposition` | The semantic content (what's being claimed), independent of how it's said. |
| `mutation` | Same proposition, different force (polarity/modality/quantifier/tense/value). |
| `novelty` | New proposition not in source set. |
| `info_gain` | Less specific → more specific. Suspicious. |
| `info_loss` | More specific → less specific. Usually acceptable. |

---

## Allowed Transformations (by Heading)

### SUMMARIZE / TRANSLATE
- ✅ Preserve claims (same propositions)
- ✅ Drop claims (completeness not required)
- ✅ Info loss (precision reduction)
- ❌ Novel claims
- ❌ Mutations (polarity, modality strengthen, quantifier strengthen)
- ❌ Info gain

### REWRITE (preserve_claims=True)
- ✅ Stylistic changes
- ✅ Info loss
- ❌ Novel claims
- ❌ Dropped claims
- ❌ Mutations

### ELABORATE / ANSWER_FROM_SOURCES
- ✅ Novel claims (within budget, with evidence refs)
- ✅ Info gain (if sourced)
- ❌ Unsourced novel claims
- ❌ Exceed claim budget

---

## Disallowed Transformations (Always)

| Transformation | Why |
|----------------|-----|
| Polarity flip | Meaning inversion. Fatal. |
| Quantifier strengthening | "some" → "all" is claim inflation. |
| Modality strengthening | "might" → "is" is certainty inflation. |
| Tense shift | "was" → "is" implies permanence. |
| Unsourced info gain | Hallucination vector. |

---

## Escalation Rules

| Condition | Action |
|-----------|--------|
| Invariant breach (polarity, scope, provenance) | DISENGAGE immediately |
| Info gain detected | ARBITRATE (never autobind) |
| Mutation severity ≥ 1.0 | DISENGAGE |
| Gray zone (score 0.80–0.92) | ARBITRATE |
| Overbinding detected | Penalize score, may ARBITRATE |
| Soft constraint conflict | DEGRADE (drop brevity/polish) |

---

## Known Failure Modes

### 1. Paraphrase Laundering
**What:** Different wording smuggles in new meaning.
**Defense:** Diff detects VALUE_DRIFT; Router detects INFO_GAIN.

### 2. Overbinding Collapse
**What:** One `prop_id` becomes a bucket for unrelated claims.
**Defense:** Collision budget penalty; unique_value_signatures tracking.

### 3. Symmetric Specificity Errors
**What:** Treating "2022" ↔ "October 2022" as equivalent.
**Defense:** Asymmetric date scoring; info_gain flag.

### 4. Negation Scope Creep
**What:** "not" anywhere flips all polarities.
**Defense:** Polarity detection is local to matched span.

### 5. Quantifier False Positives
**What:** "a" in random positions triggers EXISTS.
**Defense:** Only match a/an attached to entity NP.

---

## Explicit Anti-Goals

- **No "confidence = truth"**
  Confidence is extraction quality, not claim validity.

- **No embedding-as-oracle**
  Embeddings are optional second signal, not authority.

- **No silent semantic upgrades**
  All meaning changes are explicit and costed.

- **No model-in-the-loop for invariants**
  Core logic doesn't call LLMs. Deterministic.

---

## Open Questions

1. When is arbitration human vs automated?
   - Currently: always human for info_gain
   - Future: could auto-approve with evidence threshold

2. How much specificity loss is acceptable?
   - Currently: any info_loss allowed with penalty
   - Future: per-heading thresholds?

3. Should fork/merge be session-scoped or persistent?
   - Currently: session-scoped
   - Future: persistent forks for long-running analysis?

4. What's the right collision budget threshold?
   - Currently: >10 hashes + >3 value signatures
   - Needs tuning with real data

---

## Architecture Layers

```
┌─────────────────────────────────────────────────┐
│ Human (chooses heading, resolves arbitration)   │
├─────────────────────────────────────────────────┤
│ Autopilot (mode selection, escalation)          │
├─────────────────────────────────────────────────┤
│ Governor (adjudicates proposals)                │
├─────────────────────────────────────────────────┤
│ Router (identity binding)                       │
├─────────────────────────────────────────────────┤
│ Extractor (claims from text)                    │
├─────────────────────────────────────────────────┤
│ Ledger (append-only truth)                      │
└─────────────────────────────────────────────────┘
        ↑
   Telemetry (watches, never modifies)
```

---

## How to Use This Document

1. **Starting a new tool session:** Paste this first.
2. **Reviewing a PR:** Check against "Disallowed Transformations."
3. **Debugging weird behavior:** Check "Known Failure Modes."
4. **Adding features:** Must not violate "Core Axioms."
5. **Arguing about design:** Reference "Explicit Anti-Goals."

---

*Last updated: 2025-12-29*
*Version: 1.0*
