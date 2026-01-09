# DESIGN_RATIONALE.md — Why It's Weird

This document explains the non-obvious design decisions.
Read this before "fixing" something that looks wrong.

---

## Why Identity is Ledger-Controlled

**The naive approach:** Identity lives in memory. Router decides.

**The problem:** Memory is ephemeral. If the router says "these are the same proposition" and then crashes, you've lost that decision. Worse, if you restart with a different router version, you might make different decisions.

**Our approach:** Router *proposes* bindings. Ledger *records* them as `PROP_BIND` / `PROP_REBIND` events. The ledger is the only source of truth.

**Why it matters:**
- Auditable: You can replay identity decisions.
- Revocable: `PROP_SPLIT` can undo a bad merge.
- Portable: The ledger survives tool changes.

---

## Why Router Thresholds Exist

**The naive approach:** Use embeddings. Cosine > 0.9 = same meaning.

**The problem:** Embeddings are oracles. They say "similar" but not "why." When they're wrong, you can't debug it. And they're often wrong in subtle ways that matter (negation, quantifiers, temporal shifts).

**Our approach:** Explicit scoring with interpretable components:
- Entity similarity (Jaccard)
- Predicate match (exact)
- Value similarity (type-aware)

**The thresholds:**
- `≥ 0.92`: Auto-rebind (high confidence)
- `0.80–0.92`: Arbitrate (gray zone)
- `< 0.80`: New identity

**Why 0.92?** It's arbitrary but conservative. We'd rather mint too many identities than merge things that shouldn't be merged. Overbinding is harder to fix than underbinding.

---

## Why We Avoid Embeddings-as-Oracle

**The appeal:** Embeddings capture semantic similarity! Just use them!

**The problems:**
1. **Black box:** When it's wrong, you can't explain why.
2. **Symmetric:** "2022" and "October 2022" might have high cosine similarity, but the direction matters.
3. **Quantifier-blind:** "some" and "all" might embed similarly.
4. **Vendor lock-in:** Your identity decisions depend on a specific model.

**Our approach:** Embeddings are an *optional second signal* for the gray zone, not the primary mechanism. Core identity is determined by structured comparison.

---

## Why Telemetry is Derived from Ledger, Not Rehydration

**The naive approach:** Keep full claim objects in memory. Compute metrics on demand.

**The problems:**
1. **Memory:** Active claims can be large.
2. **Consistency:** In-memory state can drift from ledger.
3. **Replayability:** Can't reconstruct what metrics looked like at step N.

**Our approach:** Telemetry is a streaming view over ledger entries.
- `tail -f ledger.jsonl | telemetry_index.update(entry)`
- Minimal state: active claim IDs, birth steps, grounded status.
- No claim text, no entity index.

**Why it matters:**
- **O(active_claims)** space, not O(all_claims).
- **Replayable:** Re-run telemetry from ledger to debug.
- **Separation:** Telemetry watches, never modifies.

---

## Why Polarity Detection is Span-Local

**The bug we avoided:** "Python is fast, not slow." → polarity = -1 for everything.

**The fix:** Only check for negation within the matched span, not the whole sentence.

**Why it's subtle:** English negation has scope. "not" in one clause doesn't negate another. A global negation scan will produce garbage.

---

## Why Quantifier Detection Ignores "a/an" in Most Cases

**The bug we avoided:** "Python is a language." → quantifier = EXISTS.

**The problem:** "a" appears constantly in English prose. Matching it everywhere produces noise.

**The fix:** Only treat "a/an" as a quantifier if it's attached to the entity NP: "A user reported..." vs "Python is a language."

---

## Why Date Binding is Asymmetric

**The naive approach:** "2022" ≈ "October 2022" (same year, close enough).

**The problem:** The *direction* matters.
- "October 2022" → "2022" is information loss (acceptable).
- "2022" → "October 2022" is information gain (suspicious).

**Our approach:**
- **Info loss:** Score 0.88, allow rebind.
- **Info gain:** Score 0.55, force arbitration.

**Why it matters:** Information gain is where hallucinations hide. The model "helpfully" adds specificity that wasn't in the source.

---

## Why We Store Raw Values AND Normalized Values

**The naive approach:** Just store normalized. It's smaller.

**The problem:** Normalization collapses differences you need to detect.
- "October 2022" and "late 2022" both normalize to "YEAR:2022".
- Without raw values, you can't detect value drift.

**Our approach:**
- `value_norm`: Used for hashing (coarse grouping).
- `value_raw`: Used for drift detection (fine-grained).
- `value_features`: Structured attributes (year, month, modifier).

---

## Why Split Needs Per-Hash Metadata

**The bug we fixed:** Split copied the parent's canonical triple to the new record.

**The problem:** If you merged "October 2022" and "November 2022" into one identity, then split, the split record would have "October 2022" — the parent's value, not the hash's own value.

**The fix:** Store `HashMeta` for each bound hash. Split uses the hash's own metadata.

---

## Why Collision Budget Exists

**The problem:** One `prop_id` can become a bucket for unrelated claims if the router is too aggressive.

**The symptom:** `unique_value_signatures` grows. Different months, different modifiers, all "same proposition."

**Our approach:** Track per-identity metrics:
- `bind_count`
- `unique_value_signatures`
- `recent_bind_rate`

Penalize scoring when a candidate is "overbinding."

---

## Known Rejected Alternatives

### 1. "Just Use LLM to Classify Meaning"
**Rejected because:** Non-deterministic, slow, vendor-locked, not auditable.

### 2. "Store All Claims in a Vector DB"
**Rejected because:** Embedding-as-oracle problem. Also, overkill for this use case.

### 3. "Make Router Authoritative (No Ledger Events)"
**Rejected because:** In-memory identity is fragile. Can't audit, can't replay, can't revoke.

### 4. "Symmetric Specificity (Both Directions OK)"
**Rejected because:** Enables paraphrase laundering. Info gain is where hallucinations hide.

### 5. "Global Negation Scan"
**Rejected because:** Scope matters. "not" in clause A doesn't negate clause B.

---

## When to Re-Litigate

These decisions are worth revisiting if:
- **Embeddings improve dramatically** at capturing negation/quantifiers.
- **Performance becomes critical** and structured comparison is too slow.
- **Domain changes** (e.g., code vs prose have different norms).

But don't re-litigate just because it "feels simpler." The complexity is load-bearing.

---

*Last updated: 2025-12-29*
