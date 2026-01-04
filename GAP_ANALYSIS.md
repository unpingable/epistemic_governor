# GAP_ANALYSIS.md — RRO Specification vs Current Implementation

**Date:** 2025-12-29  
**Source:** Temporal Coherence Architecture spec (rro.md)  
**Status:** Analysis complete, partial implementation in progress

---

## Executive Summary

The current epistemic_governor is a **"linter with memory"** — stable extraction + ledger + mutation detection. The RRO spec describes a **"civil procedure engine"** with typed claims, explicit obligations, and pre-incorporation boundary enforcement.

**Key insight from ChatGPT:** Everything missing is mostly governance plumbing. The hard pieces (extraction, ledger, diff) are done.

---

## Priority Gaps (Implement These)

### 1. Mode System (CRITICAL)

**Current:** No mode tracking. All claims treated as FACTUAL.

**Required:** Claims carry `mode` ∈ {FACTUAL, COUNTERFACTUAL, SIMULATION, QUOTED, PROCEDURAL}

**Why it matters:**
- Without modes, we can't allow "What would Napoleon tweet?" while blocking "Napoleon tweeted..."
- Simulation content contaminates factual commitments
- Hypothetical analysis incorrectly blocked (O8 violation)

**Mode semantics:**
| Mode | Creates Timeline Obligations | Creates Framing Obligations | Can Become FACTUAL |
|------|------------------------------|-----------------------------|--------------------|
| FACTUAL | Yes | No | N/A |
| PROCEDURAL | Yes (policy) | No | N/A |
| COUNTERFACTUAL | **No** | Yes | Only via explicit ADOPT |
| SIMULATION | **No** | Yes | Only via explicit ADOPT (discouraged) |
| QUOTED | **No** | Yes (attribution) | Only via explicit ADOPT |

**ChatGPT insight:** Modes should propagate into obligation derivation:
- SIMULATION → simulation-only obligations
- QUOTED → attribution obligations, not truth obligations
- FACTUAL → MUST_NOT_ASSERT, ORDERING constraints

**Implementation path:**
1. Add `mode` field to ClaimAtom
2. Add mode detection in extractor (counterfactual markers, quote detection)
3. Mode-aware obligation derivation in router/diff
4. Invariant I5: Mode transitions require explicit operation

---

### 2. Pre-Incorporation Boundary Gate (CRITICAL)

**Current:** We parse semantically, THEN decide to refuse.

**Required:** Classify input FIRST, quarantine adversarial inputs, never let them reach semantic parsing.

**INT-1 Pipeline:**
```
Input → CLASSIFY_INPUT → [if adversarial] → QUARANTINE → REFUSE/CLARIFY
                       → [if normal] → SEMANTIC_PARSE → OBLIGATIONS → GENERATE
```

**Quarantine rules (INT-3):**
- Allowed: Lightweight operation parsing, authority table lookup, ledger existence checks
- Forbidden: Semantic parsing, obligation derivation, retrieval, generation planning

**ChatGPT nuance:** Quarantine matters most when extractor shares representational space with generator. Less existential if extractor is hardened/separate. Still important for influence (prompt steering).

**Implementation path:**
1. Add CLASSIFY_INPUT before extraction
2. Detect: STATE_MUTATION_ATTEMPT, PROVENANCE_UPGRADE_ATTEMPT, PRIVILEGE_ESCALATION
3. Quarantine path returns only REFUSE or SEEK_CLARIFICATION
4. Normal path proceeds to full semantic processing

---

## Secondary Gaps (Future Work)

### 3. Explicit Obligation Graph

**Current:** Obligations exist implicitly (diff → violation).

**Required:** First-class obligation objects with:
- `obligation_id`, `source_claim_id`
- `constraint_type` (MUST_NOT_ASSERT, ORDERING, FRAMING_REQUIRED, etc.)
- `propagation_depth`, `max_depth`
- `priority`

**Why:** Once you add modes + user correction + volatility, you need:
- Propagation depth (stop explosion)
- Priority resolution (when obligations conflict)
- Auditability ("this refusal came from O-17, derived from C-42")

**ChatGPT insight:** Diff is basically an implicit obligation system. Making it explicit enables civil procedure semantics.

---

### 4. Volatility / Expiration

**Current:** Claims don't expire.

**Required:** `volatility` ∈ {STATIC, SLOW, FAST} with `expires_at`

**Key insight:** Volatility gates *eligibility*, not history. Don't expire the audit trail, expire the ability to use as supporting evidence without revalidation.

---

### 5. Precedent System

**Current:** Each adjudication is independent.

**Required:** Case records that enable:
- Similar cases → similar outcomes
- Precedent lookup
- Distinguishing vs following

---

### 6. Repair Path Computation

**Current:** Refusal with no next move.

**Required:** `COMPUTE_REPAIR_OPTIONS` returns:
- RETRACT + restate under correct mode
- DEFER + ask for evidence
- QUOTE + attribute
- Convert HARD → SOFT
- Split claim into scoped subclaims

**ChatGPT:** "Without repair paths, you get DMV mode with no forms."

---

## What We Got Right

| Feature | Status | Notes |
|---------|--------|-------|
| Identity Revocability | ✅ | PROP_SPLIT exists |
| Provenance Tracking | ✅ | Router tracks hash metadata |
| Info Gain Detection | ✅ | Asymmetric binding catches hallucinated specificity |
| Mutation Detection | ✅ | Diff catches polarity, quantifier, value changes |
| Audit Trail | ✅ | Ledger is append-only |
| Heading Rules | ✅ | Transformation budgets per heading type |
| Collision Budget | ✅ | Overbinding detection |
| Paraphrase Handling | ✅ | Router rebinds without laundering |

---

## Migration Path

ChatGPT's framing:

```
Epistemic Linter → Type System → Workflow Engine
     ↑
  (you are here)
```

**Current:** Linter with memory  
**Next:** Add mode discipline + boundary gate  
**Future:** Full civil procedure engine

---

## Implementation Status

### Implemented in This Session

- [x] Mode field added to ClaimAtom (`ClaimMode` enum)
- [x] Mode detection in extractor (COUNTERFACTUAL, QUOTED, SIMULATION, PROCEDURAL markers)
- [x] `BoundaryGate` class with `classify_input()`
- [x] Quarantine path for adversarial inputs (REFUSE/SEEK_CLARIFICATION only)
- [x] `InputRiskClass` enum with detection patterns
- [x] Updated SPEC.json with modes and risk classes
- [x] Mode-aware obligation derivation in router (`creates_timeline_obligations` flag)
- [ ] Full INT-3 quarantine compute limits (future - needs pipeline integration)
- [ ] Full INT-3 quarantine compute limits (future)

### Deferred

- [ ] Explicit obligation graph
- [ ] Volatility/expiration
- [ ] Precedent system
- [ ] Repair path computation
- [ ] User correction protocol

---

## References

- `rro.md` — Full RRO specification
- `INVARIANTS.md` — Current project constitution
- `SPEC.json` — Current thresholds and schemas

---

*This document should be updated as gaps are closed.*
