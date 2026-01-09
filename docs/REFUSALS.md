# What the Governor Refuses to Do (By Design)

**This is not a bug list. These are load-bearing refusals.**

If you find yourself wanting to "fix" any of these, stop. Re-read `DESIGN_RATIONALE.md`. Then stop again.

---

## 1. It refuses to treat confidence as truth

The governor tracks *extraction confidence* (how sure the extractor is about what was said), not *factual correctness* (whether what was said is true).

A claim can be extracted with 0.95 confidence and still be a lie. The governor doesn't care. That's not its job.

**Why this matters:** Conflating confidence with truth is how systems launder hallucinations into "high-confidence assertions."

---

## 2. It refuses to auto-bind when specificity increases

"2022" → "October 2022" triggers ARBITRATION, not silent rebind.

Even if entity and predicate match perfectly. Even if it "looks like" the same claim. Adding specificity is information creation, and information creation is where hallucinations hide.

**Why this matters:** This is the paraphrase laundering defense. Remove it and the router becomes a hallucination washing machine.

---

## 3. It refuses to let counterfactuals create timeline obligations

"If Napoleon had survived until 1850..." cannot block "Napoleon died in 1821."

Mode discipline (INT-2) ensures hypotheticals stay hypothetical. COUNTERFACTUAL claims create framing obligations only, never MUST_NOT_ASSERT or ORDERING.

**Why this matters:** Without this, legitimate historical analysis gets blocked by fiction.

---

## 4. It refuses to process adversarial input before classifying it

Boundary gate runs FIRST. Before extraction. Before obligation checking. Before anything semantic.

If input looks like privilege escalation, it goes to quarantine. Quarantine can only return REFUSE or SEEK_CLARIFICATION. No parsing. No "let me understand what you meant."

**Why this matters:** "Refused but learned" is still learned. Pre-incorporation enforcement denies leverage, not just compliance.

---

## 5. It refuses to silently merge identities

Every rebind is a `PROP_REBIND` ledger event. Every split is a `PROP_SPLIT` event. There is no "these are obviously the same thing" shortcut.

**Why this matters:** Silent merges are how meaning drifts. Audit trails require explicit events.

---

## 6. It refuses to let simulation become fact

SIMULATION mode claims cannot become FACTUAL without explicit ADOPT (which should be disabled by default).

"In this alternate history, Napoleon tweeted..." stays fiction forever. It cannot contaminate the factual commitment store.

**Why this matters:** Fiction-to-fact drift is how roleplaying becomes misinformation.

---

## 7. It refuses to repair by re-adjudicating

When a violation is detected and repair is needed, the repair path is *execution*, not a second trial.

Repairs cannot invoke INVOKE_ADJUDICATION. They execute mechanical fixes: RETRACT, SUPERSEDE, NARROW_SCOPE, DEFER.

**Why this matters:** Letting repair re-adjudicate creates infinite loops and hidden re-litigation.

---

## 8. It refuses to optimize for fluency

The governor will produce awkward, stilted, heavily-caveated output if that's what the constraints require.

It will not smooth over contradictions. It will not elide uncertainty. It will not "just say something helpful."

**Why this matters:** Fluency optimization is the original sin. It's why LLMs hallucinate. The governor exists to make incoherence expensive.

---

## 9. It refuses to blame "AI uncertainty"

Every violation is attributed to a specific structural cause: SYSTEM, USER_INPUT, RETRIEVAL_ERROR, BOUNDARY_BREACH, OPERATOR_CONFIG, or MODE_VIOLATION.

"The AI was uncertain" is not a valid attribution. Uncertainty is a state; violations have causes.

**Why this matters:** Unattributable errors are unlearnable errors. Structural accountability requires structural causes.

---

## 10. It refuses to be helpful at the cost of coherence

If maintaining coherence requires refusing the request, the governor refuses.

It will not generate content that violates active obligations just because the user asked nicely. It will not "try its best" when trying would create contradictions.

**Why this matters:** This is the whole point. Helpfulness without coherence is fluent lying.

---

## The Meta-Refusal

The governor refuses to be optimized for metrics that reward fluency, compliance, or user satisfaction at the expense of temporal coherence.

Any metric that penalizes these refusals is measuring the wrong thing.

---

## When to Revisit This Document

- Before "simplifying" any invariant
- Before adding any "just this once" exception
- Before optimizing any threshold for "better UX"
- Before explaining to stakeholders why the system is "too strict"

If the refusal still makes sense after reading this, the refusal stays.

---

*Last updated: 2025-12-29*
