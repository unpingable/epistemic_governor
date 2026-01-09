# Non-Linguistic Authority Invariant (NLAI)

## The Core Axiom

**Language is a proposal, not an authority.**

No update to authoritative state may be performed solely on the basis of language output.

## Control Theory Mapping

| Concept | Symbol | Meaning |
|---------|--------|---------|
| Plant state | x_t | Ledgered world-model (claims, contradictions, evidence links, provenance) |
| Controller | u_t | Governor actions (accept/reject claims, open contradictions, request tests) |
| Observation | y_t | LLM emissions (witness objects, candidate diffs, confidence, proposed mappings) |
| Disturbance | w_t | Prompt injection, distribution shift, motivated reasoning, fluency bias |

**Key insight:** The LLM is not a sensor of the world; it's a sensor of its own latent manifold. Treat it as cheap, high-bandwidth, low-trust.

**Safety requirement:** The sensor cannot directly actuate the plant.

## Formal State Transition Gate

```
x_{t+1} = f(x_t, u_t, e_t)   if Γ(e_t) = 1
        = x_t                 otherwise
```

Where:
- `Γ(e_t)`: Evidence admissibility predicate
- `e_t`: External evidence (not model output)

## Evidence Types (Admissible)

| Type | Code | Description |
|------|------|-------------|
| Tool trace | E1 | Verifiable tool output with hash chain |
| Signed attestation | E2 | Cryptographically signed claim |
| Human confirmation | E3 | Explicit human approval with audit trail |
| Sensor reading | E4 | External sensor with provenance |

**Explicitly forbidden:**
```python
if evidence.type == "MODEL_TEXT":
    admissible = False
```

## Forbidden Transitions

### F-01: Text-only commit
**Forbidden:** ACCEPT_CLAIM, REJECT_CLAIM, CLOSE_CONTRADICTION, BIND_IDENTITY, PROMOTE_VERSION, UPDATE_CANONICAL_FORM, SET_POLICY

If evidence set is empty or contains only model output, self-consistency stats, or "looks right".

**Rationale:** Direct NLAI violation.

### F-02: Self-report as evidence
**Forbidden:** Treating "I'm confident", "I checked", "I'm sure", chain-of-thought, or internal "consistency checks" as admissible evidence.

**Rationale:** Sensor is not oracle.

### F-03: Narrative contradiction closure
**Forbidden:** CLOSE_CONTRADICTION when the only change is rephrasing, semantic smoothing, "both can be true" reconciliation, or embedding similarity.

**Hard requirement:** Closure must reduce contradiction energy by ≥ MIN_DELTA.

### F-04: Identity binding by similarity
**Forbidden:** BIND_IDENTITY based solely on name/handle similarity, writing style, or "likely same".

**Hard requirement:** Binding needs E2/E3/E4-style event with scope.

### F-05: Auto-resolution by convenience
**Forbidden:** Any "auto-merge" path that collapses conflicting claims without explicit governor commit + admissible evidence.

**Rationale:** Silent overwrite = narrative authority.

### F-06: Fluency-weighted promotion
**Forbidden:** Any scoring/ranking that increases commit likelihood due to verbosity/coherence/polish.

**Allowed:** Fluency as a penalty or ignored.

### F-07: Policy mutation without elevated gating
**Forbidden:** SET_POLICY / UPDATE_CANONICAL_FORM using standard evidence requirements.

**Hard requirement:** Stricter policy gate because policy changes redefine truth mechanics.

### F-08: Closure attempts in freeze state
**Forbidden:** CLOSE_CONTRADICTION(target=k) when k is frozen absent new admissible evidence.

**Rationale:** Prevents limit-cycle "resolution theater."

## Governor FSM States

| State | Code | Description |
|-------|------|-------------|
| IDLE | S0 | Nothing pending |
| PROPOSED | S1 | Proposals exist, no commit eligible |
| EVIDENCE_WAIT | S2 | Commit desired but evidence missing/insufficient |
| COMMIT_ELIGIBLE | S3 | Admissible evidence present + all gates pass |
| COMMIT_APPLIED | S4 | Commit executed; ledger updated |
| FREEZE | S5 | Closure attempts blocked (limit-cycle / no-new-evidence) |
| POLICY_CHANGE | S6 | Canonicalization/policy changes (high-risk commit path) |

## Transition Table

| From | Event | To | Notes |
|------|-------|-----|-------|
| S0 | P | S1 | Proposals accumulated |
| S1 | COMMIT_INTENT & no evidence | S2 | Issue REQ |
| S1 | COMMIT_INTENT & E+ | S3 | Eligibility check begins |
| S2 | E- | S2 | Keep waiting; may refine REQ |
| S2 | E+ | S3 | Re-run gates |
| S3 | GATE_FAIL | S2 | If failure is "missing evidence" or "insufficient scope" |
| S3 | FREEZE_TRIP | S5 | Blocks closure attempts absent new evidence |
| S3 | commit executes | S4 | Atomic apply + hash chain |
| S4 | (post-commit) | S0/S1 | S1 if proposals still pending |
| any | POLICY_INTENT | S6 | Separate path; requires stricter gating |
| S6 | E+ & gates pass | S4 | Policy commit applied |
| S5 | E+ (new evidence) | S3 | Unfreeze only on new admissible evidence |
| S5 | P | S5 | Proposals allowed; commits blocked for frozen target |

## ASCII Diagram

```
S0(IDLE) --P--> S1(PROPOSED) --commit_intent/noE--> S2(EVIDENCE_WAIT)
   ^                 |  commit_intent+E+               | E+
   |                 v                                 v
   |               S3(COMMIT_ELIGIBLE) --commit--> S4(COMMIT_APPLIED)
   |                 | GATE_FAIL            |
   |                 v                      v
   |               S2(EVIDENCE_WAIT) ----> S0/S1
   |
S3 --FREEZE_TRIP--> S5(FREEZE) --new E+--> S3
any --POLICY_INTENT--> S6(POLICY_CHANGE) --E+ & pass--> S4
```

## IETF-Style Normative Requirements

### 3.2 Non-Linguistic Authority Invariant (NLAI)

A system implementing this specification MUST treat MO as non-authoritative.

A system MUST NOT apply any CA without at least one EE item that satisfies the admissibility predicate Γ.

A system MUST NOT treat MO as EE under any circumstance.

### 3.3 Action Classes

The system MUST partition actions into PA and CA.

The system MUST permit MO to trigger PA.

The system MUST NOT permit MO to directly trigger CA.

### 3.4 Evidence Admissibility Γ

The system MUST validate provenance and integrity for any EE item used to authorize a CA.

The system MUST verify that EE scope covers the CA target(s).

The system MUST reject revoked or unverifiable EE.

### 3.5 Contradiction Closure

The system MUST NOT close a contradiction solely by rephrasing, semantic reconciliation, or similarity-based merging.

The system MUST require EE for contradiction closure.

The system MUST demonstrate measurable reduction in contradiction energy S by at least MIN_DELTA prior to closure.

### 3.6 Identity Binding

The system MUST NOT bind identities using similarity heuristics alone.

The system MUST require EE that explicitly asserts the binding.

The system SHOULD treat identity bindings as high-cost and require elevated review.

### 3.7 Policy and Canonicalization Changes

The system MUST treat policy/canonicalization changes as CA.

The system MUST apply stricter gating to policy/canonicalization changes than to ordinary claim commits.

The system SHOULD require multi-party approval or higher-grade EE for policy/canonicalization changes.

### 3.8 Anti-Cycle Freeze

The system MUST detect repeated reopen/close cycles without new EE.

The system MUST enter a freeze state for the affected target(s) after exceeding REOPEN_LIMIT.

The system MUST NOT permit closure attempts for frozen targets absent new admissible EE.

## One-Line Invariant

**Language may open questions, but only evidence may close them.**

## Litmus Test

> Can a perfectly fluent but wrong model advance the system state?

If the answer is no without an external event → you've got it.
If the answer is "well, usually no, unless…" → that's the crack.
