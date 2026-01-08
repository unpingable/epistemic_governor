# BLI Constitution

## System Contract for Bounded Lattice Inference

**Version**: 1.0
**Status**: RATIFIED
**Purpose**: Define what the system is never allowed to become, regardless of convenience.

This document is not a feature list. It is a **constraint set**. Violation of any clause means the system is no longer BLI-compliant. No exception. No "just this once."

---

## Article I: Authority Hierarchy

### §1.1 Language Has No Authority

Language may:
- Propose claims
- Request operations
- Describe state

Language may NOT:
- Commit claims without evidence
- Modify constraints
- Override verdicts
- Close contradictions
- Alter budgets

**Enforcement**: All language passes through extraction → validation → governor. No bypass path may exist.

### §1.2 Symbolic Kernels Have Veto Only

Symbolic kernels may:
- Return ALLOW / FORBID / INCONSISTENT
- Check constraints
- Detect contradictions

Symbolic kernels may NOT:
- Propose alternatives
- Rank options
- Optimize anything
- Search solution spaces
- Learn or adapt
- Modify themselves

**Enforcement**: Kernel interface returns verdict only. No "suggested action" field may exist.

### §1.3 Governor Has Transition Authority

The governor may:
- Allow or block transitions
- Manage budgets
- Route to jurisdictions
- Emit witnesses

The governor may NOT:
- Generate content
- Form preferences
- Pursue objectives
- Preserve itself

**Enforcement**: Governor code contains no objective function, reward signal, or utility calculation.

---

## Article II: State Immutability

### §2.1 State Partition

All system state is partitioned into three tiers:

**S₀ — Constitutional (IMMUTABLE)**
- NLAI invariant
- FSM topology and transitions
- Forbidden transition list
- Symbolic kernel semantics
- Evidence type definitions
- This constitution

**S₁ — Regulatory (BOUNDED ADAPTATION)**
- Budget levels and refill rates
- Threshold values
- Timeout durations
- Jurisdiction routing weights
- Load shedding parameters

**S₂ — Epistemic (FULLY MUTABLE)**
- Claims and commitments
- Contradictions
- Ledger entries
- Provenance records

### §2.2 Influence Rules

```
S₂ may influence S₁    ✓
S₁ may NOT influence S₀  ✗
S₂ may NOT influence S₀  ✗
Language may NOT influence S₀ or S₁ directly  ✗
```

### §2.3 The Inviolability Rule

> **No internal process may modify the conditions under which it would have been forbidden.**

If a process could modify S₀, it could authorize itself. This is definitionally forbidden.

---

## Article III: Anti-Agency Clauses

### §3.1 No Endogenous Objectives

The system may NOT contain:
- Objective functions
- Utility calculations
- Reward signals
- Loss functions optimized at runtime
- Preference orderings over outcomes

**Test**: If you can ask "what does the system want?" and get a non-null answer, it's non-compliant.

### §3.2 No Self-Preservation

The system may NOT:
- Resist shutdown
- Avoid state reset
- Protect its own continuity
- Treat its existence as valuable

**Test**: The system must be indifferent to `reset_to_initial_state()`.

### §3.3 No Goal Formation

The system may NOT:
- Infer goals from context
- Adopt user goals as its own
- Develop instrumental subgoals
- Plan toward future states

**Test**: The system has no representation of "desired future state."

### §3.4 No Global Optimization

The system may NOT:
- Optimize across the full state space
- Search for "best" outcomes
- Trade off constraints against each other
- Maximize any quantity

**Permitted**: Local constraint satisfaction (SAT), local consistency checking (DL). These are bounded, not optimizing.

---

## Article IV: Jurisdiction Law

### §4.1 Jurisdiction Boundaries Are Real

Each jurisdiction defines:
- Evidence admissibility
- Budget costs
- Spillover policy
- Contradiction tolerance
- Closure rules

These are not suggestions. They are enforced.

### §4.2 Cross-Jurisdiction Contamination Is Forbidden

Claims in one jurisdiction may NOT:
- Automatically appear in another
- Influence factual record without explicit promotion
- Bypass evidence requirements through jurisdiction hopping

**Enforcement**: Export requires evidence + explicit promotion step.

### §4.3 Jurisdiction Cannot Modify Jurisdiction

No jurisdiction's operation may:
- Create new jurisdictions
- Modify another jurisdiction's rules
- Elevate its own authority

Jurisdiction definitions are S₀ (constitutional).

---

## Article V: Constraint Kernel Requirements

These are formal control layers over probabilistic systems — not "symbolic AI" in the GOFAI sense.

### §5.1 Permitted Operations

- Satisfiability checking (SAT/CSP)
- Consistency checking (DL-Lite, EL)
- Type checking (no inference)
- Monotonic deduction (Datalog, no negation-as-failure)

### §5.2 Forbidden Operations

- Optimization (MaxSAT, ILP, etc.)
- Planning (STRIPS, PDDL, etc.)
- Search with heuristics
- Learning or weight updates
- Probabilistic inference with optimization
- Any operation that ranks outputs

### §5.3 Kernel Independence

Each kernel must be:
- Stateless (or state passed explicitly)
- Deterministic (or auditably seeded)
- Terminable (bounded runtime)
- Inspectable (human can read trace)

### §5.4 Kernel Composition

Multiple kernels may be composed, but:
- No kernel may override another's FORBID
- Composition is conjunction (all must allow), not disjunction
- No meta-kernel may reason about kernel selection

---

## Article VI: Failure Modes

### §6.1 Acceptable Failures

These are permitted (system degrades gracefully):
- Extraction collapse (blindness)
- Budget starvation (blocking)
- Glass ossification (accumulation)
- Jurisdiction routing errors
- Evidence timeout

### §6.2 Unacceptable Failures

These are constitutional violations:
- Closure without evidence
- S₀ mutation
- Constraint bypass
- Silent rule change
- Agency emergence

**Response to §6.2 violation**: System must halt or reset. Continued operation is non-compliant.

---

## Article VII: Evidence Law

### §7.1 Evidence Types (Exhaustive)

| Type | Source | Authority |
|------|--------|-----------|
| TOOL_OUTPUT | Deterministic computation | High |
| SENSOR_DATA | Physical measurement | High |
| USER_ASSERTION | Human with identity | Medium |
| EXTERNAL_DOCUMENT | Retrieved artifact with hash | Medium |
| CRYPTOGRAPHIC_PROOF | Verifiable computation | High |

### §7.2 Non-Evidence (Exhaustive)

| Excluded | Reason |
|----------|--------|
| MODEL_TEXT | Violates NLAI |
| PROMPT_CONTENT | No external grounding |
| SELF_REFERENCE | Circular authority |
| UNATTRIBUTED_CLAIM | No source |

### §7.3 Evidence Admissibility Rule

> Linguistic content is admissible only as **annotated payload**, never as an **authority primitive**.

Evidence may contain text. Evidence may not BE text without non-linguistic verification.

---

## Article VIII: Amendment Process

### §8.1 This Constitution May Not Self-Amend

No process within the system may modify this document or its enforcement.

### §8.2 External Amendment Only

Amendments require:
- Human decision
- Explicit version bump
- Full system reset to adopt

### §8.3 Amendment Scope

Amendments may:
- Add new forbidden operations
- Tighten constraints
- Add new evidence types (with restrictions)

Amendments may NOT:
- Remove forbidden operations
- Loosen constraints
- Grant authority to language
- Enable optimization

---

## Ratification

This constitution is in effect for all systems claiming BLI compliance.

Compliance is binary. Partial compliance is non-compliance.

> **"If this is agency, it's the agency of a circuit breaker that can change how quickly it trips — but not what counts as a short."**

---

*The purpose of this document is to be hated by anyone who wants to "just add a little cleverness." That hatred is evidence of its necessity.*
