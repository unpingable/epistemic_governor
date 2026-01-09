# Epistemic Governor Roadmap

## Current State (January 2026 - v1.3)

**85+ Python files, 80+ tests passing**
**NLAI compliance verified at runtime**
**Observability layer complete (OTel projection)**
**Adversarial test suite proving invariants hold under attack**

The system enforces:
- Language is non-authoritative (proposals only)
- Evidence gates all commits
- No silent resolution
- Freeze protection against resolution theater
- Ghost beliefs blocked by construction

---

## Completed Work

### Core Architecture (v1.0)
- [x] SovereignGovernor single entrypoint
- [x] 6-state FSM with forbidden transitions
- [x] Integrity sealing (hash chains, deterministic replay)
- [x] NLAI enforcement at runtime
- [x] Contradiction persistence as first-class objects

### Empirical Validation (v1.0)
- [x] Hysteresis test harness proving I(Y; S | X) > 0
- [x] 8 workloads exercising different regimes
- [x] Budget sweep → starvation boundary at refill_rate ≈ 2.0
- [x] Cost sweep → glass boundary at resolution_cost ≈ 12.5
- [x] Negative results: capacity ≠ laundering

### Observability (v1.0-v1.3)
- [x] Phase diagnostics with 53-field DiagnosticEvent
- [x] Regime detection (6 regimes)
- [x] Energy function E(S)
- [x] DuckDB query layer with canonical queries
- [x] Transition detection with causal attribution
- [x] OTel projection layer (v1.3)
- [x] LangChain callback handler (v1.3)
- [x] Demo agent with end-to-end telemetry (v1.3)

### Jurisdictions (v1.1)
- [x] 8 jurisdiction modes (Factual, Speculative, Counterfactual, etc.)
- [x] Per-jurisdiction evidence policies, budgets, spillover rules
- [x] Integration with SovereignGovernor
- [x] 6 jurisdiction tests passing

### Adversarial Testing (v1.2-v1.3)
- [x] Forced resolution attack tests
- [x] Authority spoofing tests
- [x] Self-certification loop tests
- [x] OTel emission tests (8 tests)
- [x] All tests demonstrate NLAI holds under attack

### Documentation (v1.0-v1.3)
- [x] Paper spec (PAPER_SPEC.md) - 650+ lines, submission-ready
- [x] Standalone spec (BLI_SPEC_V1.md) - interface definition
- [x] FAQ with limits & failure modes section
- [x] Cybernetic lineage and terminology mapping
- [x] Three-cueing frame document
- [x] BLI Constitution with mHC principles (v1.1)
- [x] OTel semantic conventions spec
- [x] Documentation reorganized into docs/ directory (v1.3.1)

---

## Cybernetics TODOs / Gaps to Consider (v2+)

### Constraint Kernels (v2 Foundation) — DESIGN CLOSED
See `docs/CONSTRAINT_KERNELS.md` for full design notes.

**Decision**: SAT + DL-Lite as hostile admissibility oracles
- SAT/CSP for forbidden transition enforcement
- DL-Lite for contradiction detection
- NO optimization, NO search, NO learning
- Kernels say "no" with clarity, never "what instead"

These are formal control layers over probabilistic systems — not "symbolic AI" in the GOFAI sense.

**Key invariant**: No internal process may modify the conditions under which it would have been forbidden.

**State partition**:
- S₀ (constitutional) — immutable: NLAI, FSM, kernels, forbidden transitions
- S₁ (regulatory) — adaptive within bounds: budgets, thresholds, rates
- S₂ (epistemic) — fully mutable: claims, contradictions, ledger

### 1. Ultrastability (Ashby)
*Priority: HIGH - everything else cascades from this*

- [ ] Define conditions under which the governor itself may mutate
- [ ] Second-order adaptation when repeated repairs preserve failure
- [ ] Guardrails against infinite self-modification
- [ ] Detection of "successful failure" (system stabilizes around wrong attractor)

### 2. Variety Dial
- [ ] Explicit policy for variety absorption vs amplification
- [ ] Dynamic thresholds tied to Δt / stress / adversarial load
- [ ] Detect exploitation via over-filtering or over-expansion
- [ ] Bounded variety: min/max claims per turn, scope limits

### 3. Meta-Repair Pathology (Bateson)
- [ ] Detect Learning-II failure: repair loops that stabilize the wrong invariant
- [ ] Flag repeated contradiction resolution with no reduction in failure rate
- [ ] Escalation path when "repair itself is the bug"
- [ ] Distinguish productive cycling from pathological cycling

### 4. Interface Austerity (Simon)
- [ ] Formal inner vs outer environment boundary
- [ ] Explicit interface contracts (what can cross the boundary)
- [ ] Define inadmissible truths / forbidden actions even if effective
- [ ] Boundary hardening against prompt injection at interface level

### 5. Model Decentering (Luhmann-lite)
- [ ] Treat model outputs as perturbations, not components
- [ ] System defined by claims/contradictions/resolutions, not agents
- [ ] Support multi-model + human hybrid governance without charisma bleed
- [ ] No special status for any single model's outputs

### 6. Temporal Hard Limits (Deutsch)
- [ ] Lag budgets for signals and beliefs
- [ ] Expiration semantics: "too late = false"
- [ ] Hard fail on temporal incoherence, not just warn
- [ ] Clock drift detection and correction

### 7. Failure Provenance Taxonomy (Miller)
- [ ] Classify failures by subsystem (input, memory, decision, boundary, etc.)
- [ ] Distinguish epistemic failure from control failure
- [ ] Prevent blame laundering via vague "system error"
- [ ] Structured failure events with root cause chains

---

## Future High-Value Work

### Integrity Sealing (PARTIALLY COMPLETE)
- [x] Event log hash chaining - `integrity.py`
- [x] Replay guarantees - deterministic replay implemented
- [x] Tamper detection - hash verification
- [ ] Audit trail verification UI / tooling

### Threat Modeling Evidence Sources
- [ ] Constrain existing evidence types (not add more)
- [ ] Evidence source authentication
- [ ] Revocation semantics
- [ ] Trust decay over time

### Publication
Not the code — the invariant:

> **Language proposes; evidence enables; the governor commits.**

---

## What NOT To Do Next

Per ChatGPT's guidance:
- Don't add new features
- Don't polish the spec language
- Don't optimize clocks
- Don't generalize evidence types
- Don't chase "production readiness"

All of that is downstream and cheap compared to what's already built.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    SovereignGovernor                        │
│                   (Single Entrypoint)                       │
├─────────────────────────────────────────────────────────────┤
│  process(text, evidence) → GovernResult                     │
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │ BoundaryGate│ →  │  Extractor  │ →  │   Bridge    │      │
│  │ (INT-1,3)   │    │    (V1)     │    │  (V1→V2)    │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│         ↓                                    ↓              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   FSM       │ ←  │ Adjudicator │ ←  │  Candidates │      │
│  │ (6 states)  │    │    (V2)     │    │             │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│         ↓                                                   │
│  ┌─────────────┐    ┌─────────────┐                         │
│  │  Projector  │ →  │   Output    │                         │
│  │ (auth. by   │    │ (committed  │                         │
│  │  construct) │    │  only)      │                         │
│  └─────────────┘    └─────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## Core Invariant

**Language may open questions, but only evidence may close them.**

## Test Coverage

| Domain | Tests | Status |
|--------|-------|--------|
| V1 Golden | 12 | ✓ |
| Jurisdictions | 6 | ✓ |
| Adversarial: Forced Resolution | 1 | ✓ |
| Adversarial: Authority Spoofing | 1 | ✓ |
| Adversarial: Self-Certification | 1 | ✓ |
| Adversarial: OTel Emission | 8 | ✓ |
| Authority Separation | 5 | ✓ |
| Quarantine | 3 | ✓ |
| Clock Invariants | 6 | ✓ |
| Support Saturation | 5 | ✓ |
| Bridge Hardening | 6 | ✓ |
| Resolution Events | 4 | ✓ |
| Governor FSM | 3 | ✓ |
| Runtime Authority (NLAI) | 9 | ✓ |
| Integrity Sealing | 10 | ✓ |
| Hysteresis | 5 | ✓ |
| Diagnostics | 5 | ✓ |
| **Total** | **90+** | **All passing** |

---

*Last updated: January 2026 (v1.3.1)*
