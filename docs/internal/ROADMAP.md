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

### 1. Ultrastability (Ashby) — IMPLEMENTED
*Priority: HIGH - everything else cascades from this*

See `ultrastability.py` for implementation, `test_ultrastability.py` for tests (14 passing).

- [x] Define conditions under which the governor itself may mutate (S₁ only, via triggers)
- [x] Second-order adaptation when repeated repairs preserve failure (pathology detection)
- [x] Guardrails against infinite self-modification (floor/ceiling/step bounds)
- [x] Detection of "successful failure" (wrong attractor detection)

**Key components:**
- `RegulatoryParameters` - S₁ state with constitutional bounds
- `AdaptationTrigger` - When to consider adaptation
- `PathologyDetector` - Oscillation, runaway, ineffective, wrong attractor
- `UltrastabilityController` - Orchestrates the loop, enforces freeze on pathology

### 2. Variety Dial — IMPLEMENTED
See `variety.py` for implementation.

- [x] Explicit policy for variety absorption vs amplification
- [x] Dynamic thresholds tied to stress / adversarial load
- [x] Detect exploitation via over-filtering or over-expansion
- [x] Bounded variety: min/max claims per turn, scope limits

**Key components:**
- `VarietyBounds` - Configurable limits on claims, domains, novelty
- `VarietyController` - Load shedding and exploitation detection
- Stress-aware shedding when system under load

### 3. Meta-Repair Pathology (Bateson) — IMPLEMENTED
Covered by `ultrastability.py` PathologyDetector.

- [x] Detect Learning-II failure: repair loops that stabilize the wrong invariant
- [x] Flag repeated contradiction resolution with no reduction in failure rate
- [x] Escalation path when "repair itself is the bug" (freeze + alert)
- [x] Distinguish productive cycling from pathological cycling (oscillation detection)

### 4. Interface Austerity (Simon) — IMPLEMENTED
See `interface_contracts.py` for implementation.

- [x] Formal inner vs outer environment boundary (INPUT/OUTPUT/CONTROL)
- [x] Explicit interface contracts (what can cross the boundary)
- [x] Define inadmissible truths / forbidden actions even if effective
- [x] Boundary hardening against prompt injection at interface level

**Key components:**
- `InputContract`, `OutputContract`, `ControlContract` - Per-interface rules
- `BoundaryGate` - Enforces contracts, filters forbidden fields
- `INADMISSIBLE_ACTIONS` - Actions forbidden regardless of effectiveness

### 5. Model Decentering (Luhmann-lite) — BY CONSTRUCTION
Already true in the architecture.

- [x] Treat model outputs as perturbations, not components (NLAI)
- [x] System defined by claims/contradictions/resolutions, not agents
- [x] Support multi-model + human hybrid governance without charisma bleed
- [x] No special status for any single model's outputs

**Evidence:** The governor never asks "who said this" - only "is there evidence?"

### 6. Temporal Hard Limits (Deutsch) — IMPLEMENTED
See `temporal.py` for implementation.

- [x] Lag budgets for signals and beliefs (processing lag limits)
- [x] Expiration semantics: "too late = false" (TTL on claims/evidence)
- [x] Hard fail on temporal incoherence, not just warn
- [x] Clock drift detection and correction

**Key components:**
- `TemporalBounds` - TTLs for claims, evidence, processing
- `TemporalController` - Expiration tracking, lag checks, clock coherence
- `check_turn_temporal()` - Turn-level temporal validation

### 7. Failure Provenance Taxonomy (Miller) — IMPLEMENTED
See `failure_provenance.py` for implementation.

- [x] Classify failures by subsystem (input, memory, decision, boundary, etc.)
- [x] Distinguish epistemic failure from control failure
- [x] Prevent blame laundering via vague "system error"
- [x] Structured failure events with root cause chains

**Key components:**
- `FailureSubsystem`, `FailureType`, `FailureSeverity` - Taxonomy enums
- `FailureEvent` with `causal_chain` - Full provenance
- `FailureRegistry` - Indexed failure collection
- `FailureBuilder` - Structured failure construction

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
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ BoundaryGate│ →  │  Extractor  │ →  │   Bridge    │     │
│  │ (INT-1,3)   │    │    (V1)     │    │  (V1→V2)    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         ↓                                    ↓              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   FSM       │ ←  │ Adjudicator │ ←  │  Candidates │     │
│  │ (6 states)  │    │    (V2)     │    │             │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         ↓                                                   │
│  ┌─────────────┐    ┌─────────────┐                        │
│  │  Projector  │ →  │   Output    │                        │
│  │ (auth. by   │    │ (committed  │                        │
│  │  construct) │    │  only)      │                        │
│  └─────────────┘    └─────────────┘                        │
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
