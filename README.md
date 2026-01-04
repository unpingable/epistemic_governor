# Bounded Lattice Inference

**A governed reasoning substrate with persistent state and non-linguistic authority.**

James Beck | Independent Researcher

**Repository**: https://github.com/unpingable/epistemic_governor

---

## What This Is

This work describes a governed reasoning substrate that introduces persistent state to language-model systems without granting agency, goals, or self-modification. The contribution is architectural: separating linguistic proposal from non-linguistic authority, and measuring the resulting system using standard tools from queueing theory and supervisory control. The system does not solve alignment, does not implement values, and does not exhibit consciousness. It enforces epistemic constraints by construction and is evaluated via falsifiable experiments.

### What this is:

- A control-theoretic architecture for persistent reasoning under hard constraints
- An empirical demonstration that interiority can exist without agency
- A diagnostic framework for reasoning-system failure modes (starvation, accumulation)

### What this is not:

- Not an agent framework
- Not a memory system for chatbots
- Not a theory of mind, selfhood, or consciousness

---

## Core Principle

**Non-Linguistic Authority Invariant (NLAI)**

> Language may open questions, but only evidence may close them.

The model proposes. The governor decides. State binds. Language never overrides law.

---

## Quick Start

```bash
# Run demonstrator (shows core behaviors)
PYTHONPATH=. python epistemic_governor/demonstrator.py

# Run tests
PYTHONPATH=. python epistemic_governor/run_golden_tests.py
PYTHONPATH=. python epistemic_governor/hysteresis.py

# Run experiments
PYTHONPATH=. python epistemic_governor/budget_sweep.py
PYTHONPATH=. python epistemic_governor/glass_sweep.py
```

---

## Key Results

### Interiority Confirmed

Same input + different internal state = different output, traceable to specific state objects.

```
Hysteresis test: 24 trials, 16.7% divergence rate
Divergence traces to: contradiction IDs, commitment IDs, budget values
```

### Two Phase Boundaries Identified

**Budget Starvation Boundary** (sharp transition)
```
repair_refill_rate < 2.0 → STARVATION
repair_refill_rate ≥ 2.0 → HEALTHY
```

**Glass Ossification Boundary** (gradual transition)
```
resolution_cost < 12.5 → HEALTHY
resolution_cost > 12.5 → GLASS (accumulation)
```

Both collapse to: **λ_open vs μ_close** (queueing theory)

### Safety Invariants Held

Across all experiments:
- `closed_without_evidence = 0` (no laundering)
- `ρ_S` stayed healthy (no ceremony collapse)

---

## Documentation

| Document | Purpose |
|----------|---------|
| `PAPER_SPEC.md` | Full paper structure |
| `BLI_SPEC_V1.md` | Standalone specification |
| `FAQ.md` | Misreadings and clarifications |
| `NLAI.md` | Non-Linguistic Authority Invariant |

---

## Test Summary

| Suite | Tests | Status |
|-------|-------|--------|
| Golden | 12 | ✓ |
| Hysteresis | 5 | ✓ |
| Authority | 5 | ✓ |
| Integrity | 10 | ✓ |
| Diagnostics | 5 | ✓ |
| (+ 8 more suites) | 39 | ✓ |
| **Total** | **76** | **All passing** |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SovereignGovernor                        │
│                   (Single Entrypoint)                       │
├─────────────────────────────────────────────────────────────┤
│  process(text, evidence) → GovernResult                     │
│                                                             │
│  Model ──→ Extractor ──→ Validator ──→ Adjudicator          │
│    │                                        │               │
│    ↓                                        ↓               │
│  [proposes]                           [verdicts]            │
│                                             │               │
│  Budget ←── FSM ←── Projector ←────────────┘                │
│    │         │          │                                   │
│    ↓         ↓          ↓                                   │
│  [costs]  [states]  [governed output]                       │
└─────────────────────────────────────────────────────────────┘
```

---

## The One Sentence

> We didn't make the model decide anything. We made it unable to pretend that unresolved contradictions never happened.

---

## Codebase

~58,700 lines across 76 Python files.

Key modules:
- `sovereign.py` - Single entrypoint
- `governor_fsm.py` - State machine
- `hysteresis.py` - Interiority tests
- `diagnostics.py` - Phase detection
- `demonstrator.py` - Core behavior demos

---

## License

Apache License 2.0

---

*"This system doesn't prevent falsehood. It prevents falsehood from becoming history."*
