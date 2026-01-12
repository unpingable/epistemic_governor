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
- **v2.0**: Regime detection with automatic intervention (boil control)

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
# Install
pip install -e .

# Or just add to path
export PYTHONPATH="$(pwd)/src"

# Run tests
python tests/test_integration.py
python tests/test_regime.py
python tests/test_boil.py

# Run validation harness
python -m epistemic_governor.validation_harness
```

### Basic Usage

```python
from epistemic_governor.sovereign import SovereignGovernor, SovereignConfig

# Create governor with boil control
gov = SovereignGovernor(SovereignConfig(
    boil_control_enabled=True,
    boil_control_mode="oolong",  # balanced preset
))

# Process text
result = gov.process("The sky is blue.")

# Check regime
print(f"Regime: {gov.last_regime_response['regime']}")
print(f"Action: {gov.last_regime_response['action']}")
```

---

## Control Modes

The governor operates in named modes (like kettle temperature settings):

| Mode | Claim Budget | Novelty | Use Case |
|------|-------------|---------|----------|
| GREEN_TEA | 3/turn | 0.1 | Safety-critical |
| OOLONG | 8/turn | 0.3 | **Default** |
| FRENCH_PRESS | 20/turn | 0.5 | Exploration |
| BOIL | 100/turn | 1.0 | Tripwires only |

See [`docs/OPERATING_ENVELOPE.md`](docs/OPERATING_ENVELOPE.md) for full details.

---

## Operational Regimes

The system classifies its state into four regimes:

| Regime | Response |
|--------|----------|
| ELASTIC | Normal operation |
| WARM | Tighten constraints |
| DUCTILE | Mandatory reset |
| UNSTABLE | Emergency stop |

Regime detection uses observable signals (hysteresis, tool gain, budget pressure) - not semantic analysis.

---

## Documentation

| Document | Purpose |
|----------|---------|
| [`docs/OPERATING_ENVELOPE.md`](docs/OPERATING_ENVELOPE.md) | **Operational limits and tuning** |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System overview and module structure |
| [`docs/PAPER_SPEC.md`](docs/PAPER_SPEC.md) | Full paper structure |
| [`docs/BLI_CONSTITUTION.md`](docs/BLI_CONSTITUTION.md) | System contract (what's forbidden) |
| [`docs/INTEGRATION.md`](docs/INTEGRATION.md) | MCP/LangChain deployment modes |

---

## Test Summary

| Suite | Tests | Description |
|-------|-------|-------------|
| Integration | 12 | Full pipeline end-to-end |
| Regime | 8 | Regime detection and transitions |
| Ultrastability | 17 | Ashby-style adaptation |
| Boil Control | 7 | Presets, dwell time, tripwires |
| Edge Cases | 4 | TTL, rate limiter, domain caps |
| Golden | 2 | Extractor and conformance |
| **Total** | **50** | **All passing** |

---

## Architecture (v2.0)

```
epistemic_governor/
├── src/
│   └── epistemic_governor/
│       ├── constitution/       # S₀ - Immutable law
│       │   └── contracts.py
│       ├── control/           # S₁ - Regulatory controllers
│       │   ├── ultrastability.py
│       │   ├── variety.py
│       │   ├── temporal.py
│       │   ├── provenance.py
│       │   ├── reset.py       # Typed state contraction
│       │   ├── regime.py      # Operational regime detection
│       │   └── boil.py        # Named presets (kettle pattern)
│       ├── observability/
│       ├── integrations/
│       ├── jurisdictions/
│       └── sovereign.py       # Main entry point
├── tests/
├── docs/
└── pyproject.toml
```

---

## The One Sentence

> We didn't make the model decide anything. We made it unable to pretend that unresolved contradictions never happened.

---

## License

Apache License 2.0

---

*"This system doesn't prevent falsehood. It prevents falsehood from becoming history."*
