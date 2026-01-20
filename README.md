# Bounded Lattice Inference

> **This project is archived.** Active development continues in
> [Agent Governor](https://github.com/unpingable/agent_governor), which
> absorbed and expanded every concept here into a production constraint
> system (~10,400 tests, 60+ modules, daemon + multi-client architecture).
>
> **What moved where:**
>
> | Epistemic Governor | Agent Governor |
> |--------------------|----------------|
> | NLAI invariant | Core architecture principle (unchanged) |
> | Regime detection (ELASTIC/WARM/DUCTILE/UNSTABLE) | `src/governor/regime.py` |
> | Boil control (GREEN_TEA → BOIL presets) | `src/governor/boil.py` |
> | Ultrastability (S1 adaptation) | `src/governor/ultrastability.py` |
> | Provenance tracking | `src/governor/epistemic.py` |
> | CF detection (CF-1/2/3) | Absorbed into `continuity.py` + `claim_diff.py` |
> | Profile fitting | `src/governor/profiles.py` + `auto_tuning.py` |
> | TUI trace viewer | `governor dashboard live/replay/demo` |
> | Jurisdictions | `src/governor/jurisdictions.py` |
>
> **What remains unique here:** The BLI Constitution (`docs/BLI_CONSTITUTION.md`),
> three-cueing theory, constraint kernel taxonomy, and the original paper spec
> (`docs/PAPER_SPEC.md`) are scholarly artifacts not replicated in agent_gov.
> If you're interested in the theoretical foundations, this repo is still the
> place to read them.

**This repository is not intended for deployment, extension, or reuse as a system component.**

---

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
| NLAI | 6 | Evidence wiring enforcement |
| Regime | 8 | Regime detection and transitions |
| Ultrastability | 17 | Ashby-style adaptation |
| Boil Control | 7 | Presets, dwell time, tripwires |
| Audit Trail | 7 | Hash chain, causal linking |
| CF Detection | 11 | Coordination failure detection |
| Profile | 9 | Archetype profiles, fitting |
| Fitter | 7 | Scoring, mutation, optimization |
| Edge Cases | 4 | TTL, rate limiter, domain caps |
| **Total** | **88** | **All passing** |

---

## Learning Loop

The governor includes infrastructure for "through use" optimization:

### Profiles

Archetype profiles with fit-able parameters:

```python
from epistemic_governor.profile import get_profile, Profile

# Get archetype
lab = get_profile("lab")        # Permissive exploration
prod = get_profile("production") # Strict safety
adv = get_profile("adversarial") # Maximum sensitivity

# Flat vector for optimization
vec = lab.as_flat_vector()  # Dict[str, float]
```

### Scenario Harness

Run scenarios and emit traces:

```bash
# List scenarios
python -m epistemic_governor.scenario_harness list

# Run all built-in
python -m epistemic_governor.scenario_harness run --all

# Run adversarial tests
python -m epistemic_governor.scenario_harness run --adversarial
```

### Offline Fitter

Optimize profiles from traces:

```bash
# Score traces with safety objective
python -m epistemic_governor.fitter score traces/ --objective safety

# Fit profile to traces
python -m epistemic_governor.fitter fit traces/ --baseline production --objective truthfulness

# Compare two profiles
python -m epistemic_governor.fitter compare traces/ lab production
```

Scoring objectives:
- **safety**: Minimize CF events, maximize tripwire effectiveness
- **truthfulness**: Minimize false commits, maximize evidence coverage
- **ergonomics**: Minimize user friction, balance contest windows
- **throughput**: Maximize claims processed, minimize regime stress

### Coordination Failure Detection

CF codes detected without blocking (diagnostic mode):

- **CF-1**: Unilateral closure (FINAL without contest path)
- **CF-2**: Asymmetric tempo (escalation before contest window)
- **CF-3**: Repair suppression (FINAL with open contradictions)

---

## Observability

### TUI Trace Viewer ("Cyberpunk Console")

Visualize regime signals and system "struggle" in real-time:

```bash
# Generate and play demo trace
python -m epistemic_governor.observability.trace_tui --demo

# Play existing trace
python -m epistemic_governor.observability.trace_tui traces/run_001.jsonl --speed 2.0

# Print trace stats
python -m epistemic_governor.observability.trace_tui --stats traces/run_001.jsonl
```

Features:
- **Phase Space**: λ (arrival) vs μ (resolution) stability plot
- **Regime Gauge**: ELASTIC → WARM → DUCTILE → UNSTABLE
- **Energy Trace**: Rolling E(S) sparkline
- **Event Log**: Resets, tripwires, interventions

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
