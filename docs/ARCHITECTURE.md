# Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Integrations                                │
│                 (LangChain, MCP, WAF adapters)                       │
│                    ↓ handle_turn() only ↓                            │
├─────────────────────────────────────────────────────────────────────┤
│                             Core                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │  Governor   │──│    FSM      │──│   Ledger    │──│   Types    │ │
│  │ (orchestr.) │  │ (6 states)  │  │ (hash chain)│  │ (Claims,..)│ │
│  └──────┬──────┘  └─────────────┘  └─────────────┘  └────────────┘ │
│         │                                                            │
│         ├─────────────────────┬──────────────────────┐               │
│         ↓                     ↓                      ↓               │
├─────────────────────────────────────────────────────────────────────┤
│                            Control                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │Ultrastable  │  │  Variety    │  │  Temporal   │  │ Provenance │ │
│  │ (S₁ adapt)  │  │(load shed)  │  │(TTL, lag)   │  │(failures)  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │
│         ↑                                                            │
│         │ bounds, forbidden transitions                              │
├─────────────────────────────────────────────────────────────────────┤
│                          Constitution                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │  Contracts  │  │  Forbidden  │  │    NLAI     │                  │
│  │ (boundary)  │  │(transitions)│  │ (invariant) │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
│                        IMMUTABLE                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## State Hierarchy

| Layer | Name | Contents | Mutability |
|-------|------|----------|------------|
| S₀ | Constitution | NLAI, FSM topology, bounds, contracts | **Immutable** |
| S₁ | Regulatory | Budgets, thresholds, timeouts | Adaptable within S₀ bounds |
| S₂ | Epistemic | Claims, contradictions, ledger | Fully mutable |

**Influence rules:**
- S₂ → S₁ ✓ (observations trigger adaptation)
- S₁ → S₀ ✗ (regulatory cannot modify constitutional)
- Language → S₀/S₁ ✗ (model cannot modify parameters)

## Package Structure

```
epistemic_governor/
├── constitution/       # S₀ - Immutable law
│   ├── contracts.py   # Interface boundary definitions
│   ├── forbidden.py   # Inadmissible actions (TODO)
│   └── nlai.py        # NLAI enforcement (TODO)
│
├── control/           # S₁ - Regulatory controllers
│   ├── ultrastability.py  # Second-order adaptation
│   ├── variety.py         # Load shedding
│   ├── temporal.py        # TTL, lag, clock
│   └── provenance.py      # Failure taxonomy
│
├── core/              # Mechanism (TODO: consolidate)
│   ├── governor.py    # Main orchestration
│   ├── fsm.py         # State machine
│   ├── ledger.py      # Persistence
│   └── types.py       # Shared types
│
├── observability/     # Telemetry
│   └── otel.py        # OpenTelemetry projection
│
├── integrations/      # External adapters
│   ├── langchain.py   # Callback handler
│   ├── demo.py        # Demo agent
│   └── mcp.py         # MCP server (TODO)
│
├── jurisdictions/     # Domain-specific policies
│   └── *.py           # Factual, Speculative, etc.
│
└── tests/             # All tests
    ├── adversarial/   # Attack tests
    └── test_*.py      # Unit tests
```

## How a Turn Flows

1. **Input** arrives at integration layer
2. **Boundary gate** (constitution/contracts) validates input
3. **Governor** receives validated input
4. **Variety controller** sheds load if needed
5. **Claim extraction** produces candidates
6. **Temporal check** validates freshness
7. **FSM** validates state transition
8. **Adjudication** checks claims against committed state (SymbolicState)
9. **Commit** (if evidence present) or **Quarantine** (if not)
10. **Output** projected from committed state only (projection enforces output rules)
11. **Telemetry** emitted via observability layer
12. **Ultrastability** considers adaptation based on metrics

## Key Invariants

### NLAI (Non-Linguistic Authority Invariant)
> Language may open questions, but only evidence may close them.

No natural-language content can directly mutate authoritative state.
Model outputs are proposals; commits require external evidence.

### Fail-Closed
If any controller or check fails, **epistemic state is not advanced**.
The system may still return a response (labeled non-assertive), but no
new commitments are added to the ledger. Ambiguity defaults to denial
of state mutation, not denial of interaction.

### Audit Trail
Every state transition is logged with:
- Timestamp
- From/to state
- Triggering event
- Turn ID

**Note:** Full forensic-grade provenance (evidence refs, causal chain, 
integrity hash) is planned but not yet implemented. Current logging
captures transition events but not full evidence linkage.

---

## Fault Domains

The architecture is organized around **fault containment boundaries** - regions where failures are allowed to exist without propagating to other regions. This is standard practice in dependable systems (Laprie, 1992) and resilience engineering.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TEMPORAL FAULT DOMAIN                         │
│  Δt boundary: stale evidence cannot influence fresh decisions        │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    AUTHORITY FAULT DOMAIN                       ││
│  │  NLAI boundary: language cannot escalate to commitment          ││
│  │  ┌─────────────────────────────────────────────────────────────┐││
│  │  │                 VARIETY FAULT DOMAIN                        │││
│  │  │  Load shedding: input volume cannot exhaust processing      │││
│  │  │  ┌─────────────────────────────────────────────────────────┐│││
│  │  │  │              ADAPTATION FAULT DOMAIN                    ││││
│  │  │  │  S₀ bounds: regulatory tuning cannot modify law         ││││
│  │  │  │                                                         ││││
│  │  │  │              [Core State: S₂]                           ││││
│  │  │  └─────────────────────────────────────────────────────────┘│││
│  │  └─────────────────────────────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

Each boundary is a **variety attenuator** (Ashby, 1956): it limits what can propagate inward.

| Fault Domain | Boundary | Failure Contained |
|--------------|----------|-------------------|
| Temporal | TTL, lag budgets | Stale data influencing decisions |
| Authority | NLAI, evidence requirement | Language escalating to commitment |
| Variety | Load shedding, rate limits | Input volume exhausting resources |
| Adaptation | S₀ bounds, freeze mechanism | Runaway self-modification |
| Interface | Contracts, injection filters | External input corrupting internals |

This is not novel - it's how flight control systems, nuclear plants, and distributed databases have worked for decades. The contribution is applying these patterns to language model governance, where the failure modes are epistemic (false confidence, hallucination, authority confusion) rather than physical.

**References:**
- Laprie, J.C. (1992). Dependability: Basic Concepts and Terminology
- Ashby, W.R. (1956). An Introduction to Cybernetics
- Leveson, N. (2011). Engineering a Safer World (STAMP/STPA)

---

## Integration Boundaries

Integrations may only import from:
- `core.governor.Governor` (main interface)
- `core.types.*` (shared types)

Integrations may **not** import from:
- `constitution.*` (internal enforcement)
- `control.*` (internal adaptation)

This prevents external code from bypassing constitutional constraints.
