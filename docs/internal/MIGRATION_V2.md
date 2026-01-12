# v2 Layout Migration Plan

## Proposed Structure

```
epistemic_governor/
├── __init__.py
├── __main__.py
│
├── constitution/           # S₀ - immutable law
│   ├── __init__.py
│   ├── nlai.py            # NLAI invariant enforcement
│   ├── forbidden.py       # forbidden transitions, inadmissible actions
│   └── contracts.py       # interface contracts (from interface_contracts.py)
│
├── control/               # S₁ - regulatory controllers
│   ├── __init__.py
│   ├── ultrastability.py  # adaptation controller
│   ├── variety.py         # variety dial
│   ├── temporal.py        # TTL, lag, clock
│   └── provenance.py      # failure taxonomy (from failure_provenance.py)
│
├── core/                  # Mechanism
│   ├── __init__.py
│   ├── types.py           # shared dataclasses (from api.py)
│   ├── claims.py          # claim types and extraction
│   ├── ledger.py          # persistence + hash chain
│   ├── fsm.py             # state machine (from governor_fsm.py)
│   └── governor.py        # main orchestration (from sovereign.py)
│
├── observability/         # Telemetry and diagnostics
│   ├── __init__.py
│   ├── diagnostics.py     # DiagnosticEvent, regime detection
│   ├── otel.py            # OTel projection (from otel_projection.py)
│   └── regimes.py         # regime definitions
│
├── integrations/          # Adapters (dumb, no logic)
│   ├── __init__.py
│   ├── langchain.py       # callback handler
│   ├── mcp.py             # MCP server adapter (stub)
│   └── demo.py            # demo agent
│
├── jurisdictions/         # Keep as-is, it's clean
│   └── ...
│
├── tests/                 # All tests together
│   ├── __init__.py
│   ├── golden/            # golden test cases
│   ├── adversarial/       # attack tests
│   ├── test_ultrastability.py
│   ├── test_temporal.py
│   ├── test_variety.py
│   ├── test_contracts.py
│   ├── test_provenance.py
│   ├── test_edge_cases.py
│   └── run_all.py
│
├── docs/
│   ├── ARCHITECTURE.md    # NEW: one diagram, one-paragraph-per-module
│   ├── CONSTITUTION.md    # renamed from BLI_CONSTITUTION.md
│   ├── INTEGRATION.md     # MCP/LangChain guide
│   ├── PAPER.md           # renamed from PAPER_SPEC.md
│   ├── FAQ.md
│   └── internal/
│       ├── ROADMAP.md
│       └── TODO.md
│
├── pyproject.toml
├── LICENSE
└── README.md
```

## File Mapping

### constitution/
| New | Old | Notes |
|-----|-----|-------|
| `nlai.py` | extract from `sovereign.py` | NLAI check functions |
| `forbidden.py` | extract from `governor_fsm.py` | forbidden transitions |
| `contracts.py` | `interface_contracts.py` | move as-is |

### control/
| New | Old | Notes |
|-----|-----|-------|
| `ultrastability.py` | `ultrastability.py` | move as-is |
| `variety.py` | `variety.py` | move as-is |
| `temporal.py` | `temporal.py` | move as-is |
| `provenance.py` | `failure_provenance.py` | rename + move |

### core/
| New | Old | Notes |
|-----|-----|-------|
| `types.py` | `api.py` | rename, keep dataclasses |
| `claims.py` | `claims.py` + `claim_extractor.py` | merge |
| `ledger.py` | `ledger.py` + `integrity.py` | merge |
| `fsm.py` | `governor_fsm.py` | rename |
| `governor.py` | `sovereign.py` | rename (main entry) |

### observability/
| New | Old | Notes |
|-----|-----|-------|
| `diagnostics.py` | `diagnostics.py` | move |
| `otel.py` | `otel_projection.py` | rename + move |
| `regimes.py` | `regimes.py` | move |

### integrations/
| New | Old | Notes |
|-----|-----|-------|
| `langchain.py` | `instrumentation/langchain_callback.py` | move |
| `demo.py` | `instrumentation/demo_agent.py` | move |
| `mcp.py` | NEW | stub for MCP adapter |

### tests/
| New | Old | Notes |
|-----|-----|-------|
| `tests/adversarial/` | `adversarial/` | move |
| `tests/golden/` | golden test data | consolidate |
| `tests/test_*.py` | various `test_*.py` | move |
| `tests/run_all.py` | `run_golden_tests.py` + others | consolidate |

## Files to DELETE (cruft)

### Generated data (should be gitignored)
- `*.json` (CONFORMANCE_*, sweep_results, router_*, diff_*, shadow_audit_*, SPEC.json)
- `diagnostic_events.jsonl`
- `diagnostic_report.txt`
- `files.txt`

### Old experiments (archive or delete)
- `autopilot_fsm.py`, `autopilot_integration.py`
- `bench_hallucination_traps.py`, `bench_resonance.py`
- `calibrate.py`
- `character.py`
- `creative.py`
- `curiosity.py`
- `homeostat.py` (superseded by ultrastability)
- `shear.py`
- `resistance.py`
- `negative_t.py`
- `heading.py`
- `envelope.py`
- `valve.py`
- `vectors.py`

### Probably redundant
- `governor.py` (if sovereign.py is the main one)
- `extractor.py` (if claim_extractor.py is used)
- `egov.py`, `kernel.py`, `modes.py` (check if used)
- `session.py`, `providers.py`, `tools.py` (check if used)

## Docs Consolidation

### Keep
- `CONSTITUTION.md` (from BLI_CONSTITUTION.md)
- `INTEGRATION.md`
- `PAPER.md` (from PAPER_SPEC.md)
- `FAQ.md`
- `internal/ROADMAP.md`

### Create
- `ARCHITECTURE.md` - one diagram, module overview

### Archive/Delete
- `BLI_SPEC_V1.md` - superseded by PAPER
- `BLI_V3_SPEC.md` - superseded by PAPER
- `CONSTRAINT_KERNELS.md` - internal design notes
- `DESIGN_RATIONALE.md` - fold into ARCHITECTURE
- `GAP_ANALYSIS.md` - internal
- `INVARIANTS.md` - fold into CONSTITUTION
- `NLAI.md` - fold into CONSTITUTION
- `OTEL_CONVENTIONS.md` - fold into INTEGRATION
- `QUICKSTART.md` - fold into README
- `REFUSALS.md` - internal
- `THREE_CUEING.md` - keep as appendix in PAPER

## Migration Steps

1. Create new directory structure
2. Move files according to mapping
3. Update imports in all files
4. Delete cruft
5. Update README with new structure
6. Run all tests to verify
7. Tag as v2.0

## Import Convention

After migration:
```python
# Constitution (rarely imported directly)
from governor.constitution import contracts, forbidden

# Control (imported by core)
from governor.control import ultrastability, variety, temporal

# Core (main public interface)
from governor.core import Governor, Claim, Evidence, Verdict
from governor.core.types import *

# Observability
from governor.observability import DiagnosticEvent, emit_otel

# Integrations (what external users import)
from governor.integrations import LangChainCallback, MCPServer
```

## Boundary Enforcement

Key rule: `integrations/` may only import from `core/` public interface.
Never: `from governor.constitution.nlai import _internal_check`

This is enforced by keeping `constitution/` and `control/` as implementation details
that `core/governor.py` wires together.
