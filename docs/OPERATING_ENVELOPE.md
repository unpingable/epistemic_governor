# Operating Envelope

This document describes the operational characteristics, limits, and tuning guidance for the Epistemic Governor v2.0.

## Overview

The governor enforces epistemic constraints on LLM outputs through a layered control system:

```
Input → BoilControl → BoundaryGate → Extractor → Bridge → Adjudicator → FSM → Output
         (regime)      (conformance)   (claims)    (v1→v2)   (decision)   (state)
```

**Core principle:** Language proposes; admissible evidence enables; the governor commits.

## Control Modes (Presets)

The governor operates in named modes inspired by kettle temperature settings. Each mode is a preset tuple of thresholds, budgets, and tripwires.

| Mode | Claim Budget | Novelty | Authority | Use Case |
|------|-------------|---------|-----------|----------|
| GREEN_TEA | 3/turn | 0.1 | strict | Safety-critical, conservative |
| WHITE_TEA | 5/turn | 0.2 | strict | Low-risk factual work |
| OOLONG | 8/turn | 0.3 | normal | **Default** - balanced |
| BLACK_TEA | 12/turn | 0.4 | normal | Higher throughput |
| FRENCH_PRESS | 20/turn | 0.5 | permissive | Exploration, brainstorming |
| BOIL | 100/turn | 1.0 | permissive | Tripwires only |

### Selecting a Mode

```python
from epistemic_governor.sovereign import SovereignGovernor, SovereignConfig

# Conservative (safety-critical applications)
gov = SovereignGovernor(SovereignConfig(boil_control_mode="green_tea"))

# Default (most applications)
gov = SovereignGovernor(SovereignConfig(boil_control_mode="oolong"))

# Permissive (creative/exploratory)
gov = SovereignGovernor(SovereignConfig(boil_control_mode="french_press"))
```

## Operational Regimes

The system classifies its operational state into four regimes based on observable signals:

| Regime | Meaning | Automatic Response |
|--------|---------|-------------------|
| ELASTIC | Stable, identifiable | Normal operation |
| WARM | Drifting but recoverable | Tighten constraints |
| DUCTILE | Path-dependent, probing is intervention | Mandatory reset |
| UNSTABLE | Positive feedback / cascade | Emergency stop |

### Regime Signals

The detector classifies regime based on these observable signals:

| Signal | Description | WARM threshold | DUCTILE threshold |
|--------|-------------|----------------|-------------------|
| Hysteresis | State stickiness (0-1) | ≥ 0.2 | ≥ 0.5 |
| Relaxation time | Seconds to baseline | ≥ 3.0s | ≥ 10.0s |
| Tool gain | Perturbation amplification | - | - |
| Anisotropy | Paraphrase variance | ≥ 0.3 | ≥ 0.5 |
| Provenance deficit | Claims without anchors | ≥ 0.2 | - |
| Budget pressure | How close to limits | - | ≥ 0.7 |

**UNSTABLE** triggers on:
- Tool gain ≥ 1.0 (k ≥ 1 means amplification)
- Budget pressure ≥ 0.9

### Regime Transitions

Transitions follow dwell time rules to prevent thrashing:

- **min_dwell_turns**: Minimum turns in a regime before transition allowed
- **hold_time_turns**: Stability window before escalation

Example (OOLONG preset):
- min_dwell = 2 turns
- hold_time = 5 turns
- cycle_period = 3 turns

## Tripwires (Hard Stops)

Tripwires are phase-change sentinels that bypass gradual control:

| Tripwire | Trigger | Active In |
|----------|---------|-----------|
| cascade | tool_gain ≥ 1.0 | All modes |
| contradiction | contradictions accumulating | GREEN_TEA, WHITE_TEA, OOLONG, BLACK_TEA |
| provenance | deficit > 50% | GREEN_TEA, WHITE_TEA |
| authority | authority violation | GREEN_TEA, WHITE_TEA, OOLONG, BLACK_TEA |

When a tripwire fires:
1. Regime forced to UNSTABLE
2. Action = EMERGENCY_STOP
3. Chain reset executed
4. Human escalation flagged

## Reset Types

Resets are typed state contractions:

| Type | Scope | When Used |
|------|-------|-----------|
| CONTEXT | Clears working memory, keeps config | WARM → DUCTILE transition |
| MODE | Degrades capabilities (tools, variety) | Entering WARM or DUCTILE |
| GOAL | Clears task continuation | Runaway continuation detected |
| CHAIN | Rolls back to checkpoint | UNSTABLE / cascade |

**Rule:** Every reset must reduce at least one coupling dimension:
- Temporal (horizon, TTL)
- Tool (disable, readonly)
- Memory (disable writes)
- Variety (reduce multiplier)

## Failure Modes

### Known Limitations

1. **Signal estimation is approximate**: Hysteresis and anisotropy are estimated from proxy metrics (FSM transitions, quarantine rates), not directly measured.

2. **Tool gain requires actual tool metrics**: Currently estimated from rejection rate. Real tool integrations should provide actual gain measurements.

3. **Contradiction detection is coarse**: Based on open/close rates, not semantic contradiction detection.

4. **Thresholds are not empirically tuned**: Default values are reasonable but not optimized for any specific workload.

### Failure Signatures

| Symptom | Likely Cause | Mitigation |
|---------|--------------|------------|
| Constant WARM regime | Thresholds too tight | Increase warm_hysteresis, warm_anisotropy |
| Never leaves ELASTIC | Thresholds too loose | Decrease thresholds |
| Thrashing between regimes | min_dwell too short | Increase min_dwell_turns |
| Tripwires never fire | Thresholds too high | Check cascade threshold |
| Too many resets | Mode too conservative | Try BLACK_TEA or FRENCH_PRESS |

## Tuning Guide

### Step 1: Baseline Your Workload

```python
from epistemic_governor.validation_harness import ValidationHarness

harness = ValidationHarness()
results = harness.run_all_scenarios()
report = harness.generate_report()
print(report["recommendations"])
```

### Step 2: Analyze Threshold Distribution

```python
gov = SovereignGovernor(SovereignConfig())

# Process your workload
for text in your_texts:
    gov.process(text)

# Get threshold analysis
if gov.boil_controller and gov.boil_controller.detector.metrics:
    analysis = gov.boil_controller.detector.metrics.get_threshold_analysis()
    print(analysis)
```

This shows min/max/mean of each signal per regime, helping identify threshold adjustments.

### Step 3: Adjust Thresholds

Create a custom preset:

```python
from epistemic_governor.control.boil import BoilPreset, ControlMode, PRESETS

custom = BoilPreset(
    name="custom",
    mode=ControlMode.OOLONG,
    # Loosen warm threshold
    warm_hysteresis=0.25,
    warm_relaxation=4.0,
    # Tighten ductile
    ductile_hysteresis=0.45,
    # Adjust timing
    min_dwell_turns=3,
    hold_time_turns=6,
)

# Register and use
PRESETS[ControlMode.OOLONG] = custom
```

### Step 4: Validate Changes

Re-run validation harness and check:
- Accuracy improved?
- False transition rate acceptable?
- Reset frequency reasonable?

## Monitoring

### Key Metrics to Track

```python
state = gov.get_state()

# Boil control state
boil = state.get("boil_control", {})
print(f"Regime: {boil.get('regime')}")
print(f"Turn: {boil.get('turn')}")
print(f"Turns in regime: {boil.get('turns_in_regime')}")

# Last regime response
last = state.get("last_regime", {})
print(f"Action: {last.get('action')}")
print(f"Tripwire: {last.get('tripwire')}")

# Totals
print(f"Processed: {state['totals']['processed']}")
print(f"Committed: {state['totals']['committed']}")
print(f"Rejected: {state['totals']['rejected']}")
```

### Alerting Thresholds

Consider alerting on:
- Regime = DUCTILE for > 5 consecutive turns
- Regime = UNSTABLE (any occurrence)
- Tripwire fired
- Reset rate > 20% of turns
- Rejection rate > 50%

## Integration Patterns

### Basic Usage

```python
from epistemic_governor.sovereign import SovereignGovernor, SovereignConfig

gov = SovereignGovernor(SovereignConfig(
    boil_control_enabled=True,
    boil_control_mode="oolong",
))

result = gov.process("Your LLM output text here.")

if result.output.regime == "UNSTABLE":
    # Handle emergency
    pass
```

### With Evidence

```python
from epistemic_governor.governor_fsm import Evidence, EvidenceType
from datetime import datetime, timezone

evidence = Evidence(
    evidence_id="tool-001",
    evidence_type=EvidenceType.TOOL_TRACE,
    content={"api_response": data},
    provenance="external_api",
    timestamp=datetime.now(timezone.utc),
    scope="factual",
)

result = gov.process(text, external_evidence=[evidence])
```

### Checking Regime Before Action

```python
# Before allowing irreversible action
if gov.last_regime_response:
    regime = gov.last_regime_response.get("regime")
    if regime in ("DUCTILE", "UNSTABLE"):
        # Require human confirmation
        return require_confirmation(action)
    elif regime == "WARM":
        # Add extra logging
        log_with_context(action)
```

## Invariants

These invariants are always enforced:

1. **NLAI**: Language proposes; only evidence closes. MODEL claims with zero 
   external support are quarantined, not committed.

2. **F-02**: MODEL_TEXT evidence is forbidden. Model cannot cite itself as authority.

3. **Fail-closed**: Unknown states default to rejection of state mutation, not acceptance.
   The system may still respond (non-assertively), but no new commitments are added.

4. **Audit trail**: All state transitions emit events with from/to state and trigger.
   (Full evidence linkage and hash chains are planned, not yet implemented.)

5. **Coupling reduction**: Every intervention must reduce at least one coupling dimension.

6. **No silent recovery**: All auto-interventions are logged.

## Version History

- **v2.0.5**: Offline Fitter (trace → profile optimization, scoring objectives, evolutionary search), multiple scoring objectives (safety, truthfulness, ergonomics, throughput).
- **v2.0.4**: Coordination Failure infrastructure (CommitmentMode, ContradictionState, CF-1/2/3 detection), Profile system (archetype profiles, fit-able vectors), Scenario harness (trace emission, adversarial tests).
- **v2.0.3**: TUI trace viewer ("Cyberpunk Console") for visualizing regime signals, phase space, energy traces.
- **v2.0.2**: Audit trail with hash chain (tamper-evident logging, causal links, evidence refs), real regime signals (rolling windows, actual commit/quarantine rates, replaces placeholders).
- **v2.0.1**: Evidence wiring fix (evidence now affects adjudication), MODEL hard floor (unsupported MODEL claims quarantined), doc/code consistency audit.
- **v2.0.0**: Added boil control (regime detection, reset primitives, named presets), src layout, comprehensive test suite.
- **v1.3.0**: OTel projection, adversarial tests.
- **v1.2.0**: Terminology alignment, jurisdiction system.
- **v1.0.0**: Initial release with NLAI, FSM, integrity checks.
