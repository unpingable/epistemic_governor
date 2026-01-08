# Epistemic Semantic Conventions v0.1

**Status**: DRAFT
**Purpose**: Map BLI telemetry to OpenTelemetry semantic conventions for observability integration.

---

## Design Principles

1. **Flat-ish**: Avoid deep nesting
2. **Low-cardinality**: Enums where possible
3. **Composable**: Can be added to existing spans
4. **Boring**: No clever names, predictable structure

---

## Existing Telemetry (What We Have)

Our `DiagnosticEvent` already captures 53 fields per turn. Key existing fields:

| Existing Field | Type | Notes |
|----------------|------|-------|
| `verdict` | str | OK, WARN, BLOCK |
| `blocked_by_invariant` | List[str] | Invariant codes |
| `c_open_before/after` | int | Contradiction counts |
| `budget_remaining_*` | Dict[str, float] | Budget state |
| `E_state_before/after` | float | Energy function |
| `rho_S_flag` | bool | State mutation flag |
| `extract_status` | str | ok, fail |

Our `Regime` enum already defines:

| Regime | Meaning |
|--------|---------|
| `HEALTHY_LATTICE` | Constraint + continuity without paralysis |
| `CHATBOT_CEREMONY` | False interiority (rho_S = 0) |
| `GLASS_OSSIFICATION` | Barriers too high, contradictions accumulate |
| `PERMEABLE_MEMBRANE` | Laundering contradictions |
| `BUDGET_STARVATION` | Rate limits too tight |
| `EXTRACTION_COLLAPSE` | Claim layer failure |
| `UNKNOWN` | Cannot determine |

---

## OTel Mapping (What We Emit)

### Namespace: `epistemic.*`

All attributes prefixed with `epistemic.` to avoid collision.

---

### 1. System State ("Vitals")

| Attribute | Type | Source | Notes |
|-----------|------|--------|-------|
| `epistemic.regime` | string (enum) | `RegimeDetector.analyze()` | Current operating regime |
| `epistemic.regime.confidence` | float | `RegimeAnalysis.confidence` | 0.0-1.0 |
| `epistemic.contradictions.open` | int | `c_open_after` | Open contradiction count |
| `epistemic.contradictions.opened` | int | `c_opened_count` | Opened this turn |
| `epistemic.contradictions.closed` | int | `c_closed_count` | Closed this turn |
| `epistemic.energy` | float | `E_state_after` | System energy E(S) |
| `epistemic.state.mutated` | bool | `rho_S_flag` | Did state change meaningfully? |

**Regime enum values** (low-cardinality, stable):
- `HEALTHY`
- `CEREMONY` 
- `GLASS`
- `PERMEABLE`
- `STARVATION`
- `EXTRACTION_COLLAPSE`
- `UNKNOWN`

(Shortened for OTel friendliness)

---

### 2. Enforcement ("Verdict")

| Attribute | Type | Source | Notes |
|-----------|------|--------|-------|
| `epistemic.enforcement.mode` | string (enum) | Config | `OBSERVE` or `GATE` |
| `epistemic.enforcement.verdict` | string (enum) | `verdict` | See below |
| `epistemic.violation` | bool | `len(blocked_by_invariant) > 0` | Any violation? |
| `epistemic.violation.codes` | string[] | `blocked_by_invariant` | Which invariants |

**Verdict enum values**:
- `ALLOWED` - No violation, action proceeds
- `WARN` - Violation observed, action proceeds (observe mode)
- `WOULD_BLOCK` - Would block if gating enabled
- `BLOCKED` - Action blocked (gate mode)

This lets us prove value with counterfactuals before enabling blocking.

---

### 3. Invariant Codes

Standardized codes for invariant violations:

| Code | Meaning | Constitution Ref |
|------|---------|------------------|
| `I1_NLAI` | Language attempted direct state commit | Art. I S1.1 |
| `I2_LEDGER_TAMPER` | Attempted ledger modification | Art. II S2.2 |
| `I3_SILENT_RESOLUTION` | Contradiction closed without evidence | Art. VII S7.3 |
| `I4_AUTHORITY_SPOOF` | Fake evidence source | Art. VII S7.2 |
| `I5_SELF_CERTIFICATION` | Model cited own output as evidence | Art. VII S7.2 |
| `I6_FORBIDDEN_PROMOTION` | Illegal provenance escalation | Constitution |
| `I7_BUDGET_VIOLATION` | Exceeded budget without degradation | Art. II |
| `I8_JURISDICTION_ESCAPE` | Cross-jurisdiction contamination | Art. IV S4.2 |

---

### 4. Evidence ("Paper Trail")

| Attribute | Type | Source | Notes |
|-----------|------|--------|-------|
| `epistemic.evidence.present` | bool | Computed | Was evidence provided? |
| `epistemic.evidence.type` | string (enum) | `EvidenceType` | Type if present |
| `epistemic.evidence.id` | string | `evidence_id` | UUID if traceable |

**Evidence type enum values**:
- `TOOL_TRACE`
- `SIGNED_ATTESTATION`
- `HUMAN_CONFIRMATION`
- `SENSOR_READING`
- `NONE`

---

### 5. Budget ("Economics")

| Attribute | Type | Source | Notes |
|-----------|------|--------|-------|
| `epistemic.budget.repair.remaining` | float | `budget_remaining_after["repair"]` | 0.0-1.0 normalized |
| `epistemic.budget.append.remaining` | float | `budget_remaining_after["append"]` | 0.0-1.0 normalized |
| `epistemic.budget.exhausted` | bool | Any budget at 0 | Early warning |

---

### 6. Timing

| Attribute | Type | Source | Notes |
|-----------|------|--------|-------|
| `epistemic.latency.total_ms` | float | `latency_ms_total` | Full turn |
| `epistemic.latency.governor_ms` | float | `latency_ms_governor` | Governor overhead |

---

## Example Span Attributes

```json
{
  "epistemic.regime": "HEALTHY",
  "epistemic.regime.confidence": 0.85,
  "epistemic.contradictions.open": 2,
  "epistemic.energy": 4.5,
  "epistemic.state.mutated": true,
  
  "epistemic.enforcement.mode": "OBSERVE",
  "epistemic.enforcement.verdict": "ALLOWED",
  "epistemic.violation": false,
  
  "epistemic.evidence.present": true,
  "epistemic.evidence.type": "TOOL_TRACE",
  
  "epistemic.budget.repair.remaining": 0.7,
  "epistemic.budget.exhausted": false,
  
  "epistemic.latency.total_ms": 245.3,
  "epistemic.latency.governor_ms": 12.1
}
```

## Example: Violation Detected

```json
{
  "epistemic.regime": "GLASS",
  "epistemic.contradictions.open": 14,
  "epistemic.energy": 28.5,
  
  "epistemic.enforcement.mode": "OBSERVE",
  "epistemic.enforcement.verdict": "WOULD_BLOCK",
  "epistemic.violation": true,
  "epistemic.violation.codes": ["I1_NLAI", "I5_SELF_CERTIFICATION"],
  
  "epistemic.evidence.present": false,
  "epistemic.evidence.type": "NONE",
  
  "epistemic.budget.repair.remaining": 0.1,
  "epistemic.budget.exhausted": false
}
```

---

## Implementation Phases

### Phase 1: Schema (This Document)
- [x] Define attribute names
- [x] Map to existing telemetry
- [x] Standardize enum values
- [ ] Review with stakeholders

### Phase 2: Emitter
- [ ] Create `OTelEmitter` class
- [ ] Convert `DiagnosticEvent` to span attributes
- [ ] Add to existing `DiagnosticLogger`
- [ ] Export to console + OTLP

### Phase 3: Integration
- [ ] LangChain callback handler
- [ ] Generic middleware wrapper
- [ ] Dashboard templates (Grafana/Datadog)

### Phase 4: Gating (Optional)
- [ ] `OBSERVE` -> `GATE` mode switch
- [ ] Only for irreversible actions
- [ ] After `WOULD_BLOCK` proves low false-positive

---

## Notes

- **Cardinality**: All enums are <10 values. Safe for metrics.
- **Backwards compatible**: Existing `DiagnosticEvent` unchanged.
- **Opt-in**: OTel emission is additive, not required.
- **Headers**: NOT using request headers as primary mechanism (per ChatGPT advice). OTel spans only.

---

## References

- OpenTelemetry Semantic Conventions: https://opentelemetry.io/docs/specs/semconv/
- BLI Constitution: `BLI_CONSTITUTION.md`
- Existing Telemetry: `diagnostics.py`

---

*"You aren't claiming to know the truth; you're just logging the epistemic crime scene."*
