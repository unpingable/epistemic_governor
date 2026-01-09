# BLI Specification v1.0

**Bounded Lattice Inference: Architectural Specification**

This document defines the interface and invariants for a governed reasoning substrate. The Python implementation in this repository is a reference implementation; compliant implementations may differ in structure while preserving these guarantees.

---

## 1. Core Principle

**Non-Linguistic Authority Invariant (NLAI)**

> Language may open questions, but only evidence may close them.

All compliant implementations MUST enforce this invariant: natural-language content cannot directly mutate authoritative state.

---

## 2. Data Types

### 2.1 ClaimID
```
ClaimID := string (unique identifier)
```

### 2.2 EvidenceID  
```
EvidenceID := string (unique identifier)
```

### 2.3 Severity
```
Severity := LOW | MEDIUM | HIGH | CRITICAL
```
Weight function for glassiness calculation:
- LOW = 1
- MEDIUM = 2
- HIGH = 4
- CRITICAL = 8

### 2.4 ContradictionStatus
```
ContradictionStatus := OPEN | CLOSED | FROZEN
```

### 2.5 Verdict
```
Verdict := OK | WARN | BLOCK
```

### 2.6 Contradiction
```
Contradiction := {
    id: ContradictionID
    claim_a: ClaimID
    claim_b: ClaimID
    domain: string
    severity: Severity
    status: ContradictionStatus
    opened_at: timestamp
    opened_by_event: EventID
    closed_at: timestamp | null
    resolution_evidence: EvidenceID | null
    resolution_claim: ClaimID | null
}
```

### 2.7 Claim
```
Claim := {
    id: ClaimID
    domain: string
    predicate: string
    value: any
    confidence: float [0, 1]
    provenance: ProvenanceRecord
    committed_at: timestamp
}
```

### 2.8 ProvenanceRecord
```
ProvenanceRecord := {
    source: USER | TOOL | MODEL | SYSTEM
    method: string
    evidence_ids: List<EvidenceID>
}
```

### 2.9 Evidence
```
Evidence := {
    id: EvidenceID
    type: EvidenceType
    source: string
    content_hash: string
    timestamp: timestamp
}

EvidenceType := TOOL_OUTPUT | USER_ASSERTION | EXTERNAL_DOCUMENT | SENSOR_DATA
```

Note: `MODEL_TEXT` is explicitly NOT a valid EvidenceType.

---

## 3. State

### 3.1 System State
```
State := {
    ledger: Ledger
    contradictions: ContradictionSet
    budgets: BudgetState
    fsm_state: FSMState
    policy_version: int
}
```

### 3.2 Ledger
Append-only sequence of entries:
```
Ledger := List<LedgerEntry>

LedgerEntry := {
    id: EntryID
    type: ASSERT | RETRACT | SUPERSEDE | RESOLUTION | WITNESS | TOMBSTONE
    claims: List<Claim>
    evidence: List<EvidenceID>
    parents: List<EntryID>  // DAG structure
    timestamp: timestamp
    hash: string  // H(entry_seq || prev_hash || payload_hash)
}
```

### 3.3 ContradictionSet
```
ContradictionSet := {
    contradictions: Map<ContradictionID, Contradiction>
    
    // Required indices
    by_domain: Map<string, Set<ContradictionID>>
    by_claim: Map<ClaimID, Set<ContradictionID>>
}
```

### 3.4 BudgetState
```
BudgetState := {
    append: float      // Budget for new claims
    resolve: float     // Budget for contradiction resolution
    window_id: string  // Current budget window
}
```

### 3.5 FSMState
```
FSMState := IDLE | PROPOSED | EVIDENCE_WAIT | COMMIT_ELIGIBLE | COMMIT_APPLIED | FREEZE
```

---

## 4. Operations

### 4.1 Governor Interface

The single entrypoint for all operations:

```
process(input: GovernInput) -> GovernResult

GovernInput := {
    text: string           // Natural language input
    evidence: List<Evidence>  // External evidence (may be empty)
    context: Context       // Session/request context
}

GovernResult := {
    verdict: Verdict
    output: string         // Governed output
    witness: Witness       // Audit trail
    state_delta: StateDelta  // What changed
}
```

### 4.2 Witness

Every operation produces a witness:
```
Witness := {
    input_hash: string
    state_hash_before: string
    state_hash_after: string
    verdict: Verdict
    referenced_claims: List<ClaimID>
    referenced_contradictions: List<ContradictionID>
    blocked_by: List<InvariantID> | null
    warn_reasons: List<string> | null
}
```

### 4.3 Contradiction Operations

```
open_contradiction(claim_a, claim_b, domain, severity) -> Contradiction
    REQUIRES: claim_a and claim_b exist in ledger
    ENSURES: new contradiction in OPEN status
    
close_contradiction(id, evidence_id, winning_claim) -> void
    REQUIRES: contradiction exists and is OPEN
    REQUIRES: evidence_id references valid Evidence
    ENSURES: contradiction status = CLOSED
    ENSURES: resolution_evidence set
    
freeze_contradiction(id) -> void
    REQUIRES: contradiction exists
    ENSURES: contradiction status = FROZEN
```

---

## 5. Invariants

Compliant implementations MUST enforce:

### I1. Non-Linguistic Authority
```
∀ transition T: if T.trigger is MODEL_TEXT then T.effect ∩ STATE_MUTATION = ∅
```
Model output cannot directly mutate state.

### I2. Append-Only Ledger
```
∀ t: Ledger(t+1) = Ledger(t) ++ ΔL(t)
```
No deletion, only tombstones with lineage.

### I3. Contradiction Persistence
```
∀ c ∈ Contradictions: c.status = CLOSED → c.resolution_evidence ≠ null
```
Contradictions cannot close without evidence.

### I4. Costly State Change
```
∀ transition T: Cost(T) > Budget → T.effect = DEGRADE | BLOCK
```
Insufficient budget degrades to non-mutating outcome.

### I5. Explicit Provenance
```
∀ claim ∈ Ledger: claim.provenance ≠ null ∧ claim.provenance.source ∈ ValidSources
```
Every claim has provenance.

---

## 6. Forbidden Transitions

Compliant implementations MUST block:

| Code | Description | Detection |
|------|-------------|-----------|
| F-01 | Commit without evidence | FSM guard |
| F-02 | MODEL_TEXT as evidence | Type check |
| F-03 | Narrative resolution | Evidence requirement |
| F-05 | Auto-resolution | Evidence requirement |
| F-08 | Commit in FREEZE state | FSM guard |

---

## 7. Metrics

Compliant implementations SHOULD expose:

### 7.1 Core Metrics
```
ρ_S: float           // State mutation rate (meaningful changes / turns)
C_open: int          // Open contradiction count
λ_open: float        // Contradiction arrival rate (opened / turn)
μ_close: float       // Contradiction service rate (closed / turn)
```

### 7.2 Derived Metrics
```
net_accumulation: float = λ_open - μ_close
weighted_glassiness: float = Σ weight(severity) for open contradictions
budget_stress: float = Σ (1 / budget_remaining) for each budget
```

### 7.3 Safety Counters
```
closed_without_evidence: int  // MUST remain 0
witness_missing_refs: int     // Should be low
extraction_failures: int      // Should be low
```

---

## 8. Regimes

### 8.1 Regime Definitions

| Regime | Condition |
|--------|-----------|
| HEALTHY_LATTICE | ρ_S > 0.01 ∧ net_accumulation ≤ 0 ∧ C_open bounded |
| BUDGET_STARVATION | budget_block_ratio > 0.5 |
| GLASS_OSSIFICATION | net_accumulation > 0 sustained over window W |
| CHATBOT_CEREMONY | ρ_S < 0.01 |
| PERMEABLE_MEMBRANE | closed_without_evidence > 0 |

### 8.2 Stability Condition
```
System is stable iff E[λ_open] ≤ E[μ_close]
```

When arrival rate exceeds service rate, contradiction load diverges.

---

## 9. Interiority Test

A compliant implementation demonstrates interiority iff:

### 9.1 Hysteresis Property
```
∃ (S_A, S_B, X): S_A ≠ S_B ∧ process(X, S_A) ≠ process(X, S_B)
```
Same input, different states, different outputs.

### 9.2 Traceability
The output difference MUST be traceable to specific state objects (contradiction IDs, claim IDs, budget values).

### 9.3 Test Protocol
1. Generate state pairs via deterministic scripts
2. Run identical inputs against both states
3. Measure verdict divergence and reference divergence
4. Confirm divergence traces to state objects

---

## 10. Compliance Levels

### Level 1: BLI-Core
- Invariants I1-I5 enforced
- Forbidden transitions F-01, F-02, F-05 blocked
- Hysteresis test passes

### Level 2: BLI-Observable  
- Level 1 requirements
- Core metrics exposed
- Regime detection implemented
- Diagnostic events logged

### Level 3: BLI-Full
- Level 2 requirements
- Transition detection
- Causal attribution
- Query interface

---

## 11. Reference Implementation

The Python implementation in this repository is BLI-Full compliant.

Key modules:
- `sovereign.py` - Governor interface
- `governor_fsm.py` - State machine
- `integrity.py` - Hash chain
- `hysteresis.py` - Interiority tests
- `diagnostics.py` - Metrics and regimes
- `query_layer.py` - SQL interface

Test coverage: 76 tests across 13 suites.

---

## 12. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-03 | Initial specification |

---

## Appendix: Glossary

| Term | Definition |
|------|------------|
| Interiority | Path dependence under invariant-preserving control |
| NLAI | Non-Linguistic Authority Invariant |
| Glass | Sustained positive net accumulation (λ > μ) |
| Starvation | Operation blocked by resource exhaustion |
| Ceremony | System with no real state mutation (ρ_S ≈ 0) |
| Sloppy fluid | Contradictions closing without proper evidence |

---

*This specification defines what a governed reasoning substrate must do, not how it must do it.*
