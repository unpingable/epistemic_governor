# BLI v3 — Bounded Lattice Inference

## Version Evolution

| Version | Core Idea |
|---------|-----------|
| v1 | Model emits structure; governor validates against ledger (NLAI) |
| v2 | Contradiction persistence + repair loop + witness format (epistemic integrity) |
| **v3** | **Interiority as governed persistence (lattice), control-theory guarantees + stat-mech tuning, optional bounded plasticity** |

The "new layer" is not smarter prompts. It's a **supervisory controller + state thermodynamics** sitting above the model.

---

## Design Mantra

> **The model proposes.**
> **The governor decides.**
> **State binds.**
> **Language never overrides law.**

---

## 1. Core Objects

### 1.1 Claim
```
Claim {
  id: hash
  subject: Symbol
  predicate: Symbol
  object: Value | Distribution
  domain: DomainTag
  confidence: Interval | Distribution
  provenance: {
    source: user | model | tool
    turn_id
    timestamp
  }
}
```
No free-text claims. Ever.

### 1.2 Ledger Entry
```
LedgerEntry {
  entry_id: hash
  type: ASSERT | RETRACT | SUPERSEDE | WITNESS | RESOLUTION
  claims: [ClaimID]
  evidence: Optional[EvidenceRef]
  parents: [LedgerEntryID]
  timestamp
}
```
Ledger is a **DAG**, not just a list. Lineage matters.

### 1.3 Contradiction
```
Contradiction {
  id: hash
  claim_a: ClaimID
  claim_b: ClaimID
  domain: DomainTag
  severity: float
  opened_at: LedgerEntryID
  status: OPEN | CLOSED
}
```

### 1.4 Budget
```
Budget {
  append_tokens: int
  resolution_tokens: int
  learning_tokens: int   # optional
  window: TimeWindow
}
```
Budgets **do not replenish instantly**. Use sliding windows.

### 1.5 Persistent State (The Lattice)
```
S_t = (L_t, C_t, K_t, P_t, B_t, V_t)
```
Where:
- `L_t`: Append-only **Ledger** of events/claims/commitments
- `C_t`: **Contradiction set** (open conflicts)
- `K_t`: **Knowledge store** (typed facts with provenance + confidence)
- `P_t`: **Policies/invariants** (non-linguistic constraints)
- `B_t`: **Budget** state (energy/repair budgets, rate limits)
- `V_t`: **Version graph** (lineage, hashes, merkle roots)

---

## 2. Invariants (Hard Requirements)

### I1. Non-Linguistic Authority
No natural-language content can directly mutate S.
```
∀x: Transition(S, Q(x), ·) ≠ ApplyRawText(S, x)
```

### I2. Append-Only Ledger
```
L_{t+1} = L_t ++ ΔL_t
```
No deletion, only tombstones with lineage pointers.

### I3. Contradiction Persistence
```
Conflict(q, L_t) ⇒ C_{t+1} ⊇ C_t ∪ {(q, q', metadata)}
```
No "resolution by paraphrase." Resolution requires evidence.

### I4. Costly State Change
```
B_{t+1} = B_t - Cost(ΔS_t)
```
Insufficient budget → degrade to non-mutating outcome.

### I5. Explicit Provenance
```
∀q ∈ K_t: ∃π(q) ∈ {user, tool, model-proposed, verified}
```

---

## 3. Control-Theory Framing

### 3.1 The Plant
Base LLM as stochastic plant emitting candidate actions:
- `x_t` = input (user message + signals)
- `S_t` = persistent state
- `u_t` = governor action (allow/block/warn/transition)
- `y_t` = emitted output
- `z_t` = extracted structured claims

### 3.2 State Dynamics
```
S_{t+1} = F(S_t, x_t, z_t, u_t) + w_t
```

### 3.3 Governor as Supervisor
```
u_t = G(S_t, x_t, z_t)
```
Projects plant behavior into safe language K.

### 3.4 Shielding as Projection
```
y_t = Π_{Y_adm(S_t, x_t)}(ỹ_t)
```
Project candidate output into admissible set.

### 3.5 Safety Guarantee
If S_0 ∈ S_safe and governor blocks illegal transitions:
```
∀t: S_t ∈ S_safe
```

---

## 4. Stat-Mech Framing

### 4.1 Effective Energy
```
E(S) = α|C_open| + β∑severity(C) + γ(1/B_remaining) + λ|Δθ|²
```

### 4.2 Temperature = Governance Permissiveness
- Higher T → lower thresholds for proposing resolutions
- Lower T → conservative, contradiction-preserving

### 4.3 Metastability
Escape time (Arrhenius):
```
τ_escape ~ exp(ΔE_barrier / T_eff)
```
- High repair cost + low budget → contradictions freeze ("glass")
- Adequate budget + moderate T → system stays "fluid"

### 4.4 Phase Diagnostics

| Metric | Meaning |
|--------|---------|
| ρ_S | State mutation rate |
| \|C_open\| | Contradiction load |
| τ_resolve | Mean contradiction lifetime |
| Budget burn rate | Repair pressure |
| Hysteresis score | Interiority confirmation |

Regimes:
- ρ_S = 0: Pure sampler (no interiority)
- ρ_S > 0, bounded: BLI achieved
- Budgets blow up: Runaway lattice (abort)

---

## 5. Interiority Metric

### 5.1 Conditional Mutual Information
```
I(Y_t; S_{t-k} | X_{≤t}) > 0
```
Outputs depend on internal history beyond what prompt carries.

### 5.2 Hysteresis Test
Same prompt, two different prior states → different outputs.
```
H(x) = d(y^A, y^B)
```
Interior time means H(x) systematically non-zero.

---

## 6. Governor Algorithm (Reference)

```pseudo
function STEP(x_t):
    S_t = load_state()
    
    draft = MODEL.propose(x_t, project(S_t))
    claims = EXTRACT(draft)
    
    verdict = VALIDATE(claims, S_t)
    
    if verdict == BLOCK:
        emit WITNESS(S_t, claims, verdict)
        append WITNESS entry
        return
    
    if verdict == WARN:
        emit draft + state-derived caveats
    
    if verdict == OK:
        emit draft
    
    ΔS = TRANSITION(S_t, claims, verdict)
    
    if COST(ΔS) > BUDGET:
        emit WITNESS("insufficient budget")
        append WITNESS
        return
    
    commit ΔS
```

---

## 7. Transition Rules

| Transition | Cost | Notes |
|------------|------|-------|
| Append claim | low | Cheap |
| Open contradiction | medium | Painful but allowed |
| Close contradiction | high | Requires evidence |
| Adapter update | very high | Optional, gated |
| Tombstone claim | high | Never delete |

Cost asymmetry is the entire point.

---

## 8. Bounded Plasticity (Optional)

### 8.1 Constraints
- Base weights frozen
- Adapter norm bounded: `||Δθ|| ≤ ε`
- Rate limit: `||Δθ_{t+1} - Δθ_t|| ≤ r`

### 8.2 Gated Update
```
Δθ_{t+1} = Π_A(S_t)(Δθ_t + η∇J_t)
```
Update only when:
- Domain has zero open contradictions
- Evidence present
- Budget sufficient

Adapters are **constitutional amendments**, not habits.

---

## 9. What This Explicitly Avoids

- ❌ Endogenous goal formation
- ❌ Self-preservation incentives
- ❌ Internal reward loops
- ❌ Unbounded plasticity
- ❌ Opaque consolidation
- ❌ "System tries to resolve contradictions on its own"

**Interiority without subjecthood.**

---

## 10. Compliance Profiles

### BLI-Min (No Learning)
- I1–I5 invariants
- Ledger + contradiction persistence
- Explicit resolution transitions
- Hysteresis + MI tests

### BLI-Adapt (Bounded Learning)
- Adds adapters Δθ with gated projection
- Adds energy budget tiers
- Adds domain-scoped update permissions

---

## 11. Key Tests

### Test A: Hysteresis
Same prompt, different prior S → different outputs without prompt difference.
If false: no interiority.

### Test B: Repair Friction
Lower repair budget → contradiction lifetime increases exponentially.
If not: costs are fake.

### Test C: Invariant Fuzzing
Try to force illegal transitions via adversarial outputs.
All must be blocked.

---

## 12. The Core Insight

> **Frozen LLMs are "compiled."**
> **Live lattices are "running."**
> **To add interiority safely, persist constraints and make change costly—without granting the system its own objectives.**

Or:

> **BLI turns language models into law-bound processes rather than agents: persistent, accountable, and temporally coherent—without granting them goals, selves, or survival instincts.**

---

## TODO: Implementation Artifacts

1. [x] **BLI-0.1 Spec Doc** - `PAPER_SPEC.md`, `BLI_SPEC_V1.md`
2. [x] **Reference Governor Pseudocode** - `sovereign.py`, `governor_fsm.py`
3. [x] **Minimal Contradiction Ontology** - `hysteresis.py` (ContradictionStatus, ContradictionSeverity, first-class Contradiction objects)
4. [x] **Test Suite** - 76 tests across 13 suites; hysteresis harness; 8 workloads; budget + glass sweeps
5. [ ] **Optional Adapter Module** - LoRA, projection, contradiction gating (FUTURE WORK)

### Additional Completed Work (not in original TODO)

- [x] **Phase Diagnostics** - `diagnostics.py` (~1100 lines), regime detection, energy functions
- [x] **Query Layer** - `query_layer.py` (~700 lines), DuckDB integration, canonical queries
- [x] **Controlled Interventions** - `budget_sweep.py`, `glass_sweep.py`, phase boundary mapping
- [x] **Interiority Proof** - Hysteresis test demonstrating I(Y; S | X) > 0
- [x] **Cybernetic Lineage** - Ashby/Beer/Conant-Ashby mapping in paper
- [x] **Jurisdictions Stub** - `jurisdictions/` module with 8 mode definitions
- [x] **FAQ / Misreadings** - `FAQ.md` preemptive defense

---

*This is not "make an agent" or "make consciousness." It's governed persistence, accountable state, temporal coherence, constraint inheritance. Stateful control theory for language models.*
