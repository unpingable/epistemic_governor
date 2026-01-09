# Constraint Kernels: Design Notes for v2

**Status**: DESIGN PHASE - Not yet implemented
**Decision**: CLOSED - SAT + DL-Lite is the foundation

---

## The Core Invariant

> **The constraint layer exists to say "no" with absolute clarity — never to say "what should happen instead."**

Constraint kernels are **hostile admissibility oracles**, not reasoners.

These are formal control layers over probabilistic systems — not "symbolic AI" in the GOFAI sense.

They answer:
- `ALLOW / FORBID`
- `CONSISTENT / INCONSISTENT`  
- `TYPED / ILL-TYPED`

They never answer:
- "What should we do instead?"
- "What's better?"
- "How should thresholds change?"

---

## What Constraint Kernels Are NOT

- Not a reasoner
- Not a planner
- Not a hypothesis generator
- Not adaptive
- Not optimizing

The moment you add:
- Search over possible states
- Preference ordering beyond hard constraints
- Any form of "best" instead of "allowed/forbidden"
- Learning or adaptation in the constraint layer

...you've smuggled agency back in through the side door.

---

## Tier 1: Natural Fit (v2 Foundation)

### SAT/CSP (non-optimizing) ⭐⭐⭐⭐⭐

**Purpose**: Forbidden transition enforcer

- Pure admissibility checking with no optimization
- Well-understood complexity bounds
- **CRITICAL**: Never touch MaxSAT or add objective functions
- Maps directly to FSM forbidden transitions

### Description Logics (minimal fragment) ⭐⭐⭐⭐⭐

**Purpose**: Contradiction detection

- EL or DL-Lite only — stay decidable
- No reasoning about actions or plans
- Contradiction detection is literally what DL is for
- Maps directly to contradiction objects

---

## Tier 2: Useful But Watch Closely

### Datalog (strict monotonic) ⭐⭐⭐⭐☆

**Purpose**: Ledger reasoning and provenance

- Great for "what follows from committed state"
- **DANGER**: Easy to drift into non-monotonic extensions
- Keep it pure: facts accumulate, never retract
- Only if needed for explicit provenance tracking

### Typed Lambda Calculus (as type checker only) ⭐⭐⭐☆☆

**Purpose**: Evidence typing and jurisdiction boundaries

- **DANGER**: Type inference can smuggle in search
- Keep it as **checking**, not inference
- More complex than probably needed initially

---

## Tier 3: Probably Don't

### Temporal Logics (LTL/CTL) ⭐⭐⭐☆☆

- Could formalize state machine properties
- **DANGER**: Model checking is expensive and can hide optimization
- Only if explicit temporal reasoning absolutely required

### Everything Else

- You almost certainly don't need it
- If you think you do, you're probably solving the wrong problem

---

## Implementation Approach

> **Reuse the math. Write the interface.**

### Why not big libraries?
- Hidden heuristics
- Implicit optimization
- Debug opacity
- Assumptions you don't control

### Why not full reinvention?
- Wasteful
- Error-prone
- Unnecessary

### The correct move
- Implement **tiny**, auditable kernels
- One file per kernel if possible
- Clear worst-case behavior
- Deterministic outputs
- No background processes

**If someone can't understand the kernel in one sitting, it's too big.**

---

## Kernel Interface (Conceptual)

```python
class ConstraintKernel:
    """
    A hostile admissibility oracle.
    
    Formal control layer, not symbolic reasoning.
    
    Guarantees:
    - No side effects
    - No state mutation
    - No search beyond bounded check
    - Deterministic output
    """
    
    def check(self, claims: List[Claim], constraints: List[Constraint]) -> Verdict:
        """
        Returns one of:
        - ALLOW: No constraint violated
        - FORBID: Hard constraint violated (with reason)
        - INCONSISTENT: Logical contradiction detected (with parties)
        """
        pass
```

---

## Self-Modification: The Safe Kind

### State Partition (Critical for v2)

**S₀ — Constitutional layer (IMMUTABLE)**
- NLAI
- FSM topology
- Constraint kernel semantics
- Evidence typing rules
- Forbidden transitions

**S₁ — Regulatory parameters (adaptive, bounded)**
- Budgets
- Thresholds
- Rates
- Timeouts
- Load shedding rules

**S₂ — Epistemic state (fully mutable)**
- Claims
- Contradictions
- Ledger entries
- Provenance

### Enforcement Rule

```
S₂ may influence S₁
S₁ may NOT influence S₀
S₂ may NOT influence S₀
Language may influence NONE directly
```

### The Invariant

> **No internal process may modify the conditions under which it would have been forbidden.**

Constraint kernels may **block** transitions.
They may never **authorize new kinds** of transitions.
Adaptation may occur **only inside pre-authorized regions**.

---

## Why This Isn't Agency

If someone asks "isn't this still agency?", the answer:

> *If this is agency, it's the agency of a circuit breaker that can change how quickly it trips — but not what counts as a short.*

We're not balancing constraint enforcement *against* self-modification.
We're using constraint enforcement to **make self-modification safe**.

This is **Ashby ultrastability**:
> Change the *parameters of regulation*, not the *laws being regulated by*.

---

## Implementation Phases

### Phase 1 (v2): SAT + minimal DL
- SAT for forbidden transition checking
- DL-Lite for contradiction semantics
- Both are mature, have proven implementations, and are **boring**

### Phase 2 (v2.x): Datalog for ledger reasoning
- Only if explicit provenance tracking needed
- Only monotonic
- Only after SAT+DL are working

### Phase 3 (probably never): Anything else

---

## The Meta-Point

The constraint layer should be so boring it's almost disappointing.

If someone looks at it and says "that's it?" — **good**.
If they say "why didn't you use [fancier thing]?" — **good**.

The power isn't in the constraint components. It's in the **composition** — how they sit under the governor's authority without ever becoming actors themselves.

---

## References

- Ashby, W.R. (1956). *An Introduction to Cybernetics*. (Ultrastability)
- Baader et al. (2003). *The Description Logic Handbook*. (DL-Lite, EL)
- Biere et al. (2009). *Handbook of Satisfiability*. (SAT algorithms)

---

*"The real innovation is the composition, not the kernels."*
