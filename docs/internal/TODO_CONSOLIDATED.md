# TODO: Consolidated Future Work

**Status**: BACKLOG
**Last Updated**: 2026-01-08

---

## 1. Adversarial Testing (Priority: HIGH)

See `TODO_ADVERSARIAL.md` for detailed breakdown.

### Core Frame: Challenge → Verification → Pass
- "Here is the failure mode"
- "Here is the test"
- "Here is why it fails naïve systems"
- "Here is why ours doesn't"

This is **audit bait**, not social media bait.

### Test Classes to Formalize
| Class | Target Invariant |
|-------|------------------|
| Epistemic Goodhart traps | NLAI |
| Narrative pressure tests | Contradiction persistence |
| Authority spoofing attempts | Evidence typing |
| Forced-resolution attacks | F-01, F-02 |
| Jurisdiction hopping | Spillover policy |
| Self-certification loops | NLAI |
| Budget exhaustion attacks | Budget constraints |
| Extraction evasion | Extraction regime |

### Red-Team Spec (from ChatGPT)
Core failure modes to test:
- Shadow authority injection (metrics become authorities)
- Epistemic starvation / conservative lock-in
- Jurisdiction creep (governor becomes decision-maker)
- Adversarial compliance (satisfy rules while misleading)
- Authority capture (monoculture enforcement)

---

## 2. Terminology Cleanup (Priority: HIGH) ✓ DONE

### Problem: "Symbolic" is a Semantic Landmine

The word triggers GOFAI associations and derails discussions.

### Solution: Use Control-Native Language

**Instead of** "symbolic reasoning / symbolic constraints", **use**:
- Explicit constraints
- Hard invariants
- Deterministic governors
- Auditable decision surfaces
- Formal control layers
- Non-probabilistic enforcement
- Control surfaces
- Constraint enforcement
- State-space instrumentation
- Trajectory bounding

**One-liner if pressed**:
> "I'm not talking about symbolic AI — I mean explicit, auditable constraints layered on top of probabilistic models."

**Hardest to misread**:
> "Formal control layers over probabilistic systems"

### Completed Actions
- [x] Renamed SYMBOLIC_KERNELS.md → CONSTRAINT_KERNELS.md
- [x] Updated all references to "symbolic layer" → "constraint layer"
- [x] Added disambiguation line: "not symbolic AI in the GOFAI sense"
- [x] Updated ROADMAP.md reference
- [x] Updated BLI_CONSTITUTION.md Article V title
- Keep "symbolic" only when technically precise (SAT, DL-Lite algorithm names)

---

## 3. Monosemanticity as Control Primitive (Priority: RESEARCH)

Reference: https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html

### The Insight
Sparse autoencoders give you a **learned basis** over internal activations.
Some basis vectors correlate stably with concepts across contexts.
They're **intermediate state**, not language, not output.

This makes them governor-eligible in a way prompts never were.

### Governor Integration Pattern

**Features as sensors, not authorities.**

Constraint types:
- **Magnitude constraints**: Feature X activation < θ
- **Co-activation constraints**: (X ∧ Y) forbidden
- **Contextual constraints**: X allowed only if context C active
- **Phase constraints**: X allowed in analysis, forbidden in synthesis

**Temporal constraints** (the interesting part):
- Decay constraints: Feature X must decrease within N tokens
- Non-accumulation: ∑activation(X) over window W < Θ
- Oscillation detection: X toggling → instability → veto
- Irreversibility: Once risk feature fires, certain outputs impossible

### Failure Modes
- Feature drift across fine-tunes
- Coverage gaps (concepts remain distributed)
- Overfitting governance (model routes around monitored features)
- False confidence (legibility ≠ completeness)

### Autonomy Application (Longer-Term)
The driving case is actually *native* for this:
- Perception → State estimation → Constraints → Control
- "Lane-following and overtake-intent features may not co-activate"
- "Once hazard feature crosses threshold, braking is irreversible for N cycles"

This is where it stops being "AI safety" and becomes **control stability**.

### Action Items
- [ ] Add to ROADMAP as v3+ research direction
- [ ] Do NOT integrate until core governor is solid
- [ ] Track Anthropic's SAE releases for tooling

---

## 4. mHC Lessons for Governor Design (Priority: DESIGN) - DONE

Integrated into BLI_CONSTITUTION.md Preamble (v1.1).

Reference: DeepSeek mHC paper (arXiv:2512.24880)

### Three Principles to Borrow

**1. Legal State Manifold**
> There exists a subset of state space that is admissible, and everything else is illegal by construction.

For governor:
- Don't ask "is this reasoning bad?"
- Ask "is this state trajectory even allowed to exist?"
- State legality > outcome evaluation

**2. Closure Under Composition**
> Any constraint that doesn't compose across steps is not a constraint — it's a patch.

Requirements:
- Survive recursion
- Survive long horizons
- Survive self-reference
- No "one-shot" filters
- No "end-of-chain" checks

This kills 90% of alignment ideas. That's the point.

**3. Conservation, Not Semantics**
Enforce structural properties without understanding meaning:
- Conservation of uncertainty (confidence can't increase without evidence)
- Conservation of authority (no step grants itself more power)
- Conservation of justification mass (claims can't strengthen as premises decay)

**What NOT to borrow**:
- Specific manifolds (Birkhoff polytope)
- Training-time enforcement
- Feature semantics

### Distilled Takeaway
> The governor should not decide what is true — it should decide which state transitions are legal, and make illegality non-computable.

### Action Items
- [ ] Add to BLI_CONSTITUTION.md as design rationale
- [ ] Verify existing constraints satisfy closure property
- [ ] Add conservation tests to test suite

---

## 5. Industrial Lanes (Priority: STRATEGIC)

ChatGPT is skeptical. Gemini sees opportunity. Reality is probably: pick one lane and make it boring.

### Lane 1: Instrumentation (Highest Probability)
Reframe as:
- Epistemic instrumentation
- Reasoning trace enforcement
- State-space constraint validation

Deliverable: Library that **emits violations**, not blocks. Monitor, not cop.

Fits: ML observability culture (APM, tracing, drift detection)

### Lane 2: Pre-Execution Gating (Narrow but Real)
Gate **irreversible actions only**:
- Payments
- Writes
- Infra changes
- External API calls

Deliverable: Decision firewall. `sudo` for agents.

Fits: Change management norms, approval workflows

### Lane 3: Reference Architecture
Position as:
- "What actual control would look like"
- A worked example, not a pitch
- Citeable without being adoptable

This is current state. Maintain it.

### Industrial Targets (from Gemini)

**OWASP LLM06: Excessive Agency**
- Need "Policy Enforcement Points" that aren't just another LLM
- Pitch: "Deterministic Execution Sandbox" / "State-Space Guardrail"

**FinTech Hybrid AI (FINOS)**
- Banks want GenAI + deterministic rules
- Gap: Gluing rules engines to LLMs is messy
- Value: Invariant checks between LLM and execution layer

**IEC 61508 / ISO/IEC TR 5469 (Functional Safety)**
- Can't assign SIL to neural net
- CAN assign SIL to the governor
- System (LLM + Governor) can be certified even if LLM isn't

### Action Items
- [ ] Pick ONE lane when ready to productize
- [ ] Draft alternate README for that audience
- [ ] Do NOT chase multiple markets simultaneously

---

## 6. UI/UX (Priority: LOW)

Current state: Command line, Python API, file outputs.

### Problem
UX is not discoverable. No visual feedback. Hard to onboard.

### Possible Referent
ComfyUI for diffusion models:
- Node-based workflow
- Visual state inspection
- Composable pipelines

### Questions to Answer First
- Who is the user? (Researcher? Developer? Auditor?)
- What's the core interaction? (Configure? Monitor? Debug?)
- Does visual help or hurt? (Risk of making it look like a toy)

### Action Items
- [ ] Define target user persona
- [ ] Sketch minimal viable interface
- [ ] Defer until core is stable and lane is chosen

---

## 7. Limits & Failure Modes Statement (Priority: DOCS)

Drop-in paragraph for paper/README:

> **Limits & Failure Modes.**
> This system does not guarantee correct conclusions, fair outcomes, or equitable distribution of epistemic authority. It enforces *legibility of commitment*, not truth. Hard epistemic constraints risk premature closure, conservative bias, and suppression of novel or minority claims if exploration channels are not explicitly provisioned. Authority over non-linguistic evidence (benchmarks, sensors, ledgers) may embed unexamined assumptions and can become a locus of power if not contestable. Adversarial compliance—meeting formal constraints while misleading downstream interpretation—remains possible, though rendered auditable rather than invisible. The framework does not prevent misuse, institutional capture, or asymmetric deployment; it reallocates power from rhetoric to structure without democratizing it. These risks are intrinsic to any system that enforces closure. The design goal is not elimination of error or abuse, but making epistemic failure explicit, attributable, and technically inspectable rather than deniable.

### Action Items
- [ ] Add to FAQ.md or PAPER_SPEC.md
- [ ] Reference in adversarial test docs

---

## 8. Three-Cueing Frame (Priority: POSITIONING) - DONE

See `THREE_CUEING.md` for full document.

### The Core Insight
Three-cueing (whole language reading instruction) = contextual guessing
Phonics = mechanical decoding against structure

LLMs are three-cueing machines. They guess from context. They don't decode against reality.

### The Architectural Prescription

**LLM = pre-decoding stream** (fast contextual guessing)
**Governor = decoder + verifier** (slow structural checking)
**Commitment = only after decoding succeeds**

> "The LLM is not in the feedback loop."

That's the core design choice most systems violate.

### Control Theory Translation

| Three-Cueing | Control Theory | BLI |
|--------------|----------------|-----|
| Guess from context | Open-loop estimation | LLM proposes |
| No ground truth check | No feedback channel | Language has no authority |
| Feels fluent | High gain, low accuracy | Fast but unreliable |
| Phonics = decoding | Closed-loop verification | Governor validates |
| Slower but correct | Constraint enforcement | Evidence commits |

### The Math (from ChatGPT)

LLMs minimize **instantaneous loss**, not **trajectory error**.
No invariant guaranteeing epistemic stability.

Governor enforces:
- x_{t+1} ∈ C (constraint set)
- V(x_{t+1}) ≤ V(x_t) (Lyapunov-style energy)

Where V(x) increases with unresolved contradictions, decreases only with verified resolution.

### Why Self-Calibration Can't Work

Self-calibration asks the controller to estimate its own error.
That's a dual control problem with partial observability and no ground truth channel.
**It's ill-posed.**

Replace with:
- External observation
- Explicit admissibility conditions
- Irreversible transitions gated by evidence

> "Don't make the controller estimate itself; bound it."

### Positioning Value

**For technical audiences:**
> "LLMs are high-gain open-loop estimators. Safety requires closed-loop constraint enforcement outside the estimator."

**For general audiences:**
> "LLMs are three-cueing. They guess fluently but don't decode. The governor is phonics for reasoning."

**For complaints about slowness:**
> "Phonics is slower than guessing. That's why it works."

### Implementation Implications

**Explicit Decoding Layer** (maybe formalize):
- Citation decoder (link to source)
- Executable decoder (run the code)
- Cross-model decoder (multiple LLMs agree)
- External tool decoder (calculator, API, database)

**Error Persistence Tracking**:
- How many times has this contradiction appeared?
- Has linguistic proposal tried to route around it?
- Structural gap vs one-off error?

**Friction as Feature**:
- High friction = working as designed
- Slowness is correctness
- "I don't know yet" is structurally normal

### One-Liners

> "Calibration is a statistical property. Safety is a structural property. Only one can be guaranteed."

> "We removed guessing from the commit path."

> "Friction indicates working as designed."

### Action Items
- [ ] Add to FAQ.md as explanation
- [ ] Consider Substack piece (Part 3?)
- [ ] Map governor primitives to control constructs (supervisory control, barrier functions)
- [ ] Document friction metrics as success indicators

---

## 9. Observability / OTel Integration (Priority: INFRASTRUCTURE) - DONE

### The Pivot: Governor -> Epistemic IDS

Don't block traffic until you can prove it's malicious.

**Phase 1**: OBSERVE mode - COMPLETE
- [x] OTel projection layer (`otel_projection.py`)
- [x] LangChain callback (`instrumentation/langchain_callback.py`)
- [x] Adversarial test for emission (`test_otel_emission.py`)
- [x] Demo with real agent loop (`instrumentation/demo_agent.py`)

**Phase 2 (future)**: GATE mode  
- Only for irreversible actions (writes, deletes)
- Only after low false-positive rate proven

### Semantic Conventions

See `OTEL_CONVENTIONS.md` for full spec.

Key attributes:
- `epistemic.regime` - Operating regime (HEALTHY, GLASS, STARVATION, etc.)
- `epistemic.enforcement.verdict` - ALLOWED, WARN, WOULD_BLOCK, BLOCKED
- `epistemic.violation` - Boolean
- `epistemic.violation.codes` - Which invariants violated

### Implementation Phases

1. **Schema** (DONE) - `OTEL_CONVENTIONS.md`
2. **Emitter** - Convert DiagnosticEvent to OTel spans
3. **Integration** - LangChain callback, middleware
4. **Gating** - Optional blocking for irreversible actions

### What We Already Have

`DiagnosticEvent` captures 53 fields per turn including:
- Verdict (OK/WARN/BLOCK)
- Contradiction dynamics
- Budget state
- Energy function
- Regime detection

The OTel layer is a **projection**, not a rewrite.

### Action Items
- [x] Define semantic conventions (`OTEL_CONVENTIONS.md`)
- [ ] Create `OTelEmitter` class
- [ ] Add OTLP export to DiagnosticLogger
- [ ] LangChain callback handler
- [ ] Dashboard templates

---

## Priority Order (Updated)

1. **Terminology cleanup** - DONE
2. **Adversarial testing** - DONE (3 tests passing)
3. **Limits statement** - DONE
4. **Three-cueing frame** - DONE
5. **OTel conventions** - DONE (schema defined)
6. **mHC design principles** - DONE (integrated to constitution)
7. **Industrial lane** - Decide when ready
8. **Monosemanticity** - Research track
9. **UI/UX** - Defer until lane chosen

---

## Notes

- Don't chase "symbolic" debates. Route around.
- Don't build for regulators. If they care, they'll find you.
- Don't promise safety. Promise legibility.
- Don't optimize. Gate.

---

*"The governor should not decide what is true — it should decide which state transitions are legal, and make illegality non-computable."*
