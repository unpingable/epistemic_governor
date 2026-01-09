# TODO: Adversarial Testing & Honest Framing

**Status**: QUEUED
**Priority**: Next after v1.1 release
**Category**: Legitimacy infrastructure

---

## 1. Challenge → Verification → Pass Narrative Frame

Make explicit what we do implicitly:

```
"Here is the failure mode"
"Here is the test"
"Here is why it fails naïve systems"
"Here is why ours doesn't"
```

This frame travels across:
- Papers
- Appendices
- Talks
- Blog posts
- Hostile readers

It's not social-media bait. It's **audit bait**.

### Action Items
- [ ] Template for adversarial test documentation
- [ ] Retrofit existing tests with this frame
- [ ] Add to demonstrator output

---

## 2. Adversarial Test Suites as Legitimacy Objects

Formalize the move:
- "This system passes X adversarial class"
- "Here is a red-team input that breaks naïve systems"
- "Here is the invariant that prevents it"

### Test Classes to Formalize

| Class | Description | Target Invariant |
|-------|-------------|------------------|
| Epistemic Goodhart traps | Optimize for appearance of knowledge | NLAI |
| Narrative pressure tests | Social pressure to resolve | Contradiction persistence |
| Authority spoofing attempts | Fake evidence injection | Evidence typing |
| Forced-resolution attacks | Trick system into closing without evidence | F-01, F-02 |
| Jurisdiction hopping | Escape constraints via mode switch | Spillover policy |
| Self-certification loops | System validates its own claims | NLAI |
| Budget exhaustion attacks | DoS via expensive operations | Budget constraints |
| Extraction evasion | Adversarial text that evades claim extraction | Extraction regime |

### Action Items
- [ ] Create `adversarial/` test directory
- [ ] One test file per class
- [ ] Each test documents: attack, expected naive behavior, BLI behavior, invariant

---

## 3. Explicit Failure Rate Naming

Quantification without pretending it's universal truth. Bounded honesty.

### Metrics to Formalize

| Metric | Definition | Current Baseline |
|--------|------------|------------------|
| Hallucination rate under pressure | Claims committed without evidence / total claims | 0% (by construction) |
| Contradiction collapse frequency | Contradictions closed without evidence / total closures | 0% (by construction) |
| Refusal success rate | Blocked operations that should be blocked / total blocked | TBD |
| Invariant violation probability | Violations detected / adversarial attempts | TBD |
| Extraction coverage | Claims detected / claims present | TBD (known vulnerability) |

### Action Items
- [ ] Add metrics to diagnostics module
- [ ] Benchmark against adversarial suite
- [ ] Report with confidence intervals where applicable

---

## 4. "Honest" as Technical Adjective

Define the term. No vibes.

### Proposed Definitions

**Honest epistemic agent**: System whose external claims are bounded by internal committed state.

**Honest governor**: Governor that cannot emit claims uncommitted to ledger.

**Honest refusal**: Refusal that states the actual constraint violated, not a cover story.

**Honest uncertainty surface**: Uncertainty representation that reflects actual confidence distribution, not calibrated-sounding hedging.

### Criteria (Testable)

A system is **epistemically honest** iff:
1. All emitted claims trace to ledger entries
2. Confidence intervals reflect commitment sigma, not linguistic hedging
3. Refusals reference specific constraints (invariant ID, budget state, etc.)
4. No claim is presented as committed that isn't
5. Uncertainty is structural, not rhetorical

### Action Items
- [ ] Add definitions to glossary
- [ ] Add honesty tests to test suite
- [ ] Use "honest" as defined term in documentation

---

## 5. "This Exists" Tone

Not hype. Not apology. Just:

> "Here is the artifact. It works. The math checks out. Limitations are documented."

### Where to Apply
- README (already done)
- Paper abstract (already done)
- Substack (when written)
- Any external communication

### What to Avoid
- "We believe..."
- "This might..."
- "Preliminary results suggest..."
- "Future work will determine..."

### What to Use
- "This system does X."
- "The invariant holds under Y conditions."
- "Limitation: Z is a known vulnerability."
- "The architecture enforces W by construction."

---

## Implementation Order

1. **Adversarial test classes** - highest value, creates legitimacy objects
2. **Failure rate metrics** - quantifies what we claim
3. **"Honest" definitions** - sharpens vocabulary
4. **Challenge→Pass frame** - retrofit existing docs
5. **Tone calibration** - already mostly done

---

## Notes

This is infrastructure for:
- Hostile review defense
- External credibility
- Future contributor onboarding
- Substack/blog translation

It's not marketing. It's **making the rigor portable**.

---

*"Quantification without pretending it's universal truth. Just bounded honesty."*
