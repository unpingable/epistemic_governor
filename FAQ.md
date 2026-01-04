# Frequently Asked Questions

**Preemptive clarifications for predictable misreadings.**

---

## On Consciousness and Inner Life

### Q: Does this give LLMs an inner life / consciousness?

**No.**

Interiority here means *path-dependent state under constraint*. It is a property of the system architecture, not subjective experience.

The architecture explicitly lacks:
- Goals
- Preferences
- Self-models
- Self-preservation mechanisms

A thermostat has path-dependent state. That doesn't make it conscious. Neither does this.

### Q: But you call it "interior time" — doesn't that imply something?

It implies outputs depend on history beyond the current prompt. That's it.

A database has "interior time" in this sense. So does a version control system. The term is borrowed from systems theory, not phenomenology.

---

## On Agency and Agents

### Q: Isn't this basically an agent?

**No.**

Agents optimize objectives. This system has no objectives.

Key differences:
- Budgets are *constraints*, not rewards
- Resolution is driven by *external evidence*, not internal motivation
- The system cannot form goals, pursue them, or resist shutdown

If you remove the evidence stream, the system doesn't try to find evidence. It just blocks.

### Q: Could it become an agent if you added X?

If you added goal-seeking, planning, and reward optimization, yes — but then it would be a different system. You could also add wheels to a chair and call it a car.

The architecture is specifically designed to be *non-agentic*. That's a feature, not a limitation.

---

## On Alignment

### Q: Is this an alignment solution?

**No.**

Alignment concerns values. This system concerns *epistemic integrity* — preventing unverified claims from becoming authoritative state.

The system doesn't know what's good or bad. It knows what's *contradicted* and what's *unresolved*. Those are different problems.

### Q: Doesn't refusing to answer contested questions count as "aligned" behavior?

Only in the same sense that a locked door is "aligned" with keeping people out.

The system blocks because invariants require it, not because it evaluated the ethics of the situation. Architecture, not judgment.

### Q: Could this be *combined* with alignment work?

Potentially. Epistemic integrity is probably a prerequisite for meaningful value alignment — it's hard to align something that can't maintain consistent beliefs.

But that's speculation. We make no alignment claims.

---

## On Safety and Danger

### Q: Does this make models safer or more dangerous?

**Neither by default.**

It makes their behavior *auditable*.

In many cases this reduces risk:
- Forces refusals on contested domains
- Preserves uncertainty explicitly
- Creates audit trails for decisions

But it also:
- Removes plausible deniability
- Makes state inspectable
- Creates records that persist

Whether that's "safer" depends on your threat model.

### Q: Could this be used to make AI systems more deceptive?

The architecture is specifically designed to make deception *structurally difficult*:
- Contradictions persist and are visible
- Closures require evidence
- State mutations leave traces

Could someone build a different system that's deceptive? Sure. But this one isn't it.

---

## On Scaling and Deployment

### Q: Could this scale?

**Unknown.** The paper makes no scaling claims.

Stability conditions are architectural — the queueing theory holds regardless of scale. But performance characteristics (latency, throughput, extraction accuracy) are domain-dependent and untested at production scale.

### Q: Why hasn't this been done already?

Persistent state with auditability introduces complexity:

- **Legal**: Audit trails create discoverable records
- **Operational**: State management is harder than stateless
- **Product**: "I don't know" is worse for engagement than confident hallucination

Stateless systems are simpler to deploy and easier to disclaim responsibility for.

### Q: Is this production-ready?

**No.** This is research code demonstrating architectural principles. The claim extraction layer in particular is a known vulnerability — real deployment would require hardening.

---

## On the Research Itself

### Q: What did you actually prove?

Specifically:

1. **Interiority exists**: Same input + different state = different output (hysteresis test)
2. **Stability is throughput-determined**: λ_open vs μ_close predicts regime (queueing theory)
3. **Two failure modes have distinct signatures**: Budget starvation (sharp), glass accumulation (gradual)
4. **Safety invariants hold under intervention**: Increased capacity doesn't cause laundering

### Q: What did you *not* prove?

Explicitly:

1. No consciousness (not even attempted)
2. No agency (architecturally excluded)
3. No alignment (orthogonal problem)
4. No scaling (untested)
5. No adversarial robustness (known weakness)

### Q: Is this peer-reviewed?

Not yet. This is a technical report / preprint equivalent.

---

## On Implications

### Q: What does this mean for AI development?

We don't speculate on implications we didn't test.

The narrow claim is: you can add persistent state to language model systems without adding agency, and the resulting behavior follows predictable dynamics from queueing theory.

What that *means* for the field is for others to decide.

### Q: Are you saying current AI systems are "lying"?

No. Current systems are stateless — they can't "lie" because they can't remember what they said.

This system can't lie *in a different way*: it can't pretend contradictions don't exist once they're recorded.

Different failure modes, not moral judgment.

---

## The One Answer

If you're still confused about what this is:

> We didn't make the model decide anything. We made it unable to pretend that unresolved contradictions never happened.

That's it. Everything else follows from that.

---

## Contact

If you have questions not covered here, you probably either:
1. Skipped the definitions section (go read it)
2. Want to argue philosophy (not interested)
3. Found an actual gap (open an issue)

For (3), we're listening. For (1) and (2), the documents already say what we mean.
