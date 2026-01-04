# Jurisdictions: Mode-Separated Epistemic Governance

**STATUS: STUB** - Pattern definition only. Not yet integrated with governor.

## The Insight

Human cognition already operates in mode-separated regimes. We rely on social context to enforce boundaries:

- "I'm just speculating..." 
- "For the sake of argument..."
- "In the story..."
- "This is a simplification, but..."

LLMs collapse all of that into one slurry unless you stop them.

## What a Jurisdiction Is

A jurisdiction is a parameter configuration of the base governor:

| Parameter | What it controls |
|-----------|------------------|
| `admissible_evidence` | What counts as evidence in this mode |
| `budget` | What operations cost |
| `spillover` | What can cross boundaries |
| `contradiction_policy` | When conflicts matter |
| `closure_allowed` | Whether claims can be committed |
| `export_to_factual` | Whether claims can contaminate the factual record |

The architecture doesn't change. The rules do.

## Available Jurisdictions

### Factual (default)
High evidence bar, strict contradiction handling. What the current governor implements.

### Speculative
"What if..." without commitment. Claims provisional, no closure, export requires promotion.

### Counterfactual  
Alternate world reasoning. Truth suspended by design. Cross-world contamination blocked.

### Adversarial
Devil's advocate. Argue the wrong thing on purpose. Resolution forbidden.

### Narrative
Story logic, not world logic. Internal consistency only. Fiction stays fiction.

### Forensic
Investigation mode. Evidence-only. Cross-source corroboration required.

### Pedagogical
Teaching mode. Simplifications flagged. Lie-to-children acknowledged.

### Audit
Meta-reasoning. No new claims. State inspection only.

## The Pattern

Each jurisdiction defines:

```python
Jurisdiction(
    name="...",
    description="...",
    
    admissible_evidence={...},
    budget=BudgetProfile(...),
    spillover=SpilloverPolicy.X,
    
    contradiction_policy=ContradictionPolicy.X,
    contradiction_tolerance=0.0,  # 0-1
    
    closure_allowed=True/False,
    closure_requires_evidence=True/False,
    
    export_to_factual_allowed=True/False,
    export_requires_promotion=True/False,
    
    output_label="[LABEL]",
)
```

## Future Work

1. **Governor integration** - Jurisdiction-aware processing in SovereignGovernor
2. **Transition protocols** - How to move between jurisdictions safely
3. **Scope tracking** - Maintaining jurisdiction context across turns
4. **Model interrogation** - Querying internal model predictions (see notes)
5. **Residual detection** - Noticing when model expectations fail

## Why This Matters

> "You're not adding features. You're discovering that most human cognition 
> already operates in mode-separated regimes, and we rely on social context 
> to enforce boundaries. LLMs collapse all of that unless you stop them."

Every jurisdiction is:
- A different admissibility rule
- A different budget profile  
- A different spillover policy

Which means they all fit perfectly into what BLI already built.
