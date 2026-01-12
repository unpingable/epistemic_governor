# Integration & Deployment Modes

**How to wire the governor into agent stacks without losing the guarantees.**

---

## The Core Distinction

| Layer | Responsibility | Examples |
|-------|----------------|----------|
| **Transport/Orchestration** | Move bytes, discover tools, manage context | MCP, LangChain, AutoGPT |
| **Policy/Governance** | Decide what's allowed to mutate state | This governor |

**MCP moves bytes. The governor decides whether bytes are allowed to mutate state.**

If you treat the governor as "just another callback," you have a suggestion box, not a governor.

---

## Deployment Modes

### 1. Advisory Mode (Weak)

The governor scores or flags proposed actions. The agent can ignore it.

```
Agent → propose_action() → Governor → {allow, deny, warn} → Agent decides
```

**Guarantees**: None. You get logs.

**Use when**: You want visibility without commitment. Dashboards, not safety.

### 2. Gateway Mode (Strong)

All tool calls route through the governor. Denial is a hard stop.

```
Agent → Governor.authorize(tool, args) → [if allowed] → Tool → Result
                                       → [if denied]  → Error + Log
```

**Guarantees**: 
- No tool executes without authorization
- All decisions logged
- Fail-closed on governor failure

**Use when**: You need actual enforcement. This is the intended mode.

### 3. Commit Gate Mode (Strongest)

The governor controls the "commit" boundary—nothing persists without explicit commit.

```
Agent → Tool → Result (ephemeral)
Agent → Governor.commit(state_change) → [if allowed] → Persist
                                      → [if denied]  → Rollback + Log
```

**Guarantees**:
- Separation of "read" from "write"
- Atomic commit semantics
- Full audit trail with causal links

**Use when**: You need transactional integrity over agent state.

---

## MCP Integration

### As an MCP Server (Recommended)

Expose the governor as an MCP server that mediates access to other tools.

```
Model ←→ Governor (MCP Server) ←→ Tool Servers
```

**Exposed tools**:
- `authorize_call(tool, args)` → `{allowed: bool, reason: str}`
- `commit(state_delta)` → `{committed: bool, witness_id: str}`
- `get_state()` → current ledger/contradiction state
- `explain_denial(witness_id)` → why something was denied

**Key behavior**:
- Tool results are treated as **evidence**, not just context
- State updates require explicit **commit**
- Denials are **logged with causal chain**

### As a Validator Tool (Weaker)

Single MCP tool that the agent must call before acting.

```python
# Agent calls this before any irreversible action
result = governor.evaluate(
    proposed_action="delete_user",
    context=current_state,
    evidence=tool_results
)

if not result.allowed:
    raise ActionDenied(result.reason)
```

**Limitation**: Only works if the agent is configured to treat denial as fatal.

---

## LangChain Integration

### Callback Handler (Advisory)

```python
from instrumentation.langchain_callback import EpistemicCallback

callback = EpistemicCallback(governor, enforcement_mode="OBSERVE")
agent.run("...", callbacks=[callback])
```

Emits telemetry but doesn't block. Good for proving value before enforcement.

### Tool Wrapper (Enforced)

```python
def governed_tool(tool_func):
    """Wrap a tool with governor authorization."""
    def wrapper(*args, **kwargs):
        # Check authorization
        decision = governor.authorize(tool_func.__name__, kwargs)
        
        if decision.verdict == "DENIED":
            raise ToolDenied(decision.reason)
        
        # Execute
        result = tool_func(*args, **kwargs)
        
        # Log result as evidence
        governor.record_evidence(tool_func.__name__, result)
        
        return result
    return wrapper
```

### Agent Executor Override (Strongest)

Replace the executor's tool dispatch to route through the governor:

```python
class GovernedAgentExecutor(AgentExecutor):
    def _call_tool(self, tool_name, tool_input):
        # All tool calls go through governor
        if not self.governor.authorize(tool_name, tool_input).allowed:
            return ToolDenied(...)
        
        result = super()._call_tool(tool_name, tool_input)
        
        # Commit if state-changing
        if self._is_state_changing(tool_name):
            self.governor.commit(tool_name, tool_input, result)
        
        return result
```

---

## What the Governor Provides

**You get:**
- Fail-closed policy gates on tool invocation
- Temporal windows (deny when evidence is stale)
- Monotonicity rules (forbid state regression, inconsistent claims)
- Auditability (ledgered decisions, replayable traces)
- Rate limiting (commitment velocity vs grounding velocity)
- Pathology detection (freeze when adaptation fails)

**You do NOT get:**
- "Stops jailbreaks" (no)
- "Makes outputs true" (no)
- "Solves alignment" (no)
- "Detects malicious intent" (no)

The governor enforces **legibility and constraint**, not **truth or safety**.

---

## Failure Modes

### Fail-Open (Dangerous)

If the governor crashes or is bypassed, actions proceed unchecked.

**Mitigation**: Wire denial as the default. Timeout = deny.

### Fail-Closed (Safe but Frustrating)

If the governor is slow or conservative, everything blocks.

**Mitigation**: 
- Tune thresholds via ultrastability (S₁ adaptation)
- Monitor block rate
- Provide explain/appeal path

### Frozen (Circuit Breaker)

If pathology is detected, the governor freezes all adaptation and may deny all actions.

**Mitigation**: 
- Human review required to unfreeze
- This is intentional—it's the "something is wrong, stop" signal

---

## The Boundary Contract

```
┌─────────────────────────────────────────────────────────┐
│                    S₀ (Constitutional)                   │
│  • NLAI invariant                                       │
│  • FSM transitions                                      │
│  • Bounds on S₁ parameters                              │
│  • Evidence type definitions                            │
│  IMMUTABLE AT RUNTIME                                   │
├─────────────────────────────────────────────────────────┤
│                    S₁ (Regulatory)                       │
│  • Budget levels                                        │
│  • Thresholds                                           │
│  • Timeouts                                             │
│  ADAPTABLE WITHIN S₀ BOUNDS                             │
├─────────────────────────────────────────────────────────┤
│                    S₂ (Epistemic)                        │
│  • Claims, contradictions                               │
│  • Ledger entries                                       │
│  • Evidence records                                     │
│  FULLY MUTABLE                                          │
└─────────────────────────────────────────────────────────┘

S₂ observations may trigger S₁ adaptation.
S₁ adaptation may NOT modify S₀.
Language may NOT directly modify S₀ or S₁.
```

If your integration allows any of these rules to be violated, you don't have a governor—you have a linter that can be ignored.

---

## Quick Start: "Make It Actually Enforce"

1. **Route all tool calls through `governor.authorize()`**
2. **Treat denial as a hard error, not a warning**
3. **Log results as evidence via `governor.record_evidence()`**
4. **Use commit semantics for state-changing operations**
5. **Monitor for freeze state and alert humans**

If you do only one thing: **wire the commit path through the governor**.

Everything else is optimization.

---

*"MCP is the plug. The governor is the breaker panel."*
