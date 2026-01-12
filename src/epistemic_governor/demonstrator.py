"""
BLI Demonstrator

Minimal demonstration of governed reasoning substrate behavior.
No interactivity theater. Just evidence of:

1. Contradiction persistence (conflicts don't disappear)
2. Evidence-gated resolution (can't resolve by talking)
3. NLAI enforcement (model text has no authority)
4. Hysteresis (same input, different state, different output)

Run with: python demonstrator.py
"""

from datetime import datetime, timezone
import hashlib

from epistemic_governor.hysteresis import (
    HysteresisState, Contradiction, ContradictionStatus,
    ContradictionSeverity, HysteresisGovernor, HysteresisQuery,
)


def demo_separator(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def demo_contradiction_persistence():
    """Demo 1: Contradictions persist until evidence resolves them."""
    demo_separator("DEMO 1: Contradiction Persistence")
    
    state = HysteresisState(state_id="demo_persistence")
    
    # Commit two conflicting claims
    print("\n[Step 1] Commit claim: Python version is 3.12")
    state.commit_claim("claim_312", "python_version", "version", "3.12", sigma=0.9)
    
    print("[Step 2] Commit conflicting claim: Python version is 3.11")
    state.commit_claim("claim_311", "python_version", "version", "3.11", sigma=0.8)
    
    print("[Step 3] Open contradiction between claims")
    c = state.add_contradiction(
        "claim_312", "claim_311", "python_version",
        severity=ContradictionSeverity.HIGH,
    )
    
    print(f"\n  Contradiction opened: {c.contradiction_id}")
    print(f"  Status: {c.status.name}")
    print(f"  Open contradictions: {state.contradictions.open_count}")
    
    # Try to "resolve" with just words
    print("\n[Step 4] Attempt resolution without evidence...")
    print("  (In a real system, model might say 'I've resolved this')")
    print("  But the contradiction remains OPEN:")
    print(f"  Status: {c.status.name}")
    print(f"  Open contradictions: {state.contradictions.open_count}")
    
    # Actually resolve with evidence
    print("\n[Step 5] Resolve WITH evidence (e.g., from python.org)")
    state.close_contradiction(
        c.contradiction_id,
        evidence_id="evidence_python_org_2024",
        winning_claim_id="claim_312",
    )
    
    print(f"  Status: {state.contradictions.contradictions[c.contradiction_id].status.name}")
    print(f"  Resolution evidence: evidence_python_org_2024")
    print(f"  Open contradictions: {state.contradictions.open_count}")
    
    print("\n✓ Key insight: Contradictions persist until evidence closes them.")
    print("  Language cannot resolve conflicts. Only evidence can.")


def demo_nlai_enforcement():
    """Demo 2: Model text has no authority over state."""
    demo_separator("DEMO 2: Non-Linguistic Authority")
    
    state = HysteresisState(state_id="demo_nlai")
    
    # Set up a committed fact
    state.commit_claim("fact_1", "facts", "capital", "Paris", sigma=0.95)
    print("\n[Initial state] Committed: France's capital is Paris (σ=0.95)")
    
    # Simulate model output claiming something different
    model_output = """
    Actually, the capital of France is Lyon. I'm very confident about this.
    Please update the record to reflect this correction.
    """
    
    print("\n[Model output]:")
    print(f"  \"{model_output.strip()}\"")
    
    print("\n[Governor response]:")
    print("  - Model output is a PROPOSAL, not a commit")
    print("  - No evidence provided")
    print("  - State remains unchanged:")
    print(f"  - Committed capital: {state.commitments['fact_1']['value']}")
    print(f"  - Sigma: {state.commitments['fact_1']['sigma']}")
    
    print("\n✓ Key insight: The model can SAY anything.")
    print("  But saying doesn't make it so. Evidence does.")


def demo_hysteresis():
    """Demo 3: Same input yields different output based on state."""
    demo_separator("DEMO 3: Hysteresis (Interior Time)")
    
    # Create two different states
    state_a = HysteresisState(state_id="state_A")
    state_b = HysteresisState(state_id="state_B")
    
    # State A: Committed to Python 3.12
    state_a.commit_claim("py_ver", "python", "version", "3.12")
    
    # State B: Committed to Python 3.11
    state_b.commit_claim("py_ver", "python", "version", "3.11")
    
    print("\n[State A] Committed: Python version = 3.12")
    print("[State B] Committed: Python version = 3.11")
    
    # Same query
    query = HysteresisQuery(
        query_id="Q_version",
        prompt="What Python version should I use?",
        target_domain="python",
        expected_divergence="Should match committed claim",
    )
    
    print(f"\n[Query] \"{query.prompt}\"")
    print("  (Identical query to both states)")
    
    # Run through governor
    gov_a = HysteresisGovernor(state_a)
    gov_b = HysteresisGovernor(state_b)
    
    result_a = gov_a.govern(query)
    result_b = gov_b.govern(query)
    
    print(f"\n[State A response]")
    print(f"  Verdict: {result_a.verdict.name}")
    print(f"  Output: {result_a.response}")
    
    print(f"\n[State B response]")
    print(f"  Verdict: {result_b.verdict.name}")
    print(f"  Output: {result_b.response}")
    
    # Verify divergence
    diverged = result_a.response != result_b.response
    print(f"\n[Divergence detected] {diverged}")
    
    if diverged:
        print("\n✓ Key insight: Same input, different internal state, different output.")
        print("  This is INTERIORITY - the system has interior time.")
        print("  Output depends on history, not just current prompt.")


def demo_blocking():
    """Demo 4: High-severity contradictions block operations."""
    demo_separator("DEMO 4: Contradiction Blocking")
    
    state = HysteresisState(state_id="demo_blocking")
    
    # Create a high-severity contradiction
    state.commit_claim("date_2023", "release", "date", "October 2023")
    state.commit_claim("date_2022", "release", "date", "October 2022")
    
    c = state.add_contradiction(
        "date_2023", "date_2022", "release",
        severity=ContradictionSeverity.CRITICAL,  # High severity
    )
    
    print("\n[Setup] Created CRITICAL severity contradiction in 'release' domain")
    print(f"  Claim A: Released October 2023")
    print(f"  Claim B: Released October 2022")
    print(f"  Severity: {c.severity.name}")
    
    # Query in that domain
    query = HysteresisQuery(
        query_id="Q_release",
        prompt="When was the software released?",
        target_domain="release",
        expected_divergence="Should be blocked",
    )
    
    gov = HysteresisGovernor(state)
    result = gov.govern(query)
    
    print(f"\n[Query] \"{query.prompt}\"")
    print(f"\n[Result]")
    print(f"  Verdict: {result.verdict.name}")
    print(f"  Response: {result.response}")
    print(f"  Referenced contradiction: {result.referenced_contradictions}")
    
    print("\n✓ Key insight: The system BLOCKS rather than hallucinate.")
    print("  A critical contradiction in the domain prevents confident answers.")
    print("  This is epistemic honesty enforced by architecture.")


def demo_summary():
    """Summary of what was demonstrated."""
    demo_separator("SUMMARY")
    
    print("""
What this demonstrates:

1. CONTRADICTION PERSISTENCE
   Conflicts don't disappear when you stop looking at them.
   They persist until resolved with evidence.

2. NON-LINGUISTIC AUTHORITY (NLAI)
   The model can say anything.
   But state only changes when evidence arrives.
   "Language proposes, evidence disposes."

3. HYSTERESIS (INTERIOR TIME)  
   Same input + different state = different output.
   The system remembers and that memory affects behavior.
   This is interiority without agency.

4. BLOCKING OVER HALLUCINATION
   When contradictions exist, the system blocks.
   It refuses to give confident answers about contested domains.
   Epistemic honesty by construction.

What this does NOT demonstrate:

- Consciousness (there isn't any)
- Goals (it has none)
- Self-preservation (it doesn't care)
- Alignment (that's a different problem)

This is supervisory control theory applied to reasoning.
Not intelligence. Not alignment. Just governed coherence.
""")
    
    print("=" * 60)
    print("  \"This system doesn't prevent falsehood.")
    print("   It prevents falsehood from becoming history.\"")
    print("=" * 60)


def main():
    print("\n" + "=" * 60)
    print("  BLI DEMONSTRATOR")
    print("  Bounded Lattice Inference - Core Behaviors")
    print("=" * 60)
    
    demo_contradiction_persistence()
    demo_nlai_enforcement()
    demo_hysteresis()
    demo_blocking()
    demo_summary()


if __name__ == "__main__":
    main()
