"""
V1 → V2 Integration Test

Demonstrates the full pipeline:
1. V1 extracts claims from text
2. Bridge converts to V2 candidates
3. V2 adjudicates with support calculus
4. Contradictions detected structurally

This is the Python 3.11 scenario that exposed the V1 bug:
- Claim 1: "Python 3.11 was released in October 2022"
- Claim 2: "Python 3.11 was released in October 2021"
- V1 failed to detect contradiction (different prop_hashes)
- V2 should catch it structurally
"""

from epistemic_governor.claim_extractor import (
    ClaimAtom, ClaimMode, Modality, Quantifier
)
from epistemic_governor.claims import Provenance, EvidenceRef
from epistemic_governor.v1_v2_bridge import claim_atom_to_candidate, bridge_claims
from symbolic_substrate import (
    Adjudicator, SymbolicState, AdjudicationDecision,
    SupportItem, ProvenanceClass,
)


def test_contradiction_detection():
    """
    Test that V2 catches contradictions that V1 missed.
    
    The bug: V1's prop_hash was too surface-form dependent.
    "released in 2022" and "released in 2021" got different hashes
    because the normalizer didn't collapse them.
    
    V2 should catch this because:
    - Same predicate type (AT_TIME)
    - Same entity (Python 3.11)
    - Different value (2022 vs 2021)
    - Overlapping temporal scope
    """
    print("=== V1 → V2 Contradiction Detection Test ===\n")
    
    # Simulate what V1 extraction produces
    claim_2022 = ClaimAtom(
        prop_hash="hash_2022",  # V1 gave different hashes
        confidence=0.85,
        polarity=1,
        modality=Modality.ASSERT,
        quantifier=Quantifier.NONE,
        tense="past",
        span=(0, 50),
        span_quote="Python 3.11 was released in October 2022",
        entities=("Python 3.11",),
        predicate="released",
        value_norm="October 2022",
        value_features={"year": 2022, "month": "October"},
    )
    
    claim_2021 = ClaimAtom(
        prop_hash="hash_2021",  # Different hash - V1's bug
        confidence=0.80,
        polarity=1,
        modality=Modality.ASSERT,
        quantifier=Quantifier.NONE,
        tense="past",
        span=(100, 150),
        span_quote="Actually it came out in October 2021",
        entities=("Python 3.11",),
        predicate="released",
        value_norm="October 2021",
        value_features={"year": 2021, "month": "October"},
    )
    
    # V1 sees these as different propositions!
    print(f"V1 prop_hashes:")
    print(f"  Claim 2022: {claim_2022.prop_hash}")
    print(f"  Claim 2021: {claim_2021.prop_hash}")
    print(f"  V1 thinks these are different propositions: {claim_2022.prop_hash != claim_2021.prop_hash}")
    
    # Bridge to V2
    candidate_2022 = claim_atom_to_candidate(claim_2022, Provenance.ASSUMED)
    candidate_2021 = claim_atom_to_candidate(claim_2021, Provenance.ASSUMED)
    
    print(f"\nV2 predicates:")
    print(f"  Candidate 2022: {candidate_2022.predicate.ptype.name}{candidate_2022.predicate.args}")
    print(f"  Candidate 2021: {candidate_2021.predicate.ptype.name}{candidate_2021.predicate.args}")
    
    # V2 adjudication with relaxed thresholds for testing
    state = SymbolicState()
    state.register_entity("Python 3.11", {"type": "software"})
    
    # Use relaxed config to let claims through for contradiction testing
    adjudicator = Adjudicator(config={
        "sigma_hard_gate": 0.95,  # Raise threshold
        "support_deficit_tolerance": 20.0,  # Very tolerant
    })
    
    # First claim - should pass with relaxed thresholds
    print("\n--- First claim (2022) with relaxed thresholds ---")
    result1 = adjudicator.adjudicate(state, candidate_2022)
    print(f"Decision: {result1.decision.name}")
    print(f"Reason: {result1.reason_code}")
    
    if result1.decision == AdjudicationDecision.ACCEPT:
        state.add_commitment(result1.commitment)
        print(f"Committed: {result1.commitment.commitment_id}")
        print(f"State now has {len(state.commitments)} commitment(s)")
    elif result1.decision == AdjudicationDecision.QUARANTINE_SUPPORT:
        # For testing, let's force-add it anyway
        print("Quarantined, but adding for contradiction test...")
        from symbolic_substrate import Commitment, Dependency, DependencyType
        import uuid
        commitment = Commitment(
            commitment_id=f"φ_test_{uuid.uuid4().hex[:8]}",
            predicate=candidate_2022.predicate,
            sigma=candidate_2022.sigma,
            t_scope=candidate_2022.t_scope,
            provclass=candidate_2022.provclass,
            support=[],
            logical_deps=[],
            evidentiary_deps=[],
            status="active",
            support_mass=0,
            support_deficit=result1.support_deficit,
        )
        state.add_commitment(commitment)
        print(f"Force-committed: {commitment.commitment_id}")
    
    # Second claim (contradicting) - should be rejected as contradiction
    print("\n--- Second claim (2021) - should contradict first ---")
    result2 = adjudicator.adjudicate(state, candidate_2021)
    print(f"Decision: {result2.decision.name}")
    print(f"Contradicts: {result2.contradicts}")
    print(f"Reason: {result2.reason_code}")
    
    # Show the structural difference
    print("\n=== Analysis ===")
    print("V1 missed this because prop_hash was surface-form dependent.")
    print("V2 catches it because:")
    print("  - Same predicate type (AT_TIME)")
    print("  - Same first arg (Python 3.11)")  
    print("  - Same second arg (state/released)")
    print("  - Different third arg (October 2022 vs October 2021)")
    print("  - Temporal scopes overlap")
    print("\nThis is structural contradiction detection, not hash matching.")
    
    detected = result2.decision == AdjudicationDecision.REJECT_CONTRADICTION
    if not detected:
        print(f"\nNote: Decision was {result2.decision.name}, checking if contradicts list is populated...")
        detected = len(result2.contradicts) > 0
    
    return detected


def test_support_calculus():
    """
    Test the support mass calculus.
    
    Shows that σ (confidence) must be funded by evidence.
    High σ with no support → rejection or quarantine.
    """
    print("\n=== Support Calculus Test ===\n")
    
    state = SymbolicState()
    adjudicator = Adjudicator()
    
    # Test different σ levels with no support
    test_cases = [
        (0.3, "low confidence"),
        (0.5, "medium confidence"),
        (0.7, "high confidence - at gate threshold"),
        (0.9, "very high confidence"),
    ]
    
    print("Testing σ levels with MODEL provenance (highest support requirement):\n")
    print(f"{'σ':<6} {'Required Support':<18} {'Decision':<25} {'Reason'}")
    print("-" * 80)
    
    for sigma, label in test_cases:
        atom = ClaimAtom(
            prop_hash=f"test_{sigma}",
            confidence=sigma,
            polarity=1,
            modality=Modality.ASSERT,
            quantifier=Quantifier.NONE,
            tense="present",
            span=(0, 20),
            span_quote=f"Test claim σ={sigma}",
            entities=("TestEntity",),
            predicate="has",
            value_norm="TestValue",
            value_features={},
        )
        
        candidate = claim_atom_to_candidate(atom, Provenance.ASSUMED)
        result = adjudicator.adjudicate(state, candidate)
        
        print(f"{sigma:<6.2f} {result.support_mass_required:<18.2f} {result.decision.name:<25} {result.reason_code}")
    
    print("\n" + "=" * 80)
    print("Key insight: S_min(σ) = a_p * σ² / (1 - σ + 0.02)")
    print("For MODEL provenance, a_p = 2.5 (harsh penalty)")
    print("High σ requires disproportionately more evidence.")


def test_three_clocks():
    """
    Test the three-clock temporal state.
    """
    print("\n=== Three Clocks Test ===\n")
    
    from symbolic_substrate import TemporalState
    
    theta = TemporalState()
    
    print(f"Initial state:")
    print(f"  t_fast: {theta.t_fast}")
    print(f"  t_integrative: {theta.t_integrative}")
    print(f"  t_developmental: {theta.t_developmental}")
    
    # Simulate several turns
    print("\nSimulating 10 turns...")
    for i in range(10):
        theta.tick_fast()
        # Every 5 turns with high stability, integrative might tick
        if i % 5 == 4:
            theta.maybe_tick_integrative(contradiction_resolved=False, stability=0.9)
    
    print(f"\nAfter 10 turns:")
    print(f"  t_fast: {theta.t_fast}")
    print(f"  t_integrative: {theta.t_integrative}")
    print(f"  t_developmental: {theta.t_developmental}")
    
    # Contradiction resolution should tick integrative
    print("\nContradiction resolved...")
    theta.maybe_tick_integrative(contradiction_resolved=True, stability=0.5)
    
    print(f"After resolution:")
    print(f"  t_integrative: {theta.t_integrative}")
    
    print("\nKey insight:")
    print("  - t_fast: advances every turn")
    print("  - t_integrative: advances when beliefs stabilize or contradictions resolve")
    print("  - t_developmental: only under sustained pressure or trauma")


if __name__ == "__main__":
    test_support_calculus()
    test_three_clocks()
    
    print("\n" + "=" * 80)
    passed = test_contradiction_detection()
    print("\n" + "=" * 80)
    
    if passed:
        print("\n✓ Contradiction detected structurally!")
    else:
        print("\n✗ Contradiction not detected - check adjudicator logic")
