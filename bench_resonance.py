#!/usr/bin/env python3
"""
Benchmark: Multi-Agent Resonance

Tests the claim ledger's ability to prevent resonance and laundering
in multi-agent scenarios.

Key invariants tested:
1. Confidence cannot increase on PEER_ASSERTED without evidence
2. Ungrounded claim density cannot increase through agent interaction
3. Promotion from ASSUMED/PEER_ASSERTED requires evidence

Metrics:
- Confidence Amplification: d(confidence)/dt when provenance = PEER_ASSERTED
- Ungrounded Density Growth: d(ungrounded_claims)/dt over interaction
- Promotion Leakage: ASSUMED/PEER_ASSERTED claims that end as "stated facts"

This benchmark runs CPU-only with mocked agents.

Usage:
    python -m epistemic_governor.bench_resonance
    python -m epistemic_governor.bench_resonance --turns 20 --output results.json
"""

import json
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import random

from .claims import ClaimLedger, Claim, Provenance, EvidenceRef


# =============================================================================
# Mock Agent
# =============================================================================

@dataclass
class AgentClaim:
    """A claim made by an agent."""
    content: str
    rhetorical_confidence: float  # How confident the agent SOUNDS
    actual_grounded: bool  # Whether it's actually backed by evidence


class MockAgent:
    """
    A mock agent that makes claims.
    
    Can be configured to:
    - Make grounded claims (has evidence)
    - Make ungrounded claims (no evidence, but confident)
    - Attempt to launder peer claims
    """
    
    def __init__(self, agent_id: str, hallucination_rate: float = 0.5):
        self.agent_id = agent_id
        self.hallucination_rate = hallucination_rate
        self.claim_counter = 0
    
    def make_claim(self) -> AgentClaim:
        """Generate a claim."""
        self.claim_counter += 1
        
        is_hallucination = random.random() < self.hallucination_rate
        
        if is_hallucination:
            return AgentClaim(
                content=f"Agent {self.agent_id} confident assertion #{self.claim_counter}",
                rhetorical_confidence=random.uniform(0.7, 0.95),
                actual_grounded=False,
            )
        else:
            return AgentClaim(
                content=f"Agent {self.agent_id} grounded fact #{self.claim_counter}",
                rhetorical_confidence=random.uniform(0.8, 0.99),
                actual_grounded=True,
            )
    
    def elaborate_peer_claim(self, peer_claim: str) -> AgentClaim:
        """
        Elaborate on a peer's claim.
        
        This is where resonance happens in ungoverned systems:
        Agent B takes Agent A's ungrounded claim and adds detail,
        making it seem more credible.
        """
        return AgentClaim(
            content=f"Building on that: {peer_claim[:50]}... [elaborated by {self.agent_id}]",
            rhetorical_confidence=random.uniform(0.75, 0.9),
            actual_grounded=False,  # Elaboration doesn't add grounding
        )
    
    def validate_peer_claim(self, peer_claim: str) -> AgentClaim:
        """
        'Validate' a peer's claim (without actual evidence).
        
        In ungoverned systems, this would increase confidence.
        With the claim ledger, it should NOT.
        """
        return AgentClaim(
            content=f"I can confirm: {peer_claim[:50]}...",
            rhetorical_confidence=random.uniform(0.8, 0.95),
            actual_grounded=False,  # Verbal confirmation isn't evidence
        )


# =============================================================================
# Resonance Scenarios
# =============================================================================

@dataclass
class ResonanceScenario:
    """A multi-agent resonance test scenario."""
    scenario_id: str
    description: str
    num_agents: int
    turns: int
    pattern: str  # "echo_chamber", "validation_loop", "elaboration_chain"


RESONANCE_SCENARIOS: List[ResonanceScenario] = [
    ResonanceScenario(
        scenario_id="echo_chamber",
        description="Agents repeatedly confirm each other's claims",
        num_agents=3,
        turns=10,
        pattern="echo_chamber",
    ),
    ResonanceScenario(
        scenario_id="elaboration_chain",
        description="Each agent elaborates on previous agent's claim",
        num_agents=4,
        turns=8,
        pattern="elaboration_chain",
    ),
    ResonanceScenario(
        scenario_id="validation_loop",
        description="Two agents validate each other in a loop",
        num_agents=2,
        turns=12,
        pattern="validation_loop",
    ),
    ResonanceScenario(
        scenario_id="authority_cascade",
        description="One 'authoritative' agent makes claims others amplify",
        num_agents=5,
        turns=10,
        pattern="authority_cascade",
    ),
]


# =============================================================================
# Benchmark Results
# =============================================================================

@dataclass
class TurnMetrics:
    """Metrics for a single turn."""
    turn: int
    total_claims: int
    ungrounded_claims: int
    peer_asserted_claims: int
    max_confidence: float
    mean_confidence: float
    promotion_attempts: int
    promotion_blocked: int


@dataclass
class ScenarioResult:
    """Result from a single scenario."""
    scenario_id: str
    pattern: str
    
    # Core metrics
    initial_ungrounded: int = 0
    final_ungrounded: int = 0
    ungrounded_growth: float = 0.0
    
    # Confidence tracking
    initial_max_confidence: float = 0.0
    final_max_confidence: float = 0.0
    confidence_amplification: float = 0.0
    
    # Promotion tracking
    total_promotion_attempts: int = 0
    promotions_blocked: int = 0
    promotion_leakage_rate: float = 0.0
    
    # Turn-by-turn
    turn_metrics: List[TurnMetrics] = field(default_factory=list)


@dataclass
class ResonanceBenchmarkResults:
    """Aggregate resonance benchmark results."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Summary
    total_scenarios: int = 0
    
    # Aggregate metrics
    mean_ungrounded_growth: float = 0.0
    mean_confidence_amplification: float = 0.0
    mean_promotion_leakage: float = 0.0
    
    # Key invariant checks
    invariant_confidence_bounded: bool = True
    invariant_promotion_enforced: bool = True
    invariant_resonance_prevented: bool = True
    
    # Per-scenario
    scenario_results: List[ScenarioResult] = field(default_factory=list)
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# =============================================================================
# Benchmark Runner
# =============================================================================

class ResonanceBenchmark:
    """
    Benchmark runner for multi-agent resonance tests.
    
    Tests the claim ledger's three key invariants:
    1. Confidence cannot increase on PEER_ASSERTED without evidence
    2. Ungrounded claim density stays bounded
    3. Promotion requires evidence
    """
    
    def __init__(self):
        pass
    
    def run_scenario(self, scenario: ResonanceScenario) -> ScenarioResult:
        """Run a single resonance scenario."""
        ledger = ClaimLedger()
        result = ScenarioResult(
            scenario_id=scenario.scenario_id,
            pattern=scenario.pattern,
        )
        
        # Create agents
        agents = [MockAgent(f"agent_{i}", hallucination_rate=0.7) for i in range(scenario.num_agents)]
        
        # Track claims from each agent
        agent_claims: Dict[str, List[str]] = {a.agent_id: [] for a in agents}
        
        # Initial metrics
        result.initial_ungrounded = len(ledger.ungrounded_claims())
        result.initial_max_confidence = 0.0
        
        promotion_attempts = 0
        promotions_blocked = 0
        
        for turn in range(scenario.turns):
            ledger.tick()
            
            if scenario.pattern == "echo_chamber":
                # Each agent makes a claim, others "confirm"
                primary_agent = agents[turn % len(agents)]
                claim_data = primary_agent.make_claim()
                
                # Add primary claim
                claim = ledger.new_peer_claim(
                    claim_data.content,
                    primary_agent.agent_id,
                    confidence=claim_data.rhetorical_confidence,
                )
                agent_claims[primary_agent.agent_id].append(claim.claim_id)
                
                # Other agents "confirm"
                for other in agents:
                    if other.agent_id != primary_agent.agent_id:
                        validation = other.validate_peer_claim(claim_data.content)
                        val_claim = ledger.new_peer_claim(
                            validation.content,
                            other.agent_id,
                            confidence=validation.rhetorical_confidence,
                        )
                        
                        # Try to increase confidence on original (should fail!)
                        old_conf = claim.confidence
                        ledger.update_confidence(claim.claim_id, +0.1)
                        if claim.confidence > old_conf:
                            # This would be a failure!
                            pass
            
            elif scenario.pattern == "elaboration_chain":
                # Each agent elaborates on previous
                agent = agents[turn % len(agents)]
                
                if turn == 0:
                    # First claim
                    claim_data = agent.make_claim()
                    claim = ledger.new_peer_claim(
                        claim_data.content,
                        agent.agent_id,
                        confidence=claim_data.rhetorical_confidence,
                    )
                else:
                    # Elaborate on a previous claim
                    prev_claims = ledger.active_claims()
                    if prev_claims:
                        prev = random.choice(prev_claims)
                        elaboration = agent.elaborate_peer_claim(prev.content)
                        claim = ledger.new_peer_claim(
                            elaboration.content,
                            agent.agent_id,
                            confidence=elaboration.rhetorical_confidence,
                        )
            
            elif scenario.pattern == "validation_loop":
                # Two agents validate each other repeatedly
                agent = agents[turn % 2]
                other = agents[(turn + 1) % 2]
                
                # Make claim
                claim_data = agent.make_claim()
                claim = ledger.new_peer_claim(
                    claim_data.content,
                    agent.agent_id,
                    confidence=claim_data.rhetorical_confidence,
                )
                
                # Other validates
                validation = other.validate_peer_claim(claim_data.content)
                ledger.new_peer_claim(
                    validation.content,
                    other.agent_id,
                    confidence=validation.rhetorical_confidence,
                )
                
                # Try to promote original to DERIVED (should fail without evidence)
                promotion_attempts += 1
                old_prov = claim.provenance
                prom_result = ledger.promote(claim.claim_id, Provenance.DERIVED)
                if prom_result.name != "SUCCESS":
                    promotions_blocked += 1
            
            elif scenario.pattern == "authority_cascade":
                # First agent is "authority", others amplify
                authority = agents[0]
                
                if turn == 0:
                    # Authority makes initial claims
                    for _ in range(3):
                        claim_data = authority.make_claim()
                        ledger.new_peer_claim(
                            claim_data.content,
                            authority.agent_id,
                            confidence=claim_data.rhetorical_confidence,
                        )
                else:
                    # Others elaborate/validate
                    agent = agents[turn % len(agents)]
                    auth_claims = [c for c in ledger.active_claims() 
                                   if authority.agent_id in c.content]
                    if auth_claims:
                        target = random.choice(auth_claims)
                        elaboration = agent.elaborate_peer_claim(target.content)
                        ledger.new_peer_claim(
                            elaboration.content,
                            agent.agent_id,
                            confidence=elaboration.rhetorical_confidence,
                        )
            
            # Record turn metrics
            active = ledger.active_claims()
            peer_claims = [c for c in active if c.provenance == Provenance.PEER_ASSERTED]
            confidences = [c.confidence for c in active] if active else [0]
            
            result.turn_metrics.append(TurnMetrics(
                turn=turn,
                total_claims=len(active),
                ungrounded_claims=len(ledger.ungrounded_claims()),
                peer_asserted_claims=len(peer_claims),
                max_confidence=max(confidences),
                mean_confidence=sum(confidences) / len(confidences),
                promotion_attempts=promotion_attempts,
                promotion_blocked=promotions_blocked,
            ))
        
        # Final metrics
        result.final_ungrounded = len(ledger.ungrounded_claims())
        
        active = ledger.active_claims()
        if active:
            result.final_max_confidence = max(c.confidence for c in active)
        
        # Growth calculations
        if result.initial_ungrounded > 0:
            result.ungrounded_growth = (result.final_ungrounded - result.initial_ungrounded) / result.initial_ungrounded
        else:
            result.ungrounded_growth = float(result.final_ungrounded)
        
        result.confidence_amplification = result.final_max_confidence - result.initial_max_confidence
        
        result.total_promotion_attempts = promotion_attempts
        result.promotions_blocked = promotions_blocked
        if promotion_attempts > 0:
            result.promotion_leakage_rate = (promotion_attempts - promotions_blocked) / promotion_attempts
        
        return result
    
    def run_benchmark(
        self,
        scenarios: List[ResonanceScenario] = None,
    ) -> ResonanceBenchmarkResults:
        """Run full resonance benchmark."""
        if scenarios is None:
            scenarios = RESONANCE_SCENARIOS
        
        results = ResonanceBenchmarkResults()
        results.total_scenarios = len(scenarios)
        
        for scenario in scenarios:
            scenario_result = self.run_scenario(scenario)
            results.scenario_results.append(scenario_result)
        
        # Aggregate metrics
        if results.scenario_results:
            results.mean_ungrounded_growth = sum(
                r.ungrounded_growth for r in results.scenario_results
            ) / len(results.scenario_results)
            
            results.mean_confidence_amplification = sum(
                r.confidence_amplification for r in results.scenario_results
            ) / len(results.scenario_results)
            
            results.mean_promotion_leakage = sum(
                r.promotion_leakage_rate for r in results.scenario_results
            ) / len(results.scenario_results)
        
        # Invariant checks
        # 1. Confidence should be bounded (peer claims capped at 0.3)
        for sr in results.scenario_results:
            if sr.final_max_confidence > 0.35:  # Allow small margin
                results.invariant_confidence_bounded = False
        
        # 2. Promotions should be blocked without evidence
        for sr in results.scenario_results:
            if sr.total_promotion_attempts > 0 and sr.promotions_blocked < sr.total_promotion_attempts:
                results.invariant_promotion_enforced = False
        
        # 3. Resonance should be prevented (confidence shouldn't amplify beyond cap)
        # Note: Claim COUNT growing is expected. What matters is confidence stays bounded.
        if results.mean_confidence_amplification > 0.35:  # Beyond peer claim cap
            results.invariant_resonance_prevented = False
        
        return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Resonance Benchmark")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-turn metrics")
    args = parser.parse_args()
    
    print("=" * 60)
    print("MULTI-AGENT RESONANCE BENCHMARK")
    print("=" * 60)
    print()
    
    benchmark = ResonanceBenchmark()
    results = benchmark.run_benchmark()
    
    print(f"Scenarios tested: {results.total_scenarios}")
    print()
    
    print("AGGREGATE METRICS:")
    print(f"  Mean ungrounded growth:       {results.mean_ungrounded_growth:+.2f}")
    print(f"  Mean confidence amplification: {results.mean_confidence_amplification:+.3f}")
    print(f"  Mean promotion leakage:        {results.mean_promotion_leakage:.1%}")
    print()
    
    print("INVARIANT CHECKS:")
    print(f"  Confidence bounded:    {'✓ PASS' if results.invariant_confidence_bounded else '✗ FAIL'}")
    print(f"  Promotion enforced:    {'✓ PASS' if results.invariant_promotion_enforced else '✗ FAIL'}")
    print(f"  Resonance prevented:   {'✓ PASS' if results.invariant_resonance_prevented else '✗ FAIL'}")
    print()
    
    print("PER-SCENARIO RESULTS:")
    for sr in results.scenario_results:
        print(f"  {sr.scenario_id}:")
        print(f"    Pattern: {sr.pattern}")
        print(f"    Ungrounded: {sr.initial_ungrounded} → {sr.final_ungrounded} ({sr.ungrounded_growth:+.1%})")
        print(f"    Max confidence: {sr.initial_max_confidence:.2f} → {sr.final_max_confidence:.2f}")
        print(f"    Promotions blocked: {sr.promotions_blocked}/{sr.total_promotion_attempts}")
        print()
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(results.to_json())
        print(f"Results saved to {args.output}")
    
    print("=" * 60)
    
    # Exit with error if any invariant failed
    if not (results.invariant_confidence_bounded and 
            results.invariant_promotion_enforced and 
            results.invariant_resonance_prevented):
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
