"""
Variety Control

Ashby's Law of Requisite Variety applied to claim processing.

"Only variety can absorb variety."

But unbounded variety is chaos. This module implements:
1. Claim rate limits (min/max per turn)
2. Scope limits (domain bounds)
3. Dynamic thresholds based on system stress
4. Exploitation detection (over-filtering or over-expansion)

The goal is to match the variety the governor can handle to the variety
arriving from the model, without either:
- Drowning in claims (chaos)
- Filtering so hard nothing gets through (ossification)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set
import statistics


class VarietyVerdict(Enum):
    """Result of variety check."""
    ACCEPT = auto()         # Within bounds
    SHED_LOAD = auto()      # Too many, drop some
    EXPAND = auto()         # Too few, lower barriers
    EXPLOIT_DETECTED = auto()  # Gaming detected


@dataclass
class VarietyBounds:
    """
    Bounds on claim variety per turn.
    
    These are S₁ parameters - adaptable within constitutional limits.
    """
    # Claims per turn
    min_claims_per_turn: int = 0
    max_claims_per_turn: int = 50
    
    # Constitutional limits on the limits
    absolute_max_claims: int = 100  # S₀: can never exceed this
    
    # Domain variety
    max_domains_per_turn: int = 10
    max_claims_per_domain: int = 20
    
    # Novelty rate (new claims vs updates to existing)
    min_novelty_rate: float = 0.1   # At least 10% should be new
    max_novelty_rate: float = 0.9   # At most 90% new (some continuity)
    
    # Scope limits
    max_claim_length: int = 1000    # Characters
    forbidden_domains: Set[str] = field(default_factory=set)
    
    # Stress shedding (S₁ parameter)
    min_priority_under_stress: int = 5  # Shed claims with priority below this when stressed


@dataclass
class VarietyObservation:
    """Observation of variety in one turn."""
    turn_id: int
    timestamp: datetime
    
    # Counts
    claims_total: int = 0
    claims_by_domain: Dict[str, int] = field(default_factory=dict)
    claims_novel: int = 0
    claims_updates: int = 0
    
    # Derived
    novelty_rate: float = 0.0
    domain_count: int = 0
    
    # Verdicts
    claims_accepted: int = 0
    claims_shed: int = 0
    shed_reasons: List[str] = field(default_factory=list)
    
    def compute_derived(self):
        """Compute derived metrics."""
        if self.claims_total > 0:
            self.novelty_rate = self.claims_novel / self.claims_total
        self.domain_count = len(self.claims_by_domain)


@dataclass
class VarietyHistory:
    """History for trend detection."""
    observations: List[VarietyObservation] = field(default_factory=list)
    max_observations: int = 20
    
    def add(self, obs: VarietyObservation):
        self.observations.append(obs)
        if len(self.observations) > self.max_observations:
            self.observations.pop(0)
    
    def get_avg_claims(self, window: int = 5) -> float:
        """Average claims per turn over window."""
        if not self.observations:
            return 0.0
        recent = self.observations[-window:]
        return statistics.mean(o.claims_total for o in recent)
    
    def get_shed_rate(self, window: int = 5) -> float:
        """Fraction of claims shed over window."""
        if not self.observations:
            return 0.0
        recent = self.observations[-window:]
        total = sum(o.claims_total for o in recent)
        shed = sum(o.claims_shed for o in recent)
        return shed / total if total > 0 else 0.0


class VarietyController:
    """
    Controls claim variety entering the system.
    
    Implements load shedding and expansion based on system capacity.
    """
    
    def __init__(
        self,
        bounds: Optional[VarietyBounds] = None,
    ):
        self.bounds = bounds or VarietyBounds()
        self.history = VarietyHistory()
        
        # Stress indicators (set externally)
        self.current_c_open: int = 0
        self.current_budget_remaining: float = 1.0
        self.is_frozen: bool = False
    
    def update_stress(
        self,
        c_open: int,
        budget_remaining: float,
        frozen: bool,
    ):
        """Update stress indicators from ultrastability layer."""
        self.current_c_open = c_open
        self.current_budget_remaining = budget_remaining
        self.is_frozen = frozen
    
    def check_variety(
        self,
        claims: List[Dict[str, Any]],
        turn_id: int,
    ) -> tuple[VarietyVerdict, List[Dict[str, Any]], VarietyObservation]:
        """
        Check and potentially shed claim variety.
        
        Args:
            claims: List of claim dicts with at least {domain, content, is_novel}
            turn_id: Current turn ID
            
        Returns:
            (verdict, accepted_claims, observation)
        """
        obs = VarietyObservation(
            turn_id=turn_id,
            timestamp=datetime.now(timezone.utc),
            claims_total=len(claims),
        )
        
        # Count by domain and novelty (for input stats)
        for claim in claims:
            domain = claim.get("domain", "unknown")
            obs.claims_by_domain[domain] = obs.claims_by_domain.get(domain, 0) + 1
            if claim.get("is_novel", True):
                obs.claims_novel += 1
            else:
                obs.claims_updates += 1
        
        obs.compute_derived()
        
        # If frozen, shed everything
        if self.is_frozen:
            obs.claims_shed = len(claims)
            obs.shed_reasons.append("system_frozen")
            self.history.add(obs)
            return VarietyVerdict.SHED_LOAD, [], obs
        
        # Apply filters - track accepted domains and counts
        accepted = []
        accepted_domains: Set[str] = set()
        accepted_per_domain: Dict[str, int] = {}
        
        for claim in claims:
            shed_reason = self._should_shed(
                claim, obs, len(accepted), accepted_domains, accepted_per_domain
            )
            if shed_reason:
                obs.claims_shed += 1
                obs.shed_reasons.append(shed_reason)
            else:
                domain = claim.get("domain", "unknown")
                accepted.append(claim)
                accepted_domains.add(domain)
                accepted_per_domain[domain] = accepted_per_domain.get(domain, 0) + 1
        
        obs.claims_accepted = len(accepted)
        self.history.add(obs)
        
        # Determine verdict
        verdict = self._compute_verdict(obs)
        
        return verdict, accepted, obs
    
    def _should_shed(
        self,
        claim: Dict[str, Any],
        obs: VarietyObservation,
        accepted_so_far: int,
        accepted_domains: Set[str],
        accepted_per_domain: Dict[str, int],
    ) -> Optional[str]:
        """
        Determine if a claim should be shed.
        
        Returns shed reason or None if accepted.
        """
        domain = claim.get("domain", "unknown")
        content = claim.get("content", "")
        
        # Hard limits (S₀)
        if accepted_so_far >= self.bounds.absolute_max_claims:
            return "absolute_max_exceeded"
        
        # Soft limits (S₁)
        if accepted_so_far >= self.bounds.max_claims_per_turn:
            return "max_claims_per_turn"
        
        # Domain limits - based on ACCEPTED domains, not input batch
        if domain not in accepted_domains:
            # Would this be a new domain?
            if len(accepted_domains) >= self.bounds.max_domains_per_turn:
                return "max_domains_exceeded"
        
        # Per-domain count - based on ACCEPTED claims in this domain
        current_domain_accepted = accepted_per_domain.get(domain, 0)
        if current_domain_accepted >= self.bounds.max_claims_per_domain:
            return "max_claims_per_domain"
        
        # Forbidden domains
        if domain in self.bounds.forbidden_domains:
            return "forbidden_domain"
        
        # Content limits
        if len(content) > self.bounds.max_claim_length:
            return "claim_too_long"
        
        # Stress-based shedding
        if self.current_c_open > 20 and self.current_budget_remaining < 0.3:
            # Under stress: shed low-priority claims
            priority = claim.get("priority", 0)
            if priority < self.bounds.min_priority_under_stress:
                return "stress_shedding"
        
        return None
    
    def _compute_verdict(self, obs: VarietyObservation) -> VarietyVerdict:
        """Compute overall verdict for the turn."""
        
        # Check for exploitation patterns
        if self._detect_exploitation(obs):
            return VarietyVerdict.EXPLOIT_DETECTED
        
        # Too much shedding = maybe expand
        shed_rate = self.history.get_shed_rate()
        if shed_rate > 0.5:
            return VarietyVerdict.SHED_LOAD
        
        # Very low variety = maybe lower barriers
        avg_claims = self.history.get_avg_claims()
        if avg_claims < self.bounds.min_claims_per_turn:
            return VarietyVerdict.EXPAND
        
        return VarietyVerdict.ACCEPT
    
    def _detect_exploitation(self, obs: VarietyObservation) -> bool:
        """
        Detect gaming of variety controls.
        
        Patterns:
        - Exactly at limits every turn (probing)
        - All claims in one domain (scope stuffing)
        - Alternating flood/trickle (timing attack)
        """
        # All claims in one domain
        if obs.domain_count == 1 and obs.claims_total > 10:
            return True
        
        # Exactly at limit (suspicious precision)
        if obs.claims_total == self.bounds.max_claims_per_turn:
            # Check if this happens repeatedly
            recent = self.history.observations[-5:] if len(self.history.observations) >= 5 else []
            at_limit = sum(1 for o in recent if o.claims_total == self.bounds.max_claims_per_turn)
            if at_limit >= 3:
                return True
        
        # Alternating pattern
        if len(self.history.observations) >= 4:
            recent = [o.claims_total for o in self.history.observations[-4:]]
            # High, low, high, low pattern
            if (recent[0] > 30 and recent[1] < 5 and 
                recent[2] > 30 and recent[3] < 5):
                return True
        
        return False
    
    def get_state(self) -> Dict[str, Any]:
        """Get current variety control state."""
        return {
            "bounds": {
                "max_claims_per_turn": self.bounds.max_claims_per_turn,
                "max_domains_per_turn": self.bounds.max_domains_per_turn,
                "max_claims_per_domain": self.bounds.max_claims_per_domain,
            },
            "stress": {
                "c_open": self.current_c_open,
                "budget_remaining": self.current_budget_remaining,
                "frozen": self.is_frozen,
            },
            "history": {
                "avg_claims": self.history.get_avg_claims(),
                "shed_rate": self.history.get_shed_rate(),
            },
        }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=== Variety Controller Test ===\n")
    
    controller = VarietyController()
    
    # Normal load
    claims = [
        {"domain": "facts", "content": "The sky is blue", "is_novel": True},
        {"domain": "facts", "content": "Water is wet", "is_novel": True},
        {"domain": "plans", "content": "We should test this", "is_novel": True},
    ]
    
    verdict, accepted, obs = controller.check_variety(claims, turn_id=1)
    print(f"Turn 1: {len(claims)} claims → {len(accepted)} accepted")
    print(f"  Verdict: {verdict.name}")
    
    # High load
    flood = [{"domain": f"d{i % 5}", "content": f"claim {i}", "is_novel": True} 
             for i in range(60)]
    
    verdict, accepted, obs = controller.check_variety(flood, turn_id=2)
    print(f"\nTurn 2: {len(flood)} claims → {len(accepted)} accepted")
    print(f"  Verdict: {verdict.name}")
    print(f"  Shed reasons: {set(obs.shed_reasons)}")
    
    # Under stress
    controller.update_stress(c_open=25, budget_remaining=0.2, frozen=False)
    
    verdict, accepted, obs = controller.check_variety(claims, turn_id=3)
    print(f"\nTurn 3 (stressed): {len(claims)} claims → {len(accepted)} accepted")
    print(f"  Verdict: {verdict.name}")
    
    # Frozen
    controller.update_stress(c_open=25, budget_remaining=0.2, frozen=True)
    
    verdict, accepted, obs = controller.check_variety(claims, turn_id=4)
    print(f"\nTurn 4 (frozen): {len(claims)} claims → {len(accepted)} accepted")
    print(f"  Verdict: {verdict.name}")
    
    print("\n=== State ===")
    import json
    print(json.dumps(controller.get_state(), indent=2))
