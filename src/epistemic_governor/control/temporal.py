"""
Temporal Limits

Hard temporal constraints on claims and evidence.

"Too late = false"

This module implements:
1. Claim expiration (TTL)
2. Evidence staleness detection
3. Lag budgets (max acceptable delay)
4. Clock coherence checking

The principle: temporal incoherence is a failure mode, not a warning.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set
import time


class TemporalVerdict(Enum):
    """Result of temporal check."""
    VALID = auto()          # Within time bounds
    EXPIRED = auto()        # Past TTL
    STALE = auto()          # Evidence too old
    LAG_EXCEEDED = auto()   # Processing took too long
    CLOCK_DRIFT = auto()    # Clock inconsistency detected


@dataclass
class TemporalBounds:
    """
    Time bounds for claims and evidence.
    
    All durations in seconds.
    """
    # Claim TTL
    default_claim_ttl: float = 3600.0       # 1 hour
    speculative_claim_ttl: float = 300.0    # 5 min for speculation
    factual_claim_ttl: float = 86400.0      # 24 hours for facts
    
    # Evidence staleness
    max_evidence_age: float = 60.0          # Evidence older than 1 min is stale
    tool_result_ttl: float = 30.0           # Tool results expire fast
    
    # Lag budgets
    max_processing_lag: float = 5.0         # Max time to process a turn
    max_commit_lag: float = 1.0             # Max time between decision and commit
    
    # Clock tolerance
    max_clock_drift: float = 1.0            # Max acceptable clock skew
    
    # Hard limits (S₀)
    absolute_max_ttl: float = 604800.0      # 1 week max TTL


@dataclass
class TimestampedItem:
    """An item with temporal metadata."""
    item_id: str
    created_at: datetime
    kind: str = "claim"  # "claim" or "evidence"
    expires_at: Optional[datetime] = None
    last_verified_at: Optional[datetime] = None
    ttl_seconds: Optional[float] = None
    
    def is_expired(self, now: Optional[datetime] = None) -> bool:
        """Check if item has expired."""
        if self.expires_at is None:
            return False
        now = now or datetime.now(timezone.utc)
        return now > self.expires_at
    
    def age_seconds(self, now: Optional[datetime] = None) -> float:
        """Get age in seconds."""
        now = now or datetime.now(timezone.utc)
        delta = now - self.created_at
        return delta.total_seconds()
    
    def time_to_expiry(self, now: Optional[datetime] = None) -> Optional[float]:
        """Get seconds until expiration, or None if no expiry."""
        if self.expires_at is None:
            return None
        now = now or datetime.now(timezone.utc)
        delta = self.expires_at - now
        return delta.total_seconds()


@dataclass
class TemporalObservation:
    """Observation of temporal health for one turn."""
    turn_id: int
    timestamp: datetime
    
    # Processing times
    processing_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processing_end: Optional[datetime] = None
    processing_lag_ms: float = 0.0
    
    # Expirations this turn
    claims_expired: int = 0
    evidence_stale: int = 0
    
    # Verdicts
    lag_violations: int = 0
    clock_drift_detected: bool = False
    
    def finish_processing(self):
        """Mark processing complete and compute lag."""
        self.processing_end = datetime.now(timezone.utc)
        delta = self.processing_end - self.processing_start
        self.processing_lag_ms = delta.total_seconds() * 1000


class TemporalController:
    """
    Enforces temporal constraints.
    
    Hard fail on temporal incoherence, not just warn.
    """
    
    def __init__(
        self,
        bounds: Optional[TemporalBounds] = None,
    ):
        self.bounds = bounds or TemporalBounds()
        
        # Track items with expiration
        self.tracked_items: Dict[str, TimestampedItem] = {}
        
        # Last known times for drift detection
        self.last_system_time: Optional[datetime] = None
        self.last_monotonic: Optional[float] = None
        
        # Stats
        self.total_expirations: int = 0
        self.total_lag_violations: int = 0
    
    def track_claim(
        self,
        claim_id: str,
        domain: str = "factual",
        created_at: Optional[datetime] = None,
    ) -> TimestampedItem:
        """
        Start tracking a claim with appropriate TTL.
        
        Args:
            claim_id: Unique claim identifier
            domain: Claim domain (affects TTL)
            created_at: Creation time (defaults to now)
        """
        now = created_at or datetime.now(timezone.utc)
        
        # Select TTL based on domain
        if domain == "speculative":
            ttl = self.bounds.speculative_claim_ttl
        elif domain == "factual":
            ttl = self.bounds.factual_claim_ttl
        else:
            ttl = self.bounds.default_claim_ttl
        
        # Enforce absolute max
        ttl = min(ttl, self.bounds.absolute_max_ttl)
        
        item = TimestampedItem(
            item_id=claim_id,
            created_at=now,
            kind="claim",
            expires_at=now + timedelta(seconds=ttl),
            ttl_seconds=ttl,
        )
        
        self.tracked_items[claim_id] = item
        return item
    
    def track_evidence(
        self,
        evidence_id: str,
        evidence_type: str = "tool_result",
        created_at: Optional[datetime] = None,
    ) -> TimestampedItem:
        """Track evidence with staleness bounds."""
        now = created_at or datetime.now(timezone.utc)
        
        if evidence_type == "tool_result":
            ttl = self.bounds.tool_result_ttl
        else:
            ttl = self.bounds.max_evidence_age
        
        item = TimestampedItem(
            item_id=evidence_id,
            created_at=now,
            kind="evidence",
            expires_at=now + timedelta(seconds=ttl),
            ttl_seconds=ttl,
        )
        
        self.tracked_items[evidence_id] = item
        return item
    
    def check_item(
        self,
        item_id: str,
        now: Optional[datetime] = None,
    ) -> TemporalVerdict:
        """
        Check temporal validity of a tracked item.
        
        Returns verdict (VALID, EXPIRED, STALE).
        Note: STALE only applies to evidence, not claims.
        """
        if item_id not in self.tracked_items:
            return TemporalVerdict.VALID  # Unknown items pass
        
        item = self.tracked_items[item_id]
        now = now or datetime.now(timezone.utc)
        
        if item.is_expired(now):
            # Don't double-count - expiration counted in expire_stale()
            return TemporalVerdict.EXPIRED
        
        # Staleness only applies to evidence, not claims
        # Claims have their own TTL which is checked via is_expired()
        if item.kind == "evidence":
            age = item.age_seconds(now)
            if age > self.bounds.max_evidence_age:
                return TemporalVerdict.STALE
        
        return TemporalVerdict.VALID
    
    def check_processing_lag(
        self,
        start_time: datetime,
        end_time: Optional[datetime] = None,
    ) -> TemporalVerdict:
        """
        Check if processing lag is within bounds.
        
        Returns LAG_EXCEEDED if too slow.
        """
        end_time = end_time or datetime.now(timezone.utc)
        lag = (end_time - start_time).total_seconds()
        
        if lag > self.bounds.max_processing_lag:
            self.total_lag_violations += 1
            return TemporalVerdict.LAG_EXCEEDED
        
        return TemporalVerdict.VALID
    
    def check_clock_coherence(self) -> TemporalVerdict:
        """
        Check for clock drift or inconsistency.
        
        Compares wall clock to monotonic clock.
        """
        now_system = datetime.now(timezone.utc)
        now_monotonic = time.monotonic()
        
        if self.last_system_time is not None and self.last_monotonic is not None:
            # Expected system time based on monotonic
            monotonic_delta = now_monotonic - self.last_monotonic
            expected_system = self.last_system_time + timedelta(seconds=monotonic_delta)
            
            # Check drift
            drift = abs((now_system - expected_system).total_seconds())
            
            if drift > self.bounds.max_clock_drift:
                return TemporalVerdict.CLOCK_DRIFT
        
        # Update tracking
        self.last_system_time = now_system
        self.last_monotonic = now_monotonic
        
        return TemporalVerdict.VALID
    
    def expire_stale(self) -> List[str]:
        """
        Find and remove expired items.
        
        Returns list of expired item IDs.
        """
        now = datetime.now(timezone.utc)
        expired = []
        
        for item_id, item in list(self.tracked_items.items()):
            if item.is_expired(now):
                expired.append(item_id)
                del self.tracked_items[item_id]
                self.total_expirations += 1
        
        return expired
    
    def refresh_evidence(
        self,
        evidence_id: str,
        now: Optional[datetime] = None,
    ) -> bool:
        """
        Refresh evidence timestamp (e.g., after re-verification).
        
        Returns True if refreshed, False if not found.
        """
        if evidence_id not in self.tracked_items:
            return False
        
        now = now or datetime.now(timezone.utc)
        item = self.tracked_items[evidence_id]
        item.last_verified_at = now
        item.expires_at = now + timedelta(seconds=item.ttl_seconds or self.bounds.max_evidence_age)
        
        return True
    
    def get_state(self) -> Dict[str, Any]:
        """Get temporal controller state."""
        now = datetime.now(timezone.utc)
        
        # Count items by status
        valid = 0
        expiring_soon = 0  # Within 10% of TTL
        expired = 0
        
        for item in self.tracked_items.values():
            if item.is_expired(now):
                expired += 1
            else:
                valid += 1
                tte = item.time_to_expiry(now)
                if tte and item.ttl_seconds and tte < item.ttl_seconds * 0.1:
                    expiring_soon += 1
        
        return {
            "tracked_items": len(self.tracked_items),
            "valid": valid,
            "expiring_soon": expiring_soon,
            "expired": expired,
            "total_expirations": self.total_expirations,
            "total_lag_violations": self.total_lag_violations,
            "bounds": {
                "default_claim_ttl": self.bounds.default_claim_ttl,
                "max_evidence_age": self.bounds.max_evidence_age,
                "max_processing_lag": self.bounds.max_processing_lag,
            },
        }


# =============================================================================
# Convenience: Turn-level temporal check
# =============================================================================

@dataclass
class TemporalTurnResult:
    """Result of temporal checks for a turn."""
    verdict: TemporalVerdict
    processing_lag_ms: float
    claims_expired: List[str]
    evidence_stale: List[str]
    clock_ok: bool
    
    @property
    def is_valid(self) -> bool:
        return self.verdict == TemporalVerdict.VALID


def check_turn_temporal(
    controller: TemporalController,
    turn_start: datetime,
    claim_ids: List[str],
    evidence_ids: List[str],
) -> TemporalTurnResult:
    """
    Run all temporal checks for a turn.
    
    Returns comprehensive result.
    """
    now = datetime.now(timezone.utc)
    
    # Check processing lag
    lag_verdict = controller.check_processing_lag(turn_start, now)
    lag_ms = (now - turn_start).total_seconds() * 1000
    
    # Check claims
    expired_claims = []
    for cid in claim_ids:
        if controller.check_item(cid, now) == TemporalVerdict.EXPIRED:
            expired_claims.append(cid)
    
    # Check evidence
    stale_evidence = []
    for eid in evidence_ids:
        v = controller.check_item(eid, now)
        if v in [TemporalVerdict.EXPIRED, TemporalVerdict.STALE]:
            stale_evidence.append(eid)
    
    # Check clock
    clock_verdict = controller.check_clock_coherence()
    
    # Overall verdict
    if lag_verdict == TemporalVerdict.LAG_EXCEEDED:
        verdict = TemporalVerdict.LAG_EXCEEDED
    elif clock_verdict == TemporalVerdict.CLOCK_DRIFT:
        verdict = TemporalVerdict.CLOCK_DRIFT
    elif expired_claims:
        verdict = TemporalVerdict.EXPIRED
    elif stale_evidence:
        verdict = TemporalVerdict.STALE
    else:
        verdict = TemporalVerdict.VALID
    
    return TemporalTurnResult(
        verdict=verdict,
        processing_lag_ms=lag_ms,
        claims_expired=expired_claims,
        evidence_stale=stale_evidence,
        clock_ok=(clock_verdict == TemporalVerdict.VALID),
    )


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=== Temporal Controller Test ===\n")
    
    controller = TemporalController()
    
    # Track some items
    claim1 = controller.track_claim("c1", domain="factual")
    claim2 = controller.track_claim("c2", domain="speculative")
    evidence1 = controller.track_evidence("e1", evidence_type="tool_result")
    
    print(f"Tracked claim c1: TTL={claim1.ttl_seconds}s, expires={claim1.expires_at}")
    print(f"Tracked claim c2: TTL={claim2.ttl_seconds}s (speculative)")
    print(f"Tracked evidence e1: TTL={evidence1.ttl_seconds}s (tool result)")
    
    # Check validity
    print(f"\nValidity checks:")
    print(f"  c1: {controller.check_item('c1').name}")
    print(f"  c2: {controller.check_item('c2').name}")
    print(f"  e1: {controller.check_item('e1').name}")
    
    # Simulate lag
    print(f"\nLag check (0s): {controller.check_processing_lag(datetime.now(timezone.utc)).name}")
    
    past = datetime.now(timezone.utc) - timedelta(seconds=10)
    print(f"Lag check (10s): {controller.check_processing_lag(past).name}")
    
    # Clock check
    print(f"\nClock coherence: {controller.check_clock_coherence().name}")
    
    # Turn-level check
    turn_start = datetime.now(timezone.utc) - timedelta(milliseconds=100)
    result = check_turn_temporal(
        controller,
        turn_start,
        claim_ids=["c1", "c2"],
        evidence_ids=["e1"],
    )
    
    print(f"\nTurn temporal result:")
    print(f"  Verdict: {result.verdict.name}")
    print(f"  Processing lag: {result.processing_lag_ms:.1f}ms")
    print(f"  Clock OK: {result.clock_ok}")
    
    # State
    print(f"\nState:")
    import json
    print(json.dumps(controller.get_state(), indent=2))
