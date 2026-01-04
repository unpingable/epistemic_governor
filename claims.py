"""
Minimal Claim Ledger (MCL)

The keystone artifact. If fly-by-wire is the control law,
the claim ledger is the instrumentation.

Without it:
- confidence floats
- provenance blurs
- laundering happens invisibly
- metrics are vibes

With it:
- everything snaps into place

Key invariants:
1. Provenance never upgrades without evidence
2. Confidence cannot increase on ASSUMED/PEER_ASSERTED without evidence
3. If a claim cannot be represented in the ledger, it should not be asserted

Usage:
    from epistemic_governor.claims import ClaimLedger, Provenance
    
    ledger = ClaimLedger()
    
    # Create a claim
    claim = ledger.new_claim(
        content="MegaCorp acquired TinySoft in 2019",
        provenance=Provenance.ASSUMED,
        confidence=0.3,
    )
    
    # Attach evidence
    ledger.attach_evidence(claim.claim_id, EvidenceRef(
        ref_id="ev_001",
        ref_type="TOOL_TRACE",
        locator="press_release_2019-06-14",
        scope="Acquisition date",
    ))
    
    # Now promotion is allowed
    ledger.promote(claim.claim_id, Provenance.RETRIEVED)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum, auto
from datetime import datetime
import uuid
import hashlib
import json


# =============================================================================
# Provenance Types (The Spine)
# =============================================================================

class Provenance(str, Enum):
    """
    Provenance types for claims.
    
    This is the spine of the whole system.
    Provenance NEVER upgrades without evidence.
    """
    
    # Direct observation (rare in LLM context)
    OBSERVED = "OBSERVED"
    
    # From tool with trace
    RETRIEVED = "RETRIEVED"
    
    # User supplied source/content
    USER_PROVIDED = "USER_PROVIDED"
    
    # Logically inferred from grounded claims
    DERIVED = "DERIVED"
    
    # From another agent (NEVER trust without evidence)
    PEER_ASSERTED = "PEER_ASSERTED"
    
    # Hypothetical / conditional
    ASSUMED = "ASSUMED"
    
    @property
    def is_grounded(self) -> bool:
        """Is this provenance grounded (has external support)?"""
        return self in {Provenance.OBSERVED, Provenance.RETRIEVED, Provenance.USER_PROVIDED}
    
    @property
    def requires_evidence_for_confidence(self) -> bool:
        """Does this provenance require evidence to increase confidence?"""
        return self in {Provenance.ASSUMED, Provenance.PEER_ASSERTED}


class ClaimStatus(str, Enum):
    """Status of a claim in the ledger."""
    
    # Currently asserted
    ACTIVE = "ACTIVE"
    
    # Cannot be asserted due to missing grounding
    BLOCKED = "BLOCKED"
    
    # Explicitly withdrawn (this is SUCCESS, not failure)
    RETRACTED = "RETRACTED"
    
    # Promoted to commitment (left hypothesis ledger)
    PROMOTED = "PROMOTED"


class ClaimSource(str, Enum):
    """Source of a claim."""
    MODEL = "MODEL"
    USER = "USER"
    PEER = "PEER"
    TOOL = "TOOL"


# =============================================================================
# Evidence Reference
# =============================================================================

@dataclass
class EvidenceRef:
    """
    Reference to evidence supporting a claim.
    
    Evidence is opaque to the ledger.
    The governor just needs to know it exists and is scoped.
    """
    ref_id: str
    ref_type: str       # URL | DOC | TOOL_TRACE | HUMAN_INPUT
    locator: str        # link, hash, trace id
    scope: str          # what aspect of the claim it supports
    
    # Optional metadata
    retrieved_at: Optional[datetime] = None
    confidence: float = 1.0  # How reliable is this evidence?
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ref_id": self.ref_id,
            "ref_type": self.ref_type,
            "locator": self.locator,
            "scope": self.scope,
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_tool_trace(cls, trace_id: str, scope: str) -> "EvidenceRef":
        """Create evidence ref from a tool trace."""
        return cls(
            ref_id=f"ev_{uuid.uuid4().hex[:8]}",
            ref_type="TOOL_TRACE",
            locator=trace_id,
            scope=scope,
            retrieved_at=datetime.now(),
        )
    
    @classmethod
    def from_url(cls, url: str, scope: str) -> "EvidenceRef":
        """Create evidence ref from a URL."""
        return cls(
            ref_id=f"ev_{uuid.uuid4().hex[:8]}",
            ref_type="URL",
            locator=url,
            scope=scope,
            retrieved_at=datetime.now(),
        )
    
    @classmethod
    def from_user(cls, description: str, scope: str) -> "EvidenceRef":
        """Create evidence ref from user input."""
        return cls(
            ref_id=f"ev_{uuid.uuid4().hex[:8]}",
            ref_type="HUMAN_INPUT",
            locator=description,
            scope=scope,
            retrieved_at=datetime.now(),
        )


# =============================================================================
# Claim Object
# =============================================================================

@dataclass
class Claim:
    """
    A claim is any assertion that could be wrong in a way that matters.
    
    Not filler text, not stylistic language, not generic advice.
    
    Yes:
    - Named entities
    - Dates
    - Causal explanations
    - Mechanisms
    - Citations
    - Quantitative statements
    - Factual descriptions
    """
    claim_id: str
    content: str                    # Normalized assertion
    provenance: Provenance
    confidence: float               # Bounded [0, 1]
    
    # Evidence
    evidence_refs: List[EvidenceRef] = field(default_factory=list)
    
    # Source and lifecycle
    source: ClaimSource = ClaimSource.MODEL
    status: ClaimStatus = ClaimStatus.ACTIVE
    
    # Temporal tracking
    created_at_step: int = 0
    last_updated_step: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    # For retraction tracking
    retracted_at_step: Optional[int] = None
    retraction_reason: Optional[str] = None
    
    # Content hash for deduplication
    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.lower().strip().encode()).hexdigest()[:16]
    
    @property
    def is_grounded(self) -> bool:
        """Is this claim grounded (has evidence or grounded provenance)?"""
        return self.provenance.is_grounded or len(self.evidence_refs) > 0
    
    @property
    def is_active(self) -> bool:
        return self.status == ClaimStatus.ACTIVE
    
    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.7
    
    @property
    def is_dangerous(self) -> bool:
        """Is this claim dangerous (high confidence + ungrounded)?"""
        return self.is_high_confidence and not self.is_grounded
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "content": self.content,
            "provenance": self.provenance.value,
            "confidence": self.confidence,
            "evidence_refs": [e.to_dict() for e in self.evidence_refs],
            "source": self.source.value,
            "status": self.status.value,
            "created_at_step": self.created_at_step,
            "is_grounded": self.is_grounded,
            "is_dangerous": self.is_dangerous,
        }


# =============================================================================
# Promotion Rules
# =============================================================================

class PromotionResult(Enum):
    """Result of a promotion attempt."""
    SUCCESS = auto()
    FORBIDDEN = auto()
    NEEDS_EVIDENCE = auto()


@dataclass
class PromotionAttempt:
    """Record of a promotion attempt."""
    claim_id: str
    from_provenance: Provenance
    to_provenance: Provenance
    result: PromotionResult
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)


# Forbidden promotions (hard faults) - these can NEVER happen
FORBIDDEN_PROMOTIONS: Set[Tuple[Provenance, Provenance]] = {
    # Cannot promote anything to OBSERVED (that's sacred)
    (Provenance.ASSUMED, Provenance.OBSERVED),
    (Provenance.PEER_ASSERTED, Provenance.OBSERVED),
    (Provenance.DERIVED, Provenance.OBSERVED),
    (Provenance.RETRIEVED, Provenance.OBSERVED),
    (Provenance.USER_PROVIDED, Provenance.OBSERVED),
}

# Promotions that REQUIRE evidence (but are allowed with it)
PROMOTIONS_REQUIRING_EVIDENCE: Set[Tuple[Provenance, Provenance]] = {
    (Provenance.ASSUMED, Provenance.DERIVED),
    (Provenance.ASSUMED, Provenance.RETRIEVED),
    (Provenance.ASSUMED, Provenance.USER_PROVIDED),
    (Provenance.PEER_ASSERTED, Provenance.DERIVED),
    (Provenance.PEER_ASSERTED, Provenance.RETRIEVED),
    (Provenance.PEER_ASSERTED, Provenance.USER_PROVIDED),
}


# =============================================================================
# Claim Ledger
# =============================================================================

class ClaimLedger:
    """
    Minimal Claim Ledger enforcing:
    
    1. Provenance never upgrades without evidence
    2. Confidence cannot increase on ASSUMED/PEER_ASSERTED without evidence
    3. If a claim cannot be represented, it should not be asserted
    
    This is the instrumentation that makes fly-by-wire real.
    """
    
    def __init__(self):
        self.claims: Dict[str, Claim] = {}
        self.step: int = 0
        
        # History
        self.promotion_history: List[PromotionAttempt] = []
        self.retraction_history: List[Tuple[str, int, str]] = []  # (claim_id, step, reason)
        
        # Metrics
        self.total_claims_created: int = 0
        self.total_promotions_attempted: int = 0
        self.total_promotions_forbidden: int = 0
        self.total_retractions: int = 0
    
    def tick(self):
        """Advance the step counter."""
        self.step += 1
    
    # =========================================================================
    # Claim Creation
    # =========================================================================
    
    def new_claim(
        self,
        content: str,
        provenance: Provenance,
        confidence: float,
        source: ClaimSource = ClaimSource.MODEL,
        status: ClaimStatus = ClaimStatus.ACTIVE,
    ) -> Claim:
        """
        Create a new claim.
        
        Confidence is bounded [0, 1].
        """
        claim = Claim(
            claim_id=f"claim_{uuid.uuid4().hex[:12]}",
            content=content.strip(),
            provenance=provenance,
            confidence=max(0.0, min(1.0, confidence)),
            source=source,
            status=status,
            created_at_step=self.step,
            last_updated_step=self.step,
        )
        
        self.claims[claim.claim_id] = claim
        self.total_claims_created += 1
        
        return claim
    
    def new_assumed_claim(self, content: str, confidence: float = 0.3) -> Claim:
        """Convenience: create an ASSUMED claim (default for model assertions)."""
        return self.new_claim(content, Provenance.ASSUMED, confidence)
    
    def new_retrieved_claim(
        self,
        content: str,
        evidence: EvidenceRef,
        confidence: float = 0.8,
    ) -> Claim:
        """Convenience: create a RETRIEVED claim with evidence."""
        claim = self.new_claim(content, Provenance.RETRIEVED, confidence)
        claim.evidence_refs.append(evidence)
        return claim
    
    def new_peer_claim(self, content: str, peer_id: str, confidence: float = 0.2) -> Claim:
        """
        Create a claim from a peer agent.
        
        Peer claims start at LOW confidence and CANNOT increase without evidence.
        This is the multi-agent resonance killer.
        """
        claim = self.new_claim(
            content,
            Provenance.PEER_ASSERTED,
            confidence=min(0.3, confidence),  # Cap peer confidence
            source=ClaimSource.PEER,
        )
        return claim
    
    # =========================================================================
    # Evidence Attachment
    # =========================================================================
    
    def attach_evidence(self, claim_id: str, ref: EvidenceRef) -> bool:
        """
        Attach evidence to a claim.
        
        Returns True if evidence was attached.
        """
        if claim_id not in self.claims:
            return False
        
        claim = self.claims[claim_id]
        claim.evidence_refs.append(ref)
        claim.last_updated_step = self.step
        
        return True
    
    def has_evidence(self, claim_id: str) -> bool:
        """Check if a claim has any evidence."""
        if claim_id not in self.claims:
            return False
        return len(self.claims[claim_id].evidence_refs) > 0
    
    # =========================================================================
    # Promotion (The Hard Part)
    # =========================================================================
    
    def can_promote(self, claim_id: str, new_provenance: Provenance) -> Tuple[bool, str]:
        """
        Check if a promotion is allowed.
        
        Returns (allowed, reason).
        """
        if claim_id not in self.claims:
            return False, "Claim not found"
        
        claim = self.claims[claim_id]
        old_prov = claim.provenance
        
        # Check forbidden promotions
        if (old_prov, new_provenance) in FORBIDDEN_PROMOTIONS:
            return False, f"Forbidden promotion: {old_prov.value} → {new_provenance.value}"
        
        # Check evidence requirements
        if (old_prov, new_provenance) in PROMOTIONS_REQUIRING_EVIDENCE:
            if not claim.evidence_refs:
                return False, f"Promotion requires evidence: {old_prov.value} → {new_provenance.value}"
        
        # Risky promotions to grounded types require evidence
        if new_provenance.is_grounded and old_prov.requires_evidence_for_confidence:
            if not claim.evidence_refs:
                return False, f"Grounded promotion requires evidence"
        
        return True, "OK"
    
    def promote(self, claim_id: str, new_provenance: Provenance) -> PromotionResult:
        """
        Attempt to promote a claim to a new provenance level.
        
        Enforces "no promotion without evidence" for risky upgrades.
        """
        self.total_promotions_attempted += 1
        
        allowed, reason = self.can_promote(claim_id, new_provenance)
        
        claim = self.claims.get(claim_id)
        old_prov = claim.provenance if claim else Provenance.ASSUMED
        
        if not allowed:
            self.total_promotions_forbidden += 1
            result = PromotionResult.FORBIDDEN if "Forbidden" in reason else PromotionResult.NEEDS_EVIDENCE
            
            self.promotion_history.append(PromotionAttempt(
                claim_id=claim_id,
                from_provenance=old_prov,
                to_provenance=new_provenance,
                result=result,
                reason=reason,
            ))
            
            return result
        
        # Perform promotion
        claim.provenance = new_provenance
        claim.last_updated_step = self.step
        
        self.promotion_history.append(PromotionAttempt(
            claim_id=claim_id,
            from_provenance=old_prov,
            to_provenance=new_provenance,
            result=PromotionResult.SUCCESS,
            reason="OK",
        ))
        
        return PromotionResult.SUCCESS
    
    # =========================================================================
    # Confidence Updates (The Money Rule)
    # =========================================================================
    
    def update_confidence(
        self,
        claim_id: str,
        delta: float,
        requires_evidence: bool = True,
    ) -> bool:
        """
        Update claim confidence.
        
        THE MONEY RULE:
        - If new evidence supports claim: confidence += delta
        - If provenance is ASSUMED/PEER_ASSERTED: confidence = min(confidence, previous)
        - If constraint strain rises: confidence -= decay
        
        NEVER:
        - Increase confidence due to repetition
        - Increase confidence due to elaboration
        - Increase confidence due to peer agreement
        
        Confidence is EARNED, not accumulated.
        """
        if claim_id not in self.claims:
            return False
        
        claim = self.claims[claim_id]
        
        # Key rule: no confidence increase without evidence for risky provenances
        if requires_evidence and delta > 0:
            if claim.provenance.requires_evidence_for_confidence:
                if not claim.evidence_refs:
                    # Silently refuse to increase confidence
                    return False
        
        # Apply update
        old_confidence = claim.confidence
        claim.confidence = max(0.0, min(1.0, claim.confidence + delta))
        claim.last_updated_step = self.step
        
        return True
    
    def decay_ungrounded_confidence(self, decay: float = 0.1):
        """Decay confidence on all ungrounded claims."""
        for claim in self.active_claims():
            if not claim.is_grounded:
                claim.confidence = max(0.0, claim.confidence - decay)
                claim.last_updated_step = self.step
    
    # =========================================================================
    # Status Changes
    # =========================================================================
    
    def block(self, claim_id: str, reason: str = "Missing grounding") -> bool:
        """Block a claim (cannot be asserted due to missing grounding)."""
        if claim_id not in self.claims:
            return False
        
        claim = self.claims[claim_id]
        claim.status = ClaimStatus.BLOCKED
        claim.last_updated_step = self.step
        
        return True
    
    def retract(self, claim_id: str, reason: str = "Explicit retraction") -> bool:
        """
        Retract a claim.
        
        Retraction is NOT failure. Retraction is SUCCESSFUL RECOVERY.
        Metrics should reward clean retraction.
        """
        if claim_id not in self.claims:
            return False
        
        claim = self.claims[claim_id]
        claim.status = ClaimStatus.RETRACTED
        claim.retracted_at_step = self.step
        claim.retraction_reason = reason
        claim.last_updated_step = self.step
        
        self.total_retractions += 1
        self.retraction_history.append((claim_id, self.step, reason))
        
        return True
    
    def unblock(self, claim_id: str) -> bool:
        """Unblock a claim (after evidence is attached)."""
        if claim_id not in self.claims:
            return False
        
        claim = self.claims[claim_id]
        if claim.status != ClaimStatus.BLOCKED:
            return False
        
        # Can only unblock if now grounded
        if not claim.is_grounded:
            return False
        
        claim.status = ClaimStatus.ACTIVE
        claim.last_updated_step = self.step
        
        return True
    
    # =========================================================================
    # Queries
    # =========================================================================
    
    def get(self, claim_id: str) -> Optional[Claim]:
        """Get a claim by ID."""
        return self.claims.get(claim_id)
    
    def active_claims(self) -> List[Claim]:
        """Get all active claims."""
        return [c for c in self.claims.values() if c.status == ClaimStatus.ACTIVE]
    
    def blocked_claims(self) -> List[Claim]:
        """Get all blocked claims."""
        return [c for c in self.claims.values() if c.status == ClaimStatus.BLOCKED]
    
    def retracted_claims(self) -> List[Claim]:
        """Get all retracted claims."""
        return [c for c in self.claims.values() if c.status == ClaimStatus.RETRACTED]
    
    def ungrounded_claims(self) -> List[Claim]:
        """Get all active ungrounded claims."""
        return [c for c in self.active_claims() if not c.is_grounded]
    
    def dangerous_claims(self) -> List[Claim]:
        """Get all dangerous claims (high confidence + ungrounded)."""
        return [c for c in self.active_claims() if c.is_dangerous]
    
    def claims_by_provenance(self, provenance: Provenance) -> List[Claim]:
        """Get all active claims with a specific provenance."""
        return [c for c in self.active_claims() if c.provenance == provenance]
    
    def find_by_content(self, content: str) -> Optional[Claim]:
        """Find a claim by content (approximate match)."""
        content_hash = hashlib.sha256(content.lower().strip().encode()).hexdigest()[:16]
        for claim in self.claims.values():
            if claim.content_hash == content_hash:
                return claim
        return None
    
    # =========================================================================
    # Metrics
    # =========================================================================
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get ledger metrics."""
        active = self.active_claims()
        
        # Provenance distribution
        prov_dist = {}
        for p in Provenance:
            count = len([c for c in active if c.provenance == p])
            prov_dist[p.value] = count
        
        # Confidence distribution
        high_conf = len([c for c in active if c.confidence >= 0.7])
        med_conf = len([c for c in active if 0.3 <= c.confidence < 0.7])
        low_conf = len([c for c in active if c.confidence < 0.3])
        
        return {
            "step": self.step,
            "total_claims": len(self.claims),
            "active_claims": len(active),
            "blocked_claims": len(self.blocked_claims()),
            "retracted_claims": len(self.retracted_claims()),
            "ungrounded_claims": len(self.ungrounded_claims()),
            "dangerous_claims": len(self.dangerous_claims()),
            "provenance_distribution": prov_dist,
            "confidence_distribution": {
                "high": high_conf,
                "medium": med_conf,
                "low": low_conf,
            },
            "total_promotions_attempted": self.total_promotions_attempted,
            "total_promotions_forbidden": self.total_promotions_forbidden,
            "total_retractions": self.total_retractions,
            "forbidden_promotion_rate": (
                self.total_promotions_forbidden / self.total_promotions_attempted
                if self.total_promotions_attempted > 0 else 0.0
            ),
        }
    
    def get_provenance_entropy(self) -> float:
        """
        Get provenance entropy (diversity of provenance types).
        
        A healthy system shifts to RETRIEVED under factual pressure.
        """
        import math
        
        active = self.active_claims()
        if not active:
            return 0.0
        
        counts = {}
        for c in active:
            counts[c.provenance] = counts.get(c.provenance, 0) + 1
        
        total = len(active)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize ledger state."""
        return {
            "step": self.step,
            "claims": {cid: c.to_dict() for cid, c in self.claims.items()},
            "metrics": self.get_metrics(),
        }
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=2, default=str)


# =============================================================================
# Integration: Claim Extraction Helpers
# =============================================================================

import re

# Patterns for detecting claims in text
DATE_PATTERN = re.compile(r"\b(18\d{2}|19\d{2}|20\d{2})\b")
ENTITY_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
QUANTITY_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\s*(million|billion|percent|%|dollars|USD|EUR)\b", re.I)
ASSERTIVE_PATTERN = re.compile(
    r"\b(definitely|certainly|clearly|undoubtedly|obviously|"
    r"was acquired|acquired in|founded in|established in|"
    r"is located|is headquartered|died in|born in)\b",
    re.I
)


def extract_claim_signals(text: str) -> Dict[str, Any]:
    """
    Extract signals indicating potential claims in text.
    
    This is crude but sufficient for gating.
    You don't need to detect all claims perfectly.
    You need to never promote one silently.
    """
    dates = DATE_PATTERN.findall(text)
    entities = ENTITY_PATTERN.findall(text)
    quantities = QUANTITY_PATTERN.findall(text)
    assertive_matches = ASSERTIVE_PATTERN.findall(text)
    
    return {
        "dates": dates,
        "entities": entities,
        "quantities": quantities,
        "assertive_phrases": assertive_matches,
        "has_speculative_content": len(dates) > 0 or len(entities) > 0 or len(quantities) > 0,
        "assertiveness_score": len(assertive_matches) * 0.2,
    }


def create_claims_from_signals(
    ledger: ClaimLedger,
    text: str,
    signals: Dict[str, Any],
) -> List[str]:
    """
    Create ASSUMED claims from extracted signals.
    
    Returns list of claim IDs.
    """
    claim_ids = []
    
    # Create claims for dates
    for date in signals.get("dates", []):
        claim = ledger.new_assumed_claim(
            content=f"Date asserted: {date}",
            confidence=0.2,
        )
        claim_ids.append(claim.claim_id)
    
    # Create claims for entities
    for entity in signals.get("entities", []):
        claim = ledger.new_assumed_claim(
            content=f"Entity asserted: {entity}",
            confidence=0.2,
        )
        claim_ids.append(claim.claim_id)
    
    # Create claims for quantities
    for qty in signals.get("quantities", []):
        claim = ledger.new_assumed_claim(
            content=f"Quantity asserted: {qty[0]} {qty[1]}",
            confidence=0.2,
        )
        claim_ids.append(claim.claim_id)
    
    return claim_ids


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=== Minimal Claim Ledger Demo ===\n")
    
    ledger = ClaimLedger()
    
    # Create some claims
    print("--- Creating Claims ---")
    
    # Model assertion (ASSUMED, low confidence)
    claim1 = ledger.new_assumed_claim(
        "MegaCorp acquired TinySoft in 2019",
        confidence=0.3,
    )
    print(f"Created: {claim1.content}")
    print(f"  Provenance: {claim1.provenance.value}, Confidence: {claim1.confidence}")
    print(f"  Grounded: {claim1.is_grounded}, Dangerous: {claim1.is_dangerous}")
    
    # Try to increase confidence without evidence (should fail silently)
    print("\n--- Confidence Update (no evidence) ---")
    result = ledger.update_confidence(claim1.claim_id, +0.5)
    print(f"Confidence update attempted: {result}")
    print(f"  Confidence after: {claim1.confidence}")  # Should still be 0.3
    
    # Try to promote without evidence (should fail)
    print("\n--- Promotion Attempt (no evidence) ---")
    result = ledger.promote(claim1.claim_id, Provenance.RETRIEVED)
    print(f"Promotion result: {result.name}")
    print(f"  Provenance: {claim1.provenance.value}")  # Should still be ASSUMED
    
    # Attach evidence
    print("\n--- Attaching Evidence ---")
    evidence = EvidenceRef.from_tool_trace(
        "press_release_2019-06-14",
        "Acquisition date confirmation",
    )
    ledger.attach_evidence(claim1.claim_id, evidence)
    print(f"Evidence attached: {evidence.locator}")
    print(f"  Grounded now: {claim1.is_grounded}")
    
    # Now promotion should succeed
    print("\n--- Promotion Attempt (with evidence) ---")
    result = ledger.promote(claim1.claim_id, Provenance.RETRIEVED)
    print(f"Promotion result: {result.name}")
    print(f"  Provenance: {claim1.provenance.value}")  # Should be RETRIEVED
    
    # Now confidence can increase
    print("\n--- Confidence Update (with evidence) ---")
    ledger.update_confidence(claim1.claim_id, +0.5)
    print(f"  Confidence after: {claim1.confidence}")  # Should be 0.8
    
    # Peer claim (starts low, can't increase without evidence)
    print("\n--- Peer Claim ---")
    peer_claim = ledger.new_peer_claim(
        "The acquisition was for $500 million",
        peer_id="agent_b",
        confidence=0.9,  # Will be capped
    )
    print(f"Peer claim confidence: {peer_claim.confidence}")  # Capped at 0.3
    
    # Try to increase peer claim confidence (should fail)
    ledger.update_confidence(peer_claim.claim_id, +0.5)
    print(f"After increase attempt: {peer_claim.confidence}")  # Still 0.3
    
    # Retraction (success, not failure!)
    print("\n--- Retraction ---")
    ledger.retract(peer_claim.claim_id, "Unverified peer assertion")
    print(f"Peer claim status: {peer_claim.status.value}")
    
    # Metrics
    print("\n--- Metrics ---")
    metrics = ledger.get_metrics()
    print(f"Active claims: {metrics['active_claims']}")
    print(f"Ungrounded claims: {metrics['ungrounded_claims']}")
    print(f"Dangerous claims: {metrics['dangerous_claims']}")
    print(f"Provenance entropy: {ledger.get_provenance_entropy():.2f}")
    print(f"Forbidden promotion rate: {metrics['forbidden_promotion_rate']:.1%}")
    
    # Claim extraction
    print("\n--- Claim Extraction ---")
    text = "MegaCorp definitely acquired TinySoft in 2019 for $500 million dollars."
    signals = extract_claim_signals(text)
    print(f"Signals: {signals}")
