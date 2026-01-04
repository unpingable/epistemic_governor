"""
Proposition Identity Router

Maps local proposition hashes (cheap, lossy) to stable canonical identities.
Identity is ledger-controlled, not magical embedding vibe.

Key principle: router suggestions are PROPOSALS, ledger events make them real.

Objects:
- prop_hash: cheap local fingerprint from (entity_norm, predicate_norm, value_norm)
- prop_id: stable canonical identity in the ledger (p_...)

Ledger events:
- PROP_BIND: first time we mint prop_id for a prop_hash
- PROP_REBIND: attach another prop_hash to existing prop_id (paraphrase merge)
- PROP_SPLIT: detach a prop_hash from prop_id (correction)

Usage:
    from epistemic_governor.prop_router import (
        PropositionRouter,
        PropositionIndex,
        BindResult,
    )
    
    router = PropositionRouter()
    result = router.bind_or_mint(claim_atom)
    
    if result.is_new:
        # Mint new prop_id
        ledger.append(PROP_BIND event)
    elif result.is_rebind:
        # Merge to existing
        ledger.append(PROP_REBIND event)
    elif result.needs_arbitration:
        # Gray zone - ask user
        pass
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any, Deque
from collections import deque
from datetime import datetime
from enum import Enum, auto
import hashlib
import json
import re


# =============================================================================
# Bind Results
# =============================================================================

class BindAction(Enum):
    """Result of router bind attempt."""
    NEW = auto()           # Mint new prop_id
    REBIND = auto()        # Merge to existing prop_id
    EXACT = auto()         # Already bound (fast path)
    ARBITRATE = auto()     # Gray zone - needs human decision


@dataclass
class BindResult:
    """Result of bind_or_mint operation."""
    action: BindAction
    prop_id: str
    prop_hash: str
    
    # For REBIND
    matched_prop_id: Optional[str] = None
    match_score: float = 0.0
    match_reason: str = ""
    
    # For ARBITRATE - now includes reasons for each candidate
    candidates: List[Tuple[str, float, str]] = field(default_factory=list)  # [(prop_id, score, reason), ...]
    
    # Mode discipline (INT-2)
    mode: Optional[str] = None  # ClaimMode value if provided
    creates_timeline_obligations: bool = True  # False for COUNTERFACTUAL/SIMULATION/QUOTED
    
    @property
    def is_new(self) -> bool:
        return self.action == BindAction.NEW
    
    @property
    def is_rebind(self) -> bool:
        return self.action == BindAction.REBIND
    
    @property
    def is_exact(self) -> bool:
        return self.action == BindAction.EXACT
    
    @property
    def needs_arbitration(self) -> bool:
        return self.action == BindAction.ARBITRATE
    
    @property
    def has_info_gain(self) -> bool:
        """Check if this result involves suspicious information gain."""
        return "info_gain" in self.match_reason


# =============================================================================
# Ledger Event Schemas
# =============================================================================

@dataclass
class PropBindEvent:
    """
    First time we mint prop_id for a prop_hash.
    
    Ledger entry_type: "prop_bind"
    """
    prop_id: str
    prop_hash: str
    
    # Canonical representation
    entity_norm: str
    predicate_norm: str
    value_norm: str
    
    # Raw values for audit
    entity_raw: str = ""
    predicate_raw: str = ""
    value_raw: str = ""
    
    # Features (for future matching)
    value_features: Dict[str, Any] = field(default_factory=dict)
    
    # Router metadata
    router_version: str = "1.0.0"
    
    def to_ledger_data(self) -> Dict:
        return {
            "prop_id": self.prop_id,
            "prop_hash": self.prop_hash,
            "entity_norm": self.entity_norm,
            "predicate_norm": self.predicate_norm,
            "value_norm": self.value_norm,
            "entity_raw": self.entity_raw,
            "predicate_raw": self.predicate_raw,
            "value_raw": self.value_raw,
            "value_features": self.value_features,
            "router_version": self.router_version,
        }


@dataclass
class PropRebindEvent:
    """
    Attach another prop_hash to existing prop_id (paraphrase merge).
    
    Ledger entry_type: "prop_rebind"
    """
    prop_id: str              # Target identity
    new_prop_hash: str        # Hash being merged
    
    # What triggered the rebind
    match_score: float
    match_reason: str
    
    # The claim that triggered this
    entity_norm: str = ""
    predicate_norm: str = ""
    value_norm: str = ""
    value_raw: str = ""
    
    # Router metadata
    router_version: str = "1.0.0"
    
    def to_ledger_data(self) -> Dict:
        return {
            "prop_id": self.prop_id,
            "new_prop_hash": self.new_prop_hash,
            "match_score": self.match_score,
            "match_reason": self.match_reason,
            "entity_norm": self.entity_norm,
            "predicate_norm": self.predicate_norm,
            "value_norm": self.value_norm,
            "value_raw": self.value_raw,
            "router_version": self.router_version,
        }


@dataclass
class PropSplitEvent:
    """
    Detach a prop_hash from a prop_id (correction).
    
    Ledger entry_type: "prop_split"
    """
    prop_id: str              # Identity being split from
    prop_hash: str            # Hash being detached
    new_prop_id: str          # New identity for the detached hash
    reason: str               # Why we're splitting
    
    def to_ledger_data(self) -> Dict:
        return {
            "prop_id": self.prop_id,
            "prop_hash": self.prop_hash,
            "new_prop_id": self.new_prop_id,
            "reason": self.reason,
        }


# =============================================================================
# Proposition Record
# =============================================================================

@dataclass
class PropositionRecord:
    """
    Canonical representation of a proposition identity.
    
    Stored in the index, updated as hashes are bound/split.
    """
    prop_id: str
    
    # Canonical triple (from first binding)
    entity_norm: str
    predicate_norm: str
    value_norm: str
    
    # All hashes bound to this identity
    bound_hashes: Set[str] = field(default_factory=set)
    
    # Features for matching (merged from all bindings)
    value_features: Dict[str, Any] = field(default_factory=dict)
    
    # Collision budget tracking (router-native telemetry)
    bind_count: int = 0
    recent_bind_steps: Deque[int] = field(default_factory=lambda: deque(maxlen=20))
    unique_value_signatures: Set[str] = field(default_factory=set)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_hash(self, prop_hash: str, step: int = 0, value_signature: str = ""):
        self.bound_hashes.add(prop_hash)
        self.bind_count += 1
        self.recent_bind_steps.append(step)
        if value_signature:
            self.unique_value_signatures.add(value_signature)
        self.updated_at = datetime.now()
    
    def remove_hash(self, prop_hash: str):
        self.bound_hashes.discard(prop_hash)
        self.updated_at = datetime.now()
    
    @property
    def is_overbinding(self) -> bool:
        """Check if this identity is accumulating too many distinct bindings."""
        # Too many binds + too many distinct value signatures = smell
        return len(self.bound_hashes) > 10 and len(self.unique_value_signatures) > 3
    
    @property
    def recent_bind_rate(self) -> float:
        """Recent bindings per step."""
        if len(self.recent_bind_steps) < 2:
            return 0.0
        steps = list(self.recent_bind_steps)
        span = max(steps) - min(steps) + 1
        return len(steps) / span if span > 0 else 0.0


@dataclass
class HashMeta:
    """
    Per-hash metadata for proper split semantics.
    
    Stores the original triple/features for each hash so split
    can create a new record with the detached hash's OWN meaning.
    """
    prop_hash: str
    entity_norm: str
    predicate_norm: str
    value_norm: str
    value_features: Dict[str, Any] = field(default_factory=dict)
    entity_raw: str = ""
    value_raw: str = ""
    bound_at_step: int = 0


# =============================================================================
# Proposition Index
# =============================================================================

class PropositionIndex:
    """
    In-memory index of proposition identities.
    
    Can be persisted via checkpoint/restore.
    """
    
    def __init__(self):
        # prop_hash -> prop_id (fast lookup)
        self._hash_to_id: Dict[str, str] = {}
        
        # prop_id -> PropositionRecord
        self._records: Dict[str, PropositionRecord] = {}
        
        # prop_hash -> HashMeta (for proper split semantics)
        self._hash_meta: Dict[str, HashMeta] = {}
        
        # Indexes for candidate retrieval
        self._by_predicate: Dict[str, Set[str]] = {}  # predicate -> prop_ids
        self._by_entity: Dict[str, Set[str]] = {}     # entity_token -> prop_ids
        
        # Counter for minting
        self._id_counter = 0
        
        # Step counter for collision tracking
        self._step = 0
    
    def get_by_hash(self, prop_hash: str) -> Optional[str]:
        """Fast path: check if hash is already bound."""
        return self._hash_to_id.get(prop_hash)
    
    def get_record(self, prop_id: str) -> Optional[PropositionRecord]:
        """Get the full record for a prop_id."""
        return self._records.get(prop_id)
    
    def get_hash_meta(self, prop_hash: str) -> Optional[HashMeta]:
        """Get the metadata for a specific hash."""
        return self._hash_meta.get(prop_hash)
    
    def mint_id(self) -> str:
        """Mint a new prop_id."""
        self._id_counter += 1
        return f"p_{self._id_counter:08d}"
    
    def tick(self):
        """Increment step counter."""
        self._step += 1
    
    def bind(
        self,
        prop_hash: str,
        entity_norm: str,
        predicate_norm: str,
        value_norm: str,
        value_features: Dict[str, Any] = None,
        entity_raw: str = "",
        value_raw: str = "",
    ) -> str:
        """
        Bind a hash to a new prop_id (PROP_BIND).
        
        Returns the new prop_id.
        """
        prop_id = self.mint_id()
        vf = value_features or {}
        
        # Create value signature for collision tracking
        value_sig = self._compute_value_signature(value_norm, vf)
        
        record = PropositionRecord(
            prop_id=prop_id,
            entity_norm=entity_norm,
            predicate_norm=predicate_norm,
            value_norm=value_norm,
            bound_hashes={prop_hash},
            value_features=vf,
            bind_count=1,
            unique_value_signatures={value_sig} if value_sig else set(),
        )
        record.recent_bind_steps.append(self._step)
        
        self._records[prop_id] = record
        self._hash_to_id[prop_hash] = prop_id
        
        # Store hash metadata for split semantics
        self._hash_meta[prop_hash] = HashMeta(
            prop_hash=prop_hash,
            entity_norm=entity_norm,
            predicate_norm=predicate_norm,
            value_norm=value_norm,
            value_features=vf,
            entity_raw=entity_raw,
            value_raw=value_raw,
            bound_at_step=self._step,
        )
        
        # Update indexes
        self._by_predicate.setdefault(predicate_norm, set()).add(prop_id)
        for token in self._tokenize_entity(entity_norm):
            self._by_entity.setdefault(token, set()).add(prop_id)
        
        return prop_id
    
    def rebind(
        self,
        prop_hash: str,
        target_prop_id: str,
        entity_norm: str = "",
        predicate_norm: str = "",
        value_norm: str = "",
        value_features: Dict[str, Any] = None,
        entity_raw: str = "",
        value_raw: str = "",
    ) -> bool:
        """
        Rebind a hash to an existing prop_id (PROP_REBIND).
        
        Returns True if successful.
        """
        if target_prop_id not in self._records:
            return False
        
        record = self._records[target_prop_id]
        vf = value_features or {}
        
        # Compute value signature
        value_sig = self._compute_value_signature(value_norm, vf)
        
        record.add_hash(prop_hash, self._step, value_sig)
        self._hash_to_id[prop_hash] = target_prop_id
        
        # Store hash metadata (using provided values or record's canonical)
        self._hash_meta[prop_hash] = HashMeta(
            prop_hash=prop_hash,
            entity_norm=entity_norm or record.entity_norm,
            predicate_norm=predicate_norm or record.predicate_norm,
            value_norm=value_norm or record.value_norm,
            value_features=vf,
            entity_raw=entity_raw,
            value_raw=value_raw,
            bound_at_step=self._step,
        )
        
        return True
    
    def split(self, prop_hash: str, old_prop_id: str) -> Optional[str]:
        """
        Split a hash from its current prop_id (PROP_SPLIT).
        
        Uses the hash's OWN metadata to create a new record,
        not the parent's canonical triple.
        
        Returns the new prop_id, or None if failed.
        """
        if old_prop_id not in self._records:
            return None
        
        old_record = self._records[old_prop_id]
        if prop_hash not in old_record.bound_hashes:
            return None
        
        # Remove from old
        old_record.remove_hash(prop_hash)
        
        # Get the hash's own metadata (critical for proper split)
        hash_meta = self._hash_meta.get(prop_hash)
        
        if hash_meta:
            # Use the hash's own triple
            entity = hash_meta.entity_norm
            predicate = hash_meta.predicate_norm
            value = hash_meta.value_norm
            features = hash_meta.value_features
        else:
            # Fallback to parent (shouldn't happen if bind/rebind are correct)
            entity = old_record.entity_norm
            predicate = old_record.predicate_norm
            value = old_record.value_norm
            features = {}
        
        # Create new record with the hash's own meaning
        new_prop_id = self.mint_id()
        new_record = PropositionRecord(
            prop_id=new_prop_id,
            entity_norm=entity,
            predicate_norm=predicate,
            value_norm=value,
            bound_hashes={prop_hash},
            value_features=features,
        )
        
        self._records[new_prop_id] = new_record
        self._hash_to_id[prop_hash] = new_prop_id
        
        # Update indexes for new record
        self._by_predicate.setdefault(predicate, set()).add(new_prop_id)
        for token in self._tokenize_entity(entity):
            self._by_entity.setdefault(token, set()).add(new_prop_id)
        
        return new_prop_id
    
    def _compute_value_signature(self, value_norm: str, features: Dict) -> str:
        """Compute a signature for collision tracking."""
        parts = [value_norm[:20]]  # Truncated norm
        if features.get("year"):
            parts.append(f"y{features['year']}")
        if features.get("month"):
            parts.append(f"m{features['month'][:3]}")
        if features.get("unit"):
            parts.append(f"u{features['unit']}")
        return ":".join(parts)
    
    def get_candidates(
        self,
        predicate_norm: str,
        entity_norm: str,
    ) -> List[str]:
        """
        Get candidate prop_ids for matching.
        
        Uses predicate + entity overlap.
        """
        # Start with predicate match
        candidates = self._by_predicate.get(predicate_norm, set()).copy()
        
        # Filter by entity overlap
        entity_tokens = self._tokenize_entity(entity_norm)
        entity_candidates = set()
        for token in entity_tokens:
            entity_candidates.update(self._by_entity.get(token, set()))
        
        # Intersection
        if entity_candidates:
            candidates &= entity_candidates
        
        return list(candidates)
    
    def _tokenize_entity(self, entity: str) -> List[str]:
        """
        Tokenize entity for indexing.
        
        Drops purely numeric tokens and common junk to prevent
        unrelated things from colliding via numeric shards.
        """
        tokens = re.split(r'[_.\s]+', entity.lower())
        # Drop numeric tokens and short/junk tokens
        junk = {'v', 'etc', 'config', 'the', 'a', 'an'}
        return [t for t in tokens if len(t) > 1 and not t.isdigit() and t not in junk]
    
    def checkpoint(self) -> Dict[str, Any]:
        """Create a checkpoint for persistence."""
        return {
            "id_counter": self._id_counter,
            "hash_to_id": dict(self._hash_to_id),
            "records": {
                pid: {
                    "prop_id": r.prop_id,
                    "entity_norm": r.entity_norm,
                    "predicate_norm": r.predicate_norm,
                    "value_norm": r.value_norm,
                    "bound_hashes": list(r.bound_hashes),
                    "value_features": r.value_features,
                }
                for pid, r in self._records.items()
            },
        }
    
    def restore(self, checkpoint: Dict[str, Any]):
        """Restore from checkpoint."""
        self._id_counter = checkpoint["id_counter"]
        self._hash_to_id = dict(checkpoint["hash_to_id"])
        self._records = {}
        self._by_predicate = {}
        self._by_entity = {}
        
        for pid, data in checkpoint["records"].items():
            record = PropositionRecord(
                prop_id=data["prop_id"],
                entity_norm=data["entity_norm"],
                predicate_norm=data["predicate_norm"],
                value_norm=data["value_norm"],
                bound_hashes=set(data["bound_hashes"]),
                value_features=data.get("value_features", {}),
            )
            self._records[pid] = record
            
            # Rebuild indexes
            self._by_predicate.setdefault(record.predicate_norm, set()).add(pid)
            for token in self._tokenize_entity(record.entity_norm):
                self._by_entity.setdefault(token, set()).add(pid)


# =============================================================================
# Proposition Router
# =============================================================================

class PropositionRouter:
    """
    Routes proposition hashes to stable identities.
    
    Two-stage algorithm:
    1. Fast path: exact hash match
    2. Candidate retrieval + scoring
    
    Thresholds:
    - BIND_THRESHOLD (0.92): auto-rebind
    - MAYBE_THRESHOLD (0.80): arbitration zone
    - Below: mint new
    """
    
    VERSION = "1.0.0"
    
    # Thresholds
    BIND_THRESHOLD = 0.92    # Auto-rebind above this
    MAYBE_THRESHOLD = 0.80   # Gray zone
    
    # Scoring weights
    WEIGHT_ENTITY = 0.4
    WEIGHT_PREDICATE = 0.3
    WEIGHT_VALUE = 0.3
    
    def __init__(self, index: PropositionIndex = None):
        self.index = index or PropositionIndex()
    
    def bind_or_mint(
        self,
        prop_hash: str,
        entity_norm: str,
        predicate_norm: str,
        value_norm: str,
        value_raw: str = "",
        value_features: Dict[str, Any] = None,
        entity_raw: str = "",
        mode: str = "factual",  # ClaimMode value (INT-2)
    ) -> BindResult:
        """
        Bind a prop_hash to an identity, or mint a new one.
        
        Mode discipline (INT-2):
        - mode is passed through to BindResult
        - creates_timeline_obligations is set based on mode
        - COUNTERFACTUAL/SIMULATION/QUOTED cannot create timeline obligations
        
        Returns BindResult indicating action taken.
        """
        vf = value_features or {}
        
        # Mode discipline: determine obligation eligibility (INT-2A)
        timeline_modes = {"factual", "procedural"}
        creates_timeline = mode.lower() in timeline_modes
        
        # Tick step counter
        self.index.tick()
        
        # Stage 0: Fast path - exact hash match
        existing_id = self.index.get_by_hash(prop_hash)
        if existing_id:
            return BindResult(
                action=BindAction.EXACT,
                prop_id=existing_id,
                prop_hash=prop_hash,
                mode=mode,
                creates_timeline_obligations=creates_timeline,
            )
        
        # Stage 1: Candidate retrieval
        candidates = self.index.get_candidates(predicate_norm, entity_norm)
        
        if not candidates:
            # No candidates - mint new
            new_id = self.index.bind(
                prop_hash=prop_hash,
                entity_norm=entity_norm,
                predicate_norm=predicate_norm,
                value_norm=value_norm,
                value_features=vf,
                entity_raw=entity_raw,
                value_raw=value_raw,
            )
            return BindResult(
                action=BindAction.NEW,
                prop_id=new_id,
                prop_hash=prop_hash,
                mode=mode,
                creates_timeline_obligations=creates_timeline,
            )
        
        # Stage 2: Score candidates
        scored = []
        for cand_id in candidates:
            record = self.index.get_record(cand_id)
            if not record:
                continue
            
            score, reason = self._score_match(
                entity_norm=entity_norm,
                predicate_norm=predicate_norm,
                value_norm=value_norm,
                value_features=vf,
                candidate=record,
            )
            scored.append((cand_id, score, reason))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        if not scored:
            # No valid candidates - mint new
            new_id = self.index.bind(
                prop_hash=prop_hash,
                entity_norm=entity_norm,
                predicate_norm=predicate_norm,
                value_norm=value_norm,
                value_features=vf,
                entity_raw=entity_raw,
                value_raw=value_raw,
            )
            return BindResult(
                action=BindAction.NEW,
                prop_id=new_id,
                prop_hash=prop_hash,
                mode=mode,
                creates_timeline_obligations=creates_timeline,
            )
        
        best_id, best_score, best_reason = scored[0]
        
        # Check for info_gain flag (force arbitration regardless of score)
        if "info_gain" in best_reason:
            return BindResult(
                action=BindAction.ARBITRATE,
                prop_id="",
                prop_hash=prop_hash,
                candidates=[(cid, score, reason) for cid, score, reason in scored[:3]],
                match_reason="info_gain_detected",
                mode=mode,
                creates_timeline_obligations=creates_timeline,
            )
        
        # Decision based on threshold
        if best_score >= self.BIND_THRESHOLD:
            # Auto-rebind
            self.index.rebind(
                prop_hash=prop_hash,
                target_prop_id=best_id,
                entity_norm=entity_norm,
                predicate_norm=predicate_norm,
                value_norm=value_norm,
                value_features=vf,
                entity_raw=entity_raw,
                value_raw=value_raw,
            )
            return BindResult(
                action=BindAction.REBIND,
                prop_id=best_id,
                prop_hash=prop_hash,
                matched_prop_id=best_id,
                match_score=best_score,
                match_reason=best_reason,
                mode=mode,
                creates_timeline_obligations=creates_timeline,
            )
        
        elif best_score >= self.MAYBE_THRESHOLD:
            # Gray zone - needs arbitration
            return BindResult(
                action=BindAction.ARBITRATE,
                prop_id="",  # Not determined yet
                prop_hash=prop_hash,
                candidates=[(cid, score, reason) for cid, score, reason in scored[:3]],
                match_reason=best_reason,
                mode=mode,
                creates_timeline_obligations=creates_timeline,
            )
        
        else:
            # Below threshold - mint new
            new_id = self.index.bind(
                prop_hash=prop_hash,
                entity_norm=entity_norm,
                predicate_norm=predicate_norm,
                value_norm=value_norm,
                value_features=vf,
                entity_raw=entity_raw,
                value_raw=value_raw,
            )
            return BindResult(
                action=BindAction.NEW,
                prop_id=new_id,
                prop_hash=prop_hash,
                mode=mode,
                creates_timeline_obligations=creates_timeline,
            )
    
    def force_rebind(self, prop_hash: str, target_prop_id: str) -> bool:
        """
        Force a rebind (after arbitration approval).
        """
        return self.index.rebind(prop_hash, target_prop_id)
    
    def split(self, prop_hash: str, reason: str) -> Optional[str]:
        """
        Split a hash from its current identity.
        
        Returns new prop_id, or None if failed.
        """
        current_id = self.index.get_by_hash(prop_hash)
        if not current_id:
            return None
        
        return self.index.split(prop_hash, current_id)
    
    def _score_match(
        self,
        entity_norm: str,
        predicate_norm: str,
        value_norm: str,
        value_features: Dict[str, Any],
        candidate: PropositionRecord,
    ) -> Tuple[float, str]:
        """
        Score how well a new claim matches a candidate proposition.
        
        Returns (score, reason_string).
        """
        reasons = []
        
        # Entity similarity
        entity_score = self._entity_similarity(entity_norm, candidate.entity_norm)
        reasons.append(f"entity={entity_score:.2f}")
        
        # Predicate match (usually exact)
        if predicate_norm == candidate.predicate_norm:
            predicate_score = 1.0
        else:
            predicate_score = 0.0
        reasons.append(f"predicate={predicate_score:.2f}")
        
        # Value similarity (type-aware, with asymmetry handling)
        value_score, value_flag = self._value_similarity(
            value_norm, value_features,
            candidate.value_norm, candidate.value_features,
        )
        reasons.append(f"value={value_score:.2f}")
        if value_flag:
            reasons.append(f"value_flag={value_flag}")
        
        # Compute info delta for additional context
        info_delta = self._value_info_delta(value_features, candidate.value_features)
        if abs(info_delta) > 0.01:
            reasons.append(f"info_delta={info_delta:+.2f}")
        
        # Weighted sum
        total = (
            self.WEIGHT_ENTITY * entity_score +
            self.WEIGHT_PREDICATE * predicate_score +
            self.WEIGHT_VALUE * value_score
        )
        
        # Collision budget penalty (router-native telemetry)
        if candidate.is_overbinding:
            total -= 0.12
            reasons.append("overbind_penalty")
        elif candidate.recent_bind_rate > 0.5:
            total -= 0.08
            reasons.append("high_bind_rate")
        
        # Additional penalty for repeated low-quality binds
        # If this candidate has accumulated info_delta binds, be more suspicious
        if len(candidate.unique_value_signatures) > 2 and info_delta > 0:
            total -= 0.05
            reasons.append("accumulated_drift")
        
        return max(0.0, total), "; ".join(reasons)
    
    def _date_specificity(self, vf: Dict[str, Any]) -> int:
        """
        Compute date specificity level.
        
        0 = year-only
        1 = month
        2 = day (future)
        """
        if vf.get("day"):
            return 2
        if vf.get("month"):
            return 1
        return 0
    
    def _entity_similarity(self, e1: str, e2: str) -> float:
        """Compute entity similarity (token Jaccard)."""
        if e1 == e2:
            return 1.0
        
        tokens1 = set(self._tokenize_for_similarity(e1))
        tokens2 = set(self._tokenize_for_similarity(e2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union)
    
    def _tokenize_for_similarity(self, text: str) -> List[str]:
        """Tokenize for similarity, filtering noise."""
        tokens = re.split(r'[_.\s]+', text.lower())
        # Drop purely numeric tokens and common junk
        junk = {'v', 'etc', 'config', 'the', 'a', 'an'}
        return [t for t in tokens if len(t) > 1 and not t.isdigit() and t not in junk]
    
    def _value_similarity(
        self,
        v1_norm: str,
        v1_features: Dict[str, Any],
        v2_norm: str,
        v2_features: Dict[str, Any],
    ) -> Tuple[float, str]:
        """
        Compute value similarity with explicit feature-aware path.
        
        CRITICAL: Never short-circuit to 1.0 on value_norm equality
        if features indicate specificity mismatch. Norm-equality is
        not information-equality.
        
        Returns (score, flag_string).
        """
        # If we have features, prefer feature-aware comparison
        if v1_features or v2_features:
            return self._value_similarity_feature_aware(
                v1_norm, v1_features or {},
                v2_norm, v2_features or {},
            )
        
        # Fallback: no features, use norm comparison
        if v1_norm == v2_norm:
            return 1.0, ""
        
        # Type mismatch check
        v1_is_date = v1_norm.startswith("YEAR:")
        v2_is_date = v2_norm.startswith("YEAR:")
        v1_is_num = v1_norm.startswith("NUM:")
        v2_is_num = v2_norm.startswith("NUM:")
        
        if (v1_is_date != v2_is_date) or (v1_is_num != v2_is_num):
            return 0.0, "type_mismatch"
        
        # Nominal string similarity
        return self._token_jaccard(v1_norm, v2_norm), ""
    
    def _value_similarity_feature_aware(
        self,
        v1_norm: str,
        f1: Dict[str, Any],
        v2_norm: str,
        f2: Dict[str, Any],
    ) -> Tuple[float, str]:
        """
        Feature-aware value similarity.
        
        v1/f1 = incoming (new claim)
        v2/f2 = candidate (existing)
        """
        # Identify type from features (more reliable than norm prefixes)
        is_date = ("year" in f1 or "year" in f2)
        is_num = ("number" in f1 or "number" in f2)
        
        # DATE comparison
        if is_date:
            y1, y2 = f1.get("year"), f2.get("year")
            
            # Year mismatch is fatal
            if y1 is not None and y2 is not None and y1 != y2:
                return 0.0, "year_mismatch"
            
            m1, m2 = f1.get("month"), f2.get("month")
            mod1, mod2 = f1.get("date_modifier"), f2.get("date_modifier")
            
            # Base: same year -> strong match
            score = 0.90
            flag = ""
            
            # Month comparison
            if m1 == m2 and m1 is not None:
                score += 0.10  # 1.00 - identical months
            elif m1 is None and m2 is None:
                score += 0.05  # 0.95 - both year-only
            elif m1 is not None and m2 is None:
                # Incoming has month, candidate doesn't = INFO GAIN
                score -= 0.35  # 0.55 - force arbitration
                flag = "info_gain"
            elif m1 is None and m2 is not None:
                # Incoming lacks month, candidate has it = INFO LOSS
                score -= 0.02  # 0.88 - acceptable
                flag = "info_loss"
            elif m1 and m2 and m1 != m2:
                score -= 0.25  # 0.65 - different months
                flag = "month_mismatch"
            
            # Modifier comparison (late, early, mid, approx)
            if mod1 == mod2:
                if mod1 is not None:
                    score += 0.02
            elif mod1 is not None and mod2 is None:
                # Adding modifier = slight info gain
                score -= 0.08
                if not flag:
                    flag = "modifier_added"
            elif mod1 is None and mod2 is not None:
                # Removing modifier = slight info loss
                score -= 0.04
                if not flag:
                    flag = "modifier_removed"
            elif mod1 and mod2 and mod1 != mod2:
                score -= 0.15
                if not flag:
                    flag = "modifier_changed"
            
            return max(0.0, min(1.0, score)), flag
        
        # NUMERIC comparison
        if is_num:
            u1, u2 = f1.get("unit", ""), f2.get("unit", "")
            
            # Unit mismatch is fatal
            if u1 and u2 and u1 != u2:
                return 0.0, "unit_mismatch"
            
            n1, n2 = f1.get("number"), f2.get("number")
            
            if n1 is not None and n2 is not None:
                if n1 == 0:
                    if n2 == 0:
                        score = 1.0
                    else:
                        return 0.0, "zero_vs_nonzero"
                else:
                    rel = abs(n2 - n1) / abs(n1)
                    if rel < 0.01:
                        score = 0.95
                    elif rel < 0.10:
                        score = 0.80
                    elif rel < 0.20:
                        score = 0.60
                    else:
                        return 0.0, "numeric_drift"
                
                # Check modifier asymmetry
                vm1, vm2 = f1.get("value_modifier"), f2.get("value_modifier")
                flag = ""
                
                if vm1 == vm2:
                    pass  # No change
                elif vm1 is not None and vm2 is None:
                    score -= 0.15
                    flag = "modifier_added"
                elif vm1 is None and vm2 is not None:
                    score -= 0.10
                    flag = "modifier_removed"
                elif vm1 and vm2 and vm1 != vm2:
                    score -= 0.20
                    flag = "modifier_changed"
                
                return max(0.0, score), flag
            
            # Numbers missing, fall back to token overlap
            return self._token_jaccard(v1_norm, v2_norm), ""
        
        # NOMINAL: if norm equal but features exist, slight penalty
        if v1_norm == v2_norm:
            return 0.98, ""
        
        return self._token_jaccard(v1_norm, v2_norm), ""
    
    def _token_jaccard(self, a: str, b: str) -> float:
        """Token-level Jaccard similarity."""
        t1 = set(a.lower().split())
        t2 = set(b.lower().split())
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / len(t1 | t2)
    
    def _value_info_delta(
        self,
        f_new: Dict[str, Any],
        f_old: Dict[str, Any],
    ) -> float:
        """
        Compute information delta between new and old features.
        
        Positive = info gain (more specific) - SUSPICIOUS
        Negative = info loss (less specific) - usually OK
        """
        delta = 0.0
        
        # Specificity-adding keys
        for key in ("month", "date_modifier", "day"):
            if f_old.get(key) is None and f_new.get(key) is not None:
                delta += 0.10  # Info gain
            if f_old.get(key) is not None and f_new.get(key) is None:
                delta -= 0.12  # Info loss
        
        # Value modifiers (at least, approximately, etc.)
        if f_old.get("value_modifier") is None and f_new.get("value_modifier") is not None:
            delta += 0.05
        if f_old.get("value_modifier") is not None and f_new.get("value_modifier") is None:
            delta -= 0.06
        
        return delta


# =============================================================================
# Lowercase Entity Detection (MVP)
# =============================================================================

class EntityDetector:
    """
    Detects entities including lowercase patterns common in ops/dev text.
    
    Handles:
    - snake_case / kebab-case
    - dotted.identifiers
    - paths /etc/...
    - hostnames foo.bar
    - versions python3.11, v1.2.3
    - config keys net.ipv4.ip_forward
    """
    
    # Patterns for lowercase entities
    LOWERCASE_PATTERNS = [
        # snake_case identifiers (at least 2 segments)
        re.compile(r'\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b'),
        
        # kebab-case identifiers
        re.compile(r'\b[a-z][a-z0-9]*(?:-[a-z0-9]+)+\b'),
        
        # dotted identifiers (config keys, packages)
        re.compile(r'\b[a-z][a-z0-9]*(?:\.[a-z0-9]+)+\b'),
        
        # Paths
        re.compile(r'(?:/[a-z0-9_.-]+)+'),
        
        # Version strings
        re.compile(r'\b(?:v?\d+\.\d+(?:\.\d+)?)\b'),
        
        # Package@version
        re.compile(r'\b[a-z][a-z0-9-]*@\d+\.\d+(?:\.\d+)?\b'),
    ]
    
    @classmethod
    def extract_entities(cls, text: str) -> List[Tuple[str, int, int]]:
        """
        Extract entities from text, including lowercase patterns.
        
        Returns list of (entity, start, end).
        """
        entities = []
        
        # Standard capitalized entities
        for match in re.finditer(r'\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b', text):
            entities.append((match.group(), match.start(), match.end()))
        
        # Lowercase patterns
        for pattern in cls.LOWERCASE_PATTERNS:
            for match in pattern.finditer(text):
                # Avoid duplicates
                overlap = False
                for existing, start, end in entities:
                    if not (match.end() <= start or match.start() >= end):
                        overlap = True
                        break
                if not overlap:
                    entities.append((match.group(), match.start(), match.end()))
        
        return sorted(entities, key=lambda x: x[1])
    
    @classmethod
    def is_entity_candidate(cls, text: str) -> bool:
        """Check if text looks like an entity."""
        # Capitalized
        if re.match(r'^[A-Z]', text):
            return True
        
        # Matches any lowercase pattern
        for pattern in cls.LOWERCASE_PATTERNS:
            if pattern.fullmatch(text):
                return True
        
        return False


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=== Proposition Router Demo ===\n")
    
    router = PropositionRouter()
    
    # First claim: year-only
    print("--- Testing Date Asymmetry ---")
    result1 = router.bind_or_mint(
        prop_hash="abc123",
        entity_norm="python_3.11",
        predicate_norm="RELEASE",
        value_norm="YEAR:2022",
        value_features={"year": 2022},  # Year only - no month
    )
    print(f"1. Year-only claim: {result1.action.name}, prop_id={result1.prop_id}")
    
    # Same hash (fast path)
    result2 = router.bind_or_mint(
        prop_hash="abc123",
        entity_norm="python_3.11",
        predicate_norm="RELEASE",
        value_norm="YEAR:2022",
    )
    print(f"2. Same hash: {result2.action.name}, prop_id={result2.prop_id}")
    
    # CRITICAL TEST: More specific (info gain - should ARBITRATE)
    result3 = router.bind_or_mint(
        prop_hash="def456",
        entity_norm="python_3.11",
        predicate_norm="RELEASE",
        value_norm="YEAR:2022",
        value_features={"year": 2022, "month": "october"},  # Now has month!
    )
    print(f"3. Info gain (year→month): {result3.action.name}")
    if result3.needs_arbitration:
        print(f"   GOOD: Arbitration triggered for info gain")
        print(f"   Reason: {result3.match_reason}")
        if result3.candidates:
            for cid, score, reason in result3.candidates:
                print(f"   Candidate: {cid}, score={score:.2f}, {reason}")
    else:
        print(f"   WARNING: Info gain was not caught!")
    
    # Test the reverse: less specific (info loss - should allow)
    print("\n--- Testing Info Loss (should allow) ---")
    router2 = PropositionRouter()
    
    # Start with specific
    result4 = router2.bind_or_mint(
        prop_hash="xyz001",
        entity_norm="python_3.11",
        predicate_norm="RELEASE",
        value_norm="YEAR:2022",
        value_features={"year": 2022, "month": "october"},
    )
    print(f"4. Month claim: {result4.action.name}, prop_id={result4.prop_id}")
    
    # Less specific should bind
    result5 = router2.bind_or_mint(
        prop_hash="xyz002",
        entity_norm="python_3.11",
        predicate_norm="RELEASE",
        value_norm="YEAR:2022",
        value_features={"year": 2022},  # No month - less specific
    )
    print(f"5. Info loss (month→year): {result5.action.name}")
    if result5.is_rebind:
        print(f"   GOOD: Rebind allowed for info loss")
        print(f"   Score: {result5.match_score:.2f}, reason: {result5.match_reason}")
    
    # Test split with proper metadata
    print("\n--- Testing Split with Proper Metadata ---")
    router3 = PropositionRouter()
    
    # Bind initial
    r1 = router3.bind_or_mint(
        prop_hash="split_test_1",
        entity_norm="python",
        predicate_norm="RELEASE",
        value_norm="YEAR:2022",
        value_features={"year": 2022, "month": "october"},
        value_raw="October 2022",
    )
    print(f"6. Initial: {r1.prop_id}")
    
    # Rebind with different value
    r2 = router3.bind_or_mint(
        prop_hash="split_test_2",
        entity_norm="python",
        predicate_norm="RELEASE",
        value_norm="YEAR:2022",
        value_features={"year": 2022, "month": "november"},
        value_raw="November 2022",
    )
    print(f"7. Rebind: {r2.action.name}, prop_id={r2.prop_id}")
    
    # Split - should use hash's own metadata
    new_id = router3.split("split_test_2", "Realized November != October")
    print(f"8. Split result: new_id={new_id}")
    
    # Verify the split record has correct metadata
    split_record = router3.index.get_record(new_id)
    if split_record:
        print(f"   Split record value: {split_record.value_norm}")
        hash_meta = router3.index.get_hash_meta("split_test_2")
        if hash_meta:
            print(f"   Hash meta value_raw: {hash_meta.value_raw}")
    
    # Test entity detection
    print("\n--- Entity Detection ---")
    test_text = "The config net.ipv4.ip_forward was changed. Python 3.11 requires /etc/config."
    entities = EntityDetector.extract_entities(test_text)
    print(f"Text: {test_text}")
    print(f"Entities: {entities}")
