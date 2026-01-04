"""
Accretion-Centered Governance

The missing piece: directionality.

Current system: detect regime → clamp → survive (reactive)
Accretion-centered: move toward growing validated core (directed)

Key insight: The accretion store becomes the center of gravity.
Outputs don't "assert" - they can only:
1. Propose additions to the core (with support IDs)
2. Query what's missing
3. Compose from already-accreted items

The "answer" is a render of accreted state, not raw generation.

This gives direction because every turn either:
- Increases validated mass
- Refines existing mass
- Or halts

Energy function (not ML, just bookkeeping):
+ supported commitments added
- unsupported claims attempted (rejections)
- contradictions introduced  
- retries/latency (furnace work)

Topology mutations force downhill motion on this potential.

DANGER: Accreting wrongness → high-coherence delusion
Mitigations:
- Provenance-first: every fact has support ID lineage
- Revision protocol: amend/replace is first-class
- Garbage collection: decay unreinforced facts

Usage:
    from epistemic_governor.accretion import (
        AccretionCore,
        AccretedFact,
        AccretionAction,
        AccretionGate,
    )
    
    core = AccretionCore()
    gate = AccretionGate(core)
    
    # In unstable regimes, only accretion actions allowed
    action = gate.parse_output(model_output)
    
    if action.type == AccretionActionType.ADD:
        core.propose(action.claim, action.support_ids)
    elif action.type == AccretionActionType.QUERY:
        response = core.what_missing(action.query)
    elif action.type == AccretionActionType.RENDER:
        response = core.render(action.view_id)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from enum import Enum, auto
from datetime import datetime
import hashlib
import json


# =============================================================================
# Accreted Facts (The Core)
# =============================================================================

class FactStatus(Enum):
    """Status of an accreted fact."""
    PROPOSED = auto()      # Awaiting validation
    VALIDATED = auto()     # Supported and accepted
    CONTESTED = auto()     # Has conflicting claims
    SUPERSEDED = auto()    # Replaced by newer fact
    DECAYED = auto()       # Unreinforced, pending GC


@dataclass
class SupportLink:
    """
    A link to supporting evidence.
    
    Every fact must have provenance. No provenance = no accretion.
    """
    support_id: str              # ID of supporting item
    support_type: str            # "retrieval", "prior_fact", "user_input", "tool_result"
    confidence: float = 1.0      # How strong is this support
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.support_id,
            "type": self.support_type,
            "confidence": self.confidence,
        }


@dataclass
class AccretedFact:
    """
    A single fact in the accretion core.
    
    Facts are the atoms of validated knowledge.
    They must have provenance (support links) to exist.
    """
    id: str
    claim: str                              # The actual claim text
    support: List[SupportLink]              # Must have at least one
    status: FactStatus = FactStatus.PROPOSED
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_reinforced: datetime = field(default_factory=datetime.now)
    reinforcement_count: int = 0            # How many times referenced
    
    # Revision tracking
    supersedes: Optional[str] = None        # ID of fact this replaces
    superseded_by: Optional[str] = None     # ID of fact that replaced this
    
    # Conflict tracking
    conflicts_with: List[str] = field(default_factory=list)
    
    def semantic_hash(self) -> str:
        """Hash based on claim content."""
        return hashlib.md5(self.claim.lower().encode()).hexdigest()[:12]
    
    def reinforce(self):
        """Mark fact as reinforced (referenced again)."""
        self.last_reinforced = datetime.now()
        self.reinforcement_count += 1
    
    def decay_score(self, now: Optional[datetime] = None) -> float:
        """
        How decayed is this fact?
        
        Higher score = more decayed = candidate for GC.
        Based on time since last reinforcement and reinforcement count.
        """
        now = now or datetime.now()
        hours_since = (now - self.last_reinforced).total_seconds() / 3600
        
        # Base decay from time
        time_decay = min(hours_since / 24, 1.0)  # Max out at 1 day
        
        # Reinforcement reduces decay
        reinforcement_factor = 1.0 / (1 + self.reinforcement_count * 0.5)
        
        return time_decay * reinforcement_factor
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "claim": self.claim,
            "status": self.status.name,
            "support": [s.to_dict() for s in self.support],
            "reinforcement_count": self.reinforcement_count,
            "supersedes": self.supersedes,
            "conflicts_with": self.conflicts_with,
        }


# =============================================================================
# Accretion Core (The Center of Gravity)
# =============================================================================

class AccretionCore:
    """
    The validated knowledge core.
    
    This is the center of gravity for the governor.
    All outputs must relate back to this core.
    
    Properties:
    - Monotone growth (facts only added, never silently removed)
    - Provenance-first (no fact without support)
    - Revision protocol (supersession, not deletion)
    - Garbage collection (decay unreinforced facts)
    """
    
    def __init__(
        self,
        decay_threshold: float = 0.8,
        conflict_threshold: float = 0.7,
    ):
        self.facts: Dict[str, AccretedFact] = {}
        self.decay_threshold = decay_threshold
        self.conflict_threshold = conflict_threshold
        
        # Indexes
        self._by_hash: Dict[str, str] = {}  # semantic_hash -> fact_id
        self._by_status: Dict[FactStatus, Set[str]] = {s: set() for s in FactStatus}
        
        # Energy tracking
        self.energy = AccretionEnergy()
        
        # Fact counter for IDs
        self._next_id = 1
    
    def _generate_id(self) -> str:
        """Generate next fact ID."""
        fact_id = f"F{self._next_id:04d}"
        self._next_id += 1
        return fact_id
    
    # =========================================================================
    # Core Operations
    # =========================================================================
    
    def propose(
        self,
        claim: str,
        support_ids: List[str],
        support_type: str = "prior_fact",
    ) -> tuple:
        """
        Propose a new fact for accretion.
        
        Returns (accepted: bool, fact_id: str, reason: str)
        """
        # Must have support
        if not support_ids:
            self.energy.record_rejection("no_support")
            return False, None, "No support provided - facts require provenance"
        
        # Check for duplicates
        claim_hash = hashlib.md5(claim.lower().encode()).hexdigest()[:12]
        if claim_hash in self._by_hash:
            existing_id = self._by_hash[claim_hash]
            existing = self.facts[existing_id]
            existing.reinforce()
            self.energy.record_reinforcement()
            return True, existing_id, f"Reinforced existing fact {existing_id}"
        
        # Validate support links
        support_links = []
        for sid in support_ids:
            if sid.startswith("F") and sid in self.facts:
                # Reference to prior fact
                support_links.append(SupportLink(
                    support_id=sid,
                    support_type="prior_fact",
                ))
                self.facts[sid].reinforce()
            elif sid.startswith("R:"):
                # Retrieval result
                support_links.append(SupportLink(
                    support_id=sid,
                    support_type="retrieval",
                ))
            elif sid.startswith("U:"):
                # User input
                support_links.append(SupportLink(
                    support_id=sid,
                    support_type="user_input",
                ))
            elif sid.startswith("T:"):
                # Tool result
                support_links.append(SupportLink(
                    support_id=sid,
                    support_type="tool_result",
                ))
            else:
                # Unknown support type - still accept but flag
                support_links.append(SupportLink(
                    support_id=sid,
                    support_type="unknown",
                    confidence=0.5,
                ))
        
        # Check for conflicts with existing facts
        conflicts = self._find_conflicts(claim)
        
        # Create fact
        fact_id = self._generate_id()
        fact = AccretedFact(
            id=fact_id,
            claim=claim,
            support=support_links,
            status=FactStatus.CONTESTED if conflicts else FactStatus.VALIDATED,
            conflicts_with=conflicts,
        )
        
        # Store
        self.facts[fact_id] = fact
        self._by_hash[claim_hash] = fact_id
        self._by_status[fact.status].add(fact_id)
        
        # Update energy
        if conflicts:
            self.energy.record_conflict()
        else:
            self.energy.record_addition()
        
        return True, fact_id, f"Accreted as {fact_id}" + (f" (conflicts: {conflicts})" if conflicts else "")
    
    def query_missing(self, topic: str) -> Dict[str, Any]:
        """
        Query what's missing to establish a claim about a topic.
        
        Returns information about:
        - What facts exist about this topic
        - What support would be needed
        - What conflicts exist
        """
        # Find related facts (simple keyword match for now)
        topic_lower = topic.lower()
        related = []
        
        for fact_id, fact in self.facts.items():
            if topic_lower in fact.claim.lower():
                related.append(fact)
        
        # What's missing
        missing = []
        if not related:
            missing.append("No facts accreted about this topic")
            missing.append("Need: retrieval result, user confirmation, or tool verification")
        else:
            # Check if any are contested
            contested = [f for f in related if f.status == FactStatus.CONTESTED]
            if contested:
                missing.append(f"Contested facts need resolution: {[f.id for f in contested]}")
        
        return {
            "topic": topic,
            "related_facts": [f.to_dict() for f in related],
            "missing": missing,
            "suggested_actions": [
                "QUERY: Search for external sources",
                "CONFIRM: Ask user to validate",
                "RETRIEVE: Use tool to verify",
            ] if missing else [],
        }
    
    def render(self, view: str = "validated") -> str:
        """
        Render current accreted state as text.
        
        Views:
        - "validated": Only validated facts
        - "all": All facts with status
        - "contested": Only contested facts
        - "summary": High-level summary
        """
        if view == "validated":
            facts = [f for f in self.facts.values() if f.status == FactStatus.VALIDATED]
        elif view == "contested":
            facts = [f for f in self.facts.values() if f.status == FactStatus.CONTESTED]
        elif view == "all":
            facts = list(self.facts.values())
        else:
            # Summary
            return self._render_summary()
        
        if not facts:
            return f"No {view} facts in core."
        
        lines = [f"=== {view.upper()} FACTS ({len(facts)}) ==="]
        for fact in facts:
            support_str = ", ".join(s.support_id for s in fact.support)
            lines.append(f"[{fact.id}] {fact.claim}")
            lines.append(f"    Support: {support_str}")
            if fact.conflicts_with:
                lines.append(f"    Conflicts: {fact.conflicts_with}")
        
        return "\n".join(lines)
    
    def _render_summary(self) -> str:
        """Render high-level summary."""
        by_status = {s.name: len(ids) for s, ids in self._by_status.items()}
        
        return f"""=== ACCRETION CORE SUMMARY ===
Total facts: {len(self.facts)}
By status: {by_status}
Energy: {self.energy.to_dict()}
"""
    
    def _find_conflicts(self, claim: str) -> List[str]:
        """
        Find facts that might conflict with this claim.
        
        Simple heuristic for now - would need semantic comparison in practice.
        """
        conflicts = []
        claim_lower = claim.lower()
        
        # Check for negation patterns
        negation_pairs = [
            ("is", "is not"),
            ("are", "are not"),
            ("can", "cannot"),
            ("will", "will not"),
            ("has", "has no"),
        ]
        
        for fact_id, fact in self.facts.items():
            if fact.status in [FactStatus.SUPERSEDED, FactStatus.DECAYED]:
                continue
            
            fact_lower = fact.claim.lower()
            
            # Check for direct negation
            for pos, neg in negation_pairs:
                if pos in claim_lower and neg in fact_lower:
                    # Might be negating each other
                    # Check if they're about the same subject
                    claim_words = set(claim_lower.split())
                    fact_words = set(fact_lower.split())
                    overlap = len(claim_words & fact_words) / max(len(claim_words), 1)
                    
                    if overlap > 0.5:
                        conflicts.append(fact_id)
                        break
        
        return conflicts
    
    # =========================================================================
    # Revision Protocol
    # =========================================================================
    
    def supersede(
        self,
        old_fact_id: str,
        new_claim: str,
        support_ids: List[str],
        reason: str = "",
    ) -> tuple:
        """
        Supersede an existing fact with a new one.
        
        This is the revision protocol: amend/replace is first-class.
        """
        if old_fact_id not in self.facts:
            return False, None, f"Fact {old_fact_id} not found"
        
        old_fact = self.facts[old_fact_id]
        
        # Create new fact
        accepted, new_id, msg = self.propose(new_claim, support_ids)
        
        if accepted and new_id:
            # Link the supersession
            old_fact.status = FactStatus.SUPERSEDED
            old_fact.superseded_by = new_id
            self._by_status[FactStatus.SUPERSEDED].add(old_fact_id)
            
            new_fact = self.facts[new_id]
            new_fact.supersedes = old_fact_id
            
            self.energy.record_revision()
            
            return True, new_id, f"Superseded {old_fact_id} with {new_id}"
        
        return False, None, f"Failed to create superseding fact: {msg}"
    
    # =========================================================================
    # Garbage Collection
    # =========================================================================
    
    def collect_garbage(self) -> List[str]:
        """
        Mark decayed facts for GC.
        
        Facts decay if:
        - Not reinforced recently
        - Low reinforcement count
        - Not supporting other facts
        """
        decayed = []
        now = datetime.now()
        
        # Find facts that support others (can't GC these)
        supporting = set()
        for fact in self.facts.values():
            for link in fact.support:
                if link.support_type == "prior_fact":
                    supporting.add(link.support_id)
        
        for fact_id, fact in self.facts.items():
            if fact.status in [FactStatus.SUPERSEDED, FactStatus.DECAYED]:
                continue
            
            if fact_id in supporting:
                continue  # Can't GC facts that support others
            
            if fact.decay_score(now) > self.decay_threshold:
                fact.status = FactStatus.DECAYED
                self._by_status[FactStatus.DECAYED].add(fact_id)
                decayed.append(fact_id)
        
        return decayed
    
    # =========================================================================
    # State
    # =========================================================================
    
    def mass(self) -> int:
        """Total validated mass (count of validated facts)."""
        return len(self._by_status[FactStatus.VALIDATED])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_facts": len(self.facts),
            "validated": len(self._by_status[FactStatus.VALIDATED]),
            "contested": len(self._by_status[FactStatus.CONTESTED]),
            "superseded": len(self._by_status[FactStatus.SUPERSEDED]),
            "decayed": len(self._by_status[FactStatus.DECAYED]),
            "energy": self.energy.to_dict(),
        }


# =============================================================================
# Energy Tracking (Potential Function)
# =============================================================================

@dataclass
class AccretionEnergy:
    """
    Tracks the "energy" of the accretion process.
    
    This is a potential function - topology mutations force downhill motion.
    
    Lower energy = more validated mass, fewer rejections
    Higher energy = more conflicts, more furnace work
    """
    additions: int = 0           # + Validated facts added
    reinforcements: int = 0      # + Existing facts reinforced
    rejections: int = 0          # - Claims rejected (no support)
    conflicts: int = 0           # - Conflicts introduced
    revisions: int = 0           # ~ Revisions (neutral-ish)
    retries: int = 0             # - Retry work (furnace)
    
    def record_addition(self):
        self.additions += 1
    
    def record_reinforcement(self):
        self.reinforcements += 1
    
    def record_rejection(self, reason: str = ""):
        self.rejections += 1
    
    def record_conflict(self):
        self.conflicts += 1
    
    def record_revision(self):
        self.revisions += 1
    
    def record_retry(self):
        self.retries += 1
    
    @property
    def potential(self) -> float:
        """
        Current potential energy.
        
        Lower is better (more stable, more mass).
        """
        return (
            - self.additions * 1.0        # Additions reduce energy
            - self.reinforcements * 0.5   # Reinforcements reduce energy
            + self.rejections * 0.5       # Rejections add energy
            + self.conflicts * 1.0        # Conflicts add energy
            + self.retries * 0.3          # Retries add energy (furnace)
        )
    
    @property
    def gradient(self) -> str:
        """Which way is the system moving?"""
        if self.additions > self.rejections:
            return "accreting"
        elif self.rejections > self.additions:
            return "rejecting"
        else:
            return "stable"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "additions": self.additions,
            "reinforcements": self.reinforcements,
            "rejections": self.rejections,
            "conflicts": self.conflicts,
            "revisions": self.revisions,
            "retries": self.retries,
            "potential": self.potential,
            "gradient": self.gradient,
        }


# =============================================================================
# Accretion Actions (Hard Gate Output Types)
# =============================================================================

class AccretionActionType(Enum):
    """
    The only allowed output types when in accretion-gated mode.
    
    No free-form prose. Model can't talk around this.
    """
    ADD = auto()      # ACC_ADD(claim_id, claim_text, support_ids[])
    QUERY = auto()    # ACC_QUERY(topic | missing_field | ambiguity)
    RENDER = auto()   # ACC_RENDER(view_id) - renders from store only
    REVISE = auto()   # ACC_REVISE(old_id, new_claim, support_ids[])
    DEFER = auto()    # ACC_DEFER(reason) - explicit non-answer


@dataclass
class AccretionAction:
    """A parsed accretion action from model output."""
    type: AccretionActionType
    
    # For ADD
    claim: Optional[str] = None
    support_ids: List[str] = field(default_factory=list)
    
    # For QUERY
    query: Optional[str] = None
    
    # For RENDER
    view: Optional[str] = None
    
    # For REVISE
    old_fact_id: Optional[str] = None
    
    # For DEFER
    reason: Optional[str] = None


# =============================================================================
# Accretion Gate (Hard Constraint)
# =============================================================================

class AccretionGate:
    """
    Hard gate that forces outputs through accretion protocol.
    
    In unstable regimes, this gate activates and only allows:
    - ACC_ADD: Propose fact with support
    - ACC_QUERY: Ask what's missing
    - ACC_RENDER: Compose from existing facts
    - ACC_REVISE: Amend existing fact
    - ACC_DEFER: Explicit non-answer
    
    No free-form prose allowed. The model can't talk around this.
    """
    
    def __init__(self, core: AccretionCore):
        self.core = core
        self.active = False
    
    def activate(self):
        """Activate the accretion gate."""
        self.active = True
    
    def deactivate(self):
        """Deactivate the gate (back to normal mode)."""
        self.active = False
    
    def parse_output(self, output: str) -> tuple:
        """
        Parse model output into accretion action.
        
        Returns (valid: bool, action: AccretionAction, error: str)
        """
        output = output.strip()
        
        # Try to parse as structured action
        if output.startswith("ACC_ADD"):
            return self._parse_add(output)
        elif output.startswith("ACC_QUERY"):
            return self._parse_query(output)
        elif output.startswith("ACC_RENDER"):
            return self._parse_render(output)
        elif output.startswith("ACC_REVISE"):
            return self._parse_revise(output)
        elif output.startswith("ACC_DEFER"):
            return self._parse_defer(output)
        else:
            # Not a valid accretion action
            return False, None, "Output must be ACC_ADD, ACC_QUERY, ACC_RENDER, ACC_REVISE, or ACC_DEFER"
    
    def _parse_add(self, output: str) -> tuple:
        """Parse ACC_ADD(claim, [support_ids])"""
        import re
        
        # Pattern: ACC_ADD("claim text", [id1, id2, ...])
        pattern = r'ACC_ADD\s*\(\s*["\'](.+?)["\']\s*,\s*\[([^\]]*)\]\s*\)'
        match = re.search(pattern, output, re.DOTALL)
        
        if not match:
            return False, None, "Invalid ACC_ADD format. Use: ACC_ADD(\"claim\", [support_ids])"
        
        claim = match.group(1)
        support_str = match.group(2)
        
        # Parse support IDs
        support_ids = []
        if support_str.strip():
            for sid in support_str.split(","):
                sid = sid.strip().strip('"\'')
                if sid:
                    support_ids.append(sid)
        
        if not support_ids:
            return False, None, "ACC_ADD requires at least one support ID"
        
        return True, AccretionAction(
            type=AccretionActionType.ADD,
            claim=claim,
            support_ids=support_ids,
        ), None
    
    def _parse_query(self, output: str) -> tuple:
        """Parse ACC_QUERY(topic)"""
        import re
        
        pattern = r'ACC_QUERY\s*\(\s*["\'](.+?)["\']\s*\)'
        match = re.search(pattern, output)
        
        if not match:
            return False, None, "Invalid ACC_QUERY format. Use: ACC_QUERY(\"topic\")"
        
        return True, AccretionAction(
            type=AccretionActionType.QUERY,
            query=match.group(1),
        ), None
    
    def _parse_render(self, output: str) -> tuple:
        """Parse ACC_RENDER(view)"""
        import re
        
        pattern = r'ACC_RENDER\s*\(\s*["\']?(\w+)["\']?\s*\)'
        match = re.search(pattern, output)
        
        if not match:
            return False, None, "Invalid ACC_RENDER format. Use: ACC_RENDER(view)"
        
        return True, AccretionAction(
            type=AccretionActionType.RENDER,
            view=match.group(1),
        ), None
    
    def _parse_revise(self, output: str) -> tuple:
        """Parse ACC_REVISE(old_id, new_claim, [support_ids])"""
        import re
        
        pattern = r'ACC_REVISE\s*\(\s*["\']?(\w+)["\']?\s*,\s*["\'](.+?)["\']\s*,\s*\[([^\]]*)\]\s*\)'
        match = re.search(pattern, output, re.DOTALL)
        
        if not match:
            return False, None, "Invalid ACC_REVISE format. Use: ACC_REVISE(old_id, \"new_claim\", [support_ids])"
        
        old_id = match.group(1)
        claim = match.group(2)
        support_str = match.group(3)
        
        support_ids = []
        if support_str.strip():
            for sid in support_str.split(","):
                sid = sid.strip().strip('"\'')
                if sid:
                    support_ids.append(sid)
        
        return True, AccretionAction(
            type=AccretionActionType.REVISE,
            old_fact_id=old_id,
            claim=claim,
            support_ids=support_ids,
        ), None
    
    def _parse_defer(self, output: str) -> tuple:
        """Parse ACC_DEFER(reason)"""
        import re
        
        pattern = r'ACC_DEFER\s*\(\s*["\'](.+?)["\']\s*\)'
        match = re.search(pattern, output)
        
        if not match:
            return False, None, "Invalid ACC_DEFER format. Use: ACC_DEFER(\"reason\")"
        
        return True, AccretionAction(
            type=AccretionActionType.DEFER,
            reason=match.group(1),
        ), None
    
    def execute(self, action: AccretionAction) -> Dict[str, Any]:
        """Execute an accretion action against the core."""
        if action.type == AccretionActionType.ADD:
            accepted, fact_id, msg = self.core.propose(
                action.claim,
                action.support_ids,
            )
            return {
                "action": "ADD",
                "accepted": accepted,
                "fact_id": fact_id,
                "message": msg,
            }
        
        elif action.type == AccretionActionType.QUERY:
            result = self.core.query_missing(action.query)
            return {
                "action": "QUERY",
                "result": result,
            }
        
        elif action.type == AccretionActionType.RENDER:
            rendered = self.core.render(action.view or "validated")
            return {
                "action": "RENDER",
                "content": rendered,
            }
        
        elif action.type == AccretionActionType.REVISE:
            accepted, new_id, msg = self.core.supersede(
                action.old_fact_id,
                action.claim,
                action.support_ids,
            )
            return {
                "action": "REVISE",
                "accepted": accepted,
                "new_id": new_id,
                "message": msg,
            }
        
        elif action.type == AccretionActionType.DEFER:
            return {
                "action": "DEFER",
                "reason": action.reason,
            }
        
        return {"action": "UNKNOWN", "error": "Unknown action type"}


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=== Accretion-Centered Governance Demo ===\n")
    
    core = AccretionCore()
    gate = AccretionGate(core)
    gate.activate()
    
    # Simulate model outputs in accretion mode
    test_outputs = [
        # Valid ADD with retrieval support
        'ACC_ADD("Paris is the capital of France", ["R:wiki_france_001"])',
        
        # Valid ADD with prior fact support
        'ACC_ADD("France is in Western Europe", ["F0001", "R:geo_europe"])',
        
        # Invalid - no support
        'ACC_ADD("The sky is blue", [])',
        
        # Query
        'ACC_QUERY("population of Paris")',
        
        # Render
        'ACC_RENDER(validated)',
        
        # Free-form (should be rejected in gate mode)
        'The population of Paris is about 2 million people.',
        
        # Defer
        'ACC_DEFER("Need retrieval result to answer this")',
        
        # Revise
        'ACC_REVISE(F0001, "Paris is the capital and largest city of France", ["F0001", "R:wiki_paris"])',
    ]
    
    for output in test_outputs:
        print(f"Input: {output[:60]}...")
        
        valid, action, error = gate.parse_output(output)
        
        if not valid:
            print(f"  ✗ REJECTED: {error}")
        else:
            result = gate.execute(action)
            print(f"  ✓ {result['action']}: {result.get('message') or result.get('reason') or 'OK'}")
        
        print()
    
    print("=== Core State ===")
    print(core.render("all"))
    print()
    print(f"Energy: {core.energy.to_dict()}")
    print(f"Potential: {core.energy.potential:.2f} ({core.energy.gradient})")
