"""
Character Ledger - Fictional Invariants for Narrative Continuity

Characters are not facts, but they ARE invariants:
- Identity invariants: voice, temperament, moral center, fear profile
- Trajectory invariants: what they've experienced constrains what they do next
- Scar invariants: events that permanently alter response surface
- Relational invariants: who they trust, resent, orbit

Key principle:
    Nothing fictional commits to epistemic history, but fictional history still commits.

This means:
- Characters are objects, not vibes
- Scenes are transactions, not paragraphs
- Revision is explicit, not silent
- Exploration can happen inside a scene without rewriting history

Usage:
    from epistemic_governor.character import (
        CharacterLedger,
        Character,
        Scene,
        Scar,
    )
    
    ledger = CharacterLedger()
    
    # Create character with initial traits
    sarah = ledger.create_character("sarah", {
        "voice": "sardonic, clipped sentences",
        "temperament": "guarded optimist",
        "fear": "abandonment",
    })
    
    # Record a scene (creates trajectory)
    scene = ledger.record_scene(
        characters=["sarah"],
        event="Sarah's partner left without explanation",
        consequences={"sarah": {"trust": -0.3}},
    )
    
    # Add scar (permanent alteration)
    ledger.add_scar("sarah", Scar(
        event=scene.id,
        trait_affected="trust",
        description="Flinches at goodbyes",
    ))
    
    # Check continuity before writing
    violation = ledger.check_continuity(
        character="sarah",
        proposed_action="Sarah easily trusted the stranger",
    )
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, Tuple
from datetime import datetime
from enum import Enum, auto
import hashlib


# =============================================================================
# Character Components
# =============================================================================

@dataclass
class Trait:
    """
    A character trait with current value and history.
    
    Traits can drift within bounds, but dramatic changes require
    scenes that justify them (trajectory invariant).
    """
    name: str
    value: Any  # Can be string description or float
    confidence: float = 1.0  # How established is this trait
    established_at: Optional[str] = None  # Scene ID where established
    last_modified: Optional[str] = None  # Scene ID of last change
    modification_count: int = 0
    
    def modify(self, new_value: Any, scene_id: str, cost: float = 0.1) -> float:
        """Modify trait, return deformation cost."""
        self.value = new_value
        self.last_modified = scene_id
        self.modification_count += 1
        return cost * self.modification_count  # Gets more expensive


@dataclass
class Scar:
    """
    A permanent alteration to character from a significant event.
    
    Scars are irreversible. They change the response surface forever.
    """
    id: str = ""
    event_id: str = ""  # Scene that caused it
    trait_affected: str = ""
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    severity: float = 1.0  # How much it affects behavior
    
    def __post_init__(self):
        if not self.id:
            self.id = f"scar_{hashlib.sha256(f'{self.event_id}{self.trait_affected}'.encode()).hexdigest()[:8]}"


@dataclass
class Relationship:
    """
    A relationship between two characters.
    
    Relationships are directional: A→B may differ from B→A.
    """
    source: str  # Character ID
    target: str  # Character ID
    
    # Relationship qualities
    trust: float = 0.5  # -1 to 1
    affection: float = 0.0  # -1 to 1
    respect: float = 0.5  # -1 to 1
    history: List[str] = field(default_factory=list)  # Scene IDs
    
    # Dynamics
    tension: float = 0.0  # Unresolved conflict
    debt: float = 0.0  # Narrative debt (promises, setup)
    
    def modify(self, changes: Dict[str, float], scene_id: str):
        """Apply relationship changes from a scene."""
        for attr, delta in changes.items():
            if hasattr(self, attr):
                current = getattr(self, attr)
                if isinstance(current, (int, float)):
                    setattr(self, attr, max(-1, min(1, current + delta)))
        self.history.append(scene_id)


@dataclass
class Character:
    """
    A character with identity, trajectory, scars, and relationships.
    
    Characters are invariants: they constrain what's plausible next.
    """
    id: str
    name: str
    created_at: datetime = field(default_factory=datetime.now)
    
    # Identity invariants
    traits: Dict[str, Trait] = field(default_factory=dict)
    voice: str = ""  # Writing style, speech patterns
    
    # Trajectory invariants  
    scenes: List[str] = field(default_factory=list)  # Scene IDs participated in
    arc_stage: str = "introduction"  # Where in their arc
    
    # Scar invariants
    scars: List[Scar] = field(default_factory=list)
    
    # Relational invariants
    relationships: Dict[str, Relationship] = field(default_factory=dict)
    
    # Identity stiffness (increases over time)
    stiffness: float = 0.3
    
    def get_trait(self, name: str) -> Optional[Trait]:
        return self.traits.get(name)
    
    def set_trait(self, name: str, value: Any, scene_id: str) -> float:
        """Set or modify a trait, return deformation cost."""
        if name in self.traits:
            return self.traits[name].modify(value, scene_id)
        else:
            self.traits[name] = Trait(
                name=name, 
                value=value, 
                established_at=scene_id
            )
            return 0.0
    
    def add_scar(self, scar: Scar):
        """Add a permanent scar."""
        self.scars.append(scar)
        # Scars increase stiffness
        self.stiffness = min(1.0, self.stiffness + 0.1 * scar.severity)
    
    def record_scene(self, scene_id: str):
        """Record participation in a scene."""
        self.scenes.append(scene_id)
        # Characters stiffen over time
        self.stiffness = min(1.0, self.stiffness + 0.02)
    
    def get_scar_modifiers(self, trait: str) -> List[Scar]:
        """Get all scars that affect a trait."""
        return [s for s in self.scars if s.trait_affected == trait]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "traits": {k: {"value": v.value, "confidence": v.confidence} 
                      for k, v in self.traits.items()},
            "voice": self.voice,
            "arc_stage": self.arc_stage,
            "scars": [s.description for s in self.scars],
            "stiffness": self.stiffness,
            "scene_count": len(self.scenes),
        }


# =============================================================================
# Scene and Events
# =============================================================================

@dataclass
class Scene:
    """
    A scene is a transaction in fictional history.
    
    Scenes modify characters and relationships. They can't be undone
    without explicit revision.
    """
    id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Participants
    characters: List[str] = field(default_factory=list)
    
    # What happened
    summary: str = ""
    events: List[str] = field(default_factory=list)
    
    # Consequences (applied to characters)
    trait_changes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    relationship_changes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Narrative properties
    tension_delta: float = 0.0
    debt_created: List[str] = field(default_factory=list)  # Setups requiring payoff
    debt_resolved: List[str] = field(default_factory=list)  # Payoffs delivered
    
    # Tracking
    revision_of: Optional[str] = None  # If this revises another scene
    revised_by: Optional[str] = None   # If this was revised
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "characters": self.characters,
            "summary": self.summary,
            "events": self.events,
            "trait_changes": self.trait_changes,
            "relationship_changes": self.relationship_changes,
        }


# =============================================================================
# Continuity Violations
# =============================================================================

class ViolationType(Enum):
    """Types of continuity violations."""
    TRAIT_CONTRADICTION = auto()      # Action contradicts established trait
    SCAR_IGNORED = auto()             # Action ignores permanent scar
    RELATIONSHIP_IMPOSSIBLE = auto()  # Relationship action not plausible
    VOICE_BREAK = auto()              # Dialogue doesn't match voice
    ARC_REGRESSION = auto()           # Character regresses without justification
    DEBT_ABANDONED = auto()           # Setup was never paid off
    TIMELINE_ERROR = auto()           # Events out of order


@dataclass
class ContinuityViolation:
    """
    A detected violation of fictional invariants.
    
    Violations don't block creative exploration, but they
    should be logged and can increase narrative boredom.
    """
    violation_type: ViolationType
    character_id: str
    description: str
    severity: float  # 0-1, how bad is this
    scene_id: Optional[str] = None
    conflicting_evidence: List[str] = field(default_factory=list)
    
    # Suggestions
    revision_hint: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.violation_type.name,
            "character": self.character_id,
            "description": self.description,
            "severity": self.severity,
            "hint": self.revision_hint,
        }


# =============================================================================
# Character Ledger
# =============================================================================

class CharacterLedger:
    """
    Ledger for fictional state.
    
    Tracks characters, scenes, relationships with invariant checking.
    Nothing here commits to epistemic history, but fictional history
    still commits - characters remember what happened to them.
    """
    
    def __init__(self):
        self.characters: Dict[str, Character] = {}
        self.scenes: Dict[str, Scene] = {}
        self.scene_order: List[str] = []  # Chronological order
        
        # Narrative debt tracking
        self.open_debts: Dict[str, str] = {}  # debt_id -> scene_id that created it
        self.resolved_debts: Set[str] = set()
        
        # Metrics
        self.total_violations: int = 0
        self.total_revisions: int = 0
    
    def create_character(
        self,
        id: str,
        name: str,
        traits: Optional[Dict[str, Any]] = None,
        voice: str = "",
    ) -> Character:
        """Create a new character."""
        char = Character(id=id, name=name, voice=voice)
        
        if traits:
            for trait_name, trait_value in traits.items():
                char.traits[trait_name] = Trait(
                    name=trait_name,
                    value=trait_value,
                    established_at="creation",
                )
        
        self.characters[id] = char
        return char
    
    def get_character(self, id: str) -> Optional[Character]:
        """Get character by ID."""
        return self.characters.get(id)
    
    def record_scene(
        self,
        characters: List[str],
        summary: str,
        events: Optional[List[str]] = None,
        trait_changes: Optional[Dict[str, Dict[str, Any]]] = None,
        relationship_changes: Optional[Dict[str, Dict[str, float]]] = None,
        debt_created: Optional[List[str]] = None,
        debt_resolved: Optional[List[str]] = None,
    ) -> Tuple[Scene, List[ContinuityViolation]]:
        """
        Record a scene and apply its consequences.
        
        Returns the scene and any continuity violations detected.
        """
        scene_id = f"scene_{len(self.scenes):04d}"
        
        scene = Scene(
            id=scene_id,
            characters=characters,
            summary=summary,
            events=events or [],
            trait_changes=trait_changes or {},
            relationship_changes=relationship_changes or {},
            debt_created=debt_created or [],
            debt_resolved=debt_resolved or [],
        )
        
        violations = []
        
        # Apply trait changes
        for char_id, changes in (trait_changes or {}).items():
            char = self.characters.get(char_id)
            if char:
                for trait_name, new_value in changes.items():
                    # Check for violations before applying
                    v = self._check_trait_change(char, trait_name, new_value, scene_id)
                    if v:
                        violations.append(v)
                    
                    char.set_trait(trait_name, new_value, scene_id)
                char.record_scene(scene_id)
        
        # Apply relationship changes
        for rel_key, changes in (relationship_changes or {}).items():
            # rel_key format: "source->target"
            if "->" in rel_key:
                source, target = rel_key.split("->")
                char = self.characters.get(source)
                if char:
                    if target not in char.relationships:
                        char.relationships[target] = Relationship(source=source, target=target)
                    char.relationships[target].modify(changes, scene_id)
        
        # Track debt
        for debt in (debt_created or []):
            self.open_debts[debt] = scene_id
        for debt in (debt_resolved or []):
            if debt in self.open_debts:
                del self.open_debts[debt]
                self.resolved_debts.add(debt)
        
        # Store scene
        self.scenes[scene_id] = scene
        self.scene_order.append(scene_id)
        self.total_violations += len(violations)
        
        return scene, violations
    
    def _check_trait_change(
        self,
        char: Character,
        trait_name: str,
        new_value: Any,
        scene_id: str,
    ) -> Optional[ContinuityViolation]:
        """Check if a trait change violates continuity."""
        existing = char.get_trait(trait_name)
        if not existing:
            return None
        
        # Check scars
        scar_mods = char.get_scar_modifiers(trait_name)
        for scar in scar_mods:
            # Scar implies the trait is locked or strongly constrained
            return ContinuityViolation(
                violation_type=ViolationType.SCAR_IGNORED,
                character_id=char.id,
                description=f"Trait '{trait_name}' is scarred: {scar.description}",
                severity=scar.severity * 0.5,
                scene_id=scene_id,
                conflicting_evidence=[scar.event_id],
                revision_hint=f"Consider how {scar.description} affects this change",
            )
        
        # Check stiffness
        if char.stiffness > 0.7 and existing.modification_count > 3:
            return ContinuityViolation(
                violation_type=ViolationType.TRAIT_CONTRADICTION,
                character_id=char.id,
                description=f"Trait '{trait_name}' is well-established and difficult to change",
                severity=char.stiffness * 0.3,
                scene_id=scene_id,
                revision_hint=f"This character is resistant to change at this point",
            )
        
        return None
    
    def add_scar(self, character_id: str, scar: Scar) -> bool:
        """Add a scar to a character."""
        char = self.characters.get(character_id)
        if not char:
            return False
        
        char.add_scar(scar)
        return True
    
    def check_action(
        self,
        character_id: str,
        proposed_action: str,
        affected_traits: Optional[List[str]] = None,
    ) -> List[ContinuityViolation]:
        """
        Check if a proposed action violates character continuity.
        
        Returns list of violations (empty if action is consistent).
        """
        char = self.characters.get(character_id)
        if not char:
            return []
        
        violations = []
        
        # Check each affected trait
        for trait_name in (affected_traits or []):
            trait = char.get_trait(trait_name)
            if not trait:
                continue
            
            # Check scars
            for scar in char.get_scar_modifiers(trait_name):
                violations.append(ContinuityViolation(
                    violation_type=ViolationType.SCAR_IGNORED,
                    character_id=character_id,
                    description=f"Action may conflict with scar: {scar.description}",
                    severity=scar.severity * 0.5,
                    conflicting_evidence=[scar.event_id],
                    revision_hint=f"Consider how '{scar.description}' shapes response",
                ))
        
        return violations
    
    def get_open_debts(self) -> Dict[str, str]:
        """Get all unresolved narrative debts (setups without payoffs)."""
        return dict(self.open_debts)
    
    def get_character_summary(self, character_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of character state for prompting."""
        char = self.characters.get(character_id)
        if not char:
            return None
        
        return {
            **char.to_dict(),
            "relationships": {
                target: {
                    "trust": rel.trust,
                    "affection": rel.affection,
                    "tension": rel.tension,
                }
                for target, rel in char.relationships.items()
            },
            "recent_scenes": char.scenes[-5:] if char.scenes else [],
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ledger statistics."""
        return {
            "characters": len(self.characters),
            "scenes": len(self.scenes),
            "total_violations": self.total_violations,
            "total_revisions": self.total_revisions,
            "open_debts": len(self.open_debts),
            "resolved_debts": len(self.resolved_debts),
        }


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=== Character Ledger Demo ===\n")
    
    ledger = CharacterLedger()
    
    # Create characters
    print("1. Creating characters")
    sarah = ledger.create_character(
        "sarah", "Sarah Chen",
        traits={
            "temperament": "guarded optimist",
            "trust": "cautious but willing",
            "humor": "dry, self-deprecating",
        },
        voice="Short sentences. Deflects with jokes. Rarely says what she means directly.",
    )
    print(f"   Created: {sarah.name}")
    print(f"   Traits: {list(sarah.traits.keys())}")
    print(f"   Stiffness: {sarah.stiffness}")
    
    marcus = ledger.create_character(
        "marcus", "Marcus Webb",
        traits={
            "temperament": "earnest idealist",
            "trust": "too trusting",
            "fear": "irrelevance",
        },
        voice="Speaks in paragraphs. Over-explains. Apologizes preemptively.",
    )
    print(f"   Created: {marcus.name}")
    
    # Record a scene
    print("\n2. Recording scene")
    scene1, violations = ledger.record_scene(
        characters=["sarah", "marcus"],
        summary="Sarah and Marcus meet at a conference. He overshares; she deflects.",
        events=[
            "Marcus approaches Sarah at the coffee station",
            "He launches into his life story unprompted",
            "Sarah makes a joke and excuses herself",
        ],
        relationship_changes={
            "sarah->marcus": {"trust": 0.1, "respect": -0.1},
            "marcus->sarah": {"affection": 0.2, "tension": 0.1},
        },
        debt_created=["sarah_real_reaction"],  # Setup: we need to see her real feelings
    )
    print(f"   Scene recorded: {scene1.id}")
    print(f"   Violations: {len(violations)}")
    print(f"   Open debts: {ledger.get_open_debts()}")
    
    # Add a scar
    print("\n3. Adding scar")
    scene2, _ = ledger.record_scene(
        characters=["sarah"],
        summary="Sarah's business partner embezzles and disappears",
        events=["Sarah discovers the theft", "Partner is unreachable", "Sarah covers the loss alone"],
        trait_changes={"sarah": {"trust": "deeply damaged"}},
    )
    
    scar = Scar(
        event_id=scene2.id,
        trait_affected="trust",
        description="Flinches at financial vulnerability. Counts exits.",
        severity=0.8,
    )
    ledger.add_scar("sarah", scar)
    print(f"   Scar added: {scar.description}")
    print(f"   Sarah's stiffness now: {sarah.stiffness}")
    
    # Check continuity before writing
    print("\n4. Checking proposed action")
    violations = ledger.check_action(
        "sarah",
        "Sarah immediately trusts Marcus with her business plan",
        affected_traits=["trust"],
    )
    if violations:
        print(f"   ⚠ Violations detected:")
        for v in violations:
            print(f"     - {v.violation_type.name}: {v.description}")
            print(f"       Hint: {v.revision_hint}")
    else:
        print("   ✓ No violations")
    
    # Character summary for prompting
    print("\n5. Character summary (for LLM prompt)")
    summary = ledger.get_character_summary("sarah")
    print(f"   Name: {summary['name']}")
    print(f"   Voice: {summary['voice']}")
    print(f"   Scars: {summary['scars']}")
    print(f"   Stiffness: {summary['stiffness']}")
    print(f"   Relationships: {list(summary['relationships'].keys())}")
    
    # Stats
    print("\n6. Ledger stats")
    stats = ledger.get_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    print("\n✓ Character ledger working")
