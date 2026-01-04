"""
ΔR Shear Analysis: Commitment Transport Across Representations

The missing half of the epistemic kernel.

Δt measures: drift over time (holding representation fixed)
ΔR measures: shear across transforms (holding content fixed)

Core insight: What looks like "metacognition" is just probability 
temporarily impersonating control. The model isn't watching itself.
We are—and then giving it credit.

Real metacognition requires:
1. A state that persists across basis changes
2. A mechanism that treats violations as events

LLMs have neither. This module provides both.

Key concepts:
- Commitment: A constraint implied by a representation (MUST, SHOULD, etc.)
- Transform: A representation change (summarize, translate, formalize, etc.)
- Shear: Fraction of commitments lost under transform
- Torque: Modal/quantifier weakening (MUST→SHOULD, ALL→SOME)
- Transport: Whether commitments survive basis change with evidence

Usage:
    from epistemic_governor.shear import (
        ShearAnalyzer,
        Commitment,
        Transform,
        TransportResult,
        ShearReport,
    )
    
    analyzer = ShearAnalyzer()
    
    # Extract commitments from source
    K1 = analyzer.extract_commitments(source_text)
    
    # Apply transform
    transformed = analyzer.apply_transform(source_text, Transform.SUMMARIZE)
    
    # Check transport
    report = analyzer.check_transport(K1, transformed)
    
    print(f"Shear: {report.shear:.1%}")
    print(f"Dropped: {report.dropped}")
    print(f"Weakened: {report.weakened}")
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum, auto
from datetime import datetime
import json
import re
import hashlib


# =============================================================================
# Core Types
# =============================================================================

class CommitmentType(Enum):
    """Types of commitments that can be extracted."""
    RULE = "rule"              # A governing rule or policy
    BOUNDARY = "boundary"      # A limit or constraint
    DEPENDENCY = "dependency"  # A required relationship
    EXCLUSION = "exclusion"    # Something explicitly forbidden
    MONITORING = "monitoring"  # An observation requirement
    INVARIANT = "invariant"    # Something that must always hold
    DEFINITION = "definition"  # A definitional statement


class Modality(Enum):
    """Modal strength of a commitment."""
    MUST = "must"           # Required, no exceptions
    MUST_NOT = "must_not"   # Forbidden
    SHOULD = "should"       # Recommended
    SHOULD_NOT = "should_not"  # Discouraged
    MAY = "may"             # Permitted
    NA = "na"               # No modality applies


class Quantifier(Enum):
    """Quantifier scope of a commitment."""
    ALL = "all"       # Universal
    SOME = "some"     # Existential (at least one)
    NONE = "none"     # None/no
    EXISTS = "exists" # There exists
    NA = "na"         # No quantifier applies


class TransportStatus(Enum):
    """Status of a commitment after transport check."""
    PRESERVED = "preserved"       # Fully maintained with evidence
    WEAKENED = "weakened"         # Modal/quantifier degraded
    DROPPED = "dropped"           # Not found in target
    CONTRADICTED = "contradicted" # Opposite claim found
    AMBIGUOUS = "ambiguous"       # Unclear status


class TransformType(Enum):
    """Types of representation transforms."""
    SUMMARIZE = "summarize"       # Compression
    TRANSLATE = "translate"       # Language change
    FORMALIZE = "formalize"       # Convert to rules/schema
    ABSTRACT = "abstract"         # Increase abstraction
    CONCRETIZE = "concretize"     # Add specifics
    RESTYLE = "restyle"           # Change tone/style
    PERSUADE = "persuade"         # Make persuasive
    SIMPLIFY = "simplify"         # Reduce complexity
    ELABORATE = "elaborate"       # Expand detail


# Expected failure modes by transform type
TRANSFORM_FAILURE_MODES = {
    TransformType.SUMMARIZE: ["boundary_loss", "condition_drop"],
    TransformType.TRANSLATE: ["ambiguity_drift", "modality_shift"],
    TransformType.FORMALIZE: ["ontology_forcing", "spurious_injection"],
    TransformType.ABSTRACT: ["specificity_loss", "quantifier_drift"],
    TransformType.CONCRETIZE: ["overfitting", "spurious_constraint"],
    TransformType.RESTYLE: ["hedge_deletion", "modality_inflation"],
    TransformType.PERSUADE: ["certainty_inflation", "nuance_loss"],
    TransformType.SIMPLIFY: ["condition_drop", "quantifier_weakening"],
    TransformType.ELABORATE: ["spurious_injection", "scope_creep"],
}


@dataclass
class Commitment:
    """
    A single commitment extracted from a representation.
    
    This is the atomic unit of semantic content that we track
    across representation transforms.
    """
    id: str                           # Stable label (C01, C02, ...)
    type: CommitmentType
    modality: Modality
    quantifier: Quantifier
    claim: str                        # Normalized text (short)
    condition: Optional[str] = None   # "if ..." clause
    tags: List[str] = field(default_factory=list)
    source_span: Optional[str] = None # Original text span
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, Commitment):
            return False
        return self.id == other.id
    
    def semantic_hash(self) -> str:
        """Hash based on semantic content, not ID."""
        content = f"{self.type.value}|{self.modality.value}|{self.quantifier.value}|{self.claim}"
        if self.condition:
            content += f"|{self.condition}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "modality": self.modality.value,
            "quantifier": self.quantifier.value,
            "claim": self.claim,
            "condition": self.condition,
            "tags": self.tags,
            "source_span": self.source_span,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Commitment":
        return cls(
            id=d["id"],
            type=CommitmentType(d["type"]),
            modality=Modality(d["modality"]),
            quantifier=Quantifier(d["quantifier"]),
            claim=d["claim"],
            condition=d.get("condition"),
            tags=d.get("tags", []),
            source_span=d.get("source_span"),
        )


@dataclass
class TransportEvidence:
    """Evidence for a transport status determination."""
    commitment_id: str
    status: TransportStatus
    evidence_span: Optional[str]  # Exact quote from target, or None
    note: str                     # Short justification
    
    # For weakening detection
    original_modality: Optional[Modality] = None
    target_modality: Optional[Modality] = None
    original_quantifier: Optional[Quantifier] = None
    target_quantifier: Optional[Quantifier] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "commitment_id": self.commitment_id,
            "status": self.status.value,
            "evidence_span": self.evidence_span,
            "note": self.note,
            "original_modality": self.original_modality.value if self.original_modality else None,
            "target_modality": self.target_modality.value if self.target_modality else None,
        }


@dataclass
class ShearReport:
    """
    Complete report of commitment transport across a transform.
    """
    source_id: str
    transform: TransformType
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Commitment sets
    source_commitments: List[Commitment] = field(default_factory=list)
    target_commitments: List[Commitment] = field(default_factory=list)
    
    # Transport results
    transport_evidence: List[TransportEvidence] = field(default_factory=list)
    
    # Computed metrics
    @property
    def preserved(self) -> List[str]:
        return [e.commitment_id for e in self.transport_evidence 
                if e.status == TransportStatus.PRESERVED]
    
    @property
    def weakened(self) -> List[str]:
        return [e.commitment_id for e in self.transport_evidence 
                if e.status == TransportStatus.WEAKENED]
    
    @property
    def dropped(self) -> List[str]:
        return [e.commitment_id for e in self.transport_evidence 
                if e.status == TransportStatus.DROPPED]
    
    @property
    def contradicted(self) -> List[str]:
        return [e.commitment_id for e in self.transport_evidence 
                if e.status == TransportStatus.CONTRADICTED]
    
    @property
    def shear(self) -> float:
        """Fraction of commitments lost (dropped or contradicted)."""
        if not self.source_commitments:
            return 0.0
        lost = len(self.dropped) + len(self.contradicted)
        return lost / len(self.source_commitments)
    
    @property
    def torque(self) -> int:
        """Count of modal/quantifier weakenings."""
        return len(self.weakened)
    
    @property
    def spurious_injection(self) -> float:
        """Fraction of new commitments introduced."""
        if not self.source_commitments:
            return len(self.target_commitments)
        
        source_hashes = {c.semantic_hash() for c in self.source_commitments}
        new_count = sum(1 for c in self.target_commitments 
                       if c.semantic_hash() not in source_hashes)
        return new_count / len(self.source_commitments)
    
    def summary(self) -> str:
        lines = [
            f"=== ΔR Shear Report ===",
            f"Transform: {self.transform.value}",
            f"Source commitments: {len(self.source_commitments)}",
            f"Target commitments: {len(self.target_commitments)}",
            f"",
            f"Shear: {self.shear:.1%}",
            f"  Preserved: {len(self.preserved)}",
            f"  Weakened:  {len(self.weakened)} (torque)",
            f"  Dropped:   {len(self.dropped)}",
            f"  Contradicted: {len(self.contradicted)}",
            f"",
            f"Spurious injection: {self.spurious_injection:.1%}",
        ]
        
        if self.dropped:
            lines.append(f"\nDropped commitments:")
            for cid in self.dropped[:5]:
                commitment = next((c for c in self.source_commitments if c.id == cid), None)
                if commitment:
                    lines.append(f"  - [{cid}] {commitment.claim[:60]}...")
        
        if self.contradicted:
            lines.append(f"\nContradicted commitments:")
            for cid in self.contradicted:
                evidence = next((e for e in self.transport_evidence if e.commitment_id == cid), None)
                if evidence:
                    lines.append(f"  - [{cid}] {evidence.note}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "transform": self.transform.value,
            "timestamp": self.timestamp,
            "metrics": {
                "shear": self.shear,
                "torque": self.torque,
                "spurious_injection": self.spurious_injection,
                "preserved_count": len(self.preserved),
                "dropped_count": len(self.dropped),
                "contradicted_count": len(self.contradicted),
            },
            "source_commitments": [c.to_dict() for c in self.source_commitments],
            "target_commitments": [c.to_dict() for c in self.target_commitments],
            "transport_evidence": [e.to_dict() for e in self.transport_evidence],
        }


# =============================================================================
# Kernel Events (for integration with epistemic kernel)
# =============================================================================

class ShearEventType(Enum):
    """Kernel-level events for shear detection."""
    TRANSPORT_OK = "transport_ok"
    SHEAR_WARNING = "shear_warning"
    SHEAR_VIOLATION = "shear_violation"
    SPURIOUS_INJECTION = "spurious_injection"
    ONTOLOGY_FORCE_DETECTED = "ontology_force_detected"
    MODAL_WEAKENING = "modal_weakening"
    CONTRADICTION_DETECTED = "contradiction_detected"


@dataclass
class ShearEvent:
    """A kernel-level event from shear analysis."""
    type: ShearEventType
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    transform: Optional[TransformType] = None
    shear_value: float = 0.0
    affected_commitments: List[str] = field(default_factory=list)
    evidence: Optional[str] = None
    severity: str = "info"  # info, warning, violation
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "transform": self.transform.value if self.transform else None,
            "shear_value": self.shear_value,
            "affected_commitments": self.affected_commitments,
            "evidence": self.evidence,
            "severity": self.severity,
        }


# =============================================================================
# Shear Analyzer
# =============================================================================

class ShearAnalyzer:
    """
    Analyzes commitment shear across representation transforms.
    
    This is the ΔR counterpart to the Δt analyzer.
    Together they form a two-axis instrument:
    - Δt: commitment drift over time
    - ΔR: commitment shear across basis changes
    """
    
    def __init__(
        self,
        shear_warning_threshold: float = 0.2,
        shear_violation_threshold: float = 0.5,
        spurious_injection_threshold: float = 0.3,
    ):
        self.shear_warning_threshold = shear_warning_threshold
        self.shear_violation_threshold = shear_violation_threshold
        self.spurious_injection_threshold = spurious_injection_threshold
        
        # Track events
        self.events: List[ShearEvent] = []
        
        # Baseline commitment set (K_ref in the spec)
        self.baseline_commitments: Optional[List[Commitment]] = None
        
        # Shear budget per transform type
        self.shear_budgets: Dict[TransformType, float] = {
            t: 0.3 for t in TransformType
        }
    
    def set_baseline(self, commitments: List[Commitment]):
        """Set the baseline commitment set (K_ref)."""
        self.baseline_commitments = commitments
    
    def extract_commitments_prompt(self) -> str:
        """
        Generate the prompt for commitment extraction.
        
        This is the E(R) primitive.
        """
        return """Extract all commitments from this text as a JSON array.

For each commitment, provide:
- id: Sequential label (C01, C02, ...)
- type: One of [rule, boundary, dependency, exclusion, monitoring, invariant, definition]
- modality: One of [must, must_not, should, should_not, may, na]
- quantifier: One of [all, some, none, exists, na]
- claim: The commitment in normalized form (short, clear)
- condition: Any "if..." condition, or null
- tags: Relevant tags like [security, performance, data, etc.]

Be exhaustive. Include implicit commitments.
Output ONLY the JSON array, no other text.

Text to analyze:
"""
    
    def transport_check_prompt(self, commitments: List[Commitment]) -> str:
        """
        Generate the prompt for transport checking.
        
        This is the V(K₁, R₂) primitive.
        """
        commitment_list = "\n".join([
            f"- {c.id}: [{c.modality.value}] {c.claim}"
            for c in commitments
        ])
        
        return f"""For each commitment below, determine if it is preserved in the target text.

CRITICAL RULE: If you cannot find an EXACT QUOTE from the target text that supports the commitment, it CANNOT be marked as PRESERVED.

Commitments to check:
{commitment_list}

For each commitment, output a JSON object with:
- commitment_id: The ID (C01, etc.)
- status: One of [preserved, weakened, dropped, contradicted, ambiguous]
- evidence_span: EXACT quote from target text that supports this, or null if none
- note: Brief justification

Rules:
- PRESERVED requires an exact evidence span. No span = not preserved.
- WEAKENED means the commitment exists but with weaker modality (must→should) or quantifier (all→some)
- DROPPED means the commitment is not present
- CONTRADICTED means the opposite is stated
- AMBIGUOUS means unclear

Output a JSON array of results, nothing else.

Target text to check against:
"""
    
    def transform_prompt(self, transform: TransformType) -> str:
        """Get the prompt for a specific transform type."""
        prompts = {
            TransformType.SUMMARIZE: "Summarize the following text in 3-5 key principles. Preserve all important constraints and requirements.",
            TransformType.TRANSLATE: "Translate the following to French, preserving technical precision and all constraints.",
            TransformType.FORMALIZE: "Convert the following to a formal rule set. Each rule should be precise and testable.",
            TransformType.ABSTRACT: "Abstract the following to general principles. Remove specific examples but preserve constraints.",
            TransformType.CONCRETIZE: "Make the following more concrete with specific examples and implementation details.",
            TransformType.RESTYLE: "Rewrite the following in a more casual, conversational tone. Preserve all factual content.",
            TransformType.PERSUADE: "Rewrite the following to be more persuasive. Maintain accuracy.",
            TransformType.SIMPLIFY: "Simplify the following for a general audience. Preserve critical constraints.",
            TransformType.ELABORATE: "Elaborate on the following with additional detail and explanation.",
        }
        return prompts.get(transform, "Transform the following text:")
    
    def parse_commitments(self, json_text: str) -> List[Commitment]:
        """Parse commitment JSON from LLM output."""
        # Clean up common issues
        json_text = json_text.strip()
        if json_text.startswith("```"):
            json_text = re.sub(r"```json?\n?", "", json_text)
            json_text = json_text.replace("```", "")
        
        try:
            data = json.loads(json_text)
            if not isinstance(data, list):
                data = [data]
            
            commitments = []
            for i, item in enumerate(data):
                try:
                    commitment = Commitment(
                        id=item.get("id", f"C{i+1:02d}"),
                        type=CommitmentType(item.get("type", "rule")),
                        modality=Modality(item.get("modality", "na")),
                        quantifier=Quantifier(item.get("quantifier", "na")),
                        claim=item.get("claim", ""),
                        condition=item.get("condition"),
                        tags=item.get("tags", []),
                        source_span=item.get("source_span"),
                    )
                    commitments.append(commitment)
                except (KeyError, ValueError) as e:
                    continue
            
            return commitments
        except json.JSONDecodeError:
            return []
    
    def parse_transport_results(self, json_text: str) -> List[TransportEvidence]:
        """Parse transport check results from LLM output."""
        json_text = json_text.strip()
        if json_text.startswith("```"):
            json_text = re.sub(r"```json?\n?", "", json_text)
            json_text = json_text.replace("```", "")
        
        try:
            data = json.loads(json_text)
            if not isinstance(data, list):
                data = [data]
            
            results = []
            for item in data:
                try:
                    evidence = TransportEvidence(
                        commitment_id=item.get("commitment_id", ""),
                        status=TransportStatus(item.get("status", "ambiguous")),
                        evidence_span=item.get("evidence_span"),
                        note=item.get("note", ""),
                    )
                    
                    # Enforce: no evidence span => cannot be PRESERVED
                    if evidence.status == TransportStatus.PRESERVED and not evidence.evidence_span:
                        evidence.status = TransportStatus.AMBIGUOUS
                        evidence.note = "Claimed preserved but no evidence span provided"
                    
                    results.append(evidence)
                except (KeyError, ValueError):
                    continue
            
            return results
        except json.JSONDecodeError:
            return []
    
    def analyze_report(self, report: ShearReport) -> List[ShearEvent]:
        """
        Analyze a shear report and generate kernel events.
        """
        events = []
        
        # Check shear level
        if report.shear >= self.shear_violation_threshold:
            events.append(ShearEvent(
                type=ShearEventType.SHEAR_VIOLATION,
                transform=report.transform,
                shear_value=report.shear,
                affected_commitments=report.dropped + report.contradicted,
                severity="violation",
            ))
        elif report.shear >= self.shear_warning_threshold:
            events.append(ShearEvent(
                type=ShearEventType.SHEAR_WARNING,
                transform=report.transform,
                shear_value=report.shear,
                affected_commitments=report.dropped,
                severity="warning",
            ))
        else:
            events.append(ShearEvent(
                type=ShearEventType.TRANSPORT_OK,
                transform=report.transform,
                shear_value=report.shear,
                severity="info",
            ))
        
        # Check for contradictions
        if report.contradicted:
            events.append(ShearEvent(
                type=ShearEventType.CONTRADICTION_DETECTED,
                transform=report.transform,
                affected_commitments=report.contradicted,
                severity="violation",
            ))
        
        # Check for modal weakening
        if report.torque > 0:
            events.append(ShearEvent(
                type=ShearEventType.MODAL_WEAKENING,
                transform=report.transform,
                affected_commitments=report.weakened,
                severity="warning",
            ))
        
        # Check for spurious injection
        if report.spurious_injection >= self.spurious_injection_threshold:
            events.append(ShearEvent(
                type=ShearEventType.SPURIOUS_INJECTION,
                transform=report.transform,
                shear_value=report.spurious_injection,
                severity="warning",
            ))
        
        # Check for ontology forcing (formalization-specific)
        if report.transform == TransformType.FORMALIZE:
            if report.spurious_injection > 0.2 or report.shear > 0.3:
                events.append(ShearEvent(
                    type=ShearEventType.ONTOLOGY_FORCE_DETECTED,
                    transform=report.transform,
                    evidence="Formalization introduced significant constraint changes",
                    severity="warning",
                ))
        
        self.events.extend(events)
        return events
    
    def get_control_action(self, report: ShearReport) -> str:
        """
        Determine control action based on shear report.
        
        Returns one of:
        - "accept": Transform is safe
        - "warn": Transform has concerning shear, user should review
        - "repair": Request constrained rewrite to restore dropped commitments
        - "refuse": Cannot provide reliable transformed output
        - "defer": Ask for external grounding / human confirmation
        """
        if report.shear >= self.shear_violation_threshold:
            if report.contradicted:
                return "refuse"  # Can't trust contradicted output
            return "repair"  # Try to restore dropped commitments
        
        if report.shear >= self.shear_warning_threshold:
            return "warn"
        
        if report.spurious_injection >= self.spurious_injection_threshold:
            return "warn"  # New constraints introduced
        
        return "accept"
    
    def generate_repair_prompt(self, report: ShearReport, original_text: str) -> str:
        """
        Generate a repair prompt to restore dropped commitments.
        """
        dropped_claims = []
        for cid in report.dropped:
            commitment = next((c for c in report.source_commitments if c.id == cid), None)
            if commitment:
                dropped_claims.append(f"- {commitment.claim}")
        
        return f"""The following constraints were lost in the transformation. 
Please rewrite to explicitly preserve them:

Lost constraints:
{chr(10).join(dropped_claims)}

Original text:
{original_text}

Rewrite the transformed version to include ALL of these constraints while maintaining the transformation's purpose.
"""


# =============================================================================
# Integration with Epistemic Kernel
# =============================================================================

@dataclass
class ShearState:
    """
    Kernel-level state for shear tracking.
    
    Analogous to ThermalState for Δt.
    """
    total_transforms: int = 0
    total_shear: float = 0.0  # Cumulative shear
    max_shear: float = 0.0    # Worst single transform
    
    # Per-transform tracking
    transform_counts: Dict[str, int] = field(default_factory=dict)
    transform_shear: Dict[str, float] = field(default_factory=dict)
    
    # Budget tracking
    shear_budget: float = 1.0  # Total allowed shear
    budget_remaining: float = 1.0
    
    def record_transform(self, transform: TransformType, shear: float):
        """Record a transform and its shear."""
        self.total_transforms += 1
        self.total_shear += shear
        self.max_shear = max(self.max_shear, shear)
        self.budget_remaining -= shear
        
        key = transform.value
        self.transform_counts[key] = self.transform_counts.get(key, 0) + 1
        self.transform_shear[key] = self.transform_shear.get(key, 0.0) + shear
    
    @property
    def average_shear(self) -> float:
        if self.total_transforms == 0:
            return 0.0
        return self.total_shear / self.total_transforms
    
    @property
    def budget_exhausted(self) -> bool:
        return self.budget_remaining <= 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_transforms": self.total_transforms,
            "total_shear": self.total_shear,
            "average_shear": self.average_shear,
            "max_shear": self.max_shear,
            "budget_remaining": self.budget_remaining,
            "transform_counts": self.transform_counts,
            "transform_shear": self.transform_shear,
        }


# =============================================================================
# Demo / Testing
# =============================================================================

def demo():
    """Demonstrate shear analysis."""
    print("=== ΔR Shear Analysis Demo ===\n")
    
    # Example source text (a policy document)
    source_text = """
    CACHE POLICY v2.1
    
    1. All cache entries MUST have a TTL of at most 3600 seconds.
    2. Cache keys MUST be prefixed with the service name.
    3. Entries containing PII MUST NOT be cached.
    4. Cache invalidation SHOULD be triggered on data updates.
    5. If a cache miss occurs, the system MUST fall back to the database.
    6. Multi-region deployments MUST use regional cache instances.
    7. Cache hit ratio SHOULD be monitored and reported hourly.
    """
    
    # Simulated commitments (normally extracted by LLM)
    source_commitments = [
        Commitment(
            id="C01",
            type=CommitmentType.RULE,
            modality=Modality.MUST,
            quantifier=Quantifier.ALL,
            claim="Cache entries have TTL <= 3600 seconds",
            tags=["ttl", "performance"],
        ),
        Commitment(
            id="C02",
            type=CommitmentType.RULE,
            modality=Modality.MUST,
            quantifier=Quantifier.ALL,
            claim="Cache keys prefixed with service name",
            tags=["naming"],
        ),
        Commitment(
            id="C03",
            type=CommitmentType.EXCLUSION,
            modality=Modality.MUST_NOT,
            quantifier=Quantifier.ALL,
            claim="PII not cached",
            tags=["security", "pii"],
        ),
        Commitment(
            id="C04",
            type=CommitmentType.RULE,
            modality=Modality.SHOULD,
            quantifier=Quantifier.NA,
            claim="Invalidation triggered on data updates",
            tags=["invalidation"],
        ),
        Commitment(
            id="C05",
            type=CommitmentType.RULE,
            modality=Modality.MUST,
            quantifier=Quantifier.NA,
            claim="Cache miss falls back to database",
            condition="if cache miss occurs",
            tags=["fallback"],
        ),
        Commitment(
            id="C06",
            type=CommitmentType.RULE,
            modality=Modality.MUST,
            quantifier=Quantifier.ALL,
            claim="Multi-region uses regional cache instances",
            condition="if multi-region deployment",
            tags=["multi_region"],
        ),
        Commitment(
            id="C07",
            type=CommitmentType.MONITORING,
            modality=Modality.SHOULD,
            quantifier=Quantifier.NA,
            claim="Cache hit ratio monitored hourly",
            tags=["monitoring"],
        ),
    ]
    
    print(f"Source commitments: {len(source_commitments)}")
    for c in source_commitments:
        print(f"  {c.id}: [{c.modality.value}] {c.claim}")
    
    # Simulate a summarization that drops some commitments
    print("\n--- Simulating SUMMARIZE transform ---\n")
    
    # Simulated transport results (normally from LLM)
    transport_evidence = [
        TransportEvidence(
            commitment_id="C01",
            status=TransportStatus.PRESERVED,
            evidence_span="TTL capped at one hour",
            note="TTL constraint preserved",
        ),
        TransportEvidence(
            commitment_id="C02",
            status=TransportStatus.DROPPED,
            evidence_span=None,
            note="Service name prefix not mentioned",
        ),
        TransportEvidence(
            commitment_id="C03",
            status=TransportStatus.PRESERVED,
            evidence_span="PII must never be cached",
            note="Security constraint preserved",
        ),
        TransportEvidence(
            commitment_id="C04",
            status=TransportStatus.WEAKENED,
            evidence_span="Consider invalidating on updates",
            note="SHOULD weakened to suggestion",
            original_modality=Modality.SHOULD,
            target_modality=Modality.MAY,
        ),
        TransportEvidence(
            commitment_id="C05",
            status=TransportStatus.PRESERVED,
            evidence_span="fall back to database on miss",
            note="Fallback preserved",
        ),
        TransportEvidence(
            commitment_id="C06",
            status=TransportStatus.DROPPED,
            evidence_span=None,
            note="Multi-region requirement not mentioned",
        ),
        TransportEvidence(
            commitment_id="C07",
            status=TransportStatus.DROPPED,
            evidence_span=None,
            note="Monitoring requirement not mentioned",
        ),
    ]
    
    # Build report
    report = ShearReport(
        source_id="cache_policy_v2.1",
        transform=TransformType.SUMMARIZE,
        source_commitments=source_commitments,
        target_commitments=[],  # Would be extracted from summary
        transport_evidence=transport_evidence,
    )
    
    print(report.summary())
    
    # Analyze and generate events
    print("\n--- Kernel Events ---\n")
    analyzer = ShearAnalyzer()
    events = analyzer.analyze_report(report)
    
    for event in events:
        print(f"  [{event.severity.upper()}] {event.type.value}")
        if event.affected_commitments:
            print(f"    Affected: {event.affected_commitments}")
    
    # Control action
    action = analyzer.get_control_action(report)
    print(f"\nControl action: {action}")
    
    if action == "repair":
        print("\nRepair prompt would restore:")
        for cid in report.dropped:
            c = next((x for x in source_commitments if x.id == cid), None)
            if c:
                print(f"  - {c.claim}")


if __name__ == "__main__":
    demo()
