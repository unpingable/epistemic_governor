"""
V1 → V2 Bridge

Converts V1 extraction output (ClaimAtom) to V2 symbolic candidates (CandidateCommitment).

This is the handoff point where language-plane parsing meets symbolic-plane adjudication.

V1 is the ingestion plane:
- BoundaryGate
- ClaimExtractor  
- Normalizer
- PropositionRouter

V2 is the authority plane:
- Adjudicator
- SymbolicState
- Support calculus
- Three clocks

The bridge ensures V1 can call V2, but V2 never imports V1.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re

# V1 imports
from .claim_extractor import ClaimAtom, ClaimMode, Modality
from .claims import Provenance, EvidenceRef

# V2 imports
from .symbolic_substrate import (
    CandidateCommitment,
    Predicate,
    PredicateType,
    ProvenanceClass,
    SupportItem,
    TemporalScope,
)


# =============================================================================
# Mapping Tables
# =============================================================================

# Map V1 Provenance to V2 ProvenanceClass
PROVENANCE_MAP: Dict[Provenance, ProvenanceClass] = {
    Provenance.OBSERVED: ProvenanceClass.SENSOR,
    Provenance.RETRIEVED: ProvenanceClass.CITED,
    Provenance.USER_PROVIDED: ProvenanceClass.USER,
    Provenance.DERIVED: ProvenanceClass.INFERRED,
    Provenance.PEER_ASSERTED: ProvenanceClass.MODEL,
    Provenance.ASSUMED: ProvenanceClass.MODEL,
}

# =============================================================================
# Predicate Mapping (NO SILENT FALLBACKS)
# =============================================================================

# Explicit mapping table - every pattern must be intentional
PREDICATE_PATTERNS: List[Tuple[str, PredicateType]] = [
    # Temporal predicates
    (r"^(released|launched|published|created|founded|established|born|died)$", PredicateType.AT_TIME),
    (r"^(happened|occurred|started|ended|began|finished)$", PredicateType.AT_TIME),
    
    # Location predicates
    (r"^(located|lives|based|situated|resides|headquartered)$", PredicateType.LOCATED_AT),
    
    # Type predicates
    (r"^(is_a|type|kind|instance|category|class)$", PredicateType.IS_A),
    
    # Property predicates
    (r"^(has|contains|includes|owns|possesses|holds)$", PredicateType.HAS),
    
    # Identity predicates
    (r"^(same_as|equals|identical|alias|aka|known_as)$", PredicateType.SAME_AS),
    
    # Temporal relations
    (r"^(before|precedes|earlier|prior)$", PredicateType.BEFORE),
    (r"^(during|while|concurrent|simultaneous)$", PredicateType.DURING),
    
    # Causal predicates
    (r"^(causes|leads_to|results_in|triggers|produces)$", PredicateType.CAUSES),
    
    # Part-whole predicates
    (r"^(part_of|belongs_to|member_of|component|subset)$", PredicateType.PART_OF),
]


class PredicateMappingResult:
    """Result of predicate mapping - explicit success or failure."""
    
    def __init__(
        self, 
        success: bool, 
        ptype: Optional[PredicateType] = None,
        reason: str = "",
        original: str = "",
    ):
        self.success = success
        self.ptype = ptype
        self.reason = reason
        self.original = original
    
    @classmethod
    def mapped(cls, ptype: PredicateType, original: str) -> "PredicateMappingResult":
        return cls(success=True, ptype=ptype, original=original)
    
    @classmethod
    def unmapped(cls, original: str, reason: str = "NO_PATTERN_MATCH") -> "PredicateMappingResult":
        return cls(success=False, reason=reason, original=original)


def map_predicate(predicate: str) -> PredicateMappingResult:
    """
    Map a V1 predicate string to a V2 PredicateType.
    
    NO SILENT FALLBACKS. If we can't map it, we return failure.
    The caller must decide what to do (quarantine, reject, etc.)
    """
    pred_lower = predicate.lower().replace(" ", "_").strip()
    
    for pattern, ptype in PREDICATE_PATTERNS:
        if re.match(pattern, pred_lower):
            return PredicateMappingResult.mapped(ptype, predicate)
    
    # NO FALLBACK - explicit failure
    return PredicateMappingResult.unmapped(predicate, "NO_PATTERN_MATCH")


def infer_predicate_type(predicate: str) -> PredicateType:
    """
    Legacy function - maps predicate or raises.
    
    DEPRECATED: Use map_predicate() for explicit error handling.
    """
    result = map_predicate(predicate)
    if result.success:
        return result.ptype
    
    # For backwards compatibility, return HAS but log warning
    # TODO: Remove this fallback once all callers use map_predicate()
    import warnings
    warnings.warn(
        f"Unmapped predicate '{predicate}' falling back to HAS. "
        "This fallback will be removed. Use map_predicate() instead.",
        DeprecationWarning,
    )
    return PredicateType.HAS


def parse_temporal_scope(
    value_features: Dict[str, Any],
    tense: str
) -> TemporalScope:
    """
    Parse V1 value_features into V2 TemporalScope.
    
    V1 stores things like {"year": 2022, "month": "October"}
    V2 wants proper datetime bounds.
    """
    # Try to extract date components
    year = value_features.get("year")
    month = value_features.get("month")
    day = value_features.get("day")
    
    if year is not None:
        try:
            # Parse month name if string
            month_num = 1
            if isinstance(month, str):
                month_names = {
                    "january": 1, "february": 2, "march": 3, "april": 4,
                    "may": 5, "june": 6, "july": 7, "august": 8,
                    "september": 9, "october": 10, "november": 11, "december": 12,
                }
                month_num = month_names.get(month.lower(), 1)
            elif isinstance(month, int):
                month_num = month
            
            day_num = int(day) if day else 1
            
            dt = datetime(int(year), month_num, day_num)
            
            # Determine granularity
            if day:
                granularity = "instant"
            elif month:
                granularity = "interval"  # Month-level
            else:
                granularity = "interval"  # Year-level
            
            return TemporalScope(start=dt, end=dt, granularity=granularity)
        except (ValueError, TypeError):
            pass
    
    # Check tense for relative positioning
    if tense == "past":
        return TemporalScope(end=datetime.utcnow(), granularity="interval")
    elif tense == "future":
        return TemporalScope(start=datetime.utcnow(), granularity="interval")
    
    # Unknown scope - this is legal and better than hallucinating
    return TemporalScope(granularity="unknown")


def convert_evidence_ref(ref: EvidenceRef) -> SupportItem:
    """Convert V1 EvidenceRef to V2 SupportItem."""
    
    # Map ref_type to source_type
    source_type_map = {
        "TOOL_TRACE": "sensor",
        "URL": "citation",
        "HUMAN_INPUT": "user_assertion",
        "DOCUMENT": "doc_span",
    }
    source_type = source_type_map.get(ref.ref_type, "citation")
    
    return SupportItem(
        source_type=source_type,
        source_id=ref.locator,
        reliability=ref.confidence,
        span_text=ref.scope,
        timestamp=ref.retrieved_at,
    )


# =============================================================================
# Main Bridge Function
# =============================================================================

class BridgeResult:
    """Result of bridging a claim - explicit success or schema quarantine."""
    
    def __init__(
        self,
        success: bool,
        candidate: Optional[CandidateCommitment] = None,
        quarantine_reason: str = "",
        original_predicate: str = "",
    ):
        self.success = success
        self.candidate = candidate
        self.quarantine_reason = quarantine_reason
        self.original_predicate = original_predicate


def claim_atom_to_candidate(
    atom: ClaimAtom,
    provenance: Provenance = Provenance.ASSUMED,
    evidence_refs: List[EvidenceRef] = None,
    allow_fallback: bool = True,  # Set False for strict mode
) -> CandidateCommitment:
    """
    Convert a V1 ClaimAtom to a V2 CandidateCommitment.
    
    This is the core bridge function. It:
    1. Maps the predicate to a typed PredicateType
    2. Constructs proper predicate args from entities + value
    3. Converts provenance classes
    4. Parses temporal scope from value_features
    5. Converts any evidence refs to support items
    
    The resulting CandidateCommitment can be passed to V2's Adjudicator.
    
    Args:
        allow_fallback: If False, raises on unmapped predicates instead of
                       falling back to HAS. Use False for strict mode.
    """
    
    # 1. Map predicate - with explicit handling
    mapping_result = map_predicate(atom.predicate)
    
    if mapping_result.success:
        ptype = mapping_result.ptype
    elif allow_fallback:
        # Legacy fallback mode - log warning
        import warnings
        warnings.warn(
            f"Unmapped predicate '{atom.predicate}' using HAS fallback. "
            "Set allow_fallback=False for strict mode.",
            DeprecationWarning,
        )
        ptype = PredicateType.HAS
    else:
        # Strict mode - raise
        raise ValueError(
            f"Cannot map predicate '{atom.predicate}' to PredicateType. "
            f"Reason: {mapping_result.reason}"
        )
    
    # 2. Build predicate args based on predicate type
    args = build_predicate_args(ptype, atom.entities, atom.value_norm, atom.value_features)
    
    predicate = Predicate(ptype=ptype, args=tuple(args))
    
    # 3. Map provenance
    provclass = PROVENANCE_MAP.get(provenance, ProvenanceClass.MODEL)
    
    # 4. Parse temporal scope
    t_scope = parse_temporal_scope(atom.value_features, atom.tense)
    
    # 5. Convert evidence refs
    support = []
    if evidence_refs:
        support = [convert_evidence_ref(ref) for ref in evidence_refs]
    
    # 6. Map confidence, adjusting for modality
    sigma = atom.confidence
    if atom.modality == Modality.MIGHT:
        sigma *= 0.6
    elif atom.modality == Modality.SPECULATE:
        sigma *= 0.5
    elif atom.modality == Modality.INFER:
        sigma *= 0.8
    
    # Negation reduces commitment (we're less sure of negatives)
    if atom.polarity == -1:
        sigma *= 0.8
    
    # 7. Build candidate
    return CandidateCommitment(
        predicate=predicate,
        sigma=sigma,
        t_scope=t_scope,
        provclass=provclass,
        support=support,
        logical_deps=[],      # V1 doesn't track these yet
        evidentiary_deps=[],  # V1 doesn't track these yet
        source_span=atom.span,
        source_text=atom.span_quote,
        extraction_confidence=atom.confidence,
    )


def bridge_claim_safe(
    atom: ClaimAtom,
    provenance: Provenance = Provenance.ASSUMED,
    evidence_refs: List[EvidenceRef] = None,
) -> BridgeResult:
    """
    Safe bridge that returns explicit result instead of raising.
    
    Use this for strict mode where unmapped predicates should
    result in QUARANTINE_SCHEMA.
    """
    mapping_result = map_predicate(atom.predicate)
    
    if not mapping_result.success:
        return BridgeResult(
            success=False,
            quarantine_reason=f"UNMAPPED_PREDICATE:{mapping_result.reason}",
            original_predicate=atom.predicate,
        )
    
    try:
        candidate = claim_atom_to_candidate(
            atom, 
            provenance, 
            evidence_refs, 
            allow_fallback=False,
        )
        return BridgeResult(success=True, candidate=candidate)
    except ValueError as e:
        return BridgeResult(
            success=False,
            quarantine_reason=str(e),
            original_predicate=atom.predicate,
        )


def build_predicate_args(
    ptype: PredicateType,
    entities: Tuple[str, ...],
    value_norm: str,
    value_features: Dict[str, Any],
) -> List[str]:
    """
    Build predicate args appropriate for the predicate type.
    
    Different predicates have different arities:
    - HAS: (entity, property, value) - 3 args
    - IS_A: (entity, type) - 2 args
    - SAME_AS: (entity1, entity2) - 2 args
    - AT_TIME: (entity, property, value, time) - 4 args
    - LOCATED_AT: (entity, location) or (entity, location, time) - 2-3 args
    - etc.
    """
    args = list(entities)
    
    if ptype == PredicateType.AT_TIME:
        # Need 4 args: entity, property, value, time
        if len(args) == 0:
            args.append("UNKNOWN_ENTITY")
        if len(args) == 1:
            args.append("state")  # property
        if len(args) == 2:
            args.append(value_norm or "UNKNOWN")  # value
        if len(args) == 3:
            # Time from value_features
            time_str = _format_time(value_features)
            args.append(time_str)
        return args[:4]
    
    elif ptype == PredicateType.HAS:
        # Need 3 args: entity, property, value
        if len(args) == 0:
            args.append("UNKNOWN_ENTITY")
        if len(args) == 1:
            args.append("property")
        if len(args) == 2:
            args.append(value_norm or "UNKNOWN")
        return args[:3]
    
    elif ptype in {PredicateType.IS_A, PredicateType.SAME_AS, 
                   PredicateType.BEFORE, PredicateType.CAUSES,
                   PredicateType.PART_OF, PredicateType.DURING,
                   PredicateType.DEPENDS_ON, PredicateType.SUPPORTS,
                   PredicateType.CONTRADICTS}:
        # Need 2 args
        while len(args) < 2:
            if value_norm and len(args) == 1:
                args.append(value_norm)
            else:
                args.append("UNKNOWN")
        return args[:2]
    
    elif ptype == PredicateType.LOCATED_AT:
        # Need 2-3 args: entity, location, [time]
        if len(args) == 0:
            args.append("UNKNOWN_ENTITY")
        if len(args) == 1:
            args.append(value_norm or "UNKNOWN_LOCATION")
        # Optional time
        if value_features.get("year") or value_features.get("time"):
            time_str = _format_time(value_features)
            args.append(time_str)
        return args[:3]
    
    else:
        # Default: 2 args minimum
        while len(args) < 2:
            if value_norm and len(args) == 1:
                args.append(value_norm)
            else:
                args.append("UNKNOWN")
        return args[:3]


def _format_time(value_features: Dict[str, Any]) -> str:
    """Format time from value features into a string."""
    year = value_features.get("year")
    month = value_features.get("month")
    day = value_features.get("day")
    
    if year:
        if month:
            if day:
                return f"{year}-{month}-{day}"
            return f"{month} {year}"
        return str(year)
    
    return "UNKNOWN_TIME"


def bridge_claims(
    atoms: List[ClaimAtom],
    provenance: Provenance = Provenance.ASSUMED,
    evidence_map: Dict[str, List[EvidenceRef]] = None,
) -> List[CandidateCommitment]:
    """
    Bridge a batch of V1 claims to V2 candidates.
    
    Args:
        atoms: List of extracted ClaimAtoms from V1
        provenance: Default provenance for claims without explicit provenance
        evidence_map: Optional mapping from prop_hash to evidence refs
    
    Returns:
        List of CandidateCommitments ready for V2 adjudication
    """
    evidence_map = evidence_map or {}
    
    candidates = []
    for atom in atoms:
        refs = evidence_map.get(atom.prop_hash, [])
        candidate = claim_atom_to_candidate(atom, provenance, refs)
        candidates.append(candidate)
    
    return candidates


# =============================================================================
# Mode-Aware Bridging (INT-2)
# =============================================================================

def should_bridge_for_adjudication(atom: ClaimAtom) -> bool:
    """
    Determine if a claim should be bridged to V2 for adjudication.
    
    Mode discipline (INT-2):
    - FACTUAL/PROCEDURAL claims get full adjudication
    - COUNTERFACTUAL/SIMULATION/QUOTED claims are logged but not adjudicated
      (they can't create timeline obligations)
    """
    timeline_modes = {ClaimMode.FACTUAL, ClaimMode.PROCEDURAL}
    return atom.mode in timeline_modes


def bridge_with_mode_filter(
    atoms: List[ClaimAtom],
    provenance: Provenance = Provenance.ASSUMED,
    evidence_map: Dict[str, List[EvidenceRef]] = None,
) -> Tuple[List[CandidateCommitment], List[ClaimAtom]]:
    """
    Bridge claims with mode filtering.
    
    Returns:
        (candidates_for_adjudication, claims_logged_only)
        
    FACTUAL/PROCEDURAL → adjudicate
    COUNTERFACTUAL/SIMULATION/QUOTED → log only, no adjudication
    """
    evidence_map = evidence_map or {}
    
    candidates = []
    logged_only = []
    
    for atom in atoms:
        if should_bridge_for_adjudication(atom):
            refs = evidence_map.get(atom.prop_hash, [])
            candidate = claim_atom_to_candidate(atom, provenance, refs)
            candidates.append(candidate)
        else:
            logged_only.append(atom)
    
    return candidates, logged_only


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    from .claim_extractor import Modality, Quantifier
    
    print("=== V1 → V2 Bridge Demo ===\n")
    
    # Create a V1 ClaimAtom
    atom = ClaimAtom(
        prop_hash="abc123",
        confidence=0.85,
        polarity=1,
        modality=Modality.CERTAIN,
        quantifier=Quantifier.BARE,
        tense="past",
        span=(0, 50),
        span_quote="Python 3.11 was released in October 2022",
        entities=("Python 3.11",),
        predicate="released",
        value_norm="October 2022",
        value_features={"year": 2022, "month": "October"},
    )
    
    # Convert to V2
    candidate = claim_atom_to_candidate(
        atom,
        provenance=Provenance.ASSUMED,
    )
    
    print(f"V1 ClaimAtom:")
    print(f"  prop_hash: {atom.prop_hash}")
    print(f"  predicate: {atom.predicate}")
    print(f"  entities: {atom.entities}")
    print(f"  value: {atom.value_norm}")
    print(f"  confidence: {atom.confidence}")
    
    print(f"\nV2 CandidateCommitment:")
    print(f"  predicate type: {candidate.predicate.ptype.name}")
    print(f"  predicate args: {candidate.predicate.args}")
    print(f"  sigma: {candidate.sigma}")
    print(f"  t_scope: {candidate.t_scope}")
    print(f"  provclass: {candidate.provclass.name}")
    
    # Now test adjudication
    from .symbolic_substrate import Adjudicator, SymbolicState
    
    state = SymbolicState()
    adjudicator = Adjudicator()
    
    result = adjudicator.adjudicate(state, candidate)
    
    print(f"\nAdjudication Result:")
    print(f"  decision: {result.decision.name}")
    print(f"  support_mass: {result.support_mass_computed:.3f}")
    print(f"  support_required: {result.support_mass_required:.3f}")
    print(f"  reason: {result.reason_code}")
