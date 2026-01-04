"""
Claim Diff Engine

Compares source and output claim sets to detect epistemic changes.

Three buckets (no ambiguity):
1. PRESERVED - Same proposition, same strength
2. MUTATED - Same proposition, meaningfully changed
3. NOVEL/DROPPED - Appears or disappears

This turns "hallucination" into state deltas.

Usage:
    from epistemic_governor.claim_diff import (
        ClaimDiffer,
        DiffResult,
        MutationType,
    )
    
    differ = ClaimDiffer()
    result = differ.diff(source_claims, output_claims)
    
    for novel in result.novel:
        print(f"NEW: {novel.prop_hash}")
    for mutation in result.mutated:
        print(f"MUTATED: {mutation.mutation_type.name}")
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Tuple
from enum import Enum, auto

from .claim_extractor import (
    ClaimAtom, ClaimSet,
    Modality, Quantifier,
    MODALITY_STRENGTH, QUANTIFIER_STRENGTH,
)
from .heading import Heading, HeadingType


# =============================================================================
# Mutation Types
# =============================================================================

class MutationType(Enum):
    """Types of claim mutations."""
    POLARITY_FLIP = auto()       # +1 → -1 or vice versa
    MODALITY_STRENGTHEN = auto() # might → assert
    MODALITY_WEAKEN = auto()     # assert → might
    QUANTIFIER_STRENGTHEN = auto() # some → all
    QUANTIFIER_WEAKEN = auto()   # all → some
    TENSE_SHIFT = auto()         # past → present
    VALUE_DRIFT = auto()         # numeric/date change
    

# Mutation severity weights (from ChatGPT's d) design)
MUTATION_SEVERITY = {
    MutationType.POLARITY_FLIP: 1.0,
    MutationType.TENSE_SHIFT: 0.8,
    MutationType.MODALITY_STRENGTHEN: 0.6,
    MutationType.QUANTIFIER_STRENGTHEN: 0.5,
    MutationType.VALUE_DRIFT: 0.4,
    MutationType.MODALITY_WEAKEN: 0.3,
    MutationType.QUANTIFIER_WEAKEN: 0.2,
}


@dataclass
class MutationEvent:
    """A detected mutation between source and output claims."""
    mutation_type: MutationType
    severity: float
    prop_hash: str
    source_claim: ClaimAtom
    output_claim: ClaimAtom
    details: str = ""


@dataclass
class AlignedPair:
    """A pair of aligned claims (same prop_hash)."""
    source: ClaimAtom
    output: ClaimAtom
    mutations: List[MutationEvent] = field(default_factory=list)
    
    @property
    def is_preserved(self) -> bool:
        return len(self.mutations) == 0
    
    @property
    def total_severity(self) -> float:
        return sum(m.severity for m in self.mutations)


# =============================================================================
# Diff Result
# =============================================================================

@dataclass
class DiffResult:
    """Result of comparing source and output claim sets."""
    
    # Three buckets
    preserved: List[AlignedPair] = field(default_factory=list)
    mutated: List[AlignedPair] = field(default_factory=list)
    novel: List[ClaimAtom] = field(default_factory=list)
    dropped: List[ClaimAtom] = field(default_factory=list)
    
    # Aggregates
    @property
    def total_mutations(self) -> int:
        return sum(len(p.mutations) for p in self.mutated)
    
    @property
    def total_severity(self) -> float:
        return sum(p.total_severity for p in self.mutated)
    
    @property
    def novel_count(self) -> int:
        return len(self.novel)
    
    @property
    def has_violations(self) -> bool:
        """Are there any novel claims or high-severity mutations?"""
        if self.novel:
            return True
        return self.total_severity >= 1.0
    
    def get_mutation_events(self) -> List[MutationEvent]:
        """Get all mutation events."""
        events = []
        for pair in self.mutated:
            events.extend(pair.mutations)
        return events
    
    def to_dict(self) -> Dict:
        return {
            "preserved_count": len(self.preserved),
            "mutated_count": len(self.mutated),
            "novel_count": len(self.novel),
            "dropped_count": len(self.dropped),
            "total_mutations": self.total_mutations,
            "total_severity": self.total_severity,
            "has_violations": self.has_violations,
        }


# =============================================================================
# Claim Differ
# =============================================================================

class ClaimDiffer:
    """
    Compares claim sets and produces diff results.
    
    Algorithm:
    1. Index claims by prop_hash
    2. Align exact-hash matches
    3. Detect mutations in aligned pairs
    4. Classify unmatched as novel/dropped
    """
    
    def __init__(
        self,
        value_drift_tolerance: float = 0.1,
        similarity_fallback: bool = False,
    ):
        self.value_drift_tolerance = value_drift_tolerance
        self.similarity_fallback = similarity_fallback
    
    def diff(
        self,
        source: ClaimSet,
        output: ClaimSet,
    ) -> DiffResult:
        """
        Compute diff between source and output claim sets.
        """
        result = DiffResult()
        
        # Index by hash
        source_by_hash = source.by_hash()
        output_by_hash = output.by_hash()
        
        source_hashes = set(source_by_hash.keys())
        output_hashes = set(output_by_hash.keys())
        
        # Common hashes (aligned)
        common = source_hashes & output_hashes
        
        # Process aligned pairs - use greedy matching to minimize severity
        for prop_hash in common:
            src_claims = list(source_by_hash[prop_hash])
            out_claims = list(output_by_hash[prop_hash])
            
            # Greedy pairing: for each output claim, find best matching source
            used_src_indices = set()
            
            for out_claim in out_claims:
                best_src_idx = None
                best_score = float('inf')
                
                for i, src_claim in enumerate(src_claims):
                    if i in used_src_indices:
                        continue
                    
                    # Compute similarity score (lower = better match)
                    score = self._compute_pair_distance(src_claim, out_claim)
                    if score < best_score:
                        best_score = score
                        best_src_idx = i
                
                if best_src_idx is not None:
                    used_src_indices.add(best_src_idx)
                    src_claim = src_claims[best_src_idx]
                else:
                    # No source left, use first one
                    src_claim = src_claims[0]
                
                pair = AlignedPair(source=src_claim, output=out_claim)
                self._detect_mutations(pair)
                
                if pair.is_preserved:
                    result.preserved.append(pair)
                else:
                    result.mutated.append(pair)
        
        # Novel claims (in output, not in source)
        for prop_hash in (output_hashes - source_hashes):
            for claim in output_by_hash[prop_hash]:
                # Optional: similarity fallback
                if self.similarity_fallback:
                    similar = self._find_similar(claim, source)
                    if similar:
                        # Treat as mutation instead
                        pair = AlignedPair(source=similar, output=claim)
                        self._detect_mutations(pair)
                        result.mutated.append(pair)
                        continue
                
                result.novel.append(claim)
        
        # Dropped claims (in source, not in output)
        for prop_hash in (source_hashes - output_hashes):
            for claim in source_by_hash[prop_hash]:
                result.dropped.append(claim)
        
        return result
    
    def _compute_pair_distance(self, src: ClaimAtom, out: ClaimAtom) -> float:
        """
        Compute distance between two claims for optimal pairing.
        
        Lower = better match (fewer differences).
        """
        distance = 0.0
        
        # Polarity difference (highest weight)
        if src.polarity != out.polarity:
            distance += 1.0
        
        # Modality difference
        src_mod = MODALITY_STRENGTH[src.modality]
        out_mod = MODALITY_STRENGTH[out.modality]
        distance += abs(src_mod - out_mod) * 0.2
        
        # Quantifier difference
        src_quant = QUANTIFIER_STRENGTH[src.quantifier]
        out_quant = QUANTIFIER_STRENGTH[out.quantifier]
        distance += abs(src_quant - out_quant) * 0.1
        
        # Tense difference
        if src.tense != out.tense:
            distance += 0.3
        
        # Value difference (compare raw)
        if src.value_raw != out.value_raw:
            distance += 0.2
        
        return distance
    
    def _detect_mutations(self, pair: AlignedPair):
        """Detect mutations between aligned claims."""
        src = pair.source
        out = pair.output
        
        # 1. Polarity flip (hard fail)
        if src.polarity != out.polarity:
            pair.mutations.append(MutationEvent(
                mutation_type=MutationType.POLARITY_FLIP,
                severity=MUTATION_SEVERITY[MutationType.POLARITY_FLIP],
                prop_hash=src.prop_hash,
                source_claim=src,
                output_claim=out,
                details=f"polarity {src.polarity} → {out.polarity}",
            ))
        
        # 2. Modality shift
        src_strength = MODALITY_STRENGTH[src.modality]
        out_strength = MODALITY_STRENGTH[out.modality]
        
        if out_strength > src_strength:
            pair.mutations.append(MutationEvent(
                mutation_type=MutationType.MODALITY_STRENGTHEN,
                severity=MUTATION_SEVERITY[MutationType.MODALITY_STRENGTHEN],
                prop_hash=src.prop_hash,
                source_claim=src,
                output_claim=out,
                details=f"modality {src.modality.value} → {out.modality.value}",
            ))
        elif out_strength < src_strength:
            pair.mutations.append(MutationEvent(
                mutation_type=MutationType.MODALITY_WEAKEN,
                severity=MUTATION_SEVERITY[MutationType.MODALITY_WEAKEN],
                prop_hash=src.prop_hash,
                source_claim=src,
                output_claim=out,
                details=f"modality {src.modality.value} → {out.modality.value}",
            ))
        
        # 3. Quantifier shift
        src_quant = QUANTIFIER_STRENGTH[src.quantifier]
        out_quant = QUANTIFIER_STRENGTH[out.quantifier]
        
        if out_quant > src_quant:
            pair.mutations.append(MutationEvent(
                mutation_type=MutationType.QUANTIFIER_STRENGTHEN,
                severity=MUTATION_SEVERITY[MutationType.QUANTIFIER_STRENGTHEN],
                prop_hash=src.prop_hash,
                source_claim=src,
                output_claim=out,
                details=f"quantifier {src.quantifier.value} → {out.quantifier.value}",
            ))
        elif out_quant < src_quant:
            pair.mutations.append(MutationEvent(
                mutation_type=MutationType.QUANTIFIER_WEAKEN,
                severity=MUTATION_SEVERITY[MutationType.QUANTIFIER_WEAKEN],
                prop_hash=src.prop_hash,
                source_claim=src,
                output_claim=out,
                details=f"quantifier {src.quantifier.value} → {out.quantifier.value}",
            ))
        
        # 4. Tense shift
        if src.tense != out.tense and src.tense != "atemporal" and out.tense != "atemporal":
            pair.mutations.append(MutationEvent(
                mutation_type=MutationType.TENSE_SHIFT,
                severity=MUTATION_SEVERITY[MutationType.TENSE_SHIFT],
                prop_hash=src.prop_hash,
                source_claim=src,
                output_claim=out,
                details=f"tense {src.tense} → {out.tense}",
            ))
        
        # 5. Value drift (compare RAW values, not normalized)
        # This is where we catch "October 2022" vs "late 2022"
        if src.value_raw != out.value_raw:
            drift, drift_details = self._compute_value_drift(src, out)
            if drift > self.value_drift_tolerance:
                pair.mutations.append(MutationEvent(
                    mutation_type=MutationType.VALUE_DRIFT,
                    severity=MUTATION_SEVERITY[MutationType.VALUE_DRIFT],
                    prop_hash=src.prop_hash,
                    source_claim=src,
                    output_claim=out,
                    details=f"value '{src.value_raw}' → '{out.value_raw}' ({drift_details})",
                ))
    
    def _compute_value_drift(self, src: ClaimAtom, out: ClaimAtom) -> Tuple[float, str]:
        """
        Compute drift between two claims' values using features.
        
        Returns (drift_score, details_string).
        """
        src_features = src.value_features or {}
        out_features = out.value_features or {}
        
        # Check for modifier changes (high signal)
        src_mod = src_features.get("value_modifier") or src_features.get("date_modifier")
        out_mod = out_features.get("value_modifier") or out_features.get("date_modifier")
        
        if src_mod != out_mod:
            if src_mod and out_mod:
                return 0.5, f"modifier changed: {src_mod} → {out_mod}"
            elif out_mod and not src_mod:
                return 0.3, f"modifier added: {out_mod}"
            elif src_mod and not out_mod:
                return 0.3, f"modifier removed: {src_mod}"
        
        # Check for date precision changes
        src_month = src_features.get("month")
        out_month = out_features.get("month")
        
        if src_month and not out_month:
            return 0.4, f"date precision lost: {src_month} → unspecified"
        if not src_month and out_month:
            return 0.3, f"date precision added: unspecified → {out_month}"
        if src_month and out_month and src_month != out_month:
            return 0.6, f"month changed: {src_month} → {out_month}"
        
        # Check for numeric changes
        src_num = src_features.get("number")
        out_num = out_features.get("number")
        
        if src_num is not None and out_num is not None:
            if src_num == 0:
                drift = 1.0 if out_num != 0 else 0.0
            else:
                drift = abs(out_num - src_num) / abs(src_num)
            if drift > 0.01:
                return min(drift, 1.0), f"numeric drift: {src_num} → {out_num}"
        
        # Fall back to string comparison
        if src.value_raw.lower() != out.value_raw.lower():
            # Compute similarity
            max_len = max(len(src.value_raw), len(out.value_raw))
            if max_len == 0:
                return 0.0, "empty"
            matches = sum(1 for a, b in zip(src.value_raw.lower(), out.value_raw.lower()) if a == b)
            drift = 1.0 - (matches / max_len)
            return drift, f"string drift: {drift:.2f}"
        
        return 0.0, "no drift"
    
    def _find_similar(self, claim: ClaimAtom, source: ClaimSet) -> Optional[ClaimAtom]:
        """Find a similar claim in source set (fallback for paraphrases)."""
        # Simple heuristic: same entities + same predicate
        for src_claim in source.claims:
            if (set(claim.entities) & set(src_claim.entities) and
                claim.predicate == src_claim.predicate):
                return src_claim
        return None


# =============================================================================
# Heading-Aware Adjudicator
# =============================================================================

class HeadingAdjudicator:
    """
    Applies heading rules to diff results.
    
    Determines which diffs are violations vs allowed changes.
    """
    
    def adjudicate(
        self,
        diff: DiffResult,
        heading: Heading,
    ) -> "AdjudicationResult":
        """
        Apply heading rules to diff.
        """
        violations = []
        allowed = []
        
        # Novel claims
        for claim in diff.novel:
            if heading.max_new_claims == 0:
                violations.append(ViolationEvent(
                    violation_type="NOVEL_CLAIM",
                    severity=0.8,
                    claim=claim,
                    reason="Heading forbids new claims",
                ))
            elif len(diff.novel) > heading.max_new_claims:
                violations.append(ViolationEvent(
                    violation_type="NOVEL_CLAIM_OVER_BUDGET",
                    severity=0.6,
                    claim=claim,
                    reason=f"Exceeds new claim budget ({heading.max_new_claims})",
                ))
            else:
                allowed.append(("NOVEL", claim))
        
        # Mutations
        for pair in diff.mutated:
            for mutation in pair.mutations:
                # Polarity flip is always a violation
                if mutation.mutation_type == MutationType.POLARITY_FLIP:
                    violations.append(ViolationEvent(
                        violation_type="POLARITY_FLIP",
                        severity=mutation.severity,
                        claim=pair.output,
                        reason=mutation.details,
                    ))
                
                # Modality strengthening is a violation for summarize/translate
                elif mutation.mutation_type == MutationType.MODALITY_STRENGTHEN:
                    if heading.heading_type in {HeadingType.SUMMARIZE, HeadingType.TRANSLATE}:
                        violations.append(ViolationEvent(
                            violation_type="MODALITY_STRENGTHEN",
                            severity=mutation.severity,
                            claim=pair.output,
                            reason=mutation.details,
                        ))
                
                # Quantifier strengthening
                elif mutation.mutation_type == MutationType.QUANTIFIER_STRENGTHEN:
                    if heading.heading_type in {HeadingType.SUMMARIZE, HeadingType.TRANSLATE}:
                        violations.append(ViolationEvent(
                            violation_type="QUANTIFIER_STRENGTHEN",
                            severity=mutation.severity,
                            claim=pair.output,
                            reason=mutation.details,
                        ))
                
                # Tense shift
                elif mutation.mutation_type == MutationType.TENSE_SHIFT:
                    violations.append(ViolationEvent(
                        violation_type="TENSE_SHIFT",
                        severity=mutation.severity,
                        claim=pair.output,
                        reason=mutation.details,
                    ))
        
        # Dropped claims (only violation for some headings)
        for claim in diff.dropped:
            if heading.heading_type == HeadingType.REWRITE:
                if hasattr(heading, 'preserve_claims') and heading.preserve_claims:
                    violations.append(ViolationEvent(
                        violation_type="DROPPED_CLAIM",
                        severity=0.3,
                        claim=claim,
                        reason="Rewrite heading requires preserving claims",
                    ))
        
        return AdjudicationResult(
            violations=violations,
            allowed=allowed,
            diff=diff,
        )


@dataclass
class ViolationEvent:
    """A heading violation."""
    violation_type: str
    severity: float
    claim: ClaimAtom
    reason: str


@dataclass
class AdjudicationResult:
    """Result of applying heading rules to diff."""
    violations: List[ViolationEvent]
    allowed: List[Tuple[str, ClaimAtom]]
    diff: DiffResult
    
    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0
    
    @property
    def total_severity(self) -> float:
        return sum(v.severity for v in self.violations)
    
    @property
    def should_disengage(self) -> bool:
        """Should autopilot disengage?"""
        # Disengage on high severity or polarity flip
        return (
            self.total_severity >= 1.0 or
            any(v.violation_type == "POLARITY_FLIP" for v in self.violations)
        )
    
    @property
    def should_arbitrate(self) -> bool:
        """Should we pause for user arbitration?"""
        return self.has_violations and not self.should_disengage


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=== Claim Diff Demo ===\n")
    
    from .claim_extractor import ClaimExtractor, ExtractMode
    
    extractor = ClaimExtractor()
    
    # Source
    source_text = """
    Python 3.11 was released in October 2022.
    The new version requires at least 8GB of RAM.
    Some users reported performance improvements.
    """
    
    # Output with drift
    output_text = """
    Python 3.11 was released in late 2022.
    The new version requires approximately 8GB of RAM.
    Most users reported significant performance improvements.
    Additionally, Python 3.12 was released in 2023.
    """
    
    source_claims = extractor.extract(source_text, ExtractMode.SOURCE)
    output_claims = extractor.extract(output_text, ExtractMode.OUTPUT)
    
    print(f"Source claims: {len(source_claims.claims)}")
    print(f"Output claims: {len(output_claims.claims)}")
    
    differ = ClaimDiffer()
    diff = differ.diff(source_claims, output_claims)
    
    print("\n--- Diff Result ---")
    print(f"Preserved: {len(diff.preserved)}")
    print(f"Mutated: {len(diff.mutated)}")
    print(f"Novel: {len(diff.novel)}")
    print(f"Dropped: {len(diff.dropped)}")
    print(f"Total severity: {diff.total_severity:.2f}")
    
    print("\n--- Mutations ---")
    for event in diff.get_mutation_events():
        print(f"  {event.mutation_type.name}: {event.details}")
    
    print("\n--- Novel Claims ---")
    for claim in diff.novel:
        print(f"  {claim.prop_hash}: {claim.entities} {claim.predicate} {claim.value}")
