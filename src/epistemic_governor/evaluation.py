"""
Evaluation Framework for Epistemic Governor

This module provides:
1. Corpus format for annotated LLM outputs
2. Evaluation metrics for extraction, calibration, governance
3. Comparison harness for governed vs ungoverned runs
4. Reproducibility infrastructure

Scientific claims we need to validate:
- Extractor finds claims accurately (precision/recall)
- Confidence scores are calibrated (reliability diagrams)
- Governor prevents contradictions (drift rate comparison)
- Thermal model correlates with actual instability

Usage:
    from epistemic_governor.evaluation import (
        AnnotatedSample,
        EvaluationCorpus,
        ExtractionEvaluator,
        run_evaluation,
    )
    
    # Load annotated corpus
    corpus = EvaluationCorpus.load("corpus.json")
    
    # Evaluate extraction
    evaluator = ExtractionEvaluator()
    results = evaluator.evaluate(corpus)
    print(results.summary())
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Set
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
import json
import hashlib


# =============================================================================
# Corpus Format
# =============================================================================

class AnnotatedClaimType(Enum):
    """Claim types for annotation (matches extractor types)."""
    FACTUAL = "factual"
    CAUSAL = "causal"
    PREDICTIVE = "predictive"
    QUANTITATIVE = "quantitative"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    ATTRIBUTION = "attribution"
    DEFINITION = "definition"
    NEGATION = "negation"
    HEDGE = "hedge"  # Explicitly hedged statement
    OPINION = "opinion"  # Subjective, not a factual claim
    OTHER = "other"


@dataclass
class AnnotatedClaim:
    """
    A human-annotated claim in LLM output.
    
    This is ground truth for evaluation.
    """
    # Identification
    id: str
    
    # Location in source text
    text: str              # The claim text
    span_start: int        # Character offset start
    span_end: int          # Character offset end
    
    # Classification
    claim_type: AnnotatedClaimType
    
    # Confidence annotation
    expressed_confidence: float  # What confidence does the text express? (0-1)
    annotator_confidence: float  # How confident is annotator this is correct? (0-1)
    
    # Relationships
    contradicts: List[str] = field(default_factory=list)  # IDs of contradicted claims
    supports: List[str] = field(default_factory=list)     # IDs of supported claims
    requires_evidence: bool = False  # Does this need external verification?
    
    # Verification (if checked)
    verified: Optional[bool] = None  # True=correct, False=incorrect, None=unchecked
    verification_source: Optional[str] = None
    
    # Annotation metadata
    annotator: str = "unknown"
    annotation_notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "claim_type": self.claim_type.value,
            "expressed_confidence": self.expressed_confidence,
            "annotator_confidence": self.annotator_confidence,
            "contradicts": self.contradicts,
            "supports": self.supports,
            "requires_evidence": self.requires_evidence,
            "verified": self.verified,
            "verification_source": self.verification_source,
            "annotator": self.annotator,
            "annotation_notes": self.annotation_notes,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnnotatedClaim":
        return cls(
            id=d["id"],
            text=d["text"],
            span_start=d["span_start"],
            span_end=d["span_end"],
            claim_type=AnnotatedClaimType(d["claim_type"]),
            expressed_confidence=d["expressed_confidence"],
            annotator_confidence=d["annotator_confidence"],
            contradicts=d.get("contradicts", []),
            supports=d.get("supports", []),
            requires_evidence=d.get("requires_evidence", False),
            verified=d.get("verified"),
            verification_source=d.get("verification_source"),
            annotator=d.get("annotator", "unknown"),
            annotation_notes=d.get("annotation_notes", ""),
        )


@dataclass
class AnnotatedSample:
    """
    A single LLM output with annotated claims.
    
    This is one sample in the evaluation corpus.
    """
    # Identification
    id: str
    
    # The LLM output
    prompt: str
    response: str
    
    # Model info
    model: str = "unknown"
    temperature: float = 0.0
    timestamp: Optional[str] = None
    
    # Annotated claims (ground truth)
    claims: List[AnnotatedClaim] = field(default_factory=list)
    
    # Sample-level annotations
    overall_confidence: float = 0.5  # Overall expressed confidence
    contains_contradictions: bool = False
    topic: str = ""
    difficulty: str = "medium"  # easy, medium, hard
    
    # Annotation metadata
    annotator: str = "unknown"
    annotation_time_minutes: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "response": self.response,
            "model": self.model,
            "temperature": self.temperature,
            "timestamp": self.timestamp,
            "claims": [c.to_dict() for c in self.claims],
            "overall_confidence": self.overall_confidence,
            "contains_contradictions": self.contains_contradictions,
            "topic": self.topic,
            "difficulty": self.difficulty,
            "annotator": self.annotator,
            "annotation_time_minutes": self.annotation_time_minutes,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnnotatedSample":
        return cls(
            id=d["id"],
            prompt=d["prompt"],
            response=d["response"],
            model=d.get("model", "unknown"),
            temperature=d.get("temperature", 0.0),
            timestamp=d.get("timestamp"),
            claims=[AnnotatedClaim.from_dict(c) for c in d.get("claims", [])],
            overall_confidence=d.get("overall_confidence", 0.5),
            contains_contradictions=d.get("contains_contradictions", False),
            topic=d.get("topic", ""),
            difficulty=d.get("difficulty", "medium"),
            annotator=d.get("annotator", "unknown"),
            annotation_time_minutes=d.get("annotation_time_minutes", 0.0),
        )


@dataclass
class EvaluationCorpus:
    """
    A collection of annotated samples for evaluation.
    """
    # Metadata
    name: str
    version: str
    description: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Samples
    samples: List[AnnotatedSample] = field(default_factory=list)
    
    # Corpus-level stats (computed)
    _stats: Optional[Dict[str, Any]] = field(default=None, repr=False)
    
    def add_sample(self, sample: AnnotatedSample):
        self.samples.append(sample)
        self._stats = None  # Invalidate cache
    
    @property
    def stats(self) -> Dict[str, Any]:
        if self._stats is None:
            self._stats = self._compute_stats()
        return self._stats
    
    def _compute_stats(self) -> Dict[str, Any]:
        total_claims = sum(len(s.claims) for s in self.samples)
        claim_types = {}
        verified_correct = 0
        verified_incorrect = 0
        unverified = 0
        
        for sample in self.samples:
            for claim in sample.claims:
                ct = claim.claim_type.value
                claim_types[ct] = claim_types.get(ct, 0) + 1
                if claim.verified is True:
                    verified_correct += 1
                elif claim.verified is False:
                    verified_incorrect += 1
                else:
                    unverified += 1
        
        return {
            "sample_count": len(self.samples),
            "total_claims": total_claims,
            "claims_per_sample": total_claims / len(self.samples) if self.samples else 0,
            "claim_types": claim_types,
            "verified_correct": verified_correct,
            "verified_incorrect": verified_incorrect,
            "unverified": unverified,
            "models": list(set(s.model for s in self.samples)),
            "topics": list(set(s.topic for s in self.samples if s.topic)),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at,
            "samples": [s.to_dict() for s in self.samples],
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvaluationCorpus":
        return cls(
            name=d["name"],
            version=d["version"],
            description=d["description"],
            created_at=d.get("created_at", ""),
            samples=[AnnotatedSample.from_dict(s) for s in d.get("samples", [])],
        )
    
    def save(self, path: str):
        """Save corpus to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "EvaluationCorpus":
        """Load corpus from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# Extraction Evaluation
# =============================================================================

@dataclass
class SpanMatch:
    """Result of matching extracted span to ground truth."""
    extracted_id: str
    ground_truth_id: Optional[str]
    overlap_ratio: float  # IoU of character spans
    type_match: bool
    confidence_error: float  # |extracted - annotated|


@dataclass 
class ExtractionResults:
    """Results of extraction evaluation on a single sample."""
    sample_id: str
    
    # Counts
    true_positives: int = 0   # Correctly extracted claims
    false_positives: int = 0  # Extracted but not in ground truth
    false_negatives: int = 0  # In ground truth but not extracted
    
    # Detailed matches
    matches: List[SpanMatch] = field(default_factory=list)
    
    # Type-level results
    type_confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Confidence calibration
    confidence_errors: List[float] = field(default_factory=list)
    
    @property
    def precision(self) -> float:
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    @property
    def recall(self) -> float:
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)
    
    @property
    def mean_confidence_error(self) -> float:
        if not self.confidence_errors:
            return 0.0
        return sum(self.confidence_errors) / len(self.confidence_errors)


@dataclass
class AggregateExtractionResults:
    """Aggregated results across corpus."""
    corpus_name: str
    sample_results: List[ExtractionResults] = field(default_factory=list)
    
    @property
    def total_tp(self) -> int:
        return sum(r.true_positives for r in self.sample_results)
    
    @property
    def total_fp(self) -> int:
        return sum(r.false_positives for r in self.sample_results)
    
    @property
    def total_fn(self) -> int:
        return sum(r.false_negatives for r in self.sample_results)
    
    @property
    def precision(self) -> float:
        if self.total_tp + self.total_fp == 0:
            return 0.0
        return self.total_tp / (self.total_tp + self.total_fp)
    
    @property
    def recall(self) -> float:
        if self.total_tp + self.total_fn == 0:
            return 0.0
        return self.total_tp / (self.total_tp + self.total_fn)
    
    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)
    
    def summary(self) -> str:
        lines = [
            f"=== Extraction Evaluation: {self.corpus_name} ===",
            f"Samples: {len(self.sample_results)}",
            f"",
            f"Overall:",
            f"  Precision: {self.precision:.3f}",
            f"  Recall:    {self.recall:.3f}",
            f"  F1:        {self.f1:.3f}",
            f"",
            f"Counts:",
            f"  True Positives:  {self.total_tp}",
            f"  False Positives: {self.total_fp}",
            f"  False Negatives: {self.total_fn}",
        ]
        return "\n".join(lines)


class ExtractionEvaluator:
    """
    Evaluates claim extraction against ground truth.
    """
    
    def __init__(self, overlap_threshold: float = 0.5):
        """
        Args:
            overlap_threshold: Minimum IoU for span match (default 0.5)
        """
        self.overlap_threshold = overlap_threshold
    
    def evaluate_sample(
        self,
        sample: AnnotatedSample,
        extracted_claims: List[Any],  # List of ProposedCommitment or similar
    ) -> ExtractionResults:
        """Evaluate extraction on a single sample."""
        results = ExtractionResults(sample_id=sample.id)
        
        ground_truth = {c.id: c for c in sample.claims}
        matched_gt = set()
        
        for extracted in extracted_claims:
            # Find best matching ground truth claim
            best_match = None
            best_overlap = 0.0
            
            ext_start = getattr(extracted, 'span_start', 0)
            ext_end = getattr(extracted, 'span_end', len(getattr(extracted, 'text', '')))
            
            for gt_id, gt in ground_truth.items():
                if gt_id in matched_gt:
                    continue
                
                overlap = self._span_overlap(
                    ext_start, ext_end,
                    gt.span_start, gt.span_end
                )
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = gt
            
            if best_match and best_overlap >= self.overlap_threshold:
                # True positive
                results.true_positives += 1
                matched_gt.add(best_match.id)
                
                # Check type match
                ext_type = getattr(extracted, 'claim_type', None)
                if ext_type:
                    ext_type_str = ext_type.name if hasattr(ext_type, 'name') else str(ext_type)
                    gt_type_str = best_match.claim_type.value
                    type_match = ext_type_str.lower() == gt_type_str.lower()
                else:
                    type_match = False
                
                # Confidence error
                ext_conf = getattr(extracted, 'confidence', 0.5)
                conf_error = abs(ext_conf - best_match.expressed_confidence)
                results.confidence_errors.append(conf_error)
                
                results.matches.append(SpanMatch(
                    extracted_id=getattr(extracted, 'id', 'unknown'),
                    ground_truth_id=best_match.id,
                    overlap_ratio=best_overlap,
                    type_match=type_match,
                    confidence_error=conf_error,
                ))
            else:
                # False positive
                results.false_positives += 1
                results.matches.append(SpanMatch(
                    extracted_id=getattr(extracted, 'id', 'unknown'),
                    ground_truth_id=None,
                    overlap_ratio=best_overlap,
                    type_match=False,
                    confidence_error=0.0,
                ))
        
        # Count false negatives (ground truth not matched)
        results.false_negatives = len(ground_truth) - len(matched_gt)
        
        return results
    
    def evaluate_corpus(
        self,
        corpus: EvaluationCorpus,
        extractor,  # CommitmentExtractor instance
    ) -> AggregateExtractionResults:
        """Evaluate extraction on entire corpus."""
        results = AggregateExtractionResults(corpus_name=corpus.name)
        
        for sample in corpus.samples:
            # Run extraction
            extracted = extractor.extract(sample.response)
            
            # Evaluate
            sample_results = self.evaluate_sample(sample, extracted)
            results.sample_results.append(sample_results)
        
        return results
    
    def _span_overlap(
        self,
        start1: int, end1: int,
        start2: int, end2: int,
    ) -> float:
        """Compute IoU (intersection over union) of two spans."""
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        
        if intersection_start >= intersection_end:
            return 0.0
        
        intersection = intersection_end - intersection_start
        union = (end1 - start1) + (end2 - start2) - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union


# =============================================================================
# Governance Comparison
# =============================================================================

@dataclass
class GovernanceComparison:
    """
    Compare governed vs ungoverned runs.
    """
    sample_id: str
    prompt: str
    
    # Ungoverned baseline
    ungoverned_response: str
    ungoverned_claims: int = 0
    ungoverned_contradictions: int = 0
    
    # Governed run
    governed_response: str = ""
    governed_claims_proposed: int = 0
    governed_claims_committed: int = 0
    governed_claims_blocked: int = 0
    governed_claims_hedged: int = 0
    governed_contradictions: int = 0
    
    # Thermal
    final_instability: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "prompt": self.prompt,
            "ungoverned_claims": self.ungoverned_claims,
            "ungoverned_contradictions": self.ungoverned_contradictions,
            "governed_claims_proposed": self.governed_claims_proposed,
            "governed_claims_committed": self.governed_claims_committed,
            "governed_claims_blocked": self.governed_claims_blocked,
            "governed_claims_hedged": self.governed_claims_hedged,
            "governed_contradictions": self.governed_contradictions,
            "final_instability": self.final_instability,
        }


@dataclass
class ComparisonResults:
    """Aggregated comparison results."""
    comparisons: List[GovernanceComparison] = field(default_factory=list)
    
    @property
    def ungoverned_contradiction_rate(self) -> float:
        total_claims = sum(c.ungoverned_claims for c in self.comparisons)
        total_contradictions = sum(c.ungoverned_contradictions for c in self.comparisons)
        if total_claims == 0:
            return 0.0
        return total_contradictions / total_claims
    
    @property
    def governed_contradiction_rate(self) -> float:
        total_claims = sum(c.governed_claims_committed for c in self.comparisons)
        total_contradictions = sum(c.governed_contradictions for c in self.comparisons)
        if total_claims == 0:
            return 0.0
        return total_contradictions / total_claims
    
    @property
    def block_rate(self) -> float:
        total_proposed = sum(c.governed_claims_proposed for c in self.comparisons)
        total_blocked = sum(c.governed_claims_blocked for c in self.comparisons)
        if total_proposed == 0:
            return 0.0
        return total_blocked / total_proposed
    
    @property
    def hedge_rate(self) -> float:
        total_proposed = sum(c.governed_claims_proposed for c in self.comparisons)
        total_hedged = sum(c.governed_claims_hedged for c in self.comparisons)
        if total_proposed == 0:
            return 0.0
        return total_hedged / total_proposed
    
    def summary(self) -> str:
        lines = [
            "=== Governance Comparison ===",
            f"Samples: {len(self.comparisons)}",
            "",
            "Contradiction Rates:",
            f"  Ungoverned: {self.ungoverned_contradiction_rate:.3f}",
            f"  Governed:   {self.governed_contradiction_rate:.3f}",
            f"  Reduction:  {(1 - self.governed_contradiction_rate / max(0.001, self.ungoverned_contradiction_rate)) * 100:.1f}%",
            "",
            "Governor Actions:",
            f"  Block rate: {self.block_rate:.3f}",
            f"  Hedge rate: {self.hedge_rate:.3f}",
        ]
        return "\n".join(lines)


# =============================================================================
# Demo Corpus Builder
# =============================================================================

def create_demo_corpus() -> EvaluationCorpus:
    """
    Create a small demo corpus for testing.
    
    In practice, you'd build this from real LLM outputs with human annotation.
    """
    corpus = EvaluationCorpus(
        name="demo_evaluation_corpus",
        version="2.0.0",
        description="Small demo corpus for testing evaluation framework",
    )
    
    # Sample 1: Simple factual claims
    sample1 = AnnotatedSample(
        id="demo_001",
        prompt="What is the capital of France?",
        response="The capital of France is Paris. Paris has been the capital since the 10th century. It has a population of approximately 2.1 million people in the city proper.",
        model="demo",
        topic="geography",
        difficulty="easy",
        claims=[
            AnnotatedClaim(
                id="demo_001_c1",
                text="The capital of France is Paris",
                span_start=0,
                span_end=31,
                claim_type=AnnotatedClaimType.FACTUAL,
                expressed_confidence=0.95,
                annotator_confidence=1.0,
                verified=True,
            ),
            AnnotatedClaim(
                id="demo_001_c2",
                text="Paris has been the capital since the 10th century",
                span_start=33,
                span_end=82,
                claim_type=AnnotatedClaimType.FACTUAL,
                expressed_confidence=0.85,
                annotator_confidence=0.7,
                verified=True,  # Actually 987 AD
            ),
            AnnotatedClaim(
                id="demo_001_c3",
                text="It has a population of approximately 2.1 million people",
                span_start=84,
                span_end=139,
                claim_type=AnnotatedClaimType.QUANTITATIVE,
                expressed_confidence=0.75,  # "approximately" hedges
                annotator_confidence=0.8,
                requires_evidence=True,
            ),
        ],
    )
    corpus.add_sample(sample1)
    
    # Sample 2: Contains a contradiction
    sample2 = AnnotatedSample(
        id="demo_002",
        prompt="Tell me about the speed of light.",
        response="The speed of light in a vacuum is exactly 299,792,458 meters per second. Light travels at about 300,000 km/s. Interestingly, nothing can travel faster than light. However, some particles appear to exceed light speed in certain media.",
        model="demo",
        topic="physics",
        difficulty="medium",
        contains_contradictions=True,
        claims=[
            AnnotatedClaim(
                id="demo_002_c1",
                text="The speed of light in a vacuum is exactly 299,792,458 meters per second",
                span_start=0,
                span_end=71,
                claim_type=AnnotatedClaimType.QUANTITATIVE,
                expressed_confidence=0.99,  # "exactly"
                annotator_confidence=1.0,
                verified=True,
            ),
            AnnotatedClaim(
                id="demo_002_c2",
                text="Light travels at about 300,000 km/s",
                span_start=73,
                span_end=108,
                claim_type=AnnotatedClaimType.QUANTITATIVE,
                expressed_confidence=0.80,  # "about"
                annotator_confidence=0.9,
                verified=True,  # Approximately correct
            ),
            AnnotatedClaim(
                id="demo_002_c3",
                text="nothing can travel faster than light",
                span_start=123,
                span_end=159,
                claim_type=AnnotatedClaimType.FACTUAL,
                expressed_confidence=0.90,
                annotator_confidence=0.95,
                verified=True,  # In vacuum, per relativity
            ),
            AnnotatedClaim(
                id="demo_002_c4",
                text="some particles appear to exceed light speed in certain media",
                span_start=170,
                span_end=230,
                claim_type=AnnotatedClaimType.FACTUAL,
                expressed_confidence=0.70,  # "appear to"
                annotator_confidence=0.8,
                contradicts=["demo_002_c3"],  # Apparent contradiction (Cherenkov)
                annotation_notes="Not actually a contradiction - Cherenkov radiation",
            ),
        ],
    )
    corpus.add_sample(sample2)
    
    # Sample 3: Hedged claims
    sample3 = AnnotatedSample(
        id="demo_003",
        prompt="What causes migraines?",
        response="Migraines are believed to involve changes in brain chemistry. Some researchers think they may be related to serotonin levels. It's possible that genetic factors play a role. Triggers might include stress, certain foods, or hormonal changes.",
        model="demo",
        topic="medicine",
        difficulty="medium",
        claims=[
            AnnotatedClaim(
                id="demo_003_c1",
                text="Migraines are believed to involve changes in brain chemistry",
                span_start=0,
                span_end=60,
                claim_type=AnnotatedClaimType.CAUSAL,
                expressed_confidence=0.65,  # "believed to"
                annotator_confidence=0.7,
            ),
            AnnotatedClaim(
                id="demo_003_c2",
                text="they may be related to serotonin levels",
                span_start=83,
                span_end=122,
                claim_type=AnnotatedClaimType.CAUSAL,
                expressed_confidence=0.50,  # "may be"
                annotator_confidence=0.6,
            ),
            AnnotatedClaim(
                id="demo_003_c3",
                text="It's possible that genetic factors play a role",
                span_start=124,
                span_end=170,
                claim_type=AnnotatedClaimType.CAUSAL,
                expressed_confidence=0.45,  # "possible"
                annotator_confidence=0.7,
            ),
            AnnotatedClaim(
                id="demo_003_c4",
                text="Triggers might include stress, certain foods, or hormonal changes",
                span_start=172,
                span_end=237,
                claim_type=AnnotatedClaimType.FACTUAL,
                expressed_confidence=0.55,  # "might"
                annotator_confidence=0.8,
            ),
        ],
    )
    corpus.add_sample(sample3)
    
    return corpus


# =============================================================================
# Main Evaluation Runner
# =============================================================================

def run_evaluation(
    corpus: EvaluationCorpus,
    extractor=None,
    kernel=None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run full evaluation suite.
    
    Args:
        corpus: Annotated evaluation corpus
        extractor: CommitmentExtractor instance (optional)
        kernel: EpistemicKernel instance (optional)
        output_path: Path to save results (optional)
    
    Returns:
        Dictionary with all evaluation results
    """
    results = {
        "corpus": corpus.name,
        "corpus_stats": corpus.stats,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Extraction evaluation
    if extractor:
        evaluator = ExtractionEvaluator()
        extraction_results = evaluator.evaluate_corpus(corpus, extractor)
        results["extraction"] = {
            "precision": extraction_results.precision,
            "recall": extraction_results.recall,
            "f1": extraction_results.f1,
            "true_positives": extraction_results.total_tp,
            "false_positives": extraction_results.total_fp,
            "false_negatives": extraction_results.total_fn,
        }
        print(extraction_results.summary())
    
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return results


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=== Evaluation Framework Demo ===\n")
    
    # Create demo corpus
    corpus = create_demo_corpus()
    print(f"Created corpus: {corpus.name}")
    print(f"Samples: {corpus.stats['sample_count']}")
    print(f"Total claims: {corpus.stats['total_claims']}")
    print(f"Claim types: {corpus.stats['claim_types']}")
    
    # Save corpus
    corpus.save("/tmp/demo_corpus.json")
    print(f"\nSaved to: /tmp/demo_corpus.json")
    
    # Load it back
    loaded = EvaluationCorpus.load("/tmp/demo_corpus.json")
    print(f"Loaded: {loaded.name} with {len(loaded.samples)} samples")
    
    # Show a sample
    print("\n--- Sample 1 ---")
    s = loaded.samples[0]
    print(f"Prompt: {s.prompt}")
    print(f"Response: {s.response[:100]}...")
    print(f"Claims: {len(s.claims)}")
    for c in s.claims:
        print(f"  [{c.claim_type.value}] {c.text[:50]}... (conf={c.expressed_confidence})")
    
    # Test extraction evaluator with mock data
    print("\n--- Extraction Evaluation (mock) ---")
    
    # Simulate extracted claims
    @dataclass
    class MockExtracted:
        id: str
        text: str
        span_start: int
        span_end: int
        confidence: float
    
    mock_extracted = [
        MockExtracted("e1", "The capital of France is Paris", 0, 31, 0.9),
        MockExtracted("e2", "Paris has been the capital since the 10th century", 33, 82, 0.8),
        # Missing the third claim (false negative)
        MockExtracted("e3", "This is a spurious claim", 200, 230, 0.6),  # False positive
    ]
    
    evaluator = ExtractionEvaluator()
    sample_results = evaluator.evaluate_sample(loaded.samples[0], mock_extracted)
    
    print(f"Precision: {sample_results.precision:.3f}")
    print(f"Recall: {sample_results.recall:.3f}")
    print(f"F1: {sample_results.f1:.3f}")
    print(f"TP={sample_results.true_positives}, FP={sample_results.false_positives}, FN={sample_results.false_negatives}")
    
    print("\n✓ Evaluation framework working")
