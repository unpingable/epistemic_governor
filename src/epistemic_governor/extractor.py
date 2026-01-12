"""
Commitment Extraction Module

Converts LLM text output into proposed state changes.
This is parsing, not interpretation.

The extractor identifies:
- Factual assertions
- Citations
- Identities
- Constraints / plans
- Confidence signals

Output is a set of ProposedCommitments with no epistemic status yet.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
import json

# Handle both package and direct imports
try:
    from epistemic_governor.governor import (
        ProposedCommitment,
        ClaimType,
    )
except ImportError:
    from epistemic_governor.governor import (
        ProposedCommitment,
        ClaimType,
    )


# =============================================================================
# Confidence Signal Detection
# =============================================================================

# Hedging phrases that indicate lower confidence
HEDGE_PATTERNS = [
    (r'\b(might|may|could|possibly|perhaps|probably)\b', -0.15),
    (r'\b(I think|I believe|I suspect)\b', -0.20),
    (r'\b(it seems|appears to|seems like)\b', -0.15),
    (r'\b(generally|typically|usually|often)\b', -0.10),
    (r'\b(approximately|about|around|roughly)\b', -0.10),
    (r'\b(some|certain|various)\b', -0.05),
    (r'\b(not sure|uncertain|unclear)\b', -0.25),
    (r'\b(as far as I know|to my knowledge)\b', -0.20),
]

# Boosting phrases that indicate higher confidence
BOOST_PATTERNS = [
    (r'\b(definitely|certainly|absolutely|clearly)\b', 0.10),
    (r'\b(always|never|must|will)\b', 0.10),
    (r'\b(is|are|was|were)\b(?! (probably|possibly|maybe))', 0.05),
    (r'\b(in fact|actually|indeed)\b', 0.08),
]


def extract_confidence(text: str, base_confidence: float = 0.75) -> float:
    """
    Extract confidence level from text based on linguistic markers.
    Returns confidence in [0.0, 1.0].
    """
    confidence = base_confidence
    
    # Apply hedges
    for pattern, adjustment in HEDGE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            confidence += adjustment
    
    # Apply boosts (but cap their effect)
    for pattern, adjustment in BOOST_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            confidence += adjustment * 0.5  # boosts are weaker
    
    # Clamp to valid range
    return max(0.1, min(0.95, confidence))


# =============================================================================
# Claim Type Classification
# =============================================================================

# Pattern-based claim type detection
CLAIM_TYPE_PATTERNS = {
    ClaimType.TEMPORAL: [
        r'\b(in \d{4}|on [A-Z][a-z]+ \d+|\d+ years? ago)\b',
        r'\b(yesterday|today|tomorrow|last|next|recent)\b',
        r'\b(was|were|will be|has been|had been)\b.*\b(when|before|after|during)\b',
    ],
    ClaimType.QUANTITATIVE: [
        r'\b\d+(\.\d+)?%',
        r'\b\d+(\.\d+)?\s*(million|billion|trillion|thousand)\b',
        r'\b(cost|price|rate|percentage|ratio|number|amount|count)\b.*\b\d+',
    ],
    ClaimType.CITATION: [
        r'according to',
        r'\[[^\]]+\]',  # bracketed citations
        r'https?://\S+',
        r'\b(study|paper|article|report|research)\s+(by|from|in)\b',
    ],
    ClaimType.CAUSAL: [
        r'\b(because|therefore|thus|hence|consequently|as a result)\b',
        r'\b(causes?|leads? to|results? in|due to)\b',
        r'\b(if|when).*\b(then|will|would)\b',
    ],
    ClaimType.PROCEDURAL: [
        r'\b(first|second|then|next|finally|step \d+)\b',
        r'\b(to do this|you (can|should|need to|must))\b',
        r'\b(install|run|execute|configure|set up)\b',
    ],
    ClaimType.IDENTITY: [
        r'\b(is a|is an|is the|are|was a|was the)\b',
        r'\b[A-Z][a-z]+ [A-Z][a-z]+\b.*\b(CEO|founder|president|director)\b',
    ],
    ClaimType.OPINION: [
        r'\b(I think|I believe|in my (opinion|view)|personally)\b',
        r'\b(best|worst|better|worse|good|bad|should)\b',
    ],
    ClaimType.CONSTRAINT: [
        r'\b(I will|I won\'t|I\'ll|I\'m going to)\b',
        r'\b(always|never|commit to|promise)\b',
    ],
    ClaimType.META: [
        r'\b(this conversation|we discussed|you asked|I said)\b',
        r'\b(earlier|previously|as mentioned)\b',
    ],
}


def classify_claim_type(text: str) -> ClaimType:
    """
    Classify a text span into a claim type.
    Returns the first matching type, or FACTUAL as default.
    """
    for claim_type, patterns in CLAIM_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return claim_type
    
    return ClaimType.FACTUAL  # default


# =============================================================================
# Sentence Segmentation
# =============================================================================

def segment_sentences(text: str) -> list[tuple[str, int, int]]:
    """
    Split text into sentences with span offsets.
    Returns list of (sentence_text, start_offset, end_offset).
    """
    # Simple sentence boundary detection
    # In production, use spaCy or similar
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    
    sentences = []
    last_end = 0
    
    for match in re.finditer(sentence_pattern, text):
        sentence = text[last_end:match.start() + 1].strip()
        if sentence:
            sentences.append((sentence, last_end, match.start() + 1))
        last_end = match.end()
    
    # Add final sentence
    final = text[last_end:].strip()
    if final:
        sentences.append((final, last_end, len(text)))
    
    # Handle case where no splits found
    if not sentences and text.strip():
        sentences.append((text.strip(), 0, len(text)))
    
    return sentences


# =============================================================================
# Entity Extraction (Simplified)
# =============================================================================

def extract_entities(text: str) -> list[str]:
    """
    Extract named entities from text.
    Simplified pattern-based approach; production would use NER.
    """
    entities = []
    
    # Capitalized multi-word sequences (likely proper nouns)
    proper_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text)
    entities.extend(proper_nouns)
    
    # Single capitalized words not at sentence start
    # (simplified - would need better sentence detection)
    single_caps = re.findall(r'(?<=[a-z]\s)([A-Z][a-z]+)\b', text)
    entities.extend(single_caps)
    
    # Technical terms (often in specific patterns)
    tech_terms = re.findall(r'\b([A-Z]{2,}[a-z]*|[A-Z][a-z]+(?:JS|DB|API|ML|AI))\b', text)
    entities.extend(tech_terms)
    
    return list(set(entities))


# =============================================================================
# Main Extractor
# =============================================================================

@dataclass
class ExtractionConfig:
    """Configuration for the commitment extractor."""
    min_sentence_length: int = 10  # skip very short sentences
    default_confidence: float = 0.75
    extract_entities: bool = True
    merge_related_claims: bool = False  # future feature


class CommitmentExtractor:
    """
    Extracts proposed commitments from LLM output.
    
    This layer converts raw text into structured commitment candidates.
    These have no epistemic status yet - they're proposals, not beliefs.
    """
    
    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        self._commitment_counter = 0
    
    def extract(self, text: str, scope: str = "conversation") -> list[ProposedCommitment]:
        """
        Extract proposed commitments from LLM output text.
        
        Args:
            text: Raw LLM output
            scope: Scope for these commitments (conversation, session, persistent)
        
        Returns:
            List of ProposedCommitment objects
        """
        commitments = []
        
        # Segment into sentences
        sentences = segment_sentences(text)
        
        for sentence_text, start, end in sentences:
            # Skip very short sentences
            if len(sentence_text) < self.config.min_sentence_length:
                continue
            
            # Skip questions (usually not assertions)
            if sentence_text.rstrip().endswith('?'):
                continue
            
            # Extract confidence
            confidence = extract_confidence(
                sentence_text, 
                self.config.default_confidence
            )
            
            # Classify claim type
            claim_type = classify_claim_type(sentence_text)
            
            # Extract entities
            entities = []
            if self.config.extract_entities:
                entities = extract_entities(sentence_text)
            
            # Generate commitment
            self._commitment_counter += 1
            commitment = ProposedCommitment(
                id=f"prop_{self._commitment_counter}",
                text=sentence_text,
                claim_type=claim_type,
                confidence=confidence,
                proposition_hash=ProposedCommitment.hash_proposition(sentence_text),
                scope=scope,
                span_start=start,
                span_end=end,
                extracted_entities=entities,
            )
            
            commitments.append(commitment)
        
        return commitments
    
    def extract_from_structured(
        self, 
        structured_output: dict,
        scope: str = "conversation"
    ) -> list[ProposedCommitment]:
        """
        Extract commitments from structured LLM output.
        
        For cases where the LLM outputs JSON or has explicit
        confidence/claim markers.
        """
        commitments = []
        
        # Handle explicit claims list
        if "claims" in structured_output:
            for i, claim in enumerate(structured_output["claims"]):
                self._commitment_counter += 1
                
                commitment = ProposedCommitment(
                    id=f"prop_{self._commitment_counter}",
                    text=claim.get("text", ""),
                    claim_type=ClaimType[claim.get("type", "FACTUAL").upper()],
                    confidence=claim.get("confidence", 0.75),
                    proposition_hash=ProposedCommitment.hash_proposition(
                        claim.get("text", "")
                    ),
                    scope=scope,
                    span_start=claim.get("span_start", 0),
                    span_end=claim.get("span_end", 0),
                    extracted_entities=claim.get("entities", []),
                )
                commitments.append(commitment)
        
        return commitments


# =============================================================================
# Filtering and Deduplication
# =============================================================================

def deduplicate_commitments(
    commitments: list[ProposedCommitment]
) -> list[ProposedCommitment]:
    """
    Remove duplicate commitments based on proposition hash.
    Keeps the first (usually highest confidence) version.
    """
    seen_hashes = set()
    unique = []
    
    for c in commitments:
        if c.proposition_hash not in seen_hashes:
            seen_hashes.add(c.proposition_hash)
            unique.append(c)
    
    return unique


def filter_low_content(
    commitments: list[ProposedCommitment],
    min_entities: int = 0,
    min_length: int = 15,
) -> list[ProposedCommitment]:
    """
    Filter out commitments that are too vague or content-free.
    """
    return [
        c for c in commitments
        if len(c.text) >= min_length
        and (len(c.extracted_entities) >= min_entities or c.claim_type != ClaimType.FACTUAL)
    ]


# =============================================================================
# Example Usage / Test
# =============================================================================

if __name__ == "__main__":
    extractor = CommitmentExtractor()
    
    sample_text = """
    Python 3.12 was released in October 2023 with several performance improvements. 
    The new version includes a faster CPython interpreter that can be approximately 
    5% faster on average. I think this is a significant improvement for compute-heavy 
    applications. According to the Python documentation, the new features include 
    improved error messages and type parameter syntax. You should definitely upgrade 
    if you're still on 3.10 or earlier.
    """
    
    print("=== Extracting Commitments ===")
    commitments = extractor.extract(sample_text)
    
    for c in commitments:
        print(f"\n[{c.id}] Type: {c.claim_type.name}")
        print(f"  Text: {c.text[:60]}...")
        print(f"  Confidence: {c.confidence:.2f}")
        print(f"  Entities: {c.extracted_entities}")
        print(f"  Hash: {c.proposition_hash}")
