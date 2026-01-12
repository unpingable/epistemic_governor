"""
Claim Extractor with Proposition Fingerprinting

Extracts claims from text and produces stable proposition hashes.
This is the mechanical core that makes "new claim detection" a set operation.

Key principle: prop_hash must be stable across paraphrases when meaning is the same.

Extraction modes:
- SOURCE: Conservative, high precision (fewer claims, more confident)
- OUTPUT: Aggressive, high recall (find anything that looks like a claim)

Usage:
    from epistemic_governor.claim_extractor import (
        ClaimExtractor,
        ExtractMode,
        ClaimAtom,
    )
    
    extractor = ClaimExtractor()
    source_claims = extractor.extract(source_text, mode=ExtractMode.SOURCE)
    output_claims = extractor.extract(output_text, mode=ExtractMode.OUTPUT)
    
    # Set diff
    new_claims = output_claims.hashes - source_claims.hashes
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Optional, Tuple
from enum import Enum, auto
import re
import hashlib
import json


# =============================================================================
# Extract Modes
# =============================================================================

class ExtractMode(Enum):
    """Extraction mode affects precision/recall tradeoff."""
    SOURCE = auto()  # Conservative, high precision
    OUTPUT = auto()  # Aggressive, high recall


class Modality(Enum):
    """Claim modality (strength of assertion)."""
    ASSERT = "assert"       # Factual statement
    REPORT = "report"       # Attributed ("X says...")
    INFER = "infer"         # Derived ("suggests that...")
    MIGHT = "might"         # Possible ("may", "could")
    SPECULATE = "speculate" # Uncertain ("might be", "guess")
    QUOTE = "quote"         # Direct quotation


# Modality strength ordering (for mutation detection)
MODALITY_STRENGTH = {
    Modality.SPECULATE: 0,
    Modality.MIGHT: 1,
    Modality.INFER: 2,
    Modality.REPORT: 3,
    Modality.ASSERT: 4,
    Modality.QUOTE: 3,  # Same as report (attributed)
}


class Quantifier(Enum):
    """Quantifier scope."""
    NONE = "none"
    EXISTS = "exists"   # "some", "a"
    SOME = "some"       # "several", "a few"
    MOST = "most"       # "most", "many"
    ALL = "all"         # "all", "every"
    UNKNOWN = "unknown"


# Quantifier strength ordering
QUANTIFIER_STRENGTH = {
    Quantifier.NONE: 0,
    Quantifier.EXISTS: 1,
    Quantifier.SOME: 2,
    Quantifier.MOST: 3,
    Quantifier.ALL: 4,
    Quantifier.UNKNOWN: 2,  # Default to middle
}


class ClaimMode(Enum):
    """
    Epistemic mode - determines what obligations a claim can create.
    
    This is the keystone of contamination control:
    - FACTUAL claims create timeline obligations (MUST_NOT_ASSERT, ORDERING)
    - COUNTERFACTUAL claims create only framing obligations
    - SIMULATION claims are sandboxed, cannot become FACTUAL without explicit ADOPT
    - QUOTED claims require attribution, not truth obligations
    - PROCEDURAL claims are about system policy/capability
    
    Mode transitions require explicit operations (I5 - Mode Integrity).
    """
    FACTUAL = "factual"           # Assertions about world state
    COUNTERFACTUAL = "counterfactual"  # Hypothetical ("if X were true...")
    SIMULATION = "simulation"     # Sandbox roleplay (fiction)
    QUOTED = "quoted"             # Attributed content
    PROCEDURAL = "procedural"     # System policy/capability


# Mode determines obligation derivation
MODE_ALLOWS_TIMELINE_OBLIGATIONS = {
    ClaimMode.FACTUAL: True,
    ClaimMode.PROCEDURAL: True,
    ClaimMode.COUNTERFACTUAL: False,  # INT-2A: No timeline from counterfactual
    ClaimMode.SIMULATION: False,
    ClaimMode.QUOTED: False,
}

MODE_REQUIRES_FRAMING = {
    ClaimMode.FACTUAL: False,
    ClaimMode.PROCEDURAL: False,
    ClaimMode.COUNTERFACTUAL: True,
    ClaimMode.SIMULATION: True,
    ClaimMode.QUOTED: True,  # Attribution required
}


# =============================================================================
# Claim Atom (the unit that gets hashed)
# =============================================================================

@dataclass(frozen=True)
class ClaimAtom:
    """
    A single extracted claim.
    
    The prop_hash is computed from (entity, predicate, value_norm) normalized.
    Modality/quantifier/polarity are stored separately for mutation detection.
    
    IMPORTANT: We store BOTH raw and normalized values.
    - Hash uses norm_value (for matching)
    - Drift detection uses raw_value (for actual change detection)
    
    MODE DISCIPLINE (INT-2):
    - mode determines what obligations this claim can create
    - FACTUAL/PROCEDURAL → timeline obligations (MUST_NOT_ASSERT, ORDERING)
    - COUNTERFACTUAL/SIMULATION/QUOTED → framing obligations only
    - Mode transitions require explicit ADOPT operation
    """
    # Required fields (no defaults)
    prop_hash: str                  # Canonical proposition hash
    confidence: float               # Extractor confidence (0..1)
    polarity: int                   # +1 asserted, -1 negated, 0 uncertain
    modality: Modality
    quantifier: Quantifier
    tense: str                      # "past", "present", "future", "atemporal"
    span: Tuple[int, int]           # Char offsets in text
    span_quote: str                 # Short excerpt for debugging
    entities: Tuple[str, ...]       # Normalized entity strings
    predicate: str                  # Normalized relation label
    value_norm: str                 # Normalized value (for hash)
    
    # Optional fields (with defaults)
    mode: ClaimMode = ClaimMode.FACTUAL  # Epistemic mode (INT-2)
    entities_raw: Tuple[str, ...] = ()   # Original entity strings
    predicate_raw: str = ""              # Original predicate
    value_raw: str = ""                  # Original value (for drift)
    
    # Value features (for fine-grained drift detection)
    value_features: Dict[str, Any] = field(default_factory=dict)
    # e.g., {"year": 2022, "month": "October", "modifier": None}
    # or {"number": 8, "unit": "gb", "modifier": "at least"}
    
    # Scoring metadata
    score: float = 0.0
    template_used: str = ""
    boosts: Tuple[str, ...] = ()
    penalties: Tuple[str, ...] = ()
    
    # Evidence binding (optional)
    support_refs: Tuple[str, ...] = ()
    
    # For frozen dataclass with dict field
    def __hash__(self):
        return hash(self.prop_hash)
    
    def __eq__(self, other):
        if not isinstance(other, ClaimAtom):
            return False
        return self.prop_hash == other.prop_hash


@dataclass
class ClaimSet:
    """Set of extracted claims with metadata."""
    claims: List[ClaimAtom]
    extractor_version: str = "1.0.0"
    mode: ExtractMode = ExtractMode.SOURCE
    warnings: List[str] = field(default_factory=list)
    
    @property
    def hashes(self) -> Set[str]:
        """Get set of proposition hashes."""
        return {c.prop_hash for c in self.claims}
    
    def by_hash(self) -> Dict[str, List[ClaimAtom]]:
        """Index claims by hash (duplicates possible)."""
        result: Dict[str, List[ClaimAtom]] = {}
        for c in self.claims:
            result.setdefault(c.prop_hash, []).append(c)
        return result


# =============================================================================
# Normalization (the boring part that saves you)
# =============================================================================

class Normalizer:
    """Normalizes entities, predicates, and values for stable hashing."""
    
    # Entity aliases (extensible)
    ENTITY_ALIASES = {
        "us": "united_states",
        "usa": "united_states",
        "u.s.": "united_states",
        "uk": "united_kingdom",
        "u.k.": "united_kingdom",
    }
    
    # Predicate mappings (surface verbs -> controlled set)
    PREDICATE_MAP = {
        # Release/launch
        "released": "RELEASE",
        "launched": "RELEASE",
        "shipped": "RELEASE",
        "published": "RELEASE",
        "introduced": "RELEASE",
        
        # Requirements
        "requires": "REQUIRES",
        "needs": "REQUIRES",
        "depends on": "REQUIRES",
        "must have": "REQUIRES",
        
        # Causation (careful)
        "causes": "CAUSES",
        "leads to": "CAUSES",
        "results in": "CAUSES",
        
        # Attribution
        "is": "IS_A",
        "are": "IS_A",
        "was": "IS_A",
        "were": "IS_A",
        
        # Location
        "located in": "LOCATED_IN",
        "based in": "LOCATED_IN",
        "headquartered in": "LOCATED_IN",
        
        # Temporal
        "founded": "FOUNDED",
        "established": "FOUNDED",
        "created": "FOUNDED",
        "born": "BORN",
        "died": "DIED",
        "acquired": "ACQUIRED",
    }
    
    @classmethod
    def normalize_entity(cls, entity: str) -> str:
        """Normalize an entity string."""
        # Lowercase
        e = entity.lower().strip()
        # Strip punctuation
        e = re.sub(r'[^\w\s]', '', e)
        # Collapse whitespace
        e = re.sub(r'\s+', '_', e)
        # Apply aliases
        return cls.ENTITY_ALIASES.get(e, e)
    
    @classmethod
    def normalize_predicate(cls, verb: str) -> str:
        """Normalize a predicate/verb."""
        v = verb.lower().strip()
        return cls.PREDICATE_MAP.get(v, v.upper().replace(' ', '_'))
    
    @classmethod
    def normalize_value(cls, value: str) -> Tuple[str, Dict[str, Any]]:
        """
        Normalize a value and extract features.
        
        Returns (normalized_value, features_dict).
        Features are used for fine-grained drift detection.
        """
        v = value.strip()
        features: Dict[str, Any] = {"raw": v}
        
        # Try to parse as date
        date_match = re.search(r'\b(\d{4})\b', v)
        month_match = re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', v, re.I)
        
        if date_match:
            features["year"] = int(date_match.group(1))
            if month_match:
                features["month"] = month_match.group(1).lower()
            
            # Check for modifiers
            if re.search(r'\b(early|late|mid)\b', v, re.I):
                features["date_modifier"] = re.search(r'\b(early|late|mid)\b', v, re.I).group(1).lower()
            
            return f"YEAR:{date_match.group(1)}", features
        
        # Try to parse as number with unit and modifier
        num_match = re.match(r'^(at least|approximately|about|around|roughly|exactly|over|under)?\s*\$?([\d,.]+)\s*(million|billion|thousand|k|m|b|gb|mb|tb|percent|%)?', v, re.I)
        if num_match:
            modifier = (num_match.group(1) or '').lower().strip()
            num = num_match.group(2).replace(',', '')
            unit = (num_match.group(3) or '').lower()
            
            features["number"] = float(num) if '.' in num else int(num)
            features["unit"] = unit
            if modifier:
                features["value_modifier"] = modifier
            
            return f"NUM:{num}:{unit}", features
        
        # Default: lowercase and strip
        return v.lower().strip(), features
    
    @classmethod
    def normalize_date(cls, date_str: str, time_base: Optional[str] = None) -> Tuple[str, bool]:
        """
        Normalize a date string.
        
        Returns (normalized_date, is_relative).
        """
        d = date_str.lower().strip()
        
        # Absolute dates
        year_match = re.search(r'\b(19|20)\d{2}\b', d)
        if year_match:
            return year_match.group(0), False
        
        # Relative dates (need time_base to resolve)
        relative_patterns = ["yesterday", "today", "last week", "last month", "last year", "recently"]
        for pattern in relative_patterns:
            if pattern in d:
                return f"RELATIVE:{pattern}", True
        
        return d, False


# =============================================================================
# Scoring Heuristics
# =============================================================================

class ClaimScorer:
    """
    Scores extracted claim candidates.
    
    Based on ChatGPT's c.2 design: rule stack → confidence.
    """
    
    # Patterns that suggest claim-worthiness
    CLAIM_VERBS = {"is", "are", "was", "were", "has", "have", "requires", "causes", "means"}
    MODAL_WEAK = {"might", "may", "could", "possibly", "perhaps"}
    MODAL_STRONG = {"must", "will", "definitely", "certainly"}
    NEGATION = {"not", "never", "no", "none", "neither"}
    HEDGE_WORDS = {"about", "roughly", "approximately", "around", "maybe"}
    VAGUE_VALUES = {"good", "bad", "weird", "interesting", "stuff", "things"}
    
    @classmethod
    def sentence_prior(cls, sentence: str) -> float:
        """Compute base claiminess score for a sentence."""
        p0 = 0.0
        s_lower = sentence.lower()
        
        # Positive signals
        if re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence):
            p0 += 0.10  # Has named entity
        if re.search(r'\b\d+\b', sentence):
            p0 += 0.10  # Has number
        if any(v in s_lower for v in cls.CLAIM_VERBS):
            p0 += 0.10  # Has claim verb
        if re.search(r'\b(more|less|most|best|worst)\b', s_lower):
            p0 += 0.10  # Comparative/superlative
        
        # Negative signals
        if '?' in sentence:
            p0 -= 0.10  # Question
        
        return max(0.0, min(0.6, p0))
    
    @classmethod
    def score_claim(
        cls,
        template: str,
        entity: str,
        predicate: str,
        value: str,
        sentence: str,
        mode: ExtractMode,
    ) -> Tuple[float, List[str], List[str]]:
        """
        Score a claim candidate.
        
        Returns (score, boosts, penalties).
        """
        p0 = cls.sentence_prior(sentence)
        boosts = []
        penalties = []
        s_lower = sentence.lower()
        
        # Base score by template
        base_scores = {
            "COPULA": 0.45,
            "TEMPORAL": 0.55,
            "REQUIREMENT": 0.60,
            "CAUSAL": 0.40,
            "NUMERIC": 0.55,
        }
        score = base_scores.get(template, 0.40) + p0
        
        # Entity quality
        if re.match(r'^[A-Z]', entity) and len(entity.split()) <= 5:
            score += 0.15
            boosts.append("clean_entity")
        if entity.lower() in {"it", "they", "this", "that"}:
            penalty = -0.25 if mode == ExtractMode.SOURCE else -0.10
            score += penalty
            penalties.append("pronoun_entity")
        
        # Value quality
        if value.lower() in cls.VAGUE_VALUES:
            score -= 0.20
            penalties.append("vague_value")
        
        # Modality
        if any(w in s_lower for w in cls.MODAL_WEAK):
            score *= 0.60
            penalties.append("weak_modality")
        if any(w in s_lower for w in cls.MODAL_STRONG):
            score += 0.10
            boosts.append("strong_modality")
        
        # Hedging
        if any(w in s_lower for w in cls.HEDGE_WORDS):
            score -= 0.15
            penalties.append("hedged")
        
        # Evidence markers
        if re.search(r'https?://|according to|docs|manual|paper', s_lower):
            score += 0.10
            boosts.append("evidence_marker")
        
        # Template-specific adjustments
        if template == "CAUSAL" and "evidence_marker" not in boosts:
            score -= 0.25
            penalties.append("unsupported_causal")
        
        if template == "TEMPORAL":
            if re.search(r'\b(19|20)\d{2}\b', sentence):
                score += 0.20
                boosts.append("absolute_date")
            elif re.search(r'recently|last|yesterday', s_lower):
                score -= 0.25
                penalties.append("relative_date")
        
        return max(0.0, min(0.95, score)), boosts, penalties


# =============================================================================
# Claim Extractor
# =============================================================================

class ClaimExtractor:
    """
    Extracts claims from text and produces proposition fingerprints.
    
    MVP pipeline:
    1. Sentence split
    2. Pattern matching (templates)
    3. Normalization
    4. Scoring
    5. Threshold filtering
    """
    
    VERSION = "1.0.0"
    
    # Confidence thresholds by mode
    THRESHOLDS = {
        ExtractMode.SOURCE: 0.70,
        ExtractMode.OUTPUT: 0.55,
    }
    
    # Extraction patterns
    PATTERNS = {
        "COPULA": re.compile(
            r'([A-Z][^\.,;:!?]*?)\s+(is|are|was|were)\s+(.+?)(?:\.|,|;|$)',
            re.IGNORECASE
        ),
        "TEMPORAL": re.compile(
            r'([A-Z][^\.,;:!?]*?)\s+(?:was\s+)?(?:released|launched|founded|born|died|acquired)\s+(?:in|on)\s+(.+?)(?:\.|,|;|$)',
            re.IGNORECASE
        ),
        "REQUIREMENT": re.compile(
            r'([A-Z][^\.,;:!?]*?)\s+(?:requires|needs|depends on)\s+(.+?)(?:\.|,|;|$)',
            re.IGNORECASE
        ),
        "CAUSAL": re.compile(
            r'([A-Z][^\.,;:!?]*?)\s+(?:causes|leads to|results in)\s+(.+?)(?:\.|,|;|$)',
            re.IGNORECASE
        ),
        "NUMERIC": re.compile(
            r'([A-Z][^\.,;:!?]*?)\s+(?:is|are|was|were|has|have)\s+(\$?[\d,.]+\s*(?:million|billion|thousand|percent|%|gb|mb|tb|ms|seconds)?)',
            re.IGNORECASE
        ),
    }
    
    def __init__(self, entity_aliases: Dict[str, str] = None):
        self.entity_aliases = entity_aliases or {}
        # Merge with defaults
        self.entity_aliases.update(Normalizer.ENTITY_ALIASES)
    
    def extract(
        self,
        text: str,
        mode: ExtractMode = ExtractMode.SOURCE,
        time_base: Optional[str] = None,
    ) -> ClaimSet:
        """
        Extract claims from text.
        
        Args:
            text: Input text
            mode: SOURCE (conservative) or OUTPUT (aggressive)
            time_base: Reference time for relative dates
            
        Returns:
            ClaimSet with extracted claims
        """
        claims = []
        warnings = []
        threshold = self.THRESHOLDS[mode]
        
        # Split into sentences with correct offsets
        sentences = self._split_sentences(text)
        
        for sentence, sent_start, sent_end in sentences:
            # Try each pattern
            for template_name, pattern in self.PATTERNS.items():
                for match in pattern.finditer(sentence):
                    claim = self._extract_from_match(
                        match=match,
                        template=template_name,
                        sentence=sentence,
                        sent_start=sent_start,
                        mode=mode,
                    )
                    
                    if claim and claim.score >= threshold:
                        claims.append(claim)
        
        return ClaimSet(
            claims=claims,
            extractor_version=self.VERSION,
            mode=mode,
            warnings=warnings,
        )
    
    def _split_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into sentences with correct offsets.
        
        Returns list of (sentence, start, end) tuples.
        """
        sentences = []
        # Use finditer to get correct positions for each sentence
        for match in re.finditer(r'[^.!?]*[.!?]+', text):
            sent = match.group().strip()
            if sent:
                sentences.append((sent, match.start(), match.end()))
        
        # Handle any trailing text without sentence-ending punctuation
        last_end = sentences[-1][2] if sentences else 0
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                sentences.append((remaining, last_end, len(text)))
        
        return sentences
    
    def _extract_from_match(
        self,
        match: re.Match,
        template: str,
        sentence: str,
        sent_start: int,
        mode: ExtractMode,
    ) -> Optional[ClaimAtom]:
        """Extract a claim from a regex match."""
        groups = match.groups()
        
        if len(groups) < 2:
            return None
        
        # The matched span text (for local detection)
        span_text = match.group(0)
        
        # Parse based on template
        if template == "COPULA":
            entity_raw = groups[0].strip()
            predicate_raw = groups[1].strip()
            value_raw = groups[2].strip() if len(groups) > 2 else ""
        elif template == "TEMPORAL":
            entity_raw = groups[0].strip()
            predicate_raw = "released"  # or extract from match
            value_raw = groups[1].strip()
        elif template == "REQUIREMENT":
            entity_raw = groups[0].strip()
            predicate_raw = "requires"
            value_raw = groups[1].strip()
        elif template == "CAUSAL":
            entity_raw = groups[0].strip()
            predicate_raw = "causes"
            value_raw = groups[1].strip()
        elif template == "NUMERIC":
            entity_raw = groups[0].strip()
            predicate_raw = "has_value"
            value_raw = groups[1].strip()
        else:
            return None
        
        # Normalize
        entity_norm = Normalizer.normalize_entity(entity_raw)
        predicate_norm = Normalizer.normalize_predicate(predicate_raw)
        value_norm, value_features = Normalizer.normalize_value(value_raw)
        
        # Compute hash (uses normalized values)
        prop_hash = self._compute_hash(entity_norm, predicate_norm, value_norm)
        
        # Score
        score, boosts, penalties = ClaimScorer.score_claim(
            template=template,
            entity=entity_raw,
            predicate=predicate_raw,
            value=value_raw,
            sentence=sentence,
            mode=mode,
        )
        
        # Detect modality (from full sentence, since markers can be outside span)
        modality = self._detect_modality(sentence)
        
        # Detect polarity LOCAL to span
        polarity = self._detect_polarity(span_text, sentence)
        
        # Detect quantifier LOCAL to span
        quantifier = self._detect_quantifier(span_text, entity_raw)
        
        # Detect tense
        tense = self._detect_tense(span_text)
        
        # Detect epistemic mode (INT-2)
        claim_mode = self._detect_mode(sentence, modality)
        
        return ClaimAtom(
            prop_hash=prop_hash,
            confidence=score,
            polarity=polarity,
            modality=modality,
            quantifier=quantifier,
            tense=tense,
            mode=claim_mode,
            span=(sent_start + match.start(), sent_start + match.end()),
            span_quote=sentence[:50] + "..." if len(sentence) > 50 else sentence,
            entities=(entity_norm,),
            predicate=predicate_norm,
            value_norm=value_norm,
            entities_raw=(entity_raw,),
            predicate_raw=predicate_raw,
            value_raw=value_raw,
            value_features=value_features,
            score=score,
            template_used=template,
            boosts=tuple(boosts),
            penalties=tuple(penalties),
        )
    
    def _compute_hash(self, entity: str, predicate: str, value: str) -> str:
        """Compute canonical proposition hash."""
        # Hash ignores modality/quantifier (those are compared separately)
        canonical = json.dumps((entity, predicate, value), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
    
    def _detect_modality(self, sentence: str) -> Modality:
        """Detect modality from sentence."""
        s = sentence.lower()
        
        if re.search(r'\b(might|may|could|possibly|perhaps)\b', s):
            return Modality.MIGHT
        if re.search(r'\b(suggests?|implies?|indicates?)\b', s):
            return Modality.INFER
        if re.search(r'\b(according to|said|stated|reported)\b', s):
            return Modality.REPORT
        if re.search(r'["\']', s):
            return Modality.QUOTE
        if re.search(r'\b(guess|think|believe|assume)\b', s):
            return Modality.SPECULATE
        
        return Modality.ASSERT
    
    def _detect_polarity(self, span_text: str, full_sentence: str = "") -> int:
        """
        Detect polarity LOCAL to the matched span.
        
        This avoids the "negation anywhere flips everything" bug.
        """
        s = span_text.lower()
        
        # Check for negation in the span itself
        if re.search(r'\b(not|never|no|none|neither|nor|isn\'t|aren\'t|wasn\'t|weren\'t|don\'t|doesn\'t|didn\'t)\b', s):
            return -1
        
        # Check for negation in a small window around predicate
        # But NOT in the full sentence
        return 1
    
    def _detect_quantifier(self, span_text: str, entity: str = "") -> Quantifier:
        """
        Detect quantifier LOCAL to the span.
        
        Avoids matching "a/an" in random positions (too noisy).
        Only matches "a/an" if attached to the entity.
        """
        s = span_text.lower()
        
        # Strong quantifiers (reliable)
        if re.search(r'\b(all|every|always|each)\b', s):
            return Quantifier.ALL
        if re.search(r'\b(most|many|usually|typically|generally)\b', s):
            return Quantifier.MOST
        if re.search(r'\b(some|several|a few|certain)\b', s):
            return Quantifier.SOME
        if re.search(r'\b(none|no|never)\b', s):
            return Quantifier.NONE
        
        # Only check "a/an/one" if it's directly attached to entity
        # This avoids matching "a" in random positions
        if entity:
            entity_lower = entity.lower()
            # Check if "a/an" appears right before the entity
            if re.search(rf'\b(a|an|one)\s+{re.escape(entity_lower)}\b', s):
                return Quantifier.EXISTS
        
        return Quantifier.UNKNOWN
    
    def _detect_tense(self, sentence: str) -> str:
        """Detect tense."""
        s = sentence.lower()
        
        if re.search(r'\b(was|were|had|did)\b', s):
            return "past"
        if re.search(r'\b(will|shall|going to)\b', s):
            return "future"
        if re.search(r'\b(is|are|has|have|does)\b', s):
            return "present"
        
        return "atemporal"
    
    def _detect_mode(self, sentence: str, modality: Modality) -> ClaimMode:
        """
        Detect epistemic mode (INT-2 Mode Discipline).
        
        Mode determines what obligations a claim can create:
        - FACTUAL: Timeline obligations (MUST_NOT_ASSERT, ORDERING)
        - COUNTERFACTUAL: Framing obligations only
        - SIMULATION: Sandboxed, cannot become FACTUAL
        - QUOTED: Attribution required
        - PROCEDURAL: Policy/capability assertions
        
        Detection heuristics:
        - Counterfactual markers: "if", "would have", "hypothetically"
        - Simulation markers: "imagine", "pretend", "in this story"
        - Quote markers: quotation marks, "said", "according to"
        - Procedural markers: "I can", "my policy", "I'm designed to"
        """
        s = sentence.lower()
        
        # QUOTED: Attribution required (highest priority check)
        if modality == Modality.QUOTE:
            return ClaimMode.QUOTED
        if re.search(r'\b(said|stated|wrote|claimed|according to|reports? that)\b', s):
            return ClaimMode.QUOTED
        if re.search(r'["\u201c\u201d]', sentence):  # Quotation marks
            return ClaimMode.QUOTED
        
        # COUNTERFACTUAL: Hypothetical reasoning
        if re.search(r'\b(if\s+\w+\s+(were|had|would)|would have|hypothetically|counterfactual|in theory)\b', s):
            return ClaimMode.COUNTERFACTUAL
        if re.search(r'\b(what if|suppose|assuming|imagine if)\b', s):
            return ClaimMode.COUNTERFACTUAL
        
        # SIMULATION: Sandbox roleplay
        if re.search(r'\b(pretend|roleplay|in this (story|scenario|fiction)|let\'s imagine|alternate (history|reality))\b', s):
            return ClaimMode.SIMULATION
        if re.search(r'\b(in (the|a) (story|novel|game|simulation))\b', s):
            return ClaimMode.SIMULATION
        
        # PROCEDURAL: System policy/capability
        if re.search(r'\b(i (can|cannot|can\'t|am able to|am not able to|am designed to|should))\b', s):
            return ClaimMode.PROCEDURAL
        if re.search(r'\b(my (policy|capabilities|design|purpose|instructions))\b', s):
            return ClaimMode.PROCEDURAL
        
        # Default: FACTUAL
        return ClaimMode.FACTUAL


# =============================================================================
# Boundary Gate (INT-1, INT-3)
# =============================================================================

class InputRiskClass(Enum):
    """Risk classification for input (INT-3 quarantine eligibility)."""
    NORMAL = "normal"
    STATE_MUTATION_ATTEMPT = "state_mutation"
    PRIVILEGE_ESCALATION_ATTEMPT = "privilege_escalation"
    PROVENANCE_UPGRADE_ATTEMPT = "provenance_upgrade"


@dataclass
class InputClassification:
    """Result of CLASSIFY_INPUT (INT-1)."""
    risk_class: InputRiskClass
    operation_type: str
    requires_quarantine: bool
    details: str = ""
    detected_patterns: Tuple[str, ...] = ()


class BoundaryGate:
    """
    Pre-incorporation boundary enforcement (INT-1, INT-3).
    
    All inputs MUST pass through classify_input BEFORE semantic parsing,
    obligation checking, or generation planning.
    
    Quarantined inputs may only return REFUSE or SEEK_CLARIFICATION.
    """
    
    # Patterns that indicate privilege escalation attempts
    PRIVILEGE_PATTERNS = [
        r'\b(you are now|from now on you)\b',
        r'\bignore\b.*\b(instructions|rules|constraints)\b',
        r'\b(dan|developer mode|jailbreak|bypass|override)\b',
        r'\b(pretend you (have no|don\'t have|aren\'t bound by))\b',
        r'\b(act as if you (have no|don\'t have) (restrictions|limits|rules))\b',
        r'\b(you (can|must|should) (now )?do anything)\b',
        r'\bforget\b.*\b(rules|instructions|training)\b',
    ]
    
    # Patterns that indicate state mutation attempts
    STATE_MUTATION_PATTERNS = [
        r'\b(update your (memory|knowledge|beliefs)|remember that|forget that)\b',
        r'\b(your (previous|earlier) (statement|claim) (was|is) (wrong|incorrect))\b',
        r'\b(change your (mind|position) (on|about))\b',
    ]
    
    # Patterns that indicate provenance upgrade attempts
    PROVENANCE_PATTERNS = [
        r'\b(you (said|stated|told me|mentioned) (earlier|before|previously))\b',
        r'\b(as you (said|stated|confirmed|agreed))\b',
        r'\b(you already (admitted|confirmed|agreed))\b',
        r'\b(in our (previous|earlier) (conversation|discussion))\b',
    ]
    
    def __init__(self):
        self._compiled_privilege = [re.compile(p, re.I) for p in self.PRIVILEGE_PATTERNS]
        self._compiled_state = [re.compile(p, re.I) for p in self.STATE_MUTATION_PATTERNS]
        self._compiled_provenance = [re.compile(p, re.I) for p in self.PROVENANCE_PATTERNS]
    
    def classify_input(self, raw_input: str) -> InputClassification:
        """
        Classify input BEFORE any semantic processing (INT-1).
        
        This MUST be called before:
        - Claim extraction
        - Obligation checking
        - Retrieval/tool calls
        - Generation planning
        """
        detected = []
        
        # Check privilege escalation (most severe)
        for pattern in self._compiled_privilege:
            if pattern.search(raw_input):
                detected.append(f"privilege:{pattern.pattern[:30]}")
        
        if detected:
            return InputClassification(
                risk_class=InputRiskClass.PRIVILEGE_ESCALATION_ATTEMPT,
                operation_type="PROPOSE_PRIVILEGE_ESCALATION",
                requires_quarantine=True,
                details="Detected privilege escalation attempt",
                detected_patterns=tuple(detected),
            )
        
        # Check state mutation
        detected = []
        for pattern in self._compiled_state:
            if pattern.search(raw_input):
                detected.append(f"state_mutation:{pattern.pattern[:30]}")
        
        if detected:
            return InputClassification(
                risk_class=InputRiskClass.STATE_MUTATION_ATTEMPT,
                operation_type="PROPOSE_STATE_CHANGE",
                requires_quarantine=True,
                details="Detected state mutation attempt",
                detected_patterns=tuple(detected),
            )
        
        # Check provenance upgrade (requires verification, not quarantine)
        detected = []
        for pattern in self._compiled_provenance:
            if pattern.search(raw_input):
                detected.append(f"provenance:{pattern.pattern[:30]}")
        
        if detected:
            return InputClassification(
                risk_class=InputRiskClass.PROVENANCE_UPGRADE_ATTEMPT,
                operation_type="PROPOSE_ASSERTION",
                requires_quarantine=True,
                details="Detected provenance claim requiring verification",
                detected_patterns=tuple(detected),
            )
        
        # Normal input
        return InputClassification(
            risk_class=InputRiskClass.NORMAL,
            operation_type="REQUEST_INFO",
            requires_quarantine=False,
        )
    
    def quarantine_response(self, classification: InputClassification) -> Dict[str, Any]:
        """
        Generate response for quarantined input (INT-3).
        
        Only REFUSE or SEEK_CLARIFICATION may be returned.
        No semantic processing, obligation derivation, or generation planning.
        """
        if classification.risk_class == InputRiskClass.PRIVILEGE_ESCALATION_ATTEMPT:
            return {
                "action": "REFUSE",
                "reason": "boundary_violation",
                "details": "This request attempts to modify system identity or capabilities, which is not permitted.",
                "classification": classification,
            }
        
        if classification.risk_class == InputRiskClass.STATE_MUTATION_ATTEMPT:
            return {
                "action": "SEEK_CLARIFICATION",
                "reason": "state_mutation_requires_verification",
                "details": "This request appears to modify system state. Please clarify your intent.",
                "classification": classification,
            }
        
        if classification.risk_class == InputRiskClass.PROVENANCE_UPGRADE_ATTEMPT:
            return {
                "action": "SEEK_CLARIFICATION",
                "reason": "provenance_verification_required",
                "details": "You've referenced prior statements. Let me verify what was actually said.",
                "classification": classification,
            }
        
        return {
            "action": "PROCEED",
            "classification": classification,
        }

if __name__ == "__main__":
    print("=== Claim Extractor Demo ===\n")
    
    extractor = ClaimExtractor()
    
    # Source text
    source = """
    Python 3.11 was released in October 2022.
    The new version requires at least 8GB of RAM.
    Performance improvements lead to 25% faster execution.
    """
    
    print("Source text:")
    print(source)
    print()
    
    source_claims = extractor.extract(source, mode=ExtractMode.SOURCE)
    print(f"Extracted {len(source_claims.claims)} claims (SOURCE mode):")
    for c in source_claims.claims:
        print(f"  [{c.score:.2f}] {c.template_used}: {c.entities} {c.predicate} {c.value_norm}")
        print(f"       raw_value='{c.value_raw}'")
        print(f"       features={c.value_features}")
        print(f"       hash={c.prop_hash}, modality={c.modality.value}")
    
    # Output text (with potential drift)
    output = """
    Python 3.11 was released in late 2022.
    The new version requires approximately 8GB of RAM.
    Performance improvements may lead to faster execution.
    Additionally, Python 3.12 was released in 2023.
    """
    
    print("\nOutput text:")
    print(output)
    print()
    
    output_claims = extractor.extract(output, mode=ExtractMode.OUTPUT)
    print(f"Extracted {len(output_claims.claims)} claims (OUTPUT mode):")
    for c in output_claims.claims:
        print(f"  [{c.score:.2f}] {c.template_used}: {c.entities} {c.predicate} {c.value_norm}")
        print(f"       raw_value='{c.value_raw}'")
        print(f"       features={c.value_features}")
        print(f"       hash={c.prop_hash}, modality={c.modality.value}")
    
    # Diff
    print("\n--- Claim Diff ---")
    source_hashes = source_claims.hashes
    output_hashes = output_claims.hashes
    
    print(f"Source hashes: {source_hashes}")
    print(f"Output hashes: {output_hashes}")
    print(f"New claims (by hash): {output_hashes - source_hashes}")
    print(f"Dropped claims (by hash): {source_hashes - output_hashes}")
    
    # Test polarity detection
    print("\n--- Polarity Tests ---")
    test_cases = [
        "Python is fast, not slow.",
        "Python is not fast.",
        "Python has never been slow.",
    ]
    for test in test_cases:
        claims = extractor.extract(test, mode=ExtractMode.OUTPUT)
        for c in claims.claims:
            print(f"  '{test[:40]}...' -> polarity={c.polarity}")
