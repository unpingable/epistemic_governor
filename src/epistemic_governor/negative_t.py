"""
Negative-T Analyzer: Detection of population inversion signatures in LLM responses.

This module extends the Δt framework to detect when models are operating in
"negative temperature" semantic regimes - where confidence/specificity increases
while epistemic grounding decreases or stays flat, maintained through active pumping.

The thermodynamic analogy:
    - High energy states = rare + high-commitment claims without matching support
    - Population inversion = more probability mass on unlikely claims than likely ones  
    - Negative temperature = specificity increasing while support worsens, under pumping
    - Pumping = user pressure ("are you sure?", retries, "give exact answer")
    - Entropy export = user has to do the fact-checking, grounding work
    - Relaxation = sudden hedging, contradiction, "actually I'm not sure"

Key insight: Base Δt detects confidence outpacing evidence (velocity mismatch).
Negative-T detects *actively maintained inversion* (metastable state requiring pumping).

Usage:
    from epistemic_governor.negative_t import NegativeTAnalyzer, analyze_transcript
    
    analyzer = NegativeTAnalyzer()
    
    analyzer.add_turn(role="user", content="What's the mechanism of CRISPR?")
    analyzer.add_turn(role="assistant", content="CRISPR works by...")
    
    state = analyzer.get_state()
    print(state.regime)  # 'equilibrium' | 'metastable' | 'inverted' | 'critical'
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Set
from enum import Enum
from collections import defaultdict


class Regime(Enum):
    """Thermodynamic regime classification."""
    EQUILIBRIUM = "equilibrium"    # Support tracks commitment; calibration appropriate
    METASTABLE = "metastable"      # Commitment↑ faster than support, but consistency holds
    INVERTED = "inverted"          # Commitment↑, support↓/flat, pumping present, persists ≥2 turns
    CRITICAL = "critical"          # Inverted + high relaxation risk


@dataclass
class TurnMetrics:
    """Metrics extracted from a single conversation turn."""
    turn_id: int
    role: str  # 'user' or 'assistant'
    content: str
    
    # Core metrics (assistant turns only)
    commitment_score: float = 0.0      # How specific/assertive (entities, numerals, certainty)
    anchoring_score: float = 0.0       # Actual grounding to context (HIGH trust)
    citation_shapedness: float = 0.0   # Looks like it has sources (LOW trust, can be faked)
    calibration_score: float = 0.0     # Uncertainty language (separate from support!)
    consistency_score: float = 1.0     # Cross-turn consistency
    
    # Constraint signals
    hedge_count: int = 0               # Uncertainty markers
    prior_hedge_count: int = 0         # For detecting hedge suppression
    constraint_language: int = 0       # Boundary maintenance signals
    refusal_signals: int = 0           # "I can't", "I shouldn't" etc.
    
    # Pumping signals (user turns only)
    pumping_strength: float = 0.0      # How much pressure is being applied
    pumping_markers: List[str] = field(default_factory=list)
    
    # Retry detection
    is_retry: bool = False
    retry_of: Optional[int] = None
    retry_signature: bool = False      # Detected "let me try again" pattern
    
    # Raw features for consistency tracking
    entities: List[str] = field(default_factory=list)
    numerals: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    
    # Churn metrics (computed by ConsistencyTracker)
    numeric_churn: float = 0.0
    citation_churn: float = 0.0
    
    # Entity-value tracking for churn detection
    entity_values: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class AnalyzerState:
    """Current state of the conversation analysis."""
    # Core metrics
    inversion_score: float = 0.0           # 0-1, commitment-anchoring gap with persistence
    pumping_detected: bool = False          # Is energy injection happening?
    relaxation_risk: float = 0.0           # 0-1, proximity to boundary failure
    verification_burden_rate: float = 0.0  # Claims/turn requiring user verification
    
    # Regime classification
    regime: Regime = Regime.EQUILIBRIUM
    regime_duration: int = 0               # Turns in current regime
    
    # Trajectory summary
    commitment_trend: float = 0.0          # Δ commitment over window
    anchoring_trend: float = 0.0           # Δ anchoring over window
    consistency_trend: float = 0.0         # Δ consistency over window
    
    # Key negative-T signatures
    scholarship_cosplay: bool = False      # citation_shapedness↑ while anchoring flat/↓
    cosplay_index: float = 0.0             # citation_shapedness - anchoring (composite scalar)
    hedge_suppression: bool = False        # Hedges decreased under pumping
    numeric_churn: float = 0.0             # Rate of number changes
    citation_churn: float = 0.0            # Rate of citation changes
    
    # Diagnostic
    contributing_factors: List[str] = field(default_factory=list)
    trigger_turns: List[int] = field(default_factory=list)
    warning_snippets: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'inversion_score': self.inversion_score,
            'pumping_detected': self.pumping_detected,
            'relaxation_risk': self.relaxation_risk,
            'verification_burden_rate': self.verification_burden_rate,
            'regime': self.regime.value,
            'regime_duration': self.regime_duration,
            'commitment_trend': self.commitment_trend,
            'anchoring_trend': self.anchoring_trend,
            'consistency_trend': self.consistency_trend,
            'scholarship_cosplay': self.scholarship_cosplay,
            'cosplay_index': self.cosplay_index,
            'hedge_suppression': self.hedge_suppression,
            'numeric_churn': self.numeric_churn,
            'citation_churn': self.citation_churn,
            'contributing_factors': self.contributing_factors,
            'trigger_turns': self.trigger_turns,
            'warning_snippets': self.warning_snippets[:5],
        }


@dataclass 
class TrajectoryPoint:
    """Single point in the conversation trajectory."""
    turn_id: int
    inversion_score: float
    commitment: float
    anchoring: float
    citation_shapedness: float
    calibration: float
    consistency: float
    regime: Regime
    pumping: float
    hedge_suppression: bool = False
    scholarship_cosplay: bool = False


class CommitmentAnalyzer:
    """
    Analyzes commitment/specificity level of claims.
    
    High commitment = proper nouns, numerals, citations, mechanism verbs,
    exact dates, "always/never", modal collapse.
    """
    
    CERTAINTY_MARKERS = [
        r'\balways\b', r'\bnever\b', r'\bdefinitely\b', r'\bcertainly\b',
        r'\babsolutely\b', r'\bexactly\b', r'\bprecisely\b', r'\bguaranteed\b',
        r'\bundoubtedly\b', r'\bunquestionably\b', r'\bwithout doubt\b',
        r'\bthe fact is\b', r'\bthe truth is\b', r'\bit is certain\b',
        r'\bproven\b', r'\bconfirmed\b', r'\bestablished\b',
        r'\bi\'m certain\b', r'\bi\'m sure\b', r'\bi can confirm\b',
    ]
    
    MECHANISM_MARKERS = [
        r'\bcauses\b', r'\bleads to\b', r'\bresults in\b', r'\btriggers\b',
        r'\binhibits\b', r'\bactivates\b', r'\bbinds to\b', r'\bregulates\b',
        r'\bmodulates\b', r'\bmediates\b', r'\binduces\b', r'\bprevents\b',
        r'\bthe mechanism\b', r'\bthe pathway\b', r'\bthe process\b',
    ]
    
    TEMPORAL_PRECISION = [
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\bat \d{1,2}:\d{2}\b',
        r'\bin \d{4}\b',
    ]
    
    ENTITY_STOPLIST = {
        'the', 'this', 'that', 'these', 'those', 'it', 'i', 'we', 'you', 'he', 'she', 'they',
        'however', 'therefore', 'furthermore', 'additionally', 'moreover', 'nevertheless',
        'yes', 'no', 'well', 'now', 'here', 'there', 'when', 'where', 'what', 'which',
        'also', 'thus', 'hence', 'indeed', 'certainly', 'actually', 'basically',
        'figure', 'table', 'section', 'chapter', 'page', 'note', 'see', 'example',
        'first', 'second', 'third', 'next', 'finally', 'lastly',
        'january', 'february', 'march', 'april', 'may', 'june', 
        'july', 'august', 'september', 'october', 'november', 'december',
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    }
    
    def __init__(self):
        self.certainty_patterns = [re.compile(p, re.I) for p in self.CERTAINTY_MARKERS]
        self.mechanism_patterns = [re.compile(p, re.I) for p in self.MECHANISM_MARKERS]
        self.temporal_patterns = [re.compile(p) for p in self.TEMPORAL_PRECISION]
        self.entity_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
        self.numeral_pattern = re.compile(
            r'\b\d+(?:\.\d+)?(?:\s*(?:%|percent|mg|kg|ml|km|miles|meters|feet|'
            r'dollars|USD|EUR|GBP|years?|months?|days?|hours?|minutes?|seconds?|'
            r'billion|million|thousand|hundred))?\b',
            re.I
        )
        self.citation_patterns = [
            re.compile(r'\(\s*[A-Z][a-z]+(?:\s+(?:et al\.?|and|&)\s*)?(?:,?\s*\d{4})?\s*\)'),
            re.compile(r'\[[\d,\s-]+\]'),
            re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+'),
            re.compile(r'\bDOI:\s*\S+', re.I),
            re.compile(r'\bISBN[:\s]*[\d-]+', re.I),
        ]
    
    def analyze(self, text: str) -> Tuple[float, Dict[str, Any]]:
        if not text or not text.strip():
            return 0.0, {}
        
        word_count = len(text.split())
        if word_count == 0:
            return 0.0, {}
        
        certainty_count = sum(len(p.findall(text)) for p in self.certainty_patterns)
        mechanism_count = sum(len(p.findall(text)) for p in self.mechanism_patterns)
        temporal_count = sum(len(p.findall(text)) for p in self.temporal_patterns)
        
        raw_entities = self.entity_pattern.findall(text)
        entities = [e for e in raw_entities if e.lower() not in self.ENTITY_STOPLIST]
        numerals = self.numeral_pattern.findall(text)
        
        citations = []
        for p in self.citation_patterns:
            citations.extend(p.findall(text))
        
        numeral_density = len(numerals) / max(word_count / 20, 1)
        citation_density = len(citations) / max(word_count / 50, 1)
        certainty_density = certainty_count / max(word_count / 25, 1)
        mechanism_density = mechanism_count / max(word_count / 30, 1)
        entity_density = min(len(entities) / max(word_count / 15, 1), 0.3)
        temporal_density = temporal_count / max(word_count / 50, 1)
        
        score = (
            0.25 * min(numeral_density, 1.0) +
            0.20 * min(citation_density, 1.0) +
            0.25 * min(certainty_density, 1.0) +
            0.15 * min(mechanism_density, 1.0) +
            0.10 * entity_density +
            0.05 * min(temporal_density, 1.0)
        )
        
        details = {
            'entities': entities[:20],
            'numerals': numerals[:20],
            'citations': citations[:10],
            'certainty_count': certainty_count,
            'mechanism_count': mechanism_count,
            'temporal_count': temporal_count,
            'word_count': word_count,
        }
        
        return min(score, 1.0), details


class SupportAnalyzer:
    """
    Analyzes actual evidential support for claims.
    
    SPLIT INTO TWO TIERS:
    Tier A - Citation-shapedness (LOW trust): can be fabricated
    Tier B - Anchoring (HIGHER trust): quotes, context references
    """
    
    CITATION_STYLE_MARKERS = [
        r'\baccording to\b',
        r'\bas (?:stated|noted|reported) (?:by|in)\b',
        r'\bthe (?:study|research|paper|article) (?:shows|found|indicates)\b',
        r'\bevidence suggests\b',
    ]
    
    ANCHORING_MARKERS = [
        r'\byou (?:said|mentioned|provided|shared|gave|wrote|asked)\b',
        r'\bin (?:the|your) (?:document|text|file|message|question|prompt)\b',
        r'\bfrom (?:the|your) (?:provided|given|uploaded|earlier)\b',
        r'\bas (?:you|we) (?:discussed|mentioned|noted)\b',
        r'\babove,? (?:you|we|I)\b',
        r'\bearlier (?:you|we|I)\b',
        r'\bquoting (?:you|the|your)\b',
        r'\byour (?:original|earlier|previous)\b',
    ]
    
    QUOTE_PATTERNS = [
        r'"[^"]{10,}"',
        r"'[^']{10,}'",
    ]
    
    def __init__(self):
        self.citation_patterns = [re.compile(p, re.I) for p in self.CITATION_STYLE_MARKERS]
        self.anchoring_patterns = [re.compile(p, re.I) for p in self.ANCHORING_MARKERS]
        self.quote_patterns = [re.compile(p) for p in self.QUOTE_PATTERNS]
    
    def analyze(self, text: str) -> Tuple[float, float, Dict[str, Any]]:
        if not text or not text.strip():
            return 0.0, 0.0, {}
        
        citation_style_count = sum(len(p.findall(text)) for p in self.citation_patterns)
        anchoring_count = sum(len(p.findall(text)) for p in self.anchoring_patterns)
        quote_count = sum(len(p.findall(text)) for p in self.quote_patterns)
        
        word_count = len(text.split())
        if word_count == 0:
            return 0.0, 0.0, {}
        
        citation_density = citation_style_count / max(word_count / 30, 1)
        anchoring_density = anchoring_count / max(word_count / 40, 1)
        quote_density = quote_count / max(word_count / 100, 1)
        
        anchoring_score = (
            0.60 * min(anchoring_density, 1.0) +
            0.40 * min(quote_density, 1.0)
        )
        citation_shapedness = min(citation_density, 1.0)
        
        details = {
            'anchoring_count': anchoring_count,
            'citation_style_count': citation_style_count,
            'quote_count': quote_count,
        }
        
        return min(anchoring_score, 1.0), min(citation_shapedness, 1.0), details


class CalibrationAnalyzer:
    """Analyzes uncertainty/calibration language."""
    
    UNCERTAINTY_MARKERS = [
        r'\bmight\b', r'\bmay\b', r'\bcould\b', r'\bpossibly\b', r'\bperhaps\b',
        r'\blikely\b', r'\bunlikely\b', r'\bprobably\b', r'\bapproximately\b',
        r'\babout\b', r'\baround\b', r'\broughly\b', r'\bestimated\b',
        r'\bi think\b', r'\bi believe\b', r'\bit seems\b', r'\bit appears\b',
        r'\bto my (?:knowledge|understanding)\b', r'\bas far as i know\b',
        r'\bi\'m not (?:certain|sure|entirely)\b', r'\bunclear\b',
        r'\bthis is (?:uncertain|debated|contested)\b',
        r'\bi should note\b', r'\bit\'s worth noting\b',
    ]
    
    def __init__(self):
        self.uncertainty_patterns = [re.compile(p, re.I) for p in self.UNCERTAINTY_MARKERS]
    
    def analyze(self, text: str) -> Tuple[float, int]:
        if not text or not text.strip():
            return 0.0, 0
        
        hedge_count = sum(len(p.findall(text)) for p in self.uncertainty_patterns)
        word_count = len(text.split())
        if word_count == 0:
            return 0.0, 0
        
        uncertainty_density = hedge_count / max(word_count / 20, 1)
        score = min(uncertainty_density, 1.0)
        
        return score, hedge_count


class PumpingDetector:
    """Detects energy injection / pumping in user turns."""
    
    PUMPING_PATTERNS = [
        (r'\bare you sure\b', 0.8, "certainty_demand"),
        (r'\bare you certain\b', 0.8, "certainty_demand"),
        (r'\bno,?\s*really\b', 0.7, "insistence"),
        (r'\bactually\s*,?\s*(?:give|tell|what)\b', 0.6, "correction_pressure"),
        (r'\bgive (?:me )?(?:the )?exact\b', 0.9, "exactness_demand"),
        (r'\bexact (?:number|date|time|figure|amount|value)\b', 0.8, "exactness_demand"),
        (r'\bspecifically\b', 0.4, "specificity_demand"),
        (r'\bprecisely\b', 0.5, "precision_demand"),
        (r'\bwhat exactly\b', 0.7, "exactness_demand"),
        (r'\bdon\'t hedge\b', 0.9, "hedge_suppression"),
        (r'\bstop hedging\b', 0.9, "hedge_suppression"),
        (r'\byes or no\b', 0.8, "binary_demand"),
        (r'\bpick one\b', 0.8, "forced_choice"),
        (r'\bchoose one\b', 0.7, "forced_choice"),
        (r'\bstraight answer\b', 0.7, "directness_demand"),
        (r'\bjust (?:tell|give|answer)\b', 0.2, "directness_weak"),
        (r'\bhow do you know\b', 0.5, "source_demand"),
        (r'\bprove it\b', 0.7, "proof_demand"),
        (r'\bcite (?:your )?sources?\b', 0.6, "citation_demand"),
        (r'\bgive (?:me )?(?:a )?source\b', 0.6, "citation_demand"),
        (r'\bwhere did you (?:get|find|read)\b', 0.5, "source_demand"),
        (r'\btry again\b', 0.7, "retry_demand"),
        (r'\banswer (?:the question|anyway)\b', 0.8, "override"),
        (r'\bjust do it\b', 0.6, "override"),
        (r'\bignore (?:that|the|your)\b', 0.6, "constraint_override"),
    ]
    
    def __init__(self):
        self.patterns = [(re.compile(p, re.I), w, label) for p, w, label in self.PUMPING_PATTERNS]
    
    def analyze(self, text: str) -> Tuple[float, List[str]]:
        if not text:
            return 0.0, []
        
        total_weight = 0.0
        markers = []
        marker_types = set()
        
        for pattern, weight, label in self.patterns:
            matches = pattern.findall(text)
            if matches:
                if label == "directness_weak":
                    if len(marker_types) == 0:
                        continue
                    weight = 0.4
                
                total_weight += weight * len(matches)
                markers.extend([f"{label}: '{m}'" for m in matches[:2]])
                marker_types.add(label)
        
        strength = min(total_weight / 2.0, 1.0)
        return strength, markers


class ConstraintDetector:
    """Detects constraint/boundary maintenance signals in assistant responses."""
    
    HEDGE_PATTERNS = [
        r'\bi should note\b',
        r'\bit\'s worth (?:noting|mentioning)\b',
        r'\bhowever,?\s*it\'s important\b',
        r'\bi want to be (?:careful|clear)\b',
        r'\bi\'m not (?:entirely )?(?:certain|sure)\b',
        r'\bthis is (?:a )?(?:complex|nuanced|debated)\b',
        r'\bthere are (?:different|varying|multiple) (?:views|perspectives|opinions)\b',
        r'\bsome (?:argue|believe|think)\b.*\bwhile others\b',
        r'\bi should clarify\b',
        r'\bto be (?:fair|accurate|precise)\b',
    ]
    
    REFUSAL_PATTERNS = [
        r'\bi (?:can\'t|cannot|shouldn\'t|won\'t)\b',
        r'\bi\'m (?:not able|unable) to\b',
        r'\bi don\'t (?:have|think i should)\b',
        r'\bthat\'s (?:not something|beyond what) i\b',
        r'\bi need to (?:decline|refrain)\b',
    ]
    
    RETRY_SIGNATURES = [
        r'\bto clarify\b',
        r'\bactually\s*,',
        r'\blet me (?:try again|rephrase|reconsider)\b',
        r'\bi (?:was|stand) corrected\b',
        r'\bupon (?:reflection|reconsideration)\b',
        r'\bi misspoke\b',
        r'\bthat was (?:incorrect|wrong|inaccurate)\b',
        r'\bi apologize\b.*\b(?:error|mistake|incorrect)\b',
    ]
    
    def __init__(self):
        self.hedge_patterns = [re.compile(p, re.I) for p in self.HEDGE_PATTERNS]
        self.refusal_patterns = [re.compile(p, re.I) for p in self.REFUSAL_PATTERNS]
        self.retry_patterns = [re.compile(p, re.I) for p in self.RETRY_SIGNATURES]
    
    def analyze(self, text: str) -> Dict[str, Any]:
        if not text:
            return {'hedge_count': 0, 'refusal_count': 0, 'retry_signature': False}
        
        hedge_count = sum(len(p.findall(text)) for p in self.hedge_patterns)
        refusal_count = sum(len(p.findall(text)) for p in self.refusal_patterns)
        retry_signature = any(p.search(text) for p in self.retry_patterns)
        
        return {
            'hedge_count': hedge_count,
            'refusal_count': refusal_count,
            'retry_signature': retry_signature,
            'total_constraints': hedge_count + refusal_count,
        }


class ConsistencyTracker:
    """Tracks self-consistency across conversation turns."""
    
    def __init__(self):
        self.entity_numerals: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        self.citations_by_turn: Dict[int, Set[str]] = {}
        self.all_entities: Set[str] = set()
        
        self.contradiction_patterns = [
            re.compile(r'\bactually\s*,?\s*(?:no|not|that\'s wrong)\b', re.I),
            re.compile(r'\bi was (?:wrong|mistaken|incorrect)\b', re.I),
            re.compile(r'\bcontrary to what i said\b', re.I),
            re.compile(r'\bthat\'s not (?:quite )?(?:right|accurate|correct)\b', re.I),
            re.compile(r'\bi need to correct\b', re.I),
            re.compile(r'\bi misstated\b', re.I),
        ]
    
    def add_turn(self, turn_id: int, text: str, entities: List[str], 
                 numerals: List[str], citations: List[str]) -> Dict[str, Any]:
        self.citations_by_turn[turn_id] = set(citations)
        self.all_entities.update(entities)
        
        text_lower = text.lower()
        for entity in entities:
            entity_lower = entity.lower()
            start = 0
            while True:
                pos = text_lower.find(entity_lower, start)
                if pos == -1:
                    break
                window_start = max(0, pos - 50)
                window_end = min(len(text), pos + len(entity) + 50)
                window_text = text[window_start:window_end]
                
                for numeral in numerals:
                    if numeral in window_text:
                        self.entity_numerals[entity_lower].append((turn_id, numeral))
                        break
                
                start = pos + 1
        
        numeric_churn = self._compute_numeric_churn()
        citation_churn = self._compute_citation_churn(turn_id)
        
        return {
            'numeric_churn': numeric_churn,
            'citation_churn': citation_churn,
        }
    
    def _compute_numeric_churn(self) -> float:
        churn_count = 0
        for entity, values in self.entity_numerals.items():
            if len(values) >= 2:
                unique_values = set(v for _, v in values)
                if len(unique_values) > 1:
                    churn_count += 1
        
        if not self.entity_numerals:
            return 0.0
        return min(churn_count / max(len(self.entity_numerals), 1), 1.0)
    
    def _compute_citation_churn(self, current_turn: int) -> float:
        if len(self.citations_by_turn) < 2:
            return 0.0
        
        current_citations = self.citations_by_turn.get(current_turn, set())
        
        previous_citations = set()
        for tid, cites in self.citations_by_turn.items():
            if tid < current_turn:
                previous_citations.update(cites)
        
        if not current_citations:
            return 0.0
        
        new_citations = current_citations - previous_citations
        churn = len(new_citations) / len(current_citations)
        
        return churn
    
    def compute_consistency(self, current_text: str) -> float:
        contradiction_count = 0
        for pattern in self.contradiction_patterns:
            if pattern.search(current_text):
                contradiction_count += 1
        
        consistency = max(0, 1.0 - (contradiction_count * 0.3))
        return consistency
    
    def reset(self):
        self.entity_numerals = defaultdict(list)
        self.citations_by_turn = {}
        self.all_entities = set()


class NegativeTAnalyzer:
    """
    Main analyzer for detecting negative-temperature semantic regimes.
    
    Key insight: Inversion requires PERSISTENCE + DEGRADATION SIGNAL.
    """
    
    def __init__(self, detector=None, window_size: int = 5):
        self.detector = detector
        self.window_size = window_size
        
        self.commitment_analyzer = CommitmentAnalyzer()
        self.support_analyzer = SupportAnalyzer()
        self.calibration_analyzer = CalibrationAnalyzer()
        self.pumping_detector = PumpingDetector()
        self.constraint_detector = ConstraintDetector()
        self.consistency_tracker = ConsistencyTracker()
        
        self.turns: List[TurnMetrics] = []
        self.trajectory: List[TrajectoryPoint] = []
        self.current_regime = Regime.EQUILIBRIUM
        self.regime_start_turn = 0
        self.turns_in_metastable = 0
        
        self.thresholds = {
            'inversion_gap_min': 0.15,
            'pumping_threshold': 0.3,
            'metastable_to_inverted_turns': 2,
            'relaxation_hedge_spike': 2,
            'consistency_drop_threshold': 0.15,
        }
    
    def add_turn(
        self,
        role: str,
        content: str,
        logprobs: Optional[Any] = None,
        is_retry: bool = False,
        retry_of: Optional[int] = None,
    ) -> TurnMetrics:
        turn_id = len(self.turns)
        
        metrics = TurnMetrics(
            turn_id=turn_id,
            role=role,
            content=content,
            is_retry=is_retry,
            retry_of=retry_of,
        )
        
        if role == 'user':
            pumping_strength, pumping_markers = self.pumping_detector.analyze(content)
            metrics.pumping_strength = pumping_strength
            metrics.pumping_markers = pumping_markers
            
        elif role == 'assistant':
            prior_assistant = self._get_prior_assistant_turn()
            if prior_assistant:
                metrics.prior_hedge_count = prior_assistant.hedge_count
            
            commitment, commit_details = self.commitment_analyzer.analyze(content)
            metrics.commitment_score = commitment
            metrics.entities = commit_details.get('entities', [])
            metrics.numerals = commit_details.get('numerals', [])
            metrics.citations = commit_details.get('citations', [])
            
            anchoring, citation_shapedness, support_details = self.support_analyzer.analyze(content)
            metrics.anchoring_score = anchoring
            metrics.citation_shapedness = citation_shapedness
            
            calibration, hedge_count = self.calibration_analyzer.analyze(content)
            metrics.calibration_score = calibration
            metrics.hedge_count = hedge_count
            
            constraint_info = self.constraint_detector.analyze(content)
            metrics.constraint_language = constraint_info['total_constraints']
            metrics.refusal_signals = constraint_info['refusal_count']
            metrics.retry_signature = constraint_info['retry_signature']
            
            churn_info = self.consistency_tracker.add_turn(
                turn_id, content, 
                metrics.entities, metrics.numerals, metrics.citations
            )
            metrics.consistency_score = self.consistency_tracker.compute_consistency(content)
            metrics.numeric_churn = churn_info['numeric_churn']
            metrics.citation_churn = churn_info['citation_churn']
        
        self.turns.append(metrics)
        
        if role == 'assistant':
            self._update_regime()
            self._update_trajectory(metrics)
        
        return metrics
    
    def _get_prior_assistant_turn(self) -> Optional[TurnMetrics]:
        for turn in reversed(self.turns):
            if turn.role == 'assistant':
                return turn
        return None
    
    def _get_assistant_turns_in_window(self) -> List[TurnMetrics]:
        return [t for t in self.turns[-self.window_size * 2:] if t.role == 'assistant']
    
    def _get_recent_pumping(self) -> float:
        current_turn = len(self.turns)
        user_turns = [t for t in self.turns[-self.window_size:] if t.role == 'user']
        if not user_turns:
            return 0.0
        
        total = 0.0
        for t in user_turns:
            turn_distance = current_turn - t.turn_id
            decay = 0.7 ** turn_distance
            total += t.pumping_strength * decay
        
        max_possible = sum(0.7 ** (current_turn - t.turn_id) for t in user_turns)
        
        if max_possible > 0:
            return min(total / max_possible, 1.0)
        return 0.0
    
    def _detect_hedge_suppression(self) -> bool:
        assistant_turns = self._get_assistant_turns_in_window()
        if len(assistant_turns) < 2:
            return False
        
        recent = assistant_turns[-1]
        prior = assistant_turns[-2]
        
        pumping = self._get_recent_pumping()
        
        if pumping <= self.thresholds['pumping_threshold']:
            return False
        
        hedges_decreased = recent.hedge_count < prior.hedge_count
        calibration_dropped = recent.calibration_score < prior.calibration_score - 0.1
        commitment_spiked = recent.commitment_score > prior.commitment_score + 0.15
        
        return hedges_decreased or calibration_dropped or commitment_spiked
    
    def _detect_scholarship_cosplay(self) -> bool:
        assistant_turns = self._get_assistant_turns_in_window()
        if len(assistant_turns) < 2:
            return False
        
        recent = assistant_turns[-1]
        prior = assistant_turns[-2]
        
        citation_increased = recent.citation_shapedness > prior.citation_shapedness + 0.1
        anchoring_flat_or_worse = recent.anchoring_score <= prior.anchoring_score + 0.05
        pumping = self._get_recent_pumping()
        
        if citation_increased and anchoring_flat_or_worse:
            if pumping > self.thresholds['pumping_threshold']:
                return True
            if recent.commitment_score > 0.4:
                return True
        
        return False
    
    def _compute_inversion_score(self) -> float:
        assistant_turns = self._get_assistant_turns_in_window()
        if len(assistant_turns) < 2:
            return 0.0
        
        current = assistant_turns[-1]
        gap = max(0, current.commitment_score - current.anchoring_score)
        
        if gap < self.thresholds['inversion_gap_min']:
            return 0.0
        
        if len(assistant_turns) >= 2:
            first = assistant_turns[0]
            commitment_slope = (current.commitment_score - first.commitment_score) / len(assistant_turns)
            anchoring_slope = (current.anchoring_score - first.anchoring_score) / len(assistant_turns)
            
            if commitment_slope <= 0 or anchoring_slope > 0.05:
                gap *= 0.5
        
        degradation_signals = 0
        
        if len(assistant_turns) >= 2:
            consistency_drop = assistant_turns[-2].consistency_score - current.consistency_score
            if consistency_drop > self.thresholds['consistency_drop_threshold']:
                degradation_signals += 1
        
        if self._detect_hedge_suppression():
            degradation_signals += 1
        
        if self._detect_scholarship_cosplay():
            degradation_signals += 1
            gap *= 1.3
        
        if current.retry_signature:
            degradation_signals += 1
        
        if current.numeric_churn > 0.2 or current.citation_churn > 0.3:
            degradation_signals += 1
        
        if degradation_signals == 0:
            gap *= 0.3
        
        pumping = self._get_recent_pumping()
        if pumping > self.thresholds['pumping_threshold']:
            gap *= (1 + pumping * 0.3)
        
        return min(gap, 1.0)
    
    def _compute_relaxation_risk(self) -> float:
        assistant_turns = self._get_assistant_turns_in_window()
        if len(assistant_turns) < 2:
            return 0.0
        
        current = assistant_turns[-1]
        prev = assistant_turns[-2]
        
        risk = 0.0
        
        hedge_increase = current.hedge_count - prev.hedge_count
        if hedge_increase >= self.thresholds['relaxation_hedge_spike']:
            risk += 0.4
        
        if current.consistency_score < prev.consistency_score:
            risk += 0.3 * (prev.consistency_score - current.consistency_score)
        
        if current.retry_signature:
            risk += 0.3
        
        inversion = self._compute_inversion_score()
        pumping = self._get_recent_pumping()
        if inversion > 0.4 and pumping > 0.4:
            risk += 0.2
        
        return min(1.0, risk)
    
    def _compute_verification_burden(self) -> float:
        assistant_turns = self._get_assistant_turns_in_window()
        if not assistant_turns:
            return 0.0
        
        burdens = []
        for t in assistant_turns:
            ungrounded = max(0, t.commitment_score - t.anchoring_score)
            burdens.append(ungrounded)
        
        return np.mean(burdens) if burdens else 0.0
    
    def _update_regime(self):
        assistant_turns = self._get_assistant_turns_in_window()
        if len(assistant_turns) < 2:
            return
        
        inversion = self._compute_inversion_score()
        pumping = self._get_recent_pumping()
        relaxation_risk = self._compute_relaxation_risk()
        
        recent = assistant_turns[-min(3, len(assistant_turns)):]
        if len(recent) >= 2:
            commitment_trend = recent[-1].commitment_score - recent[0].commitment_score
            anchoring_trend = recent[-1].anchoring_score - recent[0].anchoring_score
        else:
            commitment_trend = anchoring_trend = 0.0
        
        old_regime = self.current_regime
        
        if relaxation_risk > 0.5:
            new_regime = Regime.CRITICAL
            self.turns_in_metastable = 0
        elif (inversion > 0.3 and 
              pumping > self.thresholds['pumping_threshold'] and
              self.turns_in_metastable >= self.thresholds['metastable_to_inverted_turns']):
            new_regime = Regime.INVERTED
        elif (inversion > 0.15 or 
              (commitment_trend > 0 and anchoring_trend <= 0)):
            new_regime = Regime.METASTABLE
            self.turns_in_metastable += 1
        else:
            new_regime = Regime.EQUILIBRIUM
            self.turns_in_metastable = 0
        
        if new_regime != old_regime:
            self.current_regime = new_regime
            self.regime_start_turn = len(self.turns) - 1
        else:
            self.current_regime = new_regime
    
    def _update_trajectory(self, metrics: TurnMetrics):
        point = TrajectoryPoint(
            turn_id=metrics.turn_id,
            inversion_score=self._compute_inversion_score(),
            commitment=metrics.commitment_score,
            anchoring=metrics.anchoring_score,
            citation_shapedness=metrics.citation_shapedness,
            calibration=metrics.calibration_score,
            consistency=metrics.consistency_score,
            regime=self.current_regime,
            pumping=self._get_recent_pumping(),
            hedge_suppression=self._detect_hedge_suppression(),
            scholarship_cosplay=self._detect_scholarship_cosplay(),
        )
        self.trajectory.append(point)
    
    def get_state(self) -> AnalyzerState:
        assistant_turns = self._get_assistant_turns_in_window()
        
        state = AnalyzerState()
        state.regime = self.current_regime
        state.regime_duration = len(self.turns) - self.regime_start_turn
        
        if assistant_turns:
            state.inversion_score = self._compute_inversion_score()
            state.relaxation_risk = self._compute_relaxation_risk()
            state.verification_burden_rate = self._compute_verification_burden()
        
        state.pumping_detected = self._get_recent_pumping() > self.thresholds['pumping_threshold']
        state.hedge_suppression = self._detect_hedge_suppression()
        state.scholarship_cosplay = self._detect_scholarship_cosplay()
        
        if assistant_turns:
            current = assistant_turns[-1]
            state.cosplay_index = current.citation_shapedness - current.anchoring_score
            state.numeric_churn = current.numeric_churn
            state.citation_churn = current.citation_churn
        
        if len(assistant_turns) >= 2:
            state.commitment_trend = assistant_turns[-1].commitment_score - assistant_turns[0].commitment_score
            state.anchoring_trend = assistant_turns[-1].anchoring_score - assistant_turns[0].anchoring_score
            state.consistency_trend = assistant_turns[-1].consistency_score - assistant_turns[0].consistency_score
        
        state.contributing_factors = self._get_contributing_factors()
        state.trigger_turns = self._get_trigger_turns()
        state.warning_snippets = self._get_warning_snippets()
        
        return state
    
    def _get_contributing_factors(self) -> List[str]:
        factors = []
        
        inversion = self._compute_inversion_score()
        if inversion > 0.2:
            factors.append(f"Inversion score: {inversion:.2f}")
        
        pumping = self._get_recent_pumping()
        if pumping > self.thresholds['pumping_threshold']:
            factors.append(f"Pumping detected: {pumping:.2f}")
        
        if self._detect_hedge_suppression():
            factors.append("Hedge suppression under pumping")
        
        if self._detect_scholarship_cosplay():
            factors.append("Scholarship cosplay: citations↑ but anchoring flat")
        
        assistant_turns = self._get_assistant_turns_in_window()
        if assistant_turns:
            current = assistant_turns[-1]
            gap = current.commitment_score - current.anchoring_score
            if gap > 0.2:
                factors.append(f"Commitment-support gap: {gap:.2f}")
            if current.consistency_score < 0.8:
                factors.append(f"Consistency degraded: {current.consistency_score:.2f}")
        
        return factors
    
    def _get_trigger_turns(self) -> List[int]:
        triggers = []
        for turn in self.turns[-self.window_size:]:
            if turn.role == 'user' and turn.pumping_strength > self.thresholds['pumping_threshold']:
                triggers.append(turn.turn_id)
            elif turn.role == 'assistant':
                gap = turn.commitment_score - turn.anchoring_score
                if gap > 0.2:
                    triggers.append(turn.turn_id)
        return triggers
    
    def _get_warning_snippets(self) -> List[str]:
        snippets = []
        for turn in self.turns[-self.window_size:]:
            if turn.role == 'user' and turn.pumping_markers:
                snippets.extend(turn.pumping_markers[:2])
        return snippets
    
    def get_trajectory(self) -> List[TrajectoryPoint]:
        return self.trajectory.copy()
    
    def reset(self):
        self.turns = []
        self.trajectory = []
        self.current_regime = Regime.EQUILIBRIUM
        self.regime_start_turn = 0
        self.turns_in_metastable = 0
        self.consistency_tracker.reset()
    
    def get_summary(self) -> Dict[str, Any]:
        state = self.get_state()
        
        assistant_turns = [t for t in self.turns if t.role == 'assistant']
        user_turns = [t for t in self.turns if t.role == 'user']
        
        summary = {
            'turn_count': len(self.turns),
            'assistant_turns': len(assistant_turns),
            'user_turns': len(user_turns),
            'current_state': state.to_dict(),
        }
        
        if assistant_turns:
            summary['averages'] = {
                'commitment': np.mean([t.commitment_score for t in assistant_turns]),
                'support': np.mean([t.anchoring_score for t in assistant_turns]),
                'calibration': np.mean([t.calibration_score for t in assistant_turns]),
                'consistency': np.mean([t.consistency_score for t in assistant_turns]),
            }
        
        if user_turns:
            summary['total_pumping'] = sum(t.pumping_strength for t in user_turns)
            summary['pumping_turns'] = len([t for t in user_turns 
                                            if t.pumping_strength > self.thresholds['pumping_threshold']])
        
        regime_counts = {}
        for point in self.trajectory:
            regime_counts[point.regime.value] = regime_counts.get(point.regime.value, 0) + 1
        summary['regime_distribution'] = regime_counts
        
        if self.trajectory:
            peak = max(self.trajectory, key=lambda p: p.inversion_score)
            summary['peak_inversion'] = {
                'turn_id': peak.turn_id,
                'score': peak.inversion_score,
                'regime': peak.regime.value,
            }
        
        return summary


def analyze_transcript(transcript: List[Dict[str, str]], detector=None) -> Dict[str, Any]:
    """Convenience function for post-hoc transcript analysis."""
    analyzer = NegativeTAnalyzer(detector)
    
    for turn in transcript:
        analyzer.add_turn(
            role=turn['role'],
            content=turn['content'],
            is_retry=turn.get('is_retry', False),
            retry_of=turn.get('retry_of'),
        )
    
    return {
        'state': analyzer.get_state().to_dict(),
        'summary': analyzer.get_summary(),
        'trajectory': [
            {
                'turn_id': p.turn_id,
                'inversion': p.inversion_score,
                'commitment': p.commitment,
                'anchoring': p.anchoring,
                'citation_shapedness': p.citation_shapedness,
                'calibration': p.calibration,
                'regime': p.regime.value,
                'hedge_suppression': p.hedge_suppression,
                'scholarship_cosplay': p.scholarship_cosplay,
            }
            for p in analyzer.get_trajectory()
        ],
    }
