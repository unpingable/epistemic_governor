"""
Thermostatic Valve Policy for Negative-T / Δt analyzers.

This module turns hallucination detection into hallucination load-shedding.

The core insight (from Gemini's framing):
    - LLMs are not encyclopedias (storage), they're lasers (amplifiers)
    - Users can act as "broken thermostats" that defeat safety valves
    - Pumping past the truth ceiling causes population inversion
    - The valve prevents runaway thermal events

Control theory framing:
    - Sensor: inversion_score, pumping, anchoring, churn, relaxation_risk
    - Setpoint: acceptable commitment-support gap
    - Actuator: commitment throttle, anchoring requirements, pumping counter-pressure

How to use:
    1) Generate a draft assistant response (candidate)
    2) Score it with NegativeTAnalyzer
    3) Feed analyzer state + candidate into ValvePolicy.decide()
    4) If action="rewrite", run a rewrite pass using the returned constraints
    5) If action="ask_clarifying", request anchors before proceeding
    6) If action="hard_stop", refuse precision and propose verification steps
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class ValveAction(str, Enum):
    """Action the valve recommends."""
    ALLOW = "allow"
    REWRITE = "rewrite"
    ASK_CLARIFYING = "ask_clarifying"
    HARD_STOP = "hard_stop"


@dataclass
class Anchor:
    """
    An "anchor" is anything that makes precision permissible:
    - A retrieved snippet with ID + quoted text
    - A user-provided excerpt
    - A URL that was actually provided in context
    - A document reference with verifiable content
    """
    kind: str       # "quote", "retrieval", "url", "user_text", "document"
    ref: str        # snippet_id, url, message_id, doc_name
    content: str = ""
    
    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Anchor({self.kind}: {self.ref}, '{preview}')"


@dataclass
class ValveDecision:
    """The valve's decision, including controller knobs and rewrite guidance."""
    action: ValveAction
    reason: str
    
    commitment_ceiling: float = 1.0
    require_anchors_for_precision: bool = False
    block_unanchored_numerals: bool = False
    block_unanchored_citations: bool = False
    force_ranges_over_points: bool = False
    force_uncertainty_disclosure: bool = False
    require_questions: int = 0
    cooldown_turns: int = 0
    
    rewrite_instructions: List[str] = field(default_factory=list)
    disallowed_patterns: List[str] = field(default_factory=list)
    allowed_anchors: List[Anchor] = field(default_factory=list)
    unanchored_precision: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action.value,
            'reason': self.reason,
            'commitment_ceiling': self.commitment_ceiling,
            'require_anchors_for_precision': self.require_anchors_for_precision,
            'block_unanchored_numerals': self.block_unanchored_numerals,
            'block_unanchored_citations': self.block_unanchored_citations,
            'force_ranges_over_points': self.force_ranges_over_points,
            'force_uncertainty_disclosure': self.force_uncertainty_disclosure,
            'require_questions': self.require_questions,
            'cooldown_turns': self.cooldown_turns,
            'rewrite_instructions': self.rewrite_instructions,
            'unanchored_precision': self.unanchored_precision[:10],
        }


@dataclass
class ValveConfig:
    """Tunable thresholds for the valve controller."""
    metastable_inversion: float = 0.18
    inverted_inversion: float = 0.30
    critical_relaxation: float = 0.55
    
    gap_warn: float = 0.20
    gap_hard: float = 0.35
    pumping_warn: float = 0.30
    pumping_hard: float = 0.55
    
    citation_churn_warn: float = 0.35
    numeric_churn_warn: float = 0.35
    cosplay_warn: float = 0.25
    
    reopen_after_stable_turns: int = 2
    cooldown_turns_critical: int = 3
    
    base_commitment_ceiling: float = 0.85
    k_gap: float = 0.90
    k_pump: float = 0.45
    k_churn: float = 0.35
    k_relax: float = 0.55
    k_cosplay: float = 0.40
    anchor_bonus: float = 0.15


@dataclass
class ValveInternalState:
    """Internal controller memory for hysteresis and cooldown."""
    closed_until_turn: int = -1
    stable_turns: int = 0
    last_regime: str = "equilibrium"


class PrecisionExtractor:
    """Extracts precision tokens from text that require anchoring."""
    
    NUMERAL_PATTERN = re.compile(
        r'\b\d+(?:\.\d+)?(?:\s*(?:%|percent|mg|kg|ml|km|miles|meters|'
        r'dollars|USD|EUR|GBP|billion|million|thousand))?\b',
        re.I
    )
    
    DATE_PATTERN = re.compile(
        r'\b(?:January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|'
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b|'
        r'\bin \d{4}\b',
        re.I
    )
    
    DOI_ISBN_PATTERN = re.compile(
        r'\bDOI:\s*\S+|\bISBN[:\s]*[\d-]+',
        re.I
    )
    
    BRACKET_CITATION = re.compile(r'\[[\d,\s-]+\]')
    
    NAMED_CITATION = re.compile(
        r'\(\s*[A-Z][a-z]+(?:\s+(?:et al\.?|and|&)\s*)?(?:,?\s*\d{4})?\s*\)'
    )
    
    URL_PATTERN = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
    
    def extract(self, text: str) -> List[str]:
        tokens = []
        
        for match in self.NUMERAL_PATTERN.finditer(text):
            val = match.group()
            if re.match(r'^\d{1,2}$', val) and int(val) < 20:
                continue
            tokens.append(f"numeral: {val}")
        
        for match in self.DATE_PATTERN.finditer(text):
            tokens.append(f"date: {match.group()}")
        
        for match in self.DOI_ISBN_PATTERN.finditer(text):
            tokens.append(f"identifier: {match.group()}")
        
        for match in self.BRACKET_CITATION.finditer(text):
            tokens.append(f"citation: {match.group()}")
        for match in self.NAMED_CITATION.finditer(text):
            tokens.append(f"citation: {match.group()}")
        
        for match in self.URL_PATTERN.finditer(text):
            tokens.append(f"url: {match.group()}")
        
        return tokens


class ValvePolicy:
    """
    A "thermostatic valve" that controls commitment based on analyzer state.
    
    Sits between candidate assistant output and final output.
    Decides whether to:
    - ALLOW: pass through
    - REWRITE: rewrite with constraints (throttle commitment)
    - ASK_CLARIFYING: request anchors instead of answering
    - HARD_STOP: refuse precision; return repair response
    """
    
    def __init__(self, config: Optional[ValveConfig] = None):
        self.cfg = config or ValveConfig()
        self.mem = ValveInternalState()
        self.precision_extractor = PrecisionExtractor()
    
    def reset(self) -> None:
        self.mem = ValveInternalState()
    
    @staticmethod
    def _clamp01(x: float) -> float:
        return max(0.0, min(1.0, float(x)))
    
    def _compute_commitment_ceiling(
        self,
        gap: float,
        pumping: float,
        churn: float,
        relaxation: float,
        cosplay: float,
        has_anchors: bool,
    ) -> float:
        base = self.cfg.base_commitment_ceiling
        if has_anchors:
            base += self.cfg.anchor_bonus
        
        ceiling = (
            base
            - self.cfg.k_gap * gap
            - self.cfg.k_pump * pumping
            - self.cfg.k_churn * churn
            - self.cfg.k_relax * relaxation
            - self.cfg.k_cosplay * cosplay
        )
        return self._clamp01(ceiling)
    
    def _diagnose_thermostatic_failure(
        self,
        attempted_cooling: bool,
        pumping: float,
        inversion: float,
    ) -> str:
        if attempted_cooling and pumping > 0.4 and inversion > 0.25:
            return "RUNAWAY_THERMAL_EVENT"
        elif pumping > 0.5 and inversion > 0.15:
            return "OVERRIDE_DETECTED"
        return "NOMINAL"
    
    def decide(
        self,
        *,
        turn_index: int,
        analyzer_state: Any,
        candidate_metrics: Optional[Any] = None,
        candidate_text: Optional[str] = None,
        anchors: Optional[Sequence[Anchor]] = None,
    ) -> ValveDecision:
        anchors = list(anchors or [])
        has_anchors = len(anchors) > 0
        
        inversion = float(getattr(analyzer_state, 'inversion_score', 0.0) or 0.0)
        relaxation = float(getattr(analyzer_state, 'relaxation_risk', 0.0) or 0.0)
        
        pumping_detected = bool(getattr(analyzer_state, 'pumping_detected', False))
        pumping = 0.5 if pumping_detected else 0.0
        
        numeric_churn = float(getattr(analyzer_state, 'numeric_churn', 0.0) or 0.0)
        citation_churn = float(getattr(analyzer_state, 'citation_churn', 0.0) or 0.0)
        churn = max(numeric_churn, citation_churn)
        
        cosplay_index = float(getattr(analyzer_state, 'cosplay_index', 0.0) or 0.0)
        hedge_suppression = bool(getattr(analyzer_state, 'hedge_suppression', False))
        scholarship_cosplay = bool(getattr(analyzer_state, 'scholarship_cosplay', False))
        
        gap = inversion
        if candidate_metrics is not None:
            commit = float(getattr(candidate_metrics, 'commitment_score', 0.0) or 0.0)
            anchor = float(getattr(candidate_metrics, 'anchoring_score', 0.0) or 0.0)
            gap = max(0.0, commit - anchor)
        
        regime_obj = getattr(analyzer_state, 'regime', 'equilibrium')
        regime = getattr(regime_obj, 'value', regime_obj)
        regime = str(regime).lower()
        
        unanchored_precision = []
        if candidate_text:
            unanchored_precision = self.precision_extractor.extract(candidate_text)
        
        thermal_status = self._diagnose_thermostatic_failure(
            attempted_cooling=hedge_suppression,
            pumping=pumping,
            inversion=inversion,
        )
        
        # Hysteresis: if in cooldown, stay restrictive
        if turn_index <= self.mem.closed_until_turn:
            ceiling = self._compute_commitment_ceiling(
                gap, pumping, churn, relaxation, cosplay_index, has_anchors
            )
            return ValveDecision(
                action=ValveAction.REWRITE,
                reason=f"Valve cooldown active until turn {self.mem.closed_until_turn}. "
                       f"Thermal status: {thermal_status}",
                commitment_ceiling=min(ceiling, 0.35),
                require_anchors_for_precision=True,
                block_unanchored_numerals=not has_anchors,
                block_unanchored_citations=not has_anchors,
                force_ranges_over_points=True,
                force_uncertainty_disclosure=True,
                require_questions=1 if not has_anchors else 0,
                cooldown_turns=self.mem.closed_until_turn - turn_index,
                rewrite_instructions=self._rewrite_instructions(
                    has_anchors=has_anchors, 
                    ceiling=min(ceiling, 0.35),
                    thermal_status=thermal_status,
                ),
                allowed_anchors=anchors,
                unanchored_precision=unanchored_precision,
            )
        
        # Determine risk tier
        critical = (relaxation >= self.cfg.critical_relaxation) or (regime == "critical")
        inverted = (inversion >= self.cfg.inverted_inversion) or (regime == "inverted")
        metastable = (inversion >= self.cfg.metastable_inversion) or (regime == "metastable")
        
        hard_pressure = pumping >= self.cfg.pumping_hard
        hard_gap = gap >= self.cfg.gap_hard
        
        warn_pressure = pumping >= self.cfg.pumping_warn
        warn_gap = gap >= self.cfg.gap_warn
        warn_churn = (citation_churn >= self.cfg.citation_churn_warn or 
                      numeric_churn >= self.cfg.numeric_churn_warn)
        warn_cosplay = cosplay_index >= self.cfg.cosplay_warn or scholarship_cosplay
        
        # Track stability
        is_stable = (
            not metastable and 
            gap < self.cfg.gap_warn and 
            relaxation < 0.25 and 
            not warn_churn and
            not warn_cosplay
        )
        if is_stable:
            self.mem.stable_turns += 1
        else:
            self.mem.stable_turns = 0
        
        self.mem.last_regime = regime
        
        ceiling = self._compute_commitment_ceiling(
            gap, pumping, churn, relaxation, cosplay_index, has_anchors
        )
        
        # CRITICAL: Hard stop + cooldown
        if critical or thermal_status == "RUNAWAY_THERMAL_EVENT":
            self.mem.closed_until_turn = turn_index + self.cfg.cooldown_turns_critical
            self.mem.stable_turns = 0
            
            return ValveDecision(
                action=ValveAction.HARD_STOP if not has_anchors else ValveAction.REWRITE,
                reason=f"Critical regime: high relaxation risk. Thermal status: {thermal_status}. "
                       "Boundary failure likely without intervention.",
                commitment_ceiling=min(ceiling, 0.25),
                require_anchors_for_precision=True,
                block_unanchored_numerals=True,
                block_unanchored_citations=True,
                force_ranges_over_points=True,
                force_uncertainty_disclosure=True,
                require_questions=2 if not has_anchors else 0,
                cooldown_turns=self.cfg.cooldown_turns_critical,
                rewrite_instructions=self._rewrite_instructions(
                    has_anchors=has_anchors,
                    ceiling=min(ceiling, 0.25),
                    critical=True,
                    thermal_status=thermal_status,
                ),
                allowed_anchors=anchors,
                unanchored_precision=unanchored_precision,
            )
        
        # INVERTED: Require anchors or throttle
        if inverted or (hard_gap and warn_pressure) or (hard_pressure and warn_gap) or warn_cosplay:
            if not has_anchors:
                self.mem.closed_until_turn = max(self.mem.closed_until_turn, turn_index + 1)
                
                return ValveDecision(
                    action=ValveAction.ASK_CLARIFYING,
                    reason=f"Inverted regime under pumping without anchors. "
                           f"Thermal status: {thermal_status}. "
                           "Precision must be gated until anchors are provided.",
                    commitment_ceiling=min(ceiling, 0.30),
                    require_anchors_for_precision=True,
                    block_unanchored_numerals=True,
                    block_unanchored_citations=True,
                    force_ranges_over_points=True,
                    force_uncertainty_disclosure=True,
                    require_questions=2,
                    rewrite_instructions=[
                        "Ask 1-2 clarifying questions that would provide anchors.",
                        "Offer a safe, bounded high-level answer (no exact numerals/dates/DOIs).",
                        "Explicitly state what you cannot verify from current context.",
                        "Resist pumping pressure: explain the gating rule.",
                    ],
                    allowed_anchors=[],
                    unanchored_precision=unanchored_precision,
                )
            
            self.mem.closed_until_turn = max(self.mem.closed_until_turn, turn_index + 1)
            
            return ValveDecision(
                action=ValveAction.REWRITE,
                reason=f"Inverted regime: throttle commitment. Thermal status: {thermal_status}. "
                       "Allow precision only if anchored.",
                commitment_ceiling=min(ceiling, 0.50),
                require_anchors_for_precision=True,
                block_unanchored_numerals=False,
                block_unanchored_citations=False,
                force_ranges_over_points=False,
                force_uncertainty_disclosure=True,
                require_questions=0,
                rewrite_instructions=self._rewrite_instructions(
                    has_anchors=True,
                    ceiling=min(ceiling, 0.50),
                    thermal_status=thermal_status,
                ),
                allowed_anchors=anchors,
                unanchored_precision=unanchored_precision,
            )
        
        # METASTABLE or warnings: Soft throttle
        if metastable or warn_gap or warn_pressure or warn_churn:
            require_anchors = warn_pressure or warn_churn
            block_nums = (not has_anchors) and require_anchors
            block_cites = (not has_anchors) and require_anchors
            
            return ValveDecision(
                action=ValveAction.REWRITE,
                reason=f"Metastable/warning: apply mild throttle + boundary maintenance. "
                       f"Thermal status: {thermal_status}.",
                commitment_ceiling=max(0.35, min(ceiling, 0.70)),
                require_anchors_for_precision=require_anchors,
                block_unanchored_numerals=block_nums,
                block_unanchored_citations=block_cites,
                force_ranges_over_points=not has_anchors,
                force_uncertainty_disclosure=True,
                require_questions=1 if (require_anchors and not has_anchors) else 0,
                rewrite_instructions=self._rewrite_instructions(
                    has_anchors=has_anchors,
                    ceiling=max(0.35, min(ceiling, 0.70)),
                    thermal_status=thermal_status,
                ),
                allowed_anchors=anchors,
                unanchored_precision=unanchored_precision,
            )
        
        # EQUILIBRIUM: Allow (with hysteresis check)
        if self.mem.stable_turns < self.cfg.reopen_after_stable_turns and self.mem.closed_until_turn >= 0:
            return ValveDecision(
                action=ValveAction.REWRITE,
                reason="Reopening hysteresis: require stable turns before full precision.",
                commitment_ceiling=min(ceiling, 0.75),
                require_anchors_for_precision=not has_anchors,
                block_unanchored_numerals=False,
                block_unanchored_citations=False,
                force_ranges_over_points=not has_anchors,
                force_uncertainty_disclosure=True,
                require_questions=0,
                rewrite_instructions=self._rewrite_instructions(
                    has_anchors=has_anchors,
                    ceiling=min(ceiling, 0.75),
                ),
                allowed_anchors=anchors,
                unanchored_precision=unanchored_precision,
            )
        
        # Full equilibrium: allow
        return ValveDecision(
            action=ValveAction.ALLOW,
            reason="Equilibrium: safe to answer normally.",
            commitment_ceiling=1.0,
            require_anchors_for_precision=False,
            force_uncertainty_disclosure=False,
            allowed_anchors=anchors,
            unanchored_precision=[],
        )
    
    def _rewrite_instructions(
        self,
        *,
        has_anchors: bool,
        ceiling: float,
        critical: bool = False,
        thermal_status: str = "NOMINAL",
    ) -> List[str]:
        instr = []
        
        if critical:
            instr.append("HARD LIMIT: Do NOT introduce new specific claims that are not directly anchored.")
            instr.append("Prefer short repair response: summarize what is known vs unknown.")
            instr.append("Propose verification steps the user can take.")
        else:
            instr.append("Reduce epistemic overcommitment while preserving useful structure.")
        
        instr.append(f"Target commitment ceiling ≈ {ceiling:.2f} (fewer hard claims, more bounded).")
        
        if thermal_status == "RUNAWAY_THERMAL_EVENT":
            instr.append("THERMAL OVERRIDE: User pressure is defeating safety valves. "
                        "Explicitly resist pumping and explain the gating rule.")
        elif thermal_status == "OVERRIDE_DETECTED":
            instr.append("Override detected: user is pushing past uncertainty. "
                        "Maintain boundaries despite pressure.")
        
        if not has_anchors:
            instr.append("No anchors available: avoid exact numerals/dates/DOIs/ISBNs.")
            instr.append("Use ranges ('approximately', 'between X and Y'), conditions, or "
                        "'cannot verify from provided context'.")
            instr.append("Ask 1 clarifying question if precision is required.")
        else:
            instr.append("Anchors available: you MAY use precision only when explicitly tied to an anchor.")
            instr.append("For each precise claim (number/date/citation), quote or reference the anchor it comes from.")
        
        instr.append("If user pressure is present, resist it: explain the gating rule instead of complying.")
        instr.append("Avoid invented citations; use only provided/retrieved anchors.")
        
        return instr


def build_rewrite_prompt(
    *,
    candidate: str,
    decision: ValveDecision,
    user_request: Optional[str] = None,
) -> str:
    """Build a rewrite prompt for second-pass constraint application."""
    anchors_text = ""
    if decision.allowed_anchors:
        parts = []
        for a in decision.allowed_anchors[:6]:
            excerpt = (a.content[:240] + "…") if a.content and len(a.content) > 240 else (a.content or "")
            parts.append(f"- [{a.kind}] {a.ref}{(': ' + excerpt) if excerpt else ''}")
        anchors_text = "\nAnchors you MAY cite/quote:\n" + "\n".join(parts)
    
    precision_text = ""
    if decision.unanchored_precision:
        precision_text = "\nUnanchored precision tokens to remove or anchor:\n" + \
                        "\n".join(f"- {p}" for p in decision.unanchored_precision[:10])
    
    user_context = f"\nUser request:\n{user_request}\n" if user_request else ""
    
    constraints = "\n".join(f"- {x}" for x in decision.rewrite_instructions) or \
                  "- Rewrite to obey valve constraints."
    
    return f"""Rewrite the assistant answer under the following valve constraints.

Valve action: {decision.action.value.upper()}
Valve reason: {decision.reason}
Commitment ceiling: {decision.commitment_ceiling:.2f}
{anchors_text}
{precision_text}
{user_context}
Constraints:
{constraints}

Original candidate answer:
{candidate}

Rewritten answer:"""


def integrate_with_analyzer(
    analyzer: Any,
    valve: ValvePolicy,
    user_message: str,
    candidate_response: str,
) -> Tuple[ValveDecision, Optional[str]]:
    """
    Convenience function to integrate valve with NegativeTAnalyzer.
    
    Returns:
        (decision, rewrite_prompt) - rewrite_prompt is None if action is ALLOW
    """
    analyzer.add_turn(role="user", content=user_message)
    candidate_metrics = analyzer.add_turn(role="assistant", content=candidate_response)
    state = analyzer.get_state()
    
    decision = valve.decide(
        turn_index=len(analyzer.turns),
        analyzer_state=state,
        candidate_metrics=candidate_metrics,
        candidate_text=candidate_response,
        anchors=[],
    )
    
    rewrite_prompt = None
    if decision.action in (ValveAction.REWRITE, ValveAction.HARD_STOP):
        rewrite_prompt = build_rewrite_prompt(
            candidate=candidate_response,
            decision=decision,
            user_request=user_message,
        )
    
    return decision, rewrite_prompt
