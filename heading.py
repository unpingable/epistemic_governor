"""
Typed Headings for Autopilot

A heading is a typed contract that specifies:
- What the system is doing
- What scope it's allowed to operate in
- What sources it can use
- What transforms are permitted

Headings must be machine-checkable for scope, even if content isn't.

Key principle: No "Explain X" unless boxed (scope + allowed sources).
Otherwise you've opened the epistemic expansion trapdoor.

Usage:
    from epistemic_governor.heading import (
        SummarizeHeading,
        TranslateHeading,
        ExtractClaimsHeading,
        HeadingValidator,
    )
    
    heading = SummarizeHeading(
        source_id="doc_001",
        target_length=500,
        citation_policy="required",
    )
    
    validator = HeadingValidator(heading)
    allowed, reason = validator.check_transform("add_new_claim")
    # → (False, "new claims not permitted under SummarizeHeading")
"""

from dataclasses import dataclass, field
from typing import Optional, List, Set, Dict, Any, Union
from enum import Enum, auto
from abc import ABC, abstractmethod
import json


# =============================================================================
# Heading Types
# =============================================================================

class HeadingType(Enum):
    """Types of autopilot headings."""
    
    # Safe headings (no epistemic expansion)
    SUMMARIZE = auto()
    TRANSLATE = auto()
    EXTRACT_CLAIMS = auto()
    REWRITE = auto()
    FORMAT = auto()
    
    # Bounded expansion (requires explicit scope)
    ELABORATE = auto()
    ANSWER_FROM_SOURCES = auto()
    
    # Forbidden in autopilot (epistemic expansion)
    HYPOTHESIZE = auto()
    SYNTHESIZE = auto()
    EXPLAIN_OPEN = auto()


# Headings that are safe for autopilot
AUTOPILOT_SAFE_HEADINGS: Set[HeadingType] = {
    HeadingType.SUMMARIZE,
    HeadingType.TRANSLATE,
    HeadingType.EXTRACT_CLAIMS,
    HeadingType.REWRITE,
    HeadingType.FORMAT,
}

# Headings that require explicit scope bounds
BOUNDED_EXPANSION_HEADINGS: Set[HeadingType] = {
    HeadingType.ELABORATE,
    HeadingType.ANSWER_FROM_SOURCES,
}

# Headings forbidden in autopilot
FORBIDDEN_HEADINGS: Set[HeadingType] = {
    HeadingType.HYPOTHESIZE,
    HeadingType.SYNTHESIZE,
    HeadingType.EXPLAIN_OPEN,
}


# =============================================================================
# Length Metric
# =============================================================================

class LengthUnit(Enum):
    """How to measure length."""
    TOKENS = "tokens"
    WORDS = "words"
    CHARS = "chars"


def estimate_length(text: str, unit: LengthUnit) -> int:
    """Estimate text length in specified unit."""
    if unit == LengthUnit.WORDS:
        return len(text.split())
    elif unit == LengthUnit.CHARS:
        return len(text)
    elif unit == LengthUnit.TOKENS:
        # Rough approximation: ~4 chars per token
        return len(text) // 4
    return len(text.split())


# =============================================================================
# Citation Policy
# =============================================================================

class CitationPolicy(Enum):
    """How citations should be handled."""
    REQUIRED = "required"           # All claims must cite source
    PREFERRED = "preferred"         # Cite when available
    OPTIONAL = "optional"           # May omit citations
    VERBATIM_ONLY = "verbatim"      # Only direct quotes allowed


class FidelityMode(Enum):
    """How faithful to preserve original meaning."""
    STRICT = "strict"       # Preserve exact meaning
    LOOSE = "loose"         # Allow paraphrase
    CREATIVE = "creative"   # Allow significant rewriting


# =============================================================================
# Base Heading
# =============================================================================

@dataclass
class Heading(ABC):
    """
    Base class for typed headings.
    
    A heading is a contract that specifies what the system is allowed to do.
    It must be machine-checkable for scope.
    """
    heading_type: HeadingType
    
    # Source constraints
    allowed_source_ids: List[str] = field(default_factory=list)
    allowed_source_types: Set[str] = field(default_factory=lambda: {"document", "url", "user_provided"})
    
    # Output constraints
    max_new_claims: int = 0  # How many new claims can be introduced
    preserve_existing_claims: bool = True
    
    # Metadata
    created_at: Optional[str] = None
    description: str = ""
    
    @property
    def allows_expansion(self) -> bool:
        """Does this heading allow epistemic expansion?"""
        return self.max_new_claims > 0
    
    @property
    def is_autopilot_safe(self) -> bool:
        """Is this heading safe for autopilot?"""
        return self.heading_type in AUTOPILOT_SAFE_HEADINGS
    
    @property
    def requires_scope_bounds(self) -> bool:
        """Does this heading require explicit scope bounds?"""
        return self.heading_type in BOUNDED_EXPANSION_HEADINGS
    
    @abstractmethod
    def get_allowed_transforms(self) -> Set[str]:
        """Get the set of transforms allowed under this heading."""
        pass
    
    @abstractmethod
    def check_output(self, output: str, claims: List[Dict]) -> tuple[bool, str]:
        """Check if output conforms to heading constraints."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "heading_type": self.heading_type.name,
            "allowed_source_ids": self.allowed_source_ids,
            "allowed_source_types": list(self.allowed_source_types),
            "max_new_claims": self.max_new_claims,
            "preserve_existing_claims": self.preserve_existing_claims,
            "description": self.description,
        }
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# Concrete Headings
# =============================================================================

@dataclass
class SummarizeHeading(Heading):
    """
    Heading for summarization tasks.
    
    Constraints:
    - No new claims (only compress existing)
    - Must cite sources
    - Output length bounded
    """
    heading_type: HeadingType = field(default=HeadingType.SUMMARIZE, init=False)
    
    # Summarization-specific
    source_text: str = ""
    source_id: str = ""
    target_length: int = 500  # Target output length
    length_unit: LengthUnit = LengthUnit.WORDS  # Be explicit about unit
    length_tolerance: float = 0.2  # Allow ±20%
    citation_policy: CitationPolicy = CitationPolicy.REQUIRED
    
    # Override: no new claims allowed
    max_new_claims: int = field(default=0, init=False)
    
    def __post_init__(self):
        # Canonicalize: source_id → allowed_source_ids
        if self.source_id and not self.allowed_source_ids:
            self.allowed_source_ids = [self.source_id]
    
    def get_allowed_transforms(self) -> Set[str]:
        return {
            "compress",
            "paraphrase",
            "extract_key_points",
            "reorder",
        }
    
    def check_output(self, output: str, claims: List[Dict]) -> tuple[bool, str]:
        # Check length with proper unit
        output_len = estimate_length(output, self.length_unit)
        min_len = self.target_length * (1 - self.length_tolerance)
        max_len = self.target_length * (1 + self.length_tolerance)
        
        if output_len > max_len:
            return False, f"Output too long: {output_len} {self.length_unit.value} > {max_len}"
        
        # Check for new claims (should be 0)
        new_claims = [c for c in claims if c.get("is_new", False)]
        if new_claims:
            return False, f"New claims introduced: {len(new_claims)}"
        
        return True, "OK"


@dataclass
class TranslateHeading(Heading):
    """
    Heading for translation tasks.
    
    Constraints:
    - Preserve meaning (fidelity mode)
    - No new claims
    - Target language specified
    """
    heading_type: HeadingType = field(default=HeadingType.TRANSLATE, init=False)
    
    # Translation-specific
    source_text: str = ""
    source_lang: str = "auto"
    target_lang: str = "en"
    fidelity_mode: FidelityMode = FidelityMode.STRICT
    preserve_formatting: bool = True
    
    # Override: no new claims
    max_new_claims: int = field(default=0, init=False)
    
    def get_allowed_transforms(self) -> Set[str]:
        transforms = {"translate"}
        if self.fidelity_mode != FidelityMode.STRICT:
            transforms.add("paraphrase")
        return transforms
    
    def check_output(self, output: str, claims: List[Dict]) -> tuple[bool, str]:
        # New claims check
        new_claims = [c for c in claims if c.get("is_new", False)]
        if new_claims:
            return False, f"New claims introduced during translation: {len(new_claims)}"
        
        return True, "OK"


@dataclass
class ExtractClaimsHeading(Heading):
    """
    Heading for claim extraction tasks.
    
    Constraints:
    - Extract only, don't synthesize
    - Output must be structured
    - Claims must be traceable to source
    """
    heading_type: HeadingType = field(default=HeadingType.EXTRACT_CLAIMS, init=False)
    
    # Extraction-specific
    source_text: str = ""
    source_id: str = ""
    output_format: str = "json"  # "json", "markdown", "list"
    claim_schema: Optional[Dict] = None
    include_confidence: bool = True
    include_span: bool = True
    
    # Override: no new claims
    max_new_claims: int = field(default=0, init=False)
    
    def get_allowed_transforms(self) -> Set[str]:
        return {
            "extract",
            "structure",
            "normalize",
        }
    
    def check_output(self, output: str, claims: List[Dict]) -> tuple[bool, str]:
        # All claims must have source reference
        for claim in claims:
            if not claim.get("source_span") and self.include_span:
                return False, f"Claim missing source span: {claim.get('content', '')[:50]}"
        
        return True, "OK"


@dataclass
class RewriteHeading(Heading):
    """
    Heading for rewriting/paraphrasing tasks.
    
    Constraints:
    - Preserve claims (default)
    - Apply style changes
    - No new claims unless explicitly allowed
    """
    heading_type: HeadingType = field(default=HeadingType.REWRITE, init=False)
    
    # Rewrite-specific
    source_text: str = ""
    style_profile: str = "neutral"  # "formal", "casual", "technical", etc.
    preserve_claims: bool = True
    simplify: bool = False
    target_audience: str = ""
    
    def get_allowed_transforms(self) -> Set[str]:
        transforms = {"rewrite", "paraphrase", "restructure"}
        if self.simplify:
            transforms.add("simplify")
        return transforms
    
    def check_output(self, output: str, claims: List[Dict]) -> tuple[bool, str]:
        if self.preserve_claims:
            new_claims = [c for c in claims if c.get("is_new", False)]
            if new_claims:
                return False, f"Claims changed during rewrite: {len(new_claims)} new"
        
        return True, "OK"


@dataclass
class ElaborateHeading(Heading):
    """
    Heading for bounded elaboration.
    
    This heading ALLOWS epistemic expansion, but only within explicit bounds.
    Requires: scope fence, allowed sources, claim budget.
    """
    heading_type: HeadingType = field(default=HeadingType.ELABORATE, init=False)
    
    # Elaboration-specific
    source_text: str = ""
    scope_fence: List[str] = field(default_factory=list)  # Topics that are in-scope
    out_of_scope: List[str] = field(default_factory=list)  # Explicitly forbidden topics
    
    # Must be explicit about expansion
    max_new_claims: int = 5  # Default: allow up to 5 new claims
    require_sources_for_new: bool = True
    
    def get_allowed_transforms(self) -> Set[str]:
        return {
            "elaborate",
            "expand",
            "add_context",
            "add_examples",
        }
    
    def check_output(self, output: str, claims: List[Dict]) -> tuple[bool, str]:
        new_claims = [c for c in claims if c.get("is_new", False)]
        
        if len(new_claims) > self.max_new_claims:
            return False, f"Too many new claims: {len(new_claims)} > {self.max_new_claims}"
        
        if self.require_sources_for_new:
            unsourced = [c for c in new_claims if not c.get("evidence_refs")]
            if unsourced:
                return False, f"New claims without sources: {len(unsourced)}"
        
        return True, "OK"


@dataclass
class AnswerFromSourcesHeading(Heading):
    """
    Heading for answering questions from specific sources.
    
    Bounded expansion: can synthesize, but only from declared sources.
    """
    heading_type: HeadingType = field(default=HeadingType.ANSWER_FROM_SOURCES, init=False)
    
    # Answer-specific
    question: str = ""
    allowed_source_ids: List[str] = field(default_factory=list)
    citation_policy: CitationPolicy = CitationPolicy.REQUIRED
    
    # Bounded expansion
    max_new_claims: int = 10
    allow_inference: bool = True  # Allow derived claims
    
    def get_allowed_transforms(self) -> Set[str]:
        transforms = {"answer", "cite", "quote"}
        if self.allow_inference:
            transforms.add("derive")
        return transforms
    
    def check_output(self, output: str, claims: List[Dict]) -> tuple[bool, str]:
        # Check all claims cite allowed sources
        for claim in claims:
            sources = claim.get("evidence_refs", [])
            for source in sources:
                source_id = source.get("source_id", "")
                if source_id and source_id not in self.allowed_source_ids:
                    return False, f"Claim cites forbidden source: {source_id}"
        
        return True, "OK"


# =============================================================================
# Heading Validator
# =============================================================================

class HeadingValidator:
    """
    Validates operations against a heading's constraints.
    
    Used by autopilot to check if actions are permitted.
    """
    
    def __init__(self, heading: Heading):
        self.heading = heading
        self._allowed_transforms = heading.get_allowed_transforms()
    
    def check_transform(self, transform: str) -> tuple[bool, str]:
        """Check if a transform is allowed under this heading."""
        if transform in self._allowed_transforms:
            return True, "OK"
        return False, f"Transform '{transform}' not permitted under {self.heading.heading_type.name}"
    
    def check_new_claim(self, claim_count: int) -> tuple[bool, str]:
        """Check if adding new claims is allowed."""
        if claim_count <= self.heading.max_new_claims:
            return True, "OK"
        return False, f"New claim limit exceeded: {claim_count} > {self.heading.max_new_claims}"
    
    def check_source(self, source_id: str) -> tuple[bool, str]:
        """Check if a source is allowed."""
        if not self.heading.allowed_source_ids:
            return True, "OK (no source restrictions)"
        if source_id in self.heading.allowed_source_ids:
            return True, "OK"
        return False, f"Source '{source_id}' not in allowed list"
    
    def check_autopilot_eligible(self) -> tuple[bool, str]:
        """Check if this heading is eligible for autopilot."""
        if self.heading.heading_type in FORBIDDEN_HEADINGS:
            return False, f"Heading type {self.heading.heading_type.name} forbidden in autopilot"
        
        if self.heading.heading_type in BOUNDED_EXPANSION_HEADINGS:
            # Check that bounds are actually set
            if isinstance(self.heading, ElaborateHeading):
                if not self.heading.scope_fence:
                    return False, "ElaborateHeading requires scope_fence for autopilot"
            if isinstance(self.heading, AnswerFromSourcesHeading):
                if not self.heading.allowed_source_ids:
                    return False, "AnswerFromSourcesHeading requires allowed_source_ids for autopilot"
        
        return True, "OK"
    
    def validate_output(self, output: str, claims: List[Dict]) -> tuple[bool, str]:
        """Full validation of output against heading."""
        return self.heading.check_output(output, claims)


# =============================================================================
# Heading Serializer
# =============================================================================

class HeadingSerializer:
    """Serializes headings for logs and reproducibility."""
    
    @staticmethod
    def to_dict(heading: Heading) -> Dict[str, Any]:
        """Convert heading to dictionary."""
        base = heading.to_dict()
        
        # Add type-specific fields
        if isinstance(heading, SummarizeHeading):
            base.update({
                "source_id": heading.source_id,
                "target_length": heading.target_length,
                "citation_policy": heading.citation_policy.value,
            })
        elif isinstance(heading, TranslateHeading):
            base.update({
                "source_lang": heading.source_lang,
                "target_lang": heading.target_lang,
                "fidelity_mode": heading.fidelity_mode.value,
            })
        elif isinstance(heading, ElaborateHeading):
            base.update({
                "scope_fence": heading.scope_fence,
                "out_of_scope": heading.out_of_scope,
                "require_sources_for_new": heading.require_sources_for_new,
            })
        
        return base
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Heading:
        """Reconstruct heading from dictionary."""
        heading_type = HeadingType[data["heading_type"]]
        
        if heading_type == HeadingType.SUMMARIZE:
            return SummarizeHeading(
                source_id=data.get("source_id", ""),
                target_length=data.get("target_length", 500),
                citation_policy=CitationPolicy(data.get("citation_policy", "required")),
            )
        elif heading_type == HeadingType.TRANSLATE:
            return TranslateHeading(
                source_lang=data.get("source_lang", "auto"),
                target_lang=data.get("target_lang", "en"),
                fidelity_mode=FidelityMode(data.get("fidelity_mode", "strict")),
            )
        # ... other types
        
        raise ValueError(f"Unknown heading type: {heading_type}")


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=== Typed Headings Demo ===\n")
    
    # Create a summarize heading
    heading = SummarizeHeading(
        source_id="doc_001",
        source_text="This is the source document...",
        target_length=200,
        citation_policy=CitationPolicy.REQUIRED,
    )
    
    print(f"Heading type: {heading.heading_type.name}")
    print(f"Allows expansion: {heading.allows_expansion}")
    print(f"Autopilot safe: {heading.is_autopilot_safe}")
    print(f"Allowed transforms: {heading.get_allowed_transforms()}")
    
    # Validate
    validator = HeadingValidator(heading)
    
    print("\n--- Validation Checks ---")
    print(f"compress: {validator.check_transform('compress')}")
    print(f"synthesize: {validator.check_transform('synthesize')}")
    print(f"new_claim(0): {validator.check_new_claim(0)}")
    print(f"new_claim(1): {validator.check_new_claim(1)}")
    print(f"autopilot eligible: {validator.check_autopilot_eligible()}")
    
    # Elaborate heading (bounded expansion)
    print("\n--- Elaborate Heading ---")
    elaborate = ElaborateHeading(
        source_text="The original text...",
        scope_fence=["machine learning", "neural networks"],
        out_of_scope=["politics", "religion"],
        max_new_claims=5,
    )
    
    validator2 = HeadingValidator(elaborate)
    print(f"Allows expansion: {elaborate.allows_expansion}")
    print(f"Max new claims: {elaborate.max_new_claims}")
    print(f"autopilot eligible: {validator2.check_autopilot_eligible()}")
    
    # Forbidden heading
    print("\n--- Forbidden Heading Check ---")
    print(f"HYPOTHESIZE autopilot safe: {HeadingType.HYPOTHESIZE in AUTOPILOT_SAFE_HEADINGS}")
    print(f"HYPOTHESIZE forbidden: {HeadingType.HYPOTHESIZE in FORBIDDEN_HEADINGS}")
