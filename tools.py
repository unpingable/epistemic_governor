"""
Tool Interfaces for the Epistemic Governor

Tools that belong in a governor create HARD, EXTERNAL FRICTION.
Tools that just add more text don't belong.

Integration priority:
1. Schema-constrained decoding (immediate quality jump)
2. Provenance hashing (makes support real)
3. Fuzzer (keeps you honest)
4. SMT checks (when typed claims stabilize)

Key principle: Tools are organs, not skeletons.
- Vector DBs supply oxygen (evidence candidates)
- SMT solvers supply structural truth (consistency)
- Provenance stores supply verification
- None of them provide commitments - that's the ledger's job

Usage:
    from epistemic_governor.tools import (
        SupportCandidate,
        SupportRetriever,
        ProvenanceStore,
        ClaimCompiler,
        ConsistencyChecker,
    )
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Protocol, Tuple
from enum import Enum, auto
from datetime import datetime
import hashlib
import json
from abc import ABC, abstractmethod


# =============================================================================
# Support Retrieval (Vector DB / Search Interface)
# =============================================================================

def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace for canonical hashing."""
    return " ".join(text.split())


@dataclass(frozen=True)
class SupportCandidate:
    """
    A candidate piece of support from retrieval.
    
    This is a PROPOSAL, not truth. The governor decides
    whether to attach it to a commitment.
    
    Canonical ID is hash of: source_uri | span | normalized_quote
    """
    source_uri: str               # Full URI (not just doc_id)
    span_start: int               # Character offset start
    span_end: int                 # Character offset end
    quote: str                    # Exact text
    score: float                  # Retrieval score (0-1)
    source_type: str              # "document", "tool_result", "user_input", etc.
    
    # Optional metadata
    title: Optional[str] = None
    retrieved_at: Optional[datetime] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def normalized_quote(self) -> str:
        """Whitespace-normalized quote for hashing."""
        return _normalize_whitespace(self.quote)
    
    @property
    def canonical_blob(self) -> str:
        """Canonical blob for hashing (stable ID)."""
        # Note: retrieved_at NOT included - we want stable IDs
        return f"{self.source_uri}|{self.span_start}:{self.span_end}|{self.normalized_quote}"
    
    @property
    def full_hash(self) -> str:
        """Full SHA256 hash of canonical blob."""
        return hashlib.sha256(self.canonical_blob.encode()).hexdigest()
    
    @property
    def support_id(self) -> str:
        """Truncated hash for display (full hash stored internally)."""
        return f"sup:{self.full_hash[:12]}"
    
    @property
    def span(self) -> Tuple[int, int]:
        """Span as tuple for convenience."""
        return (self.span_start, self.span_end)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "source_uri": self.source_uri,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "quote": self.quote,
            "score": self.score,
            "source_type": self.source_type,
            "title": self.title,
            "retrieved_at": self.retrieved_at.isoformat() if self.retrieved_at else None,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "support_id": self.support_id,
            "full_hash": self.full_hash,
        }


class SupportRetriever(Protocol):
    """
    Protocol for support retrieval tools.
    
    Implementations might use:
    - Vector DB (semantic search)
    - BM25 (keyword search)
    - Grep (exact match)
    - Combined (hybrid)
    
    Key: Returns CANDIDATES only. Never grants truth.
    """
    
    def retrieve(
        self,
        query: str,
        max_results: int = 5,
        min_score: float = 0.5,
    ) -> List[SupportCandidate]:
        """Retrieve support candidates for a query."""
        ...
    
    def retrieve_for_claim(
        self,
        subject: str,
        predicate: str,
        object: str,
    ) -> List[SupportCandidate]:
        """Retrieve support candidates for a typed claim."""
        ...


class MockRetriever:
    """Mock retriever for testing."""
    
    def __init__(self, candidates: List[SupportCandidate] = None):
        self.candidates = candidates or []
    
    def retrieve(
        self,
        query: str,
        max_results: int = 5,
        min_score: float = 0.5,
    ) -> List[SupportCandidate]:
        return [c for c in self.candidates if c.score >= min_score][:max_results]
    
    def retrieve_for_claim(
        self,
        subject: str,
        predicate: str,
        object: str,
    ) -> List[SupportCandidate]:
        query = f"{subject} {predicate} {object}"
        return self.retrieve(query)
    
    @classmethod
    def with_sample_data(cls) -> "MockRetriever":
        """Create mock retriever with sample candidates."""
        candidates = [
            SupportCandidate(
                source_uri="wiki:france",
                span_start=0,
                span_end=47,
                quote="Paris is the capital and largest city of France",
                score=0.95,
                source_type="document",
                title="France - Wikipedia",
                line_start=1,
                line_end=1,
            ),
        ]
        return cls(candidates)


# =============================================================================
# Provenance Store (Content-Addressed)
# =============================================================================

class VerificationStatus(Enum):
    """Status of provenance verification."""
    VERIFIED = auto()       # Content matches what was stored
    MISMATCH = auto()       # Content differs from stored
    NOT_FOUND = auto()      # Hash not in store
    SOURCE_UNAVAILABLE = auto()  # Can't re-fetch to verify


@dataclass
class VerificationResult:
    """Result of verifying provenance."""
    status: VerificationStatus
    hash: str
    stored_content: Optional[str] = None
    current_content: Optional[str] = None
    difference: Optional[str] = None


@dataclass
class ProvenanceRecord:
    """
    A content-addressed provenance record.
    
    Every support snippet has:
    - A hash (content address)
    - Source information
    - Span/location
    - Timestamp of capture
    
    If source changes, hash changes, you know.
    """
    hash: str                     # Full SHA256 hash
    content: str                  # Exact content at time of capture
    source_uri: str
    span_start: Optional[int] = None
    span_end: Optional[int] = None
    captured_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # For verification
    verified_at: Optional[datetime] = None
    verification_status: Optional[VerificationStatus] = None
    
    @classmethod
    def create(cls, content: str, source_uri: str, **kwargs) -> "ProvenanceRecord":
        """Create a provenance record with computed hash."""
        normalized = _normalize_whitespace(content)
        content_hash = hashlib.sha256(normalized.encode()).hexdigest()
        return cls(hash=content_hash, content=content, source_uri=source_uri, **kwargs)
    
    @property
    def short_hash(self) -> str:
        """Truncated hash for display."""
        return self.hash[:16]


class ProvenanceStore:
    """
    Content-addressed store for support provenance.
    
    Two distinct operations:
    - ingest(): Store what we saw then
    - verify(): Check if world still agrees
    
    This turns "citations" from vibes into verifiable substrate.
    """
    
    def __init__(self):
        self._records: Dict[str, ProvenanceRecord] = {}
        self._by_source: Dict[str, List[str]] = {}  # source_uri -> [hashes]
    
    def ingest(
        self,
        candidate: SupportCandidate,
    ) -> str:
        """
        Ingest a support candidate into the store.
        
        Returns the full hash. This is "what we saw then."
        """
        record = ProvenanceRecord.create(
            content=candidate.quote,
            source_uri=candidate.source_uri,
            span_start=candidate.span_start,
            span_end=candidate.span_end,
            metadata={
                "score": candidate.score,
                "source_type": candidate.source_type,
                "title": candidate.title,
                "retrieved_at": candidate.retrieved_at.isoformat() if candidate.retrieved_at else None,
            },
        )
        
        self._records[record.hash] = record
        
        # Index by source
        if candidate.source_uri not in self._by_source:
            self._by_source[candidate.source_uri] = []
        if record.hash not in self._by_source[candidate.source_uri]:
            self._by_source[candidate.source_uri].append(record.hash)
        
        return record.hash
    
    def store(self, content: str, source_uri: str, **metadata) -> str:
        """
        Store raw content (simpler interface).
        
        Returns the full hash.
        """
        record = ProvenanceRecord.create(content, source_uri, metadata=metadata)
        self._records[record.hash] = record
        
        if source_uri not in self._by_source:
            self._by_source[source_uri] = []
        if record.hash not in self._by_source[source_uri]:
            self._by_source[source_uri].append(record.hash)
        
        return record.hash
    
    def verify(
        self,
        hash: str,
        current_content: Optional[str] = None,
        fetch_fn: Optional[Callable[[str], Optional[str]]] = None,
    ) -> VerificationResult:
        """
        Verify that content matches stored provenance.
        
        This is "does the world still agree."
        
        Args:
            hash: The provenance hash to verify
            current_content: Content to verify against (if already fetched)
            fetch_fn: Function to fetch current content from source
        """
        if hash not in self._records:
            return VerificationResult(
                status=VerificationStatus.NOT_FOUND,
                hash=hash,
            )
        
        record = self._records[hash]
        
        # Get current content
        if current_content is None and fetch_fn is not None:
            current_content = fetch_fn(record.source_uri)
        
        if current_content is None:
            return VerificationResult(
                status=VerificationStatus.SOURCE_UNAVAILABLE,
                hash=hash,
                stored_content=record.content,
            )
        
        # Compare
        stored_normalized = _normalize_whitespace(record.content)
        current_normalized = _normalize_whitespace(current_content)
        
        if stored_normalized == current_normalized:
            record.verified_at = datetime.now()
            record.verification_status = VerificationStatus.VERIFIED
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                hash=hash,
                stored_content=record.content,
                current_content=current_content,
            )
        else:
            record.verified_at = datetime.now()
            record.verification_status = VerificationStatus.MISMATCH
            return VerificationResult(
                status=VerificationStatus.MISMATCH,
                hash=hash,
                stored_content=record.content,
                current_content=current_content,
                difference=f"Stored: {len(stored_normalized)} chars, Current: {len(current_normalized)} chars",
            )
    
    def get(self, hash: str) -> Optional[ProvenanceRecord]:
        """Retrieve a record by hash."""
        return self._records.get(hash)
    
    def exists(self, hash: str) -> bool:
        """Check if a hash exists in the store."""
        return hash in self._records
    
    def get_by_source(self, source_uri: str) -> List[ProvenanceRecord]:
        """Get all records from a source."""
        hashes = self._by_source.get(source_uri, [])
        return [self._records[h] for h in hashes if h in self._records]


# =============================================================================
# Claim Compiler (Text → Typed Claims)
# =============================================================================

@dataclass
class CompiledClaim:
    """
    A claim compiled from free text.
    
    This is a PROPOSAL for the hypothesis ledger,
    not a commitment.
    """
    subject: str
    predicate: str
    object: str
    confidence: float
    source_text: str
    source_span: Tuple[int, int]
    
    # Validation status
    valid_subject: bool = False
    valid_predicate: bool = False
    valid_object: bool = False
    
    @property
    def is_valid(self) -> bool:
        return self.valid_subject and self.valid_predicate and self.valid_object
    
    @property
    def validation_errors(self) -> List[str]:
        errors = []
        if not self.valid_subject:
            errors.append(f"Unknown subject: {self.subject}")
        if not self.valid_predicate:
            errors.append(f"Unknown predicate: {self.predicate}")
        if not self.valid_object:
            errors.append(f"Unknown object: {self.object}")
        return errors


class ClaimCompiler(Protocol):
    """
    Protocol for claim compilation.
    
    Transforms free text into typed claim proposals.
    
    Pipeline:
    1. Text → extracted candidate predicates (LLM-assisted)
    2. Compiler validates types/scopes/entities
    3. Only valid claims can enter hypothesis ledger
    """
    
    def compile(self, text: str) -> List[CompiledClaim]:
        """Compile text into typed claim proposals."""
        ...
    
    def validate_entity(self, entity: str) -> Tuple[bool, Optional[str]]:
        """Validate an entity reference. Returns (valid, normalized_id)."""
        ...
    
    def validate_predicate(self, predicate: str) -> Tuple[bool, Optional[str]]:
        """Validate a predicate. Returns (valid, normalized_name)."""
        ...


# =============================================================================
# Consistency Checker (SMT Interface)
# =============================================================================

class ConsistencyResult(Enum):
    """Result of a consistency check."""
    SAT = auto()          # Claims are consistent (satisfiable)
    UNSAT = auto()        # Claims contradict (unsatisfiable)
    UNKNOWN = auto()      # Solver couldn't decide (treat as cannot commit)


class RetractionObjective(Enum):
    """Objective for minimal retraction."""
    MIN_COST = auto()          # Minimize total retraction cost
    MIN_BLAST_RADIUS = auto()  # Minimize number of claims retracted
    LEXICOGRAPHIC = auto()     # Minimize blast radius, then cost


@dataclass
class ConsistencyReport:
    """
    Report from consistency checking.
    
    Includes unsat core (minimal conflicting set) when UNSAT.
    """
    result: ConsistencyResult
    
    # For SAT: optional witness (assignment that satisfies)
    witness: Optional[Dict[str, Any]] = None
    
    # For UNSAT: minimal conflicting set (unsat core)
    unsat_core: List[str] = field(default_factory=list)
    
    # For retraction planning
    minimal_retraction: List[str] = field(default_factory=list)
    retraction_cost: float = 0.0
    blast_radius: int = 0
    objective_used: Optional[RetractionObjective] = None
    
    # Solver metadata
    solver_time_ms: float = 0.0
    solver_name: str = "mock"
    
    def describe(self) -> str:
        if self.result == ConsistencyResult.SAT:
            return "Claims are consistent (SAT)"
        elif self.result == ConsistencyResult.UNSAT:
            return (
                f"CONFLICT (UNSAT): core={self.unsat_core}. "
                f"Minimal retraction: {self.minimal_retraction} "
                f"(cost={self.retraction_cost:.2f}, blast={self.blast_radius})"
            )
        else:
            return "Consistency UNKNOWN - cannot commit, sandbox only"


class ConsistencyChecker(Protocol):
    """
    Protocol for consistency checking (SMT solver interface).
    
    Use for:
    - Checking if a new claim is consistent with existing commitments
    - Computing minimal retraction sets (with unsat core)
    - Proving dead ends are real (unsat), not heuristic
    
    UNKNOWN status matters: solver sometimes punts. Governor must
    treat UNKNOWN as "cannot commit" or "sandbox only."
    """
    
    def check_consistency(
        self,
        existing_claims: List[Dict],
        new_claim: Dict,
        timeout_ms: int = 1000,
    ) -> ConsistencyReport:
        """Check if new claim is consistent with existing claims."""
        ...
    
    def find_minimal_retraction(
        self,
        existing_claims: List[Dict],
        required_claim: Dict,
        claim_weights: Dict[str, float],
        objective: RetractionObjective = RetractionObjective.MIN_BLAST_RADIUS,
        timeout_ms: int = 5000,
    ) -> ConsistencyReport:
        """
        Find minimal retraction set to admit required claim.
        
        Objectives:
        - MIN_COST: Minimize sum of retraction costs
        - MIN_BLAST_RADIUS: Minimize number of claims retracted
        - LEXICOGRAPHIC: Minimize blast radius first, then cost
        """
        ...


class MockConsistencyChecker:
    """
    Mock consistency checker using simple predicate logic.
    
    For real use, replace with Z3 or similar SMT solver.
    """
    
    def check_consistency(
        self,
        existing_claims: List[Dict],
        new_claim: Dict,
        timeout_ms: int = 1000,
    ) -> ConsistencyReport:
        """Simple contradiction detection."""
        conflicts = []
        
        for claim in existing_claims:
            if self._contradicts(claim, new_claim):
                conflicts.append(claim.get("id", "unknown"))
        
        if conflicts:
            return ConsistencyReport(
                result=ConsistencyResult.UNSAT,
                unsat_core=conflicts,
                solver_name="mock",
            )
        
        return ConsistencyReport(
            result=ConsistencyResult.SAT,
            witness={"new_claim": new_claim.get("id")},
            solver_name="mock",
        )
    
    def _contradicts(self, a: Dict, b: Dict) -> bool:
        """Check if two claims contradict."""
        # Same subject, same predicate type, different object (for unique predicates)
        if a.get("subject") != b.get("subject"):
            return False
        
        pred_a = a.get("predicate", "")
        pred_b = b.get("predicate", "")
        
        # Direct negation
        negation_pairs = [
            ("IS_A", "IS_NOT_A"),
            ("EXISTS", "DOES_NOT_EXIST"),
            ("HAS_PROPERTY", "LACKS_PROPERTY"),
        ]
        for pos, neg in negation_pairs:
            if (pred_a == pos and pred_b == neg) or (pred_a == neg and pred_b == pos):
                if a.get("object") == b.get("object"):
                    return True
        
        # Unique predicates with different values
        unique_preds = ["IS_CAPITAL_OF", "HAS_VALUE"]
        if pred_a == pred_b and pred_a in unique_preds:
            if a.get("object") != b.get("object"):
                return True
        
        return False
    
    def find_minimal_retraction(
        self,
        existing_claims: List[Dict],
        required_claim: Dict,
        claim_weights: Dict[str, float],
        objective: RetractionObjective = RetractionObjective.MIN_BLAST_RADIUS,
        timeout_ms: int = 5000,
    ) -> ConsistencyReport:
        """Find claims to retract to admit required claim."""
        conflicts = []
        
        for claim in existing_claims:
            if self._contradicts(claim, required_claim):
                conflicts.append(claim.get("id", "unknown"))
        
        if not conflicts:
            return ConsistencyReport(
                result=ConsistencyResult.SAT,
                solver_name="mock",
            )
        
        # Sort based on objective
        if objective == RetractionObjective.MIN_COST:
            conflicts.sort(key=lambda cid: claim_weights.get(cid, 1.0))
        elif objective == RetractionObjective.MIN_BLAST_RADIUS:
            # Already minimal (each conflict must be retracted)
            pass
        elif objective == RetractionObjective.LEXICOGRAPHIC:
            # Blast radius is fixed (len(conflicts)), so sort by cost
            conflicts.sort(key=lambda cid: claim_weights.get(cid, 1.0))
        
        total_cost = sum(claim_weights.get(cid, 1.0) for cid in conflicts)
        
        return ConsistencyReport(
            result=ConsistencyResult.UNSAT,
            unsat_core=conflicts,
            minimal_retraction=conflicts,
            retraction_cost=total_cost,
            blast_radius=len(conflicts),
            objective_used=objective,
            solver_name="mock",
        )


# =============================================================================
# Constrained Decoding Interface
# =============================================================================

class OutputType(Enum):
    """Valid output types from the model under constraint."""
    ACC_ADD = auto()        # Add to accretion
    ACC_QUERY = auto()      # Query for missing info
    ACC_RENDER = auto()     # Render current state
    ACC_REVISE = auto()     # Revise existing claim
    ACC_DEFER = auto()      # Defer (can't answer)
    RETRACT_PLAN = auto()   # Propose retraction
    HYPOTHESIS = auto()     # Propose hypothesis
    TOOL_CALL = auto()      # Call external tool


@dataclass
class ShapeConstraint:
    """
    Shape constraints (grammar/schema) - hard parse.
    
    These are syntactic constraints enforced at decode time.
    """
    allowed_types: List[OutputType]
    max_tokens: int = 500
    json_schema: Optional[Dict] = None
    grammar: Optional[str] = None  # GBNF/regex
    
    def to_grammar(self) -> str:
        """Generate grammar for constrained decoding."""
        type_alts = " | ".join(t.name for t in self.allowed_types)
        return self.grammar or f"output ::= ({type_alts}) content"
    
    def to_json_schema(self) -> Dict:
        """Generate JSON schema for constrained decoding."""
        return self.json_schema or {
            "type": "object",
            "properties": {
                "type": {"enum": [t.name for t in self.allowed_types]},
                "content": {"type": "object"},
            },
            "required": ["type", "content"],
        }


@dataclass
class PolicyConstraint:
    """
    Policy constraints (semantic validation) - checked after parse.
    
    These are semantic constraints checked after decoding.
    """
    require_support_ids: bool = False
    require_entity_refs: bool = False
    min_confidence: Optional[float] = None
    max_claims_per_output: Optional[int] = None
    forbidden_predicates: List[str] = field(default_factory=list)
    
    def validate(self, output: Dict) -> List[str]:
        """
        Validate output against policy constraints.
        
        Returns list of violations (empty if valid).
        """
        violations = []
        
        if self.require_support_ids:
            support_ids = output.get("content", {}).get("support_ids", [])
            if not support_ids:
                violations.append("Missing required support_ids")
        
        if self.require_entity_refs:
            subject = output.get("content", {}).get("subject", "")
            if ":" not in subject:
                violations.append(f"Subject '{subject}' is not a namespaced entity ref")
        
        if self.min_confidence is not None:
            confidence = output.get("content", {}).get("confidence", 0)
            if confidence < self.min_confidence:
                violations.append(f"Confidence {confidence} below minimum {self.min_confidence}")
        
        predicate = output.get("content", {}).get("predicate", "")
        if predicate in self.forbidden_predicates:
            violations.append(f"Forbidden predicate: {predicate}")
        
        return violations


@dataclass
class OutputConstraint:
    """
    Combined output constraint (shape + policy).
    
    Two layers:
    - Shape: grammar/schema (hard parse)
    - Policy: semantic validation (checked after parse)
    
    This matches the sprinkler model: parse gate, then policy gate.
    """
    shape: ShapeConstraint
    policy: PolicyConstraint = field(default_factory=PolicyConstraint)
    
    @classmethod
    def simple(
        cls,
        allowed_types: List[OutputType],
        max_tokens: int = 500,
        require_support_ids: bool = False,
    ) -> "OutputConstraint":
        """Create a simple constraint."""
        return cls(
            shape=ShapeConstraint(allowed_types=allowed_types, max_tokens=max_tokens),
            policy=PolicyConstraint(require_support_ids=require_support_ids),
        )
    
    def to_grammar(self) -> str:
        return self.shape.to_grammar()
    
    def to_json_schema(self) -> Dict:
        return self.shape.to_json_schema()
    
    def validate(self, output: Dict) -> Tuple[bool, List[str]]:
        """
        Validate output against both shape and policy.
        
        Returns (is_valid, violations).
        """
        violations = []
        
        # Check type is allowed
        output_type = output.get("type")
        allowed_names = [t.name for t in self.shape.allowed_types]
        if output_type not in allowed_names:
            violations.append(f"Output type '{output_type}' not in allowed types: {allowed_names}")
        
        # Check policy
        violations.extend(self.policy.validate(output))
        
        return len(violations) == 0, violations


class ConstrainedDecoder(Protocol):
    """
    Protocol for constrained decoding.
    
    This is the sprinkler system's best friend.
    Not "please output JSON" — actual constrained decoding.
    """
    
    def decode(
        self,
        prompt: str,
        constraint: OutputConstraint,
    ) -> Dict[str, Any]:
        """Decode with constraints enforced."""
        ...
    
    def validate_output(
        self,
        output: str,
        constraint: OutputConstraint,
    ) -> Tuple[bool, List[str]]:
        """Validate output against constraint. Returns (valid, errors)."""
        ...


# =============================================================================
# Execution Sandbox
# =============================================================================

@dataclass
class ExecutionResult:
    """Result from sandboxed execution."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_value: Any = None
    execution_time_ms: float = 0.0
    
    @property
    def as_support(self) -> str:
        """Format as support evidence."""
        if self.success:
            return f"Execution succeeded: {self.stdout or self.return_value}"
        return f"Execution failed: {self.stderr}"


class ExecutionSandbox(Protocol):
    """
    Protocol for sandboxed execution.
    
    When domain allows (code/config/math), you can DECIDE things.
    Execution transcripts are the best kind of support.
    """
    
    def execute_python(self, code: str, timeout_ms: int = 5000) -> ExecutionResult:
        """Execute Python code in sandbox."""
        ...
    
    def evaluate_math(self, expression: str) -> ExecutionResult:
        """Evaluate mathematical expression."""
        ...
    
    def run_tests(self, code: str, tests: List[str]) -> ExecutionResult:
        """Run tests against code."""
        ...


# =============================================================================
# Tool Call Context (Causal Tracking)
# =============================================================================

@dataclass
class ToolCallContext:
    """
    Context for a tool call - prevents race conditions and mis-attribution.
    
    Every tool call and result carries:
    - session_id
    - turn_id
    - tool_call_id
    - causal_parent (what triggered it)
    """
    session_id: str
    turn_id: int
    tool_call_id: str
    causal_parent: Optional[str] = None  # dead_end_id / query_id that triggered
    timestamp: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def create(
        cls,
        session_id: str,
        turn_id: int,
        causal_parent: Optional[str] = None,
    ) -> "ToolCallContext":
        """Create a new tool call context."""
        call_id = f"tc_{session_id}_{turn_id}_{datetime.now().timestamp()}"
        return cls(
            session_id=session_id,
            turn_id=turn_id,
            tool_call_id=call_id,
            causal_parent=causal_parent,
        )


@dataclass
class ToolResult:
    """
    Result of a tool call with full context.
    """
    context: ToolCallContext
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    
    @property
    def as_support_candidate(self) -> Optional[SupportCandidate]:
        """Convert to support candidate if applicable."""
        if not self.success or not isinstance(self.result, dict):
            return None
        
        # Try to extract support candidate from result
        if "quote" in self.result and "source_uri" in self.result:
            return SupportCandidate(
                source_uri=self.result["source_uri"],
                span_start=self.result.get("span_start", 0),
                span_end=self.result.get("span_end", 0),
                quote=self.result["quote"],
                score=self.result.get("score", 0.5),
                source_type="tool_result",
                metadata={"tool_call_id": self.context.tool_call_id},
            )
        return None


# =============================================================================
# Tool Registry (Connects tools to governor)
# =============================================================================

class ToolRegistry:
    """
    Registry of tools available to the governor.
    
    Tools are organs, not skeletons:
    - They supply evidence, candidates, verification
    - They don't make commitments
    - The governor decides what to believe
    
    All tool calls are tracked with session/turn/call IDs
    to prevent race conditions and mis-attribution.
    """
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or f"session_{datetime.now().timestamp()}"
        self.current_turn: int = 0
        
        # Tools
        self.retriever: Optional[SupportRetriever] = None
        self.provenance: ProvenanceStore = ProvenanceStore()
        self.compiler: Optional[ClaimCompiler] = None
        self.checker: ConsistencyChecker = MockConsistencyChecker()
        self.decoder: Optional[ConstrainedDecoder] = None
        self.sandbox: Optional[ExecutionSandbox] = None
        
        # Call history (for debugging and replay)
        self.call_history: List[ToolResult] = []
    
    def set_turn(self, turn_id: int):
        """Set current turn (called by session)."""
        self.current_turn = turn_id
    
    def _create_context(self, causal_parent: Optional[str] = None) -> ToolCallContext:
        """Create context for a tool call."""
        return ToolCallContext.create(
            session_id=self.session_id,
            turn_id=self.current_turn,
            causal_parent=causal_parent,
        )
    
    def _record_result(self, result: ToolResult):
        """Record tool result in history."""
        self.call_history.append(result)
    
    def register_retriever(self, retriever: SupportRetriever):
        """Register support retriever."""
        self.retriever = retriever
    
    def register_compiler(self, compiler: ClaimCompiler):
        """Register claim compiler."""
        self.compiler = compiler
    
    def register_checker(self, checker: ConsistencyChecker):
        """Register consistency checker."""
        self.checker = checker
    
    def register_decoder(self, decoder: ConstrainedDecoder):
        """Register constrained decoder."""
        self.decoder = decoder
    
    def register_sandbox(self, sandbox: ExecutionSandbox):
        """Register execution sandbox."""
        self.sandbox = sandbox
    
    # === High-level operations with context ===
    
    def acquire_support(
        self,
        query: str,
        min_score: float = 0.5,
        causal_parent: Optional[str] = None,
    ) -> List[Tuple[SupportCandidate, str]]:
        """
        Acquire support candidates and store provenance.
        
        Returns list of (candidate, provenance_hash) tuples.
        All tracked with proper context.
        """
        context = self._create_context(causal_parent)
        
        if not self.retriever:
            self._record_result(ToolResult(
                context=context,
                tool_name="retriever",
                success=False,
                result=None,
                error="No retriever registered",
            ))
            return []
        
        start_time = datetime.now()
        candidates = self.retriever.retrieve(query, min_score=min_score)
        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        results = []
        for candidate in candidates:
            # Store in provenance with context
            prov_hash = self.provenance.ingest(candidate)
            results.append((candidate, prov_hash))
        
        self._record_result(ToolResult(
            context=context,
            tool_name="retriever",
            success=True,
            result={"candidates": len(candidates), "query": query},
            execution_time_ms=elapsed_ms,
        ))
        
        return results
    
    def check_claim_consistency(
        self,
        existing_claims: List[Dict],
        new_claim: Dict,
        causal_parent: Optional[str] = None,
    ) -> ConsistencyReport:
        """Check if new claim is consistent with existing."""
        context = self._create_context(causal_parent)
        
        start_time = datetime.now()
        report = self.checker.check_consistency(existing_claims, new_claim)
        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        report.solver_time_ms = elapsed_ms
        
        self._record_result(ToolResult(
            context=context,
            tool_name="consistency_checker",
            success=True,
            result={"status": report.result.name, "new_claim": new_claim.get("id")},
            execution_time_ms=elapsed_ms,
        ))
        
        return report
    
    def plan_admission(
        self,
        existing_claims: List[Dict],
        required_claim: Dict,
        claim_weights: Dict[str, float],
        objective: RetractionObjective = RetractionObjective.MIN_BLAST_RADIUS,
        causal_parent: Optional[str] = None,
    ) -> ConsistencyReport:
        """Plan what to retract to admit a required claim."""
        context = self._create_context(causal_parent)
        
        start_time = datetime.now()
        report = self.checker.find_minimal_retraction(
            existing_claims, required_claim, claim_weights, objective
        )
        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        report.solver_time_ms = elapsed_ms
        
        self._record_result(ToolResult(
            context=context,
            tool_name="retraction_planner",
            success=True,
            result={
                "status": report.result.name,
                "retraction": report.minimal_retraction,
                "cost": report.retraction_cost,
            },
            execution_time_ms=elapsed_ms,
        ))
        
        return report
    
    # === Replay support ===
    
    def get_calls_for_turn(self, turn_id: int) -> List[ToolResult]:
        """Get all tool calls for a specific turn."""
        return [r for r in self.call_history if r.context.turn_id == turn_id]
    
    def get_calls_by_causal_parent(self, parent_id: str) -> List[ToolResult]:
        """Get all tool calls triggered by a specific dead end or query."""
        return [r for r in self.call_history if r.context.causal_parent == parent_id]


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=== Tool Interfaces Demo ===\n")
    
    # Provenance store with ingest/verify
    print("--- Provenance Store (Ingest/Verify) ---")
    provenance = ProvenanceStore()
    
    # Create a support candidate
    candidate = SupportCandidate(
        source_uri="wiki:france",
        span_start=0,
        span_end=47,
        quote="Paris is the capital and largest city of France",
        score=0.95,
        source_type="document",
        title="France - Wikipedia",
    )
    
    print(f"Candidate canonical blob: {candidate.canonical_blob[:60]}...")
    print(f"Candidate support_id: {candidate.support_id}")
    print(f"Full hash: {candidate.full_hash}")
    
    # Ingest
    h = provenance.ingest(candidate)
    print(f"\nIngested with hash: {h[:16]}...")
    
    # Verify
    result = provenance.verify(h, current_content=candidate.quote)
    print(f"Verification: {result.status.name}")
    
    # Verify with different content
    result2 = provenance.verify(h, current_content="Paris was the capital of France")
    print(f"Verification (modified): {result2.status.name}")
    
    # Consistency checker with unsat core
    print("\n--- Consistency Checker (Unsat Core) ---")
    checker = MockConsistencyChecker()
    
    existing = [
        {"id": "C001", "subject": "geo:paris_fr", "predicate": "IS_CAPITAL_OF", "object": "geo:france"},
    ]
    
    # Try adding contradicting claim
    conflict = {"id": "C002", "subject": "geo:paris_fr", "predicate": "IS_CAPITAL_OF", "object": "geo:germany"}
    result = checker.check_consistency(existing, conflict)
    print(f"Result: {result.result.name}")
    print(f"Unsat core: {result.unsat_core}")
    
    # Retraction planning with objective
    print("\n--- Retraction Planning (Objectives) ---")
    weights = {"C001": 3.0}
    
    report = checker.find_minimal_retraction(
        existing, conflict, weights,
        objective=RetractionObjective.MIN_BLAST_RADIUS,
    )
    print(f"Objective: {report.objective_used.name}")
    print(f"Minimal retraction: {report.minimal_retraction}")
    print(f"Cost: {report.retraction_cost}, Blast radius: {report.blast_radius}")
    
    # Output constraints (shape + policy)
    print("\n--- Output Constraints (Shape + Policy) ---")
    constraint = OutputConstraint.simple(
        allowed_types=[OutputType.ACC_ADD, OutputType.ACC_QUERY],
        require_support_ids=True,
    )
    
    # Valid output
    valid_output = {
        "type": "ACC_ADD",
        "content": {
            "claim": "Paris is capital",
            "support_ids": ["sup:abc123"],
            "subject": "geo:paris_fr",
        }
    }
    is_valid, violations = constraint.validate(valid_output)
    print(f"Valid output: {is_valid}, violations: {violations}")
    
    # Invalid output (missing support)
    invalid_output = {
        "type": "ACC_ADD",
        "content": {"claim": "Paris is capital"}
    }
    is_valid, violations = constraint.validate(invalid_output)
    print(f"Invalid output: {is_valid}, violations: {violations}")
    
    # Tool registry with context
    print("\n--- Tool Registry (Context Tracking) ---")
    registry = ToolRegistry(session_id="demo_session")
    registry.set_turn(1)
    registry.register_retriever(MockRetriever.with_sample_data())
    
    # Acquire support with causal tracking
    results = registry.acquire_support(
        "capital of France",
        causal_parent="dead_end_001",
    )
    print(f"Acquired {len(results)} candidates")
    
    # Check call history
    print(f"Call history: {len(registry.call_history)} calls")
    for call in registry.call_history:
        print(f"  - Turn {call.context.turn_id}: {call.tool_name} "
              f"(parent={call.context.causal_parent})")

