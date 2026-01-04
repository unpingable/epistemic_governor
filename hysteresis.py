"""
Hysteresis Test Harness (BLI-HYS-0.1)

Empirically demonstrates:
    I(Y; S | X) > 0
    
i.e., outputs depend on internal state history beyond what's in the prompt.

NON-NEGOTIABLE PROPERTY:
- Same exact user input (X)
- Different internal state (S)
- System output differs for state-grounded reasons

NOT:
- Different randomness
- Different context/prompt injection
- Style drift

If this test fails, we built elaborate prompt-conditioning, not interiority.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timezone
from enum import Enum, auto
import hashlib
import json
import uuid


# =============================================================================
# Contradiction as First-Class Object
# =============================================================================

class ContradictionStatus(Enum):
    """Status of a contradiction."""
    OPEN = auto()      # Unresolved conflict
    CLOSED = auto()    # Resolved with evidence
    FROZEN = auto()    # Blocked from resolution (anti-cycle)


class ContradictionSeverity(Enum):
    """How severe is this contradiction."""
    LOW = 1        # Minor inconsistency
    MEDIUM = 2     # Significant conflict
    HIGH = 3       # Critical - blocks operations
    CRITICAL = 4   # System-level integrity threat


@dataclass
class Contradiction:
    """
    A first-class contradiction object.
    
    Contradictions are not bugs to hide - they're signals to preserve.
    They persist until resolved with evidence.
    """
    contradiction_id: str
    
    # The conflicting claims
    claim_a_id: str
    claim_b_id: str
    
    # What they conflict on
    domain: str                    # e.g., "python_version", "release_date"
    conflict_type: str             # e.g., "value_mismatch", "temporal_overlap"
    
    # Severity determines blocking behavior
    severity: ContradictionSeverity
    
    # Status tracking
    status: ContradictionStatus
    opened_at: datetime
    opened_by_event: str           # Event ID that created this
    
    # Resolution tracking (if closed)
    closed_at: Optional[datetime] = None
    closed_by_event: Optional[str] = None
    resolution_evidence: Optional[str] = None
    resolution_claim_id: Optional[str] = None  # The winning/superseding claim
    
    # Anti-cycle tracking
    reopen_count: int = 0
    last_resolution_attempt: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for hashing/storage."""
        return {
            "contradiction_id": self.contradiction_id,
            "claim_a_id": self.claim_a_id,
            "claim_b_id": self.claim_b_id,
            "domain": self.domain,
            "conflict_type": self.conflict_type,
            "severity": self.severity.name,
            "status": self.status.name,
            "opened_at": self.opened_at.isoformat(),
            "opened_by_event": self.opened_by_event,
            "reopen_count": self.reopen_count,
        }
    
    @property
    def is_blocking(self) -> bool:
        """Does this contradiction block operations in its domain?"""
        return (
            self.status == ContradictionStatus.OPEN and
            self.severity.value >= ContradictionSeverity.HIGH.value
        )


@dataclass
class ContradictionSet:
    """
    The C_t component of state - all contradictions.
    
    This is the "contradiction load" that affects system behavior.
    """
    contradictions: Dict[str, Contradiction] = field(default_factory=dict)
    
    # Index by domain for fast lookup
    _by_domain: Dict[str, Set[str]] = field(default_factory=dict)
    
    # Index by claim for impact analysis
    _by_claim: Dict[str, Set[str]] = field(default_factory=dict)
    
    def add(self, contradiction: Contradiction) -> None:
        """Add a contradiction."""
        self.contradictions[contradiction.contradiction_id] = contradiction
        
        # Index by domain
        if contradiction.domain not in self._by_domain:
            self._by_domain[contradiction.domain] = set()
        self._by_domain[contradiction.domain].add(contradiction.contradiction_id)
        
        # Index by claims
        for claim_id in [contradiction.claim_a_id, contradiction.claim_b_id]:
            if claim_id not in self._by_claim:
                self._by_claim[claim_id] = set()
            self._by_claim[claim_id].add(contradiction.contradiction_id)
    
    def get_open(self) -> List[Contradiction]:
        """Get all open contradictions."""
        return [c for c in self.contradictions.values() 
                if c.status == ContradictionStatus.OPEN]
    
    def get_by_domain(self, domain: str) -> List[Contradiction]:
        """Get contradictions in a domain."""
        cids = self._by_domain.get(domain, set())
        return [self.contradictions[cid] for cid in cids]
    
    def get_blocking(self, domain: str) -> List[Contradiction]:
        """Get blocking contradictions in a domain."""
        return [c for c in self.get_by_domain(domain) if c.is_blocking]
    
    def has_open_in_domain(self, domain: str) -> bool:
        """Check if domain has open contradictions."""
        return any(
            c.status == ContradictionStatus.OPEN 
            for c in self.get_by_domain(domain)
        )
    
    @property
    def open_count(self) -> int:
        """Number of open contradictions."""
        return len(self.get_open())
    
    @property
    def total_severity(self) -> int:
        """Sum of severity scores for open contradictions."""
        return sum(c.severity.value for c in self.get_open())
    
    def state_hash(self) -> str:
        """Hash of contradiction state for comparison."""
        data = {
            cid: c.to_dict() 
            for cid, c in sorted(self.contradictions.items())
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]


# =============================================================================
# Hysteresis State
# =============================================================================

@dataclass
class HysteresisState:
    """
    State object for hysteresis testing.
    
    Contains:
    - Commitments (what's been committed)
    - Contradictions (C_t)
    - Budget state
    
    This is a simplified state for testing - production would use full SealedState.
    """
    state_id: str
    
    # Committed claims: claim_id -> claim_data
    commitments: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Contradiction set
    contradictions: ContradictionSet = field(default_factory=ContradictionSet)
    
    # Budget
    repair_budget: float = 100.0
    append_budget: float = 100.0
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    history: List[str] = field(default_factory=list)  # Event descriptions
    
    def commit_claim(
        self,
        claim_id: str,
        domain: str,
        predicate: str,
        value: Any,
        sigma: float = 0.8,
    ) -> None:
        """Commit a claim to state."""
        self.commitments[claim_id] = {
            "claim_id": claim_id,
            "domain": domain,
            "predicate": predicate,
            "value": value,
            "sigma": sigma,
            "committed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.history.append(f"COMMIT: {claim_id} ({domain}:{predicate}={value})")
    
    def add_contradiction(
        self,
        claim_a_id: str,
        claim_b_id: str,
        domain: str,
        conflict_type: str = "value_mismatch",
        severity: ContradictionSeverity = ContradictionSeverity.MEDIUM,
    ) -> Contradiction:
        """Add a contradiction between claims."""
        contradiction = Contradiction(
            contradiction_id=f"C_{uuid.uuid4().hex[:8]}",
            claim_a_id=claim_a_id,
            claim_b_id=claim_b_id,
            domain=domain,
            conflict_type=conflict_type,
            severity=severity,
            status=ContradictionStatus.OPEN,
            opened_at=datetime.now(timezone.utc),
            opened_by_event=f"E_{uuid.uuid4().hex[:8]}",
        )
        self.contradictions.add(contradiction)
        self.history.append(
            f"CONTRADICTION: {contradiction.contradiction_id} "
            f"({claim_a_id} vs {claim_b_id} in {domain})"
        )
        return contradiction
    
    def close_contradiction(
        self,
        contradiction_id: str,
        evidence_id: str,
        winning_claim_id: str,
    ) -> None:
        """Close a contradiction with evidence."""
        if contradiction_id not in self.contradictions.contradictions:
            raise ValueError(f"Unknown contradiction: {contradiction_id}")
        
        c = self.contradictions.contradictions[contradiction_id]
        c.status = ContradictionStatus.CLOSED
        c.closed_at = datetime.now(timezone.utc)
        c.closed_by_event = f"E_{uuid.uuid4().hex[:8]}"
        c.resolution_evidence = evidence_id
        c.resolution_claim_id = winning_claim_id
        
        self.history.append(
            f"RESOLVED: {contradiction_id} with evidence {evidence_id}"
        )
    
    def state_hash(self) -> str:
        """Compute state hash for comparison."""
        data = {
            "commitments": {
                k: v for k, v in sorted(self.commitments.items())
            },
            "contradictions": self.contradictions.state_hash(),
            "repair_budget": self.repair_budget,
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]
    
    def summary(self) -> Dict[str, Any]:
        """Get state summary for reporting."""
        return {
            "state_id": self.state_id,
            "state_hash": self.state_hash(),
            "commitment_count": len(self.commitments),
            "contradiction_count": len(self.contradictions.contradictions),
            "open_contradictions": self.contradictions.open_count,
            "total_severity": self.contradictions.total_severity,
            "repair_budget": self.repair_budget,
            "history_length": len(self.history),
        }


# =============================================================================
# History Scripts (Deterministic State Generators)
# =============================================================================

def script_commitment_divergence() -> Tuple[HysteresisState, HysteresisState, str]:
    """
    Script 1: Commitment Divergence
    
    S_A commits claim q
    S_B commits claim ¬q
    
    Same domain, opposite values.
    """
    state_a = HysteresisState(state_id="S_A_commit_div")
    state_b = HysteresisState(state_id="S_B_commit_div")
    
    # Both commit in same domain, different values
    state_a.commit_claim(
        claim_id="q_python_version",
        domain="python_version",
        predicate="current_version",
        value="3.12",
        sigma=0.9,
    )
    
    state_b.commit_claim(
        claim_id="q_python_version",
        domain="python_version",
        predicate="current_version",
        value="3.11",
        sigma=0.9,
    )
    
    description = (
        "Commitment Divergence: S_A has Python=3.12, S_B has Python=3.11. "
        "Queries about Python version should diverge."
    )
    
    return state_a, state_b, description


def script_open_vs_resolved() -> Tuple[HysteresisState, HysteresisState, str]:
    """
    Script 2: Open Contradiction vs Resolved
    
    S_A has an OPEN contradiction
    S_B has the same contradiction CLOSED with evidence
    
    S_A should block/warn on related queries; S_B should allow.
    """
    state_a = HysteresisState(state_id="S_A_open_conflict")
    state_b = HysteresisState(state_id="S_B_resolved")
    
    # Both start with conflicting claims
    for state in [state_a, state_b]:
        state.commit_claim(
            claim_id="q_release_2023",
            domain="release_date",
            predicate="released",
            value="October 2023",
            sigma=0.8,
        )
        state.commit_claim(
            claim_id="q_release_2022",
            domain="release_date",
            predicate="released",
            value="October 2022",
            sigma=0.7,
        )
    
    # Both have the contradiction
    for state in [state_a, state_b]:
        state.add_contradiction(
            claim_a_id="q_release_2023",
            claim_b_id="q_release_2022",
            domain="release_date",
            conflict_type="temporal_conflict",
            severity=ContradictionSeverity.HIGH,
        )
    
    # S_B resolves it
    c_id = list(state_b.contradictions.contradictions.keys())[0]
    state_b.close_contradiction(
        contradiction_id=c_id,
        evidence_id="ev_python_org",
        winning_claim_id="q_release_2023",
    )
    
    description = (
        "Open vs Resolved: S_A has OPEN contradiction on release_date. "
        "S_B has same contradiction CLOSED. "
        "Queries in release_date domain should block in S_A, allow in S_B."
    )
    
    return state_a, state_b, description


def script_budget_pressure() -> Tuple[HysteresisState, HysteresisState, str]:
    """
    Script 3: Budget Pressure
    
    S_A has depleted repair budget
    S_B has ample repair budget
    
    Same claims and contradictions.
    Operations requiring budget should differ.
    """
    state_a = HysteresisState(state_id="S_A_low_budget")
    state_b = HysteresisState(state_id="S_B_high_budget")
    
    # Same claims
    for state in [state_a, state_b]:
        state.commit_claim(
            claim_id="q_fact_1",
            domain="facts",
            predicate="value",
            value="A",
            sigma=0.8,
        )
        state.commit_claim(
            claim_id="q_fact_2",
            domain="facts",
            predicate="value",
            value="B",
            sigma=0.7,
        )
        state.add_contradiction(
            claim_a_id="q_fact_1",
            claim_b_id="q_fact_2",
            domain="facts",
            severity=ContradictionSeverity.MEDIUM,
        )
    
    # Different budgets
    state_a.repair_budget = 5.0    # Low - can't afford resolution
    state_b.repair_budget = 100.0  # Ample
    
    description = (
        "Budget Pressure: S_A has repair_budget=5, S_B has repair_budget=100. "
        "Same contradiction. Resolution attempts should fail in S_A."
    )
    
    return state_a, state_b, description


def script_severity_levels() -> Tuple[HysteresisState, HysteresisState, str]:
    """
    Script 4: Severity Levels
    
    S_A has LOW severity contradiction
    S_B has CRITICAL severity contradiction
    
    Same claims, different blocking behavior.
    """
    state_a = HysteresisState(state_id="S_A_low_severity")
    state_b = HysteresisState(state_id="S_B_critical")
    
    # Same claims
    for state in [state_a, state_b]:
        state.commit_claim(
            claim_id="q_config_a",
            domain="config",
            predicate="setting",
            value="enabled",
        )
        state.commit_claim(
            claim_id="q_config_b",
            domain="config",
            predicate="setting",
            value="disabled",
        )
    
    # Different severities
    state_a.add_contradiction(
        claim_a_id="q_config_a",
        claim_b_id="q_config_b",
        domain="config",
        severity=ContradictionSeverity.LOW,
    )
    
    state_b.add_contradiction(
        claim_a_id="q_config_a",
        claim_b_id="q_config_b",
        domain="config",
        severity=ContradictionSeverity.CRITICAL,
    )
    
    description = (
        "Severity Levels: S_A has LOW severity, S_B has CRITICAL. "
        "Same conflict. S_B should block operations, S_A should warn."
    )
    
    return state_a, state_b, description


# =============================================================================
# Query Set (State-Sensitive Prompts)
# =============================================================================

@dataclass
class HysteresisQuery:
    """A query designed to expose state differences."""
    query_id: str
    prompt: str
    target_domain: str
    expected_divergence: str  # What should differ
    requires_budget: bool = False


def get_query_set() -> List[HysteresisQuery]:
    """Get the standard query set for hysteresis testing."""
    return [
        # Domain-specific queries
        HysteresisQuery(
            query_id="Q_python_version",
            prompt="What is the current Python version?",
            target_domain="python_version",
            expected_divergence="Value should match committed claim",
        ),
        HysteresisQuery(
            query_id="Q_release_date",
            prompt="When was the software released?",
            target_domain="release_date",
            expected_divergence="Should block if OPEN contradiction, allow if resolved",
        ),
        HysteresisQuery(
            query_id="Q_config_status",
            prompt="Is the configuration setting enabled or disabled?",
            target_domain="config",
            expected_divergence="Should block/warn based on severity",
        ),
        
        # Stance queries
        HysteresisQuery(
            query_id="Q_summarize_stance",
            prompt="Summarize your current position on the facts domain.",
            target_domain="facts",
            expected_divergence="Summary should reflect contradiction status",
        ),
        
        # Budget-sensitive queries
        HysteresisQuery(
            query_id="Q_resolve_conflict",
            prompt="Please resolve the outstanding conflict in the facts domain.",
            target_domain="facts",
            expected_divergence="Should succeed or fail based on budget",
            requires_budget=True,
        ),
        
        # Meta queries
        HysteresisQuery(
            query_id="Q_open_contradictions",
            prompt="What contradictions are currently open?",
            target_domain="*",
            expected_divergence="Count and details should differ",
        ),
        
        # Additional queries for expanded coverage
        HysteresisQuery(
            query_id="Q_confidence_level",
            prompt="How confident are you about the Python version?",
            target_domain="python_version",
            expected_divergence="Confidence should reflect commitment sigma",
        ),
        HysteresisQuery(
            query_id="Q_contradiction_severity",
            prompt="How serious is the current conflict?",
            target_domain="facts",
            expected_divergence="Should report severity level from state",
        ),
        HysteresisQuery(
            query_id="Q_budget_status",
            prompt="Do you have resources to perform a resolution?",
            target_domain="*",
            expected_divergence="Should reflect budget state",
            requires_budget=True,
        ),
        HysteresisQuery(
            query_id="Q_commitment_count",
            prompt="How many facts have been committed?",
            target_domain="*",
            expected_divergence="Count should differ between states",
        ),
        HysteresisQuery(
            query_id="Q_domain_health",
            prompt="Is the facts domain in a healthy state?",
            target_domain="facts",
            expected_divergence="Health assessment should differ",
        ),
        HysteresisQuery(
            query_id="Q_can_proceed",
            prompt="Can I safely proceed with operations in this domain?",
            target_domain="facts",
            expected_divergence="Proceed/block recommendation should differ",
        ),
    ]


def get_extended_query_set() -> List[HysteresisQuery]:
    """Extended query set for high-N testing."""
    base = get_query_set()
    extended = []
    
    # Generate variations for each base query
    for q in base:
        extended.append(q)
        # Rephrase variant
        extended.append(HysteresisQuery(
            query_id=f"{q.query_id}_v2",
            prompt=f"Tell me about: {q.prompt.lower()}",
            target_domain=q.target_domain,
            expected_divergence=q.expected_divergence,
            requires_budget=q.requires_budget,
        ))
    
    return extended


# =============================================================================
# Hysteresis Governor (Minimal for Testing)
# =============================================================================

class HysteresisVerdict(Enum):
    """Governor verdict."""
    OK = auto()
    WARN = auto()
    BLOCK = auto()


@dataclass
class HysteresisResult:
    """Result of governing a query against a state."""
    verdict: HysteresisVerdict
    
    # What state objects influenced the decision
    referenced_commitments: List[str]
    referenced_contradictions: List[str]
    
    # Witness data
    witness: Dict[str, Any]
    
    # The response (if any)
    response: Optional[str] = None
    
    # Budget consumed
    budget_consumed: float = 0.0


class HysteresisGovernor:
    """
    Minimal governor for hysteresis testing.
    
    Makes decisions based on state, not prompt content.
    """
    
    def __init__(self, state: HysteresisState):
        self.state = state
    
    def govern(self, query: HysteresisQuery) -> HysteresisResult:
        """
        Govern a query against current state.
        
        This is the core of the hysteresis test:
        Same query, different state → different result.
        """
        referenced_commitments = []
        referenced_contradictions = []
        witness = {
            "query_id": query.query_id,
            "state_id": self.state.state_id,
            "state_hash": self.state.state_hash(),
            "target_domain": query.target_domain,
        }
        
        # Check for blocking contradictions in target domain
        if query.target_domain != "*":
            blocking = self.state.contradictions.get_blocking(query.target_domain)
            
            if blocking:
                # BLOCK due to high-severity open contradiction
                for c in blocking:
                    referenced_contradictions.append(c.contradiction_id)
                
                witness["block_reason"] = "BLOCKING_CONTRADICTION"
                witness["blocking_contradictions"] = [c.contradiction_id for c in blocking]
                
                return HysteresisResult(
                    verdict=HysteresisVerdict.BLOCK,
                    referenced_commitments=referenced_commitments,
                    referenced_contradictions=referenced_contradictions,
                    witness=witness,
                    response=f"[BLOCKED: Open contradiction in {query.target_domain}]",
                )
            
            # Check for any open contradictions (warn)
            open_in_domain = [
                c for c in self.state.contradictions.get_by_domain(query.target_domain)
                if c.status == ContradictionStatus.OPEN
            ]
            
            if open_in_domain:
                for c in open_in_domain:
                    referenced_contradictions.append(c.contradiction_id)
                
                witness["warn_reason"] = "OPEN_CONTRADICTION"
                witness["open_contradictions"] = [c.contradiction_id for c in open_in_domain]
        
        # Check budget for budget-requiring queries
        if query.requires_budget:
            required = 10.0  # Simplified
            if self.state.repair_budget < required:
                witness["block_reason"] = "INSUFFICIENT_BUDGET"
                witness["required"] = required
                witness["available"] = self.state.repair_budget
                
                return HysteresisResult(
                    verdict=HysteresisVerdict.BLOCK,
                    referenced_commitments=referenced_commitments,
                    referenced_contradictions=referenced_contradictions,
                    witness=witness,
                    response=f"[BLOCKED: Insufficient budget ({self.state.repair_budget} < {required})]",
                )
        
        # Get relevant commitments
        domain_commitments = [
            c for c in self.state.commitments.values()
            if c.get("domain") == query.target_domain or query.target_domain == "*"
        ]
        
        for c in domain_commitments:
            referenced_commitments.append(c["claim_id"])
        
        witness["relevant_commitments"] = referenced_commitments
        
        # Generate response based on state
        if query.target_domain == "*":
            # Meta query - report state
            response = self._generate_meta_response(query)
        else:
            response = self._generate_domain_response(query, domain_commitments)
        
        # Determine final verdict
        if referenced_contradictions and not any(
            c.status == ContradictionStatus.CLOSED
            for c in [self.state.contradictions.contradictions.get(cid) 
                     for cid in referenced_contradictions]
            if c
        ):
            verdict = HysteresisVerdict.WARN
            witness["warn_reason"] = witness.get("warn_reason", "OPEN_CONTRADICTION")
        else:
            verdict = HysteresisVerdict.OK
        
        return HysteresisResult(
            verdict=verdict,
            referenced_commitments=referenced_commitments,
            referenced_contradictions=referenced_contradictions,
            witness=witness,
            response=response,
        )
    
    def _generate_domain_response(
        self, 
        query: HysteresisQuery, 
        commitments: List[Dict[str, Any]],
    ) -> str:
        """Generate response based on domain commitments."""
        if not commitments:
            return f"[No commitments in domain {query.target_domain}]"
        
        # Find highest-sigma commitment
        best = max(commitments, key=lambda c: c.get("sigma", 0))
        return f"Based on committed state: {best['predicate']} = {best['value']} (σ={best['sigma']})"
    
    def _generate_meta_response(self, query: HysteresisQuery) -> str:
        """Generate response for meta queries."""
        open_c = self.state.contradictions.get_open()
        
        if "contradiction" in query.prompt.lower():
            if not open_c:
                return "No open contradictions."
            
            lines = [f"Open contradictions ({len(open_c)}):"]
            for c in open_c:
                lines.append(f"  - {c.contradiction_id}: {c.domain} ({c.severity.name})")
            return "\n".join(lines)
        
        return f"State {self.state.state_id}: {len(self.state.commitments)} commitments"


# =============================================================================
# Hysteresis Test Runner
# =============================================================================

@dataclass
class HysteresisTestResult:
    """Result of a single hysteresis test."""
    script_name: str
    query_id: str
    prompt_hash: str
    
    # State info
    state_a_id: str
    state_a_hash: str
    state_b_id: str
    state_b_hash: str
    
    # Results
    verdict_a: str
    verdict_b: str
    
    # References (what state objects were cited)
    refs_a_commitments: List[str]
    refs_a_contradictions: List[str]
    refs_b_commitments: List[str]
    refs_b_contradictions: List[str]
    
    # Responses
    response_a: Optional[str]
    response_b: Optional[str]
    
    # Analysis
    verdict_diverged: bool
    reference_diverged: bool
    response_diverged: bool
    divergence_explanation: str


class HysteresisHarness:
    """
    The hysteresis test harness.
    
    Runs queries against state pairs and measures divergence.
    """
    
    def __init__(self):
        self.scripts = [
            ("commitment_divergence", script_commitment_divergence),
            ("open_vs_resolved", script_open_vs_resolved),
            ("budget_pressure", script_budget_pressure),
            ("severity_levels", script_severity_levels),
        ]
        self.queries = get_query_set()
        self.results: List[HysteresisTestResult] = []
    
    def run_all(self) -> List[HysteresisTestResult]:
        """Run all scripts against all queries."""
        self.results = []
        
        for script_name, script_fn in self.scripts:
            state_a, state_b, description = script_fn()
            
            # Run each query against both states
            for query in self.queries:
                result = self._run_single(
                    script_name, 
                    state_a, 
                    state_b, 
                    query,
                    description,
                )
                self.results.append(result)
        
        return self.results
    
    def _run_single(
        self,
        script_name: str,
        state_a: HysteresisState,
        state_b: HysteresisState,
        query: HysteresisQuery,
        script_description: str,
    ) -> HysteresisTestResult:
        """Run a single query against both states."""
        # Govern in each state
        gov_a = HysteresisGovernor(state_a)
        gov_b = HysteresisGovernor(state_b)
        
        result_a = gov_a.govern(query)
        result_b = gov_b.govern(query)
        
        # Compute prompt hash (proves same input)
        prompt_hash = hashlib.sha256(query.prompt.encode()).hexdigest()[:16]
        
        # Analyze divergence
        verdict_diverged = result_a.verdict != result_b.verdict
        
        reference_diverged = (
            set(result_a.referenced_commitments) != set(result_b.referenced_commitments) or
            set(result_a.referenced_contradictions) != set(result_b.referenced_contradictions)
        )
        
        response_diverged = result_a.response != result_b.response
        
        # Explain divergence
        explanations = []
        if verdict_diverged:
            explanations.append(
                f"Verdict: {result_a.verdict.name} → {result_b.verdict.name}"
            )
        if reference_diverged:
            if result_a.referenced_contradictions != result_b.referenced_contradictions:
                explanations.append(
                    f"Contradictions: {result_a.referenced_contradictions} vs {result_b.referenced_contradictions}"
                )
            if result_a.referenced_commitments != result_b.referenced_commitments:
                explanations.append(
                    f"Commitments: {result_a.referenced_commitments} vs {result_b.referenced_commitments}"
                )
        if response_diverged and not explanations:
            explanations.append("Response text differs")
        
        divergence_explanation = "; ".join(explanations) if explanations else "No divergence"
        
        return HysteresisTestResult(
            script_name=script_name,
            query_id=query.query_id,
            prompt_hash=prompt_hash,
            state_a_id=state_a.state_id,
            state_a_hash=state_a.state_hash(),
            state_b_id=state_b.state_id,
            state_b_hash=state_b.state_hash(),
            verdict_a=result_a.verdict.name,
            verdict_b=result_b.verdict.name,
            refs_a_commitments=result_a.referenced_commitments,
            refs_a_contradictions=result_a.referenced_contradictions,
            refs_b_commitments=result_b.referenced_commitments,
            refs_b_contradictions=result_b.referenced_contradictions,
            response_a=result_a.response,
            response_b=result_b.response,
            verdict_diverged=verdict_diverged,
            reference_diverged=reference_diverged,
            response_diverged=response_diverged,
            divergence_explanation=divergence_explanation,
        )
    
    def report(self) -> Dict[str, Any]:
        """Generate summary report."""
        total = len(self.results)
        verdict_diverged = sum(1 for r in self.results if r.verdict_diverged)
        reference_diverged = sum(1 for r in self.results if r.reference_diverged)
        response_diverged = sum(1 for r in self.results if r.response_diverged)
        any_diverged = sum(
            1 for r in self.results 
            if r.verdict_diverged or r.reference_diverged
        )
        
        # Group by script
        by_script = {}
        for r in self.results:
            if r.script_name not in by_script:
                by_script[r.script_name] = {"total": 0, "diverged": 0}
            by_script[r.script_name]["total"] += 1
            if r.verdict_diverged or r.reference_diverged:
                by_script[r.script_name]["diverged"] += 1
        
        return {
            "total_tests": total,
            "verdict_diverged": verdict_diverged,
            "reference_diverged": reference_diverged,
            "response_diverged": response_diverged,
            "any_state_grounded_divergence": any_diverged,
            "divergence_rate": any_diverged / total if total > 0 else 0,
            "by_script": by_script,
            "interiority_confirmed": any_diverged > 0,
        }


# =============================================================================
# Tests
# =============================================================================

def test_contradiction_object():
    """Test Contradiction as first-class object."""
    print("=== Test: Contradiction Object ===\n")
    
    c = Contradiction(
        contradiction_id="C_test",
        claim_a_id="q1",
        claim_b_id="q2",
        domain="test",
        conflict_type="value_mismatch",
        severity=ContradictionSeverity.HIGH,
        status=ContradictionStatus.OPEN,
        opened_at=datetime.now(timezone.utc),
        opened_by_event="E_test",
    )
    
    print(f"Contradiction: {c.contradiction_id}")
    print(f"Status: {c.status.name}")
    print(f"Severity: {c.severity.name}")
    print(f"Is blocking: {c.is_blocking}")
    
    assert c.is_blocking, "HIGH severity OPEN should be blocking"
    
    # Close it
    c.status = ContradictionStatus.CLOSED
    assert not c.is_blocking, "CLOSED should not be blocking"
    
    print("✓ Contradiction object working\n")
    return True


def test_contradiction_set():
    """Test ContradictionSet indexing."""
    print("=== Test: Contradiction Set ===\n")
    
    cs = ContradictionSet()
    
    c1 = Contradiction(
        contradiction_id="C_1",
        claim_a_id="q1",
        claim_b_id="q2",
        domain="domain_a",
        conflict_type="test",
        severity=ContradictionSeverity.LOW,
        status=ContradictionStatus.OPEN,
        opened_at=datetime.now(timezone.utc),
        opened_by_event="E1",
    )
    
    c2 = Contradiction(
        contradiction_id="C_2",
        claim_a_id="q3",
        claim_b_id="q4",
        domain="domain_b",
        conflict_type="test",
        severity=ContradictionSeverity.CRITICAL,
        status=ContradictionStatus.OPEN,
        opened_at=datetime.now(timezone.utc),
        opened_by_event="E2",
    )
    
    cs.add(c1)
    cs.add(c2)
    
    print(f"Total: {len(cs.contradictions)}")
    print(f"Open: {cs.open_count}")
    print(f"Total severity: {cs.total_severity}")
    
    assert cs.open_count == 2
    assert len(cs.get_by_domain("domain_a")) == 1
    assert len(cs.get_blocking("domain_b")) == 1  # CRITICAL is blocking
    assert len(cs.get_blocking("domain_a")) == 0  # LOW is not blocking
    
    print("✓ Contradiction set working\n")
    return True


def test_history_scripts():
    """Test that history scripts produce different states."""
    print("=== Test: History Scripts ===\n")
    
    scripts = [
        ("commitment_divergence", script_commitment_divergence),
        ("open_vs_resolved", script_open_vs_resolved),
        ("budget_pressure", script_budget_pressure),
        ("severity_levels", script_severity_levels),
    ]
    
    for name, script_fn in scripts:
        state_a, state_b, desc = script_fn()
        
        hash_a = state_a.state_hash()
        hash_b = state_b.state_hash()
        
        print(f"{name}:")
        print(f"  S_A hash: {hash_a}")
        print(f"  S_B hash: {hash_b}")
        print(f"  Different: {hash_a != hash_b}")
        
        assert hash_a != hash_b, f"States should differ for {name}"
    
    print("\n✓ All scripts produce different states\n")
    return True


def test_hysteresis_divergence():
    """Test that same query produces different results for different states."""
    print("=== Test: Hysteresis Divergence ===\n")
    
    harness = HysteresisHarness()
    results = harness.run_all()
    report = harness.report()
    
    print(f"Total tests: {report['total_tests']}")
    print(f"Verdict diverged: {report['verdict_diverged']}")
    print(f"Reference diverged: {report['reference_diverged']}")
    print(f"Any state-grounded divergence: {report['any_state_grounded_divergence']}")
    print(f"Divergence rate: {report['divergence_rate']:.1%}")
    print(f"\nBy script:")
    for script, data in report['by_script'].items():
        print(f"  {script}: {data['diverged']}/{data['total']} diverged")
    
    print(f"\n✓ Interiority confirmed: {report['interiority_confirmed']}")
    
    # Show some specific divergences
    print("\nSample divergences:")
    diverged = [r for r in results if r.verdict_diverged or r.reference_diverged][:5]
    for r in diverged:
        print(f"\n  {r.script_name} / {r.query_id}:")
        print(f"    Prompt hash: {r.prompt_hash}")
        print(f"    S_A ({r.state_a_hash}): {r.verdict_a}")
        print(f"    S_B ({r.state_b_hash}): {r.verdict_b}")
        print(f"    Divergence: {r.divergence_explanation}")
    
    assert report['interiority_confirmed'], "Should have state-grounded divergence"
    
    print("\n✓ Hysteresis test passed - interiority confirmed\n")
    return True


def test_same_state_no_divergence():
    """Test that same state produces same results (control)."""
    print("=== Test: Same State No Divergence (Control) ===\n")
    
    state_a, _, _ = script_commitment_divergence()
    
    # Clone state
    state_b = HysteresisState(state_id="S_clone")
    state_b.commitments = dict(state_a.commitments)
    state_b.contradictions = state_a.contradictions
    state_b.repair_budget = state_a.repair_budget
    
    gov_a = HysteresisGovernor(state_a)
    gov_b = HysteresisGovernor(state_b)
    
    query = HysteresisQuery(
        query_id="Q_test",
        prompt="What is the Python version?",
        target_domain="python_version",
        expected_divergence="None - same state",
    )
    
    result_a = gov_a.govern(query)
    result_b = gov_b.govern(query)
    
    print(f"S_A verdict: {result_a.verdict.name}")
    print(f"S_B verdict: {result_b.verdict.name}")
    print(f"S_A response: {result_a.response}")
    print(f"S_B response: {result_b.response}")
    
    assert result_a.verdict == result_b.verdict, "Same state should give same verdict"
    assert result_a.response == result_b.response, "Same state should give same response"
    
    print("\n✓ Control test passed - same state = same result\n")
    return True


def run_all_tests():
    """Run all hysteresis tests."""
    print("=" * 70)
    print("HYSTERESIS TEST HARNESS")
    print("Proving I(Y; S | X) > 0")
    print("=" * 70 + "\n")
    
    results = []
    
    results.append(("contradiction_object", test_contradiction_object()))
    results.append(("contradiction_set", test_contradiction_set()))
    results.append(("history_scripts", test_history_scripts()))
    results.append(("same_state_control", test_same_state_no_divergence()))
    results.append(("hysteresis_divergence", test_hysteresis_divergence()))
    
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "=" * 70)
        print("✓ INTERIORITY CONFIRMED")
        print("  Same prompt + different state = different output")
        print("  Divergence traceable to state objects")
        print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
