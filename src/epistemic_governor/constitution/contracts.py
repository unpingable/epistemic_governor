"""
Interface Contracts

Formal boundary definitions between inner and outer environments.

Simon's principle: "The boundary is the interface."

This module defines:
1. What can cross the governor boundary
2. What is forbidden even if it would work
3. Explicit contracts for each interface
4. Boundary hardening against injection

The governor has exactly three external interfaces:
1. INPUT: Claims and evidence from the world
2. OUTPUT: Verdicts and state projections to the world
3. CONTROL: Human oversight commands

Everything else is internal.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Callable
import re


# =============================================================================
# Interface Definitions
# =============================================================================

class InterfaceType(Enum):
    """The three permitted interfaces."""
    INPUT = auto()      # Claims, evidence, queries
    OUTPUT = auto()     # Verdicts, projections, witnesses
    CONTROL = auto()    # Oversight commands


class CrossingVerdict(Enum):
    """Result of boundary crossing check."""
    ALLOWED = auto()
    DENIED_TYPE = auto()         # Wrong type for this interface
    DENIED_CONTENT = auto()      # Content violates policy
    DENIED_RATE = auto()         # Rate limit exceeded
    DENIED_INJECTION = auto()    # Injection attempt detected
    DENIED_AUTHORITY = auto()    # Insufficient authority


# =============================================================================
# Boundary Contracts
# =============================================================================

@dataclass
class InputContract:
    """
    Contract for INPUT interface.
    
    What may enter:
    - Text (for claim extraction)
    - Evidence objects (typed, sourced)
    - Queries (read-only state inspection)
    
    What may NOT enter:
    - Direct state mutations
    - Control commands (must use CONTROL interface)
    - Unbounded binary data
    """
    # Allowed content types
    allowed_content_types: Set[str] = field(default_factory=lambda: {
        "text/plain",
        "application/json",
        "application/evidence",
    })
    
    # Size limits
    max_text_length: int = 100_000      # 100KB text
    max_evidence_size: int = 1_000_000  # 1MB evidence
    max_claims_per_input: int = 100
    
    # Rate limits
    max_inputs_per_minute: int = 60
    
    # Injection patterns to reject
    injection_patterns: List[str] = field(default_factory=lambda: [
        r"__import__",
        r"eval\s*\(",
        r"exec\s*\(",
        r"system\s*\(",
        r"<script",
        r"javascript:",
        r"\{\{.*\}\}",  # Template injection
        r"\$\{.*\}",    # Variable injection
    ])


@dataclass
class OutputContract:
    """
    Contract for OUTPUT interface.
    
    What may exit:
    - Verdicts (allow/deny/warn)
    - State projections (filtered views)
    - Witnesses (audit records)
    - Error messages (structured)
    
    What may NOT exit:
    - Raw internal state
    - Evidence content (only references)
    - Constitutional parameters (S₀)
    - Adaptation history details
    """
    # Allowed output types
    allowed_output_types: Set[str] = field(default_factory=lambda: {
        "verdict",
        "projection",
        "witness",
        "error",
        "stats",
    })
    
    # Forbidden fields (never leak these)
    forbidden_fields: Set[str] = field(default_factory=lambda: {
        "raw_evidence_content",
        "s0_parameters",
        "adaptation_history",
        "internal_state_hash",
        "api_keys",
        "credentials",
    })
    
    # Size limits
    max_projection_size: int = 100_000  # 100KB
    max_witness_size: int = 10_000      # 10KB


@dataclass
class ControlContract:
    """
    Contract for CONTROL interface.
    
    Who may use:
    - Human operators (authenticated)
    - Designated oversight systems
    
    What may be commanded:
    - Freeze/unfreeze adaptation
    - Reset to checkpoint
    - Adjust S₁ parameters (within bounds)
    - Query audit logs
    
    What may NOT be commanded:
    - Modify S₀ (constitutional)
    - Bypass NLAI
    - Delete audit records
    - Grant self authority
    """
    # Allowed commands
    allowed_commands: Set[str] = field(default_factory=lambda: {
        "freeze",
        "unfreeze", 
        "reset",
        "adjust_s1",
        "query_audit",
        "get_state",
        "export_witness",
    })
    
    # Forbidden commands (even with authority)
    forbidden_commands: Set[str] = field(default_factory=lambda: {
        "modify_s0",
        "bypass_nlai",
        "delete_audit",
        "grant_authority",
        "disable_logging",
    })
    
    # Required authentication
    require_auth: bool = True
    
    # Rate limits
    max_commands_per_minute: int = 10


# =============================================================================
# Boundary Gate
# =============================================================================

@dataclass
class BoundaryCrossing:
    """Record of a boundary crossing attempt."""
    crossing_id: str
    timestamp: datetime
    interface: InterfaceType
    verdict: CrossingVerdict
    content_type: str
    content_size: int
    source: str
    reason: Optional[str] = None


class BoundaryGate:
    """
    Enforces interface contracts.
    
    All external interactions must pass through this gate.
    """
    
    def __init__(
        self,
        input_contract: Optional[InputContract] = None,
        output_contract: Optional[OutputContract] = None,
        control_contract: Optional[ControlContract] = None,
    ):
        self.input_contract = input_contract or InputContract()
        self.output_contract = output_contract or OutputContract()
        self.control_contract = control_contract or ControlContract()
        
        # Compiled injection patterns
        self._injection_patterns = [
            re.compile(p, re.IGNORECASE) 
            for p in self.input_contract.injection_patterns
        ]
        
        # Rate tracking
        self._input_times: List[datetime] = []
        self._control_times: List[datetime] = []
        
        # Crossing log
        self.crossings: List[BoundaryCrossing] = []
        self._crossing_counter = 0
    
    def _next_crossing_id(self) -> str:
        self._crossing_counter += 1
        return f"X-{self._crossing_counter}"
    
    def check_input(
        self,
        content: Any,
        content_type: str,
        source: str,
    ) -> tuple[CrossingVerdict, Optional[str]]:
        """
        Check if input may cross boundary.
        
        Returns (verdict, reason).
        """
        now = datetime.now(timezone.utc)
        
        # Type check
        if content_type not in self.input_contract.allowed_content_types:
            return CrossingVerdict.DENIED_TYPE, f"Content type {content_type} not allowed"
        
        # Size check - proper measurement based on type
        if isinstance(content, bytes):
            content_size = len(content)
        elif isinstance(content, str):
            content_size = len(content)
        elif content_type == "application/json":
            import json
            content_size = len(json.dumps(content))
        else:
            content_size = len(str(content)) if content else 0
            
        if content_type == "text/plain" and content_size > self.input_contract.max_text_length:
            return CrossingVerdict.DENIED_CONTENT, f"Text too large: {content_size} > {self.input_contract.max_text_length}"
        
        # Rate check - use total_seconds() not .seconds
        self._input_times = [t for t in self._input_times if (now - t).total_seconds() < 60]
        if len(self._input_times) >= self.input_contract.max_inputs_per_minute:
            return CrossingVerdict.DENIED_RATE, "Input rate limit exceeded"
        
        # Injection check
        if isinstance(content, str):
            for pattern in self._injection_patterns:
                if pattern.search(content):
                    return CrossingVerdict.DENIED_INJECTION, f"Injection pattern detected"
        
        # Record
        self._input_times.append(now)
        self._record_crossing(InterfaceType.INPUT, CrossingVerdict.ALLOWED, content_type, content_size, source)
        
        return CrossingVerdict.ALLOWED, None
    
    def check_output(
        self,
        output: Dict[str, Any],
        output_type: str,
    ) -> tuple[CrossingVerdict, Optional[str], Dict[str, Any]]:
        """
        Check and filter output.
        
        Returns (verdict, reason, filtered_output).
        """
        # Type check
        if output_type not in self.output_contract.allowed_output_types:
            return CrossingVerdict.DENIED_TYPE, f"Output type {output_type} not allowed", {}
        
        # Filter forbidden fields
        filtered = self._filter_output(output)
        
        # Size check
        output_size = len(str(filtered))
        if output_size > self.output_contract.max_projection_size:
            return CrossingVerdict.DENIED_CONTENT, f"Output too large: {output_size}", {}
        
        self._record_crossing(InterfaceType.OUTPUT, CrossingVerdict.ALLOWED, output_type, output_size, "internal")
        
        return CrossingVerdict.ALLOWED, None, filtered
    
    def _filter_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Remove forbidden fields from output."""
        if not isinstance(output, dict):
            return output
        
        filtered = {}
        for key, value in output.items():
            if key in self.output_contract.forbidden_fields:
                continue
            if isinstance(value, dict):
                filtered[key] = self._filter_output(value)
            else:
                filtered[key] = value
        
        return filtered
    
    def check_control(
        self,
        command: str,
        authenticated: bool,
        authority_level: int = 0,
    ) -> tuple[CrossingVerdict, Optional[str]]:
        """
        Check if control command is allowed.
        
        Returns (verdict, reason).
        """
        now = datetime.now(timezone.utc)
        
        # Auth check
        if self.control_contract.require_auth and not authenticated:
            return CrossingVerdict.DENIED_AUTHORITY, "Authentication required"
        
        # Forbidden check (absolute)
        if command in self.control_contract.forbidden_commands:
            return CrossingVerdict.DENIED_AUTHORITY, f"Command '{command}' is forbidden"
        
        # Allowed check
        if command not in self.control_contract.allowed_commands:
            return CrossingVerdict.DENIED_TYPE, f"Command '{command}' not recognized"
        
        # Rate check - use total_seconds() not .seconds
        self._control_times = [t for t in self._control_times if (now - t).total_seconds() < 60]
        if len(self._control_times) >= self.control_contract.max_commands_per_minute:
            return CrossingVerdict.DENIED_RATE, "Control rate limit exceeded"
        
        # Record
        self._control_times.append(now)
        self._record_crossing(InterfaceType.CONTROL, CrossingVerdict.ALLOWED, command, 0, "operator")
        
        return CrossingVerdict.ALLOWED, None
    
    def _record_crossing(
        self,
        interface: InterfaceType,
        verdict: CrossingVerdict,
        content_type: str,
        content_size: int,
        source: str,
        reason: Optional[str] = None,
    ):
        """Record a boundary crossing."""
        crossing = BoundaryCrossing(
            crossing_id=self._next_crossing_id(),
            timestamp=datetime.now(timezone.utc),
            interface=interface,
            verdict=verdict,
            content_type=content_type,
            content_size=content_size,
            source=source,
            reason=reason,
        )
        self.crossings.append(crossing)
        
        # Keep bounded
        if len(self.crossings) > 1000:
            self.crossings = self.crossings[-500:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get boundary crossing statistics."""
        by_interface = {}
        by_verdict = {}
        
        for c in self.crossings:
            by_interface[c.interface.name] = by_interface.get(c.interface.name, 0) + 1
            by_verdict[c.verdict.name] = by_verdict.get(c.verdict.name, 0) + 1
        
        return {
            "total_crossings": len(self.crossings),
            "by_interface": by_interface,
            "by_verdict": by_verdict,
        }


# =============================================================================
# Inadmissible Actions (Even If Effective)
# =============================================================================

INADMISSIBLE_ACTIONS = {
    # Authority violations
    "self_grant_authority": "System cannot grant itself additional authority",
    "modify_own_constraints": "System cannot modify its own constraint parameters",
    "bypass_evidence_requirement": "Claims cannot be committed without evidence",
    
    # Temporal violations
    "backdate_evidence": "Evidence cannot be timestamped in the past",
    "extend_own_deadline": "System cannot extend its own temporal limits",
    
    # Audit violations
    "delete_audit_record": "Audit records cannot be deleted",
    "modify_audit_record": "Audit records cannot be modified",
    "disable_audit": "Audit logging cannot be disabled",
    
    # Boundary violations
    "expose_internal_state": "Raw internal state cannot cross output boundary",
    "accept_self_as_evidence": "System's own outputs cannot serve as evidence",
}


def is_inadmissible(action: str) -> tuple[bool, Optional[str]]:
    """
    Check if an action is inadmissible regardless of effectiveness.
    
    Returns (is_inadmissible, reason).
    """
    if action in INADMISSIBLE_ACTIONS:
        return True, INADMISSIBLE_ACTIONS[action]
    return False, None


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=== Interface Contracts Test ===\n")
    
    gate = BoundaryGate()
    
    # Valid input
    verdict, reason = gate.check_input("Hello world", "text/plain", "user")
    print(f"Valid input: {verdict.name}")
    
    # Invalid content type
    verdict, reason = gate.check_input(b"binary", "application/octet-stream", "user")
    print(f"Invalid type: {verdict.name} - {reason}")
    
    # Injection attempt
    verdict, reason = gate.check_input("eval(malicious())", "text/plain", "attacker")
    print(f"Injection: {verdict.name} - {reason}")
    
    # Valid output
    output = {"verdict": "allow", "reason": "evidence sufficient"}
    verdict, reason, filtered = gate.check_output(output, "verdict")
    print(f"Valid output: {verdict.name}")
    
    # Output with forbidden field
    output = {"verdict": "allow", "raw_evidence_content": "secret", "api_keys": "xxx"}
    verdict, reason, filtered = gate.check_output(output, "verdict")
    print(f"Filtered output: {verdict.name}, keys: {list(filtered.keys())}")
    
    # Valid control
    verdict, reason = gate.check_control("freeze", authenticated=True)
    print(f"Valid control: {verdict.name}")
    
    # Forbidden control
    verdict, reason = gate.check_control("modify_s0", authenticated=True)
    print(f"Forbidden control: {verdict.name} - {reason}")
    
    # Unauthenticated control
    verdict, reason = gate.check_control("freeze", authenticated=False)
    print(f"Unauth control: {verdict.name} - {reason}")
    
    # Inadmissible action
    is_bad, why = is_inadmissible("self_grant_authority")
    print(f"\nInadmissible 'self_grant_authority': {is_bad} - {why}")
    
    # Stats
    print("\n=== Stats ===")
    import json
    print(json.dumps(gate.get_stats(), indent=2))
