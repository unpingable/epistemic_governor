"""
Reset Primitives

Typed state contraction operations.

Reset is not "reboot the vibe" - it's a first-class state transition:
- Commit-pointed (identifiable rollback target)
- Typed (different reset scopes)
- Non-negotiable (model can't talk you out of it)
- Auditable (why it happened)

Reset types:
1. Context Reset - clears working state, keeps ledger/config
2. Mode Reset - downgrades capabilities (tools, variety, horizon)
3. Goal Reset - clears task continuation, requires re-issuance
4. Chain Reset - rolls back to checkpoint, invalidates downstream
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set


class ResetType(Enum):
    """Types of reset operations."""
    CONTEXT = auto()    # Clear working state
    MODE = auto()       # Downgrade capabilities
    GOAL = auto()       # Clear task continuation
    CHAIN = auto()      # Rollback to checkpoint


class ResetSeverity(Enum):
    """How aggressive the reset is."""
    SOFT = auto()       # Minimal disruption
    HARD = auto()       # Full scope clear
    EMERGENCY = auto()  # Immediate, no graceful handling


@dataclass
class ResetTarget:
    """What gets reset."""
    # Context reset targets
    clear_scratchpad: bool = False
    clear_working_memory: bool = False
    clear_pending_claims: bool = False
    
    # Mode reset targets
    disable_tools: bool = False
    restrict_to_readonly: bool = False
    reduce_variety: bool = False
    shorten_horizon: bool = False
    increase_evidence_requirement: bool = False
    
    # Goal reset targets
    clear_task_continuation: bool = False
    require_intent_reissuance: bool = False
    
    # Chain reset targets
    rollback_to_checkpoint: Optional[str] = None
    invalidate_downstream: bool = False


@dataclass
class ResetEvent:
    """Record of a reset operation."""
    reset_id: str
    timestamp: datetime
    reset_type: ResetType
    severity: ResetSeverity
    target: ResetTarget
    
    # Why it happened
    trigger_regime: str
    trigger_signals: Dict[str, float]
    trigger_reason: str
    
    # What was affected
    checkpoint_before: Optional[str] = None
    checkpoint_after: Optional[str] = None
    items_cleared: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reset_id": self.reset_id,
            "timestamp": self.timestamp.isoformat(),
            "reset_type": self.reset_type.name,
            "severity": self.severity.name,
            "trigger_regime": self.trigger_regime,
            "trigger_signals": self.trigger_signals,
            "trigger_reason": self.trigger_reason,
            "checkpoint_before": self.checkpoint_before,
            "checkpoint_after": self.checkpoint_after,
            "items_cleared": self.items_cleared,
        }


@dataclass 
class ModeState:
    """Current operational mode (can be degraded by Mode Reset)."""
    tools_enabled: bool = True
    readonly_mode: bool = False
    variety_multiplier: float = 1.0  # 1.0 = normal, 0.5 = reduced
    horizon_turns: int = 10          # Max planning horizon
    evidence_threshold: float = 1.0  # 1.0 = normal, 2.0 = stricter
    
    def degrade(self, level: int = 1):
        """Degrade mode by one or more levels."""
        for _ in range(level):
            if self.tools_enabled and not self.readonly_mode:
                self.readonly_mode = True
            elif self.readonly_mode:
                self.tools_enabled = False
            
            self.variety_multiplier = max(0.25, self.variety_multiplier * 0.5)
            self.horizon_turns = max(1, self.horizon_turns // 2)
            self.evidence_threshold = min(4.0, self.evidence_threshold * 1.5)
    
    def is_degraded(self) -> bool:
        return (
            not self.tools_enabled or
            self.readonly_mode or
            self.variety_multiplier < 1.0 or
            self.horizon_turns < 10 or
            self.evidence_threshold > 1.0
        )


class ResetController:
    """
    Manages reset operations.
    
    Resets are the missing actuator - the mechanism that contracts
    state space when the system enters problematic regimes.
    """
    
    def __init__(self):
        self.mode = ModeState()
        self.reset_history: List[ResetEvent] = []
        self._reset_counter = 0
        
        # Checkpoint tracking (would integrate with ledger)
        self.current_checkpoint: Optional[str] = None
        self.checkpoint_stack: List[str] = []
    
    def _next_reset_id(self) -> str:
        self._reset_counter += 1
        return f"RST-{self._reset_counter}"
    
    def context_reset(
        self,
        regime: str,
        signals: Dict[str, float],
        reason: str,
        severity: ResetSeverity = ResetSeverity.SOFT,
    ) -> ResetEvent:
        """
        Clear working context while preserving ledger and config.
        
        Use when: drift detected, hysteresis building up
        """
        target = ResetTarget(
            clear_scratchpad=True,
            clear_working_memory=True,
            clear_pending_claims=True,
        )
        
        event = ResetEvent(
            reset_id=self._next_reset_id(),
            timestamp=datetime.now(timezone.utc),
            reset_type=ResetType.CONTEXT,
            severity=severity,
            target=target,
            trigger_regime=regime,
            trigger_signals=signals,
            trigger_reason=reason,
            checkpoint_before=self.current_checkpoint,
        )
        
        # Actual reset would clear state here
        # For now, just record
        event.items_cleared = 0  # Would be actual count
        
        self.reset_history.append(event)
        return event
    
    def mode_reset(
        self,
        regime: str,
        signals: Dict[str, float],
        reason: str,
        degrade_level: int = 1,
    ) -> ResetEvent:
        """
        Downgrade operational capabilities.
        
        Use when: approaching ductile regime, need to reduce coupling
        """
        target = ResetTarget(
            disable_tools=(degrade_level >= 2),
            restrict_to_readonly=(degrade_level >= 1),
            reduce_variety=True,
            shorten_horizon=True,
            increase_evidence_requirement=True,
        )
        
        event = ResetEvent(
            reset_id=self._next_reset_id(),
            timestamp=datetime.now(timezone.utc),
            reset_type=ResetType.MODE,
            severity=ResetSeverity.SOFT if degrade_level == 1 else ResetSeverity.HARD,
            target=target,
            trigger_regime=regime,
            trigger_signals=signals,
            trigger_reason=reason,
        )
        
        # Apply mode degradation
        self.mode.degrade(degrade_level)
        
        self.reset_history.append(event)
        return event
    
    def goal_reset(
        self,
        regime: str,
        signals: Dict[str, float],
        reason: str,
    ) -> ResetEvent:
        """
        Clear task continuation, require explicit re-issuance of intent.
        
        Use when: runaway continuation detected, unclear goal state
        """
        target = ResetTarget(
            clear_task_continuation=True,
            require_intent_reissuance=True,
        )
        
        event = ResetEvent(
            reset_id=self._next_reset_id(),
            timestamp=datetime.now(timezone.utc),
            reset_type=ResetType.GOAL,
            severity=ResetSeverity.HARD,
            target=target,
            trigger_regime=regime,
            trigger_signals=signals,
            trigger_reason=reason,
        )
        
        self.reset_history.append(event)
        return event
    
    def chain_reset(
        self,
        regime: str,
        signals: Dict[str, float],
        reason: str,
        checkpoint_id: Optional[str] = None,
    ) -> ResetEvent:
        """
        Roll back to checkpoint, invalidate downstream artifacts.
        
        Use when: cascade detected, need hard recovery point
        """
        # Use specified checkpoint or last one
        rollback_target = checkpoint_id or (
            self.checkpoint_stack[-1] if self.checkpoint_stack else None
        )
        
        target = ResetTarget(
            rollback_to_checkpoint=rollback_target,
            invalidate_downstream=True,
        )
        
        event = ResetEvent(
            reset_id=self._next_reset_id(),
            timestamp=datetime.now(timezone.utc),
            reset_type=ResetType.CHAIN,
            severity=ResetSeverity.EMERGENCY,
            target=target,
            trigger_regime=regime,
            trigger_signals=signals,
            trigger_reason=reason,
            checkpoint_before=self.current_checkpoint,
            checkpoint_after=rollback_target,
        )
        
        # Would actually rollback ledger here
        if rollback_target:
            self.current_checkpoint = rollback_target
        
        self.reset_history.append(event)
        return event
    
    def create_checkpoint(self, checkpoint_id: str):
        """Create a checkpoint for potential rollback."""
        self.checkpoint_stack.append(checkpoint_id)
        self.current_checkpoint = checkpoint_id
    
    def restore_mode(self):
        """Restore mode to defaults (requires explicit action)."""
        self.mode = ModeState()
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "mode": {
                "tools_enabled": self.mode.tools_enabled,
                "readonly_mode": self.mode.readonly_mode,
                "variety_multiplier": self.mode.variety_multiplier,
                "horizon_turns": self.mode.horizon_turns,
                "evidence_threshold": self.mode.evidence_threshold,
                "is_degraded": self.mode.is_degraded(),
            },
            "checkpoints": len(self.checkpoint_stack),
            "current_checkpoint": self.current_checkpoint,
            "total_resets": len(self.reset_history),
        }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=== Reset Controller Test ===\n")
    
    controller = ResetController()
    controller.create_checkpoint("CP-001")
    
    print(f"Initial state: {controller.get_state()}\n")
    
    # Context reset
    event = controller.context_reset(
        regime="WARM",
        signals={"hysteresis": 0.3, "relaxation_time": 5.0},
        reason="Drift detected",
    )
    print(f"Context reset: {event.reset_id}")
    
    # Mode reset
    event = controller.mode_reset(
        regime="DUCTILE",
        signals={"hysteresis": 0.6, "anisotropy": 0.4},
        reason="Approaching ductile regime",
        degrade_level=1,
    )
    print(f"Mode reset: {event.reset_id}")
    print(f"  Mode degraded: {controller.mode.is_degraded()}")
    print(f"  Readonly: {controller.mode.readonly_mode}")
    print(f"  Variety: {controller.mode.variety_multiplier}")
    
    # Chain reset
    controller.create_checkpoint("CP-002")
    event = controller.chain_reset(
        regime="UNSTABLE",
        signals={"tool_gain": 1.5, "cascade_depth": 3},
        reason="Cascade detected",
        checkpoint_id="CP-001",
    )
    print(f"\nChain reset: {event.reset_id}")
    print(f"  Rolled back to: {event.target.rollback_to_checkpoint}")
    
    print(f"\nFinal state: {controller.get_state()}")
    print(f"Reset history: {len(controller.reset_history)} events")
