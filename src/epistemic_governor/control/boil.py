"""
Boil Control - Named Regime Presets

Inspired by electric kettles: they don't "optimize tea," they enforce 
temperature bands + dwell time with crude but reliable control.

Key insights from the kettle pattern:
1. Discrete regimes, not continuous sliders (green tea vs french press vs boil)
2. Bands + hysteresis, not exact targets (hold [85-Δ, 85+Δ], don't thrash)
3. Dwell time matters (hold stability for N cycles before allowing changes)
4. "Boil" is a phase-change sentinel, not "slightly hotter" (hard tripwires)

This sits above RegimeDetector as the preset layer:
- User picks a mode
- Mode configures thresholds + dwell time + tripwires
- RegimeDetector does the actual classification
- BoilController enforces hold times and prevents thrashing
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from epistemic_governor.control.regime import (
    RegimeDetector,
    RegimeSignals,
    RegimeThresholds,
    OperationalRegime,
)
from epistemic_governor.control.reset import ResetController


class ControlMode(Enum):
    """Named control modes - like kettle temperature presets."""
    GREEN_TEA = auto()      # Delicate - low temp, tight bounds, cautious
    WHITE_TEA = auto()      # Light extraction - slightly more tolerance
    OOLONG = auto()         # Medium - balanced extraction
    BLACK_TEA = auto()      # Robust - higher tolerance, more extraction
    FRENCH_PRESS = auto()   # Aggressive extraction - near limits but bounded
    BOIL = auto()           # Phase-change sentinel - hard tripwires only


@dataclass
class BoilPreset:
    """
    Named regime preset - the kettle setting.
    
    Each preset is a tuple of:
    - Setpoints: what we're targeting (budgets, tolerances)
    - Bands: hysteresis to prevent thrashing
    - Timing: cycle period, hold time before escalation
    - Tripwires: hard-stop conditions (the "boil" sentinels)
    """
    name: str
    mode: ControlMode
    
    # === Setpoints (what we're targeting) ===
    claim_budget_per_turn: int = 10          # max claims before pushback
    novelty_tolerance: float = 0.3           # speculation band [0-1]
    authority_posture: str = "normal"        # "strict", "normal", "permissive"
    variety_multiplier: float = 1.0          # scale variety bounds
    horizon_turns: int = 10                  # planning horizon
    
    # === Bands (hysteresis thresholds) ===
    # These configure RegimeThresholds
    warm_hysteresis: float = 0.2
    warm_relaxation: float = 3.0
    warm_anisotropy: float = 0.3
    ductile_hysteresis: float = 0.5
    ductile_relaxation: float = 10.0
    ductile_anisotropy: float = 0.5
    unstable_tool_gain: float = 1.0
    
    # === Timing ===
    cycle_period_turns: int = 3              # re-evaluate every N turns
    hold_time_turns: int = 5                 # stability required before mode change
    min_dwell_turns: int = 2                 # minimum time in regime before allowing transition
    
    # === Tripwires (hard-stop sentinels) ===
    contradiction_trip: bool = True          # instant intervention on contradiction
    provenance_trip: bool = True             # instant intervention on missing provenance  
    authority_trip: bool = True              # instant intervention on authority spoof
    cascade_trip: bool = True                # instant intervention on tool cascade (k > 1)
    
    def to_thresholds(self) -> RegimeThresholds:
        """Convert preset to RegimeThresholds."""
        return RegimeThresholds(
            warm_hysteresis=self.warm_hysteresis,
            warm_relaxation=self.warm_relaxation,
            warm_anisotropy=self.warm_anisotropy,
            ductile_hysteresis=self.ductile_hysteresis,
            ductile_relaxation=self.ductile_relaxation,
            ductile_anisotropy=self.ductile_anisotropy,
            unstable_tool_gain=self.unstable_tool_gain,
        )


# === Standard Presets ===

PRESETS: Dict[ControlMode, BoilPreset] = {
    ControlMode.GREEN_TEA: BoilPreset(
        name="green_tea",
        mode=ControlMode.GREEN_TEA,
        # Delicate - tight bounds, low extraction
        claim_budget_per_turn=3,
        novelty_tolerance=0.1,
        authority_posture="strict",
        variety_multiplier=0.5,
        horizon_turns=5,
        # Tight bands - intervene early
        warm_hysteresis=0.15,
        warm_relaxation=2.0,
        ductile_hysteresis=0.35,
        ductile_relaxation=6.0,
        unstable_tool_gain=0.8,
        # Long hold times - stability prioritized
        cycle_period_turns=2,
        hold_time_turns=8,
        min_dwell_turns=3,
        # All tripwires active
        contradiction_trip=True,
        provenance_trip=True,
        authority_trip=True,
        cascade_trip=True,
    ),
    
    ControlMode.WHITE_TEA: BoilPreset(
        name="white_tea",
        mode=ControlMode.WHITE_TEA,
        claim_budget_per_turn=5,
        novelty_tolerance=0.2,
        authority_posture="strict",
        variety_multiplier=0.7,
        horizon_turns=7,
        warm_hysteresis=0.18,
        warm_relaxation=2.5,
        ductile_hysteresis=0.4,
        ductile_relaxation=8.0,
        unstable_tool_gain=0.9,
        cycle_period_turns=2,
        hold_time_turns=6,
        min_dwell_turns=3,
    ),
    
    ControlMode.OOLONG: BoilPreset(
        name="oolong",
        mode=ControlMode.OOLONG,
        # Balanced - medium extraction
        claim_budget_per_turn=8,
        novelty_tolerance=0.3,
        authority_posture="normal",
        variety_multiplier=1.0,
        horizon_turns=10,
        # Standard bands
        warm_hysteresis=0.2,
        warm_relaxation=3.0,
        ductile_hysteresis=0.5,
        ductile_relaxation=10.0,
        unstable_tool_gain=1.0,
        cycle_period_turns=3,
        hold_time_turns=5,
        min_dwell_turns=2,
    ),
    
    ControlMode.BLACK_TEA: BoilPreset(
        name="black_tea",
        mode=ControlMode.BLACK_TEA,
        # Robust - more tolerance
        claim_budget_per_turn=12,
        novelty_tolerance=0.4,
        authority_posture="normal",
        variety_multiplier=1.2,
        horizon_turns=12,
        # Looser bands
        warm_hysteresis=0.25,
        warm_relaxation=4.0,
        ductile_hysteresis=0.55,
        ductile_relaxation=12.0,
        unstable_tool_gain=1.1,
        cycle_period_turns=4,
        hold_time_turns=4,
        min_dwell_turns=2,
    ),
    
    ControlMode.FRENCH_PRESS: BoilPreset(
        name="french_press",
        mode=ControlMode.FRENCH_PRESS,
        # Aggressive extraction - near limits but still bounded
        claim_budget_per_turn=20,
        novelty_tolerance=0.5,
        authority_posture="permissive",
        variety_multiplier=1.5,
        horizon_turns=15,
        # Wide bands - let it run
        warm_hysteresis=0.3,
        warm_relaxation=5.0,
        ductile_hysteresis=0.6,
        ductile_relaxation=15.0,
        unstable_tool_gain=1.2,
        # Shorter hold times - more responsive
        cycle_period_turns=5,
        hold_time_turns=3,
        min_dwell_turns=1,
        # Still trip on hard failures
        contradiction_trip=True,
        provenance_trip=False,  # Allow some ungrounded exploration
        authority_trip=True,
        cascade_trip=True,
    ),
    
    ControlMode.BOIL: BoilPreset(
        name="boil",
        mode=ControlMode.BOIL,
        # Phase-change sentinel - tripwires only, no gradual control
        claim_budget_per_turn=100,  # effectively unlimited
        novelty_tolerance=1.0,
        authority_posture="permissive",
        variety_multiplier=2.0,
        horizon_turns=20,
        # Very wide bands - only trip on hard failures
        warm_hysteresis=0.8,
        warm_relaxation=20.0,
        ductile_hysteresis=0.9,
        ductile_relaxation=30.0,
        unstable_tool_gain=1.5,
        cycle_period_turns=10,
        hold_time_turns=1,
        min_dwell_turns=1,
        # Only cascade trip active - pure sentinel mode
        contradiction_trip=False,
        provenance_trip=False,
        authority_trip=False,
        cascade_trip=True,  # The one hard limit: k > 1 = shut down
    ),
}


@dataclass
class DwellState:
    """Track time spent in current regime for dwell enforcement."""
    regime: OperationalRegime
    entered_at_turn: int
    stable_since_turn: int  # last turn where we didn't want to transition


@dataclass
class BoilEvent:
    """Record of a boil control event."""
    timestamp: datetime
    event_type: str  # "mode_change", "tripwire", "dwell_hold", "transition_allowed"
    details: Dict[str, Any]


class BoilController:
    """
    Boil Control - named regime presets with dwell time enforcement.
    
    Sits above RegimeDetector:
    1. User picks a mode (GREEN_TEA, FRENCH_PRESS, etc.)
    2. Mode configures thresholds + timing
    3. Controller enforces hold times and prevents thrashing
    4. Tripwires provide hard-stop sentinels
    """
    
    def __init__(self, mode: ControlMode = ControlMode.OOLONG):
        self.preset = PRESETS[mode]
        self.detector = RegimeDetector(
            thresholds=self.preset.to_thresholds(),
            collect_metrics=True,
        )
        
        self.turn_counter = 0
        self.dwell = DwellState(
            regime=OperationalRegime.ELASTIC,
            entered_at_turn=0,
            stable_since_turn=0,
        )
        self.event_log: List[BoilEvent] = []
        
        # Track pending transitions blocked by dwell
        self.pending_transition: Optional[OperationalRegime] = None
        self.pending_reason: Optional[str] = None
    
    def set_mode(self, mode: ControlMode):
        """Change control mode (like changing kettle temperature)."""
        old_mode = self.preset.mode
        self.preset = PRESETS[mode]
        self.detector = RegimeDetector(
            thresholds=self.preset.to_thresholds(),
            reset_controller=self.detector.reset_controller,  # Keep reset state
            collect_metrics=True,
        )
        
        self._log_event("mode_change", {
            "from": old_mode.name,
            "to": mode.name,
            "turn": self.turn_counter,
        })
    
    def check_tripwires(self, signals: RegimeSignals) -> Optional[str]:
        """
        Check hard-stop sentinels (the "boil" triggers).
        
        Returns tripwire name if triggered, None otherwise.
        """
        p = self.preset
        
        if p.cascade_trip and signals.tool_gain_estimate >= 1.0:
            return "cascade"
        
        if p.contradiction_trip and signals.c_accumulating:
            return "contradiction"
        
        if p.provenance_trip and signals.provenance_deficit_rate > 0.5:
            return "provenance"
        
        # Authority trip would need additional signal - placeholder
        # if p.authority_trip and signals.authority_violation:
        #     return "authority"
        
        return None
    
    def process_turn(self, signals: RegimeSignals) -> Dict[str, Any]:
        """
        Process a turn through boil control.
        
        Returns response dict with:
        - regime: current operational regime
        - action: what to do (CONTINUE, TIGHTEN, RESET, EMERGENCY_STOP)
        - tripwire: if a hard sentinel triggered
        - dwell_blocked: if a transition was blocked by dwell time
        - preset: current preset name
        """
        self.turn_counter += 1
        
        # 1. Check tripwires first (hard sentinels bypass everything)
        tripwire = self.check_tripwires(signals)
        if tripwire:
            self._log_event("tripwire", {
                "tripwire": tripwire,
                "turn": self.turn_counter,
                "signals": signals.to_dict(),
            })
            
            # Force UNSTABLE regime
            self.dwell = DwellState(
                regime=OperationalRegime.UNSTABLE,
                entered_at_turn=self.turn_counter,
                stable_since_turn=self.turn_counter,
            )
            self.detector.current_regime = OperationalRegime.UNSTABLE
            
            # Execute emergency response
            response = self.detector.respond(signals)
            response["tripwire"] = tripwire
            response["preset"] = self.preset.name
            response["action"] = "EMERGENCY_STOP"  # Force action
            response["regime"] = "UNSTABLE"  # Force regime
            return response
        
        # 2. Get what regime detector thinks we should be in
        proposed_regime, reason = self.detector.classify(signals)
        current_regime = self.dwell.regime
        
        # 3. Check dwell time constraints
        turns_in_regime = self.turn_counter - self.dwell.entered_at_turn
        
        if proposed_regime != current_regime:
            # Transition requested
            if turns_in_regime < self.preset.min_dwell_turns:
                # Block transition - haven't dwelled long enough
                self._log_event("dwell_hold", {
                    "from": current_regime.name,
                    "to": proposed_regime.name,
                    "turns_in_regime": turns_in_regime,
                    "min_dwell": self.preset.min_dwell_turns,
                    "turn": self.turn_counter,
                })
                
                self.pending_transition = proposed_regime
                self.pending_reason = reason
                
                # Stay in current regime
                response = self._make_response(current_regime, signals)
                response["dwell_blocked"] = True
                response["pending_transition"] = proposed_regime.name
                return response
            else:
                # Transition allowed
                self._log_event("transition_allowed", {
                    "from": current_regime.name,
                    "to": proposed_regime.name,
                    "reason": reason,
                    "turn": self.turn_counter,
                })
                
                self.dwell = DwellState(
                    regime=proposed_regime,
                    entered_at_turn=self.turn_counter,
                    stable_since_turn=self.turn_counter,
                )
                self.pending_transition = None
                self.pending_reason = None
        else:
            # Staying in same regime - update stability
            self.dwell.stable_since_turn = self.turn_counter
        
        # 4. Execute response for current regime
        # Update detector's current regime to match dwell-enforced regime
        self.detector.current_regime = self.dwell.regime
        response = self.detector.respond(signals)
        response["preset"] = self.preset.name
        response["turns_in_regime"] = turns_in_regime
        response["dwell_blocked"] = False
        
        return response
    
    def _make_response(self, regime: OperationalRegime, signals: RegimeSignals) -> Dict[str, Any]:
        """Make a response for a regime without triggering detector's reset logic."""
        response = {
            "regime": regime.name,
            "preset": self.preset.name,
            "turn": self.turn_counter,
        }
        
        if regime == OperationalRegime.ELASTIC:
            response["action"] = "CONTINUE"
        elif regime == OperationalRegime.WARM:
            response["action"] = "TIGHTEN"
        elif regime == OperationalRegime.DUCTILE:
            response["action"] = "RESET"
        else:
            response["action"] = "EMERGENCY_STOP"
        
        return response
    
    def _log_event(self, event_type: str, details: Dict[str, Any]):
        """Log a boil control event."""
        self.event_log.append(BoilEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            details=details,
        ))
    
    def get_state(self) -> Dict[str, Any]:
        """Get current controller state."""
        return {
            "mode": self.preset.mode.name,
            "preset": self.preset.name,
            "turn": self.turn_counter,
            "regime": self.dwell.regime.name,
            "turns_in_regime": self.turn_counter - self.dwell.entered_at_turn,
            "pending_transition": self.pending_transition.name if self.pending_transition else None,
            "detector_state": self.detector.get_state(),
            "events": len(self.event_log),
        }
    
    def get_preset_info(self) -> Dict[str, Any]:
        """Get current preset configuration."""
        p = self.preset
        return {
            "name": p.name,
            "mode": p.mode.name,
            "claim_budget": p.claim_budget_per_turn,
            "novelty_tolerance": p.novelty_tolerance,
            "authority_posture": p.authority_posture,
            "variety_multiplier": p.variety_multiplier,
            "horizon_turns": p.horizon_turns,
            "cycle_period": p.cycle_period_turns,
            "hold_time": p.hold_time_turns,
            "min_dwell": p.min_dwell_turns,
            "tripwires": {
                "contradiction": p.contradiction_trip,
                "provenance": p.provenance_trip,
                "authority": p.authority_trip,
                "cascade": p.cascade_trip,
            },
        }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=== Boil Control Test ===\n")
    
    # Test with OOLONG (balanced) preset
    controller = BoilController(ControlMode.OOLONG)
    print(f"Preset: {controller.get_preset_info()['name']}")
    print(f"Mode: {controller.preset.mode.name}")
    print()
    
    # Simulate a sequence
    print("Simulating turns...")
    
    # Normal operation
    for i in range(3):
        signals = RegimeSignals(hysteresis_magnitude=0.1, tool_gain_estimate=0.3)
        response = controller.process_turn(signals)
        print(f"  Turn {controller.turn_counter}: {response['regime']} - {response['action']}")
    
    # Stress increases
    print("\nStress increasing...")
    signals = RegimeSignals(
        hysteresis_magnitude=0.4,
        relaxation_time_seconds=8.0,
        anisotropy_score=0.4,
    )
    response = controller.process_turn(signals)
    print(f"  Turn {controller.turn_counter}: {response['regime']} - {response['action']}")
    print(f"    dwell_blocked: {response.get('dwell_blocked', False)}")
    
    # More stress - should transition to WARM after dwell
    for i in range(3):
        response = controller.process_turn(signals)
        print(f"  Turn {controller.turn_counter}: {response['regime']} - {response['action']}")
    
    # Tripwire test
    print("\nTripwire test (cascade)...")
    signals = RegimeSignals(tool_gain_estimate=1.2)  # k > 1
    response = controller.process_turn(signals)
    print(f"  Turn {controller.turn_counter}: {response['regime']} - {response['action']}")
    print(f"    tripwire: {response.get('tripwire', 'none')}")
    
    # Mode change test
    print("\nChanging to GREEN_TEA mode...")
    controller.set_mode(ControlMode.GREEN_TEA)
    print(f"  New preset: {controller.get_preset_info()['name']}")
    print(f"  Claim budget: {controller.preset.claim_budget_per_turn}")
    print(f"  Tripwires: {controller.get_preset_info()['tripwires']}")
    
    print(f"\nFinal state: {controller.get_state()}")
    print(f"Events logged: {len(controller.event_log)}")
