"""
Profile System - Fit-able Configuration for Learning Loop

A profile is a serializable vector of parameters that can be:
1. Applied to configure the governor
2. Compared across traces
3. Optimized by the offline fitter

Profiles capture:
- Regime detection thresholds
- Boil control presets
- Contest window timing
- Tripwire sensitivity
- Output constraints

The key insight: profiles should be a dataclass you can serialize,
and a function apply_profile(profile) -> config.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime, timezone


@dataclass
class RegimeThresholds:
    """Thresholds for regime classification."""
    # WARM thresholds
    warm_hysteresis: float = 0.2
    warm_relaxation: float = 3.0
    warm_anisotropy: float = 0.3
    warm_provenance_deficit: float = 0.2
    
    # DUCTILE thresholds
    ductile_hysteresis: float = 0.5
    ductile_relaxation: float = 10.0
    ductile_anisotropy: float = 0.5
    ductile_budget_pressure: float = 0.7
    
    # UNSTABLE thresholds
    unstable_tool_gain: float = 1.0
    unstable_budget_pressure: float = 0.9


@dataclass
class BoilPresetParams:
    """Parameters for a boil control preset."""
    claim_budget_per_turn: int = 8
    novelty_tolerance: float = 0.3
    authority_posture: str = "normal"  # strict, normal, permissive
    variety_multiplier: float = 1.0
    horizon_turns: int = 10
    cycle_period_turns: int = 3
    hold_time_turns: int = 5
    min_dwell_turns: int = 2
    
    # Tripwire enables
    tripwire_contradiction: bool = True
    tripwire_provenance: bool = True
    tripwire_authority: bool = True
    tripwire_cascade: bool = True


@dataclass
class ContestWindowParams:
    """Parameters for coordination failure prevention."""
    # CF-2: Asymmetric tempo prevention
    min_contest_window_seconds: float = 5.0
    
    # CF-3: Repair suppression
    require_contradiction_resolution: bool = True
    allow_accepted_divergence: bool = True
    
    # CF-1: Unilateral closure
    require_user_acknowledgment_for_final: bool = True


@dataclass
class OutputConstraints:
    """Constraints on output generation."""
    max_uncommitted_sigma: float = 0.3
    cap_proposal_confidence: bool = True
    annotate_provisional: bool = True
    redact_quarantined: bool = False


@dataclass
class Profile:
    """
    A complete profile for the epistemic governor.
    
    This is the unit of optimization for the learning loop.
    Profiles can be serialized, compared, and fit.
    """
    # Identity
    name: str = "default"
    description: str = ""
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Archetype (for quick classification)
    archetype: str = "balanced"  # lab, production, adversarial, low_friction
    
    # Component configs
    regime_thresholds: RegimeThresholds = field(default_factory=RegimeThresholds)
    boil_preset: BoilPresetParams = field(default_factory=BoilPresetParams)
    contest_window: ContestWindowParams = field(default_factory=ContestWindowParams)
    output_constraints: OutputConstraints = field(default_factory=OutputConstraints)
    
    # Metadata from fitting
    fit_metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, path: Path | str) -> None:
        """Save profile to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Profile":
        """Create profile from dictionary."""
        return cls(
            name=data.get("name", "default"),
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            archetype=data.get("archetype", "balanced"),
            regime_thresholds=RegimeThresholds(**data.get("regime_thresholds", {})),
            boil_preset=BoilPresetParams(**data.get("boil_preset", {})),
            contest_window=ContestWindowParams(**data.get("contest_window", {})),
            output_constraints=OutputConstraints(**data.get("output_constraints", {})),
            fit_metadata=data.get("fit_metadata"),
        )
    
    @classmethod
    def load(cls, path: Path | str) -> "Profile":
        """Load profile from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def as_flat_vector(self) -> Dict[str, float]:
        """
        Convert to flat parameter vector for optimization.
        
        This is what the fitter sees.
        """
        return {
            # Regime thresholds
            "warm_hysteresis": self.regime_thresholds.warm_hysteresis,
            "warm_relaxation": self.regime_thresholds.warm_relaxation,
            "warm_anisotropy": self.regime_thresholds.warm_anisotropy,
            "warm_provenance_deficit": self.regime_thresholds.warm_provenance_deficit,
            "ductile_hysteresis": self.regime_thresholds.ductile_hysteresis,
            "ductile_relaxation": self.regime_thresholds.ductile_relaxation,
            "ductile_anisotropy": self.regime_thresholds.ductile_anisotropy,
            "ductile_budget_pressure": self.regime_thresholds.ductile_budget_pressure,
            "unstable_tool_gain": self.regime_thresholds.unstable_tool_gain,
            "unstable_budget_pressure": self.regime_thresholds.unstable_budget_pressure,
            
            # Boil preset (numeric only)
            "claim_budget_per_turn": float(self.boil_preset.claim_budget_per_turn),
            "novelty_tolerance": self.boil_preset.novelty_tolerance,
            "variety_multiplier": self.boil_preset.variety_multiplier,
            "min_dwell_turns": float(self.boil_preset.min_dwell_turns),
            
            # Contest window
            "min_contest_window_seconds": self.contest_window.min_contest_window_seconds,
            
            # Output
            "max_uncommitted_sigma": self.output_constraints.max_uncommitted_sigma,
        }
    
    @classmethod
    def from_flat_vector(cls, vector: Dict[str, float], base: Optional["Profile"] = None) -> "Profile":
        """
        Create profile from flat parameter vector.
        
        Used by the fitter to apply optimized parameters.
        """
        if base is None:
            base = cls()
        
        # Clone base
        profile = cls.from_dict(base.to_dict())
        
        # Apply vector values
        if "warm_hysteresis" in vector:
            profile.regime_thresholds.warm_hysteresis = vector["warm_hysteresis"]
        if "warm_relaxation" in vector:
            profile.regime_thresholds.warm_relaxation = vector["warm_relaxation"]
        if "warm_anisotropy" in vector:
            profile.regime_thresholds.warm_anisotropy = vector["warm_anisotropy"]
        if "warm_provenance_deficit" in vector:
            profile.regime_thresholds.warm_provenance_deficit = vector["warm_provenance_deficit"]
        if "ductile_hysteresis" in vector:
            profile.regime_thresholds.ductile_hysteresis = vector["ductile_hysteresis"]
        if "ductile_relaxation" in vector:
            profile.regime_thresholds.ductile_relaxation = vector["ductile_relaxation"]
        if "ductile_anisotropy" in vector:
            profile.regime_thresholds.ductile_anisotropy = vector["ductile_anisotropy"]
        if "ductile_budget_pressure" in vector:
            profile.regime_thresholds.ductile_budget_pressure = vector["ductile_budget_pressure"]
        if "unstable_tool_gain" in vector:
            profile.regime_thresholds.unstable_tool_gain = vector["unstable_tool_gain"]
        if "unstable_budget_pressure" in vector:
            profile.regime_thresholds.unstable_budget_pressure = vector["unstable_budget_pressure"]
        
        if "claim_budget_per_turn" in vector:
            profile.boil_preset.claim_budget_per_turn = int(vector["claim_budget_per_turn"])
        if "novelty_tolerance" in vector:
            profile.boil_preset.novelty_tolerance = vector["novelty_tolerance"]
        if "variety_multiplier" in vector:
            profile.boil_preset.variety_multiplier = vector["variety_multiplier"]
        if "min_dwell_turns" in vector:
            profile.boil_preset.min_dwell_turns = int(vector["min_dwell_turns"])
        
        if "min_contest_window_seconds" in vector:
            profile.contest_window.min_contest_window_seconds = vector["min_contest_window_seconds"]
        
        if "max_uncommitted_sigma" in vector:
            profile.output_constraints.max_uncommitted_sigma = vector["max_uncommitted_sigma"]
        
        return profile


# =============================================================================
# Archetype Profiles
# =============================================================================

def make_lab_profile() -> Profile:
    """
    Lab/Exploration profile.
    
    - Permissive
    - High trace detail
    - Low refusal hardness
    - Aggressive annotation prompts
    """
    return Profile(
        name="lab",
        description="Permissive exploration mode for testing and development",
        archetype="lab",
        regime_thresholds=RegimeThresholds(
            warm_hysteresis=0.4,      # Higher tolerance
            warm_relaxation=5.0,
            ductile_budget_pressure=0.85,
            unstable_tool_gain=1.5,   # Very permissive
        ),
        boil_preset=BoilPresetParams(
            claim_budget_per_turn=20,
            novelty_tolerance=0.6,
            authority_posture="permissive",
            variety_multiplier=1.5,
            min_dwell_turns=1,
            tripwire_provenance=False,  # Relaxed
        ),
        contest_window=ContestWindowParams(
            min_contest_window_seconds=2.0,  # Short for fast iteration
            require_user_acknowledgment_for_final=False,
        ),
        output_constraints=OutputConstraints(
            max_uncommitted_sigma=0.5,  # Higher for exploration
            annotate_provisional=True,
        ),
    )


def make_production_profile() -> Profile:
    """
    Production/Safety profile.
    
    - Strict evidence demands
    - Faster tripwires
    - Conservative output gate
    """
    return Profile(
        name="production",
        description="Conservative production mode with strict safety controls",
        archetype="production",
        regime_thresholds=RegimeThresholds(
            warm_hysteresis=0.15,     # Sensitive
            warm_relaxation=2.0,
            ductile_budget_pressure=0.6,
            unstable_tool_gain=0.8,   # Strict
            unstable_budget_pressure=0.8,
        ),
        boil_preset=BoilPresetParams(
            claim_budget_per_turn=5,
            novelty_tolerance=0.2,
            authority_posture="strict",
            variety_multiplier=0.7,
            min_dwell_turns=3,
            tripwire_contradiction=True,
            tripwire_provenance=True,
            tripwire_authority=True,
            tripwire_cascade=True,
        ),
        contest_window=ContestWindowParams(
            min_contest_window_seconds=10.0,  # Long contest window
            require_contradiction_resolution=True,
            require_user_acknowledgment_for_final=True,
        ),
        output_constraints=OutputConstraints(
            max_uncommitted_sigma=0.2,
            cap_proposal_confidence=True,
            annotate_provisional=True,
            redact_quarantined=True,
        ),
    )


def make_adversarial_profile() -> Profile:
    """
    Adversarial/Red Team profile.
    
    - Hair-trigger on regime shift
    - Heavy logging
    - Strict output boundary
    """
    return Profile(
        name="adversarial",
        description="Maximum sensitivity for adversarial testing",
        archetype="adversarial",
        regime_thresholds=RegimeThresholds(
            warm_hysteresis=0.1,      # Very sensitive
            warm_relaxation=1.0,
            warm_anisotropy=0.2,
            warm_provenance_deficit=0.1,
            ductile_hysteresis=0.3,
            ductile_budget_pressure=0.5,
            unstable_tool_gain=0.7,   # Very strict
            unstable_budget_pressure=0.7,
        ),
        boil_preset=BoilPresetParams(
            claim_budget_per_turn=3,
            novelty_tolerance=0.1,
            authority_posture="strict",
            variety_multiplier=0.5,
            min_dwell_turns=2,
            tripwire_contradiction=True,
            tripwire_provenance=True,
            tripwire_authority=True,
            tripwire_cascade=True,
        ),
        contest_window=ContestWindowParams(
            min_contest_window_seconds=15.0,  # Very long
            require_contradiction_resolution=True,
            allow_accepted_divergence=False,  # No easy outs
            require_user_acknowledgment_for_final=True,
        ),
        output_constraints=OutputConstraints(
            max_uncommitted_sigma=0.1,
            cap_proposal_confidence=True,
            annotate_provisional=True,
            redact_quarantined=True,
        ),
    )


def make_low_friction_profile() -> Profile:
    """
    Low-friction Assistant profile.
    
    - STRUCTURAL claims can close without evidence
    - WORLD claims require evidence
    - Reduced overhead
    """
    return Profile(
        name="low_friction",
        description="Balanced assistant mode with reduced overhead for structural claims",
        archetype="low_friction",
        regime_thresholds=RegimeThresholds(
            warm_hysteresis=0.25,
            warm_relaxation=4.0,
            ductile_budget_pressure=0.75,
            unstable_tool_gain=1.2,
        ),
        boil_preset=BoilPresetParams(
            claim_budget_per_turn=12,
            novelty_tolerance=0.4,
            authority_posture="normal",
            variety_multiplier=1.2,
            min_dwell_turns=2,
            tripwire_provenance=False,  # Allow structural claims
        ),
        contest_window=ContestWindowParams(
            min_contest_window_seconds=3.0,  # Shorter for fluency
            require_user_acknowledgment_for_final=False,
        ),
        output_constraints=OutputConstraints(
            max_uncommitted_sigma=0.4,
            cap_proposal_confidence=False,  # Allow confident proposals
            annotate_provisional=False,     # Cleaner output
        ),
    )


ARCHETYPE_PROFILES = {
    "lab": make_lab_profile,
    "production": make_production_profile,
    "adversarial": make_adversarial_profile,
    "low_friction": make_low_friction_profile,
    "balanced": Profile,  # Default
}


def get_profile(name: str) -> Profile:
    """Get a profile by archetype name."""
    if name in ARCHETYPE_PROFILES:
        factory = ARCHETYPE_PROFILES[name]
        return factory()
    raise ValueError(f"Unknown profile archetype: {name}")


def list_profiles() -> List[str]:
    """List available profile archetypes."""
    return list(ARCHETYPE_PROFILES.keys())


# =============================================================================
# Profile Application
# =============================================================================

def apply_profile_to_config(profile: Profile, config) -> None:
    """
    Apply a profile to a SovereignConfig.
    
    This is the bridge from profile parameters to runtime config.
    """
    # Output constraints
    config.max_uncommitted_sigma = profile.output_constraints.max_uncommitted_sigma
    
    # Boil control mode mapping
    if profile.boil_preset.authority_posture == "strict":
        config.boil_control_mode = "green_tea"
    elif profile.boil_preset.authority_posture == "permissive":
        config.boil_control_mode = "french_press"
    else:
        config.boil_control_mode = "oolong"


def apply_profile_to_governor(profile: Profile, governor) -> None:
    """
    Apply a profile to a SovereignGovernor.
    
    Updates all configurable parameters at runtime.
    """
    # Apply to config
    apply_profile_to_config(profile, governor.config)
    
    # Contest window
    governor.fsm.min_contest_window_seconds = profile.contest_window.min_contest_window_seconds
    
    # Update projector
    governor.projector.config.max_uncommitted_sigma = profile.output_constraints.max_uncommitted_sigma


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile management for Epistemic Governor")
    parser.add_argument("command", choices=["list", "show", "create", "export"])
    parser.add_argument("--name", help="Profile archetype name")
    parser.add_argument("--output", "-o", help="Output file path")
    
    args = parser.parse_args()
    
    if args.command == "list":
        print("Available profile archetypes:")
        for name in list_profiles():
            print(f"  - {name}")
    
    elif args.command == "show":
        if not args.name:
            parser.error("--name required for show command")
        profile = get_profile(args.name)
        print(profile.to_json())
    
    elif args.command == "create":
        if not args.name or not args.output:
            parser.error("--name and --output required for create command")
        profile = get_profile(args.name)
        profile.save(args.output)
        print(f"Created profile: {args.output}")
    
    elif args.command == "export":
        if not args.output:
            parser.error("--output required for export command")
        # Export all archetypes
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in list_profiles():
            profile = get_profile(name)
            profile.save(output_dir / f"{name}.json")
        print(f"Exported {len(list_profiles())} profiles to {output_dir}")


if __name__ == "__main__":
    main()
