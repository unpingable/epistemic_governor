"""
Config Loader - Load regime thresholds from config files.

This separates tunable values from source code:
- Source defines structure (what thresholds exist)
- Config files define values (what the thresholds are)

Usage:
    from epistemic_governor.config_loader import load_thresholds, load_preset
    
    # Load thresholds
    thresholds = load_thresholds()  # uses default config
    thresholds = load_thresholds("path/to/custom.json")
    
    # Load a preset
    preset_config = load_preset("oolong")
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from epistemic_governor.control.regime import RegimeThresholds


# Default config location
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "regime_thresholds.json"


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the full config file."""
    path = config_path or DEFAULT_CONFIG_PATH
    
    if not path.exists():
        # Return empty dict if no config - will use code defaults
        return {}
    
    with open(path) as f:
        return json.load(f)


def load_thresholds(config_path: Optional[Path] = None) -> RegimeThresholds:
    """
    Load RegimeThresholds from config file.
    
    Falls back to code defaults if config missing.
    """
    config = load_config(config_path)
    thresholds_dict = config.get("thresholds", {})
    
    if not thresholds_dict:
        return RegimeThresholds()  # Code defaults
    
    return RegimeThresholds(
        warm_hysteresis=thresholds_dict.get("warm_hysteresis", 0.2),
        warm_relaxation=thresholds_dict.get("warm_relaxation", 3.0),
        warm_anisotropy=thresholds_dict.get("warm_anisotropy", 0.3),
        warm_provenance_deficit=thresholds_dict.get("warm_provenance_deficit", 0.2),
        ductile_hysteresis=thresholds_dict.get("ductile_hysteresis", 0.5),
        ductile_relaxation=thresholds_dict.get("ductile_relaxation", 10.0),
        ductile_anisotropy=thresholds_dict.get("ductile_anisotropy", 0.5),
        ductile_budget_pressure=thresholds_dict.get("ductile_budget_pressure", 0.7),
        unstable_tool_gain=thresholds_dict.get("unstable_tool_gain", 1.0),
        unstable_budget_pressure=thresholds_dict.get("unstable_budget_pressure", 0.9),
    )


def load_preset(preset_name: str, config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load a preset configuration by name.
    
    Returns dict with preset values, or empty dict if not found.
    """
    config = load_config(config_path)
    presets = config.get("presets", {})
    return presets.get(preset_name, {})


def save_thresholds(
    thresholds: RegimeThresholds,
    config_path: Optional[Path] = None,
    notes: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save thresholds back to config file.
    
    Preserves existing presets and metadata.
    """
    path = config_path or DEFAULT_CONFIG_PATH
    
    # Load existing config to preserve presets
    existing = load_config(path) if path.exists() else {}
    
    # Update thresholds
    existing["thresholds"] = {
        "warm_hysteresis": thresholds.warm_hysteresis,
        "warm_relaxation": thresholds.warm_relaxation,
        "warm_anisotropy": thresholds.warm_anisotropy,
        "warm_provenance_deficit": thresholds.warm_provenance_deficit,
        "ductile_hysteresis": thresholds.ductile_hysteresis,
        "ductile_relaxation": thresholds.ductile_relaxation,
        "ductile_anisotropy": thresholds.ductile_anisotropy,
        "ductile_budget_pressure": thresholds.ductile_budget_pressure,
        "unstable_tool_gain": thresholds.unstable_tool_gain,
        "unstable_budget_pressure": thresholds.unstable_budget_pressure,
    }
    
    # Update notes if provided
    if notes:
        existing["notes"] = {**existing.get("notes", {}), **notes}
    
    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


def get_config_path() -> Path:
    """Return the default config path for reference."""
    return DEFAULT_CONFIG_PATH


if __name__ == "__main__":
    # Quick test
    print(f"Config path: {DEFAULT_CONFIG_PATH}")
    print(f"Exists: {DEFAULT_CONFIG_PATH.exists()}")
    
    if DEFAULT_CONFIG_PATH.exists():
        thresholds = load_thresholds()
        print(f"\nLoaded thresholds:")
        print(f"  warm_hysteresis: {thresholds.warm_hysteresis}")
        print(f"  ductile_hysteresis: {thresholds.ductile_hysteresis}")
        print(f"  unstable_tool_gain: {thresholds.unstable_tool_gain}")
        
        preset = load_preset("oolong")
        print(f"\nOolong preset:")
        print(f"  claim_budget: {preset.get('claim_budget_per_turn')}")
