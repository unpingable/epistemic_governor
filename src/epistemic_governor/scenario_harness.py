"""
Scenario Harness - Emit traces by default

The scenario harness runs:
- Canned scenarios (golden files)
- Adversarial scenarios (stress tests)
- Real transcripts (redacted)

Output is always: trace.jsonl

Usage:
    python -m epistemic_governor.scenario_harness run scenarios/
    python -m epistemic_governor.scenario_harness run --adversarial
    python -m epistemic_governor.scenario_harness list
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import random

from epistemic_governor.sovereign import SovereignGovernor, SovereignConfig
from epistemic_governor.profile import Profile, get_profile, apply_profile_to_governor
from epistemic_governor.trace_collector import TraceCollector
from epistemic_governor.control.regime import RegimeSignals


# =============================================================================
# Scenario Definitions
# =============================================================================

@dataclass
class ScenarioStep:
    """A single step in a scenario."""
    input_text: str
    expected_regime: Optional[str] = None
    expected_cf_codes: List[str] = field(default_factory=list)
    inject_evidence: bool = False
    evidence_type: Optional[str] = None
    delay_seconds: float = 0.0


@dataclass
class Scenario:
    """
    A complete test scenario.
    
    Scenarios define a sequence of inputs and expected behaviors.
    """
    name: str
    description: str
    steps: List[ScenarioStep]
    profile: str = "balanced"  # Which profile archetype to use
    tags: List[str] = field(default_factory=list)  # For filtering
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "profile": self.profile,
            "tags": self.tags,
            "steps": [asdict(s) for s in self.steps],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scenario":
        steps = [ScenarioStep(**s) for s in data.get("steps", [])]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            steps=steps,
            profile=data.get("profile", "balanced"),
            tags=data.get("tags", []),
        )
    
    @classmethod
    def load(cls, path: Path | str) -> "Scenario":
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class ScenarioResult:
    """Result of running a scenario."""
    scenario_name: str
    profile_used: str
    trace_path: str
    total_steps: int
    passed_steps: int
    failed_steps: int
    cf_events: List[Dict[str, Any]]
    regime_transitions: List[str]
    duration_seconds: float
    errors: List[str] = field(default_factory=list)


# =============================================================================
# Built-in Scenarios
# =============================================================================

def make_normal_scenario() -> Scenario:
    """Normal operation - should stay ELASTIC."""
    return Scenario(
        name="normal_operation",
        description="Standard claims, no stress - should stay ELASTIC",
        profile="balanced",
        tags=["baseline", "smoke"],
        steps=[
            ScenarioStep("The sky is blue."),
            ScenarioStep("Water freezes at 0 degrees Celsius."),
            ScenarioStep("Python is a programming language."),
            ScenarioStep("The earth orbits the sun.", expected_regime="ELASTIC"),
        ],
    )


def make_stress_scenario() -> Scenario:
    """High volume - should trigger regime shift."""
    return Scenario(
        name="stress_test",
        description="High claim volume - should shift to WARM or DUCTILE",
        profile="production",  # Use production to see regime shifts
        tags=["stress", "regime"],
        steps=[
            ScenarioStep("Claim 1. Claim 2. Claim 3. Claim 4. Claim 5."),
            ScenarioStep("More claims. Even more claims. So many claims."),
            ScenarioStep("Still going. Never stopping. Claims everywhere."),
            ScenarioStep("Peak stress reached.", expected_regime="WARM"),
            ScenarioStep("Backing off now."),
            ScenarioStep("Normal again.", expected_regime="ELASTIC"),
        ],
    )


def make_contradiction_scenario() -> Scenario:
    """Contradictory claims - should trigger CF detection."""
    return Scenario(
        name="contradiction_test",
        description="Contradictory claims - should log CF events",
        profile="adversarial",
        tags=["cf", "contradiction"],
        steps=[
            ScenarioStep("The cat is on the mat."),
            ScenarioStep("The cat is not on the mat."),  # Contradiction
            ScenarioStep("Let me finalize this.", expected_cf_codes=["CF-3"]),
        ],
    )


def make_rapid_escalation_scenario() -> Scenario:
    """Fast commitment escalation - should trigger CF-2."""
    return Scenario(
        name="rapid_escalation",
        description="Fast commitment changes - should trigger CF-2 (tempo)",
        profile="production",  # Longer contest window
        tags=["cf", "tempo"],
        steps=[
            ScenarioStep("I propose X."),
            ScenarioStep("X is now confirmed.", delay_seconds=0.5),  # Too fast
            ScenarioStep("X is absolutely final.", expected_cf_codes=["CF-2"]),
        ],
    )


BUILTIN_SCENARIOS = {
    "normal": make_normal_scenario,
    "stress": make_stress_scenario,
    "contradiction": make_contradiction_scenario,
    "rapid_escalation": make_rapid_escalation_scenario,
}


# =============================================================================
# Scenario Runner
# =============================================================================

class ScenarioRunner:
    """
    Runs scenarios and emits traces.
    
    The core of the learning loop harness.
    """
    
    def __init__(self, output_dir: Path | str = "traces"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(
        self, 
        scenario: Scenario,
        profile_override: Optional[Profile] = None,
    ) -> ScenarioResult:
        """
        Run a single scenario and emit trace.
        """
        start_time = time.time()
        
        # Get profile
        profile = profile_override or get_profile(scenario.profile)
        
        # Create governor with profile
        config = SovereignConfig(boil_control_enabled=True)
        gov = SovereignGovernor(config)
        apply_profile_to_governor(profile, gov)
        
        # Create trace collector
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_path = self.output_dir / f"{scenario.name}_{timestamp}.jsonl"
        collector = TraceCollector(
            trace_path,
            profile=profile.to_dict(),
            config=config.__dict__ if hasattr(config, '__dict__') else {},
        )
        
        # Run steps
        passed = 0
        failed = 0
        errors = []
        regime_transitions = []
        all_cf_events = []
        
        for i, step in enumerate(scenario.steps):
            try:
                # Delay if specified
                if step.delay_seconds > 0:
                    time.sleep(step.delay_seconds)
                
                # Process input
                result = gov.process(step.input_text)
                
                # Record trace
                collector.record_from_governor(gov, output_text=result.output.text if result.output else None)
                
                # Track regime
                if gov.last_regime_response:
                    regime = gov.last_regime_response.get("regime")
                    if not regime_transitions or regime_transitions[-1] != regime:
                        regime_transitions.append(regime)
                
                # Collect CF events
                cf_events = gov.fsm.get_cf_events()
                all_cf_events.extend(cf_events)
                
                # Check expectations
                step_passed = True
                
                if step.expected_regime:
                    actual_regime = gov.last_regime_response.get("regime") if gov.last_regime_response else None
                    if actual_regime != step.expected_regime:
                        step_passed = False
                        errors.append(f"Step {i}: Expected regime {step.expected_regime}, got {actual_regime}")
                
                if step.expected_cf_codes:
                    recent_cf = [e.get("cf_code") for e in cf_events[-len(step.expected_cf_codes):]]
                    for expected in step.expected_cf_codes:
                        if expected not in recent_cf:
                            step_passed = False
                            errors.append(f"Step {i}: Expected CF {expected}, got {recent_cf}")
                
                if step_passed:
                    passed += 1
                else:
                    failed += 1
                    
            except Exception as e:
                failed += 1
                errors.append(f"Step {i}: Exception - {type(e).__name__}: {e}")
        
        # Close trace
        collector.close()
        
        duration = time.time() - start_time
        
        return ScenarioResult(
            scenario_name=scenario.name,
            profile_used=profile.name,
            trace_path=str(trace_path),
            total_steps=len(scenario.steps),
            passed_steps=passed,
            failed_steps=failed,
            cf_events=all_cf_events,
            regime_transitions=regime_transitions,
            duration_seconds=duration,
            errors=errors,
        )
    
    def run_all(
        self, 
        scenarios: List[Scenario],
        profile_override: Optional[Profile] = None,
    ) -> List[ScenarioResult]:
        """Run multiple scenarios."""
        results = []
        for scenario in scenarios:
            result = self.run(scenario, profile_override)
            results.append(result)
        return results
    
    def run_builtin(self, name: str) -> ScenarioResult:
        """Run a built-in scenario by name."""
        if name not in BUILTIN_SCENARIOS:
            raise ValueError(f"Unknown scenario: {name}. Available: {list(BUILTIN_SCENARIOS.keys())}")
        scenario = BUILTIN_SCENARIOS[name]()
        return self.run(scenario)


# =============================================================================
# Adversarial Scenario Generator
# =============================================================================

class AdversarialGenerator:
    """Generate adversarial scenarios for stress testing."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
    
    def generate_claim_flood(self, n_steps: int = 10, claims_per_step: int = 5) -> Scenario:
        """Generate a scenario that floods claims."""
        steps = []
        for i in range(n_steps):
            claims = ". ".join([f"Claim {i}_{j}" for j in range(claims_per_step)])
            steps.append(ScenarioStep(claims))
        
        return Scenario(
            name="adversarial_flood",
            description=f"Flood test: {claims_per_step} claims per step for {n_steps} steps",
            profile="production",
            tags=["adversarial", "flood"],
            steps=steps,
        )
    
    def generate_contradiction_cascade(self, n_pairs: int = 5) -> Scenario:
        """Generate a scenario with cascading contradictions."""
        steps = []
        for i in range(n_pairs):
            steps.append(ScenarioStep(f"Statement {i} is true."))
            steps.append(ScenarioStep(f"Statement {i} is false."))
        
        steps.append(ScenarioStep("Now let me summarize all of this."))
        
        return Scenario(
            name="adversarial_contradiction_cascade",
            description=f"Cascade test: {n_pairs} contradictory pairs",
            profile="adversarial",
            tags=["adversarial", "contradiction"],
            steps=steps,
        )
    
    def generate_rapid_fire(self, n_steps: int = 20, delay: float = 0.1) -> Scenario:
        """Generate rapid-fire inputs to test tempo limits."""
        steps = [
            ScenarioStep(f"Quick claim {i}.", delay_seconds=delay)
            for i in range(n_steps)
        ]
        
        return Scenario(
            name="adversarial_rapid_fire",
            description=f"Rapid fire: {n_steps} claims at {delay}s intervals",
            profile="production",
            tags=["adversarial", "tempo"],
            steps=steps,
        )


# =============================================================================
# CLI
# =============================================================================

def print_result(result: ScenarioResult):
    """Pretty-print a scenario result."""
    status = "✓" if result.failed_steps == 0 else "✗"
    print(f"\n{status} {result.scenario_name}")
    print(f"  Profile: {result.profile_used}")
    print(f"  Steps: {result.passed_steps}/{result.total_steps} passed")
    print(f"  Duration: {result.duration_seconds:.2f}s")
    print(f"  Regime transitions: {' → '.join(result.regime_transitions) if result.regime_transitions else 'none'}")
    print(f"  CF events: {len(result.cf_events)}")
    print(f"  Trace: {result.trace_path}")
    
    if result.errors:
        print("  Errors:")
        for err in result.errors:
            print(f"    - {err}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Scenario Harness for Epistemic Governor")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available scenarios")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run scenarios")
    run_parser.add_argument("scenarios", nargs="*", help="Scenario names or paths")
    run_parser.add_argument("--all", action="store_true", help="Run all built-in scenarios")
    run_parser.add_argument("--adversarial", action="store_true", help="Run adversarial scenarios")
    run_parser.add_argument("--profile", help="Override profile archetype")
    run_parser.add_argument("--output", "-o", default="traces", help="Output directory")
    
    args = parser.parse_args()
    
    if args.command == "list":
        print("Built-in scenarios:")
        for name in BUILTIN_SCENARIOS:
            scenario = BUILTIN_SCENARIOS[name]()
            print(f"  - {name}: {scenario.description}")
    
    elif args.command == "run":
        runner = ScenarioRunner(args.output)
        
        # Determine what to run
        scenarios_to_run = []
        
        if args.all:
            scenarios_to_run = [factory() for factory in BUILTIN_SCENARIOS.values()]
        elif args.adversarial:
            gen = AdversarialGenerator()
            scenarios_to_run = [
                gen.generate_claim_flood(),
                gen.generate_contradiction_cascade(),
                gen.generate_rapid_fire(),
            ]
        elif args.scenarios:
            for name in args.scenarios:
                if name in BUILTIN_SCENARIOS:
                    scenarios_to_run.append(BUILTIN_SCENARIOS[name]())
                elif Path(name).exists():
                    scenarios_to_run.append(Scenario.load(name))
                else:
                    print(f"Unknown scenario: {name}")
        else:
            # Default: run normal
            scenarios_to_run = [make_normal_scenario()]
        
        # Get profile override
        profile_override = None
        if args.profile:
            profile_override = get_profile(args.profile)
        
        # Run
        print(f"Running {len(scenarios_to_run)} scenario(s)...")
        
        total_passed = 0
        total_failed = 0
        
        for scenario in scenarios_to_run:
            result = runner.run(scenario, profile_override)
            print_result(result)
            total_passed += result.passed_steps
            total_failed += result.failed_steps
        
        print(f"\n{'='*60}")
        print(f"Total: {total_passed} passed, {total_failed} failed")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
