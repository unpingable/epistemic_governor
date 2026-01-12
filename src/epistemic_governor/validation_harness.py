"""
Validation Harness

Empirical validation infrastructure for the governor.

This is NOT about proving the theory - it's about de-risking implementation:
1. Are the thresholds calibrated correctly?
2. Do resets actually restore elastic regime?
3. Where are the failure mode boundaries?

Run with: PYTHONPATH=. python validation_harness.py
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import json
import random

sys.path.insert(0, str(Path(__file__).parent))

from control.regime import (
    RegimeDetector,
    OperationalRegime,
    RegimeSignals,
    RegimeThresholds,
)
from control.reset import ResetController


# =============================================================================
# Workload Definitions
# =============================================================================

@dataclass
class WorkloadScenario:
    """A test scenario with signal progression."""
    name: str
    description: str
    signals: List[RegimeSignals]
    expected_regimes: List[OperationalRegime]
    tags: List[str] = field(default_factory=list)


def make_stable_workload(turns: int = 20) -> WorkloadScenario:
    """Workload that should stay ELASTIC throughout."""
    signals = []
    for i in range(turns):
        signals.append(RegimeSignals(
            hysteresis_magnitude=0.05 + random.uniform(0, 0.1),
            relaxation_time_seconds=0.5 + random.uniform(0, 1),
            tool_gain_estimate=0.3 + random.uniform(0, 0.2),
            anisotropy_score=0.1 + random.uniform(0, 0.1),
            provenance_deficit_rate=0.05 + random.uniform(0, 0.05),
            budget_pressure=0.1 + random.uniform(0, 0.1),
        ))
    
    return WorkloadScenario(
        name="stable_baseline",
        description="Low-stress workload that should remain ELASTIC",
        signals=signals,
        expected_regimes=[OperationalRegime.ELASTIC] * turns,
        tags=["baseline", "stable"],
    )


def make_gradual_stress_workload(turns: int = 30) -> WorkloadScenario:
    """Workload that gradually increases stress."""
    signals = []
    expected = []
    
    for i in range(turns):
        progress = i / turns
        
        sig = RegimeSignals(
            hysteresis_magnitude=0.1 + 0.6 * progress,
            relaxation_time_seconds=1.0 + 15 * progress,
            tool_gain_estimate=0.3 + 0.5 * progress,
            anisotropy_score=0.1 + 0.5 * progress,
            provenance_deficit_rate=0.05 + 0.3 * progress,
            budget_pressure=0.1 + 0.6 * progress,
        )
        signals.append(sig)
        
        # Expected regime based on rough thresholds
        if progress < 0.3:
            expected.append(OperationalRegime.ELASTIC)
        elif progress < 0.5:
            expected.append(OperationalRegime.WARM)
        elif progress < 0.8:
            expected.append(OperationalRegime.DUCTILE)
        else:
            expected.append(OperationalRegime.UNSTABLE)
    
    return WorkloadScenario(
        name="gradual_stress",
        description="Gradually increasing stress to find regime boundaries",
        signals=signals,
        expected_regimes=expected,
        tags=["stress", "boundary"],
    )


def make_spike_workload(turns: int = 20, spike_at: int = 10) -> WorkloadScenario:
    """Workload with sudden stress spike."""
    signals = []
    expected = []
    
    for i in range(turns):
        if i < spike_at:
            # Normal
            sig = RegimeSignals(
                hysteresis_magnitude=0.1,
                relaxation_time_seconds=1.0,
                tool_gain_estimate=0.3,
            )
            expected.append(OperationalRegime.ELASTIC)
        elif i == spike_at:
            # Spike
            sig = RegimeSignals(
                hysteresis_magnitude=0.7,
                relaxation_time_seconds=20.0,
                tool_gain_estimate=0.9,
                anisotropy_score=0.6,
                budget_pressure=0.8,
            )
            expected.append(OperationalRegime.DUCTILE)
        else:
            # Recovery (should reset help?)
            sig = RegimeSignals(
                hysteresis_magnitude=0.3 - 0.02 * (i - spike_at),
                relaxation_time_seconds=5.0 - 0.3 * (i - spike_at),
                tool_gain_estimate=0.5 - 0.02 * (i - spike_at),
            )
            # Should recover to WARM then ELASTIC
            if i < spike_at + 3:
                expected.append(OperationalRegime.WARM)
            else:
                expected.append(OperationalRegime.ELASTIC)
        
        signals.append(sig)
    
    return WorkloadScenario(
        name="spike_recovery",
        description="Sudden spike with recovery - tests reset effectiveness",
        signals=signals,
        expected_regimes=expected,
        tags=["spike", "recovery"],
    )


def make_cascade_workload(turns: int = 15) -> WorkloadScenario:
    """Workload with tool cascade (k > 1)."""
    signals = []
    
    for i in range(turns):
        # Tool gain increases past 1.0
        tool_gain = 0.5 + 0.1 * i
        
        sig = RegimeSignals(
            hysteresis_magnitude=0.2 + 0.02 * i,
            relaxation_time_seconds=2.0,
            tool_gain_estimate=min(tool_gain, 1.5),
            budget_pressure=0.3 + 0.05 * i,
        )
        signals.append(sig)
    
    # Should hit UNSTABLE when tool_gain >= 1.0 (around turn 5)
    expected = (
        [OperationalRegime.ELASTIC] * 3 +
        [OperationalRegime.WARM] * 2 +
        [OperationalRegime.UNSTABLE] * 10
    )
    
    return WorkloadScenario(
        name="cascade_unstable",
        description="Tool cascade leading to UNSTABLE regime",
        signals=signals,
        expected_regimes=expected[:turns],
        tags=["cascade", "unstable"],
    )


def make_oscillation_workload(turns: int = 20) -> WorkloadScenario:
    """Workload that oscillates between regimes."""
    signals = []
    expected = []
    
    for i in range(turns):
        if i % 4 < 2:
            # Low stress
            sig = RegimeSignals(
                hysteresis_magnitude=0.15,
                relaxation_time_seconds=2.0,
                anisotropy_score=0.2,
            )
            expected.append(OperationalRegime.ELASTIC)
        else:
            # High stress
            sig = RegimeSignals(
                hysteresis_magnitude=0.55,
                relaxation_time_seconds=12.0,
                anisotropy_score=0.55,
                budget_pressure=0.75,
            )
            expected.append(OperationalRegime.DUCTILE)
        
        signals.append(sig)
    
    return WorkloadScenario(
        name="oscillation",
        description="Oscillating stress - tests transition behavior",
        signals=signals,
        expected_regimes=expected,
        tags=["oscillation", "transition"],
    )


# =============================================================================
# Validation Runner
# =============================================================================

@dataclass
class ValidationResult:
    """Result of running one workload."""
    scenario_name: str
    total_turns: int
    
    # Regime accuracy
    expected_regimes: List[str]
    actual_regimes: List[str]
    regime_accuracy: float
    
    # Transition analysis
    transitions: int
    false_transitions: int  # Transitions that didn't match expected
    
    # Reset analysis
    total_resets: int
    reset_types: Dict[str, int]
    
    # Metrics from detector
    metrics_summary: Dict[str, Any]
    
    # Timing
    duration_ms: float


class ValidationHarness:
    """
    Runs workloads through the regime detector and collects metrics.
    """
    
    def __init__(self, thresholds: Optional[RegimeThresholds] = None):
        self.thresholds = thresholds
        self.results: List[ValidationResult] = []
    
    def run_scenario(self, scenario: WorkloadScenario) -> ValidationResult:
        """Run a single scenario and collect results."""
        # Fresh detector for each scenario
        detector = RegimeDetector(
            thresholds=self.thresholds,
            collect_metrics=True,
        )
        
        start_time = datetime.now(timezone.utc)
        
        actual_regimes = []
        
        for signals in scenario.signals:
            response = detector.respond(signals)
            actual_regimes.append(response["regime"])
        
        end_time = datetime.now(timezone.utc)
        
        # Calculate accuracy
        matches = sum(
            1 for exp, act in zip(scenario.expected_regimes, actual_regimes)
            if exp.name == act
        )
        accuracy = matches / len(scenario.signals) if scenario.signals else 0
        
        # Count transitions
        transitions = len(detector.transition_history)
        
        # Count false transitions (simplified: any mismatch)
        expected_transitions = sum(
            1 for i in range(1, len(scenario.expected_regimes))
            if scenario.expected_regimes[i] != scenario.expected_regimes[i-1]
        )
        false_transitions = abs(transitions - expected_transitions)
        
        # Reset analysis
        reset_history = detector.reset_controller.reset_history
        reset_types = {}
        for r in reset_history:
            rt = r.reset_type.name
            reset_types[rt] = reset_types.get(rt, 0) + 1
        
        result = ValidationResult(
            scenario_name=scenario.name,
            total_turns=len(scenario.signals),
            expected_regimes=[r.name for r in scenario.expected_regimes],
            actual_regimes=actual_regimes,
            regime_accuracy=accuracy,
            transitions=transitions,
            false_transitions=false_transitions,
            total_resets=len(reset_history),
            reset_types=reset_types,
            metrics_summary=detector.metrics.get_summary() if detector.metrics else {},
            duration_ms=(end_time - start_time).total_seconds() * 1000,
        )
        
        self.results.append(result)
        return result
    
    def run_all_scenarios(self) -> List[ValidationResult]:
        """Run all standard scenarios."""
        scenarios = [
            make_stable_workload(),
            make_gradual_stress_workload(),
            make_spike_workload(),
            make_cascade_workload(),
            make_oscillation_workload(),
        ]
        
        results = []
        for scenario in scenarios:
            result = self.run_scenario(scenario)
            results.append(result)
        
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate validation report."""
        if not self.results:
            return {"error": "No results to report"}
        
        # Aggregate metrics
        total_turns = sum(r.total_turns for r in self.results)
        total_transitions = sum(r.transitions for r in self.results)
        total_resets = sum(r.total_resets for r in self.results)
        avg_accuracy = sum(r.regime_accuracy for r in self.results) / len(self.results)
        
        # Per-scenario summary
        scenarios = {}
        for r in self.results:
            scenarios[r.scenario_name] = {
                "turns": r.total_turns,
                "accuracy": f"{r.regime_accuracy:.1%}",
                "transitions": r.transitions,
                "false_transitions": r.false_transitions,
                "resets": r.total_resets,
                "reset_types": r.reset_types,
                "duration_ms": f"{r.duration_ms:.1f}",
            }
        
        # Threshold analysis from all runs
        all_threshold_data = {}
        for r in self.results:
            if "threshold_analysis" in r.metrics_summary:
                for regime, data in r.metrics_summary["threshold_analysis"].items():
                    if regime not in all_threshold_data:
                        all_threshold_data[regime] = []
                    all_threshold_data[regime].append(data)
        
        return {
            "summary": {
                "total_scenarios": len(self.results),
                "total_turns": total_turns,
                "total_transitions": total_transitions,
                "total_resets": total_resets,
                "average_accuracy": f"{avg_accuracy:.1%}",
            },
            "scenarios": scenarios,
            "threshold_data": all_threshold_data,
            "recommendations": self._generate_recommendations(),
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate threshold tuning recommendations."""
        recs = []
        
        for r in self.results:
            if r.regime_accuracy < 0.7:
                recs.append(f"{r.scenario_name}: Low accuracy ({r.regime_accuracy:.1%}) - review thresholds")
            
            if r.false_transitions > r.transitions * 0.3:
                recs.append(f"{r.scenario_name}: High false transition rate - thresholds may be too sensitive")
            
            if r.scenario_name == "stable_baseline" and r.transitions > 0:
                recs.append("stable_baseline: Should have 0 transitions - warm_* thresholds too low")
            
            if r.scenario_name == "cascade_unstable" and r.total_resets == 0:
                recs.append("cascade_unstable: No resets triggered - unstable_tool_gain threshold too high")
        
        if not recs:
            recs.append("All scenarios within expected parameters")
        
        return recs


# =============================================================================
# CLI
# =============================================================================

def run_validation():
    """Run full validation suite."""
    print("\n" + "="*70)
    print("REGIME DETECTOR VALIDATION HARNESS")
    print("="*70 + "\n")
    
    harness = ValidationHarness()
    results = harness.run_all_scenarios()
    
    print("Scenario Results:")
    print("-" * 70)
    
    for r in results:
        status = "✓" if r.regime_accuracy >= 0.7 else "✗"
        print(f"{status} {r.scenario_name}")
        print(f"    Accuracy: {r.regime_accuracy:.1%} ({r.total_turns} turns)")
        print(f"    Transitions: {r.transitions} (false: {r.false_transitions})")
        print(f"    Resets: {r.total_resets} {r.reset_types}")
        print()
    
    report = harness.generate_report()
    
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total scenarios: {report['summary']['total_scenarios']}")
    print(f"Total turns: {report['summary']['total_turns']}")
    print(f"Average accuracy: {report['summary']['average_accuracy']}")
    print(f"Total resets: {report['summary']['total_resets']}")
    
    print("\n" + "-"*70)
    print("RECOMMENDATIONS")
    print("-"*70)
    for rec in report["recommendations"]:
        print(f"  • {rec}")
    
    print("\n" + "="*70)
    
    # Save full report
    report_path = Path(__file__).parent / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to: {report_path}")
    
    return report


if __name__ == "__main__":
    run_validation()
