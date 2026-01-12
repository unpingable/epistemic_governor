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

def run_validation(config_path: Optional[Path] = None):
    """Run full validation suite."""
    print("\n" + "="*70)
    print("REGIME DETECTOR VALIDATION HARNESS")
    print("="*70 + "\n")
    
    # Load thresholds from config if provided
    if config_path:
        from epistemic_governor.config_loader import load_thresholds
        thresholds = load_thresholds(config_path)
        print(f"Using thresholds from: {config_path}")
    else:
        thresholds = None
        print("Using default thresholds")
    
    harness = ValidationHarness(thresholds=thresholds)
    results = harness.run_all_scenarios()
    
    print("\nScenario Results:")
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


def collect_labeled_data(scenarios: List[WorkloadScenario]) -> List[Tuple[RegimeSignals, OperationalRegime, str]]:
    """Collect (signals, label, scenario_name) tuples for tuning."""
    data = []
    for scenario in scenarios:
        for signals, label in zip(scenario.signals, scenario.expected_regimes):
            data.append((signals, label, scenario.name))
    return data


def split_train_holdout(
    data: List[Tuple[RegimeSignals, OperationalRegime, str]],
    holdout_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List, List]:
    """Deterministic split for reproducibility."""
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    cutoff = int(len(indices) * (1 - holdout_ratio))
    train_idx = set(indices[:cutoff])
    train = [data[i] for i in range(len(data)) if i in train_idx]
    holdout = [data[i] for i in range(len(data)) if i not in train_idx]
    return train, holdout


def evaluate_thresholds(
    data: List[Tuple[RegimeSignals, OperationalRegime, str]],
    thresholds: RegimeThresholds,
) -> Dict[str, Any]:
    """Evaluate macro-F1 and false transition rate on labeled data."""
    detector = RegimeDetector(thresholds=thresholds, collect_metrics=False)
    y_true = []
    y_pred = []
    by_scenario: Dict[str, List[str]] = {}
    by_scenario_true: Dict[str, List[str]] = {}

    for signals, label, scenario_name in data:
        response = detector.respond(signals)
        y_true.append(label.name)
        y_pred.append(response["regime"])
        by_scenario.setdefault(scenario_name, []).append(response["regime"])
        by_scenario_true.setdefault(scenario_name, []).append(label.name)

    # Macro-F1
    regimes = [r.name for r in OperationalRegime]
    f1s = []
    for regime in regimes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == regime and p == regime)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != regime and p == regime)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == regime and p != regime)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    # Accuracy
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0

    # False transition rate
    false_transitions = 0
    total_transitions = 0
    for name, preds in by_scenario.items():
        truths = by_scenario_true[name]
        expected_transitions = sum(1 for i in range(1, len(truths)) if truths[i] != truths[i - 1])
        observed_transitions = sum(1 for i in range(1, len(preds)) if preds[i] != preds[i - 1])
        total_transitions += observed_transitions
        false_transitions += abs(observed_transitions - expected_transitions)

    false_transition_rate = false_transitions / total_transitions if total_transitions else 0.0

    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "false_transition_rate": false_transition_rate,
        "total_samples": len(data),
    }


def search_thresholds(
    train_data: List[Tuple[RegimeSignals, OperationalRegime, str]],
    seed: int = 42,
    trials: int = 200,
    penalty_weight: float = 0.5,
) -> Tuple[RegimeThresholds, Dict[str, Any]]:
    """Random search thresholds with a false-transition penalty."""
    rng = random.Random(seed)
    best_score = float("-inf")
    best_thresholds = RegimeThresholds()
    best_metrics: Dict[str, Any] = {}

    for trial in range(trials):
        candidate = RegimeThresholds(
            warm_hysteresis=rng.uniform(0.05, 0.4),
            warm_relaxation=rng.uniform(1.0, 6.0),
            warm_anisotropy=rng.uniform(0.1, 0.6),
            warm_provenance_deficit=rng.uniform(0.05, 0.4),
            ductile_hysteresis=rng.uniform(0.3, 0.7),
            ductile_relaxation=rng.uniform(6.0, 18.0),
            ductile_anisotropy=rng.uniform(0.35, 0.7),
            ductile_budget_pressure=rng.uniform(0.5, 0.9),
            unstable_tool_gain=rng.uniform(0.9, 1.2),
            unstable_budget_pressure=rng.uniform(0.75, 0.98),
        )
        metrics = evaluate_thresholds(train_data, candidate)
        score = metrics["macro_f1"] - penalty_weight * metrics["false_transition_rate"]
        
        if score > best_score:
            best_score = score
            best_thresholds = candidate
            best_metrics = {"score": score, "trial": trial, **metrics}

    return best_thresholds, best_metrics


def run_tuning(
    output_config: Optional[Path] = None,
    seed: int = 42,
    holdout_ratio: float = 0.2,
    trials: int = 200,
    penalty_weight: float = 0.5,
):
    """
    Tune RegimeThresholds on validation scenarios.
    
    Outputs tuned thresholds to a config file (not source code).
    """
    print("\n" + "="*70)
    print("REGIME THRESHOLD TUNING")
    print("="*70 + "\n")

    # Collect data from scenarios
    scenarios = [
        make_stable_workload(),
        make_gradual_stress_workload(),
        make_spike_workload(),
        make_cascade_workload(),
        make_oscillation_workload(),
    ]

    data = collect_labeled_data(scenarios)
    train, holdout = split_train_holdout(data, holdout_ratio=holdout_ratio, seed=seed)

    print(f"Data: {len(data)} samples ({len(train)} train, {len(holdout)} holdout)")
    print(f"Trials: {trials}, Penalty weight: {penalty_weight}")
    print()

    # Baseline
    baseline = RegimeThresholds()
    baseline_train = evaluate_thresholds(train, baseline)
    baseline_holdout = evaluate_thresholds(holdout, baseline)

    # Search
    print("Searching...")
    tuned_thresholds, tuned_train = search_thresholds(
        train, seed=seed, trials=trials, penalty_weight=penalty_weight
    )
    tuned_holdout = evaluate_thresholds(holdout, tuned_thresholds)

    # Report
    print("\n" + "-"*70)
    print("Results (macro-F1 / accuracy / false-transition-rate)")
    print("-"*70)
    print(f"Baseline train:   {baseline_train['macro_f1']:.3f} / {baseline_train['accuracy']:.3f} / {baseline_train['false_transition_rate']:.3f}")
    print(f"Baseline holdout: {baseline_holdout['macro_f1']:.3f} / {baseline_holdout['accuracy']:.3f} / {baseline_holdout['false_transition_rate']:.3f}")
    print(f"Tuned train:      {tuned_train['macro_f1']:.3f} / {tuned_train['accuracy']:.3f} / {tuned_train['false_transition_rate']:.3f}")
    print(f"Tuned holdout:    {tuned_holdout['macro_f1']:.3f} / {tuned_holdout['accuracy']:.3f} / {tuned_holdout['false_transition_rate']:.3f}")

    print("\n" + "-"*70)
    print("Tuned thresholds:")
    print("-"*70)
    print(f"warm_hysteresis:        {tuned_thresholds.warm_hysteresis:.3f}")
    print(f"warm_relaxation:        {tuned_thresholds.warm_relaxation:.3f}")
    print(f"warm_anisotropy:        {tuned_thresholds.warm_anisotropy:.3f}")
    print(f"warm_provenance_deficit:{tuned_thresholds.warm_provenance_deficit:.3f}")
    print(f"ductile_hysteresis:     {tuned_thresholds.ductile_hysteresis:.3f}")
    print(f"ductile_relaxation:     {tuned_thresholds.ductile_relaxation:.3f}")
    print(f"ductile_anisotropy:     {tuned_thresholds.ductile_anisotropy:.3f}")
    print(f"ductile_budget_pressure:{tuned_thresholds.ductile_budget_pressure:.3f}")
    print(f"unstable_tool_gain:     {tuned_thresholds.unstable_tool_gain:.3f}")
    print(f"unstable_budget_pressure:{tuned_thresholds.unstable_budget_pressure:.3f}")

    # Save to config file
    if output_config:
        from epistemic_governor.config_loader import save_thresholds
        save_thresholds(
            tuned_thresholds,
            output_config,
            notes={
                "tuning_date": datetime.now(timezone.utc).isoformat(),
                "tuning_method": f"random_search_{trials}_trials",
                "validation_accuracy": tuned_holdout["accuracy"],
                "holdout_macro_f1": tuned_holdout["macro_f1"],
                "seed": seed,
            }
        )
        print(f"\nSaved tuned thresholds to: {output_config}")
    else:
        # Save to default tuning output
        output_path = Path(__file__).parent / "tuned_thresholds.json"
        report = {
            "tuning_date": datetime.now(timezone.utc).isoformat(),
            "config": {
                "seed": seed,
                "holdout_ratio": holdout_ratio,
                "trials": trials,
                "penalty_weight": penalty_weight,
            },
            "baseline": {"train": baseline_train, "holdout": baseline_holdout},
            "tuned": {"train": tuned_train, "holdout": tuned_holdout},
            "thresholds": {
                "warm_hysteresis": tuned_thresholds.warm_hysteresis,
                "warm_relaxation": tuned_thresholds.warm_relaxation,
                "warm_anisotropy": tuned_thresholds.warm_anisotropy,
                "warm_provenance_deficit": tuned_thresholds.warm_provenance_deficit,
                "ductile_hysteresis": tuned_thresholds.ductile_hysteresis,
                "ductile_relaxation": tuned_thresholds.ductile_relaxation,
                "ductile_anisotropy": tuned_thresholds.ductile_anisotropy,
                "ductile_budget_pressure": tuned_thresholds.ductile_budget_pressure,
                "unstable_tool_gain": tuned_thresholds.unstable_tool_gain,
                "unstable_budget_pressure": tuned_thresholds.unstable_budget_pressure,
            },
        }
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved tuning report to: {output_path}")

    return tuned_thresholds


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Regime validation and tuning")
    parser.add_argument("command", nargs="?", default="validate", 
                       choices=["validate", "tune"],
                       help="Command to run (default: validate)")
    parser.add_argument("--config", type=Path, 
                       help="Config file for thresholds")
    parser.add_argument("--output", type=Path,
                       help="Output config file for tuned thresholds")
    parser.add_argument("--trials", type=int, default=200,
                       help="Number of tuning trials")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    if args.command == "tune":
        run_tuning(
            output_config=args.output,
            seed=args.seed,
            trials=args.trials,
        )
    else:
        run_validation(config_path=args.config)
