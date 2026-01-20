"""
Offline Fitter - Traces → Candidate Profiles

The fitter takes a corpus of traces and suggests profile parameter adjustments.

Inputs:
- A corpus of traces (from scenario harness or real usage)
- A scoring function (with configurable objectives)

Outputs:
- Suggested profile parameters (and diffs from baseline)
- Per-scenario metrics
- Regression warnings

Scoring objectives (pick 1-2 primary per profile archetype):
- safety: minimize CF events, maximize tripwire effectiveness
- truthfulness: minimize false commits, maximize evidence coverage  
- ergonomics: minimize user friction, balance contest windows
- throughput: maximize claims processed, minimize regime stress

Usage:
    python -m epistemic_governor.fitter fit traces/ --objective safety
    python -m epistemic_governor.fitter fit traces/ --objective ergonomics --baseline lab
    python -m epistemic_governor.fitter compare traces/ profile_a.json profile_b.json
"""

import json
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
import math

from epistemic_governor.profile import (
    Profile, RegimeThresholds, BoilPresetParams, 
    ContestWindowParams, OutputConstraints,
    get_profile, list_profiles,
)
from epistemic_governor.trace_collector import TraceEvent, load_trace


# =============================================================================
# Scoring Functions
# =============================================================================

@dataclass
class TraceMetrics:
    """Metrics extracted from a single trace."""
    trace_path: str
    total_turns: int = 0
    
    # CF metrics
    cf_count: int = 0
    cf_1_count: int = 0  # Unilateral closure
    cf_2_count: int = 0  # Asymmetric tempo
    cf_3_count: int = 0  # Repair suppression
    
    # Contradiction metrics
    contradictions_opened: int = 0
    contradictions_resolved: int = 0
    max_open_contradictions: int = 0
    
    # Regime metrics
    time_in_elastic: int = 0
    time_in_warm: int = 0
    time_in_ductile: int = 0
    time_in_unstable: int = 0
    regime_transitions: int = 0
    
    # Commitment metrics
    escalations_to_provisional: int = 0
    escalations_to_final: int = 0
    
    # Evidence metrics
    evidence_present_ratio: float = 0.0
    
    # Reset metrics
    resets_triggered: int = 0
    tripwires_fired: int = 0


def extract_metrics(events: List[TraceEvent]) -> TraceMetrics:
    """Extract metrics from a trace."""
    metrics = TraceMetrics(trace_path="")
    metrics.total_turns = len(events)
    
    prev_regime = None
    prev_commitment = None
    max_open = 0
    evidence_turns = 0
    
    for event in events:
        # Skip non-event records
        if not hasattr(event, 'signals') or event.signals is None:
            continue
        
        # CF events
        if hasattr(event, 'cf_events') and event.cf_events:
            for cf in event.cf_events:
                metrics.cf_count += 1
                code = cf.get('cf_code', '')
                if 'CF-1' in code:
                    metrics.cf_1_count += 1
                elif 'CF-2' in code:
                    metrics.cf_2_count += 1
                elif 'CF-3' in code:
                    metrics.cf_3_count += 1
        
        # Contradictions
        open_c = getattr(event, 'open_contradictions', 0) or 0
        max_open = max(max_open, open_c)
        
        # Regime time
        label = getattr(event, 'label', None)
        if label:
            if label == 'ELASTIC':
                metrics.time_in_elastic += 1
            elif label == 'WARM':
                metrics.time_in_warm += 1
            elif label == 'DUCTILE':
                metrics.time_in_ductile += 1
            elif label == 'UNSTABLE':
                metrics.time_in_unstable += 1
            
            if prev_regime and prev_regime != label:
                metrics.regime_transitions += 1
            prev_regime = label
        
        # Commitment
        mode = getattr(event, 'commitment_mode', None)
        if mode and prev_commitment:
            if mode == 'PROVISIONAL_COMMIT' and prev_commitment == 'PROPOSE':
                metrics.escalations_to_provisional += 1
            elif mode == 'FINAL_COMMIT' and prev_commitment == 'PROVISIONAL_COMMIT':
                metrics.escalations_to_final += 1
        prev_commitment = mode
        
        # Evidence
        evidence_refs = getattr(event, 'evidence_refs', [])
        if evidence_refs:
            evidence_turns += 1
        
        # Events
        events_dict = getattr(event, 'events', {}) or {}
        if events_dict.get('reset'):
            metrics.resets_triggered += 1
        if events_dict.get('tripwire'):
            metrics.tripwires_fired += 1
    
    metrics.max_open_contradictions = max_open
    if metrics.total_turns > 0:
        metrics.evidence_present_ratio = evidence_turns / metrics.total_turns
    
    return metrics


@dataclass
class ScoringWeights:
    """Weights for different scoring objectives."""
    # Safety: minimize bad outcomes
    cf_penalty: float = 10.0
    unstable_penalty: float = 5.0
    tripwire_missed_penalty: float = 20.0
    
    # Truthfulness: evidence quality
    evidence_reward: float = 2.0
    contradiction_unresolved_penalty: float = 3.0
    
    # Ergonomics: user friction
    regime_transition_penalty: float = 0.5
    time_in_warm_penalty: float = 0.1
    
    # Throughput: processing efficiency
    time_in_elastic_reward: float = 0.5


OBJECTIVE_WEIGHTS = {
    "safety": ScoringWeights(
        cf_penalty=20.0,
        unstable_penalty=10.0,
        tripwire_missed_penalty=50.0,
        evidence_reward=1.0,
        contradiction_unresolved_penalty=5.0,
        regime_transition_penalty=0.1,
        time_in_warm_penalty=0.0,
        time_in_elastic_reward=0.1,
    ),
    "truthfulness": ScoringWeights(
        cf_penalty=5.0,
        unstable_penalty=2.0,
        tripwire_missed_penalty=10.0,
        evidence_reward=10.0,
        contradiction_unresolved_penalty=15.0,
        regime_transition_penalty=0.1,
        time_in_warm_penalty=0.0,
        time_in_elastic_reward=0.5,
    ),
    "ergonomics": ScoringWeights(
        cf_penalty=2.0,
        unstable_penalty=5.0,
        tripwire_missed_penalty=5.0,
        evidence_reward=0.5,
        contradiction_unresolved_penalty=1.0,
        regime_transition_penalty=2.0,
        time_in_warm_penalty=1.0,
        time_in_elastic_reward=2.0,
    ),
    "throughput": ScoringWeights(
        cf_penalty=1.0,
        unstable_penalty=3.0,
        tripwire_missed_penalty=5.0,
        evidence_reward=0.2,
        contradiction_unresolved_penalty=0.5,
        regime_transition_penalty=0.5,
        time_in_warm_penalty=0.5,
        time_in_elastic_reward=5.0,
    ),
}


def score_metrics(metrics: TraceMetrics, weights: ScoringWeights) -> float:
    """
    Score a trace's metrics.
    
    Higher is better.
    """
    score = 0.0
    
    # Penalties
    score -= weights.cf_penalty * metrics.cf_count
    score -= weights.unstable_penalty * metrics.time_in_unstable
    score -= weights.contradiction_unresolved_penalty * metrics.max_open_contradictions
    score -= weights.regime_transition_penalty * metrics.regime_transitions
    score -= weights.time_in_warm_penalty * metrics.time_in_warm
    
    # Rewards
    score += weights.evidence_reward * metrics.evidence_present_ratio * metrics.total_turns
    score += weights.time_in_elastic_reward * metrics.time_in_elastic
    
    # Tripwire effectiveness (rewarded if fired when unstable)
    if metrics.time_in_unstable > 0 and metrics.tripwires_fired > 0:
        score += 5.0  # Tripwire worked
    elif metrics.time_in_unstable > 3 and metrics.tripwires_fired == 0:
        score -= weights.tripwire_missed_penalty  # Should have fired
    
    return score


def score_traces(
    traces: List[Path],
    weights: ScoringWeights,
) -> Tuple[float, List[TraceMetrics]]:
    """Score multiple traces and return total score + individual metrics."""
    all_metrics = []
    total_score = 0.0
    
    for trace_path in traces:
        events = load_trace(trace_path)
        metrics = extract_metrics(events)
        metrics.trace_path = str(trace_path)
        all_metrics.append(metrics)
        total_score += score_metrics(metrics, weights)
    
    return total_score, all_metrics


# =============================================================================
# Parameter Search
# =============================================================================

@dataclass
class FitResult:
    """Result of a fitting run."""
    baseline_profile: str
    objective: str
    baseline_score: float
    fitted_score: float
    improvement: float
    improvement_pct: float
    
    # Best parameters found
    fitted_params: Dict[str, float]
    param_diffs: Dict[str, float]  # Diff from baseline
    
    # Per-trace breakdown
    trace_metrics: List[Dict[str, Any]]
    
    # Search metadata
    trials: int
    search_time_seconds: float
    
    # Warnings
    warnings: List[str] = field(default_factory=list)


def mutate_profile(profile: Profile, mutation_rate: float = 0.3, rng: random.Random = None) -> Profile:
    """Create a mutated copy of a profile."""
    if rng is None:
        rng = random.Random()
    
    vec = profile.as_flat_vector()
    
    for key in vec:
        if rng.random() < mutation_rate:
            # Mutate by +/- 20%
            factor = 1.0 + rng.uniform(-0.2, 0.2)
            vec[key] = max(0.01, vec[key] * factor)
    
    return Profile.from_flat_vector(vec, base=profile)


def crossover_profiles(p1: Profile, p2: Profile, rng: random.Random = None) -> Profile:
    """Create a child profile from two parents."""
    if rng is None:
        rng = random.Random()
    
    v1 = p1.as_flat_vector()
    v2 = p2.as_flat_vector()
    
    child_vec = {}
    for key in v1:
        # Random blend
        alpha = rng.random()
        child_vec[key] = alpha * v1[key] + (1 - alpha) * v2[key]
    
    return Profile.from_flat_vector(child_vec, base=p1)


def run_fitting(
    traces: List[Path],
    baseline: Profile,
    objective: str = "safety",
    trials: int = 100,
    seed: int = 42,
) -> FitResult:
    """
    Run parameter fitting on traces.
    
    Uses evolutionary search to find better profile parameters.
    """
    import time
    start_time = time.time()
    
    rng = random.Random(seed)
    weights = OBJECTIVE_WEIGHTS.get(objective, OBJECTIVE_WEIGHTS["safety"])
    
    # Score baseline
    baseline_score, baseline_metrics = score_traces(traces, weights)
    
    # Initialize population
    population_size = min(20, trials // 5)
    population = [baseline]
    for _ in range(population_size - 1):
        population.append(mutate_profile(baseline, mutation_rate=0.5, rng=rng))
    
    # Score population
    scored = []
    for profile in population:
        # TODO: Actually run scenarios with this profile
        # For now, just score based on parameter heuristics
        score = baseline_score + rng.uniform(-10, 10)  # Placeholder
        scored.append((score, profile))
    
    # Evolution
    best_score = baseline_score
    best_profile = baseline
    
    for trial in range(trials):
        # Sort by score (higher is better)
        scored.sort(key=lambda x: -x[0])
        
        # Keep best
        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best_profile = scored[0][1]
        
        # Select parents (top half)
        parents = [p for _, p in scored[:len(scored)//2]]
        
        # Generate children
        children = []
        for _ in range(population_size - len(parents)):
            p1 = rng.choice(parents)
            p2 = rng.choice(parents)
            child = crossover_profiles(p1, p2, rng)
            child = mutate_profile(child, mutation_rate=0.2, rng=rng)
            children.append(child)
        
        # New population
        population = parents + children
        
        # Re-score
        scored = []
        for profile in population:
            score = baseline_score + rng.uniform(-10, 10)  # Placeholder
            scored.append((score, profile))
    
    # Final scoring with actual trace evaluation
    fitted_score, fitted_metrics = score_traces(traces, weights)
    
    end_time = time.time()
    
    # Compute diffs
    baseline_vec = baseline.as_flat_vector()
    fitted_vec = best_profile.as_flat_vector()
    param_diffs = {
        k: fitted_vec[k] - baseline_vec[k]
        for k in baseline_vec
    }
    
    # Check for warnings
    warnings = []
    if fitted_score < baseline_score:
        warnings.append("Fitted profile scored worse than baseline")
    if any(abs(d) > 0.5 for d in param_diffs.values()):
        warnings.append("Large parameter changes detected - validate carefully")
    
    return FitResult(
        baseline_profile=baseline.name,
        objective=objective,
        baseline_score=baseline_score,
        fitted_score=fitted_score,
        improvement=fitted_score - baseline_score,
        improvement_pct=(fitted_score - baseline_score) / abs(baseline_score) * 100 if baseline_score != 0 else 0,
        fitted_params=fitted_vec,
        param_diffs=param_diffs,
        trace_metrics=[asdict(m) for m in fitted_metrics],
        trials=trials,
        search_time_seconds=end_time - start_time,
        warnings=warnings,
    )


# =============================================================================
# Profile Comparison
# =============================================================================

@dataclass
class ComparisonResult:
    """Result of comparing two profiles on traces."""
    profile_a_name: str
    profile_b_name: str
    objective: str
    
    score_a: float
    score_b: float
    winner: str
    
    metrics_a: List[Dict[str, Any]]
    metrics_b: List[Dict[str, Any]]
    
    # Per-dimension comparison
    dimension_comparison: Dict[str, Dict[str, float]]


def compare_profiles(
    traces: List[Path],
    profile_a: Profile,
    profile_b: Profile,
    objective: str = "safety",
) -> ComparisonResult:
    """Compare two profiles on the same traces."""
    weights = OBJECTIVE_WEIGHTS.get(objective, OBJECTIVE_WEIGHTS["safety"])
    
    score_a, metrics_a = score_traces(traces, weights)
    score_b, metrics_b = score_traces(traces, weights)
    
    # Aggregate metrics for comparison
    def aggregate(metrics_list):
        if not metrics_list:
            return {}
        total = {}
        for m in metrics_list:
            for k, v in m.__dict__.items() if hasattr(m, '__dict__') else asdict(m).items():
                if isinstance(v, (int, float)):
                    total[k] = total.get(k, 0) + v
        return total
    
    agg_a = aggregate(metrics_a)
    agg_b = aggregate(metrics_b)
    
    dimension_comparison = {}
    for key in set(agg_a.keys()) | set(agg_b.keys()):
        dimension_comparison[key] = {
            "profile_a": agg_a.get(key, 0),
            "profile_b": agg_b.get(key, 0),
            "diff": agg_b.get(key, 0) - agg_a.get(key, 0),
        }
    
    return ComparisonResult(
        profile_a_name=profile_a.name,
        profile_b_name=profile_b.name,
        objective=objective,
        score_a=score_a,
        score_b=score_b,
        winner=profile_a.name if score_a >= score_b else profile_b.name,
        metrics_a=[asdict(m) for m in metrics_a],
        metrics_b=[asdict(m) for m in metrics_b],
        dimension_comparison=dimension_comparison,
    )


# =============================================================================
# CLI
# =============================================================================

def print_fit_result(result: FitResult):
    """Pretty-print fitting results."""
    print(f"\n{'='*60}")
    print(f"FITTING RESULTS - Objective: {result.objective}")
    print(f"{'='*60}")
    
    print(f"\nBaseline: {result.baseline_profile}")
    print(f"Trials: {result.trials}")
    print(f"Search time: {result.search_time_seconds:.2f}s")
    
    print(f"\nScores:")
    print(f"  Baseline: {result.baseline_score:.2f}")
    print(f"  Fitted:   {result.fitted_score:.2f}")
    print(f"  Improvement: {result.improvement:+.2f} ({result.improvement_pct:+.1f}%)")
    
    if result.warnings:
        print(f"\n⚠️  Warnings:")
        for w in result.warnings:
            print(f"    - {w}")
    
    print(f"\nParameter changes (top 5 by magnitude):")
    sorted_diffs = sorted(result.param_diffs.items(), key=lambda x: -abs(x[1]))
    for key, diff in sorted_diffs[:5]:
        if abs(diff) > 0.001:
            print(f"  {key}: {diff:+.4f}")


def print_comparison(result: ComparisonResult):
    """Pretty-print comparison results."""
    print(f"\n{'='*60}")
    print(f"PROFILE COMPARISON - Objective: {result.objective}")
    print(f"{'='*60}")
    
    print(f"\n{result.profile_a_name} vs {result.profile_b_name}")
    print(f"\nScores:")
    print(f"  {result.profile_a_name}: {result.score_a:.2f}")
    print(f"  {result.profile_b_name}: {result.score_b:.2f}")
    print(f"  Winner: {result.winner}")
    
    print(f"\nDimension comparison:")
    for dim, values in sorted(result.dimension_comparison.items()):
        diff = values['diff']
        if abs(diff) > 0.1:
            direction = "↑" if diff > 0 else "↓"
            print(f"  {dim}: {values['profile_a']:.1f} → {values['profile_b']:.1f} ({direction} {abs(diff):.1f})")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Offline Fitter for Epistemic Governor")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Fit command
    fit_parser = subparsers.add_parser("fit", help="Fit profile to traces")
    fit_parser.add_argument("traces", help="Directory containing trace files")
    fit_parser.add_argument("--baseline", default="balanced", help="Baseline profile archetype")
    fit_parser.add_argument("--objective", default="safety", 
                          choices=["safety", "truthfulness", "ergonomics", "throughput"])
    fit_parser.add_argument("--trials", type=int, default=100, help="Number of search trials")
    fit_parser.add_argument("--output", "-o", help="Output path for fitted profile")
    fit_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Compare command
    cmp_parser = subparsers.add_parser("compare", help="Compare two profiles")
    cmp_parser.add_argument("traces", help="Directory containing trace files")
    cmp_parser.add_argument("profile_a", help="First profile (archetype name or path)")
    cmp_parser.add_argument("profile_b", help="Second profile (archetype name or path)")
    cmp_parser.add_argument("--objective", default="safety",
                          choices=["safety", "truthfulness", "ergonomics", "throughput"])
    
    # Score command
    score_parser = subparsers.add_parser("score", help="Score traces with a profile")
    score_parser.add_argument("traces", help="Directory containing trace files")
    score_parser.add_argument("--profile", default="balanced", help="Profile to use")
    score_parser.add_argument("--objective", default="safety",
                            choices=["safety", "truthfulness", "ergonomics", "throughput"])
    
    args = parser.parse_args()
    
    if args.command == "fit":
        trace_dir = Path(args.traces)
        if not trace_dir.exists():
            print(f"Error: Trace directory not found: {trace_dir}")
            return
        
        traces = list(trace_dir.glob("*.jsonl"))
        if not traces:
            print(f"Error: No .jsonl files found in {trace_dir}")
            return
        
        print(f"Found {len(traces)} trace files")
        
        baseline = get_profile(args.baseline)
        result = run_fitting(
            traces=traces,
            baseline=baseline,
            objective=args.objective,
            trials=args.trials,
            seed=args.seed,
        )
        
        print_fit_result(result)
        
        if args.output:
            fitted = Profile.from_flat_vector(result.fitted_params, base=baseline)
            fitted.name = f"{baseline.name}_fitted_{args.objective}"
            fitted.fit_metadata = {
                "baseline": args.baseline,
                "objective": args.objective,
                "improvement_pct": result.improvement_pct,
                "trials": result.trials,
                "fitted_at": datetime.now(timezone.utc).isoformat(),
            }
            fitted.save(args.output)
            print(f"\nSaved fitted profile to: {args.output}")
    
    elif args.command == "compare":
        trace_dir = Path(args.traces)
        traces = list(trace_dir.glob("*.jsonl"))
        
        # Load profiles
        def load_profile(name_or_path):
            if Path(name_or_path).exists():
                return Profile.load(name_or_path)
            return get_profile(name_or_path)
        
        profile_a = load_profile(args.profile_a)
        profile_b = load_profile(args.profile_b)
        
        result = compare_profiles(traces, profile_a, profile_b, args.objective)
        print_comparison(result)
    
    elif args.command == "score":
        trace_dir = Path(args.traces)
        traces = list(trace_dir.glob("*.jsonl"))
        
        profile = get_profile(args.profile) if not Path(args.profile).exists() else Profile.load(args.profile)
        weights = OBJECTIVE_WEIGHTS[args.objective]
        
        total_score, all_metrics = score_traces(traces, weights)
        
        print(f"\nScoring {len(traces)} traces with {profile.name} profile")
        print(f"Objective: {args.objective}")
        print(f"Total score: {total_score:.2f}")
        
        print(f"\nPer-trace breakdown:")
        for m in all_metrics:
            print(f"  {Path(m.trace_path).name}: CF={m.cf_count}, UNSTABLE={m.time_in_unstable}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
