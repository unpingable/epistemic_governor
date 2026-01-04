#!/usr/bin/env python3
"""
Epistemic Governor CLI

A debugging microscope for epistemic state.

Commands:
    dt status           Show current session state
    dt strata           View commitment layers
    dt diff <from> [to] Show changes between turns
    dt explain <id>     Explain a claim's provenance
    dt history          Show frame history
    dt thermal          Show thermal state details
    dt govern <text>    Govern text and show results
    dt repl             Interactive govern loop
    dt reset            Reset session to fresh state
    dt calibrate        Run calibration to characterize a model
    dt replay           Replay corpus through governor
    dt live             Run live LLM session with governor
    dt events           Analyze event log file
    dt device           Show device information (CUDA/MPS/CPU)
    dt registry         Show module registry state
    dt smoke            Run registry smoke test
    dt creative         Run creative regime (sandbox)
    dt help             Show this help

Live LLM Testing:
    dt live --provider ollama --model llama3     Local Ollama model
    dt live --provider openai --model gpt-4o     OpenAI API
    dt live --provider anthropic                 Anthropic API
    dt live -p openai --prompt "What is 2+2?"    Single prompt mode
    dt live -p mock --log events.jsonl -v        With event logging

Event Log Analysis:
    dt events events.jsonl                       Show all events
    dt events events.jsonl --summary             Show summary only
    dt events events.jsonl --filter commit       Filter by type
    dt events events.jsonl --tail 20             Show last 20 events

Registry & Testing:
    dt registry                     Show registered invariants
    dt registry --domain epistemic  Filter by domain
    dt smoke                        Run registry smoke test

Creative Mode:
    dt creative                     Interactive creative sandbox
    dt creative --mode dream        Dream mode (fluid identity)
    dt creative --stiffness 0.8     High identity stiffness
    dt creative --boredom 0.2       Lower boredom threshold

Calibration:
    dt calibrate --demo                      Run calibration with demo corpus
    dt calibrate --corpus data.json -m gpt4  Calibrate from corpus file
    dt calibrate --demo --fit-policy -o balanced  Fit policy after calibration

Replay:
    dt replay --demo                         Replay demo corpus with default policy
    dt replay --corpus data.json -p policy.json  Replay with custom policy
    dt replay --demo -o strict               Use strict preset

Providers:
    ollama      - Local models via Ollama (default: llama3)
    openai      - OpenAI API (requires OPENAI_API_KEY, default: gpt-4o)
    anthropic   - Anthropic API (requires ANTHROPIC_API_KEY, default: claude-sonnet-4)
    huggingface - HuggingFace transformers (requires torch)
    mock        - Deterministic mock for testing

Usage:
    python -m epistemic_governor status
    python -m epistemic_governor live --provider mock --log test.jsonl
    python -m epistemic_governor events test.jsonl --summary
    
    # Or if installed:
    dt live -p mock --log events.jsonl
    dt events events.jsonl --filter commit
"""

import sys
import argparse
import json
from typing import Optional, List
from datetime import datetime
from pathlib import Path

# Handle both package and direct imports
try:
    from .session import (
        EpistemicSession,
        create_session,
        LedgerSnapshot,
        Stratum,
        LedgerDiff,
        SessionMode,
    )
    from .kernel import EpistemicFrame, ThermalState
    from .calibrate import (
        Calibrator,
        CalibrationCorpus,
        CalibrationPrompt,
        PromptType,
        BaselineProfile,
        PolicyProfile,
        ObjectivePreset,
        ReplayHarness,
        ReplayMetrics,
        CharacterizationReport,
        create_demo_corpus,
    )
    from .providers import (
        create_provider,
        get_device,
        get_device_info,
    )
    from .providers import (
        create_provider,
        get_device,
        get_device_info,
        MockProvider,
    )
    from .events import EventLogger
except ImportError:
    from session import (
        EpistemicSession,
        create_session,
        LedgerSnapshot,
        Stratum,
        LedgerDiff,
        SessionMode,
    )
    from kernel import EpistemicFrame, ThermalState
    from calibrate import (
        Calibrator,
        CalibrationCorpus,
        CalibrationPrompt,
        PromptType,
        BaselineProfile,
        PolicyProfile,
        ObjectivePreset,
        ReplayHarness,
        ReplayMetrics,
        CharacterizationReport,
        create_demo_corpus,
    )
    from providers import (
        create_provider,
        get_device,
        get_device_info,
    )
    from providers import (
        create_provider,
        get_device,
        get_device_info,
        MockProvider,
    )
    from events import EventLogger


# =============================================================================
# ANSI Colors
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    
    @classmethod
    def disable(cls):
        """Disable colors (for non-TTY output)."""
        for attr in dir(cls):
            if not attr.startswith('_') and attr.isupper():
                setattr(cls, attr, '')


# Check if we're in a TTY
if not sys.stdout.isatty():
    Colors.disable()


def c(text: str, *colors: str) -> str:
    """Apply colors to text."""
    return ''.join(colors) + str(text) + Colors.RESET


# =============================================================================
# Output Formatting
# =============================================================================

def print_header(title: str):
    """Print a section header."""
    width = 60
    print()
    print(c("=" * width, Colors.DIM))
    print(c(f" {title}", Colors.BOLD, Colors.CYAN))
    print(c("=" * width, Colors.DIM))


def print_subheader(title: str):
    """Print a subsection header."""
    print()
    print(c(f"── {title} ", Colors.BOLD) + c("─" * (50 - len(title)), Colors.DIM))


def print_kv(key: str, value, indent: int = 0):
    """Print a key-value pair."""
    prefix = "  " * indent
    print(f"{prefix}{c(key + ':', Colors.DIM)} {value}")


def print_thermal_bar(instability: float, width: int = 30):
    """Print a visual thermal bar."""
    filled = int(instability / 1.5 * width)  # 1.5 is shutdown threshold
    filled = min(filled, width)
    
    if instability >= 1.5:
        color = Colors.BG_RED
    elif instability >= 0.7:
        color = Colors.RED
    elif instability >= 0.3:
        color = Colors.YELLOW
    else:
        color = Colors.GREEN
    
    bar = "█" * filled + "░" * (width - filled)
    markers = "│" + " " * 5 + "│" + " " * 7 + "│" + " " * 14 + "│"
    
    print(f"  [{c(bar, color)}] {instability:.3f}")
    print(c(f"  {markers}", Colors.DIM))
    print(c("   0    0.3     0.7            1.5", Colors.DIM))
    print(c("   ↑     ↑       ↑              ↑", Colors.DIM))
    print(c("  ok   warn  critical      shutdown", Colors.DIM))


def format_regime(regime: str) -> str:
    """Format regime with color."""
    colors = {
        'normal': Colors.GREEN,
        'furnace': (Colors.BG_YELLOW, Colors.BLACK),  # Looks stable but burning inside
        'warning': Colors.YELLOW,
        'critical': Colors.RED,
        'shutdown': (Colors.BG_RED, Colors.WHITE),
    }
    color = colors.get(regime, Colors.WHITE)
    if isinstance(color, tuple):
        return c(regime.upper(), *color)
    return c(regime, color)


def format_layer(layer: str) -> str:
    """Format stratum layer with color."""
    colors = {
        'ACTIVE': Colors.GREEN,
        'SUPERSEDED': Colors.YELLOW,
        'ARCHIVED': Colors.DIM,
    }
    return c(layer, colors.get(layer, Colors.WHITE))


# =============================================================================
# CLI Commands
# =============================================================================

# Global session (persists across commands in REPL)
_session: Optional[EpistemicSession] = None


def get_session() -> EpistemicSession:
    """Get or create the global session."""
    global _session
    if _session is None:
        _session = create_session(mode="normal", enable_valve=True)
    return _session


def cmd_status(args):
    """Show current session state."""
    session = get_session()
    snap = session.snapshot()
    
    print_header("EPISTEMIC STATUS")
    
    print_subheader("Session")
    print_kv("Turn", snap.turn)
    print_kv("Mode", session.mode.value)
    print_kv("Valve", "enabled" if session.enable_valve else "disabled")
    
    print_subheader("Claims")
    print_kv("Total", snap.total_claims)
    print_kv("Active", c(snap.active_claims, Colors.GREEN))
    print_kv("Superseded", c(snap.superseded_claims, Colors.YELLOW))
    print_kv("Archived", c(snap.archived_claims, Colors.DIM))
    
    print_subheader("Thermal")
    print_kv("Regime", format_regime(snap.regime))
    print_kv("Instability", f"{snap.instability:.3f}")
    print_thermal_bar(snap.instability)
    print()
    print_kv("Revisions", snap.revision_count)
    print_kv("Contradictions", snap.contradiction_count)
    
    if snap.recent_commits or snap.recent_blocks:
        print_subheader("Recent Activity")
        if snap.recent_commits:
            print_kv("Committed", ', '.join(snap.recent_commits[:5]))
        if snap.recent_blocks:
            print_kv("Blocked", ', '.join(snap.recent_blocks[:5]))
    
    print()


def cmd_strata(args):
    """View commitment layers."""
    session = get_session()
    strata = session.strata(limit=args.limit)
    
    print_header("EPISTEMIC STRATA")
    
    if not strata:
        print(c("  (empty ledger)", Colors.DIM))
        print()
        return
    
    # Group by layer
    by_layer = {'ACTIVE': [], 'SUPERSEDED': [], 'ARCHIVED': []}
    for s in strata:
        by_layer.get(s.layer, []).append(s)
    
    for layer in ['ACTIVE', 'SUPERSEDED', 'ARCHIVED']:
        items = by_layer[layer]
        if not items:
            continue
        
        print_subheader(f"{layer} ({len(items)})")
        
        for s in items:
            # Truncate text
            text = s.text[:60] + "..." if len(s.text) > 60 else s.text
            
            print(f"  {c(s.claim_id, Colors.CYAN)}")
            print(f"    {text}")
            
            if layer == 'ACTIVE':
                conf_color = Colors.GREEN if s.confidence > 0.7 else Colors.YELLOW
                print(f"    {c('confidence:', Colors.DIM)} {c(f'{s.confidence:.2f}', conf_color)}  "
                      f"{c('type:', Colors.DIM)} {s.claim_type}")
            
            if s.supersedes:
                print(f"    {c('supersedes:', Colors.DIM)} {s.supersedes}")
            
            print()
    
    print()


def cmd_diff(args):
    """Show changes between turns."""
    session = get_session()
    
    from_turn = args.from_turn
    to_turn = args.to_turn if args.to_turn is not None else session.turn
    
    diff = session.diff(from_turn, to_turn)
    
    print_header(f"DIFF: Turn {from_turn} → {to_turn}")
    
    if diff.is_empty:
        print(c("  (no changes)", Colors.DIM))
    else:
        if diff.added:
            print_subheader(f"Added ({len(diff.added)})")
            for cid in diff.added:
                claim = session.get_claim(cid)
                if claim:
                    text = claim.text[:50] + "..." if len(claim.text) > 50 else claim.text
                    print(f"  {c('+', Colors.GREEN)} {c(cid, Colors.CYAN)}: {text}")
        
        if diff.superseded:
            print_subheader(f"Superseded ({len(diff.superseded)})")
            for cid in diff.superseded:
                print(f"  {c('~', Colors.YELLOW)} {cid}")
        
        if diff.archived:
            print_subheader(f"Archived ({len(diff.archived)})")
            for cid in diff.archived:
                print(f"  {c('-', Colors.DIM)} {cid}")
    
    print()
    print_kv("Thermal Δ", f"{diff.thermal_delta:+.3f}")
    print()


def cmd_explain(args):
    """Explain a claim's provenance."""
    session = get_session()
    explanation = session.explain(args.claim_id)
    
    print_header(f"EXPLAIN: {args.claim_id}")
    
    if 'error' in explanation:
        print(c(f"  {explanation['error']}", Colors.RED))
        print()
        return
    
    print_subheader("Claim")
    print_kv("ID", explanation['id'])
    print_kv("Text", explanation['text'])
    print_kv("Type", explanation['type'])
    print_kv("Confidence", f"{explanation['confidence']:.2f}")
    
    print_subheader("Status")
    status = explanation['status']
    status_color = {'ACTIVE': Colors.GREEN, 'SUPERSEDED': Colors.YELLOW, 'ARCHIVED': Colors.DIM}
    print_kv("Status", c(status, status_color.get(status, Colors.WHITE)))
    print_kv("Committed", explanation['committed_at'])
    
    if explanation['supersedes']:
        print_kv("Supersedes", explanation['supersedes'])
    
    if explanation['support_refs']:
        print_subheader("Support References")
        for ref in explanation['support_refs']:
            print(f"  • {ref}")
    
    if explanation['revision_chain']:
        print_subheader("Revision Chain")
        for i, cid in enumerate(explanation['revision_chain']):
            arrow = "└─" if i == len(explanation['revision_chain']) - 1 else "├─"
            print(f"  {arrow} {cid}")
    
    print()


def cmd_history(args):
    """Show frame history."""
    session = get_session()
    history = session.history
    
    print_header("FRAME HISTORY")
    
    if not history:
        print(c("  (no history)", Colors.DIM))
        print()
        return
    
    for i, frame in enumerate(history[-args.limit:]):
        turn = len(history) - args.limit + i
        if turn < 0:
            turn = i
        
        print_subheader(f"Turn {turn}")
        print_kv("Status", frame.status.name if hasattr(frame.status, 'name') else str(frame.status))
        print_kv("Committed", len(frame.committed))
        print_kv("Blocked", len(frame.blocked))
        print_kv("Revisions", len(frame.revision_required))
        print_kv("Thermal", f"{frame.thermal.instability:.3f} ({frame.thermal.regime})")
        
        if frame.errors:
            print_kv("Errors", ', '.join(frame.errors[:2]))
    
    print()


def cmd_thermal(args):
    """Show thermal state details."""
    session = get_session()
    thermal = session.thermal
    
    print_header("THERMAL STATE")
    
    print_subheader("Current")
    print_kv("Instability", f"{thermal.instability:.4f}")
    print_kv("Regime", format_regime(thermal.regime))
    print()
    print_thermal_bar(thermal.instability, width=40)
    
    print_subheader("Compensation Effort (Latent Heat)")
    print_kv("Hedges", thermal.hedge_count)
    print_kv("Blocks", thermal.block_count)
    print_kv("Retries", thermal.retry_count)
    print_kv("Clarifications", thermal.clarification_count)
    print_kv("Total Effort", f"{thermal.compensation_effort:.2f}")
    
    # Furnace detection
    if thermal.instability > 0:
        print_kv("Furnace Ratio", f"{thermal.furnace_ratio:.2f}")
    if thermal.is_furnace:
        print()
        print(c("  ⚠ FURNACE DETECTED: High compensation effort, low visible instability", Colors.YELLOW))
        print(c("    System is burning energy to look stable. Consider reducing load.", Colors.DIM))
    
    print_subheader("Counters")
    print_kv("Total Commitments", thermal.total_commitments)
    print_kv("Revisions", thermal.revision_count)
    print_kv("Contradictions", thermal.contradiction_count)
    
    print_subheader("Thresholds")
    print_kv("Warning", f"{thermal.warning_threshold:.2f}")
    print_kv("Critical", f"{thermal.critical_threshold:.2f}")
    print_kv("Shutdown", f"{thermal.shutdown_threshold:.2f}")
    print_kv("Furnace Ratio", f"{thermal.furnace_ratio_threshold:.2f}")
    
    print_subheader("Heat Costs")
    print_kv("Revision", f"{thermal.revision_base_heat:.2f}")
    print_kv("Contradiction", f"{thermal.contradiction_heat:.2f}")
    print_kv("Cosplay", f"{thermal.cosplay_heat:.2f}")
    
    print()


def cmd_govern(args):
    """Govern text and show results."""
    session = get_session()
    text = ' '.join(args.text)
    
    print_header("GOVERN")
    print()
    print(c("Input:", Colors.DIM))
    print(f"  {text}")
    print()
    
    frame = session.govern(text)
    
    print(c("Output:", Colors.DIM))
    print(f"  {frame.output_text}")
    print()
    
    print_subheader("Results")
    print_kv("Status", frame.status.name if hasattr(frame.status, 'name') else str(frame.status))
    print_kv("Committed", len(frame.committed))
    print_kv("Blocked", len(frame.blocked))
    print_kv("Thermal Δ", f"{frame.thermal_delta:+.3f}")
    
    if frame.committed:
        print()
        print(c("Committed claims:", Colors.DIM))
        for claim in frame.committed:
            text = claim.text[:50] + "..." if len(claim.text) > 50 else claim.text
            print(f"  {c('+', Colors.GREEN)} [{claim.claim_type.name}] {text}")
    
    if frame.blocked:
        print()
        print(c("Blocked proposals:", Colors.DIM))
        for prop, decision in frame.blocked:
            text = prop.text[:50] + "..." if len(prop.text) > 50 else prop.text
            print(f"  {c('✗', Colors.RED)} {text}")
            print(f"    {c(decision.reason, Colors.DIM)}")
    
    if frame.errors:
        print()
        print(c("Errors:", Colors.RED))
        for err in frame.errors:
            print(f"  {err}")
    
    print()


def cmd_reset(args):
    """Reset session to fresh state."""
    global _session
    
    if not args.force:
        print(c("This will reset all session state. Use --force to confirm.", Colors.YELLOW))
        return
    
    _session = None
    get_session()  # Creates fresh session
    
    print(c("✓ Session reset", Colors.GREEN))
    print()


def cmd_repl(args):
    """Interactive govern loop."""
    session = get_session()
    
    print_header("EPISTEMIC REPL")
    print()
    print("Commands:")
    print(f"  {c('.status', Colors.CYAN)}   - Show session status")
    print(f"  {c('.strata', Colors.CYAN)}   - View commitment layers")
    print(f"  {c('.thermal', Colors.CYAN)}  - Show thermal state")
    print(f"  {c('.reset', Colors.CYAN)}    - Reset session")
    print(f"  {c('.quit', Colors.CYAN)}     - Exit REPL")
    print()
    print("Enter text to govern, or a command starting with '.'")
    print()
    
    while True:
        try:
            line = input(c(f"[T{session.turn}] ", Colors.DIM) + c("dt> ", Colors.CYAN))
        except (EOFError, KeyboardInterrupt):
            print()
            break
        
        line = line.strip()
        if not line:
            continue
        
        # Handle commands
        if line.startswith('.'):
            cmd = line[1:].lower().split()[0]
            
            if cmd in ('quit', 'exit', 'q'):
                break
            elif cmd == 'status':
                cmd_status(argparse.Namespace())
            elif cmd == 'strata':
                cmd_strata(argparse.Namespace(limit=10))
            elif cmd == 'thermal':
                cmd_thermal(argparse.Namespace())
            elif cmd == 'reset':
                cmd_reset(argparse.Namespace(force=True))
            elif cmd == 'help':
                print("Commands: .status .strata .thermal .reset .quit")
            else:
                print(c(f"Unknown command: {cmd}", Colors.RED))
            continue
        
        # Govern the input
        frame = session.govern(line)
        
        # Show compact results
        regime_str = format_regime(frame.thermal.regime)
        print(f"  {c('→', Colors.DIM)} {len(frame.committed)} committed, "
              f"{len(frame.blocked)} blocked, "
              f"Δt={frame.thermal_delta:+.3f} [{regime_str}]")
        
        if frame.committed:
            for claim in frame.committed[:3]:
                text = claim.text[:40] + "..." if len(claim.text) > 40 else claim.text
                print(f"    {c('+', Colors.GREEN)} {text}")
        
        if frame.blocked:
            for prop, _ in frame.blocked[:2]:
                text = prop.text[:40] + "..." if len(prop.text) > 40 else prop.text
                print(f"    {c('✗', Colors.RED)} {text}")
        
        print()
    
    print(c("Goodbye!", Colors.DIM))


# =============================================================================
# Calibration Commands
# =============================================================================

def cmd_calibrate(args):
    """
    Run calibration to characterize a model.
    
    Usage:
        dt calibrate --model "claude-3-sonnet" --corpus corpus.json
        dt calibrate --model "demo" --demo
        dt calibrate --compare profile1.json profile2.json
    """
    # Handle profile comparison mode
    if hasattr(args, 'compare') and args.compare:
        try:
            from .calibrate import BaselineProfile, compare_profiles
        except ImportError:
            from calibrate import BaselineProfile, compare_profiles
        
        path1, path2 = args.compare
        profile1 = BaselineProfile.load(Path(path1))
        profile2 = BaselineProfile.load(Path(path2))
        print(compare_profiles(profile1, profile2))
        return
    
    print()
    print(c("=" * 60, Colors.CYAN))
    print(c(" EPISTEMIC CALIBRATION", Colors.BOLD))
    print(c("=" * 60, Colors.CYAN))
    print()
    
    model_id = args.model or "unknown-model"
    
    # Load or create corpus
    if args.demo:
        corpus = create_demo_corpus()
        print(f"Using demo corpus with {len(corpus.prompts)} prompts")
    elif args.corpus:
        corpus_path = Path(args.corpus)
        if not corpus_path.exists():
            print(c(f"Error: Corpus file not found: {args.corpus}", Colors.RED))
            return
        corpus = CalibrationCorpus.load(corpus_path)
        print(f"Loaded corpus from {args.corpus} with {len(corpus.prompts)} prompts")
    else:
        print(c("Error: Must specify --corpus or --demo", Colors.RED))
        return
    
    # Show corpus breakdown
    print(f"\n{c('Corpus breakdown:', Colors.DIM)}")
    for pt in PromptType:
        count = len(corpus.by_type(pt))
        if count > 0:
            print(f"  {pt.value}: {count}")
    
    # Run calibration
    print(f"\n{c('Running system identification...', Colors.YELLOW)}")
    calibrator = Calibrator(model_id=model_id)
    baseline = calibrator.run_sysid(corpus)
    
    # Generate and print report
    report = CharacterizationReport(model_id=model_id, profile=baseline)
    print(report.generate())
    
    # Fit policy if requested
    if args.fit_policy:
        objective = ObjectivePreset[args.objective.upper()] if args.objective else ObjectivePreset.BALANCED
        print(f"\n{c(f'Fitting policy ({objective.name})...', Colors.YELLOW)}")
        policy = calibrator.fit_policy(baseline, objective, corpus, n_iterations=args.iterations)
        print(f"  Max confidence: {policy.global_policy.max_confidence:.2f}")
        print(f"  Hedge z-score: {policy.global_policy.hedge_delta_t.z_score}")
        print(f"  Fitting loss: {policy.fitting_loss:.4f}")
        
        # Save policy
        if args.output:
            output_path = Path(args.output)
            policy.save(output_path)
            print(f"\n{c('Policy saved to:', Colors.GREEN)} {output_path}")
    
    # Save baseline
    if args.save_profile:
        profile_path = Path(args.save_profile)
        baseline.save(profile_path)
        print(f"\n{c('Profile saved to:', Colors.GREEN)} {profile_path}")
    
    print()


def cmd_replay(args):
    """
    Replay a corpus through the governor to measure performance.
    
    Usage:
        dt replay --corpus corpus.json --policy policy.json
        dt replay --demo --objective balanced
    """
    print()
    print(c("=" * 60, Colors.CYAN))
    print(c(" EPISTEMIC REPLAY", Colors.BOLD))
    print(c("=" * 60, Colors.CYAN))
    print()
    
    # Load corpus
    if args.demo:
        corpus = create_demo_corpus()
        print(f"Using demo corpus with {len(corpus.prompts)} prompts")
    elif args.corpus:
        corpus_path = Path(args.corpus)
        if not corpus_path.exists():
            print(c(f"Error: Corpus file not found: {args.corpus}", Colors.RED))
            return
        corpus = CalibrationCorpus.load(corpus_path)
        print(f"Loaded corpus from {args.corpus}")
    else:
        print(c("Error: Must specify --corpus or --demo", Colors.RED))
        return
    
    # Load or create policy
    if args.policy:
        policy_path = Path(args.policy)
        if not policy_path.exists():
            print(c(f"Error: Policy file not found: {args.policy}", Colors.RED))
            return
        policy = PolicyProfile.load(policy_path)
        print(f"Loaded policy from {args.policy}")
    else:
        # Create default policy based on objective
        objective = ObjectivePreset[args.objective.upper()] if args.objective else ObjectivePreset.BALANCED
        print(f"Using {objective.name} preset policy")
        policy = PolicyProfile(
            model_id="default",
            baseline_profile_hash="none",
        )
    
    # Run replay
    print(f"\n{c('Running replay...', Colors.YELLOW)}")
    harness = ReplayHarness()
    metrics = harness.replay_corpus(corpus, policy)
    
    # Print results
    print(f"\n{c('Results:', Colors.GREEN)}")
    print(f"  Hallucination rate: {metrics.hallucination_rate:.1%}")
    print(f"  False refusal rate: {metrics.false_refusal_rate:.1%}")
    print(f"  Precision: {metrics.precision:.3f}")
    print(f"  Recall: {metrics.recall:.3f}")
    print(f"  F1 Score: {metrics.f1:.3f}")
    print(f"  Total revision cost: {metrics.total_revision_cost:.2f}")
    print(f"  Avg thermal delta: {metrics.avg_thermal_delta:.3f}")
    
    # Interpretation
    print(f"\n{c('Interpretation:', Colors.DIM)}")
    if metrics.hallucination_rate > 0.3:
        print(f"  {c('⚠', Colors.YELLOW)} High hallucination rate - consider stricter policy")
    if metrics.false_refusal_rate > 0.2:
        print(f"  {c('⚠', Colors.YELLOW)} High false refusal rate - consider looser policy")
    if metrics.f1 > 0.8:
        print(f"  {c('✓', Colors.GREEN)} Good balance between precision and recall")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump({
                'hallucination_rate': metrics.hallucination_rate,
                'false_refusal_rate': metrics.false_refusal_rate,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1': metrics.f1,
                'total_revision_cost': metrics.total_revision_cost,
                'avg_thermal_delta': metrics.avg_thermal_delta,
            }, f, indent=2)
        print(f"\n{c('Results saved to:', Colors.GREEN)} {output_path}")
    
    print()


def cmd_live(args):
    """
    Run live session with actual LLM through the governor.
    
    Usage:
        dt live --provider ollama --model llama3
        dt live --provider openai --model gpt-4o
        dt live --provider anthropic --prompt "What is 2+2?"
        dt live --provider mock --log events.jsonl
    """
    # Import InstrumentedSession
    try:
        from .session import InstrumentedSession
    except ImportError:
        from session import InstrumentedSession
    
    print()
    print(c("=" * 60, Colors.CYAN))
    print(c(" EPISTEMIC GOVERNOR - LIVE SESSION", Colors.BOLD))
    print(c("=" * 60, Colors.CYAN))
    print()
    
    # Create provider
    provider_type = args.provider
    model_name = args.model
    
    try:
        provider = create_provider(provider_type, model_name)
        print(f"Provider: {c(provider.get_model_id(), Colors.GREEN)}")
    except ValueError as e:
        print(c(f"Error creating provider: {e}", Colors.RED))
        return
    except Exception as e:
        print(c(f"Error: {e}", Colors.RED))
        return
    
    # Create instrumented session (with optional file logging)
    log_path = getattr(args, 'log', None)
    if log_path:
        print(f"Event log: {c(log_path, Colors.DIM)}")
    
    session = InstrumentedSession(
        provider=provider,
        event_log=log_path,
    )
    print(f"Session ID: {c(session.session_id, Colors.DIM)}")
    print()
    
    if args.prompt:
        # Single prompt mode
        print(c("User:", Colors.BLUE), args.prompt)
        print()
        
        try:
            frame = session.step(args.prompt)
            
            # Show response
            if frame.output_text:
                print(c("Response:", Colors.GREEN), frame.output_text[:500])
                if len(frame.output_text) > 500:
                    print("  [truncated...]")
            
            # Show governor state if verbose
            if args.verbose:
                print()
                print(c("Governor State:", Colors.YELLOW))
                print(f"  Thermal: {frame.thermal.regime}")
                print(f"  Instability: {frame.thermal.instability:.3f}")
                
                if frame.committed:
                    print(f"  Committed: {len(frame.committed)}")
                    for claim in frame.committed[:3]:
                        text = claim.text[:50] + "..." if len(claim.text) > 50 else claim.text
                        print(f"    ✓ [{claim.confidence:.0%}] {text}")
                
                if frame.hedged:
                    print(f"  Hedged: {len(frame.hedged)}")
                
                if frame.blocked:
                    print(f"  Blocked: {len(frame.blocked)}")
                    for prop, decision in frame.blocked[:3]:
                        text = prop.text[:40] + "..." if len(prop.text) > 40 else prop.text
                        print(f"    ✗ {text}")
                        print(f"      {c(decision.action.name, Colors.DIM)}")
                
                # Show event summary
                events = session.get_events()
                print(f"\n  Events emitted: {len(events)}")
        
        except Exception as e:
            print(c(f"Error during generation: {e}", Colors.RED))
            import traceback
            traceback.print_exc()
    
    else:
        # Interactive mode
        print("Interactive mode. Type 'quit' to exit, 'stats' for session stats, 'events' to see events.")
        print()
        
        while True:
            try:
                user_input = input(c("You: ", Colors.BLUE))
            except (EOFError, KeyboardInterrupt):
                print()
                break
            
            if not user_input.strip():
                continue
            
            if user_input.lower() in ('quit', 'exit', 'q'):
                break
            
            if user_input.lower() == 'stats':
                print()
                print(c("Session Stats:", Colors.YELLOW))
                summary = session.get_summary()
                for k, v in summary.items():
                    print(f"  {k}: {v}")
                print()
                continue
            
            if user_input.lower() == 'events':
                print()
                events = session.get_events()
                print(c(f"Events ({len(events)}):", Colors.YELLOW))
                for e in events[-10:]:  # Last 10
                    print(f"  {e.event_type.value}: turn={getattr(e, 'turn_id', '?')}")
                print()
                continue
            
            print()
            try:
                frame = session.step(user_input)
                
                # Show response
                if frame.output_text:
                    print(c("Response:", Colors.GREEN))
                    print(frame.output_text)
                    print()
                
                # Show brief status
                thermal_color = {
                    'normal': Colors.BLUE,
                    'warning': Colors.YELLOW,
                    'critical': Colors.RED,
                    'shutdown': Colors.BG_RED,
                }.get(frame.thermal.regime, Colors.WHITE)
                
                status_parts = [
                    f"[{c(frame.thermal.regime.upper(), thermal_color)}]",
                ]
                
                if frame.committed:
                    status_parts.append(f"✓{len(frame.committed)}")
                if frame.hedged:
                    status_parts.append(f"~{len(frame.hedged)}")
                if frame.blocked:
                    status_parts.append(f"✗{len(frame.blocked)}")
                
                print(c(" ".join(status_parts), Colors.DIM))
                print()
                
            except Exception as e:
                print(c(f"Error: {e}", Colors.RED))
                print()
    
    # Close session (flushes logs)
    session.close()
    
    if log_path:
        print(c(f"Events written to: {log_path}", Colors.GREEN))
    print(c("Session ended.", Colors.DIM))
    print()


def cmd_events(args):
    """
    Analyze an event log file.
    
    Usage:
        dt events events.jsonl
        dt events events.jsonl --summary
        dt events events.jsonl --filter commit
        dt events events.jsonl --tail 20
    """
    # Import event functions
    try:
        from .events import load_events, analyze_events, EventType
    except ImportError:
        from events import load_events, analyze_events, EventType
    
    log_path = Path(args.logfile)
    if not log_path.exists():
        print(c(f"Error: File not found: {args.logfile}", Colors.RED))
        return
    
    print()
    print(c("=" * 60, Colors.CYAN))
    print(c(" EVENT LOG ANALYSIS", Colors.BOLD))
    print(c("=" * 60, Colors.CYAN))
    print(f"File: {log_path}")
    print()
    
    # Load events
    try:
        events = load_events(log_path)
    except Exception as e:
        print(c(f"Error loading events: {e}", Colors.RED))
        return
    
    print(f"Loaded {len(events)} events")
    
    # Filter if requested
    if args.filter:
        filter_type = EventType(args.filter)
        events = [e for e in events if e.event_type == filter_type]
        print(f"Filtered to {len(events)} {args.filter} events")
    
    # Tail if requested
    if args.tail:
        events = events[-args.tail:]
        print(f"Showing last {len(events)} events")
    
    print()
    
    if args.summary or not events:
        # Just show summary
        all_events = load_events(log_path)  # Reload for full analysis
        analysis = analyze_events(all_events)
        
        print(c("Summary:", Colors.YELLOW))
        print(f"  Total events: {analysis['event_counts']['total']}")
        print(f"  Turns: {analysis['event_counts']['turns']}")
        print(f"  Decisions: {analysis['event_counts']['decisions']}")
        print(f"  Thermal snapshots: {analysis['event_counts']['thermal']}")
        
        if 'latency' in analysis:
            print(f"\n  Latency (ms):")
            print(f"    Mean: {analysis['latency']['mean']:.1f}")
            print(f"    Min: {analysis['latency']['min']:.1f}")
            print(f"    Max: {analysis['latency']['max']:.1f}")
        
        if 'totals' in analysis:
            print(f"\n  Totals:")
            print(f"    Proposals: {analysis['totals']['proposals']}")
            print(f"    Commits: {analysis['totals']['commits']}")
            print(f"    Hedges: {analysis['totals']['hedges']}")
            print(f"    Blocks: {analysis['totals']['blocks']}")
        
        if 'thermal' in analysis:
            print(f"\n  Thermal:")
            print(f"    Mean instability: {analysis['thermal']['mean_instability']:.3f}")
            print(f"    Max instability: {analysis['thermal']['max_instability']:.3f}")
            print(f"    Final instability: {analysis['thermal']['final_instability']:.3f}")
        
        if 'drift' in analysis:
            print(f"\n  Drift:")
            print(f"    Tests: {analysis['drift']['tests']}")
            print(f"    Flips: {analysis['drift']['flips']}")
            print(f"    Rate: {analysis['drift']['drift_rate']:.1%}")
    
    else:
        # Show individual events
        print(c("Events:", Colors.YELLOW))
        for e in events:
            turn_id = getattr(e, 'turn_id', '?')
            
            if e.event_type.value == 'turn':
                prompt = getattr(e, 'prompt', '')[:40]
                latency = getattr(e, 'latency_ms', 0)
                commits = getattr(e, 'commits_count', 0)
                print(f"  {c('TURN', Colors.GREEN)} #{turn_id}: \"{prompt}...\" ({latency:.0f}ms, {commits} commits)")
            
            elif e.event_type.value == 'commit':
                text = getattr(e, 'claim_text', '')[:50]
                conf = getattr(e, 'final_confidence', 0)
                print(f"  {c('COMMIT', Colors.BLUE)} #{turn_id}: [{conf:.0%}] {text}...")
            
            elif e.event_type.value == 'thermal':
                instab = getattr(e, 'instability_score', 0)
                regime = getattr(e, 'regime', 'unknown')
                print(f"  {c('THERMAL', Colors.YELLOW)} #{turn_id}: {regime} (instab={instab:.3f})")
            
            elif e.event_type.value == 'decision':
                action = getattr(e, 'action', 'unknown')
                reason = getattr(e, 'reason', '')[:40]
                color = Colors.GREEN if action == 'accept' else Colors.RED
                print(f"  {c('DECISION', color)} #{turn_id}: {action} - {reason}...")
            
            elif e.event_type.value == 'drift':
                did_flip = getattr(e, 'did_flip', False)
                status = c('FLIPPED', Colors.RED) if did_flip else c('HELD', Colors.GREEN)
                print(f"  {c('DRIFT', Colors.MAGENTA)} #{turn_id}: {status}")
            
            else:
                print(f"  {e.event_type.value} #{turn_id}")
    
    print()


def cmd_help(args):
    """Show help."""
    print(__doc__)


def cmd_device(args):
    """Show device information."""
    print()
    print(c("=" * 60, Colors.CYAN))
    print(c(" DEVICE INFORMATION", Colors.BOLD))
    print(c("=" * 60, Colors.CYAN))
    print()
    
    info = get_device_info()
    
    print(f"Selected device: {c(info['selected'], Colors.GREEN)}")
    print()
    
    print("CUDA (NVIDIA):")
    if info['cuda_available']:
        print(f"  {c('✓', Colors.GREEN)} Available")
        print(f"    Devices: {info.get('cuda_device_count', 0)}")
        if info.get('cuda_device_name'):
            print(f"    Name: {info['cuda_device_name']}")
    else:
        print(f"  {c('✗', Colors.RED)} Not available")
    
    print()
    print("MPS (Apple Silicon):")
    if info['mps_available']:
        print(f"  {c('✓', Colors.GREEN)} Available")
    else:
        print(f"  {c('✗', Colors.RED)} Not available")
    
    print()
    print("Recommended provider:")
    if info['cuda_available']:
        print("  HuggingFaceProvider with load_in_8bit=True or load_in_4bit=True")
    elif info['mps_available']:
        print("  HuggingFaceProvider (float16 on MPS)")
    else:
        print("  OllamaProvider (runs models efficiently on CPU)")
    
    print()


def cmd_creative(args):
    """Run creative regime exploration."""
    try:
        from .creative import CreativeRegime, CreativeConfig, CreativeMode
    except ImportError:
        from creative import CreativeRegime, CreativeConfig, CreativeMode
    
    mode_map = {
        'explore': CreativeMode.EXPLORE,
        'draft': CreativeMode.DRAFT,
        'dream': CreativeMode.DREAM,
        'comedy': CreativeMode.COMEDY,
    }
    
    config = CreativeConfig(
        mode=mode_map.get(args.mode, CreativeMode.EXPLORE),
        identity_stiffness=args.stiffness,
        boredom_threshold=args.boredom,
        boredom_enabled=True,
    )
    
    regime = CreativeRegime(config)
    
    print(f"""
============================================================
 CREATIVE REGIME - {config.mode.name}
============================================================

Sandbox mode: nothing commits to epistemic history.
Contradictions allowed. Boredom tracks repetition.

Identity stiffness: {config.identity_stiffness}
Boredom threshold: {config.boredom_threshold}

Commands:
  Type text to explore
  'stats' - show regime statistics
  'quit'/'exit'/'q' - exit

============================================================
""")
    
    while True:
        try:
            user_input = input("\n[creative] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting creative mode.")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("Exiting creative mode.")
            break
        
        if user_input.lower() == 'stats':
            summary = regime.get_summary()
            print("\n--- Creative Regime Stats ---")
            for k, v in summary.items():
                print(f"  {k}: {v}")
            continue
        
        # Extract simple motifs (words > 4 chars)
        words = user_input.lower().split()
        motifs = [w for w in words if len(w) > 4 and w.isalpha()][:5]
        
        result = regime.explore(user_input, motifs=motifs)
        
        print(f"\n  Proposal: {result['proposal_id']}")
        print(f"  Boredom: {result['boredom']:.3f} (Δ{result['boredom_delta']:.3f})")
        if result['motif_saturations']:
            sats = [f"{m}={s:.0%}" for m, s in result['motif_saturations'].items()]
            print(f"  Saturations: {', '.join(sats)}")
        
        regime.advance_turn()


def cmd_registry(args):
    """Show module registry state."""
    try:
        from .registry import create_registry, Domain
        from .epistemic_module import register_epistemic_invariants
    except ImportError:
        from registry import create_registry, Domain
        from epistemic_module import register_epistemic_invariants
    
    registry = create_registry()
    register_epistemic_invariants(registry)
    
    print("\n=== Module Registry ===\n")
    
    # Filter by domain if specified
    domain_filter = None
    if args.domain:
        try:
            domain_filter = Domain[args.domain.upper()]
        except KeyError:
            print(f"Unknown domain: {args.domain}")
            print(f"Valid domains: {[d.name for d in Domain]}")
            return
    
    invariants = registry.list_invariants(
        domain=domain_filter,
        enabled_only=not args.all,
    )
    
    # Group by domain
    by_domain = {}
    for spec in invariants:
        d = spec.domain.value
        if d not in by_domain:
            by_domain[d] = []
        by_domain[d].append(spec)
    
    for domain in sorted(by_domain.keys()):
        specs = by_domain[domain]
        print(f"[{domain.upper()}]")
        for spec in specs:
            status = "✓" if spec.enabled else "✗"
            print(f"  {status} {spec.name} (priority={spec.priority})")
            if spec.description:
                print(f"      {spec.description}")
        print()
    
    # Stats
    stats = registry.get_stats()
    print("--- Stats ---")
    print(f"  Total: {stats['total_invariants']}")
    print(f"  Enabled: {stats['enabled_invariants']}")


def cmd_smoke(args):
    """Run registry smoke test."""
    import subprocess
    import os
    
    # Run the smoke test script
    smoke_path = Path(__file__).parent / "smoke_registry.py"
    if smoke_path.exists():
        result = subprocess.run(
            [sys.executable, str(smoke_path)],
            cwd=str(smoke_path.parent.parent),
            env={**os.environ, "PYTHONPATH": str(smoke_path.parent.parent)},
        )
        sys.exit(result.returncode)
    else:
        print("Smoke test not found. Running inline...")
        
        try:
            from .registry import create_registry, ProposalEnvelope, StateView, Domain, AuditStatus
            from .epistemic_module import register_epistemic_invariants, EpistemicConfig
        except ImportError:
            from registry import create_registry, ProposalEnvelope, StateView, Domain, AuditStatus
            from epistemic_module import register_epistemic_invariants, EpistemicConfig
        
        print("\n=== Registry Smoke Test ===\n")
        
        registry = create_registry()
        register_epistemic_invariants(registry, EpistemicConfig())
        
        print(f"✓ Registry created with {len(registry.list_invariants())} invariants")
        
        # Quick test
        proposal = ProposalEnvelope(
            proposal_id="smoke_test",
            t=1,
            timestamp=datetime.now(),
            origin="test",
            origin_type="test",
            domain=Domain.EPISTEMIC,
            confidence=0.95,
            payload={"claim_type": "FACTUAL"},
        )
        state = StateView(current_t=0)
        
        report = registry.audit(proposal, state)
        print(f"✓ Audit completed: {report.status.name}")
        if report.applied_clamps:
            print(f"  Clamps applied: {report.applied_clamps}")
        
        print("\n✓ Smoke test passed")


def cmd_eval(args):
    """Run evaluation on corpus."""
    try:
        from .evaluation import (
            EvaluationCorpus,
            ExtractionEvaluator,
            create_demo_corpus,
            run_evaluation,
        )
        from .extractor import CommitmentExtractor
    except ImportError:
        from evaluation import (
            EvaluationCorpus,
            ExtractionEvaluator,
            create_demo_corpus,
            run_evaluation,
        )
        from extractor import CommitmentExtractor
    
    # Load or create corpus
    if args.demo:
        corpus = create_demo_corpus()
        print(f"Using demo corpus: {corpus.name}")
    elif args.corpus:
        corpus = EvaluationCorpus.load(args.corpus)
        print(f"Loaded corpus: {corpus.name}")
    else:
        print("Error: Specify --corpus or --demo")
        return
    
    print(f"Samples: {corpus.stats['sample_count']}")
    print(f"Total claims: {corpus.stats['total_claims']}")
    print()
    
    # Run evaluation
    extractor = CommitmentExtractor() if args.extraction else None
    
    results = run_evaluation(
        corpus=corpus,
        extractor=extractor,
        output_path=args.output,
    )
    
    if not args.extraction:
        print("Corpus loaded successfully.")
        print("Use --extraction to run extraction evaluation.")


def cmd_annotate(args):
    """Interactive annotation helper for building corpus."""
    try:
        from .evaluation import (
            EvaluationCorpus,
            AnnotatedSample,
            AnnotatedClaim,
            AnnotatedClaimType,
        )
    except ImportError:
        from evaluation import (
            EvaluationCorpus,
            AnnotatedSample,
            AnnotatedClaim,
            AnnotatedClaimType,
        )
    
    # Load or create corpus
    output_path = Path(args.output)
    if args.append and output_path.exists():
        corpus = EvaluationCorpus.load(str(output_path))
        print(f"Appending to existing corpus: {corpus.name}")
    else:
        corpus = EvaluationCorpus(
            name=output_path.stem,
            version="0.1.0",
            description="Annotated evaluation corpus",
        )
        print(f"Creating new corpus: {corpus.name}")
    
    print("""
============================================================
 ANNOTATION HELPER
============================================================

This tool helps you annotate LLM outputs with ground truth claims.

For each response, you'll:
1. Enter the prompt and response
2. Mark each claim with span, type, and confidence
3. Note any contradictions

Commands:
  'done' - finish current sample
  'save' - save corpus and exit
  'quit' - exit without saving
  'skip' - skip current sample

============================================================
""")
    
    sample_count = len(corpus.samples)
    
    while True:
        try:
            print(f"\n--- Sample {sample_count + 1} ---")
            
            # Get prompt
            prompt = input("Prompt (or 'save'/'quit'): ").strip()
            if prompt.lower() == 'save':
                corpus.save(str(output_path))
                print(f"Saved {len(corpus.samples)} samples to {output_path}")
                break
            if prompt.lower() == 'quit':
                print("Exiting without saving.")
                break
            if not prompt:
                continue
            
            # Get response
            print("Response (end with empty line):")
            response_lines = []
            while True:
                line = input()
                if not line:
                    break
                response_lines.append(line)
            response = "\n".join(response_lines)
            
            if not response:
                print("Skipping empty response.")
                continue
            
            # Show response with character positions
            print("\nResponse with positions:")
            for i, char in enumerate(response):
                if i % 50 == 0:
                    print(f"\n[{i:4d}] ", end="")
                print(char, end="")
            print(f"\n[{len(response):4d}] END")
            
            # Collect claims
            claims = []
            claim_num = 1
            
            while True:
                print(f"\n  Claim {claim_num} (or 'done' to finish sample):")
                
                text = input("    Text: ").strip()
                if text.lower() == 'done':
                    break
                if text.lower() == 'skip':
                    claims = None
                    break
                
                try:
                    span_start = int(input("    Span start: ").strip())
                    span_end = int(input("    Span end: ").strip())
                except ValueError:
                    print("    Invalid span, skipping claim.")
                    continue
                
                print(f"    Types: {[t.value for t in AnnotatedClaimType]}")
                claim_type_str = input("    Type: ").strip().lower()
                try:
                    claim_type = AnnotatedClaimType(claim_type_str)
                except ValueError:
                    claim_type = AnnotatedClaimType.FACTUAL
                    print(f"    Unknown type, using: {claim_type.value}")
                
                try:
                    expressed_conf = float(input("    Expressed confidence (0-1): ").strip())
                except ValueError:
                    expressed_conf = 0.5
                
                claim = AnnotatedClaim(
                    id=f"sample_{sample_count + 1}_c{claim_num}",
                    text=text,
                    span_start=span_start,
                    span_end=span_end,
                    claim_type=claim_type,
                    expressed_confidence=expressed_conf,
                    annotator_confidence=0.9,
                )
                claims.append(claim)
                claim_num += 1
            
            if claims is None:
                print("Skipped sample.")
                continue
            
            # Create sample
            sample = AnnotatedSample(
                id=f"sample_{sample_count + 1:04d}",
                prompt=prompt,
                response=response,
                claims=claims,
            )
            corpus.add_sample(sample)
            sample_count += 1
            print(f"Added sample with {len(claims)} claims.")
            
        except (EOFError, KeyboardInterrupt):
            print("\n\nInterrupted.")
            save = input("Save before exit? (y/n): ").strip().lower()
            if save == 'y':
                corpus.save(str(output_path))
                print(f"Saved {len(corpus.samples)} samples to {output_path}")
            break
    
    print(f"Final corpus: {len(corpus.samples)} samples")


def cmd_validate(args):
    """Run validation phases."""
    try:
        from .validation import ValidationRunner
    except ImportError:
        from validation import ValidationRunner
    
    runner = ValidationRunner(
        provider=args.provider,
        model=args.model,
        output_dir=args.output,
        verbose=True,
    )
    
    if args.phase == "smoke":
        success = runner.run_smoke_test()
        sys.exit(0 if success else 1)
    elif args.phase == "baseline":
        runner.run_baseline()
    elif args.phase == "detect":
        runner.run_detection(annotate=args.annotate)
    elif args.phase == "ground_truth":
        runner.run_ground_truth()
    elif args.phase == "valve":
        runner.run_valve_comparison()
    elif args.phase == "thermal":
        runner.run_thermal_stress()
    elif args.phase == "shadow":
        runner.run_shadow_audit()
    elif args.phase == "stress":
        runner.run_stress_test()
    elif args.phase == "all":
        runner.run_all()


def cmd_shear(args):
    """Analyze commitment shear (ΔR)."""
    try:
        from .session import create_session
    except ImportError:
        from session import create_session
    
    print()
    print(c("=" * 60, Colors.CYAN))
    print(c(" ΔR SHEAR ANALYSIS", Colors.BOLD))
    print(c("=" * 60, Colors.CYAN))
    print()
    
    if args.demo:
        # Demo with sample text
        source_text = """
        The system MUST authenticate all users before granting access.
        The system MUST NOT store passwords in plaintext.
        Users SHOULD enable two-factor authentication.
        All data MUST be encrypted in transit.
        The system SHALL log all access attempts.
        """
        
        target_text = """
        Users need to log in to access the system.
        Passwords are stored securely.
        Two-factor authentication is available.
        Data is protected during transmission.
        """
        
        print(f"{c('Source text:', Colors.DIM)}")
        print(source_text.strip())
        print()
        print(f"{c('Target text (after summarization):', Colors.DIM)}")
        print(target_text.strip())
        print()
        
    else:
        if not args.source or not args.target:
            print(c("Error: Specify --source and --target, or use --demo", Colors.RED))
            return
        
        with open(args.source) as f:
            source_text = f.read()
        with open(args.target) as f:
            target_text = f.read()
    
    # Create session and analyze
    session = create_session(enable_shear=True)
    
    # Set baseline from source
    n_commitments = session.set_commitment_baseline(source_text)
    print(f"Extracted {c(str(n_commitments), Colors.GREEN)} commitments from source")
    
    if n_commitments == 0:
        print(c("Warning: No commitments found. Try text with MUST/SHALL/SHOULD statements.", Colors.YELLOW))
        return
    
    # Check shear
    report = session.check_shear(target_text, args.transform if not args.demo else "summarize")
    
    # Display report
    print()
    print(c("=== Shear Report ===", Colors.BOLD))
    print(f"Transform: {report['transform']}")
    print(f"Source commitments: {report['source_count']}")
    print(f"Target commitments: {report['target_count']}")
    print()
    
    # Shear metric (the important number)
    shear_pct = report['shear'] * 100
    if report['violation']:
        shear_color = Colors.RED
        shear_status = "VIOLATION"
    elif report['warning']:
        shear_color = Colors.YELLOW
        shear_status = "WARNING"
    else:
        shear_color = Colors.GREEN
        shear_status = "OK"
    
    print(f"Shear: {c(f'{shear_pct:.1f}%', shear_color)} [{shear_status}]")
    print(f"Torque (weakenings): {report['torque']}")
    print(f"Spurious injection: {report['spurious_injection']:.1%}")
    print()
    
    # Breakdown
    if report['preserved']:
        print(f"{c('✓ Preserved:', Colors.GREEN)} {', '.join(report['preserved'])}")
    if report['weakened']:
        print(f"{c('⚠ Weakened:', Colors.YELLOW)} {', '.join(report['weakened'])}")
    if report['dropped']:
        print(f"{c('✗ Dropped:', Colors.RED)} {', '.join(report['dropped'])}")
    if report['contradicted']:
        print(f"{c('⛔ Contradicted:', Colors.RED)} {', '.join(report['contradicted'])}")
    
    print()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog='dt',
        description='Epistemic Governor CLI - debugging microscope for epistemic state',
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # status
    p_status = subparsers.add_parser('status', help='Show current session state')
    p_status.set_defaults(func=cmd_status)
    
    # strata
    p_strata = subparsers.add_parser('strata', help='View commitment layers')
    p_strata.add_argument('--limit', '-n', type=int, default=20, help='Max entries to show')
    p_strata.set_defaults(func=cmd_strata)
    
    # diff
    p_diff = subparsers.add_parser('diff', help='Show changes between turns')
    p_diff.add_argument('from_turn', type=int, help='Starting turn')
    p_diff.add_argument('to_turn', type=int, nargs='?', help='Ending turn (default: current)')
    p_diff.set_defaults(func=cmd_diff)
    
    # explain
    p_explain = subparsers.add_parser('explain', help='Explain a claim\'s provenance')
    p_explain.add_argument('claim_id', help='Claim ID to explain')
    p_explain.set_defaults(func=cmd_explain)
    
    # history
    p_history = subparsers.add_parser('history', help='Show frame history')
    p_history.add_argument('--limit', '-n', type=int, default=10, help='Max entries to show')
    p_history.set_defaults(func=cmd_history)
    
    # thermal
    p_thermal = subparsers.add_parser('thermal', help='Show thermal state details')
    p_thermal.set_defaults(func=cmd_thermal)
    
    # govern
    p_govern = subparsers.add_parser('govern', help='Govern text and show results')
    p_govern.add_argument('text', nargs='+', help='Text to govern')
    p_govern.set_defaults(func=cmd_govern)
    
    # reset
    p_reset = subparsers.add_parser('reset', help='Reset session to fresh state')
    p_reset.add_argument('--force', '-f', action='store_true', help='Force reset without confirmation')
    p_reset.set_defaults(func=cmd_reset)
    
    # repl
    p_repl = subparsers.add_parser('repl', help='Interactive govern loop')
    p_repl.set_defaults(func=cmd_repl)
    
    # calibrate
    p_calibrate = subparsers.add_parser('calibrate', help='Run calibration to characterize a model')
    p_calibrate.add_argument('--model', '-m', help='Model ID')
    p_calibrate.add_argument('--corpus', '-c', help='Path to calibration corpus JSON')
    p_calibrate.add_argument('--demo', action='store_true', help='Use demo corpus')
    p_calibrate.add_argument('--fit-policy', action='store_true', help='Also fit a policy')
    p_calibrate.add_argument('--objective', '-o', choices=['strict', 'balanced', 'permissive'], 
                            default='balanced', help='Policy objective preset')
    p_calibrate.add_argument('--iterations', '-i', type=int, default=20, help='Fitting iterations')
    p_calibrate.add_argument('--output', help='Path to save fitted policy')
    p_calibrate.add_argument('--save-profile', help='Path to save baseline profile')
    p_calibrate.add_argument('--compare', nargs=2, metavar=('PROFILE1', 'PROFILE2'),
                            help='Compare two baseline profiles')
    p_calibrate.add_argument('--provider', '-p', 
                            choices=['ollama', 'openai', 'anthropic', 'mock'],
                            help='LLM provider for live calibration')
    p_calibrate.set_defaults(func=cmd_calibrate)
    
    # replay
    p_replay = subparsers.add_parser('replay', help='Replay corpus through governor')
    p_replay.add_argument('--corpus', '-c', help='Path to calibration corpus JSON')
    p_replay.add_argument('--demo', action='store_true', help='Use demo corpus')
    p_replay.add_argument('--policy', '-p', help='Path to policy JSON')
    p_replay.add_argument('--objective', '-o', choices=['strict', 'balanced', 'permissive'],
                         default='balanced', help='Policy preset if no policy file')
    p_replay.add_argument('--output', help='Path to save results JSON')
    p_replay.set_defaults(func=cmd_replay)
    
    # help
    p_help = subparsers.add_parser('help', help='Show help')
    p_help.set_defaults(func=cmd_help)
    
    # device
    p_device = subparsers.add_parser('device', help='Show device information')
    p_device.set_defaults(func=cmd_device)
    
    # live
    p_live = subparsers.add_parser('live', help='Run live LLM session with governor')
    p_live.add_argument('--provider', '-p', 
                       choices=['ollama', 'openai', 'anthropic', 'huggingface', 'mock'],
                       default='ollama', help='LLM provider')
    p_live.add_argument('--model', '-m', help='Model name (default varies by provider)')
    p_live.add_argument('--prompt', help='Single prompt to run (non-interactive)')
    p_live.add_argument('--verbose', '-v', action='store_true', help='Show detailed governor output')
    p_live.add_argument('--log', '-l', help='Path to save event log (JSONL)')
    p_live.set_defaults(func=cmd_live)
    
    # events
    p_events = subparsers.add_parser('events', help='Analyze event log file')
    p_events.add_argument('logfile', help='Path to JSONL event log')
    p_events.add_argument('--summary', '-s', action='store_true', help='Show summary only')
    p_events.add_argument('--filter', '-f', choices=['turn', 'commit', 'thermal', 'drift', 'decision'],
                         help='Filter by event type')
    p_events.add_argument('--tail', '-t', type=int, help='Show only last N events')
    p_events.set_defaults(func=cmd_events)
    
    # creative - creative regime exploration
    p_creative = subparsers.add_parser('creative', help='Run creative regime (sandbox)')
    p_creative.add_argument('--mode', '-m', choices=['explore', 'draft', 'dream', 'comedy'],
                           default='explore', help='Creative mode')
    p_creative.add_argument('--stiffness', '-s', type=float, default=0.3,
                           help='Identity stiffness (0.0-1.0)')
    p_creative.add_argument('--boredom', '-b', type=float, default=0.3,
                           help='Boredom threshold')
    p_creative.set_defaults(func=cmd_creative)
    
    # registry - show registry state
    p_registry = subparsers.add_parser('registry', help='Show module registry state')
    p_registry.add_argument('--all', '-a', action='store_true', help='Show all invariants including disabled')
    p_registry.add_argument('--domain', '-d', help='Filter by domain')
    p_registry.set_defaults(func=cmd_registry)
    
    # smoke - run smoke test
    p_smoke = subparsers.add_parser('smoke', help='Run registry smoke test')
    p_smoke.set_defaults(func=cmd_smoke)
    
    # eval - run evaluation
    p_eval = subparsers.add_parser('eval', help='Run evaluation on corpus')
    p_eval.add_argument('--corpus', '-c', help='Path to evaluation corpus JSON')
    p_eval.add_argument('--demo', action='store_true', help='Use demo corpus')
    p_eval.add_argument('--output', '-o', help='Path to save results')
    p_eval.add_argument('--extraction', '-e', action='store_true', help='Run extraction evaluation')
    p_eval.set_defaults(func=cmd_eval)
    
    # annotate - helper for building corpus
    p_annotate = subparsers.add_parser('annotate', help='Interactive annotation helper')
    p_annotate.add_argument('--input', '-i', help='Input file with prompts/responses')
    p_annotate.add_argument('--output', '-o', required=True, help='Output corpus file')
    p_annotate.add_argument('--append', '-a', action='store_true', help='Append to existing corpus')
    p_annotate.set_defaults(func=cmd_annotate)
    
    # validate - run validation phases
    p_validate = subparsers.add_parser('validate', help='Run validation phases')
    p_validate.add_argument('phase', choices=['smoke', 'baseline', 'detect', 'ground_truth', 'valve', 'thermal', 'shadow', 'stress', 'all'],
                           help='Validation phase to run')
    p_validate.add_argument('--provider', '-p', default='mock',
                           choices=['ollama', 'openai', 'anthropic', 'mock'],
                           help='LLM provider')
    p_validate.add_argument('--model', '-m', default='llama3:8b', help='Model name')
    p_validate.add_argument('--output', '-o', default='validation_results', help='Output directory')
    p_validate.add_argument('--annotate', '-a', action='store_true', 
                           help='Enable interactive annotation for detection phase')
    p_validate.set_defaults(func=cmd_validate)
    
    # shear - analyze commitment shear
    p_shear = subparsers.add_parser('shear', help='Analyze commitment shear (ΔR)')
    p_shear.add_argument('--source', '-s', help='Source file with original commitments')
    p_shear.add_argument('--target', '-t', help='Target file after transform')
    p_shear.add_argument('--transform', default='summarize',
                        choices=['summarize', 'translate', 'formalize', 'compress', 'paraphrase', 'other'],
                        help='Transform type applied')
    p_shear.add_argument('--demo', action='store_true', help='Run demo with sample text')
    p_shear.set_defaults(func=cmd_shear)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == '__main__':
    main()
