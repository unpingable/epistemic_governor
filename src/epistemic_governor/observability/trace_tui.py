"""
Trace TUI - The "Cyberpunk Console" for Epistemic Governor

Replays JSONL telemetry to visualize system thermodynamics and control effort.

Usage:
    python -m epistemic_governor.observability.trace_tui traces/run_001.jsonl
    python -m epistemic_governor.observability.trace_tui traces/run_001.jsonl --speed 2.0
    python -m epistemic_governor.observability.trace_tui --demo

Design:
    - Input: JSONL trace (from trace_collector or OTel export)
    - Output: Rich TUI with Phase Space, Regime Gauge, Energy Trace
    - Policy: Read-only. Telemetry describes; nothing commits.

The dashboard visualizes the "struggle" of the system to maintain coherence:
    - Phase Space: λ (contradiction arrival) vs μ (resolution rate)
    - Regime Gauge: ELASTIC → WARM → DUCTILE → UNSTABLE
    - Energy Trace: Rolling E(S) over time
    - Event Log: Recent interventions and decisions
"""

import argparse
import json
import time
import math
import sys
from collections import deque
from dataclasses import dataclass
from typing import Dict, Any, Deque, List, Optional
from pathlib import Path

try:
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.align import Align
    from rich.table import Table
    from rich.text import Text
    from rich.console import Console
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich library not installed. Install with: pip install rich")


# =============================================================================
# Phase Space Plot
# =============================================================================

class PhaseSpacePlot:
    """
    Visualizes stability condition: E[λ_open] ≤ E[μ_close].
    
    X-axis: Arrival Rate (λ) - contradictions/claims opened
    Y-axis: Service Rate (μ) - contradictions/claims resolved
    
    The dot position shows current system state:
    - Below diagonal (green): Stable - resolving faster than opening
    - Above diagonal (red): Unstable - opening faster than resolving
    """
    
    def __init__(self, history_len: int = 30):
        self.history: Deque[tuple] = deque(maxlen=history_len)
        self.width = 40
        self.height = 15
    
    def update(self, lambda_val: float, mu_val: float):
        """Add a new data point."""
        self.history.append((lambda_val, mu_val))
    
    def render(self) -> Panel:
        """Render the phase space plot."""
        # Create grid
        grid = [[" " for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw stability boundary (diagonal y=x)
        for i in range(min(self.width, self.height)):
            r = (self.height - 1) - i
            c = int(i * self.width / self.height)
            if 0 <= r < self.height and 0 <= c < self.width:
                grid[r][c] = "[dim]·[/]"
        
        # Draw axes labels
        # Y-axis (μ - resolution rate)
        for r in range(self.height):
            grid[r][0] = "[dim]│[/]"
        # X-axis (λ - arrival rate)  
        for c in range(self.width):
            grid[self.height - 1][c] = "[dim]─[/]"
        grid[self.height - 1][0] = "[dim]└[/]"
        
        # Plot history trail
        if self.history:
            max_lambda = max(max(h[0] for h in self.history), 1)
            max_mu = max(max(h[1] for h in self.history), 1)
            scale_x = (self.width - 2) / max_lambda
            scale_y = (self.height - 2) / max_mu
            
            for i, (l, m) in enumerate(self.history):
                x = int(l * scale_x) + 1
                y = int(m * scale_y)
                r = (self.height - 2) - y
                
                # Clamp to grid
                r = max(0, min(self.height - 2, r))
                x = max(1, min(self.width - 1, x))
                
                # Color: red if unstable (λ > μ), green if stable
                color = "red" if l > m else "green"
                opacity = "dim" if i < len(self.history) - 5 else "bold"
                char = "●" if i == len(self.history) - 1 else "·"
                
                grid[r][x] = f"[{opacity} {color}]{char}[/]"
        
        # Render to string
        rows = ["".join(row) for row in grid]
        chart = "\n".join(rows)
        
        return Panel(
            chart,
            title="[bold cyan]Phase Space[/] [dim](λ vs μ)[/]",
            subtitle="[dim]▲μ resolve  →λ arrive | [green]●stable[/] [red]●unstable[/][/]",
            border_style="cyan",
        )


# =============================================================================
# Regime Gauge
# =============================================================================

class RegimeGauge:
    """
    Visualizes current operational regime and budget pressure.
    
    Regimes (from OPERATING_ENVELOPE.md):
    - ELASTIC: Normal operation, system breathing
    - WARM: Elevated load, variety attenuator active
    - DUCTILE: High stress, context resets likely
    - UNSTABLE: Critical, tripwires may fire
    """
    
    COLORS = {
        "ELASTIC": "green",
        "HEALTHY_LATTICE": "green",
        "WARM": "yellow",
        "DUCTILE": "orange1",
        "UNSTABLE": "red",
        "GLASS_OSSIFICATION": "magenta",
        "BUDGET_STARVATION": "blue",
        "FREEZE": "cyan",
    }
    
    ICONS = {
        "ELASTIC": "🟢",
        "WARM": "🟡", 
        "DUCTILE": "🟠",
        "UNSTABLE": "🔴",
    }
    
    def render(
        self, 
        regime: str, 
        stress: float, 
        energy: float,
        tool_gain: float = 0.5,
        hysteresis: float = 0.0,
    ) -> Panel:
        """Render the regime gauge."""
        color = self.COLORS.get(regime, "white")
        icon = self.ICONS.get(regime, "⚪")
        
        # Budget pressure bar
        bar_width = 25
        filled = int(stress * bar_width)
        bar = f"[{color}]" + "█" * filled + "[dim]░[/]" * (bar_width - filled) + "[/]"
        
        # Tool gain indicator
        gain_color = "green" if tool_gain < 0.7 else "yellow" if tool_gain < 1.0 else "red"
        
        # Energy color
        e_color = "green" if energy < 10 else "yellow" if energy < 50 else "red"
        
        # Build content table
        content = Table.grid(padding=(0, 1))
        content.add_column(justify="right", style="bold")
        content.add_column(justify="left")
        
        content.add_row("Regime:", f"[bold {color}]{icon} {regime}[/]")
        content.add_row("Pressure:", f"{bar} {stress:.0%}")
        content.add_row("Energy:", f"[{e_color}]{energy:.1f}[/]")
        content.add_row("Tool Gain:", f"[{gain_color}]{tool_gain:.2f}[/]")
        content.add_row("Hysteresis:", f"[dim]{hysteresis:.2f}[/]")
        
        return Panel(
            Align.center(content, vertical="middle"),
            title="[bold]System Vitals[/]",
            border_style=color,
        )


# =============================================================================
# Energy Sparkline
# =============================================================================

class EnergySparkline:
    """
    Rolling sparkline of E(S) - epistemic energy function.
    
    Energy = f(open_contradictions, budget_pressure)
    Higher energy = more unresolved state = more stress
    """
    
    def __init__(self, width: int = 60):
        self.history: Deque[float] = deque(maxlen=width)
        self.width = width
    
    def update(self, energy: float):
        self.history.append(energy)
    
    def render(self) -> Panel:
        """Render the sparkline."""
        if not self.history:
            return Panel("[dim]No data[/]", title="Energy E(S)")
        
        # Normalize to 0-1 range
        max_e = max(max(self.history), 1)
        normalized = [e / max_e for e in self.history]
        
        # Convert to spark characters
        chars = " ▁▂▃▄▅▆▇█"
        spark = ""
        for v in normalized:
            idx = int(v * (len(chars) - 1))
            # Color based on value
            if v < 0.3:
                spark += f"[green]{chars[idx]}[/]"
            elif v < 0.6:
                spark += f"[yellow]{chars[idx]}[/]"
            else:
                spark += f"[red]{chars[idx]}[/]"
        
        # Stats
        current = self.history[-1] if self.history else 0
        avg = sum(self.history) / len(self.history)
        
        content = f"{spark}\n[dim]Current: {current:.1f} | Avg: {avg:.1f} | Max: {max_e:.1f}[/]"
        
        return Panel(
            Align.center(content),
            title="[bold]Energy Trace[/] [dim]E(S)[/]",
            border_style="blue",
        )


# =============================================================================
# Event Log
# =============================================================================

class EventLog:
    """Scrolling log of recent interventions and decisions."""
    
    def __init__(self, max_entries: int = 10):
        self.entries: Deque[str] = deque(maxlen=max_entries)
    
    def add(self, verdict: str, turn_id: str, details: str = ""):
        """Add a log entry."""
        if verdict == "COMMIT":
            symbol = "[bold green]✓[/]"
        elif verdict == "QUARANTINE":
            symbol = "[bold yellow]⚠[/]"
        elif verdict == "REJECT":
            symbol = "[bold red]✗[/]"
        elif verdict == "FORBIDDEN":
            symbol = "[bold magenta]⛔[/]"
        elif verdict == "RESET":
            symbol = "[bold cyan]↺[/]"
        elif verdict == "TRIPWIRE":
            symbol = "[bold red]🚨[/]"
        else:
            symbol = "[dim]·[/]"
        
        detail_str = f" [dim]{details}[/]" if details else ""
        self.entries.append(f"{symbol} [dim]T{turn_id}:[/] {verdict}{detail_str}")
    
    def render(self) -> Panel:
        """Render the event log."""
        if not self.entries:
            content = "[dim]No events yet[/]"
        else:
            content = "\n".join(self.entries)
        
        return Panel(
            content,
            title="[bold]Event Log[/]",
            border_style="white",
        )


# =============================================================================
# Contradiction Counter
# =============================================================================

class ContradictionPanel:
    """Shows open contradiction count and resolution rate."""
    
    def __init__(self):
        self.c_open = 0
        self.c_history: Deque[int] = deque(maxlen=20)
    
    def update(self, c_open: int, c_opened: int = 0, c_closed: int = 0):
        self.c_open = c_open
        self.c_history.append(c_open)
    
    def render(self) -> Panel:
        """Render contradiction status."""
        # Mini sparkline
        if self.c_history:
            max_c = max(max(self.c_history), 1)
            chars = "▁▂▃▄▅▆▇█"
            spark = ""
            for c in self.c_history:
                idx = int((c / max_c) * (len(chars) - 1))
                spark += chars[idx]
        else:
            spark = "─" * 20
        
        # Color based on count
        if self.c_open == 0:
            color = "green"
            status = "Clear"
        elif self.c_open < 5:
            color = "yellow"
            status = "Active"
        else:
            color = "red"
            status = "Backlog"
        
        content = f"[bold {color}]{self.c_open}[/] open ({status})\n[dim]{spark}[/]"
        
        return Panel(
            Align.center(content),
            title="[bold]Contradictions[/]",
            border_style=color,
        )


# =============================================================================
# Main TUI Runner
# =============================================================================

def run_tui(trace_path: str, speed: float = 1.0):
    """Run the TUI with trace playback."""
    if not RICH_AVAILABLE:
        print("Error: Rich library required. Install with: pip install rich")
        return
    
    console = Console()
    
    # Initialize widgets
    phase_plot = PhaseSpacePlot()
    regime_gauge = RegimeGauge()
    energy_spark = EnergySparkline()
    event_log = EventLog()
    contradiction_panel = ContradictionPanel()
    
    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(name="top", size=12),
        Layout(name="middle", size=6),
        Layout(name="bottom", ratio=1),
    )
    layout["top"].split_row(
        Layout(name="gauge", ratio=1),
        Layout(name="phase", ratio=1),
    )
    layout["middle"].split_row(
        Layout(name="energy", ratio=2),
        Layout(name="contradictions", ratio=1),
    )
    
    # Load trace
    trace_path = Path(trace_path)
    if not trace_path.exists():
        console.print(f"[red]Error: Trace file not found: {trace_path}[/]")
        return
    
    with open(trace_path, 'r') as f:
        lines = f.readlines()
    
    console.print(f"[cyan]Loading trace: {trace_path} ({len(lines)} records)[/]")
    console.print("[dim]Press Ctrl+C to stop[/]\n")
    time.sleep(1)
    
    # Playback
    with Live(layout, refresh_per_second=10, screen=True, console=console) as live:
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
                
                # Extract fields (support both flat and nested formats)
                attrs = record.get("attributes", record)
                signals = record.get("signals", attrs)
                
                # Core metrics
                regime = attrs.get("epistemic.regime", 
                         attrs.get("label",
                         signals.get("regime", "UNKNOWN")))
                
                energy = float(attrs.get("epistemic.energy", 
                              signals.get("energy", 0)))
                
                stress = float(attrs.get("epistemic.budget.stress",
                              signals.get("budget_pressure", 0)))
                
                tool_gain = float(signals.get("tool_gain_estimate", 
                                 attrs.get("tool_gain", 0.5)))
                
                hysteresis = float(signals.get("hysteresis_magnitude",
                                  attrs.get("hysteresis", 0)))
                
                # Phase space metrics
                lambda_val = float(attrs.get("epistemic.contradictions.opened",
                                  signals.get("contradiction_open_rate", 0)))
                mu_val = float(attrs.get("epistemic.contradictions.closed",
                              signals.get("contradiction_close_rate", 0)))
                
                c_open = int(attrs.get("epistemic.contradictions.open",
                            attrs.get("c_open", 0)))
                
                # Event info
                turn_id = str(attrs.get("epistemic.turn.id", 
                             attrs.get("turn", i)))
                
                events = attrs.get("events", {})
                verdict = "OK"
                details = ""
                
                if events.get("reset"):
                    verdict = "RESET"
                    details = events.get("reset_type", "")
                elif events.get("tripwire"):
                    verdict = "TRIPWIRE"
                    details = events.get("tripwire_type", "")
                elif attrs.get("epistemic.enforcement.verdict"):
                    verdict = attrs["epistemic.enforcement.verdict"]
                
                # Update widgets
                phase_plot.update(lambda_val * 10 + 1, mu_val * 10 + 1)  # Scale for visibility
                energy_spark.update(energy)
                contradiction_panel.update(c_open)
                
                if verdict != "OK":
                    event_log.add(verdict, turn_id, details)
                
                # Update layout
                layout["top"]["gauge"].update(
                    regime_gauge.render(regime, stress, energy, tool_gain, hysteresis)
                )
                layout["top"]["phase"].update(phase_plot.render())
                layout["middle"]["energy"].update(energy_spark.render())
                layout["middle"]["contradictions"].update(contradiction_panel.render())
                layout["bottom"].update(event_log.render())
                
                # Playback timing
                time.sleep(0.5 / speed)
                
            except json.JSONDecodeError as e:
                event_log.add("ERROR", str(i), f"JSON parse error")
            except KeyboardInterrupt:
                break
            except Exception as e:
                event_log.add("ERROR", str(i), str(e)[:30])
    
    console.print("\n[cyan]Playback complete.[/]")


def generate_demo_trace(output_path: str = "demo_trace.jsonl"):
    """Generate a demo trace for testing the TUI."""
    import random
    
    records = []
    c_open = 0
    regime = "ELASTIC"
    
    # Simulate 50 turns with varying conditions
    for turn in range(50):
        # Simulate regime transitions
        if turn == 15:
            regime = "WARM"
        elif turn == 25:
            regime = "DUCTILE"
        elif turn == 35:
            regime = "UNSTABLE"
        elif turn == 40:
            regime = "WARM"
        elif turn == 45:
            regime = "ELASTIC"
        
        # Simulate metrics
        lambda_rate = random.uniform(0.1, 0.5)
        if regime in ["DUCTILE", "UNSTABLE"]:
            lambda_rate += 0.3
        
        mu_rate = random.uniform(0.1, 0.4)
        if regime == "UNSTABLE":
            mu_rate *= 0.5
        
        c_open = max(0, c_open + int(lambda_rate * 3) - int(mu_rate * 3))
        
        energy = c_open * 1.5 + random.uniform(0, 5)
        stress = min(1.0, (turn % 20) / 20.0 + (0.3 if regime == "UNSTABLE" else 0))
        
        events = {}
        if turn == 35:
            events["tripwire"] = True
            events["tripwire_type"] = "cascade"
        if turn == 40:
            events["reset"] = True
            events["reset_type"] = "CONTEXT"
        
        record = {
            "turn": turn,
            "label": regime,
            "signals": {
                "regime": regime,
                "energy": energy,
                "budget_pressure": stress,
                "tool_gain_estimate": 0.3 + (0.4 if regime in ["WARM", "DUCTILE"] else 0) + (0.3 if regime == "UNSTABLE" else 0),
                "hysteresis_magnitude": random.uniform(0.1, 0.4),
                "contradiction_open_rate": lambda_rate,
                "contradiction_close_rate": mu_rate,
            },
            "c_open": c_open,
            "events": events,
        }
        records.append(record)
    
    # Write trace
    with open(output_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    
    print(f"Generated demo trace: {output_path} ({len(records)} records)")
    return output_path


def print_trace_stats(trace_path: str):
    """Print statistics about a trace file."""
    path = Path(trace_path)
    if not path.exists():
        print(f"Error: File not found: {trace_path}")
        return
    
    records = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    
    if not records:
        print("No valid records found")
        return
    
    # Count regimes
    regimes = {}
    for r in records:
        regime = r.get("label", r.get("signals", {}).get("regime", "UNKNOWN"))
        regimes[regime] = regimes.get(regime, 0) + 1
    
    print(f"\nTrace: {trace_path}")
    print(f"Records: {len(records)}")
    print("\nRegime distribution:")
    for regime, count in sorted(regimes.items(), key=lambda x: -x[1]):
        pct = count / len(records) * 100
        print(f"  {regime}: {count} ({pct:.1f}%)")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Epistemic Governor Trace Viewer - The Cyberpunk Console",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Play back a trace file
    python -m epistemic_governor.observability.trace_tui traces/run_001.jsonl
    
    # Play at 2x speed
    python -m epistemic_governor.observability.trace_tui traces/run_001.jsonl --speed 2.0
    
    # Generate and play demo trace
    python -m epistemic_governor.observability.trace_tui --demo
    
    # Print trace statistics
    python -m epistemic_governor.observability.trace_tui --stats traces/run_001.jsonl
        """
    )
    
    parser.add_argument("trace", nargs="?", help="Path to JSONL trace file")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--demo", action="store_true", help="Generate and play demo trace")
    parser.add_argument("--stats", action="store_true", help="Print trace statistics only")
    
    args = parser.parse_args()
    
    if args.demo:
        trace_path = generate_demo_trace()
        if RICH_AVAILABLE:
            run_tui(trace_path, args.speed)
    elif args.stats and args.trace:
        print_trace_stats(args.trace)
    elif args.trace:
        run_tui(args.trace, args.speed)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
