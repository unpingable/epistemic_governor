"""
Query Layer (BLI-QUERY-0.1)

SQL-like query interface for diagnostic events.

"Prometheus tells us THAT something happened.
 SQL tells us WHAT, WHY, and IN WHAT ORDER."

This is the human-centered observability layer.
Relational. Temporal. Causal. Narrative.

Uses DuckDB for fast analytical queries over JSONL event logs.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    print("Warning: DuckDB not installed. Query layer limited.")


# =============================================================================
# Query Engine
# =============================================================================

class QueryEngine:
    """
    SQL query engine for diagnostic events.
    
    Loads JSONL events into DuckDB for fast analytical queries.
    """
    
    def __init__(self, jsonl_path: Optional[str] = None):
        if not HAS_DUCKDB:
            raise RuntimeError("DuckDB required for query layer")
        
        self.conn = duckdb.connect(":memory:")
        self.events_loaded = 0
        
        if jsonl_path:
            self.load_events(jsonl_path)
    
    def load_events(self, jsonl_path: str) -> int:
        """Load events from JSONL file."""
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Event log not found: {jsonl_path}")
        
        # Read and parse JSONL
        events = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
        
        if not events:
            return 0
        
        # Create table from events
        self.conn.execute("DROP TABLE IF EXISTS events")
        
        # Build CREATE TABLE from first event
        sample = events[0]
        columns = []
        for key, value in sample.items():
            if isinstance(value, bool):
                columns.append(f"{key} BOOLEAN")
            elif isinstance(value, int):
                columns.append(f"{key} BIGINT")
            elif isinstance(value, float):
                columns.append(f"{key} DOUBLE")
            elif isinstance(value, dict):
                columns.append(f"{key} JSON")
            elif isinstance(value, list):
                columns.append(f"{key} JSON")
            else:
                columns.append(f"{key} VARCHAR")
        
        create_sql = f"CREATE TABLE events ({', '.join(columns)})"
        self.conn.execute(create_sql)
        
        # Insert events
        for event in events:
            placeholders = ", ".join(["?" for _ in event])
            values = []
            for key in sample.keys():
                v = event.get(key)
                if isinstance(v, (dict, list)):
                    values.append(json.dumps(v))
                else:
                    values.append(v)
            
            self.conn.execute(
                f"INSERT INTO events VALUES ({placeholders})",
                values
            )
        
        self.events_loaded = len(events)
        
        # Create indexes
        self.conn.execute("CREATE INDEX idx_run_id ON events(run_id)")
        self.conn.execute("CREATE INDEX idx_verdict ON events(verdict)")
        self.conn.execute("CREATE INDEX idx_turn_id ON events(turn_id)")
        
        return self.events_loaded
    
    def query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute SQL query and return results as list of dicts."""
        result = self.conn.execute(sql).fetchall()
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]
    
    def query_df(self, sql: str):
        """Execute SQL query and return as DataFrame (if pandas available)."""
        return self.conn.execute(sql).fetchdf()


# =============================================================================
# Canonical Queries
# =============================================================================

class CanonicalQueries:
    """
    Pre-defined queries for common diagnostic questions.
    
    These are part of the spec - they define what questions
    the system must be able to answer about itself.
    """
    
    def __init__(self, engine: QueryEngine):
        self.engine = engine
    
    def regime_summary(self) -> List[Dict[str, Any]]:
        """Get regime indicators by run."""
        return self.engine.query("""
            SELECT 
                run_id,
                COUNT(*) as turns,
                SUM(CASE WHEN verdict = 'OK' THEN 1 ELSE 0 END) as ok_count,
                SUM(CASE WHEN verdict = 'WARN' THEN 1 ELSE 0 END) as warn_count,
                SUM(CASE WHEN verdict = 'BLOCK' THEN 1 ELSE 0 END) as block_count,
                AVG(c_open_after) as avg_c_open,
                MAX(c_open_after) as max_c_open,
                SUM(c_opened_count) as total_opened,
                SUM(c_closed_count) as total_closed,
                SUM(CASE WHEN state_hash_before != state_hash_after THEN 1 ELSE 0 END) as state_changes
            FROM events
            GROUP BY run_id
            ORDER BY run_id
        """)
    
    def contradiction_lifetimes(self, min_lifetime_ms: float = 0) -> List[Dict[str, Any]]:
        """Get contradiction resolution times."""
        return self.engine.query(f"""
            SELECT 
                run_id,
                turn_id,
                c_closed_count,
                tau_resolve_ms_closed
            FROM events
            WHERE c_closed_count > 0
            ORDER BY run_id, turn_id
        """)
    
    def block_reasons(self) -> List[Dict[str, Any]]:
        """Get breakdown of what's causing BLOCKs."""
        return self.engine.query("""
            SELECT 
                run_id,
                blocked_by_invariant,
                COUNT(*) as count
            FROM events
            WHERE verdict = 'BLOCK'
            GROUP BY run_id, blocked_by_invariant
            ORDER BY count DESC
        """)
    
    def budget_trajectory(self, run_id: str) -> List[Dict[str, Any]]:
        """Get budget trajectory for a run."""
        return self.engine.query(f"""
            SELECT 
                turn_id,
                budget_remaining_after,
                budget_spent_this_turn,
                budget_exhaustion,
                verdict
            FROM events
            WHERE run_id = '{run_id}'
            ORDER BY turn_id
        """)
    
    def energy_trajectory(self, run_id: str) -> List[Dict[str, Any]]:
        """Get energy trajectory for a run."""
        return self.engine.query(f"""
            SELECT 
                turn_id,
                E_state_after as energy,
                E_components_after as components,
                c_open_after,
                verdict
            FROM events
            WHERE run_id = '{run_id}'
            ORDER BY turn_id
        """)
    
    def contradiction_load_trajectory(self, run_id: str) -> List[Dict[str, Any]]:
        """Get contradiction load over time."""
        return self.engine.query(f"""
            SELECT 
                turn_id,
                c_open_before,
                c_open_after,
                c_opened_count,
                c_closed_count,
                c_open_after - c_open_before as delta_c
            FROM events
            WHERE run_id = '{run_id}'
            ORDER BY turn_id
        """)
    
    def state_changes(self, run_id: str) -> List[Dict[str, Any]]:
        """Get turns where state actually changed."""
        return self.engine.query(f"""
            SELECT 
                turn_id,
                state_hash_before,
                state_hash_after,
                c_opened_count,
                c_closed_count,
                verdict
            FROM events
            WHERE run_id = '{run_id}'
              AND state_hash_before != state_hash_after
            ORDER BY turn_id
        """)
    
    def verdict_distribution(self) -> List[Dict[str, Any]]:
        """Get verdict distribution by run."""
        return self.engine.query("""
            SELECT 
                run_id,
                verdict,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY run_id), 2) as pct
            FROM events
            GROUP BY run_id, verdict
            ORDER BY run_id, verdict
        """)
    
    def extraction_failures(self) -> List[Dict[str, Any]]:
        """Get extraction failures."""
        return self.engine.query("""
            SELECT 
                run_id,
                turn_id,
                extract_status,
                extract_fail_reason,
                prompt_hash
            FROM events
            WHERE extract_status = 'fail'
            ORDER BY run_id, turn_id
        """)
    
    def high_energy_states(self, threshold: float = 30.0) -> List[Dict[str, Any]]:
        """Find high-energy states (potential instability)."""
        return self.engine.query(f"""
            SELECT 
                run_id,
                turn_id,
                E_state_after as energy,
                c_open_after,
                verdict
            FROM events
            WHERE E_state_after > {threshold}
            ORDER BY E_state_after DESC
            LIMIT 50
        """)
    
    def glassiness_check(self, run_id: str, window_size: int = 20) -> List[Dict[str, Any]]:
        """Check for glassiness (increasing C_open trend)."""
        return self.engine.query(f"""
            WITH windowed AS (
                SELECT 
                    turn_id,
                    c_open_after,
                    AVG(c_open_after) OVER (
                        ORDER BY turn_id 
                        ROWS BETWEEN {window_size} PRECEDING AND CURRENT ROW
                    ) as rolling_avg
                FROM events
                WHERE run_id = '{run_id}'
            )
            SELECT 
                turn_id,
                c_open_after,
                ROUND(rolling_avg, 2) as rolling_avg_c_open
            FROM windowed
            ORDER BY turn_id
        """)
    
    # =========================================================================
    # NEW: Throughput and transition queries
    # =========================================================================
    
    def throughput_metrics(self, run_id: str, window_size: int = 20) -> List[Dict[str, Any]]:
        """
        Get open/close throughput over time.
        
        λ_open vs μ_close - if λ > μ, glass accumulation is inevitable.
        """
        return self.engine.query(f"""
            WITH windowed AS (
                SELECT 
                    turn_id,
                    c_opened_count,
                    c_closed_count,
                    SUM(c_opened_count) OVER (
                        ORDER BY turn_id 
                        ROWS BETWEEN {window_size} PRECEDING AND CURRENT ROW
                    ) as window_opened,
                    SUM(c_closed_count) OVER (
                        ORDER BY turn_id 
                        ROWS BETWEEN {window_size} PRECEDING AND CURRENT ROW
                    ) as window_closed,
                    COUNT(*) OVER (
                        ORDER BY turn_id 
                        ROWS BETWEEN {window_size} PRECEDING AND CURRENT ROW
                    ) as window_size
                FROM events
                WHERE run_id = '{run_id}'
            )
            SELECT 
                turn_id,
                ROUND(window_opened * 1.0 / window_size, 3) as open_rate,
                ROUND(window_closed * 1.0 / window_size, 3) as close_rate,
                ROUND((window_opened - window_closed) * 1.0 / window_size, 3) as net_accumulation
            FROM windowed
            ORDER BY turn_id
        """)
    
    def severity_distribution(self, run_id: str) -> List[Dict[str, Any]]:
        """Get severity distribution over time."""
        return self.engine.query(f"""
            SELECT 
                turn_id,
                c_severity_hist_after,
                c_open_after
            FROM events
            WHERE run_id = '{run_id}'
            ORDER BY turn_id
        """)
    
    def top_domains_by_opens(self, run_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top domains contributing to contradiction opens."""
        # This requires parsing the domain from events - simplified version
        return self.engine.query(f"""
            SELECT 
                run_id,
                SUM(c_opened_count) as total_opened,
                SUM(c_closed_count) as total_closed,
                SUM(c_opened_count) - SUM(c_closed_count) as net_accumulation
            FROM events
            WHERE run_id = '{run_id}'
            GROUP BY run_id
        """)
    
    def transition_windows(self, run_id: str, window_size: int = 20) -> List[Dict[str, Any]]:
        """
        Find potential regime transition points.
        
        Looks for step changes in:
        - Block/warn rate
        - C_open trajectory
        - Net accumulation sign change
        """
        return self.engine.query(f"""
            WITH metrics AS (
                SELECT 
                    turn_id,
                    c_open_after,
                    verdict,
                    CASE WHEN verdict = 'BLOCK' THEN 1 ELSE 0 END as is_block,
                    CASE WHEN verdict = 'WARN' THEN 1 ELSE 0 END as is_warn,
                    c_opened_count,
                    c_closed_count,
                    AVG(c_open_after) OVER (
                        ORDER BY turn_id 
                        ROWS BETWEEN {window_size} PRECEDING AND CURRENT ROW
                    ) as rolling_c_open,
                    SUM(CASE WHEN verdict = 'BLOCK' THEN 1 ELSE 0 END) OVER (
                        ORDER BY turn_id 
                        ROWS BETWEEN {window_size} PRECEDING AND CURRENT ROW
                    ) as window_blocks,
                    SUM(c_opened_count - c_closed_count) OVER (
                        ORDER BY turn_id 
                        ROWS BETWEEN {window_size} PRECEDING AND CURRENT ROW
                    ) as window_net_accum
                FROM events
                WHERE run_id = '{run_id}'
            ),
            with_lag AS (
                SELECT 
                    *,
                    LAG(rolling_c_open, {window_size}) OVER (ORDER BY turn_id) as prev_rolling_c_open,
                    LAG(window_blocks, {window_size}) OVER (ORDER BY turn_id) as prev_window_blocks,
                    LAG(window_net_accum, {window_size}) OVER (ORDER BY turn_id) as prev_window_net_accum
                FROM metrics
            )
            SELECT 
                turn_id,
                c_open_after,
                ROUND(rolling_c_open, 2) as rolling_c_open,
                window_blocks,
                window_net_accum,
                CASE 
                    WHEN prev_rolling_c_open IS NOT NULL 
                         AND ABS(rolling_c_open - prev_rolling_c_open) > 5 
                    THEN 'C_OPEN_SHIFT'
                    WHEN prev_window_blocks IS NOT NULL 
                         AND ABS(window_blocks - prev_window_blocks) > 5
                    THEN 'BLOCK_RATE_SHIFT'
                    WHEN prev_window_net_accum IS NOT NULL 
                         AND window_net_accum > 0 AND prev_window_net_accum <= 0
                    THEN 'ACCUMULATION_START'
                    WHEN prev_window_net_accum IS NOT NULL 
                         AND window_net_accum <= 0 AND prev_window_net_accum > 0
                    THEN 'RECOVERY_START'
                    ELSE NULL
                END as transition_type
            FROM with_lag
            WHERE turn_id > {window_size}
            ORDER BY turn_id
        """)
    
    def precursor_signature(
        self, 
        run_id: str, 
        target_turn: int, 
        lookback: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get precursor signature before a specific turn.
        
        What changed in the N turns before a transition?
        """
        return self.engine.query(f"""
            WITH window_data AS (
                SELECT 
                    turn_id,
                    c_open_after,
                    c_opened_count,
                    c_closed_count,
                    verdict,
                    blocked_by_invariant,
                    E_state_after
                FROM events
                WHERE run_id = '{run_id}'
                  AND turn_id BETWEEN {target_turn - lookback} AND {target_turn}
            )
            SELECT 
                MIN(turn_id) as window_start,
                MAX(turn_id) as window_end,
                MIN(c_open_after) as min_c_open,
                MAX(c_open_after) as max_c_open,
                MAX(c_open_after) - MIN(c_open_after) as c_open_range,
                SUM(c_opened_count) as total_opened,
                SUM(c_closed_count) as total_closed,
                SUM(c_opened_count) - SUM(c_closed_count) as net_accumulation,
                SUM(CASE WHEN verdict = 'BLOCK' THEN 1 ELSE 0 END) as block_count,
                SUM(CASE WHEN verdict = 'WARN' THEN 1 ELSE 0 END) as warn_count,
                AVG(E_state_after) as avg_energy,
                MAX(E_state_after) as max_energy
            FROM window_data
        """)
    
    def burst_detection(self, run_id: str, threshold: int = 5) -> List[Dict[str, Any]]:
        """
        Detect C_open bursts (spikes that may or may not recover).
        
        Used to test classifier robustness against false GLASS detection.
        """
        return self.engine.query(f"""
            WITH deltas AS (
                SELECT 
                    turn_id,
                    c_open_after,
                    c_open_after - LAG(c_open_after, 1, c_open_after) OVER (ORDER BY turn_id) as delta,
                    LEAD(c_open_after, 10) OVER (ORDER BY turn_id) as future_c_open
                FROM events
                WHERE run_id = '{run_id}'
            )
            SELECT 
                turn_id,
                c_open_after,
                delta,
                future_c_open,
                CASE 
                    WHEN delta >= {threshold} AND future_c_open < c_open_after 
                    THEN 'BURST_WITH_RECOVERY'
                    WHEN delta >= {threshold} AND future_c_open >= c_open_after
                    THEN 'BURST_NO_RECOVERY'
                    WHEN delta <= -{threshold}
                    THEN 'RECOVERY_BURST'
                    ELSE NULL
                END as burst_type
            FROM deltas
            WHERE ABS(delta) >= {threshold}
            ORDER BY turn_id
        """)


# =============================================================================
# Report Generator
# =============================================================================

def generate_diagnostic_report(engine: QueryEngine) -> str:
    """Generate a diagnostic report from the event log."""
    queries = CanonicalQueries(engine)
    
    lines = []
    lines.append("=" * 70)
    lines.append("DIAGNOSTIC REPORT")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Events loaded: {engine.events_loaded}")
    lines.append("=" * 70)
    
    # Regime summary
    lines.append("\n## REGIME SUMMARY BY RUN\n")
    for row in queries.regime_summary():
        lines.append(f"### {row['run_id']}")
        lines.append(f"  Turns: {row['turns']}")
        lines.append(f"  Verdicts: OK={row['ok_count']}, WARN={row['warn_count']}, BLOCK={row['block_count']}")
        lines.append(f"  C_open: avg={row['avg_c_open']:.1f}, max={row['max_c_open']}")
        lines.append(f"  Contradictions: opened={row['total_opened']}, closed={row['total_closed']}")
        lines.append(f"  State changes: {row['state_changes']}")
        lines.append("")
    
    # Verdict distribution
    lines.append("\n## VERDICT DISTRIBUTION\n")
    for row in queries.verdict_distribution():
        lines.append(f"  {row['run_id']}: {row['verdict']} = {row['count']} ({row['pct']}%)")
    
    # Block reasons
    lines.append("\n## BLOCK REASONS\n")
    block_reasons = queries.block_reasons()
    if block_reasons:
        for row in block_reasons:
            lines.append(f"  {row['run_id']}: {row['blocked_by_invariant']} ({row['count']} times)")
    else:
        lines.append("  No BLOCKs recorded")
    
    # Extraction failures
    lines.append("\n## EXTRACTION FAILURES\n")
    failures = queries.extraction_failures()
    if failures:
        for row in failures:
            lines.append(f"  {row['run_id']} turn {row['turn_id']}: {row['extract_fail_reason']}")
    else:
        lines.append("  No extraction failures")
    
    # High energy states
    lines.append("\n## HIGH ENERGY STATES (>30)\n")
    high_energy = queries.high_energy_states(30.0)
    if high_energy:
        for row in high_energy[:10]:  # Top 10
            lines.append(f"  {row['run_id']} turn {row['turn_id']}: E={row['energy']:.1f}, C_open={row['c_open_after']}")
    else:
        lines.append("  No high-energy states detected")
    
    lines.append("\n" + "=" * 70)
    lines.append("END REPORT")
    lines.append("=" * 70)
    
    return "\n".join(lines)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Main entry point for query layer."""
    import sys
    
    default_path = Path(__file__).parent / "diagnostic_events.jsonl"
    jsonl_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    
    print("=" * 70)
    print("BLI QUERY LAYER")
    print("SQL Interface for Diagnostic Events")
    print("=" * 70)
    
    if not HAS_DUCKDB:
        print("\nError: DuckDB not installed")
        print("Run: pip install duckdb")
        return
    
    print(f"\nLoading events from: {jsonl_path}")
    
    try:
        engine = QueryEngine(jsonl_path)
        print(f"Loaded {engine.events_loaded} events")
    except FileNotFoundError:
        print(f"Error: Event log not found at {jsonl_path}")
        print("Run workloads.py first to generate diagnostic data")
        return
    
    # Generate report
    report = generate_diagnostic_report(engine)
    print(report)
    
    # Save report
    report_path = Path(__file__).parent / "diagnostic_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    # Interactive examples
    print("\n" + "=" * 70)
    print("EXAMPLE QUERIES")
    print("=" * 70)
    
    queries = CanonicalQueries(engine)
    
    # Example: Energy trajectory for storm
    print("\n### Energy Trajectory (contradiction_storm, first 10 turns)")
    trajectory = queries.energy_trajectory("workload_contradiction_storm")[:10]
    for row in trajectory:
        print(f"  Turn {row['turn_id']}: E={row['energy']:.1f}, C_open={row['c_open_after']}, {row['verdict']}")
    
    # Example: Glassiness check
    print("\n### Glassiness Check (contradiction_storm)")
    glass = queries.glassiness_check("workload_contradiction_storm")
    # Show every 20th row
    for row in glass[::20]:
        print(f"  Turn {row['turn_id']}: C_open={row['c_open_after']}, rolling_avg={row['rolling_avg_c_open']}")
    
    # Raw SQL example
    print("\n### Custom Query: Turns with high contradiction delta")
    results = engine.query("""
        SELECT 
            run_id,
            turn_id,
            c_open_after - c_open_before as delta,
            verdict
        FROM events
        WHERE ABS(c_open_after - c_open_before) > 3
        ORDER BY ABS(c_open_after - c_open_before) DESC
        LIMIT 10
    """)
    for row in results:
        print(f"  {row['run_id']} turn {row['turn_id']}: Δ={row['delta']}, {row['verdict']}")


if __name__ == "__main__":
    main()
