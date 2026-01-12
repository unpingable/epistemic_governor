"""
Clock Invariant Tests

Test that the three clocks (Δt₁/₂/₃) behave correctly:
1. Monotonic - clocks never go backward
2. Decoupled - each clock can tick independently
3. Observable - clock deltas appear in telemetry

These are NOT tuning tests - just structural invariants.
"""

from dataclasses import dataclass
from typing import List, Dict, Any

try:
    from symbolic_substrate import TemporalState
except ImportError:
    from symbolic_substrate import TemporalState


# =============================================================================
# Clock Telemetry
# =============================================================================

@dataclass
class ClockDelta:
    """Records a clock tick event."""
    clock: str  # "fast", "integrative", "developmental"
    old_value: int
    new_value: int
    trigger: str  # What caused the tick


class ClockObserver:
    """
    Observes clock changes for telemetry.
    
    Wraps TemporalState to track all clock deltas.
    """
    
    def __init__(self, theta: TemporalState = None):
        self.theta = theta or TemporalState()
        self.deltas: List[ClockDelta] = []
    
    def tick_fast(self):
        """Tick fast clock with observation."""
        old = self.theta.t_fast
        self.theta.tick_fast()
        self.deltas.append(ClockDelta(
            clock="fast",
            old_value=old,
            new_value=self.theta.t_fast,
            trigger="turn",
        ))
    
    def maybe_tick_integrative(
        self, 
        contradiction_resolved: bool, 
        stability: float
    ) -> bool:
        """Tick integrative clock with observation."""
        old = self.theta.t_integrative
        ticked = self.theta.maybe_tick_integrative(contradiction_resolved, stability)
        if ticked:
            trigger = "contradiction_resolved" if contradiction_resolved else "stability_streak"
            self.deltas.append(ClockDelta(
                clock="integrative",
                old_value=old,
                new_value=self.theta.t_integrative,
                trigger=trigger,
            ))
        return ticked
    
    def maybe_tick_developmental(
        self, 
        trauma: bool, 
        sustained_pressure_turns: int
    ) -> bool:
        """Tick developmental clock with observation."""
        old = self.theta.t_developmental
        ticked = self.theta.maybe_tick_developmental(trauma, sustained_pressure_turns)
        if ticked:
            trigger = "trauma" if trauma else "sustained_pressure"
            self.deltas.append(ClockDelta(
                clock="developmental",
                old_value=old,
                new_value=self.theta.t_developmental,
                trigger=trigger,
            ))
        return ticked
    
    def get_state(self) -> Dict[str, int]:
        """Get current clock state."""
        return {
            "t_fast": self.theta.t_fast,
            "t_integrative": self.theta.t_integrative,
            "t_developmental": self.theta.t_developmental,
        }
    
    def get_telemetry(self) -> Dict[str, Any]:
        """Get clock telemetry."""
        return {
            "state": self.get_state(),
            "total_deltas": len(self.deltas),
            "deltas_by_clock": {
                "fast": sum(1 for d in self.deltas if d.clock == "fast"),
                "integrative": sum(1 for d in self.deltas if d.clock == "integrative"),
                "developmental": sum(1 for d in self.deltas if d.clock == "developmental"),
            },
            "triggers": [d.trigger for d in self.deltas],
        }


# =============================================================================
# Tests
# =============================================================================

def test_monotonicity():
    """Test: Clocks never go backward."""
    print("=== Test: Clock monotonicity ===\n")
    
    observer = ClockObserver()
    
    # Track max values seen
    max_fast = 0
    max_integrative = 0
    max_developmental = 0
    
    # Simulate 100 turns with various events
    for i in range(100):
        observer.tick_fast()
        
        # Every 10 turns, check for integrative tick
        if i % 10 == 9:
            observer.maybe_tick_integrative(
                contradiction_resolved=(i % 20 == 19),
                stability=0.9 if i % 5 == 4 else 0.5,
            )
        
        # Trauma at turn 50
        if i == 50:
            observer.maybe_tick_developmental(trauma=True, sustained_pressure_turns=0)
        
        # Check monotonicity
        state = observer.get_state()
        
        assert state["t_fast"] >= max_fast, f"Fast clock went backward at turn {i}"
        assert state["t_integrative"] >= max_integrative, f"Integrative clock went backward at turn {i}"
        assert state["t_developmental"] >= max_developmental, f"Developmental clock went backward at turn {i}"
        
        max_fast = state["t_fast"]
        max_integrative = state["t_integrative"]
        max_developmental = state["t_developmental"]
    
    print(f"Final state: {observer.get_state()}")
    print(f"All clocks monotonic through 100 turns")
    print("✓ Monotonicity maintained\n")
    return True


def test_decoupling():
    """Test: Each clock can tick independently."""
    print("=== Test: Clock decoupling ===\n")
    
    observer = ClockObserver()
    
    # Tick only fast clock
    for _ in range(10):
        observer.tick_fast()
    
    state1 = observer.get_state()
    print(f"After 10 fast ticks: {state1}")
    
    assert state1["t_fast"] == 10, "Fast should be 10"
    assert state1["t_integrative"] == 0, "Integrative should be 0"
    assert state1["t_developmental"] == 0, "Developmental should be 0"
    
    # Tick integrative without affecting fast
    initial_fast = state1["t_fast"]
    observer.maybe_tick_integrative(contradiction_resolved=True, stability=0.5)
    
    state2 = observer.get_state()
    print(f"After integrative tick: {state2}")
    
    assert state2["t_fast"] == initial_fast, "Fast should not change"
    assert state2["t_integrative"] == 1, "Integrative should be 1"
    
    # Tick developmental without affecting others
    initial_integrative = state2["t_integrative"]
    observer.maybe_tick_developmental(trauma=True, sustained_pressure_turns=0)
    
    state3 = observer.get_state()
    print(f"After developmental tick: {state3}")
    
    assert state3["t_integrative"] == initial_integrative, "Integrative should not change"
    assert state3["t_developmental"] == 1, "Developmental should be 1"
    
    print("✓ Clocks tick independently\n")
    return True


def test_observability():
    """Test: Clock deltas appear in telemetry."""
    print("=== Test: Clock observability ===\n")
    
    observer = ClockObserver()
    
    # Generate some events
    observer.tick_fast()
    observer.tick_fast()
    observer.maybe_tick_integrative(contradiction_resolved=True, stability=0.5)
    observer.maybe_tick_developmental(trauma=True, sustained_pressure_turns=0)
    
    telemetry = observer.get_telemetry()
    
    print(f"Telemetry: {telemetry}")
    
    assert telemetry["total_deltas"] == 4, "Should have 4 deltas"
    assert telemetry["deltas_by_clock"]["fast"] == 2, "Should have 2 fast deltas"
    assert telemetry["deltas_by_clock"]["integrative"] == 1, "Should have 1 integrative delta"
    assert telemetry["deltas_by_clock"]["developmental"] == 1, "Should have 1 developmental delta"
    
    # Check triggers are recorded
    assert "turn" in telemetry["triggers"], "Should record turn trigger"
    assert "contradiction_resolved" in telemetry["triggers"], "Should record contradiction trigger"
    assert "trauma" in telemetry["triggers"], "Should record trauma trigger"
    
    print("✓ All clock deltas observable in telemetry\n")
    return True


def test_fast_always_ticks():
    """Test: Fast clock ticks every turn regardless of other conditions."""
    print("=== Test: Fast clock always ticks ===\n")
    
    observer = ClockObserver()
    
    for i in range(50):
        observer.tick_fast()
    
    assert observer.theta.t_fast == 50, "Fast should equal turn count"
    
    print(f"Fast clock: {observer.theta.t_fast}")
    print("✓ Fast clock ticks reliably every turn\n")
    return True


def test_integrative_conditions():
    """Test: Integrative clock only ticks under specific conditions."""
    print("=== Test: Integrative tick conditions ===\n")
    
    # Test 1: Contradiction resolution triggers tick
    observer1 = ClockObserver()
    ticked = observer1.maybe_tick_integrative(contradiction_resolved=True, stability=0.0)
    assert ticked, "Should tick on contradiction resolution"
    print("✓ Contradiction resolution triggers integrative tick")
    
    # Test 2: High stability streak triggers tick
    observer2 = ClockObserver()
    for _ in range(10):
        observer2.tick_fast()
        ticked = observer2.maybe_tick_integrative(contradiction_resolved=False, stability=0.9)
    
    assert observer2.theta.t_integrative > 0, "Should tick after stability streak"
    print(f"✓ Stability streak triggers integrative tick (t_integrative={observer2.theta.t_integrative})")
    
    # Test 3: Low stability doesn't trigger
    observer3 = ClockObserver()
    for _ in range(20):
        observer3.tick_fast()
        observer3.maybe_tick_integrative(contradiction_resolved=False, stability=0.3)
    
    assert observer3.theta.t_integrative == 0, "Should not tick with low stability"
    print("✓ Low stability does not trigger integrative tick")
    
    print()
    return True


def test_developmental_conditions():
    """Test: Developmental clock only ticks under extreme conditions."""
    print("=== Test: Developmental tick conditions ===\n")
    
    # Test 1: Normal operation - no tick
    observer1 = ClockObserver()
    for _ in range(100):
        observer1.tick_fast()
        observer1.maybe_tick_developmental(trauma=False, sustained_pressure_turns=5)
    
    assert observer1.theta.t_developmental == 0, "Should not tick under normal conditions"
    print("✓ Normal operation: no developmental tick")
    
    # Test 2: Trauma triggers immediate tick
    observer2 = ClockObserver()
    ticked = observer2.maybe_tick_developmental(trauma=True, sustained_pressure_turns=0)
    assert ticked, "Trauma should trigger tick"
    print("✓ Trauma triggers developmental tick")
    
    # Test 3: Sustained pressure triggers tick
    observer3 = ClockObserver()
    ticked = observer3.maybe_tick_developmental(trauma=False, sustained_pressure_turns=25)
    assert ticked, "Sustained pressure should trigger tick"
    print("✓ Sustained pressure triggers developmental tick")
    
    print()
    return True


def run_all_tests():
    """Run all clock invariant tests."""
    print("=" * 60)
    print("CLOCK INVARIANT TESTS")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("monotonicity", test_monotonicity()))
    results.append(("decoupling", test_decoupling()))
    results.append(("observability", test_observability()))
    results.append(("fast_always_ticks", test_fast_always_ticks()))
    results.append(("integrative_conditions", test_integrative_conditions()))
    results.append(("developmental_conditions", test_developmental_conditions()))
    
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
