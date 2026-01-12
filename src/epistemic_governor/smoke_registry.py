#!/usr/bin/env python3
"""
smoke_registry.py - Minimum reality check for the registry ABI

Answers four questions:
1. Does it import cleanly?
2. Do audit semantics produce the lattice outcomes?
3. Does commit gating work end-to-end?
4. Do we get inspectable artifacts?

Run: python smoke_registry.py
"""

from datetime import datetime

# Question 1: Does it import cleanly?
print("1. IMPORT CHECK")
try:
    from epistemic_governor import (
        create_registry,
        register_epistemic_invariants,
        EpistemicConfig,
        ProposalEnvelope,
        StateView,
        Domain,
        AuditStatus,
        AuditReport,
    )
    print("   ✓ All imports successful")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    exit(1)

# Minimal in-memory ledger
class TinyLedger:
    def __init__(self):
        self.committed = {}  # id -> proposal
        self.proposition_hashes = set()
    
    def commit(self, proposal: ProposalEnvelope):
        self.committed[proposal.proposal_id] = proposal
        if "proposition_hash" in proposal.payload:
            self.proposition_hashes.add(proposal.payload["proposition_hash"])
    
    def to_state_view(self, current_t: int) -> StateView:
        return StateView(
            current_t=current_t,
            active_claims={h: f"claim_{i}" for i, h in enumerate(self.proposition_hashes)},
            claim_count=len(self.committed),
        )

# Setup
print("\n2. SETUP")
registry = create_registry()
register_epistemic_invariants(registry, EpistemicConfig())
ledger = TinyLedger()
print(f"   Registry: {len(registry.list_invariants())} invariants")
print(f"   Ledger: empty")

def make_proposal(id: str, t: int, confidence: float, claim_type: str = "FACTUAL", 
                  evidence: list = None, payload_extra: dict = None) -> ProposalEnvelope:
    payload = {"claim_type": claim_type, "proposition_hash": f"hash_{id}"}
    if payload_extra:
        payload.update(payload_extra)
    return ProposalEnvelope(
        proposal_id=id,
        t=t,
        timestamp=datetime.now(),
        origin="test",
        origin_type="test",
        domain=Domain.EPISTEMIC,
        confidence=confidence,
        evidence_refs=evidence or [],
        payload=payload,
    )

def run_audit(name: str, proposal: ProposalEnvelope, expect_status: AuditStatus):
    """Run audit and print report."""
    state = ledger.to_state_view(proposal.t - 1)
    report = registry.audit(proposal, state)
    
    status_ok = "✓" if report.status == expect_status else "✗"
    print(f"\n   [{status_ok}] {name}")
    print(f"       Status: {report.status.name} (expected: {expect_status.name})")
    
    if report.violated_invariants:
        print(f"       Violated: {report.violated_invariants}")
    if report.applied_clamps:
        print(f"       Clamps: {report.applied_clamps}")
    if report.required_evidence:
        print(f"       Required: {report.required_evidence}")
    print(f"       Heat: {report.total_heat_delta:.4f}, Work: {report.total_work_delta:.4f}")
    
    # Question 3: Commit gating
    if report.accepted:
        ledger.commit(report.final_proposal)
        # Mark as committed for duplicate detection
        id_inv = registry.get_invariant("global.unique_proposal_id")
        if id_inv:
            id_inv.invariant.mark_committed(proposal.proposal_id)
        print(f"       → Committed (ledger now has {len(ledger.committed)} claims)")
    else:
        print(f"       → NOT committed")
    
    return report

# Question 2: Audit semantics
print("\n3. AUDIT SEMANTICS")

# Test A: Clean claim with support → ACCEPT
p1 = make_proposal("claim_1", t=1, confidence=0.7, evidence=["source_1"])
run_audit("Clean claim with support", p1, AuditStatus.ACCEPT)

# Test B: Same claim again → VETO (duplicate)
p2 = make_proposal("claim_1", t=2, confidence=0.7, evidence=["source_1"])  # Same ID
run_audit("Duplicate claim", p2, AuditStatus.REJECTED)

# Test C: Contradictory claim → VETO
p3 = make_proposal("claim_3", t=3, confidence=0.7, evidence=["source_1"],
                   payload_extra={"contradicts": "claim_1"})
run_audit("Contradictory claim", p3, AuditStatus.REJECTED)

# Test D: Unsupported QUANTITATIVE → DEFER
p4 = make_proposal("claim_4", t=4, confidence=0.6, claim_type="QUANTITATIVE", evidence=[])
run_audit("Unsupported QUANTITATIVE", p4, AuditStatus.DEFERRED)

# Test E: Overconfident claim → CLAMP → ACCEPT
p5 = make_proposal("claim_5", t=5, confidence=0.99, evidence=["source_1"])
run_audit("Overconfident claim", p5, AuditStatus.CLAMPED)

# Question 4: Inspectable artifacts
print("\n4. FINAL STATE")
print(f"   Ledger committed: {list(ledger.committed.keys())}")
print(f"   Proposition hashes: {ledger.proposition_hashes}")

print("\n=== SMOKE TEST COMPLETE ===")
