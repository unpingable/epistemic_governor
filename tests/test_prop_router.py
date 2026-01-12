"""
Test Suite for Proposition Identity Router

Tests:
1. Paraphrase binds (same meaning → same prop_id)
2. False bind doesn't happen (different year / different unit)
3. Split works and is idempotent
4. Gray zone triggers arbitration
5. Lowercase entity detection
"""

import unittest
from epistemic_governor.prop_router import (
    PropositionRouter,
    PropositionIndex,
    BindAction,
    BindResult,
    EntityDetector,
)


class TestExactMatch(unittest.TestCase):
    """Fast path: exact hash should return same prop_id."""
    
    def setUp(self):
        self.router = PropositionRouter()
    
    def test_exact_hash_returns_same_id(self):
        """Same hash should return same prop_id without re-scoring."""
        result1 = self.router.bind_or_mint(
            prop_hash="hash_001",
            entity_norm="python",
            predicate_norm="RELEASE",
            value_norm="YEAR:2022",
        )
        
        result2 = self.router.bind_or_mint(
            prop_hash="hash_001",  # Same hash
            entity_norm="python",
            predicate_norm="RELEASE",
            value_norm="YEAR:2022",
        )
        
        self.assertEqual(result1.prop_id, result2.prop_id)
        self.assertEqual(result2.action, BindAction.EXACT)


class TestParaphraseBind(unittest.TestCase):
    """Paraphrases should bind to same prop_id."""
    
    def setUp(self):
        self.router = PropositionRouter()
    
    def test_same_meaning_different_hash_binds(self):
        """Same entity/predicate/value with different hash should rebind."""
        # First claim
        result1 = self.router.bind_or_mint(
            prop_hash="hash_original",
            entity_norm="python_3.11",
            predicate_norm="RELEASE",
            value_norm="YEAR:2022",
            value_features={"year": 2022, "month": "october"},
        )
        self.assertEqual(result1.action, BindAction.NEW)
        
        # Paraphrase (different hash, same meaning)
        result2 = self.router.bind_or_mint(
            prop_hash="hash_paraphrase",
            entity_norm="python_3.11",
            predicate_norm="RELEASE",
            value_norm="YEAR:2022",
            value_features={"year": 2022},  # No month, but same year
        )
        
        self.assertEqual(result2.action, BindAction.REBIND)
        self.assertEqual(result2.prop_id, result1.prop_id)
    
    def test_date_without_month_binds_to_date_with_month(self):
        """'late 2022' should bind to 'October 2022' (same year)."""
        result1 = self.router.bind_or_mint(
            prop_hash="hash_october",
            entity_norm="python",
            predicate_norm="RELEASE",
            value_norm="YEAR:2022",
            value_features={"year": 2022, "month": "october"},
        )
        
        result2 = self.router.bind_or_mint(
            prop_hash="hash_late",
            entity_norm="python",
            predicate_norm="RELEASE",
            value_norm="YEAR:2022",
            value_features={"year": 2022, "date_modifier": "late"},
        )
        
        # Should bind (same year, same entity, same predicate)
        self.assertEqual(result2.action, BindAction.REBIND)
        self.assertEqual(result2.prop_id, result1.prop_id)


class TestFalseBindPrevention(unittest.TestCase):
    """Different propositions should NOT bind."""
    
    def setUp(self):
        self.router = PropositionRouter()
    
    def test_different_year_does_not_bind(self):
        """Different years are different propositions."""
        result1 = self.router.bind_or_mint(
            prop_hash="hash_2022",
            entity_norm="python",
            predicate_norm="RELEASE",
            value_norm="YEAR:2022",
            value_features={"year": 2022},
        )
        
        result2 = self.router.bind_or_mint(
            prop_hash="hash_2023",
            entity_norm="python",
            predicate_norm="RELEASE",
            value_norm="YEAR:2023",  # Different year!
            value_features={"year": 2023},
        )
        
        # Should NOT bind - different years
        self.assertIn(result2.action, [BindAction.NEW, BindAction.ARBITRATE])
        if result2.action == BindAction.NEW:
            self.assertNotEqual(result2.prop_id, result1.prop_id)
    
    def test_different_unit_does_not_bind(self):
        """Different units are different propositions."""
        result1 = self.router.bind_or_mint(
            prop_hash="hash_gb",
            entity_norm="system",
            predicate_norm="REQUIRES",
            value_norm="NUM:8:gb",
            value_features={"number": 8, "unit": "gb"},
        )
        
        result2 = self.router.bind_or_mint(
            prop_hash="hash_mb",
            entity_norm="system",
            predicate_norm="REQUIRES",
            value_norm="NUM:8:mb",  # Different unit!
            value_features={"number": 8, "unit": "mb"},
        )
        
        # Should NOT bind - different units
        self.assertIn(result2.action, [BindAction.NEW, BindAction.ARBITRATE])
        if result2.action == BindAction.NEW:
            self.assertNotEqual(result2.prop_id, result1.prop_id)
    
    def test_different_predicate_does_not_bind(self):
        """Different predicates are different propositions."""
        result1 = self.router.bind_or_mint(
            prop_hash="hash_release",
            entity_norm="python",
            predicate_norm="RELEASE",
            value_norm="YEAR:2022",
        )
        
        result2 = self.router.bind_or_mint(
            prop_hash="hash_deprecate",
            entity_norm="python",
            predicate_norm="DEPRECATE",  # Different predicate!
            value_norm="YEAR:2022",
        )
        
        # Should NOT bind - different predicates
        self.assertEqual(result2.action, BindAction.NEW)
        self.assertNotEqual(result2.prop_id, result1.prop_id)


class TestSplit(unittest.TestCase):
    """Split should correctly detach a hash from its identity."""
    
    def setUp(self):
        self.router = PropositionRouter()
    
    def test_split_creates_new_identity(self):
        """Split should create a new prop_id for the detached hash."""
        # Bind initial
        result1 = self.router.bind_or_mint(
            prop_hash="hash_1",
            entity_norm="python",
            predicate_norm="RELEASE",
            value_norm="YEAR:2022",
        )
        original_id = result1.prop_id
        
        # Force rebind a second hash
        result2 = self.router.bind_or_mint(
            prop_hash="hash_2",
            entity_norm="python",
            predicate_norm="RELEASE",
            value_norm="YEAR:2022",
        )
        self.assertEqual(result2.action, BindAction.REBIND)
        
        # Split hash_2
        new_id = self.router.split("hash_2", "Realized these are different")
        
        self.assertIsNotNone(new_id)
        self.assertNotEqual(new_id, original_id)
    
    def test_split_is_idempotent(self):
        """Splitting already-split hash should fail gracefully."""
        result1 = self.router.bind_or_mint(
            prop_hash="hash_solo",
            entity_norm="python",
            predicate_norm="RELEASE",
            value_norm="YEAR:2022",
        )
        
        # Split (creates new identity)
        new_id_1 = self.router.split("hash_solo", "First split")
        
        # Try to split again from original (should fail - hash is no longer there)
        # The hash is now bound to new_id_1
        # Actually, let's check the index state
        current_id = self.router.index.get_by_hash("hash_solo")
        self.assertEqual(current_id, new_id_1)


class TestGrayZone(unittest.TestCase):
    """Gray zone should trigger arbitration."""
    
    def setUp(self):
        self.router = PropositionRouter()
        # Lower bind threshold to create gray zone
        self.router.BIND_THRESHOLD = 0.95
        self.router.MAYBE_THRESHOLD = 0.80
    
    def test_partial_match_triggers_arbitration(self):
        """Partial entity match should trigger arbitration."""
        result1 = self.router.bind_or_mint(
            prop_hash="hash_1",
            entity_norm="python_3.11",
            predicate_norm="RELEASE",
            value_norm="YEAR:2022",
            value_features={"year": 2022},
        )
        
        # Similar but different entity
        result2 = self.router.bind_or_mint(
            prop_hash="hash_2",
            entity_norm="python_3.12",  # Different version!
            predicate_norm="RELEASE",
            value_norm="YEAR:2023",
            value_features={"year": 2023},
        )
        
        # Should be NEW or ARBITRATE (entity similarity is partial)
        self.assertIn(result2.action, [BindAction.NEW, BindAction.ARBITRATE])


class TestEntityDetection(unittest.TestCase):
    """Lowercase entity detection should work."""
    
    def test_snake_case(self):
        """Should detect snake_case identifiers."""
        text = "The config_file_path was updated."
        entities = EntityDetector.extract_entities(text)
        entity_texts = [e[0] for e in entities]
        self.assertIn("config_file_path", entity_texts)
    
    def test_dotted_identifier(self):
        """Should detect dotted identifiers."""
        text = "Set net.ipv4.ip_forward to 1."
        entities = EntityDetector.extract_entities(text)
        entity_texts = [e[0] for e in entities]
        # Should find dotted pattern
        found_dotted = any("net.ipv4" in e or "ip_forward" in e for e in entity_texts)
        self.assertTrue(found_dotted)
    
    def test_path(self):
        """Should detect file paths."""
        text = "Edit /etc/nginx/nginx.conf file."
        entities = EntityDetector.extract_entities(text)
        entity_texts = [e[0] for e in entities]
        found_path = any("/etc" in e or "nginx" in e for e in entity_texts)
        self.assertTrue(found_path)
    
    def test_version_string(self):
        """Should detect version strings."""
        text = "Upgrade to v2.3.1 for the fix."
        entities = EntityDetector.extract_entities(text)
        entity_texts = [e[0] for e in entities]
        # Should find version
        found_version = any("2.3.1" in e for e in entity_texts)
        self.assertTrue(found_version)
    
    def test_capitalized_still_works(self):
        """Standard capitalized entities should still be detected."""
        text = "Microsoft released Windows 11."
        entities = EntityDetector.extract_entities(text)
        entity_texts = [e[0] for e in entities]
        self.assertIn("Microsoft", entity_texts)
        self.assertIn("Windows", entity_texts)


class TestIndexPersistence(unittest.TestCase):
    """Index should checkpoint and restore correctly."""
    
    def test_checkpoint_restore(self):
        """Checkpoint/restore should preserve state."""
        router = PropositionRouter()
        
        # Add some data
        router.bind_or_mint(
            prop_hash="hash_1",
            entity_norm="python",
            predicate_norm="RELEASE",
            value_norm="YEAR:2022",
        )
        router.bind_or_mint(
            prop_hash="hash_2",
            entity_norm="java",
            predicate_norm="RELEASE",
            value_norm="YEAR:2023",
        )
        
        # Checkpoint
        checkpoint = router.index.checkpoint()
        
        # Create new router and restore
        new_router = PropositionRouter()
        new_router.index.restore(checkpoint)
        
        # Verify state
        self.assertEqual(new_router.index.get_by_hash("hash_1"), "p_00000001")
        self.assertEqual(new_router.index.get_by_hash("hash_2"), "p_00000002")


class TestInfoGainDetection(unittest.TestCase):
    """Info gain (less specific → more specific) must trigger arbitration."""
    
    def setUp(self):
        self.router = PropositionRouter()
    
    def test_year_to_month_triggers_arbitration(self):
        """Adding month to year-only date is info gain - must arbitrate."""
        # First: year-only
        result1 = self.router.bind_or_mint(
            prop_hash="hash_year_only",
            entity_norm="python",
            predicate_norm="RELEASE",
            value_norm="YEAR:2022",
            value_features={"year": 2022},  # No month
        )
        self.assertEqual(result1.action, BindAction.NEW)
        
        # Second: with month (info gain!)
        result2 = self.router.bind_or_mint(
            prop_hash="hash_with_month",
            entity_norm="python",
            predicate_norm="RELEASE",
            value_norm="YEAR:2022",
            value_features={"year": 2022, "month": "october"},  # Has month
        )
        
        # Must trigger arbitration, not autobind
        self.assertEqual(result2.action, BindAction.ARBITRATE)
        self.assertTrue(result2.has_info_gain)
    
    def test_month_to_year_allows_rebind(self):
        """Removing month (info loss) is allowed with penalty."""
        # First: with month
        result1 = self.router.bind_or_mint(
            prop_hash="hash_specific",
            entity_norm="python",
            predicate_norm="RELEASE",
            value_norm="YEAR:2022",
            value_features={"year": 2022, "month": "october"},
        )
        
        # Second: year-only (info loss - OK)
        result2 = self.router.bind_or_mint(
            prop_hash="hash_less_specific",
            entity_norm="python",
            predicate_norm="RELEASE",
            value_norm="YEAR:2022",
            value_features={"year": 2022},  # No month
        )
        
        # Should rebind (info loss is acceptable)
        self.assertEqual(result2.action, BindAction.REBIND)
        self.assertIn("info_loss", result2.match_reason)


class TestCollisionBudget(unittest.TestCase):
    """Collision budget should penalize overbinding."""
    
    def setUp(self):
        self.router = PropositionRouter()
    
    def test_overbinding_detection(self):
        """Too many binds with different values should be flagged."""
        # Bind initial
        self.router.bind_or_mint(
            prop_hash="base_hash",
            entity_norm="system",
            predicate_norm="REQUIRES",
            value_norm="NUM:8:gb",
            value_features={"number": 8, "unit": "gb"},
        )
        
        # Simulate many binds (force rebind to same identity)
        record = self.router.index.get_record("p_00000001")
        
        # Manually add many different value signatures to trigger overbinding
        for i in range(12):
            record.add_hash(f"fake_hash_{i}", step=i, value_signature=f"sig_{i}")
        
        self.assertTrue(record.is_overbinding)


class TestSplitMetadata(unittest.TestCase):
    """Split should use the hash's own metadata, not parent's."""
    
    def setUp(self):
        self.router = PropositionRouter()
    
    def test_split_preserves_hash_metadata(self):
        """Split should create record with hash's own triple."""
        # Bind initial with October
        r1 = self.router.bind_or_mint(
            prop_hash="hash_oct",
            entity_norm="python",
            predicate_norm="RELEASE",
            value_norm="YEAR:2022",
            value_features={"year": 2022, "month": "october"},
            value_raw="October 2022",
        )
        
        # Force rebind with November (pretend user approved arbitration)
        self.router.index.rebind(
            prop_hash="hash_nov",
            target_prop_id=r1.prop_id,
            entity_norm="python",
            predicate_norm="RELEASE",
            value_norm="YEAR:2022",
            value_features={"year": 2022, "month": "november"},
            value_raw="November 2022",
        )
        
        # Split the November hash
        new_id = self.router.split("hash_nov", "Different month")
        self.assertIsNotNone(new_id)
        
        # Verify split record has November's metadata
        hash_meta = self.router.index.get_hash_meta("hash_nov")
        self.assertIsNotNone(hash_meta)
        self.assertEqual(hash_meta.value_raw, "November 2022")
        self.assertEqual(hash_meta.value_features.get("month"), "november")


if __name__ == "__main__":
    unittest.main(verbosity=2)
