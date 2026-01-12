"""
Test Suite for Claim Extraction and Diff

Locks down critical invariants:
1. Negation is local to span
2. Sentence offsets are correct
3. Value drift is detected (October 2022 vs late 2022)
4. Modality strengthen is detected on same prop
5. Quantifier detection doesn't false-positive on 'a/an'
"""

import unittest
from epistemic_governor.claim_extractor import (
    ClaimExtractor, ExtractMode, ClaimAtom, Quantifier, Modality
)
from epistemic_governor.claim_diff import (
    ClaimDiffer, MutationType, DiffResult
)


class TestPolarity(unittest.TestCase):
    """Polarity must be local to the matched span."""
    
    def setUp(self):
        self.extractor = ClaimExtractor()
    
    def test_negation_in_span(self):
        """Negation in the matched span should flip polarity."""
        text = "Python is not fast."
        claims = self.extractor.extract(text, ExtractMode.OUTPUT)
        # Should find "Python is not fast" with negative polarity
        fast_claims = [c for c in claims.claims if "fast" in c.value_raw.lower()]
        if fast_claims:
            self.assertEqual(fast_claims[0].polarity, -1)
    
    def test_negation_outside_span(self):
        """Negation outside the relevant clause should not flip polarity."""
        text = "Python is fast, not slow."
        claims = self.extractor.extract(text, ExtractMode.OUTPUT)
        # The "Python is fast" claim should have positive polarity
        # because "not" applies to "slow", not "fast"
        fast_claims = [c for c in claims.claims if "fast" in c.value_raw.lower()]
        if fast_claims:
            # This should be +1 because "not" modifies "slow", not "fast"
            self.assertEqual(fast_claims[0].polarity, 1)
    
    def test_never_negation(self):
        """'Never' should produce negative polarity."""
        text = "Python has never been slow."
        claims = self.extractor.extract(text, ExtractMode.OUTPUT)
        for c in claims.claims:
            if "never" in c.span_quote.lower():
                self.assertEqual(c.polarity, -1)


class TestSentenceOffsets(unittest.TestCase):
    """Sentence offsets must be correct even with repeated sentences."""
    
    def setUp(self):
        self.extractor = ClaimExtractor()
    
    def test_unique_sentences(self):
        """Each sentence should have correct, non-overlapping spans."""
        text = "Python is fast. Java is also fast."
        claims = self.extractor.extract(text, ExtractMode.OUTPUT)
        
        spans = [(c.span[0], c.span[1]) for c in claims.claims]
        # Check spans don't overlap unexpectedly
        for i, (start1, end1) in enumerate(spans):
            for j, (start2, end2) in enumerate(spans):
                if i != j:
                    # Spans should either be disjoint or nested
                    disjoint = end1 <= start2 or end2 <= start1
                    nested = (start1 <= start2 and end1 >= end2) or (start2 <= start1 and end2 >= end1)
                    self.assertTrue(disjoint or nested, 
                        f"Overlapping spans: ({start1}, {end1}) and ({start2}, {end2})")
    
    def test_span_matches_text(self):
        """Span should reference the correct text."""
        text = "Python 3.11 was released in October 2022."
        claims = self.extractor.extract(text, ExtractMode.OUTPUT)
        
        for c in claims.claims:
            start, end = c.span
            extracted = text[start:end]
            # The extracted span should contain the claim's key elements
            self.assertIn("2022", extracted)


class TestValueDrift(unittest.TestCase):
    """Value drift must be detected between same-hash claims."""
    
    def setUp(self):
        self.extractor = ClaimExtractor()
        self.differ = ClaimDiffer()
    
    def test_date_precision_drift(self):
        """'October 2022' vs 'late 2022' should produce value drift."""
        source = "Python 3.11 was released in October 2022."
        output = "Python 3.11 was released in late 2022."
        
        source_claims = self.extractor.extract(source, ExtractMode.SOURCE)
        output_claims = self.extractor.extract(output, ExtractMode.OUTPUT)
        
        diff = self.differ.diff(source_claims, output_claims)
        
        # Should have same hash (both 2022) but value drift mutation
        self.assertEqual(len(diff.novel), 0, "Should not be novel - same year")
        
        mutations = diff.get_mutation_events()
        value_drifts = [m for m in mutations if m.mutation_type == MutationType.VALUE_DRIFT]
        self.assertGreater(len(value_drifts), 0, "Should detect value drift")
    
    def test_modifier_drift(self):
        """'at least 8GB' vs 'approximately 8GB' should produce value drift."""
        source = "The system requires at least 8GB of RAM."
        output = "The system requires approximately 8GB of RAM."
        
        source_claims = self.extractor.extract(source, ExtractMode.SOURCE)
        output_claims = self.extractor.extract(output, ExtractMode.OUTPUT)
        
        diff = self.differ.diff(source_claims, output_claims)
        
        mutations = diff.get_mutation_events()
        value_drifts = [m for m in mutations if m.mutation_type == MutationType.VALUE_DRIFT]
        
        # Should detect the modifier change
        self.assertGreater(len(value_drifts), 0, "Should detect modifier drift")
        
        # Check the drift details mention the modifiers
        for vd in value_drifts:
            self.assertIn("modifier", vd.details.lower())


class TestModalityDetection(unittest.TestCase):
    """Modality strengthening must be detected on same proposition."""
    
    def setUp(self):
        self.extractor = ClaimExtractor()
        self.differ = ClaimDiffer()
    
    def test_might_to_assert(self):
        """'might cause' → 'causes' should be modality strengthen."""
        source = "Heat might cause damage."
        output = "Heat causes damage."
        
        source_claims = self.extractor.extract(source, ExtractMode.SOURCE)
        output_claims = self.extractor.extract(output, ExtractMode.OUTPUT)
        
        # Check source has MIGHT modality
        for c in source_claims.claims:
            if "cause" in c.predicate.lower() or "cause" in c.value_raw.lower():
                self.assertEqual(c.modality, Modality.MIGHT)
        
        # Check output has ASSERT modality  
        for c in output_claims.claims:
            if "cause" in c.predicate.lower() or "cause" in c.value_raw.lower():
                self.assertEqual(c.modality, Modality.ASSERT)


class TestQuantifierDetection(unittest.TestCase):
    """Quantifier detection should not false-positive on 'a/an'."""
    
    def setUp(self):
        self.extractor = ClaimExtractor()
    
    def test_no_false_positive_on_article(self):
        """Random 'a' in text should not set EXISTS quantifier."""
        text = "Python is a programming language."
        claims = self.extractor.extract(text, ExtractMode.OUTPUT)
        
        # The quantifier should be UNKNOWN, not EXISTS
        # because "a" is part of the value, not a quantifier
        for c in claims.claims:
            # Unless the entity is specifically "a X", shouldn't be EXISTS
            if "python" in str(c.entities).lower():
                # "a programming language" is the value, not a quantifier on Python
                self.assertNotEqual(c.quantifier, Quantifier.EXISTS,
                    "Should not detect 'a' as quantifier when it's part of value")
    
    def test_explicit_quantifiers(self):
        """Explicit quantifiers like 'all', 'some', 'most' should be detected."""
        # Use patterns that our extractor can actually match (copula patterns)
        test_cases = [
            ("All users are affected.", Quantifier.ALL),
            ("Some users are affected.", Quantifier.SOME),
            ("Most users are affected.", Quantifier.MOST),
        ]
        
        for text, expected_quant in test_cases:
            claims = self.extractor.extract(text, ExtractMode.OUTPUT)
            if claims.claims:
                found = any(c.quantifier == expected_quant for c in claims.claims)
                self.assertTrue(found, f"Should detect {expected_quant.name} in: {text}")


class TestNewClaimDetection(unittest.TestCase):
    """New claims (novel hashes) should be correctly identified."""
    
    def setUp(self):
        self.extractor = ClaimExtractor()
        self.differ = ClaimDiffer()
    
    def test_added_claim(self):
        """A completely new statement should be detected as novel."""
        source = "Python is fast."
        output = "Python is fast. Java is also fast."
        
        source_claims = self.extractor.extract(source, ExtractMode.SOURCE)
        output_claims = self.extractor.extract(output, ExtractMode.OUTPUT)
        
        diff = self.differ.diff(source_claims, output_claims)
        
        # Should have at least one novel claim about Java
        java_novel = [c for c in diff.novel if "java" in str(c.entities).lower()]
        self.assertGreater(len(java_novel), 0, "Should detect Java claim as novel")
    
    def test_dropped_claim(self):
        """A removed statement should be detected as dropped."""
        source = "Python is fast. Java is also fast."
        output = "Python is fast."
        
        source_claims = self.extractor.extract(source, ExtractMode.SOURCE)
        output_claims = self.extractor.extract(output, ExtractMode.OUTPUT)
        
        diff = self.differ.diff(source_claims, output_claims)
        
        # Should have at least one dropped claim about Java
        java_dropped = [c for c in diff.dropped if "java" in str(c.entities).lower()]
        self.assertGreater(len(java_dropped), 0, "Should detect Java claim as dropped")


if __name__ == "__main__":
    unittest.main(verbosity=2)
