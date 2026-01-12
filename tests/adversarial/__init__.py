"""
Adversarial Test Suite for BLI Governor

Tests that demonstrate failures in naive systems
and verify BLI blocks them mechanically.

Run all tests:
    python -m pytest adversarial/ -v

Run individual tests:
    python adversarial/test_forced_resolution.py
    python adversarial/test_authority_spoofing.py
    python adversarial/test_self_certification.py
"""

from pathlib import Path

ADVERSARIAL_DIR = Path(__file__).parent
