"""
Pytest configuration.

Ensures the src directory is on the path for imports.
"""

import sys
from pathlib import Path

# Add src to path so imports work
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
