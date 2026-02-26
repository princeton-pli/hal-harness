"""Add reliability_eval directory to sys.path for imports."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "reliability_eval"))
