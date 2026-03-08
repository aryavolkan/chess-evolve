import sys
from pathlib import Path

# Make python/ modules importable by their bare names
sys.path.insert(0, str(Path(__file__).parent / "python"))
