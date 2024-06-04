from pathlib import Path
import sys

__LOC__ = Path(__file__).resolve()
ROOT_DIR = __LOC__.parents[1]

# Ensure project's ROOT directory is in Python path so we can import from `src`
sys.path.append(ROOT_DIR)
