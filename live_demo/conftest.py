# Lets the live-demo tests import the `feeder` package (run: .venv/bin/pytest -q live_demo).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
