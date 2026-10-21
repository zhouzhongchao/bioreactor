"""Source and writable-data roots, independent of the working directory."""

import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("TRO_DATA_ROOT", str(CODE_ROOT.parent / "data"))).resolve()
CONTROL_LOG_DIR = DATA_ROOT / "output" / "control_logs"

