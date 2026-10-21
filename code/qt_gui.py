"""Launch the desktop GUI without connecting to hardware automatically."""

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    source_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(source_root))
    runpy.run_path(str(source_root / "control" / "qt_gui.py"), run_name="__main__")
