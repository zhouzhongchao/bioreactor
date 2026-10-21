"""Convenience launcher; application source lives in code/."""

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "code" / "qt_gui.py"),
        run_name="__main__",
    )
