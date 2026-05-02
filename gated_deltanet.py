"""Gated DeltaNet (Yang, Kautz, Hatamizadeh 2024, arXiv:2412.06464).

Just deltanet.py with --gated. Beta handles surgical overwrite at the
current key direction; alpha (the gate added here) decays the entire
prior state uniformly. Two complementary forgetting mechanisms.

Run:
    uv run python gated_deltanet.py
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    deltanet = Path(__file__).parent / "deltanet.py"
    sys.exit(subprocess.call([sys.executable, str(deltanet), "--gated", *sys.argv[1:]]))
