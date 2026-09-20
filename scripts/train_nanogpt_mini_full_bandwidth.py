"""Entrypoint for compiled full-bandwidth nanoGPT-mini; queue GPU runs with mlq."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.nanogpt_mini.nanogpt_mini_full_bandwidth_train import main

if __name__ == "__main__":
    main()
