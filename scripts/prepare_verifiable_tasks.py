#!/usr/bin/env python3
"""Prepare bounded source-backed verifiable tasks; submit through mlq."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from postraining.task_data import main

if __name__ == "__main__":
    main()
