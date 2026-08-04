"""Operational entry point for the LeJEPA answer-encoder experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postraining.answer_encoder.train import main


if __name__ == "__main__":
    main()
