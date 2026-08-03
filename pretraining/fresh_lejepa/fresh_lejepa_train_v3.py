"""Fresh LeJEPA V3: V2 compact probes with 0.1 residual dropout."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from pathlib import Path

from pretraining.fresh_lejepa import fresh_lejepa_train as v1
from pretraining.fresh_lejepa import fresh_lejepa_train_v2 as v2


ARCHITECTURE = "fresh_lejepa_attached_target_additive_codebook_dropout01_probes_v3"


class FreshLeJEPAGPTV3(v2.FreshLeJEPAGPTV2):
    probe_dropout = 0.1


def main() -> None:
    v1.FreshLeJEPAGPT = FreshLeJEPAGPTV3
    v1.EXPERIMENT_ARCHITECTURE = ARCHITECTURE
    v1.EXPERIMENT_SOURCE = Path(__file__)
    v1.main()


if __name__ == "__main__":
    main()
