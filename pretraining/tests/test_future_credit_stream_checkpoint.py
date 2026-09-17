"""CPU-only checkpoint provenance regression; no model execution."""
from dataclasses import asdict

import pytest
import torch

from pretraining.future_credit_stream import training
from pretraining.future_credit_stream.config import ARCHITECTURE, Config


def test_resume_rejects_changed_validation_byte_accounting(tmp_path, monkeypatch):
    # Use an isolated source tree: never modify the actual upstream baseline.
    module = tmp_path / 'pretraining/future_credit_stream/training.py'
    for relative in (
        'pretraining/future_credit_stream/training.py',
        'pretraining/future_credit_stream/model.py',
        'pretraining/future_credit_stream/objective.py',
        'pretraining/future_credit_stream/data.py',
        'pretraining/future_credit_stream/config.py',
        'pretraining/nanogpt_mini/nanogpt_mini_model.py',
        'pretraining/nextlat.py',
        'train_gpt.py',
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('original source\n')
    monkeypatch.setattr(training, '__file__', str(module))
    monkeypatch.setattr(training, 'REPO_ROOT', tmp_path)
    config = Config()
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save({'architecture': ARCHITECTURE, 'config': asdict(config),
                'metadata': {'sources': training.source_hashes()}}, checkpoint)
    training.load_checkpoint(checkpoint, config)

    (tmp_path / 'train_gpt.py').write_text('changed validation byte accounting\n')
    with pytest.raises(ValueError):
        training.load_checkpoint(checkpoint, config)
