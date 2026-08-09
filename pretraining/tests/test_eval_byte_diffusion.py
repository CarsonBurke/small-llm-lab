from types import SimpleNamespace

import pytest

from pretraining.byte_diffusion.training import CHECKPOINT_SCHEMA
from scripts.eval_byte_diffusion import validate_evaluation_provenance


def _trainer(*, train_hash: str = "train-a", validation_hash: str = "val-a"):
    return SimpleNamespace(
        train_cursor=SimpleNamespace(dataset_sha256=train_hash),
        validation_sha256=validation_hash,
    )


def _payload() -> dict:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "dataset_sha256": "train-a",
        "validation_sha256": "val-a",
        "dataset_provenance": {
            "payload_sha256": "payload-a",
            "source_manifests": [{"sha256": "source-a"}],
        },
    }


def test_proxy_evaluation_requires_exact_train_validation_and_payload() -> None:
    payload = _payload()
    provenance = payload["dataset_provenance"]
    validate_evaluation_provenance(
        payload, _trainer(), provenance, full_validation=False
    )
    with pytest.raises(ValueError, match="proxy validation"):
        validate_evaluation_provenance(
            payload,
            _trainer(validation_hash="val-b"),
            provenance,
            full_validation=False,
        )
    with pytest.raises(ValueError, match="training split"):
        validate_evaluation_provenance(
            payload,
            _trainer(train_hash="train-b"),
            provenance,
            full_validation=False,
        )
    with pytest.raises(ValueError, match="payload/source"):
        validate_evaluation_provenance(
            payload,
            _trainer(),
            {"payload_sha256": "payload-b", "source_manifests": []},
            full_validation=False,
        )


def test_full_evaluation_uses_bound_source_identity_not_proxy_hash() -> None:
    payload = _payload()
    validate_evaluation_provenance(
        payload,
        _trainer(validation_hash="full-validation"),
        payload["dataset_provenance"],
        full_validation=True,
    )
    payload["schema"] = "byte_diffusion_training/obsolete"
    with pytest.raises(ValueError, match="schema"):
        validate_evaluation_provenance(
            payload,
            _trainer(validation_hash="full-validation"),
            payload["dataset_provenance"],
            full_validation=True,
        )
