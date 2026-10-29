import json

import pytest

from scripts.recover_latent_jepa_references import SOURCE_HASHES, first_panel_rows, philox, recover


def test_philox_random123_zero_vector():
    # Published Philox4x32-10 all-zero counter/key known-answer vector.
    assert philox(0, 0) == (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)


def test_replay_fixed_inputs():
    rows = first_panel_rows()
    assert rows[0] == [56, 59, 10, 84, 14, 55, 35, 44, 53, 33, 48, 74]
    assert rows[3][-3:] == [11, 22, 21]
    assert [sum(row) for row in rows] == [565, 699, 764, 653]


def test_recovery_rejects_changed_sampler(tmp_path):
    (tmp_path / 'source').mkdir()
    for name in SOURCE_HASHES:
        (tmp_path / 'source' / name).write_text('changed')
    (tmp_path / 'manifest.json').write_text(json.dumps({'device': 'NVIDIA GeForce RTX 5090', 'torch': '2.13.0+cu130'}))
    with pytest.raises(ValueError, match='unsupported saved sampler'):
        recover(tmp_path)
