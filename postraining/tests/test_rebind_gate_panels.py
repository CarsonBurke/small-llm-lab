import json
import sys

import pytest

from scripts.rebind_posttraining_gate_panels import main, sha256


def fixture_args(tmp_path, monkeypatch):
    panel = tmp_path / "panel.parquet"
    panel.write_bytes(b"unchanged panel bytes")
    corpus = tmp_path / "corpus.parquet"
    corpus.write_bytes(b"different SFT lineage")
    templates = tmp_path / "templates"
    templates.mkdir()
    manifest = {
        "sources": [{"path": str(panel), "sha256": sha256(panel), "rows": 512}],
        "sft_corpus": "old.parquet", "sft_corpus_sha256": "oldhash",
        "diagnostic_panel": {"purpose": "RL-pool diagnostic"},
    }
    template = templates / "math.manifest.json"
    template.write_text(json.dumps(manifest))
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        "rebind", "--template-panel-dir", str(templates),
        "--sft-corpus", str(corpus), "--output", str(output), "--sources", "math",
    ])
    return panel, corpus, template, output, manifest


def test_rebinding_preserves_panel_contract_and_existing_output(tmp_path, monkeypatch):
    _, corpus, template, output, original = fixture_args(tmp_path, monkeypatch)
    main()
    result = json.loads((output / "math.manifest.json").read_text())
    assert result["sources"] == original["sources"]
    assert result["diagnostic_panel"] == original["diagnostic_panel"]
    assert result["sft_corpus_sha256"] == sha256(corpus)
    assert json.loads(template.read_text()) == original
    with pytest.raises(FileExistsError):
        main()


def test_changed_panel_fails_before_creating_output(tmp_path, monkeypatch):
    panel, _, _, output, _ = fixture_args(tmp_path, monkeypatch)
    panel.write_bytes(b"mutated panel")
    with pytest.raises(ValueError, match="panel hash changed"):
        main()
    assert not output.exists()
