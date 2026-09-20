"""Pure argument/metadata contracts; no model execution or CUDA allocation."""
from dataclasses import asdict

import pytest

from scripts.train_recurrent_slots import Stagnation, parse_args


def test_budget_is_explicit_and_matches_reference():
    with pytest.raises(SystemExit):
        parse_args(["--name", "contract"])
    args = parse_args(["--name", "contract", "--steps", "1000"])
    assert args.steps == 1000
    assert args.val_every == 20


@pytest.mark.parametrize("extra", [
    ["--steps", "999"], ["--steps", "1001"], ["--steps", "0"],
    ["--val-every", "10"], ["--val-every", "40"],
    ["--microbatch", "0"], ["--microbatch", "3"],
    ["--segment-size", "0"], ["--segment-size", "15"],
    ["--slots", "0"], ["--name", "../escape"], ["--name", "."],
])
def test_invalid_or_unmatched_protocol_is_rejected(extra):
    with pytest.raises(SystemExit):
        parse_args(["--name", "contract", "--steps", "1000", *extra])


def test_stagnation_requires_warmup_and_sustained_plateau():
    cull = Stagnation()
    for step in range(0, 600, 20):
        assert not cull.update(step, 2.0)
    assert cull.update(600, 2.0)


def test_improving_run_is_never_culled():
    cull = Stagnation()
    for step in range(0, 1001, 20):
        assert not cull.update(step, 2.0 - step * 0.0002)


@pytest.mark.parametrize("values", [
    # Raw BPB alone improves; EMA remains worse than its historic best.
    dict(ema=1.0, best=1.5, best_ema=1.0),
    # EMA alone improves; raw BPB remains worse than its historic best.
    dict(ema=1.8, best=1.0, best_ema=1.8),
])
def test_either_raw_or_smoothed_improvement_resets_patience(values):
    cull = Stagnation(last_improvement=200, **values)
    assert not cull.update(600, 1.4)
    assert cull.last_improvement == 600
    # One subsequent regression is never sufficient to prune.
    assert not cull.update(620, 2.0)


def test_checkpointed_cull_state_preserves_future_decisions():
    cull = Stagnation()
    for step in range(0, 601, 20):
        assert not cull.update(step, 2.0 - step * 0.0002)
    resumed = Stagnation(**asdict(cull))
    decisions = []
    for step in range(620, 1001, 20):
        decision = cull.update(step, 2.0)
        assert resumed.update(step, 2.0) == decision
        assert asdict(resumed) == asdict(cull)
        decisions.append(decision)
    assert not decisions[0]
    assert decisions[-1]


@pytest.fixture
def reference_fixture(tmp_path):
    import json
    from scripts.train_recurrent_slots import matched_reference
    args = parse_args(["--name", "contract", "--steps", "1000"])
    data, tokenizer = tmp_path / "data", tmp_path / "tokenizer.model"
    reference = dict(
        overrides=dict(DATA_PATH=str(data), TOKENIZER_PATH=str(tokenizer),
                       SEED="1337", MBS="64", VOCAB_SIZE="1024",
                       SEQ_LEN="1024", VAL_TOKENS="1048576"),
        final_val_bpb=1.32651234, final_val_step=1000,
        steps=1000, completed_steps=1000, val_every=20,
        returncode=0, training_returncode=0, metric_integrity_errors=[],
        val_entries=[dict(step=step, val_bpb=1.32651234)
                     for step in range(0, 1001, 20)])
    path = tmp_path / "reference.json"

    def check():
        path.write_text(json.dumps(reference))
        return matched_reference(args, data, tokenizer, path)

    return args, reference, check


def test_matched_reference_uses_exact_observation_and_records_evidence(reference_fixture):
    _, reference, check = reference_fixture
    bpb, matched, metadata = check()
    assert matched
    assert bpb == reference["final_val_bpb"]
    assert len(metadata["sha256"]) == 64
    assert all(metadata["matched_fields"].values())


@pytest.mark.parametrize("key,value", [
    ("completed_steps", 999), ("steps", 2000), ("val_every", 40),
    ("returncode", 1), ("training_returncode", 1),
    ("metric_integrity_errors", ["missing validation"]),
    ("final_val_bpb", float("nan")),
])
def test_incomplete_or_invalid_reference_cannot_promote(reference_fixture, key, value):
    _, reference, check = reference_fixture
    reference[key] = value
    _, matched, metadata = check()
    assert not matched
    assert not metadata["matched_fields"]["completed_reference"]


@pytest.mark.parametrize("key,value", [
    ("DATA_PATH", "/different/data"), ("TOKENIZER_PATH", "/different/tokenizer"),
    ("SEED", "42"), ("VOCAB_SIZE", "2048"),
    ("SEQ_LEN", "512"), ("VAL_TOKENS", "524288"),
])
def test_unmatched_reference_cannot_promote(reference_fixture, key, value):
    _, reference, check = reference_fixture
    reference["overrides"][key] = value
    _, matched, _ = check()
    assert not matched


@pytest.mark.parametrize("key", ["DATA_PATH", "TOKENIZER_PATH", "SEED", "MBS",
                                 "VOCAB_SIZE", "SEQ_LEN", "VAL_TOKENS"])
def test_missing_reference_contract_fails_closed(reference_fixture, key):
    _, reference, check = reference_fixture
    del reference["overrides"][key]
    with pytest.raises((KeyError, ValueError)):
        check()


def test_changed_microbatch_is_unmatched_and_records_execution_difference(reference_fixture):
    args, _, check = reference_fixture
    args.microbatch = 512
    _, matched, metadata = check()
    assert not matched
    assert not metadata["matched_fields"]["microbatch"]
    assert metadata["comparison_scope"] == "matched_update_token_and_validation_budget"
    assert metadata["reference_microbatch"] == 64
    assert metadata["run_microbatch"] == 512
    assert not metadata["same_execution_microbatch"]
    args.microbatch = 64
    assert check()[1]
    assert check()[2]["same_execution_microbatch"]
    args.seed = 42
    assert not check()[1]

@pytest.mark.parametrize("mutation", ["missing_errors", "missing_panel", "short_panel", "stale_score"])
def test_reference_requires_verified_complete_validation_panel(reference_fixture, mutation):
    _, reference, check = reference_fixture
    if mutation == "missing_errors":
        del reference["metric_integrity_errors"]
    elif mutation == "missing_panel":
        del reference["val_entries"]
    elif mutation == "short_panel":
        reference["val_entries"].pop(10)
    else:
        reference["val_entries"][-1]["val_bpb"] += 0.01
    assert not check()[1]
