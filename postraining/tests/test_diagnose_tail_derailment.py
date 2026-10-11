"""CPU contracts of ``postraining.diagnose_tail_derailment``."""

from __future__ import annotations

import math

import pytest
import torch

from postraining.diagnose_tail_derailment import (
    JUNK_PROBABILITY,
    STEP_FIELDS,
    SURPRISAL_EDGES,
    aligned_row_steps,
    bin_index,
    credit_cells,
    derail_onset,
    memory_read_disabled,
    parse_arm,
    recorded_sampling,
    step_statistics,
)
from postraining.latent_rollout import (
    rollout_continuations,
    split_rollout_groups,
    trim_stream,
)
from postraining.tests.test_hidden_carry import _prompts
from postraining.tests.test_memory_carry import _memory_wrapper

STOP_IDS = tuple(range(1, 9))
THINK_ID = 9


def test_parse_arm_options():
    arm = parse_arm("dg=runs/a.pt,nomem,top_p=0.95")
    assert (arm.label, str(arm.path), arm.nomem, arm.top_p) == ("dg", "runs/a.pt", True, 0.95)
    assert parse_arm("sft=b.pt").top_p == 1.0
    for bad in ("nolabel", "x=a.pt,bogus", "x=a.pt,top_p=0"):
        with pytest.raises(ValueError):
            parse_arm(bad)


def test_step_statistics_definitions():
    probs = torch.tensor([[0.5, 0.3, 0.2 - 2e-5, 1e-5, 1e-5]])
    logits = probs.log() + 3.0  # softmax is shift-invariant
    stats = step_statistics(logits, torch.tensor([1]), think_id=2, stop_ids=(3, 4))
    field = {name: stats[0, i].item() for i, name in enumerate(STEP_FIELDS)}
    assert field["entropy"] == pytest.approx(float(-(probs * probs.log()).sum()), rel=1e-5)
    assert field["max_prob"] == pytest.approx(0.5, rel=1e-5)
    assert field["junk_mass"] == pytest.approx(2e-5, rel=1e-3)
    assert 1e-5 < JUNK_PROBABILITY
    assert field["think_logprob"] == pytest.approx(math.log(0.2 - 2e-5), rel=1e-5)
    assert field["stop_logprob"] == pytest.approx(math.log(2e-5), rel=1e-4)
    assert field["sampled_logprob"] == pytest.approx(math.log(0.3), rel=1e-5)


def test_derail_onset_is_the_first_full_window_end():
    surprisal = torch.tensor([1.0] * 10 + [9.0] * 8 + [1.0] * 4)
    # The first 4-token window with mean > 5 ends at index 12 (mean 7).
    assert derail_onset(surprisal, window=4, nats=5.0) == 12
    assert derail_onset(torch.ones(30), window=4, nats=5.0) is None
    assert derail_onset(torch.full((3,), 20.0), window=4, nats=5.0) is None


def test_credit_cells_bin_dg_and_pg_coefficients():
    surprisal = torch.tensor([0.1, 5.0, 12.0, 20.0])
    advantages = torch.tensor([0.2, -0.1, 0.05, -0.3])
    cells = credit_cells(surprisal, advantages, correct=False)
    assert cells["outcome"] == "failed"
    bins = bin_index(surprisal, SURPRISAL_EDGES).tolist()
    # [0, 0.5) is bin 0, [4, 6) bin 4, [10, 13) bin 7, [13, inf) bin 8.
    assert bins == [0, 4, 7, 8]
    assert sum(cells["tokens"]) == 4
    expected_dg = torch.sigmoid(advantages * surprisal) * advantages
    assert cells["dg"][7] == pytest.approx(float(expected_dg[2]), rel=1e-6)
    assert cells["dg"][8] == pytest.approx(float(expected_dg[3]), rel=1e-6)
    assert cells["pg"][4] == pytest.approx(-0.1, rel=1e-6)
    assert cells["positive"][7] == 1.0 and cells["positive"][8] == 0.0


def _recorded_rollout(wrapper, prompts, samples, new_tokens=12, seed=7):
    width = max(p.numel() for p in prompts)
    prompt_ids = torch.zeros((len(prompts), width), dtype=torch.long)
    for index, prompt in enumerate(prompts):
        prompt_ids[index, width - prompt.numel():] = prompt
    lengths = torch.tensor([p.numel() for p in prompts])
    record: list[torch.Tensor] = []
    with recorded_sampling(THINK_ID, STOP_IDS, record), torch.no_grad():
        batched = rollout_continuations(
            wrapper,
            prompt_ids,
            new_tokens,
            new_tokens,
            1.0,
            1.0,
            generator=torch.Generator().manual_seed(seed),
            stop_ids=STOP_IDS,
            prompt_lengths=lengths,
            pin_emit=True,
            hidden_carry=True,
            record_likelihoods=True,
            compact_finished=False,
            prompt_repeats=samples,
        )
    groups = [
        trim_stream(group)
        for group in split_rollout_groups(batched, samples, lengths.repeat_interleave(samples))
    ]
    return torch.stack(record), groups


def test_step_record_aligns_with_split_left_padded_groups():
    wrapper = _memory_wrapper()
    prompts = [_prompts("nano", 1, length, seed)[0] for length, seed in ((7, 11), (4, 12))]
    samples = 3
    steps, groups = _recorded_rollout(wrapper, prompts, samples)
    lengths = set()
    for group_index, group in enumerate(groups):
        for member in range(samples):
            slots, row_steps = aligned_row_steps(
                steps, group_index * samples + member, group.emit_mask[member], group.old_token_logprobs[member]
            )
            assert row_steps.shape == (slots.numel(), len(STEP_FIELDS))
            lengths.add(slots.numel())
    # Early stops make rows unequal, so alignment is not vacuous.
    assert len(lengths) > 1
    # The wrong row is caught.
    group = groups[1]
    with pytest.raises(RuntimeError, match="misaligned"):
        for member in range(samples):
            aligned_row_steps(steps, member, group.emit_mask[member], group.old_token_logprobs[member])


def test_memory_read_disabled_zeroes_and_restores():
    wrapper = _memory_wrapper()
    weight = wrapper.combiner.reader.out.weight
    before = weight.detach().clone()
    assert before.abs().sum() > 0
    with memory_read_disabled(wrapper):
        assert torch.count_nonzero(weight) == 0
    assert torch.equal(weight, before)


def test_summarize_reports_derailment_and_credit():
    from postraining.diagnose_tail_derailment import summarize

    def steps(surprisal):
        out = torch.zeros(len(surprisal), len(STEP_FIELDS))
        out[:, STEP_FIELDS.index("sampled_logprob")] = -torch.tensor(surprisal)
        out[:, STEP_FIELDS.index("entropy")] = torch.tensor(surprisal)
        out[:, STEP_FIELDS.index("think_logprob")] = math.log(0.5)
        return out.half()

    sane = [1.0] * 40
    derailed = [1.0] * 20 + [9.0] * 60
    rows, tokens, credit = [], [], []
    for index, (surprisal, ended, correct) in enumerate(
        ((sane, True, True), (derailed, False, False), ([], False, False))
    ):
        rows.append(
            {
                "source": "math" if index < 2 else "code",
                "correct": correct,
                "ended": ended,
                "emitted": len(surprisal),
                "first_is_think": index == 0,
                "think_closed": index == 0,
                "onset": derail_onset(torch.tensor(surprisal)) if surprisal else None,
            }
        )
        tokens.append({"steps": steps(surprisal) if surprisal else torch.zeros(0, len(STEP_FIELDS)).half()})
        if surprisal:
            credit.append(credit_cells(torch.tensor(surprisal), torch.full((len(surprisal),), 0.1), correct))
    summary = summarize(rows, tokens, credit)
    assert summary["derailed"] == pytest.approx(1 / 3)
    assert summary["derailed_ended"] == 0.0
    assert summary["onset_quantiles"][0.5] == rows[1]["onset"]
    assert summary["step0_think_probability"] == pytest.approx(0.5, rel=1e-3)
    assert summary["credit"]["correct"][2]["tokens"] == 40
    assert summary["credit"]["failed"][6]["tokens"] == 60
    assert summary["profile_derailed"]["64-256"]["tokens"] == 16
    assert summary["by_source"]["code"]["rows"] == 1


MIXTURE = "postraining/data/reviewed_candidate_20260926_v2/mixture.manifest.json"


@pytest.mark.skipif(not __import__("pathlib").Path(MIXTURE).exists(), reason="mixture data not present")
def test_panel_prompts_covers_every_verifier_in_the_run_mixture():
    from postraining.diagnose_tail_derailment import panel_prompts

    panel = panel_prompts(MIXTURE, 64, 20261010, answer_fence=True)
    assert len(panel) == 64
    kinds = {row.get("_verifier_kind", "math") for _, row in panel}
    assert kinds <= {"math", "python_mbpp"} and "python_mbpp" in kinds
    assert panel == panel_prompts(MIXTURE, 64, 20261010, answer_fence=True)
