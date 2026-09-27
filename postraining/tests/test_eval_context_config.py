"""Frozen evaluation can explicitly extrapolate without changing training budgets."""

import pytest

from postraining.core import validate_posttraining_context_budget
from postraining.reasoning_modes import mode_rollout_budget, training_rollout_budget
from postraining.vapo.config import build_arg_parser, resolve_rollout_defaults, validate_args


def parse(*extra):
    parser = build_arg_parser()
    args = parser.parse_args(["--checkpoint", "checkpoint.pt", "--output", "out", *extra])
    validate_args(parser, args)
    return args


@pytest.mark.parametrize("mode", ["--rollout-only", "--bench-only"])
def test_context_extension_requires_frozen_evaluation(mode):
    args = parse(mode, "--eval-context-tokens", "4352")
    assert args.eval_context_tokens == 4352


@pytest.mark.parametrize("flags", [[], ["--bpb-only"], ["--rollout-only", "--eval-context-tokens", "0"]])
def test_invalid_context_extension_rejected(flags):
    with pytest.raises(SystemExit):
        parse("--eval-context-tokens", "4352", *flags)


def test_training_retains_checkpoint_context_by_default():
    assert parse().eval_context_tokens is None



def resolve(*extra, trained=1024, checkpoint=1024):
    args = parse(*extra)
    context = resolve_rollout_defaults(args, is_nano=True,
                                      checkpoint_context_tokens=checkpoint,
                                      trained_context_tokens=trained)
    return args, context


def test_new_cot_default_fits_2048_responses_and_matching_evaluations():
    args, context = resolve('--reasoning-mode', 'cot', '--prompt-tokens', '256')
    assert args.steps + args.value_warmup_steps == 1000
    assert args.continuation_tokens == args.bench_max_tokens == args.aime_max_tokens == 2048
    assert context == 2304
    emitted, stream = training_rollout_budget('cot', args.continuation_tokens,
                                              answer_tokens=24, prompt_tokens=256,
                                              context_tokens=context)
    assert emitted == stream == 2048
    validate_posttraining_context_budget(256, stream, context)


def test_evaluations_inherit_explicit_rollout_budget_but_allow_override():
    args, context = resolve('--reasoning-mode', 'cot', '--continuation-tokens', '768',
                            '--prompt-tokens', '256', '--aime-max-tokens', '2048')
    assert args.bench_max_tokens == 768
    assert args.aime_max_tokens == 2048
    assert context == 2304


def test_explicit_historical_cot_budget_preserves_old_context():
    args, context = resolve('--reasoning-mode', 'cot', '--continuation-tokens', '768',
                            '--prompt-tokens', '256')
    assert context == 1024
    assert args.bench_max_tokens == args.aime_max_tokens == 768


def test_actual_sft_window_is_honored_separately_from_pretraining_metadata():
    _, context = resolve('--reasoning-mode', 'cot', trained=5120)
    assert context == 5120


def test_latent_default_retains_thought_capacity_and_fits_evaluation():
    args, context = resolve('--reasoning-mode', 'latent')
    assert context == 4608
    assert training_rollout_budget('latent', args.continuation_tokens, answer_tokens=24,
                                  prompt_tokens=args.prompt_tokens, context_tokens=context) == (2048, 4096)
    assert mode_rollout_budget('latent', args.aime_max_tokens, answer_tokens=24,
                              prompt_tokens=args.prompt_tokens, context_tokens=context) == (2048, 4096)


def test_explicit_frozen_context_remains_a_hard_limit():
    args, context = resolve('--reasoning-mode', 'cot', '--rollout-only',
                            '--eval-context-tokens', '1024')
    assert context == 1024
    with pytest.raises(ValueError, match='exceeds'):
        validate_posttraining_context_budget(args.prompt_tokens, args.continuation_tokens, context)


def test_answer_only_does_not_extend_context_for_unused_response_cap():
    _, context = resolve('--reasoning-mode', 'none', '--no-think-tokens', '--no-answer-fence')
    assert context == 1024
