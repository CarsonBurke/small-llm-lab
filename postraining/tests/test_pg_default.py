"""CLI defaults and critic-only transfer guardrails."""
import pytest
from postraining.vapo.config import build_arg_parser, validate_args


def parse(*flags):
    parser = build_arg_parser()
    args = parser.parse_args(["--checkpoint", "sft.pt", "--output", "out", *flags])
    validate_args(parser, args)
    return args


def test_pg_is_default_and_dg_remains_explicit_opt_in():
    assert not parse().delightful_policy_gradient
    assert parse("--delightful-policy-gradient").delightful_policy_gradient


def test_critic_only_transfer_keeps_sft_actor_and_skips_warmup():
    args = parse("--critic-only-init", "critic.pt", "--value-warmup-steps", "0")
    assert args.actor_init is None and args.actor_critic_init is None and args.resume is None
    assert not args.delightful_policy_gradient


@pytest.mark.parametrize("flags", [[], ["--value-warmup-steps", "0", "--resume", "resume.pt"],
    ["--value-warmup-steps", "0", "--actor-init", "actor.pt"],
    ["--value-warmup-steps", "0", "--critic-init", "actor"]])
def test_critic_only_transfer_rejects_conflicting_initialization(flags):
    with pytest.raises(SystemExit):
        parse("--critic-only-init", "critic.pt", *flags)
