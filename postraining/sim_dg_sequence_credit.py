"""Toy simulation: token-level Delightful PG under sequence-level rewards.

Question: does DG's surprisal gate help or hurt when one sequence reward is
broadcast over many tokens, most of which do not cause success (the regime of
our math RL: binary reward, a few percent success, lucky guesses, and a long
low-probability token tail)?

A response is ``filler`` tokens, optionally followed by one answer token. The
estimators mirror the author's reference code (google-deepmind/egg,
``egg/losses/dg.py`` and ``reinforce.py``): sequence reward minus the prompt
group's mean reward, broadcast to every response token, coefficient
``sigmoid(U * l / eta) * U`` for DG (``U`` for PG) with ``l`` the current
policy's stop-gradient surprisal, and the loss normalized by the token count.
The ``gae`` advantage instead models our run: length-adaptive-lambda GAE over
a critic that knows only the running mean return, plus per-token value noise.

Two policy parameterizations share one initial distribution:

- ``tabular``: an independent logit delta for every (prompt, position). Every
  effect is attributable to the estimator, but nothing is shared, so DG's
  claimed benefit (reallocating a shared gradient budget across contexts, DG
  paper Prop 2) cannot appear.
- ``mlp``: the initial logits plus a residual MLP over fixed random context
  features, output layer zero-initialized, so step 0 matches ``tabular`` and
  every context competes for the same parameters.

Tasks:

- ``lottery``: reward = answer correct; filler has no causal effect at all.
- ``coherence``: reward = answer correct AND a Bernoulli draw whose success
  probability is the fraction of filler tokens inside each position's
  initial top-k "coherent" set, so junk filler is genuinely harmful but each
  junk token moves success only slightly (sigma/Delta >> 1, the gambling
  regime of Osband, "Does This Gradient Spark Joy?", section 4.2).
- ``chain``: reward = answer correct AND each of ``causal`` evenly spaced
  filler positions emits its own target token (rare but causal tokens, the
  "breakthrough" case DG is designed for); binary.
- ``chain_rare``: one causal position whose target starts at 1-10%.
- ``chain_coherence``: ``chain`` plus the junk penalty over the non-causal
  filler.
- ``graded``: egg's token-reversal analogue. Every position has a target
  token over a binary vocabulary and reward is the fraction correct; all
  tokens are causal.
- ``sequential``: the same with egg's ``reward_to_first_error`` (credit only
  up to the first wrong token) over 16 positions, the setting where the DG
  paper reports its larger gains.

Estimators are ``{gate}_{advantage}``. Besides ``pg`` and ``dg`` (fixed
temperature ``eta``), two gates set the temperature from the batch's own
advantage scale, keeping the raw advantage as the magnitude:

- ``dgrms``: ``eta * RMS(A)`` over the run's tokens.
- ``dgbase``: ``eta * b``, with ``b`` the mean reward the baseline stands for:
  the prompt's group mean for ``group``, and for ``gae`` and ``oracle`` the
  critics' running mean return (the ``gae`` critic's whole value level; a
  critic has no per-prompt constant baseline). For binary rewards a failure's
  group advantage is then ``-b``, the DG paper's ``b / eta = 1`` regime
  (Prop 1); fractional rewards (``graded``, ``sequential``) do not map onto it.

``dgrms`` includes the zero advantages of all-equal groups, so its scale
also tracks the fraction of mixed groups.

``expected_success`` is the exact expected reward for the answer tasks and
the mean per-position target probability for ``graded``/``sequential``.

Every (estimator, seed) run is one slice of a leading batch dimension: the
parameters are disjoint slices, each run's loss depends only on its own
slice, and Adam/SGD act elementwise, so batching is exact. Runs with the same
seed share their initial policy and features; all runs draw samples from one
generator.

Runs on the CPU; it executes no language model. The GPU is hidden before
torch loads because the optimizer's graph-capture health check would
otherwise open a CUDA context on the shared device.

    python -m postraining.sim_dg_sequence_credit --task coherence --policy mlp
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

ADVANTAGES = ("group", "gae", "oracle")
GATES = ("pg", "dg", "dgrms", "dgbase")
ESTIMATORS = tuple(f"{g}_{a}" for a in ADVANTAGES for g in ("pg", "dg"))


@dataclass(frozen=True)
class Task:
    name: str
    prompts: int
    filler: int
    vocab: int
    coherent_top_k: int
    filler_entropy: float
    # Initial per-prompt answer success probability, log-uniform in [lo, hi];
    # None means the per-position target tasks (graded/sequential).
    answer_success: tuple[float, float] | None
    junk_penalty: bool = False
    causal: int = 0
    causal_success: tuple[float, float] = (0.2, 0.6)


TASKS = {
    "lottery": Task("lottery", 64, 48, 256, 5, 1.0, (0.002, 0.2)),
    "coherence": Task("coherence", 64, 48, 256, 5, 1.0, (0.004, 0.4), junk_penalty=True),
    "chain": Task("chain", 64, 48, 256, 5, 1.0, (0.05, 0.5), causal=3),
    "chain_rare": Task(
        "chain_rare", 64, 48, 256, 5, 1.0, (0.1, 0.6), causal=1, causal_success=(0.01, 0.1)
    ),
    "chain_coherence": Task(
        "chain_coherence", 64, 48, 256, 5, 1.0, (0.05, 0.5), junk_penalty=True, causal=3
    ),
    "graded": Task("graded", 64, 10, 2, 1, 0.6, None),
    "sequential": Task("sequential", 64, 16, 2, 1, 0.6, None),
}


def zipf_logits(shape: tuple[int, ...], vocab: int, entropy: float, generator) -> torch.Tensor:
    """Randomly permuted Zipf logits whose exponent is bisected to ``entropy``."""
    log_ranks = torch.arange(1, vocab + 1, dtype=torch.float64).log()
    low, high = 0.0, 20.0
    for _ in range(60):
        exponent = (low + high) / 2
        probs = torch.softmax(-exponent * log_ranks, 0)
        value = float(-(probs * probs.log()).sum())
        low, high = (exponent, high) if value > entropy else (low, exponent)
    base = (-exponent * log_ranks).float()
    return base[torch.rand((*shape, vocab), generator=generator).argsort(-1)]


def set_probability(logits: torch.Tensor, target: torch.Tensor, probability: torch.Tensor) -> None:
    """Set each row's ``target`` logit in place so its probability is exact."""
    rest = logits.scatter(-1, target[..., None], float("-inf")).logsumexp(-1)
    logits.scatter_(-1, target[..., None], (rest + torch.log(probability / (1 - probability)))[..., None])


def log_uniform(bounds: tuple[float, float], shape: tuple[int, ...], generator) -> torch.Tensor:
    low, high = (math.log(p) for p in bounds)
    return torch.exp(low + (high - low) * torch.rand(shape, generator=generator))


def initial_policy(task: Task, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    filler = zipf_logits((task.prompts, task.filler), task.vocab, task.filler_entropy, generator)
    policy = {"coherent": filler.topk(task.coherent_top_k, -1).indices}
    if task.answer_success is None:
        policy["target"] = torch.randint(0, task.vocab, (task.prompts, task.filler), generator=generator)
        policy["logits"] = filler
        return policy
    # Coherence is defined by the initial top-k, before causal targets move.
    if task.causal:
        positions = torch.linspace(0, task.filler - 1, task.causal + 2)[1:-1].round().long()
        causal_target = torch.randint(0, task.vocab, (task.prompts, task.causal), generator=generator)
        sub = filler[:, positions]
        set_probability(
            sub, causal_target, log_uniform(task.causal_success, (task.prompts, task.causal), generator)
        )
        filler[:, positions] = sub
        policy.update(causal_positions=positions, causal_target=causal_target)
    answer = zipf_logits((task.prompts,), task.vocab, 1.5, generator)
    target = torch.randint(0, task.vocab, (task.prompts,), generator=generator)
    set_probability(answer, target, log_uniform(task.answer_success, (task.prompts,), generator))
    policy.update(logits=torch.cat((filler, answer[:, None]), 1), target=target)
    return policy


class Policy:
    """Initial logits plus a trainable delta, batched over runs."""

    def __init__(self, base: torch.Tensor, kind: str, args, seeds: list[int]):
        self.base, self.kind = base, kind
        n_runs, prompts, positions, vocab = base.shape
        if kind == "tabular":
            self.delta = torch.zeros_like(base, requires_grad=True)
            self.params = [self.delta]
            return
        features, first = [], []
        for seed in seeds:
            generator = torch.Generator().manual_seed(20_000 + seed)
            features.append(torch.randn(prompts * positions, args.features, generator=generator))
            first.append(
                torch.randn(args.features, args.hidden, generator=generator) / math.sqrt(args.features)
            )
        self.features = torch.stack(features)
        self.first = torch.stack(first).requires_grad_()
        self.bias = torch.zeros(n_runs, 1, args.hidden, requires_grad=True)
        self.second = torch.zeros(n_runs, args.hidden, vocab, requires_grad=True)
        self.params = [self.first, self.bias, self.second]

    def logits(self) -> torch.Tensor:
        if self.kind == "tabular":
            return self.base + self.delta
        hidden = F.gelu(torch.bmm(self.features, self.first) + self.bias)
        return self.base + torch.bmm(hidden, self.second).view_as(self.base)


def gae_advantages(values: torch.Tensor, rewards: torch.Tensor, lam: float) -> torch.Tensor:
    """GAE (gamma 1) over per-token state values; the reward lands on the last token.

    ``values`` is (runs, prompts, samples, length), the value of the state
    before each token, and ``rewards`` (runs, prompts, samples).
    """
    next_values = torch.cat((values[..., 1:], rewards[..., None]), -1)
    deltas = next_values - values
    advantages = torch.empty_like(values)
    running = torch.zeros_like(rewards)
    for t in range(values.shape[-1] - 1, -1, -1):
        running = deltas[..., t] + lam * running
        advantages[..., t] = running
    return advantages


def exclusive_prefix(x: torch.Tensor, op: str) -> torch.Tensor:
    """Length n+1: entry t reduces x[..., :t] (empty reduction at t = 0)."""
    identity = torch.ones_like(x[..., :1]) if op == "prod" else torch.zeros_like(x[..., :1])
    return torch.cat((identity, x.cumprod(-1) if op == "prod" else x.cumsum(-1)), -1)


def inclusive_suffix(x: torch.Tensor, op: str) -> torch.Tensor:
    """Length n+1: entry t reduces x[..., t:] (empty reduction at t = n)."""
    identity = torch.ones_like(x[..., :1]) if op == "prod" else torch.zeros_like(x[..., :1])
    flipped = x.flip(-1)
    reduced = (flipped.cumprod(-1) if op == "prod" else flipped.cumsum(-1)).flip(-1)
    return torch.cat((reduced, identity), -1)


def oracle_values(task: Task, probs, tokens, context) -> torch.Tensor:
    """Exact expected reward of the state before each token, under ``probs``.

    With these values, lambda-0 GAE gives every token exactly its causal
    effect V(s_{t+1}) - V(s_t) plus, on the last token, the terminal noise;
    a non-causal token's advantage is exactly zero.
    """
    filler_probs = probs[:, :, : task.filler]
    filler_tokens = tokens[..., : task.filler]
    if task.answer_success is None:
        p = filler_probs.gather(-1, context["target"][..., None]).squeeze(-1)[:, :, None]
        correct = (filler_tokens == context["target"][:, :, None]).float()
        if task.name != "sequential":
            realized = exclusive_prefix(correct, "sum")
            return ((realized + inclusive_suffix(p, "sum").expand_as(realized)) / task.filler)[..., :-1]
        # Reward (1/F) sum_k prod_{j<=k} c_j: realized prefix credit, then the
        # expected credit of the remaining run while the prefix is still correct.
        alive = exclusive_prefix(correct, "prod")
        earned = exclusive_prefix(correct.cumprod(-1), "sum")
        remaining = torch.zeros_like(p[..., :1])
        tails = [remaining]
        for j in range(task.filler - 1, -1, -1):
            remaining = p[..., j : j + 1] * (1 + remaining)
            tails.append(remaining)
        tail = torch.cat(tails[::-1], -1)
        return ((earned + alive * tail) / task.filler)[..., :-1]
    answer = probs[:, :, -1].gather(-1, context["target"][..., None]).squeeze(-1)[:, :, None, None]
    value = answer.expand(*filler_tokens.shape[:-1], task.filler + 1)
    if task.causal:
        positions, causal_target = context["causal_positions"], context["causal_target"]
        realized = torch.ones_like(filler_tokens, dtype=probs.dtype)
        realized[..., positions] = (filler_tokens[..., positions] == causal_target[:, :, None]).to(probs.dtype)
        expected = torch.ones_like(filler_probs[..., 0])
        expected[..., positions] = filler_probs[:, :, positions].gather(-1, causal_target[..., None]).squeeze(-1)
        value = value * exclusive_prefix(realized, "prod") * inclusive_suffix(expected, "prod")[:, :, None]
    if task.junk_penalty:
        noncausal = context["noncausal"].to(probs.dtype)
        in_set = (filler_tokens[..., None] == context["coherent"][:, :, None]).any(-1).to(probs.dtype)
        mass = filler_probs.gather(-1, context["coherent"]).sum(-1)
        fraction = exclusive_prefix(in_set * noncausal, "sum") + inclusive_suffix(mass * noncausal, "sum")[:, :, None]
        value = value * fraction / noncausal.sum()
    return value


def gate_temperature(
    advantage: torch.Tensor,
    rewards: torch.Tensor,
    running_mean: torch.Tensor,
    gate_mask: dict[str, torch.Tensor],
    group_mask: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    """Per-run gate temperature, broadcastable to ``advantage``.

    ``advantage`` is (runs, prompts, samples, length), ``rewards`` (runs,
    prompts, samples), ``running_mean`` (runs,), the critics' value level
    before this step's update, and the masks (runs, 1, 1, 1). A zero scale
    means every advantage it covers is zero (group) or the critic has seen no
    reward yet; the temperature falls back to ``eta`` there.
    """
    rms = advantage.square().mean((1, 2, 3), keepdim=True).sqrt()
    prompt_base = rewards.mean(-1)[..., None, None]
    critic_base = running_mean[:, None, None, None]
    base = torch.where(group_mask, prompt_base, critic_base)
    one = torch.ones_like(base)
    scale = torch.where(gate_mask["dgrms"], rms, torch.where(gate_mask["dgbase"], base, one))
    return eta * torch.where(scale > 0, scale, one)


def entropy(logits: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, -1)
    return -(logp.exp() * logp).sum(-1)


def simulate(task: Task, estimators: list[str], args) -> dict:
    for estimator in estimators:
        gate, _, kind = estimator.partition("_")
        if gate not in GATES or kind not in ADVANTAGES:
            raise ValueError(f"unknown estimator {estimator!r}")
    runs = [(estimator, seed) for estimator in estimators for seed in range(args.seeds)]
    inits = [initial_policy(task, seed) for seed in range(args.seeds)]
    stack = lambda key: torch.stack([inits[seed][key] for _, seed in runs])
    policy = Policy(stack("logits"), args.policy, args, [seed for _, seed in runs])
    coherent, target = stack("coherent"), stack("target")
    has_answer = task.answer_success is not None
    if task.causal:
        causal_positions = inits[0]["causal_positions"]
        causal_target = stack("causal_target")
        noncausal = torch.ones(task.filler, dtype=torch.bool)
        noncausal[causal_positions] = False
    else:
        noncausal = torch.ones(task.filler, dtype=torch.bool)
    optimizer = (
        torch.optim.Adam(policy.params, lr=args.lr)
        if args.optimizer == "adam"
        else torch.optim.SGD(policy.params, lr=args.lr)
    )
    gate_kind = [e.split("_", 1)[0] for e, _ in runs]
    gate_mask = {
        gate: torch.tensor([g == gate for g in gate_kind])[:, None, None, None] for gate in GATES
    }
    is_dg = ~gate_mask["pg"]
    advantage_kind = [e.split("_", 1)[1] for e, _ in runs]
    kind_mask = {
        kind: torch.tensor([k == kind for k in advantage_kind])[:, None, None, None]
        for kind in ADVANTAGES
    }
    context = {"target": target, "coherent": coherent, "noncausal": noncausal}
    if task.causal:
        context.update(causal_positions=causal_positions, causal_target=causal_target)
    n_runs, prompts, samples = len(runs), task.prompts, args.samples
    length = task.filler + has_answer
    # Horizon floor of the trainer's length_adaptive_lambda at alpha = 0.05.
    lam = 1 - 1 / max(0.05 * length, min(length, 20))
    running_mean = torch.zeros(n_runs)
    generator = torch.Generator().manual_seed(10_000)
    history = []
    for step in range(args.steps + 1):
        logp = F.log_softmax(policy.logits(), -1)  # (runs, prompts, length, vocab)
        with torch.no_grad():
            drawn = torch.multinomial(
                logp.exp().flatten(0, 2), samples, replacement=True, generator=generator
            ).view(n_runs, prompts, length, samples)
        # (runs, prompts, samples, length)
        token_logp = logp.gather(-1, drawn).transpose(2, 3)
        tokens = drawn.transpose(2, 3)
        filler_tokens = tokens[..., : task.filler]
        with torch.no_grad():
            if not has_answer:
                correct = (filler_tokens == target[:, :, None]).float()
                # egg's reward_to_first_error: credit the correct prefix only.
                rewards = (correct.cumprod(-1) if task.name == "sequential" else correct).mean(-1)
            else:
                rewards = (tokens[..., -1] == target[:, :, None]).float()
                if task.causal:
                    hits = filler_tokens[..., causal_positions] == causal_target[:, :, None]
                    rewards = rewards * hits.all(-1).float()
                if task.junk_penalty:
                    in_set = (filler_tokens[..., None] == coherent[:, :, None]).any(-1)
                    fraction = in_set[..., noncausal].float().mean(-1)
                    rewards = rewards * (torch.rand(rewards.shape, generator=generator) < fraction)
            group = (rewards - rewards.mean(-1, keepdim=True))[..., None].expand_as(token_logp)
            noise = lambda std: std * torch.randn(token_logp.shape, generator=generator)
            advantage = torch.zeros_like(token_logp)
            if kind_mask["group"].any():
                advantage = torch.where(kind_mask["group"], group, advantage)
            if kind_mask["gae"].any():
                values = running_mean[:, None, None, None] + noise(args.value_noise)
                advantage = torch.where(kind_mask["gae"], gae_advantages(values, rewards, lam), advantage)
            if kind_mask["oracle"].any():
                # A partial critic: the running mean plus a fraction of the
                # exact value structure, plus iid per-state error.
                exact = oracle_values(task, logp.exp(), tokens, context)
                mean = running_mean[:, None, None, None]
                values = mean + args.oracle_shrink * (exact - mean) + noise(args.oracle_value_noise)
                advantage = torch.where(
                    kind_mask["oracle"], gae_advantages(values, rewards, args.oracle_lambda), advantage
                )
            surprisal = -token_logp.detach()
            temperature = gate_temperature(
                advantage, rewards, running_mean, gate_mask, kind_mask["group"], args.eta
            )
            gate = torch.where(
                is_dg, torch.sigmoid(advantage * surprisal / temperature), torch.ones_like(advantage)
            )
            running_mean = 0.99 * running_mean + 0.01 * rewards.mean((1, 2))
        # Token-mean loss per run, summed across runs (disjoint parameters).
        loss = -(gate * advantage * token_logp).sum() / (prompts * samples * length)
        if step % args.log_every == 0:
            with torch.no_grad():
                probs = logp.exp()
                filler_probs = probs[:, :, : task.filler]
                coherent_mass = filler_probs.gather(-1, coherent).sum(-1)
                if not has_answer:
                    success = filler_probs.gather(-1, target[..., None]).squeeze(-1).mean(-1)
                else:
                    success = probs[:, :, -1].gather(-1, target[..., None]).squeeze(-1)
                    if task.causal:
                        causal_probs = filler_probs[:, :, causal_positions]
                        success = success * causal_probs.gather(-1, causal_target[..., None]).squeeze(-1).prod(-1)
                    if task.junk_penalty:
                        success = success * coherent_mass[..., noncausal].mean(-1)
                mixed = (rewards.amax(-1) > rewards.amin(-1)).float().mean(1)
                history.append(
                    {
                        "step": step,
                        "expected_success": success.mean(1).tolist(),
                        "filler_entropy": entropy(logp[:, :, : task.filler][..., noncausal, :]).mean((1, 2)).tolist(),
                        "coherent_mass": coherent_mass[..., noncausal].mean((1, 2)).tolist(),
                        "gate_mean": gate.mean((1, 2, 3)).tolist(),
                        # NaN when a run has no tokens of that sign this step.
                        **{
                            f"gate_{name}_mean": ((gate * side).sum((1, 2, 3)) / side.sum((1, 2, 3))).tolist()
                            for name, side in (("pos", (advantage > 0).float()), ("neg", (advantage < 0).float()))
                        },
                        "filler_surprisal": surprisal[..., : task.filler].mean((1, 2, 3)).tolist(),
                        "mixed_group_fraction": mixed.tolist(),
                        "reward_mean": rewards.mean((1, 2)).tolist(),
                    }
                )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return {"runs": runs, "history": history}


def finite_mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else math.nan


def summarize(result: dict, task: Task, args) -> None:
    runs, history = result["runs"], result["history"]
    estimators = list(dict.fromkeys(e for e, _ in runs))
    keys = ("expected_success", "filler_entropy", "coherent_mass", "gate_pos_mean", "gate_neg_mean")
    checkpoints = sorted({0, len(history) // 3, 2 * len(history) // 3, len(history) - 1})
    print(
        f"task {task.name}, {args.policy} policy, {args.optimizer} lr {args.lr:g}, "
        f"eta {args.eta:g}, {args.seeds} seeds"
    )
    for index in checkpoints:
        row = history[index]
        cells = []
        for estimator in estimators:
            slots = [i for i, (e, _) in enumerate(runs) if e == estimator]
            mean = {k: finite_mean([row[k][i] for i in slots]) for k in keys}
            cells.append(
                f"{estimator} acc {mean['expected_success']:.3f} "
                f"H {mean['filler_entropy']:.2f} coh {mean['coherent_mass']:.3f} "
                f"gate +{mean['gate_pos_mean']:.2f}/-{mean['gate_neg_mean']:.2f}"
            )
        print(f"  step {row['step']:5d} | " + " | ".join(cells))
    # Area under the accuracy curve separates speed from final accuracy.
    auc = []
    for estimator in estimators:
        slots = [i for i, (e, _) in enumerate(runs) if e == estimator]
        values = [sum(row["expected_success"][i] for i in slots) / len(slots) for row in history]
        auc.append(f"{estimator} {sum(values) / len(values):.3f}")
    print("  mean accuracy over training | " + " | ".join(auc))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--estimators", default=",".join(ESTIMATORS))
    parser.add_argument("--policy", choices=("tabular", "mlp"), default="tabular")
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--value-noise", type=float, default=0.01)
    parser.add_argument("--oracle-lambda", type=float, default=0.0)
    parser.add_argument("--oracle-value-noise", type=float, default=0.0)
    parser.add_argument("--oracle-shrink", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    task = TASKS[args.task]
    result = simulate(task, args.estimators.split(","), args)
    summarize(result, task, args)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"args": vars(args), **result}, handle)


if __name__ == "__main__":
    main()
