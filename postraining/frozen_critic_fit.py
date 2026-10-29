"""Frozen-policy token-Q fitting, with independent causal critic state.

The critic reads tokens, never the actor's beliefs or memory. For this
deterministic memory policy, token history determines its state. Gold context
is critic-only; training and evaluation positions are reasoning actions only.
Actor probabilities are cached by the rollout tool at the *actual* branch
state, including its memory and sampling distribution, never reconstructed
by a plain token forward here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import hashlib
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch import Tensor, nn

from postraining.model_io import fresh_trunk
from postraining.muon import Muon


def token_q_latents(embeddings: Tensor) -> Tensor:
    """This harness's action latents: its scratch trunk's input latents at
    unit RMS, in fp32 (the trainer's token Q reads readout rows instead,
    ``SeparateCritic.action_rows``)."""
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        embeddings = embeddings.float()
        return torch.nn.functional.rms_norm(embeddings, (embeddings.size(-1),))


@dataclass(frozen=True)
class FitConfig:
    steps: int = 1000
    eval_every: int = 20
    batch_size: int = 4
    head_learning_rate: float = 1e-4
    trunk_learning_rate: float = 2e-5
    gold_context: bool = False
    loss_weighting: str = "rollout"
    head_hidden: int = 1024
    seed: int = 42

    def __post_init__(self):
        if min(self.steps, self.eval_every, self.batch_size, self.head_hidden) < 1:
            raise ValueError("update, evaluation, batch and hidden sizes must be positive")
        if self.loss_weighting not in ("rollout", "token"):
            raise ValueError("loss weighting must be rollout or token")
        if any(not math.isfinite(lr) or lr <= 0 for lr in
               (self.head_learning_rate, self.trunk_learning_rate)):
            raise ValueError("learning rates must be positive and finite")


class ScaledTokenQHead(nn.Module):
    """The original token Q head with its inverse-dimension direction
    readout, local to this study: a relu^2 layer over the belief, zero
    readouts, and no linear belief path (the trainer's ``TokenQHead``)."""

    def __init__(self, model_dim: int, hidden: int, reads_actor_belief: bool = False):
        super().__init__()
        if reads_actor_belief:
            raise ValueError("this study's critic reads tokens only")
        self.belief = nn.Linear(model_dim, hidden, bias=False)
        self.state = nn.Linear(hidden, 1)
        self.direction = nn.Linear(hidden, model_dim, bias=False)
        with torch.no_grad():
            self.state.weight.zero_()
            self.state.bias.zero_()
            self.direction.weight.zero_()

    def forward(self, beliefs: Tensor, actor_beliefs: Tensor | None = None):
        if actor_beliefs is not None:
            raise ValueError("this study's critic reads tokens only")
        hidden = torch.nn.functional.relu(self.belief(beliefs)).square()
        return self.state(hidden).squeeze(-1), self.direction(hidden) / self.direction.out_features


class FrozenStudyCritic(nn.Module):
    def __init__(self, trunk: nn.Module, hidden: int):
        super().__init__()
        self.trunk = trunk
        self.q_head = ScaledTokenQHead(trunk.tok_emb.embedding_dim, hidden, False)

    def forward(self, token_ids: Tensor, positions: Tensor, actions: Tensor):
        beliefs = self.trunk.temporal_belief_from_token_latent(
            self.trunk.embed_tokens(token_ids)
        )
        compact = beliefs.gather(1, positions[..., None].expand(-1, -1, beliefs.size(-1)))
        with torch.autocast(device_type="cuda", enabled=False):
            states, directions = self.q_head(compact.float())
            action_latents = token_q_latents(self.trunk.embed_tokens(actions))
            values = states + (directions * action_latents).sum(-1)
        return values, states, directions

    def action_latents(self):
        tokens = torch.arange(self.trunk.tok_emb.num_embeddings,
                              device=self.trunk.tok_emb.weight.device)
        return token_q_latents(self.trunk.embed_tokens(tokens))


def masked_regression_loss(values: Tensor, targets: Tensor, mask: Tensor,
                           weighting: str) -> Tensor:
    """Equal rollout weight or equal reasoning-action weight, explicitly."""
    squared = (values - targets[:, None]).square() * mask
    if weighting == "rollout":
        return (squared.sum(-1) / mask.sum(-1)).mean()
    if weighting == "token":
        return squared.sum() / mask.sum()
    raise ValueError("unknown loss weighting")


def prepare_rollout(record: dict, gold_context: bool) -> dict:
    """Normalize the saved schema; positions index the state before an action."""
    tokens = [int(t) for t in record["token_ids"]]
    positions = [int(p) for p in record["reasoning_positions"]]
    if not positions or any(p < 0 or p + 1 >= len(tokens) for p in positions):
        raise ValueError("each rollout needs valid reasoning pre-action positions")
    context = [int(t) for t in record.get("gold_context_ids", [])] if gold_context else []
    if gold_context and not context:
        raise ValueError("gold-context arm requires gold_context_ids")
    reward = float(record["reward"])
    if reward not in (0.0, 1.0):
        raise ValueError("critic study expects binary verifier rewards")
    return {"tokens": context + tokens[:max(positions) + 2],
            "positions": [p + len(context) for p in positions],
            "actions": [tokens[p + 1] for p in positions], "reward": reward}


def _batch(records: list[dict], device: torch.device):
    # Right padding cannot affect a causal prefix; no padded action is scored.
    width = ((max(len(r["tokens"]) for r in records) + 127) // 128) * 128
    count = ((max(len(r["positions"]) for r in records) + 127) // 128) * 128
    ids = torch.zeros((len(records), width), dtype=torch.long)
    positions = torch.zeros((len(records), count), dtype=torch.long)
    actions = torch.zeros_like(positions)
    mask = torch.zeros((len(records), count), dtype=torch.float32)
    for i, record in enumerate(records):
        n = len(record["positions"])
        ids[i, :len(record["tokens"])] = torch.tensor(record["tokens"])
        positions[i, :n] = torch.tensor(record["positions"])
        actions[i, :n] = torch.tensor(record["actions"])
        mask[i, :n] = 1
    targets = torch.tensor([r["reward"] for r in records], dtype=torch.float32)
    return tuple(t.to(device) for t in (ids, positions, actions, mask, targets))


def _optimizers(critic: FrozenStudyCritic, config: FitConfig):
    # Reuse the established pretraining partition, including KDA exclusions.
    from postraining.train_latent_vapo import muon_matrix_parameters
    matrices = muon_matrix_parameters(critic.trunk.blocks, set())
    matrix_ids = {id(p) for p in matrices}
    head_ids = {id(p) for p in critic.q_head.parameters()}
    # The critic's unused LM renderer receives no gradients and no updates.
    adam = [p for p in critic.trunk.parameters() if id(p) not in matrix_ids]
    optimizers = [torch.optim.AdamW([
        {"params": adam, "lr": config.trunk_learning_rate},
        {"params": list(critic.q_head.parameters()), "lr": config.head_learning_rate},
    ], weight_decay=0.0, fused=True)]
    if matrices:
        optimizers.append(Muon(matrices, lr=config.trunk_learning_rate * (0.025 / 0.015) * 1.45))
    actual = [id(p) for opt in optimizers for g in opt.param_groups for p in g["params"]]
    expected = {id(p) for p in critic.parameters() if p.requires_grad}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise AssertionError("critic optimizer partition must cover each parameter exactly once")
    assert head_ids.isdisjoint(matrix_ids)
    return optimizers


def prepare_branch(record: dict, gold_context: bool) -> dict:
    prefix = [int(t) for t in record["prefix_tokens"]]
    context = [int(t) for t in record.get("gold_context_ids", [])] if gold_context else []
    if not prefix or (gold_context and not context):
        raise ValueError("branch prefix and requested gold context must be nonempty")
    return {"tokens": context + prefix,
            "positions": [len(context) + len(prefix) - 1],
            "actions": [int(record["forced_token"])], "reward": 0.0}


def normalize_branches(branches: list[dict]) -> list[dict]:
    """Expand rollout-tool pair records without losing exact state π."""
    flat = []
    for record in branches:
        if "actions" not in record:
            flat.append(record)
            continue
        if len(record["actions"]) != 2:
            raise ValueError("branch study requires exactly two actions per pair")
        for arm, action in enumerate(record["actions"]):
            flat.append({**record, "prefix_tokens": record["prefix_ids"],
                         "forced_token": action["token_id"], "arm": arm,
                         "rewards": action["rewards"]})
    return flat


def group_branch_states(branches: list[dict]) -> list[list[dict]]:
    """One critic forward per shared state, with both actions kept separate.

    Even a causal chunk kernel can change prefix roundoff when its future
    inputs differ. Evaluating the prefix once makes a common state baseline
    structural, rather than depending on separately rounded trunk forwards.
    """
    grouped: dict[tuple[str, str], dict[int, dict]] = {}
    for record in normalize_branches(branches):
        key = (str(record["prompt_id"]), str(record["pair_id"]))
        arms = grouped.setdefault(key, {})
        arm = int(record["arm"])
        if arm not in (0, 1) or arm in arms:
            raise ValueError("invalid or duplicate branch action")
        arms[arm] = record
    states = []
    for arms in grouped.values():
        if set(arms) != {0, 1}:
            raise ValueError("each shared state requires both action arms")
        for field in ("prefix_tokens", "policy_probabilities", "gold_context_ids", "reference_rewards"):
            if arms[0].get(field) != arms[1].get(field):
                raise ValueError(f"branch actions disagree on shared state field {field}")
        states.append([arms[0], arms[1]])
    return states


def _probabilities(record: dict, device: torch.device, vocab_size: int) -> Tensor:
    probabilities = torch.as_tensor(record["policy_probabilities"],
                                    dtype=torch.float32, device=device)
    if probabilities.shape != (vocab_size,):
        raise ValueError("branch must store the full actual policy distribution")
    if not bool(torch.isfinite(probabilities).all() & (probabilities >= 0).all()):
        raise ValueError("invalid branch policy distribution")
    if not math.isclose(float(probabilities.sum()), 1.0, abs_tol=2e-5):
        raise ValueError("branch policy probabilities do not sum to one")
    return probabilities


@torch.no_grad()
def evaluate_branches(critic: FrozenStudyCritic, forward, branches: list[dict],
                      config: FitConfig) -> list[dict]:
    """Emit reasoning-action Q and exact policy-centered A, preserving K outcomes."""
    device = critic.trunk.tok_emb.weight.device
    branch_states = group_branch_states(branches)
    critic.eval()
    output = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        latents = critic.action_latents().float()
        for start in range(0, len(branch_states), config.batch_size):
            pairs = branch_states[start:start + config.batch_size]
            records = [pair[0] for pair in pairs]
            ids, positions, actions, _, _ = _batch(
                [prepare_branch(r, config.gold_context) for r in records], device)
            _, states, directions = forward(ids, positions, actions)
            # The full table is small: only one evaluated reasoning state/row.
            with torch.autocast("cuda", enabled=False):
                scores = directions[:, 0].float() @ latents.T
                probabilities = torch.stack([
                    _probabilities(r, device, latents.size(0)) for r in records])
                means = (scores * probabilities).sum(-1)
                selected_actions = torch.tensor([[r["forced_token"] for r in pair] for pair in pairs],
                                                dtype=torch.long, device=device)
                # Add/subtract the shared baseline in double precision so a
                # tiny action gap is not lost to independent fp32 additions.
                taken = scores.gather(1, selected_actions).double()
                advantages = taken - means[:, None].double()
                policy_q = states[:, 0].double() + means.double()
                table_values = states[:, :1].double() + taken
            for i, pair in enumerate(pairs):
                for arm, record in enumerate(pair):
                    output.append({"pair_id": record["pair_id"], "arm": int(record["arm"]),
                                   "prompt_id": record["prompt_id"],
                                   "forced_token": int(record["forced_token"]),
                                   "rewards": list(record["rewards"]),
                                   "reference_rewards": list(record["reference_rewards"]),
                                   "q": float(table_values[i, arm]),
                                   "advantage": float(advantages[i, arm]),
                                   "policy_q": float(policy_q[i])})
    return output


def _append(path: Path, record: Any):
    with path.open("a") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")


def truncate_journal(path: Path, step: int) -> None:
    """Discard writes newer than the last atomically committed checkpoint."""
    if not path.exists():
        return
    lines = path.read_text().splitlines()
    retained = []
    previous = -1
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError(f"corrupt journal before final line: {path}")
            break  # An interrupted append is uncommitted.
        current = record["step"]
        if current <= previous:
            raise ValueError(f"nonmonotonic journal: {path}")
        previous = current
        if current <= step:
            retained.append(json.dumps(record, allow_nan=False) + "\n")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(retained))
    os.replace(temporary, path)


def _atomic_save(payload: dict, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def group_prediction_rows(flat: list[dict]) -> list[dict]:
    """Group action outputs for the shared prompt-clustered diagnostics."""
    pairs: dict[tuple[str, str], dict[int, dict]] = {}
    for record in flat:
        key = (str(record["prompt_id"]), str(record["pair_id"]))
        actions = pairs.setdefault(key, {})
        arm = int(record["arm"])
        if arm not in (0, 1) or arm in actions:
            raise ValueError("invalid or duplicate branch action")
        actions[arm] = record
    output = []
    for (prompt, pair), actions in pairs.items():
        if set(actions) != {0, 1}:
            raise ValueError("each branch state must contain both actions")
        if actions[0]["reference_rewards"] != actions[1]["reference_rewards"]:
            raise ValueError("both actions must share the same independent reference outcomes")
        output.append({"prompt_id": prompt, "state_id": pair,
                       "reference_rewards": actions[0]["reference_rewards"],
                       "actions": [{"token_id": actions[i]["forced_token"],
                                    "q": actions[i]["q"],
                                    "advantage": actions[i]["advantage"],
                                    "rewards": actions[i]["rewards"]} for i in (0, 1)]})
    return output


def evaluate_study_checkpoint(policy, checkpoint: str | Path,
                              branches: list[dict], output_path: str | Path) -> list[dict]:
    """Evaluate the arm chosen using tuning data, once on untouched test data."""
    backbone = getattr(policy, "backbone", policy)
    device = next(backbone.parameters()).device
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("critic evaluation requires CUDA bf16")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("schema") != "frozen_scaled_token_q_v1":
        raise ValueError("unexpected critic schema")
    config = FitConfig(**payload["config"])
    critic = FrozenStudyCritic(fresh_trunk(backbone, device), config.head_hidden).to(device)
    critic.load_state_dict(payload["critic"], strict=True)
    forward = torch.compile(critic, fullgraph=False, dynamic=True)
    predictions = evaluate_branches(critic, forward, branches, config)
    Path(output_path).write_text(json.dumps(predictions) + "\n")
    del forward, critic, payload
    torch.cuda.empty_cache()
    return predictions


def predict_saved_critic(policy, arm_output: str | Path, branches: list[dict]) -> list[dict]:
    directory = Path(arm_output)
    return evaluate_study_checkpoint(policy, directory / "critic.pt", branches,
                                     directory / "predictions_test.json")


def fit_study_arm(policy, train_rollouts: list[dict], tune_branches: list[dict],
                  output_dir: str | Path, config: FitConfig,
                  test_branches: list[dict] | None = None) -> dict:
    """Fit one fresh critic. Must be called only inside an mlq workload.

    ``policy`` is the frozen LatentThoughtModel (or raw nano backbone). Its
    weights seed only the architecture, not critic parameters. No actor forward
    is performed here. Test data is never used to select an update or arm.
    """
    backbone = getattr(policy, "backbone", policy)
    device = next(backbone.parameters()).device
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("critic fitting requires CUDA bf16; no CPU/eager fallback")
    actor = getattr(policy, "wrapper", policy)
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    actor.eval()
    if not train_rollouts or not tune_branches:
        raise ValueError("training and independent tuning branch data are required")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    configuration = asdict(config)
    config_path = directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("existing arm configuration differs; exact resume requires an identical config")
    data_digest = hashlib.sha256(json.dumps(
        {"train": train_rollouts, "tune": tune_branches}, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    result_path = directory / "result.json"
    if result_path.exists():
        completed = json.loads(result_path.read_text())
        if (completed.get("config") != configuration or completed.get("data_sha256") != data_digest
                or completed.get("steps") != config.steps):
            raise ValueError("completed arm contract does not match this request")
        if not (directory / "critic.pt").is_file():
            raise FileNotFoundError("completed arm is missing critic.pt")
        return completed
    recovery_path = directory / "recovery.pt"
    recovery = torch.load(recovery_path, map_location="cpu", weights_only=False) if recovery_path.exists() else None
    if recovery is not None:
        if (recovery.get("schema") != "frozen_critic_recovery_v1"
                or recovery.get("config") != configuration or recovery.get("data_sha256") != data_digest):
            raise ValueError("recovery checkpoint contract differs from this request")
        if not 0 <= recovery["step"] <= config.steps:
            raise ValueError("recovery checkpoint update is outside the requested budget")
        for name in ("metrics.jsonl", "predictions_tune.jsonl"):
            truncate_journal(directory / name, recovery["step"])
    elif (directory / "metrics.jsonl").exists():
        raise FileExistsError("metrics exist without a recovery checkpoint; refusing to restart the update budget")
    else:
        # Failure before the initial checkpoint can leave an uncommitted step-0
        # prediction append; recreating it does not repeat an optimizer update.
        (directory / "predictions_tune.jsonl").unlink(missing_ok=True)
    (directory / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    usable = [record for record in train_rollouts if record["reasoning_positions"]]
    corpus_diagnostics = {
        "rollouts_total": len(train_rollouts),
        "rollouts_usable": len(usable),
        "rollouts_empty_reasoning": len(train_rollouts) - len(usable),
        "reward_rate_all": sum(float(r["reward"]) for r in train_rollouts) / len(train_rollouts),
        "reward_rate_usable": (sum(float(r["reward"]) for r in usable) / len(usable)
                               if usable else None),
    }
    (directory / "corpus_diagnostics.json").write_text(json.dumps(corpus_diagnostics, indent=2) + "\n")
    if not usable:
        raise ValueError("no training rollouts contain usable reasoning actions")
    train = [prepare_rollout(record, config.gold_context) for record in usable]
    randomizer = random.Random(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    critic = FrozenStudyCritic(fresh_trunk(backbone, device), config.head_hidden).to(device)
    optimizers = _optimizers(critic, config)
    completed_step = 0
    elapsed_offset = 0.0
    first_update_done = False
    if recovery is not None:
        critic.load_state_dict(recovery["critic"], strict=True)
        if len(optimizers) != len(recovery["optimizers"]):
            raise ValueError("recovery optimizer partition changed")
        for optimizer, state in zip(optimizers, recovery["optimizers"], strict=True):
            optimizer.load_state_dict(state)
        randomizer.setstate(recovery["randomizer_state"])
        torch.set_rng_state(recovery["torch_rng_state"])
        torch.cuda.set_rng_state_all(recovery["cuda_rng_states"])
        completed_step = recovery["step"]
        elapsed_offset = recovery["elapsed_seconds"]
        first_update_done = recovery["first_update_done"]
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(directory / "tensorboard"),
                           purge_step=completed_step + 1 if recovery is not None else None)
    forward = torch.compile(critic, fullgraph=False, dynamic=True)
    if not first_update_done:
        first_weights = {name: p.detach().clone() for name, p in critic.named_parameters()}
    if recovery is None:
        initial = evaluate_branches(critic, forward, tune_branches, config)
        _append(directory / "predictions_tune.jsonl", {"step": 0, "predictions": initial})
        predictions = initial
    else:
        initial = recovery["initial_predictions"]
        predictions = recovery["predictions"]
    start_time = time.monotonic()
    def save_recovery(step: int):
        _atomic_save({"schema": "frozen_critic_recovery_v1", "config": configuration,
                      "data_sha256": data_digest, "critic": critic.state_dict(),
                      "optimizers": [opt.state_dict() for opt in optimizers],
                      "randomizer_state": randomizer.getstate(),
                      "torch_rng_state": torch.get_rng_state(),
                      "cuda_rng_states": torch.cuda.get_rng_state_all(),
                      "step": step, "elapsed_seconds": elapsed_offset + time.monotonic() - start_time,
                      "first_update_done": first_update_done,
                      "initial_predictions": initial, "predictions": predictions}, recovery_path)
    if recovery is None:
        save_recovery(0)
    del recovery
    for step in range(completed_step + 1, config.steps + 1):
        critic.train()
        batch = [train[randomizer.randrange(len(train))] for _ in range(config.batch_size)]
        ids, positions, actions, mask, targets = _batch(batch, device)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            values, _, _ = forward(ids, positions, actions)
            loss = masked_regression_loss(values, targets, mask, config.loss_weighting)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite critic loss at step {step}")
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        if not first_update_done:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                after, _, _ = forward(ids, positions, actions)
            displacement = {}
            for group in ("trunk", "q_head"):
                members = [(name, p) for name, p in critic.named_parameters()
                           if name.startswith(group + ".")]
                displacement[group + "_l2"] = math.sqrt(sum(
                    float((p.detach().float() - first_weights[name].float()).square().sum())
                    for name, p in members))
                displacement[group + "_initial_l2"] = math.sqrt(sum(
                    float(first_weights[name].float().square().sum()) for name, _ in members))
            displacement.update({"q_before_mean": float((values.detach() * mask).sum() / mask.sum()),
                                 "q_after_mean": float((after * mask).sum() / mask.sum()),
                                 "q_update_rms": float((((after - values.detach()).square() * mask).sum()
                                                         / mask.sum()).sqrt())})
            first_probe = evaluate_branches(critic, forward, tune_branches, config)
            displacement["fixed_probe_centered_advantage_update_rms"] = math.sqrt(
                sum((after["advantage"] - before["advantage"]) ** 2
                    for before, after in zip(initial, first_probe, strict=True)) / len(initial))
            (directory / "first_update.json").write_text(json.dumps(displacement, indent=2) + "\n")
            first_update_done = True
            del first_weights
        if step % config.eval_every == 0 or step == config.steps:
            from postraining.frozen_critic_gates import evaluate_predictions
            predictions = evaluate_branches(critic, forward, tune_branches, config)
            _append(directory / "predictions_tune.jsonl", {"step": step, "predictions": predictions})
            metric = {"step": step, "train_loss": float(loss.detach()),
                      "elapsed_seconds": elapsed_offset + time.monotonic() - start_time,
                      "tune_predictions": len(predictions),
                      "tune_branch_q_mse": sum((p["q"] - sum(p["rewards"]) / len(p["rewards"])) ** 2
                                               for p in predictions) / len(predictions),
                      "tune_advantage_rms": math.sqrt(sum(p["advantage"] ** 2 for p in predictions)
                                                      / len(predictions))}
            # Interim tuning diagnostics do not authorize actor updates. Final
            # arm selection and untouched-test inference remain with the driver.
            quality = evaluate_predictions(group_prediction_rows(predictions), seed=config.seed,
                                           bootstrap_samples=200, permutation_samples=200)
            for key, value in quality.items():
                if value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)):
                    metric["tune_" + key] = value
            _append(directory / "metrics.jsonl", metric)
            for key, value in metric.items():
                if key != "step" and value is not None:
                    writer.add_scalar(key, value, step)
            writer.flush()
            save_recovery(step)
            print(json.dumps({"arm": directory.name, **metric}), flush=True)
    _atomic_save({"critic": critic.state_dict(), "config": asdict(config),
                "schema": "frozen_scaled_token_q_v1", "step": config.steps},
               directory / "critic.pt")
    if test_branches:
        predictions = evaluate_branches(critic, forward, test_branches, config)
        (directory / "predictions_test.json").write_text(json.dumps(predictions) + "\n")
    result = {"steps": config.steps, "config": asdict(config),
              "data_sha256": data_digest,
              "critic_checkpoint": str(directory / "critic.pt"),
              "elapsed_seconds": elapsed_offset + time.monotonic() - start_time,
              "predictions": predictions}
    temporary_result = result_path.with_suffix(".json.tmp")
    temporary_result.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary_result, result_path)
    del optimizers, forward, critic
    writer.close()
    torch.cuda.empty_cache()
    return result
