#!/usr/bin/env python3
"""Policy and critic at fixed decimal decision states across a MAC run.

Run through mlq. States come from one trace capture (by default the start
policy's), so every snapshot is read on identical prefixes: the change in
pi(a|s) and in the critic's policy-centered advantage A(s,a) is the
snapshot's, not a change of states. Two decisions are read:

* answer_start: the first token after <answer> in traces whose result line
  already fixes the answer (correct whole-number result and point count).
  The correct first token is known; the question is whether the critic ranks
  it above the policy's alternatives.
* setup_wording: the token after "Drop the points and add" in the taught
  "add whole numbers" line, where the run drifted to " up".

For each snapshot the actor is its policy and the critic is the critic saved
with it, as the run's MAC update read them. The SFT start is read with the
study critic. Teacher forcing runs the plain causal trunk forward, which is
the cot policy's own readout (the nano renderer is belief-only).
"""
from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal
from pathlib import Path
import sys

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_gpt as baseline  # noqa: F401  (import order: patches must load first)
from postraining.core import encode_prompt, load_posttraining_tokenizer
from postraining.latent_thought import LatentThoughtModel, combiner_init_kwargs_from_checkpoint
from postraining.model_io import copy_trunk, load_model
from postraining.nano_backbone import rational_softcap
from postraining.value_model import SeparateCritic
from pretraining.fresh_lejepa.fresh_lejepa_train import FreshHyperparameters

ANSWER_OPEN = 50259
TOP = 8


def places(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else 0


def determined_answer(attempt: dict) -> bool:
    """The trace's result line fixes the correct answer: only rendering is left."""
    family = attempt["dataset_index"].split("/")[1]
    text = attempt["emitted_text"]
    a, b = re.findall(r"-?\d+(?:\.\d+)?", attempt["prompt"])[:2]
    if family in ("add_decimal", "sub_decimal"):
        k = max(places(a), places(b))
        A, B = int(Decimal(a) * 10**k), int(Decimal(b) * 10**k)
        want = A + B if family == "add_decimal" else A - B
        found = re.search(r"whole-number (?:sum|difference) is (-?\d+); put the point back (\d+) place", text)
    elif family == "mul_decimal":
        k = places(a) + places(b)
        want = int(a.replace(".", "")) * int(b.replace(".", ""))
        found = re.search(r"Place the point (\d+) digits? from the right of (-?\d+)", text)
        if found:
            return int(found.group(1)) == k and int(found.group(2)) == want
        return False
    else:
        return False
    return bool(found) and int(found.group(1)) == want and int(found.group(2)) == k


def build_states(attempts: list[dict], tokenizer, prompt_tokens: int, per_kind: int):
    states = []
    whole = tokenizer.encode(" whole")
    up = tokenizer.encode(" up")
    if len(whole) != 1 or len(up) != 1:
        raise ValueError("wording candidates must be single tokens")
    for attempt in attempts:
        prompt = encode_prompt(tokenizer, attempt["prompt"], prompt_tokens)
        emitted = [int(t) for t in attempt["emitted_token_ids"]]
        family = attempt["dataset_index"].split("/")[1]
        if "decimal" in family and determined_answer(attempt) and ANSWER_OPEN in emitted:
            cut = emitted.index(ANSWER_OPEN) + 1
            truth = tokenizer.encode(str(attempt["ground_truth"]))
            states.append({"kind": "answer_start", "family": family,
                           "ids": prompt + emitted[:cut], "correct": truth[0],
                           "prompt": attempt["prompt"], "truth": attempt["ground_truth"]})
        if family == "add_decimal":
            marker = tokenizer.encode("Drop the points and add")
            for start in range(len(emitted) - len(marker)):
                if emitted[start:start + len(marker)] == marker:
                    cut = start + len(marker)
                    if emitted[cut] == whole[0]:
                        states.append({"kind": "setup_wording", "family": family,
                                       "ids": prompt + emitted[:cut],
                                       "candidates": {"whole": whole[0], "up": up[0]},
                                       "prompt": attempt["prompt"]})
                    break
    selected, counts = [], {}
    for state in states:
        key = (state["kind"], state["family"])
        if counts.get(key, 0) < per_kind:
            counts[key] = counts.get(key, 0) + 1
            selected.append(state)
    return selected, counts


@torch.no_grad()
def read_state(actor, critic, latents, ids, device):
    tokens = torch.tensor([ids], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        belief = actor.backbone.temporal_belief_from_token_latent(actor.backbone.embed_tokens(tokens))
        critic_belief = critic.trunk.temporal_belief_from_token_latent(critic.trunk.embed_tokens(tokens))
    with torch.autocast("cuda", enabled=False):
        logits = rational_softcap(actor.backbone._raw_logits(belief[0, -1:]).float(),
                                  actor.backbone.logit_softcap)[0]
        policy = logits.softmax(-1)
        # The actor reads the twins' mean, which for a bilinear Q is the
        # mean of the heads' state values and directions.
        heads = [head(critic_belief[0, -1:].float(), None) for head in critic.q_heads]
        state = torch.stack([value for value, _ in heads]).mean(0)
        direction = torch.stack([direction for _, direction in heads]).mean(0)
        q = state[0] + latents @ direction[0]
        advantage = q - (policy * q).sum()
    return policy, advantage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--study-critic", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--per-kind", type=int, default=96)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    manifest = json.loads((args.run / "manifest.json").read_text())
    run_args = manifest["args"]
    base = load_model(run_args["checkpoint"], device)
    tokenizer = load_posttraining_tokenizer(
        base.architecture, FreshHyperparameters.tokenizer_path, think_tokens=True,
        answer_tokens=True, tokenizer_provenance=base.model_config.get("tokenizer_provenance"))
    attempts = json.loads(args.traces.read_text())["attempts"]
    states, counts = build_states(attempts, tokenizer, int(run_args["prompt_tokens"]), args.per_kind)
    print({f"{k}/{f}": v for (k, f), v in counts.items()}, flush=True)
    # Any of the run's checkpoints carries its policy geometry; step 0 alone
    # (a study critic on the start policy) needs no snapshot.
    snapshots = [step for step in args.steps if step]
    first = torch.load(args.run / "snapshots" / f"step_{max(snapshots):06d}.pt" if snapshots
                       else args.run / "latent_vapo_checkpoint.pt",
                       map_location="cpu", weights_only=False, mmap=True)
    actor = LatentThoughtModel(base, **combiner_init_kwargs_from_checkpoint(first)).to(device).eval()
    del first
    # The SFT policy, before any snapshot overwrites the shared actor.
    start_state = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    critic = SeparateCritic(copy_trunk(base, device),
                            q_head_hidden=int(run_args["critic_q_head_hidden"]),
                            q_twin=bool(run_args.get("critic_q_twin", False))).to(device).eval()
    results = []
    for step in args.steps:
        if step == 0:
            actor.load_state_dict(start_state, strict=True)
            critic.load_state_dict(torch.load(args.study_critic, map_location="cpu", weights_only=False)["critic"], strict=True)
        else:
            payload = torch.load(args.run / "snapshots" / f"step_{step:06d}.pt", map_location="cpu", weights_only=False)
            actor.load_state_dict(payload["model"], strict=True)
            critic.load_state_dict(payload["critic"], strict=True)
            del payload
        latents = critic.action_latents().float()
        for index, state in enumerate(states):
            policy, advantage = read_state(actor, critic, latents, state["ids"], device)
            top = torch.topk(policy, TOP).indices
            record = {"step": step, "state": index, "kind": state["kind"], "family": state["family"],
                      "top_tokens": top.tolist(), "top_probabilities": policy[top].tolist(),
                      "top_advantages": advantage[top].tolist(),
                      "mac_push_top": (policy[top] * advantage[top]).tolist()}
            if state["kind"] == "answer_start":
                c = state["correct"]
                record.update(correct=c, correct_probability=float(policy[c]),
                              correct_advantage=float(advantage[c]),
                              critic_prefers_correct=bool(int(top[advantage[top].argmax()]) == c),
                              policy_prefers_correct=bool(int(top[0]) == c))
            else:
                for name, token in state["candidates"].items():
                    record[f"{name}_probability"] = float(policy[token])
                    record[f"{name}_advantage"] = float(advantage[token])
            results.append(record)
        print(f"step {step} read {len(states)} states", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "mac_decision_state_probe/v1", "run": str(args.run),
                                       "traces": str(args.traces), "states": [
                                           {k: v for k, v in s.items() if k != "ids"} | {"length": len(s["ids"])}
                                           for s in states], "results": results}) + "\n")


if __name__ == "__main__":
    main()
