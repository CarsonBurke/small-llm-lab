"""One-time v1 -> v2 combiner surgery on a training checkpoint.

The v1 combiner injected ``gain * W(h)`` with ``gain`` zero-init — a
multiplicative saddle: W's gradient is scaled by the near-zero gain while
the gain's gradient is an inner product with a frozen random projection, so
neither factor escapes (observed: gain oscillating at ~1e-4 for 8k+ steps,
W frozen at init). v2 removes the gain and zero-inits W, whose gradient is
then the full-rank loss-direction-outer-hidden from the first step.

The fold ``W' := gain * W`` is exactly function-preserving, so a mid-run
checkpoint converts without losing the trunk/critic/type_bias training that
already happened. Optimizer surgery on both AdamW state dicts:

- The gain's state entry is removed and every higher parameter index shifts
  down by one (indices are global across param groups in construction
  order: actor = [trunk..., combiner..., renderer...], critic = one group).
- The carry matrix keeps its index but DROPS its Adam moments: they were
  accumulated in the gain-scaled gradient parameterization and are
  meaningless for W'. AdamW treats a missing state entry as a
  never-stepped parameter and lazily reinitializes.

The gain is identified as the unique scalar-shaped state entry per
optimizer (the only 0-dim parameter in the KDA/GPT-2-vocab lineup; a tied
nano backbone's ``readout_scale`` would add a second scalar to the actor's
renderer group and make the scan raise loudly rather than guess), cross-checked
against its known construction position in the actor's combiner group
(``param_groups[1]["params"][0]``). ``Module.parameters()`` yields a
module's direct ``nn.Parameter`` attributes before its children's, so the
v1 combiner order is [gain, type_bias, carry.weight, norms..., mlps...] and
the carry matrix sits two slots after the gain. Removing the gain leaves
[type_bias, carry.weight, ...] — exactly the v2 construction order, which
is what makes the plain shift-down remap sound.

Usage:
    python -m postraining.fold_combiner_gain <in.pt> <out.pt>
"""

from __future__ import annotations

import os
import sys

import torch

from postraining.latent_thought import THOUGHT_INPUT_SCHEMA

V1_SCHEMA = "gated_hidden_residual_prenorm_mlp/v1"


def fold_model_state(state: dict, prefix: str = "combiner.") -> None:
    gain = state.pop(prefix + "gain")
    if gain.ndim != 0:
        raise ValueError(f"{prefix}gain is not a scalar: {tuple(gain.shape)}")
    key = prefix + "carry.weight"
    state[key] = (gain.double() * state[key].double()).to(state[key].dtype)
    print(f"  {key}: folded gain {float(gain):+.6e}, "
          f"new rms {float(state[key].square().mean().sqrt()):.3e}")


def remap_optimizer_state(sd: dict, expected_gain_index: int | None) -> None:
    scalars = [
        index
        for index, entry in sd["state"].items()
        if entry["exp_avg"].ndim == 0
    ]
    if len(scalars) != 1:
        raise ValueError(
            f"expected exactly one scalar parameter (the gain), found state "
            f"indices {scalars}"
        )
    gain_index = scalars[0]
    if expected_gain_index is not None and gain_index != expected_gain_index:
        raise ValueError(
            f"scalar state entry at index {gain_index} but construction "
            f"order places the gain at {expected_gain_index}"
        )
    carry_index = gain_index + 2
    carry_entry = sd["state"].get(carry_index)
    if carry_entry is not None and (
        carry_entry["exp_avg"].ndim != 2
        or carry_entry["exp_avg"].shape[0] != carry_entry["exp_avg"].shape[1]
    ):
        raise ValueError(
            f"index {carry_index} should be the square carry matrix, got "
            f"{tuple(carry_entry['exp_avg'].shape)}"
        )
    sd["state"] = {
        (index if index < gain_index else index - 1): entry
        for index, entry in sd["state"].items()
        if index not in (gain_index, carry_index)
    }
    for group in sd["param_groups"]:
        group["params"] = [
            index if index < gain_index else index - 1
            for index in group["params"]
            if index != gain_index
        ]
    print(f"  dropped gain index {gain_index}, reset carry moments at "
          f"{carry_index}")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: python -m postraining.fold_combiner_gain <in.pt> <out.pt>"
        )
    source, target = sys.argv[1], sys.argv[2]
    if os.path.realpath(source) == os.path.realpath(target):
        raise SystemExit(
            "refusing to overwrite the source checkpoint; write the folded "
            "copy to a new path and keep the v1 original"
        )
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("thought_input_schema") != V1_SCHEMA:
        raise ValueError(
            f"checkpoint schema is {payload.get('thought_input_schema')!r}, "
            f"can only fold {V1_SCHEMA!r}"
        )
    print("model (actor):")
    fold_model_state(payload["model"])
    print("critic:")
    fold_model_state(payload["critic"])
    print("actor optimizer:")
    actor_sd = payload["optimizers"]["actor"]
    remap_optimizer_state(
        actor_sd, expected_gain_index=actor_sd["param_groups"][1]["params"][0]
    )
    print("critic optimizer:")
    remap_optimizer_state(payload["optimizers"]["critic"], None)
    for extra in set(payload["optimizers"]) - {"actor", "critic"}:
        if extra not in ("actor_muon", "critic_muon"):
            raise ValueError(
                f"unexpected optimizer {extra!r}; this fold only handles "
                "the adamw and muon trunk layouts"
            )
        # Muon holds only block matrices (the combiner is not under
        # ``backbone.blocks``), so its state passes through unchanged;
        # assert that no scalar snuck in. Muon's state keys differ from
        # AdamW's ("momentum", not "exp_avg"), so scan every tensor value.
        if any(
            isinstance(value, torch.Tensor) and value.ndim == 0
            for entry in payload["optimizers"][extra]["state"].values()
            for value in entry.values()
        ):
            raise ValueError(f"{extra} unexpectedly holds a scalar parameter")
        print(f"{extra}: untouched (no combiner parameters)")
    payload["thought_input_schema"] = THOUGHT_INPUT_SCHEMA
    payload["args"].pop("hidden_carry_gain_init", None)
    torch.save(payload, target)
    print(f"wrote {target} with schema {THOUGHT_INPUT_SCHEMA!r}")


if __name__ == "__main__":
    main()
