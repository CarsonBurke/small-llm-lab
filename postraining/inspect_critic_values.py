"""Roll trained-policy answers and show the critic's value at every token.

Loads a latent-VAPO checkpoint (policy AND critic), samples distinct math
prompts through the exact training rollout path (same encoding, budgets,
pinning, and verifier scoring), then reports the critic's per-slot value
estimates over each trajectory — the numbers GAE consumes as V(s).

    python3 -m postraining.inspect_critic_values \
        --wrapper-checkpoint postraining/runs/<name>/latent_vapo_checkpoint.pt \
        --rows 16 --samples 1

Writes an HTML report (tokens colored by value) plus a JSON dump next to the
checkpoint under critic_inspection/, and prints a per-prompt summary.
"""

from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path

import torch

import train_gpt as baseline  # noqa: F401  (import order: patches must load first)
from fresh_lejepa_train import FreshHyperparameters
from postraining.core import (
    answer_style,
    encode_prompt,
    load_posttraining_tokenizer,
    load_unique_math_rows,
)
from postraining.latent_rollout import (
    PAD_SLOT,
    THOUGHT_SLOT,
    TOKEN_SLOT,
    rollout_continuations,
    trim_stream,
)
from postraining.latent_thought import (
    LatentThoughtModel,
    migrate_legacy_wrapper_checkpoint,
    rollout_policy_schema_for_mode,
    validate_renderer_checkpoint,
    wrapper_init_kwargs_from_checkpoint,
)
from postraining.hl_gauss import anchored_unit_geometry
from postraining.model_io import fresh_trunk, load_model
from postraining.train_latent_vapo import score_math_rollout
from postraining.train_vapo import prompt_text
from postraining.value_model import SeparateCritic


def value_color(value: float, low: float, high: float) -> str:
    """Red (low) -> yellow -> green (high) on the report's value range."""
    span = max(high - low, 1e-9)
    unit = min(max((value - low) / span, 0.0), 1.0)
    hue = 120 * unit  # 0 red .. 120 green
    return f"hsl({hue:.0f} 70% 42%)"


def gpt2_unicode_to_bytes() -> dict[str, int]:
    """Invert GPT-2's bytes_to_unicode table (the byte-level BPE alphabet)."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    chars = printable[:]
    shift = 0
    for byte in range(256):
        if byte not in printable:
            chars.append(256 + shift)
            shift += 1
    ordered_bytes = printable + [b for b in range(256) if b not in printable]
    return {chr(c): b for c, b in zip(chars, ordered_bytes)}


def token_display_text(tokenizer, token_id: int, unicode_to_byte) -> str:
    """Decode ONE token to display text, byte-exact.

    A lone BPE token can hold a fragment of a multi-byte UTF-8 character;
    ``decode([id])`` then yields U+FFFD.  Recover the token's raw bytes via
    the GPT-2 byte alphabet and show undecodable fragments as ``⟨HH⟩`` hex
    markers instead.  Falls back to plain decode for non-GPT-2 tokenizers.
    """
    piece = getattr(tokenizer, "id_to_piece", None)
    if piece is None or unicode_to_byte is None:
        return tokenizer.decode([token_id])
    text = piece(token_id)
    if any(ch not in unicode_to_byte for ch in text):
        return text  # special token like <|endoftext|>: show it literally
    raw = bytes(unicode_to_byte[ch] for ch in text)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return "".join(
            chr(b) if b < 0x80 else f"⟨{b:02X}⟩" for b in raw
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wrapper-checkpoint",
        default="postraining/runs/rl_gpt2vocab_cot_lr5e5/latent_vapo_checkpoint.pt",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="base pretraining checkpoint; resolved from the run's "
        "manifest.json when omitted",
    )
    parser.add_argument("--math-data", default="postraining/data/dapo-math-17k.parquet")
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    if args.rows < 1 or args.samples < 1:
        parser.error("--rows and --samples must be positive")

    wrapper_path = Path(args.wrapper_checkpoint)
    if args.checkpoint is None:
        manifest = wrapper_path.parent / "manifest.json"
        if not manifest.exists():
            parser.error(f"cannot resolve the base checkpoint: {manifest} not found")
        args.checkpoint = json.loads(manifest.read_text())["base"]["checkpoint"]
        print(f"base checkpoint (from manifest): {args.checkpoint}")

    device = torch.device("cuda")
    backbone = load_model(args.checkpoint, device)
    backbone.eval()
    payload = torch.load(wrapper_path, map_location="cpu", weights_only=False)
    saved_args = payload.get("args", {})
    reasoning_mode = saved_args.get("reasoning_mode", "latent")
    wrapper = LatentThoughtModel(
        backbone, **wrapper_init_kwargs_from_checkpoint(payload)
    ).to(device)
    migrate_legacy_wrapper_checkpoint(payload, wrapper)
    validate_renderer_checkpoint(
        payload,
        str(wrapper_path),
        expected_rollout_policy_schema=rollout_policy_schema_for_mode(
            reasoning_mode
        ),
        expected_thought_input_schema=wrapper.thought_input_schema,
    )
    wrapper.load_state_dict(payload["model"], strict=True)
    wrapper.eval()
    step = payload.get("step")
    print(f"policy+critic: {wrapper_path} (step {step}, mode {reasoning_mode})")

    # Reconstruct the training run's support geometry: pre-v22 checkpoints
    # lack the anchored-support args and used the legacy [0, 1]-edge grid.
    if saved_args.get("value_anchored_support", False):
        value_num_bins, value_v_min, value_v_max = anchored_unit_geometry(
            saved_args.get("value_bins", 101),
            saved_args.get("value_margin_bins", 4),
        )
    else:
        value_num_bins = saved_args.get("value_bins", 101)
        value_v_min, value_v_max = 0.0, 1.0
    critic = SeparateCritic(
        fresh_trunk(backbone, device),
        num_bins=value_num_bins,
        sigma_ratio=saved_args.get("value_sigma_ratio", 2.0),
        v_min=value_v_min,
        v_max=value_v_max,
        prior_value=saved_args.get("value_prior", 0.05),
        adapter_init=saved_args.get("critic_adapter_init", "identity"),
    ).to(device)
    critic.load_state_dict(payload["critic"], strict=True)
    critic.eval()

    tokenizer = load_posttraining_tokenizer(
        backbone.architecture, FreshHyperparameters.tokenizer_path
    )
    stop_ids = tuple(
        dict.fromkeys(
            t for t in (tokenizer.eos_id(), tokenizer.bos_id()) if t >= 0
        )
    )
    prompt_budget = saved_args.get("prompt_tokens", 512)
    response_budget = saved_args.get("continuation_tokens", 1024)
    pin_emit = reasoning_mode != "latent"
    # Pinned modes spend the whole stream on tokens; latent gets 4x slack.
    stream_budget = (
        response_budget if pin_emit else 4 * response_budget
    )

    rows = load_unique_math_rows(args.math_data)
    picked = random.Random(args.seed).sample(range(len(rows)), args.rows)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    unicode_to_byte = (
        gpt2_unicode_to_bytes()
        if "gpt2vocab" in backbone.architecture
        else None
    )

    records = []
    for prompt_index, row_index in enumerate(picked):
        row = rows[row_index]
        encoded = encode_prompt(tokenizer, prompt_text(row), prompt_budget)
        prompt_ids = torch.tensor(encoded, dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16
        ):
            batch = trim_stream(
                rollout_continuations(
                    wrapper,
                    prompt_ids[None],
                    response_budget,
                    stream_budget,
                    saved_args.get("temperature", 1.0),
                    saved_args.get("top_p", 1.0),
                    generator=generator,
                    stop_ids=stop_ids or None,
                    pin_emit=pin_emit,
                    record_likelihoods=False,
                    cache_dtype=torch.bfloat16,
                    prompt_repeats=args.samples,
                )
            )
            values = critic.values(batch).float().cpu()
        truth = row["reward_model"]["ground_truth"]
        score_math_rollout(
            batch, truth, tokenizer, stop_ids, answer_style(row),
            saved_args.get("nearby_reward_max", 0.1),
        )
        kinds = batch.kind.cpu()
        tokens = batch.token_ids.cpu()
        actions = batch.action_mask.cpu()
        rewards = batch.reward_scalar.cpu()
        for sample_index in range(kinds.size(0)):
            slots = []
            for position in range(kinds.size(1)):
                kind = int(kinds[sample_index, position])
                if kind == PAD_SLOT:
                    continue
                token_id = int(tokens[sample_index, position])
                slots.append(
                    {
                        "position": position,
                        "kind": (
                            "thought" if kind == THOUGHT_SLOT
                            else "token"
                        ),
                        "action": bool(actions[sample_index, position]),
                        "token_id": token_id,
                        "text": (
                            token_display_text(
                                tokenizer, token_id, unicode_to_byte
                            )
                            if kind == TOKEN_SLOT
                            else "<think>"
                        ),
                        "value": float(values[sample_index, position]),
                    }
                )
            generated = [
                s for s in slots if s["action"] and s["kind"] == "token"
            ]
            # Decode the whole sequence at once: multi-byte characters span
            # token boundaries, so per-token decodes cannot be concatenated.
            emitted_text = tokenizer.decode(
                [s["token_id"] for s in generated]
            )
            record = {
                "prompt_index": prompt_index,
                "dataset_index": row_index,
                "sample_index": sample_index,
                "ground_truth": str(truth),
                "reward": float(rewards[sample_index]),
                "prompt_tail": tokenizer.decode(encoded[-64:]),
                "emitted_text": emitted_text,
                "value_at_prompt_end": next(
                    (s["value"] for s in slots if s["action"]), None
                ),
                "slots": slots,
            }
            records.append(record)
            generated_values = [s["value"] for s in generated]
            thought_count = sum(
                1 for s in slots if s["action"] and s["kind"] == "thought"
            )
            print(
                f"prompt {prompt_index:2d} (row {row_index}) "
                f"reward {record['reward']:.3f} truth {truth!r:>12} | "
                f"V(start) {record['value_at_prompt_end']:.3f} "
                f"V(mean) {sum(generated_values) / max(len(generated_values), 1):.3f} "
                f"V(last) {generated_values[-1] if generated_values else float('nan'):.3f} "
                f"thinks {thought_count:3d} | {emitted_text!r}"
            )

    out_dir = Path(
        args.out_dir
        if args.out_dir is not None
        else wrapper_path.parent / "critic_inspection"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"step_{step:06d}.json"
    json_path.write_text(json.dumps({"step": step, "records": records}, indent=1))

    all_values = [
        s["value"] for r in records for s in r["slots"] if s["action"]
    ]
    low, high = min(all_values), max(all_values)
    sections = []
    for record in records:
        spans = []
        action_slots = [s for s in record["slots"] if s["action"]]
        index = 0
        while index < len(action_slots):
            slot = action_slots[index]
            if slot["kind"] == "thought":
                # Collapse a consecutive THINK run into one labeled span.
                run = [slot]
                while (
                    index + len(run) < len(action_slots)
                    and action_slots[index + len(run)]["kind"] == "thought"
                ):
                    run.append(action_slots[index + len(run)])
                index += len(run)
                mean_value = sum(s["value"] for s in run) / len(run)
                color = value_color(mean_value, low, high)
                label = (
                    "&lt;think&gt;" if len(run) == 1
                    else f"&lt;think ×{len(run)}&gt;"
                )
                per_slot = " ".join(f"{s['value']:.3f}" for s in run)
                spans.append(
                    f'<span class="tok think" style="background:{color}" '
                    f'title="think run ×{len(run)}, V mean '
                    f'{mean_value:.4f}, per-slot [{per_slot}], pos '
                    f'{run[0]["position"]}-{run[-1]["position"]}">'
                    f"{label}</span>"
                )
                continue
            index += 1
            color = value_color(slot["value"], low, high)
            label = html.escape(slot["text"]).replace("\n", "\\n") or "·"
            spans.append(
                f'<span class="tok" style="background:{color}" '
                f'title="V={slot["value"]:.4f} pos={slot["position"]} '
                f'{slot["kind"]}">{label}</span>'
            )
        verdict = "good" if record["reward"] >= 0.5 else "bad"
        sections.append(
            f'<section><h3>prompt {record["prompt_index"]} '
            f'(row {record["dataset_index"]}) — reward '
            f'<span class="{verdict}">{record["reward"]:.3f}</span>, '
            f'truth <code>{html.escape(record["ground_truth"])}</code>, '
            f'V(start) {record["value_at_prompt_end"]:.3f}</h3>'
            f'<p class="prompt">…{html.escape(record["prompt_tail"])}</p>'
            f'<p class="stream">{"".join(spans)}</p></section>'
        )
    html_path = out_dir / f"step_{step:06d}.html"
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Critic values · step {step}</title><style>"
        "body{background:#0b0e14;color:#e8edf7;font:15px/1.6 system-ui;"
        "max-width:1200px;margin:2rem auto;padding:0 1rem}"
        ".tok{padding:0 .1em;margin:0 1px;border-radius:3px;color:#fff;"
        "white-space:pre-wrap}"
        ".think{outline:1px dashed #e8edf7;font-style:italic}"
        ".prompt{color:#9aa8bd;font-size:.85em}"
        ".good{color:#47d18c}.bad{color:#ff6b7a}"
        "code{background:#131924;padding:0 .3em;border-radius:3px}"
        f"</style></head><body><h1>Critic per-token values · step {step}</h1>"
        f"<p>value color scale: red {low:.3f} → green {high:.3f}; hover a "
        "token for its exact V. Values are the critic's HL-Gauss expected "
        "scalar at each action slot — what GAE uses as V(s).</p>"
        f"{''.join(sections)}</body></html>"
    )
    print(f"\nwrote {json_path}\nwrote {html_path}")


if __name__ == "__main__":
    main()
