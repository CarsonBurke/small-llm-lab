"""Prepare, train and losslessly transport Mini character-model comparisons.

Run preparation/training and CUDA encode/decode/generation through mlq. Physical
packets store raw latent/residual bits; this CLI does not claim entropy compression.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.nanogpt_mini.native_bits_data import (
    PreparedData,
    encode_text,
    prepare_data,
    sha256_file,
)
from pretraining.nanogpt_mini.native_bits_wire import (
    MAX_COUNT,
    MAX_PACKET_BYTES,
    pack_packet,
    unpack_packet,
)


def _fresh_output(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    return output


def _read_source(path: str | Path) -> tuple[bytes, str]:
    with Path(path).open("rb") as handle:
        source = handle.read(4 * MAX_COUNT + 1)
    if len(source) > 4 * MAX_COUNT:
        raise ValueError(
            f"text exceeds the packet limit of {MAX_COUNT} Unicode characters"
        )
    text = source.decode("utf-8", errors="strict")
    if len(text) > MAX_COUNT:
        raise ValueError(
            f"text exceeds the packet limit of {MAX_COUNT} Unicode characters"
        )
    return source, text


def _load_pinned_checkpoint(
    path: Path, expected_digest: str | None = None
) -> tuple[dict, str]:
    digest = sha256_file(path)
    if expected_digest is not None and digest != expected_digest:
        raise ValueError(
            "packet checkpoint SHA256 mismatch; refusing to decode with different weights/alphabet"
        )
    from pretraining.nanogpt_mini.nanogpt_mini_native_bits_train import load_checkpoint

    checkpoint = load_checkpoint(path)
    if sha256_file(path) != digest:
        raise ValueError("checkpoint changed while loading")
    return checkpoint, digest


def encode(args: argparse.Namespace) -> dict:
    output = _fresh_output(args.output)
    source, text = _read_source(args.input)
    checkpoint, checkpoint_digest = _load_pinned_checkpoint(Path(args.checkpoint))
    from pretraining.nanogpt_mini.nanogpt_mini_native_bits_train import (
        checkpoint_model_kind,
        checkpoint_transport_widths,
    )

    ids = encode_text(text, checkpoint["alphabet"])
    latent_width, identity_width = checkpoint_transport_widths(checkpoint)
    model = None
    if len(ids):
        import torch

        from pretraining.nanogpt_mini.nanogpt_mini_native_bits_train import load_model

        model, device = load_model(checkpoint)
        with torch.inference_mode():
            exported = model.export(
                torch.tensor(ids.astype("int64"), device=device).unsqueeze(0)
            )
    else:
        exported = {"latents": [], "residual": []}
    source_digest = hashlib.sha256(source).hexdigest()
    packet = pack_packet(
        **exported,
        checkpoint_sha256=checkpoint_digest,
        source_sha256=source_digest,
        latent_bits=latent_width,
        identity_bits=identity_width,
    )
    transported = unpack_packet(packet)
    if model is not None:
        with torch.inference_mode():
            recovered = model.recover(transported["latents"], transported["residual"])
    else:
        recovered = []
    restored = "".join(
        checkpoint["alphabet"][identity] for identity in recovered
    ).encode("utf-8")
    if (
        restored != source
        or hashlib.sha256(restored).hexdigest() != transported["source_sha256"]
    ):
        raise RuntimeError(
            "exact UTF-8 reconstruction failed before writing the packet"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(packet)
    return {
        "output": str(output),
        "model_kind": checkpoint_model_kind(checkpoint),
        "latent_bits": latent_width,
        "identity_bits": identity_width,
        "source_characters": len(text),
        "source_bytes": len(source),
        "packet_bytes": len(packet),
        "raw_payload_bits": len(text) * (latent_width + identity_width),
        "actual_bits_per_source_byte": len(packet) * 8 / len(source)
        if source
        else None,
        "checkpoint_sha256": checkpoint_digest,
        "source_sha256": source_digest,
        "exact_reconstruction_verified": True,
        "transport": "raw packed bits, not entropy compressed; size is not the modeled coding bound",
    }


def decode(args: argparse.Namespace) -> dict:
    output = _fresh_output(args.output)
    with Path(args.input).open("rb") as handle:
        payload = handle.read(MAX_PACKET_BYTES + 1)
    if len(payload) > MAX_PACKET_BYTES:
        raise ValueError("packet exceeds the 64 MiB input limit")
    packet = unpack_packet(payload)
    # Authenticate checkpoint association before constructing or moving any model to CUDA.
    checkpoint, digest = _load_pinned_checkpoint(
        Path(args.checkpoint), packet["checkpoint_sha256"]
    )
    from pretraining.nanogpt_mini.nanogpt_mini_native_bits_train import (
        checkpoint_model_kind,
        checkpoint_transport_widths,
    )

    expected_widths = checkpoint_transport_widths(checkpoint)
    if (packet["latent_bits"], packet["identity_bits"]) != expected_widths:
        raise ValueError("packet stream widths differ from the checkpoint")
    if packet["latents"]:
        import torch

        from pretraining.nanogpt_mini.nanogpt_mini_native_bits_train import load_model

        model, _ = load_model(checkpoint)
        with torch.inference_mode():
            identities = model.recover(packet["latents"], packet["residual"])
    else:
        identities = []
    text = "".join(checkpoint["alphabet"][identity] for identity in identities)
    source = text.encode("utf-8", errors="strict")
    source_digest = hashlib.sha256(source).hexdigest()
    if source_digest != packet["source_sha256"]:
        raise ValueError(
            "decoded UTF-8 source digest mismatch; damaged packet or incompatible numerical backend"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(source)
    return {
        "output": str(output),
        "model_kind": checkpoint_model_kind(checkpoint),
        "source_characters": len(text),
        "source_bytes": len(source),
        "checkpoint_sha256": digest,
        "source_sha256": source_digest,
        "exact_reconstruction_verified": True,
    }


def generate(args: argparse.Namespace) -> dict:
    output = _fresh_output(args.output)
    if args.characters < 0:
        raise ValueError("--characters must be nonnegative")
    if args.draft_tokens is not None and args.draft_tokens < 0:
        raise ValueError("--draft-tokens must be nonnegative")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("--temperature must be finite and nonnegative")
    if not 0 <= args.seed < 2**32:
        raise ValueError("--seed must fit an unsigned 32-bit integer")
    checkpoint, digest = _load_pinned_checkpoint(Path(args.checkpoint))
    from pretraining.nanogpt_mini.nanogpt_mini_native_bits_train import (
        DYNAMICS_MODEL_KINDS,
        checkpoint_model_kind,
        load_model,
        source_hashes,
    )

    model_kind = checkpoint_model_kind(checkpoint)
    if model_kind not in DYNAMICS_MODEL_KINDS | {"adaptive", "embedder", "softmax"}:
        raise ValueError(
            "generate requires a softmax, adaptive, dynamics or embedder checkpoint"
        )
    if model_kind not in DYNAMICS_MODEL_KINDS and args.draft_tokens is not None:
        raise ValueError("--draft-tokens requires a dynamics checkpoint")
    prompt_ids = encode_text(args.prompt, checkpoint["alphabet"]).tolist()
    if model_kind == "adaptive":
        from pretraining.nanogpt_mini.adaptive_generation import (
            generate as generate_characters,
        )
    elif model_kind == "embedder":
        from pretraining.nanogpt_mini.embedder_generation import (
            generate as generate_characters,
        )
    elif model_kind == "softmax":
        from pretraining.nanogpt_mini.character_generation import (
            generate as generate_characters,
        )
    else:
        from pretraining.nanogpt_mini.speculative_generation import (
            generate as generate_characters,
        )
    model, _ = load_model(checkpoint)
    model.requires_grad_(False)
    generation_options = (
        {"draft_tokens": args.draft_tokens if args.draft_tokens is not None else 2}
        if model_kind in DYNAMICS_MODEL_KINDS
        else {}
    )
    result = generate_characters(
        model,
        prompt_ids=prompt_ids,
        max_new_characters=args.characters,
        temperature=args.temperature,
        seed=args.seed,
        **generation_options,
    )
    generated_text = "".join(
        checkpoint["alphabet"][identity] for identity in result["generated_ids"]
    )
    report = {
        **result,
        "model_kind": model_kind,
        "checkpoint_sha256": digest,
        "inference_source_hashes": source_hashes(model_kind),
        "prompt": args.prompt,
        "generated_text": generated_text,
        "text": args.prompt + generated_text,
        "requested_characters": args.characters,
        "temperature": args.temperature,
        "seed": args.seed,
        "output": str(output),
    }
    serialized = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    return report


def train(args: argparse.Namespace) -> int:
    # Let the existing runner own JSONL, TensorBoard, raw logs and run metadata.
    from pretraining.nanogpt_mini.nanogpt_mini_native_bits_train import (
        DYNAMICS_MODEL_KINDS,
        TrainConfig,
        validate_model_environment,
    )

    # The runner inherits these variables; reject stale model-specific settings
    # here instead of launching a job that can only fail in the training process.
    validate_model_environment(args.model, os.environ)
    if args.frozen_codes and args.model != "native":
        raise ValueError("--frozen-codes is only valid with --model native")
    if args.fixed_codec and args.model != "bitflow":
        raise ValueError("--fixed-codec is only valid with --model bitflow")
    if args.model in {"softmax", "embedder"} and (
        args.mixture_components is not None or args.code_bits is not None
    ):
        raise ValueError("--mixture-components and --code-bits require a bit model")
    if args.latent_bits is not None and args.model != "native":
        raise ValueError("--latent-bits is only valid with --model native")
    if args.mask_surrogate is not None and args.model != "bitflow":
        raise ValueError("--mask-surrogate is only valid with --model bitflow")
    if args.density_head is not None and args.model != "bitflow":
        raise ValueError("--density-head is only valid with --model bitflow")
    if args.prefix_width is not None and args.model not in {
        "bitflow",
        "adaptive",
        *DYNAMICS_MODEL_KINDS,
    }:
        raise ValueError(
            "--prefix-width is only valid with --model bitflow, adaptive or dynamics"
        )
    for name in ("local_layers", "local_dim"):
        if getattr(args, name) is not None and args.model != "adaptive":
            raise ValueError(
                f"--{name.replace('_', '-')} is only valid with --model adaptive"
            )
    for name in (
        "embedder_dim",
        "gate_mode",
        "fixed_stride",
        "emission_cost",
        "baseline_loss_weight",
    ):
        if getattr(args, name) is not None and args.model != "embedder":
            raise ValueError(
                f"--{name.replace('_', '-')} is only valid with --model embedder"
            )
    for name in (
        "dynamics_width",
        "rollout_horizon",
        "latent_weight",
        "rollout_weight",
        "gate_weight",
    ):
        if getattr(args, name) is not None and args.model not in DYNAMICS_MODEL_KINDS:
            raise ValueError(
                f"--{name.replace('_', '-')} is only valid with --model dynamics"
            )
    if args.refresh_cost is not None and args.model not in DYNAMICS_MODEL_KINDS | {
        "adaptive"
    }:
        raise ValueError(
            "--refresh-cost is only valid with --model adaptive or dynamics"
        )
    if args.model in DYNAMICS_MODEL_KINDS | {"adaptive", "embedder"}:
        for name in (
            "mixture_components",
            "flow_layers",
            "codec_dim",
            "codec_weight",
            "code_lr",
        ):
            if getattr(args, name) is not None:
                raise ValueError(
                    f"--{name.replace('_', '-')} is not used by --model {args.model}"
                )
    density_head = args.density_head or "mixture"
    if args.prefix_width is not None:
        if args.model == "bitflow" and density_head != "prefix":
            raise ValueError("--prefix-width requires --density-head prefix")
        if args.prefix_width <= 0:
            raise ValueError("--prefix-width must be positive")
    if density_head == "prefix":
        if not args.fixed_codec:
            raise ValueError("--density-head prefix requires --fixed-codec")
        if args.mixture_components not in (None, 1):
            raise ValueError("--density-head prefix requires --mixture-components 1")
    config = TrainConfig(
        data_path=str(Path(args.data).expanduser().resolve(strict=True)),
        run_id=args.name,
        model_kind=args.model,
        iterations=args.steps,
        val_every=args.val_every,
        seq_len=args.seq_len,
        mbs=args.mbs,
        batch_characters=args.batch_characters,
        val_characters=args.val_characters,
        codec_weight=args.codec_weight if args.codec_weight is not None else 1.0,
        code_lr=args.code_lr if args.code_lr is not None else 0.004,
        seed=args.seed,
        resume=str(Path(args.resume).expanduser().resolve(strict=True))
        if args.resume
        else None,
    )
    if config.resume:
        run_dir = REPO_ROOT / "ablation_results" / config.run_id
        if Path(config.resume).is_relative_to(run_dir):
            raise ValueError(
                "ablation clears its run directory; resume into a NEW --name or copy the checkpoint outside that directory"
            )
    routed_config = None
    if config.model_kind in DYNAMICS_MODEL_KINDS | {"adaptive", "embedder"}:
        from pretraining.nanogpt_mini.nanogpt_mini_native_bits_train import (
            AdaptiveConfig,
            DynamicsConfig,
            EmbedderConfig,
        )

        data = PreparedData(config.data_path)
        if config.val_characters > data.validation.size:
            raise ValueError("--val-characters exceeds the heldout split")
        common = {
            "vocab_size": len(data.alphabet),
            "num_layers": args.num_layers,
            "model_dim": args.model_dim,
        }
        if config.model_kind == "embedder":
            routed_config = EmbedderConfig(
                **common,
                embedder_dim=args.embedder_dim
                if args.embedder_dim is not None
                else 128,
                gate_mode=args.gate_mode if args.gate_mode is not None else "learned",
                fixed_stride=args.fixed_stride if args.fixed_stride is not None else 4,
                emission_cost=args.emission_cost
                if args.emission_cost is not None
                else 0.02,
                baseline_loss_weight=(
                    args.baseline_loss_weight
                    if args.baseline_loss_weight is not None
                    else 0.01
                ),
            )
        else:
            common.update(
                code_bits=args.code_bits if args.code_bits is not None else 0,
                prefix_width=args.prefix_width
                if args.prefix_width is not None
                else 128,
                refresh_cost=args.refresh_cost
                if args.refresh_cost is not None
                else 0.02,
            )
        if config.model_kind == "adaptive":
            routed_config = AdaptiveConfig(
                **common,
                local_layers=args.local_layers if args.local_layers is not None else 2,
                local_dim=args.local_dim if args.local_dim is not None else 128,
            )
        elif config.model_kind in DYNAMICS_MODEL_KINDS:
            routed_config = DynamicsConfig(
                **common,
                dynamics_width=args.dynamics_width
                if args.dynamics_width is not None
                else 128,
                rollout_horizon=args.rollout_horizon
                if args.rollout_horizon is not None
                else 2,
                latent_weight=args.latent_weight
                if args.latent_weight is not None
                else 1.0,
                rollout_weight=args.rollout_weight
                if args.rollout_weight is not None
                else 1.0,
                gate_weight=args.gate_weight if args.gate_weight is not None else 1.0,
            )
    overrides = {
        "NATIVE_DATA_PATH": config.data_path,
        "SEQ_LEN": str(config.seq_len),
        "MBS": str(config.mbs),
        "BATCH_CHARACTERS": str(config.batch_characters),
        "VAL_CHARACTERS": str(config.val_characters),
        "SEED": str(config.seed),
        "MODEL_KIND": config.model_kind,
        "NUM_LAYERS": str(args.num_layers),
        "MODEL_DIM": str(args.model_dim),
        "NATIVE_RESUME": config.resume or "",
    }
    if config.model_kind not in DYNAMICS_MODEL_KINDS | {"embedder"}:
        overrides.update(
            CODEC_WEIGHT=str(config.codec_weight),
            CODE_LR=str(config.code_lr),
        )
    if config.model_kind not in {"softmax", "embedder"}:
        overrides.update(
            CODE_BITS=str(
                args.code_bits
                if args.code_bits is not None
                else 32
                if config.model_kind == "native"
                else 0
            ),
        )
    if config.model_kind in {"native", "bitflow"}:
        overrides.update(
            CODEC_DIM=str(args.codec_dim if args.codec_dim is not None else 64),
            MIXTURE_COMPONENTS=str(
                args.mixture_components
                if args.mixture_components is not None
                else 1
                if config.model_kind == "native" or density_head == "prefix"
                else 8
            ),
        )
    if config.model_kind == "native":
        overrides.update(
            LATENT_BITS=str(args.latent_bits if args.latent_bits is not None else 8),
            LEARN_CODES="0" if args.frozen_codes else "1",
        )
    elif config.model_kind == "bitflow":
        overrides.update(
            FLOW_LAYERS=str(args.flow_layers if args.flow_layers is not None else 4),
            LEARN_CODEC="0" if args.fixed_codec else "1",
            MASK_SURROGATE=args.mask_surrogate or "sigmoid",
            DENSITY_HEAD=density_head,
            PREFIX_WIDTH=str(
                args.prefix_width if args.prefix_width is not None else 128
            ),
        )
    elif config.model_kind == "embedder":
        overrides.update(
            EMBEDDER_DIM=str(routed_config.embedder_dim),
            GATE_MODE=routed_config.gate_mode,
            FIXED_STRIDE=str(routed_config.fixed_stride),
            EMISSION_COST=str(routed_config.emission_cost),
            BASELINE_LOSS_WEIGHT=str(routed_config.baseline_loss_weight),
        )
    elif routed_config is not None:
        overrides.update(
            PREFIX_WIDTH=str(routed_config.prefix_width),
            REFRESH_COST=str(routed_config.refresh_cost),
        )
        if config.model_kind == "adaptive":
            overrides.update(
                LOCAL_LAYERS=str(routed_config.local_layers),
                LOCAL_DIM=str(routed_config.local_dim),
            )
        else:
            overrides.update(
                DYNAMICS_WIDTH=str(routed_config.dynamics_width),
                ROLLOUT_HORIZON=str(routed_config.rollout_horizon),
                LATENT_WEIGHT=str(routed_config.latent_weight),
                ROLLOUT_WEIGHT=str(routed_config.rollout_weight),
                GATE_WEIGHT=str(routed_config.gate_weight),
            )
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/ablation.py"),
        "--name",
        config.run_id,
        "--script",
        "pretraining/nanogpt_mini/nanogpt_mini_native_bits_train.py",
        "--steps",
        str(config.iterations),
        "--val-every",
        str(config.val_every),
        "--env",
        *(f"{key}={value}" for key, value in overrides.items()),
    ]
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="strict UTF-8 to shuffled opaque character IDs (queue via mlq)"
    )
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--train-text")
    prepare.add_argument("--val-text")
    prepare.add_argument(
        "--byte-cache",
        help="version-2 UTF8-byte source cache; not raw token-storage bytes",
    )
    prepare.add_argument("--alphabet-seed", type=int, default=1337)
    training = commands.add_parser(
        "train", help="delegate to the ablation runner (queue via mlq)"
    )
    training.add_argument("--data", required=True)
    training.add_argument("--name", required=True)
    training.add_argument(
        "--model",
        choices=(
            "native",
            "bitflow",
            "softmax",
            "adaptive",
            "dynamics",
            "dynamics_bounded",
            "embedder",
        ),
        default="native",
    )
    training.add_argument("--steps", type=int, default=2000)
    training.add_argument("--val-every", type=int, default=20)
    training.add_argument("--seq-len", type=int, default=1024)
    training.add_argument("--mbs", type=int, default=8)
    training.add_argument("--batch-characters", type=int, default=8192)
    training.add_argument("--val-characters", type=int, default=65536)
    training.add_argument(
        "--code-bits",
        type=int,
        help="native code-table width (default 32), or bitflow/adaptive/dynamics width (default alphabet minimum)",
    )
    training.add_argument(
        "--latent-bits", type=int, help="native bottleneck width (default 8)"
    )
    training.add_argument(
        "--mixture-components",
        type=int,
        help="Bernoulli-mixture components (default 1 native/prefix, 8 bitflow mixture)",
    )
    training.add_argument(
        "--density-head",
        choices=("mixture", "prefix"),
        help="bitflow density head (default mixture); prefix requires --fixed-codec",
    )
    training.add_argument(
        "--prefix-width",
        type=int,
        help="hidden width for adaptive/dynamics or --density-head prefix (default 128)",
    )
    training.add_argument(
        "--flow-layers", type=int, help="bitflow coupling layers (default 4)"
    )
    training.add_argument(
        "--local-layers", type=int, help="adaptive local character layers (default 2)"
    )
    training.add_argument(
        "--local-dim", type=int, help="adaptive local character width (default 128)"
    )
    training.add_argument(
        "--dynamics-width",
        type=int,
        help="dynamics transition/error-predictor width (default 128)",
    )
    training.add_argument(
        "--rollout-horizon",
        type=int,
        help="dynamics training supervision depth, not an inference refresh cap (default 2)",
    )
    training.add_argument(
        "--latent-weight",
        type=float,
        help="dynamics latent smooth-L1 loss weight (default 1)",
    )
    training.add_argument(
        "--rollout-weight",
        type=float,
        help="dynamics rollout bitwise KL weight (default 1)",
    )
    training.add_argument(
        "--gate-weight",
        type=float,
        help="dynamics causal error-predictor loss weight (default 1)",
    )
    training.add_argument(
        "--refresh-cost",
        type=float,
        help="adaptive nats per global update or dynamics predicted-KL refresh threshold in nats (default 0.02)",
    )
    training.add_argument(
        "--embedder-dim",
        type=int,
        help="persistent recurrent character state width (default 128)",
    )
    training.add_argument(
        "--gate-mode",
        choices=("learned", "fixed"),
        help="embedder emission policy (default learned); fixed is the matched control",
    )
    training.add_argument(
        "--fixed-stride",
        type=int,
        help="fixed embedder characters per event (default 4)",
    )
    training.add_argument(
        "--emission-cost",
        type=float,
        help="embedder nats charged per global event (default 0.02)",
    )
    training.add_argument(
        "--baseline-loss-weight",
        type=float,
        help="embedder detached-input mean-future-NLL critic weight (default 0.01)",
    )
    training.add_argument(
        "--mask-surrogate",
        choices=("sigmoid", "identity"),
        help="bitflow mask backward rule (default sigmoid); checkpointed for exact resumption",
    )
    training.add_argument("--num-layers", type=int, default=6)
    training.add_argument("--model-dim", type=int, default=512)
    training.add_argument("--codec-dim", type=int, help="codec width (default 64)")
    training.add_argument(
        "--codec-weight", type=float, help="native codec weight (default 1)"
    )
    training.add_argument(
        "--code-lr", type=float, help="native code-table learning rate (default 0.004)"
    )
    training.add_argument("--seed", type=int, default=1337)
    training.add_argument(
        "--frozen-codes",
        action="store_true",
        help="native-only fixed random code table; remaining components still learn",
    )
    training.add_argument(
        "--fixed-codec",
        action="store_true",
        help="bitflow-only frozen bijective codec control with the same prior initialization",
    )
    training.add_argument("--resume")
    for action in ("encode", "decode"):
        child = commands.add_parser(
            action,
            help=f"{action} raw packed latent/residual transport (CUDA via mlq for nonempty text)",
        )
        child.add_argument("--checkpoint", required=True)
        child.add_argument("--input", required=True)
        child.add_argument("--output", required=True)
    generation = commands.add_parser(
        "generate",
        help="cached adaptive/dynamics/embedder character generation (CUDA via mlq)",
    )
    generation.add_argument("--checkpoint", required=True)
    generation.add_argument("--prompt", required=True)
    generation.add_argument("--characters", type=int, default=128)
    generation.add_argument("--temperature", type=float, default=0.0)
    generation.add_argument("--seed", type=int, default=1337)
    generation.add_argument(
        "--draft-tokens",
        type=int,
        help="dynamics proposal budget (default 2); 0 selects target-only decoding",
    )
    generation.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        output = prepare_data(
            args.output,
            train_text=args.train_text,
            val_text=args.val_text,
            byte_cache=args.byte_cache,
            alphabet_seed=args.alphabet_seed,
        )
        print(
            json.dumps(
                {"prepared": str(output), "metadata": str(output / "metadata.json")},
                sort_keys=True,
            )
        )
        return 0
    if args.command == "train":
        return train(args)
    print(
        json.dumps(
            {"encode": encode, "decode": decode, "generate": generate}[args.command](
                args
            ),
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
