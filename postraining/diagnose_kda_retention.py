"""Measure how long a KDA mixer's delta-rule state actually remembers.

The KDA recurrence is linear in its state. Acting on a row of the
value-major state (a vector in key space), one step is

    s <- (s * exp(g_t)) (I - beta_t k_t k_t^T)        [+ the new write]

so the Jacobian of the state at ``t + tau`` with respect to the state at
``t`` is the product of those per-step maps. Every factor has norm at most
one (``exp(g) <= 1``, ``beta in [0, 1]``, ``|k| = 1``): the recurrence cannot
explode, and how fast the product shrinks is both the memory horizon of the
mixer and the reach of any gradient that crosses time through it. This is
the Jacobian-product argument of arXiv:2605.12683 (appendix A) applied to
the data-dependent transitions of a trained checkpoint, on real tokens.

Three quantities, per KDA layer and horizon ``tau``:

- ``own_key``: ``|k_s^T P(s -> s+tau)|``, the surviving fraction of the
  association written at ``s`` along its own key. This is what a later
  query can still read back.
- ``gate_only``: the same with the delta-rule erase removed (decay gate
  alone), which separates forgetting by decay from overwriting.
- ``operator``: the spectral norm of ``P``, the best-retained direction of
  the whole key space, and ``energy``: ``|P|_F^2 / Dk``, the surviving
  fraction of an isotropic write.

With ``--shard`` the rows are consecutive windows of a pretraining token
stream and the per-position cross-entropy is reported in position buckets,
which shows whether the whole model (KDA plus the dense layers) gains
anything from context beyond a given length. With ``--traces`` the rows are
SFT documents framed and packed exactly as ``sft_trace_train`` packs them.

    python -m postraining.diagnose_kda_retention \\
        --checkpoint <model.pt> --shard <fineweb_val_000000.bin> \\
        --seq-len 5120 --out <report.json>

It executes the model on the GPU, so it runs through ``mlq``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from postraining.model_io import load_model
from pretraining.nanogpt_mini.nanogpt_mini_kda_model import (
    KimiDeltaAttention,
    _l2norm,
    kda_decay_gate,
)

SHARD_MAGIC = 20240520
SHARD_HEADER_INTS = 256
QUANTILES = (0.5, 0.9, 0.99)
# Per-token decay factors that keep half of a write for at least this long.
HALF_LIFE_THRESHOLDS = (16, 64, 256, 1024, 4096)


def horizons_for(seq_len: int) -> list[int]:
    """Powers of two below the row length."""
    return [1 << e for e in range(seq_len.bit_length()) if (1 << e) < seq_len]


def load_shard_rows(path: Path, seq_len: int, rows: int) -> np.ndarray:
    """``rows`` windows of ``seq_len + 1`` tokens, evenly spaced in a shard."""
    header = np.fromfile(path, dtype=np.int32, count=SHARD_HEADER_INTS)
    if header[0] != SHARD_MAGIC or header[1] != 1:
        raise ValueError(f"{path} is not a version-1 token shard")
    tokens = np.memmap(
        path, dtype=np.uint16, mode="r", offset=SHARD_HEADER_INTS * 4,
        shape=(int(header[2]),),
    )
    width = seq_len + 1
    if tokens.shape[0] < rows * width:
        raise ValueError(
            f"{path} holds {tokens.shape[0]} tokens, fewer than {rows} rows "
            f"of {width}"
        )
    starts = np.linspace(0, tokens.shape[0] - width, rows).astype(np.int64)
    return np.stack([tokens[s:s + width] for s in starts]).astype(np.int64)


def load_trace_rows(
    path: Path, model, seq_len: int, rows: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Packed SFT rows and each row's count of real (non-padding) tokens."""
    import pyarrow.parquet as pq

    from postraining.core import load_posttraining_tokenizer
    from postraining.math_prompt import INSTRUCTION_SUFFIX_ANSWER
    from postraining.sft_trace_train import (
        IGNORE_INDEX,
        pack_rows,
        tokenize_documents,
    )
    from pretraining.fresh_lejepa.fresh_lejepa_train import FreshHyperparameters

    tokenizer = load_posttraining_tokenizer(
        model.architecture,
        FreshHyperparameters.tokenizer_path,
        think_tokens=True,
        answer_tokens=True,
        tokenizer_provenance=model.model_config.get("tokenizer_provenance"),
    )
    table = pq.read_table(path, columns=["problem", "document", "doc_tokens"])
    # Enough documents to fill the requested rows several times over, drawn
    # uniformly so the panel is not one source's contiguous block.
    budget = 8 * rows * seq_len
    order = list(range(table.num_rows))
    random.Random(seed).shuffle(order)
    doc_tokens = table.column("doc_tokens").to_numpy()
    chosen, total = [], 0
    for index in order:
        chosen.append(index)
        total += int(doc_tokens[index])
        if total >= budget:
            break
    documents = table.take(chosen).select(["problem", "document"]).to_pylist()
    tokenized = tokenize_documents(
        tokenizer, documents, seq_len,
        instruction_suffix=INSTRUCTION_SUFFIX_ANSWER,
    )
    packed = pack_rows(tokenized, seq_len, tokenizer.eos_id(), rng=None)[:rows]
    if len(packed) < rows:
        raise ValueError(
            f"packed only {len(packed)} rows of {seq_len}; wanted {rows}"
        )
    inputs = np.stack([row for row, _ in packed]).astype(np.int64)
    valid = np.array(
        [int(np.flatnonzero(t != IGNORE_INDEX)[-1]) + 1 for _, t in packed]
    )
    return inputs, valid


class MixerCapture:
    """Forward hooks collecting each KDA layer's keys, gates and betas."""

    def __init__(self, model):
        self.layers = [
            (index, block.attn)
            for index, block in enumerate(model.blocks)
            if isinstance(block.attn, KimiDeltaAttention)
        ]
        if not self.layers:
            raise ValueError("checkpoint has no KDA layers")
        self.captured: dict[tuple[int, str], Tensor] = {}
        self.handles = []
        for index, attn in self.layers:
            for name in ("k_conv1d", "f_b_proj", "b_proj"):
                self.handles.append(
                    getattr(attn, name).register_forward_hook(
                        self._store(index, name)
                    )
                )

    def _store(self, index: int, name: str):
        def hook(_module, _inputs, output):
            self.captured[index, name] = output.detach()

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def transitions(self) -> tuple[Tensor, Tensor, Tensor]:
        """``(k, decay, beta)`` stacked over layers, fp32.

        k: unit keys ``[L, B, T, H, Dk]``; decay: per-step factors
        ``exp(g)`` of the same shape; beta: ``[L, B, T, H]``.
        """
        keys, decays, betas = [], [], []
        for index, attn in self.layers:
            raw_k = self.captured[index, "k_conv1d"]
            B, T, _ = raw_k.shape
            heads = (B, T, attn.num_heads, attn.head_dim)
            keys.append(_l2norm(raw_k.float().view(heads)))
            decays.append(
                kda_decay_gate(
                    self.captured[index, "f_b_proj"].view(heads),
                    attn.A_log,
                    attn.dt_bias,
                ).exp()
            )
            betas.append(torch.sigmoid(self.captured[index, "b_proj"].float()))
        return torch.stack(keys), torch.stack(decays), torch.stack(betas)


def retention_curves(
    k: Tensor,
    decay: Tensor,
    beta: Tensor,
    starts: list[int],
    operator_starts: list[int],
    horizons: list[int],
) -> dict[str, Tensor]:
    """Propagate probes through the recorded transitions.

    Returns ``own_key``/``gate_only`` of shape ``[L, B, H, S, n_h]`` and
    ``operator``/``energy`` of shape ``[L, B, H, S_op, n_h]``; entries whose
    horizon runs past the row end are NaN.
    """
    L, B, T, H, Dk = k.shape
    device = k.device
    n_h = len(horizons)
    horizon_index = {tau: i for i, tau in enumerate(horizons)}

    def nan(count: int) -> Tensor:
        return torch.full((L, B, H, count, n_h), math.nan, device=device)

    out = {
        "own_key": nan(len(starts)),
        "gate_only": nan(len(starts)),
        "operator": nan(len(operator_starts)),
        "energy": nan(len(operator_starts)),
    }
    start_at = torch.tensor(starts, device=device)
    op_start_at = torch.tensor(operator_starts, device=device)
    probe = torch.zeros(L, B, H, len(starts), Dk, device=device)
    gated = torch.zeros_like(probe)
    operator = torch.zeros(L, B, H, len(operator_starts), Dk, Dk, device=device)
    identity = torch.eye(Dk, device=device)

    for t in range(T):
        k_t = k[:, :, t]  # [L, B, H, Dk]
        decay_t = decay[:, :, t]
        beta_t = beta[:, :, t]

        # s <- (s * exp(g)) (I - beta k k^T), for row-vector probes.
        probe = probe * decay_t[:, :, :, None, :]
        overlap = torch.einsum("lbhsd,lbhd->lbhs", probe, k_t)
        probe = probe - (
            (beta_t[..., None] * overlap)[..., None] * k_t[:, :, :, None, :]
        )
        gated = gated * decay_t[:, :, :, None, :]
        operator = operator * decay_t[:, :, :, None, None, :]
        projected = torch.einsum("lbhsed,lbhd->lbhse", operator, k_t)
        operator = operator - (
            (beta_t[..., None, None] * projected)[..., None]
            * k_t[:, :, :, None, None, :]
        )

        # A probe is born as the key written at its start step.
        born = start_at == t
        if born.any():
            probe[:, :, :, born] = k_t[:, :, :, None, :]
            gated[:, :, :, born] = k_t[:, :, :, None, :]
        op_born = op_start_at == t
        if op_born.any():
            operator[:, :, :, op_born] = identity

        for name, state, origin in (
            ("own_key", probe, start_at),
            ("gate_only", gated, start_at),
        ):
            age = t - origin
            for slot in torch.nonzero(age > 0).flatten().tolist():
                column = horizon_index.get(int(age[slot]))
                if column is not None:
                    out[name][:, :, :, slot, column] = state[
                        :, :, :, slot
                    ].norm(dim=-1)
        age = t - op_start_at
        for slot in torch.nonzero(age > 0).flatten().tolist():
            column = horizon_index.get(int(age[slot]))
            if column is not None:
                matrix = operator[:, :, :, slot]
                out["operator"][:, :, :, slot, column] = (
                    torch.linalg.matrix_norm(matrix, ord=2)
                )
                out["energy"][:, :, :, slot, column] = (
                    matrix.square().sum((-2, -1)) / Dk
                )
    return out


def mask_past_row_end(
    curves: dict[str, Tensor],
    starts: list[int],
    operator_starts: list[int],
    horizons: list[int],
    valid: Tensor,
) -> None:
    """NaN every probe whose horizon ends in a row's padding, in place."""
    taus = torch.tensor(horizons, device=valid.device)
    for names, origins in (
        (("own_key", "gate_only"), starts),
        (("operator", "energy"), operator_starts),
    ):
        ends = torch.tensor(origins, device=valid.device)[:, None] + taus
        outside = ends[None] >= valid[:, None, None]  # [B, S, n_h]
        for name in names:
            curves[name].masked_fill_(outside[None, :, None], math.nan)


def summarize(values: Tensor) -> dict[str, float] | None:
    """Distribution of one horizon's retention over rows, heads and starts."""
    finite = values[~values.isnan()]
    if finite.numel() == 0:
        return None
    quantiles = torch.quantile(
        finite.double(), torch.tensor(QUANTILES, dtype=torch.float64,
                                      device=finite.device),
    )
    summary = {
        f"p{round(q * 100)}": float(v)
        for q, v in zip(QUANTILES, quantiles, strict=True)
    }
    summary["mean"] = float(finite.mean())
    summary["max"] = float(finite.max())
    summary["count"] = int(finite.numel())
    return summary


def curve_report(curves: dict[str, Tensor], horizons: list[int]) -> dict:
    return {
        name: {
            str(tau): summarize(values[..., column])
            for column, tau in enumerate(horizons)
        }
        for name, values in curves.items()
    }


def gate_report(decay: Tensor, beta: Tensor, valid: Tensor) -> dict:
    """Per-token decay factors and write strengths over real tokens."""
    T = decay.size(1)
    real = torch.arange(T, device=decay.device)[None] < valid[:, None]
    factors = decay[real]  # [N, H, Dk]
    report = {
        "decay_factor_mean": float(factors.mean()),
        "beta_mean": float(beta[real].mean()),
        "half_life_at_least": {
            str(tokens): float(
                (factors >= 0.5 ** (1.0 / tokens)).float().mean()
            )
            for tokens in HALF_LIFE_THRESHOLDS
        },
    }
    # A key dimension's own horizon: its geometric-mean decay over the panel.
    mean_log = factors.clamp_min(1e-30).log().mean(0)  # [H, Dk]
    half_life = math.log(0.5) / mean_log
    report["dimension_half_life"] = {
        "p50": float(half_life.median()),
        "p90": float(torch.quantile(half_life.flatten(), 0.9)),
        "max": float(half_life.max()),
    }
    return report


def position_buckets(seq_len: int) -> list[tuple[int, int]]:
    edges = [0] + [tau for tau in horizons_for(seq_len) if tau >= 16] + [seq_len]
    return list(zip(edges[:-1], edges[1:], strict=True))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--shard", help="pretraining token shard (.bin)")
    source.add_argument("--traces", help="SFT corpus parquet")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--starts", type=int, default=64)
    parser.add_argument("--operator-starts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        parser.error(f"{out_path} exists; reports are not overwritten")
    device = torch.device("cuda")
    model = load_model(args.checkpoint, device).eval()

    T = args.seq_len
    targets = None
    if args.shard:
        windows = load_shard_rows(Path(args.shard), T, args.rows)
        inputs = torch.from_numpy(windows[:, :-1]).to(device)
        targets = torch.from_numpy(windows[:, 1:]).to(device)
        valid = torch.full((args.rows,), T, device=device)
    else:
        rows, valid_counts = load_trace_rows(
            Path(args.traces), model, T, args.rows, args.seed
        )
        inputs = torch.from_numpy(rows).to(device)
        valid = torch.from_numpy(valid_counts).to(device)

    capture = MixerCapture(model)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        belief = model.temporal_belief_from_token_latent(
            model.embed_tokens(inputs)
        )
    capture.close()
    k, decay, beta = capture.transitions()

    horizons = horizons_for(T)
    # Probes start in the first half so every start reaches horizons of at
    # least T/2; the operator probes are the expensive ones and start early.
    starts = sorted({int(s) for s in np.linspace(0, T // 2, args.starts)})
    operator_starts = sorted(
        {int(s) for s in np.linspace(0, T // 4, args.operator_starts)}
    )
    curves = retention_curves(k, decay, beta, starts, operator_starts, horizons)
    mask_past_row_end(curves, starts, operator_starts, horizons, valid)

    report = {
        "schema": "diagnose_kda_retention/v1",
        "checkpoint": args.checkpoint,
        "source": args.shard or args.traces,
        "source_kind": "shard" if args.shard else "traces",
        "seq_len": T,
        "train_context_tokens": model.train_context_tokens,
        "rows": args.rows,
        "real_tokens": int(valid.sum()),
        "horizons": horizons,
        "starts": starts,
        "operator_starts": operator_starts,
        "all_layers": curve_report(curves, horizons),
        "layers": {},
    }
    for slot, (index, _attn) in enumerate(capture.layers):
        layer = curve_report(
            {name: values[slot] for name, values in curves.items()}, horizons
        )
        layer["gates"] = gate_report(decay[slot], beta[slot], valid)
        report["layers"][str(index)] = layer

    if targets is not None:
        losses = []
        for row in range(args.rows):  # one row of fp32 logits at a time
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits = model.project_logits(belief[row])
            losses.append(
                F.cross_entropy(logits.float(), targets[row], reduction="none")
            )
        nats = torch.stack(losses)
        report["cross_entropy_by_position"] = {
            f"{lo}-{hi}": {
                "nats": float(nats[:, lo:hi].mean()),
                "bits": float(nats[:, lo:hi].mean() / math.log(2)),
                "sem_nats": float(
                    nats[:, lo:hi].mean(1).std() / math.sqrt(args.rows)
                ),
            }
            for lo, hi in position_buckets(T)
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"{'layer':>6} " + " ".join(f"{tau:>7}" for tau in horizons))
    for name in ("own_key", "gate_only", "operator"):
        print(f"-- {name}: median (max) retention by horizon")
        for label, block in [("all", report["all_layers"])] + [
            (key, value) for key, value in report["layers"].items()
        ]:
            cells = []
            for tau in horizons:
                stats = block[name][str(tau)]
                cells.append("      -" if stats is None else f"{stats['p50']:7.4f}")
            print(f"{label:>6} " + " ".join(cells))
        cells = []
        for tau in horizons:
            stats = report["all_layers"][name][str(tau)]
            cells.append("      -" if stats is None else f"{stats['max']:7.4f}")
        print(f"{'max':>6} " + " ".join(cells))
    if targets is not None:
        print("-- cross-entropy by position (bits/token)")
        for bucket, stats in report["cross_entropy_by_position"].items():
            print(f"{bucket:>12}: {stats['bits']:.4f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
