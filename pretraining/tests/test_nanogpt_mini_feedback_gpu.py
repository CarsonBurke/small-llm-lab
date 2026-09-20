"""CUDA contracts of the latent-feedback nanogpt-mini variants at the real width (bf16, compiled)."""

import gc
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
import time

import pytest
import torch

from pretraining.nanogpt_mini.nanogpt_mini_feedback_model import (DEFAULT_DECAYS, FeedbackGPT, MemoryState, decay_mask,
                                                                  linear_memory_read_chunked, linear_memory_read_kernel,
                                                                  linear_memory_features, linear_memory_read_dense, linear_memory_read_parallel)
from pretraining.nanogpt_mini.nanogpt_mini_feedback_model import (
    _lam_gla_forward,
    chunk_simple_gla,
)
from pretraining.nanogpt_mini.nanogpt_mini_model import GPT

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
DEVICE = torch.device("cuda")
VOCAB, LAYERS, DIM = 1024, 6, 512


def _init(model: torch.nn.Module, seed: int, live: bool) -> None:
    torch.manual_seed(seed)
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                w.normal_(std=0.05) if live else w.zero_()
            elif "embed" in name:
                w.normal_()
            else:
                w.normal_(std=0.33**0.5 / w.size(-1)**0.5)
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.fill_(1.0)
        elif name in ("decay_logit", "temperature"):
            pass
        else:
            raise AssertionError(name)
    if hasattr(model, "reset_extra_parameters"):
        model.reset_extra_parameters()


def _model(mode: str, live: bool = True, fp32: bool = False, **kwargs) -> FeedbackGPT:
    torch.manual_seed(0)
    model = FeedbackGPT(VOCAB, LAYERS, DIM, mode=mode, **kwargs).to(DEVICE)
    if fp32:
        model = model.float()
    _init(model, 11, live)
    return model.eval()


def _batch(B: int, T: int, seed: int = 2):
    g = torch.Generator(device="cpu").manual_seed(seed)
    inputs = torch.randint(0, VOCAB, (B, T), generator=g).to(DEVICE, torch.int32)
    targets = torch.randint(0, VOCAB, (B, T), generator=g).to(DEVICE, torch.int64)
    return inputs, targets


def _prefix_recompute(model: FeedbackGPT, inputs, targets):
    B, T = inputs.shape
    states = torch.zeros(B, T, DIM, device=DEVICE, dtype=model.embed.weight.dtype)
    total = torch.zeros((), device=DEVICE)
    for t in range(T):
        s = model.pass_forward(inputs[:, :t + 1], states[:, :t + 1] if model.mode != "none" else None)
        total = total + model.head_loss(s[:, t], targets[:, t])
        states[:, t] = s[:, t]
    return total


REPO = pathlib.Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mode", ["none", "top"])
def test_trainer_runs_end_to_end_with_deep_validation_and_checkpoint(mode, tmp_path):
    """Exercises every trainer branch (deep eval at step 0 and at the end, the none gate, the checkpoint
    written before the final validation) on two steps. A path test, not evidence."""
    run_id = f"fbtest_{mode}_{os.getpid()}"
    with socket.socket() as sock:  # the trainer's default rendezvous port may be held by another run
        sock.bind(("127.0.0.1", 0))
        port = str(sock.getsockname()[1])
    env = dict(os.environ, RUN_ID=run_id, FB_MODE=mode, ITERATIONS="2", VAL_LOSS_EVERY="1", TRAIN_LOG_EVERY="1",
               VAL_TOKENS="131072", FB_SEQ_TOKENS="8192", DATA_PATH="data/datasets/fineweb10B_sp1024", MASTER_PORT=port)
    checkpoint = REPO / "logs" / f"{run_id}_final_model.pt"
    gc.collect()
    torch.cuda.empty_cache()  # the subprocess needs the device; this process holds nothing yet (runs first)
    try:
        result = subprocess.run([sys.executable, "pretraining/nanogpt_mini/nanogpt_mini_feedback_train.py"],
                                cwd=REPO, env=env, capture_output=True, text=True, timeout=1500)
        assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
        vals = re.findall(r"step:(\d+)/2 val_loss:\S+ val_bpb:(\S+)(.*)", result.stdout)
        assert [v[0] for v in vals] == ["0", "1", "2"], result.stdout
        for step, _, extras in vals:
            assert ("val_bpb_seq:" in extras) == (step in ("0", "2")), (step, extras)
            assert ("val_bpb_p8:" in extras) == (mode not in ("none", "top") and step in ("0", "2")), (step, extras)
            assert ("mem_decay0:" in extras) == (mode in ("lam", "top"))
            assert ("val_bpb_noread:" in extras) == (mode == "top")
        assert result.stdout.index("saved checkpoint") < result.stdout.index("step:2/2 val_loss")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert saved["architecture"] == "nanogpt_mini_feedback_v1" and saved["model_config"]["mode"] == mode
    finally:
        checkpoint.unlink(missing_ok=True)
        for log in (REPO / "logs").glob(f"{run_id}*"):
            log.unlink() if log.is_file() else None


# The trainer's none-mode gate fires at 0.002 bpb ~ 0.0055 nats per token; the tests hold the same bound.
GATE_NATS = 0.005


# bf16 noise floor of the recurrence in the feedback modes (their multiplicative state path amplifies
# rounding; job 7825 measured 0.018-0.023 nats per token for glu on random weights). Reported, and bounded
# loosely; the algorithmic identity is pinned in fp32 at the gate tolerance.
BF16_FLOOR_NATS = 0.05


@pytest.mark.parametrize("fp32", [True, False], ids=["fp32", "bf16"])
@pytest.mark.parametrize("mode,window", [("none", 0), ("glu", 0), ("add", 0), ("lam", 0), ("lam", 8), ("glu", 8), ("lam", 64), ("top", 0), ("top", 8)])
def test_sequential_evaluator_matches_prefix_recompute_at_real_width(mode, window, fp32):
    model = _model(mode, attn_window=window, fp32=fp32)
    inputs, targets = _batch(4, 256 if window == 64 else 32)
    with torch.no_grad():
        expected = _prefix_recompute(model, inputs, targets)
        actual = model.loss_sequential(inputs, targets)
    gap = abs(float(actual) - float(expected)) / inputs.numel()
    print(json.dumps({"seq_vs_prefix_nats": round(gap, 5), "mode": mode, "window": window, "fp32": fp32}))
    bound = GATE_NATS if fp32 or mode == "none" else BF16_FLOOR_NATS
    assert gap < bound, (mode, window, fp32, float(actual), float(expected))


@pytest.mark.parametrize("window", [0, 64])
def test_sequential_evaluator_matches_the_parallel_forward_at_full_length(window):
    """Exactly the trainer's none-mode gate: pass-1 loss vs the token-by-token recurrence at T=1024."""
    model = _model("none", attn_window=window)
    inputs, targets = _batch(8, 1024)
    with torch.no_grad():
        parallel = model.pass_losses(inputs, targets, 1)[0]
        sequential = model.loss_sequential(inputs, targets)
    assert abs(float(parallel) - float(sequential)) / inputs.numel() < GATE_NATS, (float(parallel), float(sequential))


def test_fused_kernel_read_matches_the_chunked_form_forward_and_backward():
    """bf16-operand kernel vs the fp32 chunked reference at production geometry, including the decay gradient."""
    torch.manual_seed(5)
    B, T, H, d = 4, 1024, 4, 128
    q, k, v = (torch.randn(B, T, H, d, device=DEVICE) for _ in range(3))
    temperature = torch.ones(H, device=DEVICE)
    logits = torch.tensor([0.0, 2.2, 3.9, 6.9], device=DEVICE)
    results = {}
    for name, fn in (("kernel", linear_memory_read_kernel), ("chunked", linear_memory_read_chunked)):
        qg, kg, vg, lg = (x.clone().requires_grad_(True) for x in (q, k, v, logits))
        qf, kf = linear_memory_features(qg, temperature), linear_memory_features(kg, temperature)
        out = fn(qf, kf, vg.to(torch.bfloat16), torch.sigmoid(lg))
        (out.float() * torch.linspace(-1, 1, d, device=DEVICE)).sum().backward()
        results[name] = (out.float().detach(), qg.grad, kg.grad, vg.grad, lg.grad)
    labels = ("out", "dq", "dk", "dv", "ddecay")
    gaps = {}
    for label, a, b in zip(labels, results["kernel"], results["chunked"]):
        gaps[label] = float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))
    print(json.dumps({"kernel_vs_chunked_rel": gaps}))
    assert gaps["out"] < 2e-2 and gaps["dq"] < 5e-2 and gaps["dk"] < 5e-2 and gaps["dv"] < 5e-2 and gaps["ddecay"] < 5e-2, gaps
    assert torch.all(results["kernel"][0][:, 0] == 0)


def test_compiled_gla_adapter_matches_fla_forward_and_backward():
    """The graph boundary must not change FLA's gate scan or decay backward."""
    torch.manual_seed(23)
    shape = (2, 257, 4, 128)
    source = [torch.randn(shape, device=DEVICE, dtype=torch.bfloat16) for _ in range(3)]
    source.append(torch.tensor(DEFAULT_DECAYS, device=DEVICE).log().expand(2, 257, 4).contiguous())
    weights = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)

    def adapted(q, k, v, gate):
        return _lam_gla_forward(q, k, v, gate)[0]

    compiled = torch.compile(adapted, fullgraph=True, dynamic=False)
    results = []
    for fn in (lambda q, k, v, g: chunk_simple_gla(q, k, v, g=g, scale=1.0)[0], compiled):
        inputs = tuple(t.detach().clone().requires_grad_() for t in source)
        output = fn(*inputs)
        gradients = torch.autograd.grad((output * weights).sum(), inputs)
        results.append((output.detach(), *gradients))
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_linear_memory_parallel_matches_the_fast_weight_state_at_full_length():
    """The bf16-input parallel read (fp32 inside) against the fp32 recurrent state over 1024 positions, 4 production decays."""
    torch.manual_seed(3)
    B, T, H, d = 2, 1024, 4, 128
    q, k, v = (torch.randn(B, T, H, d, device=DEVICE) for _ in range(3))
    temperature = torch.ones(H, device=DEVICE)
    qf, kf = linear_memory_features(q, temperature), linear_memory_features(k, temperature)
    decays = torch.tensor(DEFAULT_DECAYS, device=DEVICE)
    v16 = v.to(torch.bfloat16)
    parallel = linear_memory_read_parallel(qf, kf, v16, decays)  # the dispatch: fused kernel on CUDA
    dense = linear_memory_read_dense(qf, kf, v16, decays)
    assert (linear_memory_read_chunked(qf, kf, v16, decays) - dense).abs().max() < 1e-4 * dense.abs().max()
    state = MemoryState(B, H, d, DEVICE)
    reads = []
    for t in range(T):
        reads.append(state.read(qf[:, t]))
        state.write(kf[:, t], v16[:, t], decays)
    recurrent = torch.stack(reads, 1)
    scale = recurrent.abs().max()
    gap = float((parallel - recurrent).abs().max() / scale)
    print(json.dumps({"dispatch_vs_state_rel": gap}))
    assert gap < 2e-2, gap
    chunked = linear_memory_read_chunked(qf, kf, v16, decays)
    assert (chunked - recurrent).abs().max() < 1e-4 * scale


def test_none_mode_matches_the_baseline_gpt():
    model = _model("none")
    base = GPT(VOCAB, LAYERS, DIM).to(DEVICE)
    base.load_state_dict(model.state_dict())
    inputs, targets = _batch(8, 256)
    with torch.no_grad():
        assert torch.equal(base(inputs, targets), model(inputs, targets, 2))


def test_window_path_matches_the_causal_kernel_when_it_covers_the_sequence():
    full, wide = _model("none"), _model("none", attn_window=256)
    inputs, targets = _batch(8, 256)
    with torch.no_grad():
        a, b = full(inputs, targets), wide(inputs, targets)
    assert abs(float(a) - float(b)) / inputs.numel() < 1e-3


@pytest.mark.parametrize("mode,window", [("none", 0), ("top", 0), ("glu", 0), ("add", 0), ("lam", 0), ("lam", 64)])
def test_real_shape_training_step_fits_and_reports_time(mode, window):
    model = _model(mode, live=False, attn_window=window).train()
    model.compile(dynamic=False)
    inputs, targets = _batch(64, 1024)
    passes = 1 if mode in ("none", "top") else 2
    torch.cuda.reset_peak_memory_stats()
    for i in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss = model(inputs, targets, passes)
        loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        model.zero_grad(set_to_none=True)
    peak = torch.cuda.max_memory_allocated() / 2**30
    assert torch.isfinite(loss)
    print(json.dumps({"mode": mode, "window": window, "passes": passes, "step_ms": round(1000 * elapsed, 1),
                      "peak_gib": round(peak, 2), "loss_per_token": round(float(loss) / inputs.numel(), 4)}))
    assert peak < 28.0
