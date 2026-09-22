#!/usr/bin/env python3
"""Queue-only controlled MiniCPM Gaussian arithmetic experiment."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from postraining.vapo.policy import VAPOPolicy
from postraining.vapo.model.lora import LoRAConfig
from postraining.fast_inference import build_fused_rollout_replica
from postraining.invariant_linear import InvariantLinear
from postraining.train_minicpm_vapo import encode_math_prompt

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
assert torch.cuda.is_available()
torch.manual_seed(1337)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
side, tokenizer = VAPOPolicy.from_family("minicpm5", 
    device=torch.device('cuda'), lora_config=LoRAConfig(initialization='nora'),
    latent_thinking=True, gradient_checkpointing=False,
)
side.eval()
ids = encode_math_prompt(
    tokenizer, {'prompt': [{'role': 'user', 'content': 'Prove that the square root of 2 is irrational.'}]},
    prompt_tokens=1024, enable_thinking=True,
).to('cuda')
report = {'prompt_tokens': ids.numel(), 'component_std': side.transition.component_std, 'cases': {}}


def compare(model, backend, tag):
    model.causal_lm.config._attn_implementation = 'sdpa'
    outputs = {}
    for rows in (1, 2):
        states = []
        handles = [layer.register_forward_hook(lambda mod, inp, out: states.append(out[0, -1].float().cpu())) for layer in model.causal_lm.model.layers]
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16), sdpa_kernel(backend):
            output = model.replay_hidden(ids[None].expand(rows, -1).contiguous(), None)[0, -1].float().cpu()
        for handle in handles:
            handle.remove()
        outputs[rows] = (states, output)
    delta = outputs[1][1] - outputs[2][1]
    report['cases'][tag] = {
        'mean_l2': delta.norm().item(),
        'gaussian_kl': (0.5 * (delta / side.transition.component_std).square().sum()).item(),
        'per_layer_l2': [(a-b).norm().item() for a,b in zip(outputs[1][0],outputs[2][0],strict=True)],
        'mean_norm': outputs[1][1].norm().item(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(tag, json.dumps(report['cases'][tag]), flush=True)


compare(side, SDPBackend.FLASH_ATTENTION, 'unmerged_flash')
compare(side, SDPBackend.MATH, 'unmerged_math_control')
with torch.inference_mode():
    fused, _ = build_fused_rollout_replica(side)
compare(fused, SDPBackend.FLASH_ATTENTION, 'fused_flash')
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
compare(fused, SDPBackend.FLASH_ATTENTION, 'fused_flash_fullaccum')
# Canonical dot reduction: every token occupies lane zero in the same tile.
# This intentionally inefficient control isolates cuBLAS shape selection.
for name, module in list(fused.causal_lm.named_modules()):
    if isinstance(module, torch.nn.Linear) and name != 'lm_head':
        replacement = InvariantLinear.from_linear(module)
        replacement.decoding = True
        parent_name, _, child = name.rpartition('.')
        setattr(fused.causal_lm.get_submodule(parent_name), child, replacement)
compare(fused, SDPBackend.FLASH_ATTENTION, 'canonical_gemm_flash')
compare(fused, SDPBackend.MATH, 'canonical_gemm_math_control')
