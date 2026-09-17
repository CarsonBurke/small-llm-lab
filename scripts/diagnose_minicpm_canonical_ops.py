#!/usr/bin/env python3
"""Queue-only operation-level diagnosis of the first long-prefix mismatch."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.diagnose_minicpm_canonical import (
    AttentionControl, cached_run, full_run, projection_variant,
)
from postraining.minicpm_vapo import MiniCPMVAPOPolicy, LoRAConfig
from postraining.fast_inference import build_fused_rollout_replica

p=argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--streams', type=Path, default=Path('ablation_results/minicpm_low_noise_20260911/canonical_attention_v1.streams.pt'))
p.add_argument('--position', type=int, default=157)
a=p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
torch.manual_seed(1337)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32=True
assert torch.cuda.is_available()
source,_=MiniCPMVAPOPolicy.from_pretrained(device=torch.device('cuda'), lora_config=LoRAConfig(), latent_thinking=True, gradient_checkpointing=False)
source.eval()
with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
    policy,_=build_fused_rollout_replica(source)
    stream=torch.load(a.streams, map_location='cpu', weights_only=False)['streams'][1]
    prompt=stream['prompt'].to('cuda')
    embeddings=stream['embeddings'].to('cuda')
    captures={}
    projection_context=projection_variant(policy,'flat')
    projection_context.__enter__()
    for mode in ('cached','full'):
        values={}
        handles=[]
        for name,module in policy.causal_lm.model.named_modules():
            if name.startswith('layers.') and (not tuple(module.children()) or name.endswith(('q_proj','k_proj','v_proj','gate_proj','up_proj','down_proj','o_proj'))):
                def observe(mod, args, output, name=name, state=[0]):
                    index=state[0]
                    state[0]+=1
                    if mode=='cached' and index!=a.position:
                        return
                    def selected(tensor):
                        if not isinstance(tensor, torch.Tensor) or tensor.ndim!=3:
                            return None
                        position=0 if mode=='cached' else prompt.numel()-1+a.position
                        return tensor[0,position].float().cpu()
                    inp=selected(args[0]) if args else None
                    out=selected(output)
                    if inp is not None:
                        values[name+':input']=inp
                    if out is not None:
                        values[name+':output']=out
                handles.append(module.register_forward_hook(observe))
        control=AttentionControl('canonical',(64,64))
        if mode=='cached':
            cached_run(policy,control,prompt,embeddings)
        else:
            full_run(policy,control,prompt,embeddings)
        for handle in handles:
            handle.remove()
        captures[mode]=values
    projection_context.__exit__(None,None,None)
    report={'position':a.position,'absolute_position':prompt.numel()-1+a.position,'comparisons':[]}
    for name,x in captures['cached'].items():
        if name not in captures['full']:
            continue
        y=captures['full'][name]
        delta=x-y
        row={'name':name,'l2':delta.norm().item(),'max_abs':delta.abs().max().item(),'different':int((delta!=0).sum())}
        report['comparisons'].append(row)
        if row['different']:
            print(json.dumps(row),flush=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
