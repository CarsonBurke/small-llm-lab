"""Which memory replay diverges under torch.compile: grad or no-grad.

Run through mlq. Reuses the CUDA parity test's live-memory actor and rollout.
"""
import torch

import postraining.tests.test_mac_update_cuda as parity
from postraining.latent_rollout import compact_slots, replay_head_inputs


def main():
    device = torch.device("cuda")
    for flag in ("allow_tf32",):
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.fx.experimental._config.use_duck_shape = False
    wrapper, _ = parity._actor_and_study_critic(device, twin=True, memory=True)
    batch = parity._scored_rollout(wrapper, device)
    compiled = torch.compile(replay_head_inputs, mode="default", fullgraph=False, dynamic=True)
    mask = batch.emit_mask.bool()
    results = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            results["eager_nograd"] = replay_head_inputs(wrapper, batch)
        results["eager_grad"] = tuple(x.detach() for x in replay_head_inputs(wrapper, batch))
        results["compiled_grad"] = tuple(x.detach() for x in compiled(wrapper, batch))
        with torch.no_grad():
            results["compiled_nograd"] = compiled(wrapper, batch)
        results["compiled_grad_again"] = tuple(x.detach() for x in compiled(wrapper, batch))
    ref_b, ref_i = results["eager_nograd"]
    for name, (b, i) in results.items():
        db = (b.float() - ref_b.float())[mask].abs()
        di = (i.float() - ref_i.float())[mask].abs()
        print(name, {"belief_max": float(db.max()), "belief_mean": float(db.mean()),
                     "input_max": float(di.max()), "input_mean": float(di.mean())}, flush=True)
    # Inputs alone: is it the memory read or the trunk?
    from postraining.latent_rollout import assemble_stream_latents
    ca = torch.compile(assemble_stream_latents, mode="default", fullgraph=False, dynamic=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            e = assemble_stream_latents(wrapper, batch)
        g = ca(wrapper, batch).detach()
        with torch.no_grad():
            n = ca(wrapper, batch)
    for name, x in (("assemble_compiled_grad", g), ("assemble_compiled_nograd", n)):
        d = (x.float() - e.float())[mask].abs()
        print(name, {"max": float(d.max()), "mean": float(d.mean())}, flush=True)


if __name__ == "__main__":
    main()
