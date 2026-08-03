"""Per-dim energy readout with one pooled B=512 SIGReg statistic per step.

Builds on the per-dimension scale arm
(``energy_readout/fresh_lejepa_train_energy_readout_perdim.py``).  The paired scheme computes
four B=128 statistics per optimizer step (one per microbatch pair); le-wm
computes exactly ONE statistic per step on its whole batch.  This module is the
fully aligned version: all eight B=64 microbatches of a step are re-encoded in
one pooled forward and a single B=512 Epps-Pulley statistic is taken.

Because the statistic is scaled by B, the null floor is B-independent while the
deviation signal grows linearly in B: pooling to 512 quadruples test power over
paired-128 at the same sigreg weight (0.09), so no weight re-tune is required.
It also removes the paired scheme's even-local-accumulation requirement, which
is what breaks at 8xH100 world_size=8 (grad_accum_steps=1).  This module still
requires the WHOLE 512-sequence step batch on one rank
(``local_sequences * grad_accum_steps == 512``); a multi-rank run would need a
cross-rank gather that is deliberately not implemented here.

Gradient bookkeeping mirrors the paired install exactly: the paired path
backwards ``stat * weight * 2 * grad_scale`` per pair (accum/2 pairs per step);
the pooled path backwards ``stat * weight * grad_accum_steps * grad_scale``
once, which is the identical total scale for any definition of ``grad_scale``.
The logged ``train_loss``/``train_components[2]`` contributions are multiplied
by ``grad_accum_steps`` so the later ``/= grad_accum_steps`` recovers the
per-step statistic unchanged.

The pooled encode runs the BN projector in train mode over the full 512x1025
token batch, so BatchNorm sees one batch statistic per step -- also the le-wm
configuration (their projector normalizes the whole batch at once).
"""

from __future__ import annotations

import inspect
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import Tensor

from pretraining.fresh_lejepa import fresh_lejepa_train_v4 as v4
import train_gpt as baseline
import energy_readout.fresh_lejepa_train_energy_readout_perdim as perdim

ARCHITECTURE = (
    "energy_readout_lejepa_tied_codebook_bnproj_perdim_b512sigreg_onepass_2k"
)


class EnergyReadoutPerDimB512SigregLeJEPA(perdim.EnergyReadoutPerDimScaleLeJEPA):
    """Per-dim BN-projector energy readout scored by one pooled SIGReg per step."""

    def pooled_sigreg_loss(self, batches: list[tuple[Tensor, Tensor]]) -> Tensor:
        """Compute one exact B=512 statistic from a full step's microbatches."""
        trajectories = [
            torch.cat((input_ids, target_ids[:, -1:]), dim=1)
            for input_ids, target_ids in batches
        ]
        trajectory_latent = self.embed_tokens(torch.cat(trajectories, dim=0))
        return self.sigreg(trajectory_latent)

    @classmethod
    def experiment_metadata(cls) -> dict[str, str | int | float]:
        metadata = super().experiment_metadata()
        metadata.update({"sigreg_batch": "pooled_512_once_per_step"})
        return metadata


def _install_pooled_accumulation(
    default_steps: int = 8,
    extra_components: tuple[tuple[str, str], ...] = (),
):
    """Fork baseline.main in memory with pooled-B512 SIGReg accumulation.

    Same machinery and signature as ``v4._install_configurable_accumulation``
    (so ``pope.main`` can call it unchanged); only the guard, warmup, and
    training microbatch blocks differ -- they collect every microbatch of a
    step and take one pooled statistic instead of pairing them.
    """
    original = baseline.main
    source = textwrap.dedent(inspect.getsource(original))
    old = (
        "if 8 % world_size != 0:\n"
        "        raise ValueError(f\"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral\")\n"
        "    grad_accum_steps = 8 // world_size"
    )
    new = (
        f"total_grad_accum_steps = int(os.environ.get('GRAD_ACCUM_STEPS', '{default_steps}'))\n"
        "    if total_grad_accum_steps <= 0 or total_grad_accum_steps % world_size != 0:\n"
        "        raise ValueError(\n"
        "            f\"GRAD_ACCUM_STEPS={total_grad_accum_steps} must be positive and divisible by WORLD_SIZE={world_size}\"\n"
        "        )\n"
        "    grad_accum_steps = total_grad_accum_steps // world_size\n"
        "    local_sequences = args.train_batch_tokens // (world_size * grad_accum_steps * args.train_seq_len)\n"
        "    if local_sequences * grad_accum_steps != 512:\n"
        "        raise ValueError(\n"
        "            f\"pooled SIGReg needs the whole step batch B=512 on one rank, \"\n"
        "            f\"got {local_sequences * grad_accum_steps} (cross-rank pooling not implemented)\"\n"
        "        )"
    )
    if source.count(old) != 1:
        raise RuntimeError("upstream grad accumulation block changed")
    source = source.replace(old, new)
    warmup_old = (
        "warmup_loss = model(x, y)\n"
        "                (warmup_loss * grad_scale).backward()"
    )
    warmup_new = (
        "warmup_loss, _warmup_components = model(x, y)\n"
        "                (warmup_loss * grad_scale).backward()\n"
        "                if micro_step == 0:\n"
        "                    pending_sigreg_batches = []\n"
        "                pending_sigreg_batches.append((x, y))\n"
        "                if micro_step == grad_accum_steps - 1:\n"
        "                    pooled_sigreg = base_model.pooled_sigreg_loss(pending_sigreg_batches)\n"
        "                    (pooled_sigreg * base_model.sigreg_loss_weight * grad_accum_steps * grad_scale).backward()"
    )
    train_init_old = 'train_loss = torch.zeros((), device=device)'
    component_count = 3 + len(extra_components)
    train_init_new = (
        'train_loss = torch.zeros((), device=device)\n'
        f'        train_components = torch.zeros({component_count}, device=device)'
    )
    train_forward_old = (
        "loss = model(x, y)\n"
        "            train_loss += loss.detach()"
    )
    train_forward_new = (
        "loss, loss_components = model(x, y)\n"
        "            train_loss += loss.detach()\n"
        "            train_components += loss_components\n"
        "            (loss * grad_scale).backward()\n"
        "            if micro_step == 0:\n"
        "                pending_sigreg_batches = []\n"
        "            pending_sigreg_batches.append((x, y))\n"
        "            if micro_step == grad_accum_steps - 1:\n"
        "                pooled_sigreg = base_model.pooled_sigreg_loss(pending_sigreg_batches)\n"
        "                (pooled_sigreg * base_model.sigreg_loss_weight * grad_accum_steps * grad_scale).backward()\n"
        "                train_loss += grad_accum_steps * base_model.sigreg_loss_weight * pooled_sigreg.detach()\n"
        "                train_components[2] += grad_accum_steps * pooled_sigreg.detach()"
    )
    train_mean_old = (
        '(loss * grad_scale).backward()\n'
        '        train_loss /= grad_accum_steps'
    )
    extra_loss_terms = "".join(
        f'\n            + base_model.{weight_attr} * train_components[{index}]'
        for index, (_, weight_attr) in enumerate(extra_components, start=3)
    )
    train_mean_new = (
        'train_loss /= grad_accum_steps\n'
        '        train_components /= grad_accum_steps\n'
        '        train_loss = (\n'
        '            train_components[0]\n'
        '            + base_model.latent_loss_weight * train_components[1]\n'
        '            + base_model.sigreg_loss_weight * train_components[2]'
        f'{extra_loss_terms}\n'
        '        )'
    )
    log_old = (
        'f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "\n'
        '                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"'
    )
    extra_log_fields = "".join(
        f'                f"{metric_name}:{{train_components[{index}].item():.4f}} "\n'
        for index, (metric_name, _) in enumerate(extra_components, start=3)
    )
    log_new = (
        'f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "\n'
        '                f"policy_loss:{train_components[0].item():.4f} "\n'
        '                f"latent_loss:{train_components[1].item():.4f} "\n'
        '                f"sigreg_loss:{train_components[2].item():.4f} "\n'
        f'{extra_log_fields}'
        '                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"'
    )
    for expected, replacement, label in (
        (warmup_old, warmup_new, "warmup output"),
        (train_init_old, train_init_new, "component accumulator"),
        (train_forward_old, train_forward_new, "training output"),
        (train_mean_old, train_mean_new, "component mean"),
        (log_old, log_new, "component logging"),
    ):
        if source.count(expected) != 1:
            raise RuntimeError(f"upstream {label} block changed")
        source = source.replace(expected, replacement)
    exec(compile(source, inspect.getsourcefile(original) or "train_gpt.py", "exec"), baseline.__dict__)
    return original


def main() -> None:
    # perdim.main() resolves its class, architecture, and __file__ as module
    # globals at call time, and pope.main() resolves the accumulation
    # installer as a v4 attribute at call time, so patching all four and
    # delegating reuses the per-dim run wiring (canaries included) verbatim.
    original_install = v4._install_configurable_accumulation
    original_class = perdim.EnergyReadoutPerDimScaleLeJEPA
    original_architecture = perdim.ARCHITECTURE
    original_file = perdim.__file__
    v4._install_configurable_accumulation = _install_pooled_accumulation
    perdim.EnergyReadoutPerDimScaleLeJEPA = EnergyReadoutPerDimB512SigregLeJEPA
    perdim.ARCHITECTURE = ARCHITECTURE
    perdim.__file__ = str(Path(__file__).resolve())
    try:
        perdim.main()
    finally:
        v4._install_configurable_accumulation = original_install
        perdim.EnergyReadoutPerDimScaleLeJEPA = original_class
        perdim.ARCHITECTURE = original_architecture
        perdim.__file__ = original_file


if __name__ == "__main__":
    main()
