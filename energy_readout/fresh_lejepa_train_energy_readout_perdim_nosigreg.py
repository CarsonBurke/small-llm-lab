"""Per-dim BN-projector energy readout with the SIGReg apparatus removed.

The weight-0 run (job 247) established that energy-CE holds the latent
geometry open on its own: at matched steps it ties the sigreg'd runs on BPB,
and checkpoint analysis at step 200 shows per-dim scale spread, effective
rank, and mean offset all comparable to the sigreg'd geometry (per-dim std
max/median 1.90 vs 1.86, effective rank 21.4 vs 12.5, |mean| 8.2 vs 9.3).
SIGReg's only measured contribution in this family is cost: the pooled
re-encode is ~11% of step time, plus the paired/pooled batch-shape guards
that break at 8xH100 world_size=8.

This module removes the apparatus instead of zero-weighting it: no sigreg
re-encode, no statistic, no RNG projection draws, no batch-shape guard.  The
accumulation installer keeps v4's component bookkeeping (the model still
returns three loss components; the sigreg slot stays 0) so metrics and
logging remain schema-compatible with every sibling run.  Grad accumulation
is constrained only by divisibility, restoring the full-scale flexibility the
paired scheme lost.
"""

from __future__ import annotations

import inspect
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.fresh_lejepa import fresh_lejepa_train_v4 as v4
import train_gpt as baseline
import energy_readout.fresh_lejepa_train_energy_readout_perdim as perdim

ARCHITECTURE = (
    "energy_readout_lejepa_tied_codebook_bnproj_perdim_nosigreg_onepass_2k"
)


class EnergyReadoutPerDimNoSigregLeJEPA(perdim.EnergyReadoutPerDimScaleLeJEPA):
    """Per-dim BN-projector energy readout trained without SIGReg."""

    # The forward already defers sigreg out of the loss; with the loop-side
    # computation removed the weight only appears in the logged-loss
    # recomposition, where the sigreg component is identically zero.
    sigreg_loss_weight = 0.0

    @classmethod
    def experiment_metadata(cls) -> dict[str, str | int | float]:
        metadata = super().experiment_metadata()
        metadata.update({"sigreg_batch": "removed"})
        return metadata


def _install_nosigreg_accumulation(
    default_steps: int = 8,
    extra_components: tuple[tuple[str, str], ...] = (),
):
    """Fork baseline.main in memory with component bookkeeping and no SIGReg.

    Same machinery and signature as ``v4._install_configurable_accumulation``
    (so ``pope.main`` can call it unchanged); the guard keeps only the
    divisibility requirement, and the warmup/training microbatch blocks carry
    no sigreg computation at all.
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
        "    grad_accum_steps = total_grad_accum_steps // world_size"
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
        "                (warmup_loss * grad_scale).backward()"
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
        "            (loss * grad_scale).backward()"
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
    # installer as a v4 attribute at call time; patch all four and delegate.
    original_install = v4._install_configurable_accumulation
    original_class = perdim.EnergyReadoutPerDimScaleLeJEPA
    original_architecture = perdim.ARCHITECTURE
    original_file = perdim.__file__
    v4._install_configurable_accumulation = _install_nosigreg_accumulation
    perdim.EnergyReadoutPerDimScaleLeJEPA = EnergyReadoutPerDimNoSigregLeJEPA
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
