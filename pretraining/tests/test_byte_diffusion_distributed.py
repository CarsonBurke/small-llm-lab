from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pretraining.byte_diffusion.config import ByteDiffusionConfig, CorruptionConfig
from pretraining.byte_diffusion.data import (
    AtomicDocument,
    AtomicIdManifest,
    DeterministicChunkCursor,
    pack_documents,
)
from pretraining.byte_diffusion.model import ByteDiffusionModel
from pretraining.byte_diffusion.training import (
    ByteDiffusionTrainer,
    DistributedContext,
    TrainingRunConfig,
)


def _chunks():
    manifest = AtomicIdManifest.reference()
    documents = tuple(
        AtomicDocument(
            f"rank-shard-{index}",
            (65 + index, 66, 67, 68, 69, 70, 71, manifest.eot_id),
        )
        for index in range(8)
    )
    return pack_documents(documents, manifest, chunk_size=8)


def _config() -> TrainingRunConfig:
    return TrainingRunConfig(
        iterations=2,
        val_loss_every=1,
        train_log_every=1,
        warmdown_iters=0,
        run_id="cpu-ddp-contract",
        seed=101,
        recipe="causal_only",
        corruption=CorruptionConfig(
            kind="absorbing_rb", canvas_length=8, branches_per_row=1
        ),
        microbatch_per_rank=1,
        gradient_accumulation=2,
        attention_policy="dense_reference",
        allow_cpu_reference=True,
        compile_model=False,
    )


def _worker(rank: int, world_size: int, init_path: str, checkpoint: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )
    try:
        chunks = tuple(_chunks())
        config = _config()
        context = DistributedContext(rank=rank, local_rank=rank, world_size=world_size)
        torch.manual_seed(55)
        trainer = ByteDiffusionTrainer(
            ByteDiffusionModel(ByteDiffusionConfig.tiny()),
            DeterministicChunkCursor(chunks, seed=config.seed),
            chunks,
            config,
            device=torch.device("cpu"),
            distributed=context,
        )
        first = trainer.run_update()
        gathered_loss = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_loss, first.total)
        assert len(set(gathered_loss)) == 1
        validation = trainer.validate()
        gathered_bpb = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_bpb, validation.bpb)
        assert len(set(gathered_bpb)) == 1

        trainer.save_checkpoint(Path(checkpoint))
        trainer.run_update()
        expected = {
            name: value.detach().clone()
            for name, value in trainer.joint.model.state_dict().items()
        }

        torch.manual_seed(999)
        resumed = ByteDiffusionTrainer(
            ByteDiffusionModel(ByteDiffusionConfig.tiny()),
            DeterministicChunkCursor(chunks, seed=config.seed),
            chunks,
            config,
            device=torch.device("cpu"),
            distributed=context,
        )
        resumed.load_checkpoint(Path(checkpoint))
        resumed.run_update()
        for name, value in resumed.joint.model.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

        # DDP averaging must leave identical parameters on every rank.
        parameter = next(resumed.joint.model.parameters()).detach()
        gathered = [torch.empty_like(parameter) for _ in range(world_size)]
        dist.all_gather(gathered, parameter)
        for peer in gathered[1:]:
            torch.testing.assert_close(peer, gathered[0], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_update_validation_and_exact_resume(tmp_path: Path) -> None:
    mp.spawn(
        _worker,
        args=(2, str(tmp_path / "gloo-init"), str(tmp_path / "checkpoint.pt")),
        nprocs=2,
        join=True,
    )
