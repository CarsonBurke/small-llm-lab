# scripts/

Operational entry points: dataset builders, launchers, benchmarks, profilers,
diagnostics, and report renderers. The model and training code they drive
lives in `postraining/` and `pretraining/`. Run everything as a module from
the repository root, for example
`.venv/bin/python -m scripts.build_math_drills --help`. Anything that executes
a model on the GPU goes through `mlq` (see `CLAUDE.md`).

The directory is flat on purpose. Run manifests and gate reports bind script
paths into their provenance hashes, and tests and package code import
`scripts.<name>`. Moving files would break resume validation and invalidate
recorded lineage. Retire a script by deleting it: git history keeps it, and
completed runs keep their own source snapshots.

## KDA8 post-training (current line)

- `launch_kda8_posttrain.sh`: SFT, then cot RL, then carry RL from the
  canonical KDA base (`sft`, `rl`, `rl-carry`). `AFTER_SUCCESS=<job>` chains
  the submission behind another mlq job.
- `benchmark_sft_step`: production SFT step throughput on real packed rows.
- `audit_problem_overlap`: how much of the pretraining corpus is a
  post-training problem.
- `build_problem_registry`: global math problem registry and contamination
  index.
- `build_math_drills`: worked arithmetic drill corpus and held-out probe panel.
- `build_deepmind_rl_prompts`, `build_ultradata_math_rl_prompts`: RL prompt
  pools.
- `build_deepmind_eval_set`, `build_aime_2025_eval_set`,
  `build_aime_2026_eval_set`: held-out evaluation parquets.
- `prepare_verifiable_tasks`: bounded, source-backed verifiable tasks.
- `eval_generated_external_bpb`: score generated responses with a frozen
  external byte model.

Training-rollout transcripts are rendered by
`python -m postraining.rollout_report`, not by a script here.

## KDA pretraining and base checkpoints

- Corpus: `prepare_k3_pretrain_sources`, `build_k3_pretrain_dataset`,
  `build_k3_pretrain_dataset_checkpointed`, `build_math_mix_dataset`,
  `build_mathglm_corpus`, `sample_tokenizer_corpus`.
- Training: `run_k3_context_curriculum` (2K, then 4K, then 8K context),
  `train_kda_state_routing`.
- Kernels: `benchmark_kda_training`, `validate_kda_gdn2_reduction`,
  `validate_compile_grad_accum`.

## Gated DeltaNet 2 and recurrent memory

- GDN2: `benchmark_gated_delta`, `profile_gated_delta`,
  `benchmark_gdn2_custom_ops`, `benchmark_gdn2_execution`,
  `benchmark_gdn2_inference`, `benchmark_gdn2_wy_tiles`, `profile_gdn2_wy`,
  `pin_gdn2_autotune`, `probe_gdn2_pool_variants`, `scan_gdn2_pool_microbatch`,
  `probe_gdn2_state_use`, `diagnose_gdn2_checkpoint`, `evaluate_gdn2_context`.
- Recurrent slots: `train_recurrent_slots`, `benchmark_recurrent_slots`,
  `benchmark_recurrent_slots_concurrent`,
  `diagnose_recurrent_slots_checkpoint`, `diagnose_recurrent_slots_packing`,
  `verify_recurrent_slots_packing`.
- Latent feedback and carry: `benchmark_feedback_lam`,
  `evaluate_feedback_memory_backends`, `compare_feedback_arms`,
  `train_latent_carry`, `benchmark_latent_carry`, `profile_latent_carry`,
  `diagnose_latent_refiner`, `benchmark_chunk_memory`.
- Frozen dynamics: `diagnose_dynamics`, `speculate_dynamics`.

## nanoGPT-mini variants and streaming FFNs

- `train_nanogpt_mini_full_bandwidth`, `train_nanogpt_mini_full_bandwidth_stream`.
- Streaming: `train_future_credit_stream`, `compare_future_credit_stream`,
  `train_stationary_stream`, `compare_stationary_stream`.
- CELF and Cola controls: `train_celf`, `diagnose_celf_checkpoint`,
  `train_cola`, `prepare_cola_bpe`, `publish_cola_proxy`.
- Characters and embedders: `native_bits`, `benchmark_embedder`,
  `benchmark_latent_moe`, `benchmark_fp8_linear`.

## Byte diffusion

- Data: `build_byte_diffusion_dataset`, `build_bolmo_dataset`,
  `materialize_byte_corpus_gpt2_view`, `fit_byte_entropy_patcher`.
- Training: `train_byte_diffusion`, `train_byte_diffusion_gemma`,
  `train_byte_duo`, `train_byte_idlm`, `preflight_byte_diffusion_resume`.
- Evaluation: `eval_byte_diffusion`, `eval_byte_duo_nelbo`,
  `eval_byte_duo_time_buckets`.
- Export: `export_byte_diffusion`, `export_byte_duo`,
  `audit_byte_diffusion_artifact`.
- Speed: `benchmark_byte_diffusion`, `benchmark_byte_diffusion_architectures`,
  `benchmark_byte_diffusion_fast_blt`, `benchmark_byte_diffusion_inference`,
  `benchmark_byte_diffusion_real_data`, `benchmark_byte_diffusion_validation`,
  `benchmark_byte_duo_distributed`, `benchmark_byte_duo_inference`,
  `profile_byte_duo`.
- Reports: `summarize_byte_duo_run`, `render_byte_diffusion_report`,
  `render_byte_duo_study_report`.
- Upstream reference: `benchmark_ar_reference`.

## MiniCPM post-training (paused line)

- Runs: `run_minicpm_vapo`, `preflight_minicpm_vapo`, `evaluate_minicpm_vapo`,
  `benchmark_minicpm_vapo_sweep`, `launch_minicpm_gate10_nora_mb4_v3_10k.sh`.
- Data and critic: `build_minicpm_math_mix`, `generate_minicpm_critic_corpus`,
  `pretrain_minicpm_critic`.
- Evaluation: `evaluate_minicpm_tasks`, `evaluate_minicpm_kodcode`,
  `evaluate_minicpm_stock_holdout`.
- Latent controller: `train_minicpm_latent_controller`,
  `profile_minicpm_latent_controller`, `analyze_minicpm_controller_geometry`,
  `benchmark_minicpm_latent`, `diagnose_minicpm_latent_bridge`,
  `diagnose_minicpm_latent_lengths`, `diagnose_minicpm_low_noise`,
  `diagnose_minicpm_antithetic`, `diagnose_minicpm_token_carry`.
- Canonical bf16 kernels: `minicpm_canonical_attention_probe`,
  `minicpm_canonical_norm_probe`, `minicpm_canonical_projection_probe`,
  `minicpm_canonical_training_probe`, `diagnose_minicpm_canonical`,
  `diagnose_minicpm_canonical_ops`, `ablate_minicpm_exact_replay_paths`,
  `minicpm_coupled_rng`.
- Sampling and Uno: `benchmark_minicpm_sampling`, `benchmark_minicpm_nextlat`,
  `benchmark_minicpm_uno`, `train_minicpm_uno`, `benchmark_split_kv_plan`.
- Model comparison: `benchmark_lora_training_models`.
- Artifact migration (tests depend on these): `migrate_minicpm_tensorboard`,
  `migrate_minicpm_vapo_v6`.

## Answer encoder (LeJEPA)

- `train_answer_encoder`, `probe_answer_patch_sets`,
  `probe_qwen3_answer_embeddings`, `benchmark_qwen3_answer_embeddings`.

## Generic tooling

- `ablation`, `plot_ablations`: legacy parameter-golf ablation runner and
  plots.
- `tb_watcher`, `tb_log_bridge`: stream metrics JSONL or console logs into
  TensorBoard.
