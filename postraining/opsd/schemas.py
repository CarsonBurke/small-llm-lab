"""Versioned contracts for OPSD checkpoints and training semantics."""

OPSD_CHECKPOINT_SCHEMA = "opsd_exact_resume/v1"
OPSD_EXPORT_SCHEMA = "opsd_backbone_export/v1"
OPSD_OBJECTIVE_SCHEMA = (
    "on_policy_fixed_step0_privileged_teacher_full_vocab_forward_kl_"
    "configured_pointwise_clip_student_grad_only/v1"
)
OPSD_PROMPT_SCHEMA = "reference_solution_then_independent_derivation/v1"
OPSD_DATA_ORDER_SCHEMA = "seeded_epoch_shuffle_without_replacement/v1"
OPSD_OPTIMIZER_SCHEMA = (
    "torch_adamw_betas0.9_0.999_eps1e-8_configured_decay_and_grad_clip/v1"
)
