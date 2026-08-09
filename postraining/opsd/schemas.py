"""Versioned contracts for OPSD checkpoints and training semantics."""

OPSD_CHECKPOINT_SCHEMA = "opsd_exact_resume/v2"
OPSD_EXPORT_SCHEMA = "opsd_backbone_export/v2"
OPSD_OBJECTIVE_SCHEMA = (
    "on_policy_fixed_step0_privileged_teacher_full_vocab_forward_kl_"
    "configured_pointwise_clip_student_grad_only/v1"
)
OPSD_PROMPT_SCHEMA = "privilege_then_bare_problem_token_completion/v3"
OPSD_DATA_ORDER_SCHEMA = (
    "seeded_epoch_shuffle_with_optional_exact_source_quota_cycle/v2"
)
OPSD_OPTIMIZER_SCHEMA = (
    "torch_adamw_betas0.9_0.999_eps1e-8_configured_decay_and_grad_clip/v1"
)
