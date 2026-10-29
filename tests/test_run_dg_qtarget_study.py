"""Queue-state and durable-resume tests; never execute a model or submit a job."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import run_dg_qtarget_study as supervisor


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.study = {"source_root": "/frozen", "source_sha256": {}, "joint_updates": 950}
        self.arm = {
            "argv": ["python", "-u", "-m", "postraining.train_latent_vapo",
                     "--output", "/run", "--steps", "950", "--value-warmup-steps", "50",
                     "--value-warmup-rewind", "--actor-init", "/init", "--fresh-critic",
                     "--zero-reward-stop-pools", "32"],
            "output": "/run", "submissions": [{"id": 1}],
        }
        self.metadata = {"path": "/run/latent_vapo_checkpoint.pt", "sha256": "abc", "step": 100, "value_warmup_step": 50}

    def test_resume_preserves_budget_and_removes_initialization(self):
        argv = supervisor.resume_argv(self.arm["argv"], Path(self.metadata["path"]))
        self.assertNotIn("--actor-init", argv)
        self.assertNotIn("--fresh-critic", argv)
        self.assertIn("--value-warmup-rewind", argv)
        self.assertEqual(argv[argv.index("--steps") + 1], "950")
        self.assertEqual(argv[argv.index("--value-warmup-steps") + 1], "50")
        self.assertEqual(argv[-2:], ["--resume", self.metadata["path"]])

    def test_only_explicit_time_limit_failures_resume(self):
        for reason in ("time limit of 1800000 ms exceeded", "time limit exceeded; orphaned process group drained"):
            job = {"id": 1, "state": "failed", "attempts": [{"message": reason}]}
            self.assertEqual(supervisor.decide(self.study, self.arm, job, "", self.metadata)[0], "resume")
        job = {"id": 1, "state": "failed", "stateReason": "CUDA out of memory"}
        self.assertEqual(supervisor.decide(self.study, self.arm, job, "", self.metadata)[0], "failed")
        job["state"] = "cancelled"
        self.assertEqual(supervisor.decide(self.study, self.arm, job, "time limit exceeded", self.metadata)[0], "failed")

    def test_timeout_without_checkpoint_stops(self):
        job = {"id": 1, "state": "failed", "stateReason": "time limit exceeded"}
        self.assertEqual(supervisor.decide(self.study, self.arm, job, "", None)[0], "needs_attention")

    def test_native_wall_clock_messages_with_step_ceiling_resume(self):
        job = {"id": 1, "state": "succeeded"}
        for step in (76, 65):
            for suffix in (f"{step}/950", str(step)):
                stdout = f"--max-train-hours 0.4 reached at step {suffix}; stopping at the pool boundary"
                metadata = {**self.metadata, "step": step}
                self.assertEqual(supervisor.decide(self.study, self.arm, job, stdout, metadata)[0], "resume")

    def test_native_cull_never_resumes(self):
        job = {"id": 1, "state": "succeeded"}
        stdout = "32 consecutive constant-reward pools at step 123/950; stopping at the pool boundary"
        self.assertEqual(supervisor.decide(self.study, self.arm, job, stdout, self.metadata)[0], "culled")

    def test_repeated_checkpoint_requires_attention(self):
        arm = copy.deepcopy(self.arm)
        arm["submissions"].append({"id": 2, "resume_checkpoint_sha256": "abc"})
        job = {"id": 2, "state": "failed", "stateReason": "time limit exceeded"}
        self.assertEqual(supervisor.decide(self.study, arm, job, "", self.metadata)[0], "needs_attention")

    def test_ceiling_requires_final_evaluations(self):
        metadata = {**self.metadata, "step": 950}
        self.assertEqual(supervisor.decide(self.study, self.arm, {"id": 1, "state": "succeeded"}, "", metadata)[0], "training_complete")

    def test_rewritten_checkpoint_without_updates_requires_attention(self):
        arm = copy.deepcopy(self.arm)
        arm["submissions"][-1]["resume_checkpoint"] = self.metadata
        rewritten = {**self.metadata, "sha256": "different"}
        job = {"id": 1, "state": "failed", "stateReason": "time limit exceeded"}
        self.assertEqual(supervisor.decide(self.study, arm, job, "", rewritten)[0], "needs_attention")
        arm["submissions"][-1]["resume_checkpoint"] = {"step": 0, "value_warmup_step": 20}
        progressed = {**rewritten, "step": 0, "value_warmup_step": 21}
        self.assertEqual(supervisor.decide(self.study, arm, job, "", progressed)[0], "resume")

    def test_final_modes_preserve_ceiling_and_checkpoint(self):
        arm = {**self.arm, "last_observation": {"checkpoint": {**self.metadata, "step": 950}}}
        for kind in ("bench", "bpb", "aime"):
            with patch.object(supervisor, "sha256", return_value="abc"):
                plan = supervisor.plan_evaluation(Path("/study.json"), self.study, "adaptive", arm, kind)
            command = plan["command"]
            argv = command[command.index("--") + 1:]
            self.assertEqual(argv[argv.index("--steps") + 1], "950")
            self.assertEqual(argv[argv.index("--resume") + 1], self.metadata["path"])
            self.assertEqual(argv[argv.index("--output") + 1], f"/run/final_eval/{kind}")
            self.assertNotIn("--actor-init", argv)
            self.assertEqual(f"--{kind}-only" in argv, kind != "aime")

    def test_pending_idempotency_is_persisted_before_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "study.json"
            study = {**self.study, "arms": {"adaptive": copy.deepcopy(self.arm)}}
            arm = study["arms"]["adaptive"]
            pending = supervisor.plan_resume(path, study, "adaptive", arm, self.metadata)
            self.assertEqual(pending["idempotency_key"], supervisor.plan_resume(path, study, "adaptive", arm, self.metadata)["idempotency_key"])
            arm["pending_submission"] = pending
            supervisor.atomic_save(path, study)

            def fake_submit(command):
                saved = json.loads(path.read_text())
                self.assertEqual(saved["arms"]["adaptive"]["pending_submission"]["command"], command)
                return {"id": 2, "args": command[command.index("--") + 1:]}

            with patch.object(supervisor, "mlq", side_effect=fake_submit):
                supervisor.submit_pending(path, study, arm)
            saved = json.loads(path.read_text())["arms"]["adaptive"]
            self.assertNotIn("pending_submission", saved)
            self.assertEqual(saved["submissions"][-1]["resume_checkpoint_sha256"], "abc")

    def test_bpb_jsonl_is_single_complete_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.jsonl"
            supervisor.atomic_save(path, {"type": "bpb", "step": 950, "val_bpb": 1.23}, jsonl=True)
            self.assertEqual(len(path.read_text().splitlines()), 1)
            self.assertEqual(supervisor.metrics_summary(path.parent)["bpb"]["val_bpb"], 1.23)

    def test_scores_must_be_finite_and_in_range(self):
        for value in (None, True, float("nan"), float("inf"), -.1, 1.1):
            self.assertFalse(supervisor.valid_score(value))
        for value in (None, True, float("nan"), float("inf"), 0, -1):
            self.assertFalse(supervisor.valid_score(value, bpb=True))
        self.assertTrue(supervisor.valid_score(0))
        self.assertTrue(supervisor.valid_score(1))
        self.assertTrue(supervisor.valid_score(1.5, bpb=True))

    def test_runtime_metadata_resolves_missing_training_fields(self):
        from postraining.vapo.config import build_arg_parser, resolve_rollout_defaults, validate_args
        from postraining.reasoning_modes import training_rollout_budget

        module = SimpleNamespace(resolve_rollout_defaults=resolve_rollout_defaults, training_rollout_budget=training_rollout_budget)
        parser = build_arg_parser()
        args = parser.parse_args(["--checkpoint", "/base", "--output", "/run", "--reasoning-mode", "memory", "--prompt-tokens", "256", "--continuation-tokens", "2048"])
        validate_args(parser, args)
        manifest = {"base": {"architecture": "nanogpt_mini_gpt2vocab_kda"}, "checkpoint_context_limit": 1024, "trained_context_tokens": 1024, "context_tokens": 2304}
        supervisor.resolve_runtime_budget(module, args, manifest)
        self.assertEqual(args.resolved_train_max_new_tokens, 2048)
        self.assertEqual(args.resolved_train_max_stream_steps, 2048)
        with self.assertRaisesRegex(ValueError, "runtime context"):
            supervisor.resolve_runtime_budget(module, args, {**manifest, "context_tokens": 4096})


if __name__ == "__main__":
    unittest.main()
