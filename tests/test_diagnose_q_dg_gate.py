"""CPU math/control fixtures only; no model construction or execution."""

import math
from pathlib import Path
from types import SimpleNamespace
import unittest

from scripts.diagnose_q_dg_gate import (
    NativeWritesDisabled, ProbeFinished, driver_argv, geometry_from_moments,
    checked_write, cpu_state, input_equality_summary, instrument, scalar_coefficient,
)


class QGateProbeTests(unittest.TestCase):
    def test_coefficient_decomposition_with_rare_positive_and_negative_actions(self):
        for advantage in (-1, -.03, 0, .001, .97):
            for surprisal in (0, .5, 3, 7, 20):
                dg = scalar_coefficient(advantage, surprisal, "dg")
                base = scalar_coefficient(advantage, surprisal, "base")
                tilt = scalar_coefficient(advantage, surprisal, "tilt")
                self.assertAlmostEqual(dg, base + tilt, places=15)
                self.assertGreaterEqual(tilt, 0)

    def test_raw_gradient_decomposition_preserves_cancellation(self):
        advantages, surprisals = [.97, -.03, .001], [3, 7, 2]
        score_vectors = [[1, -1], [-1, 1], [2, -2]]
        gradients = {}
        for mode in ("dg", "base", "tilt"):
            gradients[mode] = [sum(scalar_coefficient(u, l, mode) * score[index]
                                   for u, l, score in zip(advantages, surprisals, score_vectors))
                               for index in range(2)]
        a, b, c = (gradients[mode] for mode in ("dg", "base", "tilt"))
        residual = [x - y - z for x, y, z in zip(a, b, c)]
        difference = [x - y for x, y in zip(a, b)]
        square = lambda values: sum(value * value for value in values)
        stats = geometry_from_moments(square(a), square(b), square(c), sum(x * y for x, y in zip(a, b)), square(residual), square(difference), sum(x * y for x, y in zip(c, difference)))
        self.assertLess(stats["algebra_residual_over_dg"], 1e-14)
        self.assertAlmostEqual(stats["direct_tilt_actual_difference_cosine"], 1)

    def test_geometry_exposes_finite_precision_residual_relative_to_tiny_tilt(self):
        stats = geometry_from_moments(1, 1, 1e-12, 1, 1e-12, 0, 0)
        self.assertEqual(stats["algebra_residual_over_direct_tilt"], 1)
        self.assertIsNone(stats["algebra_residual_over_actual_difference"])
        self.assertEqual(stats["actual_dg_minus_constant_half_norm"], 0)

    def test_driver_keeps_full_budget_and_disables_native_evaluations(self):
        initial = ["python", "-u", "-m", "postraining.train_latent_vapo", "--steps", "950", "--value-warmup-steps", "50", "--actor-init", "/old", "--fresh-critic", "--output", "/train", "--max-train-hours", ".4", "--bpb-every", "20", "--q-critic-lambda", "1"]
        argv = driver_argv(initial, Path("/archived/input.pt"), Path("/diagnostic/_driver"))
        self.assertEqual(argv[argv.index("--steps") + 1], "950")
        self.assertEqual(argv[argv.index("--value-warmup-steps") + 1], "50")
        self.assertEqual(argv[argv.index("--q-critic-lambda") + 1], "1")
        self.assertNotIn("--actor-init", argv)
        self.assertNotIn("--fresh-critic", argv)
        self.assertNotIn("--max-train-hours", argv)
        for argument in ("--bpb-every", "--bench-every", "--aime-every", "--post-update-kl-every", "--snapshot-every", "--rollout-sample-every"):
            self.assertEqual(argv[argv.index(argument) + 1], "0")

    def test_native_steps_checkpoints_and_events_are_suppressed(self):
        calls = []
        module = SimpleNamespace(step_optimizers=lambda: calls.append("step"), save_checkpoint=lambda: calls.append("checkpoint"), JsonlLogger=lambda: calls.append("logger"), SummaryWriter=lambda: calls.append("writer"))
        disabled = NativeWritesDisabled()
        with self.assertRaises(ProbeFinished):
            with instrument(module, {"step_optimizers": disabled.step, "save_checkpoint": disabled.checkpoint, "JsonlLogger": disabled.logger, "SummaryWriter": disabled.tensorboard}):
                module.step_optimizers()
                module.save_checkpoint()
                module.JsonlLogger().log(type="train")
                module.SummaryWriter().add_scalar("train/loss", 1)
                raise ProbeFinished()
        self.assertEqual(calls, [])
        self.assertEqual(disabled.suppressed_steps, 1)
        self.assertEqual(disabled.suppressed_checkpoints, 1)
        self.assertEqual(disabled.suppressed_events, 2)
        module.step_optimizers()
        self.assertEqual(calls, ["step"])

    def test_reloading_optimizer_state_does_not_mutate_saved_step(self):
        class Scalar:
            def __init__(self, value):
                self.value = value

            def detach(self):
                return self

            def cpu(self):
                return self

            def clone(self):
                return Scalar(self.value)

        torch = SimpleNamespace(is_tensor=lambda value: isinstance(value, Scalar))
        saved = {"state": {0: {"step": Scalar(65)}}}
        dg_state = cpu_state(saved, torch)
        dg_state["state"][0]["step"].value += 1
        base_state = cpu_state(saved, torch)
        self.assertEqual(saved["state"][0]["step"].value, 65)
        self.assertEqual(base_state["state"][0]["step"].value, 65)

    def test_invalid_numeric_evidence_fails_before_writing(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            geometry_from_moments(float("nan"), 1, 1, 1, 1, 1, 1)
        writes = []
        with self.assertRaises(ValueError):
            checked_write(lambda *args: writes.append(args), Path("/unused"), {"score": float("inf")})
        self.assertEqual(writes, [])

    def test_comparison_requires_exact_active_input_counts_order_and_values(self):
        self.assertTrue(input_equality_summary(100, 100, True, 0, 0)["exact_equal"])
        for arguments in ((100, 99, True, 0, 0), (100, 100, False, None, None), (100, 100, True, 1e-12, 0), (100, 100, True, 0, 1e-12)):
            self.assertFalse(input_equality_summary(*arguments)["exact_equal"])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            input_equality_summary(100, 100, True, float("nan"), 0)


if __name__ == "__main__":
    unittest.main()
