"""Test package.

Makes every test module import ONCE, as `postraining.tests.test_x`.
Without this, pytest imports these files as top-level `test_x` while
`test_replay_host_syncs.py` imports `postraining.tests.test_latent_rollout`
by path, so two distinct module objects for the same file coexist in
`sys.modules` -- the classic source of "passes alone, fails in the suite".
"""
