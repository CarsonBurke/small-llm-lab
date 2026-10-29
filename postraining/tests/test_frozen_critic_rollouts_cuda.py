"""Production memory-prefix intervention parity; execute only through mlq."""
import pytest
import torch

from postraining.frozen_critic_rollouts import _BranchArena, UninformativeBranch
from postraining.graph_decode import PinnedDecodeArena
from postraining.latent_rollout import trajectory_token_seeds
from postraining.tests.test_memory_carry_cuda import _wrapper, _make_live

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires queued CUDA')


class _TraceArena(PinnedDecodeArena):
    def _run(self, rows):
        self.trace.append((self.pending[0].clone(), self.policy_logits[0].clone(),
                           self.belief[0].clone(), self.memory_state[0].clone()))
        super()._run(rows)


class _StateBranchArena(_BranchArena):
    def _run(self, rows):
        if self.prefix_cursor == len(self.forced_prefix):
            self.branch_belief = self.belief[0].clone()
            self.branch_memory = self.memory_state[0].clone()
        super()._run(rows)


@torch.inference_mode()
def test_captured_memory_prefix_and_branch_distribution_match_source():
    torch.set_float32_matmul_precision('high')
    wrapper = _wrapper(torch.device('cuda'))
    _make_live(wrapper, 701)
    wrapper.requires_grad_(False)
    options = dict(rows=8, kv_width=64, temperature=1.0, stop_ids=(),
                   device=torch.device('cuda'), hidden_carry=True, compile=True,
                   cuda_graphs=True, replay_storage=False, record_policy_logits=True)
    prompt = torch.tensor([[1, 7, 9, 12]], device='cuda')
    seeds = trajectory_token_seeds(98, range(1), 6, torch.device('cuda'))
    kwargs = dict(prompt_repeats=6, max_new_tokens=20, max_stream_steps=20,
                  token_seeds=seeds, offload_device=torch.device('cpu'))
    source = _TraceArena(wrapper, **options)
    source.trace = []
    batch = source.rollout(prompt, torch.tensor([4]), **kwargs)
    forced = batch.token_ids[0, 4:13].tolist()
    expected_logits = source.trace[len(forced)][1]
    expected_belief = source.trace[len(forced)][2]
    expected_memory = source.trace[len(forced)][3]
    branch = _StateBranchArena(wrapper, **options)
    branch.configure([], None, 2, require_distinct=True,
                     forbidden_tokens=tuple(range(wrapper.backbone.tok_emb.num_embeddings)))
    with pytest.raises(UninformativeBranch, match='nonreasoning_action_pairs'):
        branch.rollout(prompt, torch.tensor([4]), **kwargs)
    assert branch.caches is None and not branch._graphs
    branch.configure(forced, None, 2)
    branched = branch.rollout(prompt, torch.tensor([4]), **kwargs)
    assert branched.token_ids[:, 4:13].tolist() == [forced] * 6
    torch.testing.assert_close(branch.branch_probabilities,
                               expected_logits.float().softmax(-1), rtol=2e-4, atol=2e-6)
    torch.testing.assert_close(branch.branch_belief, expected_belief, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(branch.branch_memory, expected_memory, rtol=2e-3, atol=2e-3)
