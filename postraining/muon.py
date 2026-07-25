"""Single-process Muon for RL post-training of nano backbones.

Orthogonalization is modded-nanogpt's Polar Express (arXiv 2505.16932): five
iterations with per-iteration coefficients, the matching 2e-2 safety factor on
the spectral normalization, and Nesterov momentum fused into the same compiled
function so the fp32 blend never reaches HBM.  This deliberately departs from
the geometry the trunk was pretrained under -- ``nanogpt_mini_gpt2vocab_train``
runs twelve naive NewtonSchulz5 iterations with ``2, -1.5, 0.5`` and scales
rectangular updates by ``max(1, rows/cols)**0.5``.  The trade is cost, not
accuracy: five Polar Express iterations spend 15 matmuls where twelve naive
ones spend 36, and in exchange they leave a designed ripple band around the
polar factor instead of converging onto it.  On a low-rank trunk gradient that
halves the size of the resulting step.  The reference also drops the
rectangular scaling, which costs the (2048, 512) matrices a further factor of
two.  Only the learning rate, weight decay and momentum are ours; those come
from the RL config, and both effects argue for revisiting the first.

The trainer is single-process, so this steps every parameter locally and skips
parameters whose grad is None (a parameter outside the current loss graph must
not decay or consume a momentum update).  A six-layer trunk holds 36 matrices
in three distinct shapes, so ``step`` stacks each shape group into one 3-D
tensor and runs a single batched Polar Express over it: three chains of fifteen
matmuls in place of 36 chains of 36.  In the live config all 36 are always
present, so the grad-free branch below is a contract, not a hot path.

The iteration amplifies input perturbations rather than damping them -- its
derivative at sigma=0 is the leading coefficient, so near-null directions grow
across iterations.  Measured on one incoherent bf16 ULP of input noise: 2.9x
out on a square iid gradient, 96x on a rank-64 one.  Any future change to how
this file batches or orders its arithmetic therefore moves the whole run, and
cannot be validated by eyeballing a single step.
"""

from __future__ import annotations

import torch
import torch._dynamo
from torch import Tensor

# Computed for num_iters=5, safety_factor=2e-2, cushion=2.  The coefficients and
# the safety factor in the normalization below are a matched pair.
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def _polar_express(
    grad: Tensor,
    momentum_buffer: Tensor,
    momentum_t: Tensor,
    split_baddbmm: bool = False,
) -> Tensor:
    """Fused Nesterov momentum + Polar Express, over a matrix or a stack.

    Momentum runs in fp32 and the result is cast to bf16 inside the same
    compiled region, so the fp32 blend is never materialized across a graph
    break.  ``momentum_t`` is a 0-D CPU tensor so changing the momentum value
    cannot trigger a recompile.  ``grad`` is scratch: it is consumed in place.
    """
    momentum = momentum_t.to(grad.dtype)
    momentum_buffer.lerp_(grad, 1 - momentum)
    g = grad.lerp_(momentum_buffer, momentum)

    X = g.bfloat16()
    is_tall = g.size(-2) > g.size(-1)

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)

    X = X.contiguous()

    # Multiply on whichever side keeps the Gram matrix at (min-dim, min-dim).
    gram = X.size(-1) if is_tall else X.size(-2)
    A = torch.empty((*X.shape[:-2], gram, gram), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)
    batched = X.ndim > 2
    matmul = torch.bmm if batched else torch.mm
    fused_matmul = torch.baddbmm if batched else torch.addmm

    for a, b, c in polar_express_coeffs:
        if is_tall:
            matmul(X.mT, X, out=A)  # A = X.mT @ X
        else:
            matmul(X, X.mT, out=A)  # A = X @ X.mT
        # A is symmetric, so A @ A.mT and A @ A agree: B = b * A + c * (A @ A).
        fused_matmul(A, A, A, beta=b, alpha=c, out=B)

        if split_baddbmm:
            # Referencing X twice makes torch take a defensive copy inside the
            # fused form; for the large matrices two kernels come out ahead.
            if is_tall:
                matmul(X, B, out=C)
            else:
                matmul(B, X, out=C)
            C.add_(X, alpha=a)
        elif is_tall:
            fused_matmul(X, X, B, beta=a, out=C)  # C = a * X + X @ B
        else:
            fused_matmul(X, B, X, beta=a, out=C)  # C = a * X + B @ X

        X, C = C, X  # Swap references to avoid unnecessary copies

    return X


# Pretraining compiles the update; CPU (tests) stays eager — the math is
# identical, only kernel fusion differs.
if torch.cuda.is_available():
    # dynamic=False is required upstream or it is much slower, but it
    # specializes on every (batch, rows, cols), and fullgraph=True turns an
    # exhausted cache into a hard error rather than a silent fallback to eager.
    # Three trunk shapes, each stepped as a full group and as a single row,
    # already reach the default limit of eight.
    torch._dynamo.config.recompile_limit = max(
        torch._dynamo.config.recompile_limit, 32
    )
    polar_express = torch.compile(_polar_express, dynamic=False, fullgraph=True)
else:
    polar_express = _polar_express


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, weight_decay: float = 0.0, mu: float = 0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        # Validate AFTER super().__init__ rather than over the argument, so
        # param-group dicts work.  ``step`` has always looped over
        # ``param_groups``, but the old loop read ``p.ndim`` straight off each
        # entry, so passing the group dicts that loop implies died on
        # AttributeError instead.  Empty input still raises ValueError -- that
        # is Optimizer's own "got an empty parameter list".
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim < 2:
                    raise ValueError(
                        "Muon only orthogonalizes matrices (ndim >= 2)"
                    )
        self._reset_buffers()

    def _reset_buffers(self) -> None:
        self._momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # Keyed by shape group and outside ``self.state``, so a checkpoint keeps
        # carrying momentum alone.
        self._momenta: dict[tuple, Tensor] = {}

    def __setstate__(self, state):
        super().__setstate__(state)
        # ``Optimizer.__getstate__`` carries only defaults, state and param
        # groups, so a deepcopied or unpickled optimizer arrives with none of
        # our attributes.  The momentum buffer rebuilds itself on the next step
        # from ``self.state``, which pickling does preserve.
        self._reset_buffers()

    @staticmethod
    def _shape_groups(group: dict) -> dict[tuple, list[Tensor]]:
        """Every parameter of the group, bucketed by shape.

        Bucketing ignores which parameters carry a grad this step, so a
        parameter keeps the same momentum row for the life of the optimizer
        even when it drops out of the loss graph for a while.
        """
        groups: dict[tuple, list[Tensor]] = {}
        for p in group["params"]:
            groups.setdefault((p.shape, p.device), []).append(p)
        return groups

    @staticmethod
    def _buffer_key(group_index: int, key: tuple) -> tuple:
        """Momentum key. The group index is what keeps two groups apart.

        ``self._momenta`` outlives any one group, so keying it on shape alone
        makes a second param group holding the same shape collide with the
        first: it would be handed a buffer sized for the OTHER group's member
        list. Fewer members than the first group is the dangerous case --
        no error, just two sets of parameters silently sharing momentum rows.
        Today every Muon here is built from a flat parameter list, which is
        exactly one group, so this has never fired.
        """
        return (group_index, *key)

    def _momentum(
        self, key: tuple, members: list[Tensor], present: list[int]
    ) -> Tensor:
        """The shape group's momentum as one fp32 buffer, one row per member.

        Holding the group contiguously is what makes batching free: the fused
        update writes momentum in place, so there is no staging copy at all.
        Only rows that are stepping are published into ``self.state``, so a
        parameter that never carries a grad never gains momentum state.
        ``load_state_dict`` hands back independent tensors, so a row that is no
        longer this buffer's is folded back in.
        """
        buffer = self._momenta.get(key)
        if buffer is None:
            buffer = torch.zeros(
                (len(members), *members[0].shape),
                dtype=torch.float32,
                device=members[0].device,
            )
            self._momenta[key] = buffer
        for index in present:
            state = self.state[members[index]]
            row = buffer[index]
            stored = state.get("momentum")
            if stored is None:
                state["momentum"] = row
            elif stored.data_ptr() != row.data_ptr():
                state["momentum"] = row.copy_(stored)
        return buffer

    @torch.no_grad()
    def step(self):
        for group_index, group in enumerate(self.param_groups):
            self._momentum_t.fill_(group["mu"])
            params: list[Tensor] = []
            updates: list[Tensor] = []
            for key, members in self._shape_groups(group).items():
                present = [i for i, p in enumerate(members) if p.grad is not None]
                if not present:
                    continue
                momentum = self._momentum(
                    self._buffer_key(group_index, key), members, present
                )
                # Every call leaves ``split_baddbmm`` off.  The reference splits
                # the fused form for matrices past 1024 rows to dodge a
                # defensive copy, but that spends a fourth launch per iteration
                # on a trunk that is launch-bound rather than FLOP-bound.  (An
                # earlier note also cited batch invariance here; there is no
                # such guarantee to protect -- see the fallback branch below.)
                if len(present) == len(members):
                    # The stack is scratch — Polar Express consumes it in place —
                    # so ``p.grad`` is left alone.  A fresh allocation each step
                    # beats a persistent buffer here: membership never changes
                    # in practice, the caching allocator hands the same block
                    # back, and the optimizer keeps no state to lose on a copy.
                    grads = torch.stack([p.grad for p in members]).float()
                    batch = polar_express(grads, momentum, self._momentum_t)
                    # Widening the bf16 result is exact and lets the foreach
                    # update below take its same-dtype fast route.  The
                    # contiguous call is a no-op while the iteration returns its
                    # input layout, and guards the fast route if that changes:
                    # ``float()`` on a transposed view keeps the transposed
                    # strides, which drops the whole list off the fast path.
                    updates.extend(batch.float().contiguous().unbind(0))
                    params.extend(members)
                else:
                    # Defensive: the live trunk always presents every member.
                    # A group missing members is not one contiguous momentum
                    # slice, so it steps a row at a time, still through bmm
                    # rather than mm.
                    #
                    # This used to claim bmm on one matrix agrees with bmm on
                    # the whole group BIT FOR BIT, and therefore that a
                    # parameter dropping out of the loss graph cannot perturb
                    # the others' arithmetic.  That is false and was measured
                    # false: cuBLAS picks its strided-batched GEMM by batch
                    # count, so batch 1 and batch 3 accumulate differently, and
                    # five cubic iterations amplify it to a few percent on the
                    # small elements.  Membership DOES perturb the result.
                    # Nothing here needs bit-exactness, so the row-at-a-time
                    # path stays -- but on its real merits: it keeps the
                    # compile cache at two entries per shape instead of one per
                    # membership count, and it avoids gathering a sub-stack.
                    for index in present:
                        grad = members[index].grad.to(torch.float32, copy=True)
                        update = polar_express(
                            grad[None],
                            momentum[index : index + 1],
                            self._momentum_t,
                        )
                        updates.append(update[0].float().contiguous())
                        params.append(members[index])
            if not params:
                continue

            if group["weight_decay"]:
                # Cautious decay: shrink a weight only where the update already
                # moves it toward zero.  The RL config runs weight_decay=0, so
                # this stays out of the common path.
                decay = group["weight_decay"] * group["lr"]
                for p, update in zip(params, updates, strict=True):
                    p.sub_(p * (update * p >= 0), alpha=decay)
            torch._foreach_add_(params, updates, alpha=-group["lr"])
