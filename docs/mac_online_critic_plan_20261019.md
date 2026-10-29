# MAC with a live critic and credit on every token (plan, 2026-10-19)

Status: stage 1 (job 14134) collapsed under all-action MAC; with only the estimator changed to sampled_baseline (job 14165) it held and improved to 0.787 by step 240 (NOTES 2026-10-19). Stage 2 queued as job 14169 (v2) on the reviewed mixture without code and science MC. Stage 3 on hold.

## Where this starts

The only Q-critic run that improved held-out reward is
`dg_full_mac_actor_20261015` (cot, arithmetic drill pool, base reward 0.59).
Its development panel peaked at 66.6% at actor step 60, then fell to 18.8% by
step 160. Two of its design choices are wrong and are fixed here; neither had
a justification that survives inspection:

1. **The critic was frozen.** It was fit for 640 updates on 32 base-policy
   pools, then never updated. By the end it scored the actor's states with
   the base policy's continuation values, and its optimism grew from +0.03 to
   +0.34. A frozen critic was a gate fixture (a 10-14 "exploitation test")
   that was carried into training.
2. **Only think-span states were credited.** `reasoning_state_mask` dropped
   every state from the closing think fence through EOS, so the answer
   policy received no actor gradient while think-span updates moved the
   shared weights that produce it. The stated reason (a gold-aware critic
   would turn answer credit into SFT on gold) is false: the gold answer is a
   function of the prompt, so the ideal Q is unchanged, and MAC moves a
   logit by `pi_a * A_a`, not SFT's `1 - pi_gold`. That run's critic had no
   gold context anyway.

That run is cot. It has no memory: `reasoning_mode` is `cot`, its config
rejects any other mode, and the only carry-shaped tensor in its checkpoints
(`combiner.carry.weight`) is the unused latent-mode combiner, all zeros.

## Objective (unchanged from 10-15 except the mask)

At every generated state s, through EOS:

```
loss(s) = - sum_a pi(a|s) * stopgrad[Q(s,a) - sum_b pi(b|s) Q(s,b)]
Q(s,a)  = c(s) + u(s) . x(a)        (token-Q bilinear head, 1024 hidden)
```

Equal weight per state. No V, no DG gate, no sampled-action term, no
entropy or KL terms.

## Stages

Each stage changes one thing against the previous one.

### Stage 1: the 10-15 run, fixed (cot, drill pool)

- Start: the same point 10-15 started from. The actor is the SFT
  checkpoint `kda8_sft_omi2_drills_code30k_science_v1_e1/sft_final_model.pt`
  (SHA c882966...). The critic is the 640-update study critic
  `dg_full_critic_convergence_20261015/actor_trunk/checkpoint.pt`.
- Changes vs 10-15: the critic refreshes on every pool, and MAC credit
  covers every generated state.
- Everything else as 10-15: drill pools `drill_rl_study_*_20261014_v1`,
  64 prompts x 16 samples per pool, one actor update per pool, 512 prompt
  tokens, temperature 1, top-p 1, development panel (512 prompts x 16)
  every 20 steps, 360 steps, learning rates as 10-15.
- Success: beats the 10-15 peak (66.6%) and holds through 360 steps,
  compared step for step against the 10-15 curve.

### Stage 2: the main corpus

The broad mixture held critic EV at or below 0.02 in every run on it, for
every critic design (10-11 to 10-14): reward near 0.02, part of it
chance-level science MC. Before a run:

- screen the mixture with the frozen policy and keep prompts in a learnable
  pass-rate band; drop or fix chance-guessable MC;
- fit the critic on that corpus first (the drill critic does not transfer;
  it needed 640 updates over 32 pools to reach EV 0.34).

### Stage 3: the thinking improvements

Memory mode, then write noise with a write-time critic, against the stage
1/2 cot result. Designed separately once stage 1 shows the critic tracks
the policy.

## Critic

- **Refresh every pool.** Q for the actor update is read before the critic
  step on the same pool, as the trainer already orders it. The critic then
  fits that pool's Monte Carlo returns at every generated position. Learning
  rates as 10-15: head 1e-4, direction 1e-4/512.
- **Timescale.** One pass per pool may be too slow, since the 10-15 critic
  took 20 passes over each pool. The decisive metric is EV on the fresh
  pool *before* fitting it. If it decays while the actor moves, add critic
  passes per pool before anything else.

## Implementation

1. Port MAC into main from `.worktrees/dg-full-actor-exec-20261015-v4`: the
   `mean_actor_critic.py` loss, the trainer branch and the config
   validation. Port only the objective; leave out the protocol machinery
   (`--mac-direction-*`, `--mac-ad-forensic-*`, `--mac-freeze-critic`, the
   mandatory direction-check resume chain). A trained critic is loaded by a
   plain initialization flag.
2. Replace `reasoning_state_mask` with the generated-state mask, removing
   the think-only path rather than keeping it as an option.
3. Merge the diagnostic full-vocabulary pass into the loss so the head reads
   the logit/Q table twice (forward, backward) instead of three times.
4. CPU tests: the mask covers fence/answer/EOS states and excludes prompt
   and pad; the centered loss has zero value and the analytic gradient;
   critic refresh ordering (actor Q from before the critic step); resume
   round-trip.
5. A CUDA test through `mlq`: compiled MAC replay matches eager on the
   production KDA trunk.

Output goes to the main checkout's `postraining/runs/<name>/`, even if the
code runs from a worktree, so TensorBoard events appear on the
`parameter-golf-postraining` tensorwatch board (:6106).

## What to watch

- Development reward curve against the 10-15 curve.
- Critic: pre-fit EV on each fresh pool, Q optimism (Q minus MC return).
- Credit split: mean |centered Q| and summed MAC weight on think vs answer
  states.
- Length by outcome (10-15 failures grew from 277 to 382 actions).
- Rollout and panel traces at every evaluation.
- Cull: development reward below 0.40 at two consecutive evaluations (the
  10-15 rule), or below the base 0.59 for three consecutive evaluations
  after step 60.

## Known risks

- The drill pool has 2,156 training prompts across 22 cells. 360 steps at 64
  prompts per step cycles it about ten times, so watch the train/heldout
  gap.
- Answer-span Q contrasts were never measured on their own; with no gold
  context, judging a digit means checking the arithmetic.
