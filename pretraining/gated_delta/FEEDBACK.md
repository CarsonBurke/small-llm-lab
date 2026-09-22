# Final-to-first GDN2 feedback

Completed experiment: `nanomini_fb_gdn2_first_attached_1k`, mlq job **8681**.
It achieved a small measured gain, but **was not retained**: improvement
over first-layer LAM was 0.0037 BPB, below the required **>0.005** at 1,000
updates. Active model/kernel/trainer integration was removed after verifying
byte-identical archived source copies. The checkpoint, tests, source,
metrics and TensorBoard events remain available.

| Model | BPB at 1,000 updates | Training ms/update |
|---|---:|---:|
| Plain transformer | 1.3433 | 529.495 |
| Standalone GDN2 | 1.36051193 | 1096.596 |
| LAM, all layers | 1.3243 | 1343.868 |
| LAM, first layer | 1.3265 | 1096.485 |
| GDN2 feedback, first layer | **1.3228** | 1202.383 |

Training timers include compilation and exclude validation. The candidate
cost 9.66% more training time than first-layer LAM and 10.53% less than
all-layer LAM. Total process wall time was approximately 1,345 seconds.
These results use one seed and support only this matched-budget comparison.

Final pass-one/two/three/eight BPBs were 1.3418 / 1.3228 / 1.3219 / 1.3218.
On the separate 262,144-token sequential panel, true recurrence scored
1.3018 versus first-layer LAM's 1.3058 and all-layer LAM's 1.3028. The
sequential result does not replace the full-panel promotion metric.

Evidence:

- [Canonical result](../../ablation_results/nanomini_fb_gdn2_first_attached_1k/result.json)
- [Comparison data](../../ablation_results/nanomini_fb_gdn2_first_attached_1k/comparison.json)
- [Learning curves](../../ablation_results/nanomini_fb_gdn2_first_attached_1k/comparison.png)
- [Checkpoint](../../ablation_results/nanomini_fb_gdn2_first_attached_1k/model.pt)
- [Archived model](../../ablation_results/nanomini_fb_gdn2_first_attached_1k/sources/pretraining/nanogpt_mini/feedback_gdn2.py)

## Tested architecture and protocol

Hypothesis: content-dependent GDN2 decay, erase and write controls improve
LAM's carry when a full-depth transformer supplies the writes and the first
layer reads that memory. This experiment preserves the six-layer D512
transformer and its ordinary causal attention.

The archived `pretraining/nanogpt_mini/feedback_gdn2.py` implements a single four-head
K128/V128 memory. Queries and readout gates come from layer 0's normalized
input; keys, values, decay, erase and write gates come from the previous
Jacobi pass's final normalized states. The GDN2 frontend retains width-four
causal convolutions, additive-epsilon L2 normalization, gated RMS readout,
and the saved standalone GDN2's Xavier gain `2**-2.5`. Only the residual
memory output projection starts at zero, matching LAM's insertion contract.

All five activated writer fields are shifted right together, with a no-op
first update. The inclusive FLA scan then reads only earlier top states.
The readout has no LAM positive-feature denominator. This is a transplant
of the GDN2 frontend and update, not an ablation of only one gate.

Training uses two Jacobi passes, both scored, with `FB_DETACH=0` and uniform
carry noise `0.02`. Neither the first-pass source nor the recurrent state is
detached. Temporal gradients span all 1,024 tokens. K-first FLA execution
saves backward intermediates behind a fullgraph custom-op adapter. The
ordinary mini optimizer groups are retained; convolutions use AdamW at
0.002, and decay rate/bias use AdamW at 0.002 without weight decay.

The matched reference is `nanomini_fb_lam_first_1k`, 1.3265 BPB. Promotion
requires more than 0.005 improvement at 1,000 updates, i.e. below 1.3215 BPB.
All-layer `nanomini_fb_lam_compiled_1k` (1.3243) and plain mini (1.3433) are
secondary comparisons. This is a matched-token/update comparison, not a
matched-compute comparison. No 16MB artifact or 8xH100 claim is implied.

The archived `scripts/train_feedback_gdn2.py --name NAME --steps 1000` copies reference
overrides, sets GDN2 with layer-0 reads, and runs the existing feedback
trainer in the queue's process group. It refuses to overwrite a previous
run. Metrics and checkpoints go to `ablation_results/NAME/`; the canonical
metrics stream feeds `tb_logs/NAME/`. Source snapshots include installed
FLA Python sources. Validation runs every 20 updates on the same 1M-token
panel, with pass-1/2/3 diagnostics and pass-8/sequential evaluation at the
start and end. The sequential panel is 262,144 tokens and must not be
compared directly against full-panel BPB.

Archived GPU contracts are in `pretraining/tests/test_feedback_gdn2_gpu.py`:
literal recurrence and all six input gradients across chunk boundaries;
attached final-state writer gradients; strict causality and row isolation;
convolution cache parity; converged Jacobi versus true sequential feedback;
and fullgraph B64/T1024 forward/backward. These are numerical/execution
tests, not reduced training runs or quality evidence.

All four final GPU contracts passed in job **8679**. The production compiled
test reserved 16,934 MiB. An independent reviewer checked the architecture,
kernel adapter, optimizer partition, runner and result interpretation.

Training ran through `mlq` at priority 1, parallel limit 1, with a 90-minute
limit and exactly 1,000 updates. Nonfinite metrics were fatal; there was no
rank-based early cull. The run completed successfully with no culling.

## Restoring the archived experiment

The snapshot root is
`ablation_results/nanomini_fb_gdn2_first_attached_1k/sources/`. It contains
the exact model, kernel adapter, tests, runner and integrated feedback
trainer, plus dependency sources and the reporting script. `config.json`
records SHA-256 hashes for training sources and installed FLA Python files.

To reproduce, use an isolated checkout and restore the new model, kernel,
test and runner files to their original relative paths, along with the
snapshotted `nanogpt_mini_feedback_train.py`. Keep the matching backbone
and dependency versions; compare them against the recorded hashes. Merely
running a nested snapshot script will not restore its package imports.
Submit restored GPU contracts and any training/evaluation through `mlq`,
using a fresh run name. The checkpoint's architecture is
`nanogpt_mini_feedback_gdn2_v1`; instantiate the restored `FeedbackGDN2GPT`
with its `model_config` and strict-load `model`. This unquantized checkpoint
is 86,189,235 bytes, not a 16MB challenge submission.

## Three-pass follow-up with a 400-update decision

The user authorized reconsideration with feedback-informed writes and an
explicit early stop if the result is not clearly more promising at update
400. `nanomini_fb_gdn2_threepass_1k` (job 8687) **was pruned at exactly 400
updates**: 1.4566 BPB, worse than the two-pass reference's 1.4510 and above
the predeclared 1.4460 continuation threshold. Update 401 did not run.
The `_1k` name denotes the planned schedule, not a completed 1,000-update
result. This run provides no final-budget quality claim.

| At update 400 | Two-pass training | Three-pass training |
|---|---:|---:|
| Headline validation BPB | 1.4510 | 1.4566 |
| Two-pass evaluation BPB | 1.4510 | 1.4578 |
| Three-pass evaluation BPB | 1.4504 | 1.4566 |
| Training seconds, excluding validation | 489.906 | 726.669 |

Three-pass training was 0.0056 BPB worse on the headline comparison and
took 48.33% more training time through update 400. Total process wall time
was 789.32 seconds. Evaluating both checkpoints with three passes also
favors the two-pass-trained model by 0.0062 BPB. Early gains fluctuated
and did not persist at the decision point.

The [result](../../ablation_results/nanomini_fb_gdn2_threepass_1k/result.json)
records `pruned=true`, `completed_steps=400`, `error=null`, and exit code
75. mlq labels that nonzero exit as failed; it is the deliberate user gate,
not a training failure. The
[decision](../../ablation_results/nanomini_fb_gdn2_threepass_1k/gate_decision.json),
checkpoint, all 21 validation records, TensorBoard events, and verified
source snapshots are preserved under the run directory. Active experiment
files and trainer integration were removed after archiving. Restoration
uses the same procedure above, with this run's `sources/` directory; it
also includes `scripts/feedback_gdn2_protocol.py` and its policy tests.

The experiment restored the archived model and kernel unchanged and set
`FB_PASSES=3`. The existing generalized loss
is `loss1 + (loss2 + loss3) / 2`, preserving total loss weighting. Gradients
from pass three reach the feedback-informed states written by pass two,
as well as the earlier first-pass computation. This remains a finite-pass
approximation to the full final-to-first loop; ordinary GDN temporal-state
gradients were already present in the two-pass run.

The planned schedule remains exactly 1,000 updates, validation every 20,
seed 1337, and the same data/token/microbatch budget. At update 400, require
three-pass headline BPB **<=1.4460**, at least 0.005 better than the prior
two-pass run's **1.4510**. The decision uses the published four-decimal
metric precision. A failed gate saves a checkpoint and decision, then exits
**before update 401**, with exit code 75 and `pruned=true` in the result.
This fixed checkpoint rule is the user's requested exception to ordinary
plateau-based culling; it does not change the 1,000-update learning-rate
schedule. Metrics include pass-two BPB as a secondary evaluation control.

Queue settings: priority 1, maximum parallel runs 1, one attempt, 90-minute
limit. Seven pure policy tests passed, including threshold equality and
nonfinite rejection. All five GPU tests passed in job 8686, including the
new third-pass gradient path. That compiled three-pass production test
reserved 25,232 MiB. Independent review verified the stop occurs before
update 401, the unchanged 1,000-update schedule, and source identity.
