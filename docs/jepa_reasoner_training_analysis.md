# JEPA-Reasoner: training, inference, and what the paper does not specify

Paper: [local PDF](../papers/lejepa_2512.19171v3.pdf) · [arXiv abstract](https://arxiv.org/abs/2512.19171) · [arXiv v3 PDF](https://arxiv.org/pdf/2512.19171v3)

Analyzed version: arXiv v3, 28 January 2026, 13 pages. The supplied PDF and the copy in `papers/` have the same SHA-256 hash.

## Short answer

JEPA-Reasoner is not a token-free language model. It does ordinary pretraining and reports supervised math-QA fine-tuning, but then branches into Reasoner SST plus separately trained Talker reconstruction. It has no RL stage.

It first trains an ordinary decoder-only Transformer with next-token cross-entropy. It then removes the LM head and reuses the Transformer as a deterministic latent-state predictor. In this self-supervised training (SST) phase, its output is trained to match the EMA embedding of the next sequence segment using scaled cosine distance. At inference it recursively produces a complete sequence of continuous latent states before a separate Talker emits any text. The Talker is trained separately, with the Reasoner frozen, to reconstruct the corresponding tokens using ordinary cross-entropy.

There is **no RL stage in the paper**. There is no reward model, verifier, preference dataset, policy-gradient loss, PPO, GRPO, or other reinforcement-learning algorithm. Consequently, the paper neither retains nor replaces the JEPA loss “during RL”: that stage simply does not exist.

The latent “thoughts” are also **not sampled**. The paper's formalization makes the Reasoner update deterministic. Only the Talker is described probabilistically when it samples output tokens. “Mixed latent vectors” mean continuous vectors that appear to combine multiple vocabulary directions; they are not a sampled tree of thoughts.

The cleanest mental model is:

```text
prompt tokens
    ↓ token embeddings
initial latent matrix
    ↓ deterministic Transformer + normalization, recursively
complete latent trajectory R
    ↓ frozen plan supplied to a separately trained Talker
output tokens
```

## Terminology: JEPA, not LeJEPA

The paper calls the method **JEPA-Reasoner** and cites I-JEPA. It does not use the term **LeJEPA** or a LeJEPA-specific objective. Its post-pretraining objective is a JEPA-style predictor/EMA-target loss implemented as scaled cosine distance. If “LeJEPA objective” was meant literally, that is a different method and should not be conflated with this paper.

## Architecture

### Reasoner

The Reasoner contains:

- A token embedding layer, which turns the initial textual input into continuous vectors.
- Modified Transformer blocks acting as a latent predictor.
- QK normalization for attention stability.
- A hybrid RMSNorm plus L2 normalization at the output during SST and inference. The L2 step puts each generated latent on the unit hypersphere.

The predictor does not apply an LM head after SST. It outputs a latent matrix for the next sequence segment, normalizes it, and feeds that matrix back through the Transformer to produce the next latent state or segment. The complete latent trajectory is produced before the Talker starts producing tokens.

The paper alternates between describing a “latent vector,” a “latent matrix,” a “next sequence segment,” and the scalar recurrence `r[t+1] = f(r[t])`. It does not provide enough shape notation or pseudocode to resolve exactly how a segment is laid out, how many token positions it spans, or how matrices from successive recurrences are concatenated.

### Talker

The Reasoner and Talker are different models.

- **Mono-Talker** has Transformer decoder blocks and an LM head, but no token embedding layer or separate encoder. It maps the Reasoner's latent sequence to the complete output token sequence in one forward pass. It is intended for reconstruction that does not need additional context. The natural-language model in Section 6 uses a 198M Mono-Talker with a 694M Reasoner.
- **Dual-Talker** has a token embedding layer, encoder blocks, decoder blocks, and an LM head. Its encoder consumes the Reasoner latents; its decoder emits tokens autoregressively while conditioned on previous tokens and continuously guided by the encoded latent trajectory. It is intended for context-aware reconstruction.

The authors emphasize that the Talker is a readout or reconstruction model, not an independent reasoner. During Talker training, the Reasoner is frozen.

There is an unresolved tension in the presentation: Section 3.2 says Dual-Talker is usually necessary for natural-language tasks, while the headline GSM8K system in Section 6 is explicitly a Reasoner plus Mono-Talker. This makes the omitted Mono-Talker mask and position-alignment details particularly important.

## Exact training objectives by stage

The following separates what the paper specifies from the details one has to infer. `mean` below could instead be a sum; the paper does not specify reduction conventions.

### Stage 1: ordinary language-model pretraining

The starting model is a standard decoder-only Transformer trained by teacher forcing on next-token prediction:

```text
L_pretrain = mean over positions t of -log p_theta(x[t] | x[<t])
```

Important details:

- The token embedding matrix is tied to a temporary LM head.
- L2 output normalization is disabled during this stage.
- The temporary LM head is discarded after pretraining.
- In the tree-search experiment, the pretraining loss is applied at every position.
- For the natural-language experiment, the paper reports 300,000 pretraining steps on C4 and WikiText.

This stage is essentially normal LM pretraining. The architectural differences are mainly preparation for the later switch to latent prediction.

The paper says tied embeddings encourage angular alignment between predictions and token embeddings. That is a heuristic motivation. Weight tying alone does not guarantee that the embedding rows are orthogonal or that the embedding Gram matrix is the identity.

### Stage 2: self-supervised latent training (SST)

The temporary LM head is removed and L2 normalization is restored. Let:

- `h_pred` be the Reasoner's normalized prediction for the next sequence segment.
- `h_target` be the corresponding target produced by the EMA target embedding layer.
- `theta` be trainable online parameters.
- `theta_target` be EMA target-embedding parameters.

The paper gives this loss:

```text
L_SST = 4 * (1 - cosine_similarity(h_pred, h_target))
```

For a sequence or matrix, the natural reading is that this is evaluated at the selected target positions and then reduced:

```text
L_SST = mean over supervised positions i of
        4 * (1 - cosine_similarity(h_pred[i], stop_gradient(h_target[i])))
```

The `stop_gradient` is implicit in using an EMA target rather than something written in the paper's equation. The paper explicitly specifies the target-embedding update:

```text
E_target ← 0.98 * E_target + 0.02 * E_online
```

The online embedding and Transformer predictor are updated by backpropagation. The target embedding is updated only by EMA. The paper motivates momentum 0.98 as a way to reduce rank collapse while retaining enough angular movement.

For unit-normalized versions of both vectors, cosine distance is also proportional to squared Euclidean distance on the unit sphere. Section 7.2 uses this relation, although the implementation description does not clearly say whether raw EMA embedding rows pass through the same explicit L2 layer as predictor outputs:

```text
1 - cosine(r, r_target) = 0.5 * squared_norm(r - r_target)
```

For tree search, SST loss is masked to positions defining the desired route. For the natural-language and math runs, the exact position mask, prompt/answer boundary treatment, next-segment offset, and reduction are not stated.

Despite being called self-supervised, SST is not learning correctness from a reward. It is learning to reproduce the latent embeddings of the ground-truth continuation in the training text. On math QA data, the target answer or continuation text—in whatever undisclosed format the QA pairs use—supplies the target trajectory indirectly through its token embeddings.

### Stage 3: Talker reconstruction training

With the Reasoner frozen, the Talker receives a latent-vector sequence and is trained to reconstruct the corresponding tokens with standard cross-entropy. The paper does not say whether those training latents come from free-running recurrence, ground-truth-derived inputs, or some combination.

For the autoregressive Dual-Talker, the factorization is naturally:

```text
L_dual = mean over positions t of
         -log q_phi(x[t] | x[<t], complete_reasoner_trajectory)
```

For the one-pass Mono-Talker, a schematic objective is:

```text
L_mono = mean over positions t of
         -log q_phi(x[t] | reasoner_latent_sequence)
```

The second formula is only a schematic factorization. The paper says Mono-Talker reconstructs the entire sequence in one forward pass, but does not specify its attention mask, positional alignment, or whether an output position can attend to all future Reasoner latents. The only exact statement is that it uses standard cross-entropy reconstruction loss.

There is no stated joint loss such as `L_SST + lambda * L_Talker`: the Talker is described as independently trained while the Reasoner is frozen.

### “SFT” on math data

Section 6.1 says the authors:

1. Pretrain a standard 0.9B Transformer for 300,000 steps on C4 and WikiText.
2. Fine-tune for 42,000 steps on “general math QA pairs.”
3. Select a checkpoint, further fine-tune a Transformer baseline, and train a JEPA-Reasoner plus Talker using the Section 4 method with the same data and hyperparameters.

The ordinary Transformer's math fine-tuning is most naturally standard teacher-forced cross-entropy, but the paper does not explicitly write that loss in Section 6. It also does not give the math dataset identity, example formatting, or prompt/response masking. The 42,000 steps refer to the initial math-QA fine-tuning; step counts for the subsequent baseline fine-tuning, Reasoner SST, and Talker training are not separately stated.

It is therefore misleading to describe this as a fully specified modern SFT recipe. For the JEPA pair, the important “post-training” operation is SST of the Reasoner plus separate CE reconstruction training of the Talker—not ordinary end-to-end response-token SFT of one model.

### RL

There is none. The paper explicitly frames its results as obtained without sophisticated reinforcement learning. It defines no RL loss at any point.

This matters technically: the Reasoner as presented is a deterministic map, not a stochastic latent policy with log-probabilities. Standard PPO or GRPO cannot simply be applied to its latent trajectory without first defining a stochastic policy or some other differentiable/reinforcement-learning interface. The paper does not propose one.

## Training versus inference

The method has a consequential train/test distinction.

During SST training, the paper says the next-segment targets are generated from the ground-truth token sequence by the EMA embedding layer. It also claims training can be done in a single parallel forward pass rather than repeatedly unrolling generated latents.

During inference, predicted latents are normalized and recursively fed back as inputs:

```text
R[0] = embed(prompt tokens)
R[j+1] = normalize(Reasoner(R[j]))
```

After the complete trajectory has been formed:

```text
Mono output = MonoTalker(R[1], ..., R[J])
```

or:

```text
x[t] is sampled from DualTalker(x[<t], R[1], ..., R[J])
```

The paper does not explain how `J` is chosen, how latent generation halts, whether trajectory length is fixed or inferred from the task, or whether the model is trained against its own recursively generated states. This leaves a potentially important exposure mismatch between parallel teacher-forced latent training and free-running latent recurrence at inference.

## How are thoughts sampled?

They are not sampled in the presented model.

The paper's own theoretical model is:

```text
r[t] = f_theta(r[t-1])                       deterministic Reasoner update
x[t] sampled from g_phi(r[t], x[<t])         probabilistic Talker output
```

Thus:

- There is no temperature, top-k, top-p, beam search, Gaussian latent, or categorical choice for Reasoner states described in the paper.
- “Autoregressive latent generation” means recurrent dependence on prior continuous states, not sampling from a latent probability distribution.
- The reported mixed latent vector is a single vector located between token-embedding directions. The tree experiment suggests it retains contributions from two sibling choices while weighting the correct child more strongly. This is evidence of superposed information, not evidence that the model samples or explicitly executes multiple thought branches.
- Token decoding parameters such as temperature and top-p are also not reported.

## Why can it generate a sensible sentence?

### It still has token-level machinery

The model is not bypassing tokens altogether:

- Input text is tokenized and embedded.
- SST targets are derived from token embeddings.
- The Talker ends in an LM head over the vocabulary.
- Talker training uses token-level cross-entropy on real sequences.
- Dual-Talker remains autoregressive over output tokens.

What has been removed from the reasoning loop is the **sampled token**, not token supervision or token generation.

### The plan is completed before speaking

The Talker cannot perturb the Reasoner because all Reasoner latents already exist before the first output token is sampled. A poor word choice can damage the local surface continuation in Dual-Talker, but it cannot make the Reasoner recompute later latent states from that bad word.

This eliminates one feedback path:

```text
sampled output token ──X──> future Reasoner state
```

It does not prove that the latent plan itself is correct or internally consistent.

### The Talker learns linguistic realization

Dual-Talker gets two strong coherence signals:

1. Its normal autoregressive token history, which supports local syntax and within-word/subword continuation.
2. The complete latent trajectory through its encoder, which supplies global semantic guidance at every decoding step.

Mono-Talker does not rely on previously sampled output tokens. It receives an ordered latent sequence and learns, by CE at every output position, to map coherent latent trajectories to coherent token sequences. Transformer interactions among the latent positions can encode sequence structure before the LM head makes position-wise token predictions. However, the paper omits the mask and exact tensor layout, so a stronger mechanistic claim is not justified.

### What about changing direction mid-sentence or mid-word?

Your intuition identifies a real failure mode, but the architecture changes where it can happen.

- The Reasoner cannot change direction *in response to a sampled word*, because it finishes first and never sees Talker output.
- It can still produce a latent trajectory whose later states conflict with earlier states. Normalization bounds vector magnitude; it does not enforce semantic consistency.
- Dual-Talker's autoregressive token history resists abrupt local changes, while latent cross-conditioning can pull it toward a different semantic continuation.
- A subword split does not create a special latent freedom: the Talker still predicts vocabulary tokens, and Dual-Talker conditions on previously emitted subword tokens in the normal way.
- Mono-Talker can in principle emit inconsistent adjacent tokens if its latent sequence is inconsistent, because no sampled-token feedback repairs them.

Appendix C contains an experiment very close to the proposed failure. The authors give Dual-Talker an initial prefix about Francis Bacon but latents from a sentence about Jean-Paul Sartre. The output rapidly switches to Sartre-related content and is only partly grammatical at the transition. That shows both sides of the result: the latent plan strongly controls semantics and the Talker often preserves surface fluency, but conflicting plans and prefixes can indeed cause a mid-sentence semantic turn. The architecture blocks output-token errors from propagating back into the Reasoner; it does not guarantee globally coherent text or eliminate surface-generation errors.

## How this differs from a normal autoregressive LM

| Property | Normal autoregressive LM | JEPA-Reasoner |
|---|---|---|
| Main recurrent object | Sampled tokens | Continuous normalized latent states |
| Reasoning and wording | Same model and sequence | Separate Reasoner and Talker |
| Initial pretraining | Next-token CE | Next-token CE |
| Later Reasoner objective | Usually still token CE, then possibly reward optimization | Scaled cosine match to EMA target embeddings |
| Text generation | Token-by-token from the same model | Separate Mono- or Dual-Talker |
| Does a sampled token affect later reasoning? | Yes | No |
| Is the full plan available before output? | Usually no | Claimed yes |
| Latent trajectory stochastic? | Hidden states depend on sampled tokens | No; deterministic in the paper |
| Explicit RL | Common in modern post-training, but optional | None |
| Correctness signal in this paper | Correct continuation tokens and possibly RL in other systems | Ground-truth continuation embeddings during SST; no reward |

The most substantive difference is the probabilistic factorization. A normal LM intertwines the reasoning state with generated tokens. JEPA-Reasoner instead assumes:

```text
P(reasoning trajectory, output text)
    = P(reasoning trajectory) * P(output text | reasoning trajectory)
```

and prevents the second factor from feeding back into the first.

## What the paper does not specify

The paper is an architectural investigation, not a reproducible training recipe. It omits several details needed to implement the reported system faithfully:

- No author code or pseudocode is linked.
- Exact construction, length, stride, and shape of a “next sequence segment.”
- Exact causal or non-causal masks during SST and inside Mono-Talker.
- How the complete latent chain's length is selected and how inference stops.
- Whether SST ever trains on recursively predicted latents rather than only ground-truth-derived inputs.
- The natural-language SST loss mask and prompt/answer boundary handling.
- The identity and formatting of the “general math QA pairs.”
- Step counts for the later baseline fine-tuning, Reasoner SST, and Talker training after the reported 42,000-step initial math fine-tune.
- Talker initialization and detailed training schedule.
- Whether Talker-training latents are obtained from free-running Reasoner recurrence, ground-truth-derived inputs, or a mixture.
- Token decoding strategy and sampling hyperparameters.
- Optimizer, learning-rate schedule, batch size, and context length for the 0.9B natural-language run. The paper gives these only for the CFG experiment.
- Any RL, preference optimization, verifier, process reward, or outcome reward.

These omissions mean the high-level loss is clear, but the exact end-to-end algorithm and the headline GSM8K result cannot be reproduced from the paper alone.

## Bottom line

JEPA-Reasoner's proposal is not “an LM that never recurses over tokens.” It is a system that moves the **reasoning recurrence** from sampled token space into continuous latent space, completes that recurrence, and then uses a token-trained language interface to verbalize it.

The loss stack is:

```text
ordinary pretraining:     next-token cross-entropy
Reasoner SST:             4 * (1 - cosine(predicted latent, EMA target latent))
Talker training:          token reconstruction cross-entropy, Reasoner frozen
reinforcement learning:   absent
```

Its strongest architectural claim—output token errors cannot alter future Reasoner states—follows directly from the decoupling. Stronger claims about correct planning, coherent latent transitions, genuine parallel search, or RL-trained reasoning do not follow from the method or evidence presented.
