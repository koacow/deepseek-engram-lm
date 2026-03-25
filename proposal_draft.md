# Does N-gram Conditional Memory Help Small Dense Transformers? A Minimal Replication of Engram

**Project Type:** Empirical

**Team Members:** Kohki Hatori, Haoran, Khoa

**Course:** CS505: Natural Language Processing, Boston University

---

## 1. Task Description

### What is the problem?

Standard transformer language models predict the next token using attention over all prior hidden states, but they have no dedicated mechanism for recalling stereotyped surface patterns — multi-token named entities, formulaic phrases, and other local n-gram regularities. Classical n-gram models capture these patterns cheaply via lookup tables, but modern neural LMs discard them entirely, forcing the network to reconstruct static associations through layers of attention and feed-forward computation.

DeepSeek's Engram paper (Cheng et al., 2025) proposes **conditional memory** as a complementary axis of model capacity: a trainable, hash-addressed n-gram embedding module that is inserted into specific transformer layers and trained jointly from scratch with the backbone. At each position $t$, the module retrieves a memory vector $\mathbf{e}_t$ by hashing the local token context, then gates its contribution to the hidden state using the current representation as a query:

$$\mathbf{e}_t = \bigoplus_{n=2}^{N} \bigoplus_{k=1}^{K} E_{n,k}\!\left[\varphi_{n,k}(x_{t-n+1}, \ldots, x_t)\right]$$

$$\alpha_t = \sigma\!\left(\frac{\mathrm{RMSNorm}(\mathbf{h}_t)^\top \mathrm{RMSNorm}(W_K \mathbf{e}_t)}{\sqrt{d}}\right), \quad \Delta\mathbf{h}_t = \alpha_t \cdot W_V \mathbf{e}_t$$

$$\mathbf{h}_t \leftarrow \mathbf{h}_t + \Delta\mathbf{h}_t + \mathrm{ShortConv}(\Delta\mathbf{h}_t)$$

where $\varphi_{n,k}$ is a deterministic multiplicative-XOR hash, $E_{n,k}$ is a learned embedding table of prime size, and $W_K, W_V$ are learned projections. The output is fused residually before the attention and FFN sublayers.

The paper validates Engram at scale (27–40B parameters, 262B training tokens) in a Mixture-of-Experts setting with a custom multi-branch residual architecture (mHC, $M=4$). **Our project asks a more fundamental question: does the core Engram mechanism — n-gram hash lookup with context-aware gating — provide a measurable benefit even in a small, standard dense transformer trained from scratch?**

### Why is it interesting?

The paper's gains are impressive (e.g., BBH +5.0, MMLU +3.4 over an iso-FLOPs MoE baseline), but they are entangled with the multi-branch (mHC) architecture, the MoE backbone, and massive scale. It is unknown whether the n-gram memory mechanism itself is the driver, or whether its benefit is contingent on these surrounding design choices. Understanding this matters for practitioners who want to apply Engram to standard architectures.

### Why is it hard?

Three challenges make this non-trivial:

1. **No training pipeline exists.** The repository only provides a forward-pass demo with mocked attention layers. We must implement a full training loop from scratch.
2. **Architecture mismatch.** The paper's Engram uses a multi-branch residual stream (`hc_mult=4`). A standard transformer (single residual stream) requires simplifying this to `hc_mult=1`, which the paper's ablations show is a meaningful design choice — so our results will directly measure the cost of this simplification.
3. **Tokenizer adaptation.** The paper's tokenizer compression (NFKC normalization, 23% vocab reduction) is built for the DeepSeek 128k tokenizer. We must adapt it for GPT-2's 50,257-token BPE vocabulary.

---

## 2. Project Outline

### Model Architecture

We will train a **small dense transformer from scratch** with the following configuration (approximately GPT-2 small scale):

- 12 layers, hidden size 768, 12 attention heads, ~85M non-embedding parameters
- Standard pre-LayerNorm transformer with causal self-attention and MLP sublayers
- Two variants: **Baseline** (no Engram) and **+Engram** (Engram modules inserted at layers 2 and 6, following the paper's finding that early injection is optimal)

For the Engram variant, we simplify `hc_mult=1` (single residual stream, no multi-branch) and use bigrams and trigrams ($N=3$), 8 hash heads per n-gram order, and embedding dimension 256 per head. The n-gram embedding table size will be scaled to add roughly 20–30% additional parameters over the baseline, keeping the comparison approximately iso-FLOPs.

### Dataset

**WikiText-103** (Merity et al., 2016) — 103M tokens of Wikipedia text, standard split. We will train for up to 100K steps with batch size 32 and sequence length 512 (~1.6B token-steps), which is feasible on 1–2 A100s in under 12 hours.

### Software

- PyTorch (custom training loop)
- HuggingFace `datasets` and `tokenizers`
- The `engram_demo_v1.py` reference implementation (for the Engram module)
- SLURM on BU's SCC cluster (A100 GPUs)

### Methods

We train four models and compare them:

| Model | Description |
|---|---|
| **Baseline** | Small dense transformer, trained from scratch |
| **+Params** | Baseline + equivalent extra parameters as a dense MLP (parameter-matched to +Engram, iso-params control) |
| **+Engram (frozen gates)** | Engram tables trained, but gating fixed at $\alpha_t = 0.5$ (ablates the context-aware gating) |
| **+Engram (full)** | Full Engram with learned context-aware gating |

The `+Params` control is critical: it isolates whether any gains come from n-gram memory specifically, or simply from having more parameters.

---

## 3. Experimental Design

### Primary Metric

- **Perplexity (PPL)** on WikiText-103 test set — the standard LM benchmark metric, directly comparable to the paper's validation loss.

### Diagnostic Metrics

To understand *where* Engram helps (motivated by the paper's sensitivity analysis, Figure 6):

- **Named entity next-token accuracy** — accuracy at positions where the ground-truth token completes a named entity (extracted via spaCy). Engram's gating visualizations show strong activation on multi-token entities ("Alexander the Great", "Princess of Wales").
- **Formulaic phrase accuracy** — accuracy on high-frequency fixed phrases (e.g., bigrams in the top 0.1% of corpus frequency). This directly tests the n-gram memory hypothesis.
- **Gating visualization** — for the +Engram model, visualize $\alpha_t$ across token positions (replicating Figure 7 of the paper) to verify the module activates on the expected token types.

### Baselines

1. **Baseline (no Engram)** — establishes the floor; main comparison point.
2. **+Params (dense MLP)** — iso-parameter control; isolates parameter count from n-gram structure.
3. **+Engram (frozen gates)** — ablates the learned gating; tests whether static retrieval alone helps.

### Ablations (time permitting)

- Engram insertion depth: layer 2 only vs. layers 2+6 vs. layer 6 only
- N-gram order: bigram-only vs. bigram+trigram
- Embedding table size: small (5M slots) vs. medium (20M slots) vs. large (80M slots)

---

## 4. Feasibility

- **Training time:** A 12-layer, 768-dim transformer on WikiText-103 for 100K steps (batch 32, seq 512) takes approximately 6–10 hours on a single A100. We can run 4 models in parallel or sequentially within the project window.
- **Engineering scope:** The Engram module forward pass is implemented in `engram_demo_v1.py`. The main new work is (a) a standard training loop (~200 lines of PyTorch), (b) adapting the tokenizer compression to GPT-2's vocab, and (c) simplifying `hc_mult` to 1. This is achievable for a team with fine-tuning experience.
- **Honest limitation:** By simplifying mHC to a single branch, our results will reflect a weakened version of Engram. We will document this explicitly and frame our contribution as an **ablation of the core n-gram memory mechanism in isolation from the multi-branch architecture**, which is itself a novel empirical finding not present in the original paper.

---

## References

**Important: verify all details manually before submission. Do not submit without checking author names, years, and venues against the actual papers.**

- Cheng, X., Zeng, W., Dai, D., et al. (2025). *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models.* DeepSeek AI. GitHub: https://github.com/deepseek-ai/Engram
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners.* OpenAI Blog.
- Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2017). *Pointer Sentinel Mixture Models.* ICLR 2017. [WikiText-103]
- Xie, Z., et al. (2025). *Manifold-Constrained Hyper-Connections.* [mHC architecture — verify full citation]
