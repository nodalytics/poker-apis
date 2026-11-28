# glance — models: where we evolve to

**Status: direction, not implementation.** Nothing here is built yet. This is
where glance should evolve: from hand-mapped screens plus classic vision to
**our own foundation model** that understands and operates any desktop app,
hands-off (see "The ultimate test" in [README.md](README.md)). The C++ engine
stays the fast, deterministic runtime that grabs, acts and verifies; models
plug in behind the same `Screen`/`Input` + state-JSON contracts.

## Why our own model

- Off-the-shelf vision-language models are slow, remote and costly per step,
  and they are not trained on *operating* apps from raw pixels at the
  latency a live loop needs.
- glance already produces the training data: timestamped grabs, structured
  states, and verified action traces (each action checked against the next
  grab). Use cases 1, 2 and 5 in the README form that data pipeline.
- Nothing about the model is settled up front. Token arrangement, encoders,
  decoders and architectures are all to be tried and measured.

## Reference: Kronos (ideas to borrow)

[Kronos](https://github.com/shiyu-coder/Kronos) (arXiv 2508.02739, MIT
license) is a decoder-only foundation model for financial candlestick
(K-line) series. Its parts:

- **Tokenizer.** Each 6-dimensional bar (OHLCV + amount) is encoded by a
  small Transformer autoencoder and quantized with Binary Spherical
  Quantization to 20 bits. The bits split into a **coarse** 10-bit subtoken
  and a **fine** 10-bit subtoken, two 1024-entry vocabularies instead of one
  of about a million. The coarse subtoken has its own reconstruction loss;
  the fine subtoken encodes the residual.
- **Predictor.** A causal Transformer (RoPE, RMSNorm) predicts
  p(coarse | past) · p(fine | past, coarse). During training the fine head
  is conditioned on the model's own *sampled* coarse token, not ground
  truth, to reduce exposure bias. Sizes run from 4M to 500M parameters with
  a 512–2048 context.
- **Data.** Over 12 billion bars from 45 exchanges at 7 frequencies, with
  cleaning (spike and illiquidity filters) and upweighting of under-
  represented markets.
- **Inference.** Temperature and top-p sampling over N Monte Carlo
  trajectories, averaged.

## Transferable ideas for glance

1. **Coarse→fine screen tokens.** Coarse tokens carry layout, regions and app
   identity; fine tokens carry text and pixel detail. Train them with
   separate reconstruction targets (low-res or layout for coarse, full detail
   for fine).
2. **Coarse→fine actions.** Predict the action type (click, type, scroll,
   key, wait) first, then its arguments (target element, coordinates, text)
   conditioned on the sampled type.
3. **Split large vocabularies** (UI element ids, coordinates, text) into
   sub-tokens instead of one huge softmax.
4. **Train on sampled coarse tokens** so training matches closed-loop
   rollouts.
5. **Modality dropout.** Randomly drop the accessibility tree, OCR text or
   pixels during training, so the model still works on apps that expose no
   accessibility data, which is exactly glance's case.
6. **Normalize and clip numeric state** (amounts, geometry, scroll offsets,
   timings), with z-scores clipped like Kronos's [-5, 5].
7. **Temporal embeddings:** time since the last input, event time and
   session position. UI state changes are timing-sensitive (animations,
   network waits).
8. **Data hygiene:** drop idle frames, dedupe near-identical grabs, and
   upweight rare apps and workflows.
9. **Monte Carlo planning:** sample N action trajectories, score them
   (with a learned value or the verify-after-act check), and act on the
   best.

Caveats: screens are far higher-dimensional than a 6-number bar, so the
visual tokenizer needs patch-based encoding and larger codebooks. Desktop
action traces are also much scarcer than market data, so bootstrapping from
a pretrained vision backbone and from accessibility trees (where apps
expose them) is likely more practical than pretraining from scratch.

## Reference: Mamba and state-space models (the session backbone)

Kronos is a model of a *frame/action* — how a bar or a screen becomes tokens.
A **state-space model (SSM)** is a candidate for the *session*: the long,
interleaved `[state, action, state', …]` stream a desktop run produces. It is
a complement to the decoder-only idea above, not a rival — the same tokens,
a different sequence backbone. SSMs matter to glance for one reason: they are
linear-time and run inference in **constant memory per step**, which is the
shape of a live, on-device loop over a session that never stops growing.

### Lineage and mechanics

- **S4** ([arXiv 2111.00396](https://arxiv.org/abs/2111.00396)): a structured
  SSM — a linear recurrence `h_t = A·h_{t-1} + B·x_t`, `y_t = C·h_t` —
  reparameterized so it trains as a long convolution in O(L). First model to
  solve Long Range Arena's length-16k Path-X.
- **Mamba / S6** ([arXiv 2312.00752](https://arxiv.org/abs/2312.00752), Gu &
  Dao; repo [state-spaces/mamba](https://github.com/state-spaces/mamba)): the
  *selective* SSM. `A, B, C` and the step size become **functions of the
  input**, so the recurrence chooses what to keep and what to forget — the
  content-based gating plain SSMs lacked on discrete data like text. A
  hardware-aware **parallel scan** keeps training O(L); inference is a true
  recurrence — O(1) work and memory per token, no KV cache — reported at ~5×
  the throughput of a same-size Transformer.
- **Mamba-2 / SSD** ([arXiv 2405.21060](https://arxiv.org/abs/2405.21060),
  ICML 2024): "state-space duality" shows SSMs and attention are two readings
  of the same semiseparable-matrix product, yielding a core layer 2–8× faster
  than Mamba's.
- **Vision SSMs** — Vim ([arXiv 2401.09417](https://arxiv.org/abs/2401.09417))
  and VMamba ([arXiv 2401.10166](https://arxiv.org/abs/2401.10166)) — run
  selective scans over *image patches* with a bidirectional / four-way (SS2D)
  scan, for a global receptive field at cost linear in the patch count.
  Relevant because glance's input is pixels, not text.

### Tradeoffs vs Transformers (settled vs contested)

- **Settled win:** long context at linear time and fixed memory. A
  Transformer's KV cache grows with the session; an SSM's state is a
  fixed-size vector, so minute ten of a run costs the same per step as minute
  one.
- **Settled weakness:** exact recall and copying. A fixed-size state cannot
  losslessly hold an arbitrary history, so "repeat the value from 2,000
  frames ago" is hard, where attention just looks it up
  ([arXiv 2410.03810](https://arxiv.org/abs/2410.03810),
  [arXiv 2410.11135](https://arxiv.org/abs/2410.11135)).
- **Contested:** how fundamental that gap is — recent work argues part of it
  is an optimization/tuning artifact, not a hard ceiling
  ([arXiv 2508.19029](https://arxiv.org/abs/2508.19029)).
- **The usual fix is a hybrid:** Jamba
  ([arXiv 2403.19887](https://arxiv.org/abs/2403.19887)) interleaves a few
  attention layers among many Mamba layers (plus MoE), recovering recall
  while keeping most of the memory win.

### How glance could use it

1. **The session as one long stream.** Model a desktop run as the interleaved
   `[UI-state tokens, action tokens, …]` sequence glance already emits (the
   verify-after-act trace; the M5 event log). An SSM consumes it as a
   streaming recurrence — constant memory, no re-attention over a growing
   history — which is far closer to "small, nimble, on-device" than quadratic
   attention over the whole session.
2. **Where it sits on the encoder/decoder axis.** A vision encoder turns each
   grab into state tokens (a Vision-Mamba patch encoder is one option); an SSM
   backbone carries session state forward and a small head emits the next
   action. This is orthogonal to Kronos: Kronos informs *how a frame/action is
   tokenized* (coarse→fine, split vocabularies), the SSM is *what runs over the
   token stream in time* — and the decoder-only predictor above could itself
   be an SSM or an attention/SSM hybrid.
3. **Honest limits.** Recall is the real risk for glance: "remember the stack
   size three streets ago" or "the value that scrolled off 2,000 frames back"
   is exactly what a fixed state tends to drop — so a serious design is a
   hybrid (a few attention layers), or it leans on the deterministic state
   JSON glance already keeps as an explicit external memory rather than asking
   the recurrence to be a database. The scarcity/cost caveat from the Kronos
   section applies in full: desktop action traces are scarce and training is
   costly. This is a research direction to measure, **not a committed plan**.

### References (SSMs)

- S4 — [arXiv 2111.00396](https://arxiv.org/abs/2111.00396); repo
  [HazyResearch/state-spaces](https://github.com/HazyResearch/state-spaces)
- Mamba (S6) — [arXiv 2312.00752](https://arxiv.org/abs/2312.00752); repo
  [state-spaces/mamba](https://github.com/state-spaces/mamba)
- Mamba-2 / SSD — [arXiv 2405.21060](https://arxiv.org/abs/2405.21060)
- Jamba — [arXiv 2403.19887](https://arxiv.org/abs/2403.19887)
- Vision Mamba (Vim) — [arXiv 2401.09417](https://arxiv.org/abs/2401.09417);
  repo [hustvl/Vim](https://github.com/hustvl/Vim)
- VMamba — [arXiv 2401.10166](https://arxiv.org/abs/2401.10166)
- SSM recall/copy limits — [arXiv 2410.03810](https://arxiv.org/abs/2410.03810),
  [arXiv 2410.11135](https://arxiv.org/abs/2410.11135); contested:
  [arXiv 2508.19029](https://arxiv.org/abs/2508.19029)

## Research axes to try

- **Token arrangement:** pixel patches vs UI-element tokens vs OCR-text
  tokens. Options include interleaving `[state, action, state', …]`,
  diff-only frames, and multi-scale (whole screen → focused region).
- **Encoders:** a ViT-style patch encoder, a CNN backbone, a VQ/BSQ
  autoencoder (Kronos-style), or element-detector features from the glance
  engine.
- **Decoders and heads:** autoregressive token decoding, factored
  coarse/fine heads, and direct coordinate regression vs discretized grids.
- **Architectures:** decoder-only Transformers, encoder-decoder, state-space
  models for long sessions, and small on-device models for the live loop
  alongside a larger planner.

## First concrete steps (when this starts)

1. Record: a trace logger in glance that writes grab + state JSON + action +
   post-action grab (the verify-after-act result) for every step.
2. Benchmark: a fixed suite of desktop tasks with success criteria checked
   from grabs, so every model variant is measured the same way.
3. Baselines: today's template/glyph engine, then an off-the-shelf
   vision-language model, before training anything of our own.
