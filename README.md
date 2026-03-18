# Neural Lossless Compression of VQ-Encoded Driving Video

**2.75x compression ratio** | 3.56 bits/token | 4M param transformer + ANS

[**Download Submission Zip**](https://drive.google.com/file/d/1_9LuFmJ-djx-hMhcWgv7TI3ALrHlK8U6/view?usp=drive_link) | [**Technical Report (PDF)**](report/report.pdf) | [**Changelog**](CHANGELOG.md)

---

## Overview

This writeup describes my approach to the commaVQ compression challenge: losslessly compress 5,000 segments of VQ-encoded driving video into the smallest possible zip file. The final submission achieves a **2.75x compression ratio** (333 MB zip from 915 MB of token data), using a 4-million parameter frame-level autoregressive transformer paired with ANS entropy coding.

The approach was developed iteratively over roughly two weeks. I want to walk through not just the final system, but the full trajectory: what I tried, what failed, and what each failure revealed about the structure of the problem.

## The Setup

The dataset consists of 5,000 driving segments. Each segment has 1,200 frames, and each frame is an 8x16 grid of discrete tokens from a vocabulary of 1,024 (10 bits per token). That is 768 million tokens total, roughly 915 MB at 10 bits each.

The submission is a zip file containing compressed data plus a `decompress.py` script. The score is `original_size / zip_size`. The constraint that matters is that the decompression model must fit inside the zip alongside the data. This creates a tradeoff: a larger model reduces the entropy-coded data size but increases the archive through its own weight footprint.

The core theoretical observation is Shannon's source coding theorem: a probabilistic model assigning probability p(x) to symbol x can compress it to -log2(p(x)) bits. Better prediction = shorter codes. The entire project reduces to: build the best predictor that fits in a few megabytes.

## Phase 1: Classical Baselines

Before training any models, I established baselines with classical compression.

**LZMA with transposition (1.61x).** The single most impactful classical trick is to transpose the data. Each segment is (1200, 8, 16). Reshaping to (128, 1200) and transposing puts each spatial position's temporal sequence contiguously in the byte stream. LZMA's sliding-window dictionary then captures temporal repetitions efficiently.

**Delta encoding (1.28x).** My first instinct was to delta-encode consecutive frames, since driving video has high temporal redundancy. This actually *hurt*. The reason is structural: VQ token indices are categorical, not ordinal. Token 500 and token 501 represent arbitrarily different visual patterns. Frame-to-frame differences of categorical labels produce high-entropy residuals, unlike pixel-space deltas where differences are typically small.

**10-bit packing (1.29x).** Also counterproductive. Packing 10-bit values into a dense bitstream breaks the byte alignment that LZMA depends on for dictionary matching. The "wasted" 6 bits per token in int16 storage actually help LZMA by maintaining alignment.

These failures were instructive. They taught me that preprocessing which looks good from an information-theoretic perspective can actively hurt if it conflicts with the assumptions of the downstream compressor. Byte alignment and dictionary structure matter for LZMA in ways that entropy calculations do not capture.

## Phase 2: Understanding the Data

I spent significant time measuring the statistical structure of the token sequences before building neural models.

**Temporal mutual information.** Adjacent frames at the same spatial position share 5.40 bits of mutual information (out of 10 bits total entropy). This decays roughly logarithmically with temporal distance: 1.53 bits at distance 20, 1.24 bits at distance 40.

**Conditional entropy ceilings.** I measured H(token | previous K tokens at same position) for K = 1, 2, 3, 5, 10, 20. Even K=20 only reduces the conditional entropy to about 4.6 bits/token when using a single position's history.

**Spatial context: the key observation.** This was the most important measurement of the project. I computed conditional entropy under different spatial conditioning:

| Conditioning | Entropy (bits/token) |
|---|---|
| None (marginal) | 10.0 |
| Previous frame, same position | 4.62 |
| Current frame, above token | 3.39 |
| Both previous frame + above token | 1.45 |

The joint conditioning reduces entropy from 10 bits to 1.45 bits, an 85% reduction. What surprised me was that spatial context from the *current* frame (3.39 bits) is actually more informative than temporal context from the *previous* frame (4.62 bits). This fundamentally reoriented the project: any approach that models positions independently is leaving most of the compressible information untouched, regardless of how much temporal context it uses.

## Phase 3: Per-Position Temporal Models

Armed with the analysis, I built progressively more capable per-position temporal models.

**v1: Pure temporal.** A small transformer predicting each of the 128 spatial positions independently, conditioned on K=20 previous tokens at the same position. This establishes the neural baseline but ignores all spatial structure.

**v2: Temporal + previous-frame neighbors.** Added 4 spatial neighbors (above, below, left, right) from previous frames as additional context. Modest improvement: the model can now see a small spatial neighborhood, but only from previous frames.

**v3: Temporal + current-frame above token.** Added the token directly above in the current frame as conditioning. This is the critical spatial signal: since we decode rows top-to-bottom, the above token is already known during decompression. Result: 3.90 bits/token actual (ANS on 100 segments).

**v4: Temporal + raster-order neighbors.** Extended v3 with above-left and above-right neighbors from the current frame. This barely helped (less than 0.05 bits improvement). The problem is fundamental: adding 2 more spatial neighbors when the frame has 128 positions is a marginal increase. The model needs a qualitatively different approach to spatial context, not incremental additions.

### Debugging the Evaluation Gap

An important aside: early evaluations of v3 showed 3.09 bits/token on segments 0-2, frames 20-219. Actual ANS compression on 100 random segments gave 3.90 bits/token. The 0.8-bit gap came from two sources: (1) segments 0-2 were unusually easy (likely highway driving with minimal visual change), and (2) the evaluation window missed the harder early frames and the full diversity of the sequence. I wrote debugging scripts to diagnose this, which confirmed that the per-segment variance is substantial but there is no consistent directional trend across frame positions.

Lesson: always evaluate on the actual task distribution. Convenient proxy metrics on small subsets will mislead you, and typically in the optimistic direction.

## Phase 4: Dead Ends

Several approaches that seemed promising turned out not to help.

**Transition tables + ANS (2.16x).** Per-position first-order Markov tables P(token | prev_token, position) give 2.16x compression. But the tables themselves are ~50 MB after LZMA, eating most of the data savings. The net compression was worse than the v3 neural approach which captures the same patterns in 2.3 MB of model weights.

**Factored table compression.** Using P(token | prev) * P(token | above) / P(token) as a product-of-experts approximation. Gets 3.62 bits/token but tables are 49 MB. Not competitive.

**Hybrid neural + transition tables.** Blending neural model probabilities with transition table probabilities. At most 0.002 bits improvement. The neural model already subsumes the patterns the tables capture.

**All-timestep training loss.** Training v3 with loss on all timesteps (not just the final prediction) for 60 epochs. Performed worse than standard training at 30 epochs. The intermediate predictions are a different task (less temporal context), and spending model capacity on them hurts the actual target.

**Mean-pooled temporal context.** A frame-level decoder attending to mean-pooled summaries of K=10 previous frames. Worse than attending to K=3 unmodified frames. Mean pooling destroys temporal ordering and token identity: the discrete structure of the codebook is lost when you average embeddings.

## Phase 5: Frame-Level Autoregressive Models

The conditional entropy analysis pointed clearly to the solution: model the entire frame jointly, with full spatial context.

### Architecture

The frame model processes each frame as a 128-position autoregressive sequence in raster order (left-to-right, top-to-bottom). It consists of:

1. **Previous frame encoder.** Each of K=3 previous frames is embedded via shared token + position embeddings, producing K*128 context vectors.

2. **Current frame decoder.** Transformer blocks with:
   - Causal self-attention: position i attends to positions 0..i-1 (raster order)
   - Cross-attention: all positions attend to the K*128 previous-frame context
   - Feed-forward network

3. **Output head.** Linear projection to 1024-way classification (vocabulary size).

The key design decision: **K=3 with full spatial context beats K=20 with narrow spatial context.** The per-position model uses K=20 because each position only sees its own temporal history. The frame model sees the entire previous frame (128 positions), providing far richer context per frame. Three frames of 128 positions each (384 context tokens) carry more usable information than 20 frames of 1 position (20 context tokens).

| Model | Temporal Context | Spatial Context | Bits/token |
|---|---|---|---|
| v3 (per-position) | K=20, 1 position | 1 above token | 3.896 |
| Frame small (K=1) | 1 full frame | Full causal | 3.698 |
| Frame small3 (K=3) | 3 full frames | Full causal | 3.605 |
| Frame medium3 (K=3) | 3 full frames | Full causal | 3.514 |

### Compression-Decompression Asymmetry

An elegant property of autoregressive models: during compression, all tokens are known, so the causal mask lets us compute all 128 probabilities in a single forward pass (teacher forcing). Compression takes 1 forward pass per frame, or ~2.7 seconds per segment.

Decompression requires sequential decoding: predict position 0, decode from ANS, feed back, predict position 1, etc. That is 128 forward passes per frame, roughly 128x slower. This asymmetry is acceptable since decompression speed is not scored, but for practical deployment I prototyped a row-level model (8 passes per frame instead of 128) that showed comparable bits/token.

### Model Configurations

| Model | d_model | Layers | Heads | Params | Compact Size |
|---|---|---|---|---|---|
| Frame small (K=1) | 128 | 4 | 4 | 1.34M | 2.6 MB |
| Frame small3 (K=3) | 128 | 4 | 4 | 1.34M | 2.3 MB |
| Frame medium3 (K=3) | 192 | 6 | 4 | 3.98M | 6.8 MB |

The medium3 model is 4.5 MB larger than small3 but saves ~0.09 bits/token, which translates to roughly 8.6 MB of compressed data savings. Net gain: ~4 MB.

### Model Compression

The trained weights are stored as:
1. FP16 quantization (no quality loss observed)
2. LZMA compression of the fp16 buffer
3. Compact JSON header for architecture config

The 3.98M parameter model compresses from 7.6 MB (raw fp16) to 6.8 MB (LZMA).

## Phase 6: ANS Entropy Coding

I use Asymmetric Numeral Systems (ANS) via the `constriction` Python library, specifically the stack-based variant.

For each segment:
1. Run the model teacher-forced over all 1,200 frames to get probability distributions for all 153,600 tokens.
2. Quantize probabilities to 24-bit fixed-point integers.
3. Encode tokens in reverse order using `AnsCoder.encode_reverse()`.
4. Store the compressed bitstream.

Decompression reverses this: decode tokens forward, regenerating probabilities as each token is revealed.

Critical implementation detail: the probability distributions used during encoding and decoding must be *exactly* identical at the bit level. Any floating-point non-determinism will cause silent decoding failures. The 24-bit quantization ensures reproducibility.

## Final Results

Full 5,000-segment compression with Frame medium3:

| Component | Size | Fraction |
|---|---|---|
| Compressed data (ANS) | 326.2 MB | 97.9% |
| Model weights (LZMA fp16) | 6.8 MB | 2.0% |
| decompress.py | ~10 KB | <0.1% |
| **Total zip** | **333.3 MB** | |
| Original data | 915 MB | |
| **Compression ratio** | **2.75x** | |
| **Bits/token** | **3.563** | |

Compression time: 3 hours 49 minutes on a single GPU.

### Score Progression

| Approach | Bits/token | Ratio | Notes |
|---|---|---|---|
| LZMA + transpose | 6.21 | 1.61x | Classical baseline |
| Delta + LZMA | 7.81 | 1.28x | Hurt: tokens not ordinal |
| Transition tables | 4.63 | 2.16x | Tables too large (49 MB) |
| Per-position v3 | 3.90 | 2.40x | Limited spatial context |
| Frame small (K=1) | 3.70 | ~2.58x | Full spatial, 1 prev frame |
| Frame small3 (K=3) | 3.61 | ~2.66x | 3 prev frames |
| **Frame medium3 (K=3)** | **3.56** | **2.75x** | **Final submission** |

## What I Would Try With More Time

1. **Larger model.** A 10M parameter model (~18 MB compact) might be net positive if it saves more than 18 MB in compressed data. The medium vs. small improvement curve suggests this is likely.

2. **Online adaptation.** Fine-tune the model on each segment before compressing it, transmitting weight deltas. This is the direction the top leaderboard entry (self-compressing neural network) takes.

3. **INT8 quantization.** More aggressive weight quantization could shrink the model by 2x with careful calibration.

4. **Non-raster orderings.** Center-out or learned orderings could provide better spatial context at each position.

5. **Row-level model.** The row model prototype (8 forward passes per frame vs. 128) showed comparable bits/token with much faster decompression. More training could make this the better choice for practical deployment.

## Reproduction

### Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- `constriction` (ANS entropy coding library)
- `datasets` (HuggingFace, for loading commaVQ)

### Steps

```bash
# Install dependencies
pip install torch constriction datasets numpy

# Train the frame model (medium3 config, ~63 minutes on GPU)
python compression/train_frame_model.py --config medium3 --epochs 30

# Compress all 5,000 segments (~4 hours on GPU)
python compression/frame_compress.py --model compression/frame_medium3.pt --skip-verify

# The submission zip is written to compression/frame_submission.zip

# Verify (WARNING: decompression is ~128x slower than compression)
bash compression/evaluate.sh compression/frame_submission.zip
```

### Quick evaluation (100 segments)

```bash
python compression/frame_compress.py --model compression/frame_medium3.pt --num-segments 100
```

## Files

| File | Description |
|---|---|
| `compression/backends.py` | Pluggable classical compression backends (LZMA, zstd, delta, bitpack) |
| `compression/compress.py` | Classical compression harness with single-blob zip format |
| `compression/decompress.py` | Classical decompression (included in LZMA-based submissions) |
| `compression/run_eval.py` | End-to-end evaluation harness |
| `compression/analyze_data.py` | Dataset statistical analysis (entropy, distributions, correlations) |
| `compression/measure_ceiling.py` | Temporal conditional entropy measurement for K=1..20 |
| `compression/measure_spatial_gain.py` | Spatial context information gain quantification |
| `compression/temporal_model.py` | Per-position temporal transformer v1 |
| `compression/temporal_model_v2.py` | v2: adds spatial neighbors from previous frames |
| `compression/temporal_model_v3.py` | v3: adds current-frame above-token context |
| `compression/temporal_model_v4.py` | v4: adds raster-order spatial neighbors |
| `compression/temporal_compress.py` | ANS compression pipeline for v2 temporal model |
| `compression/temporal_v3_compress.py` | ANS compression pipeline for v3 (row-by-row) |
| `compression/temporal_v4_compress.py` | ANS compression pipeline for v4 (raster-order) |
| `compression/transition_compress.py` | Transition table baseline with ANS |
| `compression/table_compress.py` | Product-of-experts table compression |
| `compression/temporal_spatial_model.py` | Combined temporal + spatial context model |
| `compression/temporal_frame_model.py` | Mean-pooled temporal context frame model |
| `compression/frame_model.py` | Frame-level autoregressive transformer (final architecture) |
| `compression/frame_compress.py` | ANS compression pipeline for frame model |
| `compression/decompress_submission.py` | Self-contained submission decompressor |
| `compression/row_model.py` | Row-level model for faster decompression |
| `compression/gpt_compress.py` | GPT reference compression (model too large to ship) |
| `compression/ans_compress.py` | GPT + ANS compression pipeline |
| `compression/train_*.py` | Training scripts for each model variant |
| `compression/experiment_*.py` | Ablation and analysis experiments |
| `compression/debug_*.py` | Debugging and diagnostic scripts |
| `report/report.typ` | Typst source for technical report |
| `report/generate_plots.py` | Matplotlib figure generation |
