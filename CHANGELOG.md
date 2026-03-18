# Changelog

Chronological record of every experiment attempted during the commaVQ compression challenge.

## Score Progression Summary

| Date | Approach | Bits/token | Ratio | Delta |
|---|---|---|---|---|
| Mar 4 | LZMA + transpose | 6.21 | 1.61x | baseline |
| Mar 4 | Delta + LZMA | 7.81 | 1.28x | -0.33x |
| Mar 4 | Bitpack + LZMA | 7.76 | 1.29x | -0.32x |
| Mar 4 | Zstd | 6.99 | 1.43x | -0.18x |
| Mar 5 | Transition tables + ANS | 4.63 | 2.16x | +0.55x |
| Mar 7 | Temporal v1 + ANS | 4.20 | ~2.30x | +0.14x |
| Mar 8 | Temporal v2 + ANS | 4.05 | ~2.35x | +0.05x |
| Mar 10 | Temporal v3 + ANS | 3.90 | 2.40x | +0.05x |
| Mar 11 | Temporal v4 + ANS | 3.85 | ~2.42x | +0.02x |
| Mar 14 | Frame small (K=1) | 3.70 | ~2.58x | +0.16x |
| Mar 15 | Frame small3 (K=3) | 3.61 | ~2.66x | +0.08x |
| Mar 16 | Frame medium3 (K=3) | 3.51 | ~2.74x | +0.08x |
| Mar 17 | Full 5000-seg run | 3.56 | 2.75x | final |

## Detailed Timeline

### Day 1: Infrastructure and Classical Baselines

**Built pluggable backend system.** Created `backends.py` with LZMA, zstd, delta-LZMA, and bitpack-LZMA backends. Each backend implements compress/decompress with a common interface. Built `compress.py` to produce single-blob zip format (avoids per-file zip metadata overhead for 5,000 segments) and `decompress.py` to reverse it.

**Benchmarked classical approaches (100 segments):**
- LZMA with transpose: 1.610x (6.21 bits/token) -- best classical result
- LZMA optimized preset: 1.610x (identical)
- Zstd level 22: 1.432x (6.99 bits/token)
- Bitpack + LZMA: 1.289x (7.76 bits/token) -- *worse*, broke byte alignment
- Delta + LZMA: 1.280x (7.81 bits/token) -- *worse*, tokens not ordinal

**Key finding:** The transpose trick (reshape to 128xN, transpose) is critical for LZMA. It groups each position's temporal sequence contiguously, enabling dictionary-based repetition detection. Delta encoding fails because VQ tokens are categorical labels with no meaningful numerical ordering.

**Built evaluation harness.** Created `run_eval.py` for end-to-end testing: compress, build zip, verify round-trip, report metrics.

### Day 2: Dataset Analysis

**Statistical analysis of token distributions.** Created `analyze_data.py`:
- Measured per-position entropy (varies from ~8.5 to ~9.8 bits across positions)
- Computed temporal mutual information decay curve (5.40 bits at d=1, logarithmic decay)
- Analyzed run-length statistics and token frequency distributions
- Computed position-specific transition probabilities

**Measured conditional entropy ceilings.** Created `measure_ceiling.py`:
- H(token | prev_1) = 4.62 bits
- H(token | prev_3) = 4.08 bits
- H(token | prev_5) = 3.82 bits
- H(token | prev_10) = 3.51 bits
- H(token | prev_20) = 3.35 bits
- Diminishing returns beyond K=10

**Quantified spatial information gain.** Created `measure_spatial_gain.py`:
- H(token | prev_frame_same_pos) = 4.62 bits
- H(token | above_token_current_frame) = 3.39 bits
- H(token | both) = 1.45 bits
- **Finding:** Spatial context from the current frame is *more informative* than temporal context from the previous frame. Joint conditioning reduces entropy by 85%.

### Day 3-4: First Neural Models

**Built temporal model v1.** Per-position transformer predicting each of 128 positions independently from K=20 previous tokens at the same position. Config: dim=128, 4 layers, 4 heads, 1.34M params.

**Trained v1.** 30 epochs, AdamW with cosine LR. Train loss converged to ~4.2 bits/token.

**Built GPT reference.** Created `gpt_compress.py` with full autoregressive GPT-2 style model for upper-bound analysis. Model is ~1.3 GB, far too large to ship, but establishes ceiling at ~2.1 bits/token.

**Built temporal model v2.** Added 4 spatial neighbors from previous frames (above, below, left, right) as additional context tokens. Modest improvement over v1.

**Built ANS compression pipeline.** Created `temporal_compress.py` integrating model inference with `constriction` ANS coder. Key implementation detail: probabilities must be quantized to 24-bit fixed-point for exact reproducibility between encode and decode.

**Built transition table baseline.** Created `transition_compress.py`: per-position P(token | prev_token) lookup tables + ANS. Result: 2.16x, demonstrating the value of position-specific modeling. But tables are ~50 MB, eating most of the savings.

### Day 5-6: Improved Temporal Models

**Built temporal model v3.** Added current-frame above-token as spatial context. During row-by-row decoding, the row above is already decoded, so this is available during decompression. Required changing the compression pipeline to process rows sequentially.

**Trained v3 and measured actual compression.** ANS compression on 100 random segments: 3.896 bits/token. Significantly worse than the 3.086 bits/token from eval on segments 0-2 (frames 20-219).

**Debugged the evaluation gap.** Created `debug_v3_gap.py` and `debug_frame_range.py`:
- Segments 0-2 were unusually easy (highway driving)
- Eval window (frames 20-219) missed the harder early frames
- Per-segment variance is substantial but no directional frame-position trend
- **Lesson:** Always evaluate on actual task distribution

**Built temporal model v4.** Added above-left and above-right neighbors from current frame (raster-order compatible). Less than 0.05 bits improvement over v3. The marginal value of 2 extra spatial neighbors is negligible compared to the 127 positions a frame-level model can see.

**Bug found and fixed in v4:** The initial version included a "left" neighbor from the current frame, but during batched compression, all 16 positions per row are processed simultaneously, so the left neighbor isn't yet decoded. Removed left neighbor, kept only 3 above-row neighbors.

### Day 7: Experiments and Dead Ends

**Extended v3 training with all-timestep loss.** Created `train_temporal_v3_long.py`: trained for 60 epochs with loss on all timesteps (not just the final prediction). Result: 3.626 bits eval, *worse* than standard 30-epoch training. The intermediate predictions are a different task, and spending model capacity on them hurts the actual target.

**Context length experiments.** Created `experiment_context_len.py`: measured mutual information for various K values. Confirmed diminishing returns beyond K=3 when using full spatial context.

**Hybrid neural + transition table experiments.** Created `experiment_hybrid.py` and `experiment_hybrid_v3_transition.py`: blended neural model probs with transition table probs. Maximum improvement: 0.002 bits. The neural model already subsumes the transition table patterns.

**Table compression with product-of-experts.** Created `table_compress.py`: P(token | prev) * P(token | above) / P(token). Gets 3.62 bits/token but tables are 49 MB LZMA. Net compression 2.405x, not competitive.

**Improvement experiments.** Created `experiment_improvements.py`: tested various architectural tweaks to the per-position model. None provided meaningful gains over v3.

### Day 8-9: The Breakthrough -- Frame-Level Models

**Identified the fundamental limitation** of per-position models: they treat each of the 128 positions independently, regardless of how many spatial neighbors you add. The conditional entropy analysis showed that *full* spatial context (all previously decoded positions in the current frame) is qualitatively different from *partial* spatial context (a few neighbors).

**Built temporal-spatial model.** Created `temporal_spatial_model.py`: attempted to combine temporal context with broader spatial context within the per-position framework. Did not significantly outperform v3.

**Built temporal-frame model.** Created `temporal_frame_model.py`: frame-level decoder with mean-pooled temporal context from K=10 previous frames. Underperformed vs. simply using K=3 unmodified frames. Mean pooling destroys temporal ordering and token identity.

**Built frame-level autoregressive model.** Created `frame_model.py`: the key breakthrough. Processes the entire frame as a 128-position autoregressive sequence with:
- Cross-attention to K previous frames (K*128 context tokens)
- Causal self-attention within current frame (raster order)
- Full spatial context: every previously decoded position is visible

**Trained frame small (K=1).** dim=128, 4 layers, 4 heads, 1.34M params. Eval: 3.698 bits/token. Already better than the v3 model with K=20, using only 1 previous frame but full spatial context.

**Trained frame small3 (K=3).** Same architecture, 3 previous frames. Eval: 3.605 bits/token. Confirmed that K=3 with full spatial context >> K=20 with narrow spatial context.

### Day 10-11: Scaling Up and Submission

**Trained frame medium3 (K=3).** dim=192, 6 layers, 4 heads, 3.98M params, 6.8 MB compact. Eval on 100 segments: 3.514 bits/token.

**Built frame compression pipeline.** Created `frame_compress.py`: full ANS pipeline for frame model. Teacher-forced compression (1 pass/frame) and sequential decompression (128 passes/frame). Verified round-trip correctness on test segments.

**Built self-contained decompressor.** Created `decompress_submission.py`: inlines the full FrameModel architecture with no external dependencies beyond torch/numpy/constriction. Auto-installs constriction if missing. Reads OUTPUT_DIR from environment per evaluate.sh contract.

**Prototyped row-level model.** Created `row_model.py` and `train_row_model.py`: processes 16 tokens per row in parallel (8 sequential steps per frame instead of 128). Early results: row small at epoch 15 = 3.636 bits/token, row medium at epoch 6 = 3.667 bits/token. Comparable quality with ~16x faster decompression. Did not finish training due to time/GPU constraints.

**Ran full 5,000-segment compression.** Frame medium3 model, ~3 hours 49 minutes on single GPU.

### Day 12: Final Results and Documentation

**Final submission results:**
- Zip size: 333.3 MB
- Compression ratio: 2.75x
- Bits/token: 3.563
- Model overhead: 6.8 MB (2% of zip)

**Generated analysis figures and technical report.** Created report/generate_plots.py (7 matplotlib figures) and report/report.typ (Typst source for detailed technical writeup).

## Approaches Not Attempted (Future Work)

- **Larger models (10M+ params):** Likely net positive but diminishing returns
- **Online adaptation:** Fine-tune per segment, transmit weight deltas
- **INT8/INT4 quantization:** Could halve model size
- **Non-raster orderings:** Center-out or learned token order
- **Ensemble methods:** Multiple small models with mixture predictions
- **Self-compressing neural network:** The approach used by the top leaderboard entry
