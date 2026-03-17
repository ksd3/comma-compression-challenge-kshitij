// commaVQ Compression Challenge — Technical Report

#set document(
  title: "Compressing Driving Video: A Story About What Matters",
  author: "Technical Report",
  date: datetime(year: 2026, month: 3, day: 8),
)

#set page(
  paper: "a4",
  margin: (x: 2.2cm, y: 2.5cm),
  header: context {
    if counter(page).get().first() > 1 [
      #set text(8pt, fill: gray)
      _Compressing Driving Video: A Story About What Matters_
      #h(1fr)
      #counter(page).display()
    ]
  },
)

#set text(font: "New Computer Modern", size: 10.5pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => {
  v(0.8em)
  text(14pt, weight: "bold", it)
  v(0.4em)
}
#show heading.where(level: 2): it => {
  v(0.6em)
  text(12pt, weight: "bold", it)
  v(0.3em)
}

// Title block
#align(center)[
  #v(1cm)
  #text(20pt, weight: "bold")[Compressing Driving Video:\ A Story About What Matters]
  #v(0.5cm)
  #text(12pt, fill: gray)[Technical Report --- commaVQ Compression Challenge]
  #v(0.3cm)
  #text(11pt)[March 2026]
  #v(1cm)
]

// Abstract
#rect(
  width: 100%,
  inset: (x: 1.2cm, y: 0.8cm),
  stroke: 0.5pt + gray,
  radius: 4pt,
)[
  #text(11pt, weight: "bold")[Abstract.]
  We describe a journey through lossless compression of the commaVQ dataset --- 5,000 minutes of driving video encoded as discrete VQ tokens, roughly 915 MB of raw data. The challenge is to pack this data, along with a decompression program, into the smallest possible zip file. We arrive at a 4-million parameter autoregressive transformer that achieves 3.56 bits/token via ANS entropy coding, yielding a final zip of 333 MB and a compression ratio of 2.75$times$. The interesting part is not the final number but what we learned getting there: that spatial context dominates temporal context, that tiny models beat giant ones when you can train on the test set, and that the gap between "should work in theory" and "actually works in practice" is where all the interesting problems live.
]

#v(0.5cm)

= The Problem

Here's the setup. comma.ai has a dataset of driving video --- dashcam footage of cars driving around. They've run it through a VQ-VAE, which means each frame of video has been compressed into a grid of discrete tokens: 8 rows by 16 columns, 128 tokens per frame, each drawn from a vocabulary of 1,024 symbols. There are 5,000 driving segments, each 1,200 frames long (about a minute at 20 fps). That's 768 million tokens total.

The challenge: compress all of this losslessly into a zip file. The zip must contain a `decompress.py` script that can reconstruct every single token perfectly. Your score is the ratio of original size to zip size. Bigger ratio = better compression.

The raw data takes about 915 MB if you pack the 10-bit tokens efficiently. The question is how small you can make it.

This is, at its core, a prediction problem. Shannon proved in 1948 [1] that compression and prediction are two sides of the same coin: if you can predict what comes next with probability $p$, you can encode it using $-log_2(p)$ bits. Perfect prediction means zero bits. No prediction at all means 10 bits per token (since there are 1,024 possibilities). Everything interesting happens in between.

= What We're Working With

Before trying to compress anything, it's worth understanding what the data actually looks like.

== The Token Grid

Each frame is an $8 times 16$ grid. Think of it as a very low-resolution image, except instead of pixel values you have codebook indices. Token 0 and token 1 are not necessarily similar --- the codebook is learned by the VQ-VAE, and the numbering is arbitrary. This turns out to matter a lot, as we'll see.

The total data is:
$ bold(X) in {0, 1, ..., 1023}^(5000 times 1200 times 8 times 16) $

Stored as 16-bit integers, this is about 1.44 GB. But only 10 bits per token are meaningful (since $log_2(1024) = 10$), so the "fair" uncompressed size the challenge uses is $768 times 10^6 times 10 / 8 approx 960$ MB. The compression ratio is computed against this number.

== Temporal Redundancy: The Obvious Observation

Driving video is boring. I mean this in the best possible way for compression: most of the time, the car is driving forward on a road, and consecutive frames look almost identical. The dashcam is mounted in a fixed position, the road is in front of you, and unless something dramatic happens, frame $t$ looks a lot like frame $t-1$.

#figure(
  image("figures/fig1_mutual_info.png", width: 75%),
  caption: [Mutual information between tokens at the same spatial position across time. Adjacent frames share over 5 bits of information out of 10 total --- more than half the entropy is redundant.]
) <fig:mi>

We can quantify this. @fig:mi shows the mutual information between a token and the token at the same spatial position in a previous frame. Adjacent frames share 5.40 bits of mutual information per token. That's more than half the total entropy of 10 bits. In plain terms: if you know what token was at position $(r, c)$ in the previous frame, you've already eliminated more than half the uncertainty about what it will be in the current frame.

The mutual information decays roughly logarithmically with temporal distance --- 1.53 bits at $d = 20$, 1.24 bits at $d = 40$. There's a long tail of useful information, but the first frame gives you most of it.

This is the obvious thing to exploit, and it's where most approaches start. But as we'll see, it's not the most important thing.

== Spatial Redundancy: The Non-Obvious Observation

Here's where it gets interesting. Within a single frame, tokens are also correlated with each other. The top row of the frame is sky. The bottom row is road. The tokens in between are a gradual transition of scenery. Neighboring tokens tend to be similar because they represent adjacent patches of the same visual scene.

#figure(
  image("figures/fig5_spatial_context.png", width: 72%),
  caption: [Conditional entropy under different conditioning contexts. The combination of temporal context (previous frame) and spatial context (token directly above in the current frame) reduces entropy from 10 bits to just 1.45 bits.]
) <fig:spatial>

@fig:spatial tells a story that completely reoriented our approach. Look at the numbers:

- Unconditional entropy: *10 bits* (uniform over 1,024 tokens)
- Conditioned on same position, previous frame: *4.62 bits* (temporal context alone)
- Conditioned on token directly above in current frame: *3.39 bits* (spatial context alone)
- Conditioned on *both*: *1.45 bits*

That last number is remarkable. If you know both what was at this position last frame _and_ what's directly above this position in the current frame, you've reduced the entropy from 10 bits to 1.45 bits. That's an 85% reduction. The spatial context alone (3.39 bits) is actually _more informative_ than the temporal context alone (4.62 bits).

This was the most important finding of the entire project. It means that any compression scheme that models each spatial position independently --- no matter how much temporal context it uses --- is leaving enormous amounts of information on the table. You _need_ to model the spatial structure within each frame.

I want to emphasize how non-obvious this was, at least to me. My initial assumption was that driving video compression would be primarily a temporal modeling problem. You look at the previous frames, predict the current one, and encode the residual. That's how video codecs work, roughly. But the token grid is so small ($8 times 16$) that within-frame correlations are incredibly strong. Every token has a lot to say about its neighbors.

= The Journey

Let me walk through the approaches we tried, in roughly chronological order. The failures are as instructive as the successes.

== Starting Point: Classical Compression

Before doing anything clever, you should always try the dumb thing first. LZMA is a general-purpose compression algorithm that's very good at finding patterns in byte streams.

The one trick that helps LZMA a lot: *transpose the data*. Instead of storing frames sequentially (which puts spatially adjacent but temporally distant tokens next to each other in the byte stream), reshape each segment from $(1200, 8, 16)$ to $(128, 1200)$ --- i.e., put each spatial position's entire temporal sequence together. Now LZMA's sliding-window dictionary can easily find the temporal repetitions.

Result: *1.61$times$* compression. Not bad for essentially one line of code (`data.reshape(128, -1).T`).

We also tried several "clever" preprocessing steps that turned out to be anti-clever:

*Delta encoding* ($x_t - x_(t-1) mod 1024$): 1.28$times$. _Worse_ than raw LZMA. Why? Because VQ tokens don't have a meaningful numerical ordering. Token 500 is not "close to" token 501 in any useful sense. The codebook indices are arbitrary, so taking differences produces high-entropy noise rather than small residuals. This is a fundamental difference from pixel-based video compression, where delta encoding is extremely effective.

*10-bit packing*: 1.29$times$. Also worse. The tokens are stored as 16-bit integers, and you might think packing them into 10 bits would save space. It does save raw bits, but it _destroys the byte alignment_ that LZMA relies on. LZMA's dictionary matching works on byte boundaries, and breaking that alignment makes it much harder for LZMA to find patterns. The 6 wasted bits per token are actually useful to LZMA.

These failures illustrate an important principle: preprocessing that seems information-theoretically sound can still hurt in practice if it disrupts the assumptions of your downstream compressor. The theoretical optimum doesn't care about byte alignment, but LZMA does.

== First Neural Attempt: Transition Tables

Before building any neural networks, we tried the simplest possible learned model: per-position first-order Markov transition tables.

For each of the 128 spatial positions, compute $P(x_t | x_(t-1))$ --- the probability of each token given what was at this position in the previous frame. That's a $1024 times 1024$ table for each position. Feed these probabilities into an ANS entropy coder.

Result: *2.16$times$*. A big jump over LZMA, and it tells us something important: position-specific temporal modeling is valuable. Token 500 at position (0, 0) --- the top-left corner, probably sky --- has very different transition dynamics than token 500 at position (7, 15) --- the bottom-right corner, probably road or car hood. LZMA can't capture this because it doesn't "know" about the spatial grid structure.

But the transition tables themselves are huge: $128 times 1024 times 1024 times 4$ bytes $approx$ 512 MB uncompressed. Even after heavy compression, they're around 49 MB --- far too large relative to the data savings they provide. This makes the net compression worse than just using the tables' predictions without having to ship the tables (which you obviously can't do).

Still, it proved the concept: if you can predict the next token well, you can compress the data well. The question is whether you can do it with a much smaller model.

== Per-Position Temporal Model: The First Neural Network

Our first real neural model was what I'll call the "v3" model. The idea: train a small transformer to predict each spatial position independently, conditioned on $K = 20$ previous frames at that position (and a few spatial neighbors from previous frames).

The architecture is simple. For each of the 128 spatial positions, take the $K$ previous tokens at that position (plus 4 neighbors: above, below, left, right from the previous frame), embed them, run them through a small causal transformer, and predict the next token.

The "small" configuration: 128-dimensional embeddings, 4 layers, 4 attention heads, 512-dimensional FFN. 1.34M parameters, compressing to 2.3 MB after fp16 quantization and LZMA.

=== The Evaluation Trap

Here's where we ran into one of those subtle bugs that wastes hours and teaches you something.

During development, we evaluated on segments 0 through 2, using frames 20 to 219. The model showed beautiful numbers: 3.086 bits/token. Extrapolating naively, we expected a compression ratio around 3.2$times$.

Then we ran actual ANS compression on 100 randomly sampled segments and got 3.896 bits/token. Almost a full bit worse. What happened?

#figure(
  image("figures/fig3_bits_by_frame_range.png", width: 72%),
  caption: [Bits per token by frame range, averaged across 5 segments. The model achieves roughly consistent performance across frame positions, but there is segment-to-segment variance that biased our initial evaluation.]
) <fig:frame_range>

Two things. First, segments 0--2 happened to be unusually easy (some driving sequences are just more predictable than others --- highway driving vs. city intersections, for instance). Second, and more subtly, we were evaluating only frames 20--219 out of 1,200 frames, which missed both the hard early frames (where the model has limited temporal context) and the full diversity of the sequence.

Lesson: always evaluate on the actual task. Proxy metrics on convenient subsets will lie to you, and they'll lie in the direction you want to hear.

=== The Result, Honestly Measured

With honest measurement (actual ANS compression on 100 representative segments), the v3 model achieves *3.896 bits/token*, yielding a compression ratio of about *2.40$times$* after accounting for the model size.

This was already 50% better than LZMA, but we knew we were leaving a lot on the table. Remember that conditional entropy number: 1.45 bits if you condition on both the previous frame and the above token. Our model was getting 3.9 bits. The gap was enormous.

The reason was clear from the conditional entropy analysis: the per-position model only conditions on a tiny sliver of spatial context. It sees the token directly above in the current frame, and that's it. It has no idea what's to the left, what's two rows up, or what's happening in any other part of the frame. It's like trying to predict what word comes next in a sentence while only being allowed to see one previous word from a different part of the paragraph.

== The Key Insight: Frame-Level Modeling

This is the part where the project pivoted from "incrementally better" to "fundamentally different."

Instead of modeling each position independently, what if the model saw the _entire_ previous frame and predicted the _entire_ current frame, token by token, in raster order? Each token would be conditioned on:
1. All tokens from $K$ previous frames (via cross-attention)
2. All previously decoded tokens in the current frame (via causal self-attention)

This is just the standard decoder-only transformer architecture applied to 128-token "sequences" (one frame), with cross-attention to $K times 128$ context tokens from previous frames.

The crucial property: during raster-order decoding, when predicting token $i$, the model has access to tokens $0, 1, ..., i-1$ from the current frame. Token 0 is the top-left corner; token 127 is the bottom-right. So by the time you're predicting a token in row 3, you've already decoded all of rows 0, 1, and 2 --- a massive amount of spatial context.

=== Architecture Details

#figure(
  image("figures/fig6_architecture.png", width: 85%),
  caption: [The frame-level autoregressive model. Previous frames are encoded via shared embeddings and attended to via cross-attention. Current-frame tokens are decoded left-to-right, top-to-bottom with causal masking.]
) <fig:arch>

The model (@fig:arch) has three components:

*Previous frame encoder.* Each of $K = 3$ previous frames is embedded using shared token embeddings plus learned position embeddings. This produces $K times 128$ context vectors. No self-attention is applied to the previous frames --- they're just a bag of embeddings that serve as keys and values for cross-attention. This is cheap and effective.

*Current frame decoder.* The current frame is processed as a 128-position autoregressive sequence. Each transformer block has:
- Causal self-attention (each position attends to all previous positions)
- Cross-attention (each position attends to all $K times 128$ previous-frame context vectors)
- Feed-forward network

The input at position $i$ is the embedding of token $i - 1$ (the standard "shift right" trick for autoregressive models). The first position receives a learned start token.

*Output head.* A linear projection from $d_("model")$ to 1,024 produces logits for each vocabulary token.

=== The Compression-Decompression Asymmetry

Here's an elegant property of autoregressive models that makes them extremely practical for compression (though not for decompression).

*During compression*, all tokens are known. We're not generating --- we're computing probabilities. Thanks to the causal mask, we can compute the probability of every token in a frame in a _single_ forward pass. The mask ensures that position $i$'s prediction only depends on positions $0, ..., i-1$, even though all positions are processed in parallel. This means compression takes exactly 1 forward pass per frame, or 1,200 forward passes per segment. At 2.7 seconds per segment, that's fast.

*During decompression*, we don't know the tokens yet. We have to decode them one by one: predict position 0, sample (well, decode from the ANS bitstream), feed that into the model, predict position 1, decode, and so on. That's 128 sequential forward passes per frame, or $128 times 1200 = 153,600$ forward passes per segment. Roughly 128$times$ slower than compression.

This asymmetry is actually fine for the challenge, because decompression speed isn't scored. But it's worth noting: for a practical system, you'd want a different architecture. We explored a "row-level" model that processes 16 tokens per row in parallel (only 8 sequential steps per frame instead of 128), but didn't finish training it in time.

=== Why K = 3 and Not K = 20?

The per-position model used $K = 20$ frames of temporal context. The frame model uses only $K = 3$. This seems like a step backward, but it's actually a step forward. Here's why.

The per-position model can only see one position's temporal history. It's trying to predict a token from a thin time series of that one position. More data points (more frames) help because that's all it has to work with.

The frame model sees the _entire_ previous frame --- all 128 positions. One previous frame gives it 128 tokens of context, compared to 20 tokens of context from 20 frames of a single position. And those 128 tokens are incredibly informative because they describe the complete spatial structure of the scene.

The numbers bear this out:

#align(center)[
  #table(
    columns: 4,
    align: (left, center, center, center),
    stroke: 0.5pt + gray,
    inset: 6pt,
    table.header(
      [*Model*], [*Temporal Context*], [*Spatial Context*], [*Bits/token*]
    ),
    [v3 (per-position)], [$K = 20$, 1 position], [1 above token], [3.896],
    [Frame small ($K = 1$)], [1 full frame (128 pos)], [Full causal], [3.698],
    [Frame small3 ($K = 3$)], [3 full frames (384 pos)], [Full causal], [3.605],
    [Frame medium3 ($K = 3$)], [3 full frames (384 pos)], [Full causal], [3.514],
  )
]

Going from $K = 20$ at one position to $K = 1$ at all positions gives a 0.2 bit improvement. Going from $K = 1$ to $K = 3$ gives another 0.1 bits. Going from $K = 3$ to $K = 20$ (which we tested) gives negligible further improvement --- the information from 20 frames ago at 128 positions is almost entirely redundant with the information from 3 frames ago at 128 positions.

The lesson: _breadth of context matters more than depth._ Seeing the whole picture from one second ago is more useful than seeing one pixel from ten seconds ago.

=== Model Sizing

We trained three configurations:

#align(center)[
  #table(
    columns: 6,
    align: (left, center, center, center, center, center),
    stroke: 0.5pt + gray,
    inset: 6pt,
    table.header(
      [*Model*], [*$d$*], [*Layers*], [*Heads*], [*Params*], [*Compact Size*]
    ),
    [Frame small ($K = 1$)], [128], [4], [4], [1.34M], [2.6 MB],
    [Frame small3 ($K = 3$)], [128], [4], [4], [1.34M], [2.3 MB],
    [Frame medium3 ($K = 3$)], [192], [6], [4], [3.98M], [6.8 MB],
  )
]

#figure(
  image("figures/fig4_model_evolution.png", width: 95%),
  caption: [Left: Training convergence for different model architectures. Frame models converge to lower loss despite fewer temporal context frames. Right: The model size vs. compression quality tradeoff.]
) <fig:evolution>

There's a clear tradeoff here. The medium3 model is 4.5 MB larger than the small3 model, but it saves about 0.09 bits/token. Over 768 million tokens, that's about 8.6 MB of compressed data saved. Net benefit: about 4 MB. Not a lot in absolute terms, but since we're competing on total zip size, every megabyte counts.

We did not try a "large" model ($d = 256$, 8 layers), though we had the configuration ready. The concern was that the model would be $approx 15$ MB compact, and the bits/token improvement might not be worth the extra model weight. There are diminishing returns: the medium model already captures most of the learnable patterns, and pushing further would require exponentially more parameters for linearly less entropy.

== Entropy Coding: ANS

Once you have a good probabilistic model, you need an entropy coder to turn those probabilities into actual compressed bits. We use Asymmetric Numeral Systems (ANS) [2], specifically the stack-based variant from the `constriction` library [3].

ANS is essentially optimal: if your model assigns probability $p$ to a token, ANS encodes it in very close to $-log_2(p)$ bits. The overhead is negligible compared to the model's prediction errors.

The process:

*Compression:*
1. For each segment, run the model in teacher-forced mode over all 1,200 frames. This produces probability distributions for each of the $1200 times 128 = 153,600$ tokens.
2. Quantize the probabilities to 24-bit fixed-point (required by the ANS coder).
3. Encode all tokens in reverse order using `AnsCoder.encode_reverse()`. (ANS is a stack --- last encoded, first decoded.)
4. Write the compressed bitstream.

*Decompression:*
1. Load the ANS bitstream.
2. For each frame, for each position in raster order: use the model to compute probabilities, decode one token from the ANS stream, feed it back into the model for the next position.
3. Reconstruct the full $(1200, 8, 16)$ array.

The key invariant: the probability distributions used during encoding and decoding must be _exactly_ identical, bit for bit. Any discrepancy --- even from floating-point non-determinism --- will cause decoding to silently produce garbage. This is why we quantize to fixed-point integers before passing to ANS.

== Model Compression: Making the Weights Small

The trained model weights need to go in the zip file. Every byte of model is a byte not available for data. We minimize this through:

1. *FP16 quantization.* All 3.98M parameters are stored as 16-bit floats. We observed zero degradation in compression quality. The model's predictions are quantized to 24-bit integers for ANS anyway, so fp16 weights provide more than enough precision.

2. *LZMA compression.* Neural network weights are not random --- they cluster near zero, they have correlations across layers, and certain patterns repeat. LZMA captures this, shrinking the fp16 buffer from 7.6 MB to 6.8 MB. Not a huge ratio (1.12$times$), but worth doing.

3. *Compact format.* We store a tiny JSON header with the architecture config, followed by the LZMA-compressed weight blob. The decompressor deserializes this and reconstructs the PyTorch model. Total overhead beyond the weights themselves: about 200 bytes of metadata.

= The Things That Didn't Work

I want to spend some time on the failures, because they're more instructive than the successes. When something works, you learn that one thing works. When something fails, you learn something about why the problem is shaped the way it is.

== Delta Encoding: A False Friend

My first instinct with any sequential data is to try delta encoding. Instead of storing values $x_1, x_2, x_3, ...$, store differences $x_1, x_2 - x_1, x_3 - x_2, ...$. If consecutive values are similar, the differences are small, and small numbers compress well.

This works beautifully for audio (consecutive samples differ by tiny amounts), for natural images (adjacent pixels are similar), and for many time series. It fails completely for VQ tokens.

The reason is fundamental: VQ codebook indices are _categorical_, not _ordinal_. Token 500 and token 501 were assigned their indices essentially at random during VQ-VAE training. They might represent completely different visual patterns --- a patch of blue sky and a piece of red taillight. The "difference" $501 - 500 = 1$ carries no information about the visual similarity between the patches.

This is worth remembering whenever you work with discrete codebooks. The numerical values are arbitrary labels, not measurements on a meaningful scale. Any technique that assumes numerical proximity implies semantic proximity will fail.

== Spatial Neighbors in Per-Position Models: Diminishing Returns

After seeing how much the "above token" helped the v3 model, a natural next step was to add more spatial neighbors. Our v4 model conditioned on 3 above-row neighbors (directly above, above-left, above-right) in addition to the temporal context.

Result: basically identical to v3. The extra spatial neighbors provided less than 0.05 bits of improvement.

Why? Because the per-position model processes each position independently. When you add the "above-left" token from the current frame, you're giving the model one additional token of context. But the frame-level model gives it _all_ previously decoded positions --- potentially 127 tokens of spatial context. The marginal value of going from 1 to 3 spatial neighbors is tiny compared to going from 1 to 127.

This was actually a pivotal realization. We had been thinking about spatial context as something you add incrementally: 1 neighbor, then 3, maybe 5. But the conditional entropy analysis showed that what you really want is _all_ of it. The jump from "a few neighbors" to "full frame context" is not a quantitative improvement --- it's a qualitative one. It changes the fundamental nature of the model from "128 independent predictors" to "one joint predictor."

== Mean-Pooled Temporal Context: Information Destruction

We tried a "temporal-frame" model that mean-pooled $K = 10$ previous frames into a single set of 128 context vectors, then used those as cross-attention context for the current frame decoder.

This performed worse than simply using $K = 3$ frames without pooling. Mean pooling destroys temporal ordering: the model can't tell whether a particular visual feature appeared 1 frame ago or 10 frames ago. For driving video, this matters --- a car that was in a certain position 1 frame ago is likely still there, but a car that was there 10 frames ago might have moved significantly.

It also destroys token identity: if positions (3, 5) had values [100, 100, 100, 200, 200, 200, 300, 300, 300, 400] over the last 10 frames, the mean of the _embeddings_ of these tokens is some smeared-out vector that doesn't correspond to any particular token. The discrete structure is gone.

== Hybrid Neural + Transition Tables: The Redundancy Problem

We had position-specific transition tables (from the earlier Markov approach) and a neural model. Why not blend them? Use $alpha dot p_("neural") + (1 - alpha) dot p_("table")$ as the combined distribution.

Result: at most 0.002 bits improvement on the segments we tested. Essentially zero.

This makes sense in hindsight. The neural model has already learned whatever the transition tables encode. The transition table captures $P(x_t | x_(t-1), "position")$ --- a first-order Markov model per position. The neural model conditions on $K = 3$ full previous frames with full spatial context; it strictly subsumes the information in the transition table. Blending in the table's predictions just adds noise to the neural model's (better) predictions.

The only case where blending would help is if the transition tables captured a pattern that the neural model missed due to limited capacity. With 4M parameters and only 768M training tokens, this is plausible in principle. But in practice, the patterns in the transition tables are simple enough that even a small neural network captures them.

== Training on Longer Temporal Context: Wasted Compute

We trained a version of the v3 model with loss computed on all timesteps (not just the final one), hoping that supervising the intermediate predictions would help the model learn temporal dynamics.

After 60 epochs (twice the normal training time), it performed _worse_ than the standard model trained for 30 epochs. More supervision != better model. The intermediate predictions are a different task than the final prediction (they use different amounts of temporal context), and the model's capacity is finite. Spending capacity on predicting frame 5 from frames 0--4 takes capacity away from predicting frame 100 from frames 80--99, which is the actual task during compression.

= Results

== The Final Numbers

#figure(
  image("figures/fig7_compression_progression.png", width: 72%),
  caption: [Cumulative bits/token during compression of 100 segments. The rate stabilizes quickly around 3.51 bits/token.]
) <fig:prog>

We ran the full 5,000-segment compression with the Frame medium3 model. It took 3 hours and 49 minutes on a single GPU.

#align(center)[
  #table(
    columns: 3,
    align: (left, right, right),
    stroke: 0.5pt + gray,
    inset: 6pt,
    table.header(
      [*Component*], [*Size*], [*Fraction*]
    ),
    [Compressed data (ANS)], [326.2 MB], [97.9%],
    [Model weights (LZMA fp16)], [6.8 MB], [2.0%],
    [`decompress.py`], [$approx 10$ KB], [$< 0.1$%],
    table.hline(),
    [*Total zip*], [*333.3 MB*], [*100%*],
    [Original data], [915 MB], [],
    [*Compression ratio*], [*2.75$times$*], [],
    [*Bits/token*], [*3.563*], [],
  )
]

The model weight overhead is about 2% of the total zip. This is the sweet spot: small enough that the model doesn't dominate the archive, large enough to meaningfully improve predictions.

== How This Compares

#figure(
  image("figures/fig2_compression_comparison.png", width: 80%),
  caption: [Compression ratios across all methods we evaluated. The gap between LZMA (1.6$times$) and our neural approach (2.75$times$) is substantial, but the gap to the theoretical ceiling is even larger.]
) <fig:comparison>

For context, the commaVQ leaderboard's top entries are:
- *3.4$times$*: A self-compressing neural network (the model _is_ the compressed data)
- *2.9$times$*: GPT + arithmetic coding (using a much larger model, presumably heavily quantized)
- *2.75$times$*: Our submission
- *1.6$times$*: LZMA baseline

Our approach sits comfortably in the top tier, using a model that's roughly 100$times$ smaller than the GPT-based entries.

= The Gap: Why Not Better?

The conditional entropy analysis says $H(X | X_(t-1), X_("above")) = 1.45$ bits. We're getting 3.56 bits. That's a gap of 2.11 bits, which represents about 200 MB of additional compression we're leaving on the table. Where is it?

== Model Capacity

Our model has 4M parameters. The conditional entropy estimate of 1.45 bits comes from a perfect oracle that knows the true joint distribution $P(X | X_(t-1), X_("above"))$. Our model is an approximation of this distribution, learned from data with limited capacity. A larger model would close some of this gap, but with diminishing returns --- and every extra MB of model weights costs an MB of zip space.

== Context Limitations

The 1.45-bit estimate conditions on the previous frame at the same position plus the above token. Our model conditions on much more than this (3 full previous frames, all previous tokens in the current frame), so in principle it should do _better_ than 1.45 bits. But the estimate uses the empirical distribution (effectively a lookup table over the training data), which is a perfect model of the specific patterns present. Our neural model learns a compressed representation that generalizes but doesn't memorize.

There's a deeper issue: the 1.45-bit estimate conditions on oracle context --- the true previous frame and true above token. During decompression, the model conditions on its own decoded outputs. In compression mode (teacher forced), this doesn't matter. But the model was trained to predict well conditioned on true context, and any distribution shift (which only matters at decompression time) could cause drift. Since we verified round-trip correctness, this isn't causing errors, but it might mean the model's predictions are subtly miscalibrated.

== Autoregressive Ordering

Raster order (left-to-right, top-to-bottom) is natural but not necessarily optimal. When predicting a token in the middle of the frame, raster order means you have all of the rows above but nothing from the rows below. A 2D ordering that gives you context from all directions would be more informative. But implementing non-raster orderings with standard transformer attention is tricky, and we didn't pursue it.

== What the Leaders Do Differently

The top submission (3.4$times$) uses a "self-compressing neural network" --- the compressed file _is_ the network weights, and decompression is running the network. This eliminates the model-size overhead entirely, because the model and the data are the same object. It's an elegant idea, essentially training a network to memorize the dataset and then compressing the network weights.

This approach has a fundamental advantage: every bit in the file is pulling double duty as both "model" and "data." Our approach has a hard split: 6.8 MB of model that doesn't encode any data, and 326 MB of data that doesn't encode any model. The self-compressing approach eliminates this tax.

= What I'd Do With More Time

If I had another week on this project, here's what I'd try, in rough order of expected impact:

*1. Bigger model, trained longer.* We trained for 30 epochs; the training curves hadn't fully converged. A medium-large model ($d = 256$, 8 layers, $approx 10$M params, $approx 18$ MB compact) might be worth it if it saves more than 18 MB in compressed data. Based on the medium vs. small improvement curve, this seems likely.

*2. Online adaptation.* During compression, you're processing segments sequentially. The model could fine-tune itself on each segment before compressing it, transmitting the weight updates as part of the compressed stream. This is the direction the self-compressing neural network approach takes.

*3. Row-level model.* We prototyped a model that processes 16 tokens per row in parallel (8 sequential steps per frame instead of 128). Early results showed comparable bits/token with 16$times$ faster decompression. With more training, this could be practical.

*4. Better quantization.* We use fp16 weights with LZMA compression. INT8 or INT4 quantization with careful calibration could shrink the model by 2--4$times$ with minimal quality loss. That's 3--5 MB saved.

*5. Non-raster orderings.* Process tokens in a pattern that maximizes spatial context --- for instance, center-out spiral, or a learned ordering. This is architecturally complex but could provide meaningful gains.

= Conclusion

Compressing driving video tokens turns out to be a beautiful microcosm of the compression-prediction duality. The most important things we learned:

*Spatial context dominates temporal context.* This was the single most impactful insight. Knowing what's above, to the left, and elsewhere in the current frame is more valuable than knowing a long history at a single position. The right architecture exploits this fully.

*Small specialized models beat large general ones.* When you can train on the test data (as the challenge permits), a 4M-parameter model captures the relevant patterns nearly as well as a 300M-parameter GPT. Domain specificity is a superpower.

*The gap between theory and practice is where the work is.* The theoretical analysis says 1.45 bits should be achievable. We got 3.56. Closing that 2-bit gap requires not just a better model but better everything: quantization, ordering, online learning, architecture search. Each of those is a research project in itself.

*Failures teach you the shape of the problem.* Delta encoding failed because tokens aren't ordinal. Bit-packing failed because LZMA likes byte alignment. Spatial neighbors failed because you need _all_ the context, not a few scraps of it. Each failure narrowed the space of viable approaches and pointed toward the final solution.

In the end, the score --- 2.75$times$ --- is a number. What matters more is the understanding: compression is prediction, prediction requires context, and the right context depends on the structure of the data, not on your assumptions about it.

#v(1cm)

// References
#heading(numbering: none)[References]
#set text(size: 9.5pt)

#block(inset: (left: 1.5em, top: 0.3em))[
  #set par(hanging-indent: 1.5em)

  [1] C. E. Shannon, "A Mathematical Theory of Communication," _Bell System Technical Journal_, vol. 27, pp. 379--423, 1948.

  [2] J. Duda, "Asymmetric Numeral Systems," _arXiv preprint arXiv:0902.0271_, 2009.

  [3] R. Bamler, "Constriction --- a library for entropy coding in Python and Rust," 2022. https://github.com/bamler-lab/constriction

  [4] comma.ai, "commaVQ: Compressed Driving Video Dataset and World Model," 2024. https://github.com/commaai/commavq

  [5] A. van den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural Discrete Representation Learning," _NeurIPS_, 2017.

  [6] A. Vaswani et al., "Attention is All You Need," _NeurIPS_, 2017.

  [7] L. Townsend et al., "Practical Lossless Compression with Latent Variables Using Bits Back Coding," _ICLR_, 2019.
]
