# HRCCNN_LM_Text — Character-Level Text Language Model

> **STATUS: PAUSED.** Empirical sweeps (documented in
> `docs/ReservoirMemoryBottleneck.md`) show that all tested configurations
> converge to BPC ~3.05 regardless of reservoir size, readout capacity, or
> training budget. The memory depth intrinsic to the hypercube reservoir is
> too shallow for language modeling — the ~20-40 character echo state
> horizon caps performance at bigram-level prediction. This is a
> fundamental property of the fixed-weight reservoir dynamics, not a tuning
> failure. Work on this example is paused until the memory bottleneck is
> addressed. The next area of exploration is **reservoir cascades**, which
> extend effective memory by chaining reservoirs operating at different
> timescales.

## Goal

Train a DIM 13 HRCCNN to predict the next character of natural English
text, one character at a time. The model is shown a continuous stream of
text (initial corpus: Tiny Shakespeare, ~1.1 MB) and must predict
`char(t+1)` given the reservoir state after consuming `char(t)`.
Inference is autoregressive: each predicted character is fed back as the
reservoir's next input.

This is an ESN language model, not a "large" language model. The
reservoir is 8192 neurons with a small CNN readout head. The question
is whether hypercube reservoir dynamics plus a learned convolutional
readout can capture enough sequential structure in natural text to
produce coherent character-level predictions — bigrams, trigrams,
word boundaries, and common word completions — without any explicit
attention mechanism or gradient-based sequence training.

## Vocabulary (96 tokens)

The vocabulary is **fixed at full printable ASCII plus newline** — not
derived from any single corpus. This makes trained models portable
across corpora without vocab mismatches.

```
Category          Chars                                  Count   Byte range
─────────────────────────────────────────────────────────────────────────────
Newline           \n                                       1     0x0A
Printable ASCII   SP ! " # $ % & ' ( ) * + , - . /       95    0x20–0x7E
                  0 1 2 3 4 5 6 7 8 9 : ; < = > ?
                  @ A B C D E F G H I J K L M N O
                  P Q R S T U V W X Y Z [ \ ] ^ _
                  ` a b c d e f g h i j k l m n o
                  p q r s t u v w x y z { | } ~
                                                        ────
                                                          96
```

This covers:

- **10 digits** (0–9) — essential for general-purpose text (dates,
  numbers, code, addresses, technical prose).
- **26 uppercase + 26 lowercase** letters.
- **32 punctuation/symbol characters** — full coverage of standard
  keyboard symbols including `#`, `@`, `{`, `}`, `<`, `>`, etc.
- **Space** and **newline** as whitespace.

Notable design choices:

- **Tab (`\t`, 0x09) is excluded.** Tabs are inconsistently used across
  corpora (sometimes 2 spaces, sometimes 4, sometimes mixed). For v1,
  corpora should be pretabbed or tab-stripped. Adding tab is a trivial
  expansion (97 tokens) if code corpora need it later.
- **Carriage return (`\r`, 0x0D) is excluded.** Corpora should use Unix
  line endings. `\r` in Windows-style `\r\n` pairs would waste a vocab
  slot on a purely mechanical artifact.
- **No high-ASCII or UTF-8.** Bytes 0x80–0xFF are outside scope. Corpora
  with Unicode should be preprocessed (transliterate accented characters,
  strip or replace non-ASCII). Expanding to full byte (256 tokens) is a
  future option but quadruples the output layer without clear benefit
  for English text.
- **Corpus chars outside the fixed vocab are errors.** The loader
  rejects any byte not in the 96-token set rather than silently
  dropping or replacing it. This is intentional — it forces corpus
  preprocessing to be explicit rather than hiding data loss.

### Class imbalance

With a fixed 96-token vocab, any given corpus will leave some tokens
with zero or near-zero frequency. On Tiny Shakespeare, the top 10
characters account for 62.5% of all positions:

| Rank | Char | Frequency |
|------|------|-----------|
| 1    | SP   | 15.2%     |
| 2    | e    | 8.5%      |
| 3    | t    | 6.0%      |
| 4    | o    | 5.9%      |
| 5    | a    | 5.0%      |
| 6    | h    | 4.6%      |
| 7    | s    | 4.5%      |
| 8    | r    | 4.4%      |
| 9    | n    | 4.4%      |
| 10   | i    | 4.1%      |

31 of the 96 vocab tokens never appear in Tiny Shakespeare (all
digits except `3`, plus `"`, `#`, `%`, `(`, `)`, `*`, `+`, `/`,
`<`, `=`, `>`, `@`, `[`, `\`, `]`, `^`, `_`, `` ` ``, `{`, `|`,
`}`, `~`). These unused classes have zero training signal — the
model will emit them only by mistake. This is expected and correct:
the fixed vocab exists so a model trained on Shakespeare can later
be evaluated on or fine-tuned with a corpus that does use those
tokens, without a vocab mismatch.

A naive majority-class baseline predicting space every step would
score 15.2% accuracy. The model must beat this trivially; the real
bar is beating bigram/trigram baselines (~30–40% character accuracy
for English text).

Uniform loss weighting is the default. If rare punctuation or
uppercase characters show systematically poor accuracy in Phase 4
diagnostics, inverse-frequency loss reweighting is the first lever.

## Text Grammar

There is no formal grammar to engineer.  Natural English text has
implicit structure at multiple scales:

- **Character level**: letter frequency, digram/trigram statistics.
  English has strong character-level redundancy (`th`, `he`, `in`,
  `er`, `an` are dominant bigrams).
- **Word level**: common words (`the`, `and`, `to`, `of`, `I`) are
  highly predictable once a prefix is seen. After `th`, `e` is
  overwhelmingly likely.
- **Line structure**: Tiny Shakespeare has a regular format:
  ```
  CHARACTER NAME:\n
  Dialogue line 1.\n
  Dialogue line 2.\n
  \n
  NEXT CHARACTER:\n
  ```
  Character names appear in ALL CAPS followed by `:`, which creates
  a strong positional signal. The model should learn that `\n` followed
  by an uppercase letter likely begins a character name.
- **Verse structure**: iambic pentameter creates rhythmic patterns in
  line length and stress. Whether the reservoir can encode anything
  about this is an open question — it would require ~40-character
  memory (one line of verse), which is at the edge of DIM 13's echo
  state horizon.

The model does not need to learn any of these structures explicitly.
It sees a stream of characters and predicts the next one. Any
structure it captures is emergent from the reservoir dynamics and
CNN readout.

## Input Encoding

Each input character is its 8-bit ASCII byte, bipolar-encoded.

- `ReservoirConfig::num_inputs = 8`.
- Channel `b` (0 <= b <= 7) carries bit `b` of the current character's
  ASCII byte, **LSB = bit 0**: `+1.0` if set, `-1.0` if cleared.
- At DIM 13, each of the 8 channels owns **1024 vertices** via
  `Reservoir`'s modular channel routing.

### ASCII bit structure for the text vocabulary

The 96-token vocab spans bytes 0x0A and 0x20–0x7E. Unlike the math
vocab (which clustered in 0x20–0x3D), the text vocab uses the full
printable ASCII range and the bit patterns carry richer semantic
structure:

```
Bit 7: ALWAYS -1 (all chars < 0x80). Dead channel — constant bias.

Bit 6: +1 for letters (0x41–0x5A, 0x61–0x7A) and symbols 0x60–0x7E,
        -1 for newline, space, digits, and punctuation (0x0A, 0x20–0x3F).
        => Roughly "letter vs. non-letter" but not perfectly clean
        (backtick, braces, pipe, tilde leak into the +1 group).
        Still a useful prior — the dominant +1 population is letters.

Bit 5: +1 for 0x20–0x3F (space, digits, punctuation) and
        0x60–0x7E (lowercase, backtick, braces, etc.),
        -1 for 0x0A (newline) and 0x40–0x5F (uppercase, @, [, \, etc.).
        Combined with bit 6:
          bit6=+1, bit5=+1 → lowercase + `` ` { | } ~ ``  (0x60–0x7E)
          bit6=+1, bit5=-1 → uppercase + @ [ \ ] ^ _      (0x40–0x5F)
          bit6=-1, bit5=+1 → digits + space + punctuation  (0x20–0x3F)
          bit6=-1, bit5=-1 → newline only                  (0x0A)
        => Case discrimination is a single-bit flip (bit 5). FREE.
        The four quadrants cleanly separate the major character classes.

Bits 4–0: Within each group, bits 4–0 encode the letter identity.
          Crucially, uppercase and lowercase versions of the same
          letter share identical bits 4–0 (e.g., 'A'=0x41 and
          'a'=0x61 differ ONLY in bit 5).
          => The reservoir sees case as a one-channel perturbation
          on top of shared letter identity. This is an extremely
          favorable prior for English text, where case rarely
          changes the next-character prediction.
```

**Key differences from the math vocab encoding:**

| Property              | Math (20 tokens)           | Text (96 tokens)                         |
|-----------------------|----------------------------|------------------------------------------|
| Byte range            | 0x20–0x3D                  | 0x0A, 0x20–0x7E                          |
| Dead channels         | bits 6, 7 always -1        | bit 7 always -1                          |
| Free discriminators   | bit 4 = "digit or equals"  | bit 6 ~ letter/non-letter, bit 5 = case  |
| Hamming distance range| 1–4 bits                   | 1–7 bits                                 |
| Semantic clustering   | digits tight (0x30–0x39)   | upper/lower share bits 0–4; digits tight |

**Encoding advantages for text:**

- **7 live channels** (vs. 6 for math). Bit 6 is now informative,
  giving the reservoir more input bandwidth per timestep.
- **Case-insensitive letter identity in 5 bits.** The reservoir
  "knows" that `T` and `t` are the same letter (bits 0–4) with a
  case flag (bit 5). This is exactly the prior English text needs.
- **Digits cluster tightly** in 0x30–0x39: bit 5 and bit 4 both set,
  bits 0–3 carry the digit value 0–9 directly. Same structure the
  math model exploited. General-purpose corpora with numbers get
  this digit-identity prior for free.
- **Punctuation and digits cluster in 0x20–0x3F** with bit 6 = -1.
  The reservoir can learn a "non-letter incoming" signal from a
  single channel.
- **Newline is unique:** the only character with bit 6 = -1 AND
  bit 5 = -1. It is maximally distinguishable from all other tokens.

**Encoding disadvantages:**

- **Hamming spread is wider.** Math tokens spanned 5 bits of
  variation; text tokens span 7. Two characters can differ in up to
  7 channels simultaneously (e.g., `\n` 0x0A vs `u` 0x75), making
  the reservoir's input-mixing problem harder.
- **Space is cold.** `' '` = 0x20: only bit 5 set, 7 channels at -1.
  Same as math — spaces are low-drive separators.
- **Cross-group confusables.** Characters that differ only in bit 5
  or bit 6 are Hamming-close despite being semantically unrelated
  (e.g., `0` 0x30 and `P` 0x50 differ only in bit 6; `A` 0x41 and
  `a` 0x61 differ only in bit 5). The reservoir's random `W_in`
  must separate these from context. For the case pair this is a
  feature; for digit/letter collisions it's a minor liability.

### Why not a learned embedding?

The reservoir's random per-vertex
`W_in` *is* the embedding. Each of 1024 vertices per channel sees a
different random projection of the 8-bit input, creating an 8192-dim
distributed representation without any trainable embedding layer.
Adding a learned embedding would require backpropagation through
the reservoir, which violates the ESN paradigm.

The ASCII encoding carries a strong character-identity prior for
free (case-insensitive letter identity in 5 bits, case in 1 bit,
letter/non-letter in 1 bit). A custom binary code might cluster
semantically-similar characters more tightly (e.g., all vowels
within Hamming distance 1), but the engineering cost outweighs the
benefit — especially since the reservoir's random projections will
scatter any input code into a high-dimensional representation anyway.

## Output Encoding

**96-way softmax classification** over the fixed vocabulary. Each
position's target is the class index of the next character.

The vocab string is the 96-token fixed set, sorted by ascending byte
value, and embedded in the model file at save time. Class index =
position in the vocab string:

- Class 0 = `\n` (0x0A)
- Class 1 = ` ` (0x20)
- Class 2 = `!` (0x21)
- Class 3 = `"` (0x22)
- ...
- Class 17 = `0` (0x30)
- Class 18 = `1` (0x31)
- ...
- Class 95 = `~` (0x7E)

The fixed vocab is identical across all models, so class indices are
portable. Eval and infer reload the vocab from the model file for
consistency verification.

## Reservoir

- **DIM 13** (N = 8192).
- **Spectral radius** 0.90, **input scaling** 0.02 — scale-invariant
  defaults.  SR=0.95 was tested and performed worse (val top-1 dropped
  from 37% to 33% at DIM 12).
- **Output fraction** **0.5**.  The CNN sees 4096 of 8192 vertices
  (effective CNN DIM = 12).  This gives the CNN the same spatial
  resolution as a full DIM 12 reservoir while the extra 4096 hidden
  vertices enrich the reservoir's internal dynamics.

| output_fraction | eff. CNN DIM | N at CNN | FLATTEN feats (nl=1, ch=4) | Head params (96 classes) |
|-----------------|-------------|----------|---------------------------|--------------------------|
| 1.0             | 13          | 8192     | 16,384                    | 1.57M                   |
| **0.5 (current)** | **12**   | **4096** | **8,192**                 | **786k**                |
| 0.25            | 11          | 2048     | 4,096                     | 393k                    |
| 0.125           | 10          | 1024     | 2,048                     | 197k                    |

**Why 0.5:** experiments at DIM 12 showed that increasing spatial
resolution (more vertices visible to the CNN) was the strongest
performance lever — stronger than deeper CNN heads or higher SR.
At 900k training positions the sample/param ratio is 900k/786k ≈
1.15:1.  This is tighter than the math model's 4.6:1 but the CNN
extracts the useful signal in a single streaming pass, limiting
overfitting.

- **Seed** — single seed, derived from `gen_seed` via golden-ratio
  XOR. Seed lottery variance is cosmetic at DIM 13.

Reservoir state steps once per input character.  The CNN readout
sees the subsampled live state directly; no states buffer.

## Readout

`Readout` in **multi-class classification** mode:

- **Output shape**: CNN backbone's feature vector projects to
  **96 logits** -> softmax -> probability distribution over vocab.
- **Training loss**: softmax + cross-entropy.
- **Inference decode**: temperature-controlled sampling (default
  temperature = 0.8). Argmax (temperature = 0) available for
  deterministic evaluation. Temperature > 0 produces more diverse
  and readable autoregressive samples.

Architecture: `Baseline<13>()` backbone with **nl=1, ch=4,
FLATTEN head**.

- **nl = 1, ch = 4** (one conv+pool layer, 4 channels). After one
  pool from effective DIM 12 → DIM 11: 2048 vertices × 4 channels =
  **8,192 features**. Linear head: 8,192 × 96 + 96 = **~786k params**.
  The lean 4-channel head keeps the parameter count manageable while
  the full DIM 12 spatial resolution provides rich per-vertex signal.
- **FLATTEN readout head.** The hypercube reservoir's per-vertex
  `W_in` means vertex identity carries character information. Each of
  the 4096 visible vertices encodes a different random projection of
  the 8-bit input — which vertices are active tells you which
  character was seen. GAP averages this away. FLATTEN preserves it
  end-to-end.
- **LR**: linear decay from `lr_max=0.0015` to `lr_max * lr_floor_frac`
  across all mini-batch updates (all passes, no reset).  Adam optimizer
  for stable online updates.

### Eval diagnostics

After streaming through the training region, the reservoir continues
into the val region.  Each val character is predicted via
`PredictLiveRaw()` and metrics are accumulated:

- **Val top-1, top-3, top-5 accuracy** — per-character prediction
  from the live reservoir state across 100k val positions.
- **Val BPC (bits per character)** — cross-entropy in bits. The
  standard metric for char-level LMs. Baselines: ~6.6 BPC for uniform
  random over 96 classes, ~1.5 BPC for a simple RNN, ~1.2 BPC for a
  well-tuned LSTM.
- **Worst N classes by accuracy** — surfaces characters the model
  consistently mispredicts.
- **Autoregressive text samples** — 3 samples from the val region,
  each with a 64-char prompt and 200 chars of generated text at the
  configured temperature.

## Training Regime

**Streaming online training.** The reservoir is driven through the
corpus one character at a time.  After each character, the CNN readout
receives a single gradient update on the live reservoir state.  No
states are stored — RAM usage is constant regardless of corpus size.

Layout in the corpus:

```
[── warmup ──][── warmup_train ──][──────── train_chars ────────][── val_chars ──][+1]
     64             32,768                  900,000                   100,000
```

- **Warmup** (64 chars): drives the reservoir without training.
  Washes out the zero-init transient.  At SR = 0.90, the zero-state
  contribution decays to < 0.1% after 64 steps.
- **Warmup-train** (32,768 chars): `esn.InitOnline()` drives the
  reservoir through these positions, collects states transiently to
  compute input standardization (per-vertex mean and 1/std), builds
  the CNN architecture, and frees the states buffer.
- **Train stream** (900,000 chars × num_passes): for each character,
  the reservoir advances one step and mini-batch gradient updates are
  applied every `mini_batch_size` steps.  The target is the next
  character's class index.  LR decays linearly from `lr_max` to
  `lr_max * lr_floor_frac` across all mini-batch updates (all passes
  combined, no reset at pass boundaries).
- **Val stream** (100,000 chars): the reservoir continues into the val
  region.  Each character is predicted via `esn.PredictLiveRaw()` and
  metrics (top-k accuracy, BPC, per-class confusion) are accumulated.
  No weight updates.  The reservoir's state at the train/val boundary
  carries context from the training region, matching what would happen
  in production.
- **+1 trailing character**: needed so the last val position has a
  target.

**Multiple passes.** The corpus can be re-fed through the reservoir
for additional training passes.  The reservoir does not reset between
passes — it continues from its current state, so each pass sees the
data in a different dynamic context.

**Total corpus usage**: ~1,033k characters out of 1,115,394 (92.6%).
The remaining ~82k chars are untouched and available for future
held-out evaluation with `RunEval`.

**Memory budget**: steady-state RAM is under 50 MiB (reservoir state +
CNN weights + optimizer).  A transient ~1.5 GiB peak occurs during
`InitOnline` (32k states for standardization), freed immediately after.
No states buffer.  DIM 14, 15, 16 are all feasible.

**Teacher forcing.** At position `t`, the reservoir has consumed
`char(t)` and the target is `char(t+1)`.  Standard next-character
prediction — every position contributes one online gradient update.

**No per-expression reset.** The reservoir runs continuously.  The
spectral radius (0.90) governs the slowest decay mode: 0.9^t gives
~35% at lag 10, ~4% at lag 30, ~0.5% at lag 50.  The true effective
context window should be empirically measured via
memory capacity measurement (total R² over lags 1–50).  Rough
estimate: **20–40 characters** at DIM 13 / SR 0.90, covering word
boundaries and short phrases.  Paragraph-level structure is out of
reach.  Increasing DIM further (14, 15) is straightforward with
streaming training — no RAM constraint.

### Implications for the task (pending MC measurement)

- **Word completion** (context ~5 chars) is well within estimated
  echo state range.  The model should learn common word continuations.
- **Character-name prediction** after `\n` + uppercase (context ~15
  chars for a typical name) is probably within range.
- **Verse structure / rhyme** (context ~40–80 chars for a couplet)
  is likely beyond the memory horizon.  Don't expect this.
- **Cross-paragraph coherence** is impossible.

## Inference

```
1. Construct ESN<13> from saved ReservoirConfig.
2. Bootstrap CNN readout (dummy Run + Train with epochs=0).
3. Restore weights via SetReadoutState(FromSerial<13>(mf.readout)).
4. ResetAndPrime(esn, prompt):
     a. esn.ResetReservoirOnly()
     b. Encode prompt to bipolar bits.
     c. esn.Warmup(prompt_bits, prompt.size())
5. Generation loop:
     esn.PredictLiveRaw(logits)       // 96 floats from live state
     if temperature == 0:
       c = argmax(logits)
     else:
       probs = softmax(logits / temperature)
       c = sample(probs)
     emit c
     esn.Warmup(bipolar_bits(c), 1)   // step with emitted char
6. Repeat for num_chars steps.
```

No hard output-length cap needed (unlike math, where `#` terminates).
The caller specifies `num_chars` (default 500).

Temperature controls generation diversity:
- **0.0**: greedy argmax. Deterministic but repetitive.
- **0.5**: moderate diversity. Good for evaluating coherence.
- **0.8** (default): natural-sounding diversity.
- **1.0+**: increasingly random. Useful for probing what the model
  "knows" vs. what it's uncertain about.

## Success Metrics

- **BPC (bits per character)**: the primary metric. Lower is better.
  Teacher-forced cross-entropy in bits over the val set.

  | Model | BPC (approx.) |
  |-------|---------------|
  | Uniform random (96 classes) | 6.6 |
  | Unigram frequency baseline | 4.2 |
  | Bigram model | ~3.0 |
  | Trigram model | ~2.5 |
  | Simple RNN (Karpathy) | ~1.5 |
  | LSTM (Karpathy) | ~1.2 |

  Realistic target for HRCCNN at DIM 13: **2.5–3.5 BPC** (trigram to
  bigram level). Breaking below 2.5 would be a strong result. Breaking
  below 2.0 would be remarkable for a non-gradient-trained sequence
  model.

- **Top-1 character accuracy**: fraction of val positions where the
  argmax prediction matches the target. Easier to interpret than BPC
  for quick sanity checks. Chance = 1/96 ~ 1.0%. Majority-class
  baseline (always predict space) = 15.2% on Tiny Shakespeare.

- **Top-3 / Top-5 accuracy**: how often the correct character is among
  the top 3 or 5 predictions. Surfaces whether the model is "close"
  when wrong. A model with 35% top-1 but 70% top-5 is learning useful
  structure — it just can't commit to the right answer.

- **Autoregressive sample quality**: subjective but informative. At
  BPC ~3.0 expect recognizable English words with occasional nonsense.
  At BPC ~2.0 expect mostly-grammatical short phrases.

## Anticipated Failure Modes

- **Vocabulary collapse**: model predicts only the top 5–10 characters
  (space, `e`, `t`, `o`, `a`, ...) and ignores the rest. With 96
  classes, 31 of which may have zero training frequency on a given
  corpus, collapse is more likely than with the math model's dense
  20-token vocab. Symptom: top-1 accuracy plateaus at ~30% with
  near-zero accuracy on punctuation and uppercase. Fix:
  inverse-frequency class weighting.

- **Bigram ceiling**: model learns strong bigram statistics (`th` ->
  `e`, `qu` -> vowel) but cannot break past bigram-level BPC (~3.0).
  This would mean the reservoir's fading memory isn't providing useful
  context beyond 2 characters. Fixes (in order): increase
  `output_fraction` to expose more reservoir state; multi-char input
  injection (see Phase 4); DIM 14 (see Phase 5).

- **Case confusion**: uppercase and lowercase versions of the same
  letter differ by one bit (bit 5). The model may conflate them,
  predicting lowercase where uppercase is needed (e.g., after `\n`
  for character names). The per-class confusion diagnostic will
  catch this. Fix: case-specific class weighting, or a custom
  input encoding that separates case more aggressively (but this
  sacrifices the "same letter = same bits 0–4" advantage).

- **Punctuation scarcity**: `,` `;` `!` `?` `.` together are ~3.5%
  of the corpus. The model may under-predict punctuation, leading
  to run-on text in autoregressive samples. Visible in per-class
  accuracy and in sample quality.

- **Newline prediction**: `\n` is 3.6% of the corpus but carries
  critical structural meaning (verse/line boundaries). If the model
  either never predicts `\n` (run-on blocks) or over-predicts it
  (fragmented output), it hasn't learned line structure.

- **Repetitive loops**: common in autoregressive char-level models.
  The model emits a common word fragment (e.g., "the the the...")
  because each emitted character reinforces the same prediction.
  Temperature sampling mitigates this somewhat; beam search would
  help more but is out of scope.

## Phased Delivery

**Phase 1 — scaffold.** (complete)
- Corpus loader, fixed 96-token vocab, bipolar encoder.
- Binary serialization with embedded vocab.
- Config-driven train/eval/infer dispatch.
- Autoregressive generation with temperature sampling.

**Phase 2 — DIM 12 batch baseline.** (complete)
- Batch training at DIM 12 with various configs.  Best result:
  nl=1/ch=4/of=1.0, val top-1 = 44.5%, BPC = 2.89 at epoch 10.
  Model plateaus immediately and overfits beyond epoch 10.
- Established that SR=0.90 is optimal (SR=0.95 hurt performance).
- Established that FLATTEN > GAP and more spatial resolution is the
  strongest lever.

**Phase 3 — streaming training + DIM 13.** (complete)
- Streaming online training: one CNN gradient step per character,
  no states buffer.  Enables DIM 13+ with negligible RAM.
- DIM 13, of=0.5: 8192-neuron reservoir, CNN sees 4096 vertices.
- Target: match or exceed DIM 12 batch baseline (44.5% / 2.89 BPC)
  with a richer reservoir.

**Phase 4 — diagnostics & tuning.**
- Entry criterion: BPC has plateaued.
- **Memory capacity measurement**: measure empirical R² vs lag
  at DIM 13 / SR 0.90.
- DIM sweep (14, 15) — streaming makes higher DIM trivial.
- nl/ch sweep if readout capacity is the bottleneck.
- **Multi-char input injection.** If BPC plateaus at bigram level
  (~3.0), inject char(t-1) alongside char(t) as 16 bipolar inputs.
  Gives the reservoir clean bigram signal without relying on echo
  memory.  Cost: halves vertices per channel (512 at DIM 13).
- **Positional encoding.** Concatenate vertex address bits as extra
  CNN input channels.  Gives the conv layers position awareness
  before the FLATTEN head.

**Phase 5 — push limits.**
- DIM 14+ (N = 16,384+) — straightforward with streaming, no RAM
  constraint.
- Alternative / larger corpora (modern English, code, multilingual).
  Larger corpora are the primary scaling lever once Tiny Shakespeare
  is exhausted.
- Multiple corpus passes to see if additional training signal exists
  beyond the first pass.

## Deliverable Structure

**Location:** `examples/HRCCNN_LM_Text/`. Single binary
`HRCCNN_LM_Text.exe` with mode selected by `config::kMode` in
`Config.h`.

**Modes:**

- **Train**: load corpus, stream through reservoir with online CNN
  updates, evaluate on val region, save model.
- **Eval**: load saved model + corpus, score teacher-forced accuracy
  and BPC on a held-out region.
- **Infer**: load saved model (no corpus needed — vocab from model
  file), autoregressively continue a prompt.

**Files:**

```
examples/HRCCNN_LM_Text/
  Config.h          Config structs (TrainCfg, EvalCfg, InferCfg)
  Corpus.h/.cpp     Corpus loading, fixed vocab, bipolar encoding
  Dataset.h/.cpp    ResetAndPrime, GenerateText (temperature sampling)
  Serialization.h/.cpp  Binary model file format (magic HCNNLMTX, v1)
  RunTrain.cpp      Streaming online train loop + val eval
  RunEval.cpp       Teacher-forced eval on held-out region
  RunInfer.cpp      Autoregressive inference from prompt
  main.cpp          Mode dispatch
```

## Serialization

- **Magic**: `HCNNLMTX` (8 bytes).
- **Vocab**: stored as a length-prefixed string (u32 length + chars).
  The vocab is the fixed 96-token set, embedded in the model file for
  forward-compatibility verification — eval/infer confirm the loaded
  vocab matches the expected fixed set.

Format: format version, DIM, training metadata (seed, positions, epochs,
git SHA), ReservoirConfig and ReadoutConfig as POD blobs, then the
readout state (weights blob, bias, feature mean/scale vectors).
| Training positions | ~200k (5k expr x 40 chars) | 900k (streamed) |
| Primary metric | exact-match accuracy | BPC |
| Output fraction | 0.125 (eff. DIM 9) | 0.5 (eff. DIM 12) |
| CNN config | nl=1, ch=8, FLATTEN | nl=1, ch=4, FLATTEN |
| Temperature sampling | no (argmax only) | yes (default 0.8) |
| Peak RAM | ~18 GiB | < 50 MiB (steady-state) |

## Open Decisions

- **DIM scaling**: DIM 13 is the current config.  Streaming training
  makes DIM 14+ trivial in terms of RAM.  Higher DIM gives the
  reservoir more internal dynamics at the cost of slower per-step
  computation.
- **output_fraction**: 0.5 balances spatial resolution against head
  params.  If the CNN needs more signal, try 1.0 with ch=2 to keep
  the param budget constant.
- **Class weighting**: uniform is the default.  Inverse-frequency if
  rare characters show near-zero accuracy.
- **Corpus expansion**: Tiny Shakespeare is ~1M chars.  Concatenating
  additional texts (modern English, code, multilingual) is a Phase 5
  option.
- **Number of passes**: the training loop supports multiple passes
  over the corpus.  The reservoir continues without reset between
  passes, so each pass sees identical text in a different dynamic
  context.  The number of passes is a tunable (`num_passes` in
  `TrainCfg`).
