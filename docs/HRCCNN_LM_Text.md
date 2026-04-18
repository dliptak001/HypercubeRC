# HRCCNN_LM_Text — Character-Level Text LM on HRCCNN

## Goal

Train a DIM 12 HRCCNN to predict the next character of natural English
text, one character at a time. The model is shown a continuous stream of
text (initial corpus: Tiny Shakespeare, ~1.1 MB) and must predict
`char(t+1)` given the reservoir state after consuming `char(t)`.
Inference is autoregressive: each predicted character is fed back as the
reservoir's next input.

This is an ESN language model, not a "large" language model. The
reservoir is 4096 neurons with a small CNN readout head. The question
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

Unlike HRCCNN_LLM_Math, there is no formal grammar to engineer.
Natural English text has implicit structure at multiple scales:

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
  memory (one line of verse), which is at the edge of DIM 12's echo
  state horizon.

The model does not need to learn any of these structures explicitly.
It sees a stream of characters and predicts the next one. Any
structure it captures is emergent from the reservoir dynamics and
CNN readout.

## Input Encoding

Identical to HRCCNN_LLM_Math: each input character is its 8-bit ASCII
byte, bipolar-encoded.

- `ReservoirConfig::num_inputs = 8`.
- Channel `b` (0 <= b <= 7) carries bit `b` of the current character's
  ASCII byte, **LSB = bit 0**: `+1.0` if set, `-1.0` if cleared.
- At DIM 12, each of the 8 channels owns **512 vertices** via
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

Same reasoning as HRCCNN_LLM_Math: the reservoir's random per-vertex
`W_in` *is* the embedding. Each of 512 vertices per channel sees a
different random projection of the 8-bit input, creating a 4096-dim
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

- **DIM 12** (N = 4096).
- **Spectral radius** 0.90, **input scaling** 0.02 — scale-invariant
  defaults.
- **Output fraction** **0.125** (default). Same as the math model.
  Gives effective CNN DIM = 9 (512 vertices), landing the FLATTEN
  head at 2,048 features — proven territory from the math model.

| output_fraction | stride | eff. CNN DIM | N at CNN | FLATTEN feats (nl=1, ch=8) | Head params (96 classes) |
|-----------------|--------|-------------|----------|---------------------------|--------------------------|
| 1.0             | 1      | 12          | 4096     | 16,384                    | 1.57M                   |
| 0.5             | 2      | 11          | 2048     | 8,192                     | 786k                    |
| 0.25            | 4      | 10          | 1024     | 4,096                     | 393k                    |
| **0.125 (default)** | **8** | **9**  | **512**  | **2,048**                 | **197k**                |
| 0.0625          | 16     | 8           | 256      | 1,024                     | 98k                     |

**Why 0.125:** at 900k training positions, the sample/param ratio is
~4.6:1. Comfortable regime — well out of memorization risk. Going up
to 0.25 would double head params to 393k (ratio 2.3:1) — still
workable. Going down to 0.0625 halves to 98k (ratio 9.2:1) — very
safe but may underfit. The math model proved 0.125 at 200k positions
with 20 classes; 96 classes have a larger head but the 900k training
set compensates.

- **Seed** — single seed, derived from `gen_seed` via golden-ratio
  XOR. Same reasoning as math: at DIM 12, seed lottery variance is
  cosmetic.

Reservoir state steps once per input character. HCNN sees raw state;
no translation layer.

## Readout

`CNNReadout` in **multi-class classification** mode:

- **Output shape**: CNN backbone's feature vector projects to
  **96 logits** -> softmax -> probability distribution over vocab.
- **Training loss**: softmax + cross-entropy.
- **Inference decode**: temperature-controlled sampling (default
  temperature = 0.8). Argmax (temperature = 0) available for
  deterministic evaluation. Temperature > 0 produces more diverse
  and readable autoregressive samples.

Starting architecture: `HRCCNNBaseline<12>()` backbone — **nl=1,
ch=8, FLATTEN head** — identical to the math model.

- **nl = 1, ch = 8** (one conv+pool layer, 8 channels). After one
  pool from effective DIM 9 → DIM 8: 256 vertices × 8 channels =
  **2,048 features**. Linear head: 2,048 × 96 + 96 = **~197k params**.
  Same architecture that worked for math; the 96-class head is 5×
  larger than math's 20-class head, but the feature extractor is
  identical.
- **FLATTEN readout head.** Same reasoning as math: the hypercube
  reservoir's per-vertex `W_in` means vertex identity carries
  character information. Each of the 512 sampled vertices encodes a
  different random projection of the 8-bit input — which vertices
  are active tells you which character was seen. GAP averages this
  away. FLATTEN preserves it end-to-end.
- **bs = 4096, ep = 100** (reduced from math's 1000 because the
  streaming corpus provides 900k positions — far more training data
  per epoch than 5k expressions × ~40 chars).
- **lr_decay_epochs = 200** (cosine decay horizon extends past
  actual training, keeping the learning rate higher for longer).

### Eval diagnostics

Each eval hook (every 25 epochs by default) reports:

- **Train top-1 accuracy** — cheap, via `esn.Accuracy()`.
- **Val top-1, top-3, top-5 accuracy** — expensive per-sample
  `PredictRaw`, but only on the 100k val positions.
- **Val BPC (bits per character)** — cross-entropy in bits. The
  standard metric for char-level LMs. Baselines: ~6.6 BPC for uniform
  random over 96 classes, ~1.5 BPC for a simple RNN, ~1.2 BPC for a
  well-tuned LSTM.
- **Worst N classes by accuracy** — surfaces characters the model
  consistently mispredicts.
- **Autoregressive text samples** — 3 samples from the val region,
  each with a 64-char prompt and 200 chars of generated text at the
  configured temperature.

Early stopping is available via `eval_patience` (disabled by default).
When enabled, training halts if val top-1 has not improved for N
consecutive eval checkpoints.

## Training Regime

**Streaming single-pass.** Unlike the math model (which resets the
reservoir per expression), the text model drives the reservoir through
one continuous span of the corpus. This is the natural regime for a
language model: the reservoir's fading memory integrates context from
all preceding characters, exactly as it would at inference time.

Layout in the corpus:

```
[── warmup ──][──────────── train_chars ──────────────][── val_chars ──][+1]
     64                   900,000                         100,000
```

- **Warmup** (64 chars): drives the reservoir without collecting states.
  Washes out the zero-init transient. At SR = 0.90, the zero-state
  contribution decays to < 0.1% after 64 steps.
- **Train positions** (900,000): `esn.Run()` collects one state per
  character. Each state's target is the next character's class index.
- **Val positions** (100,000): collected in the same `esn.Run()` call,
  contiguous with training positions. The reservoir's state at the
  train/val boundary carries context from the training region, matching
  what would happen in production — no artificial state reset.
  **Val is only scored every 25 epochs** (controlled by
  `eval_every_epochs`), not after every epoch — the val states sit
  idle between eval hooks and add no per-epoch cost.
- **+1 trailing character**: needed so the last val position has a
  target.

**Total corpus usage**: 1,000,065 characters out of 1,115,394 (89.7%).
Uses nearly the full corpus. The remaining ~115k chars are untouched
and available for future held-out evaluation with `RunEval`.

**Memory budget**: 1M positions x 4096 floats x 4 bytes = 15.6 GiB
for the states buffer. CNN training creates a standardized copy of
the subsampled states (512 floats/position at output_fraction=0.125)
= ~2 GiB, plus transient buffers. Peak RAM ~18 GiB. Requires 32 GiB
system RAM.

**Teacher forcing.** At position `t`, the reservoir has consumed
`char(t)` and the target is `char(t+1)`. Standard next-character
prediction — every position contributes a training example. No
masking of any region.

**No per-expression reset.** The reservoir runs continuously. The
spectral radius (0.90) governs the slowest decay mode: 0.9^t gives
~35% at lag 10, ~4% at lag 30, ~0.5% at lag 50. But this is an upper
bound — most eigenvalues are smaller, and the tanh nonlinearity
typically reduces effective memory below the linear prediction. The
true effective context window should be empirically measured via
`diagnostics/MemoryCapacity.h` (total R² over lags 1–50) before
drawing conclusions. Rough estimate: **20–40 characters** at DIM 12 /
SR 0.90, covering word boundaries and short phrases. Paragraph-level
structure is out of reach. If MC measurement shows this is
insufficient, DIM 14 (4× neurons, 4× RAM per position) is a Phase 5
experiment — contingent on a library optimization to store only
subsampled states, which would break the current RAM deadlock
(DIM 14 × 900k positions = 56 GiB states alone).

### Implications for the task (pending MC measurement)

- **Word completion** (context ~5 chars) is well within estimated
  echo state range. The model should learn common word continuations.
- **Character-name prediction** after `\n` + uppercase (context ~15
  chars for a typical name) is probably within range.
- **Verse structure / rhyme** (context ~40–80 chars for a couplet)
  is likely beyond the memory horizon. Don't expect this.
- **Cross-paragraph coherence** is impossible.

## Inference

```
1. Construct ESN<12> from saved ReservoirConfig.
2. Bootstrap CNN readout (dummy Run + Train with epochs=0).
3. Restore weights via SetReadoutState(FromSerial<12>(mf.readout)).
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

  Realistic target for HRCCNN at DIM 12: **2.5–3.5 BPC** (trigram to
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

**Phase 1 — scaffold.**
- Corpus loader, fixed 96-token vocab, bipolar encoder.
- Binary serialization with embedded vocab.
- Config-driven train/eval/infer dispatch.
- Streaming reservoir drive (warmup + continuous Run).
- Autoregressive generation with temperature sampling.
- **Acceptance criterion**: builds clean, config is sane, can load
  corpus and print vocab/size. No training yet.

**Phase 2 — DIM 12 v1.**
- Default config: DIM 12, nl=1/ch=8/FLATTEN, 900k train / 100k val,
  output_fraction=0.125, 100 epochs, bs=4096.
- Target: top-1 accuracy > 25% (beating majority-class baseline by
  a clear margin) and BPC < 4.0 (beating unigram baseline).
- Eval hooks report top-k accuracy, BPC, per-class confusion, and
  autoregressive samples every 25 epochs.
- If BPC plateaus above 4.0, the approach has a structural problem.

**Phase 3 — scale & iterate.**
- Increase epochs if not overfitting (900k positions already uses
  ~90% of Tiny Shakespeare).
- Try larger/different corpora to increase training data.
- Target: BPC < 3.0 (bigram level or better).

**Phase 4 — diagnostics & tuning.**
- Entry criterion: BPC has plateaued over 3+ eval checkpoints.
- **Memory capacity measurement**: run `diagnostics/MemoryCapacity.h`
  at DIM 12 / SR 0.90 to get empirical R² vs lag. Establishes the
  true context window before chasing tuning levers.
- output_fraction sweep (0.25, 0.5, 1.0).
- nl/ch sweep (nl=1..3, ch=8..32).
- GAP vs FLATTEN comparison.
- Per-position accuracy analysis (does accuracy vary with position
  in a word? in a line?).
- Context window probe: predict characters that require N-gram
  context of increasing length; measure where accuracy drops off.
- **Multi-char input injection.** If BPC plateaus at bigram level
  (~3.0), the reservoir's recurrent memory may not be carrying
  char(t-1) at sufficient fidelity. Injecting char(t-1) alongside
  char(t) as a second 8-bit bipolar input (`num_inputs=16`, 256
  vertices/channel at DIM 12) gives the reservoir a clean,
  undistorted copy of the previous character. The recurrent signal
  (lossy, context-mixed) and the direct injection (pristine) are
  complementary — one carries history, the other carries identity.
  Cost: halves vertices per channel. Three chars (`num_inputs=24`,
  ~170 vertices/channel) is possible but starts thinning the
  random-projection bandwidth. Only pursue if v1 establishes that
  temporal context — not readout capacity or training data — is the
  bottleneck.

**Phase 5 — push limits.**
- DIM 14 (N = 16,384) if DIM 12 caps below BPC 2.5. Contingent on a
  library optimization to store only subsampled states per position
  (currently the full N-vector is cached), which would break the RAM
  deadlock at higher DIM.
- Alternative / larger corpora (modern English, code, multilingual).
  Larger corpora are the primary scaling lever once Tiny Shakespeare
  is exhausted.

## Deliverable Structure

**Location:** `examples/HRCCNN_LM_Text/`. Single binary
`HRCCNN_LM_Text.exe` with mode selected by `config::kMode` in
`Config.h`.

**Modes:**

- **Train**: load corpus, drive reservoir, train CNN readout, save
  model with periodic checkpoints.
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
  RunTrain.cpp      Streaming train loop + eval hooks + checkpoints
  RunEval.cpp       Teacher-forced eval on held-out region
  RunInfer.cpp      Autoregressive inference from prompt
  main.cpp          Mode dispatch
```

## Serialization

Same structure as HRCCNN_LLM_Math with two differences:

- **Magic**: `HCNNLMTX` (8 bytes) instead of `HCNNLLMM`.
- **Vocab**: stored as a length-prefixed string (u32 length + chars).
  The vocab is the fixed 96-token set, embedded in the model file for
  forward-compatibility verification — eval/infer confirm the loaded
  vocab matches the expected fixed set.

The rest of the format is identical: format version, DIM, training
metadata (seed, chunks, epochs, git SHA), ReservoirConfig and
CNNReadoutConfig as POD blobs, then the readout state (weights blob,
bias, feature mean/scale vectors).

## Comparison with HRCCNN_LLM_Math

| Aspect | Math | Text |
|--------|------|------|
| Vocab size | 20 | 96 (fixed printable ASCII + newline) |
| Vocab source | hardcoded | fixed (not corpus-derived) |
| Token semantics | structured (digits, ops, EOS) | natural (letters, punctuation) |
| Input encoding | 8-bit bipolar (6 live channels) | 8-bit bipolar (7 live channels) |
| Corpus | generated on the fly | static file (Tiny Shakespeare) |
| Reservoir regime | per-expression reset + priming | continuous streaming |
| Context needed | full expression (~40 chars) | local (~20–40 chars estimated) |
| Training positions | ~200k (5k expr x 40 chars) | 900k (contiguous) |
| Primary metric | exact-match accuracy | BPC |
| Output fraction | 0.125 (eff. DIM 9) | 0.125 (eff. DIM 9) |
| CNN config | nl=1, ch=8, FLATTEN | nl=1, ch=8, FLATTEN |
| Temperature sampling | no (argmax only) | yes (default 0.8) |
| Early stopping | no | yes (optional) |

## Open Decisions

- **output_fraction**: 0.125 is the starting point (same as math).
  If overfitting: increase training data before reducing fraction.
  If underfitting: try 0.25 (doubles features and head params).
- **GAP vs FLATTEN**: FLATTEN is default for vertex-identity
  preservation. If overfitting persists after increasing training
  data, GAP is a fallback that cuts head params from ~197k to ~1k.
- **Class weighting**: uniform is default. Inverse-frequency if rare
  characters show near-zero accuracy.
- **Corpus expansion**: Tiny Shakespeare is ~1M chars. If the model
  shows capacity for more data, concatenating additional texts (e.g.,
  other Shakespeare plays, or mixing in modern English) is a Phase 5
  option.
- **Multi-scale input**: feeding both the raw byte and a derived
  "character class" (lowercase, uppercase, punctuation, whitespace)
  as a second input channel. Deferred unless the single-byte encoding
  proves insufficient.
