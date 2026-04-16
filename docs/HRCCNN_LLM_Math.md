# HRCCNN_LLM_Math — Character-Level Math LM on HRCCNN

## Goal

Train a DIM 12 HRCCNN to predict the next character of arithmetic expressions,
one character at a time. The model is shown `"5.00 * (3.00 + 4.50) = "` and
must emit `"37.50#"`, where `#` terminates the answer. Inference is
autoregressive: each predicted character is fed back as the reservoir's next
input until `#` appears (or a hard output-length cap triggers).

Strategic experiment, not just a demo. ESNs have not (to our knowledge) been
asked to perform bounded arithmetic as a character-stream task. The reservoir
isn't computing — the CNN readout is, conditioned on reservoir state. The
open question is whether 4096 neurons of fading-memory state plus a small
CNN head can encode enough about "which operation, which operands, which
partial answer so far" to produce the next digit correctly.

## Vocabulary (20 tokens)

```
0 1 2 3 4 5 6 7 8 9   digits     (10)
+ - * /               operators  (4)
( )                   grouping   (2)
=                     equals     (1)
.                     decimal    (1)
' '                   space      (1)
#                     EOS        (1)
```

The `-` token is dual-role: subtraction operator and unary sign for negative
numbers. Context disambiguates. Space is kept to preserve readability
(`"a + b = c"`, not `"a+b=c"`); worth revisiting if it wastes reservoir
cycles without earning its keep.

## Expression Grammar

- Nesting depth ≤ 2. **Depth definition by example:**
  - `a op b` → depth 1.
  - `(a op b) op c` → depth 2 (one level of parens).
  - `(c op (a op b)) op d` → depth 3 (two levels; out of scope).
- **LHS operand values in `[-100, 100]`** at ≤2dp resolution (per the
  canonical form: trailing zeros dropped).
- **RHS result is not range-clipped.** It inherits whatever magnitude
  the expression produces, printed in canonical form (≤2dp, trailing
  zeros dropped). Bounded only by the natural grammar ceiling: depth-2
  multiplicative expressions over LHS ∈ [-100, 100] top out at ~1e8,
  so RHS integer parts are at most 9 digits wide. Float32 headroom
  (~3.4e38) is astronomical relative to this; no float-level clipping
  needed.

EBNF sketch (LHS and RHS grammars are intentionally asymmetric):

```
line        := expr ' = ' result '#'
expr        := operand ( ' ' op ' ' operand )*
operand     := number_lhs | '(' expr ')'
number_lhs  := [ '-' ] digit{1,3} [ '.' digit{1,2} ]    // |v| ≤ 100
result      := [ '-' ] digit{1,9} [ '.' digit{1,2} ]    // unbounded magnitude
op          := '+' | '-' | '*' | '/'
```

Same canonical-form rule (drop trailing zeros, drop bare decimal)
applies to both `number_lhs` and `result`.

**Unary minus.** A leading `-` may prefix any `number` in any position
— after `=`, after `(`, after a binary operator, or at the start of a
bare expression. Parentheses are **not** required to wrap a negative
operand (e.g., `3.00 * -5.00 = -15.00#` is well-formed). The model is
expected to infer operator precedence from context; unary vs. binary
`-` is disambiguated by position, not grammar.

**Output-magnitude spread.** The grammar allows RHS magnitudes from
~0 (rounded from `0.01 * 0.01`) up to `(100 * 100) * (100 * 100) = 1e8`
— about 8 orders of magnitude. That full range would stress the
readout's magnitude encoding hard: variable-length integer parts up to
9 digits wide, with correct digit count. For v1 we narrow this to
`|RHS| ≤ 999` via the training-data filter (below), keeping integer
parts ≤ 3 digits while the grammar itself stays open-ended. Removing
the filter later tests generalization to larger magnitudes.

**Operand cap.** At depth ≤ 2 with strictly binary ops, the natural upper
bound is 4 atomic operands. That's fine; no explicit cap needed beyond
the depth constraint. Worst-case line length is roughly 40–50 characters.

**Generator distribution policy:**

- **Operator mix:** uniform across `+ - * /`. Do not inherit natural-text
  operator frequencies — they would starve `/` relative to `+`. Uniform
  keeps each operator class well-represented in the training target
  distribution.
- **Depth mix:** 50/50 between depth-1 (`a op b`) and depth-2. Within
  the depth-2 bucket, the three structural forms are sampled uniformly
  at 1/3 each:
  - `(a op b) op c` (left-paren)
  - `a op (b op c)` (right-paren)
  - `(a op b) op (c op d)` (both-paren, 4 atoms)

  Balanced sampling forces the model to handle all three structures;
  a model trained on depth-1-dominated data will fail at depth-2
  inference, and a model trained on only left-paren will fail on
  right- or both-paren forms.
- **Operand distribution:** uniform in `[-100, 100]` at ≤2-decimal
  resolution (20,001 discrete values per operand slot).
- **Training-data RHS filter (v1 only):** after rounding/canonicalizing,
  reject any expression with `|RHS| > 999`. This is a *training-set*
  filter, not a grammar constraint — the `result` production still
  permits up to 9 integer digits, and the model's architecture is
  unchanged. The filter keeps v1 evaluation focused on ≤ 3-digit
  integer parts, so decimal-misalignment and integer-digit-miscount
  failures happen within a bounded window. Removing the filter
  (Phase 3+ or Phase 5) is a distinct generalization test.
- **Rejection-rate consequence:** with the `|RHS| ≤ 999` filter in
  place, multiplicative depth-2 expressions reject far more often than
  additive ones (e.g., `a * b * c` with each in [-100, 100] exceeds
  999 for most operand triples). The *effective* operand distribution
  seen by the model at multiplicative positions is narrower than at
  additive positions. Watching for this bias is a Phase 4 diagnostic.
- **Division by zero:** generator rejects any expression whose grammar
  would force a `0.00` divisor at any `/` position. Rejection sampling.
- **Unary minus frequency:** any generated `number` carries a leading
  `-` with 50% probability (independent of position), subject to the
  `[-100.00, 100.00]` range constraint.

## Number Format & Division Policy

**Maximum 2 decimal places, trailing zeros dropped.** Numbers are
printed in a canonical form: at most 2 digits after the decimal
point, and trailing zeros (including the decimal point itself if the
fractional part collapses to zero) are dropped.

| Value (rounded to 2dp) | Canonical form |
|---|---|
| `5.00` | `5` |
| `5.50` | `5.5` |
| `5.25` | `5.25` |
| `-12.00` | `-12` |
| `-12.40` | `-12.4` |
| `0.00` | `0` |
| `-0.00` | `0` |

This is strictly more natural than the uniform-width 2dp format and a
genuinely harder formatting task: at each digit position the model has
to decide "emit another digit" vs. "end the number." That decision is
a cleaner test of whether the model is actually computing rather than
pattern-matching a fixed numeric width. If v1 cannot learn the stop
condition, falling back to uniform 2dp is a simplifying lever.

Division rarely produces a clean ≤2-decimal result. Three candidate
policies:

| Option | Meaning | Tradeoff |
|---|---|---|
| A | Round true result to 2 decimals via `std::round` (half-away-from-zero), then canonicalize | Target is approximate; model learns rounding + trailing-zero dropping |
| B | Reject expressions whose result isn't exactly ≤2-decimal | Aggressive filter; starves grammar diversity |
| C | Restrict divisor to clean values (integers, powers of 10) | Narrows grammar, cleanly preserves format |

**Default: Option A** with `std::round` (half-away-from-zero) followed
by canonicalization. Rounding is a legitimate "do the math" skill and
the model should learn it. Both the rounding and the canonicalization
are applied **only during training-dataset construction** — the model
just sees the canonical target string. Revisit if training stalls
specifically on division.

**Canonicalization algorithm** (pseudocode, applied to any numeric
value before stringifying):

```
canonicalize(v: double) -> string:
    # 1. Round to 2 decimals (half-away-from-zero).
    r = std::round(v * 100.0) / 100.0

    # 2. Collapse -0 and ±0.00 to "0".
    if r == 0.0:
        return "0"

    # 3. Split sign, integer part, and fractional cents.
    sign       = (r < 0) ? "-" : ""
    r_abs      = |r|
    int_part   = (int) floor(r_abs)
    frac_cents = (int) round((r_abs - int_part) * 100)   // 0..99

    # 4. Drop trailing zeros in the fractional part.
    if frac_cents == 0:
        return sign + to_string(int_part)            // e.g., "5"
    if frac_cents % 10 == 0:
        return sign + to_string(int_part) + "." +
               to_string(frac_cents / 10)            // e.g., "5.5"
    return sign + to_string(int_part) + "." +
           zero_pad_2(frac_cents)                    // e.g., "5.25"
```

Edge cases this handles correctly: `-0.00` → `"0"`, `5.004` → `"5"`
(rounds to 5.00, then strips), `5.005` → `"5.01"` (half-away-from-zero
at the 3rd decimal), `100.0` → `"100"`, `-99.9` → `"-99.9"`.

## Input Encoding

Each input character is fed to the reservoir as its 8-bit ASCII byte,
bipolar-encoded:

- `ReservoirConfig::num_inputs = 8`.
- Channel `b` (0 ≤ b ≤ 7) carries bit `b` of the current character's
  ASCII byte, **LSB = bit 0** (standard C convention `(c >> b) & 1`):
  `+1.0` if the bit is set, `-1.0` if cleared.
- `Reservoir`'s per-vertex channel routing (channel `k` owns vertices
  `k, k+K, 2K, ...`, `Reservoir.h:18`) gives each of the 8 channels
  **512 vertices at DIM 12** — a substantial random-projection slice
  per bit.

The LSB=bit-0 convention is load-bearing for the "low nibble of a
digit *is* its value" claim below — bits 0–3 of `'0'..'9'` = 0x30..0x39
carry the digit value 0..9 directly.

No character embedding layer; no vocab↔index table on the input side.
The reservoir's random per-vertex `W_in` *is* the embedding.

**Why 8-bit bipolar over 20-way one-hot:**

- **~2.5× more vertices per channel** at DIM 12 (512 vs. ≈ 205 for
  one-hot). More random-projection bandwidth per live signal.
- **Every channel drives every step at ±1** — all ~4096 `W_in` weights
  contribute at every timestep. One-hot leaves 19/20 of `W_in` idle
  per step, thinning reservoir drive.
- **Free semantic structure from ASCII layout** (vocab-specific):
  - Digits `'0'..'9'` = 0x30–0x39 — bit 4 set, low 4 bits = digit value 0–9.
  - `'='` = 0x3D — also bit 4 set.
  - Everything else (`+ - * / ( ) . ' ' #`) sits in 0x20–0x2F — bit 4 cleared.
  - ⇒ **bit 4 is effectively a "digit-or-equals" channel for free**, and
    the low nibble of a digit *is* its value. The reservoir does not
    have to learn from scratch that `'5'` and `'6'` are neighbors.

**Caveats:**

- Nothing in the vocab exceeds 0x3D, so **bits 6 and 7 are always -1**.
  Those channels act as a constant bias — harmless (the reservoir
  absorbs it) but wasted. Kept for encoding simplicity; revisit only
  if input bandwidth becomes a bottleneck.
- **Space (`' '` = 0x20) is the coldest input in the vocab** — only
  bit 5 is set; 7 channels drive at −1, 1 at +1. That's a feature,
  not a bug: spaces act as low-drive separators that let the reservoir
  briefly relax between tokens without resetting.
- ASCII Hamming structure is a *prior* the reservoir cannot see past.
  For our vocab it clusters mostly usefully (digits tight, symbols
  tight), but some semantically-paired operators are not Hamming-close
  (e.g., `'+'` 0x2B and `'-'` 0x2D differ in bit 1; `'*'` 0x2A and
  `'/'` 0x2F differ in bits 0 and 2). Accept the prior rather than
  engineer a custom binary code — the elegance of "just feed the
  ASCII byte" outweighs the minor structural imperfection.

**Output side is unchanged.** The classification target remains 20-way
softmax over the vocab. Bit-encoding only the input does not collapse
the output space.

## Reservoir

- **DIM 12** (N = 4096).
- **Spectral radius** 0.90, **input scaling** 0.02 — scale-invariant
  defaults for the hypercube topology; no per-size re-tune needed.
- **Output fraction** **0.125** (default; configurable knob). See
  "Output fraction & sub-hypercube subsampling" below.
- **Seed** — **single seed, all phases, no survey.** At DIM 12 both the
  reservoir-seed lottery and the CNN-init-seed lottery are nearly
  cosmetic (per `diagnostics/NARMA10.md`'s dispersion-collapse table —
  CV 1.5% at DIM 9, shrinking further at DIM 12). Pick any seed once
  and lock it in the driver. Multi-seed averaging is explicitly out of
  scope for this example.

Reservoir state steps once per input character. HCNN sees raw reservoir
state; no translation layer (HCNN always bypasses it).

### Output fraction & sub-hypercube subsampling

The reservoir runs at full DIM=12 (4096 vertices) for dynamics, but the
CNN readout operates on a **stride-selected sub-hypercube** of the
state. This is how feature dim is kept manageable without sacrificing
reservoir capacity.

HypercubeCNN's convolution uses XOR-neighbor masks, so the subsample
**must preserve hypercube structure** — equivalently, the stride must
be a power of 2. A stride of `2^k` takes vertices at indices
`0, 2^k, 2·2^k, 3·2^k, ...`, which form a valid sub-hypercube of
effective dimension `DIM − k`.

| output_fraction | stride | effective CNN DIM | N at CNN | FLATTEN feat dim (nl=1, ch=8) | Reduction |
|---|---|---|---|---|---|
| 1.0 | 1 | 12 | 4096 | 16,384 | 1× |
| 0.5 | 2 | 11 | 2048 | 8,192 | 2× |
| 0.25 | 4 | 10 | 1024 | 4,096 | 4× |
| **0.125 (default)** | **8** | **9** | **512** | **2,048** | **8×** |
| 0.0625 | 16 | 8 | 256 | 1,024 | 16× |

**Why 0.125 is the default:** lands feature dim in the ~2k range,
which is the sweet spot for our 5k-expression training budget (200k
positions) — head params ~41k, sample/param ratio ~5:1, well
out of memorization-risk territory. Up or down one power of 2 is a
cheap sweep if v1 underfits or overfits. Non-power-of-2 values are
**unsafe for HCNN** (they break the XOR-neighbor spatial assumption
inside HypercubeCNN); the driver will validate and refuse them.

**Why this is safe for reservoir capacity:** the 4096-vertex reservoir
still runs in full. The CNN just sees every 8th vertex's state. Since
per-vertex `W_in` is random and reservoir dynamics mix rapidly across
neighbors, the sampled 512 vertices still carry a rich compression of
the full-state signal — not 512/4096 of the information, but
considerably more than that ratio would suggest.

Historical note: ESN used to force `output_fraction = 1.0` when the
readout was HCNN. That forcing has been relaxed — HCNN callers are
now responsible for choosing a power-of-2-aligned subsample, and the
`HRCCNN_LLM_Math` driver drives `Reservoir + CNNReadout` directly
anyway (bypassing ESN's generic pipeline).

## Readout

`CNNReadout` in **multi-class classification** mode:

- **Output shape**: one linear layer projects the CNN backbone's
  flattened feature vector to **20 logits** → softmax → probability
  distribution over the vocab. Not 20 independent heads — one head
  with a 20-way softmax.
- **Training loss**: softmax + cross-entropy over class-index targets.
  Handled internally by `CNNReadout`'s multi-class mode — no readout-
  side code changes needed; the head shape is configuration.
- **Inference decode**: **argmax** over the 20 probabilities. Picks
  the single strongest token per step.

Starting architecture: `HRCCNNBaseline<12>()` backbone (**nl=1, ch=8**
per `readout/HCNNPresets.h`; FLATTEN head; lr=0.0015) with **training
overrides: ep=1000, bs=4096**. The CNN sees the stride-8 sub-hypercube
of the reservoir (effective DIM=9, N=512 at the CNN input) per the
output_fraction=0.125 default; see the Reservoir section's
subsampling table.

**Head: FLATTEN.** After one conv+pool pair operating on the
sub-hypercube: 256 surviving vertices × 8 channels = **2,048
features**. Linear head projecting 2048 → 20 logits = **~41k head
params** (plus ~a few hundred conv params, negligible in comparison).

Vertex identity is load-bearing on this task — the hypercube
reservoir's random per-vertex `W_in` means every (sampled) vertex
encodes a different linear combination of the input character's bits,
so `'5'` and `'6'` land very differently across the state. A
per-vertex FLATTEN head preserves that structure end-to-end. (The
sub-hypercube subsampling trades away 7/8 of the vertices but the
remaining 512 still carry diffuse character-bit information — any
pooling-over-vertices operation that destroys per-vertex identity
defeats the purpose of a hypercube reservoir for this task.)

**Sample/parameter ratio: ~5:1.** 200k training positions
(5k expressions × ~40 chars) against ~41k head params. Healthy
regime — well out of both memorization-risk (would be ~1:1 or lower)
and capacity-starvation (would be 100:1 or more). Standard weight-
decay regularization on the head is still a sensible default for
noise tolerance, but early stopping on validation loss should be
sufficient.

**Gradient-update budget.** With 200k training positions, bs=4096, and
ep=1000: **~49k gradient updates**, essentially 1× the NARMA
calibration — not much headroom but not starved. If character-level
accuracy keeps climbing at end-of-training without overfitting, raise
the epoch count.

Expect further retuning if FLATTEN underfits the arithmetic semantics
despite the parameter budget — the natural knobs are (in order):
bumping `output_fraction` to 0.25 or higher (more vertices → more
features), widening `ch`, or deepening `nl`.

**Decode knobs deferred until v1 works:**

- **Temperature sampling** (instead of argmax): useful diagnostic if
  the model gets stuck on a wrong early-RHS token and can't recover.
  Not needed until we have a working model to debug.
- **Top-k / nucleus *sampling*** (as a generation strategy): irrelevant
  for this task — we want one correct answer per expression, not
  generation diversity. Not to be confused with the next item.
- **Softmax distribution *inspection*** (as a diagnostic): genuinely
  useful — surfaces per-step top-N contenders, confidence margins,
  and entropy of the predictive distribution. Distinct from sampling:
  the model still emits argmax, but we *log* the full distribution
  for offline analysis. Tracked as a Phase 4 item.

## Training Regime

**Per-expression reset + LHS priming.** Each expression is an
independent training unit. Before feeding an expression's data:

1. `reservoir.Reset()` (zero the state — library semantics settled as
   Option 1, nothing else).
2. Prime: feed the LHS through the reservoir **twice**, discarding all
   states. Each priming pass runs only the hypercube update — the CNN
   readout is not invoked. This washes out the zero-init transient
   (at SR=0.90 the zero-state contribution decays to ~4% after ~30
   character steps, so two LHS passes lands the reservoir firmly in
   echo-state territory before the collecting pass begins).
3. Collecting pass: feed the full expression (LHS + RHS), collecting
   `(state, next-char target)` pairs at every position for teacher-
   forced training.

Priming is applied identically at inference time so that train-state
≡ inference-state when predictions are correct (standard teacher-
forcing guarantee). The protocol belongs in the `HRCCNN_LLM_Math`
driver, not in the Reservoir class — `Reset()` stays minimal,
priming is the caller's job.

`#` is a pure EOS marker with no dual purpose. This aligns training
exactly with the inference regime (the user presents one equation;
we reset, prime, generate).

```
sample: "5 + 3 = 8#"                           # depth-1, integer operands
sample: "-2.5 * 4 = -10#"                      # depth-1, mixed widths, unary minus
sample: "(1.5 + 2.5) * 3 = 12#"                # depth-2, paren on left
sample: "5 * (2 - -3) = 25#"                   # depth-2, unary minus inside parens
sample: "10 / -4 = -2.5#"                      # depth-1, rounded division to 1dp
sample: "(2.5 + 7.5) / 3 = 3.33#"              # depth-2, rounded division to 2dp
sample: "0.25 * 0.4 = 0.1#"                    # small operands, canonicalized result
...
```

- Before each expression: `reservoir.Reset()` (zero the state), then
  two priming passes over the LHS (no readout, no state collection).
- Collecting pass: feed the full expression character by character.
- Teacher forcing: at position `t`, input = char(t), target = char(t+1).
  Every position contributes a training example.
- **State indexing convention:** the reservoir state the readout sees
  when predicting char(t+1) is the state **after consuming char(t)**
  (i.e., after `InjectInput` + `Step`). This is the standard ESN
  teacher-forcing convention; stating it explicitly so nobody writes
  an off-by-one collection loop. For an expression of `L` characters,
  a collecting pass yields `L−1` (state, target) pairs: positions 0
  through L−2, with char(L−1) = `#` as the final target.
- Target training-set size: **5k independent expressions** for the v1
  first pass — ~3.2 GB of reservoir states at DIM 12, fast iteration.
  Scale up (10k → 25k → larger) as results warrant.
- Validation: independently-generated 2k-expression set (sized to
  match the v1 training scale; scales up alongside training).
- **State buffer storage:** `float32` (matches the rest of the
  library's conventions; the 3.2 GB figure assumes this). `float16`
  is a lever available later if training-set upscaling ever hits the
  64 GB ceiling.

**Implications of the reset:**

- **Batching is now trivial.** Each expression is an independent (state
  trajectory, target trajectory) pair. Batched training over N
  expressions in parallel is straightforward — previously the
  single-stream model made batching awkward.
- **`#` is now only the EOS character.** It no longer doubles as a
  context-reset marker, so the model's `#` prediction is a pure
  "answer is complete" signal, not "new context begins."
- **No cross-expression carryover.** The reservoir never sees a
  previous expression's residual state, so training samples are IID.
- **Transient washout is handled by priming**, not accepted. See
  the LHS-priming protocol above. Priming is free (reservoir-only,
  no readout), so there is no reason to tolerate weak-state positions.

**Should LHS positions contribute loss?** At deployment the model only
emits RHS characters; LHS is always given. Two views:

- Train on everything: every position is a free language-model
  pretext signal — the model learns numeric format, operator grammar,
  `=` position, etc. "for free."
- Mask LHS: training loss matches the deployment objective.

**Default: train on everything.** The LHS pretext should make formatting
solid before computation even kicks in.

**Class-imbalance.** Digits dominate target positions (probably >60%);
operators, parens, `=`, `.`, and `#` together are the tail. Watch for
the head collapsing to always-predict-digit. If it does, inverse-
frequency loss reweighting in the softmax head is the first lever.
With canonical (trailing-zeros-dropped) numbers, the `.` character is
scarcer than under a uniform-2dp format (only present when a number
has a fractional part), so it may need extra weighting if
decimal-placement errors dominate.

## Inference

```
1. reservoir.Reset()                          # zero the state
2. Prime: feed LHS twice (reservoir-only, no readout).
3. Final LHS pass: feed "5.00 + 3.00 = " one char at a time.
4. Generation loop:
     p ← softmax(CNNReadout(state))
     c ← argmax(p)                            # argmax first; sampling later
     emit c
     if c == '#': break
     state ← reservoir.step(bipolar_bits(c))  # 8-dim ±1 from ASCII byte
5. Concatenated emitted chars = predicted RHS.
```

Hard output-length cap of ~12 chars prevents runaway generation when
`#` is never emitted.

Training and inference reservoir dynamics are identical: reset, feed,
(optionally) generate. The only difference is the input source during
the generation phase — teacher-forced ground truth vs. the model's
own argmax.

## Success Metrics

- **Exact-match accuracy**: fraction of held-out expressions where the
  full generated RHS (through `#`) equals ground truth. The real metric.
- **Character-level accuracy**: per-position argmax-matches-target rate.
  Climbs earlier and smoother than exact-match; the training dashboard.
- **Format accuracy**: fraction of emitted RHS strings that parse as a
  valid canonical number followed by `#` — i.e., `[-]?\d{1,3}(\.\d{1,2})?#`
  with no trailing zeros in the fractional part and no bare trailing `.`.
  Separates "learned numeric format" from "learned computation." Format
  should saturate first.

Expected ordering during training: format ≫ character ≫ exact-match.
If format accuracy plateaus below 90%, the model is undersized or the
head is under-tuned — diagnose before scaling.

## Anticipated Failure Modes

- **Long-context decay**: by the time the model emits the last digit of
  the answer, it may have forgotten the early operands. ESN memory
  fades geometrically. At ~30-char lag with DIM 12, this is plausibly
  near the edge. Mitigation: DIM 14–16 if DIM 12 caps out.
- **Operator confusion**: `/` is the rarest and hardest; expect
  operator-stratified accuracy to show a `/ << +` gap. Data generator
  should enforce a uniform operator distribution, not natural frequency.
- **Decimal misalignment / integer-digit miscount**: emitting
  `"1.05"` for true `"10.5"`, `"0.5"` for true `"5"`, or `"99"` for
  true `"999"`. Indicates weak magnitude encoding in the reservoir's
  projection of operand digits. Under the v1 training filter
  (`|RHS| ≤ 999`) this is bounded to at most 3 integer digits; when
  the filter is removed later (Phase 3+ or 5), integer-digit miscount
  becomes a first-class failure mode with RHS parts up to 9 digits
  wide.
- **Premature-stop / non-stop on number emission**: under the canonical
  format the model must decide, at each digit position, whether to
  emit another digit or terminate the number. Failure mode: always
  emitting exactly 2 decimal digits (reversion to uniform format) or
  stopping after the integer part when fractional digits were needed.
- **EOS over/under-prediction**: model emits `#` too early (truncated
  answer) or never (runaway). Class reweighting on `#` loss addresses
  both directions.

## Phased Delivery

**Phase 1 — scaffold (no HRC).**
- Expression generator: grammar (with the distribution policy above),
  division rounding + canonicalization, division-by-zero rejection.
  No RHS value clipping — the result inherits whatever magnitude the
  grammar produces.
- 8-bit bipolar **input** encoder (char → 8-dim ±1 vector for the
  reservoir).
- Vocab ↔ class-index tables for the **output** side (20-way target
  labels during training + argmax decode at inference).
- **Verification acceptance criterion**: generate 1k samples; for each
  line `L`, parse the LHS with an independent shunting-yard evaluator,
  round-to-2dp per the division policy, and compare to the printed
  RHS. Any mismatch is a fail-fast generator bug. Phase 1 ships when
  this passes at 100%.

**Phase 2 — DIM 12 v1.**
- DIM 12, full vocab, full grammar, FLATTEN head (HRCCNNBaseline<12>
  nl=1/ch=8 with the training overrides from the Readout section).
- **5k independent training expressions**, 2k validation.
- Target: format accuracy > 90% and any non-trivial exact-match. The
  goal is to confirm the approach converges at all, not to maximize.
- If format accuracy saturates well below 90% or character accuracy
  plateaus near chance (~5%, i.e. 1/20), the approach has a structural
  problem — diagnose (Phase 4 tools) before pushing more data at it.
- If v1 looks promising but capped, **upscale training to 10k** within
  this phase and rerun.

**Phase 3 — further scale-up (only if Phase 2 warrants).**
- Same config as Phase 2; push training to 25k → 50k expressions.
- Goal: drive exact-match toward the Phase 5 threshold (~80%).
- Track all three metrics (format, character, exact-match) across
  epochs.

**Phase 4 — diagnostics & tuning.**
- **Entry criterion**: format accuracy has saturated above 90% and
  either character accuracy has plateaued over 5 consecutive
  checkpoints, or exact-match is climbing but operator/magnitude bias
  is suspected. Don't enter Phase 4 to chase a still-improving loss.
- Operator-stratified accuracy (where does it fail?).
- Operand-magnitude-stratified accuracy (does performance degrade
  for `|x| > 50`?).
- **Softmax distribution logging** during held-out inference: for
  each RHS position, capture top-N contenders, confidence margin
  (`p_argmax - p_second`), and distribution entropy. Surfaces where
  the model is confident-and-wrong vs. uncertain-and-wrong — very
  different failure modes that call for different fixes.
- **`output_fraction` sweep** (0.25 → 0.125 → 0.0625): measures how
  much reservoir signal the CNN can actually exploit. Going up one
  power of 2 doubles feature dim and doubles head params; going down
  halves both. If v1 accuracy keeps climbing as we expose more of the
  reservoir, that's a sign we're in capacity-starvation; if accuracy
  is flat across the sweep, 0.125 is fine and feature dim isn't the
  bottleneck.
- Head architecture variations if FLATTEN underfits (e.g., FLATTEN +
  hidden layer between conv features and softmax).
- `nl` / `ch` sweep only if the output_fraction sweep and head
  variations don't help. Single seed throughout — no seed survey.

**Phase 5 — scale out / generalize if needed.**
- If DIM 12 caps below, say, 80% exact-match despite Phase 4, try
  DIM 14 (N=16k, ~4× training cost). DIM 16 is the hard ceiling
  (library-supported).
- **Generalization test**: remove the `|RHS| ≤ 999` training filter
  and evaluate whether the Phase 3 model still holds up on the full
  grammar-permitted RHS range (~1e8). Expect a drop; the magnitude
  of the drop measures how much of v1's success was within-bounds
  pattern-matching vs. genuine multi-digit integer-emission skill.

## Deliverable Structure

**Location:** `examples/HRCCNN_LLM_Math/`. Single binary
`HRCCNN_LLM_Math.exe` with subcommand-style invocation — this is the
first of what may become several *LLM-style* example tasks (math is
the quasi-LLM proving ground; text / code / other symbolic streams
could follow if the approach pans out).

**Subcommands:**

```
HRCCNN_LLM_Math train --output <model.bin> [--samples N] [--seed S]
    Generate training data on the fly, run the full reset/prime/collect/
    train loop, and serialize the resulting model to <model.bin>.

HRCCNN_LLM_Math eval --model <model.bin> [--samples N]
    Load a serialized model; generate a fresh held-out set; report
    format / character / exact-match accuracy.

HRCCNN_LLM_Math infer --model <model.bin> --input "<LHS string>"
    Load a serialized model; run the inference protocol (reset, prime
    LHS twice, argmax-generate RHS) on a single user-supplied LHS.
    Prints the emitted RHS.
```

`main.cpp` is thin — it parses the subcommand, dispatches to one of
three top-level functions (`RunTrain`, `RunEval`, `RunInfer`), each
of which lives in its own `.cpp` alongside `main.cpp`. The expression
generator, encoder/decoder tables, and canonical formatter are their
own compilation units.

## Serialization

**Critical requirement**, not an afterthought. Training at DIM 12
with 5k+ expressions is expected to be slow enough that no experiment
should be run twice. Every trained model persists to disk at the end
of `train`, and every subsequent `eval` / `infer` invocation loads
from disk.

**On-disk format** (binary, little-endian, single file):

```
struct ModelFile {
    // Header
    char     magic[8];          // "HCNNLLMM"
    uint32_t format_version;    // 1 for v1
    uint32_t dim;               // must match the compile-time DIM
                                // the binary was built against

    // Reproducibility metadata (informational; not required to
    // reconstruct the model, but helps with audit)
    uint64_t training_seed;
    uint32_t training_samples;
    uint32_t training_epochs;
    char     git_sha[40];       // git HEAD at training time

    // Configs — verbatim struct copies, so any schema change bumps
    // format_version
    ReservoirConfig  reservoir_cfg;
    CNNReadoutConfig cnn_cfg;

    // CNN weights — flat float32 array in the same layout as
    // CNNReadout::GetWeights() returns. Size is determined by cnn_cfg.
    uint64_t weights_count;
    float    weights[weights_count];
};
```

**Load protocol:**

1. Read header; verify magic, format_version, and `dim` matches the
   binary's compile-time DIM (fail loudly if not — DIM is a template
   parameter, not a runtime choice).
2. Reconstruct `Reservoir<DIM>` from `reservoir_cfg` (same seed →
   same recurrent weights + W_in).
3. Reconstruct `CNNReadout<DIM>` from `cnn_cfg` (empty weights).
4. Call `CNNReadout::SetWeights(weights)` to restore the trained head.
5. Model is ready for `eval` or `infer`.

**Design notes:**

- Reservoir weights are *not* serialized — they're fully determined
  by `reservoir_cfg` (seed + hyperparameters) and are recomputed at
  load time. This is tiny in storage but costs an `Initialize()` call
  on load (~100ms at DIM 12, fine).
- CNN weights are the only training product that needs to persist.
  At the default config (output_fraction=0.125, nl=1/ch=8, FLATTEN),
  the head is 2048 × 20 + 20 ≈ **41k floats** plus a few hundred
  conv params — roughly **165 KB** per checkpoint. Trivial in disk
  terms. A Phase 4 output_fraction=0.25 sweep would double this to
  ~330 KB, still trivial; output_fraction=1.0 (no subsampling) would
  land at ~1.3 MB.
- Saving the full training config inside the file (not just the
  weights) means a single `.bin` is fully self-contained for `infer`
  — you don't need to remember what hyperparameters a given checkpoint
  used.
- `git_sha` is captured at train time from `git rev-parse HEAD` via a
  helper. Makes it trivial to correlate a saved checkpoint with the
  exact source it was trained under.

## Open Decisions

- Keep or drop space from the vocab. Default keep; revisit at phase 3.
- Per-class loss weighting (uniform vs inverse-frequency). Default
  uniform; switch to inverse-frequency if operators/EOS under-train.
- Reservoir `Reset()` semantics — see task #1.
- v1 training-set size given 64 GB RAM budget — see task #2.
