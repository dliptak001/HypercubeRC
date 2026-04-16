# math_llm — Character-Level Math LM on HRCCNN

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

- Nesting depth ≤ 2 (at most one level of parentheses).
- Operand values in `[-100.00, 100.00]`, always printed with exactly 2
  decimal places.
- Answer printed with exactly 2 decimal places followed by `#`.

EBNF sketch:

```
line    := expr ' = ' number '#'
expr    := operand ( ' ' op ' ' operand )*
operand := number | '(' expr ')'
number  := [ '-' ] digit{1,3} '.' digit digit
op      := '+' | '-' | '*' | '/'
```

**Value-range risk.** Depth-2 multiplication permits results like
`100.00 * 100.00 * 100.00 = 1e6` — blows the 2-decimal output format and
smears the target distribution over an enormous range. Mitigation:
**value-clipped generation** — reject any expression whose true result
falls outside `[-999.99, 999.99]`. Training data is generated on the fly,
so rejection sampling is nearly free.

**Operand cap.** At depth ≤ 2 with strictly binary ops, the natural upper
bound is 4 atomic operands. That's fine; no explicit cap needed beyond
the depth constraint. Worst-case line length is roughly 40–50 characters.

## Number Format & Division Policy

All numbers printed with exactly 2 decimal places. `5` becomes `"5.00"`,
never `"5"` or `"5.0"`. Uniform width reduces the formatting burden on the
readout — after a `.` it always emits two digits, then space or `#`.

Division rarely produces a clean 2-decimal result. Three candidate
policies:

| Option | Meaning | Tradeoff |
|---|---|---|
| A | Round true result to 2 decimals | Target is approximate; model learns rounding |
| B | Reject expressions whose result isn't exactly 2-decimal | Aggressive filter; starves grammar diversity |
| C | Restrict divisor to clean values (integers, powers of 10) | Narrows grammar, cleanly preserves format |

**Default: Option A.** Rounding is a legitimate "do the math" skill and
the model should learn it. Revisit if training stalls specifically on
division.

## Input Encoding

Each input character is fed to the reservoir as its 8-bit ASCII byte,
bipolar-encoded:

- `ReservoirConfig::num_inputs = 8`.
- Channel `b` (0 ≤ b ≤ 7) carries bit `b` of the current character's
  ASCII byte: `+1.0` if the bit is set, `-1.0` if cleared.
- `Reservoir`'s per-vertex channel routing (channel `k` owns vertices
  `k, k+K, 2K, ...`, `Reservoir.h:18`) gives each of the 8 channels
  **512 vertices at DIM 12** — a substantial random-projection slice
  per bit.

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
- **Output fraction** 1.0 (HCNN sees full state).
- **Seed** — no task-specific survey exists yet. First pass uses the
  DIM 9 NARMA winner as a proxy, or falls back to 42. A dedicated
  seed survey at DIM 12 on this task is a phase-4 item.

Reservoir state steps once per input character. HCNN sees raw reservoir
state; no translation layer (HCNN always bypasses it).

## Readout

- `CNNReadout` in **multi-class classification** mode, 20 softmax heads.
- Starting architecture: `HRCCNNBaseline<12>()` (nl=2, ch=16, FLATTEN,
  ep=2000, bs=2048, lr=0.0015).
- That baseline is calibrated for regression on NARMA, not 20-way
  classification on arithmetic. Expect per-task retuning. Concretely:
  deeper `nl` and wider `ch` may be needed because the output head's
  job is far more complex than one regression scalar.

## Training Regime

**One long stream**, no resets. Expressions concatenated with `#` as
the only separator:

```
5.00 + 3.00 = 8.00#-2.50 * 4.00 = -10.00#(1.00 + 2.00) / 3.00 = 1.00#...
```

- Reservoir warms up once at the start, then runs continuously.
- `#` is dual-purpose: the EOS emitted at inference, and the delimiter
  that cues "new expression begins" during training. The reservoir
  learns `#` as a context-reset marker without an explicit reset.
- Teacher forcing: at training position `t`, input = char(t), target =
  char(t+1). Every position contributes a training example.
- Target training-stream length: **~1M characters** for first pass
  (~25k expressions at ~40 chars each). Scale up if accuracy plateaus
  without overfitting.
- Validation: independently-generated 10k-expression stream.

**Should LHS positions contribute loss?** At deployment the model only
emits RHS characters; LHS is always given. Two views:

- Train on everything: every position is a free language-model
  pretext signal — the model learns numeric format, operator grammar,
  `=` position, etc. "for free."
- Mask LHS: training loss matches the deployment objective.

**Default: train on everything.** The LHS pretext should make formatting
solid before computation even kicks in.

**Class-imbalance.** Digits dominate target positions (probably >70%);
operators, parens, `=`, and `#` together are a small tail. Watch for
the head collapsing to always-predict-digit. If it does, inverse-
frequency loss reweighting in the softmax head is the first lever.

## Inference

```
1. Warm up reservoir on LHS: feed "5.00 + 3.00 = " one char at a time.
2. Loop:
     p ← softmax(CNNReadout(state))
     c ← argmax(p)          # start with argmax; sampling later
     emit c
     if c == '#': break
     state ← reservoir.step(bipolar_bits(c))   # 8-dim ±1 from ASCII byte
3. Concatenated emitted chars = predicted RHS.
```

Hard output-length cap of ~12 chars prevents runaway generation when
`#` is never emitted.

Training and inference reservoir dynamics are identical — only the
input source differs (teacher-forced ground truth vs. model's own
previous output).

## Success Metrics

- **Exact-match accuracy**: fraction of held-out expressions where the
  full generated RHS (through `#`) equals ground truth. The real metric.
- **Character-level accuracy**: per-position argmax-matches-target rate.
  Climbs earlier and smoother than exact-match; the training dashboard.
- **Format accuracy**: fraction of emitted RHS strings that parse as a
  valid `[-]d+\.dd#` regardless of value. Separates "learned numeric
  format" from "learned computation." Format should saturate first.

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
- **Decimal misalignment**: emitting `"1.05"` for true `"10.50"`.
  Indicates weak magnitude encoding in the reservoir's projection of
  operand digits. A distinct failure pattern worth watching for.
- **EOS over/under-prediction**: model emits `#` too early (truncated
  answer) or never (runaway). Class reweighting on `#` loss addresses
  both directions.

## Phased Delivery

**Phase 1 — scaffold (no HRC).**
- Expression generator: grammar, value clipping, division rounding.
- Vocab ↔ index tables; one-hot encoder/decoder.
- Verify 1k generated samples parse and evaluate correctly with an
  independent reference evaluator (e.g., `std::stod` on reshaped
  infix, or a tiny shunting-yard parser).

**Phase 2 — smallest viable model.**
- DIM 8 (N=256), not 12. Iteration speed matters for the scaffold.
- Restrict grammar to depth 0, operators `+ -` only, operands in
  `[-10.00, 10.00]`.
- Target: exact-match > 50% on held-out `+ -` single-op expressions.
- If this fails, DIM 12 won't rescue the approach — something
  structural is wrong. Diagnose before scaling up.

**Phase 3 — scale to target.**
- DIM 12, full vocab, full grammar.
- 1M-character training stream.
- Track all three metrics (format, character, exact-match) across
  epochs.

**Phase 4 — diagnostics & tuning.**
- Seed survey at DIM 12 on this task (reservoir and CNN-init seeds).
- Operator-stratified accuracy (where does it fail?).
- Operand-magnitude-stratified accuracy (does performance degrade
  for `|x| > 50`?).
- Head architecture sweep if exact-match stalls (probe `nl`, `ch`).

**Phase 5 — scale out if needed.**
- If DIM 12 caps below, say, 80% exact-match, try DIM 14 (N=16k,
  ~4× training cost). DIM 16 is the hard ceiling (library-supported).

## Open Decisions

- DIM 12 seed: use DIM 9 NARMA winner as proxy until a task-specific
  survey runs.
- Keep or drop space from the vocab. Default keep; revisit at phase 3.
- Per-class loss weighting (uniform vs inverse-frequency). Default
  uniform; switch to inverse-frequency if operators/EOS under-train.
- Sampling vs. argmax at inference. Start argmax; sampling only if
  we want diversity metrics.
