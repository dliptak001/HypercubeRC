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

- Nesting depth ≤ 2. **Depth definition by example:**
  - `a op b` → depth 1.
  - `(a op b) op c` → depth 2 (one level of parens).
  - `(c op (a op b)) op d` → depth 3 (two levels; out of scope).
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

**Unary minus.** A leading `-` may prefix any `number` in any position
— after `=`, after `(`, after a binary operator, or at the start of a
bare expression. Parentheses are **not** required to wrap a negative
operand (e.g., `3.00 * -5.00 = -15.00#` is well-formed). The model is
expected to infer operator precedence from context; unary vs. binary
`-` is disambiguated by position, not grammar.

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
| A | Round true result to 2 decimals via `std::round` (half-away-from-zero) | Target is approximate; model learns rounding |
| B | Reject expressions whose result isn't exactly 2-decimal | Aggressive filter; starves grammar diversity |
| C | Restrict divisor to clean values (integers, powers of 10) | Narrows grammar, cleanly preserves format |

**Default: Option A** with `std::round` (half-away-from-zero). Rounding
is a legitimate "do the math" skill and the model should learn it. This
rounding is applied **only during training-dataset construction** — the
model just sees the final 2-decimal target. Revisit if training stalls
specifically on division.

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
- **Seed** — **single seed, all phases, no survey.** Uses the DIM 9
  NARMA winner as a proxy. Multi-seed averaging is explicitly out of
  scope for this example.

Reservoir state steps once per input character. HCNN sees raw reservoir
state; no translation layer (HCNN always bypasses it).

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

Starting architecture: `HRCCNNBaseline<12>()` backbone (nl=2, ch=16,
FLATTEN, lr=0.0015) with **task-specific training overrides: ep=1000,
bs=4096**. The baseline `ep=2000, bs=2048` is calibrated for ~50k
gradient updates on regression over NARMA's small dataset. This task
has ~10× more training positions, so we halve epochs and double batch
size to land at a reasonable update budget (~250k gradient updates,
~5× the NARMA calibration — more than baseline, appropriate for the
richer per-position classification objective). Expect further
retuning — deeper `nl` and wider `ch` may be needed because the
readout's job here is far more complex than one regression scalar.

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
forcing guarantee). The protocol belongs in the math_llm driver, not
in the Reservoir class — `Reset()` stays minimal, priming is the
caller's job.

`#` is a pure EOS marker with no dual purpose. This aligns training
exactly with the inference regime (the user presents one equation;
we reset, prime, generate).

```
sample: "5.00 + 3.00 = 8.00#"
sample: "-2.50 * 4.00 = -10.00#"
sample: "(1.00 + 2.00) / 3.00 = 1.00#"
...
```

- Before each expression: `reservoir.Reset()` (zero the state), then
  two priming passes over the LHS (no readout, no state collection).
- Collecting pass: feed the full expression character by character.
- Teacher forcing: at position `t`, input = char(t), target = char(t+1).
  Every position contributes a training example.
- Target training-set size: **5k independent expressions** for the v1
  first pass — ~3.2 GB of reservoir states at DIM 12, fast iteration.
  Scale up (10k → 25k → larger) as results warrant.
- Validation: independently-generated 2k-expression set (sized to
  match the v1 training scale; scales up alongside training).

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

**Class-imbalance.** Digits dominate target positions (probably >70%);
operators, parens, `=`, and `#` together are a small tail. Watch for
the head collapsing to always-predict-digit. If it does, inverse-
frequency loss reweighting in the softmax head is the first lever.

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
- 8-bit bipolar **input** encoder (char → 8-dim ±1 vector for the
  reservoir).
- Vocab ↔ class-index tables for the **output** side (20-way target
  labels during training + argmax decode at inference).
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
- **5k independent training expressions** (v1). Upscale in 2× steps
  only if v1 accuracy is promising but capped.
- Track all three metrics (format, character, exact-match) across
  epochs.

**Phase 4 — diagnostics & tuning.**
- Operator-stratified accuracy (where does it fail?).
- Operand-magnitude-stratified accuracy (does performance degrade
  for `|x| > 50`?).
- **Softmax distribution logging** during held-out inference: for
  each RHS position, capture top-N contenders, confidence margin
  (`p_argmax - p_second`), and distribution entropy. Surfaces where
  the model is confident-and-wrong vs. uncertain-and-wrong — very
  different failure modes that call for different fixes.
- Head architecture sweep if exact-match stalls (probe `nl`, `ch`).
  Single seed throughout — no seed survey.

**Phase 5 — scale out if needed.**
- If DIM 12 caps below, say, 80% exact-match, try DIM 14 (N=16k,
  ~4× training cost). DIM 16 is the hard ceiling (library-supported).

## Open Decisions

- Keep or drop space from the vocab. Default keep; revisit at phase 3.
- Per-class loss weighting (uniform vs inverse-frequency). Default
  uniform; switch to inverse-frequency if operators/EOS under-train.
- Reservoir `Reset()` semantics — see task #1.
- v1 training-set size given 64 GB RAM budget — see task #2.
