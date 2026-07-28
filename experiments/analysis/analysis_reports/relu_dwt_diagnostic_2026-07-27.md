# Why a rectified head destroys DWT preprocessing (2026-07-27)

Four predictions were registered in `m4_dwt_relu_diagnostic.yaml`. **Two are falsified, both in the
opposite direction to the one I predicted**, and the mechanism that survives is not the one I proposed.

## The effect, reproduced

M4-Daily, five seeds, forecasts saved.

| design | head | sMAPE |
|---|---|---|
| DWT preprocessing | linear | 9.100 ± 12.553 |
| DWT preprocessing | **rectified** | **41.571 ± 13.764** |
| in block frozen basis | linear | 3.265 ± 0.117 |
| in block frozen basis | **rectified** | **3.266 ± 0.082** |

The same intervention is inert for one design and catastrophic for the other, and the block is
identical in both.

## What the forecasts actually show

| design | head | level | roughness | preds ≤ 0 | at sMAPE ceiling |
|---|---|---|---|---|---|
| DWT preprocessing | linear | 0.951 | 7.72 | 0.029 | 0.029 |
| DWT preprocessing | rectified | **0.722** | **26.38** | 0.000 | 0.000 |
| in block | linear | 0.988 | 0.086 | 0.000 | 0.000 |
| in block | rectified | 0.988 | 0.052 | 0.000 | 0.000 |

`level` is mean forecast over mean target, so 1.0 is unbiased. `roughness` is the mean absolute first
difference of the forecast divided by the same statistic on the targets, so 1.0 is as jagged as the
data and below 1.0 is smoother.

## Predictions, scored

**1. The rectified arm overshoots. FALSIFIED, with the sign reversed.** It *undershoots* badly: the
median signed error is **−12.7 percent** of the mean absolute target, and only **3.5 percent** of points
lie above the target against 24.6 percent for the linear head. The level ratio is 0.722, so the
rectified model predicts about three quarters of the true level.

**2. The overshoot exceeds the 2.5 percent analytic bias. MOOT**, since there is no overshoot. The
analytic argument was not merely too small, it pointed the wrong way.

**3. No run reaches exactly 200. CONFIRMED.** Zero predictions at or below zero, zero points at the
ceiling, exactly as a rectifier guarantees. The failure is definitively **not** the sign collapse
documented elsewhere in this work, which is what this prediction was for.

**4. The detail stacks go dead, producing smoother forecasts. FALSIFIED, and dramatically reversed.**
The rectified forecasts are **26.4 times rougher than the targets**, against 7.7 for the same design
with a linear head. Far from going quiet, the detail stacks became the dominant source of variation.

## The mechanism that survives

Not bias, not dead stacks, not runaway scale. **Rectification destroys cancellation.**

A detail sub-band is a zero mean oscillation, and its job is to make signed corrections that partly
cancel one another before the sum reaches the output. Rectifying every stack's forecast removes the
negative half of each correction, so five bands that used to cancel now only ever add, and they add as
half wave spikes rather than smooth signed curves. That is exactly the 26.4 times roughness.

The undershoot follows from it. Since the model can no longer subtract anywhere, the only defence
against those spikes overshooting is to hold the approximation band's contribution low, so it learns a
level about 28 percent beneath the truth and lets the spikes make up the difference unevenly. Both
symptoms come from one cause, and both hurt sMAPE.

The in block design is untouched because its blocks are in series over a shared residual, so how much
each contributes is *learned* rather than fixed by a preprocessor. Any positive target admits a
decomposition into non negative parts, and the cascade simply finds one, approaching from below.
Roughness even falls slightly, 0.086 to 0.052.

## Consequences

**The comparison stands as reported.** The ag0 result is the valid one, and the agf regime remains
discarded as an unfair test. This diagnostic confirms *why* it is unfair rather than changing the
verdict.

**The rectifier rule needs the wording it now has.** Not "rectify when the targets are positive" but
"rectify when each block's own output should be positive". A decomposition based architecture fails the
second test while passing the first, and the failure is silent, large, and would look like a bad
architecture rather than a bad flag.

**My proposed explanation was wrong twice over** and is retracted: half wave rectification bias
accounted for 2.5 percent of the target and pointed toward overshoot, where the measured effect is a 28
percent undershoot with a 26 fold roughness increase. Keeping the analytic story would have been
plausible, quantitatively wrong, and directionally wrong.

## Reproduction

```bash
python experiments/run_from_yaml.py experiments/configs/m4_dwt_relu_diagnostic.yaml
```

20 of 20 rows in `experiments/results/m4/m4_dwt_relu_diagnostic_results.csv`, forecast tensors in
`experiments/results/predictions/m4_dwt_relu_diagnostic_*.npz`.
