# N-BEATS divergence on M4 is a sign failure, not a fit failure (2026-07-27)

Four predictions were registered in `experiments/configs/m4_divergence_signs.yaml` before any run
started. All four are confirmed, and the decisive one is confirmed to within 0.11 sMAPE. This is the
strongest single result in the stability line of work, and unlike everything else in it, it is a direct
measurement of the forecasts themselves rather than an inference from summary statistics.

## Design

M4-Weekly and M4-Daily, paper sample protocol, 30 stacks of `Generic`, one block per stack, no weight
sharing, the exact configuration whose divergence rate motivated the question. Ten seeds per arm, two
arms, `save_predictions: true` so that every forecast tensor is on disk. The only difference between
arms is `active_g`.

| arm | `active_g` | Weekly | Daily |
|---|---|---|---|
| `Generic_30s_sw0_ag0` | False | 11.768 ± 7.220, **3/10 diverged** | 4.585 ± 4.445, **1/10 diverged** |
| `Generic_30s_sw0_agTrue` | True | 7.230 ± 0.297, 0/10 | 3.301 ± 0.121, 0/10 |

The instability reproduced exactly as expected, which is what makes the forecasts worth opening.

## Prediction 1: diverged runs emit negative forecasts, and those points sit at the ceiling

Confirmed, and the fractions land inside the 5 to 20 percent window registered in advance.

| period | seed | sMAPE | percent of points negative | percent at ceiling | mean sMAPE of the negative points | most negative forecast |
|---|---|---|---|---|---|---|
| Weekly | 45 | 22.195 | 7.778 | 7.778 | **200.000** | −7,801.5 |
| Weekly | 47 | 22.212 | 7.692 | 7.692 | **200.000** | −14,513.2 |
| Weekly | 51 | 22.279 | 7.799 | 7.799 | **200.000** | −10,594.1 |
| Daily | 43 | 17.233 | 7.143 | 7.143 | **200.000** | −24,313.4 |

Two details are worth pausing on. The negative fraction and the ceiling fraction are **identical to
every digit** in all four runs, so there is no second route to the ceiling operating here: nothing
reaches 200 except by having a negative sign. And the mean per point sMAPE over exactly the negative
points is 200.000 rather than merely close to it, which is the analytic prediction and not a fitted
quantity.

## Prediction 2: converged runs are essentially clean

Confirmed. All nine converged Daily runs contain **zero** negative forecast points, with a minimum
forecast of about +132 in every one. Weekly is nearly as clean: three of seven converged runs are at
exactly zero, and the remaining four carry 0.086 to 0.129 percent negatives, which is between one and
two points in a thousand. Those trace amounts cost between 0.17 and 0.25 sMAPE, so even the "clean"
Weekly runs are paying a small, previously invisible sign tax.

## Prediction 3: rectifying both paths eliminates the mechanism outright

Confirmed with nothing to qualify. Across all twenty `active_g=True` runs the maximum negative fraction
is **0.000000**, the maximum exact zero fraction is **0.0000**, and the smallest forecast value produced
anywhere is **+112.6**. Zero divergences. The intervention closes the failure mode by construction
rather than by making it rarer, which is exactly the claim.

## Prediction 4, the decisive one: deleting the negative points recovers the converged score

Confirmed, and by a wider margin than I expected. Rescoring each diverged run on only its non negative
points:

| period | seed | full sMAPE | negatives deleted | converged mean of the same arm | residual gap |
|---|---|---|---|---|---|
| Weekly | 45 | 22.195 | **7.199** | 7.284 | −0.085 |
| Weekly | 47 | 22.212 | **7.397** | 7.284 | +0.113 |
| Weekly | 51 | 22.279 | **7.245** | 7.284 | −0.039 |
| Daily | 43 | 17.233 | **3.174** | 3.180 | −0.006 |

A run that scores 22.279 and a run that scores 7.105 are, on 92 percent of their points, the same model.
The residual gaps are smaller than the seed to seed spread of the converged runs themselves, and two of
the four are negative, meaning the diverged model is *better* than the converged average on the points
it did not botch. There is no general loss of fit hiding underneath. The arithmetic closes too:
`0.9286 × 3.174 + 0.0714 × 200 = 17.23`, against the observed 17.233.

## What this settles, and what it does not

It settles the mechanism. Divergence in these configurations is a small minority of catastrophic sign
errors, amplified to the metric ceiling by sMAPE's behaviour when forecast and target have opposite
signs. It is not gradient explosion, not a failure to learn the series, and not a capacity problem. It
also explains cleanly why an output rectifier is one of the interventions that removes the instability
without appealing to expressiveness or to magnitude bounding, both of which I argued for previously and
neither of which survives: ReLU is unbounded above, and `active_g="forecast"` never touches the backcast
at all.

Three caveats belong with it.

The result is measured under **sMAPE on strictly positive M4 series**. On a metric without a ceiling the
same sign errors would still hurt, and given magnitudes near −24,000 against targets near +130 they
would hurt considerably more, but the clean 200 signature is specific to sMAPE and the recovery
arithmetic would not be as tidy. On data that genuinely takes negative values, an output rectifier is
not available as a remedy at all.

The tax is real and separately measured. `active_g=True` removes the failure and costs +0.129 sMAPE on
converged runs at p = 3.7e-06 on the fifty seed M4-Yearly corpus. In this study the same pattern shows
on Daily, where the rectified arm's converged mean of 3.301 is worse than the unrectified arm's 3.180.
Insurance with a premium, and both numbers should be reported together.

Zero output remains an open route. A rectifier cannot emit a negative but can emit exactly zero, and
`sMAPE(0, A) = 200` as well. Of the 112 runs at exactly 200 across the whole M4 corpus, 108 are `ag0`
and 4 are `agf`, and those surviving 4 must be the dead zero mode. Nothing in this study contains a
single exact zero, so the route is real but rare, and a strictly positive activation such as `Softplus`
would close it. Untested.

## Reproduction

```bash
python experiments/run_from_yaml.py experiments/configs/m4_divergence_signs.yaml
```

Rows in `experiments/results/m4/m4_divergence_signs_results.csv`, forecast tensors in
`experiments/results/predictions/m4_divergence_signs_*.npz`. Every number above is recomputed from those
tensors; the recomputed sMAPE agrees with the logged sMAPE to four decimal places on all forty runs,
which also validates the evaluation path.
