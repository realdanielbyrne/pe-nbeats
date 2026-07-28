# A configuration recipe for N-BEATS, derived from measurement

Everything here is a rule a practitioner can apply without reading the paper, each with the evidence
behind it and a confidence grade. Rules graded **strong** rest on controlled sweeps of hundreds to
thousands of runs across all six M4 periods. **Moderate** rests on one or two datasets. **Provisional**
is a mechanistic prediction with limited direct measurement, flagged rather than hidden.

## On the name

I would not call this N-BEATS 2. The recipe mostly *subtracts*: a narrower trunk, fewer stacks, a
frozen synthesis head. A version number implies a successor architecture, invites comparison as a rival,
and overstates what is on offer. Three better options, in order of preference:

1. **A named preset on the existing model**, `NBeats(profile="slim")`. This is literally what it is, a
   configuration of N-BEATS, and it is the lowest friction path into Darts or pytorch-forecasting: no
   new class, no new documentation page, no migration.
2. **N-BEATS-Slim** as a distinct class, if a name is wanted for citation.
3. N-BEATS 2, only if the community treats the changes as a redesign rather than a retuning.

The recipe's real selling point is not the defaults but that **it tells you in advance when they
apply**, which is the pre-flight check below.

## The defaults

| Setting | Default | Why | Grade |
|---|---|---|---|
| Stacks | **10** | 30 stacks bought nothing on any M4 period for N-BEATS-I+G (−0.200 to +0.026, none significant) and collapsed the Generic family on Quarterly (+4.96, p = 0.001) and Daily (+5.60) | strong |
| Trunk width | **128** | On M4, 1,323,550 against 64,645,150 parameters scored 13.707 against 13.645; Kruskal-Wallis p = 0.776 across a 48.8 fold range. No validated way to know in advance when this is unsafe, see below | strong on M4 |
| Forecast basis rank | **complete**, `d_f = H` | At complete rank a freely learned projection absorbs any rotation, so identity is provably inert and the choice of matrix is free. Below it, accuracy degrades predictably with captured energy | strong |
| Backcast basis rank | **truncate**, `d_b ≈ H` | The backcast has no target, so its rank incurs no approximation error; 11 of 12 paired M4 comparisons favour the truncated basis | strong |
| Basis family | **irrelevant at complete rank** | 36 paired comparisons at identical parameter counts, none significant, minimum p = 0.084; family null over eight datasets, Fisher p = 0.125 | strong |
| Output head | **rectified**, subject to the condition below | Total collapses fall from 107 of 23,786 runs to 4 of 6,372 with a rectified forecast, and to 0 of 136 with both paths rectified | strong |
| Trend branch | **pair it** when the spectrum is concentrated | Takes a DCT from 28.748 to 9.111 and a Karhunen-Loève basis from 27.607 to 9.291, both from 0 of 10 seeds converging to 10 of 10, p = 0.00018 | strong |
| Loss | sMAPE or MASE as the task requires | — | — |

## The pre-flight check, which is the actual contribution

Two numbers computed from the training windows, no GPU, no training:

**Effective rank** = participation ratio of the coefficient spectrum, roughly how many coefficients
carry the signal. It decides whether a plain linear head will train at all. Below about 4, pair a trend
branch or rectify the output; M4-Hourly sits at 1.05 and fails on every seed without one. This rule is
supported: the same quantity tracks measured convergence rate at rho = +0.949, and both remedies are
measured to work.

**Compressibility** = fraction of the horizon needed to capture 99.9% of the *detrended* target energy.
Reported as a diagnostic. **It is not a validated predictor of trunk sensitivity and should not be used
as one.** I built it for that purpose and it failed against my own data: M4-Yearly and Monthly sit at
0.500 and 0.889, which any threshold separating them from Traffic-96 at 1.000 would misclassify as
needing a wide trunk, yet both are measured inert across a 48.8 fold range. It does separate rank
deficient from full rank targets cleanly on synthetic data at any offset, so it measures something real;
it just does not measure this. Reported with that caveat attached rather than dropped, because the
negative result is worth knowing.

**So there is currently no way to know in advance whether the narrow trunk is safe on a new dataset.**
The honest procedure is to try 128 and 512 on a held out split. That is one integer and two runs, which
is a far cheaper answer than the sweep it replaces, but it is not the free lunch a predictor would be.

Shipped as `lightningnbeats.preflight.recommend(windows, forecast_length)`, returning a configuration
and the measurements that produced it. This is the piece a library should expose, because it converts
"which settings do I use" from a sweep into a function call.

## The one condition on the rectified head, which is easy to get wrong

The rule is **not** "rectify when the targets are positive". It is **"rectify when each block's own
output should be positive"**, and those come apart.

In a doubly residual stack over a non negative series, each block emits a partial reconstruction of a
positive target, so rectifying is free. In any architecture whose blocks emit *signed components that
are supposed to cancel*, rectifying forbids them from doing their job. A decomposition based design is
the clear case: measured on M4-Daily, five of six DWT sub-bands are about half negative by
construction, since that is what a detail band is. Rectifying such a model degrades it catastrophically,
by 9 to 40 sMAPE across periods.

I found this by applying the rectifier symmetrically to two architectures for fairness and discovering
that symmetric in the configuration file is not symmetric in effect. [strong]

## What NOT to do, all measured

- **Do not add stacks to fix a deficit.** On Traffic the frozen basis arms move −0.12 to +0.97 going
  from 10 to 30 stacks, mostly worse, while the Generic baseline *gains* 0.65 (p = 0.014). Depth widens
  the gap rather than closing it. [strong]
- **Do not use an autoencoder bottleneck for accuracy.** At matched budget it does not beat a plain
  narrow trunk: 13.655 at 8.31M against 13.564 for a Generic baseline at 8.22M on M4-Yearly. Narrowing
  the trunk is simpler, better supported, and a one line change. It remains useful only when the
  latent dimension itself is the binding constraint. [strong]
- **Do not add the learnable gate.** Null across twelve comparisons, smallest p = 0.24, at +320
  parameters. [strong]
- **Do not use variational blocks.** Catastrophic on M4, 5 to 10 times worse. [strong]
- **Do not rectify both paths by default.** It removes collapse but taxes every run that would have
  converged: +0.129 sMAPE at p = 3.7e-06 on 50 seeds. Rectify the forecast only. [strong]
- **Do not use a leaky rectifier to avoid dead outputs.** sMAPE scores any negative forecast at exactly
  200 regardless of magnitude, so a leaky unit trades the rare zero mode for the common negative mode.
  Use `Softplus` if exact zeros must be excluded. [strong on the metric, untested end to end]
- **Do not choose a wavelet family.** It is a well powered null. Spend the effort on rank instead.
  [strong]
- **Do not truncate a localised basis inside a scale band.** On M4-Monthly the bands sit at 9 rows: at
  6 rows the basis captures 0.658 and scores 73.380; at 9 it captures 0.996 and scores 13.850. Three
  rows are worth 59.5 sMAPE. Complete rank sidesteps this entirely. [strong]

## Why the recipe works, in one paragraph

N-BEATS as published carries roughly fifty times more trunk capacity than its accuracy requires on M4,
and the excess was invisible because trunk width had never been swept. Replacing each block's
unconstrained linear synthesis head with a projection through a frozen orthonormal basis makes the rank
an explicit knob and removes the head's ability to emit values outside the target's range, which is the
documented failure mode. Which basis fills that slot does not matter once it is complete. What matters
is whether the target is compressible enough for the narrow trunk, and that is measurable before
training.

## Honest limits

The trunk rule is measured on M4, where targets are highly compressible. The incompressible branch,
which is where Traffic-96 lands, is currently a mechanistic prediction with a study in flight; until it
returns, 512 is a safe default there rather than a measured optimum. The frozen basis arms lose outright
to the paper baselines on Traffic-96, and the recipe does not claim otherwise. Whether that deficit is
the trunk or the architecture is exactly what the running study decides.
