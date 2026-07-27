# What `active_g` actually does, and what it costs (2026-07-27)

Written because I gave three wrong explanations for this in sequence and want the reasoning preserved
rather than rediscovered. Each wrong version is recorded with why it failed.

## The setting matters and I kept conflating two of them

```python
if self.active_g:
    if self.active_g != "forecast":
        backcast = self.activation(backcast)   # SKIPPED when active_g == "forecast"
    if self.active_g != "backcast":
        forecast = self.activation(forecast)
```

* `active_g=True` activates **both** paths. This is what the convergence and ablation studies used.
* `active_g="forecast"` (`agf`) activates **only the forecast**. The backcast, and therefore the whole
  residual cascade, is untouched. This is what the M4 sweeps and every result in the basis studies use.

Any explanation appealing to the residual cascade can apply to the first and cannot apply to `agf`.

## Three explanations I gave, two of them wrong

**Wrong 1: it adds expressiveness.** A Generic block is `Linear(MLP(x))` with a four layer ReLU trunk.
Measured directly, `|| f(2x) - 2 f(x) || = 0.0314`, so the block is already nonlinear and the readout is
not confined to a plane. Decisively, the instability is removed by four independent interventions and
**three of them reduce expressiveness**: a frozen low rank basis, a bottleneck trunk, and a backcast
penalty. If insufficient expressiveness were the problem, removing expressiveness would worsen it.

**Wrong 2: it bounds the magnitude a block injects.** ReLU is unbounded above. What is true is that it
is 1-Lipschitz, so it cannot amplify, whereas a linear map amplifies by its largest singular value. But
"bounds the magnitude" is wrong, and for `agf` the argument is doubly wrong because `agf` does not
touch the backcast at all.

**Current account: it constrains SIGN, and sMAPE punishes sign errors at the ceiling.** For positive
targets and a negative forecast, `|F-A| = |F|+|A|`, so that point scores exactly 200, the sMAPE
maximum. A converged 30 stack Generic on M4-Daily scores 3.1 and the diverged seeds score 17.4, 31.2
and 31.6. The fraction of points at 200 required to produce those:

| observed sMAPE | fraction of points negative |
|---|---|
| 17.4 | 7.3% |
| 31.2 | 14.3% |
| 31.6 | 14.5% |

A small minority of sign errors reproduces the divergences exactly, and ReLU on the forecast makes
that failure mode structurally impossible. **This is arithmetic consistent with the numbers, not yet
evidence.** `m4_divergence_signs.yaml` saves per run forecasts to test it, with the decisive check
being whether deleting the negative points from a diverged run recovers the converged score.

## The accuracy cost of `active_g=True`, isolated

M4-Yearly, 10 stack Generic, **50 seeds per arm** (`convergence_study_results_v1.csv`):

| arm | `active_g` | mean | sd | diverged | **converged only mean** |
|---|---|---|---|---|---|
| baseline | False | 14.167 | 4.444 | 1/50 | **13.539** |
| activeG | True | 13.668 | 0.121 | 0/50 | **13.668** |
| sumLosses | False | 13.561 | 0.126 | 0/50 | **13.561** |
| activeG+sumL | True | 13.646 | 0.164 | 0/50 | 13.646 |

On raw means `active_g=True` looks like a win, 13.668 against 14.167. **The entire advantage is one
diverged baseline seed at 44.95.** Restricting to converged runs inverts it:

* activeG against baseline: **+0.129 worse, MWU p = 3.7e-06**
* activeG against sumLosses: **+0.107 worse, MWU p = 6.1e-05**

So `active_g=True` is insurance with a premium: it removes a rare catastrophic failure and taxes every
run that would have been fine. And the premium is avoidable, since `sum_losses` reaches the same 0/50
divergence rate at 13.561, statistically indistinguishable from the converged baseline.

**Why the tax exists.** With both paths rectified, no block can make a downward correction to the
running forecast or a negative contribution to the residual. In a doubly residual cascade whose whole
design is iterative correction, forbidding corrections in one direction costs real capacity.
`sum_losses` stabilises by penalising backcast reconstruction error instead, which constrains nothing
about sign.

**Practical rule.** Prefer `sum_losses` to `active_g=True` for stability. Reserve `active_g="forecast"`
for cases where it is measured to help, which on M4 is Hourly only, where it is worth −0.23 to −0.37 on
frozen basis arms and is preferred in 9 of 9 matched stems.

## Where `agf` helps for reasons unrelated to sign

Two further mechanisms, both established in the basis studies and neither involving the cascade:

1. **Subspace escape at truncated rank.** `sigma(B^T theta)` is not confined to `row(B)`, so the
   subspace floor does not apply. Worth 25 to 32 percent of achievable error for low capture bases and
   4 to 7 percent for high capture ones. Trained: SVD db3 at `d=16/48` goes 73.946 to 10.526, while the
   same basis at complete rank moves only 9.309 to 9.072.
2. **Free DC for energy concentrating bases.** ReLU has positive output mean even for zero mean input,
   so the DC level need not come from one enormous coefficient. This repairs the complete rank
   conditioning failure: DCT 27.054 to 9.102, KL 23.472 to 9.081.

So `active_g` is at least three separate mechanisms wearing one flag. The paper should not present it
as a single device.

## The conditioning failure, for the record

At complete rank all bases represent the target exactly, yet DCT and KL fail under a linear head while
db3 and random draws do not. The cause is the coefficient vector each basis requires:

| basis | share of coefficient energy in coefficient 0 | dynamic range | ag0 sMAPE |
|---|---|---|---|
| DCT | 0.9757 | 3126 | 27.054 |
| KL | 0.9757 | 3431 | 23.472 |
| band ordered db3 | 0.1597 | 668 | not run |
| SVD db3 | 0.0055 | 59 | 9.309 |
| random orthonormal | 0.0027 | 25 | 9.333 |

M4-Hourly windows carry 0.9757 of their energy in the mean, and DCT row 0 is exactly constant while
KL's leading eigenvector is nearly so, so each aligns one row with almost all the energy. The block
must then learn one coefficient near 225,000 and forty seven near 240 from a single linear layer with
one learning rate. **The KL cell was registered as a falsification test before it was run and it
failed as predicted**, which is the main evidence for this account.

Band ordered wavelets spread the same energy over six scaling coefficients, giving 0.9972 capture in
the top six at a dynamic range of 668, roughly five times better conditioned than the DCT. That is a
genuine practical advantage and an untested cell.

**Scope caveat.** Remove the window mean and the dynamic ranges collapse to 99 for band ordered db3 and
17 for SVD db3. The entire conditioning problem is an artifact of unnormalised, DC dominated data. It
should be expected to vanish under normalisation, which is consistent with Traffic, where occupancy is
naturally bounded and the leading eigenvector holds only 0.61 of the energy.
