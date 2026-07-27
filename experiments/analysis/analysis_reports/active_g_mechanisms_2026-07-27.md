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
run that would have been fine. On this one period `sum_losses` looks like the same insurance without
the premium, 0/50 at 13.561, and that is what I wrongly generalised. See the corrected rule below: it
does not hold once other periods are included.

**Why the tax exists.** With both paths rectified, no block can make a downward correction to the
running forecast or a negative contribution to the residual. In a doubly residual cascade whose whole
design is iterative correction, forbidding corrections in one direction costs real capacity.

**Practical rule, CORRECTED.** An earlier version of this file recommended `sum_losses` over
`active_g=True` as a free substitute stabiliser, on the strength of the Yearly cell above. That is
withdrawn. Widening the search: `Generic_sumLosses` on Monthly collapsed 1 of 5 seeds to exactly 200
test sMAPE, and pooled over 30,434 M4 runs the collapse rates are

| setting | collapses at sMAPE 200 |
|---|---|
| plain linear head | 107 / 23,786 (0.45%) |
| `sum_losses=True`, ag0 | 1 / 70 (1.4%) |
| `active_g="forecast"` | 4 / 6,372 (0.06%) |
| `active_g=True` | 0 / 136 |

so the activation is the reliable stabiliser and `sum_losses` is not established as one. Reserve
`active_g="forecast"` for cases where it is measured to help, which on M4 is Hourly only.

**A measurement bug that invalidates the Milk study.** `validation_step` adds
`0.25 * loss_fn(backcast, zeros)` when `sum_losses=True`, and sMAPE against zeros is 200 for any
nonzero backcast, so that term contributes about 50 by construction. Measured offsets between
`sum_losses` twins on Milk are 51.077 and 49.956 against a predicted 50.000. The Milk 2x2, 400 runs,
therefore compares two different objectives rather than two model qualities, and **no `best_val_loss`
comparison across `sum_losses` settings anywhere in this repository is valid.**

**Existing-data corroboration for the sign account.** Of 112 runs at exactly sMAPE 200 across all M4
results, 108 are `active_g=False` and 4 are `agf`. An output activation removes 96 percent of total
collapses. The surviving 4 are informative: ReLU cannot emit a negative but can emit all zeros, and
`sMAPE(0, A) = 200` too, so agf closes the negative route and leaves the rarer dead-zero route open.

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

Canonical numbers, recomputed 2026-07-27 over all 214,810 M4-Hourly forecast windows by
`experiments/scripts/basis_conditioning.py`. **These supersede the figures in the first version of this
table**, which came from a different window set and are not reproducible; the ordering and every
conclusion are unchanged, but the magnitudes moved (DCT in particular, 3126 against 1081).

| basis | top-1 energy share | effective rank | max/min | ag0 sMAPE | seeds that converge |
|---|---|---|---|---|---|
| KL | 0.9757 | **1.05** | 3722.3 | 23.472 | **0 / 10** |
| DCT | 0.7992 | 1.54 | 1081.2 | 27.054 | 0 / 10 |
| band ordered db3 | 0.1685 | 6.03 | 604.9 | **9.569** (median 9.194) | **9 / 10** |
| SVD db3 | 0.1459 | 17.86 | 38.7 | 9.309 | 10 / 10 |
| random orthonormal | 0.1184 | 17.12 | 24.9 | 9.333 | 10 / 10 |

**Two corrections to how this was previously reported.** First, the earlier table used "share of
coefficient 0", which is not comparable across these bases: coefficient 0 is the constant row for the
DCT and the leading eigenvector for KL, but an arbitrary row for SVD db3 and for a random draw, whose
rows carry no energy ordering at all. That understated the scrambled bases at 0.0055 and 0.0027 when
their true largest coefficients hold 0.1459 and 0.1184. **Top-1 share**, the energy fraction in the
largest coefficient whichever it is, is ordering independent and is the statistic of record.

Second, "dynamic range" invites confusion with the matrix condition number, which is exactly 1 for
every basis here since all are orthonormal. The quantity is the spread of the **target coefficient
distribution** the head must learn to emit, not a property of the basis matrix.

**Robustness.** Ranking the five bases against their measured convergence rate, with Spearman so that
the two arms tied at 10/10 are not penalised by an arbitrary tiebreak:

| statistic | rho | verdict |
|---|---|---|
| top-1 share | **−0.949** | at the ceiling attainable under the tie structure |
| effective rank | **+0.949** | at the ceiling |
| max/min | **−0.949** | at the ceiling |
| p95/p5 | −0.632 | fails |
| max/median | −0.791 | fails |

The two percentile measures fail for a diagnostic reason rather than a random one: both discard or
downweight the extreme coefficient, and the extreme coefficient is the coordinate that causes the
failure. **Effective rank**, the participation ratio `1 / sum_k e_k^2`, is the measure to report. It
depends on no order statistic and it states the failure condition without apology: a linear synthesis
head cannot train when the target effectively occupies one coordinate out of forty eight.

**Lower dispersion is better for training and worse for approximation, and the two are in direct
conflict.** Concentration is precisely what makes KL and DCT the best bases at truncated rank, where
they reach 0.9714 and 0.9630 captured energy on Traffic against 0.3035 for a wavelet. The resolution is
not to prefer a flatter basis and surrender the capture, but to repair the optimisation side and keep
the approximation win, which is what both an output rectifier and a Trend block do.

The band ordered cell was run on 2026-07-27 with predictions registered in
`experiments/configs/m4_conditioning_ag0.yaml`, and it lands where conditioning says it should rather
than where capture says it should: capture is flat to the fourth decimal across all five rows and
therefore predicts no differences at all, while dynamic range sorts them into the three tiers observed.
Per seed, the separation is not a shifted mean but a failure rate:

```
band ordered db3, ag0:  8.927  9.036  9.133  9.166  9.171  9.217  9.241  9.282  9.294  13.220
KL,               ag0: 16.944 20.962 20.967 20.978 21.229 21.357 25.086 28.916 29.106 29.172
```

Nine of ten band ordered seeds land in a 0.37 wide band whose median of 9.194 is slightly better than
SVD db3's mean, and one seed fails. Not a single KL seed succeeds. **One registered prediction failed
here and is recorded as such:** I predicted a standard deviation under 0.5 and measured 1.288, because
I expected the intermediate conditioning to produce clean convergence rather than a reduced failure
rate. The direction was right and the mechanism claim is unchanged, but the specific prediction was
wrong and the honest reading is that dynamic range 668 buys a 90 percent success rate rather than
immunity. Under `agf` the arm collapses to 9.103 ± 0.075 with all ten seeds converging, alongside KL at
9.081 and DCT at 9.102, exactly as predicted.

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
