# A Trend block repairs the frozen basis conditioning failure (2026-07-27)

Five predictions were registered in `experiments/configs/m4_trend_repairs_conditioning.yaml` before any
run started. Three are confirmed outright, one is confirmed on its stated tolerance but overstated in
its wording, and one is half wrong. The headline is not in doubt: **prepending a Trend block converts
two total failures into two clean successes, and it does so without any activation change.**

## The claim being tested

At complete forecast rank, DCT and KL fail under a plain linear synthesis head on M4-Hourly while
SVD-ordered wavelets and random draws do not, and captured energy is flat to the fourth decimal across
all of them, so expressiveness explains none of it. The proposed variable is how the target's energy is
distributed across coefficients: M4-Hourly windows carry 0.9757 of their energy in a single direction
for KL and 0.7992 for the DCT, so the head must learn one enormous coefficient alongside forty seven
small ones under a single learning rate.

A Trend block spans a low order polynomial including a constant. In a doubly residual cascade it
subtracts that component from the running residual, so the frozen basis block never sees it. If
conditioning is the operative variable, a Trend block should repair exactly what an output rectifier
repairs, by the same mechanism and with neither one adding expressiveness.

**Training free, measured before the runs.** The degree 2 subspace accounts for **0.978786** of
M4-Hourly forecast energy, and removing it transforms the coefficient statistics:

| basis | top-1 share raw → detrended | effective rank raw → detrended |
|---|---|---|
| KL | 0.9757 → 0.4573 | 1.05 → 2.53 |
| DCT | 0.7992 → 0.3836 | 1.54 → 3.31 |
| band ordered db3 | 0.1685 → 0.2348 | 6.03 → 6.37 |

## Results

Ten seeds per cell, sixty runs, `ag0` throughout so nothing here depends on an activation.

| arm | mean | median | sd | min | max | converging |
|---|---|---|---|---|---|---|
| **Trend + DCT** | **9.111** | 9.100 | 0.152 | 8.930 | 9.341 | **10 / 10** |
| **Trend + band ordered db3** | **9.132** | 9.129 | 0.109 | 8.908 | 9.353 | **10 / 10** |
| **Trend + KL** | **9.291** | 9.285 | 0.112 | 9.110 | 9.553 | **10 / 10** |
| no trend, DCT | 28.748 | 27.076 | 6.469 | 21.030 | 37.398 | 0 / 10 |
| no trend, KL | 27.607 | 28.977 | 6.332 | 13.199 | 37.426 | 0 / 10 |
| no trend, band ordered db3 | 9.289 | 9.280 | 0.103 | 9.115 | 9.455 | 10 / 10 |

| basis | no trend | with trend | Δ | MWU p |
|---|---|---|---|---|
| DCT | 28.748 | 9.111 | **−19.636** | 0.00018 |
| KL | 27.607 | 9.291 | **−18.316** | 0.00018 |
| band ordered db3 | 9.289 | 9.132 | −0.157 | 0.0036 |

## Prediction by prediction

**1. DCT and band ordered db3 converge 10/10 when paired with Trend. CONFIRMED.** DCT goes from zero
seeds converging to ten, spanning 8.930 to 9.341. Band ordered db3 likewise, 8.908 to 9.353.

**2. KL converges on at least 8 of 10. CONFIRMED and exceeded**, at 10 of 10. I registered the weaker
bound because KL retains the worst residual conditioning after trend removal, effective rank 2.53
against 3.31 and 6.37. That caution turned out to be unnecessary for convergence, though it does show up
in the ranking: KL is the worst of the three repaired arms, which is the direction the residual
conditioning predicts.

**3. The no trend control does not repair anything. CONFIRMED decisively.** At five stacks with no
Trend block, DCT and KL still fail on every seed, 0 of 10 each. So the repair is the trend branch and
not the block count, and this is the control that makes the causal claim rather than a correlational
one.

One wrinkle worth recording rather than smoothing over: band ordered db3 converges 10/10 at five stacks
without a Trend block, where the ten stack arm converged 9/10. So for the *marginal* basis, depth does
carry a little risk. For the two bases that actually fail, depth is irrelevant in both directions.

**4. The repaired arms land within about 0.2 of each other. CONFIRMED on the tolerance, overstated in
wording.** The spread is 0.180. But "statistically indistinguishable", which is how I phrased the
underlying reasoning, is too strong: KL against band ordered db3 is significant at p = 0.0091, and
survives Bonferroni over three comparisons at 0.027. DCT against band ordered db3 is a genuine null at
p = 0.68. The honest statement is that basis identity is worth at most 0.18 sMAPE once training
succeeds, against the 19.6 it is worth when training fails.

**5. Band ordered db3 becomes the best ag0 arm. HALF WRONG.** The half that held is the important half:
the single failing seed disappears, and the mean falls from 9.569 to 9.132 while the converged median
barely moves, 9.194 to 9.129. That is precisely the "failure rate, not shifted mean" signature the
conditioning account predicts, and it is the cleanest demonstration of it anywhere in this work. The
half that failed is the ranking claim. I predicted band ordered db3 would end up uniquely best; it ties
with Trend + DCT at 9.132 against 9.111, p = 0.68. I also registered a mean of 9.15 to 9.25 and measured
9.132, marginally below my own band.

## Why capacity does not explain this

The trend paired arms carry 2,717,455 parameters against 1,418,240 for the five stack controls, a 1.9x
difference. That confound is closed by the trunk width study run the same day, which found **no
detectable accuracy difference across a 48.8x parameter range** on M4 (Kruskal p = 0.776 and 0.773).
A 1.9x change in capacity cannot plausibly move sMAPE by 19.6 when a 48.8x change moves it by 0.098.

The caveat applies honestly to the small arm, though: for band ordered db3 the effect is only −0.157,
and there capacity is not excluded as a partial contributor.

## What this adds

The conditioning account now has three independent interventions that repair the same failure by the
same mechanism, none of which adds expressiveness:

1. an output rectifier (`agf`), which supplies a DC level for free: DCT 27.054 → 9.102, KL 23.472 → 9.081
2. **a Trend block, which subtracts the DC level before the basis sees it: DCT 28.748 → 9.111, KL 27.607 → 9.291**
3. normalising the data, which removes the concentration at source and is why Traffic never exhibits it

This matters for the paper beyond the mechanism, because the architecture that actually ships,
Trend paired with a frozen basis, is immune to the failure by construction. The pathology is real,
reproducible and precisely predicted, and it is also already solved by the standard configuration.

## Reproduction

```bash
python experiments/scripts/basis_conditioning.py --dataset m4 --period Hourly
python experiments/run_from_yaml.py experiments/configs/m4_trend_repairs_conditioning.yaml
```

Rows in `experiments/results/m4/m4_trend_repairs_conditioning_results.csv`, 60 of 60, no divergences
flagged by the runner in any arm.
