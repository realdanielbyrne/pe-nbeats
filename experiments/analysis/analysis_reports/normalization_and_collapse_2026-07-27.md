# Normalisation and total collapse: a claim I could not support, and why (2026-07-27)

This file exists to stop a plausible and attractive claim from entering the paper. The claim is that
normalising the data eliminates the total-collapse failure mode, and it is supported by a comparison
that does not survive inspection. It is recorded here with the two confounds that nearly fooled me and
the third that actually did.

## The claim, and the evidence that seemed to support it

Pooling every run in the repository by dataset:

| dataset | normalize | runs | at exactly sMAPE 200 |
|---|---|---|---|
| Weather | true | 7,896 | **0** |
| Traffic | false | 1,543 | 444 (28.8%) |
| M4 | n/a, raw | 31,528 | 111 (0.35%) |

That looks like a strong effect, and it gets stronger under the first two controls rather than weaker.

**Confound 1, block family. Closed.** Traffic's collapses concentrate in AELG and VAE arms, which are
independently known to be unstable. But Weather ran the same families at scale and never collapsed:
AE 0/2,186, AELG 0/3,069, Generic 0/479, VAE 0/770.

**Confound 2, backcast to horizon ratio. Nearly fatal, and it eliminates the naive reading.** Traffic's
collapses are entirely a short backcast phenomenon:

| Traffic L/H | runs | collapses |
|---|---|---|
| 2 | 1,193 | 444 (37.2%) |
| 5 | 350 | **0** |

So blaming Traffic's rate on the normalise flag is simply wrong. It is explained by L/H without any
reference to scaling. Restricting both datasets to L/H = 2, where the stress actually lives, the
contrast still looked decisive, family matched and with Fisher exact underflowing:

| family | Traffic, normalize=false | Weather, normalize=true |
|---|---|---|
| AELG | 422 / 890 | 0 / 2,171 |
| VAE | 22 / 250 | 0 / 629 |
| all | 444 / 1,193 | **0 / 5,795** |

## Confound 3: the metric is not on the same scale, and this one is fatal

`normalize=True` resolves to `normalization_style='global'`, a **per column z-score**. Z-scored targets
cross zero. sMAPE is `200|F-A|/(|F|+|A|)`, so with a denominator that routinely approaches zero the
metric lands in an entirely different regime, and the Weather corpus shows it:

```
Weather sMAPE over 7,896 runs:  median 65.9   75th 66.7   99th 101.4   MAX 104.4
```

A *good* Weather model scores about 66 where a good M4 model scores about 13, and **no Weather run in
the entire corpus exceeds 104.4**. Reaching exactly 200 requires every forecast point to sit at the
ceiling simultaneously, which on z-scored targets that straddle zero does not happen.

**So Weather's 0 of 7,896 is a property of the metric regime, not evidence about normalisation.** The
comparison is void. Both earlier confound checks were correct and both were beside the point.

## The experiment this kills

I had written `traffic_normalize_collapse.yaml` to settle it causally: identical architecture, seeds,
horizon and backcast, varying only `protocol.normalize`, on the Traffic AELG arm that historically
collapses on 47.4 percent of seeds. It was deleted unrun, because it carries the same defect. The
`normalize=true` arm would be scored on z-scored targets and therefore could not reach 200 for reasons
having nothing to do with stability, so it would have confirmed its own prediction by construction.

A valid version requires denormalising predictions before computing metrics. The loaders already expose
the hook, `return_scales` with `normalization_style='window'`, but the benchmark path does not use it,
and `run_unified_benchmark.py` computes metrics in whatever space the datamodule produced. Until that
is wired, **no cross-normalisation metric comparison in this repository is valid**, which is a second
instance of the same class of bug as the `sum_losses` validation offset.

## What survives

The mechanistic half, which is measured training free and does not depend on any metric:

Normalisation roughly triples the effective rank of the target coefficient distribution on Traffic-96 at
L/H = 2, measured by `experiments/scripts/basis_conditioning.py`:

| basis | eff. rank raw | eff. rank normalised | top-1 share raw → normalised |
|---|---|---|---|
| KL | 2.54 | 7.43 | 0.6120 → 0.2475 |
| DCT | 3.67 | 10.09 | 0.5024 → 0.2077 |
| band ordered db3 | 12.69 | 23.17 | 0.1114 → 0.0755 |
| SVD db3 | 34.04 | 80.84 | 0.1196 → 0.0223 |
| random orthonormal | 58.22 | 80.52 | 0.0474 → 0.0290 |

Since effective rank is what predicts trainability on M4-Hourly, with rho = +0.949 against measured
convergence rate, it is reasonable to expect normalisation to help. **That expectation is a conjecture
and must be labelled one.** It has not been tested end to end anywhere in this work.

## What the paper should claim instead

Nothing about normalisation is needed, because the two interventions that ARE established do the same
job and are measured directly on the metric they are claimed to improve:

| intervention | evidence | metric space |
|---|---|---|
| output rectifier | M4 collapse 107/23,786 → 4/6,372 (`agf`) → 0/136 (`active_g=True`); DCT 27.054 → 9.102; KL 23.472 → 9.081 | raw, unnormalised |
| Trend pairing | DCT 28.748 → 9.111 and KL 27.607 → 9.291, both 0/10 → 10/10 converging, p = 0.00018, with a no-trend control still at 0/10 | raw, unnormalised |

Both are demonstrated on unnormalised data, which is the regime the no-preprocessing claim is about. The
argument is strictly stronger without the normalisation arm, since it never asks the reader to accept a
comparison across metric spaces.
