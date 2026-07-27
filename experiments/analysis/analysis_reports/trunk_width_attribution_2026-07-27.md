# The compression is the trunk, and the trunk is inert across a 49x parameter range (2026-07-27)

Four predictions were registered in `experiments/configs/m4_trunk_width_attribution.yaml` before any run
started. All four are confirmed, and the first is confirmed far more strongly than I dared register. This
study exists because a question exposed that the headline compression comparison in this line of work
varies **two** things at once and only ever names one of them.

## The confound

Every compression number compares a Trend+Seasonality backbone against a Trend+Wavelet backbone. Those
two arms differ in the synthesis basis, which is the thing the paper talks about, and also in the trunk
width, which it does not: `Seasonality` defaults to `s_width=2048` while every wavelet and generic block
defaults to 512. Measured on M4-Yearly at 10 alternating stacks with no weight sharing:

| backbone | trunk | parameters |
|---|---|---|
| Trend+Seasonality | `s_width=2048` | 64,645,150 |
| Trend+Seasonality | `s_width=512` | 5,140,510 |
| Trend+HaarV3 | `g_width=512` | 5,084,190 |

At matched trunk width the two backbones sit **1.1 percent** apart. So the entire 12.6x headline gap is
one integer in a constructor, and the frozen basis contributes essentially nothing to it. That much is
arithmetic rather than a result. The result is whether accuracy survives the narrowing, and nothing in
roughly 30,000 runs had ever tested it.

## Design

M4-Yearly and M4-Monthly, paper sample protocol, 10 alternating stacks, one block per stack, no weight
sharing, `ag0` throughout so nothing depends on the activation, identical seeds across every arm. The
trend branch is pinned at `t_width=256` everywhere. Only the second block type and its trunk width vary.
Ten seeds per cell, 160 runs, zero divergences anywhere.

## Prediction 1: narrowing the Seasonality trunk is close to free

Registered as "within 0.15 sMAPE on Yearly for `s_width` 2048 to 512." Measured at +0.026, and the
prediction holds all the way down to 128 rather than stopping at 512.

| period | 2048 | 1024 | 512 | 256 | 128 |
|---|---|---|---|---|---|
| **Yearly** | 13.645 | 13.609 | 13.671 | 13.680 | 13.707 |
| Δ vs 2048 | — | −0.036 | **+0.026** | +0.035 | +0.061 |
| MWU p | — | 0.521 | 0.910 | 0.734 | 0.273 |
| **Monthly** | 13.893 | 14.057 | 14.162 | 13.993 | 14.071 |
| Δ vs 2048 | — | +0.164 | +0.269 | +0.100 | +0.178 |
| MWU p | — | 0.473 | 0.212 | 0.623 | 0.427 |

Monthly's deltas are larger but non monotone, since 256 beats 512, and no cell approaches significance.
That is the signature of noise rather than of a real width effect.

## Prediction 3: trunk width is inert, full stop

Kruskal-Wallis over all five Seasonality rungs:

| period | H | p | total spread |
|---|---|---|---|
| Yearly | 1.781 | **0.776** | 0.098 sMAPE |
| Monthly | 1.796 | **0.773** | 0.269 sMAPE |

Between 1,323,550 and 64,645,150 trainable parameters, a **48.8x range**, there is no detectable
accuracy difference on either period. The trunk is not four times wider than the task needs. It is
roughly fifty times larger than the task needs, and the paper default sits at the far useless end of a
flat curve.

## Prediction 2: the family null survives width matching

Registered as indistinguishable at every matched rung after Bonferroni over six comparisons. Confirmed
with room to spare; no raw p falls below 0.34.

| period | width | T+Seasonality | T+HaarV3 | Δ | p | Bonferroni |
|---|---|---|---|---|---|---|
| Yearly | 512 | 13.671 | 13.626 | +0.045 | 0.910 | ns |
| Yearly | 256 | 13.680 | 13.605 | +0.075 | 0.571 | ns |
| Yearly | 128 | 13.707 | 13.611 | +0.096 | 0.345 | ns |
| Monthly | 512 | 14.162 | 14.062 | +0.100 | 0.473 | ns |
| Monthly | 256 | 13.993 | 13.936 | +0.057 | 0.385 | ns |
| Monthly | 128 | 14.071 | 13.944 | +0.127 | 0.427 | ns |

Every one of the six sign differences favours the wavelet, which is worth stating honestly rather than
hiding: a sign test on six of six gives p = 0.031, so there may be a genuine effect on the order of
0.08 sMAPE. It is an order of magnitude smaller than the effect the paper claimed, it does not survive
correction on any individual cell, and it does not justify a basis narrative.

## Prediction 4, and the answer to the question that prompted this

**Yes. Legacy N-BEATS reaches the same compression by editing one integer, and it costs nothing
measurable.** Trend+Seasonality at `s_width=128` uses 1,323,550 parameters against 64,645,150 at the
default, a 48.8x reduction, and scores 13.707 against 13.645. The frozen basis is not what buys the
compression, and after this study there is no honest way to claim otherwise.

The falsification condition was that Trend+Seasonality would collapse on narrowing while Trend+Wavelet
held, which would have made the frozen basis the enabler of a narrow trunk. It did not happen. Neither
family degrades, so there is nothing to enable.

## What this does to the paper

It kills the compression claim as an argument for the basis, and replaces it with a considerably blunter
and better supported claim about the architecture: **N-BEATS as published carries roughly fifty times more
trunk parameters than its accuracy requires, and this is invisible in the original work because trunk
width was never swept.** That is a finding about N-BEATS rather than a finding about wavelets, it is
larger in magnitude than anything the paper currently claims, and it is exactly the over-parameterization
framing the revision is being built around.

What survives for the frozen basis is orthogonal to compression and unaffected by this result: the basis
is what makes the *head* cheap and structured, the capture law predicts which basis works before training,
and the conditioning account predicts which ones are trainable under a linear synthesis head. None of
those arguments ever depended on parameter count, and none of them are weakened by learning that the
trunk was the real story all along.

## Reproduction

```bash
python experiments/run_from_yaml.py experiments/configs/m4_trunk_width_attribution.yaml
```

Rows in `experiments/results/m4/m4_trunk_width_attribution_results.csv`, 160 of 160, no divergences.
Required plumbing `s_width` and `ae_width` through `run_from_yaml.py` and `run_unified_benchmark.py`;
before this study neither was reachable from a config, which is the likely reason trunk width had never
been swept.
