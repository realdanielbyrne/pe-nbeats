# Narrative status, 2026-07-27: what the five earlier recommendation documents got wrong

There are five narrative documents in this directory, all dated 2026-05-02, all written before the
NeurIPS reviews and before the experiments run since. **Every one of them foregrounds a thesis the
evidence has since contradicted.** They are kept for traceability and should not be used to guide
writing. This file is the current position, and each of the five now carries a banner pointing here.

| document | its primary thesis | status |
|---|---|---|
| `NARRATIVE_RECOMMENDATIONS.md` | "The Cost of Flexibility: the Generic block is dangerous; structured alternatives eliminate failure modes and achieve 10 to 50 times compression" | half survives, see below |
| `PAPER_NARRATIVE_RECOMMENDATIONS.md` | "Wavelet/TrendWavelet families dominate or tie best M4 results across most periods" | **dead** |
| `COMBINED_NARRATIVE_RECOMMENDATIONS.md` | "Multi-resolution N-BEATS", twin Tier-A claims | **dead** |
| `COMBINED_PAPER_NARRATIVE.md` | same, plus LLM applicability | **dead**, and the LLM material leaves the paper |
| `claude_narrative_recs.md` | wavelets plus bottlenecks as the two central contributions | **dead** |

## The four claims that died, and what replaces each

**1. "Wavelets dominate M4."** Dead. At the complete forecast rank every M4 configuration in the paper
uses, the frozen basis is square and orthogonal, so a freely learned projection absorbs any rotation
and basis identity provably cannot change what the model represents. Measured: 36 paired comparisons
at identical parameter counts, none significant, minimum p = 0.084. Replacement claim: **captured
energy, not basis family, governs accuracy, and it is computable before training.**

**2. "Multi-resolution is the mechanism."** Dead twice over. The shipped construction orthogonalises by
SVD and orders rows by singular value, which destroys the dyadic level ordering, so there was never
any multiresolution selection happening. And even with the ordering repaired, selecting rows at build
time is *linear* approximation, where the wavelet advantage, a best `m` term theorem requiring per
signal adaptivity, does not exist. Replacement claim: **locality costs spatial coverage under fixed
truncation**, measurable as 59.5 sMAPE for three basis rows at a Monthly band boundary.

**3. "Wavelets plus autoencoders deliver 10 to 50 times compression."** Reattributed, not dead. The
compression is real but does not come from the basis. Ninety six percent of the gap against the Fourier
baseline is **trunk width**, the Seasonality block carrying 13.08M of its 13.67M parameters in a 2048
wide trunk, and the remainder comes from the backcast head, which dominates because `L = 5H`. Every
basis compresses identically because a frozen matrix has no trainable parameters at all. Replacement
claim: **compression is structural and basis agnostic.**

**4. "Better accuracy."** Withdrawn in favour of a frontier claim. Parity on four of six M4 periods at
nine to sixty eight times fewer parameters, with two significant losses stated in the same table, and
Traffic losing outright at 26.24 against 29.37. The paper claims a position on the accuracy and size
frontier, not an accuracy win.

## The one claim that survives, and is now better supported

**"The Generic block is dangerous"** was the emotional anchor of `NARRATIVE_RECOMMENDATIONS.md`, and it
holds up. What is new is the mechanism and the size of the effect. Across 30,434 M4 runs, the rate of
total collapse to exactly 200 sMAPE, the metric ceiling:

| setting | collapses |
|---|---|
| plain linear synthesis head | **107 / 23,786** (0.45%) |
| `active_g="forecast"` | 4 / 6,372 (0.06%) |
| `active_g=True` | 0 / 136 |

And the mechanism is now identifiable. sMAPE is `200|F-A|/(|F|+|A|)`, so for a positive target **any
negative forecast scores exactly 200 regardless of magnitude**, and so does exactly zero. A converged
30 stack Generic on M4-Daily scores 3.1 while its diverged seeds score 17.4, 31.2 and 31.6, which
corresponds to 7.3, 14.3 and 14.5 percent of points sitting at the ceiling. An unconstrained linear
head can emit negatives; a rectified one cannot. That is why the intervention works, and it is a
sharper story than "flexibility is dangerous" because it names the failure and predicts where it
appears. `m4_divergence_signs.yaml` is saving per run forecasts to test it directly.

## Clarifications the earlier documents get wrong in detail

**`active_g` is at least three mechanisms, not one.** The documents treat it as a single stabilising
knob. It is a sign constraint on the output, a subspace escape at truncated rank, and a free DC source
for energy concentrating bases, and which one is operating depends entirely on the configuration.
Critically, `active_g=True` rectifies both paths while `active_g="forecast"` rectifies only the
forecast and leaves the residual cascade untouched, so no explanation appealing to the cascade can
cover both. Full account in
`experiments/analysis/analysis_reports/active_g_mechanisms_2026-07-27.md`.

**`active_g=True` is not free.** It removes collapse and taxes accuracy: on 50 seeds its raw mean beats
baseline only because one baseline seed diverged, and restricted to converged runs it is +0.129 worse
at p = 3.7e-06. Rectifying both paths forbids a block from making a downward correction, in a cascade
whose whole design is iterative correction.

**`sum_losses` is not a substitute stabiliser.** I briefly thought it was and withdrew that. It
collapsed 1 of 5 seeds on Monthly where the baseline did not, and pooled it is 1/70 against 107/23,786,
directionally worse. Separately there is a **measurement bug worth knowing before any of this is
written up**: `validation_step` adds `0.25 * loss_fn(backcast, zeros)` under `sum_losses`, and sMAPE
against zeros is 200 for any nonzero backcast, so that term adds about 50 by construction. Measured
offsets on the 400 run Milk study are 51.077 and 49.956 against a predicted 50.000, so that study
compares two objectives rather than two models, and no `best_val_loss` comparison across that flag
anywhere in the repository is valid.

**Would a leaky rectifier remove the residual zero collapses? No, it would make things worse.** The
four surviving collapses under `agf` must be all zero outputs, since a rectifier cannot emit a
negative, and a leaky rectifier does essentially eliminate exact zeros. But sMAPE punishes a negative
forecast **identically** to a zero one, exactly 200, and the magnitude of the negative is irrelevant:
`F = -0.001` scores 200 just as `F = -1000` does. A leaky rectifier emits negatives whenever its input
is negative, so it would trade the rare zero mode, 4 of 112 collapses, for the common negative mode,
108 of 112. The existing ablation agrees: as a trunk activation, LeakyReLU has the worst collapse rate
of any activation tested, 2 of 5 seeds on Monthly, against zero for ELU and GELU.

What the question is really reaching for is a **strictly positive** output activation, and `Softplus`
is already registered in `constants.ACTIVATIONS`. It emits no negatives and no zeros except under
extreme float underflow, `softplus(-100) = 3.8e-44` rather than 0, so it closes both routes at once. It
is untested here and is the obvious next experiment if output collapse is worth eliminating entirely.
The cost to weigh against it is that softplus never returns exactly zero, so a block can no longer
switch itself off, which may matter in a deep cascade.

## Where to look instead

* Current thesis, title and abstract: `rebuttal/retitle_and_abstract.md`
* Theory tests with registered predictions and their outcomes:
  `experiments/analysis/analysis_reports/frozen_basis_theory_tests_2026-07-26.md`
* The `active_g` and stability account:
  `experiments/analysis/analysis_reports/active_g_mechanisms_2026-07-27.md`
* Prospective prediction and its resolution:
  `experiments/analysis/analysis_reports/registered_prediction_traffic_kl_vs_dct.md`
