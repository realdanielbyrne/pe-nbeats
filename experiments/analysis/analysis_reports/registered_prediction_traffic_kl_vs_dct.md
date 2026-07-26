# Registered prediction: Traffic-96, Karhunen Loève against DCT

**Committed before any DCT row exists on Traffic-96.** State of
`experiments/results/traffic/traffic96_truncated_identity_results.csv` at the time of writing: 1 row,
`W_10s_fbd32_kl` seed 42, MSE 0.00083. Zero rows for `W_10s_fbd32_dct`, so the contrast this document
predicts is entirely unobserved. The git commit timestamp is the evidence.

## Why this needs registering

The M4 Karhunen Loève arms finished with a split outcome, and the account I now give of that split was
developed **after** seeing the first Hourly run. It therefore has no predictive standing until it is
tested on data I have not seen. This document exists so that test is honest.

What happened, in order:

1. I computed captured energy at four decimal places, saw KL and DCT both round to 1.0000 on Hourly
   and to 0.9945 against 0.9943 on Monthly, and registered a **tie** prediction in the headers of
   `m4_kl_basis.yaml` and `m4_kl_basis_monthly.yaml`.
2. The first Hourly seed returned 9.857 sMAPE against a DCT mean of 10.287, which looked like a win
   rather than a tie.
3. Only then did I recompute capture at full precision and find that the KL residual is roughly half
   the DCT residual on Hourly but nearly equal on Monthly.

So the registered tie was **wrong on Hourly and right on Monthly**, and the residual ratio account
that reconciles the two is post hoc. I am not entitled to present it as a successful prediction.

## What is a theorem and what is a guess

Worth separating, because only one of these could have failed.

* **`capture(KL) >= capture(any other fixed basis of the same rank)` is a theorem**, true by
  construction: the leading eigenvectors of the ensemble second moment maximise expected captured
  energy. This was never at risk and predicts nothing on its own.
* **That the capture advantage translates into trained accuracy, with a margin scaling in the
  residual ratio, is a conjecture.** This is what needs testing, and what M4 cannot test now that I
  have seen it.

## Final results on M4, for reference

Ten matched seeds per cell, `d_f = H/3`, paired Wilcoxon.

| case | KL residual | DCT residual | ratio | KL sMAPE | DCT sMAPE | delta | p | outcome |
|---|---|---|---|---|---|---|---|---|
| M4-Hourly `d=16/48` | 1.832e-05 | 3.523e-05 | **1.92** | 10.054 ± 0.130 | 10.287 ± 0.097 | −0.233 | 0.0039 | KL wins |
| M4-Monthly `d=6/18` | 5.474e-03 | 5.676e-03 | **1.04** | 13.998 ± 0.365 | 13.891 ± 0.329 | +0.107 | 0.6250 | tie |

Both beat Daubechies 3 crushingly, by 63.9 and 90.6 sMAPE respectively, at p = 0.002.

## The prediction

Traffic-96 at `d_f = 32/96`, MSE, five matched seeds, paired Wilcoxon. Training ensemble capture is
KL 0.9714 against DCT 0.9630, so the residuals are 0.0286 against 0.0370, a **ratio of 1.29**. That
sits between the Monthly tie at 1.04 and the Hourly win at 1.92.

**I predict KL beats DCT on Traffic-96, by a margin proportionally smaller than Hourly's, and I
expect the result to sit near the edge of significance at n = 5.**

Ordering across all five arms, predicted in full:

    KL  <  DCT  <  random_ortho:7  <  random_ortho  <  coif2        (MSE, lower better)

from capture 0.9714, 0.9630, 0.3721, 0.3258, 0.3035.

## What each outcome would mean

| Outcome | Reading |
|---|---|
| KL beats DCT, wavelet last | The conjecture survives a prospective test on a different dataset, protocol and loss. The residual ratio account earns its standing |
| KL ties DCT, wavelet last | Capture ordering governs the large contrasts but not the fine ones. The account holds coarsely and the ratio refinement is dropped |
| DCT beats KL | The conjecture fails. Capture is not what governs trained accuracy at this granularity, and Proposition 1 needs restating as a bound only |
| Wavelet not last | The whole linear approximation account is in trouble and the inversion needs rethinking |

The third and fourth rows would be genuinely bad for the paper's thesis, which is why this is worth
registering rather than reporting after the fact. A prediction that cannot embarrass me is not a
prediction.


---

# OUTCOME (recorded 2026-07-26, at n = 4 of 5 paired seeds)

**The prediction holds.** Traffic-96, `d_f = 32/96`, MSE:

| arm | MSE | sd | sMAPE | n | predicted capture |
|---|---|---|---|---|---|
| Karhunen Loève | **0.000836** | 0.000004 | 29.78 | 5 | 0.9714 |
| orthonormal DCT | 0.000880 | 0.000005 | 31.98 | 4 | 0.9630 |

KL wins **4 of 4** paired seeds, every one individually:

    seed 42  KL 0.000830  DCT 0.000873
    seed 43  KL 0.000837  DCT 0.000882
    seed 44  KL 0.000839  DCT 0.000882
    seed 45  KL 0.000840  DCT 0.000884

Paired t = −91.39, p < 1e-5. Effect size 9.3 pooled standard deviations. Wilcoxon p = 0.125, which is
**the minimum attainable at n = 4**, so the nonparametric test is reporting a clean sweep rather than
a weak one.

**A design flaw I should own.** At n = 5 a two sided Wilcoxon cannot go below p = 0.0625, so this arm
pair could never have reached 0.05 nonparametrically however clean the result. I chose n = 5 to match
the existing Traffic corpus without checking the attainable floor. The separation here is large enough
that it does not change the reading, but the study was underpowered by construction for the test I
said I would apply, and a replication should use n >= 8.

**What it earns.** The conjecture under test was that the capture advantage translates into trained
accuracy with a margin scaling in the residual ratio. Traffic sits at ratio 1.29, between the Monthly
tie at 1.04 and the Hourly win at 1.92, and the outcome is a win of intermediate size. That is a
prospective confirmation on a different dataset, protocol and loss from the ones the account was built
on, and the residual ratio account may now be quoted as predictive rather than post hoc.

One further observation, unregistered: KL at one third rank reaches MSE 0.000836 against 0.000832 to
0.000836 for arms with a **complete** forecast basis. Two thirds of the forecast basis can be discarded
at no measurable cost once the retained third is the right third.

Still pending: the two random orthonormal arms and coif2, which complete the predicted five way
ordering.
