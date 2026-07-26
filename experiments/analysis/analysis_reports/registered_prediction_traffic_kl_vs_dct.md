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
