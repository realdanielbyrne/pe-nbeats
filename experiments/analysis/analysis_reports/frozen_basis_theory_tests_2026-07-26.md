# Frozen basis theory tests, M4 and Traffic (2026-07-26)

Consolidated results from eight studies run to test, rather than illustrate, the account of what a
frozen orthonormal synthesis basis buys. Every study registered its prediction in its config header
before training. Predictions that failed are recorded as failures.

Protocol throughout: `nbeats_paper`, paper step LR, seeds 42 to 51, ten seeds per cell unless stated.
Arms within a comparison are parameter identical by construction.

---

## 1. Captured energy predicts the ordering, in both activation regimes

M4-Hourly, pure WaveletV3, ten stacks, truncated forecast rank `d_f = 16` of 48. Capture is computed
from the basis geometry alone with no training.

| basis | captured energy | ag0 sMAPE | agf sMAPE | agf gain |
|---|---|---|---|---|
| Karhunen Loève | 0.999981 | 10.054 ± 0.130 | **9.063 ± 0.139** | −0.991 |
| orthonormal DCT | 0.999964 | 10.287 ± 0.097 | 9.263 ± 0.192 | −1.024 |
| band ordered db3 | 0.999885 | 10.587 ± 0.091 | 9.369 ± 0.129 | −1.218 |
| SVD ordered db3 | 0.347765 | 73.946 ± 4.480 | 10.526 ± 0.355 | −63.420 |

**The capture column reproduces the sMAPE ordering without a single inversion, in both columns.** It
does so even where capture separates only in the fifth and sixth decimal place. This is the strongest
form of the claim available: a training free scalar that ranks configurations correctly before any
GPU time is spent.

## 2. An output activation voids the subspace bound

Proposition 1 bounds a block's error below by the target energy outside `row(B)`. That presumes the
block emits `B^T theta`. Under `active_g` it emits `sigma(B^T theta)`, and a rectified vector is not
confined to `row(B)`, so the bound does not apply. The escape is worth most where capture is worst.

Computed with no training, best achievable relative error of one block on real M4-Hourly targets:

| basis | capture | linear head | ReLU head | reduction |
|---|---|---|---|---|
| SVD db3 | 0.346 | 0.8089 | 0.6058 | 25.1% |
| random orthonormal | 0.306 | 0.8332 | 0.5664 | 32.0% |
| band ordered db3 | 0.983 | 0.1321 | 0.1262 | 4.4% |
| DCT | 0.987 | 0.1154 | 0.1071 | 7.2% |

The trained runs agree: the agf gain in Section 1 is −63.4 for the one low capture basis and −1.0 to
−1.2 for the three high capture ones. Spread across bases falls from 63.892 to 1.463, a 97.7%
compression, **but the ordering is untouched and identity remains significant, KW p = 1.7e-06.**

Consequence for the paper: every statement of the capture law needs "under linear synthesis" attached
where magnitudes are quoted. The ordering claim survives without qualification.

## 3. The identity null has an address

M4-Hourly, pure WaveletV3, **complete** forecast rank `d_f = H = 48`, ten seeds, three placements of
the nonlinearity relative to the frozen basis.

| basis | none (linear) | after basis (agf) | between (probe) |
|---|---|---|---|
| db3 | 9.309 ± 0.147 | 9.072 ± 0.200 | 44.045 ± 3.247 |
| DCT | 27.054 ± 5.691 | 9.102 ± 0.207 | 15.036 ± 0.651 |
| random orthonormal | 9.333 ± 0.192 | 9.164 ± 0.147 | 55.523 ± 3.151 |
| random orthonormal:7 | 9.299 ± 0.197 | 9.206 ± 0.172 | 49.484 ± 2.496 |
| **identity across bases** | KW p = 6.4e-05, spread 17.755 | **KW p = 0.457, spread 0.133** | KW p = 1.9e-07, spread 40.487 |

**Registered prediction CONFIRMED.** Interposing the activation between projection and basis destroys
the null and the accuracy together. The rotation absorption argument requires that no nonlinearity sit
there, and that is exactly where it breaks.

**Registered prediction FAILED.** I predicted the null would hold equally with no activation and with
activation after the basis. It holds beautifully in the second case and fails in the first, entirely
because of an ag0 plus DCT optimisation pathology. The null is therefore an
activation after basis result; under pure linear synthesis it holds for three bases of four.

### The DCT pathology, and why it is not a counterexample

At complete rank the four bases are provably equal in what they can represent, so 27.054 against 9.309
cannot be a representational difference. Two devices that leave expressiveness untouched both repair
it: an output activation (27.054 to 9.102, p = 0.002) and a trend branch (9.200 in the identity
control). That both work is the evidence the failure is optimisation. **Proposition 2 guarantees equal
expressiveness, not equal trainability**, and the paper must say so.

**Monthly replication identifies the failure mode.** There the same cell is bimodal: every basis has
seeds converging to between 13.3 and 14.7, and only the failure rate differs, 0 of 10 for
random ortho:7, 1 of 10 for db3, 2 of 10 for random ortho, 4 of 10 for the DCT. On Hourly the DCT
instead fails on all ten. So **basis identity does not move the converged solution, it moves the
probability of converging**, which is exactly what an expressiveness guarantee without a trainability
guarantee predicts, and it is the same bimodal divergence the paper already documents for
unconstrained linear heads.

Unexplained and recorded as such: interposition reorders the bases, with DCT best at 15.036 while db3
and the random draws collapse to 44 and 55. Captured energy does not govern that regime.

## 4. Spatial coverage, not locality, is what truncation costs a wavelet

Captured energy for a band ordered wavelet grows almost linearly in `d` inside a band and jumps at the
boundary, because each retained translate covers only a fraction of the time axis.

| cell | capture | sMAPE |
|---|---|---|
| Hourly `d=5`, inside a band of 6 | 0.831 | 44.484 ± 0.060 |
| Hourly `d=5`, DCT at matched rank | 0.995 | 18.455 ± 0.090 |
| Hourly `d=16`, past the boundary at 12 | 0.9999 | 10.587 ± 0.091 |
| Monthly `d=6`, inside a band of 9 | 0.658 | 73.380 ± 6.508 |
| **Monthly `d=9`, exactly on the boundary** | 0.996 | **13.850 ± 0.377** |
| Monthly `d=9`, DCT at matched rank | 0.997 | 13.720 ± 0.234 |

Three extra basis rows are worth **59.5 sMAPE** on Monthly, which no rank argument alone explains and
which the coverage argument predicts. Once coverage is complete the wavelet reaches parity with the
DCT, 13.850 against 13.720.

## 5. Retraction: the wavelet was never given a fair test

The shipped basis path SVDs the raw DWT synthesis matrix and orders rows by singular value, which
destroys the dyadic ordering, so truncation keeps an arbitrary subspace. Building in periodization
mode and orthonormalising by QR preserves the ordering (nesting residual 1.3e-15 against 4.5 for SVD)
and lifts Hourly `d=16` capture from 0.348 to 0.999885 and sMAPE from 73.946 to 10.587.

**The claim that a wavelet finishes behind random orthonormal draws is withdrawn.** It measured the
orthogonalisation, not the wavelet. The `basis_offset` mechanism retraction is likewise partial: under
band preserving construction the offsets do select genuine frequency bands.

## 6. Offset tiering and energy ordered bases are incompatible

An offset discards the leading rows, which is survivable only if the ordering is not aligned with where
the energy sits. Mean per block capture across the five Hourly tiers:

| basis | offset tiering | rank tiering |
|---|---|---|
| SVD db3 | 0.645 | 0.726 |
| DCT | 0.200 | 1.000 |
| KL | 0.200 | 1.000 |
| band ordered db3 | 0.200 | 1.000 |

So the SVD scrambling that Section 5 calls a defect is precisely what makes offset tiering viable. The
two facts are the same fact. `stack_basis_dims` adds the nested alternative, rows `[0 : d_k]`.

**Registered prediction FAILED.** I predicted DCT plus offset would be far worse than db3 plus offset.
Measured 8.971 (n=6) against 8.885 (n=10), a gap of 0.086. It failed because that study runs at agf,
and by Section 2 an output activation voids the subspace bound, so per block capture stops dominating.
The prediction was registered before that interaction was understood.

## 7. The Hourly parity claim is a selection artifact

The reported best frozen basis arm on Hourly, `T+DB3V3_30s_tiered_agf` at 8.751, is the best of sixteen
configurations in its study and its cell holds n = 8 of 10. An independent rebuild of the same
configuration gives **8.885 ± 0.221 at n = 10**, which is worse than the 8.758 baseline. The study's
runner up, 8.921, also loses. Hourly should be reported as near parity, selection limited.

## 8. When the capture floor does not bind

Traffic-96 KL at `d_f = 32` of 96, capture 0.9714, gives MSE 0.000837 (n = 4) against 0.000832 to
0.000836 for complete forecast arms: indistinguishable, despite 2.9% of energy being unreachable. The
subspace floor only binds when the tail energy exceeds the error the model would otherwise achieve. On
M4-Hourly with db3 at capture 0.35 the tail is 65% and dominates; on Traffic at 0.97 it is irrelevant.

## 9. The Karhunen Loève arm

KL beats the DCT where their residuals genuinely separate and ties it where they do not, which is what
the optimality of KL requires.

| period | KL residual | DCT residual | ratio | Δ sMAPE | p | outcome |
|---|---|---|---|---|---|---|
| Hourly `d=16/48` | 1.83e-05 | 3.52e-05 | 1.92 | −0.233 | 0.0039 | KL wins |
| Monthly `d=6/18` | 5.47e-03 | 5.68e-03 | 1.04 | +0.107 | 0.6250 | tie |

The registered prediction was a tie on both, made from capture rounded to four decimals, and was too
weak on Hourly. The residual ratio account reconciling the two was developed after seeing the first
Hourly seed and is therefore post hoc; a prospective test is registered for Traffic-96 in
`registered_prediction_traffic_kl_vs_dct.md` and is still running.

---

## Standing summary

| claim | status |
|---|---|
| Captured energy orders configurations correctly, before training | Holds, both activation regimes, no inversions |
| Error is bounded below by tail energy outside the subspace | Holds under linear synthesis only; an output activation voids it |
| Basis identity is irrelevant at complete rank | Holds cleanly under agf, spread 0.133; ag0 has one optimisation pathology |
| The completeness argument's precondition is no nonlinearity between projection and basis | Confirmed by interposition, spread 40.5 |
| Wavelets lose to random subspaces | **Withdrawn**, an artifact of SVD row ordering |
| Locality is a liability under fixed truncation | Holds, and the mechanism is spatial coverage |
| Compression comes from the backcast head and trunk | Holds, 96% of the Fourier baseline gap is trunk width |
| Hourly reaches parity | Weakened to near parity, selection limited |

## Still running

`m4_p2_boundary` Monthly half (120 runs), `m4_hourly_tiered_rank` (66 of 100 remaining),
`traffic96_truncated_identity` (15 of 25 remaining). Its registered prospective test has RESOLVED in
favour of the account: Karhunen Loève 0.000836 against DCT 0.000880, winning 5 of 5 paired seeds at
9.3 pooled sd, paired t p = 1.8e-07. Wilcoxon gives 0.0625, the minimum attainable at n = 5, so that
arm pair was underpowered by construction for the test I registered; replication needs n >= 8.
