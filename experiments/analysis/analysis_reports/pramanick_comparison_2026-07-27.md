# DWT preprocessing against an in block frozen basis, under one protocol (2026-07-27)

Three parties asked for this comparison and none of my work contained it: reviewer ovPT Question 1,
reviewer K8yb Weakness (i), and the area chair listed it second of five rebuttal considerations. It is
now run: 420 runs, both designs swept across three parameter budgets, all six M4 periods, ten seeds,
one protocol, matched seeds.

**The in block frozen basis wins on all ten budget matched comparisons on the median, sign test
p = 0.0020.** Two of my four registered predictions failed, and both are reported below rather than
rescoped.

## The two designs

**Theirs.** Decompose the input series with a discrete wavelet transform and train a separate Generic
stack on each resulting sub-band. The wavelet is an external preprocessor and the network never sees
one again.

**Mine.** Leave the input alone and replace each block's linear synthesis head with a factorisation
through a frozen orthonormal basis. The wavelet is inside the block and there is no preprocessing.

`DWTPreprocessNBeats` splits the length `L` input into `level + 1` components summing back to it
exactly, using a precomputed `L x L` matrix per band, then routes each to its own stack of Generic
blocks with the standard doubly residual topology and sums the per band forecasts. Reconstruction error
is 2.4e-07, so the decomposition discards nothing.

## Fairness, and one bug I introduced and caught

Identical protocol, loss, optimizer, schedule, early stopping, seeds and evaluation code throughout.
Both designs swept across three widths rather than pinned at one point, so a poor width choice cannot
decide the outcome. db3 for both, stated rather than tuned.

**The bug.** My first analysis paired arms by *width label*, `gw128` against `gw128`. The two designs
spend parameters differently, so on M4-Yearly that pairs 348,672 parameters against 1,309,470, a 3.8x
handicap presented as a matched cell. It inflated my margin from +0.143 to +3.209. Every comparison
below instead pairs each DWT arm with whichever in block arm is nearest in **actual parameter count for
that period**, prints the residual gap, and discards any pair more than 25 percent apart.

## Results

| period | budget gap | DWT mean | in block mean | DWT median | in block median | Δ median | p |
|---|---|---|---|---|---|---|---|
| Yearly | 1.7% | 13.810 | 13.667 | 13.802 | 13.719 | **+0.083** | 0.054 |
| Yearly | 3.0% | 13.897 | 13.580 | 13.888 | 13.575 | **+0.313** | 0.0008 |
| Quarterly | 16.4% | 10.442 | 10.118 | 10.434 | 10.107 | **+0.328** | 0.0002 |
| Quarterly | 23.0% | 10.440 | 10.133 | 10.456 | 10.130 | **+0.327** | 0.0002 |
| Monthly | 9.0% | 14.172 | 13.876 | 14.107 | 13.861 | **+0.246** | 0.121 |
| Weekly | 14.2% | 10.740 | 7.277 | 7.795 | 7.240 | **+0.555** | 0.014 |
| Weekly | 23.8% | 9.216 | 7.020 | 7.708 | 6.884 | **+0.824** | 0.0017 |
| Daily | 13.8% | 9.142 | 3.214 | 3.646 | 3.240 | **+0.406** | 0.0010 |
| Daily | 23.9% | 6.261 | 22.983 | 3.467 | 3.364 | **+0.103** | 0.162 |
| Hourly | 20.3% | 11.460 | 9.337 | 10.615 | 9.308 | **+1.308** | 0.0002 |

Positive favours the in block design. **Ten of ten on the median (p = 0.0020), nine of ten on the mean
(p = 0.0215).**

**The budget gaps run in both directions, which strengthens the result.** At the small rung the in
block arm is the larger one, by 14 to 20 percent, so those cells flatter me. But at the large rung on
Quarterly, Weekly and Daily the **DWT arm carries 23 to 24 percent more parameters and still loses**,
by 0.327, 0.824 and 0.103 on the median. The design does not win when handed a parameter advantage.

**Against the Generic baseline, DWT preprocessing never wins on any period**, ranging from +0.21 on
Yearly to +2.70 on Hourly, and significant nearly everywhere. The in block arms match the baseline on
Yearly (−0.023, p = 0.68) and Quarterly (−0.073, p = 0.064) and lose modestly on Hourly (+0.347).

## The one cell where the mean favours their design, and why it does not count

Daily at the large rung reads −16.722 on the mean. That is entirely one seed of **my** arm collapsing:

```
T+Db3V3_10s_gw512, Daily:   3.13  3.13  3.214  3.221  3.327  3.400  3.433  3.468  3.507  200.0
DWTpre_db3_2b_gw512, Daily: 3.32  3.36  3.387  3.421  3.462  3.471  3.480  3.483  3.676  31.554
```

Nine of my ten seeds sit in 3.13 to 3.51 and one lands at **exactly 200.0**, the sMAPE ceiling. On the
median my arm still wins the cell, 3.364 against 3.467. This is the only run at exactly 200 in all 420,
and it is mine. It is also exactly the failure mode documented elsewhere in this work: a negative
forecast on a strictly positive series, which sMAPE scores at its maximum regardless of magnitude. So
the single cell where the preprocessing design appears to win is my own instability, and the account
that predicts it is my own.

## Registered predictions, scored

**1. The two designs land within about 0.15 sMAPE of each other at comparable budgets. FAILED.** Only
1 of 10 comparisons falls within 0.15. I predicted parity on the reasoning that at complete forecast
rank a learned projection absorbs any rotation, so neither wavelet placement should matter. That
reasoning was about the *synthesis* basis and does not transfer to a design that decomposes the
*input*: splitting the input across independent stacks removes each stack's access to the other bands,
which is a real architectural constraint rather than a change of coordinates. The prediction was wrong
and the direction favours me, which is exactly when a failed prediction most deserves stating.

**2. Neither design beats the Generic baseline meaningfully at its own budget. HALF FAILED.** True of
the in block design on four periods. False of the preprocessing design, which is significantly worse
than the baseline nearly everywhere.

**3. Both reach within about 0.2 sMAPE of the baseline at roughly 1.3M parameters. MIXED.** The in
block arm does on Yearly (+0.064) and Quarterly (−0.073); the preprocessing arm does not, anywhere.

**4. The preprocessing design suffers most at short horizons, where few decomposition levels are
admissible. FAILED, and in the opposite direction.** Mean gap by horizon: Yearly H=6 +0.230,
Quarterly H=8 +0.316, Monthly H=18 +0.296, Weekly H=13 +2.830, Hourly H=48 +2.123. The gap is
*smallest* at the short horizons and largest at the long ones. My reasoning was that a coarse sub-band
split would hurt; the data says the opposite, and the plausible reading is that long horizons need the
cross band interaction that independent per band stacks cannot provide.

**Falsification condition: preprocessing wins by more than 0.3 sMAPE on four or more periods. NOT
MET** — one period on the mean, zero on the median, and that one is my own collapsed seed.

## Scope, stated plainly

This is my reimplementation from a published description, without the authors' code, on data their
paper did not use, under a protocol they did not choose. Their evaluation is on stock market series.
The mitigations available to me were taken and are listed above, but a reader should weight this as
what it is: a good faith reimplementation by an interested party, not an independent replication.

I also did not sweep the wavelet family for their design. My own family test is a well powered null
(36 paired comparisons, minimum p = 0.084), which is the reason, but it remains an untested assumption
for their architecture specifically.

## Reproduction

```bash
python experiments/run_from_yaml.py experiments/configs/m4_pramanick_comparison.yaml
python experiments/scripts/analyze_pramanick.py
```

420 of 420 rows in `experiments/results/m4/m4_pramanick_comparison_results.csv`.
