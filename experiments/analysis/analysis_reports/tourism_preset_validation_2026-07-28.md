# The recommended defaults, tested out of sample on Tourism (2026-07-28)

240 runs, three horizons, ten seeds, on a dataset used to derive none of the defaults. Four predictions
were registered in `tourism_preset_validation.yaml`. **The headline defaults survive. One rule does not,
and it is the one I was most confident about.**

## Results

| Tourism-Yearly, H=4 | sMAPE | params | | Tourism-Quarterly, H=8 | sMAPE | params |
|---|---|---|---|---|---|---|
| SlimTB rand w128 | **21.688** | 529K | | SlimB haar w128 | **18.008** | 568K |
| SlimTB haar w128 | 21.701 | 529K | | SlimTB haar w128 | 18.274 | 560K |
| SlimTB dct w128 | 21.853 | 529K | | SlimB dct w128 | 18.296 | 568K |
| SlimTB dct w512 | 22.124 | 8.02M | | SlimTB rand w128 | 18.313 | 560K |
| NBEATS-G 10s | 22.251 | 8.11M | | NBEATS-G 10s | 18.383 | 8.34M |
| SlimB haar w128 | 22.388 | 532K | | SlimTB dct w128 | 18.640 | 560K |
| NBEATS-IG 10s | 23.585 | 35.6M | | NBEATS-IG 10s | 18.928 | 36.3M |
| SlimB dct w128 | 28.231 | 532K | | SlimTB dct w512 | 19.262 | 8.14M |

Tourism-Monthly, H=24: SlimB haar 23.065, SlimB dct 23.112, SlimTB haar 23.458, NBEATS-G 23.549,
SlimTB dct 23.721, SlimTB rand 23.756, SlimTB dct w512 23.759, NBEATS-IG 24.129.

## Predictions, scored

**1. SlimTB with DCT within 0.5 sMAPE of the better baseline on all three periods, at roughly an order
of magnitude fewer parameters. CONFIRMED.** Yearly **−0.398** (ahead of the baseline), Quarterly +0.257,
Monthly +0.172; none significant; 14 to 15 times fewer parameters throughout. The defaults transfer to a
dataset that produced none of them.

**2. DCT and haar indistinguishable where both are valid. CONFIRMED per period, with a caveat I am
flagging rather than burying.** No period separates them (p = 0.73, 0.054, 0.10), but DCT is behind haar
on **all three**, by +0.152, +0.366 and +0.264. Three of three in one direction is weak evidence of a
small real effect. The case for DCT was never accuracy, and it survives; the claim that it is *equal* on
accuracy is now marginal.

**3. SlimB, lacking a trend branch, is worse on at least two of three periods. FAILED, and inverted.**

| period | basis | no trend | with trend | p |
|---|---|---|---|---|
| Quarterly | dct | **18.296** | 18.640 | **0.021** |
| Monthly | dct | **23.112** | 23.721 | **0.0073** |
| Monthly | haar | **23.065** | 23.458 | 0.064 |
| Yearly | dct | 28.231 (sd 8.12) | **21.853** (sd 0.46) | 0.79 |

On two of three periods the trend branch makes things **significantly worse**, costing 0.3 to 0.6 sMAPE.
It earns itself on Yearly, and not on the mean: `SlimB_dct` has a *better* median there, 23.898 against
its own mean of 28.231, and a standard deviation of 8.12 against 0.46. The trend branch is buying **tail
protection, not accuracy**, and only where the horizon is very short.

**4. Nothing collapses to the metric ceiling. CONFIRMED.** Zero runs at 200 across all 240; worst run
anywhere is 40.07. The rectified head did its job on non negative targets.

**Bonus, the narrow trunk holds.** w128 beats w512 on all three periods, significantly on Quarterly
(−0.622, p = 0.031). This is the first out of sample confirmation of the trunk result, which until now
rested on M4 alone.

## What must change in the recipe

**The trend branch moves from "always on" to conditional.** I recommended it as free insurance. It is
not free: it costs 0.3 to 0.6 sMAPE on the two longer Tourism horizons. What it buys is protection
against the heavy tail that appears at very short horizons, where `SlimB_dct` on H=4 has a standard
deviation of 8.12. The corrected rule keys it to the diagnostic that already exists: pair a trend branch
when the target's effective rank is low or the window mean dominates, and otherwise leave it off and
save the parameters. Tourism-Yearly at H=4 is exactly the concentrated case; Monthly at H=24 is not.

**Nothing else changes.** Ten stacks, complete forecast rank, narrow trunk, rectified head on non
negative targets, and basis chosen on engineering grounds all survive contact with a dataset that had no
hand in producing them.

## Reproduction

```bash
python experiments/run_from_yaml.py experiments/configs/tourism_preset_validation.yaml
```

240 of 240 rows in `experiments/results/tourism/tourism_preset_validation_results.csv`. The first 170
runs crashed on a rank deficient `ordered:haar` basis at H=24, fixed in `blocks.py` with a level clamp
and a hard error; the affected cells were rerun and no other study used that path at an affected length.
