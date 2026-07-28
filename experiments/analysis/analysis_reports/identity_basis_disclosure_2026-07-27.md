# Several configurations labelled "wavelet" used the identity matrix (2026-07-27)

Found while deciding a default basis for practitioners. It is an error in the submitted paper, it
affects a large fraction of the reported cells, and it partly explains the paper's own central null.

## What happens

A discrete wavelet transform needs enough samples to decompose. `pywt.dwt_max_level(H, filter_length)`
returns 0 when the target is shorter than the filter can act on, and `_WaveletGeneratorV3` correctly
respects that rather than forcing an invalid decomposition. The consequence, which I did not follow
through, is that the resulting orthonormal basis has no multiscale structure at all. Inspecting the
shipped forecast basis directly, every row has support 1: **it is the identity matrix.**

| period | H | haar | db2 | db3 | coif2 | sym10 |
|---|---|---|---|---|---|---|
| Yearly | 6 | wavelet | wavelet | **IDENTITY** | **IDENTITY** | **IDENTITY** |
| Quarterly | 8 | wavelet | wavelet | **IDENTITY** | **IDENTITY** | **IDENTITY** |
| Weekly | 13 | wavelet | wavelet | wavelet | **IDENTITY** | **IDENTITY** |
| Daily | 14 | wavelet | wavelet | wavelet | **IDENTITY** | **IDENTITY** |
| Monthly | 18 | wavelet | wavelet | wavelet | **IDENTITY** | **IDENTITY** |
| Hourly | 48 | wavelet | wavelet | wavelet | wavelet | wavelet |

So **coif2 and sym10 were the identity on five of six M4 periods, and db3 on two.** Only haar and db2
were genuine wavelets throughout, and only Hourly exercised the full shortlist.

## What it does and does not invalidate

**Accuracy numbers stand.** Those models trained and scored what they scored; nothing is retracted
numerically.

**Labels and interpretation do not.** A configuration named `TW_30s_td3_bdeq_sym10` on Monthly was not
using a sym10 wavelet on the forecast path. It was a trend branch plus a plain linear forecast head.
Every claim of the form "family X helps on period Y" for coif2 or sym10 outside Hourly was describing
the identity.

## It is corroboration, not just an error

Proposition 2 says that at complete forecast rank a freely learned projection absorbs any rotation, so
basis identity cannot change what the model represents. The strongest possible instance of that claim
is a basis that *is* the identity, and those cells score the same as the genuine wavelets beside them.
The family null was measured across 36 paired comparisons at minimum p = 0.084; part of the reason is
now mechanical, in that several of the families being compared were the same matrix.

This cuts both ways and both should be said. It weakens the null as *evidence* about wavelets, since
fewer distinct families were actually tested than the tables imply. It strengthens the proposition,
since the identity is the limiting case and behaves exactly as predicted.

## Consequences for the paper

1. Disclose the table above, and mark every affected cell in the results.
2. Restate the family null over the families that were genuinely distinct on each period. On Hourly all
   five were real, and the null holds there on its own.
3. Drop coif2 and sym10 from the M4 shortlist outside Hourly, since they cannot be instantiated there.
4. Raise an error or a warning rather than silently returning the identity. Returning a valid but
   meaningless basis is worse than refusing.

## Consequence for practitioners, which is why this matters beyond the paper

A user with quarterly or yearly data, which is most business forecasting, who selects a sensible
looking wavelet family gets the identity and no warning. This is the deciding argument for **DCT as the
default basis**: it is defined at every horizon, has no minimum length, and cannot silently degenerate.
The recommendation now carries the table above.
