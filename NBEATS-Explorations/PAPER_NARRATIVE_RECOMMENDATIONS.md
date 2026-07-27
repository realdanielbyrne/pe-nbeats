> **SUPERSEDED, 2026-07-27.** This document predates the NeurIPS reviews and the experiments run
> since, and its primary thesis has been contradicted by them. Wavelet family is a well powered null,
> the multiresolution mechanism does not exist under the shipped construction, and the compression is
> 96 percent trunk width rather than basis. Do not use it to guide writing. See
> [`NARRATIVE_SUPERSEDED_2026-07-27.md`](NARRATIVE_SUPERSEDED_2026-07-27.md) for the current position
> and for which of its claims survive.

# N-BEATS-Lightning Paper Narrative: Structured Recommendations

**Date:** 2026-05-02  
**Synthesis:** Based on `comprehensive_m4_paper_sample`, `sym10_fills`, `tiered_offset`, and `plateau` experiments.

---

## 1. Core Contributions: Tier Rankings

### Tier 1 — Central Scientific Contributions

1. **WaveletV3 orthonormal basis expansion**
   - Replaces polynomial Trend and Fourier Seasonality with multi-resolution, localized bases.
   - Preserves interpretability (fixed, non-learned bases).
   - Empirically central: Wavelet/TrendWavelet families dominate or tie best M4 results across most periods.

2. **Alternating Trend + Wavelet stack composition**
   - Original N-BEATS uses homogeneous Generic or fixed Trend/Seasonality stacks.
   - Best M4 generalist: `T+Sym10V3_30s_bdeq` (mean rank 9.17/68, top-3 on 3 periods).
   - Key insight: dedicated alternating blocks outperform unified blocks and pure Generic.

3. **Structural AE/AELG regularization for efficiency**
   - Pure Generic-AE/AELG are weak; paired with Trend/Wavelet bases they become powerful.
   - AELG learned gates enable parameter efficiency: `ld=16` often matches `ld=32`.
   - Sub-1M champion: `TWAE_10s_ld32_ag0` (0.55M, rank 27.67/68).
   - Parameter-extreme: `TWAELG_10s_ld16_db3_ag0` (0.51M, tied with `ld=32` on 5/6 periods).

4. **`active_g=forecast` as scoped stabilization**
   - Not a global default; Hourly-specific and some Yearly wavelet variants.
   - Convergence stabilization: helps Generic blocks avoid bimodal collapse.
   - Critical distinction: `ag0` is paper-faithful; `agf` is repo-novel extension.

### Tier 2 — Supporting Contributions

5. **Paper-faithful training protocol + sub-epoch validation**
   - `nbeats_paper` sampling, `val_check_interval=100`, `min_delta=0.001` fix early-stopping collapse.
   - Essential for credible paper comparison.

6. **Scheduler robustness**
   - Primary claims use multistep LR (paper-faithful).
   - PlateauLR confirms architecture ordering stability.

7. **Tiered basis offsets**
   - Conceptually promising frequency-band partitioning.
   - Works best on longer horizons (Hourly).
   - Caution: short M4 horizons clamp offsets.

### Tier 3 — Appendix/Negative Results

8. **BottleneckGeneric**—elegant but weak on M4; negative evidence.
9. **Skip connections**—targeted GenericAELG rescue, not broadly applicable.
10. **V1/V2 wavelets, activations, NHiTS, VAE, meta-learning**—V1/V2 as engineering rejects (1 paragraph: ill-conditioning); others to appendix or future work.

---

## 2. Narrative Arc

**Core Thesis:** N-BEATS' residual framework is powerful, but original bases (polynomial, Fourier) are limited. Orthonormal wavelets + dedicated stack composition yield better localized, multi-resolution decomposition.

**Flow:** original bases → identify gaps (local shocks, regime shifts) → introduce WaveletV3 → show alternating beats unified → add AE/AELG efficiency → handle optimization pathologies → conclude: **structured bases beat brute-force capacity**.

---

## 3. Essential Main-Paper Results

**Canonical leaderboard** (comprehensive_m4_paper_sample + sym10_fills):
- Per-period SMAPE/OWA/params, mean rank across 6 periods
- Label all `agf` as "repo-novel extension"

**Per-period SOTA:** Yearly `TALG+DB3V3ALG_10s_ag0` (1.04M) | Quarterly `T+HaarV3_10s_bdeq` (5.13M) | Monthly `TW_30s_td3_bdeq_haar` (6.78M) | Weekly `T+Coif2V3_30s_bdeq` (15.75M) | Daily `TAELG+Coif2V3ALG_30s_ag0` (3.73M) | Hourly `NBEATS-IG_30s_agf` (43.6M)

**Generalist:** `T+Sym10V3_30s_bdeq` (rank 9.17/68) | Paper-faithful: `NBEATS-IG_30s_ag0` (rank 11.17/68)

**Parameter efficiency:** Sub-1M `TWAE_10s_ld32_ag0` (0.55M) | Parameter-extreme `TWAELG_10s_ld16_db3_ag0` (0.51M)

**Architecture family:** Alt Trend+Wavelet dominates | Unified TrendWavelet competitive | AE/AELG improves efficiency | Pure Generic-AE/AELG fails

**Wavelet + depth:** Shortlist haar/db3/coif2/sym10 | **Drop coif3** | Haar ≈ 10s | Long-support prefer 30s

**`active_g`:** Hourly 9/9 pairs prefer `agf` (5/9 sig) | Quarterly `ag0` preferred | **Hourly-only switch**

---

## 4. Appendix Content

**Include:** Full M4 tables, Generic collapse analysis, scheduler robustness, hyperparameter grids, tiered-offset clamping details.

**Brief mention:** V1/V2 as engineering rejects (1 paragraph), skip connections, activation comparisons.

**Future work:** NHiTS, VAE, meta-learning.

---

## 5. Three Most Promising Paper Angles

**Angle A (Strongest):** "Multi-resolution N-BEATS: orthonormal wavelets and trend-wavelet composition"—clear contribution, strong M4 evidence.

**Angle B:** "Structured basis expansions beat brute-force capacity"—sub-1M AE/AELG at 30–50× fewer params.

**Angle C:** "N-BEATS depends on basis structure and validation cadence"—robustness and methodology angle.

---

**Recommendation:** Lead with Angle A (wavelets + composition). Use parameter efficiency and robustness as support. Reserve other contributions to appendix or future work.
