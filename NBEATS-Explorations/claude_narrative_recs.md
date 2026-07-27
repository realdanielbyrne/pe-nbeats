> **SUPERSEDED, 2026-07-27.** This document predates the NeurIPS reviews and the experiments run
> since, and its primary thesis has been contradicted by them. Wavelet family is a well powered null,
> the multiresolution mechanism does not exist under the shipped construction, and the compression is
> 96 percent trunk width rather than basis. Do not use it to guide writing. See
> [`NARRATIVE_SUPERSEDED_2026-07-27.md`](NARRATIVE_SUPERSEDED_2026-07-27.md) for the current position
> and for which of its claims survive.

# Research Paper Narrative — `lightningnbeats`

## Context

The `lightningnbeats` repository extends Oreshkin et al. (2020) N-BEATS with ~10 distinct modifications spanning new basis-expansion blocks (orthonormal wavelets, AE / AELG / VAE bottlenecks, TrendWavelet hybrids), training-side knobs (`active_g`, `sum_losses`, ResNet skips), schedule innovations (plateau LR, tiered per-stack offsets), and an NHiTS port. Empirical evidence is mature on M4 (45 result CSVs, two LR protocols), partial on Tourism/Traffic/Weather, and exploratory on Milk/NHiTS. The question: which subset deserves a single coherent paper, and how should the story be told?

The recommendation below treats the work as **one paper with a tight thesis, not a survey of every knob**. Less is more — a paper that argues two strong claims will be cited; a paper that lists ten will not.

---

## 1. Core Contributions (rank-ordered by scientific weight)

| Tier | Contribution | Why it matters | Evidence depth |
|---|---|---|---|
| **A — Headline** | **Orthonormal DWT bases** + **TrendWavelet / TrendWaveletGeneric hybrid blocks** | Replaces Oreshkin's *global*, time-invariant Fourier (seasonality) and polynomial (trend) bases with *time-localized* multi-resolution bases via SVD-orthogonalized DWT construction. Hybrid blocks fuse structured (trend+wavelet) and unstructured (low-rank generic) decompositions in a single additive head. | M4 paper-sample + sliding sweeps; per-period SOTA on Yearly, Monthly, Weekly. Generalist `T+Sym10V3_30s_bdeq` mean rank 6.83/53. |
| **A — Headline** | **AE / AELG bottleneck root blocks** (parameter efficiency) | Encoder–decoder substrate replacing the 4-FC backbone; AELG adds a learnable sigmoid gate that *discovers* effective latent dimensionality during training. Combined with TrendWavelet, achieves SOTA-matching accuracy at **0.48–2.9M parameters vs 19–43M for paper baselines** (12–80× reduction). This is the deployment-ready story. | 871-run M4-Yearly omnibus; sub-1M champion `TWAELG_10s_ld32_db3_*`. |
| **B — Strong support** | **Tiered offsets / per-stack LR cascading** | Different stacks specialize on different frequency bands of the orthonormal DWT — operationalizes the wavelet pyramid in the *training schedule*, not just the basis. Selectivity (helps wavelet blocks, leaves legacy blocks unchanged) is the strongest mechanistic evidence in the repo for the multi-resolution hypothesis. | M4 Hourly tiered sweep (sym10 +0.15–0.19 SMAPE); Yearly falsification still pending. |
| **B — Strong support** | **Plateau LR schedule** | Required to make paper-faithful comparisons stable; without sub-epoch validation (`val_check_interval=100`, `min_delta=0.001`, `patience=20`) the `nbeats_paper` sampling protocol collapses to `best_epoch=0/1`. | Used uniformly across the comprehensive M4 sweeps. |
| **C — Engineering / supporting** | `active_g` (=`forecast`), `sum_losses`, ResNet skip (`skip_distance`/`skip_alpha`), VAE root blocks, configurable activations/losses, `BottleneckGeneric`. | Each is empirically useful in narrow regimes (`active_g=forecast` on Yearly/Hourly only; skips rescue ≥20-stack GenericAELG; VAE useful on smooth Weather, fails on Traffic). None is a standalone scientific claim. | Various ablation CSVs. |
| **C — Out of scope** | NHiTS port | NHiTS itself (Challu et al. 2022) is a published architecture; the port is engineering, not a contribution. | Pilot only. |

**Recommendation:** Title and abstract should foreground Tier A only. Tier B appears as supporting sections. Tier C lives in the appendix.

---

## 2. Narrative Arc

A single throughline ties the work together: **N-BEATS' inductive bias (basis expansion) is its strength, but the paper's *choice* of bases — global Fourier and polynomials — is its weakest assumption.** Real time series are non-stationary and locally structured; global bases waste capacity reconstructing the wrong things.

Suggested arc (six beats):

1. **Premise.** N-BEATS' doubly-residual basis-expansion stack is the right scaffold; treat this as accepted.
2. **Diagnosis.** Polynomial trend + Fourier seasonality are *global* bases. They cannot localize transient structure, force the network to over-parameterize the Generic stack to compensate, and produce a bimodal collapse failure mode (Generic-only at 30 stacks, +2 SMAPE) that motivated `active_g` as a band-aid in prior repo work.
3. **Solution part 1 — better bases.** Orthonormal DWT provides time-frequency localization. Drop-in replacement for Fourier inside the existing stack interface.
4. **Solution part 2 — hybridization.** Real signals are not purely wavelet-decomposable either. TrendWavelet additively combines polynomial trend + DWT bases in one block; TrendWaveletGeneric adds a low-rank learned residual. This is the structural sweet spot empirically.
5. **Solution part 3 — capacity control.** Wavelet bases tempt over-parameterization; AE/AELG bottlenecks impose explicit rank control on each block's basis coefficients. The learned gate (`AERootBlockLG`) lets the network discover its own effective rank. Result: matched accuracy at 12–80× fewer parameters.
6. **Solution part 4 — schedule alignment.** Tiered per-stack offsets match the training schedule to the wavelet pyramid: early stacks see low-frequency bands at high LR, later stacks refine high-frequency residuals. The selectivity result (only helps wavelet blocks) is the cleanest mechanistic confirmation that the basis change, not generic regularization, is doing the work.

This arc lets every Tier A and Tier B contribution land naturally; Tier C items become methods-section footnotes.

---

## 3. Reporting Priorities (Main Results)

Anchor the main results section on M4 because (a) it is the original benchmark, (b) the repo has the most replicates, and (c) two LR protocols exist for a clean robustness check.

**Main Table 1 — M4 per-period SMAPE/OWA, paper-sample protocol with plateau LR.**
Columns: best paper-faithful baseline (`NBEATS-G_*_ag0` or `NBEATS-IG_*_ag0`), best repo extension, parameter count, ratio. Use `comprehensive_m4_paper_sample_plateau_results.csv` as the source. Show Tier A wins on Yearly/Monthly/Weekly and IG retaining Quarterly/Hourly.

**Main Table 2 — Parameter efficiency frontier.**
Pareto plot of SMAPE vs parameter count across all configs, M4 averaged. Highlight the AELG sub-1M cluster matching 19–43M paper baselines.

**Main Table 3 — Sym10 tiered-offset on Hourly.**
The sharpest mechanistic result: tiered offsets help `T+Sym10V3` and `TW` blocks but leave NBEATS-G/IG unchanged. Selectivity is the headline. Add the Yearly falsification as a robustness panel once it lands.

**Main Figure 1 — Architecture schematic.**
Side-by-side: Oreshkin's RootBlock (4-FC → polynomial/Fourier head) vs AERootBlockLG → TrendWavelet head. One figure carries the "what changed" story.

**Main Figure 2 — Basis visualization.**
First few orthonormal DWT basis vectors next to the polynomial-trend and Fourier-seasonality bases at matched length. Makes time-localization vs global-support immediately legible to a reader who has not held a wavelet before.

**Stat hygiene:** report Wilcoxon signed-rank or paired-MWU across seeds; pool periods only when reporting generalist rankings; never mix `nbeats_paper` and `sliding` protocols in absolute SMAPE comparisons.

---

## 4. Appendix Selection

Move to the appendix:

- **NBEATS-G bimodal collapse + `active_g` rescue** — The Milk 63%-stuck result is striking but it is a failure-mode study of Oreshkin's architecture, not a contribution of ours. Appendix as motivating evidence for the diagnosis in §2.2.
- **Convergence studies (Parts 6 & 8)** — sum_losses × active_g 2×2 factorial, divergence detection. Useful for reproducibility, not narrative.
- **Activation / loss / optimizer registries** — engineering breadth, no scientific claim.
- **NHiTS port and pooling/sampling-window pilots** — single appendix section noting the port works and blocks plug in unchanged; cite Challu et al. for the architecture itself.
- **Alternate datasets (Tourism, Traffic, Weather, Milk)** — these *do* matter, but as a robustness appendix, not the main results. Headline finding to retain in the appendix: AsymWavelet diagnostic on Traffic-96 (84% convergence with L=5H lookback) refuting prior architectural attribution of the Traffic-96 divergence.
- **Hyperparameters tables, YAML schema, claim-file infrastructure** — pure reproducibility material.
- **Per-dataset hyperparameter tuning notes** (e.g., basis_dim sensitivity, latent_dim sweeps, gate-function studies).
- **VAE root block and KL-weight studies** — VAE is interesting on smooth data (Weather) but fails on spiky data (Traffic); the asymmetric story is a footnote, not a thesis.
- **`BottleneckGeneric`, `GenericAEBackcast`** — block-zoo entries that did not produce per-period SOTA on M4.

Empirically-falsified knobs that should be **dropped, not appendixed:** `_sd5` skip variants on M4, all `BNG*`/`BNAE*` BottleneckGeneric families, `_coif3` configs, pure GenericAE/AELG without weight sharing.

---

## 5. Three Highest-Promise Angles (pick one as primary)

Three viable framings; all three cannot fit in one paper.

### Angle A — "Multi-resolution N-BEATS" *(recommended)*
**Thesis:** replacing global Fourier/polynomial bases with orthonormal DWT bases (and their hybrids) is the structurally correct generalization of N-BEATS.
**Evidence backbone:** WaveletV3 + TrendWavelet wins on M4 Yearly/Monthly/Weekly; tiered-offset selectivity is the mechanistic confirmation.
**Why this wins:** it is the cleanest scientific story, has the strongest empirical support, and the rest of the repo (AE/AELG, plateau LR, active_g) becomes natural supporting machinery rather than separate claims. Targets ICML / NeurIPS / TMLR.

### Angle B — "Parameter-efficient time-series forecasting via learned-gate AE bottlenecks"
**Thesis:** AELG bottlenecks discover effective rank during training and recover SOTA accuracy at 12–80× fewer parameters.
**Evidence backbone:** sub-1M-param `TWAELG_10s_ld32_db3_*` matching 19–43M paper baselines on M4-Yearly/Daily/Hourly; AELG ld=16 vs AE ld=32 parity.
**Why this could win:** edge-deployment angle is fashionable and the parameter-count headline is dramatic. Risk: gating mechanisms are well-trodden; reviewers may push back on novelty without a stronger learning-theoretic story.

### Angle C — "Schedule-induced frequency specialization"
**Thesis:** tiered per-stack LR / offset schedules align training dynamics with the wavelet pyramid, producing frequency-band specialization across the stack.
**Evidence backbone:** sym10 tiered Hourly result; selectivity (legacy blocks unchanged); Yearly falsification (pending).
**Why this is risky now:** evidence base is narrow (one period, one family); needs the four open follow-ups in `project_tiered_offset_open_questions.md` to land before it carries a paper. Strong workshop paper today, possible standalone paper after follow-ups.

**Recommendation: write Angle A.** It uses the deepest evidence base and gives the AE/AELG and tiered-offset work natural homes as supporting sections rather than competing claims.

---

## Critical files referenced

- [src/lightningnbeats/blocks/blocks.py](../src/lightningnbeats/blocks/blocks.py) — block hierarchies, `_WaveletGeneratorV3`, AE/AELG/VAE roots
- [src/lightningnbeats/models.py](../src/lightningnbeats/models.py) — stack composition, `stack_basis_offsets`, skip connections, `sum_losses`
- [experiments/results/m4/comprehensive_m4_paper_sample_plateau_results.csv](../experiments/results/m4/comprehensive_m4_paper_sample_plateau_results.csv) — main-table source
- [experiments/results/m4/tiered_offset_m4_allperiods_paperlr_results.csv](../experiments/results/m4/tiered_offset_m4_allperiods_paperlr_results.csv) — tiered-offset main-table source
- [experiments/analysis/analysis_reports/](../experiments/analysis/analysis_reports/) — pre-computed statistical tests
- `EXTENSIONS.md` — extension-by-extension empirical record

## Verification (pre-submission)

- Run the four open follow-ups from `project_tiered_offset_open_questions.md` (TWAE ld32 re-run, tiered+agf, tiered 30s, **Yearly falsification**) before claiming the tiered-offset selectivity result.
- Re-run pairwise Wilcoxon on `comprehensive_m4_paper_sample_plateau_results.csv` for the per-period table to ensure top-of-table claims are still significant.
- Confirm Tourism/Traffic/Weather appendix tables use the protocols documented in their YAML configs and not a stale older sweep.
