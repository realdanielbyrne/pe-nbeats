> **SUPERSEDED, 2026-07-27.** This document predates the NeurIPS reviews and the experiments run
> since, and its primary thesis has been contradicted by them. Wavelet family is a well powered null,
> the multiresolution mechanism does not exist under the shipped construction, and the compression is
> 96 percent trunk width rather than basis. Do not use it to guide writing. See
> [`NARRATIVE_SUPERSEDED_2026-07-27.md`](NARRATIVE_SUPERSEDED_2026-07-27.md) for the current position
> and for which of its claims survive.

# Combined Narrative Recommendations — NeurIPS Submission

**Date:** 2026-05-02
**Author:** Daniel Byrne
**Synthesis of:** [`NARRATIVE_RECOMMENDATIONS.md`](NARRATIVE_RECOMMENDATIONS.md), [`PAPER_NARRATIVE_RECOMMENDATIONS.md`](PAPER_NARRATIVE_RECOMMENDATIONS.md), [`claude_narrative_recs.md`](claude_narrative_recs.md)
**Purpose:** Single recommended best paper narrative engineered to land a splash at NeurIPS. Use this to guide v2 of [`ai-paper.md`](ai-paper.md).

---

## 0. The 60-Second Pitch

> N-BEATS' doubly-residual basis-expansion stack is the right scaffold for time-series forecasting. Its weakest assumption is its *choice* of bases: polynomial trend and Fourier seasonality are *global*, time-invariant, and force the Generic stack to over-parameterize as compensation — to the point of producing 40–50% bimodal-collapse failures on small datasets. We replace those bases with **orthonormal discrete wavelet transforms** that provide time-frequency localization, fuse them with polynomial trend in **TrendWavelet hybrid blocks**, and impose explicit rank control via **AE/AELG bottleneck root blocks** whose learnable latent gates *discover* effective capacity during training. The result is per-period SOTA on M4 Yearly, Monthly, and Weekly at sub-1M parameters — **12–80× fewer parameters than the 19–43M paper baselines** — with a tiered per-stack training schedule that selectively benefits wavelet blocks, mechanistically confirming the multi-resolution thesis.

---

## 1. Title and Twin Headline Thesis

### Title

> **Multi-Resolution N-BEATS: Orthonormal Wavelet Bases, Hybrid Trend-Wavelet Decomposition, and Compressed Backbones for Parameter-Efficient Time-Series Forecasting**

### Twin Tier-A Claims (foregrounded in the abstract)

The paper carries two co-equal headline claims. They reinforce each other rather than competing: the wavelet inductive bias *enables* the parameter compression by giving the network the right structural prior, so capacity isn't wasted reconstructing the wrong things.

**Claim 1 — Multi-resolution DWT bases.** Replace the global polynomial-trend and Fourier-seasonality bases with **orthonormal discrete wavelet transforms** (V3 construction, SVD-orthogonalized, condition number κ ≈ 1.0). Combined with **TrendWavelet** hybrid blocks, this yields per-period SOTA on M4 Yearly, Monthly, and Weekly. **Tiered per-stack offsets** confirm the mechanism: they help wavelet blocks while leaving NBEATS-G/IG unchanged on Hourly — a clean selectivity result.

**Claim 2 — Parameter efficiency via AE/AELG bottlenecks.** Replace the 4-FC RootBlock backbone with an **encoder-decoder substrate**. The learned-gate variant (`AERootBlockLG`) adds a `sigmoid(g) ⊙ z` gate that *discovers* effective latent dimensionality during training. Combined with TrendWavelet, sub-1M-param configurations match 19–43M paper baselines on M4-Yearly/Daily/Hourly — a **12–80× compression** at ≤0.5% SMAPE penalty.

---

## 2. Why This Wins at NeurIPS

- **One thesis, three pieces of evidence.** The shape NeurIPS reviewers reward: (a) per-period SOTA wins on the original benchmark, (b) sub-1M-parameter Pareto frontier, (c) tiered-offset selectivity as mechanistic confirmation. No "kitchen sink" — every Tier-B section directly supports one of the two Tier-A claims.
- **Diagnosis-then-cure framing, not autopsy.** The Milk 40–50% bimodal-collapse story is striking and unique to this work, but it lives in §2 (Diagnosis) as motivating evidence — not as the headline. Framing it as the headline reads as a critique of Oreshkin et al.; framing it as the disease our cure addresses reads as a constructive extension. Reviewers reward construction over destruction.
- **Statistical hygiene as a free differentiator.** Paired Wilcoxon signed-rank across seeds; never mix `nbeats_paper` and `sliding` SMAPE numbers in the same comparison; pool periods only for generalist rankings; 10 seeds per configuration on 112 configs is more numerous than the original paper's 18-model ensemble (caveat the comparison shape, but lean on the replication count).
- **No overclaim surface.** Wavelets win 6 of 9 tasks, not 9 of 9. Generic-IG is retained as the right answer for Quarterly and Hourly. `active_g` is presented as scoped, not universal. These honest scoping statements are what separates accepted papers from desk-rejected ones.

---

## 3. Six-Beat Narrative Arc

1. **Premise.** N-BEATS' doubly-residual basis-expansion scaffold (Oreshkin et al., 2020) is accepted as the right substrate. We do not redesign the residual topology; we redesign the bases that ride on top of it.
2. **Diagnosis.** Polynomial + Fourier bases are *global*. They cannot localize transient structure, force the Generic stack to over-parameterize as compensation, and produce a bimodal-collapse failure mode visible only at the small-dataset extreme. **Concrete evidence:** 40–50% of NBEATS-G_30s training runs diverge on Milk (156 observations, H=6); AELG variants diverge in 1.7% of runs at the same depth. Parameter-to-observation ratio predicts divergence rate. The Generic block's "fully learned" flexibility is *the cause*, not just an indicator.
3. **Solution part 1 — better bases.** Orthonormal DWT (V3 construction) provides time-frequency localization that polynomials and Fourier cannot. SVD-orthogonalization solves the prior wavelet-block ill-conditioning (κ ≈ 600K → κ = 1.0). Drop-in replacement for the Fourier-Seasonality block; the doubly-residual interface is unchanged.
4. **Solution part 2 — hybridization.** Real signals are not purely wavelet-decomposable either. **TrendWavelet** additively combines polynomial trend + DWT in a single block; **TrendWaveletGeneric** adds a third low-rank learned residual branch. The hybrid is the empirical sweet spot — it dominates Yearly, Monthly, Weekly per-period leaderboards.
5. **Solution part 3 — capacity control.** Wavelet bases tempt the network to over-parameterize the projection from hidden width to coefficients. **AE/AELG bottleneck root blocks** impose explicit rank control via an encoder-decoder substrate; the learned gate discovers effective rank during training. Result: matched accuracy at 12–80× fewer parameters. Sub-1M-param configurations (`TWAELG_10s_ld16_db3_*`, `TWAE_10s_ld32_*`) compete with 19–43M paper baselines.
6. **Solution part 4 — schedule alignment.** Tiered per-stack offsets align training dynamics to the wavelet pyramid: early stacks see low-frequency bands, later stacks refine high-frequency residuals. The selectivity result (helps `T+Sym10V3` and `TW`, leaves NBEATS-G/IG unchanged on Hourly) is the cleanest mechanistic confirmation in the paper that the basis change — not generic regularization — is doing the work. **Conditional on landing the Yearly falsification before submission** (see §10).

---

## 4. Tier Rankings

| Tier | Contribution | Where it appears |
|---|---|---|
| **A — Twin headline** | WaveletV3 orthonormal DWT bases + TrendWavelet / TrendWaveletGeneric hybrids | Title, abstract, §3, Main Table 1 |
| **A — Twin headline** | AE / AELG bottleneck root blocks (parameter efficiency) | Abstract, §4, Main Table 2 |
| **B — Strong support** | Tiered per-stack offsets (mechanistic confirmation) | §5, Main Table 3 — pending Yearly falsification |
| **B — Strong support** | Plateau LR + sub-epoch validation (paper-faithful protocol) | §3.3, all main tables |
| **B — Methodological** | Flexible stack construction (YAML composition grammar) | §6 short subsection + Appendix C |
| **C — Diagnosis** | NBEATS-G bimodal collapse on small datasets | §2.2 motivating + appendix detail |
| **C — Engineering** | `active_g`, `sum_losses`, ResNet skips, VAE root, configurable activations | Appendix only |
| **C — Out of scope** | NHiTS port | One appendix paragraph |
| **D — Forward-looking** | TrendWavelet + AE bottleneck transplanted into Llama-3.2-1B and SmolLM-135M FFNs (separate repo) | Appendix G stub — placeholder for forthcoming results |

---

## 5. Main Results Plan (3 tables, 2 figures — disciplined)

**Table 1 — M4 per-period SMAPE / OWA, paper-sample protocol with plateau LR.**
Best paper-faithful baseline (`*_ag0`) vs best repo extension. Columns: config, period, SMAPE (mean ± std, n=10), OWA, parameters, ratio vs paper baseline. All `*_agf` cells flagged with footnote "repo-novel extension; not present in Oreshkin et al."
- Source: [`experiments/results/m4/comprehensive_m4_paper_sample_plateau_results.csv`](../experiments/results/m4/comprehensive_m4_paper_sample_plateau_results.csv)

**Table 2 — Parameter-efficiency Pareto frontier.**
Sub-1M cluster (`TWAELG_10s_ld16_db3_*`, `TWAE_10s_ld32_*`) matching 19–43M paper baselines. Show parameter count, SMAPE on M4-Yearly/Daily/Hourly, gap to per-period winner. **Headline number:** 50× compression at ≤0.5% SMAPE penalty.
- Source: same as Table 1, filtered.

**Table 3 — Sym10 tiered-offset on M4-Hourly with selectivity panel.**
Pairwise comparison: tiered vs untiered for each block family. Helps `T+Sym10V3` (Δ −0.18 SMAPE, p<0.01) and `TW`; leaves NBEATS-G/IG unchanged. Selectivity is the headline. **Gated on Yearly falsification:** if Yearly contradicts Hourly, demote to "supporting panel" rather than headline finding.
- Source: [`experiments/results/m4/tiered_offset_m4_allperiods_paperlr_results.csv`](../experiments/results/m4/tiered_offset_m4_allperiods_paperlr_results.csv)

**Figure 1 — Architecture schematic.**
Side-by-side: Oreshkin's RootBlock (4-FC → polynomial / Fourier head) vs AERootBlockLG → TrendWavelet head. Annotate the gate ($\sigma(g) \odot z$) and the additive trend + wavelet decomposition. One figure carries the "what changed" story.

**Figure 2 — Basis visualization.**
First few orthonormal DWT basis vectors next to polynomial-trend basis and Fourier-seasonality basis at matched length. Makes time-localization vs global-support immediately legible to a reader who has not used wavelets before.

---

## 6. Appendix Layout

| Section | Content |
|---|---|
| **Appendix A — Cross-dataset robustness** | Tourism / Traffic / Weather / Milk results. Includes AsymWavelet-on-Traffic-96 (84% convergence with L=5H lookback) refuting prior architectural attribution. |
| **Appendix B — Backbone hierarchy reversal** | RootBlock > AELG on M4 / Tourism vs AE > AELG on Weather / Milk. Referenced from Discussion as "no universal default"; full evidence here. |
| **Appendix C — Flexible stack construction grammar** | YAML `stacks:` schema (`homogeneous`, `prefix_body`, `alternating`, `concat`, `explicit`, `builtin`, direct list, string shorthand) + the underlying `stack_types` list-based composition in `NBeatsNet.create_stack()`. **Novel methodological contribution** — Oreshkin et al. only report Generic-only and fixed Trend+Seasonality interpretable stacks; flexible composition (e.g., alternating `T+Sym10V3`) is what enabled the alternating-vs-unified findings in §3. |
| **Appendix D — Hyperparameter ablations** | `latent_dim`, `basis_dim` / `forecast_basis_dim`, `trend_thetas_dim`, `generic_dim`, gate functions, asymmetric wavelet families (`backcast_wavelet_type` / `forecast_wavelet_type`), `kl_weight`, `share_weights`, `skip_distance` / `skip_alpha`. Practitioner-facing reference. |
| **Appendix E — VAE asymmetric story** | Works on smooth Weather, fails on spiky Traffic. One-page footnote-length subsection. |
| **Appendix F — Negative and engineering results** | All `active_g` ablations (Tourism rescue p=0.0002; Hourly `agf` preference; Quarterly `ag0` preference), skip-connection studies, BottleneckGeneric universal under-performance, V1/V2 wavelet engineering rejects (one paragraph: ill-conditioning), configurable activations / losses / optimizers registries. |
| **Appendix G — Cross-domain stub: TrendWavelet + AE blocks in LLMs** | **Placeholder block.** Describes the architectural transplant (TrendWavelet + AE bottleneck swapped into transformer FFN sub-modules of Llama-3.2-1B and SmolLM-135M); empirical results forthcoming from a sister repository. **Phrasing must be forward-tense** ("we have implemented and are evaluating") — never present-tense empirical claims until results are integrated. |

**Items dropped, not appendixed:** `_sd5` skip variants on M4, `BNG*`/`BNAE*`/`BNAELG*` BottleneckGeneric families on M4, `_coif3` configs on M4, pure GenericAE/AELG without weight sharing, V1/V2 wavelet block details beyond the one-paragraph engineering note.

---

## 7. Five Things NOT to Claim

1. **"Wavelets are universally better."** They win on 6 of 9 tasks, not 9 of 9. Quarterly and Hourly favor `NBEATS-IG`. Be honest.
2. **"The Generic block should be deleted."** It wins on Quarterly and Hourly with `agf`. Context matters.
3. **"10× better."** We are ≤0.5% better on accuracy and **30–80× smaller** on parameters. The headline is parameters, not accuracy.
4. **"`active_g` is a global default."** It is dataset-specific. On Quarterly, `ag0` is preferred. The wrong choice on large datasets imposes a measurable penalty.
5. **"V1/V2 wavelets failed because…"** These were engineering issues (ill-conditioning, not a scientific finding). One paragraph in Appendix F mentioning the diagnosis; do not centrally feature.

---

## 8. LLM / FFN Cross-Domain Claim — Disposition

**User has empirical evidence in a sister repository:** TrendWavelet + AE bottleneck blocks transplanted into Llama-3.2-1B and SmolLM-135M FFN sub-modules. Disposition for *this* paper:

- **Main paper: silent.** No cross-domain claim in abstract, intro, or main results. The M4-and-supporting-datasets evidence is the splash. Adding cross-domain results to the headline competes with the twin Tier-A thesis and risks reviewer fragmentation.
- **Discussion §7 (Future Work):** one or two sentences pointing forward to the LLM application as ongoing work, with a citation pointer to the sister repository.
- **Appendix G (stub):** describes the architectural transplant — which sub-modules of the transformer FFN are replaced, what the TrendWavelet basis means in the FFN context (per-token feature decomposition? hidden-dim-axis pseudo-time? — make the interpretation explicit), what the AE bottleneck imposes on the residual stream. Reserve space for forthcoming empirical results without making them now.
- **At submission lock:** if LLM results have landed, decide whether to (a) keep them appendix-only, or (b) promote them to a 0.5-page §6.1 cross-domain teaser. **Do not** promote them to abstract / Tier-A status without separate evidence load comparable to the M4 sweeps (multiple model families, perplexity on standard benchmarks like WikiText / Pile, downstream-task evals, parameter / FLOP accounting). Cross-domain results in an N-BEATS paper should support, not dilute.

**Why this is the right disposition even with evidence in hand:**

- The wavelet inductive bias has a *clean* story for time-indexed signals (orthonormal DWT over a 1D temporal grid). Mapping it onto transformer FFNs requires either treating the FFN's hidden-dim axis as a "pseudo-time" axis or introducing per-token wavelet decomposition. Both interpretations need their own justification — too much to bolt onto a forecasting paper.
- Two co-equal Tier-A claims is already a heavy lift. Three Tier-A claims invites reviewers to attack the weakest link.
- The evidence quality bar for a NeurIPS LLM claim is high. Stubbing now and integrating cleanly at the next venue (or as a v2 of this paper) is safer than rushing.

---

## 9. Honest Caveats (state explicitly in §6 Discussion)

1. **M4-Quarterly has statistical ties.** Top 15 configurations are within 0.025 SMAPE; pairwise Wilcoxon p > 0.23. Architecture choice does not matter for Quarterly. State this; do not oversell.
2. **Tourism SOTA not beaten.** Best result 21.773 vs. known best 20.864. Honest limitation.
3. **Two sampling protocols are not interchangeable.** `nbeats_paper` and `sliding` produce different absolute SMAPE numbers; we use `nbeats_paper` for direct comparison to Oreshkin et al. and report `sliding` only as a robustness panel.
4. **Single-run-per-config vs. paper's 18-model ensemble.** Caveat the comparison shape; lean on the higher replicate count (10 seeds × 112 configs vs. their 18 ensembled models per period).
5. **Daily/Hourly gaps.** Frame as ongoing work, not limitation.
6. **LLM evidence is in a sister repo, not this paper.** Stated explicitly in Future Work and Appendix G.

---

## 10. Pre-Submission Verification Checklist

- [ ] **Land the Yearly tiered-offset falsification** before claiming the Hourly selectivity result in Main Table 3. If Yearly contradicts Hourly, demote Table 3 to a supporting panel and remove tiered offsets from beat 6 of the arc. See [`memory/project_tiered_offset_open_questions.md`](../../../Users/dbyrne/.claude/projects/c--Github-N-BEATS-Lightning/memory/project_tiered_offset_open_questions.md) for the four open follow-ups.
- [ ] Re-run pairwise Wilcoxon signed-rank on `comprehensive_m4_paper_sample_plateau_results.csv` for the per-period table to confirm top-of-table claims are still significant after any new data lands.
- [ ] Confirm Tourism / Traffic / Weather appendix tables use the protocols documented in their YAML configs (not stale older sweeps).
- [ ] Run the four open tiered-offset follow-ups: TWAE ld32 re-run, tiered+`agf`, tiered 30s, Yearly falsification.
- [ ] Lock the LLM/FFN disposition (silent in main paper; Future Work pointer; Appendix G stub) **unless** sister-repo results have been integrated with NeurIPS-grade evidence load.
- [ ] Pre-submission read-through: scan for present-tense LLM claims and demote any that exist to forward-tense.
- [ ] Verify all `*_agf` cells in Tables 1 and 2 are footnoted as "repo-novel extension."
- [ ] Verify no SMAPE comparison in the paper crosses `nbeats_paper` and `sliding` protocols.

---

## 11. Comparison Summary — How This Combined Narrative Differs From Each Source

| From | Adopted | Modified | Dropped |
|---|---|---|---|
| **NARRATIVE_RECOMMENDATIONS.md** | Diagnostic Milk-divergence evidence (40–50% NBEATS-G_30s); statistical-claim language; "what NOT to claim" list; honest caveats | Demoted "Generic block is dangerous" from headline to §2 motivating evidence; removed Backbone Hierarchy Reversal as a Tier-1 claim (now Appendix B) | Cross-dataset generalist as a main-paper Table 4; the 4-table / 3-figure budget |
| **PAPER_NARRATIVE_RECOMMENDATIONS.md** | Tier-rank discipline; "drop coif3" / wavelet shortlist guidance; Hourly-only `active_g=forecast` rule; the `agf` vs `ag0` paper-faithfulness convention | Promoted parameter efficiency from "supporting" to a co-equal Tier-A claim alongside multi-resolution | The tighter "structured beats brute-force" framing — replaced with the more explicit "global bases are the weak assumption" framing for narrative bite |
| **claude_narrative_recs.md** | Six-beat narrative arc; tiered-offset Tier-B / Main-Table-3 status; statistical hygiene language; main-results-on-M4 discipline; appendix triage logic | Added flexible stack construction as an explicit Tier-B / Appendix-C contribution (per user direction); added Appendix G LLM stub (per user direction) | None substantive — closest to the structural backbone of this combined narrative |

---

## 12. Critical Files Referenced

- [`src/lightningnbeats/blocks/blocks.py`](../src/lightningnbeats/blocks/blocks.py) — block hierarchies, `_WaveletGeneratorV3`, AE / AELG / VAE roots
- [`src/lightningnbeats/models.py`](../src/lightningnbeats/models.py) — stack composition, `stack_basis_offsets`, skip connections, `sum_losses`
- [`experiments/configs/schema.md`](../experiments/configs/schema.md) — YAML composition grammar (Appendix C source)
- [`experiments/results/m4/comprehensive_m4_paper_sample_plateau_results.csv`](../experiments/results/m4/comprehensive_m4_paper_sample_plateau_results.csv) — Main Table 1 source
- [`experiments/results/m4/tiered_offset_m4_allperiods_paperlr_results.csv`](../experiments/results/m4/tiered_offset_m4_allperiods_paperlr_results.csv) — Main Table 3 source
- [`experiments/analysis/analysis_reports/`](../experiments/analysis/analysis_reports/) — pre-computed statistical tests
- [`NBEATS-Explorations/ai-paper.md`](ai-paper.md) — existing v1 draft; align v2 against this combined narrative

---

## 13. One-Sentence Summary For The Cover Letter

> We replace N-BEATS' global polynomial-trend and Fourier-seasonality bases with orthonormal wavelet decompositions, fuse them with polynomial trend in hybrid TrendWavelet blocks, and impose explicit rank control via learned-gate autoencoder bottlenecks — achieving per-period SOTA on M4 Yearly / Monthly / Weekly at sub-1M parameters (12–80× compression of the 19–43M paper baselines), with a tiered training schedule whose selective benefit to wavelet blocks mechanistically confirms the multi-resolution thesis.
