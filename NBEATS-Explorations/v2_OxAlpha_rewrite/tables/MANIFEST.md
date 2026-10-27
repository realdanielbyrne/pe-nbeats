# Table Manifest — "Frozen Basis Time Series Forecasting"

Tables agent output for the v2 Ox-Alpha rewrite. Governing documents:
`rebuttal/notation_and_taxonomy_spec.md` §5 (schema/markers), §6a–6c (string
map, aggregation rule), §7 (identity marking), §8 (table-by-table
disposition); `rebuttal/paper_rewrite_plan.md` §11 (corrections ledger),
§12 (loose ends), §14 item 5 (single sanctioned params change).

Marker conventions used in every surviving results table:
`†` = forecast-head activation (`active_g`), a `lightningnbeats` extension,
not paper-faithful · `ᵍ` = historical run used the gated AE trunk (gate
null; AELG→AE merge per spec §4) · `‡` = frozen basis at that horizon was
the identity matrix (scoped to the affected periods; never on Hourly).

## 29 legacy labels → disposition

| # | Legacy label | Disposition |
|---|---|---|
| 1 | `tab:backbones` | **Kept-in-file `backbones.tex`** — restructured under the new thesis: uniform-U vs AE-d trunk framing, §2 symbols, gate row demoted to ᵍ-provenance |
| 2 | `tab:param-counts` | **Kept-in-file `param-counts.tex`** — restructured; trunk counts verbatim; width presented as inert (KW p = 0.776/0.773); count convention stated |
| 3 | `tab:branch-ae` | **Dropped** — spec §8 #3: AE-for-accuracy is null; one paragraph survives in prose (0.057-sMAPE band) |
| 4 | `tab:datasets` | **Kept-in-file `datasets.tex`** — keep-as-is with the sanctioned correction: M4-Monthly H = 18 (was 24; plan §12 loose end) |
| 5 | `tab:main_leaderboard` | **Kept-in-file `main_leaderboard.tex`** — restructured to §5 schema; ‡ inline on Yearly/Weekly/Monthly/Daily winners; † on Hourly baseline row; caption carries the full identity disclosure (spec §8 caption obligation) |
| 6 | `tab:sub1m_frontier` | **Skipped — sibling-owned** (`tables/frontier.tex`, incl. the single sanctioned params change 0.85M → 1.00M) |
| 7 | `tab:hyperparams_defaults` | **Skipped — sibling-owned** (`sections/A1_appendix_hyperparameters.tex`) |
| 8 | `tab:sweep_grids` | **Skipped — sibling-owned** (`sections/A1_appendix_hyperparameters.tex`) |
| 9 | `tab:lr_per_period` | **Skipped — sibling-owned** (`sections/A1_appendix_hyperparameters.tex`; γ removed from caption per spec §3 is A1's obligation) |
| 10 | `tab:protocol_per_period` | **Skipped — sibling-owned** (`sections/A1_appendix_hyperparameters.tex`) |
| 11 | `tab:wavelet_per_period` | **Dropped** — spec §8 #11: "best wavelet per period" is meaningless under the null and partly identity-vs-identity; superseded by the N4 identity/family-null table |
| 12 | `tab:quickpick` | **Kept-in-file `quickpick.tex`** — restructured to §5 schema; ttd/ld/bd/ag headers replaced by p (caption), d_b, Trunk, † marker |
| 13 | `tab:configs_m4` | **Skipped — sibling-owned** (`sections/A2_appendix_configs.tex`) |
| 14 | `tab:configs_ood` | **Skipped — sibling-owned** (`sections/A2_appendix_configs.tex`) |
| 15 | `tab:active_g_ablation` | **Skipped — sibling-owned** (`sections/A3_appendix_ablations.tex`) |
| 16 | `tab:tiered_per_period` | **Skipped — sibling-owned** (`sections/A3_appendix_ablations.tex`) |
| 17 | `tab:lr_matched` | **Skipped — sibling-owned** (`sections/A3_appendix_ablations.tex`) |

| 18 | `tab:m4_yearly_full` | **Kept-in-file `m4_yearly_full.tex`** — §5 schema + ‡/†/ᵍ + LR column; one §6c deletion (sym10 twin of TWAE_10s_ld32_ag0, 13.553±0.165 deleted; representative 13.546±0.102 kept) |
| 19 | `tab:m4_quarterly_full` | **Kept-in-file `m4_quarterly_full.tex`** — two §6c deletions (T+DB3V3_10s_tiered twin 10.356±0.053; T+Coif2V3_30s_bd2eq sliding twin 10.146±0.045) |
| 20 | `tab:m4_monthly_full` | **Kept-in-file `m4_monthly_full.tex`** — one §6c deletion (TW_30s_td3_bdeq_coif3 13.296±0.182 n=5, identical up to family to the kept representative 13.240±0.334) |
| 21 | `tab:m4_weekly_full` | **Kept-in-file `m4_weekly_full.tex`** — four §6c deletions (step-paper db3/sym10 twins of the kept 6.735±0.203; plateau-reval db3 twin 6.955±0.233; sliding bdeq twins 6.675±0.185 and 6.686±0.168 of the kept 6.671±0.208) |
| 22 | `tab:m4_daily_full` | **Kept-in-file `m4_daily_full.tex`** — one §6c deletion (T+DB3V3_10s_tiered twin 3.023±0.039) |
| 23 | `tab:m4_hourly_full` | **Kept-in-file `m4_hourly_full.tex`** — one §6c deletion (TWAELG_10s_ld32_sym10_agf 8.933±0.110, twin of the kept 8.924±0.129), so the paper-sample list renumbers to rank 9; no ‡ anywhere (all families genuine at H=48, spec §7 / checklist item 5) |
| 24 | `tab:generalist_paper` | **Kept-in-file `generalist_paper.tex`** — ranks verbatim (108-config pool) + historical-pool footnote; ‡ scoped per period via superscripts ᵃ (Y/Q/W/D/M) and ᵇ (Y/Q) |
| 25 | `tab:generalist_sliding` | **Kept-in-file `generalist_sliding.tex`** — ranks verbatim incl. three-way tie at 18.3 (112-config pool) + pool footnote; ‡ᵇ scoping; leader (genuine everywhere) unmarked |
| 26 | `tab:tourism_winners` | **Kept-in-file `tourism_winners.tex`** — restructured to schema; out-of-sweep 20.864 noted as prior finding in caption (plan §12); no ‡ asserted pending plan §10 item 7 verification |
| 27 | `tab:llm_params` | **Dropped** — leaves with the withdrawn Transformer study (plan §3; P1 item 3) |
| 28 | `tab:llm_trajectory` | **Dropped** — same |
| 29 | `tab:llm_kd` | **Dropped** — same |

## Files created by this agent (tables/)

| File | Label |
|---|---|
| `backbones.tex` | `tab:backbones` |
| `param-counts.tex` | `tab:param-counts` |
| `datasets.tex` | `tab:datasets` (Monthly H corrected to 18) |
| `main_leaderboard.tex` | `tab:main_leaderboard` (caption-inline identity disclosure) |
| `quickpick.tex` | `tab:quickpick` |
| `m4_yearly_full.tex` | `tab:m4_yearly_full` (1 dedup deletion) |
| `m4_quarterly_full.tex` | `tab:m4_quarterly_full` (2 dedup deletions) |
| `m4_monthly_full.tex` | `tab:m4_monthly_full` (1 dedup deletion) |
| `m4_weekly_full.tex` | `tab:m4_weekly_full` (4 dedup deletions) |
| `m4_daily_full.tex` | `tab:m4_daily_full` (1 dedup deletion) |
| `m4_hourly_full.tex` | `tab:m4_hourly_full` (1 dedup deletion; no ‡) |
| `generalist_paper.tex` | `tab:generalist_paper` (pool footnote) |
| `generalist_sliding.tex` | `tab:generalist_sliding` (pool footnote) |
| `tourism_winners.tex` | `tab:tourism_winners` |

Pre-existing sibling files not touched: `corrections_ledger.tex`
(`tab:corrections_ledger`), `frontier.tex` (`tab:sub1m_frontier`).

## Sibling-owned labels skipped (do not redefine here)

Method section (`02_unified_formulation.tex`): `tab:dimensions`,
`tab:variant_taxonomy`, `tab:notation_disambiguation`.
Appendix A1: `tab:hyperparams_defaults`, `tab:sweep_grids`,
`tab:lr_per_period`, `tab:protocol_per_period`.
Appendix A2: `tab:configs_m4`, `tab:configs_ood`.
Appendix A3: `tab:active_g_ablation`, `tab:tiered_per_period`,
`tab:lr_matched`.

## Numeric changes register (everything else is bit-identical to v1)

1. `datasets.tex`: M4 horizon set `{6, 8, 24, …}` → `{6, 8, 18, …}`
   (sanctioned correction, plan §12).
2. §6c family-dedup deletions listed above (deletions only; every
   surviving score ± std, param count, rank digit, and n is verbatim
   from v1). No score, std, or rank was recomputed or pooled.
3. No other numeric cell changed anywhere; the frontier 0.85M→1.00M
   change lives only in the sibling-owned `frontier.tex`.

## Spec ambiguities resolved

* **coif3 row of the old monthly top-5**: not covered by the §7 identity
  matrix, but §6c collapses "rows identical up to family" without limiting
  families to the matrix, so it was deleted as a twin of the sym10-class
  representative rather than printed without provenance.
* **Generalist ‡ scoping**: per-period rank cells cannot carry a per-cell
  dagger without destroying the verbatim digits, so scoped superscripts
  (‡ᵃ = Y/Q/W/D/M, ‡ᵇ = Y/Q) attach to the configuration name and are
  defined in the caption.
* **Tourism-Yearly winner (`TW_10s_td3_bdeq_db3`)**: spec §6b flags its
  identity status at Tourism horizons as unverified (plan §10 item 7);
  printed with no ‡ rather than guessing.
* **Legacy stray `*` on the Tourism reference value (IG-10)**: had no
  defined legend; dropped (formatting, not a number).
* **`tab:param-counts` vs "counts include the frozen bases"**: the legacy
  numbers are trunk-only counts and may not be recomputed, so they are kept
  verbatim and the caption states explicitly that results-table counts are
  inclusive of the frozen bases (convention stated once per legend).
* **Hourly paper-sample list length**: after the sanctioned §6c deletion it
  holds nine representatives instead of ten; ranks renumbered 1–9, scores
  untouched.

## Undischarged items (for the assembler / verification pass)

* Plan §10 item 7: verify `TW_10s_td3_bdeq_db3` (Tourism) and
  `TrendWaveletAELG_coif3_eq_bcast_td3_ld16` instantiability/identity at
  Tourism horizons before camera-ready freeze.
* The unified legend table (spec §5 rule 5, new table N3) is not owned by
  any fragment I produce; captions here define the markers inline and so
  remain self-contained regardless of where N3 lands.
* Cross-reference targets used: `Section~\ref{sec:method_identity}`
  (already referenced by `A4_appendix_full_tables_intro.tex`). The
  assembler must ensure `\citet{oreshkin2020nbeats}` resolves against
  `references.bib`.

