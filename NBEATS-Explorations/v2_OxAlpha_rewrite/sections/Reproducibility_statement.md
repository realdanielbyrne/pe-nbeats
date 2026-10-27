# Reproducibility Statement — Working Note (authors only, not paper text)

*Status: 2026-08 camera-ready assembly. Companion to the appendix fragments in
`sections/` (A0–A4). This note inventories the reproducibility bundle promised
in all five rebuttal responses and tracked as paper_rewrite_plan.md §9 row
"Configs, per-run CSVs, registered prediction files ship with the revision"
and §13 step 9. Items marked MISSING must be exported before the bundle ships.*

## 1. Bundle contents to ship

The supplement bundle should contain three artifact classes:
1. **Configs** — every YAML that generated a number printed in the paper.
2. **Per-run CSVs** — one row per run (seed-level), including failures.
3. **Registered prediction files** — per-run forecast arrays (.npz) for the
   registered-prediction studies.

## 2. Configs (`experiments/configs/`) — ALL PRESENT

Every config cited in A2 exists in the repo today. Key files verified:

| File | Status |
|---|---|
| `comprehensive_sweep_m4.yaml` | present |
| `comprehensive_m4_paper_sample.yaml` | present (source of the verbatim block in `app:configs:yaml`) |
| `comprehensive_m4_paper_sample_plateau.yaml` | present — see caveat C1 below |
| `comprehensive_m4_paper_sample_sym10_fills.yaml` | present |
| `tiered_offset_m4_allperiods.yaml`, `_paperlr.yaml` | present |
| `m4_hourly_sym10_tiered_offset.yaml`, `_paperlr.yaml` | present |
| `nbeats_g.yaml`, `nbeats_ig.yaml` | present |
| Tourism / Traffic / Weather / Milk sweeps + omnibus + NHiTS set | present |
| Post-submission studies: `m4_trunk_width_attribution.yaml`, `m4_basis_identity_control.yaml`, `m4_complete_basis.yaml`, `m4_kl_basis*.yaml`, `m4_2x2_ae_row.yaml`, `m4_weight_decay_arm.yaml`, `m4_ae_taper_sweep.yaml`, `m4_divergence_signs.yaml`, `m4_dwt_relu_diagnostic.yaml`, `m4_p2_boundary.yaml`, `m4_pramanick_comparison*.yaml` | present |
| `tourism_preset_validation.yaml` | present |

**Caveat C1 (must resolve before shipping).**
`comprehensive_m4_paper_sample_plateau.yaml` header/description say factor
0.75, cooldown 0 ("Monthly-only for initial validation") while its periods
list all six; the paper text (A0 protocol + setup section) states factor
0.5, patience 3, cooldown 1. Decide which is canonical: either update the
YAML to match the paper or restate the paper numbers. The plateau CSV
(`comprehensive_m4_paper_sample_plateau_results.csv`) is the ground truth —
diff its effective scheduler settings against the YAML before release.

**Caveat C2 (must resolve before shipping).**
Early-stopping patience under the faithful paper-sample protocol: the
shipped YAML (`comprehensive_m4_paper_sample.yaml`, `patience: 15`) and the
v1 appendix say 15 val-check units; the rewritten setup section says 20.
The appendix fragments (A0, A1) follow the v1/YAML value of 15 so that text
matches the released file. Reconcile the setup section (or the YAML)
before the bundle ships — pick one number everywhere.

## 3. Per-run CSVs (`experiments/results/`) — PRESENT

Canonical M4 files referenced by A0/A1 (all exist under
`experiments/results/m4/`):

- Paper-sample faithful: `comprehensive_m4_paper_sample_results.csv`,
  `comprehensive_m4_paper_sample_sym10_fills_results.csv`,
  `tiered_offset_m4_allperiods_paperlr_results.csv`,
  `m4_hourly_sym10_tiered_offset_paperlr_results.csv`
- Paper-sample plateau: `comprehensive_m4_paper_sample_plateau_results.csv`,
  `tiered_offset_m4_allperiods_results.csv`,
  `m4_hourly_sym10_tiered_offset_results.csv`
- Sliding: `comprehensive_sweep_m4_results.csv`
- Post-submission: `m4_trunk_width_attribution_results.csv`,
  `m4_2x2_ae_row_results.csv`, `m4_weight_decay_arm_results.csv`,
  `m4_ae_taper_sweep_results.csv`, `m4_basis_identity_control_results.csv`,
  `m4_complete_basis_results.csv`, `m4_kl_basis*_results.csv`,
  `m4_divergence_signs_results.csv`, `m4_dwt_relu_diagnostic_results.csv`,
  `m4_p2_boundary_results.csv`, `m4_pramanick_comparison*_results.csv`,
  `tiered_offset_m4_weekly_plateau_validation_results.csv`

OOD (exist): `tourism/comprehensive_sweep_tourism_results.csv`,
`tourism/tourism_preset_validation_results.csv`,
`tourism/tourism_aelg_sota_confirmation_results.csv`;
`traffic/comprehensive_sweep_traffic_results.csv`,
`traffic/traffic96_*_results.csv`, `traffic/Traffic-AsymWavelet-Diagnostic_results.csv`;
`weather/comprehensive_sweep_weather_results.csv` (+ gate/KL/skip studies);
`milk/milk_convergence{,_10stack}_results.csv`,
`milk/unified_benchmark_results.csv`.

Analysis reports: `experiments/analysis/analysis_reports/m4_overall_leaderboard_2026-05-03.md`
exists; regenerable via `m4_overall_leaderboard.py`. Note the spec §6b flag:
the Hourly capture-ordering value still needs pinning (plan §10 item 6,
10.587 vs 10.580 ± 0.102) before any prose repeats it.

## 4. Registered prediction files (.npz) — PARTIAL

Present:

- `experiments/results/predictions/` — ~4,954 .npz files (Traffic/Weather
  asymmetric-basis diagnostics, M4 study runs).
- `experiments/results/m4/predictions/` — ~1,479 .npz files.
- `experiments/results/milk/predictions/` — ~3,136 .npz files.

MISSING (must be exported or their absence disclosed):

- **Tourism registered preset-validation predictions.** No `predictions/`
  directory under `experiments/results/tourism/`; only the summary CSV
  exists. If raw forecasts cannot be regenerated cheaply, disclose in the
  checklist answer that Tourism ships at summary level.
- **Traffic capture-law five-basis run predictions.** Verify whether the
  Traffic basis-completeness runs saved predictions; if not, mark the same
  way.
- **Weather-96 preset/frontier arm predictions**, if the checklist answer
  claims them.

## 5. Verification gates before shipping

1. Run `verify_paper_numbers.py` against the restructured tables (spec §11
   item 4; expected values unchanged except the sanctioned Hourly params
   restatement 0.85M → 1.00M).
2. Resolve caveat C1 (plateau scheduler discrepancy).
3. Pin the Hourly capture-law value (§10 item 6) and the Tourism
   instantiability flags (§10 item 7).
4. Grep the final .tex: no `1.4e-24`, no "12–80", no Llama/perplexity
   content outside the withdrawal note; statistics quoted as exact p = 0.017
   (n = 5 floor) and rho = −0.98, p = 9.0e-18.
5. Assemble the bundle (configs + CSVs + .npz sets above), checksum it, and
   reference it from `checklist.tex`.
