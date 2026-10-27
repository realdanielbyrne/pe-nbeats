# Style Guide — k3_v2_rewrite (BINDING for all section fragments)

Every fragment is a LaTeX **fragment**: it must start with `\section{...}` or `\subsection{...}`
(no `\documentclass`, no `\begin{document}`), and will be pulled into `main.tex` via `\input`.

## Labels (use exactly these; cite with `\ref{}` / `\label{}`)

Sections: `sec:intro`, `sec:method`, `sec:trunk`, `sec:completeness`, `sec:truncation`,
`sec:trainability`, `sec:tourism`, `sec:scoped`, `sec:related_recipe_limits`,
`sec:transfer_summary` (NHiTS subsection of sec:scoped), `sec:limitations` (subsection of 9),
`sec:experimental_setup` (app A), `sec:wavelet` (app C wavelet construction paragraph),
`sec:ablations` (app D), `app:protocols`, `app:tables`, `app:configs`, `app:ood`,
`app:drop`, `app:repro`.

Tables: `tab:dims` (N1), `tab:taxonomy` (N2), `tab:legend` (N3), `tab:identity` (N4),
`tab:datasets`, `tab:main_leaderboard`, `tab:sub1m_frontier`, `tab:trunk_width`,
`tab:hourly_third_rank`, `tab:traffic_capture`, `tab:effective_rank`, `tab:sign_failure`,
`tab:tourism_predictions`, `tab:pramanick`, `tab:corrections`, `tab:active_g_ablation`,
`tab:tiered_per_period`, `tab:lr_matched`, `tab:hyperparams_defaults`, `tab:sweep_grids`,
`tab:configs_m4`, `tab:configs_ood`, `tab:m4_yearly_full`, `tab:m4_quarterly_full`,
`tab:m4_monthly_full`, `tab:m4_weekly_full`, `tab:m4_daily_full`, `tab:m4_hourly_full`,
`tab:generalist_paper`, `tab:generalist_sliding`, `tab:tourism_winners`.

Figures: `fig:block_diagram` (F1), `fig:acc_params` (F2), `fig:rank_error` (F3).

## Macros (defined in main.tex preamble — use them, do not redefine)

`\L` = lookback length L, `\Ho` = horizon H, `\S` = number of stacks S, `\U` = trunk width U,
`\db` = d_b, `\df` = d_f, `\Bb` = B_b, `\Bf` = B_f, `\Pb` = P_b, `\Pf` = P_f,
`\smape` = sMAPE.

## Notation rules (spec §3, binding)

- `\phi` = trunk activation (ReLU). NEVER `\sigma` for the trunk.
- `\mathrm{sigm}(\cdot)` = gate sigmoid (appears at most once, in §2).
- `v` = learned gate vector (appears at most once, in §2).
- `\times 0.5` = LR halving factor. NEVER `\gamma`.
- `B_b \in \mathbb{R}^{d_b \times L}`, `B_f \in \mathbb{R}^{d_f \times H}` with orthonormal rows;
  `P_b`, `P_f` the projection matrices. NEVER `\Psi`.
- `\theta^b`, `\theta^f` = coefficient vectors. `p` = polynomial order. `r` = generic-branch rank.
- Baselines spelled `N-BEATS-G`, `N-BEATS-I+G` (Oreshkin's names).

## Banned in main body AND appendix tables (spec §11.7)

- Wavelet family tokens (`haar`, `db2`, `db3`, `coif2`, `sym10`, ...) outside Appendix C
  (hyperparameter appendix, API names once) and Table N4 (`tab:identity`).
- Config-string suffix tokens `_10s`, `_30s`, `ld`, `td`, `ttd`, `bd`, `ag0`, `agf`, `sw`, `V3`
  outside Appendix C and N4. Tables use structured columns instead (spec §5).
- `sum_losses` comparisons, normalization comparisons, `1.4e-24`, the strings "12-80" and
  "60--95%" (except in the corrections ledger as corrected history), Llama/perplexity content
  outside the single withdrawal note in §8.
- No sentence may compare results across `sum_losses` settings or across normalization.

## Number rules

- All numbers must come from the plan/rebuttal/current-paper values listed in your task brief.
  Do not invent, recompute, or re-aggregate any number. Representative-row collapse only.
- Markers: `†` = active_g=forecast (lightningnbeats extension, not paper-faithful);
  `ᵍ` = AELG row merged into AE pool; `‡` = identity-basis cell (scope per spec §7).
- Params include frozen bases: Hourly sub-1M entry is 1.00M, not 0.85M.
- Oreshkin citation year: 2020 everywhere.

## Tone

- Camera-ready prose, no "laboratory notebook" narration. Claims scoped exactly as the plan
  scopes them. Where an experiment is unrun, say so plainly with the reason (never an ambiguous gap).
- Failed registered predictions are reported as failures (§7), not buried.
