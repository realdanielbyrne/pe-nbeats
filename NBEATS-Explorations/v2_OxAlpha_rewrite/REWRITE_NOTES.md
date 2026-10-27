# Rewrite Notes — v2_OxAlpha_rewrite

*2026-08-23. Assembler ledger for the camera-ready rewrite of NeurIPS 2026
submission 30981 into "Frozen Basis Time Series Forecasting", executed per
[`../rebuttal/paper_rewrite_plan.md`](../rebuttal/paper_rewrite_plan.md) with the
notation/taxonomy governed by [`../rebuttal/notation_and_taxonomy_spec.md`](../rebuttal/notation_and_taxonomy_spec.md).*

## What is here

| Path | Contents |
|---|---|
| `arxiv_preprint_v2.tex` | Main file; preamble inherited from v1; inputs all fragments below |
| `v1_arxiv_preprint_baseline.tex` | Untouched copy of the v1 source of truth |
| `sections/00_titlepage_abstract.tex` | Title (both decision-gate options documented) + abstract |
| `sections/01_introduction.tex` | Thesis, contributions, scope conditions, frontier framing, corrections framing, Transformer withdrawal note |
| `sections/02_unified_formulation.tex` | Method from spec §§2–7: formulation eq., dimension table, synthesis heads, rotation absorption, predictors, taxonomy, disambiguation, marking rules |
| `sections/03_trunk_result.tex` | Trunk result + unrun disclosures (§10 items 1–3, 5) |
| `sections/04_completeness_result.tex` | Rotation absorption, 36 comparisons, four scope conditions, boundary test |
| `sections/05_predictors.tex` | Captured energy + effective rank evidence incl. Traffic-96 and Hourly capture law |
| `sections/06_experimental_setup.tex`, `06_results.tex` | Setup; results with scoped NHiTS subsection; Tourism 20.864 handled as prior exploratory finding |
| `sections/07_recipe_and_guidance.tex` | Practitioner recipe (main text, boxed quick-pick — decision gate resolved) |
| `sections/08_related_work.tex`, `sections/09_limitations_conclusion.tex` | Related work repositioned; limitations mirroring abstract; short conclusion (decision gate: included) |
| `sections/A0…A4_appendix_*.tex` | Protocol, hyperparameters, configs (verbatim YAML shipped), ablations, full-tables lead-in |
| `tables/*.tex` + `tables/MANIFEST.md` | All surviving tables as standalone booktabs files; MANIFEST carries the complete 29-legacy-label disposition register |
| `Reproducibility_statement.md` | Authors-only bundle checklist (configs / CSVs / prediction exports, MISSING items marked) |

## Decision gates (plan §13) — resolved

1. **Title**: bare form — *Frozen Basis Time Series Forecasting*. Subtitle option preserved as a comment in `00_titlepage_abstract.tex`.
2. **Conclusion**: included (short form).
3. **Practitioner recipe**: main text with boxed summary.
4. **Unrun experiments** (§10 items 1–3 AE row / taper sweep / weight-decay arm, item 5 Traffic trunk-width rerun): disclosed plainly in `03_trunk_result.tex` and recapped in Limitations.
5. **`paper.md`**: archived as stale (per plan §1); not regenerated.

## Verification performed at assembly

- **Label closure**: zero duplicate `\label`s across `sections/` + `tables/`; every `\ref`/`\eqref` target resolves.
- **Citation closure**: every `\citep`/`\citet` key exists in `references.bib`.
- **Banned-token sweep**: no `1.4e-24`; "12–80×" appears only inside `corrections_ledger.tex` as corrected history; typo class clean; Llama/perplexity content confined to the withdrawal note and ledger history.
- **Statistics consistency**: p-floor 0.017 (n = 5); run-level ρ = −0.98, p = 9.0e-18; KW p = 0.776/0.773; completeness min p = 0.084; Hourly db3 pinned to **10.580 ± 0.102** everywhere printable (the rebuttal's 10.587 appears only in a comment and the working note).
- **Abstract-number traceability** (plan §14 item 6): every abstract figure verified present in the body (48.8-fold; 64,645,150 → 1,323,550; 0.098; 0.084; −1.0; +0.949; 200 ceiling; 0.11; collapse counts; 29.37 vs 26.24; 240 runs).
- **Environment/brace balance**: verified programmatically across all fragments.
- **Sanctioned params change** (§14 item 5): frontier Hourly compact arm restated 1.00M inclusive of frozen basis matrices, footnoted, ratio kept at 51×.

## Known open items for the authors

1. **LaTeX build**: no LaTeX toolchain is available in this environment (and Docker socket is permission-denied), so the document has **not been compiled**. Run `pdflatex arxiv_preprint_v2 && bibtex arxiv_preprint_v2 && pdflatex ×2`. Fragments were validated structurally only.
2. **`verify_paper_numbers.py`** (plan §14 item 5) has not been run against the restructured tables.
3. **Spec §11 checklist** items requiring a compiled build (script cross-check, number invariance) are pending that build.
4. **Plateau-scheduler discrepancy** (repo YAML factor 0.75/cooldown 0 vs paper 0.5/patience 3/cooldown 1) and **early-stop patience** (15 vs 20) flagged by the appendix pass — reconcile before submission.
5. **Plan §10 items 6–7** (pin 10.580 against CSVs; Tourism instantiability flags) still open; §10 items 1–3 and 5 unrun and disclosed as such.
6. Missing prediction exports (Tourism preset validation, Traffic five-basis) marked MISSING in `Reproducibility_statement.md`.
7. If §10 experiments land before submission, replace the disclosure paragraphs in `03_trunk_result.tex` / `09_limitations_conclusion.tex` with results.
