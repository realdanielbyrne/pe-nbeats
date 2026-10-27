# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Reply in the most concise form possible. Skip pleasantries,
preambles, and recaps of my question. No phrases like
"I'd be happy to", "Great question", or "Let me explain".

Drop articles and filler words wherever the meaning stays clear. Prefer short declarative sentences. If a tool call is needed, run it first and show only the result. Do not narrate your steps.

Do not proactively suggest follow-up actions, next steps, testing recommendations, code improvements, alternative approaches, or future work after completing a task or answering a question. Only provide such suggestions when explicitly requested.

## Project Overview

PyTorch Lightning implementation of the N-BEATS (Neural Basis Expansion Analysis for Time Series) forecasting algorithm, published as the `pe-nbeats` PyPI package (import `pe_nbeats`; formerly `lightningnbeats`). Extends the original paper with wavelet basis expansion blocks, autoencoder variants, bottleneck generic blocks, and fully customizable stack compositions.

### Requirements

- **Python** >= 3.12, < 3.15
- **PyTorch** >= 2.1.0
- **Lightning** >= 2.1.0

Supports multiple accelerators: **CUDA** (NVIDIA GPUs), **MPS** (Apple Silicon via Metal Performance Shaders), and **CPU**. The `get_best_accelerator()` utility in `__init__.py` detects the best available accelerator (CUDA > MPS > CPU).

## Build & Install

```bash
pip install -e .                    # editable install from source
pip install pe-nbeats          # install from PyPI
python -m build                      # build distribution package
```

Dependencies are defined in `pyproject.toml`. Uses setuptools as build backend.

## Running Examples

Examples are in `examples/` and are designed to run as scripts or Jupyter cell-by-cell (`#%%` markers):

```bash
python examples/M4AllBlks.py       # M4 dataset benchmark across all block types
python examples/TourismAllBlks.py  # Tourism dataset benchmark
```

## YAML-driven Experiment Launcher (Recommended)

`experiments/run_from_yaml.py` is the **unified experiment launcher** that standardizes the workflow across all study types. It accepts all parameters through a YAML configuration file and wraps `run_single_experiment()` from `run_unified_benchmark.py`.

For Traffic/Weather benchmark configs that define a top-level `protocol:` block, the launcher forwards `train_ratio` / `include_target` into dataset loading, passes `normalize` / `val_ratio` into the columnar datamodule, and treats `protocol.loss` / `protocol.forecast_multiplier` / `protocol.batch_size` as fallbacks when the same keys are not explicitly set under `training`.

YAML / unified runs also use atomic claim files under `experiments/results/.claims/` to prevent duplicate work across parallel workers. Claim filenames follow the NHiTS-style pattern `<claim_config>__<dataset>__<horizon>__run<idx>.claim`; for multi-pass/search jobs the pass or experiment tag is prefixed into `claim_config` so `baseline` and `activeG_fcast` stay distinct. `hardware.worker_id` (or `--worker-id`) is recorded in the claim metadata.

```bash
# Run a pre-built config (dry-run first to check the plan)
python experiments/run_from_yaml.py experiments/configs/nbeats_g.yaml --dry-run
python experiments/run_from_yaml.py experiments/configs/nbeats_g.yaml

# Override settings on the command line
python experiments/run_from_yaml.py experiments/configs/nbeats_g.yaml \
    --dataset m4 --periods Yearly --n-runs 3 --max-epochs 50

# GenericAE successive halving search
python experiments/run_from_yaml.py experiments/configs/genericae_pure.yaml

# TrendAE alternating benchmark
python experiments/run_from_yaml.py experiments/configs/trendae_ae_alternating.yaml

# Analyze existing results without retraining
python experiments/run_from_yaml.py experiments/configs/nbeats_g.yaml --analyze-only

# Enable W&B logging
python experiments/run_from_yaml.py experiments/configs/nbeats_g.yaml --wandb
```

### Example YAML configs (in `experiments/configs/`)

| File | Description |
|------|-------------|
| `nbeats_g.yaml` | NBEATS-G — 30×Generic, two-pass (baseline + activeG_fcast), M4 all periods |
| `nbeats_ig.yaml` | NBEATS-I+G — Trend+Seasonality+28×Generic, two-pass |
| `genericae_pure.yaml` | GenericAE pure-stack successive halving search (40 configs, 3 rounds) |
| `trendae_ae_alternating.yaml` | TrendAE+AE alternating patterns benchmark |

### YAML schema

Full documentation of all supported keys: [`experiments/configs/schema.md`](experiments/configs/schema.md)

Key concepts:

- **`stacks`**: Defines the stack architecture using one of: `homogeneous`, `prefix_body`, `alternating`, `concat`, `explicit`, `builtin` (references `UNIFIED_CONFIGS`), direct list, or string shorthand (`"Generic:30"`).
- **`passes`**: Two-pass design (baseline + activeG_fcast). When absent, a single pass uses `experiment_name`.
- **Atomic claim files**: YAML/unified runs claim jobs before training via `experiments/results/.claims/`. Two-pass/search jobs fold the pass tag into the claim-side config name so each `(pass, config, dataset, period, run)` tuple is independently claimable.
- **`search`**: Successive halving with `rounds` list (each with `max_epochs`, `n_runs`, `keep_fraction` / `top_k`).
- **`extra_csv_columns`** + **`extra_fields`**: Extend the CSV schema for study-specific metadata.
- All CLI flags override YAML values. Resumability is built-in (skips already-written CSV rows).

### Replicating existing study scripts

| Old script | Equivalent YAML approach |
|-----------|--------------------------|
| `run_unified_benchmark.py` | Use `configs: [{builtin: NBEATS-G}, ...]` + `passes:` |
| `run_generic_ae_study.py` | See `configs/genericae_pure.yaml` |
| `run_trendae_study.py` | See `configs/trendae_ae_alternating.yaml` |
| Custom one-off experiment | Single top-level `stacks:` + `training:` |

The `experiments/run_experiments.py` script is **deprecated** (use YAML configs via `run_from_yaml.py` instead). It runs systematic benchmarks with multiple seeds across datasets:

```bash
python experiments/run_experiments.py --dataset m4 --part 1 --periods Yearly --max-epochs 50
python experiments/run_experiments.py --dataset traffic --part 1 --periods Traffic-96 --max-epochs 100
python experiments/run_experiments.py --dataset m4 --part 2 --periods Yearly Monthly --max-epochs 100
python experiments/run_experiments.py --dataset m4 --part all
python experiments/run_experiments.py --part 6 --max-epochs 100
python experiments/run_experiments.py --part 8 --max-epochs 100
python experiments/run_experiments.py --part 8 --max-epochs 100 --convergence-config Generic30_sumLosses
```

`--dataset`: `m4` (default), `traffic`, or `weather`. `--part 1`: Block-type benchmark (paper baselines + novel blocks at 30-stack scale). `--part 2`: Ablation studies (active_g, sum_losses, activations). `--part 6`: Convergence study (ignores `--dataset`; runs across both M4-Yearly and Weather-96 with random seeds). `--part 8`: sum_losses convergence study — 2x2 factorial (active_g x sum_losses) with 200 runs/config, per-epoch tracking, divergence detection, multi-dataset (ignores `--dataset`). `--convergence-config`: filter to a single config for parallel Part 6 or Part 8 execution. `--periods`: one or more of `Yearly`, `Quarterly`, `Monthly`, `Weekly`, `Daily`, `Hourly` for M4; `Traffic-96` for Traffic; `Weather-96` for Weather. Results are saved to dataset-specific subdirectories under `experiments/results/<dataset>/`.

## Testing

Tests are in `tests/` and use pytest:

```bash
pytest tests/                        # run all tests
pytest tests/ -v                     # verbose output
pytest tests/test_blocks.py          # run specific test file
pytest tests/test_blocks.py -k "TestGenericArchitecture"  # run single test class
pytest tests/test_blocks.py -k "test_output_shapes"       # run single test method
```

For the complete testing guide, including conventions for adding coverage and common troubleshooting tips, see [`TESTING.md`](TESTING.md).

Test files: `test_blocks.py` (block shapes, attributes, registries), `test_loaders.py` (DataModule setup, splits), `test_models.py` (width selection, optimizer dispatch, forward pass, sum_losses). No linter or formatter is configured.

## Architecture

### Package Structure (`src/pe_nbeats/`)

- **`models.py`** — Two model classes sharing a common `_NBeatsBase(pl.LightningModule)` superclass:
  - `NBeatsNet` — The standard N-BEATS model with doubly residual stacking.
  - `NHiTSNet` — NHiTS variant that adds multi-rate input pooling (`n_pools_kernel_size` per stack) and hierarchical forecast interpolation (`n_freq_downsample` per stack). All existing block types plug in unchanged; pooling/interpolation happens outside the block interface. Additional params: `n_pools_kernel_size`, `n_freq_downsample`, `interpolation_mode` (default `'linear'`).
  - Both accept a `stack_types` list of strings to define architecture. Handle forward pass with backward/forward residual connections, training/validation/test steps, loss configuration, and optimizer setup.
- **`blocks/blocks.py`** — All block implementations (the largest file). Two parallel inheritance hierarchies:
  - `RootBlock(nn.Module)` — Standard backbone: 4 FC layers with activation. Parent of `Generic`, `BottleneckGeneric`, `Seasonality`, `Trend`, `AutoEncoder`, `VAE`, `GenericAEBackcast`, `WaveletV3`, `TrendWavelet`, `TrendWaveletGeneric`, and concrete wavelet subclasses.
  - `AERootBlock(nn.Module)` — Autoencoder backbone: encoder (units → units/2 → latent_dim) then decoder (latent_dim → units/2 → units). Parent of `GenericAE`, `BottleneckGenericAE`, `TrendAE`, `SeasonalityAE`, `AutoEncoderAE`, `GenericAEBackcastAE`, `TrendWaveletAE`, and `TrendWaveletGenericAE`.
  - `AERootBlockLG(nn.Module)` — Learned-Gate AE backbone: same encoder-decoder structure as `AERootBlock` but adds a learnable `nn.Parameter` gate vector (`latent_gate`) of size `latent_dim`. Applies `sigmoid(gate) * z` after the latent layer, allowing the network to discover effective latent dimensionality during training. Parent of `GenericAELG`, `BottleneckGenericAELG`, `TrendAELG`, `SeasonalityAELG`, `AutoEncoderAELG`, `GenericAEBackcastAELG`, `TrendWaveletAELG`, and `TrendWaveletGenericAELG`.
  - `AERootBlockVAE(nn.Module)` — Variational AE backbone: replaces the deterministic bottleneck with a stochastic latent space. Uses two heads (`fc2_mu`, `fc2_logvar`) and the reparameterization trick (`z = mu + std * eps` during training, `z = mu` during eval). Stores `self.kl_loss` after each forward pass; KL loss is collected in `NBeatsNet.training_step()` and added to total loss with weight 0.1 (controlled by the `kl_weight` parameter, default `0.1`). Parent of `GenericVAE`, `BottleneckGenericVAE`, `TrendVAE`, `SeasonalityVAE`, `AutoEncoderVAE`, `GenericAEBackcastVAE`, and `TrendWaveletGenericVAE`.
  - Wavelet blocks (`HaarWaveletV3`, `DB2WaveletV3`, etc.) are thin subclasses that only set the wavelet type string. `WaveletV3` uses orthonormal DWT bases.
  - `TrendWavelet` merges polynomial trend and orthonormal DWT basis expansions in a single block. `TrendWaveletGeneric` extends it with a third learned low-rank generic basis branch; AE / LG / VAE descendants preserve the same additive decomposition.
  - Basis generators (`_SeasonalityGenerator`, `_TrendGenerator`, `_WaveletGeneratorV3`) produce non-trainable basis matrices registered as buffers.
  - V1 and V2 wavelet blocks were removed due to instability (NaN failures and MASE blow-ups) documented in the companion paper (now in the private `pe-nbeats-paper` repo). Use V3 wavelet variants.
- **`loaders.py`** — PyTorch Lightning DataModules and Datasets. Two data layout conventions:
  - **Row-oriented** (M4 format): rows = series, cols = time observations. `RowCollectionTimeSeriesDataModule` splits by time dimension; validation = last `backcast + forecast` columns.
  - **Columnar** (Tourism format): cols = series, rows = time observations. `ColumnarCollectionTimeSeriesDataModule` supports `no_val` mode. Short series are padded with zeros.
  - `TimeSeriesDataModule` / `TimeSeriesDataset` — single univariate series with 80/20 random split.
  - Test variants (`RowCollectionTimeSeriesTestModule`, `ColumnarCollectionTimeSeriesTestDataModule`) concatenate train tail + test head for evaluation.
- **`losses.py`** — Custom loss functions: `SMAPELoss`, `MAPELoss`, `MASELoss`, `NormalizedDeviationLoss`
- **`constants.py`** — String registries: `ACTIVATIONS`, `LOSSES`, `OPTIMIZERS`, `BLOCKS`. All configuration is resolved by string lookup against these lists.
- **`data/benchmark_dataset.py`** — `BenchmarkDataset(ABC)`: abstract base class all datasets implement. Interface: `train_data`, `test_data`, `forecast_length`, `frequency`, `name`, `supports_owa`, `compute_owa()`, `get_training_series()`.
- **`data/M4/`** — `M4Dataset(BenchmarkDataset)` loader with bundled CSV data files for all 6 M4 periods. Includes Naive2 baseline constants and OWA computation.
- **`data/Traffic/`** — `TrafficDataset(BenchmarkDataset)` loader for PeMS Traffic (862 sensors, hourly). Downloads from Hugging Face on first use, caches at `~/.cache/pe_nbeats/Traffic/`. Parameterized by `horizon` (96, 192, 336, 720).
- **`data/Weather/`** — `WeatherDataset(BenchmarkDataset)` loader for Weather (21 meteorological indicators, 10-min intervals). Downloads from Hugging Face on first use, caches at `~/.cache/pe_nbeats/Weather/`. Parameterized by `horizon` (96, 192, 336, 720).
- **`data/Tourism/`** — `TourismDataset(BenchmarkDataset)` loader for Tourism dataset. Columnar format (columns = series, rows = observations). Parameterized by `period` (Yearly, Quarterly, Monthly).
- **`data/Milk/`** — `MilkDataset(BenchmarkDataset)` loader for the Milk Production univariate time series.

### Key Design Patterns

- **`stack_types` is required**: `NBeatsNet` raises `ValueError` if `stack_types` is not provided. There is no default architecture.
- **String-based block dispatch**: `NBeatsNet.create_stack()` uses `getattr(b, stack_type)(...)` to instantiate blocks by name. Valid names must appear in `constants.BLOCKS`.
- **All blocks return `(backcast, forecast)` tuples**. The forward pass subtracts backcast from input (residual) and adds forecast to output.
- **`active_g` parameter**: Non-standard extension that applies activation to the final linear layers of Generic-type blocks. Default `False` (paper-faithful).
- **Weight sharing**: When `share_weights=True`, blocks within a stack reuse the first block's parameters.
- **`sum_losses`**: Adds weighted backcast reconstruction loss (0.25 × loss vs zeros) to forecast loss, pushing backcasts to fully reconstruct the input.
- **AE descendant dimension split**: In `AutoEncoderAE` / `AutoEncoderAELG` / `AutoEncoderVAE` and `GenericAEBackcastAE` / `GenericAEBackcastAELG` / `GenericAEBackcastVAE`, `latent_dim` controls the branch-local AE bottlenecks. `thetas_dim` remains reserved for explicit theta/basis projections such as the `GenericAEBackcast*` forecast heads.
- **Generic vs BottleneckGeneric**: `Generic` matches the paper (single linear projection to target length). `BottleneckGeneric` projects through `thetas_dim` bottleneck first (rank-d factorized basis expansion).
- **`trend_thetas_dim` / `generic_dim` routing**: `Trend`, `TrendWavelet`, and `TrendWaveletGeneric` families use `trend_thetas_dim` for polynomial order. `TrendWaveletGeneric*` additionally use `generic_dim` for the learned generic branch rank.
- **`forecast_basis_dim`**: Optional override for WaveletV3 forecast path basis dimensionality. When set, the forecast linear projection uses `forecast_basis_dim` (clamped to `forecast_length`) while the backcast path continues to use `basis_dim`. Allows asymmetric regularization when backcast and forecast lengths differ significantly. Default `None` (both paths use `basis_dim`).
- **Asymmetric wavelet families** (`backcast_wavelet_type`, `forecast_wavelet_type`): Optional overrides that allow using different wavelet families for the backcast and forecast paths. When set, the specified family is used for `_WaveletGeneratorV3` on that path; when `None`, both paths fall back to `wavelet_type`. This is useful when backcast and forecast lengths differ significantly (e.g., Traffic-96: L=480, H=96), allowing long-support wavelets (e.g., `sym20`) for the long backcast and short-support wavelets (e.g., `db3`, `coif2`) for the short forecast. Supported by all generic WaveletV3 base classes (`WaveletV3`, `WaveletV3AE`, `WaveletV3AELG`, `WaveletV3VAE2`, `WaveletV3VAE`) plus the hybrid `TrendWavelet` / `TrendWaveletGeneric` families (`TrendWavelet`, `TrendWaveletAE`, `TrendWaveletAELG`, `TrendWaveletGeneric`, `TrendWaveletGenericAE`, `TrendWaveletGenericAELG`, `TrendWaveletGenericVAE`). Generic base classes are the recommended public API; family-named subclasses (e.g., `HaarWaveletV3`) remain as compatibility wrappers.
- **WaveletV3 short-target behavior**: `_WaveletGeneratorV3` respects `pywt.dwt_max_level(...)=0` and keeps `level=0` rather than forcing an invalid level-1 decomposition. Short targets paired with long-support families (`db20`, `sym20`, `coif10`) therefore become approximation-only orthonormal bases rather than true multilevel decompositions; prefer short-support families (`haar`, `db2`, `db3`) for short horizons.
- **`skip_distance` / `skip_alpha` parameters**: ResNet-style skip connections that periodically re-inject the original input into the residual stream. After every `skip_distance` stacks (except the last), `alpha * original_input` is added to the residual `x`. Combats signal/gradient decay in deep stacks. `skip_alpha` can be a float or `"learnable"` (creates an `nn.Parameter` initialized to 0.01). Default: disabled (`skip_distance=0`). Supported by both `NBeatsNet` and `NHiTSNet`.

### Width Parameter Mapping

The `create_stack` method selects hidden layer width by block type:

| Width param | Default | Block types |
|---|---|---|
| `g_width` | 512 | `Generic`, `BottleneckGeneric`, `GenericAE`, `BottleneckGenericAE`, `GenericAEBackcast`, `GenericAEBackcastAE`, all wavelet blocks |
| `s_width` | 2048 | `Seasonality`, `SeasonalityAE` |
| `t_width` | 256 | `Trend`, `TrendAE`, `TrendWavelet`, `TrendWaveletAE`, `TrendWaveletAELG`, `TrendWaveletGeneric`, `TrendWaveletGenericAE`, `TrendWaveletGenericAELG`, `TrendWaveletGenericVAE` |
| `ae_width` | 512 | `AutoEncoder`, `AutoEncoderAE` |

### Adding a New Block Type

1. Create the block class in `blocks/blocks.py`, inheriting from `RootBlock`, `AERootBlock`, `AERootBlockLG`, or `AERootBlockVAE`. Must implement `forward()` returning `(backcast, forecast)`.
2. Add the class name string to the `BLOCKS` list in `constants.py`.
3. If the block needs a new width parameter, add the mapping in `NBeatsNet.create_stack()` in `models.py`.
4. Add a shape test in `tests/test_blocks.py` — the parametrized `TestAllBlocksOutputShapes` will automatically cover it if it's in `BLOCKS`.

### Adding a New Dataset

1. Create a new directory under `src/pe_nbeats/data/<DatasetName>/` with `__init__.py` and a dataset module.
2. Create a class inheriting from `BenchmarkDataset` in `data/benchmark_dataset.py`. Must set `train_data` (DataFrame, columnar), `test_data`, `forecast_length`, `frequency`, and `name`.
3. Override `supports_owa = True` and `compute_owa()` if Naive2 baselines exist.
4. Add the import to `src/pe_nbeats/data/__init__.py`.
5. In `experiments/run_unified_benchmark.py`: extend `load_dataset()` and add the dataset choice to the CLI `--dataset` argument. Create a YAML config in `experiments/configs/` to drive experiments via `run_from_yaml.py`.

## Empirical Defaults from M4 Sweeps

### Headline (2026-07-28): parameter savings come from trunk width

Supersedes the wavelet/AE framing in the tables below. Evidence: `trunk_width_attribution_2026-07-27.md`, `tourism_preset_validation_2026-07-28.md`, `nbeats_slim_recommendations.md`, `identity_basis_disclosure_2026-07-27.md`.

- **Trunk width is the mechanism.** Width 2048→128 is accuracy-neutral on M4 (KW p≈0.78 over 48.8× params) and on Tourism. Width-matched basis comparisons are null; AE backbone and learned gate are null at matched budget. Exception: Traffic-96 improves with width (w128 26.49 → w512 25.45).
- **Basis identity is inert at full forecast rank.** Default basis is `wavelet_type="dct"`. Legacy SVD wavelet bases scramble band order (`basis_offset` selects no band; use `"ordered:<family>"`), and coif2/sym10 (5/6 M4 periods) and db3 (H ≤ 8) are silently the identity. Both cases now emit `UserWarning`.
- **Out-of-box API:** `NBeatsNet.slim(L, H, trend_branch=, rectify=, width=128)` and `NBeatsNet.from_recommendation(pe_nbeats.recommend(windows), L)`. Constructor defaults stay paper-faithful.
- **`sum_losses`:** `val_loss`/`test_loss` are forecast-only since 0.2.0; combined objective logged as `val_total_loss`. Pre-0.2.0 `best_val_loss` for `sum_losses=True` rows includes the backcast term.
- Per-period tables below remain valid as measured numbers, but wavelet-family labels for coif2/sym10 outside Hourly describe an identity basis.

These are recommended starting points for new M4 experiments. Drawn from two complementary sweeps; full evidence in `experiments/analysis/analysis_reports/`.

### Best per-period configs (M4) — refreshed 2026-05-03

| Period | Paper-sample best (LR) | Sliding best |
|---|---|---|
| Yearly    | `T+Coif2V3_30s_bdeq` plateau (15.2M, 13.542) — sub-1M: `TWAE_10s_ld32_ag0` plateau (0.48M, 13.546) | `TW_10s_td3_bdeq_coif2` (2.1M, 13.499) |
| Quarterly | `NBEATS-IG_10s_ag0` plateau (19.6M, **10.313**) | `NBEATS-IG_10s_ag0` (19.6M, 10.127) |
| Monthly   | `TW_30s_td3_bdeq_sym10` plateau (6.8M, **13.240**) | `TW_30s_td3_bd2eq_coif2` (7.1M, 13.279) |
| Weekly    | `T+Coif2V3_30s_bdeq` step_paper (15.8M, 6.735) — best plateau-tiered `TAE+Sym10V3AE_30s_tiered` 6.935 ± 0.223 (n=10) loses by +0.20¹ | `T+Db3V3_30s_bdeq` (15.8M, 6.671) |
| Daily     | `T+Sym10V3_10s_tiered_ag0` plateau (5.25M, **3.012**) — Pareto: `TAELG+Sym10V3AELG_10s_tiered` plateau (1.14M, 3.013) | `NBEATS-G_30s_ag0` (26.0M, 2.588) |
| Hourly    | `NBEATS-IG_30s_agf` step_paper (43.6M, 8.758) | `NBEATS-IG_30s_agf` (43.6M, 8.587) |

¹ The single-seed Weekly p3_recommended hit 6.559 SMAPE in `tiered_offset_m4_weekly_plateau_validation_results.csv` was a 1.4-σ bottom-tail draw; the n=10 plateau expansion (`tiered_offset_m4_allperiods_results.csv`, same plateau cell) returns 6.987 ± 0.301 on the same `T+Sym10V3_30s_tiered_ag0` config (best plateau-tiered Weekly cell of any backbone is `TAE+Sym10V3AE_30s_tiered` 6.935 ± 0.223, MWU p=0.12 vs 6.735 step_paper SOTA, n=10/10). Weekly stays on **step_paper LR + non-tiered**; tiered offset is locked out at H=13.

Different protocols crown different per-period winners. Pick one protocol per study; do not mix `nbeats_paper` and `sliding` for absolute SMAPE comparisons.

### Best M4 generalists

- **Paper-sample, all-6-period coverage (CROWN):** `T+Sym10V3_10s_tiered_ag0` (mean rank **13.33/108**). 5.07–6.06M params, top-11 on every M4 period (Daily #1, Hourly #5, Quarterly #6, Yearly #11, Weekly #16, Monthly #41). Hourly cell uses plateau LR + descend tiering direction (canonicalized from `T+Sym10V3_10s_bdEQ_descend` in `m4_hourly_sym10_tiered_offset_results.csv`); the other 5 periods use the all-periods CSVs with `tiered=ascend`.
- **Paper-sample, paper-faithful alternative:** `NBEATS-IG_30s_ag0` (mean rank 16.0/68). 38.13–43.58M params. Use when reproducing Oreshkin et al. 2020 numbers or when zero divergence is required.
- **Sliding:** `T+HaarV3_30s_bd2eq` (mean rank 12.67/112) — unrivaled cross-period generalist; 16.25M params. Smaller alternative: `TALG+HaarV3ALG_30s_ag0` (3.66M, rank 21.0).

### Sub-1M parameter champion (M4)

`TWAELG_10s_ld32_db3_*` — 0.48–0.85M params, top-5 on Yearly, Daily, Hourly. On Daily, `TWGAELG_10s_ld16_db3_ag0` (0.52M, 3.051) is best parameter efficiency anywhere on M4.

### Defaults for new M4 experiments

- **LR scheduler:** plateau (ReduceLROnPlateau) is the paper-sample default for Quarterly / Monthly / Daily — strictly beats step_paper by Δ−0.03 to −0.20 SMAPE (n=18–52 matched pairs). **Weekly is locked on step_paper LR**: n=10 plateau expansion (2026-05-04) confirms plateau LR regresses on Weekly across all tested patience cells (1/2/3/5) regardless of tiering. Yearly stays on step_paper (neutral). Cosine-warmup is the sliding default. See [`m4_overall_leaderboard_2026-05-03.md`](experiments/analysis/analysis_reports/m4_overall_leaderboard_2026-05-03.md) §4.1.
- **Sampling protocol:** sliding wins absolute SMAPE on long-horizon periods (Daily H=14, Hourly H=48 — gaps of −0.42 / −0.17 vs paper-sample). Paper-sample wins or ties on short horizons (Y/M). Pick one protocol per study; for paper-faithful comparison to Oreshkin et al. 2020, use paper-sample with plateau LR.
- **Wavelet shortlist:** haar, db3, coif2, sym10. Coif3 produces no per-period SOTA — drop from default M4 sweeps.
- **`active_g`:** default `False` (paper-faithful). `active_g=forecast` (`agf`) helps on **Hourly only** as a robust default; loses or ties elsewhere. Never use as a global default.
- **Stack architecture:** when ≥15M params is acceptable, prefer alternating `T+<wav>V3` (RootBlock) over unified `TW`/`TWAE`/`TWAELG` on Weekly; on Monthly, plateau LR has reversed this — `TW_30s_td3_bdeq_sym10` plateau (13.240) is now the leader. Reserve unified for sub-1.5M parameter targets.
- **AE vs AELG:** equivalent at matched configurations on 5/6 periods (Daily exception: RB beats AELG by 0.17 SMAPE). Pick AELG when latent-dim/parameter count is the binding constraint (its native `ld=16` halves AE-`ld=32` cost with no consistent SMAPE penalty). **VAE family is strictly unusable on M4** — pure VAE configs collapse to 55–68 SMAPE on Q/M/W/D.
- **Stack depth:** novel wavelet/trend-bias families are stable across 10s↔30s; paper Generic blocks (`NBEATS-G`, `Generic_*`) collapse at 30 stacks on Quarterly/Weekly/Monthly (+2 SMAPE, bimodal collapse, std 7.4–9.5) and should be capped at 10 stacks there. NBEATS-IG is the stable paper baseline at any depth.
- **`basis_dim`:** `bdeq` (`basis_dim=forecast_length`) is the M4 default. `bd2eq` is statistically harmful on Daily/Quarterly under sliding (Wilcoxon p=0.003 pooled).
- **Tiered basis-offset scope:** keep tiered offset for **Daily only** in production defaults (decisive win — 8/10 paper-sample Daily top-10 are tiered, Cliff's d 0.54–0.78). The earlier "tiered helps Monthly" claim is **retracted** — under matched plateau LR, non-tiered `TW_30s_td3_bdeq_sym10` (13.240) beats tiered `T+DB3V3_30s_tiered_agf` (13.344). Yearly/Quarterly tiered improvements are within seed noise. Hourly tiered does NOT beat the paper baseline. **Weekly tiered is RESOLVED as a regression** (2026-05-04, n=10 plateau): best tiered Weekly cell `TAE+Sym10V3AE_30s_tiered` 6.935 vs canonical `T+Coif2V3_30s_bdeq` 6.735 (+0.20). Single-seed 6.559 from p3_recommended did not survive — Weekly is a real cascade reversal at H=13.
- **Drop:** all `BNG*` / `BNAE*` / `BNAELG*` (BottleneckGeneric family — universally worst on M4); pure `GenericVAE_*` and all VAE configs (5–10× SMAPE worse on M4); `GAE_*` / `GAELG_*` / `GenericAE_*_sw0` / `GenericAELG_*_sw0` (pure Generic AE/AELG bottoms out at mean rank ~38); all `_sd5` skip variants (never help on M4); all `*_coif3` variants on M4 (no per-period SOTA); `_30s_agf` tiered configs at `_10s` depth (run-to-run divergence outliers); step_paper LR on Q/M/D (use plateau instead).

### Early-stopping settings for `sampling_style: nbeats_paper`

The paper-sample protocol requires sub-epoch validation to avoid `best_epoch=0/1` collapse:

```yaml
training:
  val_check_interval: 100   # validate every 100 steps (≈10 checks per Lightning epoch)
  min_delta: 0.001          # require 0.1% improvement to reset patience
  patience: 20              # in val-check units (= 200 raw steps minimum before stop can fire)
```

Without these, paper-sample early-stopping fires on the first (warmup-corrupted) validation check.

### Convention: `agf` vs `ag0`

`active_g` is a repository extension to N-BEATS, not part of Oreshkin et al. 2020. Configs labelled `*_ag0` (`active_g=False`) reproduce the published architecture. Configs labelled `*_agf` (`active_g=forecast`) apply a novel repo extension on top of the paper backbone — they share the architecture but are not paper-faithful in the strict sense. When reporting "paper baseline" SMAPE, only `*_ag0` results count.

## CI/CD

GitHub Actions workflow (`.github/workflows/python-publish.yml`) runs the core test suite on CPU, builds, runs `twine check`, then publishes `pe-nbeats` to PyPI via Trusted Publishing on GitHub release. No linter is configured.
