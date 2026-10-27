# AGENTS.md

Guidance for coding agents working in this repository.

## Project Overview

PyTorch Lightning implementation of the N-BEATS (Neural Basis Expansion Analysis for Time Series) forecasting algorithm, published as the `pe-nbeats` PyPI package (import `pe_nbeats`; formerly `lightningnbeats`). Extends the original paper with wavelet basis expansion blocks, autoencoder variants, bottleneck generic blocks, and fully customizable stack compositions.

## Requirements

- Python >= 3.12, < 3.15
- PyTorch >= 2.1.0
- Lightning >= 2.1.0

Supports CUDA, MPS, and CPU. The `get_best_accelerator()` utility in `src/pe_nbeats/__init__.py` detects the best available accelerator (CUDA > MPS > CPU).

## Build & Install

```bash
pip install -e .
pip install pe-nbeats
python -m build
```

Dependencies are defined in `pyproject.toml` (setuptools build backend).

## Running Examples

Examples are in `examples/` and are designed to run as scripts or Jupyter cell-by-cell (`#%%` markers):

```bash
python examples/M4AllBlks.py
python examples/TourismAllBlks.py
```

The `experiments/run_experiments.py` script runs systematic benchmarks with multiple seeds across datasets:

```bash
python experiments/run_experiments.py --dataset m4 --part 1 --periods Yearly --max-epochs 50
python experiments/run_experiments.py --dataset traffic --part 1 --periods Traffic-96 --max-epochs 100
python experiments/run_experiments.py --dataset m4 --part 2 --periods Yearly Monthly --max-epochs 100
python experiments/run_experiments.py --dataset m4 --part all
python experiments/run_experiments.py --part 6 --max-epochs 100
python experiments/run_experiments.py --part 8 --max-epochs 100
python experiments/run_experiments.py --part 8 --max-epochs 100 --convergence-config Generic30_sumLosses
```

Notes:

- `--dataset`: `m4` (default), `traffic`, or `weather`
- `--part 1`: Block-type benchmark (paper baselines + novel blocks at 30-stack scale)
- `--part 2`: Ablation studies (active_g, sum_losses, activations)
- `--part 6`: Convergence study (ignores `--dataset`; runs across M4-Yearly and Weather-96 with random seeds)
- `--part 8`: sum_losses convergence study (2x2 factorial: active_g x sum_losses) with 200 runs/config, per-epoch tracking, divergence detection, multi-dataset (ignores `--dataset`)
- `--convergence-config`: filter to a single config for parallel Part 6 or Part 8 execution
- `--periods`: one or more of `Yearly`, `Quarterly`, `Monthly`, `Weekly`, `Daily`, `Hourly` for M4; `Traffic-96` for Traffic; `Weather-96` for Weather
- Results are saved under `experiments/results/<dataset>/`

## Testing

Tests are in `tests/` and use pytest:

```bash
pytest tests/
pytest tests/ -v
pytest tests/test_blocks.py
pytest tests/test_blocks.py -k "TestGenericArchitecture"
pytest tests/test_blocks.py -k "test_output_shapes"
```

For the complete testing guide, including conventions for adding coverage and common troubleshooting tips, see [`TESTING.md`](TESTING.md).

Test files: `test_blocks.py` (block shapes, attributes, registries), `test_loaders.py` (DataModule setup, splits), `test_models.py` (width selection, optimizer dispatch, forward pass, sum_losses). No linter or formatter is configured.

## Architecture

### Package Structure (`src/pe_nbeats/`)

- `models.py`: `NBeatsNet(pl.LightningModule)` main model. Accepts `stack_types` list of strings to define architecture. Handles forward pass with backward/forward residual connections, training/validation/test steps, loss configuration, and optimizer setup.
- `blocks/blocks.py`: all block implementations (largest file). Two parallel inheritance hierarchies:
  - `RootBlock(nn.Module)`: standard backbone. Parent of `Generic`, `BottleneckGeneric`, `Seasonality`, `Trend`, `AutoEncoder`, `VAE`, `GenericAEBackcast`, `WaveletV3`, `TrendWavelet`, `TrendWaveletGeneric`, and concrete wavelet subclasses.
  - `AERootBlock(nn.Module)`: autoencoder backbone. Parent of `GenericAE`, `BottleneckGenericAE`, `TrendAE`, `SeasonalityAE`, `AutoEncoderAE`, `GenericAEBackcastAE`, `TrendWaveletAE`, and `TrendWaveletGenericAE`.
  - `AERootBlockLG(nn.Module)`: Learned-Gate AE backbone: same encoder-decoder as `AERootBlock` but adds a learnable `nn.Parameter` gate vector (`latent_gate`) of size `latent_dim`. Applies `sigmoid(gate) * z` after the latent layer. Parent of `GenericAELG`, `BottleneckGenericAELG`, `TrendAELG`, `SeasonalityAELG`, `AutoEncoderAELG`, `GenericAEBackcastAELG`, `TrendWaveletAELG`, and `TrendWaveletGenericAELG`.
  - `AERootBlockVAE(nn.Module)`: Variational AE backbone: stochastic latent space with `fc2_mu`/`fc2_logvar` heads and reparameterization trick. Stores `self.kl_loss`; collected in `training_step()` with weight 0.001. Parent of `GenericVAE`, `BottleneckGenericVAE`, `TrendVAE`, `SeasonalityVAE`, `AutoEncoderVAE`, `GenericAEBackcastVAE`, and `TrendWaveletGenericVAE`.
  - Wavelet blocks (`HaarWaveletV3`, `DB2WaveletV3`, etc.) are thin subclasses that set the wavelet type string.
  - `TrendWavelet` merges polynomial trend and orthonormal DWT basis expansions in one block. `TrendWaveletGeneric` adds a third learned low-rank generic basis branch; AE / LG / VAE descendants keep that three-way additive decomposition.
  - Basis generators (`_SeasonalityGenerator`, `_TrendGenerator`, `_WaveletGeneratorV3`) produce non-trainable basis matrices registered as buffers.
  - V1 and V2 wavelet blocks were removed due to instability. Use V3 wavelet variants.
- `loaders.py`: Lightning DataModules and Datasets. Two data layout conventions:
  - Row-oriented (M4 format): rows = series, cols = time observations. Validation = last `backcast + forecast` columns.
  - Columnar (Tourism format): cols = series, rows = time observations. Supports `no_val` mode. Short series are padded with zeros.
  - `TimeSeriesDataModule` / `TimeSeriesDataset`: single univariate series with 80/20 random split.
  - Test variants concatenate train tail + test head for evaluation.
- `losses.py`: `SMAPELoss`, `MAPELoss`, `MASELoss`, `NormalizedDeviationLoss`
- `constants.py`: string registries `ACTIVATIONS`, `LOSSES`, `OPTIMIZERS`, `BLOCKS`
- `data/benchmark_dataset.py`: `BenchmarkDataset(ABC)` interface for datasets.
- `data/M4/`: `M4Dataset` loader with bundled CSV data files for all 6 M4 periods. Includes Naive2 baseline constants and OWA computation.
- `data/Traffic/`: `TrafficDataset` loader for PeMS Traffic. Downloads from Hugging Face on first use, caches at `~/.cache/pe_nbeats/Traffic/`. Parameterized by `horizon`.
- `data/Weather/`: `WeatherDataset` loader for Weather. Downloads from Hugging Face on first use, caches at `~/.cache/pe_nbeats/Weather/`. Parameterized by `horizon`.

### Key Design Patterns

- `stack_types` is required. `NBeatsNet` raises `ValueError` if it is not provided.
- String-based block dispatch: `NBeatsNet.create_stack()` uses `getattr(b, stack_type)(...)` and valid names must appear in `constants.BLOCKS`.
- All blocks return `(backcast, forecast)` tuples. Forward pass subtracts backcast from input (residual) and adds forecast to output.
- `active_g`: applies activation to the final linear layers of Generic-type blocks. Default `False`.
- Weight sharing: when `share_weights=True`, blocks within a stack reuse the first block's parameters.
- `sum_losses`: adds weighted backcast reconstruction loss (0.25 x loss vs zeros) to forecast loss.
- `trend_thetas_dim` / `generic_dim`: Trend, TrendWavelet, and TrendWaveletGeneric families use `trend_thetas_dim` for polynomial order; `TrendWaveletGeneric*` additionally use `generic_dim` for the learned generic branch rank.
- `forecast_basis_dim`: optional override for WaveletV3 forecast basis dimension. Default `None` (both paths use `basis_dim`).
- Asymmetric wavelet families (`backcast_wavelet_type`, `forecast_wavelet_type`) are supported by the generic WaveletV3 base classes and the hybrid `TrendWavelet` / `TrendWaveletGeneric` families.

### Width Parameter Mapping

`create_stack()` selects hidden layer width by block type:

| Width param | Default | Block types |
|---|---|---|
| `g_width` | 512 | `Generic`, `BottleneckGeneric`, `GenericAE`, `BottleneckGenericAE`, `GenericAEBackcast`, `GenericAEBackcastAE`, all wavelet blocks |
| `s_width` | 2048 | `Seasonality`, `SeasonalityAE` |
| `t_width` | 256 | `Trend`, `TrendAE`, `TrendWavelet`, `TrendWaveletGeneric`, `TrendWaveletGenericAE`, `TrendWaveletGenericAELG`, `TrendWaveletGenericVAE` |
| `ae_width` | 512 | `AutoEncoder`, `AutoEncoderAE` |

## Adding a New Block Type

1. Create the block class in `src/pe_nbeats/blocks/blocks.py`, inheriting from `RootBlock`, `AERootBlock`, `AERootBlockLG`, or `AERootBlockVAE`. Implement `forward()` returning `(backcast, forecast)`.
2. Add the class name string to the `BLOCKS` list in `src/pe_nbeats/constants.py`.
3. If the block needs a new width parameter, add the mapping in `NBeatsNet.create_stack()` in `src/pe_nbeats/models.py`.
4. Add a shape test in `tests/test_blocks.py`. The parametrized `TestAllBlocksOutputShapes` will cover it if it's in `BLOCKS`.

## Adding a New Dataset

1. Create a new directory under `src/pe_nbeats/data/<DatasetName>/` with `__init__.py` and a dataset module.
2. Create a class inheriting from `BenchmarkDataset` in `src/pe_nbeats/data/benchmark_dataset.py`. Set `train_data`, `test_data`, `forecast_length`, `frequency`, and `name`.
3. Override `supports_owa = True` and `compute_owa()` if Naive2 baselines exist.
4. Add the import to `src/pe_nbeats/data/__init__.py`.
5. In `experiments/run_experiments.py`: add a horizons dict, add to `DATASET_DEFAULTS`, extend `load_dataset()`, and add the dataset choice to the CLI `--dataset` argument.

## CI/CD

GitHub Actions workflow in `.github/workflows/python-publish.yml` runs the core tests on CPU, builds, runs `twine check`, and publishes `pe-nbeats` to PyPI via Trusted Publishing on GitHub release.
