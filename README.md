# PE-NBEATS: Parameter-Efficient N-BEATS

A [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) implementation of [N-BEATS](https://arxiv.org/abs/1905.10437) (Oreshkin et al., 2020) and N-HiTS, built around one practical observation: **most of N-BEATS's parameters are not doing any work.**

The published configurations spend almost all of their budget on trunk width, the hidden size of each block's fully connected layers. Sweeping that width on M4 from 2048 down to 128 left accuracy statistically unchanged across a 49-fold range in parameter count, and the result held out of sample on Tourism. A model with a narrow trunk and a frozen orthonormal basis at full forecast rank therefore lands within noise of the paper baselines at roughly a tenth to a fiftieth of their size.

PE-NBEATS packages that as a one-call preset, together with a pre-flight tool that reads your training windows and picks the settings that do matter. The full paper-faithful architecture and a large catalogue of research blocks (polynomial, Fourier, DCT, wavelet, autoencoder) remain available and freely composable.

```bash
pip install pe-nbeats
```

## Quickstart

```python
import lightning.pytorch as pl
import numpy as np
import torch

from pe_nbeats import NBeatsNet, get_best_accelerator, recommend
from pe_nbeats.data import MilkDataset
from pe_nbeats.loaders import TimeSeriesDataModule

milk = MilkDataset()                       # bundled monthly series, H = 6
series = milk.train_data["milk"].to_numpy(dtype=np.float32)
H, L = milk.forecast_length, 4 * milk.forecast_length

# Measure the target before training. No GPU, no fitting.
windows = np.lib.stride_tricks.sliding_window_view(series, H)
rec = recommend(windows)
print(rec.report())

model = NBeatsNet.from_recommendation(rec, backcast_length=L)
dm = TimeSeriesDataModule(series, batch_size=64, backcast_length=L, forecast_length=H)
trainer = pl.Trainer(accelerator=get_best_accelerator(), devices=1, max_epochs=100)
trainer.fit(model, datamodule=dm)

model.eval()
with torch.no_grad():
    _, forecast = model(torch.from_numpy(series[-L:]).unsqueeze(0))
```

`recommend()` returns a `Recommendation` whose `report()` prints the measurements behind each choice, each tagged with how strongly the evidence supports it.

## The slim preset

`NBeatsNet.slim(backcast_length, forecast_length, ...)` builds the right-sized model directly:

| Setting | Slim default | Paper default | Why |
|---|---|---|---|
| Trunk width | 128 | 512 (Generic), 2048 (Seasonality), 256 (Trend) | Accuracy-neutral from 128 to 2048 on M4 (Kruskal-Wallis p ≈ 0.78) and confirmed on Tourism. |
| Stacks | 10 | 30 | 30 stacks bought nothing on any M4 period and destabilised the Generic family on Quarterly and Daily. |
| Forecast head | frozen DCT basis at full rank | learned linear (Generic) or fixed Fourier/polynomial | At full rank the learned projection absorbs any rotation, so the choice of basis cannot limit accuracy. DCT is defined at every horizon. |
| Trend branch | `trend_branch=` (off unless needed) | Trend stack in N-BEATS-I | Rescues targets whose energy sits almost entirely in the window mean; costs 0.3 to 0.6 sMAPE elsewhere. |
| Rectified forecast | `rectify=` (off unless targets are non-negative) | off | Removes the collapse mode on non-negative series. Never use it on signed targets. |

`trend_branch=True` alternates `Trend` and basis blocks, the analogue of N-BEATS-I+G. For a Milk-sized problem the result has about 0.54M parameters against 19.5M for N-BEATS-I+G. Every other `NBeatsNet` keyword (`loss`, `learning_rate`, `lr_scheduler_config`, …) passes through.

```python
model = NBeatsNet.slim(backcast_length=30, forecast_length=6, trend_branch=True, rectify=True)
```

**Where the narrow trunk does not hold.** On Traffic-96, a long-horizon target that does not compress, widening the trunk from 128 to 512 improved sMAPE by about 1.0. There is no validated way yet to predict this from the data alone, so when accuracy matters on a new dataset, try `width=128` and `width=512` on a held-out split. That is two runs, not a sweep.

## Requirements

- **Python** ≥ 3.12, < 3.15
- **PyTorch** ≥ 2.1.0
- **Lightning** ≥ 2.1.0

Supports CUDA, Apple MPS, Intel XPU, and CPU. `pe_nbeats.get_best_accelerator()` returns the highest-priority device available.

Optional extras: `pe-nbeats[viz]` (matplotlib, seaborn), `pe-nbeats[experiments]` (wandb, tensorboard, scipy, statsmodels, pyyaml, tqdm, needed by `experiments/`), `pe-nbeats[notebooks]`, `pe-nbeats[all]`.

### From source

```bash
git clone https://github.com/realdanielbyrne/pe-nbeats.git
cd pe-nbeats
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,experiments]"
```

### CUDA

PyPI hosts a CPU-only `torch` wheel. To use a CUDA build, install the package first, then upgrade `torch` from PyTorch's CUDA index:

```bash
pip install -r requirements-cuda.txt                       # CUDA 12.8 (recommended)
# or pick a specific build:
pip install torch --index-url https://download.pytorch.org/whl/cu128 --upgrade
pip install torch --index-url https://download.pytorch.org/whl/cu121 --upgrade
```

Match your driver at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

### Multi-GPU

The last block's backcast head never reaches the loss, which is inherent to the doubly residual design. Lightning's default DDP strategy rejects unused parameters, so either pass `devices=1` or use `strategy="ddp_find_unused_parameters_true"`.

### Migrating from `lightningnbeats`

This package was previously published as `lightningnbeats`. `import lightningnbeats` still works through a compatibility alias that emits a `DeprecationWarning`; replace it with `import pe_nbeats`. Cached Traffic and Weather downloads under `~/.cache/lightningnbeats/` are reused.

## Paper-faithful N-BEATS

`stack_types` takes any list of registered block names, so the original architectures are one line each:

```python
NBeatsNet(L, H, stack_types=["Generic"] * 30)                                   # N-BEATS-G
NBeatsNet(L, H, stack_types=["Trend", "Seasonality"], n_blocks_per_stack=3,
          share_weights=True)                                                    # N-BEATS-I
NBeatsNet(L, H, stack_types=["Trend", "Seasonality"] + ["Generic"] * 8)          # N-BEATS-I+G
```

Constructor defaults (`g_width=512`, `s_width=2048`, `t_width=256`) match the paper, so existing configurations reproduce unchanged. `NHiTSNet` accepts the same blocks and adds multi-rate input pooling (`n_pools_kernel_size`) and hierarchical interpolation (`n_freq_downsample`).

## Pre-flight check

`pe_nbeats.recommend(windows, forecast_length=None)` computes two quantities from horizon-length windows of the training split:

- **Effective rank**, roughly how many basis coefficients carry the signal. Below about 4 a plain linear head may fail to train, and the recommendation turns the trend branch on.
- **Window-mean share**, the fraction of target energy in the window mean. Above one half it also turns the trend branch on.

It turns the rectified head on only when every window is non-negative, and always recommends DCT at full forecast rank with a 128-wide trunk. `pe_nbeats.analyze(windows)` returns the raw statistics without a recommendation.

## Research blocks

Every entry in `pe_nbeats.constants.BLOCKS` is a valid `stack_types` string. They are kept for reproducibility and experimentation. Read the caveats before relying on any of them for accuracy.

### Backbones

| Backbone | Topology | Notes |
|---|---|---|
| `RootBlock` | input → U → U → U → U | Paper-faithful 4-layer trunk. |
| `AERootBlock` | input → U/2 → d → U/2 → U | Hourglass trunk; `latent_dim` sets `d`. |
| `AERootBlockLG` | as AE | Adds a learned `sigmoid(γ) ⊙ z` gate on the latent. |
| `AERootBlockVAE` | μ/log σ² heads | Stochastic latent; KL term scaled by `kl_weight`. **Experimental.** |

### Families

| Family | RootBlock | AERootBlock | AERootBlockLG | AERootBlockVAE |
| --- | --- | --- | --- | --- |
| Generic (paper) | `Generic` | `GenericAE` | `GenericAELG` | `GenericVAE` |
| Bottleneck Generic | `BottleneckGeneric` | `BottleneckGenericAE` | `BottleneckGenericAELG` | `BottleneckGenericVAE` |
| Backcast-only AE | `GenericAEBackcast` | `GenericAEBackcastAE` | `GenericAEBackcastAELG` | `GenericAEBackcastVAE` |
| Branch-specific AE | `AutoEncoder` | `AutoEncoderAE` | `AutoEncoderAELG` | `AutoEncoderVAE` |
| Trend (polynomial) | `Trend` | `TrendAE` | `TrendAELG` | `TrendVAE` |
| Seasonality (Fourier) | `Seasonality` | `SeasonalityAE` | `SeasonalityAELG` | `SeasonalityVAE` |
| Frozen basis (DCT / wavelet / random) | `WaveletV3` | `WaveletV3AE` | `WaveletV3AELG` | `WaveletV3VAE` |
| Trend + frozen basis | `TrendWavelet` | `TrendWaveletAE` | `TrendWaveletAELG` | — |
| Trend + frozen basis + Generic | `TrendWaveletGeneric` | `TrendWaveletGenericAE` | `TrendWaveletGenericAELG` | `TrendWaveletGenericVAE` |

`WaveletV3` and the `TrendWavelet*` families take `wavelet_type`, which accepts:

- `"dct"`: orthonormal DCT-II, ordered low to high frequency. Recommended.
- `"random_ortho"` or `"random_ortho:<seed>"`: a seeded random orthonormal basis.
- `"ordered:<family>"`, e.g. `"ordered:haar"`: a periodized DWT basis that keeps the dyadic band order, so `basis_offset` selects a frequency band.
- `"<family>"`, e.g. `"haar"` or `"db3"`: the legacy SVD-orthogonalised DWT basis, kept so earlier results reproduce.

Family-named subclasses (`HaarWaveletV3`, `DB3WaveletV3`, …) remain as compatibility wrappers.

### What the evidence says about these blocks

- **Basis identity is inert at full forecast rank.** Wavelet families were a well-powered null against each other, and a random orthonormal basis performs as well as any wavelet. The DCT default is an engineering choice, not an accuracy claim.
- **Wavelets need room to decompose.** When `pywt.dwt_max_level(H, filter_length) == 0` the basis has no multiscale structure and is effectively the identity. `coif2` and `sym10` hit this at every M4 horizon except Hourly, and `db3` at H ≤ 8. The library now warns when it happens.
- **The legacy SVD path scrambles band order.** With `"<family>"` bases, `basis_offset` does not pick a frequency band, and the library warns when one is set. Use `"ordered:<family>"`.
- **Autoencoder trunks and the learned gate are not the source of the savings.** At matched parameter budgets they do not beat a plain narrow trunk, and the gate was null across twelve comparisons. Use them when the latent dimension itself is the constraint.
- **VAE blocks collapse on M4** (5 to 10 times worse sMAPE). Treat them as experimental.
- **`active_g="forecast"`** rectifies the forecast output. It is safe only when each block's output should be non-negative. Never combine it with signed, cancelling components such as DWT sub-band inputs. `active_g=True`, which also rectifies the backcast, costs accuracy.
- **`sum_losses=True`** adds `0.25 · L(backcast, 0)` to the training loss. `val_loss` and `test_loss` always report the forecast loss alone; the combined objective is logged as `val_total_loss`.
- **`skip_distance` / `skip_alpha`** reinject the input every few stacks. They have not helped on M4 and are off by default.

### Width and dimension parameters

| Parameter | Default | Applies to |
|---|---|---|
| `g_width` | 512 | Generic, BottleneckGeneric, GenericAEBackcast, frozen-basis blocks |
| `s_width` | 2048 | Seasonality family |
| `t_width` | 256 | Trend and TrendWavelet* families |
| `ae_width` | 512 | AutoEncoder family |
| `latent_dim` | 5 | AE / LG / VAE backbones |
| `thetas_dim` | 5 | Generic-type theta projections |
| `trend_thetas_dim` | 3 | Polynomial order for Trend and TrendWavelet* |
| `basis_dim` | 32 | Backcast basis rank (and forecast rank unless overridden) |
| `forecast_basis_dim` | `None` | Forecast basis rank; set to `forecast_length` for full rank |
| `generic_dim` | 5 | Learned branch rank in TrendWaveletGeneric* |

## Losses

Besides the standard `torch.nn` losses, `loss=` accepts `"SMAPELoss"`, `"MAPELoss"`, `"MASELoss"` and `"NormalizedDeviationLoss"`.

## Data

### Bundled and downloadable datasets

```python
from pe_nbeats.data import M4Dataset, MilkDataset, TourismDataset, TrafficDataset, WeatherDataset

MilkDataset()                    # bundled
TourismDataset("Quarterly")      # bundled
M4Dataset("Yearly")              # downloaded from the official M4 repository on first use
TrafficDataset(horizon=96)       # downloaded from Hugging Face on first use
WeatherDataset(horizon=96)       # downloaded from Hugging Face on first use
```

Downloads are cached under `~/.cache/pe_nbeats/`. Every dataset exposes `train_data` and `test_data` as DataFrames with one column per series, plus `forecast_length` and `frequency`. `M4Dataset` also provides `compute_owa(smape, mase)`.

### DataModules

- `TimeSeriesDataModule(data, batch_size, backcast_length, forecast_length)`: one univariate series, sliding windows with a random 80/20 split.
- `ColumnarCollectionTimeSeriesDataModule(dataframe, backcast_length, forecast_length, ...)`: many series stored as columns (M4, Tourism, Traffic, Weather as loaded above). It supports normalisation (`normalization_style="global"` or `"window"`), a dedicated `val_data`, and paper-style sampling (`sampling_style="nbeats_paper"`).
- `ColumnarCollectionTimeSeriesTestDataModule(train_data, test_data, ...)`: joins the tail of each training series to its test horizon for evaluation.
- `RowCollectionTimeSeriesDataModule` / `RowCollectionTimeSeriesTestModule`: the same for data with one row per series.

```python
from pe_nbeats import NBeatsNet, get_best_accelerator
from pe_nbeats.data import M4Dataset
from pe_nbeats.loaders import (
    ColumnarCollectionTimeSeriesDataModule,
    ColumnarCollectionTimeSeriesTestDataModule,
)
import lightning.pytorch as pl

m4 = M4Dataset("Yearly")
H = m4.forecast_length
L = 5 * H

model = NBeatsNet.slim(L, H, trend_branch=True, rectify=True)
dm = ColumnarCollectionTimeSeriesDataModule(m4.train_data, L, H, batch_size=1024)
test_dm = ColumnarCollectionTimeSeriesTestDataModule(m4.train_data, m4.test_data, L, H, batch_size=1024)

trainer = pl.Trainer(accelerator=get_best_accelerator(), devices=1, max_epochs=50)
trainer.fit(model, datamodule=dm)
trainer.test(model, datamodule=test_dm)
```

## Experiments

The repository's benchmark sweeps are driven by YAML through `experiments/run_from_yaml.py`. Install with `pip install -e ".[experiments]"`.

```bash
python experiments/run_from_yaml.py experiments/configs/nbeats_g.yaml --dry-run
python experiments/run_from_yaml.py experiments/configs/tourism_preset_validation.yaml
```

The schema reference is in [`experiments/configs/schema.md`](experiments/configs/schema.md), and analysis reports are in [`experiments/analysis/analysis_reports/`](experiments/analysis/analysis_reports/).

## Development

```bash
pytest tests/
```

Conventions are in [TESTING.md](TESTING.md). To add a block, subclass `RootBlock`, `AERootBlock`, `AERootBlockLG` or `AERootBlockVAE` in `src/pe_nbeats/blocks/blocks.py` and return `(backcast, forecast)` from `forward()`. Then register the name in `BLOCKS` in `src/pe_nbeats/constants.py`. The parametrised shape tests pick it up automatically.

## License

MIT
