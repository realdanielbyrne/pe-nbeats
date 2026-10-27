# YAML Configuration Schema for `run_from_yaml.py`

Complete reference for all supported keys in experiment YAML files.  
Run experiments with:

```bash
python experiments/run_from_yaml.py experiments/configs/my_config.yaml
python experiments/run_from_yaml.py experiments/configs/my_config.yaml --dry-run
```

---

## Top-Level Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `experiment_name` | str | `"yaml_experiment"` | Label stored in the `experiment` column of every CSV row. Also used as the default CSV filename prefix. |
| `config_name` | str | `experiment_name` | Config name for single-config mode (see below). |
| `category` | str | `"custom"` | Default category label for all configs. |
| `description` | str | — | Human-readable description (not written to CSV). |
| `dataset` | str \| list[str] | `"m4"` | Dataset(s) to run. Choices: `m4`, `tourism`, `traffic`, `weather`, `milk`. Use a list to run multiple datasets sequentially. |
| `periods` | list[str] \| null | all for dataset | Period names to run. `null` or absent = all periods for the dataset. |

---

## Stack Architecture

Choose **one** of the following patterns:

### `stacks` / `stack_types` (top-level or per config)

Accepts any of the formats described in the **Stack Specification** section below.

### `configs` (multi-config mode)

A list of config objects, each defining its own stack and per-config overrides:

```yaml
configs:
  - name: NBEATS-G
    category: paper_baseline
    stacks:
      type: homogeneous
      block: Generic
      n: 30
  - name: NBEATS-I+G
    stacks:
      type: prefix_body
      prefix: [Trend, Seasonality]
      body: Generic
      total: 30
```

When `configs` is absent, a single config is built from the top-level `stacks` key.

---

## Stack Specification Formats

Used in both `stacks` (top-level), `configs[i].stacks`, and `configs[i].stack_types`.

### Explicit list

```yaml
stacks:
  - Generic
  - Generic
  - Generic
  # ... (30 total)
```

### String shorthand (homogeneous)

```yaml
stacks: "Generic:30"   # or "Generic*30" or "Generic x30"
```

### `homogeneous`

```yaml
stacks:
  type: homogeneous
  block: Generic
  n: 30             # or n_stacks: 30
```

### `prefix_body`

Useful for NBEATS-I+G style architectures.

```yaml
stacks:
  type: prefix_body
  prefix:
    - Trend
    - Seasonality
  body: Generic     # or remainder: Generic
  total: 30         # n_body = total - len(prefix) = 28
```

### `alternating`

Repeats a sequence of block types:

```yaml
stacks:
  type: alternating
  blocks: [TrendAE, GenericAE]
  repeats: 5         # → [TrendAE, GenericAE] * 5 = 10 stacks
```

### `concat`

Concatenate multiple sub-specs (any format):

```yaml
stacks:
  type: concat
  parts:
    - {type: explicit, blocks: [Trend, Seasonality]}
    - {type: homogeneous, block: GenericAE, n: 8}
```

### `builtin`

Reference a pre-defined config from `UNIFIED_CONFIGS` in `run_unified_benchmark.py`:

```yaml
stacks:
  builtin: NBEATS-I+G
```

Valid builtin names: `NBEATS-G`, `NBEATS-I`, `NBEATS-I+G`, `GenericAE`,
`BottleneckGenericAE`, `AutoEncoder`, `BottleneckGeneric`, `Coif2WaveletV3`,
`DB4WaveletV3`, `Trend+Coif2WaveletV3`, `Trend+DB3WaveletV3`,
`Trend+HaarWaveletV3`, `Generic+DB3WaveletV3`, `NBEATS-I+GenericAE`,
`NBEATS-I+BottleneckGeneric`, `NBEATS-I-AE`.

---

## `training` — Training Hyperparameters

```yaml
training:
  active_g: false          # false | "forecast" | "backcast"
  sum_losses: false        # true = add 0.25× backcast reconstruction loss
  activation: ReLU         # any name from constants.ACTIVATIONS
  loss: SMAPELoss          # any name from constants.LOSSES
  optimizer: Adam          # Adam | SGD | RMSprop | Adagrad | Adadelta | AdamW
  learning_rate: 0.001
  weight_decay: null       # L2 weight decay; null = optimizer's own default (Adam 0.0, paper-faithful). Explicit values (incl. 0.0) are forwarded.
  max_epochs: 100
  patience: 10             # EarlyStopping patience
  n_blocks_per_stack: 1
  share_weights: true
  batch_size: null         # null = auto per dataset/period from BATCH_SIZES table
  forecast_multiplier: null  # null = auto per dataset (M4→5, Tourism/Traffic/Weather→2)
  skip_distance: 0         # Re-inject original input every N stacks (0 = disabled)
  skip_alpha: 0.0          # Mixing weight for skip injection (float or "learnable")
```

Can be overridden at the config level or the pass level.

### Skip Connections (ResNet-style signal re-injection)

When `skip_distance > 0`, the forward pass periodically adds `alpha * original_input`
to the residual stream after every `skip_distance` stacks. This combats signal and
gradient decay in deep stack architectures (e.g., 30-stack Generic).

- `skip_distance: 0` — Disabled (default, backward compatible).
- `skip_distance: 5` — Re-inject after stacks 5, 10, 15, 20, 25 (in a 30-stack model).
- `skip_alpha: 0.1` — Fixed mixing weight.
- `skip_alpha: "learnable"` — Creates an `nn.Parameter` initialized to 0.01;
  the model learns the optimal injection strength during training.

The last stack is always excluded from re-injection.

```yaml
# Example: fixed alpha
training:
  skip_distance: 5
  skip_alpha: 0.1

# Example: learnable alpha
training:
  skip_distance: 5
  skip_alpha: learnable
```

---

## `protocol` — Dataset / Evaluation Protocol

```yaml
protocol:
  normalize: false          # true = z-score each series using train-split stats
  train_ratio: null         # Traffic/Weather only; null = dataset default (0.8)
  val_ratio: null           # null = last backcast+forecast rows form validation
  include_target: false     # Traffic/Weather only; true keeps the OT column
  loss: null                # fallback when training.loss is not explicitly set
  forecast_multiplier: null # fallback when training.forecast_multiplier is not explicitly set
  batch_size: null          # fallback when training.batch_size is not explicitly set
  datamodule: columnar      # "columnar" (default) or "univariate" (80/20 random split)
  sampling_style: sliding   # "sliding" (default) | "nhits_paper" | "nbeats_paper"
  steps_per_epoch: null     # required when sampling_style is 'nhits_paper' or 'nbeats_paper'
  sampling_weights: uniform # "uniform" (default) or "by_series" (equal-probability per series)
  lh_multiplier: null       # required when sampling_style='nbeats_paper'; Lh = lh_multiplier * H.
                            # May be a scalar (same for all periods) or a period-keyed dict
                            # (e.g. {Yearly: 1.5, Quarterly: 1.5, Monthly: 10, ...}).
                            # A 'default' key in the dict is used as fallback for unlisted periods.
  acknowledge_epoch_semantics: false  # opt-in when using max_epochs under nhits_paper/nbeats_paper
```

Notes:

- The YAML launcher reads `protocol` from the top level and forwards the relevant values into dataset loading and datamodule construction.
- `normalize` and `val_ratio` are passed into `ColumnarCollectionTimeSeriesDataModule`.
- `train_ratio` and `include_target` are applied when loading the Traffic or Weather dataset.
- `loss`, `forecast_multiplier`, and `batch_size` act as **fallbacks** only when the same fields are not explicitly set under `training`.
- An explicit `training.batch_size` takes precedence over the tuned `BATCH_SIZES` table. When `training.batch_size` is not set, the tuned table provides the default.
- `datamodule: univariate` selects `TimeSeriesDataModule` with a flat numpy array and 80/20 random train/val split (via `torch.utils.data.random_split`). This matches the behavior of standalone convergence scripts (e.g. `run_milk_convergence_10stack.py`). Only meaningful for single-series datasets like Milk.
- `sampling_style` controls training-batch sampling. Three modes are supported:
  - `'sliding'` (default): enumerates every valid `(column, start_idx)` window and shuffles one full pass per epoch.
  - `'nhits_paper'`: reproduces the NHiTS iteration-based protocol (Challu et al. 2023): each Lightning epoch draws exactly `steps_per_epoch * batch_size` windows **with replacement** from all valid windows. No insample length constraint.
  - `'nbeats_paper'`: same replacement-sampling protocol as `nhits_paper`, plus the **Lh insample constraint** from Oreshkin et al. (2020): each series' valid windows are restricted to the last `lh_multiplier * H` timesteps (Lh = 7H recommended for M4, matching the paper's maximum ensemble lookback). Requires `lh_multiplier` to be set.
  **Training-only**: validation and test dataloaders always use dense sliding windows with no replacement.
- `steps_per_epoch` is **required** when `sampling_style` is `'nhits_paper'` or `'nbeats_paper'` (hard error if missing) and must be a positive int. Ignored for `'sliding'`.
- `lh_multiplier` is **required** when `sampling_style='nbeats_paper'`. Sets Lh = lh_multiplier × H. Use `7` to match the N-BEATS paper's maximum ensemble lookback (Lh = 7H). Ignored for other sampling styles.
- `sampling_weights='uniform'` samples uniformly over the enumerated window list — biases toward long series proportionally to their window count. `'by_series'` weights each index by `1 / n_windows_in_its_column`, giving every series an equal per-step draw probability. Ignored for `'sliding'`.
- **Epoch semantics under `'nhits_paper'` / `'nbeats_paper'`**: one "epoch" equals `steps_per_epoch` gradient updates rather than one full pass over the data, so `training.max_epochs` has a different meaning than under `'sliding'`. If `training.max_steps` is unset, you must set `acknowledge_epoch_semantics: true` to opt in (a `UserWarning` then reports the total `max_epochs * steps_per_epoch` step budget). Preferred path: set `training.max_steps` for an unambiguous total training budget.

---

## `block_params` — Block-Specific Parameters

```yaml
block_params:
  thetas_dim: 5            # Bottleneck projection dimension
  latent_dim: 4            # AE backbone latent space size
  ae_intermediate_width: null  # AE backbone taper (intermediate) layer width (null = units // 2)
  basis_dim: 128           # WaveletV3 basis dimension
  forecast_basis_dim: null # WaveletV3 asymmetric forecast basis (null = same as basis_dim)
  trend_thetas_dim: null   # Trend polynomial degree override (null = use thetas_dim)
  generic_dim: 5           # Learned generic branch rank for TrendWaveletGeneric* blocks
  wavelet_type: null       # Wavelet family for wavelet-capable blocks (null = 'db3')
  backcast_wavelet_type: null  # Override wavelet family for backcast path only (null = use wavelet_type)
  forecast_wavelet_type: null  # Override wavelet family for forecast path only (null = use wavelet_type)
```

---

## `lr_scheduler` — Learning Rate Scheduler

Cosine annealing with warmup (default):

```yaml
lr_scheduler:
  type: cosine             # optional; "cosine" is the default
  warmup_epochs: 15        # Hold LR constant for this many epochs
  T_max: null              # CosineAnnealing period; null = max_epochs - warmup_epochs
  eta_min: 0.000001        # Minimum LR after cosine decay
```

ReduceLROnPlateau (recommended when using early stopping):

```yaml
lr_scheduler:
  type: plateau
  factor: 0.5              # Multiply LR by this on trigger (default 0.5)
  patience: 10             # Checks (in interval units) with no val_loss improvement before reducing (default 10)
  min_lr: 0.00001          # LR floor (default 1e-5); also accepts eta_min as alias
  mode: min                # "min" for val_loss; "max" for accuracy-style metrics
  cooldown: 0              # Checks to wait after a reduction before resuming (default 0)
  monitor: val_loss        # Metric to watch (default "val_loss")
  interval: epoch          # "epoch" (default) or "step" — Lightning step cadence
  frequency: 1             # Number of intervals between scheduler.step calls (default 1)
                           # Under nbeats_paper sampling, set interval=step and
                           # frequency=val_check_interval to step the scheduler
                           # exactly once per validation check; this matches the
                           # signal that drives EarlyStopping and is essential
                           # when runs early-stop in only a few Lightning epochs.
```

StepLR (fixed-interval decay):

```yaml
lr_scheduler:
  type: step
  step_size: 50            # Required: decay every N epochs
  gamma: 0.5               # Decay factor (default 0.5)
```

MultiStepLR — paper-faithful policy (Oreshkin et al. 2020: halve LR every 10% of training):

```yaml
lr_scheduler:
  type: multistep
  n_milestones: 10         # Evenly-spaced drops; milestones computed as
                           # [round(max_epochs * i / n_milestones) for i in 1..n_milestones]
  gamma: 0.5               # LR multiplier at each milestone (default 0.5)

# OR provide explicit milestone epochs:
lr_scheduler:
  type: multistep
  milestones: [15, 30, 45, 60, 75, 90, 105, 120, 135, 150]
  gamma: 0.5
```

> **Early stopping note:** When using `multistep`, set `training.patience` to at least
> `max_epochs / n_milestones + 5` so early stopping cannot fire between scheduled drops.
> For 150 epochs / 10 milestones (interval = 15 epochs), use `patience: 20`.

Set to `null` or omit to disable the scheduler entirely (constant LR).

---

## `passes` — Two-Pass Design

Define multiple training passes (e.g., baseline + active_g) over the same config set.
Each pass becomes a separate `experiment` label in the CSV.

```yaml
passes:
  - name: baseline
    training:
      active_g: false
  - name: activeG_fcast
    training:
      active_g: forecast
```

When `passes` is absent, a single pass is created whose name is `experiment_name`.

---

## `runs` — Run Configuration

```yaml
runs:
  n_runs: 10           # Number of independent runs per (config × pass × period)
  base_seed: 42        # Base seed for sequential mode (run i → seed = base_seed + i)
  seed_mode: sequential  # sequential | random | fixed
  seed: null           # Fixed seed used when seed_mode: fixed (null = 42)
  seeds: null          # Explicit per-run seed list; wraps if shorter than n_runs
```

### Seed modes

| `seed_mode` | Behaviour |
|---|---|
| `sequential` | `seed = base_seed + run_idx` — default, fully reproducible |
| `random` | A fresh `numpy.random.randint` seed per run — stochastic exploration |
| `fixed` | Every run uses the same seed (`runs.seed`, default 42) |

An explicit `seeds` list always takes precedence over `seed_mode`.  
If the list is shorter than `n_runs` it is cycled:

```yaml
runs:
  n_runs: 5
  seeds: [42, 137, 999]   # run 0→42, run 1→137, run 2→999, run 3→42, run 4→137
```

---

## `output` — Output Paths

```yaml
output:
  results_dir: experiments/results          # Base results directory
  csv_filename: null                        # null → "{experiment_name}_results.csv"
  save_predictions: true                    # Save NPZ files per run
  predictions_subdir: predictions           # Subdirectory under results_dir/{dataset}/
```

CSV is written to `{results_dir}/{dataset}/{csv_filename}`.  
Predictions are written to `{results_dir}/{dataset}/{predictions_subdir}/`.

---

## `logging` — Logging Configuration

```yaml
logging:
  wandb:
    enabled: false
    project: nbeats-lightning
    group: null              # null = auto ("unified/{period}")
  tensorboard: false         # TensorBoard disabled by default (saves disk space)
  csv_log: true
```

---

## `hardware` — Hardware Settings

```yaml
hardware:
  accelerator: auto          # auto | cuda | mps | cpu
  num_workers: 0             # DataLoader worker processes
  gpu_id: null               # null = auto; int = pin to specific GPU index
  worker_id: ""             # Optional label recorded in atomic claim metadata
```

YAML / unified runs also use atomic claim files under
`experiments/results/.claims/`. Claim filenames keep the NHiTS-style shape
`<claim_config>__<dataset>__<horizon>__run<idx>.claim`; for two-pass and search
runs, the pass / experiment tag is prefixed into `claim_config` so `baseline`
and `activeG_fcast` remain independently claimable.

---

## `extra_csv_columns` — Extended CSV Schema

Add study-specific columns to the CSV. These are filled from `extra_fields` in each
config (or from `extra_fields` at the top level for all configs):

```yaml
extra_csv_columns:
  - search_round
  - block_type
  - latent_dim_cfg
  - thetas_dim_cfg
```

Per-config population:

```yaml
configs:
  - name: GenericAE_ld4_ag0
    stacks: {type: homogeneous, block: GenericAE, n: 10}
    extra_fields:
      block_type: GenericAE
      latent_dim_cfg: 4
      thetas_dim_cfg: 5
```

The special key `search_round` is auto-populated by the successive halving engine.

---

## `search` — Successive Halving

```yaml
search:
  enabled: true
  rounds:
    - max_epochs: 8      # Epochs for round 1
      n_runs: 3          # Runs per config in round 1
      keep_fraction: 0.50   # Keep top 50% → promotes to round 2
    - max_epochs: 15
      n_runs: 3
      keep_fraction: 0.50   # Keep top 50% → promotes to round 3
    - max_epochs: 50
      n_runs: 3
      top_k: 5           # top_k overrides keep_fraction; use for final round
  use_meta_forecaster: false  # Reserved; meta-forecaster integration planned
```

Ranking criterion: median `best_val_loss` across runs.  
Configs with >50 % divergence rate are ranked last.

---

## `analysis` — Post-Experiment Analysis

```yaml
analysis:
  enabled: false     # Print ranking table after all runs complete
  ranking: true      # Include mean-OWA ranking table in output
```

Use `--analyze-only` CLI flag to run analysis on existing results without training.

---

## Config-Level Overrides

Each entry in `configs` can override **any** top-level key:

```yaml
configs:
  - name: MyConfig
    category: custom              # overrides top-level category
    stacks:
      type: homogeneous
      block: GenericAE
      n: 10
    n_blocks_per_stack: 2         # overrides training.n_blocks_per_stack
    share_weights: false          # overrides training.share_weights
    training:
      max_epochs: 200             # overrides top-level training.max_epochs
      active_g: forecast
    block_params:
      latent_dim: 8               # overrides top-level block_params.latent_dim
    lr_scheduler:
      warmup_epochs: 20           # overrides top-level lr_scheduler
    extra_fields:
      my_column: my_value         # populates custom CSV column
```

---

## CLI Overrides

All CLI flags override the YAML values:

| Flag | Effect |
|------|--------|
| `--dataset` | Override `dataset` |
| `--periods P1 P2` | Override `periods` |
| `--n-runs N` | Override `runs.n_runs` |
| `--max-epochs N` | Override `training.max_epochs` |
| `--batch-size N` | Override `training.batch_size` |
| `--accelerator` | Override `hardware.accelerator` |
| `--num-workers N` | Override `hardware.num_workers` |
| `--worker-id ID` | Override `hardware.worker_id` for claim metadata |
| `--wandb` | Enable W&B logging |
| `--wandb-project NAME` | Set W&B project |
| `--no-save-predictions` | Disable NPZ saving |
| `--results-dir PATH` | Override `output.results_dir` |
| `--dry-run` | Print plan, skip training |
| `--analyze-only` | Skip training, print analysis |

---

## Replicating Existing Scripts

### `run_unified_benchmark.py`

```yaml
experiment_name: unified_benchmark
dataset: m4
stacks:
  builtin: NBEATS-G    # or any other builtin name
passes:
  - name: baseline
    training: {active_g: false}
  - name: activeG_fcast
    training: {active_g: forecast}
runs:
  n_runs: 10
```

Or run all 16 built-in configs at once:

```yaml
configs:
  - {builtin: NBEATS-G}
  - {builtin: NBEATS-I}
  - {builtin: NBEATS-I+G}
  - {builtin: GenericAE}
  # ...
```

### `run_generic_ae_study.py`

See [`genericae_pure.yaml`](./genericae_pure.yaml) — a complete replication
of the 40-config successive halving search.

### `run_trendae_study.py`

See [`trendae_ae_alternating.yaml`](./trendae_ae_alternating.yaml) —
key alternating TrendAE+AE patterns.

### Single Custom Experiment

```yaml
experiment_name: my_wavelet_test
dataset: m4
periods: [Yearly]
stacks:
  type: alternating
  blocks: [Trend, Coif2WaveletV3]
  repeats: 15
training:
  active_g: false
  max_epochs: 100
block_params:
  basis_dim: 128
runs:
  n_runs: 5
```

---

## Valid Block Type Names

All block types supported in `stack_types` (from `constants.BLOCKS`):

**Standard backbone:** `Generic`, `BottleneckGeneric`, `Trend`, `Seasonality`,
`AutoEncoder`, `VAE`, `GenericAEBackcast`

**AE backbone:** `GenericAE`, `BottleneckGenericAE`, `TrendAE`, `SeasonalityAE`,
`AutoEncoderAE`, `GenericAEBackcastAE`

**Learned-Gate AE:** `GenericAELG`, `BottleneckGenericAELG`, `TrendAELG`,
`SeasonalityAELG`, `AutoEncoderAELG`, `GenericAEBackcastAELG`

**Variational AE:** `GenericVAE`, `BottleneckGenericVAE`, `TrendVAE`,
`SeasonalityVAE`, `AutoEncoderVAE`, `GenericAEBackcastVAE`

**Wavelet V3 (DWT):** `HaarWaveletV3`, `DB2WaveletV3`, `DB3WaveletV3`,
`DB4WaveletV3`, `DB10WaveletV3`, `DB20WaveletV3`, `Coif1WaveletV3`,
`Coif2WaveletV3`, `Coif3WaveletV3`, `Coif10WaveletV3`, `Symlet2WaveletV3`,
`Symlet3WaveletV3`, `Symlet10WaveletV3`, `Symlet20WaveletV3`

**Trend+Wavelet AE (combined polynomial + DWT):** `TrendWaveletAE`, `TrendWaveletAELG`

**Wavelet V3AE (DWT + AE bottleneck):** `HaarWaveletV3AE`, `DB2WaveletV3AE`,
`DB3WaveletV3AE`, `DB4WaveletV3AE`, `DB10WaveletV3AE`, `DB20WaveletV3AE`,
`Coif1WaveletV3AE`, `Coif2WaveletV3AE`, `Coif3WaveletV3AE`, `Coif10WaveletV3AE`,
`Symlet2WaveletV3AE`, `Symlet3WaveletV3AE`, `Symlet10WaveletV3AE`,
`Symlet20WaveletV3AE`

**Wavelet V3AELG (DWT + Learned-Gate AE bottleneck):** `HaarWaveletV3AELG`,
`DB2WaveletV3AELG`, `DB3WaveletV3AELG`, `DB4WaveletV3AELG`, `DB10WaveletV3AELG`,
`DB20WaveletV3AELG`, `Coif1WaveletV3AELG`, `Coif2WaveletV3AELG`,
`Coif3WaveletV3AELG`, `Coif10WaveletV3AELG`, `Symlet2WaveletV3AELG`,
`Symlet3WaveletV3AELG`, `Symlet10WaveletV3AELG`, `Symlet20WaveletV3AELG`

---

## Compact Grid Study YAML (TrendWaveletAE / TrendWaveletAELG)

`experiments/run_trendwaveletae_study.py` uses a compact grid schema where
configs are generated at runtime from a search space specification.

### `block_type` — Scalar or List

The `architecture.block_type` field accepts either a single string or a list
of strings. When a list is provided, the grid is expanded across all block
types, producing distinct configs for each:

```yaml
# Scalar (single block type, backward compatible)
architecture:
  block_type: TrendWaveletAE
  repeats: 10

# List (multiple block types in one study)
architecture:
  block_type: [TrendWaveletAE, TrendWaveletAELG]
  repeats: 10
```

With a list, the total config count is `len(block_types) * len(wavelet_types)

- len(basis_labels) *len(trend_dims)* len(latent_dims)`. Each config's
canonical ID encodes the block type as the first field (e.g.,
`TrendWaveletAE|haar|eq_fcast|td3|ld8`), so AE and AELG configs have
distinct IDs and can coexist in the same CSV.

### v2 Study Example

The v2 configs (`trendwaveletae_v2_*.yaml`) use a refined search space
based on empirical findings from the v1 study:

```yaml
architecture:
  block_type: [TrendWaveletAE, TrendWaveletAELG]
  repeats: 10

search_space:
  wavelet_types: [haar, db3, db20, coif2, sym10]    # 5 (one per family)
  basis_labels: [eq_fcast, lt_fcast]                  # 2 (bcast eliminated)
  trend_dims: [3]                                     # 1 (hardcoded winner)
  latent_dims: [8, 12]                                # 2 (confirm + probe upward)

search:
  rounds:
    - {max_epochs: 15, n_runs: 3, keep_fraction: 0.50}
    - {max_epochs: 50, n_runs: 5, top_k: 10}
```

This produces 2 x 5 x 2 x 1 x 2 = **40 configs** with 2 rounds of
successive halving (~220 runs per dataset vs ~2,133 in v1).

---

## Compact Grid Study YAML (WaveletV3AE)

`experiments/run_wavelet_v3ae_study.py` uses a compact grid schema where
configs are generated at runtime (instead of listing 336 entries explicitly).

```yaml
study_name: wavelet_v3ae_study_m4
dataset: m4
period: Yearly

architecture:
  trend_block: TrendAE
  repeats: 5            # m4/tourism=5, traffic/weather=10

training:
  active_g: false
  sum_losses: false
  activation: ReLU
  n_blocks_per_stack: 1
  share_weights: true
  loss: SMAPELoss
  optimizer: Adam
  learning_rate: 0.001

lr_scheduler:
  warmup_epochs: 15
  eta_min: 0.000001

search_space:
  wavelets: [HaarWaveletV3AE, DB2WaveletV3AE, ... Symlet20WaveletV3AE]
  basis_labels: [eq_fcast, lt_fcast, eq_bcast, lt_bcast]
  trend_thetas_dims: [3, 5]
  latent_dims: [2, 5, 8]

search:
  rounds:
    - {max_epochs: 10, n_runs: 3, keep_fraction: 0.67}
    - {max_epochs: 15, n_runs: 3, keep_fraction: 0.67}
    - {max_epochs: 50, n_runs: 3, top_k: 10}

output:
  results_dir: experiments/results
  search_csv_filename: wavelet_v3ae_study_results.csv
  cross_csv_path: experiments/results/wavelet_v3ae_cross_dataset_results.csv
```

### Notes

- Basis labels are mapped at runtime per dataset/period:
  - `eq_fcast = forecast_length`
  - `lt_fcast = max(forecast_length//2, forecast_length-2)`
  - `eq_bcast = backcast_length`
  - `lt_bcast = backcast_length//2`
- Colliding basis values are **not** deduplicated; all labels remain in the
  grid so each dataset keeps `14 * 4 * 2 * 3 = 336` configs.
- Search CSV includes:
  `search_round`, `basis_dim`, `basis_offset`, `trend_thetas_dim_cfg`,
  `wavelet_family`, `bd_label`, `latent_dim_cfg`,
  `meta_predicted_best`, `meta_convergence_score`.

---

## Valid Dataset Periods

| Dataset | Valid Periods |
|---------|--------------|
| `m4` | `Yearly`, `Quarterly`, `Monthly`, `Weekly`, `Daily`, `Hourly` |
| `tourism` | `Tourism-Yearly`, `Tourism-Quarterly`, `Tourism-Monthly` |
| `traffic` | `Traffic-96`, `Traffic-192` |
| `weather` | `Weather-96`, `Weather-192` |
| `milk` | `Milk` |
