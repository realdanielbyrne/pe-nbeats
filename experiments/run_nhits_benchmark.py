"""
NHiTS-Protocol Benchmark for Novel Block Types

Aligns our pipeline with the NHiTS evaluation protocol, then benchmarks
our best novel block types through both NBeatsNet and NHiTSNet.

Protocol differences addressed:
  1. Dataset-dependent normalization: Weather=Z-score, Traffic=none (0-1 scaled)
  2. 70/10/20 train/val/test split (vs our 80/20)
  3. All columns including OT target (21 Weather cols, 862 Traffic cols)
  4. MSE loss (vs our SMAPE)
  5. L = 5*H lookback (matching NHiTS)
  6. 8 runs per config with seeds 1-8

Usage:
    python experiments/run_nhits_benchmark.py --dry-run
    python experiments/run_nhits_benchmark.py --dataset weather --horizons 96
    python experiments/run_nhits_benchmark.py --dataset traffic --horizons 96 192 336 720
    python experiments/run_nhits_benchmark.py --analyze
"""

import argparse
import ast
import csv
import gc
import json
import math
import os
import re
import shutil
import signal
import sys
import tempfile
import time
import warnings
from contextlib import contextmanager

# File locking — cross-platform
if sys.platform == "win32":
    import msvcrt

    @contextmanager
    def _exclusive_lock(file_obj):
        file_obj.seek(0)
        while True:
            try:
                msvcrt.locking(file_obj.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                time.sleep(0.05)
        try:
            yield
        finally:
            file_obj.seek(0)
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    @contextmanager
    def _exclusive_lock(file_obj):
        fcntl.flock(file_obj, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(file_obj, fcntl.LOCK_UN)


import numpy as np
import pandas as pd
import torch
import yaml
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# Allow running from project root or experiments/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pe_nbeats.models import NBeatsNet, NHiTSNet
from pe_nbeats.loaders import (
    ColumnarCollectionTimeSeriesDataModule,
    ColumnarCollectionTimeSeriesTestDataModule,
)
from pe_nbeats.data import TrafficDataset, WeatherDataset

torch.set_float32_matmul_precision("medium")


# ---------------------------------------------------------------------------
# Signal Handling
# ---------------------------------------------------------------------------

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    if _shutdown_requested:
        print("\n[SIGNAL] Force exit.")
        os._exit(1)
    _shutdown_requested = True
    print(
        "\n[SIGNAL] Shutdown requested. Finishing current run... "
        "(signal again to force)"
    )


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# NHiTS Protocol Constants
# ---------------------------------------------------------------------------

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
LOSS = "MSELoss"
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
FORECAST_MULTIPLIER = 5  # L = 5*H
THETAS_DIM = 5
LATENT_DIM = 16  # Research-confirmed optimal (significantly better than ld=4/8)
# BASIS_DIM is set dynamically to forecast_length (eq_fcast) per horizon
MAX_EPOCHS = 100
PATIENCE = 10
N_RUNS = 8
BASE_SEED = 1  # Seeds 1-8 matching NHiTS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CSV_FILENAME = "nhits_benchmark_results.csv"

# Horizon-adaptive NHiTS pooling/interpolation overrides.
# Determined via 36-config grid search (experiments/configs/nhits_pooling_search.yaml)
# over pool ∈ {[2,2,2],[4,2,1],[2,2,1],[8,4,1],[16,8,1],[32,16,1]} and
# freq ∈ {[168,24,1],[144,36,1],[72,18,1],[48,12,1],[24,12,1],[1,1,1]}.
# Winners per horizon (see experiments/analyze_pooling_search.py).
NHITS_HORIZON_OVERRIDES = {
    96: {
        "n_pools_kernel_size": [2, 2, 2],
        "n_freq_downsample": [180, 60, 1],
    },
    192: {
        "n_pools_kernel_size": [4, 2, 1],
        "n_freq_downsample": [24, 12, 1],
    },
    336: {
        "n_pools_kernel_size": [16, 8, 1],
        "n_freq_downsample": [24, 12, 1],
    },
    720: {
        "n_pools_kernel_size": [32, 16, 1],
        "n_freq_downsample": [24, 12, 1],
    },
}
CLAIMS_DIR = os.path.join(RESULTS_DIR, ".claims")
STALE_CLAIM_SECONDS = 7200  # 2 hours — assume crashed worker

# ---------------------------------------------------------------------------
# CSV Schema
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "model_type",
    "config_name",
    "dataset",
    "horizon",
    "backcast_length",
    "n_stacks",
    "n_blocks_per_stack",
    "run",
    "seed",
    "mse",
    "mae",
    "denorm_mse",
    "denorm_mae",
    "smape",
    "n_params",
    "training_time_seconds",
    "epochs_trained",
    "stopping_reason",
    "best_val_loss",
    "loss_function",
    "active_g",
    "normalize",
    "train_ratio",
    "val_ratio",
    "include_target",
    "stack_types",
    "n_pools_kernel_size",
    "n_freq_downsample",
    "sampling_style",
    "steps_per_epoch",
    "sampling_weights",
]


# ---------------------------------------------------------------------------
# Benchmark Configs
# ---------------------------------------------------------------------------


def get_benchmark_configs():
    """Return the 9 benchmark configurations.

    Selection rationale (evidence-based):
      Reference baselines (2 configs — vanilla blocks, same pipeline):
        1. Generic-10              — original N-BEATS-G (10×Generic)
        2. NHiTS-Generic           — original NHiTS-G (3×Generic + pooling)

      NBeatsNet novel blocks (4 configs, 10 stacks each):
        3. GenericAELG-10          — pure generic baseline
        4. BottleneckGenericAELG-10 — promoted; consistent convergence on Traffic
        5. TrendAELG+Sym20V3AELG   — alternating; Sym20 is universal best wavelet
        6. TrendWaveletAELG-10     — unified trend+wavelet block

      NHiTSNet novel blocks (3 configs, 3 stacks, hierarchical pooling):
        7. NHiTS-GenericAELG                     — NHiTS generic baseline
        8. NHiTS-TrendAELG+Coif2V3AELG+GenericAELG — Coif2 is best forecast wavelet
        9. NHiTS-TrendWaveletAELG                — unified in NHiTS framework
    """
    configs = []

    # --- Reference baselines (vanilla blocks, same pipeline) ---
    configs.append(
        {
            "model_type": "NBeatsNet",
            "config_name": "Generic-10",
            "stack_types": ["Generic"] * 10,
            "n_blocks_per_stack": 1,
        }
    )
    configs.append(
        {
            "model_type": "NHiTSNet",
            "config_name": "NHiTS-Generic",
            "stack_types": ["Generic"] * 3,
            "n_blocks_per_stack": 1,
            "n_pools_kernel_size": [8, 4, 1],
            "n_freq_downsample": [24, 12, 1],
        }
    )

    # --- NBeatsNet novel configs (4) ---
    configs.append(
        {
            "model_type": "NBeatsNet",
            "config_name": "GenericAELG-10",
            "stack_types": ["GenericAELG"] * 10,
            "n_blocks_per_stack": 1,
        }
    )
    configs.append(
        {
            "model_type": "NBeatsNet",
            "config_name": "BottleneckGenericAELG-10",
            "stack_types": ["BottleneckGenericAELG"] * 10,
            "n_blocks_per_stack": 1,
        }
    )
    configs.append(
        {
            "model_type": "NBeatsNet",
            "config_name": "TrendAELG+Sym20V3AELG",
            "stack_types": ["TrendAELG", "Symlet20WaveletV3AELG"] * 5,
            "n_blocks_per_stack": 1,
        }
    )
    configs.append(
        {
            "model_type": "NBeatsNet",
            "config_name": "TrendWaveletAELG-10",
            "stack_types": ["TrendWaveletAELG"] * 10,
            "n_blocks_per_stack": 1,
        }
    )

    # --- NHiTSNet configs (3) ---
    configs.append(
        {
            "model_type": "NHiTSNet",
            "config_name": "NHiTS-GenericAELG",
            "stack_types": ["GenericAELG"] * 3,
            "n_blocks_per_stack": 1,
            "n_pools_kernel_size": [8, 4, 1],
            "n_freq_downsample": [24, 12, 1],
        }
    )
    configs.append(
        {
            "model_type": "NHiTSNet",
            "config_name": "NHiTS-TrendAELG+Coif2V3AELG",
            "stack_types": ["TrendAELG", "Coif2WaveletV3AELG", "GenericAELG"],
            "n_blocks_per_stack": 1,
            "n_pools_kernel_size": [8, 4, 1],
            "n_freq_downsample": [24, 12, 1],
        }
    )
    configs.append(
        {
            "model_type": "NHiTSNet",
            "config_name": "NHiTS-TrendWaveletAELG",
            "stack_types": ["TrendWaveletAELG"] * 3,
            "n_blocks_per_stack": 1,
            "n_pools_kernel_size": [8, 4, 1],
            "n_freq_downsample": [24, 12, 1],
        }
    )

    # --- active_g=forecast variants (Generic-family only) ---
    # Evidence: active_g=forecast provides ~11.6% OWA improvement for Generic blocks
    # but is neutral-to-harmful for TrendWavelet blocks (see wavelet_study_3).
    configs.append(
        {
            "model_type": "NBeatsNet",
            "config_name": "Generic-10-agF",
            "stack_types": ["Generic"] * 10,
            "n_blocks_per_stack": 1,
            "active_g": "forecast",
        }
    )
    configs.append(
        {
            "model_type": "NBeatsNet",
            "config_name": "GenericAELG-10-agF",
            "stack_types": ["GenericAELG"] * 10,
            "n_blocks_per_stack": 1,
            "active_g": "forecast",
        }
    )
    configs.append(
        {
            "model_type": "NBeatsNet",
            "config_name": "GenericAE-10-agF",
            "stack_types": ["GenericAE"] * 10,
            "n_blocks_per_stack": 1,
            "active_g": "forecast",
        }
    )
    configs.append(
        {
            "model_type": "NBeatsNet",
            "config_name": "BottleneckGenericAELG-10-agF",
            "stack_types": ["BottleneckGenericAELG"] * 10,
            "n_blocks_per_stack": 1,
            "active_g": "forecast",
        }
    )
    configs.append(
        {
            "model_type": "NBeatsNet",
            "config_name": "BottleneckGenericAE-10-agF",
            "stack_types": ["BottleneckGenericAE"] * 10,
            "n_blocks_per_stack": 1,
            "active_g": "forecast",
        }
    )
    configs.append(
        {
            "model_type": "NHiTSNet",
            "config_name": "NHiTS-Generic-agF",
            "stack_types": ["Generic"] * 3,
            "n_blocks_per_stack": 1,
            "n_pools_kernel_size": [8, 4, 1],
            "n_freq_downsample": [24, 12, 1],
            "active_g": "forecast",
        }
    )
    configs.append(
        {
            "model_type": "NHiTSNet",
            "config_name": "NHiTS-GenericAELG-agF",
            "stack_types": ["GenericAELG"] * 3,
            "n_blocks_per_stack": 1,
            "n_pools_kernel_size": [8, 4, 1],
            "n_freq_downsample": [24, 12, 1],
            "active_g": "forecast",
        }
    )
    configs.append(
        {
            "model_type": "NHiTSNet",
            "config_name": "NHiTS-GenericAE-agF",
            "stack_types": ["GenericAE"] * 3,
            "n_blocks_per_stack": 1,
            "n_pools_kernel_size": [8, 4, 1],
            "n_freq_downsample": [24, 12, 1],
            "active_g": "forecast",
        }
    )

    return configs


# ---------------------------------------------------------------------------
# YAML Config Loading
# ---------------------------------------------------------------------------


def parse_stack_types(raw):
    """Parse stack_types from YAML, handling both list and string expression forms.

    Supports:
      - Already a list: ['Generic', 'Generic', 'Generic']
      - Python expression string: "['Generic'] * 10"
      - Python expression string: "['TrendAELG', 'Symlet20WaveletV3AELG'] * 5"
    """
    if isinstance(raw, list):
        return raw

    if not isinstance(raw, str):
        raise ValueError(
            f"stack_types must be a list or string expression, got {type(raw)}"
        )

    raw = raw.strip()

    # Handle "['X'] * N" or "['A', 'B'] * N" patterns safely
    # Split on ' * ' to separate list literal from multiplier
    match = re.match(r"^(\[.*\])\s*\*\s*(\d+)$", raw)
    if match:
        list_part = ast.literal_eval(match.group(1))
        multiplier = int(match.group(2))
        return list_part * multiplier

    # Try parsing as a plain list literal: "['A', 'B', 'C']"
    try:
        result = ast.literal_eval(raw)
        if isinstance(result, list):
            return result
    except (ValueError, SyntaxError):
        pass

    raise ValueError(f"Cannot parse stack_types expression: {raw!r}")


def load_yaml_configs(yaml_path):
    """Load benchmark configs from a YAML file.

    Returns:
        dict with keys: configs, protocol, training, block_params, runs,
                        dataset, horizons, experiment_name
    """
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    # --- Extract protocol settings (with defaults matching hardcoded constants) ---
    protocol_raw = yaml_data.get("protocol", {})
    protocol = {
        "train_ratio": protocol_raw.get("train_ratio", TRAIN_RATIO),
        "val_ratio": protocol_raw.get("val_ratio", VAL_RATIO),
        "loss": protocol_raw.get("loss", LOSS),
        "forecast_multiplier": protocol_raw.get(
            "forecast_multiplier", FORECAST_MULTIPLIER
        ),
        "batch_size": protocol_raw.get("batch_size", BATCH_SIZE),
        "normalize": protocol_raw.get(
            "normalize", None
        ),  # None = auto-detect by dataset
        "normalization_style": protocol_raw.get("normalization_style", None),
        "include_target": protocol_raw.get("include_target", True),
        "sampling_style": protocol_raw.get("sampling_style", "sliding"),
        "steps_per_epoch": protocol_raw.get("steps_per_epoch", None),
        "sampling_weights": protocol_raw.get("sampling_weights", "uniform"),
        "acknowledge_epoch_semantics": protocol_raw.get(
            "acknowledge_epoch_semantics", False
        ),
    }

    # --- Extract training settings ---
    training_raw = yaml_data.get("training", {})

    def _coerce_float(v):
        # YAML 1.1 (PyYAML) does not parse unquoted '1e-3' as float — it stays
        # a string. Coerce here so numerics written in scientific form survive.
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return v
        return v

    def _coerce_int(v):
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                return v
        return v

    # Parse optional nested lr_scheduler block (wins over top-level warmup_epochs)
    lr_sched_raw = training_raw.get("lr_scheduler")
    if isinstance(lr_sched_raw, dict):
        lr_scheduler = {
            "type": str(lr_sched_raw.get("type", "cosine")).lower(),
            "warmup_epochs": _coerce_int(lr_sched_raw.get("warmup_epochs")),
            "T_max": _coerce_int(lr_sched_raw.get("T_max")),
            "eta_min": _coerce_float(lr_sched_raw.get("eta_min")),
            "step_size": _coerce_int(lr_sched_raw.get("step_size")),
            "gamma": _coerce_float(lr_sched_raw.get("gamma")),
            "interval": lr_sched_raw.get("interval"),
            # plateau-specific fields
            "factor": _coerce_float(lr_sched_raw.get("factor")),
            "patience": _coerce_int(lr_sched_raw.get("patience")),
            "min_lr": _coerce_float(
                lr_sched_raw.get("min_lr", lr_sched_raw.get("eta_min"))
            ),
            "mode": lr_sched_raw.get("mode", "min"),
            "cooldown": _coerce_int(lr_sched_raw.get("cooldown")),
            "monitor": lr_sched_raw.get("monitor", "val_loss"),
        }
    else:
        lr_scheduler = None

    training = {
        "max_epochs": _coerce_int(training_raw.get("max_epochs", MAX_EPOCHS)),
        "patience": _coerce_int(training_raw.get("patience", PATIENCE)),
        "activation": training_raw.get("activation", "ReLU"),
        "active_g": training_raw.get("active_g", False),
        "sum_losses": training_raw.get("sum_losses", False),
        "warmup_epochs": _coerce_int(training_raw.get("warmup_epochs")),
        "learning_rate": _coerce_float(training_raw.get("learning_rate")),
        "num_workers": _coerce_int(training_raw.get("num_workers")),
        "max_steps": _coerce_int(training_raw.get("max_steps")),
        "early_stopping": training_raw.get("early_stopping", True),
        "val_check_interval": _coerce_int(training_raw.get("val_check_interval")),
        "lr_scheduler": lr_scheduler,
    }

    # --- Extract block params ---
    block_params_raw = yaml_data.get("block_params", {})
    block_params = {
        "thetas_dim": block_params_raw.get("thetas_dim", THETAS_DIM),
        "latent_dim": block_params_raw.get("latent_dim", LATENT_DIM),
        "basis_dim": block_params_raw.get("basis_dim", "eq_fcast"),
    }

    # --- Extract runs settings ---
    runs_raw = yaml_data.get("runs", {})
    runs = {
        "n_runs": runs_raw.get("n_runs", N_RUNS),
        "base_seed": runs_raw.get("base_seed", BASE_SEED),
    }

    # --- Extract dataset and horizons ---
    dataset = yaml_data.get("dataset", None)
    horizons = yaml_data.get("horizons", None)
    experiment_name = yaml_data.get("experiment_name", "yaml_experiment")
    output_csv = yaml_data.get("output_csv", None)

    # --- Parse configs ---
    configs_raw = yaml_data.get("configs", [])
    configs = []
    for cfg in configs_raw:
        stack_types = parse_stack_types(cfg["stack_types"])

        parsed = {
            "config_name": cfg["name"],
            "model_type": cfg.get("model", "NBeatsNet"),
            "stack_types": stack_types,
            "n_blocks_per_stack": cfg.get("n_blocks_per_stack", 1),
        }

        # Per-config active_g override (falls back to training-level default)
        if "active_g" in cfg:
            parsed["active_g"] = cfg["active_g"]
        elif training["active_g"]:
            parsed["active_g"] = training["active_g"]

        # NHiTSNet-specific pooling params
        if "n_pools_kernel_size" in cfg:
            parsed["n_pools_kernel_size"] = cfg["n_pools_kernel_size"]
        if "n_freq_downsample" in cfg:
            parsed["n_freq_downsample"] = cfg["n_freq_downsample"]

        configs.append(parsed)

    # --- Top-level flags ---
    skip_horizon_overrides = yaml_data.get("skip_horizon_overrides", False)

    return {
        "configs": configs,
        "protocol": protocol,
        "training": training,
        "block_params": block_params,
        "runs": runs,
        "dataset": dataset,
        "horizons": horizons,
        "experiment_name": experiment_name,
        "output_csv": output_csv,
        "skip_horizon_overrides": skip_horizon_overrides,
    }


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def set_seed(seed):
    pl.seed_everything(seed, workers=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_smape(y_pred, y_true):
    eps = 1e-8
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return np.mean(numerator / denominator) * 100.0


def compute_mae(y_pred, y_true):
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_mse(y_pred, y_true):
    return float(np.mean((y_true - y_pred) ** 2))


def compute_denormalized_mse_mae(preds, targets, col_indices, col_means, col_stds):
    """Compute MSE and MAE after reversing z-score normalization.

    When col_means/col_stds are None (no normalization was applied),
    returns standard MSE/MAE on the raw predictions.

    Args:
        preds: array of shape (n_samples, forecast_length)
        targets: array of shape (n_samples, forecast_length)
        col_indices: list of (col_name, start_idx) from ColumnarTimeSeriesDataset
        col_means: dict mapping column names to training means (or None)
        col_stds: dict mapping column names to training stds (or None)

    Returns:
        (denorm_mae, denorm_mse) on original-scale data
    """
    if col_means is None or col_stds is None:
        return float(np.mean(np.abs(targets - preds))), float(
            np.mean((targets - preds) ** 2)
        )

    preds_denorm = np.empty_like(preds)
    targets_denorm = np.empty_like(targets)
    for i, (col, _start) in enumerate(col_indices):
        mu = col_means[col]
        sigma = col_stds[col]
        preds_denorm[i] = preds[i] * sigma + mu
        targets_denorm[i] = targets[i] * sigma + mu

    denorm_mae = float(np.mean(np.abs(targets_denorm - preds_denorm)))
    denorm_mse = float(np.mean((targets_denorm - preds_denorm) ** 2))
    return denorm_mae, denorm_mse


def resolve_accelerator():
    if torch.cuda.is_available():
        return "cuda", torch.device("cuda")
    elif torch.backends.mps.is_available():
        return "mps", torch.device("mps")
    return "cpu", torch.device("cpu")


# ---------------------------------------------------------------------------
# Callbacks (from run_unified_benchmark.py)
# ---------------------------------------------------------------------------


class DivergenceDetector(pl.Callback):
    def __init__(self, relative_threshold=3.0, consecutive_epochs=3):
        super().__init__()
        self.relative_threshold = relative_threshold
        self.consecutive_epochs = consecutive_epochs
        self.best_val_loss = float("inf")
        self.bad_epoch_count = 0
        self.diverged = False

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        v = trainer.callback_metrics.get("val_loss")
        if v is None:
            return
        val_loss = float(v)
        if not math.isfinite(val_loss):
            self.diverged = True
            trainer.should_stop = True
            return
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.bad_epoch_count = 0
            return
        if (
            self.best_val_loss > 0
            and val_loss > self.best_val_loss * self.relative_threshold
        ):
            self.bad_epoch_count += 1
        else:
            self.bad_epoch_count = 0
        if self.bad_epoch_count >= self.consecutive_epochs:
            self.diverged = True
            trainer.should_stop = True


class ConvergenceTracker(pl.Callback):
    def __init__(self):
        super().__init__()
        self.val_losses = []
        self.train_losses = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        v = trainer.callback_metrics.get("val_loss")
        if v is not None:
            self.val_losses.append(float(v))

    def on_train_epoch_end(self, trainer, pl_module):
        v = trainer.callback_metrics.get("train_loss")
        if v is not None:
            self.train_losses.append(float(v))


class LearningRateLogger(pl.Callback):
    """Logs the learning rate at the start of each training epoch."""

    def on_train_epoch_start(self, trainer, pl_module):
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        pl_module.log("lr_epoch", lr, prog_bar=True)
        print(f"  Epoch {trainer.current_epoch + 1}: lr = {lr:.6f}")


# ---------------------------------------------------------------------------
# CSV Helpers
# ---------------------------------------------------------------------------


def init_csv(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
        return

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            existing_header = next(reader)
        except StopIteration:
            existing_header = []
    if existing_header == CSV_COLUMNS:
        return

    # Migrate header
    print(
        f"  [MIGRATE] {os.path.basename(path)}: "
        f"header {len(existing_header)} cols -> {len(CSV_COLUMNS)} cols"
    )
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        old_header = next(reader)
        raw_rows = list(reader)

    migrated = []
    for raw in raw_rows:
        row_dict = {}
        for i, col_name in enumerate(old_header):
            if col_name in CSV_COLUMNS and i < len(raw):
                row_dict[col_name] = raw[i]
        for col in CSV_COLUMNS:
            if col not in row_dict:
                row_dict[col] = ""
        migrated.append(row_dict)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(migrated)


def append_result(path, row_dict):
    lock_path = path + ".lock"
    with open(lock_path, "w") as lock_f:
        with _exclusive_lock(lock_f):
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writerow(row_dict)


def result_exists(path, config_name, dataset, horizon, run):
    if not os.path.exists(path):
        return False
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row.get("config_name") == config_name
                and row.get("dataset") == dataset
                and row.get("horizon") == str(horizon)
                and row.get("run") == str(run)
            ):
                return True
    return False


# ---------------------------------------------------------------------------
# Claim-based Job Locking (parallel-safe)
# ---------------------------------------------------------------------------


def _claim_key(config_name, dataset, horizon, run):
    """Deterministic filename for a job's claim file."""
    return f"{config_name}__{dataset}__{horizon}__run{run}.claim"


def claim_job(config_name, dataset, horizon, run, worker_id=""):
    """Atomically claim a job. Returns True if this process won the claim.

    Uses O_CREAT | O_EXCL for race-free file creation. Writes worker_id
    and timestamp so stale claims can be detected.
    """
    os.makedirs(CLAIMS_DIR, exist_ok=True)
    claim_path = os.path.join(
        CLAIMS_DIR, _claim_key(config_name, dataset, horizon, run)
    )
    try:
        fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(
                json.dumps(
                    {
                        "worker_id": worker_id,
                        "pid": os.getpid(),
                        "claimed_at": time.time(),
                        "config_name": config_name,
                        "dataset": dataset,
                        "horizon": horizon,
                        "run": run,
                    }
                )
            )
        return True
    except FileExistsError:
        # Another worker already claimed it — check for staleness
        try:
            mtime = os.path.getmtime(claim_path)
            if time.time() - mtime > STALE_CLAIM_SECONDS:
                print(
                    f"  [STALE] Reclaiming stale job: {config_name}/{dataset}-{horizon}/run{run}"
                )
                os.unlink(claim_path)
                return claim_job(config_name, dataset, horizon, run, worker_id)
        except OSError:
            pass
        return False


def release_claim(config_name, dataset, horizon, run):
    """Remove claim file after job completes (CSV row is the permanent record)."""
    claim_path = os.path.join(
        CLAIMS_DIR, _claim_key(config_name, dataset, horizon, run)
    )
    try:
        os.unlink(claim_path)
    except OSError:
        pass


def job_is_claimed(config_name, dataset, horizon, run):
    """Check if another worker has claimed this job (non-stale)."""
    claim_path = os.path.join(
        CLAIMS_DIR, _claim_key(config_name, dataset, horizon, run)
    )
    if not os.path.exists(claim_path):
        return False
    try:
        mtime = os.path.getmtime(claim_path)
        if time.time() - mtime > STALE_CLAIM_SECONDS:
            return False  # stale claim
    except OSError:
        return False
    return True


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(model, test_dm, device):
    """Run the test dataloader and return predictions and targets.

    When the test DataModule uses ``normalization_style='window'`` with
    ``return_scales=True``, each batch is ``(x, y, mu, sigma)`` in normalized
    space. We return both the normalized arrays (``preds``, ``targets``) and
    the original-scale arrays (``preds_raw``, ``targets_raw``) so callers can
    report paper-aligned MSE/MAE in the denormalized domain. In the default
    ``(x, y)`` case the raw arrays are identical to the normalized ones.
    """
    model.eval()
    model.to(device)
    all_preds, all_targets = [], []
    all_preds_raw, all_targets_raw = [], []
    test_dm.setup("test")
    with torch.no_grad():
        for batch in test_dm.test_dataloader():
            if len(batch) == 4:
                x, y, mu, sigma = batch
                x = x.to(device)
                _, forecast = model(x)
                f_np = forecast.cpu().numpy()
                y_np = y.numpy()
                mu_np = mu.numpy().reshape(-1, 1)
                sigma_np = sigma.numpy().reshape(-1, 1)
                all_preds.append(f_np)
                all_targets.append(y_np)
                all_preds_raw.append(f_np * sigma_np + mu_np)
                all_targets_raw.append(y_np * sigma_np + mu_np)
            else:
                x, y = batch
                x = x.to(device)
                _, forecast = model(x)
                f_np = forecast.cpu().numpy()
                y_np = y.numpy()
                all_preds.append(f_np)
                all_targets.append(y_np)
                all_preds_raw.append(f_np)
                all_targets_raw.append(y_np)
    return (
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_preds_raw, axis=0),
        np.concatenate(all_targets_raw, axis=0),
    )


# ---------------------------------------------------------------------------
# Single Experiment
# ---------------------------------------------------------------------------


def run_single_experiment(
    config,
    dataset_name,
    horizon,
    run_idx,
    csv_path,
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE,
    batch_size=BATCH_SIZE,
    worker_id="",
    protocol=None,
    block_params=None,
    training=None,
    runs=None,
    skip_horizon_overrides=False,
):
    """Run a single NHiTS-protocol experiment."""
    global _shutdown_requested
    if _shutdown_requested:
        return

    config_name = config["config_name"]
    model_type = config["model_type"]

    # Check resumability — already completed in CSV
    if result_exists(csv_path, config_name, dataset_name, horizon, run_idx):
        print(f"  [SKIP] {config_name} / {dataset_name}-{horizon} / run {run_idx}")
        return

    # Claim-based locking — attempt atomic claim for parallel workers
    if not claim_job(config_name, dataset_name, horizon, run_idx, worker_id):
        print(
            f"  [CLAIMED] {config_name} / {dataset_name}-{horizon} / run {run_idx} "
            f"(another worker)"
        )
        return

    try:
        _run_experiment_body(
            config,
            dataset_name,
            horizon,
            run_idx,
            csv_path,
            max_epochs,
            patience,
            batch_size,
            worker_id,
            protocol=protocol,
            block_params=block_params,
            training=training,
            runs=runs,
            skip_horizon_overrides=skip_horizon_overrides,
        )
    except Exception as e:
        print(
            f"  [ERROR] {config_name} / {dataset_name}-{horizon} / run {run_idx}: {e}"
        )
        release_claim(config_name, dataset_name, horizon, run_idx)
        raise


def _run_experiment_body(
    config,
    dataset_name,
    horizon,
    run_idx,
    csv_path,
    max_epochs,
    patience,
    batch_size,
    worker_id,
    protocol=None,
    block_params=None,
    training=None,
    runs=None,
    skip_horizon_overrides=False,
):
    """Inner body of run_single_experiment (wrapped in try/finally for claim safety)."""
    # Resolve protocol/block/training/runs params — use provided dicts or fall back to globals
    p_train_ratio = protocol["train_ratio"] if protocol else TRAIN_RATIO
    p_val_ratio = protocol["val_ratio"] if protocol else VAL_RATIO
    p_loss = protocol["loss"] if protocol else LOSS
    p_forecast_multiplier = (
        protocol["forecast_multiplier"] if protocol else FORECAST_MULTIPLIER
    )
    p_include_target = protocol["include_target"] if protocol else True

    # Normalize: use protocol value if explicitly set, otherwise auto-detect by dataset
    if protocol and protocol.get("normalize") is not None:
        p_normalize = protocol["normalize"]
    else:
        p_normalize = dataset_name == "weather"

    # Normalization style: explicit protocol value wins; else legacy bool maps to
    # 'global' / 'none'. 'window' selects RevIN-style per-input-window z-score.
    p_normalization_style = protocol.get("normalization_style") if protocol else None
    if p_normalization_style is None:
        p_normalization_style = "global" if p_normalize else "none"
    elif p_normalization_style not in ("none", "global", "window"):
        raise ValueError(
            f"protocol.normalization_style must be one of "
            f"{{'none','global','window'}}, got {p_normalization_style!r}"
        )

    p_sampling_style = (
        protocol.get("sampling_style", "sliding") if protocol else "sliding"
    )
    p_steps_per_epoch = protocol.get("steps_per_epoch") if protocol else None
    p_sampling_weights = (
        protocol.get("sampling_weights", "uniform") if protocol else "uniform"
    )
    p_ack_epoch_semantics = bool(
        protocol.get("acknowledge_epoch_semantics", False) if protocol else False
    )
    if p_sampling_style not in ("sliding", "nhits_paper"):
        raise ValueError(
            f"protocol.sampling_style must be one of {{'sliding','nhits_paper'}}, "
            f"got {p_sampling_style!r}"
        )
    if p_sampling_weights not in ("uniform", "by_series"):
        raise ValueError(
            f"protocol.sampling_weights must be one of {{'uniform','by_series'}}, "
            f"got {p_sampling_weights!r}"
        )
    if p_sampling_style == "nhits_paper":
        if p_steps_per_epoch is None:
            raise ValueError(
                "protocol.sampling_style='nhits_paper' requires "
                "protocol.steps_per_epoch to be set."
            )
        if (
            not isinstance(p_steps_per_epoch, int)
            or isinstance(p_steps_per_epoch, bool)
            or p_steps_per_epoch <= 0
        ):
            raise ValueError(
                f"protocol.steps_per_epoch must be a positive int, "
                f"got {p_steps_per_epoch!r}"
            )

    bp_thetas_dim = block_params["thetas_dim"] if block_params else THETAS_DIM
    bp_latent_dim = block_params["latent_dim"] if block_params else LATENT_DIM
    bp_basis_dim_spec = block_params["basis_dim"] if block_params else "eq_fcast"

    t_activation = training["activation"] if training else "ReLU"
    t_sum_losses = training["sum_losses"] if training else False
    t_learning_rate = training.get("learning_rate") if training else None
    if t_learning_rate is None:
        t_learning_rate = LEARNING_RATE

    t_num_workers = training.get("num_workers") if training else None
    if t_num_workers is None:
        t_num_workers = 0

    t_max_steps = training.get("max_steps") if training else None
    t_early_stopping = training.get("early_stopping", True) if training else True
    t_val_check_interval = training.get("val_check_interval") if training else None

    # Epochs-vs-steps guard: paper-style sampling redefines "epoch" as
    # steps_per_epoch gradient updates, so max_epochs has different semantics
    # than sliding mode. Force the user to either specify max_steps
    # (unambiguous) or explicitly acknowledge the epoch semantics shift.
    if p_sampling_style == "nhits_paper" and t_max_steps is None:
        if not p_ack_epoch_semantics:
            raise ValueError(
                "sampling_style='nhits_paper' changes epoch semantics: one epoch = "
                "steps_per_epoch gradient updates. Specify training.max_steps "
                "for an unambiguous budget, or set "
                "protocol.acknowledge_epoch_semantics: true to keep using "
                "training.max_epochs (total steps will be "
                "max_epochs * steps_per_epoch)."
            )
        warnings.warn(
            f"sampling_style='nhits_paper' with max_epochs={max_epochs} and no "
            f"max_steps: total training = {max_epochs * p_steps_per_epoch} "
            f"gradient steps (steps_per_epoch={p_steps_per_epoch}).",
            UserWarning,
            stacklevel=2,
        )

    # Assemble lr_scheduler_config. Preferred path is training['lr_scheduler']
    # (nested block supporting {'type': 'cosine'|'step', ...}); fall back to
    # legacy top-level training['warmup_epochs'] for existing YAML configs.
    lr_scheduler_config = None
    t_lr_sched = training.get("lr_scheduler") if training else None
    if isinstance(t_lr_sched, dict):
        sched_type = str(t_lr_sched.get("type", "cosine")).lower()
        if sched_type in ("cosine", "cosine_warmup", "sequential"):
            lr_scheduler_config = {
                "type": "cosine",
                "warmup_epochs": t_lr_sched.get("warmup_epochs", 15),
                "T_max": t_lr_sched.get("T_max") or max_epochs,
                "eta_min": t_lr_sched.get("eta_min", 1e-6),
            }
        elif sched_type == "plateau":
            lr_scheduler_config = {
                "type": "plateau",
                "factor": (
                    t_lr_sched.get("factor")
                    if t_lr_sched.get("factor") is not None
                    else 0.5
                ),
                "patience": (
                    t_lr_sched.get("patience")
                    if t_lr_sched.get("patience") is not None
                    else 10
                ),
                "min_lr": (
                    t_lr_sched.get("min_lr")
                    if t_lr_sched.get("min_lr") is not None
                    else t_lr_sched.get("eta_min", 1e-5)
                ),
                "mode": t_lr_sched.get("mode", "min"),
                "cooldown": (
                    t_lr_sched.get("cooldown")
                    if t_lr_sched.get("cooldown") is not None
                    else 0
                ),
                "monitor": t_lr_sched.get("monitor", "val_loss"),
            }
        elif sched_type == "step":
            if t_lr_sched.get("step_size") is None:
                raise ValueError(
                    "training.lr_scheduler.type='step' requires 'step_size'."
                )
            lr_scheduler_config = {
                "type": "step",
                "step_size": t_lr_sched["step_size"],
                "gamma": t_lr_sched.get("gamma", 0.5),
            }
        else:
            raise ValueError(
                f"Unknown training.lr_scheduler.type {sched_type!r}; "
                "expected one of {{'cosine', 'plateau', 'step'}}."
            )
        if t_lr_sched.get("interval"):
            lr_scheduler_config["interval"] = t_lr_sched["interval"]
    elif training and training.get("warmup_epochs") is not None:
        lr_scheduler_config = {
            "type": "cosine",
            "warmup_epochs": training["warmup_epochs"],
            "T_max": max_epochs,
            "eta_min": 1e-6,
        }

    r_base_seed = runs["base_seed"] if runs else BASE_SEED

    config_name = config["config_name"]
    model_type = config["model_type"]

    seed = r_base_seed + run_idx
    set_seed(seed)

    # Load dataset with protocol params (70/10/20 LTSF split)
    if dataset_name == "weather":
        dataset = WeatherDataset(
            horizon,
            train_ratio=p_train_ratio,
            val_ratio=p_val_ratio,
            include_target=p_include_target,
        )
    elif dataset_name == "traffic":
        dataset = TrafficDataset(
            horizon,
            train_ratio=p_train_ratio,
            val_ratio=p_val_ratio,
            include_target=p_include_target,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    forecast_length = dataset.forecast_length
    backcast_length = forecast_length * p_forecast_multiplier

    stack_types = config["stack_types"]
    n_stacks = len(stack_types)
    n_blocks_per_stack = config.get("n_blocks_per_stack", 1)

    accelerator, device = resolve_accelerator()

    # Precision
    if accelerator == "cuda" and torch.cuda.is_bf16_supported():
        precision = "bf16-mixed"
    elif accelerator == "cuda":
        precision = "32-true"
    else:
        precision = "32-true"

    # basis_dim: "eq_fcast" means match to forecast_length, otherwise use the integer value
    if bp_basis_dim_spec == "eq_fcast":
        basis_dim = forecast_length
    else:
        basis_dim = int(bp_basis_dim_spec)

    # Create model
    model_kwargs = dict(
        backcast_length=backcast_length,
        forecast_length=forecast_length,
        stack_types=stack_types,
        n_blocks_per_stack=n_blocks_per_stack,
        share_weights=True,
        thetas_dim=bp_thetas_dim,
        loss=p_loss,
        active_g=config.get("active_g", False),
        sum_losses=t_sum_losses,
        activation=t_activation,
        latent_dim=bp_latent_dim,
        basis_dim=basis_dim,
        learning_rate=t_learning_rate,
        optimizer_name="Adam",
        no_val=False,
        lr_scheduler_config=lr_scheduler_config,
    )

    pools = []
    freqs = []
    if model_type == "NHiTSNet":
        # Per-config values are optional. When omitted, fall back to
        # NHiTSNet's own defaults ([2]*n_stacks / [1]*n_stacks). Horizon
        # overrides still take precedence unless skip_horizon_overrides
        # is set (e.g. during parameter search).
        pools = list(config.get("n_pools_kernel_size", [2] * n_stacks))
        freqs = list(config.get("n_freq_downsample", [1] * n_stacks))
        if not skip_horizon_overrides and forecast_length in NHITS_HORIZON_OVERRIDES:
            ho = NHITS_HORIZON_OVERRIDES[forecast_length]
            pools = ho.get("n_pools_kernel_size", pools)
            freqs = ho.get("n_freq_downsample", freqs)
        model_kwargs["n_pools_kernel_size"] = pools
        model_kwargs["n_freq_downsample"] = freqs
        ModelClass = NHiTSNet
    else:
        ModelClass = NBeatsNet

    model = ModelClass(**model_kwargs)
    n_params = count_parameters(model)

    train_data = dataset.train_data
    val_data = dataset.val_data
    test_data = dataset.test_data

    dm = ColumnarCollectionTimeSeriesDataModule(
        train_data,
        backcast_length=backcast_length,
        forecast_length=forecast_length,
        batch_size=batch_size,
        no_val=False,
        normalize=p_normalize,
        normalization_style=p_normalization_style,
        val_data=val_data,
        num_workers=t_num_workers,
        sampling_style=p_sampling_style,
        steps_per_epoch=p_steps_per_epoch,
        sampling_weights=p_sampling_weights,
    )

    # We need to call setup() on dm first to get normalization stats
    dm.setup()

    # In 'window' mode we ask the test DM to emit (x, y, mu, sigma) so predictions
    # can be denormalized back to the original scale before computing metrics.
    test_dm = ColumnarCollectionTimeSeriesTestDataModule(
        train_data,
        test_data,
        backcast_length=backcast_length,
        forecast_length=forecast_length,
        batch_size=batch_size,
        col_means=dm.col_means,
        col_stds=dm.col_stds,
        normalization_style=p_normalization_style,
        return_scales=(p_normalization_style == "window"),
        num_workers=t_num_workers,
    )

    # Trainer
    ckpt_tmp_dir = tempfile.mkdtemp(prefix="nhits_ckpt_")
    chk_callback = ModelCheckpoint(
        dirpath=ckpt_tmp_dir,
        filename="best-checkpoint",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        verbose=False,
    )
    # Long-horizon training (720-step forecast / 3600-step backcast) is
    # inherently noisier; the standard 3× / 3-epoch threshold fires on
    # normal fluctuations.  Use a looser threshold for those runs.
    _div_threshold = 6.0 if forecast_length >= 720 else 3.0
    _div_epochs = 5 if forecast_length >= 720 else 3
    divergence_detector = DivergenceDetector(
        relative_threshold=_div_threshold, consecutive_epochs=_div_epochs
    )
    convergence_tracker = ConvergenceTracker()
    lr_logger = LearningRateLogger()

    trainer_callbacks = [
        chk_callback,
        divergence_detector,
        convergence_tracker,
        lr_logger,
    ]
    if t_early_stopping:
        trainer_callbacks.insert(1, early_stop_callback)

    trainer_kwargs = dict(
        accelerator=accelerator,
        devices=1,
        max_epochs=max_epochs,
        precision=precision,
        gradient_clip_val=1.0,
        callbacks=trainer_callbacks,
        logger=False,
        enable_progress_bar=True,
        deterministic=False,
        log_every_n_steps=1,
    )
    if t_max_steps is not None:
        trainer_kwargs["max_steps"] = t_max_steps
    if t_val_check_interval is not None:
        trainer_kwargs["val_check_interval"] = t_val_check_interval
    trainer = pl.Trainer(**trainer_kwargs)

    # Train
    stack_summary = (
        f"{n_stacks}x{stack_types[0]}"
        if len(set(stack_types)) == 1
        else f"{n_stacks} mixed"
    )
    print(
        f"  [RUN]  {config_name} / {dataset_name}-{horizon} / run {run_idx} "
        f"(seed={seed}, {stack_summary}, params={n_params:,})"
    )
    t0 = time.time()
    trainer.fit(model, datamodule=dm)
    training_time = time.time() - t0

    # Load best checkpoint
    best_path = trainer.checkpoint_callback.best_model_path
    if best_path:
        model = ModelClass.load_from_checkpoint(best_path, weights_only=False)
    shutil.rmtree(ckpt_tmp_dir, ignore_errors=True)
    epochs_trained = trainer.current_epoch

    # Stopping reason
    if divergence_detector.diverged:
        stopping_reason = "DIVERGED"
    elif (
        t_early_stopping
        and hasattr(early_stop_callback, "stopped_epoch")
        and early_stop_callback.stopped_epoch > 0
    ):
        stopping_reason = "EARLY_STOPPED"
    elif t_max_steps is not None and trainer.global_step >= t_max_steps:
        stopping_reason = "MAX_STEPS"
    else:
        stopping_reason = "MAX_EPOCHS"

    best_val_loss = (
        float(trainer.checkpoint_callback.best_model_score)
        if trainer.checkpoint_callback.best_model_score is not None
        else float("nan")
    )

    # Inference. In 'global' / 'none' modes preds_raw == preds and targets_raw ==
    # targets. In 'window' mode the raw arrays are denormalized per-sample using
    # the (mu, sigma) emitted by the test dataset.
    preds, targets, preds_raw, targets_raw = run_inference(model, test_dm, device)

    # Metrics on normalized data (primary — matches NHiTS evaluation)
    mse_norm = compute_mse(preds, targets)
    mae_norm = compute_mae(preds, targets)

    # Denormalized metrics (original scale) and sMAPE
    if p_normalization_style == "window":
        denorm_mse = float(np.mean((targets_raw - preds_raw) ** 2))
        denorm_mae = float(np.mean(np.abs(targets_raw - preds_raw)))
    else:
        col_indices = test_dm.test_dataset.col_indices
        denorm_mae, denorm_mse = compute_denormalized_mse_mae(
            preds, targets, col_indices, dm.col_means, dm.col_stds
        )
    smape = compute_smape(preds_raw, targets_raw)

    print(
        f"         MSE={mse_norm:.4f}  MAE={mae_norm:.4f}  "
        f"dMSE={denorm_mse:.4f}  dMAE={denorm_mae:.4f}  sMAPE={smape:.2f}  "
        f"time={training_time:.1f}s  epochs={epochs_trained}  [{stopping_reason}]"
    )

    # Save result
    row = {
        "model_type": model_type,
        "config_name": config_name,
        "dataset": dataset_name,
        "horizon": horizon,
        "backcast_length": backcast_length,
        "n_stacks": n_stacks,
        "n_blocks_per_stack": n_blocks_per_stack,
        "run": run_idx,
        "seed": seed,
        "mse": f"{mse_norm:.6f}",
        "mae": f"{mae_norm:.6f}",
        "denorm_mse": f"{denorm_mse:.6f}" if math.isfinite(denorm_mse) else "nan",
        "denorm_mae": f"{denorm_mae:.6f}" if math.isfinite(denorm_mae) else "nan",
        "smape": f"{smape:.6f}" if math.isfinite(smape) else "nan",
        "n_params": n_params,
        "training_time_seconds": f"{training_time:.2f}",
        "epochs_trained": epochs_trained,
        "stopping_reason": stopping_reason,
        "best_val_loss": (
            f"{best_val_loss:.8f}" if math.isfinite(best_val_loss) else "nan"
        ),
        "loss_function": p_loss,
        "active_g": str(config.get("active_g", False)),
        "normalize": p_normalize,
        "train_ratio": p_train_ratio,
        "val_ratio": p_val_ratio,
        "include_target": p_include_target,
        "stack_types": str(list(dict.fromkeys(stack_types))),
        "n_pools_kernel_size": str(pools) if pools else "",
        "n_freq_downsample": str(freqs) if freqs else "",
        "sampling_style": p_sampling_style,
        "steps_per_epoch": p_steps_per_epoch if p_steps_per_epoch is not None else "",
        "sampling_weights": p_sampling_weights,
    }
    append_result(csv_path, row)
    release_claim(config_name, dataset_name, horizon, run_idx)

    # Cleanup
    del model, trainer, dm, test_dm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_results(csv_path):
    """Print a comparison table from existing results."""
    if not os.path.exists(csv_path):
        print(f"No results file found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Results file is empty.")
        return

    print("\n" + "=" * 100)
    print("NHiTS-Protocol Benchmark Results")
    print("=" * 100)

    # Published NHiTS baselines for comparison
    nhits_baselines = {
        ("weather", 96): {"mse": 0.158, "mae": 0.209},
        ("weather", 192): {"mse": 0.211, "mae": 0.253},
        ("weather", 336): {"mse": 0.272, "mae": 0.296},
        ("weather", 720): {"mse": 0.348, "mae": 0.349},
        ("traffic", 96): {"mse": 0.401, "mae": 0.267},
        ("traffic", 192): {"mse": 0.411, "mae": 0.270},
        ("traffic", 336): {"mse": 0.423, "mae": 0.278},
        ("traffic", 720): {"mse": 0.461, "mae": 0.299},
    }

    for dataset_name in df["dataset"].unique():
        print(f"\n{'─' * 100}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'─' * 100}")

        ds_df = df[df["dataset"] == dataset_name]

        for horizon in sorted(ds_df["horizon"].unique()):
            h_df = ds_df[ds_df["horizon"] == horizon]

            baseline = nhits_baselines.get((dataset_name, int(horizon)), {})
            baseline_mse = baseline.get("mse", float("nan"))
            baseline_mae = baseline.get("mae", float("nan"))

            print(
                f"\n  Horizon: {int(horizon)}  "
                f"(NHiTS published: MSE={baseline_mse:.3f}, MAE={baseline_mae:.3f})"
            )
            print(
                f"  {'Config':<40s} {'Model':<12s} {'MSE':>8s} {'MAE':>8s} "
                f"{'vs NHiTS':>9s} {'Runs':>5s}"
            )
            print(f"  {'─' * 85}")

            summary = (
                h_df.groupby(["config_name", "model_type"])
                .agg(
                    mse_mean=(
                        "mse",
                        lambda x: pd.to_numeric(x, errors="coerce").mean(),
                    ),
                    mae_mean=(
                        "mae",
                        lambda x: pd.to_numeric(x, errors="coerce").mean(),
                    ),
                    mse_std=("mse", lambda x: pd.to_numeric(x, errors="coerce").std()),
                    n_runs=("run", "count"),
                )
                .reset_index()
                .sort_values("mse_mean")
            )

            for _, row in summary.iterrows():
                mse_m = row["mse_mean"]
                mae_m = row["mae_mean"]
                mse_s = row["mse_std"]
                n = int(row["n_runs"])
                gap = (
                    (mse_m - baseline_mse) / baseline_mse * 100
                    if math.isfinite(baseline_mse) and baseline_mse > 0
                    else float("nan")
                )
                gap_str = f"{gap:+.1f}%" if math.isfinite(gap) else "N/A"

                print(
                    f"  {row['config_name']:<40s} {row['model_type']:<12s} "
                    f"{mse_m:8.4f} {mae_m:8.4f} {gap_str:>9s} {n:5d}"
                )

    print(f"\n{'=' * 100}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="NHiTS-Protocol Benchmark for Novel Block Types"
    )
    parser.add_argument(
        "--yaml",
        type=str,
        default=None,
        help="Path to YAML config file (overrides hardcoded configs)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=None,
        choices=["weather", "traffic"],
        help="Dataset(s) to benchmark (default: weather traffic)",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=None,
        help="Forecast horizons to test (default: 96 192 336 720)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=None,
        help="Number of runs per config (default: 8)",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=None, help="Maximum training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=None, help="Early stopping patience"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument(
        "--configs",
        type=str,
        nargs="*",
        default=None,
        help="Filter to specific config names",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print experiment plan without running"
    )
    parser.add_argument(
        "--scheduler-type",
        choices=["cosine", "plateau", "step", "none"],
        default=None,
        help=(
            "Override lr_scheduler type from YAML. "
            "'cosine' = warmup + CosineAnnealingLR; "
            "'plateau' = ReduceLROnPlateau (monitors val_loss); "
            "'step' = StepLR (requires step_size in YAML); "
            "'none' = constant LR (disables scheduler)."
        ),
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze existing results and print comparison table",
    )
    parser.add_argument(
        "--csv-path", type=str, default=None, help="Override CSV output path"
    )
    parser.add_argument(
        "--worker-id",
        type=str,
        default="",
        help="Worker identifier for parallel execution (logged in claim files)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU index to use (sets CUDA_VISIBLE_DEVICES)",
    )

    args = parser.parse_args()

    # Pin to specific GPU if requested
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # --- Load configs: from YAML or hardcoded defaults ---
    yaml_data = None
    protocol = None
    block_params = None
    training = None
    runs = None

    skip_horizon_overrides = False
    if args.yaml:
        yaml_data = load_yaml_configs(args.yaml)
        all_configs = yaml_data["configs"]
        protocol = yaml_data["protocol"]
        block_params = yaml_data["block_params"]
        training = yaml_data["training"]
        runs = yaml_data["runs"]
        skip_horizon_overrides = yaml_data.get("skip_horizon_overrides", False)
    else:
        all_configs = get_benchmark_configs()

    # Resolve CSV path: CLI --csv-path > YAML output_csv > default CSV_FILENAME.
    # A YAML-supplied bare filename is joined with RESULTS_DIR; absolute paths
    # or paths containing a directory component are used as-is.
    if args.csv_path:
        csv_path = args.csv_path
    else:
        yaml_output_csv = yaml_data.get("output_csv") if yaml_data else None
        if yaml_output_csv:
            if os.path.isabs(yaml_output_csv) or os.path.dirname(yaml_output_csv):
                csv_path = yaml_output_csv
            else:
                csv_path = os.path.join(RESULTS_DIR, yaml_output_csv)
        else:
            csv_path = os.path.join(RESULTS_DIR, CSV_FILENAME)

    if args.analyze:
        analyze_results(csv_path)
        return

    # --- Resolve CLI overrides (CLI > YAML > hardcoded defaults) ---
    datasets = args.dataset
    if datasets is None:
        if yaml_data and yaml_data.get("dataset"):
            # YAML dataset can be a string or list
            ds = yaml_data["dataset"]
            datasets = [ds] if isinstance(ds, str) else list(ds)
        else:
            datasets = ["weather", "traffic"]

    horizons = args.horizons
    if horizons is None:
        if yaml_data and yaml_data.get("horizons"):
            horizons = yaml_data["horizons"]
        else:
            horizons = [96, 192, 336, 720]

    n_runs = args.n_runs
    if n_runs is None:
        if runs:
            n_runs = runs["n_runs"]
        else:
            n_runs = N_RUNS

    max_epochs = args.max_epochs
    if max_epochs is None:
        if training:
            max_epochs = training["max_epochs"]
        else:
            max_epochs = MAX_EPOCHS

    patience = args.patience
    if patience is None:
        if training and "patience" in training:
            patience = training["patience"]
        else:
            patience = PATIENCE

    if args.scheduler_type is not None:
        if training is None:
            training = {}
        if args.scheduler_type == "none":
            training["lr_scheduler"] = None
        else:
            if not isinstance(training.get("lr_scheduler"), dict):
                training["lr_scheduler"] = {}
            training["lr_scheduler"]["type"] = args.scheduler_type

    batch_size = args.batch_size
    if batch_size is None:
        if protocol:
            batch_size = protocol["batch_size"]
        else:
            batch_size = BATCH_SIZE

    # Filter configs by name if requested
    if args.configs:
        all_configs = [c for c in all_configs if c["config_name"] in args.configs]
        if not all_configs:
            available = (
                [c["config_name"] for c in load_yaml_configs(args.yaml)["configs"]]
                if args.yaml
                else [c["config_name"] for c in get_benchmark_configs()]
            )
            print(f"No configs matched: {args.configs}")
            print(f"Available: {available}")
            return

    # Build job list
    jobs = []
    for dataset_name in datasets:
        for horizon in horizons:
            for config in all_configs:
                for run_idx in range(n_runs):
                    jobs.append(
                        {
                            "config": config,
                            "dataset_name": dataset_name,
                            "horizon": horizon,
                            "run_idx": run_idx,
                        }
                    )

    total_jobs = len(jobs)
    worker_label = f" (worker: {args.worker_id})" if args.worker_id else ""
    source_label = f" [YAML: {args.yaml}]" if args.yaml else " [hardcoded]"
    print(f"\nNHiTS-Protocol Benchmark{worker_label}{source_label}")
    print(f"  Datasets:   {datasets}")
    print(f"  Horizons:   {horizons}")
    print(f"  Configs:    {len(all_configs)}")
    print(f"  Runs/cfg:   {n_runs}")
    print(f"  Total jobs: {total_jobs}")
    print(f"  CSV path:   {csv_path}")
    if args.gpu is not None:
        print(f"  GPU:        {args.gpu}")

    # Protocol summary
    p_train_ratio = protocol["train_ratio"] if protocol else TRAIN_RATIO
    p_val_ratio = protocol["val_ratio"] if protocol else VAL_RATIO
    p_loss = protocol["loss"] if protocol else LOSS
    p_fcast_mult = protocol["forecast_multiplier"] if protocol else FORECAST_MULTIPLIER
    if protocol and protocol.get("normalize") is not None:
        norm_desc = f"all={'yes' if protocol['normalize'] else 'no'}"
    else:
        norm_desc = ", ".join(
            f"{d}={'yes' if d == 'weather' else 'no'}" for d in datasets
        )
    print(
        f"  Protocol:   normalize=[{norm_desc}], train_ratio={p_train_ratio}, "
        f"val_ratio={p_val_ratio}, loss={p_loss}, L={p_fcast_mult}H"
    )
    print()

    if args.dry_run:
        print("Configs to run:")
        for config in all_configs:
            n = len(config["stack_types"])
            unique = list(dict.fromkeys(config["stack_types"]))
            print(
                f"  {config['config_name']:<40s} {config['model_type']:<12s} "
                f"{n} stacks: {unique}"
            )
        print(f"\n{total_jobs} total experiments (dry run — not executing)")
        return

    init_csv(csv_path)

    completed = 0
    for job in jobs:
        if _shutdown_requested:
            print("\n[SHUTDOWN] Stopping after current run.")
            break

        run_single_experiment(
            config=job["config"],
            dataset_name=job["dataset_name"],
            horizon=job["horizon"],
            run_idx=job["run_idx"],
            csv_path=csv_path,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            worker_id=args.worker_id,
            protocol=protocol,
            block_params=block_params,
            training=training,
            runs=runs,
            skip_horizon_overrides=skip_horizon_overrides,
        )
        completed += 1

    print(f"\nCompleted {completed}/{total_jobs} experiments.")
    print(f"Results saved to: {csv_path}")

    # Auto-analyze if we completed all jobs
    if completed == total_jobs:
        analyze_results(csv_path)


if __name__ == "__main__":
    main()
