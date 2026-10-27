"""N-BEATS Meta-Forecaster for Hyperparameter Search.

Trains a small N-BEATS model on historical val_loss curves from prior experiments.
After training, it takes a short (6-epoch) val_loss snippet and forecasts the next
6 epochs, enabling early pruning of unpromising hyperparameter configurations.

Usage:
    from tools.meta_forecaster import MetaForecaster

    mf = MetaForecaster(cache_dir="experiments/results/.meta_cache")
    mf.train([
        "experiments/results/m4/unified_benchmark_results.csv",
        "experiments/results/m4/convergence_study_results_v1.csv",
    ])

    # After running a config for 6 epochs:
    result = mf.predict([90.2, 42.1, 31.5, 22.0, 19.3, 17.8])
    print(result["predicted_best"])       # predicted minimum loss
    print(result["convergence_score"])    # lower = more promising
"""

import csv
import json
import math
import os
import sys
import tempfile

import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# Allow running from project root or experiments/
# __file__ is experiments/tools/meta_forecaster.py; climb two levels to reach the project src/
_EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_EXPERIMENTS_DIR, "..", "src"))

from pe_nbeats.models import NBeatsNet
from pe_nbeats.loaders import TimeSeriesDataModule


class MetaForecaster:
    """N-BEATS model that predicts val_loss curves from early training epochs.

    Uses existing experiment results as training data. After training, takes
    a 6-epoch val_loss snippet and forecasts the next 6 epochs, enabling
    early pruning of unpromising hyperparameter configs.
    """

    BACKCAST_LENGTH = 6
    FORECAST_LENGTH = 6
    META_STACK_TYPES = ["TrendAE", "HaarWaveletV3"] * 3  # 6 alternating stacks
    META_T_WIDTH = 32                        # TrendAE hidden width
    META_G_WIDTH = 32                        # WaveletV3 hidden width
    META_BLOCKS_PER_STACK = 1
    META_LATENT_DIM = 2                      # AE bottleneck
    META_TREND_THETAS_DIM = 3                # Cubic polynomial basis
    META_BASIS_DIM = 4                       # Wavelet basis functions
    META_MAX_EPOCHS = 200
    META_PATIENCE = 20
    MODEL_FILENAME = "meta_forecaster_v2.ckpt"

    def __init__(self, cache_dir):
        """Initialize with a directory for saving/loading the trained model.

        Parameters
        ----------
        cache_dir : str
            Directory where the trained meta-model checkpoint is stored.
        """
        self.cache_dir = cache_dir
        self.model = None
        os.makedirs(cache_dir, exist_ok=True)

    @property
    def checkpoint_path(self):
        return os.path.join(self.cache_dir, self.MODEL_FILENAME)

    @staticmethod
    def extract_curves_from_csv(csv_path, min_length=12):
        """Parse val_loss_curve JSON column from a results CSV.

        Parameters
        ----------
        csv_path : str
            Path to an experiment results CSV with a val_loss_curve column.
        min_length : int
            Minimum curve length to include (shorter curves are skipped).

        Returns
        -------
        list[list[float]]
            List of val_loss curves, each a list of per-epoch floats.
        """
        curves = []
        if not os.path.exists(csv_path):
            return curves

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = row.get("val_loss_curve", "")
                if not raw or raw == "[]":
                    continue
                try:
                    parsed = json.loads(raw)
                    curve = [float(v) for v in parsed]
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue

                # Skip curves with non-finite values
                if any(not math.isfinite(v) for v in curve):
                    continue
                if len(curve) >= min_length:
                    curves.append(curve)

        return curves

    @staticmethod
    def _prepare_training_data(curves):
        """Create sliding-window samples from val_loss curves.

        Each curve is normalized by dividing by curve[0] so all curves start
        at 1.0. Then sliding windows of (backcast=6, forecast=6) are extracted.

        Parameters
        ----------
        curves : list[list[float]]
            Raw val_loss curves (each >= 12 elements).

        Returns
        -------
        numpy.ndarray
            1D array of all normalized curve segments concatenated, suitable
            for TimeSeriesDataModule with backcast=6, forecast=6.
        """
        window_size = MetaForecaster.BACKCAST_LENGTH + MetaForecaster.FORECAST_LENGTH
        segments = []

        for curve in curves:
            # Normalize: divide by first value so curves start at 1.0
            initial = curve[0]
            if initial <= 0 or not math.isfinite(initial):
                continue
            normalized = [v / initial for v in curve]

            # Skip curves with extreme values after normalization
            if any(v > 10.0 or v < 0 for v in normalized):
                continue

            # Extract sliding windows — the TimeSeriesDataModule handles
            # the actual backcast/forecast splitting, so we just need
            # a long enough 1D segment. We concatenate all windows with a
            # small NaN gap to prevent cross-curve leakage.
            if len(normalized) >= window_size:
                segments.append(np.array(normalized, dtype=np.float32))

        if not segments:
            raise ValueError(
                "No valid training data after filtering. Need curves with "
                f">= {window_size} epochs that don't diverge."
            )

        # Concatenate all curves with a gap of NaNs between them.
        # TimeSeriesDataModule uses a sliding window, so the NaN gaps
        # will create invalid samples that we need to handle differently.
        # Instead, we'll build the dataset ourselves.
        return segments

    @staticmethod
    def _build_dataset_arrays(segments):
        """Convert normalized curve segments into backcast/forecast arrays.

        Parameters
        ----------
        segments : list[numpy.ndarray]
            List of normalized curve segments.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            (X, Y) arrays where X has shape (n_samples, BACKCAST_LENGTH)
            and Y has shape (n_samples, FORECAST_LENGTH).
        """
        bc = MetaForecaster.BACKCAST_LENGTH
        fc = MetaForecaster.FORECAST_LENGTH
        window = bc + fc

        X_list = []
        Y_list = []

        for seg in segments:
            n_windows = len(seg) - window + 1
            for i in range(n_windows):
                x = seg[i : i + bc]
                y = seg[i + bc : i + bc + fc]
                X_list.append(x)
                Y_list.append(y)

        X = np.stack(X_list)
        Y = np.stack(Y_list)
        return X, Y

    def train(self, csv_paths, min_curve_length=12, force_retrain=False):
        """Train the meta-forecaster on val_loss curves from existing CSVs.

        Parameters
        ----------
        csv_paths : list[str]
            Paths to experiment results CSVs with val_loss_curve columns.
        min_curve_length : int
            Minimum curve length to include as training data.
        force_retrain : bool
            If True, retrain even if a cached checkpoint exists.

        Returns
        -------
        dict
            Training metrics: n_curves, n_samples, train_loss, val_loss.
        """
        if not force_retrain and os.path.exists(self.checkpoint_path):
            print(f"  [META] Loading cached meta-forecaster from {self.checkpoint_path}")
            self.load()
            return {"cached": True}

        # Collect curves from all CSVs
        all_curves = []
        for path in csv_paths:
            curves = self.extract_curves_from_csv(path, min_curve_length)
            print(f"  [META] Loaded {len(curves)} curves from {os.path.basename(path)}")
            all_curves.extend(curves)

        if not all_curves:
            raise ValueError(
                f"No valid training curves found in {len(csv_paths)} CSVs. "
                f"Need curves with >= {min_curve_length} epochs."
            )

        # Prepare training data
        segments = self._prepare_training_data(all_curves)
        X, Y = self._build_dataset_arrays(segments)
        n_samples = len(X)

        print(f"  [META] Training data: {len(all_curves)} curves -> "
              f"{n_samples} sliding-window samples")

        # Build a combined array for TimeSeriesDataModule-compatible format.
        # We'll use a RowCollectionTimeSeriesDataModule-style approach but
        # simpler: just create a custom DataModule.
        combined = np.concatenate([X, Y], axis=1)  # (n_samples, 13)

        dm = _MetaDataModule(
            combined,
            backcast_length=self.BACKCAST_LENGTH,
            forecast_length=self.FORECAST_LENGTH,
            batch_size=min(256, n_samples),
        )

        # Build the meta-model
        model = NBeatsNet(
            backcast_length=self.BACKCAST_LENGTH,
            forecast_length=self.FORECAST_LENGTH,
            stack_types=self.META_STACK_TYPES,
            t_width=self.META_T_WIDTH,
            g_width=self.META_G_WIDTH,
            n_blocks_per_stack=self.META_BLOCKS_PER_STACK,
            share_weights=True,
            loss="SMAPELoss",
            learning_rate=1e-3,
            activation="ReLU",
            no_val=False,
            latent_dim=self.META_LATENT_DIM,
            trend_thetas_dim=self.META_TREND_THETAS_DIM,
            basis_dim=self.META_BASIS_DIM,
        )

        # Train
        ckpt_dir = tempfile.mkdtemp(prefix="meta_ckpt_")
        checkpoint_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )
        early_stop_cb = EarlyStopping(
            monitor="val_loss",
            patience=self.META_PATIENCE,
            mode="min",
            verbose=False,
        )

        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=self.META_MAX_EPOCHS,
            callbacks=[checkpoint_cb, early_stop_cb],
            enable_progress_bar=True,
            logger=False,
            gradient_clip_val=1.0,
        )

        print(f"  [META] Training meta-forecaster (backcast={self.BACKCAST_LENGTH}, "
              f"forecast={self.FORECAST_LENGTH}, {n_samples} samples)...")
        trainer.fit(model, datamodule=dm)

        # Load best checkpoint
        best_path = checkpoint_cb.best_model_path
        if best_path:
            model = NBeatsNet.load_from_checkpoint(best_path, weights_only=False)

        best_val = float(checkpoint_cb.best_model_score or float("nan"))
        epochs = trainer.current_epoch

        # Save to cache
        trainer_for_save = pl.Trainer(accelerator="cpu", devices=1, logger=False)
        trainer_for_save.strategy.connect(model)
        trainer_for_save.save_checkpoint(self.checkpoint_path)

        # Cleanup temp dir
        import shutil
        shutil.rmtree(ckpt_dir, ignore_errors=True)

        self.model = model
        self.model.eval()

        metrics = {
            "cached": False,
            "n_curves": len(all_curves),
            "n_samples": n_samples,
            "best_val_loss": best_val,
            "epochs_trained": epochs,
        }
        print(f"  [META] Training complete: val_loss={best_val:.6f}, "
              f"epochs={epochs}, saved to {self.checkpoint_path}")
        return metrics

    def load(self):
        """Load a previously trained model from cache_dir."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"No meta-forecaster checkpoint at {self.checkpoint_path}. "
                f"Run train() first."
            )
        self.model = NBeatsNet.load_from_checkpoint(
            self.checkpoint_path, weights_only=False
        )
        self.model.eval()

    def predict(self, early_curve):
        """Predict future val_loss from a short observed curve.

        Parameters
        ----------
        early_curve : list[float]
            Val_loss values for the first BACKCAST_LENGTH epochs.

        Returns
        -------
        dict
            - predicted_curve: list of FORECAST_LENGTH predicted val_loss values
            - predicted_best: minimum of observed + predicted curve
            - predicted_best_epoch: epoch index of the predicted minimum
            - convergence_score: predicted_best / early_curve[0] (lower = better)
        """
        if self.model is None:
            raise RuntimeError("Meta-forecaster not trained/loaded. Call train() or load() first.")

        if len(early_curve) < self.BACKCAST_LENGTH:
            raise ValueError(
                f"Need at least {self.BACKCAST_LENGTH} epochs, got {len(early_curve)}"
            )

        # Normalize
        initial = early_curve[0]
        if initial <= 0 or not math.isfinite(initial):
            return {
                "predicted_curve": [float("nan")] * self.FORECAST_LENGTH,
                "predicted_best": float("nan"),
                "predicted_best_epoch": -1,
                "convergence_score": float("nan"),
            }

        normalized = [v / initial for v in early_curve[:self.BACKCAST_LENGTH]]
        x = torch.tensor([normalized], dtype=torch.float32)

        with torch.no_grad():
            _, forecast = self.model(x)

        pred_normalized = forecast.squeeze(0).cpu().numpy().tolist()
        pred_denorm = [v * initial for v in pred_normalized]

        full_curve = list(early_curve[:self.BACKCAST_LENGTH]) + pred_denorm
        best_val = min(full_curve)
        best_epoch = int(np.argmin(full_curve))
        convergence_score = best_val / initial if initial > 0 else float("nan")

        return {
            "predicted_curve": pred_denorm,
            "predicted_best": best_val,
            "predicted_best_epoch": best_epoch,
            "convergence_score": convergence_score,
        }

    def predict_batch(self, curves):
        """Vectorized prediction for multiple configs at once.

        Parameters
        ----------
        curves : list[list[float]]
            List of early val_loss curves, each of length BACKCAST_LENGTH.

        Returns
        -------
        list[dict]
            List of prediction results, one per input curve.
        """
        if self.model is None:
            raise RuntimeError("Meta-forecaster not trained/loaded.")

        results = []
        valid_indices = []
        normalized_batch = []
        initials = []

        for i, curve in enumerate(curves):
            if (len(curve) < self.BACKCAST_LENGTH
                    or curve[0] <= 0
                    or not math.isfinite(curve[0])):
                results.append({
                    "predicted_curve": [float("nan")] * self.FORECAST_LENGTH,
                    "predicted_best": float("nan"),
                    "predicted_best_epoch": -1,
                    "convergence_score": float("nan"),
                })
            else:
                initial = curve[0]
                initials.append(initial)
                normalized = [v / initial for v in curve[:self.BACKCAST_LENGTH]]
                normalized_batch.append(normalized)
                valid_indices.append(i)
                results.append(None)  # placeholder

        if normalized_batch:
            x = torch.tensor(normalized_batch, dtype=torch.float32)
            with torch.no_grad():
                _, forecasts = self.model(x)

            forecasts_np = forecasts.cpu().numpy()

            for j, idx in enumerate(valid_indices):
                initial = initials[j]
                pred_normalized = forecasts_np[j].tolist()
                pred_denorm = [v * initial for v in pred_normalized]

                full_curve = list(curves[idx][:self.BACKCAST_LENGTH]) + pred_denorm
                best_val = min(full_curve)
                best_epoch = int(np.argmin(full_curve))
                convergence_score = best_val / initial if initial > 0 else float("nan")

                results[idx] = {
                    "predicted_curve": pred_denorm,
                    "predicted_best": best_val,
                    "predicted_best_epoch": best_epoch,
                    "convergence_score": convergence_score,
                }

        return results


# ---------------------------------------------------------------------------
# Custom DataModule for meta-forecaster training
# ---------------------------------------------------------------------------

class _MetaDataModule(pl.LightningDataModule):
    """Simple DataModule for pre-split (X, Y) pairs stored as row samples."""

    def __init__(self, data, backcast_length, forecast_length, batch_size=256):
        super().__init__()
        self.data = data  # (n_samples, backcast + forecast)
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.batch_size = batch_size

    def setup(self, stage=None):
        n = len(self.data)
        indices = np.random.permutation(n)
        split = int(0.8 * n)

        train_data = self.data[indices[:split]]
        val_data = self.data[indices[split:]]

        self.train_dataset = _PairedDataset(
            train_data, self.backcast_length, self.forecast_length
        )
        self.val_dataset = _PairedDataset(
            val_data, self.backcast_length, self.forecast_length
        )

    def train_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )


class _PairedDataset(torch.utils.data.Dataset):
    """Dataset of (backcast, forecast) pairs from pre-split rows."""

    def __init__(self, data, backcast_length, forecast_length):
        self.data = torch.from_numpy(data).float()
        self.bc = backcast_length
        self.fc = forecast_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return row[: self.bc], row[self.bc : self.bc + self.fc]
