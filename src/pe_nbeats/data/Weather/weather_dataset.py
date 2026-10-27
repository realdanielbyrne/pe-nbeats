import numpy as np
import pandas as pd

from .._download import cached_file
from ..benchmark_dataset import BenchmarkDataset

# THUML Time-Series-Library dataset on Hugging Face (CC-BY-4.0).
# 21 meteorological indicators, ~52,696 timesteps (10-minute intervals).
_DOWNLOAD_URL = (
    "https://huggingface.co/datasets/thuml/Time-Series-Library/"
    "resolve/main/weather/weather.csv?download=true"
)


class WeatherDataset(BenchmarkDataset):
    """Weather benchmark dataset (21 meteorological indicators, 10-min intervals).

    Downloads the standard weather.csv from THUML/Hugging Face on first use
    and caches it at ~/.cache/pe_nbeats/Weather/weather.csv.

    Uses the standard LTSF chronological split (70 / 10 / 20) by default:
      - train_data: first 70% of rows
      - val_data:   next 10% of rows (dedicated validation set)
      - test_data:  final 20% of rows (full rolling-window evaluation)

    Parameters
    ----------
    horizon : int
        Forecast horizon (commonly 96, 192, 336, or 720).
    train_ratio : float
        Fraction of data used for training (default 0.7).
    val_ratio : float
        Fraction of data used for validation (default 0.1).  The remainder
        after ``train_ratio + val_ratio`` becomes the test set.
    include_target : bool
        If True, keep the OT column (21 columns total). Default False
        drops OT (20 columns), matching previous behavior.
    """

    supports_owa = False

    def __init__(self, horizon, train_ratio=0.7, val_ratio=0.1, include_target=False):
        self.horizon = horizon
        self.forecast_length = horizon
        self.frequency = 144  # 10-min intervals, daily seasonality = 6*24 = 144

        df = self._load_or_download()

        # Drop the date column — keep only numeric indicator columns
        if "date" in df.columns:
            df = df.drop(columns=["date"])
        # Drop the OT target column unless include_target is set
        if not include_target and "OT" in df.columns:
            df = df.drop(columns=["OT"])

        n_total = len(df)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # Chronological split: train | val | test
        self.train_data = df.iloc[:n_train].reset_index(drop=True)
        self.val_data = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
        self.test_data = df.iloc[n_train + n_val :].reset_index(drop=True)

    @property
    def name(self):
        return f"Weather-{self.horizon}"

    @staticmethod
    def _load_or_download():
        """Load cached CSV or download from Hugging Face."""
        return pd.read_csv(cached_file("Weather", "weather.csv", _DOWNLOAD_URL))
