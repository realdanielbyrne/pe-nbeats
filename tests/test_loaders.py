"""Tests for data loaders — RowCollection setup, Columnar iloc, validation."""
import inspect
import math

import pytest
import numpy as np
import pandas as pd
import torch
from torch.utils.data import RandomSampler, WeightedRandomSampler

from pe_nbeats.loaders import (
    ColumnarCollectionTimeSeriesDataModule,
    ColumnarCollectionTimeSeriesTestDataModule,
    RowCollectionTimeSeriesDataModule,
)


BACKCAST_LENGTH = 8
FORECAST_LENGTH = 4


def _make_row_dataframe(n_series=10, n_timesteps=20):
    """Create a DataFrame where rows=series, cols=time steps."""
    data = np.random.rand(n_series, n_timesteps)
    return pd.DataFrame(data)


def _make_columnar_dataframes(n_timesteps=30, n_series=5, forecast_length=4):
    """Create train/test DataFrames where cols=series, rows=time steps."""
    train_data = pd.DataFrame(
        np.random.rand(n_timesteps, n_series),
        columns=[f"ts_{i}" for i in range(n_series)])
    test_data = pd.DataFrame(
        np.random.rand(forecast_length, n_series),
        columns=[f"ts_{i}" for i in range(n_series)])
    return train_data, test_data


# --- RowCollectionTimeSeriesDataModule tests ---

class TestRowCollectionSetup:
    """Verify RowCollectionTimeSeriesDataModule.setup() works correctly."""

    def test_setup_does_not_crash(self):
        df = _make_row_dataframe(n_series=10, n_timesteps=20)
        dm = RowCollectionTimeSeriesDataModule(
            data=df, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, batch_size=4)
        dm.setup()

    def test_setup_splits_time_dimension(self):
        n_timesteps = 20
        df = _make_row_dataframe(n_series=10, n_timesteps=n_timesteps)
        dm = RowCollectionTimeSeriesDataModule(
            data=df, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, batch_size=4)
        dm.setup()
        expected_train_cols = n_timesteps - FORECAST_LENGTH
        expected_val_cols = BACKCAST_LENGTH + FORECAST_LENGTH
        assert dm.train_data.shape[1] == expected_train_cols
        assert dm.val_data.shape[1] == expected_val_cols

    def test_setup_preserves_all_series(self):
        n_series = 15
        df = _make_row_dataframe(n_series=n_series, n_timesteps=20)
        dm = RowCollectionTimeSeriesDataModule(
            data=df, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, batch_size=4)
        dm.setup()
        assert dm.train_data.shape[0] == n_series
        assert dm.val_data.shape[0] == n_series

    def test_setup_returns_numpy_arrays(self):
        df = _make_row_dataframe(n_series=10, n_timesteps=20)
        dm = RowCollectionTimeSeriesDataModule(
            data=df, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, batch_size=4)
        dm.setup()
        assert isinstance(dm.train_data, np.ndarray)
        assert isinstance(dm.val_data, np.ndarray)

    def test_train_dataloader(self):
        df = _make_row_dataframe(n_series=10, n_timesteps=20)
        dm = RowCollectionTimeSeriesDataModule(
            data=df, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, batch_size=4)
        dm.setup()
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        x, y = batch
        assert x.shape[1] == BACKCAST_LENGTH
        assert y.shape[1] == FORECAST_LENGTH


# --- ColumnarCollectionTimeSeriesTestDataModule tests ---

class TestColumnarTestModule:
    """Verify ColumnarCollectionTimeSeriesTestDataModule iloc and validation."""

    def test_iloc_slicing_works(self):
        train_data, test_data = _make_columnar_dataframes(
            n_timesteps=30, n_series=5, forecast_length=FORECAST_LENGTH)
        dm = ColumnarCollectionTimeSeriesTestDataModule(
            train_data=train_data, test_data=test_data,
            backcast_length=BACKCAST_LENGTH, forecast_length=FORECAST_LENGTH)
        expected_rows = BACKCAST_LENGTH + FORECAST_LENGTH
        assert len(dm.test_data) == expected_rows

    def test_backcast_exceeds_train_length_raises(self):
        train_data, test_data = _make_columnar_dataframes(
            n_timesteps=5, n_series=3, forecast_length=FORECAST_LENGTH)
        with pytest.raises(ValueError, match="cannot exceed training data length"):
            ColumnarCollectionTimeSeriesTestDataModule(
                train_data=train_data, test_data=test_data,
                backcast_length=10, forecast_length=FORECAST_LENGTH)

    def test_non_default_index(self):
        train_data, test_data = _make_columnar_dataframes(
            n_timesteps=30, n_series=5, forecast_length=FORECAST_LENGTH)
        train_data.index = range(100, 130)
        test_data.index = range(200, 204)
        dm = ColumnarCollectionTimeSeriesTestDataModule(
            train_data=train_data, test_data=test_data,
            backcast_length=BACKCAST_LENGTH, forecast_length=FORECAST_LENGTH)
        expected_rows = BACKCAST_LENGTH + FORECAST_LENGTH
        assert len(dm.test_data) == expected_rows
        assert dm.test_data.index[0] == 0


# --- sampling_style='nhits_paper' tests ---

def _build_columnar_dm(
    n_timesteps=30,
    n_series=5,
    batch_size=4,
    forecast_length=FORECAST_LENGTH,
    backcast_length=BACKCAST_LENGTH,
    **kwargs,
):
    train_data, _ = _make_columnar_dataframes(
        n_timesteps=n_timesteps,
        n_series=n_series,
        forecast_length=forecast_length,
    )
    return ColumnarCollectionTimeSeriesDataModule(
        train_data,
        backcast_length=backcast_length,
        forecast_length=forecast_length,
        batch_size=batch_size,
        no_val=True,
        **kwargs,
    )


class TestSlidingDefault:
    """Regression: default sampling_style is 'sliding', behavior unchanged."""

    def test_default_sampling_style_is_sliding(self):
        dm = _build_columnar_dm()
        assert dm.sampling_style == "sliding"
        assert dm.steps_per_epoch is None
        assert dm.sampling_weights == "uniform"

    def test_sliding_train_dataloader_uses_shuffle_not_sampler(self):
        dm = _build_columnar_dm(batch_size=4)
        dm.setup()
        loader = dm.train_dataloader()
        # With shuffle=True and no explicit sampler, PyTorch assigns a
        # RandomSampler over the full dataset (len == full dataset).
        assert len(loader.sampler) == len(dm.train_dataset)
        expected_batches = math.ceil(len(dm.train_dataset) / 4)
        assert sum(1 for _ in loader) == expected_batches

    def test_sliding_epoch_covers_all_windows(self):
        dm = _build_columnar_dm(batch_size=4)
        dm.setup()
        loader = dm.train_dataloader()
        total = sum(batch[0].shape[0] for batch in loader)
        assert total == len(dm.train_dataset)


class TestPaperSamplingValidation:
    """Validation errors raised in __init__."""

    def test_paper_requires_steps_per_epoch(self):
        with pytest.raises(ValueError, match="steps_per_epoch"):
            _build_columnar_dm(sampling_style="nhits_paper")

    def test_rejects_invalid_style(self):
        with pytest.raises(ValueError, match="sampling_style"):
            _build_columnar_dm(sampling_style="random")

    def test_rejects_invalid_weights(self):
        with pytest.raises(ValueError, match="sampling_weights"):
            _build_columnar_dm(
                sampling_style="nhits_paper",
                steps_per_epoch=10,
                sampling_weights="foo",
            )

    def test_rejects_zero_steps_per_epoch(self):
        with pytest.raises(ValueError, match="steps_per_epoch"):
            _build_columnar_dm(sampling_style="nhits_paper", steps_per_epoch=0)

    def test_rejects_negative_steps_per_epoch(self):
        with pytest.raises(ValueError, match="steps_per_epoch"):
            _build_columnar_dm(sampling_style="nhits_paper", steps_per_epoch=-1)


class TestPaperSamplingBehavior:
    """Runtime behavior under sampling_style='nhits_paper'."""

    def test_uniform_sampler_num_samples(self):
        dm = _build_columnar_dm(
            batch_size=4,
            sampling_style="nhits_paper",
            steps_per_epoch=7,
        )
        dm.setup()
        loader = dm.train_dataloader()
        assert isinstance(loader.sampler, RandomSampler)
        assert loader.sampler.replacement is True
        assert len(loader.sampler) == 7 * 4

    def test_uniform_epoch_batch_count(self):
        dm = _build_columnar_dm(
            batch_size=4,
            sampling_style="nhits_paper",
            steps_per_epoch=5,
        )
        dm.setup()
        loader = dm.train_dataloader()
        assert sum(1 for _ in loader) == 5

    def test_uniform_replacement_distribution(self):
        # Tiny dataset + num_samples >> len(dataset) ensures collisions.
        torch.manual_seed(0)
        dm = _build_columnar_dm(
            n_timesteps=BACKCAST_LENGTH + FORECAST_LENGTH + 1,  # 2 windows per col
            n_series=2,
            batch_size=2,
            sampling_style="nhits_paper",
            steps_per_epoch=50,
        )
        dm.setup()
        loader = dm.train_dataloader()
        indices = list(iter(loader.sampler))
        assert len(indices) == 50 * 2
        # With 4 unique indices but 100 draws, at least one must repeat.
        counts = {i: indices.count(i) for i in set(indices)}
        assert max(counts.values()) > 1

    def test_by_series_sampler_type(self):
        dm = _build_columnar_dm(
            batch_size=4,
            sampling_style="nhits_paper",
            steps_per_epoch=10,
            sampling_weights="by_series",
        )
        dm.setup()
        loader = dm.train_dataloader()
        assert isinstance(loader.sampler, WeightedRandomSampler)
        # With uniform column lengths, weights sum ≈ n_columns.
        n_columns = len(dm.train_dataset.data_dict)
        assert dm._by_series_weights is not None
        assert abs(float(dm._by_series_weights.sum()) - n_columns) < 1e-6

    def test_by_series_equalizes_series_frequency(self):
        # Column A: 104 rows (100 windows). Column B: 14 rows (10 windows).
        backcast, forecast = BACKCAST_LENGTH, FORECAST_LENGTH  # 8+4=12 min_length
        long_len = 100 + backcast + forecast - 1 + 1  # 112
        short_len = 10 + backcast + forecast - 1 + 1  # 22
        df = pd.DataFrame({
            "A": np.concatenate([np.random.rand(long_len),
                                 np.full(long_len - short_len, np.nan)])[:long_len],
            # Ensure B has NaN tail so it drops to short_len observations.
            "B": np.concatenate([np.random.rand(short_len),
                                 np.full(long_len - short_len, np.nan)]),
        })
        dm = ColumnarCollectionTimeSeriesDataModule(
            df,
            backcast_length=backcast,
            forecast_length=forecast,
            batch_size=1,
            no_val=True,
            sampling_style="nhits_paper",
            steps_per_epoch=1000,
            sampling_weights="by_series",
        )
        dm.setup()
        # Sanity: the enumerated window counts per column must differ.
        col_counts = {}
        for col, _ in dm.train_dataset.col_indices:
            col_counts[col] = col_counts.get(col, 0) + 1
        assert col_counts["A"] > col_counts["B"]

        torch.manual_seed(0)
        loader = dm.train_dataloader()
        drawn_cols = [
            dm.train_dataset.col_indices[i][0] for i in iter(loader.sampler)
        ]
        a = drawn_cols.count("A")
        b = drawn_cols.count("B")
        # Target ratio is 1.0; with 1000 draws allow ±15%.
        ratio = a / b
        assert 0.85 <= ratio <= 1.15


class TestPaperSamplingValTestUnaffected:
    """Requirement 4: validation / test dataloaders stay dense sliding-window."""

    def test_paper_sampling_does_not_affect_val_dataloader(self):
        train_data, _ = _make_columnar_dataframes(n_timesteps=30, n_series=4)
        dm = ColumnarCollectionTimeSeriesDataModule(
            train_data,
            backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH,
            batch_size=4,
            sampling_style="nhits_paper",
            steps_per_epoch=3,
        )
        dm.setup()
        loader = dm.val_dataloader()
        n_val = len(dm.val_dataset)
        total = sum(batch[0].shape[0] for batch in loader)
        assert total == n_val  # no replacement, no truncation
        assert sum(1 for _ in loader) == math.ceil(n_val / 4)

    def test_columnar_test_datamodule_signature_unchanged(self):
        sig = inspect.signature(ColumnarCollectionTimeSeriesTestDataModule.__init__)
        assert "sampling_style" not in sig.parameters
        assert "steps_per_epoch" not in sig.parameters
        assert "sampling_weights" not in sig.parameters

