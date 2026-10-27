from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    WeightedRandomSampler,
    random_split,
)
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl


class RowCollectionTimeSeriesDataset(Dataset):
    def __init__(self, data, backcast_length, forecast_length):
        """
        The RowCollectionTimeSeriesDataset class is a PyTorch Dataset that takes a
        collection of time series as input and returns a single sample of the time
        series. Used for training a time series model whose input is a collection
        of time series organized such that rows represent individual time series and
        columns give the subsequent observations. Each timeset does not have to be the
        the same length.

        Parameters
        ----------
          train_data (numpy.ndarray):
            The univariate time series data. The data organization is assumed to be a
            numpy.ndarray with rows representingtime series and columns representing time steps.
          backcast (int, optional):
            The length of the historical data.
          forecast (int, optional):
            The length of the future data to predict.
        """

        super(RowCollectionTimeSeriesDataset, self).__init__()
        self.data = data
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.items = []

        total_len = self.backcast_length + self.forecast_length
        for row in range(self.data.shape[0]):
            col_starts = np.arange(0, self.data.shape[1] - total_len + 1)
            seqs = [self.data[row, start : start + total_len] for start in col_starts]
            valid_indices = [i for i, seq in enumerate(seqs) if not np.isnan(seq).any()]
            self.items.extend([(row, col_starts[i]) for i in valid_indices])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        row, col = self.items[idx]
        x = self.data[row, col : col + self.backcast_length]
        y = self.data[
            row,
            col
            + self.backcast_length : col
            + self.backcast_length
            + self.forecast_length,
        ]

        return torch.from_numpy(x.copy()).float(), torch.from_numpy(y.copy()).float()


class RowCollectionTimeSeriesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data,
        backcast_length,
        forecast_length,
        batch_size=1024,
        split_ratio=0.8,
        fill_short_ts=True,
        num_workers=0,
        pin_memory=False,
    ):
        """The RowCollectionTimeSeriesDataModule class is a PyTorch Lightning DataModule
        is used for training a time series model with a dataset that is a collection of time series
        organized into rows where each row represents a time series, and each column represents
        subsequent observations.

        Parameters
        ----------
            data (pd.Dataframe):
              The univariate time series data. The data organization is assumed to be a
              pandas dataframe with rows representing time series and columns representing time steps.
            backcast_length (int, optional):
              The length of the historical data.
            forecast_length (int, optional):
              The length of the future data to predict.
            batch_size (int, optional):
              The batch size. Defaults to 1024.
            split_ratio (float, optional):
              The ratio of the data to use for training/validation.
            fill_short_ts (bool, optional):
              If True, short training sequences are filled. Defaults to True.
        """
        super(RowCollectionTimeSeriesDataModule, self).__init__()

        self.data = data
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.fill_short_ts = fill_short_ts
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str = None):
        shuffled = self.data.sample(frac=1, axis=0).reset_index(drop=True)
        total_len = self.backcast_length + self.forecast_length

        # Split the original data into training and validation sets
        self.train_data = shuffled.iloc[:, : -self.forecast_length].values.astype(
            np.float32
        )
        self.val_data = shuffled.iloc[
            :, -self.backcast_length - self.forecast_length :
        ].values.astype(np.float32)

        if self.fill_short_ts:
            for dataset in [self.train_data, self.val_data]:
                for row in range(dataset.shape[0]):
                    nan_indices = np.isnan(dataset[row])
                    row_length = np.sum(~nan_indices)
                    elements_to_add = total_len - row_length
                    if elements_to_add > 0:
                        fill_val = 0.0

                        imputed_values = np.full(elements_to_add, fill_val)
                        new_row = np.concatenate(
                            [imputed_values, dataset[row][~nan_indices]]
                        )
                        nan_padding = np.full(dataset.shape[1] - len(new_row), np.nan)
                        dataset[row] = np.concatenate([new_row, nan_padding])

        self.train_dataset = RowCollectionTimeSeriesDataset(
            self.train_data, self.backcast_length, self.forecast_length
        )
        self.val_dataset = RowCollectionTimeSeriesDataset(
            self.val_data, self.backcast_length, self.forecast_length
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )


class RowCollectionTimeSeriesTestModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data,
        test_data,
        backcast_length,
        forecast_length,
        batch_size=1024,
        fill_short_ts: bool = True,
        num_workers=0,
        pin_memory=False,
    ):
        """The RowCollectionTimeSeriesTestModule class is a PyTorch Lightning DataModule
        used for testing a time series model whose input is a collection of time series.
        The final `backcast` samples of each time series in `train_data` are concatenated
        with the first `forecast` samples of the corresponding time series in `test_data`.
        If a time series in `train_data` is shorter than `backcast`, the missing values are
        imputed with the median of the training row.

        Parameters
        ----------
          backcast_length (int, optional):
            The length of the historical data.
          forecast_length (int, optional):
            The length of the future data to predict.
          batch_size (int, optional):
            The batch size. Defaults to 1024.
        """

        super(RowCollectionTimeSeriesTestModule, self).__init__()
        self.train_data = train_data
        self.test_data_raw = test_data

        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.batch_size = batch_size
        self.fill_short_ts = fill_short_ts
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str = None):
        # Create test data by concatenating last `backcast` samples from
        # train_data and first `forecast` samples from test_data

        test_data_sequences = []
        total_len = self.backcast_length + self.forecast_length

        for train_row, test_row in zip(
            self.train_data.values, self.test_data_raw.values
        ):
            train_row = train_row[~np.isnan(train_row)]
            sequence = np.concatenate(
                (train_row[-self.backcast_length :], test_row[: self.forecast_length])
            )

            if self.fill_short_ts:
                nan_indices = np.isnan(sequence)
                seq_length = np.sum(~nan_indices)
                elements_to_add = total_len - seq_length

                if elements_to_add > 0:
                    fill_val = 0.0

                    # Create an array of imputed values
                    imputed_values = np.full(elements_to_add, fill_val)

                    # Insert the imputed values before the first backcast observation
                    sequence = np.concatenate([imputed_values, sequence])

            if sequence.shape[0] == self.backcast_length + self.forecast_length:
                test_data_sequences.append(sequence)

        self.test_data = np.array(test_data_sequences)
        self.test_dataset = RowCollectionTimeSeriesDataset(
            self.test_data, self.backcast_length, self.forecast_length
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )


class TimeSeriesDataset(Dataset):
    def __init__(self, data, backcast_length, forecast_length):
        """A simple PyTorch Dataset that takes a time series as input and returns a single sample of
        the time series. Used for training a simple time series model.

        Parameters
        ----------
            data (numpy.ndarray): The univariate time series data.
            backcast_length (int): The length of the historical data.
            forecast_length (int): The length of the future data to predict.
        """
        self.data = data
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.total_length = backcast_length + forecast_length

    def __len__(self):
        return len(self.data) - self.total_length + 1

    def __getitem__(self, index):
        start_idx = index
        end_idx = index + self.total_length
        x = self.data[start_idx : end_idx - self.forecast_length]
        y = self.data[start_idx + self.backcast_length : end_idx]
        return torch.from_numpy(x.copy()).float(), torch.from_numpy(y.copy()).float()


class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data,
        batch_size,
        backcast_length,
        forecast_length,
        num_workers=0,
        pin_memory=False,
    ):
        """
        The TimeSeriesDataModule class is a PyTorch Lightning DataModule that takes a univariate
        time series as input and returns batches of samples of the time series.

        Parameters
        ----------
            data (numpy.ndarray): The univariate time series data.
            batch_size (int): The batch size.
            backcast_length (int): The length of the historical data.
            forecast_length (int): The length of the future data to predict.
        """
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        dataset = TimeSeriesDataset(
            self.data, self.backcast_length, self.forecast_length
        )
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )


class ForecastingDataset(Dataset):
    def __init__(self, historical_data):
        """The ForecastingDataset class is a PyTorch Dataset that takes a historical
        time series as input and returns a single sample of the time series. Used for
        inferencing, predicting, for a time series model.

        Parameters
        ----------
            historical_data (pytorch.tensor): A single time series of historical data.
        """
        super().__init__()
        self.historical_data = historical_data

    def __len__(self):
        return len(self.historical_data)

    def __getitem__(self, idx):
        return self.historical_data[idx]


class ColumnarTimeSeriesDataset(Dataset):
    def __init__(
        self,
        dataframe,
        backcast_length,
        forecast_length,
        col_means=None,
        col_stds=None,
        normalization_style=None,
        return_scales=False,
        eps=1e-5,
        max_insample_length=None,
    ):
        """Sliding-window dataset over a columnar DataFrame.

        Parameters
        ----------
        normalization_style : {None, 'none', 'global', 'window'}, optional
            - None / 'none' : no normalization (unless ``col_means`` is given,
              which triggers legacy 'global' behavior for backward compat).
            - 'global' : apply pre-computed per-column z-score using
              ``col_means`` / ``col_stds`` once at dataset construction.
            - 'window' : compute per-window mean/std from the input ``x``
              inside ``__getitem__`` (RevIN-style), scale both ``x`` and
              ``y``. Incompatible with ``col_means``/``col_stds``.
        return_scales : bool, optional
            When True in 'window' mode, ``__getitem__`` returns
            ``(x_norm, y_norm, mu, sigma)`` so callers can denormalize
            predictions for test-time metrics. Defaults to False.
        eps : float, optional
            Stability term added to per-window std. Defaults to 1e-5.
        max_insample_length : int or None, optional
            When set, restricts valid training windows to only those whose
            start index falls within the last ``max_insample_length`` positions
            of each series. Implements the Lh insample constraint from the
            N-BEATS paper (Oreshkin et al., 2020): set to ``lh_multiplier *
            forecast_length`` (e.g. 7 * H) to match the paper's maximum
            ensemble lookback. Ignored when None (all valid windows used).
        """
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.min_length = backcast_length + forecast_length
        self.col_means = col_means
        self.col_stds = col_stds

        if normalization_style is None:
            normalization_style = "global" if col_means is not None else "none"
        if normalization_style not in ("none", "global", "window"):
            raise ValueError(
                f"normalization_style must be one of "
                f"{{'none','global','window'}}, got {normalization_style!r}"
            )
        if normalization_style == "window" and col_means is not None:
            raise ValueError(
                "normalization_style='window' is incompatible with precomputed "
                "col_means/col_stds; pass normalization_style='global' instead."
            )
        self.normalization_style = normalization_style
        self.return_scales = return_scales
        self.eps = float(eps)

        # Drop columns with insufficient data and convert to dictionary of NumPy arrays
        self.data_dict = {}
        for col in dataframe.columns:
            series = self.pad_series(dataframe[col].dropna().values).astype(np.float32)
            if (
                self.normalization_style == "global"
                and col_means is not None
                and col in col_means
            ):
                series = (series - col_means[col]) / col_stds[col]
            self.data_dict[col] = series

        # Precompute column indices and starting positions.
        # When max_insample_length is set (N-BEATS Lh constraint), restrict each
        # series to only windows starting within its last max_insample_length positions.
        self.col_indices = [
            (col, idx)
            for col, series in self.data_dict.items()
            for idx in range(
                max(0, len(series) - self.min_length + 1 - max_insample_length)
                if max_insample_length is not None
                else 0,
                len(series) - self.min_length + 1,
            )
        ]

    def pad_series(self, series):
        valid_entries = len(series)
        if valid_entries < self.min_length:
            # Calculate the number of zeros to pad
            pad_size = self.min_length - valid_entries
            # Create a padding array of zeros
            padding = np.zeros(pad_size)

            # Pad the series with zeros at the beginning
            series = np.concatenate((padding, series), axis=0)
        return series

    def __len__(self):
        return len(self.col_indices)

    def __getitem__(self, idx):
        col, start_idx = self.col_indices[idx]
        series = self.data_dict[col]
        x = series[start_idx : start_idx + self.backcast_length].copy()
        y = series[
            start_idx + self.backcast_length : start_idx + self.min_length
        ].copy()

        if self.normalization_style == "window":
            mu = float(np.mean(x))
            sigma = float(np.std(x))
            if sigma < self.eps:
                sigma = 1.0
            x = (x - mu) / sigma
            y = (y - mu) / sigma
            x_t = torch.from_numpy(x).float()
            y_t = torch.from_numpy(y).float()
            if self.return_scales:
                return (
                    x_t,
                    y_t,
                    torch.tensor(mu, dtype=torch.float32),
                    torch.tensor(sigma, dtype=torch.float32),
                )
            return x_t, y_t

        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


class ColumnarCollectionTimeSeriesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataframe,
        backcast_length,
        forecast_length,
        batch_size=1024,
        no_val=False,
        num_workers=0,
        pin_memory=False,
        normalize=False,
        normalization_style=None,
        val_ratio=None,
        val_data=None,
        sampling_style="sliding",
        steps_per_epoch=None,
        sampling_weights="uniform",
        lh_multiplier=None,
    ):
        """
        The ColumnarCollectionTimeSeriesDataModule class is a PyTorch Datamodule that takes a
        collection of time series as input and returns a single sample of the time
        series. The input dataset is a collection of time series organized such that columns
        represent individual time series and rows represent subsequent observations. This is how
        the Tourism dataset is organized.

        Args:
            dataframe (Pandas): Pandas dataframe with columns representing time series and rows representing time steps.
            backcast_length (int, optional):  Number of past observations to use for prediction. Defaults to 10.
            forecast_length (int, optional): Number of future observations to predict. Defaults to 4.
            batch_size (int, optional): The batch size. Defaults to 32.
            normalize (bool, optional): Legacy switch. If True and ``normalization_style``
                is None, falls back to ``normalization_style='global'``. Defaults to False.
            normalization_style ({None, 'none', 'global', 'window'}, optional):
                'global' applies per-column z-score using training-split statistics
                (legacy behavior). 'window' applies RevIN-style per-input-window
                z-score inside ``ColumnarTimeSeriesDataset.__getitem__``; train/val
                datasets skip global statistics in this mode. Defaults to None
                (resolved from ``normalize``).
            val_ratio (float, optional): Fraction of training data to use as validation set.
                When None (default), validation is the last backcast+forecast rows.
                Ignored when ``val_data`` is provided.
            val_data (pd.DataFrame, optional): Dedicated external validation DataFrame.
                When provided, ``val_ratio`` is ignored and ``dataframe`` is used
                entirely for training.
            sampling_style ({'sliding', 'nhits_paper', 'nbeats_paper'}, optional):
                Training-batch sampling protocol. ``'sliding'`` (default) shuffles
                the enumerated sliding-window dataset for one full pass per epoch.
                ``'nhits_paper'`` reproduces the NHiTS iteration-based uniform sampling
                protocol: each epoch draws ``steps_per_epoch * batch_size`` windows
                with replacement (no insample length constraint).
                ``'nbeats_paper'`` adds the N-BEATS Lh insample constraint on top of
                the same replacement-sampling protocol (requires ``lh_multiplier``).
                Affects ``train_dataloader()`` only; validation and test dataloaders
                always use dense sliding windows.
            steps_per_epoch (int, optional): Required when ``sampling_style`` is
                ``'nhits_paper'`` or ``'nbeats_paper'``.
                Number of gradient updates per Lightning epoch. Must be a positive int.
            sampling_weights ({'uniform', 'by_series'}, optional): Weighting scheme
                when ``sampling_style`` is ``'nhits_paper'`` or ``'nbeats_paper'``.
                ``'uniform'`` (default) draws uniformly over the enumerated window
                index list, which biases toward long series in proportion to their
                window count. ``'by_series'`` weights each index by
                ``1 / n_windows_in_its_column`` so every series is equally likely
                per step. Ignored when ``sampling_style='sliding'``.
            lh_multiplier (int or None, optional): Required when
                ``sampling_style='nbeats_paper'``. Restricts each series' valid
                training windows to only those starting within the last
                ``lh_multiplier * forecast_length`` timesteps of the series. Implements
                the Lh insample constraint from the N-BEATS paper (Oreshkin et al.,
                2020). Use ``lh_multiplier=7`` to match the paper's maximum ensemble
                lookback (Lh = 7H). Ignored for other sampling styles.
        """
        super().__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.batch_size = batch_size
        self.no_val = no_val
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.normalize = normalize

        # Resolve normalization_style. Explicit value wins; otherwise derive
        # from the legacy ``normalize`` bool for backward compatibility.
        if normalization_style is None:
            normalization_style = "global" if normalize else "none"
        if normalization_style not in ("none", "global", "window"):
            raise ValueError(
                f"normalization_style must be one of "
                f"{{'none','global','window'}}, got {normalization_style!r}"
            )
        self.normalization_style = normalization_style
        self.val_ratio = val_ratio
        self._external_val_data = val_data

        if sampling_style not in ("sliding", "nhits_paper", "nbeats_paper"):
            raise ValueError(
                f"sampling_style must be one of {{'sliding','nhits_paper','nbeats_paper'}}, "
                f"got {sampling_style!r}"
            )
        if sampling_weights not in ("uniform", "by_series"):
            raise ValueError(
                f"sampling_weights must be one of {{'uniform','by_series'}}, "
                f"got {sampling_weights!r}"
            )
        if sampling_style in ("nhits_paper", "nbeats_paper"):
            if steps_per_epoch is None:
                raise ValueError(
                    f"sampling_style={sampling_style!r} requires steps_per_epoch to be set."
                )
            if not isinstance(steps_per_epoch, int) or isinstance(
                steps_per_epoch, bool
            ):
                raise ValueError(
                    f"steps_per_epoch must be a positive int, got {steps_per_epoch!r}"
                )
            if steps_per_epoch <= 0:
                raise ValueError(
                    f"steps_per_epoch must be a positive int, got {steps_per_epoch!r}"
                )
        if sampling_style == "nbeats_paper":
            if lh_multiplier is None:
                raise ValueError(
                    "sampling_style='nbeats_paper' requires lh_multiplier to be set "
                    "(e.g. lh_multiplier=1.5 for Yearly/Quarterly, 10 for other M4 periods)."
                )
            if isinstance(lh_multiplier, bool) or not isinstance(lh_multiplier, (int, float)):
                raise ValueError(
                    f"lh_multiplier must be a positive number, got {lh_multiplier!r}"
                )
            if lh_multiplier <= 0:
                raise ValueError(
                    f"lh_multiplier must be a positive number, got {lh_multiplier!r}"
                )
        self.sampling_style = sampling_style
        self.steps_per_epoch = steps_per_epoch
        self.sampling_weights = sampling_weights
        self.lh_multiplier = lh_multiplier
        self._by_series_weights = None

        self.total_length = backcast_length + forecast_length
        self.dataframe = dataframe
        self.col_means = None
        self.col_stds = None

    def setup(self, stage=None):
        if self.no_val:
            train_data = self.dataframe
            val_data = pd.DataFrame()
        elif self._external_val_data is not None:
            # Dedicated external validation set — use all of dataframe for training
            train_data = self.dataframe
            val_data = self._external_val_data
        elif self.val_ratio is not None:
            n_total = len(self.dataframe)
            n_val = int(n_total * self.val_ratio)
            train_data = self.dataframe.iloc[:-n_val]
            val_data = self.dataframe.iloc[-n_val:]
        else:
            train_data = self.dataframe.iloc[: -self.forecast_length]
            val_data = self.dataframe.iloc[
                -self.forecast_length - self.backcast_length :
            ]

        # Compute per-column normalization stats from training split (global mode only)
        if self.normalization_style == "global":
            eps = 1e-8
            self.col_means = {}
            self.col_stds = {}
            for col in train_data.columns:
                vals = train_data[col].dropna().values.astype(np.float32)
                mu = float(np.mean(vals)) if len(vals) > 0 else 0.0
                sigma = float(np.std(vals)) if len(vals) > 0 else 1.0
                if sigma < eps:
                    sigma = 1.0
                self.col_means[col] = mu
                self.col_stds[col] = sigma

        self.val_dataset = ColumnarTimeSeriesDataset(
            val_data,
            self.backcast_length,
            self.forecast_length,
            col_means=self.col_means,
            col_stds=self.col_stds,
            normalization_style=self.normalization_style,
        )
        max_insample_length = (
            int(round(self.lh_multiplier * self.forecast_length))
            if self.sampling_style == "nbeats_paper"
            else None
        )
        self.train_dataset = ColumnarTimeSeriesDataset(
            train_data,
            self.backcast_length,
            self.forecast_length,
            col_means=self.col_means,
            col_stds=self.col_stds,
            normalization_style=self.normalization_style,
            max_insample_length=max_insample_length,
        )

        # Cache per-index weights for 'by_series' sampling so each series is
        # drawn with equal probability regardless of how many windows it offers.
        if self.sampling_style in ("nhits_paper", "nbeats_paper") and self.sampling_weights == "by_series":
            col_counts = {}
            for col, _ in self.train_dataset.col_indices:
                col_counts[col] = col_counts.get(col, 0) + 1
            self._by_series_weights = torch.tensor(
                [1.0 / col_counts[col] for col, _ in self.train_dataset.col_indices],
                dtype=torch.double,
            )
        else:
            self._by_series_weights = None

    def train_dataloader(self):
        if self.sampling_style == "sliding":
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.num_workers > 0,
            )

        # Both 'nhits_paper' and 'nbeats_paper' use replacement sampling with a fixed
        # step budget per epoch. For 'nbeats_paper', the Lh constraint has already
        # been applied to col_indices in setup(), so no dataloader changes are needed.
        num_samples = self.steps_per_epoch * self.batch_size
        if self.sampling_weights == "by_series":
            if self._by_series_weights is None:
                raise RuntimeError(
                    "by_series weights are unset; call setup() before train_dataloader()."
                )
            sampler = WeightedRandomSampler(
                self._by_series_weights,
                num_samples=num_samples,
                replacement=True,
            )
        else:
            sampler = RandomSampler(
                self.train_dataset,
                replacement=True,
                num_samples=num_samples,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )


class ColumnarCollectionTimeSeriesTestDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data,
        test_data,
        backcast_length,
        forecast_length,
        batch_size=1024,
        num_workers=0,
        pin_memory=False,
        col_means=None,
        col_stds=None,
        normalization_style=None,
        return_scales=False,
    ):
        """Takes two collections of time series organized into columns
        where each row represents a time step and each column represents
        an individual time series. The module combines the training data
        with the holdout data to create a single test dataset.

        Args:
            train_data (pd.Dataframe): The training data.
            test_data (pd.Dataframe): The holdout test data.
            backcast_length (_type_): The length of the historical data.
            forecast_length (_type_): The length of the future data to predict.
            batch_size (int, optional): The batch size. Defaults to 1024.
            col_means (dict, optional): Per-column means from training data for normalization.
            col_stds (dict, optional): Per-column stds from training data for normalization.
            normalization_style ({None, 'none', 'global', 'window'}, optional):
                'global' uses ``col_means``/``col_stds`` (legacy). 'window' applies
                RevIN-style per-input-window normalization. Defaults to None
                (resolved from ``col_means``: 'global' if present, else 'none').
            return_scales (bool, optional): In 'window' mode, yield
                ``(x, y, mu, sigma)`` from the test dataloader so callers can
                denormalize predictions before computing metrics. Defaults to False.
        """
        super(ColumnarCollectionTimeSeriesTestDataModule, self).__init__()

        if backcast_length > len(train_data):
            raise ValueError(
                f"backcast_length ({backcast_length}) cannot exceed training data length ({len(train_data)})"
            )

        if normalization_style is None:
            normalization_style = "global" if col_means is not None else "none"
        if normalization_style not in ("none", "global", "window"):
            raise ValueError(
                f"normalization_style must be one of "
                f"{{'none','global','window'}}, got {normalization_style!r}"
            )

        self.test_data = pd.concat(
            [train_data.iloc[-backcast_length:], test_data]
        ).reset_index(drop=True)
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.col_means = col_means if normalization_style == "global" else None
        self.col_stds = col_stds if normalization_style == "global" else None
        self.normalization_style = normalization_style
        self.return_scales = return_scales

    def setup(self, stage: str = None):
        self.test_dataset = ColumnarTimeSeriesDataset(
            self.test_data,
            self.backcast_length,
            self.forecast_length,
            col_means=self.col_means,
            col_stds=self.col_stds,
            normalization_style=self.normalization_style,
            return_scales=self.return_scales,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
