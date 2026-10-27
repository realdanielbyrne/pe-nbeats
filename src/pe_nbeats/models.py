import math
from numbers import Real
from time import time
import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    MultiStepLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)
from torch.nn import functional as F
import lightning.pytorch as pl

from .losses import *
from .blocks import blocks as b
from .constants import (
    BLOCKS,
    DEPRECATED_BLOCKS,
    LOSSES,
    OPTIMIZERS,
    get_deprecated_block_message,
)

# Generic wavelet base classes that accept wavelet_type / backcast_wavelet_type /
# forecast_wavelet_type kwargs.  Family-named subclasses (e.g. HaarWaveletV3)
# hardcode their wavelet family and do NOT accept these parameters.
_GENERIC_WAVELET_V3 = {
    "WaveletV3",
    "WaveletV3AE",
    "WaveletV3AELG",
    "WaveletV3VAE",
}


def _validate_stack_type(stack_type: str) -> None:
    if stack_type in DEPRECATED_BLOCKS:
        raise ValueError(get_deprecated_block_message(stack_type))
    if stack_type not in BLOCKS:
        raise ValueError(
            f"Unknown stack type: {stack_type}. Please select one of {BLOCKS}"
        )


def _validate_active_g_value(v) -> None:
    """Reject anything that isn't bool / 'backcast' / 'forecast'."""
    if isinstance(v, bool):
        return
    if isinstance(v, str) and v in ("backcast", "forecast"):
        return
    raise ValueError(
        f"active_g must be True, False, 'backcast', 'forecast', or a list of "
        f"those values (one per stack); got {v!r}"
    )


def _normalize_active_g(active_g, n_stacks: int) -> list:
    """Normalize ``active_g`` into a per-stack list of length ``n_stacks``.

    Scalars (bool / 'backcast' / 'forecast') are broadcast to every stack.
    A list input must have one entry per stack; each entry is validated
    independently and may differ from its neighbors.
    """
    if isinstance(active_g, list):
        if len(active_g) != n_stacks:
            raise ValueError(
                f"active_g list length {len(active_g)} must equal the number "
                f"of stacks ({n_stacks})."
            )
        for v in active_g:
            _validate_active_g_value(v)
        return list(active_g)
    _validate_active_g_value(active_g)
    return [active_g] * n_stacks


class _NBeatsBase(pl.LightningModule):
    """Shared training infrastructure for NBeatsNet and NHiTSNet.

    Provides configure_loss, configure_optimizers, and all Lightning step
    methods. Subclasses must implement forward() and set self.stacks before
    or during their own __init__.
    """

    def __init__(
        self,
        loss: str = "SMAPELoss",
        frequency: int = 1,
        no_val: bool = False,
        optimizer_name: str = "Adam",
        learning_rate: float = 1e-3,
        weight_decay: float = None,
        sum_losses: bool = False,
        kl_weight: float = 0.001,
        lr_scheduler_config: dict = None,
    ):
        super(_NBeatsBase, self).__init__()
        self.loss = loss
        self.frequency = frequency
        self.no_val = no_val
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.sum_losses = sum_losses
        self.kl_weight = kl_weight
        self.lr_scheduler_config = lr_scheduler_config
        self.loss_fn = self.configure_loss()

    def configure_loss(self):
        if self.loss not in LOSSES:
            raise ValueError(
                f"Unknown loss function name: {self.loss}. Please select one of {LOSSES}"
            )
        if self.loss == "MASELoss":
            return MASELoss(self.frequency)
        if self.loss == "MAPELoss":
            return MAPELoss()
        if self.loss == "SMAPELoss":
            return SMAPELoss()
        if self.loss == "NormalizedDeviationLoss":
            return NormalizedDeviationLoss()
        else:
            return getattr(nn, self.loss)()

    def configure_optimizers(self):
        if self.optimizer_name not in OPTIMIZERS:
            raise ValueError(
                f"Unknown optimizer name: {self.optimizer_name}. Please select one of {OPTIMIZERS}"
            )

        # weight_decay=None means "don't pass the kwarg", preserving each
        # optimizer's own default (Adam 0.0, AdamW 0.01, ...). An explicit
        # value — including 0.0 — is always forwarded.
        _opt_kwargs = {"lr": self.learning_rate}
        if self.weight_decay is not None:
            _opt_kwargs["weight_decay"] = self.weight_decay
        optimizer = getattr(optim, self.optimizer_name)(
            self.parameters(), **_opt_kwargs
        )

        if self.lr_scheduler_config is not None:
            cfg = self.lr_scheduler_config
            sched_type = str(cfg.get("type", "cosine")).lower()
            interval = cfg.get("interval", "epoch")

            if sched_type in ("cosine", "cosine_warmup", "sequential"):
                warmup_epochs = cfg.get("warmup_epochs", 15)
                t_max = cfg.get("T_max", 35)
                eta_min = cfg.get("eta_min", 1e-6)

                warmup_scheduler = ConstantLR(
                    optimizer, factor=1.0, total_iters=warmup_epochs
                )
                cosine_scheduler = CosineAnnealingLR(
                    optimizer, T_max=t_max, eta_min=eta_min
                )
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs],
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": interval},
                }
            elif sched_type == "plateau":
                monitor = str(cfg.get("monitor", "val_loss"))
                factor = float(cfg.get("factor", 0.5))
                plateau_patience = int(cfg.get("patience", 10))
                min_lr = float(cfg.get("min_lr", cfg.get("eta_min", 1e-5)))
                mode = str(cfg.get("mode", "min"))
                cooldown = int(cfg.get("cooldown", 0))
                frequency = int(cfg.get("frequency", 1))
                scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode=mode,
                    factor=factor,
                    patience=plateau_patience,
                    min_lr=min_lr,
                    cooldown=cooldown,
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": monitor,
                        "interval": interval,
                        "frequency": frequency,
                    },
                }
            elif sched_type == "step":
                step_size = cfg.get("step_size")
                if step_size is None:
                    raise ValueError(
                        "lr_scheduler_config={'type': 'step'} requires 'step_size'."
                    )
                gamma = cfg.get("gamma", 0.5)
                scheduler = StepLR(
                    optimizer, step_size=int(step_size), gamma=float(gamma)
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": interval},
                }
            elif sched_type == "multistep":
                milestones = cfg.get("milestones")
                if not milestones:
                    raise ValueError(
                        "lr_scheduler_config={'type': 'multistep'} requires a 'milestones' list."
                    )
                gamma = float(cfg.get("gamma", 0.5))
                scheduler = MultiStepLR(
                    optimizer, milestones=[int(m) for m in milestones], gamma=gamma
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": interval},
                }
            else:
                raise ValueError(
                    f"Unknown lr_scheduler_config type {sched_type!r}; "
                    "expected one of {{'cosine', 'plateau', 'step', 'multistep'}}."
                )

        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        backcast, forecast = self(x)
        loss = self.loss_fn(forecast, y)

        if self.sum_losses:
            backcast_loss = self.loss_fn(backcast, torch.zeros_like(backcast))
            loss = loss + backcast_loss * 0.25

        # Collect KL divergence loss from any VAE blocks
        kl_loss = loss.new_tensor(0.0)
        for stack in self.stacks:
            for block in stack:
                if isinstance(block, (b.AERootBlockVAE, b.VAE)):
                    kl_loss = kl_loss + block.kl_loss
        loss = loss + self.kl_weight * kl_loss

        self.log("train_loss", loss, prog_bar=True)
        # LR logging requires an attached Trainer; skip when the step is
        # invoked directly (e.g. in unit tests).
        if getattr(self, "_trainer", None) is not None and self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("lr", lr, prog_bar=False)
            self.log("lr_1e3", lr * 1000, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.no_val:
            return None
        x, y = batch
        backcast, forecast = self(x)
        # val_loss is the forecast loss alone, so early stopping, checkpointing
        # and reported numbers mean the same thing whether or not sum_losses is
        # on. The backcast penalty is a training regulariser; the combined
        # objective is logged separately for monitoring.
        loss = self.loss_fn(forecast, y)
        if self.sum_losses:
            backcast_loss = self.loss_fn(backcast, torch.zeros_like(backcast))
            self.log("val_total_loss", loss + backcast_loss * 0.25)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        backcast, forecast = self(x)
        loss = self.loss_fn(forecast, y)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        self.eval()
        _, forecast = self(batch)
        return forecast.detach().cpu().numpy()


class NBeatsNet(_NBeatsBase):
    SEASONALITY = "seasonality"
    TREND = "trend"
    GENERIC = "generic"
    AE = "ae"

    def __init__(
        self,
        backcast_length: int,
        forecast_length: int,
        stack_types: list = None,
        n_blocks_per_stack: int = 1,
        g_width: int = 512,
        s_width: int = 2048,
        t_width: int = 256,
        ae_width: int = 512,
        share_weights: bool = False,
        thetas_dim: int = 5,
        learning_rate: float = 1e-3,
        weight_decay: float = None,
        loss: str = "SMAPELoss",
        no_val: bool = False,
        optimizer_name: str = "Adam",  # 'Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'AdamW'
        activation: str = "ReLU",  # 'ReLU', 'RReLU', 'PReLU', 'ELU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'Sigmoid', 'GELU'
        frequency: int = 1,
        active_g=False,
        latent_dim: int = 5,
        latent_gate_fn: str = "sigmoid",
        ae_intermediate_width: int = None,
        sum_losses: bool = False,
        kl_weight: float = 0.001,
        basis_dim: int = 32,
        basis_offset: int = 0,
        stack_basis_offsets: list = None,
        stack_basis_dims: list = None,
        forecast_basis_dim: int = None,
        theta_activation: bool = False,
        wavelet_type: str = "db3",
        backcast_wavelet_type: str = None,
        forecast_wavelet_type: str = None,
        trend_thetas_dim: int | None = 3,
        generic_dim: int = 5,
        lr_scheduler_config: dict = None,
        skip_distance: int = 0,
        skip_alpha: float | str = 0.1,
    ):
        """A PyTorch Lightning module for the N-BEATS network for time series forecasting.

        N-Beats is based on the idea of neural basis expansion, where the input time series
        is decomposed into a linear combination of basis functions learned by the network.
        The network consists of multiple stacks of blocks, each block containing a fully
        connected layer followed by an activation function and a linear layer. The output
        of each block is added to the input of the next block (backward residual
        connection) and also to the final output of the stack (forward residual
        connection). The final output of the stack is split into two parts: the backcast,
        which is used to reconstruct the input time series, and the forecast, which is used
        to predict the future values.

        Based on the paper by Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2019).
        N-BEATS: Neural basis expansion analysis for interpretable time series forecasting.
        arXiv preprint arXiv:1905.10437. https://openreview.net/forum?id=r1ecqn4YwB

        Attributes
        ----------
        SEASONALITY : str
            Constant for seasonality stack type.
        TREND : str
            Constant for trend stack type.
        GENERIC : str
            Constant for generic stack type.

        Parameters
        ----------
        backcast_length : int, optional
            The length of the historical data.  It is customary to use a multiple of the
            forecast (H)orizon (2H,3H,4H,5H,...).
        forecast_length : int, optional
            The length of the forecast horizon.
        generic_architecture : bool, optional
            If True, use the generic architecture, otherwise use the interpretable
            architecture.
            Default True.
        n_blocks_per_stack : int, optional
            The number of blocks per stack.
            Default 1.
        n_stacks : int, optional
            The number of stacks, by default 5 when Generic, else fixed at 2  (1 Seasonal
            + 1 Trend) when interpretable.
        g_width, s_width, t_width : int, optional
            The width of the fully connected layers in a Generic block(g), Seasonal
            Block(s), or Trend Block(t).
            Default (g = 512, s = 2048, t = 256).
        share_weights : bool, optional
            If True, share initial weights across blocks.
        thetas_dim : int, optional
            The dimensionality of the waveform generator parameters in the Generic and
            Trend blocks.
        learning_rate : float, optional
            The learning rate for the optimizer, by default 1e-5.
        weight_decay : float, optional
            L2 weight-decay coefficient passed to the optimizer constructor.
            Default None: the kwarg is not passed at all, so each optimizer
            keeps its own default (Adam 0.0 — paper-faithful — AdamW 0.01).
            An explicit value, including 0.0, is always forwarded.
        loss : str, optional
            The loss function to use, defined in LOSSES.
        no_val : bool, optional
            If True, skip validation during training.
            Default False.
        optimizer_name : str, optional
            The name of the optimizer to use. Allowed methDefined in OPTIMIZERS.
            Default 'Adam'.
        activation : str, optional
            The activation function to use.  Defined in ACTIVATIONS.
            Default : 'ReLU'.
        frequency : int, optional
            The frequency of the data.  Used only when MASELoss is used as teh loss funtion. Default 1.
        active_g : bool, str, or list, optional
            Controls whether activation is applied after basis expansion.
            False: no activation (paper-faithful). True: activation on both backcast and forecast.
            'backcast': activation on backcast path only. 'forecast': activation on forecast path only.
            Also accepts a list of length ``len(stack_types)`` with an independent
            per-stack value (e.g. ``[True, True, False]`` to leave the last
            stack's output unclamped). Default : False.
        sum_losses : bool, optional
            If True, the total loss is defined as forecast_loss + 1/4 Backcast_loss.  This is an experimental feature. Default False.
        kl_weight : float, optional
            Multiplier applied to the summed KL divergence from VAE-family blocks
            during training. Set to 0.0 to disable KL regularization. Default 0.1.
        latent_dim : int, optional
            The dimensionality of the latent space in the AutoEncoder blocks. Default 5.
        latent_gate_fn : str, optional
            Latent gate function for AELG blocks. One of ``'sigmoid'``,
            ``'wavy_sigmoid'``, or ``'wavelet_sigmoid'``. Default ``'sigmoid'``.
        ae_intermediate_width : int, optional
            Width of the intermediate (taper) layers in AE-backboned blocks
            (encoder units -> ae_intermediate_width -> latent_dim, decoder
            mirrored). Default None (uses units // 2, the historical shape).
        basis_dim : int, optional
            The dimensionality of the basis space in the Wavelet blocks. Default 32.
        basis_offset : int, optional
            Row offset into the SVD-ordered basis for WaveletV3 blocks, selecting which
            frequency band is used (0 = lowest/smoothest, higher values shift toward
            higher frequencies). Default 0.
        stack_basis_offsets : list of int, optional
            Per-stack basis offsets for WaveletV3 blocks. If provided, must be the same
            length as stack_types. stack_basis_offsets[i] overrides basis_offset for
            stack i. Non-wavelet stacks receive the offset value but ignore it.
            Default None (all stacks use basis_offset).
        forecast_basis_dim : int, optional
            Override basis dimensionality for the forecast path of WaveletV3 blocks.
            When set, the forecast linear projection uses this value (clamped to
            forecast_length) while the backcast path continues to use basis_dim.
            Allows asymmetric regularization when backcast and forecast lengths differ.
            Default None (both paths use basis_dim).
        trend_thetas_dim : int or None, optional
            The polynomial degree for Trend and TrendAE blocks, independent of the
            global ``thetas_dim`` used by other block types.  Any positive integer
            is accepted (e.g. 2 = linear, 3 = cubic, 5 = degree-4 polynomial).
            Default 3. If None, falls back to ``thetas_dim``.
        lr_scheduler_config : dict, optional
            Learning-rate scheduler configuration. The ``type`` key (default
            ``'cosine'``) selects the scheduler family:

            - ``'cosine'``: ConstantLR warmup + CosineAnnealingLR decay wrapped
              in a SequentialLR. Keys: ``warmup_epochs`` (int, default 15),
              ``T_max`` (int, default 35), ``eta_min`` (float, default 1e-6).
            - ``'step'``: ``torch.optim.lr_scheduler.StepLR``. Required key:
              ``step_size`` (int). Optional: ``gamma`` (float, default 0.5).
            - ``'multistep'``: ``torch.optim.lr_scheduler.MultiStepLR``.
              Required key: ``milestones`` (list[int], pre-computed by
              ``resolve_lr_scheduler``). Optional: ``gamma`` (float, default
              0.5). Paper-faithful: use ``n_milestones=10`` in YAML and set
              early-stopping ``patience > max_epochs / n_milestones``.

            The optional ``interval`` key (``'epoch'`` or ``'step'``, default
            ``'epoch'``) controls when Lightning steps the scheduler. Default
            None (no scheduler, constant LR).

        Inputs
        ------
        stack_input of shape `(batch_size, input_chunk_length)`
            Tensor containing the input sequence.

        Outputs
        -------
        stack_residual of shape `(batch_size, input_chunk_length)`
            Tensor containing the 'backcast' of the block, which represents an approximation of `x`
            given the constraints of the functional space determined by `g`.
        stack_forecast of shape `(batch_size, output_chunk_length)`
            Tensor containing the forward forecast of the stacks.
        """

        # Fail fast on obviously invalid scalar strings; full per-stack
        # validation (including list form) runs below once n_stacks is known.
        if isinstance(active_g, str) and active_g not in ("backcast", "forecast"):
            raise ValueError(
                f"active_g must be True, False, 'backcast', or 'forecast', got '{active_g}'"
            )
        if latent_gate_fn not in b.LATENT_GATE_FNS:
            raise ValueError(
                f"latent_gate_fn must be one of {sorted(b.LATENT_GATE_FNS)}, got '{latent_gate_fn}'"
            )
        if trend_thetas_dim is not None and (
            not isinstance(trend_thetas_dim, int) or trend_thetas_dim < 1
        ):
            raise ValueError(
                f"trend_thetas_dim must be a positive integer, got {trend_thetas_dim}"
            )
        if not isinstance(skip_distance, int) or skip_distance < 0:
            raise ValueError(
                f"skip_distance must be a non-negative integer, got {skip_distance}"
            )
        if isinstance(skip_alpha, str):
            if skip_alpha != "learnable":
                raise ValueError(
                    f"skip_alpha must be a float or 'learnable', got '{skip_alpha}'"
                )
        elif not isinstance(skip_alpha, (int, float)) or isinstance(skip_alpha, bool):
            raise ValueError(
                f"skip_alpha must be a float or 'learnable', got {skip_alpha!r}"
            )
        if (
            not isinstance(kl_weight, Real)
            or isinstance(kl_weight, bool)
            or kl_weight < 0
        ):
            raise ValueError(
                f"kl_weight must be a non-negative real number, got {kl_weight!r}"
            )

        super(NBeatsNet, self).__init__(
            loss=loss,
            frequency=frequency,
            no_val=no_val,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            sum_losses=sum_losses,
            kl_weight=kl_weight,
            lr_scheduler_config=lr_scheduler_config,
        )

        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.n_blocks_per_stack = n_blocks_per_stack
        self.share_weights = share_weights
        self.g_width = g_width
        self.s_width = s_width
        self.t_width = t_width
        self.ae_width = ae_width
        self.thetas_dim = thetas_dim
        self.activation = activation
        self.active_g = active_g
        self.latent_dim = latent_dim
        self.latent_gate_fn = latent_gate_fn
        self.ae_intermediate_width = ae_intermediate_width
        self.basis_dim = basis_dim
        self.basis_offset = basis_offset
        self.stack_basis_offsets = stack_basis_offsets
        self.stack_basis_dims = stack_basis_dims
        self.forecast_basis_dim = forecast_basis_dim
        self.theta_activation = theta_activation
        self.wavelet_type = wavelet_type
        self.backcast_wavelet_type = backcast_wavelet_type
        self.forecast_wavelet_type = forecast_wavelet_type
        self.trend_thetas_dim = (
            self.thetas_dim if trend_thetas_dim is None else trend_thetas_dim
        )
        self.generic_dim = generic_dim
        self.skip_distance = skip_distance
        self.skip_alpha = skip_alpha

        self.save_hyperparameters()

        if (
            skip_distance > 0
            and isinstance(skip_alpha, str)
            and skip_alpha == "learnable"
        ):
            self._skip_alpha_learnable = True
            self.skip_alpha_param = nn.Parameter(torch.tensor(0.01))
        else:
            self._skip_alpha_learnable = False

        if stack_types is not None:
            self.stack_types = stack_types
            self.n_stacks = len(stack_types)
        else:
            raise ValueError("Stack architecture must be specified.")

        # Normalize active_g into a per-stack list; create_stack() reads from
        # self._active_g_per_stack[stack_idx] so each stack can be configured
        # independently while self.active_g preserves the original user input.
        self._active_g_per_stack = _normalize_active_g(active_g, self.n_stacks)

        self.stacks = nn.ModuleList()
        for stack_idx, stack_type in enumerate(self.stack_types):
            self.stacks.append(self.create_stack(stack_type, stack_idx))

    @classmethod
    def slim(
        cls,
        backcast_length: int,
        forecast_length: int,
        trend_branch: bool = False,
        rectify: bool = False,
        width: int = 128,
        n_stacks: int = 10,
        basis: str = "dct",
        **kwargs,
    ):
        """Right-sized N-BEATS: narrow trunk, frozen orthonormal basis, 10 stacks.

        Most of the paper configuration's parameters sit in trunk width, which
        was accuracy-neutral on M4 and Tourism between 128 and 2048. This preset
        keeps the doubly residual architecture and changes only the settings
        that were measured to matter:

        * ``width`` sets every block's hidden width (default 128). Wider trunks
          did help on Traffic-96; try 128 and 512 on a held-out split when the
          target is long-horizon and incompressible.
        * Each block projects its forecast through a frozen ``basis`` at
          complete rank (``forecast_basis_dim = forecast_length``), so the
          basis choice cannot limit what is representable. The backcast basis
          is truncated to ``forecast_length`` rows. DCT is the default because
          it is defined at every horizon; wavelets degenerate silently at short
          horizons.
        * ``trend_branch=True`` alternates ``Trend`` and basis blocks (the
          analogue of N-BEATS-I+G). Use it when the target's effective rank is
          low or the window mean dominates; otherwise it costs accuracy.
        * ``rectify=True`` sets ``active_g="forecast"``. Only use it for
          non-negative targets.

        :func:`pe_nbeats.preflight.recommend` measures the training windows and
        decides ``trend_branch`` and ``rectify``; see :meth:`from_recommendation`.

        Any other ``NBeatsNet`` keyword (``loss``, ``learning_rate``,
        ``lr_scheduler_config``, ...) is passed through ``kwargs``.
        """
        if trend_branch:
            if n_stacks % 2:
                raise ValueError("trend_branch=True needs an even n_stacks")
            stack_types = ["Trend", "WaveletV3"] * (n_stacks // 2)
        else:
            stack_types = ["WaveletV3"] * n_stacks
        params = dict(
            g_width=width,
            t_width=width,
            share_weights=True,
            wavelet_type=basis,
            basis_dim=forecast_length,
            forecast_basis_dim=forecast_length,
            trend_thetas_dim=3,
            active_g="forecast" if rectify else False,
        )
        params.update(kwargs)
        return cls(
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            stack_types=stack_types,
            **params,
        )

    @classmethod
    def from_recommendation(cls, rec, backcast_length: int, **kwargs):
        """Build :meth:`slim` from a :class:`pe_nbeats.preflight.Recommendation`."""
        params = dict(
            trend_branch=rec.needs_trend_branch,
            rectify=rec.needs_rectifier,
            width=rec.recommended_trunk_width,
            basis=rec.recommended_basis,
        )
        params.update(kwargs)
        return cls.slim(backcast_length, rec.forecast_length, **params)

    def create_stack(self, stack_type, stack_idx=0):

        blocks = nn.ModuleList()
        _validate_stack_type(stack_type)

        if self.stack_basis_offsets and stack_idx < len(self.stack_basis_offsets):
            effective_offset = self.stack_basis_offsets[stack_idx]
        else:
            effective_offset = self.basis_offset

        # Per-stack basis RANK, the nested counterpart of stack_basis_offsets.
        # An offset selects rows [o : full_rank], which discards the LEADING rows.
        # That is survivable for a basis whose ordering is not aligned with where
        # the data's energy sits, and ruinous for one that is: on M4-Hourly, rows
        # [8:48] of an SVD-ordered db3 basis still capture 0.895 of the target
        # energy, while the same window of a DCT captures 0.0003, because the DCT
        # keeps its constant row first and the offset throws it away. Varying the
        # rank instead selects rows [0 : d_k], so every stack retains the leading
        # rows and only its capacity differs. That is the form of tiering an
        # energy-ordered basis can actually use.
        if self.stack_basis_dims and stack_idx < len(self.stack_basis_dims):
            effective_basis_dim = self.stack_basis_dims[stack_idx]
        else:
            effective_basis_dim = self.basis_dim

        # AE-backbone taper width. Forwarded only when explicitly set so the
        # family-named compatibility wrappers (fixed signatures) are untouched
        # and default behavior (units // 2) is preserved exactly.
        _ae_iw_kwargs = (
            {}
            if self.ae_intermediate_width is None
            else {"ae_intermediate_width": self.ae_intermediate_width}
        )

        for block_id in range(self.n_blocks_per_stack):
            if self.share_weights and block_id != 0:
                # share weights across blocks
                block = blocks[-1]
            else:
                if stack_type in [
                    "Generic",
                    "NHiTSGenericRoot",
                    "BottleneckGeneric",
                    "GenericAE",
                    "BottleneckGenericAE",
                    "GenericAEBackcast",
                    "GenericAEBackcastAE",
                    "GenericAELG",
                    "BottleneckGenericAELG",
                    "GenericAEBackcastAELG",
                    "GenericVAE",
                    "BottleneckGenericVAE",
                    "GenericAEBackcastVAE",
                ]:
                    units = self.g_width
                elif stack_type in [
                    "Seasonality",
                    "SeasonalityAE",
                    "SeasonalityAELG",
                    "SeasonalityVAE",
                ]:
                    units = self.s_width
                elif stack_type in [
                    "Trend",
                    "TrendAE",
                    "TrendWavelet",
                    "TrendWaveletAE",
                    "TrendWaveletAELG",
                    "TrendAELG",
                    "TrendVAE",
                    "TrendWaveletGeneric",
                    "TrendWaveletGenericAE",
                    "TrendWaveletGenericAELG",
                    "TrendWaveletGenericVAE",
                ]:
                    units = self.t_width
                elif stack_type in [
                    "AutoEncoder",
                    "AutoEncoderAE",
                    "AutoEncoderAELG",
                    "AutoEncoderVAE",
                    "VAE",
                ]:
                    units = self.ae_width
                else:
                    units = self.g_width

                # Use trend_thetas_dim for Trend/TrendAE when set, else global thetas_dim
                if stack_type in (
                    "Trend",
                    "TrendAE",
                    "TrendWavelet",
                    "TrendAELG",
                    "TrendVAE",
                    "TrendWaveletGeneric",
                    "TrendWaveletGenericAE",
                    "TrendWaveletGenericAELG",
                    "TrendWaveletGenericVAE",
                ):
                    effective_td = self.trend_thetas_dim
                else:
                    effective_td = self.thetas_dim

                ae_latent_blocks = (
                    "GenericAE",
                    "BottleneckGenericAE",
                    "SeasonalityAE",
                    "AutoEncoderAE",
                    "TrendAE",
                    "GenericAEBackcastAE",
                    "GenericAELG",
                    "BottleneckGenericAELG",
                    "SeasonalityAELG",
                    "AutoEncoderAELG",
                    "TrendAELG",
                    "GenericAEBackcastAELG",
                    "GenericVAE",
                    "BottleneckGenericVAE",
                    "SeasonalityVAE",
                    "AutoEncoderVAE",
                    "TrendVAE",
                    "GenericAEBackcastVAE",
                )
                if stack_type == "TrendWavelet":
                    block = getattr(b, stack_type)(
                        units,
                        self.backcast_length,
                        self.forecast_length,
                        trend_dim=self.trend_thetas_dim,
                        wavelet_dim=effective_basis_dim,
                        basis_offset=effective_offset,
                        share_weights=self.share_weights,
                        activation=self.activation,
                        active_g=self._active_g_per_stack[stack_idx],
                        forecast_basis_dim=self.forecast_basis_dim,
                        wavelet_type=self.wavelet_type,
                        backcast_wavelet_type=self.backcast_wavelet_type,
                        forecast_wavelet_type=self.forecast_wavelet_type,
                    )
                elif stack_type == "TrendWaveletGeneric":
                    block = getattr(b, stack_type)(
                        units,
                        self.backcast_length,
                        self.forecast_length,
                        trend_dim=self.trend_thetas_dim,
                        wavelet_dim=effective_basis_dim,
                        generic_dim=self.generic_dim,
                        basis_offset=effective_offset,
                        share_weights=self.share_weights,
                        activation=self.activation,
                        active_g=self._active_g_per_stack[stack_idx],
                        forecast_basis_dim=self.forecast_basis_dim,
                        wavelet_type=self.wavelet_type,
                        backcast_wavelet_type=self.backcast_wavelet_type,
                        forecast_wavelet_type=self.forecast_wavelet_type,
                    )
                elif stack_type in ("TrendWaveletAE", "TrendWaveletAELG"):
                    block_kwargs = dict(
                        trend_dim=self.trend_thetas_dim,
                        wavelet_dim=effective_basis_dim,
                        basis_offset=effective_offset,
                        share_weights=self.share_weights,
                        activation=self.activation,
                        active_g=self._active_g_per_stack[stack_idx],
                        latent_dim=self.latent_dim,
                        forecast_basis_dim=self.forecast_basis_dim,
                        wavelet_type=self.wavelet_type,
                        backcast_wavelet_type=self.backcast_wavelet_type,
                        forecast_wavelet_type=self.forecast_wavelet_type,
                    )
                    if "AELG" in stack_type:
                        block_kwargs["latent_gate_fn"] = self.latent_gate_fn
                    block_kwargs.update(_ae_iw_kwargs)
                    block = getattr(b, stack_type)(
                        units,
                        self.backcast_length,
                        self.forecast_length,
                        **block_kwargs,
                    )
                elif stack_type in (
                    "TrendWaveletGenericAE",
                    "TrendWaveletGenericAELG",
                    "TrendWaveletGenericVAE",
                ):
                    block_kwargs = dict(
                        trend_dim=self.trend_thetas_dim,
                        wavelet_dim=effective_basis_dim,
                        generic_dim=self.generic_dim,
                        basis_offset=effective_offset,
                        share_weights=self.share_weights,
                        activation=self.activation,
                        active_g=self._active_g_per_stack[stack_idx],
                        latent_dim=self.latent_dim,
                        forecast_basis_dim=self.forecast_basis_dim,
                        wavelet_type=self.wavelet_type,
                        backcast_wavelet_type=self.backcast_wavelet_type,
                        forecast_wavelet_type=self.forecast_wavelet_type,
                    )
                    if "AELG" in stack_type:
                        block_kwargs["latent_gate_fn"] = self.latent_gate_fn
                    block_kwargs.update(_ae_iw_kwargs)
                    block = getattr(b, stack_type)(
                        units,
                        self.backcast_length,
                        self.forecast_length,
                        **block_kwargs,
                    )
                elif stack_type in ae_latent_blocks:
                    block_args = [
                        units,
                        self.backcast_length,
                        self.forecast_length,
                        effective_td,
                        self.share_weights,
                        self.activation,
                        self._active_g_per_stack[stack_idx],
                        self.latent_dim,
                    ]
                    if "AELG" in stack_type:
                        block = getattr(b, stack_type)(
                            *block_args, self.latent_gate_fn, **_ae_iw_kwargs
                        )
                    else:
                        block = getattr(b, stack_type)(*block_args, **_ae_iw_kwargs)
                elif "Wavelet" in stack_type:
                    # Only generic base classes accept wavelet_type/backcast/forecast overrides;
                    # family-named subclasses (e.g. HaarWaveletV3) hardcode their wavelet family.
                    _wavelet_kwargs = {}
                    if stack_type in _GENERIC_WAVELET_V3:
                        _wavelet_kwargs = dict(
                            wavelet_type=self.wavelet_type,
                            backcast_wavelet_type=self.backcast_wavelet_type,
                            forecast_wavelet_type=self.forecast_wavelet_type,
                        )
                        if stack_type == "WaveletV3":
                            # Diagnostic probe for the completeness argument;
                            # only the plain WaveletV3 base class accepts it.
                            _wavelet_kwargs["theta_activation"] = self.theta_activation
                        else:
                            # AE-backed generic wavelet classes take the taper width.
                            _wavelet_kwargs.update(_ae_iw_kwargs)
                    if any(token in stack_type for token in ("V3VAE", "V3AE")):
                        if "AELG" in stack_type:
                            _wavelet_kwargs["latent_gate_fn"] = self.latent_gate_fn
                        block = getattr(b, stack_type)(
                            units,
                            self.backcast_length,
                            self.forecast_length,
                            effective_basis_dim,
                            effective_offset,
                            self.share_weights,
                            self.activation,
                            self._active_g_per_stack[stack_idx],
                            forecast_basis_dim=self.forecast_basis_dim,
                            latent_dim=self.latent_dim,
                            **_wavelet_kwargs,
                        )
                    elif "V3" in stack_type:
                        block = getattr(b, stack_type)(
                            units,
                            self.backcast_length,
                            self.forecast_length,
                            effective_basis_dim,
                            effective_offset,
                            self.share_weights,
                            self.activation,
                            self._active_g_per_stack[stack_idx],
                            forecast_basis_dim=self.forecast_basis_dim,
                            **_wavelet_kwargs,
                        )
                    else:
                        block = getattr(b, stack_type)(
                            units,
                            self.backcast_length,
                            self.forecast_length,
                            self.basis_dim,
                            self.share_weights,
                            self.activation,
                            self._active_g_per_stack[stack_idx],
                        )
                else:
                    block = getattr(b, stack_type)(
                        units,
                        self.backcast_length,
                        self.forecast_length,
                        effective_td,
                        self.share_weights,
                        self.activation,
                        self._active_g_per_stack[stack_idx],
                    )

            blocks.append(block)

        return blocks

    def forward(self, x):
        x = torch.squeeze(x, -1)
        y = torch.zeros(
            x.shape[0], self.forecast_length, device=x.device, dtype=x.dtype
        )

        if self.skip_distance > 0:
            x_original = x
            alpha = (
                self.skip_alpha_param if self._skip_alpha_learnable else self.skip_alpha
            )

        n_stacks = len(self.stacks)
        for stack_id in range(n_stacks):
            for block_id in range(len(self.stacks[stack_id])):
                x_hat, y_hat = self.stacks[stack_id][block_id](x)
                x = x - x_hat
                y = y + y_hat
            if (
                self.skip_distance > 0
                and (stack_id + 1) % self.skip_distance == 0
                and stack_id < n_stacks - 1
            ):
                x = x + alpha * x_original

        stack_residual = x
        stack_forecast = y
        return stack_residual, stack_forecast


class NHiTSNet(_NBeatsBase):
    """PyTorch Lightning module implementing the NHiTS architecture.

    NHiTS (Neural Hierarchical Interpolation for Time Series) extends N-BEATS
    with two key innovations applied at the model-level forward loop:

    1. **Multi-rate input pooling**: each stack receives a MaxPool1d-downsampled
       version of the current residual, controlled by ``n_pools_kernel_size``.
    2. **Hierarchical forecast interpolation**: each stack produces a forecast at
       a reduced resolution (``forecast_length // n_freq_downsample[i]`` steps),
       which is then interpolated back to the full forecast length before
       accumulation.

    All existing block types from ``blocks.py`` (Generic, AE, LG, VAE, Wavelet,
    TrendWavelet…) plug in unchanged because the pooling and interpolation are
    entirely outside the block interface.

    Based on: Challu, C., Olivares, K. G., Oreshkin, B. N., et al. (2022).
    NHITS: Neural Hierarchical Interpolation for Time Series Forecasting.
    arXiv:2201.12886. https://arxiv.org/abs/2201.12886

    Parameters
    ----------
    backcast_length : int
        Length of the input (lookback) window.
    forecast_length : int
        Length of the forecast horizon.
    stack_types : list[str]
        Ordered list of block type names, one entry per stack.
    n_pools_kernel_size : list[int], optional
        MaxPool1d kernel size per stack. Length must match ``stack_types``.
        Default: ``[2] * n_stacks`` (halves the input at every stack).
    n_freq_downsample : list[int], optional
        Forecast downsampling ratio per stack. Each stack produces
        ``max(1, forecast_length // ratio)`` output steps before interpolation.
        Length must match ``stack_types``.
        Default: ``[1] * n_stacks`` (no downsampling — same as N-BEATS).
    interpolation_mode : str, optional
        Mode passed to ``torch.nn.functional.interpolate`` for both backcast
        and forecast upsampling. ``'linear'`` (default), ``'nearest'``, or
        ``'area'``.
    n_blocks_per_stack : int, optional
        Number of blocks per stack. Default 1.
    g_width, s_width, t_width, ae_width : int, optional
        Hidden layer widths per block family (same mapping as NBeatsNet).
    share_weights : bool, optional
        Reuse first block parameters across blocks in the same stack.
    thetas_dim : int, optional
        Basis expansion dimension for Generic / Trend blocks. Default 5.
    learning_rate : float, optional
        Optimizer learning rate. Default 1e-3.
    loss : str, optional
        Loss function name (see LOSSES). Default ``'SMAPELoss'``.
    no_val : bool, optional
        Skip validation steps when True. Default False.
    optimizer_name : str, optional
        Optimizer name (see OPTIMIZERS). Default ``'Adam'``.
    activation : str, optional
        Activation function name (see ACTIVATIONS). Default ``'ReLU'``.
    frequency : int, optional
        Series frequency; used only with ``MASELoss``. Default 1.
    active_g : bool, str, or list, optional
        Activation on basis-expansion output. False / True / ``'backcast'`` /
        ``'forecast'``. Also accepts a list of length ``len(stack_types)`` with
        an independent per-stack value (e.g. ``[True, True, False]`` to leave
        the last stack's output unclamped). Default False.
    latent_dim : int, optional
        Latent bottleneck size for AE-family blocks. Default 5.
    latent_gate_fn : str, optional
        Latent gate function for AELG blocks. One of ``'sigmoid'``,
        ``'wavy_sigmoid'``, or ``'wavelet_sigmoid'``. Default ``'sigmoid'``.
    sum_losses : bool, optional
        Add 0.25 × backcast-reconstruction loss to forecast loss. Default False.
    kl_weight : float, optional
        Multiplier applied to the summed KL divergence from VAE-family blocks
        during training. Set to 0.0 to disable KL regularization. Default 0.1.
    basis_dim : int, optional
        Wavelet basis dimensionality. Default 32.
    basis_offset : int, optional
        Row offset into the SVD-ordered WaveletV3 basis. Default 0.
    stack_basis_offsets : list[int], optional
        Per-stack override of ``basis_offset``. Default None.
    forecast_basis_dim : int, optional
        Override basis dim for WaveletV3 forecast path. Default None.
    wavelet_type : str, optional
        PyWavelets wavelet name for TrendWavelet blocks. Default ``'db3'``.
    trend_thetas_dim : int or None, optional
        Polynomial degree for Trend/TrendAE blocks. Default 3.
    lr_scheduler_config : dict, optional
        Learning-rate scheduler configuration. ``type`` selects the family:
        ``'cosine'`` (default) uses ConstantLR warmup + CosineAnnealingLR
        (keys: ``warmup_epochs``, ``T_max``, ``eta_min``); ``'step'`` uses
        ``torch.optim.lr_scheduler.StepLR`` (required: ``step_size``;
        optional: ``gamma`` default 0.5); ``'multistep'`` uses
        ``torch.optim.lr_scheduler.MultiStepLR`` (required: ``milestones``
        list[int]; optional: ``gamma`` default 0.5 — paper-faithful with
        ``n_milestones=10`` and ``patience > max_epochs / n_milestones``).
        Optional ``interval`` key (``'epoch'``/``'step'``) controls the
        Lightning update frequency. Default None.
    """

    def __init__(
        self,
        backcast_length: int,
        forecast_length: int,
        stack_types: list = None,
        n_pools_kernel_size: list = None,
        n_freq_downsample: list = None,
        interpolation_mode: str = "linear",
        n_blocks_per_stack: int = 1,
        g_width: int = 512,
        s_width: int = 2048,
        t_width: int = 256,
        ae_width: int = 512,
        share_weights: bool = False,
        thetas_dim: int = 5,
        learning_rate: float = 1e-3,
        weight_decay: float = None,
        loss: str = "SMAPELoss",
        no_val: bool = False,
        optimizer_name: str = "Adam",
        activation: str = "ReLU",
        frequency: int = 1,
        active_g=False,
        latent_dim: int = 5,
        latent_gate_fn: str = "sigmoid",
        ae_intermediate_width: int = None,
        sum_losses: bool = False,
        kl_weight: float = 0.001,
        basis_dim: int = 32,
        basis_offset: int = 0,
        stack_basis_offsets: list = None,
        stack_basis_dims: list = None,
        forecast_basis_dim: int = None,
        theta_activation: bool = False,
        wavelet_type: str = "db3",
        backcast_wavelet_type: str = None,
        forecast_wavelet_type: str = None,
        trend_thetas_dim: int | None = 3,
        generic_dim: int = 5,
        lr_scheduler_config: dict = None,
        skip_distance: int = 0,
        skip_alpha: float | str = 0.1,
    ):
        if stack_types is None:
            raise ValueError("Stack architecture must be specified.")
        # Fail fast on obviously invalid scalar strings; full per-stack
        # validation (including list form) runs below once n_stacks is known.
        if isinstance(active_g, str) and active_g not in ("backcast", "forecast"):
            raise ValueError(
                f"active_g must be True, False, 'backcast', or 'forecast', got '{active_g}'"
            )
        if latent_gate_fn not in b.LATENT_GATE_FNS:
            raise ValueError(
                f"latent_gate_fn must be one of {sorted(b.LATENT_GATE_FNS)}, got '{latent_gate_fn}'"
            )
        if trend_thetas_dim is not None and (
            not isinstance(trend_thetas_dim, int) or trend_thetas_dim < 1
        ):
            raise ValueError(
                f"trend_thetas_dim must be a positive integer, got {trend_thetas_dim}"
            )
        if not isinstance(skip_distance, int) or skip_distance < 0:
            raise ValueError(
                f"skip_distance must be a non-negative integer, got {skip_distance}"
            )
        if isinstance(skip_alpha, str):
            if skip_alpha != "learnable":
                raise ValueError(
                    f"skip_alpha must be a float or 'learnable', got '{skip_alpha}'"
                )
        elif not isinstance(skip_alpha, (int, float)) or isinstance(skip_alpha, bool):
            raise ValueError(
                f"skip_alpha must be a float or 'learnable', got {skip_alpha!r}"
            )
        if (
            not isinstance(kl_weight, Real)
            or isinstance(kl_weight, bool)
            or kl_weight < 0
        ):
            raise ValueError(
                f"kl_weight must be a non-negative real number, got {kl_weight!r}"
            )

        n_stacks = len(stack_types)

        # Default pool / downsample lists when not supplied
        if n_pools_kernel_size is None:
            n_pools_kernel_size = [2] * n_stacks
        if n_freq_downsample is None:
            n_freq_downsample = [1] * n_stacks

        if len(n_pools_kernel_size) != n_stacks:
            raise ValueError(
                f"n_pools_kernel_size length {len(n_pools_kernel_size)} must equal "
                f"the number of stacks ({n_stacks})."
            )
        if len(n_freq_downsample) != n_stacks:
            raise ValueError(
                f"n_freq_downsample length {len(n_freq_downsample)} must equal "
                f"the number of stacks ({n_stacks})."
            )

        super(NHiTSNet, self).__init__(
            loss=loss,
            frequency=frequency,
            no_val=no_val,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            sum_losses=sum_losses,
            kl_weight=kl_weight,
            lr_scheduler_config=lr_scheduler_config,
        )

        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.stack_types = stack_types
        self.n_stacks = n_stacks
        self.n_pools_kernel_size = n_pools_kernel_size
        self.n_freq_downsample = n_freq_downsample
        self.interpolation_mode = interpolation_mode
        self.n_blocks_per_stack = n_blocks_per_stack
        self.g_width = g_width
        self.s_width = s_width
        self.t_width = t_width
        self.ae_width = ae_width
        self.share_weights = share_weights
        self.thetas_dim = thetas_dim
        self.activation = activation
        self.active_g = active_g
        self.latent_dim = latent_dim
        self.latent_gate_fn = latent_gate_fn
        self.ae_intermediate_width = ae_intermediate_width
        self.basis_dim = basis_dim
        self.basis_offset = basis_offset
        self.stack_basis_offsets = stack_basis_offsets
        self.stack_basis_dims = stack_basis_dims
        self.forecast_basis_dim = forecast_basis_dim
        self.theta_activation = theta_activation
        self.wavelet_type = wavelet_type
        self.backcast_wavelet_type = backcast_wavelet_type
        self.forecast_wavelet_type = forecast_wavelet_type
        self.trend_thetas_dim = (
            self.thetas_dim if trend_thetas_dim is None else trend_thetas_dim
        )
        self.generic_dim = generic_dim
        self.skip_distance = skip_distance
        self.skip_alpha = skip_alpha

        # Precompute per-stack effective block lengths.
        # Must use math.ceil to match F.max_pool1d(..., ceil_mode=True).
        self.pooled_backcast_lengths = [
            max(1, math.ceil(backcast_length / pool_size))
            for pool_size in n_pools_kernel_size
        ]
        self.reduced_forecast_lengths = [
            max(1, forecast_length // freq_down) for freq_down in n_freq_downsample
        ]

        self.save_hyperparameters()

        if (
            skip_distance > 0
            and isinstance(skip_alpha, str)
            and skip_alpha == "learnable"
        ):
            self._skip_alpha_learnable = True
            self.skip_alpha_param = nn.Parameter(torch.tensor(0.01))
        else:
            self._skip_alpha_learnable = False

        # Normalize active_g into a per-stack list; _create_nhits_stack reads
        # from self._active_g_per_stack[stack_idx] so each stack can be
        # configured independently while self.active_g preserves the original
        # user input.
        self._active_g_per_stack = _normalize_active_g(active_g, self.n_stacks)

        self.stacks = nn.ModuleList()
        for stack_idx, stack_type in enumerate(self.stack_types):
            pooled_bl = self.pooled_backcast_lengths[stack_idx]
            reduced_fl = self.reduced_forecast_lengths[stack_idx]
            self.stacks.append(
                self._create_nhits_stack(stack_type, stack_idx, pooled_bl, reduced_fl)
            )

    # ------------------------------------------------------------------
    # Stack construction
    # ------------------------------------------------------------------

    def _create_nhits_stack(
        self,
        stack_type: str,
        stack_idx: int,
        effective_backcast_length: int,
        effective_forecast_length: int,
    ) -> nn.ModuleList:
        """Create blocks for one NHiTS stack with per-stack effective lengths.

        Mirrors NBeatsNet.create_stack() but uses ``effective_backcast_length``
        and ``effective_forecast_length`` (after pooling / downsampling) so the
        blocks operate on the correct input/output dimensions.
        """
        blocks = nn.ModuleList()
        _validate_stack_type(stack_type)

        if self.stack_basis_offsets and stack_idx < len(self.stack_basis_offsets):
            effective_offset = self.stack_basis_offsets[stack_idx]
        else:
            effective_offset = self.basis_offset

        # Per-stack basis RANK, the nested counterpart of stack_basis_offsets.
        # An offset selects rows [o : full_rank], which discards the LEADING rows.
        # That is survivable for a basis whose ordering is not aligned with where
        # the data's energy sits, and ruinous for one that is: on M4-Hourly, rows
        # [8:48] of an SVD-ordered db3 basis still capture 0.895 of the target
        # energy, while the same window of a DCT captures 0.0003, because the DCT
        # keeps its constant row first and the offset throws it away. Varying the
        # rank instead selects rows [0 : d_k], so every stack retains the leading
        # rows and only its capacity differs. That is the form of tiering an
        # energy-ordered basis can actually use.
        if self.stack_basis_dims and stack_idx < len(self.stack_basis_dims):
            effective_basis_dim = self.stack_basis_dims[stack_idx]
        else:
            effective_basis_dim = self.basis_dim

        ae_latent_blocks = (
            "GenericAE",
            "BottleneckGenericAE",
            "SeasonalityAE",
            "AutoEncoderAE",
            "TrendAE",
            "GenericAEBackcastAE",
            "GenericAELG",
            "BottleneckGenericAELG",
            "SeasonalityAELG",
            "AutoEncoderAELG",
            "TrendAELG",
            "GenericAEBackcastAELG",
            "GenericVAE",
            "BottleneckGenericVAE",
            "SeasonalityVAE",
            "AutoEncoderVAE",
            "TrendVAE",
            "GenericAEBackcastVAE",
        )

        # AE-backbone taper width. Forwarded only when explicitly set so the
        # family-named compatibility wrappers (fixed signatures) are untouched
        # and default behavior (units // 2) is preserved exactly.
        _ae_iw_kwargs = (
            {}
            if self.ae_intermediate_width is None
            else {"ae_intermediate_width": self.ae_intermediate_width}
        )

        for block_id in range(self.n_blocks_per_stack):
            if self.share_weights and block_id != 0:
                block = blocks[-1]
            else:
                # --- Width selection (identical mapping to NBeatsNet) ---
                if stack_type in [
                    "Generic",
                    "NHiTSGenericRoot",
                    "BottleneckGeneric",
                    "GenericAE",
                    "BottleneckGenericAE",
                    "GenericAEBackcast",
                    "GenericAEBackcastAE",
                    "GenericAELG",
                    "BottleneckGenericAELG",
                    "GenericAEBackcastAELG",
                    "GenericVAE",
                    "BottleneckGenericVAE",
                    "GenericAEBackcastVAE",
                ]:
                    units = self.g_width
                elif stack_type in [
                    "Seasonality",
                    "SeasonalityAE",
                    "SeasonalityAELG",
                    "SeasonalityVAE",
                ]:
                    units = self.s_width
                elif stack_type in [
                    "Trend",
                    "TrendAE",
                    "TrendWavelet",
                    "TrendWaveletAE",
                    "TrendWaveletAELG",
                    "TrendAELG",
                    "TrendVAE",
                    "TrendWaveletGeneric",
                    "TrendWaveletGenericAE",
                    "TrendWaveletGenericAELG",
                    "TrendWaveletGenericVAE",
                ]:
                    units = self.t_width
                elif stack_type in [
                    "AutoEncoder",
                    "AutoEncoderAE",
                    "AutoEncoderAELG",
                    "AutoEncoderVAE",
                    "VAE",
                ]:
                    units = self.ae_width
                else:
                    units = self.g_width

                # Trend blocks use trend_thetas_dim; all others use thetas_dim
                if stack_type in (
                    "Trend",
                    "TrendAE",
                    "TrendWavelet",
                    "TrendAELG",
                    "TrendVAE",
                    "TrendWaveletGeneric",
                    "TrendWaveletGenericAE",
                    "TrendWaveletGenericAELG",
                    "TrendWaveletGenericVAE",
                ):
                    effective_td = self.trend_thetas_dim
                else:
                    effective_td = self.thetas_dim

                # --- Block instantiation (same dispatch as NBeatsNet.create_stack) ---
                if stack_type == "TrendWavelet":
                    block = getattr(b, stack_type)(
                        units,
                        effective_backcast_length,
                        effective_forecast_length,
                        trend_dim=self.trend_thetas_dim,
                        wavelet_dim=effective_basis_dim,
                        basis_offset=effective_offset,
                        share_weights=self.share_weights,
                        activation=self.activation,
                        active_g=self._active_g_per_stack[stack_idx],
                        forecast_basis_dim=self.forecast_basis_dim,
                        wavelet_type=self.wavelet_type,
                        backcast_wavelet_type=self.backcast_wavelet_type,
                        forecast_wavelet_type=self.forecast_wavelet_type,
                    )
                elif stack_type == "TrendWaveletGeneric":
                    block = getattr(b, stack_type)(
                        units,
                        effective_backcast_length,
                        effective_forecast_length,
                        trend_dim=self.trend_thetas_dim,
                        wavelet_dim=effective_basis_dim,
                        generic_dim=self.generic_dim,
                        basis_offset=effective_offset,
                        share_weights=self.share_weights,
                        activation=self.activation,
                        active_g=self._active_g_per_stack[stack_idx],
                        forecast_basis_dim=self.forecast_basis_dim,
                        wavelet_type=self.wavelet_type,
                        backcast_wavelet_type=self.backcast_wavelet_type,
                        forecast_wavelet_type=self.forecast_wavelet_type,
                    )
                elif stack_type in ("TrendWaveletAE", "TrendWaveletAELG"):
                    block_kwargs = dict(
                        trend_dim=self.trend_thetas_dim,
                        wavelet_dim=effective_basis_dim,
                        basis_offset=effective_offset,
                        share_weights=self.share_weights,
                        activation=self.activation,
                        active_g=self._active_g_per_stack[stack_idx],
                        latent_dim=self.latent_dim,
                        forecast_basis_dim=self.forecast_basis_dim,
                        wavelet_type=self.wavelet_type,
                        backcast_wavelet_type=self.backcast_wavelet_type,
                        forecast_wavelet_type=self.forecast_wavelet_type,
                    )
                    if "AELG" in stack_type:
                        block_kwargs["latent_gate_fn"] = self.latent_gate_fn
                    block_kwargs.update(_ae_iw_kwargs)
                    block = getattr(b, stack_type)(
                        units,
                        effective_backcast_length,
                        effective_forecast_length,
                        **block_kwargs,
                    )
                elif stack_type in (
                    "TrendWaveletGenericAE",
                    "TrendWaveletGenericAELG",
                    "TrendWaveletGenericVAE",
                ):
                    block_kwargs = dict(
                        trend_dim=self.trend_thetas_dim,
                        wavelet_dim=effective_basis_dim,
                        generic_dim=self.generic_dim,
                        basis_offset=effective_offset,
                        share_weights=self.share_weights,
                        activation=self.activation,
                        active_g=self._active_g_per_stack[stack_idx],
                        latent_dim=self.latent_dim,
                        forecast_basis_dim=self.forecast_basis_dim,
                        wavelet_type=self.wavelet_type,
                        backcast_wavelet_type=self.backcast_wavelet_type,
                        forecast_wavelet_type=self.forecast_wavelet_type,
                    )
                    if "AELG" in stack_type:
                        block_kwargs["latent_gate_fn"] = self.latent_gate_fn
                    block_kwargs.update(_ae_iw_kwargs)
                    block = getattr(b, stack_type)(
                        units,
                        effective_backcast_length,
                        effective_forecast_length,
                        **block_kwargs,
                    )
                elif stack_type in ae_latent_blocks:
                    block_args = [
                        units,
                        effective_backcast_length,
                        effective_forecast_length,
                        effective_td,
                        self.share_weights,
                        self.activation,
                        self._active_g_per_stack[stack_idx],
                        self.latent_dim,
                    ]
                    if "AELG" in stack_type:
                        block = getattr(b, stack_type)(
                            *block_args, self.latent_gate_fn, **_ae_iw_kwargs
                        )
                    else:
                        block = getattr(b, stack_type)(*block_args, **_ae_iw_kwargs)
                elif "Wavelet" in stack_type:
                    _wavelet_kwargs = {}
                    if stack_type in _GENERIC_WAVELET_V3:
                        _wavelet_kwargs = dict(
                            wavelet_type=self.wavelet_type,
                            backcast_wavelet_type=self.backcast_wavelet_type,
                            forecast_wavelet_type=self.forecast_wavelet_type,
                        )
                        if stack_type == "WaveletV3":
                            # Diagnostic probe for the completeness argument;
                            # only the plain WaveletV3 base class accepts it.
                            _wavelet_kwargs["theta_activation"] = self.theta_activation
                        else:
                            # AE-backed generic wavelet classes take the taper width.
                            _wavelet_kwargs.update(_ae_iw_kwargs)
                    if any(token in stack_type for token in ("V3VAE", "V3AE")):
                        if "AELG" in stack_type:
                            _wavelet_kwargs["latent_gate_fn"] = self.latent_gate_fn
                        block = getattr(b, stack_type)(
                            units,
                            effective_backcast_length,
                            effective_forecast_length,
                            self.basis_dim,
                            effective_offset,
                            self.share_weights,
                            self.activation,
                            self._active_g_per_stack[stack_idx],
                            forecast_basis_dim=self.forecast_basis_dim,
                            latent_dim=self.latent_dim,
                            **_wavelet_kwargs,
                        )
                    elif "V3" in stack_type:
                        block = getattr(b, stack_type)(
                            units,
                            effective_backcast_length,
                            effective_forecast_length,
                            self.basis_dim,
                            effective_offset,
                            self.share_weights,
                            self.activation,
                            self._active_g_per_stack[stack_idx],
                            forecast_basis_dim=self.forecast_basis_dim,
                            **_wavelet_kwargs,
                        )
                    else:
                        block = getattr(b, stack_type)(
                            units,
                            effective_backcast_length,
                            effective_forecast_length,
                            self.basis_dim,
                            self.share_weights,
                            self.activation,
                            self._active_g_per_stack[stack_idx],
                        )
                else:
                    block = getattr(b, stack_type)(
                        units,
                        effective_backcast_length,
                        effective_forecast_length,
                        effective_td,
                        self.share_weights,
                        self.activation,
                        self._active_g_per_stack[stack_idx],
                    )

            blocks.append(block)

        return blocks

    # ------------------------------------------------------------------
    # Interpolation helper
    # ------------------------------------------------------------------

    def _interpolate(self, tensor: torch.Tensor, target_size: int) -> torch.Tensor:
        """Upsample a (batch, time) tensor to ``target_size`` time steps."""
        interp_kwargs = {"mode": self.interpolation_mode}
        if self.interpolation_mode not in ("nearest", "area"):
            interp_kwargs["align_corners"] = False
        return F.interpolate(
            tensor.unsqueeze(1), size=target_size, **interp_kwargs
        ).squeeze(1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """NHiTS forward pass with multi-rate pooling and hierarchical interpolation.

        For each stack ``i``:
          1. Downsample the residual with ``MaxPool1d(kernel=n_pools_kernel_size[i])``.
          2. Pass the pooled signal through each block to get ``(x_hat, y_hat)``.
          3. Interpolate ``x_hat`` back to ``backcast_length`` and subtract from residual.
          4. Interpolate ``y_hat`` up to ``forecast_length`` and add to forecast.

        Returns
        -------
        (residual, forecast) : tuple[Tensor, Tensor]
            Shapes ``(batch, backcast_length)`` and ``(batch, forecast_length)``.
        """
        x = torch.squeeze(x, -1)
        y = torch.zeros(
            x.shape[0], self.forecast_length, device=x.device, dtype=x.dtype
        )

        if self.skip_distance > 0:
            x_original = x
            alpha = (
                self.skip_alpha_param if self._skip_alpha_learnable else self.skip_alpha
            )

        n_stacks = len(self.stacks)
        for stack_id in range(n_stacks):
            pool_size = self.n_pools_kernel_size[stack_id]

            # Multi-rate input pooling
            if pool_size > 1:
                x_pooled = F.max_pool1d(
                    x.unsqueeze(1),
                    kernel_size=pool_size,
                    stride=pool_size,
                    ceil_mode=True,
                ).squeeze(1)
            else:
                x_pooled = x

            for block_id in range(len(self.stacks[stack_id])):
                x_hat, y_hat = self.stacks[stack_id][block_id](x_pooled)

                # Backcast: interpolate from pooled length back to original length
                if x_hat.shape[-1] != self.backcast_length:
                    x_hat_full = self._interpolate(x_hat, self.backcast_length)
                else:
                    x_hat_full = x_hat

                # Forecast: interpolate from reduced length up to full forecast length
                if y_hat.shape[-1] != self.forecast_length:
                    y_hat_full = self._interpolate(y_hat, self.forecast_length)
                else:
                    y_hat_full = y_hat

                x = x - x_hat_full
                y = y + y_hat_full

            if (
                self.skip_distance > 0
                and (stack_id + 1) % self.skip_distance == 0
                and stack_id < n_stacks - 1
            ):
                x = x + alpha * x_original

        return x, y


def _mra_projectors(length, wavelet, level, mode="periodization"):
    """Linear operators that split a signal into its DWT sub-band components.

    A discrete wavelet transform, zeroing every band but one, and the inverse
    transform are all linear, so extracting sub-band `j` is a fixed `length x
    length` matrix. Building them once at construction keeps the forward pass a
    single batched matmul instead of a per-batch call into pywt, and it makes
    the decomposition exact rather than approximate.

    Returns an array of shape `(n_bands, length, length)` whose sum over the
    band axis is the identity, so the components reconstruct the input exactly.
    """
    import pywt

    probe = pywt.wavedec(np.zeros(length), wavelet, level=level, mode=mode)
    n_bands = len(probe)
    projectors = np.zeros((n_bands, length, length), dtype=np.float64)

    for t in range(length):
        impulse = np.zeros(length)
        impulse[t] = 1.0
        coeffs = pywt.wavedec(impulse, wavelet, level=level, mode=mode)
        for j in range(n_bands):
            isolated = [np.zeros_like(c) for c in coeffs]
            isolated[j] = coeffs[j]
            projectors[j, :, t] = pywt.waverec(isolated, wavelet, mode=mode)[:length]

    reconstruction_error = np.abs(projectors.sum(axis=0) - np.eye(length)).max()
    if reconstruction_error > 1e-8:
        raise ValueError(
            f"MRA projectors do not sum to the identity (max deviation "
            f"{reconstruction_error:.2e}); the decomposition would lose signal."
        )
    return projectors


class DWTPreprocessNBeats(_NBeatsBase):
    """N-BEATS with an external DWT preprocessor, one Generic stack per sub-band.

    This is a reimplementation of the design of Pramanick et al. (2024), which
    applies a discrete wavelet transform to the input series and trains a
    separate Generic stack on each resulting sub-band. It exists as a baseline:
    the contrast against it is external decomposition of the *input* against an
    in block frozen synthesis basis, which is what this repository implements
    elsewhere.

    The input of length `L` is split into `level + 1` components that sum back
    to it exactly. Each component is routed to its own independent stack of
    Generic blocks carrying the standard doubly residual topology, and the
    per band forecasts are summed. Because the decomposition is a partition of
    the signal rather than a transform of it, no inverse transform is needed on
    the forecast side, and the horizon never has to be given a wavelet
    structure it does not possess.

    Args:
        backcast_length (int): Length of the input window, `L`.
        forecast_length (int): Length of the horizon, `H`.
        wavelet_type (str): Wavelet family for the decomposition, e.g. `db3`.
        n_levels (int): Decomposition depth `J`, giving `J + 1` sub-bands.
            Clamped to `pywt.dwt_max_level` for the given length and family.
        n_blocks_per_band (int): Generic blocks in each sub-band's stack.
        g_width (int): Hidden width of the fully connected trunk in each block.
        share_weights (bool): Share block parameters within a sub-band stack.
        activation (str): Trunk activation.
        active_g (bool): Apply an activation to the block's synthesis output.
    """

    def __init__(
        self,
        backcast_length: int,
        forecast_length: int,
        wavelet_type: str = "db3",
        n_levels: int = 3,
        n_blocks_per_band: int = 2,
        g_width: int = 512,
        share_weights: bool = False,
        activation: str = "ReLU",
        active_g: bool = False,
        thetas_dim: int = 5,
        loss: str = "SMAPELoss",
        frequency: int = 1,
        no_val: bool = False,
        optimizer_name: str = "Adam",
        learning_rate: float = 1e-3,
        weight_decay: float = None,
        sum_losses: bool = False,
        kl_weight: float = 0.001,
        lr_scheduler_config: dict = None,
        **unused,
    ):
        super(DWTPreprocessNBeats, self).__init__(
            loss=loss,
            frequency=frequency,
            no_val=no_val,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            sum_losses=sum_losses,
            kl_weight=kl_weight,
            lr_scheduler_config=lr_scheduler_config,
        )
        import pywt

        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.wavelet_type = wavelet_type
        self.g_width = g_width
        self.t_width = g_width  # reported by the benchmark runner

        max_level = pywt.dwt_max_level(backcast_length, pywt.Wavelet(wavelet_type).dec_len)
        self.n_levels = int(min(n_levels, max_level))
        if self.n_levels < 1:
            raise ValueError(
                f"backcast_length={backcast_length} admits no DWT level for "
                f"'{wavelet_type}'; the decomposition would be a no op."
            )

        projectors = _mra_projectors(backcast_length, wavelet_type, self.n_levels)
        self.n_bands = projectors.shape[0]
        self.register_buffer(
            "band_projectors", torch.tensor(projectors, dtype=torch.float32)
        )

        # One independent Generic stack per sub-band.
        self.stacks = nn.ModuleList()
        for _ in range(self.n_bands):
            band_stack = nn.ModuleList()
            for block_idx in range(n_blocks_per_band):
                if share_weights and block_idx > 0:
                    band_stack.append(band_stack[0])
                else:
                    band_stack.append(
                        b.Generic(
                            units=g_width,
                            backcast_length=backcast_length,
                            forecast_length=forecast_length,
                            thetas_dim=thetas_dim,
                            share_weights=False,
                            activation=activation,
                            active_g=active_g,
                        )
                    )
            self.stacks.append(band_stack)

        self.save_hyperparameters()

    def forward(self, x):
        x = torch.squeeze(x, -1)

        # (batch, L) x (bands, L, L) -> (bands, batch, L); components sum to x.
        components = torch.einsum("bl,jkl->jbk", x, self.band_projectors)

        forecast = torch.zeros(
            x.shape[0], self.forecast_length, device=x.device, dtype=x.dtype
        )
        residual = torch.zeros_like(x)

        for band_idx in range(self.n_bands):
            band_residual = components[band_idx]
            for block in self.stacks[band_idx]:
                backcast_hat, forecast_hat = block(band_residual)
                band_residual = band_residual - backcast_hat
                forecast = forecast + forecast_hat
            residual = residual + band_residual

        return residual, forecast
