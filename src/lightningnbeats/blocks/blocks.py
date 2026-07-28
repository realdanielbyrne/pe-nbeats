import os
import torch
from torch import nn
from torch.nn import functional as F
from ..constants import ACTIVATIONS, DEPRECATED_BLOCKS, get_deprecated_block_message
import numpy as np
import pywt
from pywt import Wavelet as PyWavelet  # type: ignore[attr-defined]

LATENT_GATE_FNS = {"sigmoid", "wavy_sigmoid", "wavelet_sigmoid"}


def wavy_sigmoid(x, x0=2.0, eps=0.02, beta=12.0):
    """Smooth sigmoid variant with a damped sinusoidal tail."""
    a = torch.sqrt(x * x + eps * eps) - eps
    tau = F.softplus(beta * (a - x0)) / beta
    return torch.sigmoid(1.6 * x) + 0.18 * torch.exp(-1.1 * tau) * torch.sin(4.2 * tau)


def wavelet_sigmoid(x, A=0.18, x0=2.0, s=1.25, w0=4.5, beta=12.0, eps=0.02):
    """Sigmoid variant with a Gaussian-windowed wavelet-style oscillation."""
    a = torch.sqrt(x * x + eps * eps) - eps
    tau = (
        F.softplus(beta * (a - x0)) / beta - F.softplus(x.new_tensor(-beta * x0)) / beta
    )
    u = tau / s
    return torch.sigmoid(1.6 * x) + A * torch.exp(-0.5 * u * u) * torch.sin(w0 * u)


def squeeze_last_dim(tensor):
    if len(tensor.shape) == 3 and tensor.shape[-1] == 1:  # (128, 10, 1) => (128, 10).
        return tensor[..., 0]
    return tensor


class RootBlock(nn.Module):
    def __init__(self, backcast_length, units, activation="ReLU"):
        """The Block class is the basic building block of the N-BEATS network.
        It consists of a stack of fully connected layers.It serves as the base
        class for the GenericBlock, SeasonalityBlock, and TrendBlock classes.

        Args:
            backcast_length (int):
              The length of the historical data.  It is customary to use a
              multiple of the forecast_length (H)orizon (2H,3H,4H,5H,...).
            units (int):
              The width of the fully connected layers in the blocks comprising
              the stacks of the generic model
            activation (str, optional):
              The activation function applied to each of the fully connected
              Linear layers. Defaults to 'ReLU'.

        Raises:
              ValueError: If the activation function is not in ACTIVATIONS.
        """
        super(RootBlock, self).__init__()
        self.units = units
        self.backcast_length = backcast_length

        if activation not in ACTIVATIONS:
            raise ValueError(f"'{activation}' is not in {ACTIVATIONS}")

        self.activation = getattr(nn, activation)()

        self.fc1 = nn.Linear(backcast_length, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        return x


class AutoEncoder(RootBlock):
    def __init__(
        self,
        units: int,
        backcast_length: int,
        forecast_length: int,
        thetas_dim: int,
        share_weights: bool,
        activation: str = "ReLU",
        active_g: bool = False,
    ):

        super(AutoEncoder, self).__init__(backcast_length, units, activation)

        self.units = units
        self.thetas_dim = thetas_dim
        self.share_weights = share_weights
        self.activation = getattr(nn, activation)()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.active_g = active_g

        # Encoders
        if share_weights:
            self.b_encoder = self.f_encoder = nn.Sequential(
                nn.Linear(units, thetas_dim),
                nn.ReLU(),
            )
        else:
            self.b_encoder = nn.Sequential(
                nn.Linear(units, thetas_dim),
                nn.ReLU(),
            )
            self.f_encoder = nn.Sequential(
                nn.Linear(units, thetas_dim),
                nn.ReLU(),
            )

        self.b_decoder = nn.Sequential(
            nn.Linear(thetas_dim, units),
            nn.ReLU(),
            nn.Linear(units, backcast_length),
        )
        self.f_decoder = nn.Sequential(
            nn.Linear(thetas_dim, units),
            nn.ReLU(),
            nn.Linear(units, forecast_length),
        )

    def forward(self, x):
        x = super(AutoEncoder, self).forward(x)
        b = self.b_encoder(x)
        b = self.b_decoder(b)

        f = self.f_encoder(x)
        f = self.f_decoder(f)

        # N-BEATS paper does not apply activation here, but Generic models will not converge sometimes without it
        if self.active_g:
            if self.active_g != "forecast":
                b = self.activation(b)
            if self.active_g != "backcast":
                f = self.activation(f)

        return b, f


class _VAEMixin:
    """Shared VAE utility methods: reparameterization, KL divergence, and logvar clamping.

    This mixin centralises the three core VAE operations to eliminate code duplication
    across all VAE-type block hierarchies (RootBlock and AERootBlockVAE).
    """

    LOGVAR_MIN: float = -10.0
    LOGVAR_MAX: float = 4.0

    def _clamp_logvar(self, logvar: torch.Tensor) -> torch.Tensor:
        """Clamp logvar to [LOGVAR_MIN, LOGVAR_MAX] for numerical stability."""
        return torch.clamp(logvar, min=self.LOGVAR_MIN, max=self.LOGVAR_MAX)

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps during training, z = mu during eval."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def _kl_divergence(
        self, mu: torch.Tensor, logvar: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """KL divergence vs unit Gaussian: -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) / batch_size."""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size


def __getattr__(name):
    if name in DEPRECATED_BLOCKS:
        raise AttributeError(get_deprecated_block_message(name))
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


class VAE(_VAEMixin, RootBlock):
    """Variational AutoEncoder block using the RootBlock backbone.

    Alternative to the AutoEncoder block that replaces the deterministic encoder
    bottleneck with a stochastic latent space using the reparameterization trick.
    Each path (backcast and forecast) has its own mu/logvar heads. During training,
    z = mu + std * eps; during eval, z = mu (deterministic). Stores self.kl_loss
    after each forward pass for collection by the training step.

    Args:
        units (int): Width of the fully connected layers.
        backcast_length (int): Length of the historical data.
        forecast_length (int): Length of the forecast horizon.
        thetas_dim (int): Latent space dimensionality for the VAE bottleneck.
        share_weights (bool): If True, backcast and forecast encoders share parameters.
        activation (str, optional): Activation function name. Defaults to 'ReLU'.
        active_g (bool, optional): Apply activation to outputs. Defaults to False.
    """

    def __init__(
        self,
        units: int,
        backcast_length: int,
        forecast_length: int,
        thetas_dim: int,
        share_weights: bool,
        activation: str = "ReLU",
        active_g: bool = False,
    ):

        super(VAE, self).__init__(backcast_length, units, activation)

        self.units = units
        self.thetas_dim = thetas_dim
        self.share_weights = share_weights
        self.activation = getattr(nn, activation)()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.active_g = active_g

        # VAE Encoders — mu and logvar heads instead of deterministic bottleneck
        if share_weights:
            self.b_mu = self.f_mu = nn.Linear(units, thetas_dim)
            self.b_logvar = self.f_logvar = nn.Linear(units, thetas_dim)
        else:
            self.b_mu = nn.Linear(units, thetas_dim)
            self.b_logvar = nn.Linear(units, thetas_dim)
            self.f_mu = nn.Linear(units, thetas_dim)
            self.f_logvar = nn.Linear(units, thetas_dim)

        self.b_decoder = nn.Sequential(
            nn.Linear(thetas_dim, units),
            nn.ReLU(),
            nn.Linear(units, backcast_length),
        )
        self.f_decoder = nn.Sequential(
            nn.Linear(thetas_dim, units),
            nn.ReLU(),
            nn.Linear(units, forecast_length),
        )

        # KL divergence loss stored after each forward pass
        self.kl_loss = torch.tensor(0.0)

    def forward(self, x):
        x = super(VAE, self).forward(x)
        batch_size = x.shape[0]

        # Backcast path
        b_mu = self.b_mu(x)
        b_logvar = self._clamp_logvar(self.b_logvar(x))
        b_z = self._reparameterize(b_mu, b_logvar)
        b = self.b_decoder(b_z)

        # Forecast path
        f_mu = self.f_mu(x)
        f_logvar = self._clamp_logvar(self.f_logvar(x))
        f_z = self._reparameterize(f_mu, f_logvar)
        f = self.f_decoder(f_z)

        # Store combined KL loss from both paths
        self.kl_loss = self._kl_divergence(
            b_mu, b_logvar, batch_size
        ) + self._kl_divergence(f_mu, f_logvar, batch_size)

        if self.active_g:
            if self.active_g != "forecast":
                b = self.activation(b)
            if self.active_g != "backcast":
                f = self.activation(f)

        return b, f


class Generic(RootBlock):
    def __init__(
        self,
        units: int,
        backcast_length: int,
        forecast_length: int,
        thetas_dim: int = 5,
        share_weights: bool = False,
        activation: str = "ReLU",
        active_g: bool = False,
    ):
        """Paper-faithful Generic Block as defined in Oreshkin et al. (2019).

        Uses a single linear layer producing theta of size (backcast_length + forecast_length),
        then slices into backcast and forecast components. This matches the paper's formulation
        exactly: 4 FC+ReLU layers followed by one linear projection, with no intermediate
        bottleneck dimension.

        Args:
            units (int):
              The width of the fully connected layers.
            backcast_length (int):
              The length of the historical data.
            forecast_length (int):
              The length of the forecast horizon.
            thetas_dim (int, optional):
              Not used in paper-faithful Generic (kept for API compatibility). Defaults to 5.
            share_weights (bool, optional):
              If True, the initial weights of the Linear layers are shared.
              Defaults to False.
            activation (str, optional):
              The activation function used in the FC layers. Defaults to 'ReLU'.
            active_g (bool, optional):
              If True, applies activation after the basis expansion.
              Not a feature in the original paper. Defaults to False.
        """
        super(Generic, self).__init__(backcast_length, units, activation)

        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.active_g = active_g

        self.theta_b_fc = nn.Linear(units, backcast_length, bias=False)
        self.theta_f_fc = nn.Linear(units, forecast_length, bias=False)

    def forward(self, x):
        x = super(Generic, self).forward(x)

        backcast = self.theta_b_fc(x)
        forecast = self.theta_f_fc(x)

        if self.active_g:
            if self.active_g != "forecast":
                backcast = self.activation(backcast)
            if self.active_g != "backcast":
                forecast = self.activation(forecast)

        return backcast, forecast


class NHiTSGenericRoot(nn.Module):
    def __init__(
        self,
        units: int,
        backcast_length: int,
        forecast_length: int,
        thetas_dim: int = 5,
        share_weights: bool = False,
        activation: str = "ReLU",
        active_g: bool = False,
    ):
        """NHiTS paper-faithful Generic block (2-layer MLP tower).

        Matches the NHiTS reference implementation (neuralforecast) which uses
        ``mlp_units = [[512, 512]]`` per block — i.e. 2 FC+activation layers,
        not the 4-layer tower inherited from N-BEATS. The backcast/forecast
        projection heads are identical to :class:`Generic` (direct projection
        to backcast_length and forecast_length, no bottleneck).

        Args are kept API-compatible with :class:`Generic` so the block can be
        dispatched through the same ``create_stack`` branch. ``thetas_dim`` and
        ``share_weights`` are accepted but unused.
        """
        super(NHiTSGenericRoot, self).__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"'{activation}' is not in {ACTIVATIONS}")

        self.units = units
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.active_g = active_g
        self.activation = getattr(nn, activation)()

        self.fc1 = nn.Linear(backcast_length, units)
        self.fc2 = nn.Linear(units, units)

        self.theta_b_fc = nn.Linear(units, backcast_length, bias=False)
        self.theta_f_fc = nn.Linear(units, forecast_length, bias=False)

    def forward(self, x):
        x = squeeze_last_dim(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))

        backcast = self.theta_b_fc(x)
        forecast = self.theta_f_fc(x)

        if self.active_g:
            if self.active_g != "forecast":
                backcast = self.activation(backcast)
            if self.active_g != "backcast":
                forecast = self.activation(forecast)

        return backcast, forecast


class BottleneckGeneric(RootBlock):
    def __init__(
        self,
        units: int,
        backcast_length: int,
        forecast_length: int,
        thetas_dim: int = 5,
        share_weights: bool = False,
        activation: str = "ReLU",
        active_g: bool = False,
    ):
        """Bottleneck Generic Block — a novel extension of the paper's Generic block.

        Uses a two-stage projection through an intermediate thetas_dim bottleneck:
        units → thetas_dim → target_length. This is equivalent to a rank-d factorization
        of the basis expansion matrix, where d = thetas_dim, providing a tunable knob
        to control basis complexity.

        Args:
            units (int):
              The width of the fully connected layers in the blocks comprising
              the stacks of the generic model.
            backcast_length (int):
              The length of the historical data.
            forecast_length (int):
              The length of the forecast horizon.
            thetas_dim (int, optional):
              The dimensionality of the bottleneck (rank of factorized basis).
              Defaults to 5.
            share_weights (bool, optional):
              If True, the initial weights of the Linear layers are shared.
              Defaults to False.
            activation (str, optional):
              The activation function used in the parent class Block, and
              optionally as the non-linear activation of the backcast_g and
              forecast_g layers. Defaults to 'ReLU'.
            active_g (bool, optional):
              If True, applies activation after the basis expansion.
              Not a feature in the original paper. Defaults to False.
        """
        super(BottleneckGeneric, self).__init__(backcast_length, units, activation)

        if share_weights:
            self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
        else:
            self.backcast_linear = nn.Linear(units, thetas_dim)
            self.forecast_linear = nn.Linear(units, thetas_dim)

        self.backcast_g = nn.Linear(thetas_dim, backcast_length, bias=False)
        self.forecast_g = nn.Linear(thetas_dim, forecast_length, bias=False)
        self.active_g = active_g

    def forward(self, x):
        x = super(BottleneckGeneric, self).forward(x)
        theta_b = self.backcast_linear(x)
        theta_f = self.forecast_linear(x)

        backcast = self.backcast_g(theta_b)
        forecast = self.forecast_g(theta_f)

        if self.active_g:
            if self.active_g != "forecast":
                backcast = self.activation(backcast)
            if self.active_g != "backcast":
                forecast = self.activation(forecast)

        return backcast, forecast


class GenericAEBackcast(RootBlock):
    def __init__(
        self,
        units: int,
        backcast_length: int,
        forecast_length: int,
        thetas_dim: int,
        share_weights: bool,
        activation: str = "ReLU",
        active_g: bool = False,
    ):

        super(GenericAEBackcast, self).__init__(backcast_length, units, activation)

        self.units = units
        self.thetas_dim = thetas_dim
        self.share_weights = share_weights
        self.activation = getattr(nn, activation)()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.active_g = active_g

        self.forecast_linear = nn.Linear(units, self.thetas_dim)
        self.forecast_g = nn.Linear(self.thetas_dim, forecast_length, bias=False)

        self.b_encoder = nn.Linear(units, self.thetas_dim)
        self.b_decoder = nn.Linear(self.thetas_dim, units)
        self.backcast_g = nn.Linear(units, backcast_length)

    def forward(self, x):
        x = super(GenericAEBackcast, self).forward(x)
        b = self.activation(self.b_encoder(x))
        b = self.activation(self.b_decoder(b))
        b = self.backcast_g(b)

        theta_f = self.forecast_linear(x)
        f = self.forecast_g(theta_f)

        # N-BEATS paper does not apply activation here, but Generic models will not converge sometimes without it
        if self.active_g:
            if self.active_g != "forecast":
                b = self.activation(b)
            if self.active_g != "backcast":
                f = self.activation(f)
        return b, f


class _SeasonalityGenerator(nn.Module):
    def __init__(self, len):
        """Generates the basis for the Fourier basic expansion of the Seasonality Block.

        Args:
            forecast_length (int): The length of the forecast_length horizon.
        """
        super().__init__()
        half_minus_one = int(len / 2 - 1)
        cos_vectors = [
            torch.cos(torch.arange(len) / len * 2 * np.pi * i)
            for i in range(1, half_minus_one + 1)
        ]
        sin_vectors = [
            torch.sin(torch.arange(len) / len * 2 * np.pi * i)
            for i in range(1, half_minus_one + 1)
        ]

        # basis is of size (2 * int(forecast_length / 2 - 1) + 1, forecast_length)
        basis = torch.stack([torch.ones(len)] + cos_vectors + sin_vectors, dim=1).T

        self.basis = nn.Parameter(basis, requires_grad=False)

    def forward(self, x):
        return torch.matmul(x, self.basis)


class Seasonality(RootBlock):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        thetas_dim=5,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
    ):

        super(Seasonality, self).__init__(backcast_length, units, activation)

        self.backcast_linear = nn.Linear(
            units, 2 * int(backcast_length / 2 - 1) + 1, bias=False
        )
        self.forecast_linear = nn.Linear(
            units, 2 * int(forecast_length / 2 - 1) + 1, bias=False
        )

        self.backcast_g = _SeasonalityGenerator(backcast_length)
        self.forecast_g = _SeasonalityGenerator(forecast_length)

    def forward(self, x):
        x = super(Seasonality, self).forward(x)
        # linear compression
        backcast_thetas = self.backcast_linear(x)
        forecast_thetas = self.forecast_linear(x)

        # fourier expansion
        backcast = self.backcast_g(backcast_thetas)
        forecast = self.forecast_g(forecast_thetas)

        return backcast, forecast


class _TrendGenerator(nn.Module):
    def __init__(self, thetas_dim, target_length):
        """Trend model. A typical characteristic of trend is that most of the time it is a
        monotonic function, or at least a slowly varying function. In order to mimic this
        behaviour this block implements a low pass filter, or slowly varying function of a
        small degree polynomial across forecast_length window.

        Args:
            expansion_coefficient_dim (int): The dimensionality of the expansion coefficients used in
            the Vandermonde expansion.  The N-BEATS paper uses 2 or 3, but any positive integer can be used.
            5 is also a reasonalbe choice.
            target_length (int): The length of the forecast_length horizon.
        """
        super().__init__()

        # basis is of size (expansion_coefficient_dim, target_length)
        basis = torch.stack(
            [
                (torch.arange(target_length) / target_length) ** i
                for i in range(thetas_dim)
            ],
            dim=1,
        ).T

        self.basis = nn.Parameter(basis, requires_grad=False)

    def forward(self, x):
        return torch.matmul(x, self.basis)


class Trend(RootBlock):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        thetas_dim,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
    ):
        """The Trend Block implements the function whose parameters are generated by the _TrendGenerator block.

        Args:
            units (int):
              The width of the fully connected layers in the blocks comprising the parent class Block.
            backcast_length (int):
              The length of the historical data.  It is customary to use a multiple of the forecast_length (H)orizon (2H,3H,4H,5H,...).
            forecast_length (int):
              The length of the forecast_length horizon.
            thetas_dim (int):
              The dimensionality of the _TrendGenerator polynomial.
            share_weights (bool, optional):
              If True, the inital weights of the Linear layers are shared. Defaults to False.
            activation (str, optional):
              The activation function passed to the parent class Block. Defaults to 'ReLU'.
        """
        super(Trend, self).__init__(backcast_length, units, activation)
        self.share_weights = share_weights
        if self.share_weights:
            self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
        else:
            self.backcast_linear = nn.Linear(units, thetas_dim)
            self.forecast_linear = nn.Linear(units, thetas_dim)

        self.backcast_g = _TrendGenerator(thetas_dim, backcast_length)
        self.forecast_g = _TrendGenerator(thetas_dim, forecast_length)

    def forward(self, x):
        x = super(Trend, self).forward(x)

        # linear compression
        backcast_thetas = self.backcast_linear(x)
        forecast_thetas = self.forecast_linear(x)

        #  Vandermonde basis expansion
        backcast = self.backcast_g(backcast_thetas)
        forecast = self.forecast_g(forecast_thetas)

        return backcast, forecast


# ---------------------------------------------------------------------------
# AutoEncoderRoot Blocks
# ---------------------------------------------------------------------------
class AERootBlock(nn.Module):
    def __init__(self, backcast_length, units, activation="ReLU", latent_dim=5):
        """The AERootBlock class is the basic building block of the N-BEATS network.
        It consists of a stack of fully connected layers organized as an Autoencoder.
        It serves as the base class for the GenericAEBlock, SeasonalityAEBlock,
        and the TrendAEBlock classes.

        Args:
            backcast_length (int):
              The length of the historical data.  It is customary to use a multiple
              of the forecast_length (H)orizon (2H,3H,4H,5H,...).
            units (int):
              The width of the fully connected layers in the blocks comprising the
              stacks of the generic model.
            activation (str, optional):
              The activation function applied to each of the fully connected Linear
              layers. Defaults to 'ReLU'.
            latent_dim (int, optional):
              The dimensionality of the latent space. Defaults to 5.

        Raises:
              ValueError: If the activation function is not in ACTIVATIONS.
        """
        super(AERootBlock, self).__init__()
        self.units = units
        self.backcast_length = backcast_length
        self.latent_dim = latent_dim

        if not activation in ACTIVATIONS:
            raise ValueError(f"'{activation}' is not in {ACTIVATIONS}")

        self.activation = getattr(nn, activation)()

        self.fc1 = nn.Linear(backcast_length, units // 2)
        self.fc2 = nn.Linear(units // 2, latent_dim)
        self.fc3 = nn.Linear(latent_dim, units // 2)
        self.fc4 = nn.Linear(units // 2, units)

    def forward(self, x):
        x = squeeze_last_dim(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        return x


class AERootBlockLG(nn.Module):
    def __init__(
        self,
        backcast_length,
        units,
        activation="ReLU",
        latent_dim=5,
        latent_gate_fn="sigmoid",
    ):
        """Learned-Gate AE backbone. Same encoder-decoder structure as AERootBlock
        but adds a learnable gate vector that applies sigmoid-scaled per-dimension
        masking at the latent bottleneck, allowing the network to discover the
        effective latent dimensionality during training.

        Args:
            backcast_length (int): Length of the historical data.
            units (int): Width of the fully connected layers.
            activation (str, optional): Activation function name. Defaults to 'ReLU'.
            latent_dim (int, optional): Maximum latent dimensionality. Defaults to 5.
            latent_gate_fn (str, optional): Latent gate function name. One of
                ``'sigmoid'``, ``'wavy_sigmoid'``, or ``'wavelet_sigmoid'``.
                Defaults to ``'sigmoid'``.
        """
        super(AERootBlockLG, self).__init__()
        self.units = units
        self.backcast_length = backcast_length
        self.latent_dim = latent_dim
        self.latent_gate_fn = latent_gate_fn

        if activation not in ACTIVATIONS:
            raise ValueError(f"'{activation}' is not in {ACTIVATIONS}")
        if latent_gate_fn not in LATENT_GATE_FNS:
            raise ValueError(
                f"latent_gate_fn must be one of {sorted(LATENT_GATE_FNS)}, got '{latent_gate_fn}'"
            )

        self.activation = getattr(nn, activation)()

        self.fc1 = nn.Linear(backcast_length, units // 2)
        self.fc2 = nn.Linear(units // 2, latent_dim)
        self.fc3 = nn.Linear(latent_dim, units // 2)
        self.fc4 = nn.Linear(units // 2, units)

        # Learned gate: sigmoid(latent_gate) scales each latent dimension
        self.latent_gate = nn.Parameter(torch.ones(latent_dim))

    def _compute_latent_gate_scale(self):
        if self.latent_gate_fn == "sigmoid":
            return torch.sigmoid(self.latent_gate)
        if self.latent_gate_fn == "wavy_sigmoid":
            return wavy_sigmoid(self.latent_gate)
        return wavelet_sigmoid(self.latent_gate)

    def forward(self, x):
        x = squeeze_last_dim(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = x * self._compute_latent_gate_scale()
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        return x


class AERootBlockVAE(_VAEMixin, nn.Module):
    def __init__(self, backcast_length, units, activation="ReLU", latent_dim=5):
        """Variational AE backbone. Replaces the deterministic bottleneck with a
        stochastic latent space using the reparameterization trick. Produces mu and
        log_var heads, samples z = mu + sigma * epsilon during training, and stores
        self.kl_loss for collection by the training step.

        Args:
            backcast_length (int): Length of the historical data.
            units (int): Width of the fully connected layers.
            activation (str, optional): Activation function name. Defaults to 'ReLU'.
            latent_dim (int, optional): Latent space dimensionality. Defaults to 5.
        """
        super(AERootBlockVAE, self).__init__()
        self.units = units
        self.backcast_length = backcast_length
        self.latent_dim = latent_dim

        if activation not in ACTIVATIONS:
            raise ValueError(f"'{activation}' is not in {ACTIVATIONS}")

        self.activation = getattr(nn, activation)()

        self.fc1 = nn.Linear(backcast_length, units // 2)
        self.fc1b = nn.Linear(units // 2, units // 4)
        self.fc2_mu = nn.Linear(units // 4, latent_dim)
        self.fc2_logvar = nn.Linear(units // 4, latent_dim)
        self.fc3 = nn.Linear(latent_dim, units // 4)
        self.fc3b = nn.Linear(units // 4, units // 2)
        self.fc4 = nn.Linear(units // 2, units)

        # KL divergence loss stored after each forward pass
        self.kl_loss = torch.tensor(0.0)

    def forward(self, x):
        x = squeeze_last_dim(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc1b(x))

        mu = self.fc2_mu(x)
        logvar = self._clamp_logvar(self.fc2_logvar(x))

        z = self._reparameterize(mu, logvar)
        self.kl_loss = self._kl_divergence(mu, logvar, x.shape[0])

        x = self.activation(self.fc3(z))
        x = self.activation(self.fc3b(x))
        x = self.activation(self.fc4(x))
        return x


class GenericAEBackcastAE(AERootBlock):
    def __init__(
        self,
        units: int,
        backcast_length: int,
        forecast_length: int,
        thetas_dim: int,
        share_weights: bool,
        activation: str = "ReLU",
        active_g: bool = False,
        latent_dim: int = 5,
    ):

        super(GenericAEBackcastAE, self).__init__(
            backcast_length, units, activation, latent_dim=latent_dim
        )

        self.units = units
        self.thetas_dim = thetas_dim
        self.share_weights = share_weights
        self.activation = getattr(nn, activation)()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.active_g = active_g
        branch_latent_dim = latent_dim

        self.forecast_linear = nn.Linear(units, thetas_dim)
        self.forecast_g = nn.Linear(thetas_dim, forecast_length, bias=False)

        self.b_encoder = nn.Linear(units, branch_latent_dim)
        self.b_decoder = nn.Linear(branch_latent_dim, units)
        self.backcast_g = nn.Linear(units, backcast_length)

    def forward(self, x):
        x = super(GenericAEBackcastAE, self).forward(x)
        b = self.activation(self.b_encoder(x))
        b = self.activation(self.b_decoder(b))
        b = self.backcast_g(b)

        theta_f = self.forecast_linear(x)
        f = self.forecast_g(theta_f)

        # N-BEATS paper does not apply activation here;
        # however Generic models will not always converge without it
        if self.active_g:
            if self.active_g != "forecast":
                b = self.activation(b)
            if self.active_g != "backcast":
                f = self.activation(f)
        return b, f


class AutoEncoderAE(AERootBlock):
    def __init__(
        self,
        units: int,
        backcast_length: int,
        forecast_length: int,
        thetas_dim: int,
        share_weights: bool,
        activation: str = "ReLU",
        active_g: bool = False,
        latent_dim: int = 5,
    ):
        """AutoEncoder block with an AE backbone.

        Args:
            units (int): The number of input and output units
            backcast_length (int): The length of the historical data.
            forecast_length (int): The length of the forecast_length horizon.
            thetas_dim (int): Unused. Kept for API compatibility with other block types.
            share_weights (bool): The weights of the encoder are shared if True.
            activation (str, optional): Activation function name. Defaults to 'ReLU'.
            active_g (bool, optional): Apply activation after decoding. Defaults to False.
            latent_dim (int, optional): Branch-local AE bottleneck size. Defaults to 5.
        """

        super(AutoEncoderAE, self).__init__(
            backcast_length, units, activation, latent_dim=latent_dim
        )

        self.units = units
        self.thetas_dim = thetas_dim
        self.share_weights = share_weights
        self.activation = getattr(nn, activation)()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.active_g = active_g
        branch_latent_dim = latent_dim

        if share_weights:
            self.b_encoder = self.f_encoder = nn.Linear(units, branch_latent_dim)
        else:
            self.b_encoder = nn.Linear(units, branch_latent_dim)
            self.f_encoder = nn.Linear(units, branch_latent_dim)

        self.b_decoder = nn.Linear(branch_latent_dim, units)
        self.f_decoder = nn.Linear(branch_latent_dim, units)
        self.backcast_g = nn.Linear(units, backcast_length)
        self.forecast_g = nn.Linear(units, forecast_length)

    def forward(self, x):
        x = super(AutoEncoderAE, self).forward(x)
        b = self.activation(self.b_encoder(x))
        b = self.activation(self.b_decoder(b))
        b = self.backcast_g(b)

        f = self.activation(self.f_encoder(x))
        f = self.activation(self.f_decoder(f))
        f = self.forecast_g(f)

        # N-BEATS paper does not apply activation here, but Generic models will not converge sometimes without it
        if self.active_g:
            if self.active_g != "forecast":
                b = self.activation(b)
            if self.active_g != "backcast":
                f = self.activation(f)

        return b, f


class GenericAE(AERootBlock):
    def __init__(
        self,
        units: int,
        backcast_length: int,
        forecast_length: int,
        thetas_dim: int = 5,
        share_weights: bool = False,
        activation: str = "ReLU",
        active_g: bool = False,
        latent_dim: int = 5,
    ):
        """Paper-faithful Generic Block with AERootBlock (bottleneck pre-split) backbone.

        Uses direct linear projections from units to target lengths (no intermediate
        thetas_dim bottleneck), matching the paper's Generic formulation.

        Args:
            units (int): Width of the fully connected layers.
            backcast_length (int): Length of the historical data.
            forecast_length (int): Length of the forecast horizon.
            thetas_dim (int, optional): Not used (kept for API compatibility). Defaults to 5.
            share_weights (bool, optional): Defaults to False.
            activation (str, optional): Defaults to 'ReLU'.
            active_g (bool, optional): If True, applies activation after projection. Defaults to False.
            latent_dim (int, optional): Latent dimension for the AE backbone. Defaults to 5.
        """
        super(GenericAE, self).__init__(
            backcast_length, units, activation, latent_dim=latent_dim
        )

        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.active_g = active_g

        self.theta_b_fc = nn.Linear(units, backcast_length, bias=False)
        self.theta_f_fc = nn.Linear(units, forecast_length, bias=False)

    def forward(self, x):
        x = super(GenericAE, self).forward(x)

        backcast = self.theta_b_fc(x)
        forecast = self.theta_f_fc(x)

        if self.active_g:
            if self.active_g != "forecast":
                backcast = self.activation(backcast)
            if self.active_g != "backcast":
                forecast = self.activation(forecast)

        return backcast, forecast


class BottleneckGenericAE(AERootBlock):
    def __init__(
        self,
        units: int,
        backcast_length: int,
        forecast_length: int,
        thetas_dim: int = 5,
        share_weights: bool = False,
        activation: str = "ReLU",
        active_g: bool = False,
        latent_dim: int = 5,
    ):
        """Bottleneck Generic Block with AERootBlock (bottleneck pre-split) backbone.

        Uses a two-stage projection through an intermediate thetas_dim bottleneck,
        equivalent to a rank-d factorization of the basis expansion matrix.

        Args:
            units (int): Width of the fully connected layers.
            backcast_length (int): Length of the historical data.
            forecast_length (int): Length of the forecast horizon.
            thetas_dim (int, optional): Bottleneck dimension. Defaults to 5.
            share_weights (bool, optional): Defaults to False.
            activation (str, optional): Defaults to 'ReLU'.
            active_g (bool, optional): If True, applies activation after expansion. Defaults to False.
            latent_dim (int, optional): Latent dimension for the AE backbone. Defaults to 5.
        """
        super(BottleneckGenericAE, self).__init__(
            backcast_length, units, activation, latent_dim=latent_dim
        )

        if share_weights:
            self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
        else:
            self.backcast_linear = nn.Linear(units, thetas_dim)
            self.forecast_linear = nn.Linear(units, thetas_dim)

        self.backcast_g = nn.Linear(thetas_dim, backcast_length, bias=False)
        self.forecast_g = nn.Linear(thetas_dim, forecast_length, bias=False)
        self.active_g = active_g

    def forward(self, x):
        x = super(BottleneckGenericAE, self).forward(x)
        theta_b = self.backcast_linear(x)
        theta_f = self.forecast_linear(x)

        backcast = self.backcast_g(theta_b)
        forecast = self.forecast_g(theta_f)

        if self.active_g:
            if self.active_g != "forecast":
                backcast = self.activation(backcast)
            if self.active_g != "backcast":
                forecast = self.activation(forecast)

        return backcast, forecast


class TrendAE(AERootBlock):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        thetas_dim,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        latent_dim=5,
    ):
        """The TrendAEBlock implements the function whose parameters are generated by the _TrendGenerator block.
        The TrendAEBlock is an AutoEncoder version of the TrendBlock where the presplit section of the network
        is an AutoEncoder.

        Args:
            units (int):
              The width of the fully connected layers in the blocks comprising the parent class Block.
            backcast_length (int):
              The length of the historical data.  It is customary to use a multiple of the forecast_length (H)orizon (2H,3H,4H,5H,...).
            forecast_length (int):
              The length of the forecast_length horizon.
            thetas_dim (int):
              The dimensionality of the _TrendGenerator polynomial.
            share_weights (bool, optional):
              If True, the inital weights of the Linear layers are shared. Defaults to False.
            activation (str, optional):
              The activation function passed to the parent class Block. Defaults to 'ReLU'.
        """
        super(TrendAE, self).__init__(
            backcast_length, units, activation, latent_dim=latent_dim
        )
        if share_weights:
            self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
        else:
            self.backcast_linear = nn.Linear(units, thetas_dim)
            self.forecast_linear = nn.Linear(units, thetas_dim)

        self.backcast_g = _TrendGenerator(thetas_dim, backcast_length)
        self.forecast_g = _TrendGenerator(thetas_dim, forecast_length)

    def forward(self, x):
        x = super(TrendAE, self).forward(x)

        # linear compression
        backcast_thetas = self.backcast_linear(x)
        forecast_thetas = self.forecast_linear(x)

        #  Vandermonde basis expansion
        backcast = self.backcast_g(backcast_thetas)
        forecast = self.forecast_g(forecast_thetas)

        return backcast, forecast


class TrendWavelet(RootBlock):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        trend_dim=4,
        wavelet_dim=16,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g=False,
        wavelet_type="db3",
        forecast_basis_dim=None,
        backcast_wavelet_type=None,
        forecast_wavelet_type=None,
    ):
        """TrendWavelet merges polynomial and orthonormal DWT basis expansions.

        Mirrors TrendWaveletAE/TrendWaveletAELG after the shared backbone, but uses
        the standard RootBlock trunk instead of an AE-style bottleneck.
        """
        super(TrendWavelet, self).__init__(backcast_length, units, activation)
        self.trend_dim = trend_dim
        self.active_g = active_g
        self.wavelet_type = wavelet_type
        self.backcast_wavelet_type = backcast_wavelet_type
        self.forecast_wavelet_type = forecast_wavelet_type

        eff_back_wavelet = (
            backcast_wavelet_type if backcast_wavelet_type is not None else wavelet_type
        )
        eff_fore_wavelet = (
            forecast_wavelet_type if forecast_wavelet_type is not None else wavelet_type
        )

        eff_back_wave = min(
            wavelet_dim, backcast_length - min(basis_offset, backcast_length - 1)
        )
        fore_wavelet_dim = (
            forecast_basis_dim if forecast_basis_dim is not None else wavelet_dim
        )
        eff_fore_wave = min(
            fore_wavelet_dim, forecast_length - min(basis_offset, forecast_length - 1)
        )

        total_backcast_dim = trend_dim + eff_back_wave
        total_forecast_dim = trend_dim + eff_fore_wave

        if share_weights and total_backcast_dim == total_forecast_dim:
            self.backcast_linear = self.forecast_linear = nn.Linear(
                units, total_backcast_dim
            )
        else:
            self.backcast_linear = nn.Linear(units, total_backcast_dim)
            self.forecast_linear = nn.Linear(units, total_forecast_dim)

        self.backcast_trend_g = _TrendGenerator(trend_dim, backcast_length)
        self.forecast_trend_g = _TrendGenerator(trend_dim, forecast_length)

        self.backcast_wavelet_g = _WaveletGeneratorV3(
            backcast_length,
            wavelet_type=eff_back_wavelet,
            basis_dim=eff_back_wave,
            basis_offset=basis_offset,
        )
        self.forecast_wavelet_g = _WaveletGeneratorV3(
            forecast_length,
            wavelet_type=eff_fore_wavelet,
            basis_dim=eff_fore_wave,
            basis_offset=basis_offset,
        )

    def forward(self, x):
        x = super(TrendWavelet, self).forward(x)

        backcast_coeffs = self.backcast_linear(x)
        forecast_coeffs = self.forecast_linear(x)

        backcast_trend_theta = backcast_coeffs[:, : self.trend_dim]
        backcast_wavelet_theta = backcast_coeffs[:, self.trend_dim :]
        forecast_trend_theta = forecast_coeffs[:, : self.trend_dim]
        forecast_wavelet_theta = forecast_coeffs[:, self.trend_dim :]

        backcast = self.backcast_trend_g(
            backcast_trend_theta
        ) + self.backcast_wavelet_g(backcast_wavelet_theta)
        forecast = self.forecast_trend_g(
            forecast_trend_theta
        ) + self.forecast_wavelet_g(forecast_wavelet_theta)

        if self.active_g:
            if self.active_g != "forecast":
                backcast = self.activation(backcast)
            if self.active_g != "backcast":
                forecast = self.activation(forecast)

        return backcast, forecast


class TrendWaveletGeneric(RootBlock):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        trend_dim=4,
        wavelet_dim=16,
        generic_dim=5,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g=False,
        wavelet_type="db3",
        forecast_basis_dim=None,
        backcast_wavelet_type=None,
        forecast_wavelet_type=None,
    ):
        """Three-way additive decomposition: trend + wavelet + learned generic basis.

        Extends TrendWavelet by adding a third parallel path — a learned (data-driven)
        basis expansion identical to BottleneckGeneric's ``backcast_g``/``forecast_g``.
        The three coefficient vectors are produced by splitting a single linear head,
        then each set is expanded through its respective basis and summed:

            output = trend_component + wavelet_component + generic_component

        The trend and wavelet bases are frozen (Vandermonde polynomials and orthonormal
        DWT respectively), while the generic basis is a trainable ``nn.Linear`` (no bias)
        providing a rank-``generic_dim`` learned factorization.

        Args:
            units (int): Width of the fully connected layers in the RootBlock backbone.
            backcast_length (int): Length of the historical input sequence.
            forecast_length (int): Length of the forecast horizon.
            trend_dim (int, optional): Polynomial degree (number of Vandermonde basis vectors).
                Defaults to 4.
            wavelet_dim (int, optional): Number of DWT basis rows. Defaults to 16.
            generic_dim (int, optional): Rank of the learned generic basis expansion.
                Defaults to 5.
            basis_offset (int, optional): Row offset into SVD-ordered DWT basis for
                frequency-band selection.  0 = lowest/smoothest.  Defaults to 0.
            share_weights (bool, optional): Share the projection head between backcast and
                forecast paths when their total coefficient dimensions match. Defaults to False.
            activation (str, optional): Activation function for the backbone. Defaults to 'ReLU'.
            active_g (bool or str, optional): Apply activation to the generic basis
                projection only (trend and wavelet projections remain linear).
                False = no activation.  True = both paths.
                'backcast'/'forecast' = single path.  Defaults to False.
            wavelet_type (str, optional): PyWavelets wavelet family string. Defaults to 'db3'.
            forecast_basis_dim (int, optional): Override for wavelet basis dim on the forecast
                path. Defaults to None (both paths use wavelet_dim).
            backcast_wavelet_type (str, optional): Override wavelet family for the backcast
                path only.  Defaults to None (uses ``wavelet_type``).
            forecast_wavelet_type (str, optional): Override wavelet family for the forecast
                path only.  Defaults to None (uses ``wavelet_type``).
        """
        super(TrendWaveletGeneric, self).__init__(backcast_length, units, activation)
        self.trend_dim = trend_dim
        self.generic_dim = generic_dim
        self.active_g = active_g
        self.wavelet_type = wavelet_type
        self.backcast_wavelet_type = backcast_wavelet_type
        self.forecast_wavelet_type = forecast_wavelet_type

        eff_back_wavelet = (
            backcast_wavelet_type if backcast_wavelet_type is not None else wavelet_type
        )
        eff_fore_wavelet = (
            forecast_wavelet_type if forecast_wavelet_type is not None else wavelet_type
        )

        eff_back_wave = min(
            wavelet_dim, backcast_length - min(basis_offset, backcast_length - 1)
        )
        fore_wavelet_dim = (
            forecast_basis_dim if forecast_basis_dim is not None else wavelet_dim
        )
        eff_fore_wave = min(
            fore_wavelet_dim, forecast_length - min(basis_offset, forecast_length - 1)
        )
        self.eff_back_wave = eff_back_wave
        self.eff_fore_wave = eff_fore_wave

        total_backcast_dim = trend_dim + eff_back_wave + generic_dim
        total_forecast_dim = trend_dim + eff_fore_wave + generic_dim

        if share_weights and total_backcast_dim == total_forecast_dim:
            self.backcast_linear = self.forecast_linear = nn.Linear(
                units, total_backcast_dim
            )
        else:
            self.backcast_linear = nn.Linear(units, total_backcast_dim)
            self.forecast_linear = nn.Linear(units, total_forecast_dim)

        # Frozen polynomial (Vandermonde) basis generators
        self.backcast_trend_g = _TrendGenerator(trend_dim, backcast_length)
        self.forecast_trend_g = _TrendGenerator(trend_dim, forecast_length)

        # Frozen orthonormal DWT basis generators
        self.backcast_wavelet_g = _WaveletGeneratorV3(
            backcast_length,
            wavelet_type=eff_back_wavelet,
            basis_dim=eff_back_wave,
            basis_offset=basis_offset,
        )
        self.forecast_wavelet_g = _WaveletGeneratorV3(
            forecast_length,
            wavelet_type=eff_fore_wavelet,
            basis_dim=eff_fore_wave,
            basis_offset=basis_offset,
        )

        # Learned (trainable) generic basis expansion — rank-d factorized matrix
        self.backcast_generic_g = nn.Linear(generic_dim, backcast_length, bias=False)
        self.forecast_generic_g = nn.Linear(generic_dim, forecast_length, bias=False)

    def forward(self, x):
        x = super(TrendWaveletGeneric, self).forward(x)

        backcast_coeffs = self.backcast_linear(x)
        forecast_coeffs = self.forecast_linear(x)

        # Three-way split: trend | wavelet | generic
        b_trend = backcast_coeffs[:, : self.trend_dim]
        b_wavelet = backcast_coeffs[
            :, self.trend_dim : self.trend_dim + self.eff_back_wave
        ]
        b_generic = backcast_coeffs[:, self.trend_dim + self.eff_back_wave :]

        f_trend = forecast_coeffs[:, : self.trend_dim]
        f_wavelet = forecast_coeffs[
            :, self.trend_dim : self.trend_dim + self.eff_fore_wave
        ]
        f_generic = forecast_coeffs[:, self.trend_dim + self.eff_fore_wave :]

        # Activate generic projection only when active_g is set
        b_generic_out = self.backcast_generic_g(b_generic)
        f_generic_out = self.forecast_generic_g(f_generic)
        if self.active_g:
            if self.active_g != "forecast":
                b_generic_out = self.activation(b_generic_out)
            if self.active_g != "backcast":
                f_generic_out = self.activation(f_generic_out)

        # Additive decomposition: structured + structured + learned
        backcast = (
            self.backcast_trend_g(b_trend)
            + self.backcast_wavelet_g(b_wavelet)
            + b_generic_out
        )
        forecast = (
            self.forecast_trend_g(f_trend)
            + self.forecast_wavelet_g(f_wavelet)
            + f_generic_out
        )

        return backcast, forecast


class TrendWaveletAE(AERootBlock):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        trend_dim=4,
        wavelet_dim=16,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g=False,
        wavelet_type="db3",
        latent_dim=5,
        forecast_basis_dim=None,
        backcast_wavelet_type=None,
        forecast_wavelet_type=None,
    ):
        """TrendWaveletAE merges polynomial (Vandermonde) and orthonormal DWT basis expansions
        into a single block with an AutoEncoder backbone (Option A — parallel basis, additive output).

        The AERootBlock backbone compresses the input into a latent representation and decodes
        it back to `units` width.  A single linear projection head per path maps the decoded
        representation to concatenated trend + wavelet coefficient vectors, which are split and
        expanded through their respective frozen basis matrices.  The two components are summed
        to produce the final backcast/forecast.

        Args:
            units (int): Width of the fully connected layers in the AE backbone.
            backcast_length (int): Length of the historical input sequence.
            forecast_length (int): Length of the forecast horizon.
            trend_dim (int, optional): Polynomial degree (number of Vandermonde basis vectors).
                Defaults to 4.
            wavelet_dim (int, optional): Number of DWT basis rows. Defaults to 16.
            basis_offset (int, optional): Row offset into SVD-ordered DWT basis for
                frequency-band selection.  0 = lowest/smoothest.  Defaults to 0.
            share_weights (bool, optional): Share the projection head between backcast and
                forecast paths when their total coefficient dimensions match. Defaults to False.
            activation (str, optional): Activation function for the AE backbone. Defaults to 'ReLU'.
            active_g (bool or str, optional): Apply activation after basis expansion.
                False = no activation.  True = both paths.
                'backcast'/'forecast' = single path.  Defaults to False.
            wavelet_type (str, optional): PyWavelets wavelet family string. Defaults to 'db3'.
            latent_dim (int, optional): Dimensionality of the AE bottleneck. Defaults to 5.
            forecast_basis_dim (int, optional): Override for wavelet basis dim on the forecast
                path.  When set the forecast wavelet head uses this value (clamped to
                forecast_length) while the backcast path continues to use wavelet_dim.
                Defaults to None (both paths use wavelet_dim).
            backcast_wavelet_type (str, optional): Override wavelet family for the backcast
                path only.  Defaults to None (uses ``wavelet_type``).
            forecast_wavelet_type (str, optional): Override wavelet family for the forecast
                path only.  Defaults to None (uses ``wavelet_type``).
        """
        super(TrendWaveletAE, self).__init__(
            backcast_length, units, activation, latent_dim=latent_dim
        )
        self.trend_dim = trend_dim
        self.active_g = active_g
        self.wavelet_type = wavelet_type
        self.backcast_wavelet_type = backcast_wavelet_type
        self.forecast_wavelet_type = forecast_wavelet_type

        # Resolve effective wavelet family per path (path-specific overrides fallback)
        eff_back_wavelet = (
            backcast_wavelet_type if backcast_wavelet_type is not None else wavelet_type
        )
        eff_fore_wavelet = (
            forecast_wavelet_type if forecast_wavelet_type is not None else wavelet_type
        )

        # Effective wavelet dimension for each path (clamped to available rows after offset)
        eff_back_wave = min(
            wavelet_dim, backcast_length - min(basis_offset, backcast_length - 1)
        )
        fore_wavelet_dim = (
            forecast_basis_dim if forecast_basis_dim is not None else wavelet_dim
        )
        eff_fore_wave = min(
            fore_wavelet_dim, forecast_length - min(basis_offset, forecast_length - 1)
        )

        # Total coefficient dims: trend + wavelet for each path
        total_backcast_dim = trend_dim + eff_back_wave
        total_forecast_dim = trend_dim + eff_fore_wave

        if share_weights and total_backcast_dim == total_forecast_dim:
            self.backcast_linear = self.forecast_linear = nn.Linear(
                units, total_backcast_dim
            )
        else:
            self.backcast_linear = nn.Linear(units, total_backcast_dim)
            self.forecast_linear = nn.Linear(units, total_forecast_dim)

        # Frozen polynomial (Vandermonde) basis generators
        self.backcast_trend_g = _TrendGenerator(trend_dim, backcast_length)
        self.forecast_trend_g = _TrendGenerator(trend_dim, forecast_length)

        # Frozen orthonormal DWT basis generators
        self.backcast_wavelet_g = _WaveletGeneratorV3(
            backcast_length,
            wavelet_type=eff_back_wavelet,
            basis_dim=eff_back_wave,
            basis_offset=basis_offset,
        )
        self.forecast_wavelet_g = _WaveletGeneratorV3(
            forecast_length,
            wavelet_type=eff_fore_wavelet,
            basis_dim=eff_fore_wave,
            basis_offset=basis_offset,
        )

    def forward(self, x):
        x = super(TrendWaveletAE, self).forward(x)

        # Project AE output to concatenated trend + wavelet coefficient vectors
        backcast_coeffs = self.backcast_linear(x)
        forecast_coeffs = self.forecast_linear(x)

        # Split coefficient vectors into trend and wavelet parts
        backcast_trend_theta = backcast_coeffs[:, : self.trend_dim]
        backcast_wavelet_theta = backcast_coeffs[:, self.trend_dim :]
        forecast_trend_theta = forecast_coeffs[:, : self.trend_dim]
        forecast_wavelet_theta = forecast_coeffs[:, self.trend_dim :]

        # Expand each through its frozen basis and sum (parallel basis, additive output)
        backcast = self.backcast_trend_g(
            backcast_trend_theta
        ) + self.backcast_wavelet_g(backcast_wavelet_theta)
        forecast = self.forecast_trend_g(
            forecast_trend_theta
        ) + self.forecast_wavelet_g(forecast_wavelet_theta)

        if self.active_g:
            if self.active_g != "forecast":
                backcast = self.activation(backcast)
            if self.active_g != "backcast":
                forecast = self.activation(forecast)

        return backcast, forecast


class TrendWaveletAELG(AERootBlockLG):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        trend_dim=4,
        wavelet_dim=16,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g=False,
        wavelet_type="db3",
        latent_dim=5,
        forecast_basis_dim=None,
        backcast_wavelet_type=None,
        forecast_wavelet_type=None,
        latent_gate_fn="sigmoid",
    ):
        """TrendWaveletAELG extends TrendWaveletAE with the Learned-Gate AE backbone.

        Identical to TrendWaveletAE in its projection heads and frozen basis expansions
        (parallel Vandermonde polynomial + orthonormal DWT, additive output), but replaces
        the plain AERootBlock backbone with AERootBlockLG.  The LG backbone adds a learnable
        ``nn.Parameter`` gate vector of size ``latent_dim``; each forward pass applies
        ``sigmoid(latent_gate) * z`` after the bottleneck, allowing the network to
        discover the effective latent dimensionality during training and selectively suppress
        uninformative latent dimensions.

        Args:
            units (int): Width of the fully connected layers in the AE backbone.
            backcast_length (int): Length of the historical input sequence.
            forecast_length (int): Length of the forecast horizon.
            trend_dim (int, optional): Polynomial degree (number of Vandermonde basis vectors).
                Defaults to 4.
            wavelet_dim (int, optional): Number of DWT basis rows. Defaults to 16.
            basis_offset (int, optional): Row offset into SVD-ordered DWT basis for
                frequency-band selection.  0 = lowest/smoothest.  Defaults to 0.
            share_weights (bool, optional): Share the projection head between backcast and
                forecast paths when their total coefficient dimensions match. Defaults to False.
            activation (str, optional): Activation function for the AE backbone. Defaults to 'ReLU'.
            active_g (bool or str, optional): Apply activation after basis expansion.
                False = no activation.  True = both paths.
                'backcast'/'forecast' = single path.  Defaults to False.
            wavelet_type (str, optional): PyWavelets wavelet family string. Defaults to 'db3'.
            latent_dim (int, optional): Dimensionality of the AE bottleneck. Defaults to 5.
            forecast_basis_dim (int, optional): Override for wavelet basis dim on the forecast
                path.  When set the forecast wavelet head uses this value (clamped to
                forecast_length) while the backcast path continues to use wavelet_dim.
                Defaults to None (both paths use wavelet_dim).
            backcast_wavelet_type (str, optional): Override wavelet family for the backcast
                path only.  Defaults to None (uses ``wavelet_type``).
            forecast_wavelet_type (str, optional): Override wavelet family for the forecast
                path only.  Defaults to None (uses ``wavelet_type``).
            latent_gate_fn (str, optional): Latent gate function name for the LG bottleneck.
                Defaults to ``'sigmoid'``.
        """
        super(TrendWaveletAELG, self).__init__(
            backcast_length,
            units,
            activation,
            latent_dim=latent_dim,
            latent_gate_fn=latent_gate_fn,
        )
        self.trend_dim = trend_dim
        self.active_g = active_g
        self.wavelet_type = wavelet_type
        self.backcast_wavelet_type = backcast_wavelet_type
        self.forecast_wavelet_type = forecast_wavelet_type

        # Resolve effective wavelet family per path (path-specific overrides fallback)
        eff_back_wavelet = (
            backcast_wavelet_type if backcast_wavelet_type is not None else wavelet_type
        )
        eff_fore_wavelet = (
            forecast_wavelet_type if forecast_wavelet_type is not None else wavelet_type
        )

        # Effective wavelet dimension for each path (clamped to available rows after offset)
        eff_back_wave = min(
            wavelet_dim, backcast_length - min(basis_offset, backcast_length - 1)
        )
        fore_wavelet_dim = (
            forecast_basis_dim if forecast_basis_dim is not None else wavelet_dim
        )
        eff_fore_wave = min(
            fore_wavelet_dim, forecast_length - min(basis_offset, forecast_length - 1)
        )

        # Total coefficient dims: trend + wavelet for each path
        total_backcast_dim = trend_dim + eff_back_wave
        total_forecast_dim = trend_dim + eff_fore_wave

        if share_weights and total_backcast_dim == total_forecast_dim:
            self.backcast_linear = self.forecast_linear = nn.Linear(
                units, total_backcast_dim
            )
        else:
            self.backcast_linear = nn.Linear(units, total_backcast_dim)
            self.forecast_linear = nn.Linear(units, total_forecast_dim)

        # Frozen polynomial (Vandermonde) basis generators
        self.backcast_trend_g = _TrendGenerator(trend_dim, backcast_length)
        self.forecast_trend_g = _TrendGenerator(trend_dim, forecast_length)

        # Frozen orthonormal DWT basis generators
        self.backcast_wavelet_g = _WaveletGeneratorV3(
            backcast_length,
            wavelet_type=eff_back_wavelet,
            basis_dim=eff_back_wave,
            basis_offset=basis_offset,
        )
        self.forecast_wavelet_g = _WaveletGeneratorV3(
            forecast_length,
            wavelet_type=eff_fore_wavelet,
            basis_dim=eff_fore_wave,
            basis_offset=basis_offset,
        )

    def forward(self, x):
        x = super(TrendWaveletAELG, self).forward(x)

        # Project LG-AE output to concatenated trend + wavelet coefficient vectors
        backcast_coeffs = self.backcast_linear(x)
        forecast_coeffs = self.forecast_linear(x)

        # Split coefficient vectors into trend and wavelet parts
        backcast_trend_theta = backcast_coeffs[:, : self.trend_dim]
        backcast_wavelet_theta = backcast_coeffs[:, self.trend_dim :]
        forecast_trend_theta = forecast_coeffs[:, : self.trend_dim]
        forecast_wavelet_theta = forecast_coeffs[:, self.trend_dim :]

        # Expand each through its frozen basis and sum (parallel basis, additive output)
        backcast = self.backcast_trend_g(
            backcast_trend_theta
        ) + self.backcast_wavelet_g(backcast_wavelet_theta)
        forecast = self.forecast_trend_g(
            forecast_trend_theta
        ) + self.forecast_wavelet_g(forecast_wavelet_theta)

        if self.active_g:
            if self.active_g != "forecast":
                backcast = self.activation(backcast)
            if self.active_g != "backcast":
                forecast = self.activation(forecast)

        return backcast, forecast


class TrendWaveletGenericAE(AERootBlock):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        trend_dim=4,
        wavelet_dim=16,
        generic_dim=5,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g=False,
        wavelet_type="db3",
        latent_dim=5,
        forecast_basis_dim=None,
        backcast_wavelet_type=None,
        forecast_wavelet_type=None,
    ):
        """Three-way additive decomposition (trend + wavelet + generic) with AE backbone.

        Extends TrendWaveletAE by adding a third parallel path — a learned (data-driven)
        basis expansion (rank-``generic_dim`` ``nn.Linear``, no bias).  The three coefficient
        vectors are produced by splitting a single linear head, expanded through their
        respective bases, and summed.

        Args:
            units (int): Width of the fully connected layers in the AE backbone.
            backcast_length (int): Length of the historical input sequence.
            forecast_length (int): Length of the forecast horizon.
            trend_dim (int, optional): Polynomial degree. Defaults to 4.
            wavelet_dim (int, optional): Number of DWT basis rows. Defaults to 16.
            generic_dim (int, optional): Rank of the learned generic basis. Defaults to 5.
            basis_offset (int, optional): DWT basis row offset. Defaults to 0.
            share_weights (bool, optional): Share projection heads when dims match. Defaults to False.
            activation (str, optional): Activation function. Defaults to 'ReLU'.
            active_g (bool or str, optional): Apply activation to the generic basis
                projection only (trend and wavelet projections remain linear).
                False = no activation.  True = both paths.
                'backcast'/'forecast' = single path.  Defaults to False.
            wavelet_type (str, optional): PyWavelets wavelet family. Defaults to 'db3'.
            latent_dim (int, optional): AE bottleneck dimensionality. Defaults to 5.
            forecast_basis_dim (int, optional): Override wavelet basis dim for forecast. Defaults to None.
            backcast_wavelet_type (str, optional): Override wavelet family for backcast. Defaults to None.
            forecast_wavelet_type (str, optional): Override wavelet family for forecast. Defaults to None.
        """
        super(TrendWaveletGenericAE, self).__init__(
            backcast_length, units, activation, latent_dim=latent_dim
        )
        self.trend_dim = trend_dim
        self.generic_dim = generic_dim
        self.active_g = active_g
        self.wavelet_type = wavelet_type
        self.backcast_wavelet_type = backcast_wavelet_type
        self.forecast_wavelet_type = forecast_wavelet_type

        eff_back_wavelet = (
            backcast_wavelet_type if backcast_wavelet_type is not None else wavelet_type
        )
        eff_fore_wavelet = (
            forecast_wavelet_type if forecast_wavelet_type is not None else wavelet_type
        )

        eff_back_wave = min(
            wavelet_dim, backcast_length - min(basis_offset, backcast_length - 1)
        )
        fore_wavelet_dim = (
            forecast_basis_dim if forecast_basis_dim is not None else wavelet_dim
        )
        eff_fore_wave = min(
            fore_wavelet_dim, forecast_length - min(basis_offset, forecast_length - 1)
        )
        self.eff_back_wave = eff_back_wave
        self.eff_fore_wave = eff_fore_wave

        total_backcast_dim = trend_dim + eff_back_wave + generic_dim
        total_forecast_dim = trend_dim + eff_fore_wave + generic_dim

        if share_weights and total_backcast_dim == total_forecast_dim:
            self.backcast_linear = self.forecast_linear = nn.Linear(
                units, total_backcast_dim
            )
        else:
            self.backcast_linear = nn.Linear(units, total_backcast_dim)
            self.forecast_linear = nn.Linear(units, total_forecast_dim)

        self.backcast_trend_g = _TrendGenerator(trend_dim, backcast_length)
        self.forecast_trend_g = _TrendGenerator(trend_dim, forecast_length)

        self.backcast_wavelet_g = _WaveletGeneratorV3(
            backcast_length,
            wavelet_type=eff_back_wavelet,
            basis_dim=eff_back_wave,
            basis_offset=basis_offset,
        )
        self.forecast_wavelet_g = _WaveletGeneratorV3(
            forecast_length,
            wavelet_type=eff_fore_wavelet,
            basis_dim=eff_fore_wave,
            basis_offset=basis_offset,
        )

        self.backcast_generic_g = nn.Linear(generic_dim, backcast_length, bias=False)
        self.forecast_generic_g = nn.Linear(generic_dim, forecast_length, bias=False)

    def forward(self, x):
        x = super(TrendWaveletGenericAE, self).forward(x)

        backcast_coeffs = self.backcast_linear(x)
        forecast_coeffs = self.forecast_linear(x)

        b_trend = backcast_coeffs[:, : self.trend_dim]
        b_wavelet = backcast_coeffs[
            :, self.trend_dim : self.trend_dim + self.eff_back_wave
        ]
        b_generic = backcast_coeffs[:, self.trend_dim + self.eff_back_wave :]

        f_trend = forecast_coeffs[:, : self.trend_dim]
        f_wavelet = forecast_coeffs[
            :, self.trend_dim : self.trend_dim + self.eff_fore_wave
        ]
        f_generic = forecast_coeffs[:, self.trend_dim + self.eff_fore_wave :]

        b_generic_out = self.backcast_generic_g(b_generic)
        f_generic_out = self.forecast_generic_g(f_generic)
        if self.active_g:
            if self.active_g != "forecast":
                b_generic_out = self.activation(b_generic_out)
            if self.active_g != "backcast":
                f_generic_out = self.activation(f_generic_out)

        backcast = (
            self.backcast_trend_g(b_trend)
            + self.backcast_wavelet_g(b_wavelet)
            + b_generic_out
        )
        forecast = (
            self.forecast_trend_g(f_trend)
            + self.forecast_wavelet_g(f_wavelet)
            + f_generic_out
        )

        return backcast, forecast


class TrendWaveletGenericAELG(AERootBlockLG):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        trend_dim=4,
        wavelet_dim=16,
        generic_dim=5,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g=False,
        wavelet_type="db3",
        latent_dim=5,
        forecast_basis_dim=None,
        backcast_wavelet_type=None,
        forecast_wavelet_type=None,
        latent_gate_fn="sigmoid",
    ):
        """Three-way additive decomposition (trend + wavelet + generic) with Learned-Gate AE backbone.

        Extends TrendWaveletAELG by adding a third parallel path — a learned generic basis.

        Args: same as TrendWaveletGenericAE.
        """
        super(TrendWaveletGenericAELG, self).__init__(
            backcast_length,
            units,
            activation,
            latent_dim=latent_dim,
            latent_gate_fn=latent_gate_fn,
        )
        self.trend_dim = trend_dim
        self.generic_dim = generic_dim
        self.active_g = active_g
        self.wavelet_type = wavelet_type
        self.backcast_wavelet_type = backcast_wavelet_type
        self.forecast_wavelet_type = forecast_wavelet_type

        eff_back_wavelet = (
            backcast_wavelet_type if backcast_wavelet_type is not None else wavelet_type
        )
        eff_fore_wavelet = (
            forecast_wavelet_type if forecast_wavelet_type is not None else wavelet_type
        )

        eff_back_wave = min(
            wavelet_dim, backcast_length - min(basis_offset, backcast_length - 1)
        )
        fore_wavelet_dim = (
            forecast_basis_dim if forecast_basis_dim is not None else wavelet_dim
        )
        eff_fore_wave = min(
            fore_wavelet_dim, forecast_length - min(basis_offset, forecast_length - 1)
        )
        self.eff_back_wave = eff_back_wave
        self.eff_fore_wave = eff_fore_wave

        total_backcast_dim = trend_dim + eff_back_wave + generic_dim
        total_forecast_dim = trend_dim + eff_fore_wave + generic_dim

        if share_weights and total_backcast_dim == total_forecast_dim:
            self.backcast_linear = self.forecast_linear = nn.Linear(
                units, total_backcast_dim
            )
        else:
            self.backcast_linear = nn.Linear(units, total_backcast_dim)
            self.forecast_linear = nn.Linear(units, total_forecast_dim)

        self.backcast_trend_g = _TrendGenerator(trend_dim, backcast_length)
        self.forecast_trend_g = _TrendGenerator(trend_dim, forecast_length)

        self.backcast_wavelet_g = _WaveletGeneratorV3(
            backcast_length,
            wavelet_type=eff_back_wavelet,
            basis_dim=eff_back_wave,
            basis_offset=basis_offset,
        )
        self.forecast_wavelet_g = _WaveletGeneratorV3(
            forecast_length,
            wavelet_type=eff_fore_wavelet,
            basis_dim=eff_fore_wave,
            basis_offset=basis_offset,
        )

        self.backcast_generic_g = nn.Linear(generic_dim, backcast_length, bias=False)
        self.forecast_generic_g = nn.Linear(generic_dim, forecast_length, bias=False)

    def forward(self, x):
        x = super(TrendWaveletGenericAELG, self).forward(x)

        backcast_coeffs = self.backcast_linear(x)
        forecast_coeffs = self.forecast_linear(x)

        b_trend = backcast_coeffs[:, : self.trend_dim]
        b_wavelet = backcast_coeffs[
            :, self.trend_dim : self.trend_dim + self.eff_back_wave
        ]
        b_generic = backcast_coeffs[:, self.trend_dim + self.eff_back_wave :]

        f_trend = forecast_coeffs[:, : self.trend_dim]
        f_wavelet = forecast_coeffs[
            :, self.trend_dim : self.trend_dim + self.eff_fore_wave
        ]
        f_generic = forecast_coeffs[:, self.trend_dim + self.eff_fore_wave :]

        b_generic_out = self.backcast_generic_g(b_generic)
        f_generic_out = self.forecast_generic_g(f_generic)
        if self.active_g:
            if self.active_g != "forecast":
                b_generic_out = self.activation(b_generic_out)
            if self.active_g != "backcast":
                f_generic_out = self.activation(f_generic_out)

        backcast = (
            self.backcast_trend_g(b_trend)
            + self.backcast_wavelet_g(b_wavelet)
            + b_generic_out
        )
        forecast = (
            self.forecast_trend_g(f_trend)
            + self.forecast_wavelet_g(f_wavelet)
            + f_generic_out
        )

        return backcast, forecast


class TrendWaveletGenericVAE(AERootBlockVAE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        trend_dim=4,
        wavelet_dim=16,
        generic_dim=5,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g=False,
        wavelet_type="db3",
        latent_dim=5,
        forecast_basis_dim=None,
        backcast_wavelet_type=None,
        forecast_wavelet_type=None,
    ):
        """Three-way additive decomposition (trend + wavelet + generic) with VAE backbone.

        Extends TrendWaveletGenericAE with the variational AE backbone (reparameterization
        trick). KL loss is stored in ``self.kl_loss`` and collected by ``NBeatsNet.training_step()``.

        Args: same as TrendWaveletGenericAE.
        """
        super(TrendWaveletGenericVAE, self).__init__(
            backcast_length, units, activation, latent_dim=latent_dim
        )
        self.trend_dim = trend_dim
        self.generic_dim = generic_dim
        self.active_g = active_g
        self.wavelet_type = wavelet_type
        self.backcast_wavelet_type = backcast_wavelet_type
        self.forecast_wavelet_type = forecast_wavelet_type

        eff_back_wavelet = (
            backcast_wavelet_type if backcast_wavelet_type is not None else wavelet_type
        )
        eff_fore_wavelet = (
            forecast_wavelet_type if forecast_wavelet_type is not None else wavelet_type
        )

        eff_back_wave = min(
            wavelet_dim, backcast_length - min(basis_offset, backcast_length - 1)
        )
        fore_wavelet_dim = (
            forecast_basis_dim if forecast_basis_dim is not None else wavelet_dim
        )
        eff_fore_wave = min(
            fore_wavelet_dim, forecast_length - min(basis_offset, forecast_length - 1)
        )
        self.eff_back_wave = eff_back_wave
        self.eff_fore_wave = eff_fore_wave

        total_backcast_dim = trend_dim + eff_back_wave + generic_dim
        total_forecast_dim = trend_dim + eff_fore_wave + generic_dim

        if share_weights and total_backcast_dim == total_forecast_dim:
            self.backcast_linear = self.forecast_linear = nn.Linear(
                units, total_backcast_dim
            )
        else:
            self.backcast_linear = nn.Linear(units, total_backcast_dim)
            self.forecast_linear = nn.Linear(units, total_forecast_dim)

        self.backcast_trend_g = _TrendGenerator(trend_dim, backcast_length)
        self.forecast_trend_g = _TrendGenerator(trend_dim, forecast_length)

        self.backcast_wavelet_g = _WaveletGeneratorV3(
            backcast_length,
            wavelet_type=eff_back_wavelet,
            basis_dim=eff_back_wave,
            basis_offset=basis_offset,
        )
        self.forecast_wavelet_g = _WaveletGeneratorV3(
            forecast_length,
            wavelet_type=eff_fore_wavelet,
            basis_dim=eff_fore_wave,
            basis_offset=basis_offset,
        )

        self.backcast_generic_g = nn.Linear(generic_dim, backcast_length, bias=False)
        self.forecast_generic_g = nn.Linear(generic_dim, forecast_length, bias=False)

    def forward(self, x):
        x = super(TrendWaveletGenericVAE, self).forward(x)

        backcast_coeffs = self.backcast_linear(x)
        forecast_coeffs = self.forecast_linear(x)

        b_trend = backcast_coeffs[:, : self.trend_dim]
        b_wavelet = backcast_coeffs[
            :, self.trend_dim : self.trend_dim + self.eff_back_wave
        ]
        b_generic = backcast_coeffs[:, self.trend_dim + self.eff_back_wave :]

        f_trend = forecast_coeffs[:, : self.trend_dim]
        f_wavelet = forecast_coeffs[
            :, self.trend_dim : self.trend_dim + self.eff_fore_wave
        ]
        f_generic = forecast_coeffs[:, self.trend_dim + self.eff_fore_wave :]

        b_generic_out = self.backcast_generic_g(b_generic)
        f_generic_out = self.forecast_generic_g(f_generic)
        if self.active_g:
            if self.active_g != "forecast":
                b_generic_out = self.activation(b_generic_out)
            if self.active_g != "backcast":
                f_generic_out = self.activation(f_generic_out)

        backcast = (
            self.backcast_trend_g(b_trend)
            + self.backcast_wavelet_g(b_wavelet)
            + b_generic_out
        )
        forecast = (
            self.forecast_trend_g(f_trend)
            + self.forecast_wavelet_g(f_wavelet)
            + f_generic_out
        )

        return backcast, forecast


class SeasonalityAE(AERootBlock):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        thetas_dim=5,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        latent_dim=5,
    ):
        """The SeasonalityAEBlock is the basic building block of the N-BEATS network.
        It consists of a stack of fully connected layers defined the parent class Block,
        followed by a linear layer, which generates the parameters of a Fourier expansion.

        The SeasonalityAEBlock is an AutoEncoder version of the SeasonalityBlock where the presplit
        section of the network is an AutoEncoder.

        Args:
            units (int):
              The width of the fully connected layers in the blocks comprising the parent
              class Block.
            backcast_length (int):
              The length of the historical data.  It is customary to use a multiple of
              the forecast_length (H)orizon (2H,3H,4H,5H,...).
            forecast_length (int):
              The length of the forecast_length horizon.
            activation (str, optional):
              The activation function passed to the parent class Block. Defaults to 'LeakyReLU'.
        """
        super(SeasonalityAE, self).__init__(
            backcast_length, units, activation, latent_dim=latent_dim
        )

        self.backcast_linear = nn.Linear(
            units, 2 * int(backcast_length / 2 - 1) + 1, bias=False
        )
        self.forecast_linear = nn.Linear(
            units, 2 * int(forecast_length / 2 - 1) + 1, bias=False
        )

        self.backcast_g = _SeasonalityGenerator(backcast_length)
        self.forecast_g = _SeasonalityGenerator(forecast_length)

    def forward(self, x):
        x = super(SeasonalityAE, self).forward(x)
        # linear compression
        backcast_thetas = self.backcast_linear(x)
        forecast_thetas = self.forecast_linear(x)

        # fourier expansion
        backcast = self.backcast_g(backcast_thetas)
        forecast = self.forecast_g(forecast_thetas)

        return backcast, forecast


# ---------------------------------------------------------------------------
# Learned-Gate (LG) AE Subclasses — AERootBlockLG backbone
# ---------------------------------------------------------------------------


class GenericAELG(AERootBlockLG):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        thetas_dim=5,
        share_weights=False,
        activation="ReLU",
        active_g=False,
        latent_dim=5,
        latent_gate_fn="sigmoid",
    ):
        super(GenericAELG, self).__init__(
            backcast_length,
            units,
            activation,
            latent_dim=latent_dim,
            latent_gate_fn=latent_gate_fn,
        )
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.active_g = active_g
        self.theta_b_fc = nn.Linear(units, backcast_length, bias=False)
        self.theta_f_fc = nn.Linear(units, forecast_length, bias=False)

    def forward(self, x):
        x = super(GenericAELG, self).forward(x)
        backcast = self.theta_b_fc(x)
        forecast = self.theta_f_fc(x)
        if self.active_g:
            if self.active_g != "forecast":
                backcast = self.activation(backcast)
            if self.active_g != "backcast":
                forecast = self.activation(forecast)
        return backcast, forecast


class BottleneckGenericAELG(AERootBlockLG):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        thetas_dim=5,
        share_weights=False,
        activation="ReLU",
        active_g=False,
        latent_dim=5,
        latent_gate_fn="sigmoid",
    ):
        super(BottleneckGenericAELG, self).__init__(
            backcast_length,
            units,
            activation,
            latent_dim=latent_dim,
            latent_gate_fn=latent_gate_fn,
        )
        if share_weights:
            self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
        else:
            self.backcast_linear = nn.Linear(units, thetas_dim)
            self.forecast_linear = nn.Linear(units, thetas_dim)
        self.backcast_g = nn.Linear(thetas_dim, backcast_length, bias=False)
        self.forecast_g = nn.Linear(thetas_dim, forecast_length, bias=False)
        self.active_g = active_g

    def forward(self, x):
        x = super(BottleneckGenericAELG, self).forward(x)
        theta_b = self.backcast_linear(x)
        theta_f = self.forecast_linear(x)
        backcast = self.backcast_g(theta_b)
        forecast = self.forecast_g(theta_f)
        if self.active_g:
            if self.active_g != "forecast":
                backcast = self.activation(backcast)
            if self.active_g != "backcast":
                forecast = self.activation(forecast)
        return backcast, forecast


class TrendAELG(AERootBlockLG):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        thetas_dim,
        share_weights=False,
        activation="ReLU",
        active_g=False,
        latent_dim=5,
        latent_gate_fn="sigmoid",
    ):
        super(TrendAELG, self).__init__(
            backcast_length,
            units,
            activation,
            latent_dim=latent_dim,
            latent_gate_fn=latent_gate_fn,
        )
        if share_weights:
            self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
        else:
            self.backcast_linear = nn.Linear(units, thetas_dim)
            self.forecast_linear = nn.Linear(units, thetas_dim)
        self.backcast_g = _TrendGenerator(thetas_dim, backcast_length)
        self.forecast_g = _TrendGenerator(thetas_dim, forecast_length)

    def forward(self, x):
        x = super(TrendAELG, self).forward(x)
        backcast_thetas = self.backcast_linear(x)
        forecast_thetas = self.forecast_linear(x)
        backcast = self.backcast_g(backcast_thetas)
        forecast = self.forecast_g(forecast_thetas)
        return backcast, forecast


class SeasonalityAELG(AERootBlockLG):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        thetas_dim=5,
        share_weights=False,
        activation="ReLU",
        active_g=False,
        latent_dim=5,
        latent_gate_fn="sigmoid",
    ):
        super(SeasonalityAELG, self).__init__(
            backcast_length,
            units,
            activation,
            latent_dim=latent_dim,
            latent_gate_fn=latent_gate_fn,
        )
        self.backcast_linear = nn.Linear(
            units, 2 * int(backcast_length / 2 - 1) + 1, bias=False
        )
        self.forecast_linear = nn.Linear(
            units, 2 * int(forecast_length / 2 - 1) + 1, bias=False
        )
        self.backcast_g = _SeasonalityGenerator(backcast_length)
        self.forecast_g = _SeasonalityGenerator(forecast_length)

    def forward(self, x):
        x = super(SeasonalityAELG, self).forward(x)
        backcast_thetas = self.backcast_linear(x)
        forecast_thetas = self.forecast_linear(x)
        backcast = self.backcast_g(backcast_thetas)
        forecast = self.forecast_g(forecast_thetas)
        return backcast, forecast


class AutoEncoderAELG(AERootBlockLG):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        thetas_dim,
        share_weights=False,
        activation="ReLU",
        active_g=False,
        latent_dim=5,
        latent_gate_fn="sigmoid",
    ):
        super(AutoEncoderAELG, self).__init__(
            backcast_length,
            units,
            activation,
            latent_dim=latent_dim,
            latent_gate_fn=latent_gate_fn,
        )
        self.units = units
        self.thetas_dim = thetas_dim
        self.share_weights = share_weights
        self.activation = getattr(nn, activation)()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.active_g = active_g
        branch_latent_dim = latent_dim
        if share_weights:
            self.b_encoder = self.f_encoder = nn.Linear(units, branch_latent_dim)
        else:
            self.b_encoder = nn.Linear(units, branch_latent_dim)
            self.f_encoder = nn.Linear(units, branch_latent_dim)
        self.b_decoder = nn.Linear(branch_latent_dim, units)
        self.f_decoder = nn.Linear(branch_latent_dim, units)
        self.backcast_g = nn.Linear(units, backcast_length)
        self.forecast_g = nn.Linear(units, forecast_length)

    def forward(self, x):
        x = super(AutoEncoderAELG, self).forward(x)
        b = self.activation(self.b_encoder(x))
        b = self.activation(self.b_decoder(b))
        b = self.backcast_g(b)
        f = self.activation(self.f_encoder(x))
        f = self.activation(self.f_decoder(f))
        f = self.forecast_g(f)
        if self.active_g:
            if self.active_g != "forecast":
                b = self.activation(b)
            if self.active_g != "backcast":
                f = self.activation(f)
        return b, f


class GenericAEBackcastAELG(AERootBlockLG):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        thetas_dim,
        share_weights=False,
        activation="ReLU",
        active_g=False,
        latent_dim=5,
        latent_gate_fn="sigmoid",
    ):
        super(GenericAEBackcastAELG, self).__init__(
            backcast_length,
            units,
            activation,
            latent_dim=latent_dim,
            latent_gate_fn=latent_gate_fn,
        )
        self.units = units
        self.thetas_dim = thetas_dim
        self.share_weights = share_weights
        self.activation = getattr(nn, activation)()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.active_g = active_g
        branch_latent_dim = latent_dim
        self.forecast_linear = nn.Linear(units, thetas_dim)
        self.forecast_g = nn.Linear(thetas_dim, forecast_length, bias=False)
        self.b_encoder = nn.Linear(units, branch_latent_dim)
        self.b_decoder = nn.Linear(branch_latent_dim, units)
        self.backcast_g = nn.Linear(units, backcast_length)

    def forward(self, x):
        x = super(GenericAEBackcastAELG, self).forward(x)
        b = self.activation(self.b_encoder(x))
        b = self.activation(self.b_decoder(b))
        b = self.backcast_g(b)
        theta_f = self.forecast_linear(x)
        f = self.forecast_g(theta_f)
        if self.active_g:
            if self.active_g != "forecast":
                b = self.activation(b)
            if self.active_g != "backcast":
                f = self.activation(f)
        return b, f


# ---------------------------------------------------------------------------
# Variational AE (VAE) Subclasses — AERootBlockVAE backbone
# ---------------------------------------------------------------------------


class GenericVAE(AERootBlockVAE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        thetas_dim=5,
        share_weights=False,
        activation="ReLU",
        active_g=False,
        latent_dim=5,
    ):
        super(GenericVAE, self).__init__(
            backcast_length, units, activation, latent_dim=latent_dim
        )
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.active_g = active_g
        self.theta_b_fc = nn.Linear(units, backcast_length, bias=False)
        self.theta_f_fc = nn.Linear(units, forecast_length, bias=False)

    def forward(self, x):
        x = super(GenericVAE, self).forward(x)
        backcast = self.theta_b_fc(x)
        forecast = self.theta_f_fc(x)
        if self.active_g:
            if self.active_g != "forecast":
                backcast = self.activation(backcast)
            if self.active_g != "backcast":
                forecast = self.activation(forecast)
        return backcast, forecast


class BottleneckGenericVAE(AERootBlockVAE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        thetas_dim=5,
        share_weights=False,
        activation="ReLU",
        active_g=False,
        latent_dim=5,
    ):
        super(BottleneckGenericVAE, self).__init__(
            backcast_length, units, activation, latent_dim=latent_dim
        )
        if share_weights:
            self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
        else:
            self.backcast_linear = nn.Linear(units, thetas_dim)
            self.forecast_linear = nn.Linear(units, thetas_dim)
        self.backcast_g = nn.Linear(thetas_dim, backcast_length, bias=False)
        self.forecast_g = nn.Linear(thetas_dim, forecast_length, bias=False)
        self.active_g = active_g

    def forward(self, x):
        x = super(BottleneckGenericVAE, self).forward(x)
        theta_b = self.backcast_linear(x)
        theta_f = self.forecast_linear(x)
        backcast = self.backcast_g(theta_b)
        forecast = self.forecast_g(theta_f)
        if self.active_g:
            if self.active_g != "forecast":
                backcast = self.activation(backcast)
            if self.active_g != "backcast":
                forecast = self.activation(forecast)
        return backcast, forecast


class TrendVAE(AERootBlockVAE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        thetas_dim,
        share_weights=False,
        activation="ReLU",
        active_g=False,
        latent_dim=5,
    ):
        super(TrendVAE, self).__init__(
            backcast_length, units, activation, latent_dim=latent_dim
        )
        if share_weights:
            self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
        else:
            self.backcast_linear = nn.Linear(units, thetas_dim)
            self.forecast_linear = nn.Linear(units, thetas_dim)
        self.backcast_g = _TrendGenerator(thetas_dim, backcast_length)
        self.forecast_g = _TrendGenerator(thetas_dim, forecast_length)

    def forward(self, x):
        x = super(TrendVAE, self).forward(x)
        backcast_thetas = self.backcast_linear(x)
        forecast_thetas = self.forecast_linear(x)
        backcast = self.backcast_g(backcast_thetas)
        forecast = self.forecast_g(forecast_thetas)
        return backcast, forecast


class SeasonalityVAE(AERootBlockVAE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        thetas_dim=5,
        share_weights=False,
        activation="ReLU",
        active_g=False,
        latent_dim=5,
    ):
        super(SeasonalityVAE, self).__init__(
            backcast_length, units, activation, latent_dim=latent_dim
        )
        self.backcast_linear = nn.Linear(
            units, 2 * int(backcast_length / 2 - 1) + 1, bias=False
        )
        self.forecast_linear = nn.Linear(
            units, 2 * int(forecast_length / 2 - 1) + 1, bias=False
        )
        self.backcast_g = _SeasonalityGenerator(backcast_length)
        self.forecast_g = _SeasonalityGenerator(forecast_length)

    def forward(self, x):
        x = super(SeasonalityVAE, self).forward(x)
        backcast_thetas = self.backcast_linear(x)
        forecast_thetas = self.forecast_linear(x)
        backcast = self.backcast_g(backcast_thetas)
        forecast = self.forecast_g(forecast_thetas)
        return backcast, forecast


class AutoEncoderVAE(AERootBlockVAE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        thetas_dim,
        share_weights=False,
        activation="ReLU",
        active_g=False,
        latent_dim=5,
    ):
        super(AutoEncoderVAE, self).__init__(
            backcast_length, units, activation, latent_dim=latent_dim
        )
        self.units = units
        self.thetas_dim = thetas_dim
        self.share_weights = share_weights
        self.activation = getattr(nn, activation)()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.active_g = active_g
        branch_latent_dim = latent_dim
        if share_weights:
            self.b_encoder = self.f_encoder = nn.Linear(units, branch_latent_dim)
        else:
            self.b_encoder = nn.Linear(units, branch_latent_dim)
            self.f_encoder = nn.Linear(units, branch_latent_dim)
        self.b_decoder = nn.Linear(branch_latent_dim, units)
        self.f_decoder = nn.Linear(branch_latent_dim, units)
        self.backcast_g = nn.Linear(units, backcast_length)
        self.forecast_g = nn.Linear(units, forecast_length)

    def forward(self, x):
        x = super(AutoEncoderVAE, self).forward(x)
        b = self.activation(self.b_encoder(x))
        b = self.activation(self.b_decoder(b))
        b = self.backcast_g(b)
        f = self.activation(self.f_encoder(x))
        f = self.activation(self.f_decoder(f))
        f = self.forecast_g(f)
        if self.active_g:
            if self.active_g != "forecast":
                b = self.activation(b)
            if self.active_g != "backcast":
                f = self.activation(f)
        return b, f


class GenericAEBackcastVAE(AERootBlockVAE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        thetas_dim,
        share_weights=False,
        activation="ReLU",
        active_g=False,
        latent_dim=5,
    ):
        super(GenericAEBackcastVAE, self).__init__(
            backcast_length, units, activation, latent_dim=latent_dim
        )
        self.units = units
        self.thetas_dim = thetas_dim
        self.share_weights = share_weights
        self.activation = getattr(nn, activation)()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.active_g = active_g
        branch_latent_dim = latent_dim
        self.forecast_linear = nn.Linear(units, thetas_dim)
        self.forecast_g = nn.Linear(thetas_dim, forecast_length, bias=False)
        self.b_encoder = nn.Linear(units, branch_latent_dim)
        self.b_decoder = nn.Linear(branch_latent_dim, units)
        self.backcast_g = nn.Linear(units, backcast_length)

    def forward(self, x):
        x = super(GenericAEBackcastVAE, self).forward(x)
        b = self.activation(self.b_encoder(x))
        b = self.activation(self.b_decoder(b))
        b = self.backcast_g(b)
        theta_f = self.forecast_linear(x)
        f = self.forecast_g(theta_f)
        if self.active_g:
            if self.active_g != "forecast":
                b = self.activation(b)
            if self.active_g != "backcast":
                f = self.activation(f)
        return b, f


# ---------------------------------------------------------------------------
# V3 Wavelet Blocks — Orthonormal DWT basis via impulse-response synthesis
#
# Root cause fix for V1/V2 instability: the rolled phi/psi basis construction
# produces ill-conditioned matrices (DB3 cond~604K). V3 replaces this entirely
# with proper DWT impulse-response synthesis + SVD orthogonalization, producing
# a genuinely orthonormal basis where ALL singular values = 1.0 (cond = 1.0).
#
# Architecture: 2-projection pattern matching Seasonality/Trend (no downsampling
# layer needed since the basis is always target_length x target_length).
# ---------------------------------------------------------------------------


class _WaveletGeneratorV3(nn.Module):
    def __init__(
        self,
        target_length,
        wavelet_type="db3",
        max_decomp_level=5,
        basis_dim=None,
        basis_offset=0,
    ):
        super().__init__()
        basis = self._build_basis(
            target_length, wavelet_type, max_decomp_level, basis_dim, basis_offset
        )
        self.basis = nn.Parameter(basis, requires_grad=False)

    @staticmethod
    def _build_control_basis(
        target_length, kind, basis_dim=None, basis_offset=0, max_decomp_level=5
    ):
        """Non-wavelet orthonormal control bases.

        Used to test whether the benefit of the frozen basis is wavelet-specific or a
        property of any frozen orthonormal low-rank projection. Returns the same
        (effective_dim, target_length) shape as the wavelet path, so a control run is
        parameter-identical to its wavelet counterpart by construction.

        ``kind`` is one of:

        * ``'dct'`` — orthonormal DCT-II rows, ordered low to high frequency.
        * ``'random_ortho'`` / ``'random_ortho:<seed>'`` — rows of a
          Haar-distributed random orthonormal matrix, deterministic given the seed.
        * ``'kl:<prefix>'`` — a precomputed Karhunen-Loeve (PCA) basis loaded from
          ``{prefix}_{target_length}.npy``, rows ordered by descending eigenvalue.
          This is the basis linear approximation theory names as optimal for a
          fixed d-dimensional subspace: it maximises expected captured energy for
          the ensemble it was estimated on. Unlike the other kinds it is
          data-dependent, estimated from the TRAINING split only. It is still
          linear approximation in the technical sense, since the rows are fixed
          before training and shared by every series.
        """
        if kind.startswith("ordered:"):
            # Band-ordering-preserving orthonormalisation of a DWT synthesis basis.
            #
            # The default path SVDs the raw basis and keeps rows by singular value,
            # which mixes every wavelet function together and destroys the dyadic
            # level ordering. Truncating it therefore keeps an arbitrary subspace
            # rather than the coarse scales, and basis_offset selects no frequency
            # band. This path instead builds the basis in 'periodization' mode,
            # where the synthesis operator is already square and orthonormal at
            # even lengths, and orthonormalises by QR. QR is Householder, hence
            # triangular, so span(first k rows) equals span(first k wavelet
            # functions) and the coarse-to-fine ordering survives truncation.
            #
            # Note what this does NOT rescue. Truncating mid-band drops translates,
            # and each translate covers only part of the time axis, so a localised
            # basis truncated inside a band leaves regions of the target
            # unreachable. Captured energy therefore grows roughly linearly in d
            # within a band and jumps at the boundary, whereas a global basis
            # concentrates its energy in the first few rows. That is a liability of
            # locality under FIXED truncation, and it does not exist under the per
            # signal selection that makes wavelets good in the first place.
            family = kind.split(":", 1)[1]
            # The periodization synthesis operator is square and orthonormal only
            # while every halving acts on an even length. Once a level produces an
            # odd length the frame stops spanning, QR silently returns fewer rows
            # than target_length, and the block's theta layer no longer matches
            # its basis. Measured before this guard: haar at length 24 gave 18
            # rows, at 48 gave 42, at 96 gave 88. Clamp the depth so that
            # target_length is divisible by 2**level.
            max_even_level = 0
            while target_length % (2 ** (max_even_level + 1)) == 0:
                max_even_level += 1
            level = min(
                pywt.dwt_max_level(target_length, PyWavelet(family).dec_len),
                max_decomp_level,
                max_even_level,
            )
            structure = pywt.wavedec(
                np.zeros(target_length), family, level=level, mode="periodization"
            )
            band_lengths = [len(c) for c in structure]
            rows = []
            for band_idx, band_len in enumerate(band_lengths):
                for j in range(band_len):
                    impulse = [np.zeros(l) for l in band_lengths]
                    impulse[band_idx][j] = 1.0
                    rows.append(
                        pywt.waverec(impulse, family, mode="periodization")[
                            :target_length
                        ]
                    )
            raw = np.array(rows, dtype=np.float64)
            q, r = np.linalg.qr(raw.T)
            diag = np.abs(np.diag(r))
            keep = diag > 1e-10 * max(diag[0], np.finfo(np.float64).tiny)
            ortho = q.T[keep]
            if ortho.shape[0] != target_length:
                raise ValueError(
                    f"ordered:{family} produced a rank {ortho.shape[0]} basis for "
                    f"target_length={target_length} at level {level}; it must be "
                    f"full rank. A rank deficient basis silently mismatches the "
                    f"theta layer. Use a family whose decomposition keeps every "
                    f"length even, or a global basis such as dct."
                )
            full_rank = ortho.shape[0]
            offset = min(basis_offset, full_rank - 1)
            available = full_rank - offset
            effective_dim = available if basis_dim is None else min(available, basis_dim)
            return torch.tensor(
                ortho[offset : offset + effective_dim, :], dtype=torch.float32
            )

        if kind.startswith("kl:"):
            prefix = kind.split(":", 1)[1]
            path = f"{prefix}_{target_length}.npy"
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"KL basis not found at {path}. Precompute it from the training "
                    f"split with experiments/scripts/build_kl_basis.py before "
                    f"running this config."
                )
            ortho = np.load(path)
            if ortho.shape != (target_length, target_length):
                raise ValueError(
                    f"KL basis at {path} has shape {ortho.shape}, expected "
                    f"({target_length}, {target_length})."
                )
        elif kind == "dct":
            n = np.arange(target_length, dtype=np.float64)
            k = np.arange(target_length, dtype=np.float64)[:, None]
            ortho = np.cos(np.pi * (n + 0.5) * k / target_length)
            ortho[0] *= np.sqrt(1.0 / target_length)
            ortho[1:] *= np.sqrt(2.0 / target_length)
        else:
            seed = 0
            if ":" in kind:
                seed = int(kind.split(":", 1)[1])
            rng = np.random.default_rng(seed)
            q, r = np.linalg.qr(rng.standard_normal((target_length, target_length)))
            # Sign-fix the QR so the draw is unique (Haar) and reproducible.
            ortho = (q * np.sign(np.diag(r))).T

        # These bases are full rank by construction, so full_rank == target_length.
        offset = min(basis_offset, target_length - 1)
        available = target_length - offset
        effective_dim = available if basis_dim is None else min(available, basis_dim)
        return torch.tensor(
            ortho[offset : offset + effective_dim, :], dtype=torch.float32
        )

    @staticmethod
    def _build_basis(
        target_length, wavelet_type, max_decomp_level, basis_dim=None, basis_offset=0
    ):
        import logging

        if (
            wavelet_type == "dct"
            or wavelet_type.startswith("random_ortho")
            or wavelet_type.startswith("kl:")
            or wavelet_type.startswith("ordered:")
        ):
            return _WaveletGeneratorV3._build_control_basis(
                target_length, wavelet_type, basis_dim, basis_offset, max_decomp_level
            )

        wavelet = PyWavelet(wavelet_type)
        max_level = pywt.dwt_max_level(target_length, wavelet.dec_len)
        # Respect PyWavelets level=0 for short sequences / long filters. Forcing level=1
        # triggers boundary-effect warnings without yielding a valid multilevel decomposition.
        level = min(max_level, max_decomp_level)

        # Get DWT coefficient structure
        dummy = np.zeros(target_length)
        coeffs = pywt.wavedec(dummy, wavelet_type, level=level)
        coeff_lengths = [len(c) for c in coeffs]

        # Build raw synthesis matrix via impulse responses
        basis_rows = []
        for band_idx, band_len in enumerate(coeff_lengths):
            for j in range(band_len):
                impulse = [np.zeros(l) for l in coeff_lengths]
                impulse[band_idx][j] = 1.0
                reconstructed = pywt.waverec(impulse, wavelet_type)
                basis_rows.append(reconstructed[:target_length])

        raw_basis = np.array(basis_rows, dtype=np.float64)

        # SVD orthogonalization — rows of Vt are ordered low→high frequency by singular value
        _, S, Vt = np.linalg.svd(raw_basis, full_matrices=False)
        tol = S[0] * max(raw_basis.shape) * np.finfo(np.float64).eps
        full_rank = int(np.sum(S > tol))

        if full_rank < target_length:
            logging.warning(
                f"WaveletV3 rank-deficient: {full_rank}/{target_length} for '{wavelet_type}'"
            )

        # Select a frequency band via offset + window:
        #   basis_offset=0,  basis_dim=32  → rows [0:32]   (low-frequency)
        #   basis_offset=32, basis_dim=32  → rows [32:64]  (mid-frequency)
        #   basis_offset=0,  basis_dim=None → all rows      (full spectrum)
        # Clamp so the slice never exceeds full_rank.
        offset = min(basis_offset, full_rank - 1)
        available = full_rank - offset
        effective_dim = available if basis_dim is None else min(available, basis_dim)
        ortho_basis = Vt[
            offset : offset + effective_dim, :
        ]  # (effective_dim, target_length)

        return torch.tensor(ortho_basis, dtype=torch.float32)

    def forward(self, x):
        # x: (batch, effective_dim), basis: (effective_dim, target_length)
        return torch.matmul(x, self.basis)


class WaveletV3(RootBlock):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        wavelet_type="db3",
        forecast_basis_dim=None,
        backcast_wavelet_type=None,
        forecast_wavelet_type=None,
        theta_activation: bool = False,
    ):
        """
        theta_activation : bool, optional
            Diagnostic probe, NOT part of the proposed method. Applies the block
            activation to the theta vector BETWEEN the learned projection and the
            frozen basis, giving ``B^T sigma(W h)`` instead of ``B^T W h``.

            This is the one placement that breaks the rotation-absorption
            argument. At complete rank the frozen basis is square and orthogonal,
            so ``B^T W`` realises any matrix and basis identity cannot matter;
            note that ``active_g`` does NOT disturb this, because it applies its
            activation AFTER the basis, leaving ``sigma(B^T W h)`` free to
            realise any ``sigma(M h)``. Interposing the nonlinearity instead
            makes the reachable set depend on the basis, since a rotation of a
            rectified vector is generally not itself a rectified vector.

            So this flag is the falsification handle: with it off, basis identity
            must be null at complete rank; with it on, identity should start to
            matter. Default False.
        """
        super(WaveletV3, self).__init__(backcast_length, units, activation)
        self.wavelet_type = wavelet_type
        self.backcast_wavelet_type = backcast_wavelet_type
        self.forecast_wavelet_type = forecast_wavelet_type
        self.share_weights = share_weights
        self.active_g = active_g
        self.theta_activation = theta_activation

        # Resolve effective wavelet family per path
        eff_back_wavelet = (
            backcast_wavelet_type if backcast_wavelet_type is not None else wavelet_type
        )
        eff_fore_wavelet = (
            forecast_wavelet_type if forecast_wavelet_type is not None else wavelet_type
        )

        # Clamp basis_dim to the available rows after offset (can't exceed sequence length).
        # basis_dim=None means no truncation; basis_offset shifts which frequency band is used.
        eff_back_dim = (
            backcast_length - min(basis_offset, backcast_length - 1)
            if basis_dim is None
            else min(
                basis_dim, backcast_length - min(basis_offset, backcast_length - 1)
            )
        )
        fore_dim = forecast_basis_dim if forecast_basis_dim is not None else basis_dim
        eff_fore_dim = (
            forecast_length - min(basis_offset, forecast_length - 1)
            if fore_dim is None
            else min(fore_dim, forecast_length - min(basis_offset, forecast_length - 1))
        )

        self.backcast_linear = nn.Linear(units, eff_back_dim, bias=False)
        self.forecast_linear = nn.Linear(units, eff_fore_dim, bias=False)

        self.backcast_g = _WaveletGeneratorV3(
            backcast_length,
            wavelet_type=eff_back_wavelet,
            basis_dim=eff_back_dim,
            basis_offset=basis_offset,
        )
        self.forecast_g = _WaveletGeneratorV3(
            forecast_length,
            wavelet_type=eff_fore_wavelet,
            basis_dim=eff_fore_dim,
            basis_offset=basis_offset,
        )

    def forward(self, x):
        x = super(WaveletV3, self).forward(x)
        backcast_thetas = self.backcast_linear(x)
        forecast_thetas = self.forecast_linear(x)
        if self.theta_activation:
            # Nonlinearity BETWEEN projection and basis; see __init__ docstring.
            backcast_thetas = self.activation(backcast_thetas)
            forecast_thetas = self.activation(forecast_thetas)
        backcast = self.backcast_g(backcast_thetas)
        forecast = self.forecast_g(forecast_thetas)
        if self.active_g:
            if self.active_g != "forecast":
                backcast = self.activation(backcast)
            if self.active_g != "backcast":
                forecast = self.activation(forecast)
        return backcast, forecast


# --- V3 Wavelet subclasses (thin wrappers setting wavelet_type) ---


class HaarWaveletV3(WaveletV3):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
    ):
        super(HaarWaveletV3, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="haar",
            forecast_basis_dim=forecast_basis_dim,
        )

    def forward(self, x):
        return super(HaarWaveletV3, self).forward(x)


class DB2WaveletV3(WaveletV3):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
    ):
        super(DB2WaveletV3, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="db2",
            forecast_basis_dim=forecast_basis_dim,
        )

    def forward(self, x):
        return super(DB2WaveletV3, self).forward(x)


class DB3WaveletV3(WaveletV3):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
    ):
        super(DB3WaveletV3, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="db3",
            forecast_basis_dim=forecast_basis_dim,
        )

    def forward(self, x):
        return super(DB3WaveletV3, self).forward(x)


class DB4WaveletV3(WaveletV3):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
    ):
        super(DB4WaveletV3, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="db4",
            forecast_basis_dim=forecast_basis_dim,
        )

    def forward(self, x):
        return super(DB4WaveletV3, self).forward(x)


class DB10WaveletV3(WaveletV3):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
    ):
        super(DB10WaveletV3, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="db10",
            forecast_basis_dim=forecast_basis_dim,
        )

    def forward(self, x):
        return super(DB10WaveletV3, self).forward(x)


class DB20WaveletV3(WaveletV3):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
    ):
        super(DB20WaveletV3, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="db20",
            forecast_basis_dim=forecast_basis_dim,
        )

    def forward(self, x):
        return super(DB20WaveletV3, self).forward(x)


class Coif1WaveletV3(WaveletV3):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
    ):
        super(Coif1WaveletV3, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="coif1",
            forecast_basis_dim=forecast_basis_dim,
        )

    def forward(self, x):
        return super(Coif1WaveletV3, self).forward(x)


class Coif2WaveletV3(WaveletV3):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
    ):
        super(Coif2WaveletV3, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="coif2",
            forecast_basis_dim=forecast_basis_dim,
        )

    def forward(self, x):
        return super(Coif2WaveletV3, self).forward(x)


class Coif3WaveletV3(WaveletV3):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
    ):
        super(Coif3WaveletV3, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="coif3",
            forecast_basis_dim=forecast_basis_dim,
        )

    def forward(self, x):
        return super(Coif3WaveletV3, self).forward(x)


class Coif10WaveletV3(WaveletV3):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
    ):
        super(Coif10WaveletV3, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="coif10",
            forecast_basis_dim=forecast_basis_dim,
        )

    def forward(self, x):
        return super(Coif10WaveletV3, self).forward(x)


class Symlet2WaveletV3(WaveletV3):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
    ):
        super(Symlet2WaveletV3, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="sym2",
            forecast_basis_dim=forecast_basis_dim,
        )

    def forward(self, x):
        return super(Symlet2WaveletV3, self).forward(x)


class Symlet3WaveletV3(WaveletV3):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
    ):
        super(Symlet3WaveletV3, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="sym3",
            forecast_basis_dim=forecast_basis_dim,
        )

    def forward(self, x):
        return super(Symlet3WaveletV3, self).forward(x)


class Symlet10WaveletV3(WaveletV3):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
    ):
        super(Symlet10WaveletV3, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="sym10",
            forecast_basis_dim=forecast_basis_dim,
        )

    def forward(self, x):
        return super(Symlet10WaveletV3, self).forward(x)


class Symlet20WaveletV3(WaveletV3):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
    ):
        super(Symlet20WaveletV3, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="sym20",
            forecast_basis_dim=forecast_basis_dim,
        )

    def forward(self, x):
        return super(Symlet20WaveletV3, self).forward(x)


# ---------------------------------------------------------------------------
# WaveletV3AE — V3 wavelet blocks with AE bottleneck backbone (Option B)
# ---------------------------------------------------------------------------


class WaveletV3AE(AERootBlock):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        wavelet_type="db3",
        forecast_basis_dim=None,
        latent_dim=5,
        backcast_wavelet_type=None,
        forecast_wavelet_type=None,
    ):
        """WaveletV3AE: orthonormal DWT basis block with AutoEncoder backbone (Option B).

        Structurally identical to WaveletV3 but inherits from AERootBlock instead of RootBlock,
        replacing the flat 4-FC-layer backbone with an encoder-decoder bottleneck backbone
        (backcast_length → units/2 → latent_dim → units/2 → units).  This reduces backbone
        trainable parameters significantly at large `units` values while preserving the frozen
        orthonormal DWT basis expansion for the projection heads.

        Args:
            units (int): Width of the fully connected layers in the AE backbone.
            backcast_length (int): Length of the historical input sequence.
            forecast_length (int): Length of the forecast horizon.
            basis_dim (int, optional): Number of DWT basis rows. Defaults to 32.
            basis_offset (int, optional): Row offset into SVD-ordered basis. Defaults to 0.
            share_weights (bool, optional): Share projection heads when dimensions match.
                Defaults to False.
            activation (str, optional): Activation for the AE backbone. Defaults to 'ReLU'.
            active_g (bool or str, optional): Apply activation after basis expansion.
                Defaults to False.
            wavelet_type (str, optional): PyWavelets wavelet family string. Defaults to 'db3'.
            forecast_basis_dim (int, optional): Override basis_dim for forecast path only.
                Defaults to None (both paths use basis_dim).
            latent_dim (int, optional): Dimensionality of the AE bottleneck. Defaults to 5.
            backcast_wavelet_type (str, optional): Override wavelet family for the backcast
                path only.  Defaults to None (uses ``wavelet_type``).
            forecast_wavelet_type (str, optional): Override wavelet family for the forecast
                path only.  Defaults to None (uses ``wavelet_type``).
        """
        super(WaveletV3AE, self).__init__(
            backcast_length, units, activation, latent_dim=latent_dim
        )
        self.wavelet_type = wavelet_type
        self.backcast_wavelet_type = backcast_wavelet_type
        self.forecast_wavelet_type = forecast_wavelet_type
        self.share_weights = share_weights
        self.active_g = active_g

        # Resolve effective wavelet family per path
        eff_back_wavelet = (
            backcast_wavelet_type if backcast_wavelet_type is not None else wavelet_type
        )
        eff_fore_wavelet = (
            forecast_wavelet_type if forecast_wavelet_type is not None else wavelet_type
        )

        eff_back_dim = (
            backcast_length - min(basis_offset, backcast_length - 1)
            if basis_dim is None
            else min(
                basis_dim, backcast_length - min(basis_offset, backcast_length - 1)
            )
        )
        fore_dim = forecast_basis_dim if forecast_basis_dim is not None else basis_dim
        eff_fore_dim = (
            forecast_length - min(basis_offset, forecast_length - 1)
            if fore_dim is None
            else min(fore_dim, forecast_length - min(basis_offset, forecast_length - 1))
        )

        self.backcast_linear = nn.Linear(units, eff_back_dim, bias=False)
        self.forecast_linear = nn.Linear(units, eff_fore_dim, bias=False)

        self.backcast_g = _WaveletGeneratorV3(
            backcast_length,
            wavelet_type=eff_back_wavelet,
            basis_dim=eff_back_dim,
            basis_offset=basis_offset,
        )
        self.forecast_g = _WaveletGeneratorV3(
            forecast_length,
            wavelet_type=eff_fore_wavelet,
            basis_dim=eff_fore_dim,
            basis_offset=basis_offset,
        )

    def forward(self, x):
        x = super(WaveletV3AE, self).forward(x)
        backcast_thetas = self.backcast_linear(x)
        forecast_thetas = self.forecast_linear(x)
        backcast = self.backcast_g(backcast_thetas)
        forecast = self.forecast_g(forecast_thetas)
        if self.active_g:
            if self.active_g != "forecast":
                backcast = self.activation(backcast)
            if self.active_g != "backcast":
                forecast = self.activation(forecast)
        return backcast, forecast


# --- V3AE Wavelet subclasses (thin wrappers setting wavelet_type) ---


class HaarWaveletV3AE(WaveletV3AE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(HaarWaveletV3AE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="haar",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(HaarWaveletV3AE, self).forward(x)


class DB2WaveletV3AE(WaveletV3AE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(DB2WaveletV3AE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="db2",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(DB2WaveletV3AE, self).forward(x)


class DB3WaveletV3AE(WaveletV3AE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(DB3WaveletV3AE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="db3",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(DB3WaveletV3AE, self).forward(x)


class DB4WaveletV3AE(WaveletV3AE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(DB4WaveletV3AE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="db4",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(DB4WaveletV3AE, self).forward(x)


class DB10WaveletV3AE(WaveletV3AE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(DB10WaveletV3AE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="db10",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(DB10WaveletV3AE, self).forward(x)


class DB20WaveletV3AE(WaveletV3AE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(DB20WaveletV3AE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="db20",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(DB20WaveletV3AE, self).forward(x)


class Coif1WaveletV3AE(WaveletV3AE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(Coif1WaveletV3AE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="coif1",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(Coif1WaveletV3AE, self).forward(x)


class Coif2WaveletV3AE(WaveletV3AE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(Coif2WaveletV3AE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="coif2",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(Coif2WaveletV3AE, self).forward(x)


class Coif3WaveletV3AE(WaveletV3AE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(Coif3WaveletV3AE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="coif3",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(Coif3WaveletV3AE, self).forward(x)


class Coif10WaveletV3AE(WaveletV3AE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(Coif10WaveletV3AE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="coif10",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(Coif10WaveletV3AE, self).forward(x)


class Symlet2WaveletV3AE(WaveletV3AE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(Symlet2WaveletV3AE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="sym2",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(Symlet2WaveletV3AE, self).forward(x)


class Symlet3WaveletV3AE(WaveletV3AE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(Symlet3WaveletV3AE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="sym3",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(Symlet3WaveletV3AE, self).forward(x)


class Symlet10WaveletV3AE(WaveletV3AE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(Symlet10WaveletV3AE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="sym10",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(Symlet10WaveletV3AE, self).forward(x)


class Symlet20WaveletV3AE(WaveletV3AE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(Symlet20WaveletV3AE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="sym20",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(Symlet20WaveletV3AE, self).forward(x)


# ---------------------------------------------------------------------------
# WaveletV3AELG — V3 wavelet blocks with Learned-Gate AE backbone
# ---------------------------------------------------------------------------


class WaveletV3AELG(AERootBlockLG):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        wavelet_type="db3",
        forecast_basis_dim=None,
        latent_dim=5,
        backcast_wavelet_type=None,
        forecast_wavelet_type=None,
        latent_gate_fn="sigmoid",
    ):
        """WaveletV3AELG: orthonormal DWT basis block with Learned-Gate AE backbone.

        Structurally identical to WaveletV3AE but inherits from AERootBlockLG instead of
        AERootBlock, adding a learnable sigmoid gate vector at the latent bottleneck that
        allows the network to discover effective latent dimensionality during training.

        Args:
            units (int): Width of the fully connected layers in the LG-AE backbone.
            backcast_length (int): Length of the historical input sequence.
            forecast_length (int): Length of the forecast horizon.
            basis_dim (int, optional): Number of DWT basis rows. Defaults to 32.
            basis_offset (int, optional): Row offset into SVD-ordered basis. Defaults to 0.
            share_weights (bool, optional): Share projection heads when dimensions match.
                Defaults to False.
            activation (str, optional): Activation for the LG-AE backbone. Defaults to 'ReLU'.
            active_g (bool or str, optional): Apply activation after basis expansion.
                Defaults to False.
            wavelet_type (str, optional): PyWavelets wavelet family string. Defaults to 'db3'.
            forecast_basis_dim (int, optional): Override basis_dim for forecast path only.
                Defaults to None (both paths use basis_dim).
            latent_dim (int, optional): Dimensionality of the LG-AE bottleneck. Defaults to 5.
            backcast_wavelet_type (str, optional): Override wavelet family for the backcast
                path only.  Defaults to None (uses ``wavelet_type``).
            forecast_wavelet_type (str, optional): Override wavelet family for the forecast
                path only.  Defaults to None (uses ``wavelet_type``).
            latent_gate_fn (str, optional): Latent gate function name for the LG bottleneck.
                Defaults to ``'sigmoid'``.
        """
        super(WaveletV3AELG, self).__init__(
            backcast_length,
            units,
            activation,
            latent_dim=latent_dim,
            latent_gate_fn=latent_gate_fn,
        )
        self.wavelet_type = wavelet_type
        self.backcast_wavelet_type = backcast_wavelet_type
        self.forecast_wavelet_type = forecast_wavelet_type
        self.share_weights = share_weights
        self.active_g = active_g

        # Resolve effective wavelet family per path
        eff_back_wavelet = (
            backcast_wavelet_type if backcast_wavelet_type is not None else wavelet_type
        )
        eff_fore_wavelet = (
            forecast_wavelet_type if forecast_wavelet_type is not None else wavelet_type
        )

        eff_back_dim = (
            backcast_length - min(basis_offset, backcast_length - 1)
            if basis_dim is None
            else min(
                basis_dim, backcast_length - min(basis_offset, backcast_length - 1)
            )
        )
        fore_dim = forecast_basis_dim if forecast_basis_dim is not None else basis_dim
        eff_fore_dim = (
            forecast_length - min(basis_offset, forecast_length - 1)
            if fore_dim is None
            else min(fore_dim, forecast_length - min(basis_offset, forecast_length - 1))
        )

        self.backcast_linear = nn.Linear(units, eff_back_dim, bias=False)
        self.forecast_linear = nn.Linear(units, eff_fore_dim, bias=False)

        self.backcast_g = _WaveletGeneratorV3(
            backcast_length,
            wavelet_type=eff_back_wavelet,
            basis_dim=eff_back_dim,
            basis_offset=basis_offset,
        )
        self.forecast_g = _WaveletGeneratorV3(
            forecast_length,
            wavelet_type=eff_fore_wavelet,
            basis_dim=eff_fore_dim,
            basis_offset=basis_offset,
        )

    def forward(self, x):
        x = super(WaveletV3AELG, self).forward(x)
        backcast_thetas = self.backcast_linear(x)
        forecast_thetas = self.forecast_linear(x)
        backcast = self.backcast_g(backcast_thetas)
        forecast = self.forecast_g(forecast_thetas)
        if self.active_g:
            if self.active_g != "forecast":
                backcast = self.activation(backcast)
            if self.active_g != "backcast":
                forecast = self.activation(forecast)
        return backcast, forecast


# --- V3AELG Wavelet subclasses (thin wrappers setting wavelet_type) ---


class HaarWaveletV3AELG(WaveletV3AELG):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
        latent_gate_fn="sigmoid",
    ):
        super(HaarWaveletV3AELG, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="haar",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
            latent_gate_fn=latent_gate_fn,
        )

    def forward(self, x):
        return super(HaarWaveletV3AELG, self).forward(x)


class DB2WaveletV3AELG(WaveletV3AELG):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
        latent_gate_fn="sigmoid",
    ):
        super(DB2WaveletV3AELG, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="db2",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
            latent_gate_fn=latent_gate_fn,
        )

    def forward(self, x):
        return super(DB2WaveletV3AELG, self).forward(x)


class DB3WaveletV3AELG(WaveletV3AELG):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
        latent_gate_fn="sigmoid",
    ):
        super(DB3WaveletV3AELG, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="db3",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
            latent_gate_fn=latent_gate_fn,
        )

    def forward(self, x):
        return super(DB3WaveletV3AELG, self).forward(x)


class DB4WaveletV3AELG(WaveletV3AELG):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
        latent_gate_fn="sigmoid",
    ):
        super(DB4WaveletV3AELG, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="db4",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
            latent_gate_fn=latent_gate_fn,
        )

    def forward(self, x):
        return super(DB4WaveletV3AELG, self).forward(x)


class DB10WaveletV3AELG(WaveletV3AELG):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
        latent_gate_fn="sigmoid",
    ):
        super(DB10WaveletV3AELG, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="db10",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
            latent_gate_fn=latent_gate_fn,
        )

    def forward(self, x):
        return super(DB10WaveletV3AELG, self).forward(x)


class DB20WaveletV3AELG(WaveletV3AELG):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
        latent_gate_fn="sigmoid",
    ):
        super(DB20WaveletV3AELG, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="db20",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
            latent_gate_fn=latent_gate_fn,
        )

    def forward(self, x):
        return super(DB20WaveletV3AELG, self).forward(x)


class Coif1WaveletV3AELG(WaveletV3AELG):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
        latent_gate_fn="sigmoid",
    ):
        super(Coif1WaveletV3AELG, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="coif1",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
            latent_gate_fn=latent_gate_fn,
        )

    def forward(self, x):
        return super(Coif1WaveletV3AELG, self).forward(x)


class Coif2WaveletV3AELG(WaveletV3AELG):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
        latent_gate_fn="sigmoid",
    ):
        super(Coif2WaveletV3AELG, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="coif2",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
            latent_gate_fn=latent_gate_fn,
        )

    def forward(self, x):
        return super(Coif2WaveletV3AELG, self).forward(x)


class Coif3WaveletV3AELG(WaveletV3AELG):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
        latent_gate_fn="sigmoid",
    ):
        super(Coif3WaveletV3AELG, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="coif3",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
            latent_gate_fn=latent_gate_fn,
        )

    def forward(self, x):
        return super(Coif3WaveletV3AELG, self).forward(x)


class Coif10WaveletV3AELG(WaveletV3AELG):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
        latent_gate_fn="sigmoid",
    ):
        super(Coif10WaveletV3AELG, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="coif10",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
            latent_gate_fn=latent_gate_fn,
        )

    def forward(self, x):
        return super(Coif10WaveletV3AELG, self).forward(x)


class Symlet2WaveletV3AELG(WaveletV3AELG):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
        latent_gate_fn="sigmoid",
    ):
        super(Symlet2WaveletV3AELG, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="sym2",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
            latent_gate_fn=latent_gate_fn,
        )

    def forward(self, x):
        return super(Symlet2WaveletV3AELG, self).forward(x)


class Symlet3WaveletV3AELG(WaveletV3AELG):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
        latent_gate_fn="sigmoid",
    ):
        super(Symlet3WaveletV3AELG, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="sym3",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
            latent_gate_fn=latent_gate_fn,
        )

    def forward(self, x):
        return super(Symlet3WaveletV3AELG, self).forward(x)


class Symlet10WaveletV3AELG(WaveletV3AELG):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
        latent_gate_fn="sigmoid",
    ):
        super(Symlet10WaveletV3AELG, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="sym10",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
            latent_gate_fn=latent_gate_fn,
        )

    def forward(self, x):
        return super(Symlet10WaveletV3AELG, self).forward(x)


class Symlet20WaveletV3AELG(WaveletV3AELG):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
        latent_gate_fn="sigmoid",
    ):
        super(Symlet20WaveletV3AELG, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="sym20",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
            latent_gate_fn=latent_gate_fn,
        )

    def forward(self, x):
        return super(Symlet20WaveletV3AELG, self).forward(x)


# ---------------------------------------------------------------------------
# WaveletV3VAE — V3 wavelet blocks with AERootBlockVAE (variational AE) backbone
# ---------------------------------------------------------------------------


class WaveletV3VAE(AERootBlockVAE):
    """WaveletV3VAE: orthonormal DWT basis block with variational AE backbone.

    Structurally identical to WaveletV3AELG but inherits from AERootBlockVAE instead of
    AERootBlockLG, replacing the learned-gate bottleneck with the reparameterization trick
    (mu + std * eps during training, mu during eval). KL loss is stored in self.kl_loss
    and collected by NBeatsNet.training_step().

    Args:
        units (int): Width of the fully connected layers in the VAE backbone.
        backcast_length (int): Length of the historical input sequence.
        forecast_length (int): Length of the forecast horizon.
        basis_dim (int, optional): Number of DWT basis rows. Defaults to 32.
        basis_offset (int, optional): Row offset into SVD-ordered basis. Defaults to 0.
        share_weights (bool, optional): Share projection heads when dims match. Defaults to False.
        activation (str, optional): Activation for the VAE backbone. Defaults to 'ReLU'.
        active_g (bool or str, optional): Apply activation after basis expansion. Defaults to False.
        wavelet_type (str, optional): PyWavelets wavelet family string. Defaults to 'db3'.
        forecast_basis_dim (int, optional): Override basis_dim for forecast path only. Defaults to None.
        latent_dim (int, optional): Dimensionality of the VAE bottleneck. Defaults to 5.
        backcast_wavelet_type (str, optional): Override wavelet family for the backcast path only. Defaults to None.
        forecast_wavelet_type (str, optional): Override wavelet family for the forecast path only. Defaults to None.
    """

    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        wavelet_type="db3",
        forecast_basis_dim=None,
        latent_dim=5,
        backcast_wavelet_type=None,
        forecast_wavelet_type=None,
    ):
        super(WaveletV3VAE, self).__init__(
            backcast_length, units, activation, latent_dim=latent_dim
        )
        self.wavelet_type = wavelet_type
        self.backcast_wavelet_type = backcast_wavelet_type
        self.forecast_wavelet_type = forecast_wavelet_type
        self.share_weights = share_weights
        self.active_g = active_g

        # Resolve effective wavelet family per path
        eff_back_wavelet = (
            backcast_wavelet_type if backcast_wavelet_type is not None else wavelet_type
        )
        eff_fore_wavelet = (
            forecast_wavelet_type if forecast_wavelet_type is not None else wavelet_type
        )

        eff_back_dim = (
            backcast_length - min(basis_offset, backcast_length - 1)
            if basis_dim is None
            else min(
                basis_dim, backcast_length - min(basis_offset, backcast_length - 1)
            )
        )
        fore_dim = forecast_basis_dim if forecast_basis_dim is not None else basis_dim
        eff_fore_dim = (
            forecast_length - min(basis_offset, forecast_length - 1)
            if fore_dim is None
            else min(fore_dim, forecast_length - min(basis_offset, forecast_length - 1))
        )

        self.backcast_linear = nn.Linear(units, eff_back_dim, bias=False)
        self.forecast_linear = nn.Linear(units, eff_fore_dim, bias=False)

        self.backcast_g = _WaveletGeneratorV3(
            backcast_length,
            wavelet_type=eff_back_wavelet,
            basis_dim=eff_back_dim,
            basis_offset=basis_offset,
        )
        self.forecast_g = _WaveletGeneratorV3(
            forecast_length,
            wavelet_type=eff_fore_wavelet,
            basis_dim=eff_fore_dim,
            basis_offset=basis_offset,
        )

    def forward(self, x):
        x = super(WaveletV3VAE, self).forward(x)
        backcast_thetas = self.backcast_linear(x)
        forecast_thetas = self.forecast_linear(x)
        backcast = self.backcast_g(backcast_thetas)
        forecast = self.forecast_g(forecast_thetas)
        if self.active_g:
            if self.active_g != "forecast":
                backcast = self.activation(backcast)
            if self.active_g != "backcast":
                forecast = self.activation(forecast)
        return backcast, forecast


# --- V3VAE Wavelet subclasses (thin wrappers setting wavelet_type) ---


class HaarWaveletV3VAE(WaveletV3VAE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(HaarWaveletV3VAE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="haar",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(HaarWaveletV3VAE, self).forward(x)


class DB2WaveletV3VAE(WaveletV3VAE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(DB2WaveletV3VAE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="db2",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(DB2WaveletV3VAE, self).forward(x)


class DB3WaveletV3VAE(WaveletV3VAE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(DB3WaveletV3VAE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="db3",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(DB3WaveletV3VAE, self).forward(x)


class DB4WaveletV3VAE(WaveletV3VAE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(DB4WaveletV3VAE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="db4",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(DB4WaveletV3VAE, self).forward(x)


class DB10WaveletV3VAE(WaveletV3VAE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(DB10WaveletV3VAE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="db10",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(DB10WaveletV3VAE, self).forward(x)


class DB20WaveletV3VAE(WaveletV3VAE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(DB20WaveletV3VAE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="db20",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(DB20WaveletV3VAE, self).forward(x)


class Coif1WaveletV3VAE(WaveletV3VAE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(Coif1WaveletV3VAE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="coif1",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(Coif1WaveletV3VAE, self).forward(x)


class Coif2WaveletV3VAE(WaveletV3VAE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(Coif2WaveletV3VAE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="coif2",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(Coif2WaveletV3VAE, self).forward(x)


class Coif3WaveletV3VAE(WaveletV3VAE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(Coif3WaveletV3VAE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="coif3",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(Coif3WaveletV3VAE, self).forward(x)


class Coif10WaveletV3VAE(WaveletV3VAE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(Coif10WaveletV3VAE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="coif10",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(Coif10WaveletV3VAE, self).forward(x)


class Symlet2WaveletV3VAE(WaveletV3VAE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(Symlet2WaveletV3VAE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="sym2",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(Symlet2WaveletV3VAE, self).forward(x)


class Symlet3WaveletV3VAE(WaveletV3VAE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(Symlet3WaveletV3VAE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="sym3",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(Symlet3WaveletV3VAE, self).forward(x)


class Symlet10WaveletV3VAE(WaveletV3VAE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(Symlet10WaveletV3VAE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="sym10",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(Symlet10WaveletV3VAE, self).forward(x)


class Symlet20WaveletV3VAE(WaveletV3VAE):
    def __init__(
        self,
        units,
        backcast_length,
        forecast_length,
        basis_dim=32,
        basis_offset=0,
        share_weights=False,
        activation="ReLU",
        active_g: bool = False,
        forecast_basis_dim=None,
        latent_dim=5,
    ):
        super(Symlet20WaveletV3VAE, self).__init__(
            units,
            backcast_length,
            forecast_length,
            basis_dim,
            basis_offset,
            share_weights,
            activation,
            active_g,
            wavelet_type="sym20",
            forecast_basis_dim=forecast_basis_dim,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        return super(Symlet20WaveletV3VAE, self).forward(x)
