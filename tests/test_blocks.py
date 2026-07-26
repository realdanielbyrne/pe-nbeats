"""Tests for block implementations — activation, instantiation, forward pass, attribute names."""
import warnings

import lightningnbeats
import pytest
import torch
from torch import nn

from lightningnbeats.blocks import blocks as b
from lightningnbeats.constants import ACTIVATIONS


BACKCAST_LENGTH = 20
FORECAST_LENGTH = 5
UNITS = 32
THETAS_DIM = 4
BASIS_DIM = 16
LATENT_DIM = 4


# --- RootBlock activation validation ---

class TestRootBlockActivation:
    """Verify RootBlock validates activation parameter."""

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="is not in"):
            b.RootBlock(BACKCAST_LENGTH, UNITS, activation="InvalidActivation")

    def test_valid_activations(self):
        for act in ACTIVATIONS:
            block = b.RootBlock(BACKCAST_LENGTH, UNITS, activation=act)
            assert isinstance(block.activation, nn.Module)

    def test_pep8_not_in_syntax(self):
        """Ensure 'activation not in' works correctly (PEP 8 fix)."""
        with pytest.raises(ValueError):
            b.RootBlock(BACKCAST_LENGTH, UNITS, activation="FakeReLU")
        block = b.RootBlock(BACKCAST_LENGTH, UNITS, activation="ReLU")
        assert block is not None


# --- AutoEncoderAE instantiation (critical bug fix) ---

class TestAutoEncoderAE:
    """Verify AutoEncoderAE can be instantiated and run forward pass."""

    def test_instantiation_no_crash(self):
        block = b.AutoEncoderAE(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU", latent_dim=LATENT_DIM)
        assert block is not None

    def test_instantiation_shared_weights(self):
        block = b.AutoEncoderAE(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=True, activation="ReLU", latent_dim=LATENT_DIM)
        assert block.b_encoder is block.f_encoder

    def test_instantiation_different_activations(self):
        for act in ["ReLU", "LeakyReLU", "GELU", "ELU"]:
            block = b.AutoEncoderAE(
                units=UNITS, backcast_length=BACKCAST_LENGTH,
                forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
                share_weights=False, activation=act, latent_dim=LATENT_DIM)
            assert block is not None

    def test_forward_pass(self):
        block = b.AutoEncoderAE(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU", latent_dim=LATENT_DIM)
        x = torch.randn(4, BACKCAST_LENGTH)
        backcast, forecast = block(x)
        assert backcast.shape == (4, BACKCAST_LENGTH)
        assert forecast.shape == (4, FORECAST_LENGTH)

    def test_uses_explicit_linear_branch_layers(self):
        block = b.AutoEncoderAE(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU", latent_dim=LATENT_DIM)
        assert isinstance(block.b_encoder, nn.Linear)
        assert isinstance(block.f_encoder, nn.Linear)
        assert isinstance(block.b_decoder, nn.Linear)
        assert isinstance(block.f_decoder, nn.Linear)


class TestAEDescendantDimensionSemantics:
    """Verify AE-like descendants use latent_dim for local bottlenecks."""

    @pytest.mark.parametrize("block_name", [
        "AutoEncoderAE", "AutoEncoderAELG", "AutoEncoderVAE",
        "GenericAEBackcastAE", "GenericAEBackcastAELG", "GenericAEBackcastVAE",
    ])
    def test_branch_bottlenecks_use_latent_dim(self, block_name):
        block_class = getattr(b, block_name)
        block = block_class(
            units=UNITS,
            backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH,
            thetas_dim=THETAS_DIM,
            share_weights=False,
            activation="ReLU",
            latent_dim=LATENT_DIM,
        )

        assert isinstance(block.b_encoder, nn.Linear)
        assert isinstance(block.b_decoder, nn.Linear)
        assert block.b_encoder.in_features == UNITS
        assert block.b_encoder.out_features == LATENT_DIM
        assert block.b_decoder.in_features == LATENT_DIM
        assert block.b_decoder.out_features == UNITS

    @pytest.mark.parametrize("block_name", [
        "AutoEncoderAE", "AutoEncoderAELG", "AutoEncoderVAE",
    ])
    def test_autoencoder_descendants_project_directly_from_units(self, block_name):
        block_class = getattr(b, block_name)
        block = block_class(
            units=UNITS,
            backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH,
            thetas_dim=THETAS_DIM,
            share_weights=False,
            activation="ReLU",
            latent_dim=LATENT_DIM,
        )

        assert isinstance(block.forecast_g, nn.Linear)
        assert block.backcast_g.in_features == UNITS
        assert block.backcast_g.out_features == BACKCAST_LENGTH
        assert block.forecast_g.in_features == UNITS
        assert block.forecast_g.out_features == FORECAST_LENGTH

    @pytest.mark.parametrize("block_name", [
        "GenericAEBackcastAE", "GenericAEBackcastAELG", "GenericAEBackcastVAE",
    ])
    def test_generic_ae_backcast_descendants_keep_thetas_dim_forecast_head(self, block_name):
        block_class = getattr(b, block_name)
        block = block_class(
            units=UNITS,
            backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH,
            thetas_dim=THETAS_DIM,
            share_weights=False,
            activation="ReLU",
            latent_dim=LATENT_DIM,
        )

        assert isinstance(block.forecast_linear, nn.Linear)
        assert isinstance(block.forecast_g, nn.Linear)
        assert block.backcast_g.in_features == UNITS
        assert block.backcast_g.out_features == BACKCAST_LENGTH
        assert block.forecast_linear.in_features == UNITS
        assert block.forecast_linear.out_features == THETAS_DIM
        assert block.forecast_g.in_features == THETAS_DIM
        assert block.forecast_g.out_features == FORECAST_LENGTH

    @pytest.mark.parametrize("block_name", [
        "AutoEncoderAE", "AutoEncoderAELG", "AutoEncoderVAE",
    ])
    def test_autoencoder_descendants_still_share_encoder_when_requested(self, block_name):
        block_class = getattr(b, block_name)
        block = block_class(
            units=UNITS,
            backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH,
            thetas_dim=THETAS_DIM,
            share_weights=True,
            activation="ReLU",
            latent_dim=LATENT_DIM,
        )

        assert block.b_encoder is block.f_encoder


# --- SeasonalityAE forward (dead code removal) ---

class TestSeasonalityAE:
    """Verify SeasonalityAE forward returns correct tensors."""

    def test_forward_returns_two_tensors(self):
        block = b.SeasonalityAE(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU", latent_dim=LATENT_DIM)
        x = torch.randn(4, BACKCAST_LENGTH)
        result = block(x)
        assert len(result) == 2
        backcast, forecast = result
        assert backcast.shape == (4, BACKCAST_LENGTH)
        assert forecast.shape == (4, FORECAST_LENGTH)


# --- Generic vs BottleneckGeneric architecture tests ---

class TestGenericArchitecture:
    """Verify paper-faithful Generic block architecture."""

    def test_generic_has_theta_fc_layers(self):
        block = b.Generic(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU")
        assert hasattr(block, "theta_b_fc")
        assert hasattr(block, "theta_f_fc")

    def test_generic_does_not_have_bottleneck_layers(self):
        block = b.Generic(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU")
        assert not hasattr(block, "backcast_linear")
        assert not hasattr(block, "forecast_linear")
        assert not hasattr(block, "backcast_g")
        assert not hasattr(block, "forecast_g")

    def test_generic_forward_pass_shape(self):
        block = b.Generic(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU")
        x = torch.randn(4, BACKCAST_LENGTH)
        backcast, forecast = block(x)
        assert backcast.shape == (4, BACKCAST_LENGTH)
        assert forecast.shape == (4, FORECAST_LENGTH)

    def test_generic_theta_b_fc_output_size(self):
        block = b.Generic(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU")
        assert block.theta_b_fc.out_features == BACKCAST_LENGTH

    def test_generic_theta_f_fc_output_size(self):
        block = b.Generic(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU")
        assert block.theta_f_fc.out_features == FORECAST_LENGTH


class TestBottleneckGenericArchitecture:
    """Verify BottleneckGeneric block architecture with thetas_dim bottleneck."""

    def test_bottleneck_generic_has_bottleneck_layers(self):
        block = b.BottleneckGeneric(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU")
        assert hasattr(block, "backcast_linear")
        assert hasattr(block, "forecast_linear")
        assert hasattr(block, "backcast_g")
        assert hasattr(block, "forecast_g")

    def test_bottleneck_generic_does_not_have_theta_fc_layers(self):
        block = b.BottleneckGeneric(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU")
        assert not hasattr(block, "theta_b_fc")
        assert not hasattr(block, "theta_f_fc")

    def test_bottleneck_generic_forward_pass_shape(self):
        block = b.BottleneckGeneric(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU")
        x = torch.randn(4, BACKCAST_LENGTH)
        backcast, forecast = block(x)
        assert backcast.shape == (4, BACKCAST_LENGTH)
        assert forecast.shape == (4, FORECAST_LENGTH)

    def test_bottleneck_generic_backcast_linear_output_size(self):
        block = b.BottleneckGeneric(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU")
        assert block.backcast_linear.out_features == THETAS_DIM

    def test_bottleneck_generic_forecast_linear_output_size(self):
        block = b.BottleneckGeneric(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU")
        assert block.forecast_linear.out_features == THETAS_DIM

    def test_bottleneck_generic_backcast_g_output_size(self):
        block = b.BottleneckGeneric(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU")
        assert block.backcast_g.out_features == BACKCAST_LENGTH

    def test_bottleneck_generic_forecast_g_output_size(self):
        block = b.BottleneckGeneric(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU")
        assert block.forecast_g.out_features == FORECAST_LENGTH


class TestGenericAEArchitecture:
    """Verify paper-faithful GenericAE block architecture."""

    def test_generic_ae_has_theta_fc_layers(self):
        block = b.GenericAE(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU", latent_dim=LATENT_DIM)
        assert hasattr(block, "theta_b_fc")
        assert hasattr(block, "theta_f_fc")

    def test_generic_ae_does_not_have_bottleneck_layers(self):
        block = b.GenericAE(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU", latent_dim=LATENT_DIM)
        assert not hasattr(block, "backcast_linear")
        assert not hasattr(block, "forecast_linear")
        assert not hasattr(block, "backcast_g")
        assert not hasattr(block, "forecast_g")

    def test_generic_ae_forward_pass_shape(self):
        block = b.GenericAE(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU", latent_dim=LATENT_DIM)
        x = torch.randn(4, BACKCAST_LENGTH)
        backcast, forecast = block(x)
        assert backcast.shape == (4, BACKCAST_LENGTH)
        assert forecast.shape == (4, FORECAST_LENGTH)


class TestBottleneckGenericAEArchitecture:
    """Verify BottleneckGenericAE block architecture with thetas_dim bottleneck."""

    def test_bottleneck_generic_ae_has_bottleneck_layers(self):
        block = b.BottleneckGenericAE(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU", latent_dim=LATENT_DIM)
        assert hasattr(block, "backcast_linear")
        assert hasattr(block, "forecast_linear")
        assert hasattr(block, "backcast_g")
        assert hasattr(block, "forecast_g")

    def test_bottleneck_generic_ae_does_not_have_theta_fc_layers(self):
        block = b.BottleneckGenericAE(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU", latent_dim=LATENT_DIM)
        assert not hasattr(block, "theta_b_fc")
        assert not hasattr(block, "theta_f_fc")

    def test_bottleneck_generic_ae_forward_pass_shape(self):
        block = b.BottleneckGenericAE(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            share_weights=False, activation="ReLU", latent_dim=LATENT_DIM)
        x = torch.randn(4, BACKCAST_LENGTH)
        backcast, forecast = block(x)
        assert backcast.shape == (4, BACKCAST_LENGTH)
        assert forecast.shape == (4, FORECAST_LENGTH)


# --- Trend block defaults tests ---

class TestTrendDefaults:
    """Verify Trend block default parameter values."""

    def test_trend_activation_defaults_to_relu(self):
        block = b.Trend(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM)
        assert isinstance(block.activation, torch.nn.ReLU)

    def test_trend_share_weights_defaults_to_false(self):
        block = b.Trend(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM)
        assert block.share_weights is False
        assert block.backcast_linear is not block.forecast_linear


class TestTrendAEDefaults:
    """Verify TrendAE block default parameter values."""

    def test_trend_ae_activation_defaults_to_relu(self):
        block = b.TrendAE(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            latent_dim=LATENT_DIM)
        assert isinstance(block.activation, torch.nn.ReLU)

    def test_trend_ae_share_weights_defaults_to_false(self):
        block = b.TrendAE(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
            latent_dim=LATENT_DIM)
        assert block.backcast_linear is not block.forecast_linear


class TestTrendWaveletProperties:
    """Verify TrendWavelet matches AE-family constructor and routing semantics."""

    def test_forward_pass(self):
        block = b.TrendWavelet(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, trend_dim=THETAS_DIM,
            wavelet_dim=BASIS_DIM, activation="ReLU")
        x = torch.randn(4, BACKCAST_LENGTH)
        backcast, forecast = block(x)
        assert backcast.shape == (4, BACKCAST_LENGTH)
        assert forecast.shape == (4, FORECAST_LENGTH)

    def test_share_weights_requires_equal_total_dims(self):
        shared = b.TrendWavelet(
            units=UNITS, backcast_length=8, forecast_length=8,
            trend_dim=THETAS_DIM, wavelet_dim=3, share_weights=True)
        unshared = b.TrendWavelet(
            units=UNITS, backcast_length=BACKCAST_LENGTH, forecast_length=FORECAST_LENGTH,
            trend_dim=THETAS_DIM, wavelet_dim=BASIS_DIM, share_weights=True)
        assert shared.backcast_linear is shared.forecast_linear
        assert unshared.backcast_linear is not unshared.forecast_linear

    def test_wavelet_type_defaults_and_overrides(self):
        default_block = b.TrendWavelet(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, trend_dim=THETAS_DIM,
            wavelet_dim=BASIS_DIM)
        override_block = b.TrendWavelet(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, trend_dim=THETAS_DIM,
            wavelet_dim=BASIS_DIM, wavelet_type="haar")
        assert default_block.wavelet_type == "db3"
        assert override_block.wavelet_type == "haar"

    def test_asymmetric_wavelet_types_are_stored(self):
        block = b.TrendWavelet(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, trend_dim=THETAS_DIM,
            wavelet_dim=BASIS_DIM, wavelet_type="db3",
            backcast_wavelet_type="sym20", forecast_wavelet_type="coif2")
        assert block.backcast_wavelet_type == "sym20"
        assert block.forecast_wavelet_type == "coif2"

    def test_effective_wavelet_dims_respect_forecast_basis_dim_and_basis_offset(self):
        block = b.TrendWavelet(
            units=UNITS, backcast_length=20, forecast_length=5,
            trend_dim=THETAS_DIM, wavelet_dim=16, forecast_basis_dim=4,
            basis_offset=3)
        assert block.backcast_linear.out_features == THETAS_DIM + min(16, 20 - 3)
        assert block.forecast_linear.out_features == THETAS_DIM + min(4, 5 - 3)

    def test_expected_generators_and_linear_sizes_exist(self):
        block = b.TrendWavelet(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, trend_dim=THETAS_DIM,
            wavelet_dim=BASIS_DIM)
        assert block.backcast_linear.in_features == UNITS
        assert block.forecast_linear.in_features == UNITS
        assert block.backcast_linear.out_features == THETAS_DIM + min(BASIS_DIM, BACKCAST_LENGTH)
        assert block.forecast_linear.out_features == THETAS_DIM + min(BASIS_DIM, FORECAST_LENGTH)
        assert hasattr(block, "backcast_trend_g")
        assert hasattr(block, "forecast_trend_g")
        assert hasattr(block, "backcast_wavelet_g")
        assert hasattr(block, "forecast_wavelet_g")


# --- Seasonality block bias tests ---

class TestSeasonalityBias:
    """Verify Seasonality block has bias=False on linear layers."""

    def test_seasonality_backcast_linear_has_no_bias(self):
        block = b.Seasonality(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM)
        assert block.backcast_linear.bias is None

    def test_seasonality_forecast_linear_has_no_bias(self):
        block = b.Seasonality(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM)
        assert block.forecast_linear.bias is None


# --- Constants registry validation tests ---

class TestBlocksRegistry:
    """Verify every entry in BLOCKS has a corresponding class in blocks module."""

    def test_all_blocks_have_classes(self):
        from lightningnbeats.constants import BLOCKS
        for block_name in BLOCKS:
            assert hasattr(b, block_name), f"Block class {block_name} not found in blocks module"

    def test_all_block_classes_instantiable(self):
        from lightningnbeats.constants import BLOCKS
        for block_name in BLOCKS:
            block_class = getattr(b, block_name)
            assert callable(block_class), f"Block {block_name} is not callable"


class TestLossesRegistry:
    """Verify every entry in LOSSES has a corresponding class."""

    def test_all_losses_exist(self):
        from lightningnbeats.constants import LOSSES
        from lightningnbeats import losses
        import torch.nn as nn

        for loss_name in LOSSES:
            if hasattr(losses, loss_name):
                assert callable(getattr(losses, loss_name))
            elif hasattr(nn, loss_name):
                assert callable(getattr(nn, loss_name))
            else:
                raise AssertionError(f"Loss {loss_name} not found in losses module or torch.nn")


# --- All blocks output shape tests ---

class TestAllBlocksOutputShapes:
    """Parametrized test for all registered blocks to verify output shapes."""

    @pytest.mark.parametrize("block_name", [
        "Generic", "BottleneckGeneric", "GenericAE", "BottleneckGenericAE",
        "GenericAEBackcast", "GenericAEBackcastAE",
        "Trend", "TrendAE", "TrendWavelet", "Seasonality", "SeasonalityAE",
        "AutoEncoder", "AutoEncoderAE",
        # VAE block (RootBlock backbone)
        "VAE",
        # Learned-Gate (LG) AE blocks
        "GenericAELG", "BottleneckGenericAELG", "TrendAELG", "SeasonalityAELG",
        "AutoEncoderAELG", "GenericAEBackcastAELG",
        # Variational AE (VAE) blocks
        "GenericVAE", "BottleneckGenericVAE", "TrendVAE", "SeasonalityVAE",
        "AutoEncoderVAE", "GenericAEBackcastVAE",
        # V3 Wavelet blocks (orthonormal DWT basis) — generic base + family wrappers
        "WaveletV3",
        "HaarWaveletV3", "DB2WaveletV3", "DB3WaveletV3", "DB4WaveletV3",
        "DB10WaveletV3", "DB20WaveletV3",
        "Coif1WaveletV3", "Coif2WaveletV3", "Coif3WaveletV3", "Coif10WaveletV3",
        "Symlet2WaveletV3", "Symlet3WaveletV3", "Symlet10WaveletV3", "Symlet20WaveletV3",
        # TrendWavelet family — merged polynomial + DWT basis
        "TrendWavelet",
        # TrendWaveletAE family — merged polynomial + DWT basis (AE backbone, Option A)
        "TrendWaveletAE",
        "TrendWaveletAELG",
        # TrendWaveletGeneric family — three-way additive: trend + wavelet + learned generic basis
        "TrendWaveletGeneric",
        "TrendWaveletGenericAE",
        "TrendWaveletGenericAELG",
        "TrendWaveletGenericVAE",
        # V3AE Wavelet blocks (orthonormal DWT basis, AE bottleneck backbone, Option B)
        "WaveletV3AE",
        "HaarWaveletV3AE", "DB2WaveletV3AE", "DB3WaveletV3AE", "DB4WaveletV3AE",
        "DB10WaveletV3AE", "DB20WaveletV3AE",
        "Coif1WaveletV3AE", "Coif2WaveletV3AE", "Coif3WaveletV3AE", "Coif10WaveletV3AE",
        "Symlet2WaveletV3AE", "Symlet3WaveletV3AE", "Symlet10WaveletV3AE", "Symlet20WaveletV3AE",
        # V3AELG Wavelet blocks (orthonormal DWT basis, Learned-Gate AE backbone)
        "WaveletV3AELG",
        "HaarWaveletV3AELG", "DB2WaveletV3AELG", "DB3WaveletV3AELG", "DB4WaveletV3AELG",
        "DB10WaveletV3AELG", "DB20WaveletV3AELG",
        "Coif1WaveletV3AELG", "Coif2WaveletV3AELG", "Coif3WaveletV3AELG", "Coif10WaveletV3AELG",
        "Symlet2WaveletV3AELG", "Symlet3WaveletV3AELG", "Symlet10WaveletV3AELG", "Symlet20WaveletV3AELG",
        # V3VAE Wavelet blocks (orthonormal DWT basis, variational AE backbone)
        "WaveletV3VAE",
    ])
    def test_block_output_shape(self, block_name):
        block_class = getattr(b, block_name)

        kwargs = {
            "units": UNITS,
            "backcast_length": BACKCAST_LENGTH,
            "forecast_length": FORECAST_LENGTH,
            "activation": "ReLU"
        }

        ae_root_blocks = ["SeasonalityAE", "GenericAEBackcastAE", "AutoEncoderAE",
                          "GenericAE", "BottleneckGenericAE", "TrendAE",
                          "GenericAELG", "BottleneckGenericAELG", "TrendAELG",
                          "SeasonalityAELG", "AutoEncoderAELG", "GenericAEBackcastAELG",
                          "GenericVAE", "BottleneckGenericVAE", "TrendVAE",
                          "SeasonalityVAE", "AutoEncoderVAE", "GenericAEBackcastVAE",
                          # TrendWaveletAE family and V3AE variants (AE backbone)
                          "TrendWaveletAE",
                          "TrendWaveletAELG",
                          # TrendWaveletGeneric AE/LG/VAE variants
                          "TrendWaveletGenericAE",
                          "TrendWaveletGenericAELG",
                          "TrendWaveletGenericVAE",
                          "WaveletV3AE",
                          "HaarWaveletV3AE", "DB2WaveletV3AE", "DB3WaveletV3AE", "DB4WaveletV3AE",
                          "DB10WaveletV3AE", "DB20WaveletV3AE",
                          "Coif1WaveletV3AE", "Coif2WaveletV3AE", "Coif3WaveletV3AE", "Coif10WaveletV3AE",
                          "Symlet2WaveletV3AE", "Symlet3WaveletV3AE", "Symlet10WaveletV3AE", "Symlet20WaveletV3AE",
                          # V3AELG variants (LG-AE backbone)
                          "WaveletV3AELG",
                          "HaarWaveletV3AELG", "DB2WaveletV3AELG", "DB3WaveletV3AELG", "DB4WaveletV3AELG",
                          "DB10WaveletV3AELG", "DB20WaveletV3AELG",
                          "Coif1WaveletV3AELG", "Coif2WaveletV3AELG", "Coif3WaveletV3AELG", "Coif10WaveletV3AELG",
                          "Symlet2WaveletV3AELG", "Symlet3WaveletV3AELG", "Symlet10WaveletV3AELG", "Symlet20WaveletV3AELG",
                          # V3VAE variants (variational AE backbone)
                          "WaveletV3VAE"]
        if block_name in ae_root_blocks:
            kwargs["latent_dim"] = LATENT_DIM

        if block_name in ("TrendWavelet", "TrendWaveletAE", "TrendWaveletAELG",
                         "TrendWaveletGeneric", "TrendWaveletGenericAE",
                         "TrendWaveletGenericAELG", "TrendWaveletGenericVAE"):
            # TrendWavelet family uses trend_dim + wavelet_dim instead of thetas_dim/basis_dim
            kwargs["trend_dim"] = THETAS_DIM
            kwargs["wavelet_dim"] = BASIS_DIM
            if "Generic" in block_name:
                kwargs["generic_dim"] = 5
        elif "Wavelet" in block_name:
            kwargs["basis_dim"] = BASIS_DIM
        else:
            kwargs["thetas_dim"] = THETAS_DIM

        if block_name in ["AutoEncoder", "AutoEncoderAE", "GenericAEBackcast", "GenericAEBackcastAE",
                         "AutoEncoderAELG", "AutoEncoderVAE",
                         "GenericAEBackcastAELG", "GenericAEBackcastVAE",
                         "VAE"]:
            kwargs["share_weights"] = False
        elif block_name in ["Trend", "TrendAE", "TrendWavelet", "TrendAELG", "TrendVAE",
                            "TrendWaveletAE", "TrendWaveletAELG",
                            "TrendWaveletGeneric", "TrendWaveletGenericAE",
                            "TrendWaveletGenericAELG", "TrendWaveletGenericVAE"]:
            kwargs["share_weights"] = False

        block = block_class(**kwargs)

        x = torch.randn(4, BACKCAST_LENGTH)
        backcast, forecast = block(x)

        assert backcast.shape == (4, BACKCAST_LENGTH), f"{block_name} backcast shape incorrect"
        assert forecast.shape == (4, FORECAST_LENGTH), f"{block_name} forecast shape incorrect"


class TestDeprecatedVAE2Access:
    """Deprecated block names should raise targeted messages on attribute lookup."""

    @pytest.mark.parametrize("block_name", [
        "GenericVAE2",
        "WaveletV3VAE2",
        "DB4WaveletV3VAE2",
    ])
    def test_blocks_module_lookup_raises_deprecation_message(self, block_name):
        with pytest.raises(AttributeError, match="deprecated and removed"):
            getattr(b, block_name)

    def test_root_package_lookup_raises_deprecation_message(self):
        with pytest.raises(AttributeError, match="deprecated and removed"):
            getattr(lightningnbeats, "GenericVAE2")


# --- AERootBlockLG / AERootBlockVAE property tests ---

class TestAERootBlockLGProperties:
    """Verify learned-gate AE blocks have a latent_gate parameter."""

    @pytest.mark.parametrize("block_name", [
        "GenericAELG", "BottleneckGenericAELG", "TrendAELG",
        "SeasonalityAELG", "AutoEncoderAELG", "GenericAEBackcastAELG",
    ])
    def test_has_latent_gate(self, block_name):
        block_class = getattr(b, block_name)
        kwargs = {"units": UNITS, "backcast_length": BACKCAST_LENGTH,
                  "forecast_length": FORECAST_LENGTH, "thetas_dim": THETAS_DIM,
                  "latent_dim": LATENT_DIM, "share_weights": False}
        block = block_class(**kwargs)
        assert hasattr(block, 'latent_gate'), f"{block_name} missing latent_gate"
        assert block.latent_gate.shape == (LATENT_DIM,)
        assert block.latent_gate.requires_grad

    @pytest.mark.parametrize("block_name", [
        "HaarWaveletV3AELG", "DB2WaveletV3AELG", "DB3WaveletV3AELG", "DB4WaveletV3AELG",
        "DB10WaveletV3AELG", "DB20WaveletV3AELG",
        "Coif1WaveletV3AELG", "Coif2WaveletV3AELG", "Coif3WaveletV3AELG", "Coif10WaveletV3AELG",
        "Symlet2WaveletV3AELG", "Symlet3WaveletV3AELG", "Symlet10WaveletV3AELG", "Symlet20WaveletV3AELG",
    ])
    def test_wavelet_v3aelg_has_latent_gate(self, block_name):
        block_class = getattr(b, block_name)
        kwargs = {"units": UNITS, "backcast_length": BACKCAST_LENGTH,
                  "forecast_length": FORECAST_LENGTH, "basis_dim": BASIS_DIM,
                  "latent_dim": LATENT_DIM}
        block = block_class(**kwargs)
        assert hasattr(block, 'latent_gate'), f"{block_name} missing latent_gate"
        assert block.latent_gate.shape == (LATENT_DIM,)
        assert block.latent_gate.requires_grad


class TestAERootBlockLGGateFunctions:
    """Verify LG blocks expose and apply configurable latent gate functions."""

    def test_default_latent_gate_fn_is_sigmoid(self):
        block = b.AERootBlockLG(
            BACKCAST_LENGTH,
            UNITS,
            activation="ReLU",
            latent_dim=LATENT_DIM,
        )
        assert block.latent_gate_fn == "sigmoid"

    def test_wavy_sigmoid_is_accepted_by_aelg_blocks(self):
        block = b.GenericAELG(
            units=UNITS,
            backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH,
            thetas_dim=THETAS_DIM,
            share_weights=False,
            activation="ReLU",
            latent_dim=LATENT_DIM,
            latent_gate_fn="wavy_sigmoid",
        )
        assert block.latent_gate_fn == "wavy_sigmoid"

    def test_wavelet_sigmoid_is_accepted_by_aelg_blocks(self):
        block = b.GenericAELG(
            units=UNITS,
            backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH,
            thetas_dim=THETAS_DIM,
            share_weights=False,
            activation="ReLU",
            latent_dim=LATENT_DIM,
            latent_gate_fn="wavelet_sigmoid",
        )
        assert block.latent_gate_fn == "wavelet_sigmoid"

    def test_invalid_latent_gate_fn_raises(self):
        with pytest.raises(ValueError, match="latent_gate_fn must be one of"):
            b.AERootBlockLG(
                BACKCAST_LENGTH,
                UNITS,
                activation="ReLU",
                latent_dim=LATENT_DIM,
                latent_gate_fn="invalid",
            )

    def test_sigmoid_gate_helper_matches_torch(self):
        block = b.AERootBlockLG(
            BACKCAST_LENGTH,
            UNITS,
            activation="ReLU",
            latent_dim=LATENT_DIM,
            latent_gate_fn="sigmoid",
        )
        with torch.no_grad():
            block.latent_gate.copy_(torch.tensor([-2.0, -0.5, 0.5, 2.0]))

        expected = torch.sigmoid(block.latent_gate)
        actual = block._compute_latent_gate_scale()

        assert torch.allclose(actual, expected, atol=1e-6)

    def test_wavy_sigmoid_gate_helper_matches_formula(self):
        block = b.AERootBlockLG(
            BACKCAST_LENGTH,
            UNITS,
            activation="ReLU",
            latent_dim=LATENT_DIM,
            latent_gate_fn="wavy_sigmoid",
        )
        with torch.no_grad():
            block.latent_gate.copy_(torch.tensor([-2.0, -0.5, 0.5, 2.0]))

        expected = b.wavy_sigmoid(block.latent_gate)
        actual = block._compute_latent_gate_scale()

        assert torch.allclose(actual, expected, atol=1e-6)

    def test_wavelet_sigmoid_gate_helper_matches_formula(self):
        block = b.AERootBlockLG(
            BACKCAST_LENGTH,
            UNITS,
            activation="ReLU",
            latent_dim=LATENT_DIM,
            latent_gate_fn="wavelet_sigmoid",
        )
        with torch.no_grad():
            block.latent_gate.copy_(torch.tensor([-2.0, -0.5, 0.5, 2.0]))

        expected = b.wavelet_sigmoid(block.latent_gate)
        actual = block._compute_latent_gate_scale()

        assert torch.allclose(actual, expected, atol=1e-6)


class TestAERootBlockVAEProperties:
    """Verify VAE blocks store kl_loss after forward pass."""

    @pytest.mark.parametrize("block_name", [
        "GenericVAE", "BottleneckGenericVAE", "TrendVAE",
        "SeasonalityVAE", "AutoEncoderVAE", "GenericAEBackcastVAE",
    ])
    def test_kl_loss_stored_after_forward(self, block_name):
        block_class = getattr(b, block_name)
        kwargs = {"units": UNITS, "backcast_length": BACKCAST_LENGTH,
                  "forecast_length": FORECAST_LENGTH, "thetas_dim": THETAS_DIM,
                  "latent_dim": LATENT_DIM, "share_weights": False}
        block = block_class(**kwargs)
        block.train()
        x = torch.randn(4, BACKCAST_LENGTH)
        backcast, forecast = block(x)
        assert hasattr(block, 'kl_loss'), f"{block_name} missing kl_loss"
        assert isinstance(block.kl_loss, torch.Tensor)
        assert block.kl_loss.item() >= 0, f"{block_name} kl_loss should be non-negative"

    @pytest.mark.parametrize("block_name", [
        "GenericVAE", "BottleneckGenericVAE", "TrendVAE",
        "SeasonalityVAE", "AutoEncoderVAE", "GenericAEBackcastVAE",
    ])
    def test_eval_mode_uses_mu_directly(self, block_name):
        block_class = getattr(b, block_name)
        kwargs = {"units": UNITS, "backcast_length": BACKCAST_LENGTH,
                  "forecast_length": FORECAST_LENGTH, "thetas_dim": THETAS_DIM,
                  "latent_dim": LATENT_DIM, "share_weights": False}
        block = block_class(**kwargs)
        block.eval()
        x = torch.randn(4, BACKCAST_LENGTH)
        # Two eval passes should produce identical results (no stochastic sampling)
        b1, f1 = block(x)
        b2, f2 = block(x)
        assert torch.allclose(b1, b2), f"{block_name} eval mode is stochastic"
        assert torch.allclose(f1, f2), f"{block_name} eval mode is stochastic"


class TestVAEBlockProperties:
    """Verify the VAE block (RootBlock backbone) stores kl_loss and behaves correctly."""

    def test_kl_loss_stored_after_forward(self):
        block = b.VAE(units=UNITS, backcast_length=BACKCAST_LENGTH,
                      forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
                      share_weights=False)
        block.train()
        x = torch.randn(4, BACKCAST_LENGTH)
        backcast, forecast = block(x)
        assert hasattr(block, 'kl_loss'), "VAE missing kl_loss"
        assert isinstance(block.kl_loss, torch.Tensor)
        assert block.kl_loss.item() >= 0, "VAE kl_loss should be non-negative"

    def test_eval_mode_is_deterministic(self):
        block = b.VAE(units=UNITS, backcast_length=BACKCAST_LENGTH,
                      forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
                      share_weights=False)
        block.eval()
        x = torch.randn(4, BACKCAST_LENGTH)
        b1, f1 = block(x)
        b2, f2 = block(x)
        assert torch.allclose(b1, b2), "VAE eval mode is stochastic"
        assert torch.allclose(f1, f2), "VAE eval mode is stochastic"

    def test_train_mode_is_stochastic(self):
        block = b.VAE(units=UNITS, backcast_length=BACKCAST_LENGTH,
                      forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
                      share_weights=False)
        block.train()
        x = torch.randn(4, BACKCAST_LENGTH)
        _, f1 = block(x)
        _, f2 = block(x)
        # With high probability, two stochastic passes produce different results
        assert not torch.allclose(f1, f2, atol=1e-6), "VAE train mode should be stochastic"

    def test_share_weights(self):
        block = b.VAE(units=UNITS, backcast_length=BACKCAST_LENGTH,
                      forecast_length=FORECAST_LENGTH, thetas_dim=THETAS_DIM,
                      share_weights=True)
        assert block.b_mu is block.f_mu, "share_weights should share mu heads"
        assert block.b_logvar is block.f_logvar, "share_weights should share logvar heads"


# --- V3 Wavelet property tests ---

V3_WAVELET_BLOCKS = [
    # Generic base classes (parameter-driven wavelet family)
    "WaveletV3", "WaveletV3AE", "WaveletV3AELG", "WaveletV3VAE",
    # Family wrappers (RootBlock backbone)
    "HaarWaveletV3", "DB2WaveletV3", "DB3WaveletV3", "DB4WaveletV3",
    "DB10WaveletV3", "DB20WaveletV3",
    "Coif1WaveletV3", "Coif2WaveletV3", "Coif3WaveletV3", "Coif10WaveletV3",
    "Symlet2WaveletV3", "Symlet3WaveletV3", "Symlet10WaveletV3", "Symlet20WaveletV3",
    # V3AE variants (same orthonormal basis, AE bottleneck backbone)
    "HaarWaveletV3AE", "DB2WaveletV3AE", "DB3WaveletV3AE", "DB4WaveletV3AE",
    "DB10WaveletV3AE", "DB20WaveletV3AE",
    "Coif1WaveletV3AE", "Coif2WaveletV3AE", "Coif3WaveletV3AE", "Coif10WaveletV3AE",
    "Symlet2WaveletV3AE", "Symlet3WaveletV3AE", "Symlet10WaveletV3AE", "Symlet20WaveletV3AE",
    # V3AELG variants (same orthonormal basis, Learned-Gate AE backbone)
    "HaarWaveletV3AELG", "DB2WaveletV3AELG", "DB3WaveletV3AELG", "DB4WaveletV3AELG",
    "DB10WaveletV3AELG", "DB20WaveletV3AELG",
    "Coif1WaveletV3AELG", "Coif2WaveletV3AELG", "Coif3WaveletV3AELG", "Coif10WaveletV3AELG",
    "Symlet2WaveletV3AELG", "Symlet3WaveletV3AELG", "Symlet10WaveletV3AELG", "Symlet20WaveletV3AELG",
]

class TestWaveletV3Properties:
    """Verify V3 wavelet blocks have orthonormal bases and stable outputs."""

    @pytest.mark.parametrize("block_name", V3_WAVELET_BLOCKS)
    def test_basis_is_orthonormal(self, block_name):
        block_class = getattr(b, block_name)
        block = block_class(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, basis_dim=BASIS_DIM,
        )
        for name, param in block.named_parameters():
            if "basis" in name and not param.requires_grad:
                basis = param.data
                identity = torch.matmul(basis, basis.T)
                expected = torch.eye(basis.shape[0])
                error = torch.max(torch.abs(identity - expected)).item()
                assert error < 1e-5, (
                    f"{block_name}.{name} not orthonormal: max error {error:.2e}"
                )

    @pytest.mark.parametrize("block_name", V3_WAVELET_BLOCKS)
    def test_basis_condition_number(self, block_name):
        block_class = getattr(b, block_name)
        block = block_class(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, basis_dim=BASIS_DIM,
        )
        for name, param in block.named_parameters():
            if "basis" in name and not param.requires_grad:
                sv = torch.linalg.svdvals(param.data)
                cond = (sv[0] / sv[-1]).item()
                assert cond < 1.1, (
                    f"{block_name}.{name} condition number {cond:.4f} >= 1.1"
                )

    @pytest.mark.parametrize("block_name", V3_WAVELET_BLOCKS)
    def test_no_downsampling_layer(self, block_name):
        block_class = getattr(b, block_name)
        block = block_class(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, basis_dim=BASIS_DIM,
        )
        assert not hasattr(block, "backcast_down_sample"), (
            f"{block_name} should not have backcast_down_sample"
        )
        assert not hasattr(block, "forecast_down_sample"), (
            f"{block_name} should not have forecast_down_sample"
        )

    @pytest.mark.parametrize("wavelet_type,wavelet_class", [
        ("haar", "HaarWaveletV3"),
        ("db2", "DB2WaveletV3"),
        ("db3", "DB3WaveletV3"),
        ("db4", "DB4WaveletV3"),
        ("db10", "DB10WaveletV3"),
        ("db20", "DB20WaveletV3"),
        ("coif1", "Coif1WaveletV3"),
        ("coif2", "Coif2WaveletV3"),
        ("coif3", "Coif3WaveletV3"),
        ("coif10", "Coif10WaveletV3"),
        ("sym2", "Symlet2WaveletV3"),
        ("sym3", "Symlet3WaveletV3"),
        ("sym10", "Symlet10WaveletV3"),
        ("sym20", "Symlet20WaveletV3"),
        # V3AE variants (same orthonormal DWT basis, AE bottleneck backbone)
        ("haar", "HaarWaveletV3AE"),
        ("db2", "DB2WaveletV3AE"),
        ("db3", "DB3WaveletV3AE"),
        ("db4", "DB4WaveletV3AE"),
        ("db10", "DB10WaveletV3AE"),
        ("db20", "DB20WaveletV3AE"),
        ("coif1", "Coif1WaveletV3AE"),
        ("coif2", "Coif2WaveletV3AE"),
        ("coif3", "Coif3WaveletV3AE"),
        ("coif10", "Coif10WaveletV3AE"),
        ("sym2", "Symlet2WaveletV3AE"),
        ("sym3", "Symlet3WaveletV3AE"),
        ("sym10", "Symlet10WaveletV3AE"),
        ("sym20", "Symlet20WaveletV3AE"),
        # V3AELG variants (same orthonormal DWT basis, Learned-Gate AE backbone)
        ("haar", "HaarWaveletV3AELG"),
        ("db2", "DB2WaveletV3AELG"),
        ("db3", "DB3WaveletV3AELG"),
        ("db4", "DB4WaveletV3AELG"),
        ("db10", "DB10WaveletV3AELG"),
        ("db20", "DB20WaveletV3AELG"),
        ("coif1", "Coif1WaveletV3AELG"),
        ("coif2", "Coif2WaveletV3AELG"),
        ("coif3", "Coif3WaveletV3AELG"),
        ("coif10", "Coif10WaveletV3AELG"),
        ("sym2", "Symlet2WaveletV3AELG"),
        ("sym3", "Symlet3WaveletV3AELG"),
        ("sym10", "Symlet10WaveletV3AELG"),
        ("sym20", "Symlet20WaveletV3AELG"),
    ])
    def test_all_families_produce_full_rank(self, wavelet_type, wavelet_class):
        block_class = getattr(b, wavelet_class)
        block = block_class(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, basis_dim=BASIS_DIM,
        )
        for name, param in block.named_parameters():
            if "basis" in name and not param.requires_grad:
                target_length = param.shape[1]
                rank = param.shape[0]
                expected_dim = min(BASIS_DIM, target_length)
                assert rank == expected_dim, (
                    f"{wavelet_class}.{name}: rank {rank} != expected {expected_dim} "
                    f"(min(BASIS_DIM={BASIS_DIM}, target_length={target_length}))"
                )

    @pytest.mark.parametrize("wavelet_type", ["db20", "sym20", "coif10"])
    def test_short_sequences_do_not_force_invalid_level_one(self, wavelet_type):
        """Regression: keep level=0 when PyWavelets reports no valid decomposition levels."""
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            basis = b._WaveletGeneratorV3._build_basis(
                target_length=5,
                wavelet_type=wavelet_type,
                max_decomp_level=5,
            )

        assert basis.shape == (5, 5)
        assert not any(
            "Level value of 1 is too high" in str(warning.message)
            for warning in caught_warnings
        ), f"{wavelet_type} should respect level=0 when max_level == 0"

    @pytest.mark.parametrize("block_name", V3_WAVELET_BLOCKS)
    def test_no_nan_output(self, block_name):
        block_class = getattr(b, block_name)
        block = block_class(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, basis_dim=BASIS_DIM,
        )
        x = torch.randn(8, BACKCAST_LENGTH)
        backcast, forecast = block(x)
        assert not torch.isnan(backcast).any(), f"{block_name} backcast contains NaN"
        assert not torch.isnan(forecast).any(), f"{block_name} forecast contains NaN"
        assert not torch.isinf(backcast).any(), f"{block_name} backcast contains Inf"
        assert not torch.isinf(forecast).any(), f"{block_name} forecast contains Inf"

    @pytest.mark.parametrize("block_name", V3_WAVELET_BLOCKS)
    def test_share_weights_with_different_lengths(self, block_name):
        """Verify share_weights=True works when backcast_length != forecast_length.

        Regression test: V3 orthonormal basis is (target_length x target_length),
        so backcast and forecast linear projections must have different output sizes.
        Sharing them caused a shape mismatch in matmul.
        """
        block_class = getattr(b, block_name)
        # Use M4-Yearly dimensions: backcast=30, forecast=6
        block = block_class(
            units=UNITS, backcast_length=30, forecast_length=6,
            basis_dim=BASIS_DIM, share_weights=True,
        )

        # Forward pass must succeed
        x = torch.randn(8, 30)
        backcast, forecast = block(x)
        assert backcast.shape == (8, 30), f"{block_name} backcast shape incorrect"
        assert forecast.shape == (8, 6), f"{block_name} forecast shape incorrect"

        # Linear layers must NOT be shared (different output sizes).
        # out_features == min(basis_dim, target_length) after truncation.
        assert block.backcast_linear is not block.forecast_linear, (
            f"{block_name} should use separate linear projections"
        )
        assert block.backcast_linear.out_features == min(BASIS_DIM, 30)
        assert block.forecast_linear.out_features == min(BASIS_DIM, 6)

    @pytest.mark.parametrize("block_name", V3_WAVELET_BLOCKS)
    def test_asymmetric_basis_dims(self, block_name):
        """Verify forecast_basis_dim overrides basis_dim for forecast path only."""
        block_class = getattr(b, block_name)
        block = block_class(
            units=UNITS, backcast_length=30, forecast_length=6,
            basis_dim=16, forecast_basis_dim=4,
        )
        x = torch.randn(4, 30)
        backcast, forecast = block(x)
        assert backcast.shape == (4, 30)
        assert forecast.shape == (4, 6)
        assert block.backcast_linear.out_features == min(16, 30)  # uses basis_dim
        assert block.forecast_linear.out_features == min(4, 6)    # uses forecast_basis_dim


# --- active_g split mode tests ---

ACTIVE_G_BLOCK_CONFIGS = [
    ("Generic", {"units": UNITS, "backcast_length": BACKCAST_LENGTH,
                 "forecast_length": FORECAST_LENGTH, "thetas_dim": THETAS_DIM}),
    ("BottleneckGeneric", {"units": UNITS, "backcast_length": BACKCAST_LENGTH,
                           "forecast_length": FORECAST_LENGTH, "thetas_dim": THETAS_DIM}),
    ("GenericAE", {"units": UNITS, "backcast_length": BACKCAST_LENGTH,
                   "forecast_length": FORECAST_LENGTH, "thetas_dim": THETAS_DIM,
                   "latent_dim": LATENT_DIM}),
    ("AutoEncoder", {"units": UNITS, "backcast_length": BACKCAST_LENGTH,
                     "forecast_length": FORECAST_LENGTH, "thetas_dim": THETAS_DIM,
                     "share_weights": False}),
    ("WaveletV3", {"units": UNITS, "backcast_length": BACKCAST_LENGTH,
                   "forecast_length": FORECAST_LENGTH, "basis_dim": BASIS_DIM}),
    ("TrendWavelet", {"units": UNITS, "backcast_length": BACKCAST_LENGTH,
                      "forecast_length": FORECAST_LENGTH, "trend_dim": THETAS_DIM,
                      "wavelet_dim": BASIS_DIM}),
]


class TestActiveGSplitModes:
    """Verify active_g='backcast' and active_g='forecast' split modes."""

    @pytest.mark.parametrize("block_name,kwargs", ACTIVE_G_BLOCK_CONFIGS)
    def test_active_g_true_both_nonnegative(self, block_name, kwargs):
        """active_g=True should produce non-negative backcast and forecast (ReLU)."""
        block = getattr(b, block_name)(active_g=True, **kwargs)
        torch.manual_seed(0)
        x = torch.randn(8, BACKCAST_LENGTH)
        backcast, forecast = block(x)
        assert (backcast >= 0).all(), f"{block_name} backcast has negatives with active_g=True"
        assert (forecast >= 0).all(), f"{block_name} forecast has negatives with active_g=True"

    @pytest.mark.parametrize("block_name,kwargs", ACTIVE_G_BLOCK_CONFIGS)
    def test_active_g_backcast_only(self, block_name, kwargs):
        """active_g='backcast' should produce non-negative backcast but allow negative forecast."""
        block = getattr(b, block_name)(active_g='backcast', **kwargs)
        torch.manual_seed(0)
        x = torch.randn(8, BACKCAST_LENGTH)
        backcast, forecast = block(x)
        assert (backcast >= 0).all(), f"{block_name} backcast has negatives with active_g='backcast'"
        # forecast should allow negatives (not activated)

    @pytest.mark.parametrize("block_name,kwargs", ACTIVE_G_BLOCK_CONFIGS)
    def test_active_g_forecast_only(self, block_name, kwargs):
        """active_g='forecast' should produce non-negative forecast but allow negative backcast."""
        block = getattr(b, block_name)(active_g='forecast', **kwargs)
        torch.manual_seed(0)
        x = torch.randn(8, BACKCAST_LENGTH)
        backcast, forecast = block(x)
        assert (forecast >= 0).all(), f"{block_name} forecast has negatives with active_g='forecast'"
        # backcast should allow negatives (not activated)

    @pytest.mark.parametrize("block_name,kwargs", ACTIVE_G_BLOCK_CONFIGS)
    def test_active_g_false_allows_negatives(self, block_name, kwargs):
        """active_g=False should allow negative values in both outputs."""
        block = getattr(b, block_name)(active_g=False, **kwargs)
        torch.manual_seed(0)
        x = torch.randn(32, BACKCAST_LENGTH)
        backcast, forecast = block(x)
        # With random init and enough samples, we expect some negatives
        has_negative = (backcast < 0).any() or (forecast < 0).any()
        assert has_negative, f"{block_name} produced no negatives with active_g=False (unlikely)"

    @pytest.mark.parametrize("block_name,kwargs", ACTIVE_G_BLOCK_CONFIGS)
    def test_active_g_output_shapes_unchanged(self, block_name, kwargs):
        """Split modes should not change output shapes."""
        for mode in [False, True, 'backcast', 'forecast']:
            block = getattr(b, block_name)(active_g=mode, **kwargs)
            x = torch.randn(4, BACKCAST_LENGTH)
            backcast, forecast = block(x)
            assert backcast.shape == (4, BACKCAST_LENGTH), \
                f"{block_name} backcast shape wrong with active_g={mode}"
            assert forecast.shape == (4, FORECAST_LENGTH), \
                f"{block_name} forecast shape wrong with active_g={mode}"


# --- Asymmetric wavelet family tests ---

class TestAsymmetricWaveletFamily:
    """Verify backcast_wavelet_type and forecast_wavelet_type produce distinct bases."""

    @pytest.mark.parametrize("block_cls_name", [
        "WaveletV3", "WaveletV3AE", "WaveletV3AELG", "WaveletV3VAE",
    ])
    def test_asymmetric_params_stored(self, block_cls_name):
        """Block stores backcast_wavelet_type and forecast_wavelet_type attributes."""
        block_class = getattr(b, block_cls_name)
        kwargs = {"units": UNITS, "backcast_length": BACKCAST_LENGTH,
                  "forecast_length": FORECAST_LENGTH, "basis_dim": BASIS_DIM,
                  "wavelet_type": "db3",
                  "backcast_wavelet_type": "sym20",
                  "forecast_wavelet_type": "coif2"}
        if block_cls_name != "WaveletV3":
            kwargs["latent_dim"] = LATENT_DIM
        block = block_class(**kwargs)
        assert block.backcast_wavelet_type == "sym20"
        assert block.forecast_wavelet_type == "coif2"

    @pytest.mark.parametrize("block_cls_name", [
        "WaveletV3", "WaveletV3AE", "WaveletV3AELG", "WaveletV3VAE",
    ])
    def test_asymmetric_output_shapes(self, block_cls_name):
        """Asymmetric wavelet families still produce correct output shapes."""
        block_class = getattr(b, block_cls_name)
        kwargs = {"units": UNITS, "backcast_length": BACKCAST_LENGTH,
                  "forecast_length": FORECAST_LENGTH, "basis_dim": BASIS_DIM,
                  "wavelet_type": "db3",
                  "backcast_wavelet_type": "sym20",
                  "forecast_wavelet_type": "coif2"}
        if block_cls_name != "WaveletV3":
            kwargs["latent_dim"] = LATENT_DIM
        block = block_class(**kwargs)
        x = torch.randn(4, BACKCAST_LENGTH)
        backcast, forecast = block(x)
        assert backcast.shape == (4, BACKCAST_LENGTH)
        assert forecast.shape == (4, FORECAST_LENGTH)

    def test_fallback_uses_wavelet_type(self):
        """When backcast/forecast overrides are None, both paths use wavelet_type."""
        block = b.WaveletV3(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, basis_dim=BASIS_DIM,
            wavelet_type="coif2")
        assert block.backcast_wavelet_type is None
        assert block.forecast_wavelet_type is None
        # Forward pass works with symmetric fallback
        x = torch.randn(4, BACKCAST_LENGTH)
        backcast, forecast = block(x)
        assert backcast.shape == (4, BACKCAST_LENGTH)
        assert forecast.shape == (4, FORECAST_LENGTH)

    def test_partial_override_backcast_only(self):
        """Only backcast_wavelet_type set; forecast falls back to wavelet_type."""
        block = b.WaveletV3AELG(
            units=UNITS, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, basis_dim=BASIS_DIM,
            latent_dim=LATENT_DIM, wavelet_type="db3",
            backcast_wavelet_type="sym20")
        assert block.backcast_wavelet_type == "sym20"
        assert block.forecast_wavelet_type is None
        x = torch.randn(4, BACKCAST_LENGTH)
        backcast, forecast = block(x)
        assert backcast.shape == (4, BACKCAST_LENGTH)
        assert forecast.shape == (4, FORECAST_LENGTH)


class TestControlBases:
    """Non-wavelet orthonormal control bases used to test whether the benefit of a
    frozen basis is wavelet-specific or a property of any frozen low-rank
    orthonormal projection."""

    KINDS = ["dct", "random_ortho", "random_ortho:7"]

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("target_length,basis_dim", [(30, 6), (96, 96), (480, 96)])
    def test_shape_matches_wavelet_path(self, kind, target_length, basis_dim):
        wav = b._WaveletGeneratorV3._build_basis(target_length, "db3", 5, basis_dim)
        ctl = b._WaveletGeneratorV3._build_basis(target_length, kind, 5, basis_dim)
        assert ctl.shape == wav.shape
        assert ctl.numel() == wav.numel()

    @pytest.mark.parametrize("kind", KINDS)
    def test_rows_are_orthonormal(self, kind):
        basis = b._WaveletGeneratorV3._build_basis(120, kind, 5, 40)
        gram = basis @ basis.T
        assert torch.allclose(gram, torch.eye(40), atol=1e-4)

    @pytest.mark.parametrize("kind", KINDS)
    def test_deterministic_across_calls(self, kind):
        a = b._WaveletGeneratorV3._build_basis(64, kind, 5, 16)
        c = b._WaveletGeneratorV3._build_basis(64, kind, 5, 16)
        assert torch.equal(a, c)

    def test_random_draws_differ_by_seed(self):
        a = b._WaveletGeneratorV3._build_basis(64, "random_ortho", 5, 16)
        c = b._WaveletGeneratorV3._build_basis(64, "random_ortho:7", 5, 16)
        assert not torch.allclose(a, c)

    @pytest.mark.parametrize("kind", KINDS)
    def test_block_forward_and_param_parity(self, kind):
        wav = b.TrendWavelet(256, 30, 6, wavelet_dim=6, wavelet_type="db3")
        ctl = b.TrendWavelet(256, 30, 6, wavelet_dim=6, wavelet_type=kind)
        assert sum(p.numel() for p in ctl.parameters()) == sum(
            p.numel() for p in wav.parameters()
        )
        backcast, forecast = ctl(torch.randn(4, 30))
        assert backcast.shape == (4, 30)
        assert forecast.shape == (4, 6)
