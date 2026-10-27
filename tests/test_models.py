"""Tests for NBeatsNet and NHiTSNet models — width selection, optimizer dispatch, NHiTS semantics."""

import pytest
import torch
from torch import optim

from lightningnbeats.models import NBeatsNet, NHiTSNet


def _make_model(stack_types, **kwargs):
    """Helper to create a minimal NBeatsNet model."""
    defaults = dict(
        backcast_length=20,
        forecast_length=5,
        stack_types=stack_types,
        n_blocks_per_stack=1,
        share_weights=False,
        thetas_dim=4,
        active_g=False,
        latent_dim=4,
        basis_dim=16,
    )
    defaults.update(kwargs)
    return NBeatsNet(**defaults)


class _ConstantLoss(torch.nn.Module):
    """Deterministic loss stub for isolating KL weighting behavior."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, forecast, y):
        return forecast.new_tensor(self.value)


# --- Width selection tests ---


class TestWidthSelection:
    """Verify each block type uses the correct width parameter."""

    def test_generic_uses_g_width(self):
        model = _make_model(
            ["Generic"], g_width=64, s_width=128, t_width=32, ae_width=96
        )
        block = model.stacks[0][0]
        assert block.units == 64

    def test_generic_ae_uses_g_width(self):
        model = _make_model(
            ["GenericAE"], g_width=64, s_width=128, t_width=32, ae_width=96
        )
        block = model.stacks[0][0]
        assert block.units == 64

    def test_generic_ae_backcast_uses_g_width(self):
        model = _make_model(
            ["GenericAEBackcast"], g_width=64, s_width=128, t_width=32, ae_width=96
        )
        block = model.stacks[0][0]
        assert block.units == 64

    def test_generic_ae_backcast_ae_uses_g_width(self):
        model = _make_model(
            ["GenericAEBackcastAE"], g_width=64, s_width=128, t_width=32, ae_width=96
        )
        block = model.stacks[0][0]
        assert block.units == 64

    def test_seasonality_uses_s_width(self):
        model = _make_model(
            ["Seasonality"], g_width=64, s_width=128, t_width=32, ae_width=96
        )
        block = model.stacks[0][0]
        assert block.units == 128

    def test_seasonality_ae_uses_s_width(self):
        model = _make_model(
            ["SeasonalityAE"], g_width=64, s_width=128, t_width=32, ae_width=96
        )
        block = model.stacks[0][0]
        assert block.units == 128

    def test_trend_uses_t_width(self):
        model = _make_model(["Trend"], g_width=64, s_width=128, t_width=32, ae_width=96)
        block = model.stacks[0][0]
        assert block.units == 32

    def test_trend_ae_uses_t_width(self):
        model = _make_model(
            ["TrendAE"], g_width=64, s_width=128, t_width=32, ae_width=96
        )
        block = model.stacks[0][0]
        assert block.units == 32

    def test_trend_wavelet_uses_t_width(self):
        model = _make_model(
            ["TrendWavelet"], g_width=64, s_width=128, t_width=32, ae_width=96
        )
        block = model.stacks[0][0]
        assert block.units == 32

    def test_autoencoder_uses_ae_width(self):
        model = _make_model(
            ["AutoEncoder"], g_width=64, s_width=128, t_width=32, ae_width=96
        )
        block = model.stacks[0][0]
        assert block.units == 96

    def test_autoencoder_ae_uses_ae_width(self):
        model = _make_model(
            ["AutoEncoderAE"], g_width=64, s_width=128, t_width=32, ae_width=96
        )
        block = model.stacks[0][0]
        assert block.units == 96

    def test_wavelet_uses_g_width_fallback(self):
        model = _make_model(
            ["HaarWaveletV3"], g_width=64, s_width=128, t_width=32, ae_width=96
        )
        block = model.stacks[0][0]
        assert block.units == 64


class TestAEDescendantRouting:
    """Verify NBeatsNet routes latent_dim and thetas_dim to repaired descendants."""

    @pytest.mark.parametrize(
        "block_name",
        [
            "AutoEncoderAE",
            "AutoEncoderAELG",
            "AutoEncoderVAE",
        ],
    )
    def test_autoencoder_descendants_route_latent_dim_to_branch_layers(
        self, block_name
    ):
        latent_dim = 7
        model = _make_model(
            [block_name], ae_width=96, thetas_dim=3, latent_dim=latent_dim
        )
        block = model.stacks[0][0]

        assert block.units == 96
        assert block.b_encoder.out_features == latent_dim
        assert block.b_decoder.in_features == latent_dim
        assert block.forecast_g.in_features == 96

    @pytest.mark.parametrize(
        "block_name",
        [
            "GenericAEBackcastAE",
            "GenericAEBackcastAELG",
            "GenericAEBackcastVAE",
        ],
    )
    def test_generic_ae_backcast_descendants_split_latent_and_theta_dims(
        self, block_name
    ):
        latent_dim = 7
        thetas_dim = 3
        model = _make_model(
            [block_name], g_width=64, thetas_dim=thetas_dim, latent_dim=latent_dim
        )
        block = model.stacks[0][0]

        assert block.units == 64
        assert block.b_encoder.out_features == latent_dim
        assert block.b_decoder.in_features == latent_dim
        assert block.forecast_linear.out_features == thetas_dim
        assert block.forecast_g.in_features == thetas_dim


class TestNHiTSAEDescendantRouting:
    """Verify NHiTSNet mirrors AE descendant latent/theta routing semantics."""

    @pytest.mark.parametrize(
        "block_name",
        [
            "GenericAEBackcastAE",
            "GenericAEBackcastAELG",
            "GenericAEBackcastVAE",
        ],
    )
    def test_generic_ae_backcast_descendants_split_latent_and_theta_dims(
        self, block_name
    ):
        latent_dim = 7
        thetas_dim = 3
        model = _make_nhits(
            [block_name],
            g_width=64,
            thetas_dim=thetas_dim,
            latent_dim=latent_dim,
            n_pools_kernel_size=[1],
            n_freq_downsample=[1],
        )
        block = model.stacks[0][0]

        assert block.units == 64
        assert block.b_encoder.out_features == latent_dim
        assert block.b_decoder.in_features == latent_dim
        assert block.forecast_linear.out_features == thetas_dim
        assert block.forecast_g.in_features == thetas_dim


# --- Optimizer dispatch tests ---


class TestOptimizerDispatch:
    """Verify optimizer_name parameter is respected."""

    def test_adam_optimizer(self):
        model = _make_model(["Generic"], optimizer_name="Adam")
        opt = model.configure_optimizers()
        assert isinstance(opt, optim.Adam)

    def test_sgd_optimizer(self):
        model = _make_model(["Generic"], optimizer_name="SGD")
        opt = model.configure_optimizers()
        assert isinstance(opt, optim.SGD)

    def test_adamw_optimizer(self):
        model = _make_model(["Generic"], optimizer_name="AdamW")
        opt = model.configure_optimizers()
        assert isinstance(opt, optim.AdamW)

    def test_rmsprop_optimizer(self):
        model = _make_model(["Generic"], optimizer_name="RMSprop")
        opt = model.configure_optimizers()
        assert isinstance(opt, optim.RMSprop)

    def test_invalid_optimizer_raises(self):
        model = _make_model(["Generic"])
        model.optimizer_name = "InvalidOptimizer"
        with pytest.raises(ValueError, match="Unknown optimizer name"):
            model.configure_optimizers()


class TestWeightDecay:
    """Verify weight_decay is stored and forwarded to the optimizer."""

    def test_weight_decay_forwarded_to_optimizer(self):
        model = _make_model(["Generic"], weight_decay=0.01)
        opt = model.configure_optimizers()
        assert model.weight_decay == 0.01
        assert all(pg["weight_decay"] == 0.01 for pg in opt.param_groups)

    def test_weight_decay_default_none_keeps_adam_zero(self):
        model = _make_model(["Generic"])
        opt = model.configure_optimizers()
        assert model.weight_decay is None
        assert all(pg["weight_decay"] == 0.0 for pg in opt.param_groups)

    def test_weight_decay_default_none_preserves_adamw_default(self):
        # Old code never passed the kwarg, so AdamW used its own 0.01 default.
        # weight_decay=None must keep that behavior byte-identical.
        model = _make_model(["Generic"], optimizer_name="AdamW")
        opt = model.configure_optimizers()
        assert all(pg["weight_decay"] == 0.01 for pg in opt.param_groups)

    def test_weight_decay_explicit_zero_overrides_adamw_default(self):
        model = _make_model(["Generic"], optimizer_name="AdamW", weight_decay=0.0)
        opt = model.configure_optimizers()
        assert all(pg["weight_decay"] == 0.0 for pg in opt.param_groups)

    def test_weight_decay_with_sgd(self):
        model = _make_model(["Generic"], optimizer_name="SGD", weight_decay=1e-4)
        opt = model.configure_optimizers()
        assert all(pg["weight_decay"] == 1e-4 for pg in opt.param_groups)

    def test_weight_decay_nhits(self):
        model = _make_nhits(["Generic"], weight_decay=0.01)
        opt = model.configure_optimizers()
        assert model.weight_decay == 0.01
        assert all(pg["weight_decay"] == 0.01 for pg in opt.param_groups)

    def test_weight_decay_nhits_default_none_keeps_adam_zero(self):
        model = _make_nhits(["Generic"])
        opt = model.configure_optimizers()
        assert model.weight_decay is None
        assert all(pg["weight_decay"] == 0.0 for pg in opt.param_groups)


class TestAEIntermediateWidthPlumbing:
    """Verify ae_intermediate_width flows from the model into AE-backed blocks."""

    def test_create_stack_forwards_taper_width(self):
        model = _make_model(["GenericAE"], g_width=64, ae_intermediate_width=48)
        block = model.stacks[0][0]
        assert block.fc1.out_features == 48
        assert block.fc4.in_features == 48

    def test_default_taper_is_units_half(self):
        model = _make_model(["GenericAE"], g_width=64)
        block = model.stacks[0][0]
        assert model.ae_intermediate_width is None
        assert block.fc1.out_features == 32

    def test_wavelet_v3ae_taper(self):
        model = _make_model(["WaveletV3AE"], g_width=64, ae_intermediate_width=48)
        assert model.stacks[0][0].fc1.out_features == 48

    def test_trendwavelet_ae_taper(self):
        model = _make_model(["TrendWaveletAE"], t_width=64, ae_intermediate_width=48)
        assert model.stacks[0][0].fc1.out_features == 48

    def test_non_ae_blocks_unaffected(self):
        model = _make_model(
            ["Generic", "Trend", "Seasonality"],
            g_width=64,
            t_width=32,
            s_width=64,
            ae_intermediate_width=48,
        )
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)

    def test_nhits_forwards_taper_width(self):
        model = _make_nhits(["GenericAE"], g_width=64, ae_intermediate_width=48)
        assert model.stacks[0][0].fc1.out_features == 48


class TestPlateauSchedulerConfig:
    """Verify configure_optimizers wires interval/frequency for plateau."""

    def test_plateau_default_epoch_cadence(self):
        cfg = {"type": "plateau", "factor": 0.5, "patience": 5}
        model = _make_model(["Generic"], lr_scheduler_config=cfg)
        result = model.configure_optimizers()
        assert isinstance(result, dict)
        sched = result["lr_scheduler"]
        assert sched["interval"] == "epoch"
        assert sched["frequency"] == 1
        assert sched["monitor"] == "val_loss"

    def test_plateau_step_cadence_forwarded(self):
        cfg = {
            "type": "plateau",
            "factor": 0.75,
            "patience": 0,
            "cooldown": 0,
            "min_lr": 1e-6,
            "interval": "step",
            "frequency": 100,
        }
        model = _make_model(["Generic"], lr_scheduler_config=cfg)
        result = model.configure_optimizers()
        sched = result["lr_scheduler"]
        assert sched["interval"] == "step"
        assert sched["frequency"] == 100
        # ReduceLROnPlateau's hyperparameters reflect the YAML inputs
        plateau = sched["scheduler"]
        assert plateau.factor == pytest.approx(0.75)
        assert plateau.patience == 0


# --- Forward pass shape tests ---


class TestForwardPass:
    """Verify model forward pass produces correct output shapes."""

    def test_generic_forward_shape(self):
        model = _make_model(["Generic"], g_width=32)
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)

    def test_trend_seasonality_forward_shape(self):
        model = _make_model(["Trend", "Seasonality"], t_width=32, s_width=64)
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)

    def test_invalid_stack_type_raises(self):
        with pytest.raises(ValueError, match="Stack architecture must be specified"):
            NBeatsNet(backcast_length=20, forecast_length=5, stack_types=None)


# --- Model defaults tests ---


class TestModelDefaults:
    """Verify NBeatsNet default parameter values."""

    def test_active_g_defaults_to_false(self):
        model = NBeatsNet(
            backcast_length=20,
            forecast_length=5,
            stack_types=["Generic"],
            n_blocks_per_stack=1,
        )
        assert model.active_g is False

    def test_learning_rate_defaults_to_1e_minus_3(self):
        model = NBeatsNet(
            backcast_length=20,
            forecast_length=5,
            stack_types=["Generic"],
            n_blocks_per_stack=1,
        )
        assert model.learning_rate == 1e-3

    def test_latent_gate_fn_defaults_to_sigmoid(self):
        model = NBeatsNet(
            backcast_length=20,
            forecast_length=5,
            stack_types=["GenericAELG"],
            n_blocks_per_stack=1,
        )
        assert model.latent_gate_fn == "sigmoid"
        assert model.stacks[0][0].latent_gate_fn == "sigmoid"

    def test_kl_weight_defaults_to_point_001(self):
        model = NBeatsNet(
            backcast_length=20,
            forecast_length=5,
            stack_types=["VAE"],
            n_blocks_per_stack=1,
        )
        assert model.kl_weight == 0.001


# --- sum_losses semantic fix tests ---


class TestSumLossesBehavior:
    """Verify sum_losses uses zero target for backcast loss."""

    def test_sum_losses_backcast_uses_zero_target_in_training(self):
        model = _make_model(["Generic"], sum_losses=True, g_width=32)
        x = torch.randn(4, 20)
        y = torch.randn(4, 5)
        batch = (x, y)

        loss = model.training_step(batch, 0)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_sum_losses_backcast_uses_zero_target_in_validation(self):
        model = _make_model(["Generic"], sum_losses=True, g_width=32, no_val=False)
        x = torch.randn(4, 20)
        y = torch.randn(4, 5)
        batch = (x, y)

        loss = model.validation_step(batch, 0)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)

    def test_sum_losses_backcast_uses_zero_target_in_test(self):
        model = _make_model(["Generic"], sum_losses=True, g_width=32)
        x = torch.randn(4, 20)
        y = torch.randn(4, 5)
        batch = (x, y)

        loss = model.test_step(batch, 0)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)


# --- KL weighting tests ---


class TestKLWeightBehavior:
    """Verify configurable KL weighting in shared training logic."""

    def test_nbeats_invalid_kl_weight_raises(self):
        with pytest.raises(
            ValueError, match="kl_weight must be a non-negative real number"
        ):
            _make_model(["VAE"], kl_weight=-0.1)
        with pytest.raises(
            ValueError, match="kl_weight must be a non-negative real number"
        ):
            _make_model(["VAE"], kl_weight="bad")
        with pytest.raises(
            ValueError, match="kl_weight must be a non-negative real number"
        ):
            _make_model(["VAE"], kl_weight=True)
        with pytest.raises(
            ValueError, match="kl_weight must be a non-negative real number"
        ):
            _make_model(["VAE"], kl_weight=False)

    def test_nbeats_zero_kl_weight_disables_kl_term(self):
        model = _make_model(["VAE"], kl_weight=0.0, ae_width=32)
        model.loss_fn = _ConstantLoss(2.0)
        model.forward = lambda x: (
            torch.zeros(
                x.shape[0], model.backcast_length, device=x.device, dtype=x.dtype
            ),
            torch.zeros(
                x.shape[0], model.forecast_length, device=x.device, dtype=x.dtype
            ),
        )
        model.stacks[0][0].kl_loss = torch.tensor(5.0)

        loss = model.training_step((torch.randn(4, 20), torch.randn(4, 5)), 0)

        assert torch.isclose(loss, torch.tensor(2.0))

    def test_nbeats_custom_kl_weight_scales_loss(self):
        model = _make_model(["VAE"], kl_weight=0.3, ae_width=32)
        model.loss_fn = _ConstantLoss(2.0)
        model.forward = lambda x: (
            torch.zeros(
                x.shape[0], model.backcast_length, device=x.device, dtype=x.dtype
            ),
            torch.zeros(
                x.shape[0], model.forecast_length, device=x.device, dtype=x.dtype
            ),
        )
        model.stacks[0][0].kl_loss = torch.tensor(5.0)

        loss = model.training_step((torch.randn(4, 20), torch.randn(4, 5)), 0)

        assert torch.isclose(loss, torch.tensor(3.5))

    def test_nhits_zero_kl_weight_disables_kl_term(self):
        model = _make_nhits(
            ["VAE"],
            kl_weight=0.0,
            ae_width=32,
            n_pools_kernel_size=[1],
            n_freq_downsample=[1],
        )
        model.loss_fn = _ConstantLoss(2.0)
        model.forward = lambda x: (
            torch.zeros(
                x.shape[0], model.backcast_length, device=x.device, dtype=x.dtype
            ),
            torch.zeros(
                x.shape[0], model.forecast_length, device=x.device, dtype=x.dtype
            ),
        )
        model.stacks[0][0].kl_loss = torch.tensor(5.0)

        loss = model.training_step((torch.randn(4, 24), torch.randn(4, 6)), 0)

        assert torch.isclose(loss, torch.tensor(2.0))

    def test_nhits_custom_kl_weight_scales_loss(self):
        model = _make_nhits(
            ["VAE"],
            kl_weight=0.3,
            ae_width=32,
            n_pools_kernel_size=[1],
            n_freq_downsample=[1],
        )
        model.loss_fn = _ConstantLoss(2.0)
        model.forward = lambda x: (
            torch.zeros(
                x.shape[0], model.backcast_length, device=x.device, dtype=x.dtype
            ),
            torch.zeros(
                x.shape[0], model.forecast_length, device=x.device, dtype=x.dtype
            ),
        )
        model.stacks[0][0].kl_loss = torch.tensor(5.0)

        loss = model.training_step((torch.randn(4, 24), torch.randn(4, 6)), 0)

        assert torch.isclose(loss, torch.tensor(3.5))


# --- NormalizedDeviationLoss configuration tests ---


class TestNormalizedDeviationLossConfiguration:
    """Verify NormalizedDeviationLoss is handled in configure_loss."""

    def test_model_with_normalized_deviation_loss(self):
        from lightningnbeats.losses import NormalizedDeviationLoss

        model = _make_model(["Generic"], loss="NormalizedDeviationLoss", g_width=32)
        assert isinstance(model.loss_fn, NormalizedDeviationLoss)

    def test_normalized_deviation_loss_forward_pass(self):
        model = _make_model(["Generic"], loss="NormalizedDeviationLoss", g_width=32)
        x = torch.randn(4, 20)
        y = torch.randn(4, 5)
        batch = (x, y)

        loss = model.training_step(batch, 0)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)


# --- BottleneckGeneric width selection tests ---


class TestBottleneckGenericWidthSelection:
    """Verify BottleneckGeneric and BottleneckGenericAE use g_width."""

    def test_bottleneck_generic_uses_g_width(self):
        model = _make_model(
            ["BottleneckGeneric"], g_width=64, s_width=128, t_width=32, ae_width=96
        )
        block = model.stacks[0][0]
        assert block.units == 64

    def test_bottleneck_generic_ae_uses_g_width(self):
        model = _make_model(
            ["BottleneckGenericAE"], g_width=64, s_width=128, t_width=32, ae_width=96
        )
        block = model.stacks[0][0]
        assert block.units == 64


# --- active_g split mode model-level tests ---


class TestActiveGSplitModesModel:
    """Verify string active_g modes are accepted and passed to blocks."""

    def test_active_g_backcast_accepted(self):
        model = _make_model(["Generic"], active_g="backcast", g_width=32)
        assert model.active_g == "backcast"
        block = model.stacks[0][0]
        assert block.active_g == "backcast"

    def test_active_g_forecast_accepted(self):
        model = _make_model(["Generic"], active_g="forecast", g_width=32)
        assert model.active_g == "forecast"
        block = model.stacks[0][0]
        assert block.active_g == "forecast"

    def test_active_g_true_still_works(self):
        model = _make_model(["Generic"], active_g=True, g_width=32)
        assert model.active_g is True
        block = model.stacks[0][0]
        assert block.active_g is True

    def test_active_g_false_still_works(self):
        model = _make_model(["Generic"], active_g=False, g_width=32)
        assert model.active_g is False

    def test_active_g_invalid_string_raises(self):
        with pytest.raises(ValueError, match="active_g must be"):
            _make_model(["Generic"], active_g="invalid")

    def test_latent_gate_fn_invalid_string_raises(self):
        with pytest.raises(ValueError, match="latent_gate_fn must be one of"):
            _make_model(["GenericAELG"], latent_gate_fn="invalid")

    def test_latent_gate_fn_routes_to_aelg_blocks(self):
        model = _make_model(["GenericAELG"], latent_gate_fn="wavy_sigmoid", g_width=32)
        assert model.latent_gate_fn == "wavy_sigmoid"
        assert model.stacks[0][0].latent_gate_fn == "wavy_sigmoid"

    def test_wavelet_latent_gate_fn_routes_to_aelg_blocks(self):
        model = _make_model(
            ["GenericAELG"], latent_gate_fn="wavelet_sigmoid", g_width=32
        )
        assert model.latent_gate_fn == "wavelet_sigmoid"
        assert model.stacks[0][0].latent_gate_fn == "wavelet_sigmoid"

    def test_active_g_backcast_forward_pass(self):
        model = _make_model(["Generic"], active_g="backcast", g_width=32)
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)

    def test_active_g_forecast_forward_pass(self):
        model = _make_model(["Generic"], active_g="forecast", g_width=32)
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)


# --- active_g per-stack list tests (NBeatsNet) ---


class TestActiveGPerStackNBeats:
    """Verify per-stack active_g list is applied independently to each stack."""

    def test_list_broadcasts_per_stack(self):
        model = _make_model(
            ["Generic", "Generic", "Generic"], active_g=[True, True, False], g_width=32
        )
        assert model.stacks[0][0].active_g is True
        assert model.stacks[1][0].active_g is True
        assert model.stacks[2][0].active_g is False

    def test_list_preserved_on_model(self):
        model = _make_model(["Generic", "Generic"], active_g=[True, False], g_width=32)
        assert model.active_g == [True, False]
        assert model._active_g_per_stack == [True, False]

    def test_mixed_modes_per_stack(self):
        model = _make_model(
            ["Generic", "Generic", "Generic"],
            active_g=[True, "forecast", False],
            g_width=32,
        )
        assert model.stacks[0][0].active_g is True
        assert model.stacks[1][0].active_g == "forecast"
        assert model.stacks[2][0].active_g is False

    def test_list_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="active_g list length"):
            _make_model(
                ["Generic", "Generic"], active_g=[True, False, True], g_width=32
            )

    def test_invalid_list_entry_raises(self):
        with pytest.raises(ValueError, match="active_g must be"):
            _make_model(["Generic", "Generic"], active_g=[True, "invalid"], g_width=32)

    def test_list_applies_to_all_blocks_within_stack(self):
        model = _make_model(
            ["Generic", "Generic"],
            active_g=[True, False],
            n_blocks_per_stack=2,
            g_width=32,
        )
        assert model.stacks[0][0].active_g is True
        assert model.stacks[0][1].active_g is True
        assert model.stacks[1][0].active_g is False
        assert model.stacks[1][1].active_g is False

    def test_forward_pass_with_per_stack_list(self):
        model = _make_model(
            ["Generic", "Generic", "Generic"], active_g=[True, True, False], g_width=32
        )
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)

    def test_last_stack_unclamped_produces_signed_forecasts(self):
        """With last stack active_g=False, its raw forecast can be negative."""
        torch.manual_seed(0)
        model = _make_model(
            ["Generic", "Generic", "Generic"], active_g=[True, True, False], g_width=32
        )
        # Probe the last block directly: its forecast output is pure linear
        # (no activation), so over a random batch it must produce some negatives.
        last_block = model.stacks[-1][0]
        _, y_last = last_block(torch.randn(64, 20))
        assert (y_last < 0).any()


# --- forecast_basis_dim passthrough tests ---


class TestForecastBasisDim:
    """Verify forecast_basis_dim is passed through to V3 wavelet blocks."""

    def test_forecast_basis_dim_passed_to_v3_blocks(self):
        model = _make_model(
            ["HaarWaveletV3"] * 2,
            basis_dim=16,
            forecast_basis_dim=4,
            backcast_length=30,
            forecast_length=6,
        )
        for stack in model.stacks:
            for block in stack:
                assert block.forecast_linear.out_features == min(4, 6)
                assert block.backcast_linear.out_features == min(16, 30)
        x = torch.randn(4, 30)
        _, forecast = model(x)
        assert forecast.shape == (4, 6)

    def test_forecast_basis_dim_none_uses_basis_dim(self):
        model = _make_model(
            ["HaarWaveletV3"],
            basis_dim=16,
            forecast_basis_dim=None,
            backcast_length=30,
            forecast_length=6,
        )
        block = model.stacks[0][0]
        assert block.forecast_linear.out_features == min(16, 6)
        assert block.backcast_linear.out_features == min(16, 30)


# --- trend_thetas_dim tests ---


class TestTrendThetasDim:
    """Verify trend_thetas_dim routes only to trend-family blocks."""

    def test_trend_thetas_dim_overrides_trend_block(self):
        """Trend block should use trend_thetas_dim instead of global thetas_dim."""
        model = _make_model(["Trend"], t_width=32, thetas_dim=8, trend_thetas_dim=3)
        block = model.stacks[0][0]
        # Trend projects to thetas_dim coefficients
        assert block.backcast_linear.out_features == 3
        assert block.forecast_linear.out_features == 3

    def test_trend_thetas_dim_overrides_trend_ae_block(self):
        """TrendAE block should use trend_thetas_dim instead of global thetas_dim."""
        model = _make_model(
            ["TrendAE"], t_width=32, thetas_dim=8, trend_thetas_dim=2, latent_dim=4
        )
        block = model.stacks[0][0]
        assert block.backcast_linear.out_features == 2
        assert block.forecast_linear.out_features == 2

    def test_trend_thetas_dim_overrides_trend_wavelet_block(self):
        """TrendWavelet should use trend_thetas_dim instead of global thetas_dim."""
        model = _make_model(
            ["TrendWavelet"], t_width=32, basis_dim=7, trend_thetas_dim=2
        )
        block = model.stacks[0][0]
        assert block.trend_dim == 2
        assert block.backcast_linear.out_features == 2 + min(7, 20)
        assert block.forecast_linear.out_features == 2 + min(7, 5)

    def test_trend_thetas_dim_does_not_affect_bottleneck_generic(self):
        """BottleneckGeneric block should still use global thetas_dim."""
        model = _make_model(
            ["BottleneckGeneric"], g_width=32, thetas_dim=8, trend_thetas_dim=3
        )
        block = model.stacks[0][0]
        # BottleneckGeneric uses thetas_dim as its bottleneck
        assert block.backcast_linear.out_features == 8
        assert block.forecast_linear.out_features == 8

    def test_trend_thetas_dim_defaults_to_3(self):
        """When trend_thetas_dim is not specified, Trend uses default of 3."""
        model = _make_model(["Trend"], t_width=32, thetas_dim=8)
        block = model.stacks[0][0]
        assert block.backcast_linear.out_features == 3

    def test_trend_thetas_dim_none_falls_back_to_global_thetas_dim(self):
        """trend_thetas_dim=None should use global thetas_dim for Trend blocks."""
        model = _make_model(["Trend"], t_width=32, thetas_dim=8, trend_thetas_dim=None)
        block = model.stacks[0][0]
        assert block.backcast_linear.out_features == 8
        assert block.forecast_linear.out_features == 8

    def test_trend_thetas_dim_invalid_raises(self):
        """trend_thetas_dim must be a positive integer."""
        with pytest.raises(
            ValueError, match="trend_thetas_dim must be a positive integer"
        ):
            _make_model(["Trend"], t_width=32, trend_thetas_dim=0)
        with pytest.raises(
            ValueError, match="trend_thetas_dim must be a positive integer"
        ):
            _make_model(["Trend"], t_width=32, trend_thetas_dim=-1)

    def test_trend_thetas_dim_accepts_any_positive_int(self):
        """trend_thetas_dim=5 should work as a valid positive integer."""
        model = _make_model(["Trend"], t_width=32, trend_thetas_dim=5)
        block = model.stacks[0][0]
        assert block.backcast_linear.out_features == 5

    def test_mixed_stack_routing(self):
        """In a Trend+BottleneckGeneric stack, each gets the correct thetas_dim."""
        model = _make_model(
            ["Trend", "BottleneckGeneric"],
            t_width=32,
            g_width=32,
            thetas_dim=7,
            trend_thetas_dim=3,
        )
        trend_block = model.stacks[0][0]
        bg_block = model.stacks[1][0]
        assert trend_block.backcast_linear.out_features == 3
        assert bg_block.backcast_linear.out_features == 7

    def test_trend_thetas_dim_forward_shape(self):
        """Forward pass works correctly with trend_thetas_dim set."""
        model = _make_model(
            ["Trend", "Generic"],
            t_width=32,
            g_width=32,
            thetas_dim=7,
            trend_thetas_dim=3,
        )
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)


# --- wavelet_type forwarding tests ---


class TestWaveletTypeForwarding:
    """Verify wavelet_type is forwarded to TrendWavelet-family blocks."""

    def test_wavelet_type_forwarded_to_trendwavelet(self):
        """TrendWavelet blocks should receive the wavelet_type parameter."""
        model = _make_model(["TrendWavelet"], t_width=64, wavelet_type="haar")
        block = model.stacks[0][0]
        assert block.wavelet_type == "haar"

    def test_wavelet_type_forwarded_to_trendwaveletae(self):
        """TrendWaveletAE blocks should receive the wavelet_type parameter."""
        model = _make_model(["TrendWaveletAE"], g_width=64, wavelet_type="haar")
        block = model.stacks[0][0]
        assert block.wavelet_type == "haar"

    def test_wavelet_type_forwarded_to_trendwaveletaelg(self):
        """TrendWaveletAELG blocks should receive the wavelet_type parameter."""
        model = _make_model(["TrendWaveletAELG"], g_width=64, wavelet_type="haar")
        block = model.stacks[0][0]
        assert block.wavelet_type == "haar"

    def test_wavelet_type_default_is_db3(self):
        """Default wavelet_type should be 'db3'."""
        model = _make_model(["TrendWavelet"], t_width=64)
        block = model.stacks[0][0]
        assert block.wavelet_type == "db3"

    def test_wavelet_type_forward_pass(self):
        """Forward pass works with non-default wavelet_type."""
        model = _make_model(["TrendWavelet"] * 2, t_width=64, wavelet_type="haar")
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)


# ---------------------------------------------------------------------------
# NHiTSNet tests
# ---------------------------------------------------------------------------


def _make_nhits(stack_types, **kwargs):
    """Helper to create a minimal NHiTSNet model."""
    defaults = dict(
        backcast_length=24,
        forecast_length=6,
        stack_types=stack_types,
        n_blocks_per_stack=1,
        share_weights=False,
        thetas_dim=4,
        active_g=False,
        latent_dim=4,
        basis_dim=16,
        g_width=32,
    )
    defaults.update(kwargs)
    return NHiTSNet(**defaults)


class TestNHiTSNetConstruction:
    """Verify NHiTSNet instantiates correctly and stores parameters."""

    def test_basic_construction(self):
        model = _make_nhits(["Generic", "Generic", "Generic"])
        assert model.n_stacks == 3
        assert len(model.stacks) == 3

    def test_stack_types_none_raises(self):
        with pytest.raises(ValueError, match="Stack architecture must be specified"):
            NHiTSNet(backcast_length=24, forecast_length=6, stack_types=None)

    def test_invalid_active_g_raises(self):
        with pytest.raises(ValueError, match="active_g must be"):
            _make_nhits(["Generic"], active_g="invalid")

    def test_invalid_latent_gate_fn_raises(self):
        with pytest.raises(ValueError, match="latent_gate_fn must be one of"):
            _make_nhits(["GenericAELG"], latent_gate_fn="invalid")

    def test_invalid_trend_thetas_dim_raises(self):
        with pytest.raises(
            ValueError, match="trend_thetas_dim must be a positive integer"
        ):
            _make_nhits(["Trend"], trend_thetas_dim=0)

    def test_invalid_kl_weight_raises(self):
        with pytest.raises(
            ValueError, match="kl_weight must be a non-negative real number"
        ):
            _make_nhits(
                ["VAE"], kl_weight=-0.1, n_pools_kernel_size=[1], n_freq_downsample=[1]
            )
        with pytest.raises(
            ValueError, match="kl_weight must be a non-negative real number"
        ):
            _make_nhits(
                ["VAE"], kl_weight="bad", n_pools_kernel_size=[1], n_freq_downsample=[1]
            )
        with pytest.raises(
            ValueError, match="kl_weight must be a non-negative real number"
        ):
            _make_nhits(
                ["VAE"], kl_weight=True, n_pools_kernel_size=[1], n_freq_downsample=[1]
            )
        with pytest.raises(
            ValueError, match="kl_weight must be a non-negative real number"
        ):
            _make_nhits(
                ["VAE"], kl_weight=False, n_pools_kernel_size=[1], n_freq_downsample=[1]
            )

    def test_n_pools_kernel_size_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="n_pools_kernel_size length"):
            _make_nhits(["Generic", "Generic"], n_pools_kernel_size=[2])

    def test_n_freq_downsample_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="n_freq_downsample length"):
            _make_nhits(["Generic", "Generic"], n_freq_downsample=[2])

    def test_default_pools_and_downsample_filled_in(self):
        model = _make_nhits(["Generic", "Generic"])
        assert model.n_pools_kernel_size == [2, 2]
        assert model.n_freq_downsample == [1, 1]

    def test_kl_weight_defaults_to_point_001(self):
        model = _make_nhits(["VAE"], n_pools_kernel_size=[1], n_freq_downsample=[1])
        assert model.kl_weight == 0.001

    def test_custom_pools_stored(self):
        model = _make_nhits(
            ["Generic", "Generic", "Generic"],
            n_pools_kernel_size=[1, 2, 4],
            n_freq_downsample=[1, 2, 6],
        )
        assert model.n_pools_kernel_size == [1, 2, 4]
        assert model.n_freq_downsample == [1, 2, 6]

    def test_pooled_backcast_lengths_computed(self):
        model = _make_nhits(
            ["Generic", "Generic", "Generic"],
            backcast_length=24,
            n_pools_kernel_size=[1, 2, 4],
        )
        # 24//1=24, 24//2=12, 24//4=6
        assert model.pooled_backcast_lengths == [24, 12, 6]

    def test_reduced_forecast_lengths_computed(self):
        model = _make_nhits(
            ["Generic", "Generic", "Generic"],
            forecast_length=6,
            n_freq_downsample=[1, 2, 6],
        )
        # 6//1=6, 6//2=3, 6//6=1
        assert model.reduced_forecast_lengths == [6, 3, 1]

    def test_blocks_use_effective_forecast_length(self):
        """Each block's forecast output should match its reduced_forecast_length."""
        model = _make_nhits(
            ["Generic", "Generic", "Generic"],
            backcast_length=24,
            forecast_length=6,
            n_pools_kernel_size=[1, 2, 4],
            n_freq_downsample=[1, 2, 6],
        )
        expected_fl = [6, 3, 1]
        for stack_id, stack in enumerate(model.stacks):
            block = stack[0]
            assert block.forecast_length == expected_fl[stack_id], (
                f"stack {stack_id}: expected forecast_length={expected_fl[stack_id]}, "
                f"got {block.forecast_length}"
            )

    def test_blocks_use_effective_backcast_length(self):
        """Each block's backcast length should match its pooled_backcast_length."""
        model = _make_nhits(
            ["Generic", "Generic", "Generic"],
            backcast_length=24,
            forecast_length=6,
            n_pools_kernel_size=[1, 2, 4],
            n_freq_downsample=[1, 2, 6],
        )
        expected_bl = [24, 12, 6]
        for stack_id, stack in enumerate(model.stacks):
            block = stack[0]
            assert block.backcast_length == expected_bl[stack_id], (
                f"stack {stack_id}: expected backcast_length={expected_bl[stack_id]}, "
                f"got {block.backcast_length}"
            )


class TestNHiTSNetForwardShape:
    """Verify NHiTSNet forward pass produces correct output shapes."""

    def test_no_pooling_forward_shape(self):
        """With pool=1 and freq_down=1 the output is identical to N-BEATS."""
        model = _make_nhits(
            ["Generic", "Generic"],
            backcast_length=20,
            forecast_length=5,
            n_pools_kernel_size=[1, 1],
            n_freq_downsample=[1, 1],
        )
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)

    def test_multirate_forward_shape(self):
        """Multi-rate pooling + interpolation must still produce correct shapes."""
        model = _make_nhits(
            ["Generic", "Generic", "Generic"],
            backcast_length=24,
            forecast_length=6,
            n_pools_kernel_size=[1, 2, 4],
            n_freq_downsample=[1, 2, 6],
        )
        x = torch.randn(4, 24)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 24)
        assert forecast.shape == (4, 6)

    def test_single_stack_forward_shape(self):
        model = _make_nhits(
            ["Generic"],
            backcast_length=16,
            forecast_length=4,
            n_pools_kernel_size=[2],
            n_freq_downsample=[2],
        )
        x = torch.randn(2, 16)
        backcast, forecast = model(x)
        assert backcast.shape == (2, 16)
        assert forecast.shape == (2, 4)

    def test_trend_seasonality_generic_forward_shape(self):
        model = _make_nhits(
            ["Trend", "Seasonality", "Generic"],
            backcast_length=24,
            forecast_length=6,
            t_width=32,
            s_width=64,
            n_pools_kernel_size=[1, 1, 2],
            n_freq_downsample=[1, 1, 2],
        )
        x = torch.randn(4, 24)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 24)
        assert forecast.shape == (4, 6)

    def test_ae_block_forward_shape(self):
        model = _make_nhits(
            ["GenericAE", "GenericAE"],
            backcast_length=20,
            forecast_length=5,
            n_pools_kernel_size=[1, 2],
            n_freq_downsample=[1, 5],
        )
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)


class TestNHiTSNetParameters:
    """Verify NHiTS-specific parameters are stored and passed correctly."""

    def test_active_g_forecast_passthrough(self):
        model = _make_nhits(["Generic"], active_g="forecast")
        assert model.active_g == "forecast"
        assert model.stacks[0][0].active_g == "forecast"

    def test_active_g_backcast_passthrough(self):
        model = _make_nhits(["Generic"], active_g="backcast")
        assert model.active_g == "backcast"
        assert model.stacks[0][0].active_g == "backcast"

    def test_latent_gate_fn_passthrough(self):
        model = _make_nhits(
            ["GenericAELG"],
            latent_gate_fn="wavy_sigmoid",
            n_pools_kernel_size=[1],
            n_freq_downsample=[1],
        )
        assert model.latent_gate_fn == "wavy_sigmoid"
        assert model.stacks[0][0].latent_gate_fn == "wavy_sigmoid"

    def test_wavelet_latent_gate_fn_passthrough(self):
        model = _make_nhits(
            ["GenericAELG"],
            latent_gate_fn="wavelet_sigmoid",
            n_pools_kernel_size=[1],
            n_freq_downsample=[1],
        )
        assert model.latent_gate_fn == "wavelet_sigmoid"
        assert model.stacks[0][0].latent_gate_fn == "wavelet_sigmoid"

    def test_interpolation_mode_stored(self):
        model = _make_nhits(["Generic"], interpolation_mode="nearest")
        assert model.interpolation_mode == "nearest"

    def test_nearest_interpolation_forward_shape(self):
        model = _make_nhits(
            ["Generic", "Generic"],
            backcast_length=20,
            forecast_length=4,
            n_pools_kernel_size=[1, 2],
            n_freq_downsample=[1, 2],
            interpolation_mode="nearest",
        )
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 4)

    def test_sum_losses_training_step(self):
        model = _make_nhits(["Generic", "Generic"], sum_losses=True)
        x = torch.randn(4, 24)
        y = torch.randn(4, 6)
        loss = model.training_step((x, y), 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_optimizer_dispatch(self):
        model = _make_nhits(["Generic"], optimizer_name="AdamW")
        opt = model.configure_optimizers()
        assert isinstance(opt, optim.AdamW)

    def test_trend_thetas_dim_routing(self):
        """Trend blocks in NHiTSNet should respect trend_thetas_dim."""
        model = _make_nhits(
            ["Trend"],
            t_width=32,
            thetas_dim=8,
            trend_thetas_dim=3,
            n_pools_kernel_size=[1],
            n_freq_downsample=[1],
        )
        block = model.stacks[0][0]
        assert block.backcast_linear.out_features == 3
        assert block.forecast_linear.out_features == 3

    def test_trend_wavelet_thetas_dim_routing(self):
        """TrendWavelet blocks in NHiTSNet should respect trend_thetas_dim."""
        model = _make_nhits(
            ["TrendWavelet"],
            t_width=32,
            basis_dim=8,
            trend_thetas_dim=3,
            n_pools_kernel_size=[1],
            n_freq_downsample=[1],
        )
        block = model.stacks[0][0]
        assert block.trend_dim == 3
        assert block.backcast_linear.out_features == 3 + min(8, 24)
        assert block.forecast_linear.out_features == 3 + min(8, 6)

    def test_active_g_list_broadcasts_per_stack(self):
        model = _make_nhits(
            ["Generic", "Generic", "Generic"],
            active_g=[True, True, False],
            n_pools_kernel_size=[1, 1, 1],
            n_freq_downsample=[1, 1, 1],
        )
        assert model.stacks[0][0].active_g is True
        assert model.stacks[1][0].active_g is True
        assert model.stacks[2][0].active_g is False
        assert model.active_g == [True, True, False]
        assert model._active_g_per_stack == [True, True, False]

    def test_active_g_list_mixed_modes(self):
        model = _make_nhits(
            ["Generic", "Generic", "Generic"],
            active_g=[True, "forecast", False],
            n_pools_kernel_size=[1, 1, 1],
            n_freq_downsample=[1, 1, 1],
        )
        assert model.stacks[0][0].active_g is True
        assert model.stacks[1][0].active_g == "forecast"
        assert model.stacks[2][0].active_g is False

    def test_active_g_list_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="active_g list length"):
            _make_nhits(
                ["Generic", "Generic"],
                active_g=[True, False, True],
                n_pools_kernel_size=[1, 1],
                n_freq_downsample=[1, 1],
            )

    def test_active_g_list_invalid_entry_raises(self):
        with pytest.raises(ValueError, match="active_g must be"):
            _make_nhits(
                ["Generic", "Generic"],
                active_g=[True, "invalid"],
                n_pools_kernel_size=[1, 1],
                n_freq_downsample=[1, 1],
            )

    def test_active_g_list_forward_pass(self):
        model = _make_nhits(
            ["Generic", "Generic", "Generic"],
            active_g=[True, True, False],
            n_pools_kernel_size=[1, 1, 1],
            n_freq_downsample=[1, 1, 1],
        )
        x = torch.randn(4, 24)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 24)
        assert forecast.shape == (4, 6)


# --- Asymmetric wavelet forwarding tests ---


class TestAsymmetricWaveletForwarding:
    """Verify NBeatsNet and NHiTSNet forward backcast/forecast_wavelet_type to blocks."""

    def test_nbeats_forwards_asymmetric_wavelet_to_waveletv3(self):
        """WaveletV3 blocks should receive asymmetric wavelet params from NBeatsNet."""
        model = _make_model(
            ["WaveletV3"] * 2,
            g_width=64,
            backcast_wavelet_type="sym20",
            forecast_wavelet_type="coif2",
        )
        block = model.stacks[0][0]
        assert block.backcast_wavelet_type == "sym20"
        assert block.forecast_wavelet_type == "coif2"

    def test_nbeats_forwards_asymmetric_wavelet_to_waveletv3aelg(self):
        """WaveletV3AELG blocks should receive asymmetric wavelet params from NBeatsNet."""
        model = _make_model(
            ["WaveletV3AELG"] * 2,
            g_width=64,
            backcast_wavelet_type="sym20",
            forecast_wavelet_type="db3",
        )
        block = model.stacks[0][0]
        assert block.backcast_wavelet_type == "sym20"
        assert block.forecast_wavelet_type == "db3"

    def test_nbeats_forwards_asymmetric_wavelet_to_trendwaveletaelg(self):
        """TrendWaveletAELG blocks should receive asymmetric wavelet params."""
        model = _make_model(
            ["TrendWaveletAELG"],
            g_width=64,
            backcast_wavelet_type="sym20",
            forecast_wavelet_type="coif2",
        )
        block = model.stacks[0][0]
        assert block.backcast_wavelet_type == "sym20"
        assert block.forecast_wavelet_type == "coif2"

    def test_nbeats_forwards_asymmetric_wavelet_to_trendwavelet(self):
        """TrendWavelet blocks should receive asymmetric wavelet params."""
        model = _make_model(
            ["TrendWavelet"],
            t_width=64,
            backcast_wavelet_type="sym20",
            forecast_wavelet_type="coif2",
        )
        block = model.stacks[0][0]
        assert block.backcast_wavelet_type == "sym20"
        assert block.forecast_wavelet_type == "coif2"

    def test_nbeats_asymmetric_forward_pass(self):
        """Forward pass with asymmetric wavelet params produces correct shapes."""
        model = _make_model(
            ["TrendAELG", "WaveletV3AELG"] * 2,
            g_width=64,
            t_width=32,
            backcast_wavelet_type="sym20",
            forecast_wavelet_type="coif2",
        )
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)

    def test_nbeats_none_asymmetric_defaults_to_wavelet_type(self):
        """When asymmetric params are None, blocks fall back to wavelet_type."""
        model = _make_model(["WaveletV3AELG"] * 2, g_width=64, wavelet_type="haar")
        block = model.stacks[0][0]
        assert block.backcast_wavelet_type is None
        assert block.forecast_wavelet_type is None
        assert block.wavelet_type == "haar"

    def test_nhits_forwards_asymmetric_wavelet(self):
        """NHiTSNet should forward asymmetric wavelet params to wavelet blocks."""
        model = _make_nhits(
            ["WaveletV3AELG"],
            g_width=64,
            n_pools_kernel_size=[1],
            n_freq_downsample=[1],
            backcast_wavelet_type="sym20",
            forecast_wavelet_type="coif2",
        )
        block = model.stacks[0][0]
        assert block.backcast_wavelet_type == "sym20"
        assert block.forecast_wavelet_type == "coif2"

    def test_nhits_forwards_asymmetric_wavelet_to_trendwavelet(self):
        """NHiTSNet should forward asymmetric wavelet params to TrendWavelet."""
        model = _make_nhits(
            ["TrendWavelet"],
            t_width=64,
            n_pools_kernel_size=[1],
            n_freq_downsample=[1],
            backcast_wavelet_type="sym20",
            forecast_wavelet_type="coif2",
        )
        block = model.stacks[0][0]
        assert block.backcast_wavelet_type == "sym20"
        assert block.forecast_wavelet_type == "coif2"

    def test_nhits_asymmetric_forward_pass(self):
        """NHiTSNet forward pass with asymmetric wavelet params produces correct shapes."""
        model = _make_nhits(
            ["TrendAELG", "WaveletV3AELG"],
            g_width=64,
            t_width=32,
            n_pools_kernel_size=[2, 1],
            n_freq_downsample=[2, 1],
            backcast_wavelet_type="sym20",
            forecast_wavelet_type="coif2",
        )
        x = torch.randn(4, 24)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 24)
        assert forecast.shape == (4, 6)


class TestWaveletVAELatentDimRouting:
    """Verify generic WaveletV3 VAE variants receive the configured latent_dim."""

    @pytest.mark.parametrize("block_name", ["WaveletV3VAE"])
    def test_nbeats_wavelet_vae_variants_route_latent_dim(self, block_name):
        latent_dim = 7
        model = _make_model([block_name], g_width=64, latent_dim=latent_dim)
        block = model.stacks[0][0]

        assert block.fc2_mu.out_features == latent_dim

    @pytest.mark.parametrize("block_name", ["WaveletV3VAE"])
    def test_nhits_wavelet_vae_variants_route_latent_dim(self, block_name):
        latent_dim = 7
        model = _make_nhits(
            [block_name],
            g_width=64,
            latent_dim=latent_dim,
            n_pools_kernel_size=[1],
            n_freq_downsample=[1],
        )
        block = model.stacks[0][0]

        assert block.fc2_mu.out_features == latent_dim


class TestDeprecatedVAE2StackTypes:
    """Deprecated VAE2 stack names should fail with a targeted message."""

    @pytest.mark.parametrize(
        "block_name",
        [
            "GenericVAE2",
            "TrendVAE2",
            "VAE2",
            "WaveletV3VAE2",
            "DB4WaveletV3VAE2",
        ],
    )
    def test_nbeats_rejects_deprecated_stack_types(self, block_name):
        with pytest.raises(ValueError, match="deprecated and removed"):
            _make_model([block_name])

    @pytest.mark.parametrize(
        "block_name",
        [
            "GenericVAE2",
            "TrendVAE2",
            "VAE2",
            "WaveletV3VAE2",
            "DB4WaveletV3VAE2",
        ],
    )
    def test_nhits_rejects_deprecated_stack_types(self, block_name):
        with pytest.raises(ValueError, match="deprecated and removed"):
            _make_nhits(
                [block_name],
                n_pools_kernel_size=[1],
                n_freq_downsample=[1],
            )


# ---------------------------------------------------------------------------
# Skip connection tests
# ---------------------------------------------------------------------------


class TestSkipConnections:
    """Verify ResNet-style skip connection behavior in NBeatsNet."""

    def test_skip_distance_default_is_zero(self):
        model = _make_model(["Generic"] * 5, g_width=32)
        assert model.skip_distance == 0

    def test_skip_alpha_default(self):
        model = _make_model(["Generic"] * 5, g_width=32)
        assert model.skip_alpha == 0.1
        assert model._skip_alpha_learnable is False

    def test_skip_distance_stored(self):
        model = _make_model(["Generic"] * 6, g_width=32, skip_distance=3)
        assert model.skip_distance == 3

    def test_skip_alpha_stored(self):
        model = _make_model(
            ["Generic"] * 6, g_width=32, skip_distance=3, skip_alpha=0.1
        )
        assert model.skip_alpha == 0.1
        assert model._skip_alpha_learnable is False

    def test_learnable_alpha_creates_parameter(self):
        model = _make_model(
            ["Generic"] * 6, g_width=32, skip_distance=3, skip_alpha="learnable"
        )
        assert model._skip_alpha_learnable is True
        assert hasattr(model, "skip_alpha_param")
        assert isinstance(model.skip_alpha_param, torch.nn.Parameter)
        assert model.skip_alpha_param.requires_grad is True

    def test_learnable_alpha_initial_value(self):
        model = _make_model(
            ["Generic"] * 6, g_width=32, skip_distance=3, skip_alpha="learnable"
        )
        assert abs(model.skip_alpha_param.item() - 0.01) < 1e-6

    def test_skip_distance_negative_raises(self):
        with pytest.raises(
            ValueError, match="skip_distance must be a non-negative integer"
        ):
            _make_model(["Generic"], skip_distance=-1)

    def test_skip_alpha_invalid_string_raises(self):
        with pytest.raises(ValueError, match="skip_alpha must be"):
            _make_model(["Generic"], skip_distance=1, skip_alpha="invalid")

    def test_skip_alpha_none_raises(self):
        with pytest.raises(ValueError, match="skip_alpha must be"):
            _make_model(["Generic"], skip_distance=1, skip_alpha=None)

    def test_skip_alpha_bool_raises(self):
        with pytest.raises(ValueError, match="skip_alpha must be"):
            _make_model(["Generic"], skip_distance=1, skip_alpha=True)

    def test_learnable_alpha_without_skip_distance_no_param(self):
        """learnable alpha with skip_distance=0 should not create nn.Parameter."""
        model = _make_model(
            ["Generic"] * 4, g_width=32, skip_distance=0, skip_alpha="learnable"
        )
        assert model._skip_alpha_learnable is False
        assert not hasattr(model, "skip_alpha_param")

    def test_forward_pass_skip_disabled(self):
        model = _make_model(["Generic"] * 5, g_width=32, skip_distance=0)
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)

    def test_forward_pass_skip_fixed_alpha(self):
        model = _make_model(
            ["Generic"] * 6, g_width=32, skip_distance=2, skip_alpha=0.1
        )
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)

    def test_forward_pass_skip_learnable_alpha(self):
        model = _make_model(
            ["Generic"] * 6, g_width=32, skip_distance=2, skip_alpha="learnable"
        )
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)

    def test_skip_modifies_residual(self):
        stack_types = ["Generic"] * 6
        model_no_skip = _make_model(stack_types, g_width=32, skip_distance=0)
        model_skip = _make_model(
            stack_types, g_width=32, skip_distance=2, skip_alpha=0.5
        )
        model_skip.load_state_dict(model_no_skip.state_dict(), strict=False)
        x = torch.randn(4, 20)
        res_no_skip, _ = model_no_skip(x)
        res_skip, _ = model_skip(x)
        assert not torch.allclose(res_no_skip, res_skip, atol=1e-6)

    def test_skip_distance_larger_than_stacks(self):
        model = _make_model(
            ["Generic"] * 3, g_width=32, skip_distance=10, skip_alpha=0.5
        )
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)

    def test_skip_distance_one(self):
        model = _make_model(
            ["Generic"] * 4, g_width=32, skip_distance=1, skip_alpha=0.1
        )
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)

    def test_skip_with_mixed_block_types(self):
        model = _make_model(
            ["Trend", "Seasonality", "Generic", "GenericAE"],
            g_width=32,
            t_width=32,
            s_width=64,
            skip_distance=2,
            skip_alpha=0.1,
        )
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)

    def test_skip_alpha_zero_is_noop(self):
        model = _make_model(
            ["Generic"] * 4, g_width=32, skip_distance=2, skip_alpha=0.0
        )
        x = torch.randn(4, 20)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 20)
        assert forecast.shape == (4, 5)


class TestNHiTSSkipConnections:
    """Verify skip connections work in NHiTSNet."""

    def test_nhits_skip_disabled_by_default(self):
        model = _make_nhits(["Generic", "Generic"])
        assert model.skip_distance == 0

    def test_nhits_skip_forward_shape(self):
        model = _make_nhits(
            ["Generic"] * 4,
            g_width=32,
            n_pools_kernel_size=[1, 2, 1, 2],
            n_freq_downsample=[1, 1, 2, 2],
            skip_distance=2,
            skip_alpha=0.1,
        )
        x = torch.randn(4, 24)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 24)
        assert forecast.shape == (4, 6)

    def test_nhits_skip_learnable_alpha(self):
        model = _make_nhits(
            ["Generic"] * 4, g_width=32, skip_distance=2, skip_alpha="learnable"
        )
        assert model._skip_alpha_learnable is True
        assert isinstance(model.skip_alpha_param, torch.nn.Parameter)
        x = torch.randn(4, 24)
        backcast, forecast = model(x)
        assert backcast.shape == (4, 24)
        assert forecast.shape == (4, 6)

    def test_nhits_skip_distance_negative_raises(self):
        with pytest.raises(
            ValueError, match="skip_distance must be a non-negative integer"
        ):
            _make_nhits(["Generic"], skip_distance=-1)

    def test_nhits_skip_alpha_invalid_string_raises(self):
        with pytest.raises(ValueError, match="skip_alpha must be"):
            _make_nhits(["Generic"], skip_distance=1, skip_alpha="bad")
