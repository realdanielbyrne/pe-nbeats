"""Tests for run_nhits_benchmark.py YAML config loading and parse_stack_types."""

import os
import sys
import tempfile
import warnings

import pytest
import yaml

_EXPERIMENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments"))
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_DIR)

import run_nhits_benchmark as rnb  # noqa: E402


# ---------------------------------------------------------------------------
# parse_stack_types
# ---------------------------------------------------------------------------

class TestParseStackTypes:
    """Tests for the parse_stack_types helper."""

    def test_list_passthrough(self):
        """A plain list should be returned as-is."""
        result = rnb.parse_stack_types(["Generic", "Generic", "Generic"])
        assert result == ["Generic", "Generic", "Generic"]

    def test_string_expression_single(self):
        """'[\"Generic\"] * 10' should produce 10 Generic entries."""
        result = rnb.parse_stack_types("['Generic'] * 10")
        assert result == ["Generic"] * 10

    def test_string_expression_multi(self):
        """'[\"TrendAELG\", \"Sym20V3AELG\"] * 5' should alternate 5 times."""
        result = rnb.parse_stack_types("['TrendAELG', 'Sym20V3AELG'] * 5")
        assert len(result) == 10
        assert result == ["TrendAELG", "Sym20V3AELG"] * 5

    def test_plain_list_string(self):
        """'[\"A\", \"B\", \"C\"]' should parse as a list."""
        result = rnb.parse_stack_types("['A', 'B', 'C']")
        assert result == ["A", "B", "C"]

    def test_invalid_expression_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            rnb.parse_stack_types("not a valid expression")

    def test_non_string_non_list_raises(self):
        with pytest.raises(ValueError, match="must be a list or string"):
            rnb.parse_stack_types(42)


# ---------------------------------------------------------------------------
# load_yaml_configs
# ---------------------------------------------------------------------------

class TestLoadYamlConfigs:
    """Tests for load_yaml_configs function.

    Uses a self-contained fixture YAML rather than the live experiment
    configs under experiments/configs/, which are edited routinely and
    previously made these tests fail whenever a sweep changed shape.
    """

    _WEATHER_LIKE_YAML = """
experiment_name: NHiTS-Protocol-Weather-Fixture
dataset: weather
horizons: [96, 192, 336, 720]

protocol:
  normalize: true
  train_ratio: 0.7
  val_ratio: 0.1
  include_target: true
  loss: MSELoss
  forecast_multiplier: 5
  batch_size: 256

training:
  max_epochs: 100
  patience: 10
  activation: ReLU
  active_g: false
  sum_losses: false

block_params:
  thetas_dim: 5
  latent_dim: 16
  basis_dim: eq_fcast

runs:
  n_runs: 8
  base_seed: 1

configs:
  - name: Generic-10
    stack_types: "['Generic'] * 10"
  - name: TrendAELG+Sym20V3AELG
    stack_types: "['TrendAELG', 'Symlet20WaveletV3AELG'] * 5"
  - name: NHiTS-Generic
    model: NHiTSNet
    stack_types: "['NHiTSGenericRoot'] * 3"
    n_pools_kernel_size: [8, 4, 1]
    n_freq_downsample: [24, 12, 1]
"""

    _TRAFFIC_LIKE_YAML = """
experiment_name: NHiTS-Protocol-Traffic-Fixture
dataset: traffic
horizons: [96]

protocol:
  normalize: false

configs:
  - name: Generic-10
    stack_types: "['Generic'] * 10"
"""

    @pytest.fixture
    def weather_yaml_path(self, tmp_path):
        path = tmp_path / "nhits_benchmark_weather_fixture.yaml"
        path.write_text(self._WEATHER_LIKE_YAML)
        return str(path)

    @pytest.fixture
    def traffic_yaml_path(self, tmp_path):
        path = tmp_path / "nhits_benchmark_traffic_fixture.yaml"
        path.write_text(self._TRAFFIC_LIKE_YAML)
        return str(path)

    def test_loads_weather_yaml(self, weather_yaml_path):
        """Should load all configs and protocol from weather YAML."""
        result = rnb.load_yaml_configs(weather_yaml_path)
        assert "configs" in result
        assert "protocol" in result
        assert "training" in result
        assert "block_params" in result
        assert "runs" in result
        assert result["dataset"] == "weather"
        assert result["horizons"] == [96, 192, 336, 720]

    def test_weather_protocol_values(self, weather_yaml_path):
        """Protocol values should match the YAML file."""
        result = rnb.load_yaml_configs(weather_yaml_path)
        p = result["protocol"]
        assert p["train_ratio"] == 0.7
        assert p["val_ratio"] == 0.1
        assert p["loss"] == "MSELoss"
        assert p["forecast_multiplier"] == 5
        assert p["batch_size"] == 256
        assert p["normalize"] is True

    def test_weather_configs_count(self, weather_yaml_path):
        """Fixture YAML should have 3 configs."""
        result = rnb.load_yaml_configs(weather_yaml_path)
        assert len(result["configs"]) == 3

    def test_traffic_normalize_false(self, traffic_yaml_path):
        """Traffic YAML should have normalize=false."""
        result = rnb.load_yaml_configs(traffic_yaml_path)
        assert result["protocol"]["normalize"] is False

    def test_stack_types_parsed_correctly(self, weather_yaml_path):
        """Stack types expressions should be expanded to lists."""
        result = rnb.load_yaml_configs(weather_yaml_path)
        configs = result["configs"]

        # Generic-10 should have 10 Generic entries
        generic10 = next(c for c in configs if c["config_name"] == "Generic-10")
        assert generic10["stack_types"] == ["Generic"] * 10

        # TrendAELG+Sym20V3AELG should alternate 5 times = 10 entries
        trend_sym = next(c for c in configs if c["config_name"] == "TrendAELG+Sym20V3AELG")
        assert len(trend_sym["stack_types"]) == 10
        assert trend_sym["stack_types"][0] == "TrendAELG"
        assert trend_sym["stack_types"][1] == "Symlet20WaveletV3AELG"

    def test_nhits_configs_have_pooling_params(self, weather_yaml_path):
        """NHiTSNet configs should have n_pools_kernel_size and n_freq_downsample."""
        result = rnb.load_yaml_configs(weather_yaml_path)
        configs = result["configs"]

        nhits_generic = next(c for c in configs if c["config_name"] == "NHiTS-Generic")
        assert nhits_generic["model_type"] == "NHiTSNet"
        assert nhits_generic["n_pools_kernel_size"] == [8, 4, 1]
        assert nhits_generic["n_freq_downsample"] == [24, 12, 1]

    def test_training_params(self, weather_yaml_path):
        result = rnb.load_yaml_configs(weather_yaml_path)
        t = result["training"]
        assert t["max_epochs"] == 100
        assert t["patience"] == 10
        assert t["activation"] == "ReLU"
        assert t["active_g"] is False
        assert t["sum_losses"] is False

    def test_block_params(self, weather_yaml_path):
        result = rnb.load_yaml_configs(weather_yaml_path)
        bp = result["block_params"]
        assert bp["thetas_dim"] == 5
        assert bp["latent_dim"] == 16
        assert bp["basis_dim"] == "eq_fcast"

    def test_runs_params(self, weather_yaml_path):
        result = rnb.load_yaml_configs(weather_yaml_path)
        r = result["runs"]
        assert r["n_runs"] == 8
        assert r["base_seed"] == 1


# ---------------------------------------------------------------------------
# sampling_style='nhits_paper' — epochs-vs-steps guard
# ---------------------------------------------------------------------------


def _paper_protocol(
    *,
    sampling_style="nhits_paper",
    steps_per_epoch=5,
    sampling_weights="uniform",
    acknowledge_epoch_semantics=False,
):
    return {
        "train_ratio": 0.7,
        "val_ratio": 0.1,
        "loss": "MSELoss",
        "forecast_multiplier": 5,
        "batch_size": 32,
        "normalize": False,
        "normalization_style": None,
        "include_target": True,
        "sampling_style": sampling_style,
        "steps_per_epoch": steps_per_epoch,
        "sampling_weights": sampling_weights,
        "acknowledge_epoch_semantics": acknowledge_epoch_semantics,
    }


def _paper_training(*, max_steps=None):
    return {
        "activation": "ReLU",
        "active_g": False,
        "sum_losses": False,
        "learning_rate": 1e-3,
        "num_workers": 0,
        "max_steps": max_steps,
        "early_stopping": True,
        "val_check_interval": None,
    }


class TestPaperSamplingEpochsGuard:
    """The epochs-vs-steps guard in _run_experiment_body."""

    @pytest.fixture
    def stub_args(self, tmp_path):
        config = {
            "config_name": "DummyCfg",
            "model_type": "NBeatsNet",
            "stack_types": ["Generic"],
            "active_g": False,
        }
        return dict(
            config=config,
            dataset_name="weather",
            horizon=96,
            run_idx=0,
            csv_path=str(tmp_path / "results.csv"),
            max_epochs=10,
            patience=3,
            batch_size=32,
            worker_id="test",
            block_params={"thetas_dim": 5, "latent_dim": 4, "basis_dim": "eq_fcast"},
            runs={"base_seed": 1},
            skip_horizon_overrides=False,
        )

    def test_paper_without_max_steps_errors(self, stub_args):
        stub_args["protocol"] = _paper_protocol()
        stub_args["training"] = _paper_training(max_steps=None)
        with pytest.raises(ValueError, match="acknowledge_epoch_semantics"):
            rnb._run_experiment_body(**stub_args)

    def test_paper_with_acknowledge_epoch_semantics_warns(self, stub_args, monkeypatch):
        stub_args["protocol"] = _paper_protocol(acknowledge_epoch_semantics=True)
        stub_args["training"] = _paper_training(max_steps=None)

        # Stop execution right after the guard runs so we don't try to load
        # the real Weather dataset.
        def _blow_up(*a, **kw):
            raise RuntimeError("short-circuit after guard")

        monkeypatch.setattr(rnb, "WeatherDataset", _blow_up)

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with pytest.raises(RuntimeError, match="short-circuit after guard"):
                rnb._run_experiment_body(**stub_args)

        messages = [str(w.message) for w in captured if issubclass(w.category, UserWarning)]
        assert any("gradient steps" in m for m in messages), messages

    def test_paper_with_max_steps_silent(self, stub_args, monkeypatch):
        stub_args["protocol"] = _paper_protocol()
        stub_args["training"] = _paper_training(max_steps=50)

        def _blow_up(*a, **kw):
            raise RuntimeError("short-circuit after guard")

        monkeypatch.setattr(rnb, "WeatherDataset", _blow_up)

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with pytest.raises(RuntimeError, match="short-circuit after guard"):
                rnb._run_experiment_body(**stub_args)

        # No UserWarning about gradient-step budget should have been emitted.
        assert not any(
            "gradient steps" in str(w.message)
            for w in captured
            if issubclass(w.category, UserWarning)
        )

    def test_paper_rejects_invalid_steps_per_epoch(self, stub_args):
        stub_args["protocol"] = _paper_protocol(steps_per_epoch=0)
        stub_args["training"] = _paper_training(max_steps=50)
        with pytest.raises(ValueError, match="steps_per_epoch"):
            rnb._run_experiment_body(**stub_args)

    def test_paper_missing_steps_per_epoch_errors(self, stub_args):
        stub_args["protocol"] = _paper_protocol(steps_per_epoch=None)
        stub_args["training"] = _paper_training(max_steps=50)
        with pytest.raises(ValueError, match="steps_per_epoch"):
            rnb._run_experiment_body(**stub_args)

