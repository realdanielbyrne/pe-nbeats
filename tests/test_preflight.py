"""Tests for the pre-flight configuration diagnostics.

The behaviours pinned here are the ones that would silently mislead a
practitioner if they regressed: the offset confound, the sign rule, and the fact
that the trunk recommendation is NOT keyed to compressibility.
"""

import numpy as np
import pytest

from lightningnbeats.preflight import analyze, recommend


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _rank_k(rng, n, horizon, k):
    return rng.standard_normal((n, k)) @ rng.standard_normal((k, horizon))


class TestCompressibility:
    """Compressibility must reflect target structure, not the offset."""

    @pytest.mark.parametrize("offset", [0.0, 5.0, 100.0])
    def test_low_rank_target_is_compressible_at_any_offset(self, rng, offset):
        windows = _rank_k(rng, 2000, 48, 2) + offset
        assert recommend(windows).compressibility <= 0.25

    @pytest.mark.parametrize("offset", [0.0, 5.0, 100.0])
    def test_white_noise_is_incompressible_at_any_offset(self, rng, offset):
        # Regression test. Computed on the raw spectrum, a +100 offset put
        # 99.99 percent of the energy in one direction and white noise looked
        # as compressible as rank-2 data. Detrending is what fixes it.
        windows = rng.standard_normal((2000, 48)) + offset
        assert recommend(windows).compressibility > 0.5

    def test_offset_does_not_change_the_verdict(self, rng):
        base = _rank_k(rng, 2000, 48, 3)
        values = {recommend(base + o).compressibility for o in (0.0, 1.0, 50.0)}
        assert len(values) == 1, f"offset changed compressibility: {values}"


class TestEffectiveRank:
    def test_rank_one_target_has_effective_rank_near_one(self, rng):
        windows = _rank_k(rng, 2000, 32, 1)
        assert analyze(windows)["effective_rank"] == pytest.approx(1.0, abs=0.05)

    def test_white_noise_has_high_effective_rank(self, rng):
        windows = rng.standard_normal((4000, 32))
        assert analyze(windows)["effective_rank"] > 20

    def test_concentrated_target_asks_for_a_trend_branch(self, rng):
        windows = _rank_k(rng, 2000, 48, 1) + 100.0
        assert recommend(windows).needs_trend_branch


class TestSignRule:
    """Rectifying is only safe when the targets cannot go negative."""

    def test_non_negative_targets_get_a_rectifier(self, rng):
        windows = np.abs(rng.standard_normal((500, 24))) + 1.0
        assert recommend(windows).needs_rectifier

    def test_signed_targets_do_not_get_a_rectifier(self, rng):
        windows = rng.standard_normal((500, 24))
        rec = recommend(windows)
        assert not rec.needs_rectifier
        assert any("do NOT rectify" in n for n in rec.notes)


class TestTrunkRecommendation:
    """The trunk rule is the measured default, deliberately not a prediction."""

    def test_trunk_is_not_keyed_to_compressibility(self, rng):
        # Keying width to compressibility was refuted against M4, where Yearly
        # and Monthly sit at 0.500 and 0.889 detrended yet tolerate width 128.
        # Both of these must return the same width despite opposite structure.
        compressible = recommend(_rank_k(rng, 2000, 48, 2))
        diffuse = recommend(rng.standard_normal((2000, 48)))
        assert compressible.compressibility < diffuse.compressibility
        assert compressible.recommended_trunk_width == diffuse.recommended_trunk_width

    def test_unvalidated_caveat_is_present(self, rng):
        rec = recommend(rng.standard_normal((500, 24)))
        assert any("unvalidated" in n for n in rec.notes)


class TestRecommendationShape:
    def test_forecast_rank_defaults_to_complete(self, rng):
        rec = recommend(rng.standard_normal((500, 36)))
        assert rec.recommended_forecast_rank == 36

    def test_report_renders(self, rng):
        text = recommend(rng.standard_normal((500, 24))).report()
        assert "pre-flight report" in text
        assert "Recommended configuration" in text

    def test_explicit_forecast_length_overrides_inference(self, rng):
        rec = recommend(rng.standard_normal((500, 24)), forecast_length=24)
        assert rec.forecast_length == 24

    def test_every_note_carries_a_confidence_grade(self, rng):
        grades = ("[strong", "[moderate", "[provisional", "[unvalidated")
        for note in recommend(rng.standard_normal((500, 24))).notes:
            if note.startswith("parameter budget"):
                continue
            assert any(g in note for g in grades), f"ungraded note: {note}"
