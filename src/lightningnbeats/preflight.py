"""Pre-flight diagnostics: choose a configuration before training anything.

The expensive parts of configuring a basis expansion forecaster are decidable
from the training data alone. Two quantities do the work, both computed from a
sample of target windows with no gradient steps:

**Captured energy** is the fraction of target energy lying inside the span of
the rows a frozen basis retains. It governs *accuracy*: across bases at
identical parameter counts it reproduces the trained ordering without inversion
on the datasets tested, including a perfect rank correlation on Traffic-96.

**Effective rank** is the participation ratio of the target's coefficient
spectrum, `1 / sum_k e_k^2`, meaning roughly how many coefficients actually
carry the signal. It governs *trainability*: a plain linear synthesis head
cannot train when the target effectively occupies one coordinate, and on
M4-Hourly it tracks the measured convergence rate at rho = +0.949 where captured
energy, flat to the fourth decimal, explains nothing.

Neither requires a GPU. `recommend()` turns them into a configuration.

    from lightningnbeats.preflight import recommend
    rec = recommend(train_windows, forecast_length=48)
    print(rec.report())

EVIDENCE AND ITS LIMITS. The rules below are graded. Those marked `strong` rest
on controlled sweeps of hundreds to thousands of runs across all six M4 periods.
Those marked `moderate` rest on one or two datasets. Those marked `provisional`
are mechanistic predictions with limited direct measurement, and are flagged
rather than hidden. Nothing here is tuned per dataset; if a rule needs tuning it
is not a default and does not belong in this file.
"""

from dataclasses import dataclass, field

import numpy as np

# Empirical thresholds. These come from the measured separation between datasets
# rather than from tuning: M4-Hourly targets sit at effective rank 1.05 to 1.54
# for energy concentrating bases and fail on every seed under a linear head,
# while Traffic-96 sits at 2.54 to 3.67 and does not.
_EFFECTIVE_RANK_FRAGILE = 4.0
_EFFECTIVE_RANK_COMFORTABLE = 10.0
_CAPTURE_TARGET = 0.999


@dataclass
class Recommendation:
    """A configuration proposal with the measurements that produced it."""

    forecast_length: int
    effective_rank: float
    top1_share: float
    dc_share: float
    capture_at_rank: dict = field(default_factory=dict)
    compressibility: float = 1.0
    recommended_forecast_rank: int = 0
    recommended_trunk_width: int = 128
    needs_rectifier: bool = False
    needs_trend_branch: bool = False
    recommended_basis: str = "dct"
    notes: list = field(default_factory=list)

    def report(self):
        lines = [
            "N-BEATS pre-flight report",
            "=" * 58,
            f"  horizon                    {self.forecast_length}",
            f"  effective rank of target   {self.effective_rank:.2f}",
            f"  largest coefficient share  {self.top1_share:.4f}",
            f"  window mean share          {self.dc_share:.4f}",
            f"  compressibility            {self.compressibility:.3f}"
            f"   (fraction of horizon needed)",
            "",
            "Recommended configuration",
            "-" * 58,
            f"  basis                      {self.recommended_basis}",
            f"  forecast basis rank        {self.recommended_forecast_rank}",
            f"  trunk width                {self.recommended_trunk_width}",
            f"  stacks                     10",
            f"  rectified forecast head    {self.needs_rectifier}",
            f"  paired trend branch        {self.needs_trend_branch}",
        ]
        if self.notes:
            lines += ["", "Notes", "-" * 58]
            lines += [f"  - {n}" for n in self.notes]
        return "\n".join(lines)


def _coefficient_spectrum(windows, basis=None):
    """Per coefficient RMS of the target ensemble under an orthonormal basis.

    With no basis supplied the Karhunen-Loeve basis of the windows is used,
    which is the fixed subspace maximising captured energy and therefore the
    most favourable case; if the spectrum is degenerate there, it is degenerate
    everywhere.
    """
    windows = np.asarray(windows, dtype=np.float64)
    if basis is None:
        second_moment = (windows.T @ windows) / windows.shape[0]
        evals, evecs = np.linalg.eigh(second_moment)
        basis = evecs[:, np.argsort(evals)[::-1]].T
    theta = windows @ basis.T
    return np.sqrt(np.mean(theta**2, axis=0)), basis


def _detrend(windows, degree=2):
    """Remove the subspace a trend branch spans, span{1, t, ..., t^degree}."""
    horizon = windows.shape[1]
    t = np.linspace(0.0, 1.0, horizon)
    vander = np.vstack([t**k for k in range(degree + 1)]).T
    return windows - windows @ (vander @ np.linalg.pinv(vander)).T


def _capture_curve(windows):
    rms, _ = _coefficient_spectrum(windows)
    energy = rms**2 / max(np.sum(rms**2), np.finfo(float).tiny)
    horizon = windows.shape[1]
    cumulative = np.cumsum(energy)
    ranks = sorted({1, 2, 4, 8, 16, 32, horizon // 4, horizon // 2, horizon})
    capture = {
        int(r): float(cumulative[min(r, horizon) - 1]) for r in ranks if 1 <= r <= horizon
    }
    return energy, capture


def analyze(windows):
    """Geometry of the target ensemble. No training, no GPU, no basis choice.

    Reported twice: on the raw targets, and on the targets after removing the
    low order polynomial a trend branch spans. The detrended view is the one the
    capacity rule uses, because a large constant offset otherwise dominates the
    spectrum and makes any series look compressible. A window mean of 100 on
    unit variance noise puts 99.99 percent of the energy in one direction, so
    the raw statistic would recommend a narrow trunk for white noise.
    """
    windows = np.asarray(windows, dtype=np.float64)
    horizon = windows.shape[1]

    energy, capture = _capture_curve(windows)
    residual = _detrend(windows)
    energy_d, capture_d = _capture_curve(residual)

    means = windows.mean(axis=1, keepdims=True)
    dc_share = float(
        np.sum(means**2) * horizon / max(np.sum(windows**2), np.finfo(float).tiny)
    )
    trend_share = 1.0 - float(
        np.sum(residual**2) / max(np.sum(windows**2), np.finfo(float).tiny)
    )

    return {
        "effective_rank": float(1.0 / np.sum(energy**2)),
        "effective_rank_detrended": float(1.0 / np.sum(energy_d**2)),
        "top1_share": float(energy.max()),
        "dc_share": dc_share,
        "trend_share": trend_share,
        "capture": capture,
        "capture_detrended": capture_d,
        "horizon": horizon,
    }


def recommend(windows, forecast_length=None, param_budget=None):
    """Propose a configuration from the target geometry alone.

    Args:
        windows: array of shape (n_windows, forecast_length) sampled from the
            training split. A few thousand windows is ample; the second moment
            converges long before the full population is used.
        forecast_length: inferred from `windows` when omitted.
        param_budget: optional hint. Only widens the trunk beyond the default,
            never narrows accuracy claims.
    """
    stats = analyze(windows)
    horizon = forecast_length or stats["horizon"]
    rec = Recommendation(
        forecast_length=horizon,
        effective_rank=stats["effective_rank"],
        top1_share=stats["top1_share"],
        dc_share=stats["dc_share"],
        capture_at_rank=stats["capture"],
    )

    # RANK. Basis identity is provably inert once the forecast basis is square,
    # since a freely learned projection absorbs any rotation, so the default is
    # complete rank and the choice of matrix is left free. Truncating below the
    # rank that captures `_CAPTURE_TARGET` costs accuracy predictably.
    rec.recommended_forecast_rank = horizon
    sufficient = [r for r, c in stats["capture"].items() if c >= _CAPTURE_TARGET]
    if sufficient:
        rec.notes.append(
            f"rank {min(sufficient)} already captures {_CAPTURE_TARGET:.3f} of "
            f"target energy; truncating there is safe if parameters are tight "
            f"[moderate]"
        )

    # TRAINABILITY. A concentrated spectrum breaks a plain linear head. Either
    # remedy works and neither adds expressive power; the trend branch is
    # preferred because rectifying both paths costs accuracy on converged runs.
    if stats["effective_rank"] < _EFFECTIVE_RANK_FRAGILE:
        rec.needs_trend_branch = True
        rec.notes.append(
            f"effective rank {stats['effective_rank']:.2f} is low, so a plain "
            f"linear head may fail to train; pair with a trend branch [strong]"
        )
    else:
        rec.notes.append(
            f"effective rank {stats['effective_rank']:.2f} is comfortable, so "
            f"leave the trend branch OFF. It is not free: measured out of sample "
            f"it costs 0.3 to 0.6 sMAPE where the target is not concentrated "
            f"(p = 0.021 and 0.0073), and buys only tail protection at very "
            f"short horizons [strong]"
        )
    if stats["dc_share"] > 0.5:
        rec.needs_trend_branch = True
        rec.notes.append(
            f"{stats['dc_share']:.2f} of target energy is in the window mean; "
            f"normalising the data removes this at source [provisional]"
        )

    # OUTPUT SIGN. Rectifying forbids a negative forecast, which is the observed
    # failure mode on strictly positive series. Only safe when the data is
    # strictly positive, so it is proposed rather than assumed.
    if np.asarray(windows).min() >= 0:
        rec.needs_rectifier = True
        rec.notes.append(
            "targets are non negative, so a rectified forecast head removes the "
            "collapse mode at no representational cost [strong]"
        )
        rec.notes.append(
            "CONDITION: rectify only if each BLOCK's output is itself a partial "
            "reconstruction of a non negative target. Never rectify an "
            "architecture whose blocks emit signed components that cancel, such "
            "as one fed sub-band decompositions: measured on M4-Daily, five of "
            "six DWT bands are about half negative by construction, and "
            "rectifying such a model degrades it catastrophically [strong]"
        )
    else:
        rec.notes.append(
            "targets take negative values, so do NOT rectify the forecast head; "
            "use the trend branch for stability instead [strong]"
        )

    # CAPACITY. Trunk width was inert across a 48.8 fold range on M4. Whether
    # that generalises depends on how compressible the target is, and effective
    # rank alone does not separate the datasets tested: M4-Hourly sits at 1.05
    # and Traffic-96 at 2.52, yet they differ enormously in how much of the
    # horizon must be retained. The discriminating statistic is the fraction of
    # the horizon needed to reach the capture target: 4 of 48 on M4-Hourly, or
    # 8 percent, against all 96 on Traffic-96.
    # Keyed to the DETRENDED spectrum: a trend branch supplies the low order
    # component anyway, so what decides capacity is how compressible the
    # remainder is. Using the raw spectrum here would call white noise with a
    # large offset compressible.
    sufficient_d = [
        r for r, c in stats["capture_detrended"].items() if c >= _CAPTURE_TARGET
    ]
    rank_needed = min(sufficient_d) if sufficient_d else horizon
    rec.compressibility = rank_needed / horizon

    # CAPACITY. Reported as a diagnostic, NOT used to pick the width.
    #
    # I tried to key the trunk width to compressibility and the rule is refuted
    # by my own measurements. Trunk width is inert on M4-Yearly and M4-Monthly
    # across a 48.8 fold range, Kruskal-Wallis p = 0.776 and 0.773, so width 128
    # is correct there. Detrended compressibility for those two periods is 0.500
    # and 0.889, which any threshold separating them from Traffic-96 at 1.000
    # would misclassify as needing a wide trunk. The statistic separates rank
    # deficient from full rank targets cleanly on synthetic data at any offset,
    # but it does not predict trunk sensitivity, and shipping it as though it
    # did would contradict the evidence.
    #
    # So the default is the measured one and the caveat is explicit. Anyone with
    # a held out split should simply try both; it is one integer and two runs.
    rec.recommended_trunk_width = 128
    rec.notes.append(
        "trunk width 128 is the measured default: on M4 it matched 2048 across "
        "a 48.8 fold parameter range, Kruskal-Wallis p = 0.776 [strong on M4]"
    )
    rec.notes.append(
        f"compressibility {rec.compressibility:.3f} ({rank_needed} of {horizon} "
        f"directions needed) is reported as a diagnostic only. I have NO "
        f"validated predictor of when a narrow trunk is unsafe; this statistic "
        f"was tested for that role and refuted against M4. If accuracy matters, "
        f"try 128 and 512 on a held out split [unvalidated]"
    )
    if param_budget:
        rec.notes.append(f"parameter budget {param_budget:,} supplied by caller")

    # BASIS. At complete rank identity is provably inert, so this is decided on
    # engineering grounds rather than accuracy. The deciding constraint is that
    # a wavelet needs enough samples to decompose: at H = 4, 6 or 8 every family
    # except haar returns level 0 and degenerates to an approximation only
    # basis, silently. Short horizons are exactly where business data lives.
    rec.recommended_basis = "dct"
    usable = _usable_wavelets(horizon)
    if usable:
        rec.notes.append(
            f"basis: DCT. At complete rank identity cannot change what is "
            f"representable, so this is an engineering choice: DCT is defined "
            f"at every horizon, deterministic, and needs no fitting. Wavelet "
            f"families usable at H={horizon}: {', '.join(usable)} [strong]"
        )
    else:
        rec.notes.append(
            f"basis: DCT, and do NOT substitute a wavelet here. At H={horizon} "
            f"no wavelet family admits a decomposition, so every one degenerates "
            f"to an approximation only basis without raising an error [strong]"
        )

    rec.notes.append(
        "cap depth at 10 stacks; 30 stacks bought nothing on any M4 period and "
        "collapsed the Generic family on Quarterly and Daily [strong]"
    )
    return rec


def _usable_wavelets(horizon, families=("haar", "db2", "db3", "coif2", "sym10")):
    """Families that can actually decompose a target of this length."""
    try:
        import pywt
    except ImportError:  # pragma: no cover - pywt is a hard dependency in practice
        return []
    return [
        f
        for f in families
        if pywt.dwt_max_level(horizon, pywt.Wavelet(f).dec_len) >= 1
    ]
