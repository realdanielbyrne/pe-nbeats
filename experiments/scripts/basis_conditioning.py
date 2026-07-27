"""Coefficient conditioning of frozen synthesis bases, and what a Trend block does to it.

WHY THIS EXISTS. At complete forecast rank every orthonormal basis represents the
target exactly, so expressiveness cannot explain why DCT and KL fail under a plain
linear synthesis head while SVD-ordered wavelets and random draws do not. The
proposed variable is how the target's energy is DISTRIBUTED across coefficients:
a single linear layer emits all H coefficients from one weight matrix under one
learning rate, so if one coefficient must be enormous and the rest small, no single
step size serves both.

WHAT IS AND IS NOT BEING MEASURED. This is NOT the matrix condition number. Every
basis here is orthonormal, so kappa(B) = 1 for all of them. What is measured is the
spread of the TARGET COEFFICIENT distribution the head must learn to produce.

STATISTICS REPORTED, and their robustness. Let rms_k be the root mean square of
coefficient k over the window ensemble, and e_k its share of coefficient energy.

    top1      max_k e_k                     robust, ordering independent
    eff_rank  1 / sum_k e_k^2               robust, participation ratio
    max/min   max rms / min rms             extreme dependent
    p95/p5    percentile ratio              NOT monotone with the outcome
    max/med   max rms / median rms          NOT monotone with the outcome

Across the five bases whose convergence is measured, top1, eff_rank and max/min all
order correctly; p95/p5 and max/med do not. The percentile measures fail precisely
because they discard the extreme coefficient, which is the coordinate that causes
the failure. Prefer eff_rank when reporting: it needs no extremes and states the
failure condition plainly, namely that a linear head cannot train when the target
lives in one coordinate out of H.

THE TREND ARM. A Trend block spans a low order Vandermonde subspace including a
constant, so in a doubly residual cascade it absorbs that component before the
frozen basis block sees it. The basis is FROZEN at build time, so what changes is
the data, not the basis; the "trend removed" rows therefore apply the same raw-built
basis to the residual after orthogonally projecting out span{1, t, ..., t^(p-1)}.

Usage:
    python experiments/scripts/basis_conditioning.py --dataset m4 --period Hourly
"""

import argparse
import os
import sys

import numpy as np
import pywt
from scipy.fft import dct as scipy_dct

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from build_kl_basis import collect_windows  # noqa: E402
from run_unified_benchmark import load_dataset  # noqa: E402

# Measured convergence at complete rank under a plain linear head, M4-Hourly.
# Sources: m4_kl_complete_results.csv, m4_conditioning_ag0_results.csv and the
# complete-rank arms of the frozen-basis theory tests.
MEASURED_CONVERGENCE = {
    "KL": "0/10",
    "DCT": "0/10",
    "band-ordered db3": "9/10",
    "SVD db3": "10/10",
    "random ortho": "10/10",
}


def _wavelet_frame(m, family):
    """Rows of the synthesis frame, grouped by DWT level (dyadic order)."""
    level = pywt.dwt_max_level(m, pywt.Wavelet(family).dec_len)
    structure = pywt.wavedec(np.zeros(m), family, level=level, mode="periodization")
    rows = []
    for i in range(len(structure)):
        for j in range(len(structure[i])):
            coeffs = [np.zeros_like(s) for s in structure]
            coeffs[i][j] = 1.0
            rows.append(pywt.waverec(coeffs, family, mode="periodization")[:m])
    return np.array(rows)


def ordered_wavelet(m, family):
    """QR of the frame, which PRESERVES the dyadic level ordering."""
    raw = _wavelet_frame(m, family)
    q, r = np.linalg.qr(raw.T)
    diag = np.abs(np.diag(r))
    keep = diag > 1e-10 * max(diag[0], np.finfo(np.float64).tiny)
    return q.T[keep]


def svd_wavelet(m, family):
    """The SHIPPED construction: SVD, ordered by singular value, which DESTROYS
    the dyadic ordering so a truncation keeps an arbitrary subspace."""
    raw = _wavelet_frame(m, family)
    _, s, vt = np.linalg.svd(raw, full_matrices=False)
    return vt[: int(np.sum(s > 1e-10 * s[0]))]


def dct_basis(m):
    return scipy_dct(np.eye(m), axis=0, norm="ortho").T


def kl_basis(windows):
    """Top eigenvectors of the UNCENTERED second moment, descending by energy."""
    s = (windows.T @ windows) / windows.shape[0]
    evals, evecs = np.linalg.eigh(s)
    return evecs[:, np.argsort(evals)[::-1]].T


def random_ortho(m, seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((m, m)))
    return q.T


def stats(basis, windows):
    theta = windows @ basis.T
    rms = np.sqrt(np.mean(theta**2, axis=0))
    energy = rms**2 / np.sum(rms**2)
    return {
        "top1": energy.max(),
        "eff_rank": 1.0 / np.sum(energy**2),
        "max_over_min": rms.max() / max(rms.min(), 1e-30),
        "p95_over_p5": np.percentile(rms, 95) / max(np.percentile(rms, 5), 1e-30),
        "max_over_med": rms.max() / np.median(rms),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="m4")
    ap.add_argument("--period", default="Hourly")
    ap.add_argument("--forecast-multiplier", type=int, default=5)
    ap.add_argument("--trend-thetas-dim", type=int, default=3)
    ap.add_argument("--max-windows", type=int, default=1_000_000)
    ap.add_argument("--normalize", action="store_true")
    args = ap.parse_args()

    loaded = load_dataset(args.dataset, args.period)
    dataset = loaded[0] if isinstance(loaded, tuple) else loaded
    h = dataset.forecast_length
    L = args.forecast_multiplier * h

    _, ys, total = collect_windows(dataset, L, h, args.normalize, args.max_windows)
    print(f"{args.dataset}/{args.period}: H={h} L={L}, {ys.shape[0]} of {total} windows\n")

    # The subspace a Trend block spans, and the residual left after it.
    t = np.linspace(0, 1, h)
    vand = np.vstack([t**k for k in range(args.trend_thetas_dim)]).T
    proj = vand @ np.linalg.pinv(vand)
    resid = ys - ys @ proj.T
    captured = 1 - np.mean(np.sum(resid**2, 1)) / np.mean(np.sum(ys**2, 1))
    print(
        f"Trend subspace (degree {args.trend_thetas_dim - 1}) captures "
        f"{captured:.6f} of forecast energy\n"
    )

    bases = {
        "KL": kl_basis(ys),
        "DCT": dct_basis(h),
        "band-ordered db3": ordered_wavelet(h, "db3"),
        "SVD db3": svd_wavelet(h, "db3"),
        "random ortho": random_ortho(h),
    }

    hdr = (
        f"{'basis':20s} {'conv':>6s} {'input':16s} {'top1':>7s} "
        f"{'eff.rank':>9s} {'max/min':>10s} {'p95/p5':>9s} {'max/med':>9s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name, basis in bases.items():
        for label, data in (("raw", ys), ("trend removed", resid)):
            s = stats(basis, data)
            conv = MEASURED_CONVERGENCE.get(name, "") if label == "raw" else ""
            print(
                f"{name:20s} {conv:>6s} {label:16s} {s['top1']:7.4f} "
                f"{s['eff_rank']:9.2f} {s['max_over_min']:10.1f} "
                f"{s['p95_over_p5']:9.1f} {s['max_over_med']:9.1f}"
            )

    # Rank correlation against the measured convergence RATE. Spearman is used
    # rather than a strict monotonicity walk because SVD db3 and random ortho
    # both converge 10/10, so their relative order carries no information and a
    # strict walk would penalise an arbitrary tiebreak.
    from scipy.stats import spearmanr

    order = [n for n in bases if n in MEASURED_CONVERGENCE]
    rate = [eval(MEASURED_CONVERGENCE[n].replace("/", "/")) for n in order]
    print("\nSpearman against measured convergence rate "
          f"({', '.join(f'{n}={r:.1f}' for n, r in zip(order, rate))}):")
    for key, better in (("top1", "lower"), ("eff_rank", "higher"),
                        ("max_over_min", "lower"), ("p95_over_p5", "lower"),
                        ("max_over_med", "lower")):
        vals = [stats(bases[n], ys)[key] for n in order]
        rho, p = spearmanr(vals, rate)
        want = -1.0 if better == "lower" else 1.0
        flag = "OK " if abs(rho - want) < 1e-9 else "   "
        print(f"  {flag}{key:14s} rho={rho:+.3f} (perfect={want:+.0f}, {better} is better)")


if __name__ == "__main__":
    main()
