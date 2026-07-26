"""Build Karhunen-Loeve (PCA) synthesis bases from the training split.

Linear approximation theory says that among all fixed d-dimensional subspaces,
the one maximising expected captured energy is spanned by the top d eigenvectors
of the ensemble second-moment matrix S = E[y y^T]. That basis is the theoretical
optimum every other frozen basis in this repository is competing against, so it
is the sharpest available test of the linear-approximation account: if the
account is right, the KL basis should beat DCT, random orthonormal and wavelet
arms at truncated rank.

The second moment is UNCENTERED on purpose. The quantity that bounds block error
is E[||B y||^2] / E[||y||^2], which is tr(B S B^T)/tr(S) with S uncentered; using
a centred covariance would optimise captured variance instead, which is not what
the synthesis head is bounded by.

Windows are drawn from the TRAINING split only, through the same datamodule the
benchmark uses, so the ensemble matches what the model actually sees (including
whatever normalisation the protocol applies).

Usage:
    python experiments/scripts/build_kl_basis.py --dataset m4 --period Hourly \
        --forecast-multiplier 5 --out experiments/bases/m4_hourly
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_unified_benchmark import load_dataset  # noqa: E402

from lightningnbeats.loaders import ColumnarCollectionTimeSeriesDataModule  # noqa: E402


def collect_windows(dataset, backcast_length, forecast_length, normalize, max_windows):
    """Return (backcast_windows, forecast_windows) as float64 arrays."""
    dm = ColumnarCollectionTimeSeriesDataModule(
        dataset.train_data,
        backcast_length=backcast_length,
        forecast_length=forecast_length,
        batch_size=1024,
        no_val=False,
        num_workers=0,
        normalize=normalize,
    )
    dm.setup()
    ds = dm.train_dataset
    n = len(ds)
    idx = np.arange(n)
    if n > max_windows:
        # Deterministic subsample. The second moment converges long before the
        # full population is used, and this keeps the script cheap on Traffic.
        idx = np.random.default_rng(0).choice(n, size=max_windows, replace=False)
        idx.sort()

    xs = np.empty((len(idx), backcast_length), dtype=np.float64)
    ys = np.empty((len(idx), forecast_length), dtype=np.float64)
    for row, i in enumerate(idx):
        x, y = ds[int(i)]
        xs[row] = np.asarray(x, dtype=np.float64)
        ys[row] = np.asarray(y, dtype=np.float64)
    return xs, ys, n


def kl_basis(windows):
    """Top eigenvectors of the uncentered second moment, rows descending by energy."""
    m = windows.shape[1]
    S = (windows.T @ windows) / windows.shape[0]
    evals, evecs = np.linalg.eigh(S)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    basis = evecs[:, order].T  # rows = eigenvectors, descending eigenvalue
    assert basis.shape == (m, m)
    # eigh returns orthonormal columns; verify the row form is orthonormal too.
    err = np.abs(basis @ basis.T - np.eye(m)).max()
    assert err < 1e-8, f"KL basis not orthonormal (max dev {err:.2e})"
    return basis, np.clip(evals, 0.0, None)


def captured(basis_rows, windows):
    """Fraction of ensemble energy captured by the row space of basis_rows."""
    proj = windows @ basis_rows.T
    return float((proj**2).sum() / (windows**2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--period", required=True)
    ap.add_argument("--forecast-multiplier", type=int, default=5)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--max-windows", type=int, default=200000)
    ap.add_argument("--train-ratio", type=float, default=None)
    ap.add_argument("--include-target", action="store_true", default=None)
    ap.add_argument("--out", required=True, help="prefix; writes {out}_{length}.npy")
    args = ap.parse_args()

    ds = load_dataset(
        args.dataset,
        args.period,
        train_ratio=args.train_ratio,
        include_target=args.include_target,
    )
    H = int(ds.forecast_length)
    L = args.forecast_multiplier * H

    xs, ys, n_total = collect_windows(ds, L, H, args.normalize, args.max_windows)
    print(f"{args.dataset}/{args.period}: H={H} L={L} "
          f"windows={xs.shape[0]} (of {n_total} available)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for name, windows, m in (("backcast", xs, L), ("forecast", ys, H)):
        basis, evals = kl_basis(windows)
        path = f"{args.out}_{m}.npy"
        np.save(path, basis)
        total = evals.sum()
        head = [1, max(1, m // 8), max(1, m // 3), max(1, m // 2), m]
        frac = ", ".join(
            f"d={d}: {evals[:d].sum()/total:.4f}" for d in sorted(set(head))
        )
        print(f"  {name:9s} -> {path}  ({m}x{m})  cumulative energy  {frac}")

    # Sanity check: KL capture must equal the eigenvalue partial sum.
    basis_f, evals_f = kl_basis(ys)
    d = max(1, H // 3)
    assert abs(captured(basis_f[:d], ys) - evals_f[:d].sum() / evals_f.sum()) < 1e-9
    print(f"  verified: KL capture at d=H/3 matches eigenvalue partial sum")


if __name__ == "__main__":
    main()
