"""Training-free captured-energy comparison across frozen bases.

Proposition 1 bounds a block's forecast error below by the ensemble energy lying
outside the row space of its frozen basis. That bound is computable with no
training at all, so the ordering of bases can be PREDICTED before any GPU time
is spent. Proposition 5 says the ordering should be

    KL >= DCT > random orthonormal > dyadic wavelet

because KL is the optimal fixed subspace by construction, DCT approaches it for
slowly decaying stationary covariance, a random subspace captures d/m in
expectation, and the wavelet's SVD row ordering is not aligned with where the
data's energy actually sits.

Reported on both the training ensemble (what the model fits) and the test
ensemble (what it is scored on), so the ordering can be checked for stability.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_unified_benchmark import load_dataset  # noqa: E402

from pe_nbeats.blocks.blocks import _WaveletGeneratorV3 as WG  # noqa: E402
from pe_nbeats.loaders import (  # noqa: E402
    ColumnarCollectionTimeSeriesDataModule,
)

CASES = [
    # (dataset, period, fmult, H, d, kl_prefix, wavelet, loader_kwargs)
    ("m4", "Hourly", 5, 48, 16, "experiments/bases/m4_hourly", "db3", {}),
    ("m4", "Monthly", 5, 18, 6, "experiments/bases/m4_monthly", "db3", {}),
    (
        "traffic",
        "Traffic-96",
        5,
        96,
        32,
        "experiments/bases/traffic96",
        "coif2",
        {"train_ratio": 0.7, "include_target": True},
    ),
]


def forecast_windows(dataset, L, H, max_windows=100000):
    dm = ColumnarCollectionTimeSeriesDataModule(
        dataset.train_data,
        backcast_length=L,
        forecast_length=H,
        batch_size=1024,
        no_val=False,
        num_workers=0,
        normalize=False,
    )
    dm.setup()
    ds = dm.train_dataset
    n = len(ds)
    idx = np.arange(n)
    if n > max_windows:
        idx = np.random.default_rng(0).choice(n, size=max_windows, replace=False)
        idx.sort()
    return np.stack([np.asarray(ds[int(i)][1], dtype=np.float64) for i in idx])


def test_windows(dataset, H):
    """Test targets as (n, H).

    M4 ships one H-length horizon per series, so the transposed test frame is
    already the target ensemble. Traffic and Weather ship a long held-out span
    per sensor instead, so the horizons are cut from it as non-overlapping
    windows.
    """
    Y = np.asarray(dataset.test_data.values, dtype=np.float64).T
    if Y.shape[1] == H:
        return Y
    n_win = Y.shape[1] // H
    if n_win < 1:
        raise ValueError(f"test span {Y.shape[1]} shorter than horizon {H}")
    return Y[:, : n_win * H].reshape(Y.shape[0] * n_win, H)


def captured(basis, W):
    proj = W @ basis.numpy().T if hasattr(basis, "numpy") else W @ basis.T
    return float((proj**2).sum() / (W**2).sum())


def main():
    print(f"{'case':<18}{'basis':<16}{'d/m':>7}{'train':>9}{'test':>9}")
    print("-" * 59)
    rows = []
    for name, period, fmult, H, d, prefix, wav, kw in CASES:
        ds = load_dataset(name, period, **kw)
        L = fmult * H
        Wtr = forecast_windows(ds, L, H)
        Wte = test_windows(ds, H)
        bases = [
            ("KL (optimal)", f"kl:{prefix}"),
            ("DCT", "dct"),
            ("random ortho", "random_ortho"),
            ("random ortho:7", "random_ortho:7"),
            (f"wavelet {wav}", wav),
        ]
        for label, kind in bases:
            B = WG._build_basis(H, kind, 5, d, 0)
            ctr, cte = captured(B, Wtr), captured(B, Wte)
            print(
                f"{f'{period} d={d}':<18}{label:<16}{d/H:>7.3f}{ctr:>9.4f}{cte:>9.4f}"
            )
            rows.append((period, label, d / H, ctr, cte))
        print("-" * 59)

    # Verdict on the predicted ordering, evaluated per case on the train ensemble.
    print("\nPredicted ordering  KL >= DCT > random > wavelet")
    for name, period, fmult, H, d, prefix, wav, kw in CASES:
        sub = {lbl: tr for (p, lbl, _, tr, _) in rows if p == period}
        kl, dct = sub["KL (optimal)"], sub["DCT"]
        rnd = max(sub["random ortho"], sub["random ortho:7"])
        wv = sub[f"wavelet {wav}"]
        ok = kl >= dct - 1e-4 and dct > rnd and rnd > wv
        gap = kl - dct
        note = "KL ~= DCT (ceiling)" if gap < 1e-3 else f"KL beats DCT by {gap:.4f}"
        print(
            f"  {period:<12} KL {kl:.4f} / DCT {dct:.4f} / rand {rnd:.4f} "
            f"/ {wav} {wv:.4f}   {'HOLDS' if ok else 'VIOLATED'}  [{note}]"
        )


if __name__ == "__main__":
    main()
