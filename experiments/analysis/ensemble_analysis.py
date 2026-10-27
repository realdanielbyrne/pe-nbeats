"""Temporary script to compute median ensemble metrics across all configs.

Does NOT import run_unified_benchmark (which installs signal handlers that
interfere with subprocess stdout capture). Instead, reimplements the needed
metric functions inline.
"""
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)
signal.signal(signal.SIGTERM, signal.SIG_DFL)

import sys, os, json
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_DIR, "..", "src"))

from pe_nbeats.data import M4Dataset

# --- Metric functions (copied from run_unified_benchmark to avoid import) ---
def compute_smape(y_pred, y_true):
    denom = (np.abs(y_true) + np.abs(y_pred))
    ratio = np.where(denom == 0, 0.0, 2.0 * np.abs(y_true - y_pred) / denom)
    return float(np.mean(ratio) * 100)

def compute_m4_mase(y_pred, y_true, train_series_list, frequency):
    n_series = y_pred.shape[0]
    mase_values = []
    for i in range(n_series):
        train_i = train_series_list[i]
        pred_i = y_pred[i]; true_i = y_true[i]
        forecast_mae = np.mean(np.abs(true_i - pred_i))
        m = max(1, frequency)
        if len(train_i) <= m: continue
        naive_errors = np.abs(train_i[m:] - train_i[:-m])
        naive_mae = np.mean(naive_errors)
        if naive_mae < 1e-10: continue
        mase_values.append(forecast_mae / naive_mae)
    return float(np.mean(mase_values)) if mase_values else float("nan")

# --- Config registry (copied from run_unified_benchmark) ---
UNIFIED_CONFIGS = {
    "NBEATS-G":                {"category": "paper_baseline"},
    "NBEATS-I":                {"category": "paper_baseline"},
    "NBEATS-I+G":              {"category": "paper_baseline"},
    "GenericAE":               {"category": "novel_ae"},
    "BottleneckGenericAE":     {"category": "novel_ae"},
    "AutoEncoder":             {"category": "novel_ae"},
    "BottleneckGeneric":       {"category": "novel_basis"},
    "Coif2WaveletV3":          {"category": "novel_basis"},
    "DB4WaveletV3":            {"category": "novel_basis"},
    "Trend+HaarWaveletV3":     {"category": "novel_mixed"},
    "Trend+DB3WaveletV3":      {"category": "novel_mixed"},
    "Trend+Coif2WaveletV3":    {"category": "novel_mixed"},
    "Generic+DB3WaveletV3":    {"category": "novel_mixed"},
    "NBEATS-I+GenericAE":      {"category": "novel_mixed"},
    "NBEATS-I+BottleneckGeneric": {"category": "novel_mixed"},
    "NBEATS-I-AE":             {"category": "novel_mixed"},
}

pred_dir = os.path.join(_DIR, "results", "m4", "predictions")
results = []

# Cache datasets
ds_cache = {}
series_cache = {}
for period in ["Yearly", "Quarterly", "Monthly", "Weekly"]:
    ds_cache[period] = M4Dataset(period, "All")
    series_cache[period] = ds_cache[period].get_training_series()
    print(f"  Loaded {period}: {ds_cache[period].train_data.shape}", flush=True)

for config in UNIFIED_CONFIGS:
    cat = UNIFIED_CONFIGS[config]["category"]
    for exp in ["baseline", "activeG_fcast"]:
        for period in ["Yearly", "Quarterly", "Monthly", "Weekly"]:
            preds_list = []
            targets = None
            for run in range(10):
                fpath = os.path.join(pred_dir, f"{exp}_{config}_{period}_run{run}.npz")
                if not os.path.exists(fpath):
                    break
                d = np.load(fpath)
                preds_list.append(d["preds"])
                if targets is None: targets = d["targets"]
            if len(preds_list) < 10:
                continue

            ens = np.median(np.stack(preds_list), axis=0)
            ds = ds_cache[period]
            s = compute_smape(ens, targets)
            m = compute_m4_mase(ens, targets, series_cache[period], ds.frequency)
            o = ds.compute_owa(s, m)
            results.append({"config": config, "category": cat, "experiment": exp,
                           "period": period, "owa": o, "smape": s, "mase": m})

df = pd.DataFrame(results)
print(f"\nTotal ensemble results: {len(df)}", flush=True)

for period in ["Yearly", "Quarterly", "Monthly", "Weekly"]:
    pdf = df[df["period"]==period].sort_values("owa").head(10)
    print(f"\n{'~'*95}")
    print(f"  {period} -- Top 10 Ensemble OWA")
    print(f"{'~'*95}")
    print(f"  {'#':<3} {'Config':<28} {'Pass':<14} {'Category':<16} {'OWA':>7} {'sMAPE':>8} {'MASE':>7}")
    for i,(_, r) in enumerate(pdf.iterrows()):
        print(f"  {i+1:<3} {r['config']:<28} {r['experiment']:<14} {r['category']:<16} {r['owa']:>7.4f} {r['smape']:>8.3f} {r['mase']:>7.3f}")

best = df.loc[df.groupby(["config","period"])["owa"].idxmin()]
avg = best.groupby(["config","category"]).agg(
    owa=("owa","mean"), smape=("smape","mean"), mase=("mase","mean"), n=("period","count")
).reset_index()
avg = avg[avg["n"]==4].sort_values("owa")

print(f"\n{'='*95}")
print(f"  CROSS-PERIOD AVERAGE (best pass per config)")
print(f"{'='*95}")
print(f"  {'#':<3} {'Config':<28} {'Category':<16} {'Avg OWA':>9} {'sMAPE':>8} {'MASE':>7}")
for i,(_, r) in enumerate(avg.iterrows()):
    tag = " <-- PAPER" if r["category"]=="paper_baseline" else ""
    print(f"  {i+1:<3} {r['config']:<28} {r['category']:<16} {r['owa']:>9.4f} {r['smape']:>8.3f} {r['mase']:>7.3f}{tag}")

print(f"\n  BY CATEGORY:")
for cat, g in avg.groupby("category"):
    print(f"    {cat:<18} mean={g['owa'].mean():.4f}  best={g['owa'].min():.4f}  n={len(g)}")

print("\nDone.", flush=True)

