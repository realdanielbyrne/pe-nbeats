"""Analysis for the three theory-testing studies.

Each section states the registered prediction, then tests it.

  E-KL  Does the basis linear approximation theory names as optimal actually win?
  E-P2  Does the completeness null survive a nonlinearity AFTER the basis and
        break under one BETWEEN the projection and the basis?
  E-TR  Does the predicted ordering hold on a different dataset, protocol and
        loss, where the predictions genuinely separate?

Run from the repository root:
    python experiments/scripts/analyze_basis_theory.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
R = "experiments/results"


def load(path):
    p = os.path.join(R, path)
    return pd.read_csv(p) if os.path.exists(p) else None


def cell(df, name, metric):
    """Seed-indexed metric series for one config, dropping failures."""
    d = df[df["config_name"] == name].copy()
    d = d[np.isfinite(d[metric])]
    if "diverged" in d.columns:
        d = d[~d["diverged"].fillna(False).astype(bool)]
    return d.set_index("seed")[metric].sort_index()


def paired(a, b):
    """Paired Wilcoxon on the seeds a and b share."""
    common = a.index.intersection(b.index)
    if len(common) < 5:
        return None
    x, y = a.loc[common], b.loc[common]
    if np.allclose(x.values, y.values):
        return dict(n=len(common), diff=0.0, p=1.0)
    st = stats.wilcoxon(x, y)
    return dict(n=len(common), diff=float(x.mean() - y.mean()), p=float(st.pvalue))


def summarise(df, metric, label="config_name"):
    g = df.groupby(label)[metric].agg(["mean", "std", "count"])
    return g.sort_values("mean")


# ---------------------------------------------------------------- E-KL
def e_kl():
    print("=" * 78)
    print("E-KL  The Karhunen-Loeve arm on M4")
    print("  Registered prediction: KL TIES the DCT, because unnormalised M4")
    print("  windows are DC dominated and both sit at the capture ceiling")
    print("  (Hourly KL 1.0000 / DCT 1.0000; Monthly 0.9945 / 0.9943).")
    print("=" * 78)
    for per, klf, base, klname, dctname, wavname in (
        ("Hourly", "m4/m4_kl_basis_results.csv", "m4/m4_truncated_identity_results.csv",
         "W_10s_fbd16_kl", "W_10s_fbd16_dct", "W_10s_fbd16_db3"),
        ("Monthly", "m4/m4_kl_basis_monthly_results.csv",
         "m4/m4_truncated_identity_monthly_results.csv",
         "W_10s_fbd6_kl", "W_10s_fbd6_dct", "W_10s_fbd6_db3"),
    ):
        k, b = load(klf), load(base)
        if k is None or b is None or k.empty:
            print(f"  {per}: not ready")
            continue
        kl = cell(k, klname, "smape")
        print(f"\n  {per}  (KL n={len(kl)}, mean sMAPE {kl.mean():.3f} +/- {kl.std():.3f})")
        for other in (dctname, wavname):
            o = cell(b, other, "smape")
            if o.empty:
                continue
            r = paired(kl, o)
            if r is None:
                print(f"    vs {other:24s} n too small")
                continue
            verdict = "TIE" if r["p"] > 0.05 else ("KL WINS" if r["diff"] < 0 else "KL LOSES")
            print(f"    vs {other:24s} mean {o.mean():8.3f}  "
                  f"delta {r['diff']:+7.3f}  p={r['p']:.4f}  n={r['n']}  {verdict}")


# ---------------------------------------------------------------- E-P2
def e_p2():
    print("\n" + "=" * 78)
    print("E-P2  Boundary test for the completeness argument")
    print("  Registered prediction: identity NULL under 'none' and 'after_basis',")
    print("  identity SIGNIFICANT under 'between'.")
    print("=" * 78)
    d = load("m4/m4_p2_boundary_results.csv")
    if d is None or d.empty:
        print("  not ready")
        return
    d = d[np.isfinite(d["smape"])]
    for per in sorted(d["period"].unique()):
        dp = d[d["period"] == per]
        print(f"\n  {per}")
        for cond in ("none", "after_basis", "between"):
            dc = dp[dp["nonlinearity_placement"] == cond]
            if dc.empty:
                continue
            groups = [g["smape"].values for _, g in dc.groupby("basis_kind")]
            names = [k for k, _ in dc.groupby("basis_kind")]
            if len(groups) < 2 or min(len(g) for g in groups) < 3:
                print(f"    {cond:12s} incomplete ({[len(g) for g in groups]})")
                continue
            kw = stats.kruskal(*groups)
            means = {n: float(np.mean(g)) for n, g in zip(names, groups)}
            spread = max(means.values()) - min(means.values())
            pred_null = cond in ("none", "after_basis")
            got_null = kw.pvalue > 0.05
            mark = "AS PREDICTED" if pred_null == got_null else "*** CONTRARY ***"
            best = min(means, key=means.get)
            worst = max(means, key=means.get)
            print(f"    {cond:12s} KW p={kw.pvalue:.4f}  spread={spread:6.3f}  "
                  f"best={best}({means[best]:.3f}) worst={worst}({means[worst]:.3f})  {mark}")


# ---------------------------------------------------------------- E-TR
def e_tr():
    print("\n" + "=" * 78)
    print("E-TR  Traffic-96 identity at truncated forecast rank (d/m = 32/96)")
    print("  Registered prediction: test error orders INVERSELY to captured")
    print("  energy: KL .9714 > DCT .9630 > rand .3721/.3258 > coif2 .3035")
    print("=" * 78)
    d = load("traffic/traffic96_truncated_identity_results.csv")
    if d is None or d.empty:
        print("  not ready")
        return
    metric = "mse" if "mse" in d.columns else "smape"
    d = d[np.isfinite(d[metric])]
    g = d.groupby("config_name").agg(
        mean=(metric, "mean"), std=(metric, "std"), n=(metric, "count"),
        cap=("predicted_capture_train", "first"),
    ).sort_values("mean")
    print(f"\n  ranked by {metric} (lower is better)")
    print(g.to_string(float_format=lambda v: f"{v:.6f}"))

    if len(g) >= 3 and g["cap"].notna().all():
        rho, p = stats.spearmanr(g["cap"], g["mean"])
        print(f"\n  Spearman(predicted capture, {metric}) = {rho:+.3f}  p={p:.4f}")
        print(f"  {'ORDERING HOLDS' if rho < 0 else '*** ORDERING CONTRARY ***'}"
              " (negative rho = more capture, less error)")

    # pairwise, the contrast the theory is sharpest about
    kl = cell(d, "W_10s_fbd32_kl", metric)
    dct = cell(d, "W_10s_fbd32_dct", metric)
    if not kl.empty and not dct.empty:
        r = paired(kl, dct)
        if r:
            print(f"\n  KL vs DCT: delta {r['diff']:+.6f}  p={r['p']:.4f}  n={r['n']}")
            print("  (theory says KL should edge DCT; a tie still supports the account,"
                  " a DCT win contradicts it)")


if __name__ == "__main__":
    e_kl()
    e_p2()
    e_tr()
    print()
