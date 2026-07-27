"""Analyse the DWT preprocessing comparison against the four registered predictions.

Predictions are stated in experiments/configs/m4_pramanick_comparison.yaml and are
reproduced here so the verdict is computed against them rather than around them:

  1. At comparable budgets the two designs land within ~0.15 sMAPE of each other
     on most periods.
  2. Neither design beats the Generic baseline by a meaningful margin at its own
     budget.
  3. Both reach within ~0.2 sMAPE of the Generic baseline at roughly 1.3M params.
  4. The DWT preprocessing arm degrades more on short horizon periods, where only
     2 or 3 decomposition levels are admissible, than on Hourly where 5 are.

  FALSIFIED IF preprocessing wins by more than 0.3 sMAPE on four or more periods
  at comparable budgets.

Usage:
    python experiments/scripts/analyze_pramanick.py
"""

import os
import sys

import pandas as pd
from scipy.stats import mannwhitneyu

CSV = "experiments/results/m4/m4_pramanick_comparison_results.csv"
PERIODS = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]
DWT_ARMS = ["DWTpre_db3_2b_gw128", "DWTpre_db3_2b_gw256", "DWTpre_db3_2b_gw512"]
MINE_ARMS = ["T+Db3V3_10s_gw128", "T+Db3V3_10s_gw256", "T+Db3V3_10s_gw512"]
BASELINE = "NBEATS-G_10s_ag0"

# Pairing by width LABEL is not pairing by budget: the two designs spend
# parameters differently, so on Yearly the DWT arm at width 128 holds 348,672
# against 1,309,470 for the in-block arm at the same label, a 3.8x handicap.
# Every comparison below therefore pairs each DWT arm with whichever in-block arm
# is closest to it in actual parameter count for that period, and prints the
# residual gap so an unmatched pair cannot masquerade as a matched one.
MAX_BUDGET_GAP = 0.25  # skip pairs further apart than this fraction


def cell(df, config, period):
    return df[(df.config_name == config) & (df.period == period)].smape


def n_params(df, config, period):
    rows = df[(df.config_name == config) & (df.period == period)]
    return None if rows.empty else int(rows.n_params.iloc[0])


def budget_matched_pairs(df, period):
    """Pair each DWT arm with the nearest in-block arm by parameter count."""
    pairs = []
    for dwt in DWT_ARMS:
        pd_params = n_params(df, dwt, period)
        if pd_params is None:
            continue
        best, best_gap = None, None
        for mine in MINE_ARMS:
            mp = n_params(df, mine, period)
            if mp is None:
                continue
            gap = abs(mp - pd_params) / max(mp, pd_params)
            if best_gap is None or gap < best_gap:
                best, best_gap = mine, gap
        if best is not None and best_gap <= MAX_BUDGET_GAP:
            pairs.append((dwt, best, best_gap, pd_params, n_params(df, best, period)))
    return pairs


def main():
    if not os.path.exists(CSV):
        sys.exit(f"No results at {CSV}")
    df = pd.read_csv(CSV)
    print(f"{len(df)} rows, {df.config_name.nunique()} configs, "
          f"{df.period.nunique()} periods\n")

    print("=" * 78)
    print("PARAMETER COUNTS (the budgets actually compared)")
    print("=" * 78)
    params = df.groupby(["period", "config_name"]).n_params.first().unstack()
    print(params.reindex([p for p in PERIODS if p in params.index]).to_string())

    print("\n" + "=" * 78)
    print("MEAN sMAPE BY CONFIG AND PERIOD")
    print("=" * 78)
    table = df.groupby(["period", "config_name"]).smape.mean().unstack()
    print(table.reindex([p for p in PERIODS if p in table.index]).round(3).to_string())

    print("\n" + "=" * 78)
    print("PREDICTION 1 and FALSIFICATION TEST: design vs design at each rung")
    print("=" * 78)
    big_pre_wins = set()
    within_015 = 0
    total = 0
    for period in PERIODS:
        for dwt_cfg, mine_cfg, gap, dp, mp in budget_matched_pairs(df, period):
            a, b = cell(df, dwt_cfg, period), cell(df, mine_cfg, period)
            if len(a) < 3 or len(b) < 3:
                continue
            total += 1
            delta = a.mean() - b.mean()  # negative favours DWT preprocessing
            p = mannwhitneyu(a, b).pvalue
            if abs(delta) <= 0.15:
                within_015 += 1
            if delta < -0.3:
                big_pre_wins.add(period)
            winner = "dwt-pre" if delta < 0 else "in-block"
            print(f"{period:10s} dwt {dp:>10,} {a.mean():7.3f}  vs  "
                  f"in-block {mp:>10,} {b.mean():7.3f}  "
                  f"budget gap {gap*100:4.1f}%  delta {delta:+7.3f}  "
                  f"p={p:.4f}  -> {winner}")
    if total:
        print(f"\nWithin 0.15 sMAPE: {within_015}/{total} budget matched comparisons")
        print(f"Periods where DWT preprocessing wins by more than 0.3: "
              f"{len(big_pre_wins)} {sorted(big_pre_wins)} "
              f"(falsification threshold is 4 or more)")

    print("\n" + "=" * 78)
    print("PREDICTION 2 and 3: each design against the Generic baseline")
    print("=" * 78)
    for period in PERIODS:
        base = cell(df, BASELINE, period)
        if len(base) < 3:
            continue
        base_params = df[(df.config_name == BASELINE)
                         & (df.period == period)].n_params.iloc[0]
        print(f"\n{period}  baseline {base.mean():.3f} at {base_params:,} params")
        for cfg in DWT_ARMS + MINE_ARMS:
            arm = cell(df, cfg, period)
            if len(arm) < 3:
                continue
            arm_params = df[(df.config_name == cfg)
                            & (df.period == period)].n_params.iloc[0]
            p = mannwhitneyu(arm, base).pvalue
            print(f"   {cfg:22s} {arm.mean():7.3f}  delta {arm.mean()-base.mean():+7.3f}"
                  f"  p={p:.4f}  {base_params/arm_params:5.1f}x smaller")

    print("\n" + "=" * 78)
    print("PREDICTION 4: does DWT preprocessing suffer more at short horizons?")
    print("=" * 78)
    print("Horizon and admissible decomposition depth drive the sub-band count.")
    for period in PERIODS:
        deltas = []
        for dwt_cfg, mine_cfg, _gap, _dp, _mp in budget_matched_pairs(df, period):
            a, b = cell(df, dwt_cfg, period), cell(df, mine_cfg, period)
            if len(a) >= 3 and len(b) >= 3:
                deltas.append(a.mean() - b.mean())
        if deltas:
            h = df[df.period == period].forecast_length.iloc[0]
            print(f"{period:10s} H={h:3d}  mean delta across rungs "
                  f"{sum(deltas)/len(deltas):+7.3f}")

    print("\n" + "=" * 78)
    print("DIVERGENCES")
    print("=" * 78)
    coll = df[df.smape.round(3) == 200.0]
    print(f"runs at exactly sMAPE 200: {len(coll)}")
    if len(coll):
        print(coll.groupby(["design", "config_name"]).size().to_string())


if __name__ == "__main__":
    main()
