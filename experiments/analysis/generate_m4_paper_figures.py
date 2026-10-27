#!/usr/bin/env python3
"""Generate M4 paper figures directly from the comprehensive sweep CSV.

This script intentionally uses only the Python standard library so it can run
in lightweight environments where matplotlib/pandas are unavailable.
"""

from __future__ import annotations

import ast
import csv
import math
import os
import statistics as stats
from dataclasses import dataclass
from typing import Iterable


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(
    ROOT, "experiments", "results", "m4", "comprehensive_sweep_m4_results.csv"
)
OUT_DIR = os.path.join(os.path.dirname(ROOT), "pe-nbeats-paper", "figures")  # sibling private paper repo
FIG5_PATH = os.path.join(OUT_DIR, "figure5_m4_training_curves.svg")


@dataclass
class Run:
    period: str
    config_name: str
    smape: float
    val_loss_curve: list[float]


def load_runs(path: str) -> list[Run]:
    runs: list[Run] = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            curve = [float(v) for v in ast.literal_eval(row["val_loss_curve"])]
            runs.append(
                Run(
                    period=row["period"],
                    config_name=row["config_name"],
                    smape=float(row["smape"]),
                    val_loss_curve=curve,
                )
            )
    return runs


def mean_curve(curves: Iterable[list[float]]) -> list[float]:
    curves = list(curves)
    max_len = max(len(curve) for curve in curves)
    out: list[float] = []
    for idx in range(max_len):
        vals = [curve[idx] for curve in curves if idx < len(curve)]
        out.append(stats.fmean(vals))
    return out


def smape_summary(runs: Iterable[Run]) -> tuple[float, float]:
    vals = [run.smape for run in runs]
    return stats.fmean(vals), stats.stdev(vals) if len(vals) > 1 else 0.0


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def curve_path(
    curve: list[float],
    x0: float,
    y0: float,
    width: float,
    height: float,
    x_max: int,
    y_min: float,
    y_max: float,
) -> str:
    if len(curve) == 1:
        x = x0
        y = y0 + height / 2
        return f"M {x:.2f} {y:.2f}"
    pts = []
    y_span = max(y_max - y_min, 1e-9)
    for idx, val in enumerate(curve):
        x = x0 + width * idx / max(x_max - 1, 1)
        y = y0 + height - ((val - y_min) / y_span) * height
        pts.append((x, y))
    return "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in pts)


def draw_axes(
    parts: list[str],
    x0: float,
    y0: float,
    width: float,
    height: float,
    y_ticks: list[float],
    y_min: float,
    y_max: float,
    x_max: int,
) -> None:
    parts.append(
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{width:.1f}" height="{height:.1f}" '
        'fill="#ffffff" stroke="#d1d5db" stroke-width="1"/>'
    )
    for tick in y_ticks:
        y = y0 + height - ((tick - y_min) / max(y_max - y_min, 1e-9)) * height
        parts.append(
            f'<line x1="{x0:.1f}" y1="{y:.2f}" x2="{x0 + width:.1f}" y2="{y:.2f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x0 - 8:.1f}" y="{y + 4:.2f}" font-size="11" '
            'text-anchor="end" fill="#4b5563">'
            f"{tick:.0f}</text>"
        )
    for frac, label in [(0.0, "0"), (0.5, str(x_max // 2)), (1.0, str(x_max))]:
        x = x0 + width * frac
        parts.append(
            f'<line x1="{x:.2f}" y1="{y0 + height:.1f}" x2="{x:.2f}" y2="{y0 + height + 5:.1f}" '
            'stroke="#9ca3af" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{y0 + height + 18:.1f}" font-size="11" '
            'text-anchor="middle" fill="#4b5563">'
            f"{label}</text>"
        )


def add_panel_title(parts: list[str], x: float, y: float, title: str, subtitle: str) -> None:
    parts.append(
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="14" font-weight="700" '
        'fill="#111827">'
        f"{svg_escape(title)}</text>"
    )
    parts.append(
        f'<text x="{x:.1f}" y="{y + 18:.1f}" font-size="11" fill="#4b5563">'
        f"{svg_escape(subtitle)}</text>"
    )


def add_legend(
    parts: list[str],
    items: list[tuple[str, str]],
    x: float,
    y: float,
    line_width: float = 3.0,
) -> None:
    cur_y = y
    for label, color in items:
        parts.append(
            f'<line x1="{x:.1f}" y1="{cur_y:.1f}" x2="{x + 18:.1f}" y2="{cur_y:.1f}" '
            f'stroke="{color}" stroke-width="{line_width:.1f}" stroke-linecap="round"/>'
        )
        parts.append(
            f'<text x="{x + 24:.1f}" y="{cur_y + 4:.1f}" font-size="11" fill="#374151">'
            f"{svg_escape(label)}</text>"
        )
        cur_y += 16


def render_figure_5(runs: list[Run], out_path: str) -> None:
    yearly_cfgs = [
        ("NBEATS-IG_10s_ag0", "#111827"),
        ("TW_10s_td3_bdeq_coif2", "#0f766e"),
        ("T+Db3V3_30s_bd2eq", "#2563eb"),
        ("TWAELG_10s_ld16_coif2_agf", "#dc2626"),
    ]
    yearly_runs = {
        cfg: [r for r in runs if r.period == "Yearly" and r.config_name == cfg]
        for cfg, _ in yearly_cfgs
    }
    quarterly_generic = [
        r for r in runs if r.period == "Quarterly" and r.config_name == "NBEATS-G_30s_ag0"
    ]
    weekly_generic_ag0 = [
        r for r in runs if r.period == "Weekly" and r.config_name == "NBEATS-G_30s_ag0"
    ]
    weekly_generic_agf = [
        r for r in runs if r.period == "Weekly" and r.config_name == "NBEATS-G_30s_agf"
    ]

    width = 1380
    height = 520
    margin = 48
    gutter = 28
    panel_width = (width - 2 * margin - 2 * gutter) / 3
    plot_height = 280
    title_y = 72
    plot_y = 110

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="48" y="28" font-size="20" font-weight="700" fill="#111827">Figure 5. M4 training-curve diagnostics from comprehensive_sweep_m4_results.csv</text>',
        '<text x="48" y="48" font-size="12" fill="#4b5563">Validation curves are shown directly from the stored val_loss_curve traces. Lower is better.</text>',
    ]

    # Panel A
    x0 = margin
    y0 = plot_y
    x_max = max(len(mean_curve([r.val_loss_curve for r in rs])) for rs in yearly_runs.values())
    mean_curves = {cfg: mean_curve([r.val_loss_curve for r in rs]) for cfg, rs in yearly_runs.items()}
    y_min = min(min(curve) for curve in mean_curves.values()) - 0.2
    y_max = max(max(curve[:10]) for curve in mean_curves.values()) + 0.5
    y_ticks = [14, 18, 22, 26, 30]
    add_panel_title(
        parts,
        x0,
        title_y,
        "A. Yearly: representative architectures converge to the same band",
        "Mean validation curves across 10 seeds for four near-frontier M4-Yearly models.",
    )
    draw_axes(parts, x0, y0, panel_width, plot_height, y_ticks, y_min, y_max, x_max)
    for cfg, color in yearly_cfgs:
        path = curve_path(mean_curves[cfg], x0, y0, panel_width, plot_height, x_max, y_min, y_max)
        parts.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.8" stroke-linecap="round"/>'
        )
    add_legend(
        parts,
        [
            ("NBEATS-IG_10s_ag0", "#111827"),
            ("TW_10s_td3_bdeq_coif2", "#0f766e"),
            ("T+Db3V3_30s_bd2eq", "#2563eb"),
            ("TWAELG_10s_ld16_coif2_agf", "#dc2626"),
        ],
        x0 + 10,
        y0 + plot_height + 34,
    )

    # Panel B
    x1 = margin + panel_width + gutter
    q_mean, q_std = smape_summary(quarterly_generic)
    q_curves = [run.val_loss_curve for run in quarterly_generic]
    x_max = max(len(curve) for curve in q_curves)
    y_min = min(min(curve) for curve in q_curves) - 0.5
    y_max = max(max(curve) for curve in q_curves) + 1.0
    y_ticks = [10, 15, 20, 25, 30]
    add_panel_title(
        parts,
        x1,
        title_y,
        "B. Quarterly: 30-stack Generic is seed-sensitive without active_g",
        f"NBEATS-G_30s_ag0: mean SMAPE {q_mean:.2f} ± {q_std:.2f} across 10 seeds.",
    )
    draw_axes(parts, x1, y0, panel_width, plot_height, y_ticks, y_min, y_max, x_max)
    for idx, curve in enumerate(sorted(q_curves, key=lambda c: min(c))):
        color = "#ef4444" if idx >= len(q_curves) - 2 else "#f59e0b"
        opacity = "0.85" if idx >= len(q_curves) - 2 else "0.45"
        path = curve_path(curve, x1, y0, panel_width, plot_height, x_max, y_min, y_max)
        parts.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-opacity="{opacity}"/>'
        )
    parts.append(
        f'<text x="{x1 + 10:.1f}" y="{y0 + plot_height + 36:.1f}" font-size="11" fill="#374151">'
        "Thin lines are individual seeds; red trajectories are the highest-loss seeds.</text>"
    )

    # Panel C
    x2 = margin + 2 * (panel_width + gutter)
    w0_mean, w0_std = smape_summary(weekly_generic_ag0)
    wf_mean, wf_std = smape_summary(weekly_generic_agf)
    all_curves = [run.val_loss_curve for run in weekly_generic_ag0 + weekly_generic_agf]
    x_max = max(len(curve) for curve in all_curves)
    y_min = min(min(curve) for curve in all_curves) - 0.5
    y_max = max(max(curve) for curve in all_curves) + 1.0
    y_ticks = [6, 10, 14, 18, 22]
    add_panel_title(
        parts,
        x2,
        title_y,
        "C. Weekly: active_g='forecast' collapses Generic variance",
        (
            f"NBEATS-G_30s ag0: {w0_mean:.2f} ± {w0_std:.2f} SMAPE; "
            f"agf: {wf_mean:.2f} ± {wf_std:.2f}."
        ),
    )
    draw_axes(parts, x2, y0, panel_width, plot_height, y_ticks, y_min, y_max, x_max)
    for curve in [run.val_loss_curve for run in weekly_generic_ag0]:
        path = curve_path(curve, x2, y0, panel_width, plot_height, x_max, y_min, y_max)
        parts.append(
            f'<path d="{path}" fill="none" stroke="#ef4444" stroke-width="1.5" '
            'stroke-linecap="round" stroke-opacity="0.35"/>'
        )
    for curve in [run.val_loss_curve for run in weekly_generic_agf]:
        path = curve_path(curve, x2, y0, panel_width, plot_height, x_max, y_min, y_max)
        parts.append(
            f'<path d="{path}" fill="none" stroke="#2563eb" stroke-width="1.5" '
            'stroke-linecap="round" stroke-opacity="0.45"/>'
        )
    mean_ag0 = mean_curve([run.val_loss_curve for run in weekly_generic_ag0])
    mean_agf = mean_curve([run.val_loss_curve for run in weekly_generic_agf])
    parts.append(
        f'<path d="{curve_path(mean_ag0, x2, y0, panel_width, plot_height, x_max, y_min, y_max)}" '
        'fill="none" stroke="#b91c1c" stroke-width="3.0" stroke-linecap="round"/>'
    )
    parts.append(
        f'<path d="{curve_path(mean_agf, x2, y0, panel_width, plot_height, x_max, y_min, y_max)}" '
        'fill="none" stroke="#1d4ed8" stroke-width="3.0" stroke-linecap="round"/>'
    )
    add_legend(
        parts,
        [("ag0 seeds + mean", "#b91c1c"), ("agf seeds + mean", "#1d4ed8")],
        x2 + 10,
        y0 + plot_height + 34,
    )

    # Shared axis labels
    for panel_x in (x0, x1, x2):
        parts.append(
            f'<text x="{panel_x + panel_width / 2:.1f}" y="{y0 + plot_height + 72:.1f}" '
            'font-size="12" text-anchor="middle" fill="#374151">Epoch</text>'
        )
        parts.append(
            f'<text x="{panel_x - 34:.1f}" y="{y0 + plot_height / 2:.1f}" font-size="12" '
            'text-anchor="middle" fill="#374151" transform="rotate(-90 '
            f'{panel_x - 34:.1f} {y0 + plot_height / 2:.1f})">Validation SMAPE</text>'
        )

    parts.append("</svg>")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))


def main() -> None:
    runs = load_runs(CSV_PATH)
    render_figure_5(runs, FIG5_PATH)
    print(f"Wrote {FIG5_PATH}")


if __name__ == "__main__":
    main()
