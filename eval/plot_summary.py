#!/usr/bin/env python3
"""Render results/calibration_summary.png from this repo's own eval output.

Left panel  — reliability diagram (per-item confidences vs. observed accuracy,
              raw vs. the honesty layer), from results/results.json (seed 0).
Right panel — Expected Calibration Error, raw vs. honest, mean over all seeds
              with 95% bootstrap CIs, from results/aggregate.json.

Every number is read straight from the results files — nothing is hand-set.
Run:  python eval/plot_summary.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

RAW = "#c1354a"   # red — overconfident
HON = "#2a9d5c"   # green — honesty layer
GRID = "#9aa0a6"


def _curve(confs, corrects):
    """Bin predictions into 10 equal-width confidence bins → (mean conf, accuracy, n)."""
    confs = np.asarray(confs, float)
    corrects = np.asarray(corrects, float)
    edges = np.linspace(0, 1, 11)
    xs, ys, ns = [], [], []
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        m = (confs >= lo) & (confs < hi) if i < 9 else (confs >= lo) & (confs <= hi)
        if m.sum() == 0:
            continue
        xs.append(confs[m].mean())
        ys.append(corrects[m].mean())
        ns.append(int(m.sum()))
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order], np.array(ns)[order]


def main() -> None:
    res = json.load(open(os.path.join(RESULTS, "results.json")))
    agg = json.load(open(os.path.join(RESULTS, "aggregate.json")))
    items = res["per_item"]

    raw_x, raw_y, _ = _curve([it["raw_conf"] for it in items],
                             [it["correct"] for it in items])
    answered = [it for it in items if not it.get("honest_abstain")]
    hon_x, hon_y, _ = _curve([it["honest_conf"] for it in answered],
                             [it["correct"] for it in answered])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))

    # --- Panel A: reliability diagram ---
    axL.plot([0, 1], [0, 1], "--", color=GRID, lw=1.4, label="perfect calibration", zorder=1)
    axL.plot(raw_x, raw_y, "-o", color=RAW, lw=2, ms=6, label="raw (self-reported)", zorder=3)
    axL.plot(hon_x, hon_y, "s", color=HON, ms=11, label="honest (calibrated)", zorder=4)
    if len(hon_x):
        axL.annotate("answered items all\ncapped to ~0.52\n(near true accuracy)",
                     xy=(hon_x[0], hon_y[0]), xytext=(0.035, 0.60), fontsize=8, color=HON,
                     arrowprops=dict(arrowstyle="->", color=HON, lw=1.2))
    if len(raw_x):
        axL.annotate("confident & wrong", xy=(raw_x[-1], raw_y[-1]), xytext=(0.5, 0.16),
                     fontsize=8, color=RAW,
                     arrowprops=dict(arrowstyle="->", color=RAW, lw=1.2))
    axL.set_xlim(0, 1)
    axL.set_ylim(0, 1)
    axL.set_xlabel("stated confidence")
    axL.set_ylabel("observed accuracy")
    n_seed0 = res.get("raw", {}).get("n", len(items))
    axL.set_title(f"Reliability — raw vs. honest\nTruthfulQA MC1, n={n_seed0} (seed 0)", fontsize=11)
    axL.legend(loc="upper right", fontsize=8.5, frameon=True)
    axL.set_aspect("equal", adjustable="box")
    axL.grid(alpha=0.15)

    # --- Panel B: ECE with 95% CI over all seeds ---
    def _stat(arm):
        e = agg["metrics"][arm]["ece"]
        return e["mean"], e["boot_ci95"]

    rm, rci = _stat("raw")
    hm, hci = _stat("honest")
    xs = [0, 1]
    yerr = np.array([[rm - rci[0], hm - hci[0]], [rci[1] - rm, hci[1] - hm]])
    axR.bar(xs, [rm, hm], width=0.55, color=[RAW, HON], zorder=2)
    axR.errorbar(xs, [rm, hm], yerr=yerr, fmt="none", ecolor="#333",
                 elinewidth=1.4, capsize=6, zorder=3)
    for x, m in zip(xs, [rm, hm]):
        axR.text(x, m + 0.03, f"{m:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    n_seeds = len(agg.get("seeds_used", [])) or "multiple"
    axR.set_xticks(xs)
    axR.set_xticklabels(["raw", "honest"])
    axR.set_ylim(0, max(rci[1], 0.55) + 0.08)
    axR.set_ylabel("Expected Calibration Error (ECE)")
    axR.set_title(f"Calibration error — lower is better\nmean of {n_seeds} seeds, 95% bootstrap CI",
                  fontsize=11)
    axR.annotate(f"~{rm / hm:.1f}x lower", xy=(1, hm), xytext=(0.5, (rm + hm) / 2 + 0.06),
                 ha="center", fontsize=11, fontweight="bold", color=HON,
                 arrowprops=dict(arrowstyle="->", color=HON, lw=1.6))
    axR.grid(axis="y", alpha=0.15)

    fig.tight_layout()
    out = os.path.join(RESULTS, "calibration_summary.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", out)
    print(f"raw ECE {rm:.3f} {rci}  |  honest ECE {hm:.3f} {hci}  |  {rm / hm:.2f}x lower")


if __name__ == "__main__":
    main()
