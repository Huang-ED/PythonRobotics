"""Figures for README.md, regenerated from the planner's own cost functions.

    python -m PathPlanning.DWAT_v5_modular.results.ablation_horizon_weighting.make_figures

fig1_same_danger_different_time.png
    Two candidate trajectories with the SAME close pass -- same minimum
    clearance, same closing speed -- one 2 s ahead and one 10 s ahead. (2 s,
    not 1 s: at 1 s part of the pass falls before t = 0 and is cut off, which
    would make even uniform averaging score the two differently.) The
    severity and the horizon aggregation are dwa_ramped's (exponential
    barrier, aggregate_compound_costs "mean") at v21.13's parameters; only
    time_decay_lambda varies. Uniform averaging scores the two passes
    identically; a discount scores the imminent one higher.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from PathPlanning.DWAT_v4_st import dwa_average_weighted_side as base  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG = json.load(open(HERE.parents[1] / "configs" / "algorithm" / "v21.13.json"))

# Reference palette (dataviz skill): categorical slots 1-2, text and surface tokens.
SOON, LATER = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GRAYS = ["#9a9994", "#6b6a66", "#2f2f2d"]       # weightings, light -> dark = lambda*T 0 -> 3

T = CFG["predict_time_obstacle"]
DT = CFG["dt"]
t = np.arange(int(round(T / DT)) + 1) * DT

D_MIN = 0.3        # the "same danger distance" [u]
CLOSING = 0.9      # head-on closing speed of the open-water suite [u/s]
T_SOON, T_LATER = 2.0, 10.0
ARMS = [("uniform, λ·T = 0", 0.0), ("v21.13, λ·T = 1", CFG["time_decay_lambda"]),
        ("strong, λ·T = 3", 3.0 / T)]


def severity(d):
    """dwa_ramped's exponential barrier: S = M * exp(-(decay / M^2) * d^2)."""
    m = CFG["max_side_weight_dist"]
    return m * np.exp(-(CFG["side_clearance_gaussian_decay"] / m ** 2) * d ** 2)


def clearance(t_cpa):
    return np.sqrt(D_MIN ** 2 + (CLOSING * (t - t_cpa)) ** 2)


def cost(s, lam):
    """The planner's horizon reduction, exactly: side gain x aggregate("mean")."""
    w = np.exp(-lam * t)
    c, _, _ = base.aggregate_compound_costs(s, w, "mean")
    return CFG["side_cost_gain"] * c


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig1():
    s_soon, s_later = severity(clearance(T_SOON)), severity(clearance(T_LATER))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), facecolor=SURFACE,
                             gridspec_kw=dict(width_ratios=[1.15, 1.15, 1.0], wspace=0.32))

    ax = axes[0]
    _style(ax)
    ax.plot(t, s_soon, color=SOON, lw=2)
    ax.plot(t, s_later, color=LATER, lw=2)
    ax.text(T_SOON + 0.3, s_soon.max() * 1.05, f"pass in {T_SOON:.0f} s", color=INK, fontsize=9)
    ax.text(T_LATER + 0.9, s_later.max() * 0.62, f"pass in {T_LATER:.0f} s", color=INK, fontsize=9)
    ax.set_xlim(0, T)
    ax.set_ylim(0, s_soon.max() * 1.18)
    ax.set_xlabel("time ahead in the candidate's rollout [s]", color=INK2, fontsize=9)
    ax.set_title(f"(a) Same close pass ({D_MIN} u), different time", color=INK, fontsize=10, loc="left")
    ax.set_ylabel("threat severity S", color=INK2, fontsize=9)

    ax = axes[1]
    _style(ax)
    # Labels at x = 28 s: uniform above its line, the two discounts below theirs.
    for (name, lam), col, above in zip(ARMS, GRAYS, (True, False, False)):
        w = np.exp(-lam * t)
        w = w / w.mean()
        ax.plot(t, w, color=col, lw=2)
        y = np.interp(28.0, t, w)
        ax.text(28.0, y + (0.08 if above else -0.08), name, color=INK, fontsize=8.5,
                ha="center", va="bottom" if above else "top")
    for x, col in ((T_SOON, SOON), (T_LATER, LATER)):
        ax.axvline(x, color=col, lw=1, ls=(0, (3, 3)))
    ax.set_xlim(0, T)
    ax.set_ylim(0, 3.4)
    ax.set_xlabel("time ahead in the candidate's rollout [s]", color=INK2, fontsize=9)
    ax.set_title("(b) Normalised time weight", color=INK, fontsize=10, loc="left")
    ax.set_ylabel("weight relative to uniform", color=INK2, fontsize=9)

    ax = axes[2]
    _style(ax)
    x = np.arange(len(ARMS))
    width = 0.36
    for i, (name, lam) in enumerate(ARMS):
        cs, cl = cost(s_soon, lam), cost(s_later, lam)
        ax.bar(i - width / 2 - 0.01, cs, width, color=SOON, edgecolor=SURFACE, linewidth=2)
        ax.bar(i + width / 2 + 0.01, cl, width, color=LATER, edgecolor=SURFACE, linewidth=2)
        ax.text(i, max(cs, cl) * 1.04 + 0.002, f"{cs / cl:.2f}×", color=INK, fontsize=9, ha="center")
    ax.set_xticks(x, [n.replace(", ", "\n") for n, _ in ARMS])
    ax.set_ylim(0, max(cost(s_soon, lam) for _, lam in ARMS) * 1.22)
    ax.set_title("(c) Dynamic-obstacle cost", color=INK, fontsize=10, loc="left")
    ax.set_ylabel("cost, as the planner scores it", color=INK2, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SOON), plt.Rectangle((0, 0), 1, 1, color=LATER)]
    ax.legend(handles, [f"pass in {T_SOON:.0f} s", f"pass in {T_LATER:.0f} s"], frameon=False,
              fontsize=8.5, loc="upper left", labelcolor=INK)

    out = HERE / "fig1_same_danger_different_time.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(fig1())
