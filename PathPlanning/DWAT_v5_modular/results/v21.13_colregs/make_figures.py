"""Figures for README.md in this folder, all read from the run logs under runs/.

Run from the repository root:
    python PathPlanning/DWAT_v5_modular/results/v21.13_colregs/make_figures.py \\
        PathPlanning/DWAT_v5_modular/results/v21.13_colregs
"""
import glob
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge, FancyArrow

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

# reference palette (dataviz skill, light mode), fixed order
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
})

RUNS = {
    "v21.12": "runs/v21.11/braking_rollout_static",
    "side_only": "runs/v21.13/side_only",
    "v21.13": "runs/v21.13/port_turn_release6",
    "port_turn": "runs/v21.13/port_turn",
}


def load(variant, group, scen):
    p = glob.glob(f"{RUNS[variant]}/{group}/{scen}__*/log_details.json")
    assert len(p) == 1, (variant, scen, p)
    d = json.load(open(p[0]))
    traj = np.asarray(d["trajectory"], float)
    obs = np.asarray([e["dynamic_obstacles_pos"] for e in d["log_entries"]], float)
    T = min(len(traj) - 1, len(obs))
    return d, traj[1:T + 1], obs[:T]          # paired as encounter_report does


def cpa(traj, obs, j=0):
    dist = np.hypot(obs[:, j, 0] - traj[:, 0], obs[:, j, 1] - traj[:, 1])
    return int(np.argmin(dist))


# ---------------------------------------------------------------- fig 1: sectors
def fig_sectors():
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    R = 1.0
    # ego heads up (+y). Bearings counter-clockwise from the bow; starboard = right.
    def wedge(a0, a1, color, alpha=0.22):
        # convert bow-relative bearing (deg, CCW, 0 = up) to matplotlib angle (deg, CCW, 0 = +x)
        ax.add_patch(Wedge((0, 0), R, 90 + a0, 90 + a1, facecolor=color, alpha=alpha, edgecolor=SURF, lw=2))
    wedge(-15, 15, BLUE)                 # head-on
    wedge(-112.5, -15, ORANGE)           # starboard: crossing give-way
    wedge(15, 112.5, AQUA)               # port: crossing stand-on
    wedge(112.5, 247.5, YELLOW)          # stern sector: overtaken
    ax.add_patch(FancyArrow(0, -0.18, 0, 0.36, width=0.06, head_width=0.16, head_length=0.12,
                            length_includes_head=True, color=INK))
    lab = dict(ha="center", va="center", fontsize=10)
    ax.text(0, 0.80, "head-on\n|β| ≤ 15°\n(and |α| ≤ 15°)\nR14 · give way", **lab)
    ax.text(0.58, 0.18, "crossing,\ngive-way\n−112.5° < β < −15°\nR15", **lab)
    ax.text(-0.58, 0.18, "crossing,\nstand-on\n15° < β < 112.5°\nR17", **lab)
    ax.text(0, -0.68, "overtaken (target abaft the beam)\n|β| > 112.5° · R13/R17 · stand on", **lab)
    ax.text(0, -0.30, "ego", ha="center", va="top", fontsize=9, color=INK2)
    ax.text(1.02, 0.0, "starboard", ha="left", va="center", color=INK2, fontsize=9, rotation=-90)
    ax.text(-1.02, 0.0, "port", ha="right", va="center", color=INK2, fontsize=9, rotation=90)
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Where the target sits off the ego's bow (β) decides the role\n"
                 "overtaking is decided by α instead: the ego abaft the TARGET's beam", fontsize=10)
    fig.savefig(OUT / "fig1_sectors.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- fig 2: before/after
def draw_case(ax, scen, variants, styles, title, window=14.0):
    centre = None
    for v in variants:
        d, tr, ob = load(v, "encounters", scen)
        k = cpa(tr, ob)
        c, ls, lbl = styles[v]
        ax.plot(tr[:, 0], tr[:, 1], color=c, ls=ls, label=lbl, zorder=3)
        ax.plot([tr[k, 0], ob[k, 0, 0]], [tr[k, 1], ob[k, 0, 1]], color=c, lw=1, ls=":", zorder=2)
        ax.plot(tr[k, 0], tr[k, 1], "o", ms=8, color=c, mec=SURF, mew=2, zorder=4)
        ax.plot(ob[k, 0, 0], ob[k, 0, 1], "s", ms=8, color=c, mec=SURF, mew=2, zorder=4)
        if centre is None:
            centre = ob[k, 0].copy()
            ax.plot(ob[:, 0, 0], ob[:, 0, 1], color=MUTED, lw=1.5, ls="--", zorder=1)
            i = max(k - 120, 1)
            ax.annotate("", xy=ob[i + 20, 0], xytext=ob[i, 0],
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5))
    ax.set_xlim(centre[0] - window, centre[0] + window)
    ax.set_ylim(centre[1] - window, centre[1] + window)
    ax.set_aspect("equal"); ax.grid(color=GRID, lw=0.8); ax.set_title(title)
    ax.text(0.02, 0.02, "ego sails south (↓)\ndashed grey = target track", transform=ax.transAxes,
            fontsize=8, color=INK2, va="bottom")


def fig_before_after():
    styles = {"v21.12": (ORANGE, "--", "v21.12 (no COLREGs)"), "v21.13": (BLUE, "-", "v21.13")}
    cases = [("owenc215_head_on", "head-on (R14)"),
             ("owenc215_cross_stbd_045", "starboard crossing 045 (R15)"),
             ("owenc215_cross_stbd_090", "starboard crossing 090 (R15)"),
             ("owenc215_cross_stbd_135", "starboard crossing 135 (R15)")]
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for ax, (s, t) in zip(axs.flat, cases):
        draw_case(ax, s, ["v21.12", "v21.13"], styles, t)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Closest approach: ● ego, ■ target, dotted line between them", y=0.955, fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig2_before_after.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------- fig 3: side_only blind spot
def fig_port_turn():
    styles = {"v21.12": (ORANGE, "--", "v21.12"),
              "side_only": (AQUA, "-.", "side term only"),
              "v21.13": (BLUE, "-", "v21.13 (+ no port turn)")}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    draw_case(ax, "owenc215_cross_stbd_045", ["v21.12", "side_only", "v21.13"], styles,
              "cross_stbd_045: where each variant passed", window=12)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    for v in ["side_only", "v21.13"]:
        d, tr, ob = load(v, "encounters", "owenc215_cross_stbd_045")
        t = (np.arange(len(tr)) + 1) * float(d["config"]["dt"])
        c, ls, lbl = styles[v]
        bx.plot(t, np.degrees(tr[:, 2]), color=c, ls=ls, label=lbl)
        role_on = [i for i, e in enumerate(d["log_entries"][:len(tr)]) if e["colregs"]["roles"][0] != "none"]
        if v == "v21.13" and role_on:
            bx.axvspan(t[role_on[0]], t[role_on[-1]], color=GRID, zorder=0)
            bx.text(t[role_on[0]] + 1, -16, "give-way role locked", fontsize=8, color=INK2, va="top")
    bx.axhline(-90, color=MUTED, lw=1, ls=":")
    bx.text(41, -87, "original course (−90°)", fontsize=8, color=INK2)
    bx.set_xlim(40, 140); bx.set_xlabel("time [s]"); bx.set_ylabel("ego heading [deg]  (up = port, down = starboard)")
    bx.grid(color=GRID, lw=0.8); bx.legend(frameon=False, fontsize=8, loc="lower left")
    bx.set_title("heading: side-only turns to PORT, v21.13 to starboard")
    fig.tight_layout(); fig.savefig(OUT / "fig3_port_turn.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------- fig 4: release hysteresis
def fig_release():
    styles = {"port_turn": (ORANGE, "--", "release at 3.0 (port_turn)"),
              "v21.13": (BLUE, "-", "release at 6.0 (v21.13)")}
    fig, axs = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    ax, rx = axs
    role_y = {"none": 0, "crossing_give_way": 1, "crossing_stand_on": 2, "overtaking": 3}
    for n, v in enumerate(["port_turn", "v21.13"]):
        d, tr, ob = load(v, "combined", "v3crowd6_v2111")
        t = (np.arange(len(tr)) + 1) * float(d["config"]["dt"])
        rng = np.hypot(ob[:, 3, 0] - tr[:, 0], ob[:, 3, 1] - tr[:, 1])
        c, ls, lbl = styles[v]
        ax.plot(t, rng, color=c, ls=ls, label=lbl)
        k = int(np.argmin(rng)); ax.plot(t[k], rng[k], "o", ms=8, color=c, mec=SURF, mew=2)
        roles = [e["colregs"]["roles"][3] for e in d["log_entries"][:len(tr)]]
        rx.plot(t, [role_y.get(r, 0) + (n - 0.5) * 0.18 for r in roles], color=c, ls=ls, lw=2)
    for y, s in [(3.0, "risk gate 3.0"), (6.0, "release 6.0")]:
        ax.axhline(y, color=MUTED, lw=1, ls=":"); ax.text(42, y + 0.25, s, fontsize=8, color=INK2)
    ax.set_ylim(0, 12); ax.set_ylabel("range to target t3"); ax.grid(color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="upper center"); ax.set_title("v3crowd6, target t3: identical runs until the lock releases at t = 93 s")
    rx.set_yticks(list(role_y.values())); rx.set_yticklabels(list(role_y.keys()), fontsize=8)
    rx.set_xlabel("time [s]"); rx.grid(color=GRID, lw=0.8); rx.set_xlim(40, 160)
    fig.tight_layout(); fig.savefig(OUT / "fig4_release.png", dpi=130); plt.close(fig)


fig_sectors(); fig_before_after(); fig_port_turn(); fig_release()
print("written:", sorted(p.name for p in OUT.glob("*.png")))
