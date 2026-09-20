"""Render a run's animation straight from log_details.json -- no re-planning.

Draws the dist_localgoal corridor, which is what the item-15 failures cross.
"""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, FFMpegWriter


def render(run_dir, out_path, stride=5, fps=25, title=None):
    d = json.load(open(os.path.join(run_dir, "log_details.json")))
    cfg = d["config"]
    traj = np.asarray(d["trajectory"], float)
    obs = np.asarray([e["dynamic_obstacles_pos"] for e in d["log_entries"]], float)
    lg = np.asarray([[e["local_goal_x"], e["local_goal_y"]] for e in d["log_entries"]], float)
    T = min(len(traj), len(obs))
    traj, obs, lg = traj[:T], obs[:T], lg[:T]

    sc = json.load(open(json.load(open(os.path.join(run_dir, "run.json")))["inputs"]["scenario"]))
    start = np.array(sc["start_position"], float)
    goal = np.array(sc["goal_position"], float)
    R = float(cfg["dist_localgoal"])
    half = max(0.5 * (cfg["robot_length"] - cfg["robot_width"]), 0.0)
    er = cfg["robot_width"] / 2.0
    ob_r = float(sc["dynamic_obstacles"][0].get("radius", 0.5))

    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=110)
    ax.set_aspect("equal")
    lo = min(traj[:, 0].min(), obs[:, :, 0].min()) - 6
    hi = max(traj[:, 0].max(), obs[:, :, 0].max()) + 6
    ax.set_xlim(lo, hi)
    ax.set_ylim(min(goal[1], traj[:, 1].min()) - 6, max(start[1], traj[:, 1].max()) + 6)

    # reference leg and the dist_localgoal corridor it must stay inside
    ax.plot([start[0], goal[0]], [start[1], goal[1]], "--", c="0.6", lw=1, zorder=1)
    for s in (-1, 1):
        ax.plot([start[0] + s * R, goal[0] + s * R], [start[1], goal[1]],
                ":", c="crimson", lw=1.2, zorder=1)
    ax.text(goal[0] + R, start[1] + 2, f" dist_localgoal = {R:g}", color="crimson", fontsize=8)
    ax.plot(*goal, "*", c="darkgreen", ms=15, zorder=3)

    trail, = ax.plot([], [], "-", c="tab:blue", lw=1.4, zorder=4)
    ego_body, = ax.plot([], [], "-", c="k", lw=2.2, zorder=6)
    ego_nose = Circle((0, 0), er, fc="none", ec="tab:blue", lw=1.0, zorder=5); ax.add_patch(ego_nose)
    ego_tail = Circle((0, 0), er, fc="none", ec="tab:blue", lw=1.0, zorder=5); ax.add_patch(ego_tail)
    lgpt, = ax.plot([], [], "x", c="darkgreen", ms=7, mew=1.6, zorder=6)
    tg = [Circle((0, 0), ob_r, fc="tab:red", alpha=.45, ec="darkred", zorder=5) for _ in range(obs.shape[1])]
    for c in tg: ax.add_patch(c)
    tgtrail = [ax.plot([], [], "-", c="tab:red", lw=.8, alpha=.5, zorder=3)[0] for _ in range(obs.shape[1])]
    txt = ax.text(.02, .98, "", transform=ax.transAxes, va="top", fontsize=9,
                  family="monospace", bbox=dict(fc="white", alpha=.75, ec="0.8"))
    ax.set_title(title or os.path.basename(run_dir), fontsize=10)
    ax.grid(alpha=.25)

    idx = list(range(0, T, stride)) + [T - 1]

    def upd(k):
        x, y, yaw = traj[k, 0], traj[k, 1], traj[k, 2]
        h = np.array([math.cos(yaw), math.sin(yaw)])
        a, b = np.array([x, y]) - half * h, np.array([x, y]) + half * h
        ego_body.set_data([a[0], b[0]], [a[1], b[1]])
        ego_nose.center = (b[0], b[1]); ego_tail.center = (a[0], a[1])
        trail.set_data(traj[:k + 1, 0], traj[:k + 1, 1])
        lgpt.set_data([lg[k, 0]], [lg[k, 1]])
        dmin = 9e9
        for j, c in enumerate(tg):
            c.center = (obs[k, j, 0], obs[k, j, 1])
            tgtrail[j].set_data(obs[:k + 1, j, 0], obs[:k + 1, j, 1])
            to = obs[k, j] - np.array([x, y])
            p = np.clip(to @ h, -half, half)
            dmin = min(dmin, float(np.hypot(*(to - p * h))) - ob_r - er)
        dev = abs(x - start[0])
        flag = "  <-- OUTSIDE CORRIDOR" if dev > R else ""
        txt.set_text(f"t = {k*0.1:6.1f} s\nclearance = {dmin:7.3f}\ncross-track = {dev:6.3f}{flag}")
        return ()

    FuncAnimation(fig, upd, frames=idx, interval=1000 / fps, blit=False).save(
        out_path, writer=FFMpegWriter(fps=fps, bitrate=2400))
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    print(render(sys.argv[1], sys.argv[2], title=sys.argv[3] if len(sys.argv) > 3 else None))
