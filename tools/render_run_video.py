"""Render a run's animation straight from log_details.json -- no re-planning.

Draws the dist_localgoal corridor, which is what the item-15 failures cross.

NAMING: output belongs beside the run as ``video_from_log.mp4``, never
``video.mp4``. The two are not interchangeable and the difference matters when
reading an old run:

    video.mp4            assembled from frames/ that the RUN itself wrote, so
                         it carries whatever the planner drew live -- the
                         candidate fan, the global-planner search.
    video_from_log.mp4   reconstructed afterwards from log_details.json. Ego,
                         targets, local goal and the predicted obstacle tracks
                         are exact; the candidate fan appears ONLY if the run
                         set ``log_candidates_every``, and the planner search
                         never does.

DEFAULT_OUTPUT_NAME below is the one to use when writing into a run directory.
"""

DEFAULT_OUTPUT_NAME = "video_from_log.mp4"
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
    # trajectory carries the initial state too, so it is one longer than
    # log_entries; clamp to whichever is shortest or the last frame overruns.
    T = min(len(traj), len(d["log_entries"]))
    if obs.size:
        T = min(T, len(obs))
    traj, lg = traj[:T], lg[:T]
    # An obstacle-free run logs an empty list per step, giving shape (T, 0) or
    # (T,) instead of (T, N, 2). Normalise so every consumer can index [:, :, k].
    obs = obs[:T].reshape(T, -1, 2) if obs.size else np.zeros((T, 0, 2))

    sc_path = json.load(open(os.path.join(run_dir, "run.json")))["inputs"]["scenario"]
    sc = json.load(open(sc_path))
    start = np.array(sc["start_position"], float)
    goal = np.array(sc["goal_position"], float)

    # Land, and the actual global path. Neither is in log_details.json, so
    # rebuild the map context from the scenario the run recorded. Without this
    # a land route renders as open water -- the v3crowd7 run that BEACHED on a
    # static obstacle showed nothing to beach on -- and the look-ahead corridor
    # gets drawn about a straight start->goal line that is not the path.
    static_xy, road_map = None, None
    try:
        import importlib
        _mc = importlib.import_module("PathPlanning.DWAT_v5_modular.map_creation")
        _main = importlib.import_module("PathPlanning.DWAT_v5_modular.main")
        _cfgd = dict(cfg)
        _mod = importlib.import_module(
            _cfgd.pop("dwa_module", "PathPlanning.DWAT_v4_st.dwa_average_weighted_side"))
        _c = _main.apply_algorithm_overrides(_mod.Config(), _cfgd)
        _ctx = _mc.create_map_context(_c, sc_path)
        static_xy = np.asarray(_ctx.map_manager.get_static_obstacles(), float)
        road_map = np.asarray(_ctx.road_map, float)
    except Exception as exc:
        print(f"  [warn] map not rebuilt, rendering without land: {exc}")
    R = float(cfg["dist_localgoal"])
    half = max(0.5 * (cfg["robot_length"] - cfg["robot_width"]), 0.0)
    er = cfg["robot_width"] / 2.0
    # An obstacle-free reference run has no targets at all, and those are
    # exactly the runs worth rendering to see PATH TRACKING on its own.
    _dyn = sc.get("dynamic_obstacles") or []
    ob_r = float(_dyn[0].get("radius", 0.5)) if _dyn else 0.5

    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=110)
    ax.set_aspect("equal")
    _xs = [traj[:, 0]] + ([obs[:, :, 0].ravel()] if obs.shape[1] else [])
    lo = min(a.min() for a in _xs) - 6
    hi = max(a.max() for a in _xs) + 6
    ax.set_xlim(lo, hi)
    _ys = [traj[:, 1]] + ([obs[:, :, 1].ravel()] if obs.shape[1] else [])
    ax.set_ylim(min(goal[1], min(a.min() for a in _ys)) - 6,
                max(start[1], max(a.max() for a in _ys)) + 6)

    # Land first, so everything else draws over it.
    if static_xy is not None and len(static_xy):
        ax.plot(static_xy[:, 0], static_xy[:, 1], "s", c="0.45", ms=2.0,
                mec="none", zorder=0, label="land")

    # The global path the follower is tracking -- the real Theta* road map when
    # it could be rebuilt, else the straight start->goal line as a fallback.
    if road_map is not None and len(road_map) >= 2:
        ax.plot(road_map[:, 0], road_map[:, 1], "--", c="0.35", lw=1.2, zorder=1)
        ax.plot(road_map[:, 0], road_map[:, 1], ".", c="0.35", ms=4, zorder=1)
    else:
        ax.plot([start[0], goal[0]], [start[1], goal[1]], "--", c="0.6", lw=1, zorder=1)
    ax.text(0.02, 0.02, f"dist_localgoal = {R:g}", transform=ax.transAxes,
            color="crimson", fontsize=8)
    ax.plot(*goal, "*", c="darkgreen", ms=15, zorder=3)

    # Candidate fan (only present when the run set log_candidates_every) and
    # the predicted obstacle tracks, which reconstruct from the logged
    # positions as pos + vel*t -- no extra logging needed for those.
    cand_lines = []
    n_cand_max = max((len(e.get("candidates") or []) for e in d["log_entries"]), default=0)
    for _ in range(n_cand_max):
        cand_lines.append(ax.plot([], [], "-", c="deepskyblue", lw=0.8, alpha=0.25, zorder=2)[0])
    ob_vel = np.zeros((obs.shape[1], 2))
    if len(obs) > 1 and obs.shape[1]:
        ob_vel = (obs[1] - obs[0]) / float(cfg["dt"])
    pred_t = np.arange(0.0, float(cfg["predict_time_obstacle"]) + float(cfg["dt"]), float(cfg["dt"]))
    pred_lines = [ax.plot([], [], "--", c="peru", lw=1.0, alpha=0.8, zorder=3)[0]
                  for _ in range(obs.shape[1])]

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

    def _cross_track(p):
        """Distance to the global path -- the polyline, not a straight line.

        |x - 50| was hardcoded for the open-water route and read 21.74 on every
        v3crowd run, whose path is (70,90) -> (60,10). Identical values across
        six different runs was the tell.
        """
        if road_map is None or len(road_map) < 2:
            return abs(p[0] - start[0])
        a, b = road_map[:-1], road_map[1:]
        ab = b - a
        L2 = np.sum(ab * ab, axis=1)
        L2 = np.where(L2 < 1e-12, 1e-12, L2)
        t = np.clip(np.sum((p - a) * ab, axis=1) / L2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        return float(np.min(np.hypot(*(p - proj).T)))

    idx = list(range(0, T, stride)) + [T - 1]

    def upd(k):
        x, y, yaw = traj[k, 0], traj[k, 1], traj[k, 2]
        h = np.array([math.cos(yaw), math.sin(yaw)])
        a, b = np.array([x, y]) - half * h, np.array([x, y]) + half * h
        ego_body.set_data([a[0], b[0]], [a[1], b[1]])
        ego_nose.center = (b[0], b[1]); ego_tail.center = (a[0], a[1])
        trail.set_data(traj[:k + 1, 0], traj[:k + 1, 1])
        cands = d["log_entries"][k].get("candidates") or []
        for li, line in enumerate(cand_lines):
            if li < len(cands):
                ct = np.asarray(cands[li], float)
                line.set_data(ct[:, 0], ct[:, 1])
            else:
                line.set_data([], [])
        for j, line in enumerate(pred_lines):
            line.set_data(obs[k, j, 0] + ob_vel[j, 0] * pred_t,
                          obs[k, j, 1] + ob_vel[j, 1] * pred_t)
        lgpt.set_data([lg[k, 0]], [lg[k, 1]])
        dmin = float("nan")
        for j, c in enumerate(tg):
            c.center = (obs[k, j, 0], obs[k, j, 1])
            tgtrail[j].set_data(obs[:k + 1, j, 0], obs[:k + 1, j, 1])
            to = obs[k, j] - np.array([x, y])
            p = np.clip(to @ h, -half, half)
            _d = float(np.hypot(*(to - p * h))) - ob_r - er
            dmin = _d if (dmin != dmin) else min(dmin, _d)
        dev = _cross_track(np.array([x, y]))
        flag = "  <-- BEYOND dist_localgoal" if dev > R else ""
        _clr = "     n/a" if dmin != dmin else f"{dmin:7.3f}"
        txt.set_text(f"t = {k*0.1:6.1f} s\nclearance  = {_clr}\ncross-track = {dev:6.3f}{flag}")
        return ()

    FuncAnimation(fig, upd, frames=idx, interval=1000 / fps, blit=False).save(
        out_path, writer=FFMpegWriter(fps=fps, bitrate=2400))
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    run_dir = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(run_dir, DEFAULT_OUTPUT_NAME)
    if os.path.isdir(out):
        out = os.path.join(out, DEFAULT_OUTPUT_NAME)
    if os.path.basename(out) == "video.mp4":
        raise SystemExit(
            "refusing to write video.mp4: that name is reserved for a video "
            "assembled from the run's own frames/. Use " + DEFAULT_OUTPUT_NAME)
    print(render(run_dir, out, title=sys.argv[3] if len(sys.argv) > 3 else None))
