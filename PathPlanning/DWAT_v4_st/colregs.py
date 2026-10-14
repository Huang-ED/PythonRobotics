"""Basic COLREGs for the DWA cost: who gives way, and on which side.

Three parts, all off unless ``colregs_mode`` is set (default ``"off"``
reproduces every existing run): one classification per control cycle, then two
costs per candidate. ``port_turn_cost`` is further gated by its own gain,
``colregs_port_turn_gain`` (default 0).

1. ``classify`` -- once per control cycle, before any candidate is scored.
   Each target that poses a risk of collision (closest point of approach
   inside ``colregs_dcpa_gate`` and ``colregs_tcpa_gate``, both vessels on
   their present velocities) gets a role from two bearings: the target's off
   the ego's bow, and the ego's off the target's bow.

       head_on             both within +-colregs_head_on_deg of the other's bow   R14
       overtaking          ego abaft the target's beam (> 112.5 deg)              R13, ego gives way
       overtaken           target abaft the ego's beam                            R13/R17, ego stands on
       crossing_give_way   target on the ego's starboard side                     R15
       crossing_stand_on   target on the ego's port side                          R17

   The role is LOCKED until the target is opening and further than
   ``colregs_release_dist``. Without the lock the ego's own starboard turn
   moves the bearing, a head-on becomes a port-side crossing, the rule
   switches off, and the ego turns back.

2. ``side_cost`` -- per candidate, only for head_on and crossing_give_way.
   At the candidate's own closest approach to the target, the target should be
   on the ego's PORT side: a starboard alteration in a head-on (port-to-port),
   passing astern of a crosser from starboard. Both rules reduce to that one
   condition. The penalty is

       gain * sum_i  sigmoid(-lateral_i / softness) * exp(-(d_i / side_dist)^2)

   smooth in the candidate, so a symmetric encounter is broken towards
   starboard instead of chattering (OPEN_PROBLEMS item 14), and fading with
   passing distance, so a wrong-side pass far off costs nothing.

3. ``port_turn_cost`` -- per candidate, same targets. The fade in (2) has a
   blind spot: a candidate holds its turn for the whole horizon, so a port
   turn is predicted to curl far away from the target and its wrong-side pass
   costs almost nothing -- although the ego will straighten up within seconds
   and cut across the target's bow (measured: owenc215_cross_stbd_045). Rules
   14, 15 and 17(c) say it directly: no alteration to port. So

       port_turn_gain * sum_i  max(0, wrap(yaw(t_eval) - yaw_ref_i))

   where yaw_ref_i is the ego's heading when target i's role was locked and
   t_eval is ``colregs_turn_eval_time`` into the candidate. Starboard
   alterations, holding course and slowing down cost nothing.

Not modelled: what a stand-on vessel should do (R17 -- the cost does not
forbid it from manoeuvring), R13's side (it names none), restricted
visibility, traffic separation, narrow channels.
"""

import math
import weakref

import numpy as np

HEAD_ON = "head_on"
OVERTAKING = "overtaking"
OVERTAKEN = "overtaken"
CROSSING_GIVE_WAY = "crossing_give_way"
CROSSING_STAND_ON = "crossing_stand_on"
NONE = "none"

SIDE_RULE_ROLES = frozenset({HEAD_ON, CROSSING_GIVE_WAY})

_STERN_SECTOR = math.radians(112.5)

# Per-run state, keyed by the run's Config: {obstacle index: (locked role,
# ego heading when it locked)}, plus what the last cycle decided, for the log.
# Obstacle indices are positions in map_manager.dynamic_obstacles, which is
# fixed for the length of a run.
_STATE = weakref.WeakKeyDictionary()


def enabled(config):
    return str(getattr(config, "colregs_mode", "off")) != "off"


def _state(config):
    s = _STATE.get(config)
    if s is None:
        s = {"locked": {}, "roles": [], "ref_yaw": [], "best_side": 0.0, "best_port_turn": 0.0}
        _STATE[config] = s
    return s


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _tcpa_dcpa(r, w):
    """Time and distance of closest approach for relative position r, velocity w."""
    ww = float(w @ w)
    if ww < 1e-12:
        return -math.inf, float(np.hypot(*r))
    tcpa = -float(r @ w) / ww
    return tcpa, float(np.hypot(*(r + w * tcpa)))


def _role(beta, alpha, head_on):
    if abs(beta) <= head_on and abs(alpha) <= head_on:
        return HEAD_ON
    if abs(alpha) > _STERN_SECTOR:
        return OVERTAKING
    if abs(beta) > _STERN_SECTOR:
        return OVERTAKEN
    # Bearings are counter-clockwise (yaw convention), so starboard is negative.
    return CROSSING_GIVE_WAY if beta < 0.0 else CROSSING_STAND_ON


def classify(x, ob_pos, ob_vel, config):
    """Role of every dynamic obstacle, in map_manager order. Updates the locks."""
    st = _state(config)
    locked = st["locked"]
    n = 0 if ob_pos is None else len(ob_pos)

    p = np.asarray(x[:2], dtype=float)
    yaw, v = float(x[2]), float(x[3])
    v_ego = v * np.array([math.cos(yaw), math.sin(yaw)])

    dcpa_gate = float(config.colregs_dcpa_gate)
    tcpa_gate = config.colregs_tcpa_gate
    tcpa_gate = float(config.predict_time_obstacle if tcpa_gate is None else tcpa_gate)
    release = config.colregs_release_dist
    release = dcpa_gate if release is None else float(release)
    head_on = math.radians(float(config.colregs_head_on_deg))
    min_speed = float(config.colregs_min_target_speed)

    roles, ref_yaw = [], []
    for i in range(n):
        q = np.asarray(ob_pos[i], dtype=float)
        u = np.asarray(ob_vel[i], dtype=float)
        r = q - p
        w = u - v_ego
        rng = float(np.hypot(*r))

        if i in locked:
            if float(r @ w) > 0.0 and rng > release:
                del locked[i]
            else:
                roles.append(locked[i][0])
                ref_yaw.append(locked[i][1])
                continue

        if float(np.hypot(*u)) < min_speed:
            roles.append(NONE)
            ref_yaw.append(yaw)
            continue
        tcpa, dcpa = _tcpa_dcpa(r, w)
        if not (0.0 < tcpa <= tcpa_gate and dcpa <= dcpa_gate):
            roles.append(NONE)
            ref_yaw.append(yaw)
            continue

        beta = _wrap(math.atan2(r[1], r[0]) - yaw)
        alpha = _wrap(math.atan2(-r[1], -r[0]) - math.atan2(u[1], u[0]))
        role = _role(beta, alpha, head_on)
        locked[i] = (role, yaw)
        roles.append(role)
        ref_yaw.append(yaw)

    st["roles"] = roles
    st["ref_yaw"] = ref_yaw
    return roles


def ref_yaw(config):
    """Heading each target's role locked at, in the order classify returned."""
    return np.asarray(_state(config)["ref_yaw"], dtype=float)


def side_cost(trajectory, ob_pos, ob_vel, config):
    """Wrong-side penalty of one candidate against the side-rule targets given."""
    if ob_pos is None or len(ob_pos) == 0:
        return 0.0
    times = np.arange(trajectory.shape[0]) * config.dt
    pred = ob_pos[:, np.newaxis, :] + ob_vel[:, np.newaxis, :] * times[np.newaxis, :, np.newaxis]
    d = pred - trajectory[np.newaxis, :, 0:2]                     # (N, T, 2)
    dist = np.hypot(d[:, :, 0], d[:, :, 1])
    k = np.argmin(dist, axis=1)                                   # each target's CPA step
    idx = np.arange(len(ob_pos))
    yaw = trajectory[k, 2]
    dx, dy = d[idx, k, 0], d[idx, k, 1]
    lateral = -np.sin(yaw) * dx + np.cos(yaw) * dy                # > 0: target to port
    soft = max(float(config.colregs_side_softness), 1e-9)
    wrong = 0.5 * (1.0 - np.tanh(lateral / (2.0 * soft)))         # sigmoid(-lateral / soft)
    near = np.exp(-(dist[idx, k] / float(config.colregs_side_dist)) ** 2)
    return float(config.colregs_side_cost_gain * np.sum(wrong * near))


def port_turn_cost(trajectory, ref, config):
    """Port alteration of one candidate past each side-rule target's reference heading."""
    gain = float(config.colregs_port_turn_gain)
    if gain == 0.0 or ref is None or len(ref) == 0:
        return 0.0
    k = min(int(round(float(config.colregs_turn_eval_time) / config.dt)), trajectory.shape[0] - 1)
    dyaw = (trajectory[k, 2] - ref + np.pi) % (2.0 * np.pi) - np.pi   # > 0: to port of ref
    return float(gain * np.sum(np.maximum(dyaw, 0.0)))


def record_best_cost(config, side, port_turn):
    """The chosen candidate's two COLREGs cost terms, for the log."""
    st = _state(config)
    st["best_side"] = float(side)
    st["best_port_turn"] = float(port_turn)


def log_entry(config):
    """One iteration's COLREGs record.

    Logs written before 2026-10-14 carry a single ``side_cost`` key instead,
    which despite its name is the TOTAL (side + port turn) of the chosen
    candidate.
    """
    st = _state(config)
    return {"roles": list(st["roles"]),
            "cost_side": st["best_side"],
            "cost_port_turn": st["best_port_turn"],
            "cost_total": st["best_side"] + st["best_port_turn"]}
