# How much does the ego oscillate, and what sets it?

The ego does not follow its global path; it swings about it. On the rescaled
v3crowd routes that swing reaches **7 u against a `dist_localgoal` of 7.0**, and
the run dies — not from a collision, but from losing the path. This document is
the parameter analysis: what governs the oscillation, how each config key enters
it, what level is tolerable, and how to set them.

Companion to `PREDICTION_HORIZON.md`, same shape: the claim, why the obvious
framing is wrong, the mathematics with its bounds, the evidence, a rule, and
what it does not settle.

---

## 1. The claim

> **The oscillation has two separate regimes and they need separate rules.**
>
> **Small-signal**, near the path, is governed by **one dimensionless number**:
> ```
>     mu = (dist_localgoal/max_speed + predict_time_to_goal) / T_yaw
>     T_yaw = max_yaw_rate / max_delta_yaw_rate
> ```
> `mu > 1` is a hard stability boundary. Raising `mu` improves **damping**.
>
> **Corrected 2026-09-23.** This document first wrote the lag as
> `T_act = max(max_speed/max_accel, T_yaw)`, carried over from
> `PREDICTION_HORIZON.md`. That is wrong here: oscillation is a *steering*
> phenomenon and surge acceleration belongs to the speed loop. An 11-config
> screen separates them decisively -- see section 4.4. The earlier evidence
> could not: only `v20.7.8-2` distinguishes the two forms, and it is stable
> under both.
>
> **Large-signal**, acquiring the track from a heading error `psi0`, is not
> governed by `mu` at all. It is a rate-limited manoeuvre whose excursion is
> ```
>     e_acq = INT v(t) sin(psi(t)) dt     over the slew
> ```
> and no amount of look-ahead changes it.

Confusing the two is how three of this investigation's wrong turns happened.

> **The two regimes are now separated by experiment, not by argument.** Varying
> `dist_localgoal` alone from 7 to 18 -- a 2.4x change in `mu` at constant
> `t_acq` -- moves the peak excursion by **0.05%** while helm saturation falls
> monotonically from 0.776 to 0.490. Section 4.4.

---

## 2. Why *not* "path tracking"

Framing this as a tracking problem invites the wrong fixes — a better follower,
a restoring term, a longer look-ahead. Two measurements kill that framing:

- **Starting aligned, the ego tracks the path exactly.** Cross-track is
  **0.00** for the whole run. There is nothing wrong with the follower.
- **`dist_localgoal` 7 -> 12 changes the launch excursion by nothing** (3.24 u
  either way), while changing the *decaying* part a lot.

So it is not "tracking quality". It is a transient response with two regimes,
and the parameters that matter are the ship's own time constants.

---

## 3. The mathematics

### 3.1 Small-signal: one number governs everything

Pure pursuit on a straight line, for small cross-track `e` and heading error
`psi`, with look-ahead `L_d`:

```
    e_dot   = v * psi
    psi_dot = v * kappa,    kappa = 2 sin(alpha)/L_d,    alpha ~ psi + e/L_d

 => e_ddot + (2v/L_d) e_dot + (2 v^2/L_d^2) e = 0
    omega_n = sqrt(2) v / L_d        zeta = 1/sqrt(2) = 0.707
```

**Ideal pure pursuit is always well damped, at any speed.** So the oscillation
cannot come from the geometry. It comes from what the ideal model leaves out:
the yaw rate cannot change instantly. Inserting a first-order lag `tau = T_yaw`:

```
    T_yaw s^3 + s^2 + (2v/L_d) s + 2 v^2/L_d^2 = 0
```

Substituting `s = sigma/T_yaw` removes every dimensional quantity:

```
    sigma^3 + sigma^2 + (2/mu) sigma + 2/mu^2 = 0,     mu = L_d/(v * T_yaw)
```

**Speed, look-ahead and actuation lag enter only through their ratio.** That is
why raising `max_delta_yaw_rate` by 2.2x and raising `dist_localgoal` from 7 to
20 were interchangeable fixes for the same divergence — both raise `mu`.

Routh-Hurwitz on `sigma^3 + sigma^2 + a sigma + b` requires `a > b`:

```
    2/mu > 2/mu^2       =>       mu > 1       i.e.   dist_localgoal > v * T_yaw
```

**The goal rollout adds lead, so it belongs in `mu`.** `calc_to_goal_cost`
scores the heading at the *end* of a `predict_time_to_goal` rollout, which is
`v * T_goal` of preview on top of the look-ahead:

```
    mu = (L_d/v + T_goal) / T_yaw
```

This is what separates `v21.5` from `v21.6`: identical ships and identical
`dist_localgoal`, differing only in `T_goal` (8.0 vs 2.0), giving `mu` = 1.10
and 0.80 -- one side of the stability boundary each. `v21.5` grazed but did not
ring; `v21.6` diverged at x1.49. Plain `L_d/(v*T_yaw)` = 0.70 for both and
cannot tell them apart.

Solving for the dominant pole gives the design curve:

| `mu` | `zeta` | overshoot | decay/cycle | `t_settle`/`T_yaw` | `T_osc`/`T_yaw` |
|---|---|---|---|---|---|
| 0.70 | −0.076 | — | **grows** | — | — |
| 1.00 | 0.000 | 100% | ×1.00 | inf | 4.4 |
| 1.54 | 0.115 | 69% | ×0.48 | 32.9 | 6.0 |
| 2.00 | 0.200 | 53% | ×0.28 | 22.7 | 7.3 |
| 3.00 | 0.362 | 30% | ×0.09 | 17.1 | 10.4 |
| **4.00** | **0.500** | **16%** | ×0.03 | **16.0** | 14.5 |
| 5.60 | 0.643 | 7% | ×0.01 | 18.6 | 24.5 |
| → inf | 0.707 | 4% | — | — | — |

**`t_settle` has a minimum near `mu ~ 4`**, not a monotone improvement: below it
the loop is under-damped, above it the period lengthens faster than the damping
improves. This is the single most useful line in the document.

### 3.2 Large-signal: acquisition is rate-limited, and `mu` does not enter

Starting with heading error `psi0` the ship cannot use the linear law — the yaw
rate saturates. With `omega(t) = min(alpha t, omega_max)`:

```
    t_acq = sqrt(2 psi0/alpha)              if psi0 < omega_max*T_yaw/2
            T_yaw/2 + psi0/omega_max        otherwise

    e_acq = INT v(t) sin(psi(t)) dt         over 0..t_acq
```

`L_d` appears nowhere. **The acquisition excursion is a property of the ship,
not of the controller** — which is exactly why tripling `dist_localgoal` left
it at 3.24 u.

### 3.3 Why the gains cannot fix it

A natural first move is to re-weight the cost. The algebra says it will not
work, and the experiment agrees. On a clear path only two terms are live:

```
    J(v,w) = G * dpsi(v,w) + S * (v_max - v_end(v))
```

Two consequences follow immediately:

1. **`argmin` is invariant under a common positive scaling**, so at most ONE
   dimensionless gain can matter: `rho = to_goal_cost_gain / speed_cost_gain`
   (shipped: 0.6). There are not four knobs here; there is one.
2. **The speed term does not contain `w`.** `v_end` depends only on the sampled
   `v`. So

   ```
       dJ/dw = G * d(dpsi)/dw      =>   stationarity:  d(dpsi)/dw = 0
   ```

   **The gain cancels out of the yaw stationarity condition.** Gains can reach
   the yaw decision only by changing which `v` is selected.

Replaying 456 real states of the ringing run and re-minimising: the chosen `w`
is **bit-identical** from `rho` = 0.075 to 20, a 267x range, `max|dw|` =
0.00000. Only `v` moves, and only past `rho ~ 5.

**Where gains do reach the oscillation** is through `e_acq`, which is
proportional to `v`:

| `rho` | mean `v` in the launch turn | predicted `e_acq` scale |
|---|---|---|
| 0.6 *(shipped)* | 0.441 | x1.00 |
| 9.6 | 0.338 | x0.765 |
| 20.0 | 0.292 | x0.663 |

That is a real mechanism, and it must be applied to **`speed_cost_gain`, never
`to_goal_cost_gain`**: `G/B` contains no `S`, so cutting `S` raises `rho` at
zero graze cost, whereas raising `G` to the same `rho` would put `G/B` at 4.76,
worse than the `T_goal = 8.0` that produced the 0.000 graze.

**It fails anyway.** All three runs ran aground on land and none reached the
goal (the collision is recorded in `collision_event`, not in `error`):

| config | `rho` | at the start | ended | where |
|---|---|---|---|---|
| `bal-spd030` | 2.0 | normal launch | **grounded**, t = 194.7 s | land cell (67,27), corner of legs 5/6 |
| `bal-spd006` | 9.6 | **stationary 0-48 s**, pivoting | **grounded**, t = 160.0 s | land cell (46,46), corner of legs 2/3 |
| `bal-spd003` | 20.0 | **stationary 0-67 s**, pivoting | **grounded**, t = 180.8 s | land cell (46,46), same corner |

The launch behaves exactly as the acquisition model says. At `rho` >= 9.6 the
ego turns the whole 99.7 deg at zero speed, so `e_acq ~ v` goes to zero and
there is no launch swing at all -- at the price of standing still for up to a
minute. What ends the runs is different: all three later clip land at a
corner where the global path runs close to it. So the gain change does not
fail by stalling the ship. It fails by grounding.

*Corrected 2026-09-23: first recorded as "stalled", from `reached_goal = False`
with `error = None`, without reading `collision_event`. The videos showed the
groundings.*

**One caveat on the theorem.** It holds only where the cost has just those two
terms. This route has land, and the static term

```
    obstacle_cost_gain * max(0, max_obstacle_cost_dist - dist_static)
```

*is* `w`-dependent and does *not* scale with `S`, so dividing through by `S`
leaves two ratios changing, not one. Measured: the static term is non-zero on
**63%** of steps, and `w` then differs on 2.2% of steps at `rho` = 2.0 and 18.9%
at `rho` = 9.6. That is why `bal-spd030`, predicted bit-identical, diverged.
**Gains are provably yaw-blind only in genuinely clear water.**

### 3.4 What counts as acceptable

Two independent criteria, and both must hold:

1. **The transient must decay before the next encounter.**
   `t_settle < t_meet` of the first target. This is the binding one in traffic.
2. **The excursion must stay inside the corridor.**
   `e_acq < dist_localgoal`, or the look-ahead circle leaves the path and the
   run aborts (`OPEN_PROBLEMS.md` item 15).

Criterion 1 is what the v3crowd runs violate.

A third, purely diagnostic measure turns out to order the configs better than
either: **helm saturation**, the fraction of control steps spent slewing at
`max_delta_yaw_rate`. It is a per-step statistic, so it needs no peaks, no
period and no window -- it is immune to all the artefacts in section 7. The ego
that spends 78% of its steps at the rate limit merely holding a straight line
has little authority left for an encounter.

---

## 4. What the evidence says

### 4.1 The stability boundary predicts every observed divergence

| config | `L_d` | `T_goal` | `T_yaw` | `mu` | predicted | **measured** |
|---|---|---|---|---|---|---|
| v20.7.8-2 | 7.0 | 1.0 | 1.00 | 15.00 | stable | max xte 0.70, no ringing |
| **v21.5** | 7.0 | 8.0 | 20.00 | **1.10** | marginal | grazed, one peak, no ringing |
| **v21.6** | 7.0 | 2.0 | 20.00 | **0.80** | **UNSTABLE** | **diverged, ×1.49 growth** |
| v21.7 preview-20 | 20.0 | 2.0 | 20.00 | 2.10 | stable | settles ×0.37 |
| v21.9 / v21.10 | 7.0 | 2.0 | 9.10 | 1.76 | stable, poorly damped | rings all run, helm sat 0.78 |
| **bal-mix-T33** | 9.0 | 2.0 | 6.74 | 2.97 | stable, damped | settles, helm sat 0.39 |

Restated in `T_yaw` (2026-09-23). `v21.5` and `v21.6` were previously listed as
one row at `mu` = 0.70; they are the same ship with different `T_goal`, and
splitting them is what shows the boundary lying between them.

The v21.6 divergence — visible in the videos as a cross-track that keeps
growing after the pass — sits at `mu = 0.80`, below the Routh-Hurwitz boundary.
It was not a tuning failure. It was linear instability.

Predicted ringing period for v21.6 is `6.0 * T_yaw` = **54.5 s**; measured
between the v21.6 peaks at t=97.6 and 125.1 s: **55 s**.

**This agreement does not transfer.** On the v3 route at `v21.10` the same
formula predicts 54.6 s and the measured ringing period is **~36 s**. The
period column of section 3.1's table should not be trusted outside the
configuration it was fitted on; the amplitude and stability results are what
carried over.

### 4.2 The acquisition integral predicts the excursion

Heading-step sweep, clean open water, no obstacles, v21.10:

| `psi0` | `e_acq` predicted | measured | error |
|---|---|---|---|
| 0° | 0.00 | **0.00** | — |
| 22.5° | 0.22 | 0.23 | −3% |
| 45° | 0.84 | 0.93 | −10% |
| 67.5° | 1.76 | 2.40 | −27% |
| 90° | 2.81 | 2.79 | **+1%** |
| 135° | 4.68 | 4.66 | **0%** |

And on the v3 route (`psi0` = 99.7°): predicted **3.27 u**, measured **3.24 u**.

**Starting aligned gives exactly 0.00.** The follower is not the problem.

### 4.3 Why v3crowd fails

`v21.10` runs at `mu = 1.76`, so `t_settle` = **299 s** — longer than the entire
200 s transit. The v3 scenarios declare no `start_yaw`, so the ego begins 99.7°
off its first leg and injects a 3.24 u transient. The first target meets it at
**t = 38 s**, while the ego is still ringing. Every encounter is handled
mid-transient, the excursions add, and cross-track passes `dist_localgoal`.

Both criteria of §3.3 are violated at once.

### 4.4 The parameter screen, and which model each result belongs to

Eleven configs, one fixed scenario (`v3_baseline_step`: the real six-corner v3
route, clean water, no targets, `start_yaw` omitted so the ego begins 99.7 deg
off its first leg). One excitation, one variable at a time.

**Names.** Every `bal-*` ("balance") config is `v21.10` with one or two keys
changed, in `configs/algorithm/v21.10/`. **"Helm" means `max_delta_yaw_rate`**,
quoted as time to full helm, `T_yaw = max_yaw_rate / max_delta_yaw_rate`:
`bal-helm35` reaches full helm in 3.5 s against `v21.10`'s 9.10 s. `bal-mix`
moves both levers about 30% instead of one lever far: `dist_localgoal` 7 -> 9
and full helm 9.10 -> 6.74 s. `bal-spd*` lower `speed_cost_gain`; `bal-tg5`
raises `predict_time_to_goal`; `bal-v040` lowers `max_speed`.

| config | `dLG` | `T_yaw` | `T_goal` | `v` | `S` | **`xte_max`** | helm sat. | goal |
|---|---|---|---|---|---|---|---|---|
| `bal-helm35` | 7 | **3.50** | 2 | 0.5 | 1.0 | **1.942** | 0.342 | reached |
| `bal-helm52` | 7 | **5.20** | 2 | 0.5 | 1.0 | **2.322** | 0.456 | reached |
| `bal-mix` | **9** | **6.74** | 2 | 0.5 | 1.0 | **2.679** | 0.447 | reached |
| `bal-mix-tg` | 9 | 6.74 | **4** | 0.5 | 1.0 | 2.706 | 0.440 | reached |
| `bal-v040` | 7 | 9.10 | 2 | **0.4** | 1.0 | 2.916 | 0.584 | reached |
| `bal-dlg10` | **10** | 9.10 | 2 | 0.5 | 1.0 | 3.238 | 0.602 | reached |
| `v21.10` | 7 | 9.10 | 2 | 0.5 | 1.0 | 3.238 | 0.776 | reached |
| `bal-tg5` | 7 | 9.10 | **5** | 0.5 | 1.0 | 3.249 | 0.406 | reached |
| `bal-spd030` | 7 | 9.10 | 2 | 0.5 | **0.30** | -- | -- | **grounded** |
| `bal-spd006` | 7 | 9.10 | 2 | 0.5 | **0.0625** | -- | -- | **grounded** |

**The separating experiment.** Vary `dist_localgoal` alone. `t_acq` is untouched
and `mu` changes by 2.4x, so the two regimes make opposite predictions:

```
    dLG        7      10      12      18        mu_yaw 1.76 -> 4.17
    xte_max    3.238   3.238   3.238   3.240    spread 0.0016 u = 0.05%
    helm_sat   0.776   0.602   0.582   0.490    monotone
```

**Both models are confirmed, each about its own regime.** Look-ahead does not
touch the peak -- `L_d` is absent from `e_acq`, exactly as section 3.2 says --
and it does improve damping, exactly as section 3.1 says.

**The acquisition model (section 3.2) predicts the peak to 5.5%.** Using `t_acq = T_yaw + psi0/w_max` and
`e_acq ~ v * t_acq`, calibrated on `v21.10` alone:

| config | `t_acq` | predicted | measured | error |
|---|---|---|---|---|
| `bal-helm35` | 12.69 | 2.246 | 1.942 | -13.5% |
| `bal-helm52` | 14.39 | 2.547 | 2.322 | -8.8% |
| `bal-mix` | 15.93 | 2.820 | 2.679 | -5.0% |
| `bal-v040` | 18.29 | 2.591 | 2.916 | **+12.6%** |
| `bal-dlg10` | 18.29 | 3.238 | 3.238 | **0.0%** |
| `bal-tg5` | 18.29 | 3.238 | 3.249 | +0.3% |

Mean |error| **5.5%**, max 13.5%, n=8. The `max_speed` case is the worst fit:
`e_acq ~ v` over-predicts the benefit of slowing.

**The small-signal model (section 3.1): the published `T_act` does not order the data; `T_yaw` does.**

| `mu` computed with | corr. with helm sat. | corr. with `xte_max` |
|---|---|---|
| `max(T_surge, T_yaw)` *(as first published)* | **-0.057** | +0.051 |
| **`T_yaw` alone** | **-0.736** | **-0.917** |

Oscillation is a steering phenomenon; surge acceleration belongs to the speed
loop. The `max()` was borrowed from `PREDICTION_HORIZON.md`, where it correctly
means "the horizon must cover both actuators" -- using it as a *steering lag*
was a category error.

**Do not read the -0.917 as independent support.** `mu_yaw` is proportional to
`1/T_yaw` and `t_acq` is nearly linear in `T_yaw`, so that correlation largely
restates the acquisition model. The independent evidence for `mu` is the helm-saturation
column of the `dLG` series above, where `t_acq` is pinned by construction.

### 4.5 The encounter suites

`v3crowd5/6/7`, five to seven targets on the six-corner route, run in both
conditions. Clearance is the worst of the three cases, measured by
`adapters/encounter_report.py`: surface to surface against the capsule hull,
so hull length counts. That is the measure behind every committed clearance,
including `v20.7.8-2`'s 0.483.

| config | aligned start | | unaligned start (99.7 deg) | |
|---|---|---|---|---|
| | reach | min clear | reach | min clear |
| `v21.10` | 3/3 | **0.724** | **0/3** | -- |
| `bal-dlg10` | 3/3 | 0.724 | **2/3** | 0.729 |
| `bal-mix` | 3/3 | 0.614 | 3/3 | 0.660 |
| **`bal-mix-T33` = `v21.11`** | 3/3 | **0.649** | 3/3 | **0.692** |
| `bal-helm52` | 3/3 | 0.519 | 3/3 | 0.567 |
| `bal-helm35` | 3/3 | **0.368** | 3/3 | 0.458 |

*Corrected 2026-09-23. This table first carried clearances from a scratch
scorer that measured from the hull centre and ignored its length, reading
0.04-0.11 high (e.g. `v21.10` 0.758, `bal-helm35` 0.414). The ranking is
unchanged; every value above is from the official scorer.*

Three things this says that the single-scenario screen could not:

- **`v21.10` was never broken on an aligned start.** It is the *best* config
  there. Its v3crowd failures are entirely the undeclared `start_yaw`.
- **The oscillation ranking inverts on clearance.** `bal-helm35` is best on
  every oscillation metric and worst on clearance -- 0.368 aligned, 0.458
  unaligned, both below `v20.7.8-2`'s 0.483, so not shippable. A nimbler ship defers its avoidance.
- **`bal-dlg10` fails by grounding**, not by the corridor abort. At t = 111 s,
  still swinging from the unaligned start, it cuts the first corner toward a
  local goal at the vertex (48,55) with land between, and touches the land cell
  at (53,59). *(First recorded here as a stall; corrected from the video.)*

`bal-mix-T33` is `bal-mix` with the horizon re-derived: lowering `T_yaw` moved
`T_act` to 8.345 s, so `predict_time_obstacle` becomes `4*T_act` = 33.38 and
`lambda` = 1/T = 0.029958. Because the barrier is a *mean* over the horizon,
shortening it strengthens the barrier -- and it recovers about a third of the
clearance `bal-mix` gave up on the aligned suites (0.614 -> 0.649 against
`v21.10`'s 0.724), and 0.660 -> 0.692 unaligned. That is `PREDICTION_HORIZON.md`
section 5.1's coupling, predicted in advance and confirmed here.

### 4.6 Open-water verification of v21.11

The full 13-case open-water suite, never run on this config before, each run
rendered inline. Official scorer, each version against its own obstacle-free
baseline (151.0 s for both):

| case | clearance v21.10 | v21.11 | delay v21.10 | v21.11 | max deviation v21.10 | v21.11 |
|---|---|---|---|---|---|---|
| `head_on` | 0.644 | 0.527 | +1.7 s | +0.8 s | 2.08 | 1.73 |
| `cross_*_045` | 0.628 | 0.522 | +0.5 | +0.3 | 1.24 | 1.09 |
| `cross_*_090` | 0.719 | 0.630 | +0.6 | +0.4 | 1.38 | 1.27 |
| `cross_*_135` | 0.899 | 0.851 | +1.2 | +0.8 | 2.19 | 2.14 |
| `overtake_ego` | 0.864 | 0.815 | +0.9 | +0.7 | 1.78 | 1.75 |
| `overtaken_ego` | 0.842 | 0.750 | +1.4 | +1.2 | 2.78 | 2.80 |
| `overtaken_ego_fast` | 0.753 | 0.663 | +1.0 | +0.8 | 1.94 | 1.91 |
| `null_safe_pass` *(control)* | 4.356 | 4.356 | 0.0 | 0.0 | **0.000** | **0.000** |
| `ow_sequence215` | 0.630 | 0.529 | +3.6 | **+7.1** | 2.30 | **5.15** |
| `ow_simultaneous215` | 0.655 | 0.505 | +3.6 | +1.0 | 2.25 | 2.19 |

**13/13 reach the goal, no collision, and the no-threat control still deviates
exactly 0.000.** Worst encounter clearance 0.628 -> **0.522**, still above
`v20.7.8-2`'s 0.483; mean over the seven geometries 0.764 -> 0.680.

The trade is the one section 4.5 predicted: **less clearance, but earlier and
smaller manoeuvres.** Every single-encounter delay is shorter and nine of ten
deviations are smaller. A ship that can turn faster waits longer to turn, and
passes closer.

**One regression, and it is not the oscillation.** In `ow_sequence215` the
second target crosses from the ego's starboard side. The ego turns *east*, the
way the target is going, runs alongside it (clearance 0.529 at t ~ 93 s), and
circles back: 5.15 u of deviation and 7.1 s of delay against 2.30 and 3.6. The
cost function has no side preference -- open item 14 -- so which way the ego
breaks in a crossing is decided by small numerical differences, and v21.11's
differences broke the other way from v21.10's. A COLREGs side rule fixes
this; no oscillation parameter does.

---

## 5. Recommendation

```
    mu = (dist_localgoal/max_speed + predict_time_to_goal) / T_yaw
    T_yaw = max_yaw_rate / max_delta_yaw_rate
```

**Superseded 2026-09-23.** This section previously recommended
`dist_localgoal ~ 18` for v21.10, derived from the `mu ~ 4` settling optimum and
flagged in section 6 as "derived, not measured". It has now been measured, and
the recommendation was wrong: `dLG` 7 -> 18 changes the peak excursion by
**0.05%**. The `mu ~ 4` optimum is real but it is an optimum for *damping*, and
damping was never what the v3crowd runs were failing on.

**Set the parameters in this order.**

1. **Never ship `mu < 1`.** That is linear instability, not a tuning choice.
   `v21.6` sat at 0.80 and diverged at x1.49.
2. **Size the peak excursion with `max_delta_yaw_rate`, not `dist_localgoal`.**
   `e_acq ~ v * t_acq` and `t_acq = T_yaw + psi0/w_max`. This is the only
   parameter that shrinks the swing. Budget it against criterion 2 of
   section 3.4.
3. **Then set `dist_localgoal` for damping and corridor width.** It buys both,
   and nothing else buys the corridor. Diminishing returns are sharp: 7 -> 10
   drops helm saturation 0.776 -> 0.602, 10 -> 18 only 0.602 -> 0.490.
4. **Re-derive `predict_time_obstacle` whenever `T_yaw` moves.** `4*T_act`, with
   `lambda = 1/T`. Skipping this costs real clearance -- 0.614 vs 0.649 on the
   aligned suites, for no other change.
5. **Do not tune the gains for this.** Section 3.3: the yaw stationarity
   condition has no gain in it, and the one path that does work -- slowing the
   ship -- pivots it in place for up to a minute and then runs it aground.
6. **Declare `start_yaw` along the first leg in every scenario.** Not because
   the follower needs it, but because an undeclared one silently injects the
   largest transient in the run, and it is the difference between `v21.10`
   passing v3crowd 3/3 and failing 0/3.

**The best config measured is `bal-mix-T33`, promoted to `v21.11`**:
`dist_localgoal` 9, `max_delta_yaw_rate` 0.028099 (full helm 6.74 s),
`predict_time_obstacle` 33.38, `time_decay_lambda` 0.029958. Three configs reach
the goal in both conditions with clearance above `v20.7.8-2`'s 0.483 --
`bal-mix` (0.614/0.660), `bal-helm52` (0.519/0.567) and `v21.11`
(**0.649/0.692**) -- and `v21.11` has the most clearance of them. Neither of its
parameters moved more than 35% from `v21.10`. `bal-helm35` is excluded: 0.368
is below the `v20.7.8-2` bar despite winning every oscillation metric.

**Its cost is real and is not hidden here:** on an aligned start, clearance
falls from `v21.10`'s 0.724 to 0.649 on v3crowd, and from 0.628 to 0.522 on
open water (section 4.6). Robustness to an undeclared heading is
bought with clearance. If every scenario declares `start_yaw`, `v21.10` is the
better config and this whole trade is unnecessary.

**The helm-rate change is a claim about the ship, not a free parameter.** Full
helm in 6.74 s is 3.0x faster than the AIS-matched 20 s (`v21.10` already took
2.2x of that). Whether the vessel being modelled can do this is a naval
architecture question this document cannot answer.

## 6. What this does not settle

- **RESOLVED: the `mu ~ 4` optimum is now measured, and it does not do what
  section 5 once claimed.** `dLG` = 18 was run. It changes the peak excursion by
  0.05% and improves helm saturation by 19% over `dLG` = 10. See section 4.4.
- **Why `e_acq ~ v` over-predicts the benefit of slowing is unexplained.**
  `bal-v040` is the worst fit in the whole table at +12.6%, and it is the only
  config that varies `max_speed`. One more speed point would settle whether the
  model is wrong or that run is.
- **`t_settle` is never measured directly.** Every damping claim here rests on
  helm saturation and tail RMS, which are proxies. The settling column of
  section 3.1's table remains untested.
- **Every failure among the candidate configs is a grounding at a corner.**
  `bal-dlg10` at (53,59), the three gain runs at (46,46) and (67,27), and the
  earlier `rejoin` at (52,58): each where the Theta* path turns close to land.
  Pure pursuit cuts corners, and by more for a longer look-ahead, which would
  put an *upper* bound on `dist_localgoal` that nothing in this document
  derives. One `dLG` case is not enough to call it; a corner-clearance measure
  run across the `dLG` series would.
- **Clearance and oscillation trade against each other and nothing here
  predicts the exchange rate.** Section 4.5 measures it for five configs; there
  is no model for it.
- **Large-signal damping is better than the linear model says.** The 67.5° step
  settled by t ~ 105 s where the linear `t_settle` is 299 s — rate saturation
  adds damping the model omits. So §3.1's settling column is conservative, by an
  unmeasured factor.
- **The two regimes are treated as separate and they are not.** A large
  acquisition transient decays at the small-signal rate, so `mu` governs its
  *tail* while the integral governs its *peak*. Nothing here derives the
  crossover.
- **Nothing pulls the ego back toward the global path** (item 15). Every result
  here is about a follower that steers only at a look-ahead point. A cross-track
  term in the cost function would change the whole analysis.
- **`mu` conflicts with nothing measured, but it has not been tested against
  cornering.** Every number here comes from straight legs or a single step.

---

## 7. Seven metric artefacts this analysis produced

Recorded because each one briefly produced a confident wrong answer:

1. **A fixed-time envelope window cannot compare different `mu`.** Damping read
   ×0.47/×0.12/×0.19/×0.45 for `dist_localgoal` 5/7/10/20.69 — apparently
   non-monotone, contradicting the theory. Their periods are 43/54/72/159 s and
   the window was 15–25 s, so it sampled 0.09 to 0.35 of a cycle depending on
   the config. **The grid's preference for `dLG` = 7 was an artefact**, and it
   was used to justify keeping the look-ahead short.
2. **Peak-picking read growth where the trace decays.** The 67.5° step scored
   "decay/cycle 1.12" from local extrema; the actual peaks are
   2.22 → 1.94 → 1.55 → 1.02 → 0.44.
3. **`|x - 50|` is not cross-track.** Hardcoded for the open-water path; on the
   v3 route (70,90) → (60,10) it read 21.74 for all six runs. Identical values
   across six different runs was the tell.
4. **A run that never manoeuvres scores as perfectly damped.** `overtake_ego`
   read ×0.00 "settles" while trailing its target for 270 s.
5. **A run that ends early scores well on a maximum statistic.** `bal-spd006`
   read `xte_max` = 1.358, the best of any config, because it ran aground at
   t = 160 s, before the parts of the route where the swings happen. Caught
   only because mean `v` 0.311 over ~50 u of a 103 u route did not add up.
   **Read the peak alongside how and when the run ended.**
6. **Comparing two runs whose scenario changed between them.** `start_yaw` was
   added to the v3crowd suites mid-investigation, so "12/12 versus 0/3" was
   measuring the scenario edit, not the configs. `git log -S` on the scenario
   file is the check; the fair baseline reversed the conclusion (section 4.5).
7. **A clearance that ignores hull length.** The scratch scorer used for
   sections 4.4-4.5 measured from the hull centre with the half-beam as radius,
   so a 1.2-long hull passing abeam read 0.04-0.13 too high. The committed
   figures -- including the 0.483 bar everything is compared against -- come
   from `encounter_report.py`, which measures to the hull axis. Comparing one
   against the other made `bal-helm35` look like 0.414 against 0.482 when it is
   0.368 against 0.483. Caught when `v21.10`'s open-water worst case read 0.718
   here against the 0.628 in its own validation commit.

And one diagnosis error of the same family: **"did not reach the goal, no
error" is not "stalled".** A collision ends the run with `error = None`; the
cause is in `collision_event`. Four runs were first recorded here as stalls and
all four were groundings.

The shared cause: a measure built for one configuration, reused on another
without rechecking what it assumed.
