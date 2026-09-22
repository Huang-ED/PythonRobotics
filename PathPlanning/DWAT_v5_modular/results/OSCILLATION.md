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
>     mu = dist_localgoal / (max_speed * T_act)
>     T_act = max(max_speed/max_accel, max_yaw_rate/max_delta_yaw_rate)
> ```
> `mu > 1` is a hard stability boundary. `mu ~ 4` minimises settling time.
>
> **Large-signal**, acquiring the track from a heading error `psi0`, is not
> governed by `mu` at all. It is a rate-limited manoeuvre whose excursion is
> ```
>     e_acq = INT v(t) sin(psi(t)) dt     over the slew
> ```
> and no amount of look-ahead changes it.

Confusing the two is how three of this investigation's wrong turns happened.

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
the yaw rate cannot change instantly. Inserting a first-order lag `tau = T_act`:

```
    T_act s^3 + s^2 + (2v/L_d) s + 2 v^2/L_d^2 = 0
```

Substituting `s = sigma/T_act` removes every dimensional quantity:

```
    sigma^3 + sigma^2 + (2/mu) sigma + 2/mu^2 = 0,     mu = L_d/(v * T_act)
```

**Speed, look-ahead and actuation lag enter only through their ratio.** That is
why raising `max_delta_yaw_rate` by 2.2x and raising `dist_localgoal` from 7 to
20 were interchangeable fixes for the same divergence — both raise `mu`.

Routh-Hurwitz on `sigma^3 + sigma^2 + a sigma + b` requires `a > b`:

```
    2/mu > 2/mu^2       =>       mu > 1       i.e.   dist_localgoal > v * T_act
```

Solving for the dominant pole gives the design curve:

| `mu` | `zeta` | overshoot | decay/cycle | `t_settle`/`T_act` | `T_osc`/`T_act` |
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

### 3.3 What counts as acceptable

Two independent criteria, and both must hold:

1. **The transient must decay before the next encounter.**
   `t_settle < t_meet` of the first target. This is the binding one in traffic.
2. **The excursion must stay inside the corridor.**
   `e_acq < dist_localgoal`, or the look-ahead circle leaves the path and the
   run aborts (`OPEN_PROBLEMS.md` item 15).

Criterion 1 is what the v3crowd runs violate.

---

## 4. What the evidence says

### 4.1 The stability boundary predicts every observed divergence

| config | `L_d` | `T_act` | `mu` | predicted | **measured** |
|---|---|---|---|---|---|
| v20.7.8-2 | 7.0 | 2.50 | 5.60 | stable | max xte 0.70, no ringing |
| **v21.5 / v21.6** | 7.0 | 20.00 | **0.70** | **UNSTABLE** | **diverged, ×1.49 growth** |
| v21.7 preview-20 | 20.0 | 20.00 | 2.00 | stable, ×0.28 | settles ×0.37 |
| v21.9 / v21.10 | 7.0 | 9.10 | 1.54 | stable, ×0.48 | settles ×0.24–0.66 |

The v21.6 divergence — visible in the videos as a cross-track that keeps
growing after the pass — sits at `mu = 0.70`, below the Routh-Hurwitz boundary.
It was not a tuning failure. It was linear instability.

Predicted ringing period at `mu = 1.54` is `6.0 * T_act` = **54.5 s**; measured
between the v21.6 peaks at t=97.6 and 125.1 s: **55 s**.

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

`v21.10` runs at `mu = 1.54`, so `t_settle` = **299 s** — longer than the entire
200 s transit. The v3 scenarios declare no `start_yaw`, so the ego begins 99.7°
off its first leg and injects a 3.24 u transient. The first target meets it at
**t = 38 s**, while the ego is still ringing. Every encounter is handled
mid-transient, the excursions add, and cross-track passes `dist_localgoal`.

Both criteria of §3.3 are violated at once.

---

## 5. Recommendation

```
    mu = dist_localgoal / (max_speed * T_act)        target mu ~ 4
```

- **Never ship `mu < 1`.** That is linear instability, not a tuning choice.
- **`mu ~ 4`** minimises settling time (`16 * T_act`) at `zeta = 0.5`.
- For v21.10 (`v` 0.5, `T_act` 9.10) that is **`dist_localgoal` ~ 18 u**, against
  the current 7.0.
- Raising `mu` by cutting `T_act` — i.e. a more responsive ship — is equally
  valid and shortens acquisition too, which `L_d` cannot.
- **Declare `start_yaw` along the first leg** in every scenario. Not because the
  follower needs it, but because an undeclared one silently injects the largest
  transient in the run. Open-water scenarios already do; no v3 scenario did.

---

## 6. What this does not settle

- **The `mu ~ 4` optimum is derived, not measured.** No run has yet been done at
  `dist_localgoal` = 18 with the encounter suite scored on both clearance and
  settling. Until then it is a prediction.
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

## 7. Four metric artefacts this analysis produced

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

The shared cause: a measure built for one configuration, reused on another
without rechecking what it assumed.
