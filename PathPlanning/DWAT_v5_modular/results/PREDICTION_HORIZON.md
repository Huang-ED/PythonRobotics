# How long should `predict_time_obstacle` be?

A single question: **what sets the right prediction horizon?** Nothing else is
decided here — no code, no config changes, no backlog items. Written 2026-09-16
while `v21.2` was running, because that config's horizon (78.76 s) was derived
by analogy rather than from an argument, and the analogy turns out to be the
wrong one.

---

## 1. The claim

> **The horizon should be measured in units of the ship's actuation time, not in
> ship lengths and not in seconds.**
>
> `predict_time_obstacle ≈ 4 · T_act`, where
> `T_act = max(max_speed/max_accel, max_yaw_rate/max_delta_yaw_rate)`.

Everything below is the argument for that, the evidence for it, and the upper
bound that stops it growing without limit.

---

## 2. Why *not* ship lengths

`v21.0`–`v21.2` were derived by matching the AIS suite's horizon expressed in
ship lengths (`v·T/L = 32.8`). That choice has no mechanism behind it. A ship
length is a *geometric* property; a horizon is a *temporal* budget. The two are
related only through speed, and speed is exactly what the planner is choosing.

The measured spread kills the idea outright:

| config | `v·T/L` | outcome |
|---|---|---|
| `v20.7.8-2` | 12.5 | passes 9 of 11 encounters |
| `ais_v1.1/ny_hist_367791550` | **73.6** | **grounds** |
| `ais_v1.1/ny_synth_00` | **2.7** | **aborts** |
| `v21.2` | 32.8 | passes `head_on`, `overtaken_ego` |

A 27× spread across the working configs, with failures at both extremes. It
predicts nothing.

---

## 3. What actually constrains the horizon

### 3.1 The lower bound: the horizon *is* the reaction time

For a closing encounter the planner sees a threat when the obstacle's
extrapolated position first intersects the ego's rolled-out path — that is, at a
separation of roughly `(v_own + v_target)·T`. So **time-to-CPA at the moment of
detection is `T` itself.** The horizon does not merely bound a computation; it
*is* the warning time the controller gets.

For that warning to be useful the ego must be able to (a) develop a manoeuvre,
(b) execute it, and (c) *see it work* before the horizon ends:

```
T  ≥  T_act        actuation: slew to full helm or full thrust
    +  T_exec       execution: achieve the required lateral offset
    +  T_verify     verification: see separation growing again
```

**`T_act` — the actuator, not the hull.** Two timescales compete and the slower
governs:

```
T_surge = max_speed / max_accel               seconds to full speed
T_yaw   = max_yaw_rate / max_delta_yaw_rate   seconds to full helm
T_act   = max(T_surge, T_yaw)
```

For `v21.2` that is `max(13.0, 20.0) = 20.0 s`, set by the rudder, not the
engine. This is the term the ship-length framing misses entirely.

**`T_exec` — the geometry of getting out of the way.** With turn radius `R`, the
along-track distance to build a lateral offset `d` is approximately `√(2·R·d)`
for `d ≪ R`, so

```
T_exec ≈ √(2·R·d) / v
```

For `v21.2` at the safety threshold (`d = 1.15 u`, `R = 2.64 u`, `v ≈ 0.3 u/s`):
`√(2·2.64·1.15) = 2.46 u`, hence `T_exec ≈ 8.2 s`. Note `T_exec` is *small
against `T_act`* — 0.4 of it. **For a ship, turning is dominated by getting the
rudder over, not by the turn itself.** This is the opposite of the wheeled-robot
regime the algorithm was written for, where `T_act` is near zero.

**`T_verify` — why `T_act + T_exec` is not enough.** This is v20.7.9's lesson in
a different guise. The planner does not execute a manoeuvre; it *scores* one, by
rolling it out and reading the cost at the end. If the horizon ends at CPA, the
rollout shows the closest approach and nothing after it, so an evasive candidate
and a doomed one score alike. The horizon must extend past CPA far enough for
separation to be visibly increasing. Empirically this is the same order as
`T_exec`.

Summing with a modest margin:

```
T  ≳  T_act + 2·T_exec  ≈  20 + 16  =  36 s        i.e.  T / T_act ≈ 1.8
```

That is the **floor**. It is not the recommendation — see §4.

### 3.2 The upper bound: the time discount already truncates it

`calc_trajectory_clearance_and_collision` weights each rollout sample by
`exp(−λ·t)`. Samples beyond a few `1/λ` contribute essentially nothing to the
cost while costing full compute time:

| `λ·T` | weight at the far end | interpretation |
|---|---|---|
| 1.0 | 37% | far horizon still matters |
| 2.0 | 14% | diminishing |
| 3.0 | **5%** | the tail is nearly free of influence |
| 7.9 | **0.04%** | the tail is decoration |

So there is a hard ceiling: **`T ≤ 3/λ`**. Beyond it you are paying linear
compute for samples the cost function has already discounted away.

This is not hypothetical. A `v21.x` config that had kept `λ = 0.1` while
extending `T` to 78.76 s would have sat at `λ·T = 7.9` — **the last 60% of the
rollout weighted below 0.5%**, pure waste. That is why `λ` was reduced to
0.01269 alongside the horizon, giving `λ·T = 1.0`.

**The two bounds must be designed together**, which gives a consistency rule:

```
λ  ≈  3 / (k · T_act)        where  k = T / T_act
```

Horizon and discount are not independent knobs. Changing one without the other
either starves the far horizon or pays for a tail nobody reads.

### 3.3 There are *two* horizons, and the ordering is a hard constraint

Everything above concerns `predict_time_obstacle` — Stage 2's horizon, `T₂`.
Stage 1 has a horizon too, and the relation between them is not free:

> **`T₁ ≤ T₂` always.**

The reason is what each stage is *for*. Stage 2 is a **gradient**: it scores
candidates, so it can push the ego gently away from a hazard long before the
hazard is critical. Stage 1 is a **veto**: it deletes candidates outright. If
Stage 1 sees further than Stage 2, it can reject a candidate over a hazard the
cost function never sees — the planner then meets a cliff with no preceding
slope, because the thing that vetoed it was never in the objective. That is
exactly the §7.3 failure in `OPEN_PROBLEMS.md`: Guard A fired with **no prior
cost gradient** steering the ego away.

**The shipped code violates this, without bound.** `closest_obstacle_on_curve`
takes no span limit — it scans the whole arc, so Stage 1's reach is a full
circle at the sampled `(v, ω)`:

| config | `T₂` | Stage 1 arc time at max ω | ratio |
|---|---|---|---|
| `v20.7.8-2` | 30.0 s | 90.0 s | **3.0×** |
| `v21.2` | 78.8 s | 66.4 s | 0.8× |
| `ais_v1.1/ny_hist_366962130` | 120.0 s | 101.1 s | 0.8× |

And those are at *maximum* yaw. The window samples ω down to ~0, where
`R = v/ω → ∞` and the scan becomes an effectively infinite ray — so across most
of the candidate grid `T₁` is unbounded while `T₂` is 30–120 s. The ordering is
not merely violated; it is inverted without limit. This is item 9, and the rule
is *why* item 9 is a defect rather than an inefficiency.

**How much horizon Stage 1 actually needs.** A braking test only cares about
distance up to the stopping distance; beyond it the test passes regardless. So
its natural horizon is the stopping time `v/a + dt`:

| config | stop time | `T₂` | `T₁/T₂` |
|---|---|---|---|
| `v20.7.8-2` | 2.6 s | 30.0 s | **0.087** |
| `v21.2` | 13.1 s | 78.8 s | **0.166** |
| `ais_v1.1/ny_hist_366962130` | 23.0 s | 120.0 s | **0.192** |

Stage 1 needs 9–19% of Stage 2's horizon. Everything past that is not merely
wasted — it is the inversion.

**Why the bound is `≤` and not `<`.** Strict inequality holds while Stage 1 is a
*braking* test, because stopping is inherently short-horizon. But under item 6,
where Stage 1 is rebuilt as a CPA test sharing Stage 2's rollout, both stages
compute the same quantity over the same horizon and differ only in threshold:
Stage 2 penalises, Stage 1 vetoes once the penalty saturates. `T₁ = T₂` is then
correct — indeed it is the point, since the two stages finally agree about
physics. `T₁ > T₂` is the incoherent case, and it is what ships today.

**Consequence for the backlog.** Implementing item 5 by reusing Stage 2's
sampled spacetime check makes `T₁ = T₂` structural rather than a convention to
maintain by hand, and item 9 dissolves rather than needing separate work. It
also settles how obstacle culling must be filed: culling Stage 2 by reachability
is provably bit-identical and is an optimisation, while culling Stage 1 by
horizon changes behaviour — but by this rule that change is the *fix*, so it
belongs with item 9 behind a config key and measured, not smuggled in as a
speed-up.

---

## 4. What the evidence says

`T/T_act` for every config that has been run:

| config | `T_surge` | `T_yaw` | `T_act` | `predT` | **`T/T_act`** | outcome |
|---|---|---|---|---|---|---|
| `v20.7.8-2/predT5` | 2.5 | 1.0 | 2.5 | 5.0 | **2.00** | **fails — by design** |
| `ais_v1.1/ny_synth_00` | 54.6 | 20.0 | 54.6 | 120.0 | **2.20** | **aborts** |
| `ais_v1.1/ny_hist_367791550` | 39.2 | 20.0 | 39.2 | 120.0 | 3.06 | grounds *(dt, item 3)* |
| `v21.2` | 13.0 | 20.0 | 20.0 | 78.8 | 3.94 | passes 2 of 2 so far |
| `ais_v1.1/ny_hist_366962130` | 13.0 | 20.0 | 20.0 | 120.0 | 6.00 | aborts *(items 1+2)* |
| `v20.7.8-2` | 2.5 | 1.0 | 2.5 | 30.0 | 12.00 | passes 9 of 11 |

Two observations, and one caution.

**The two lowest values are the two horizon-starved failures.** `predT5` at 2.00
is the deliberate demonstration that a short horizon cannot see an overtake
coming. `ny_synth_00` at 2.20 is the genuine braking case — a 147 m ship needing
54.6 s to stop, given a horizon only 2.2 stopping-times long. The metric puts
them next to each other, which no other framing does: in ship lengths they are
2.1 and 2.7, indistinguishable from nothing in particular.

**The failures above 3.0 have known, different causes.** `ny_hist_367791550`
grounds because `dt` is too coarse (item 3); `ny_hist_366962130` aborts on the
phantom vessel and the disc hull (items 1+2). Neither is a horizon failure, so
neither contradicts the metric.

> **Caution.** Six configs, two supporting failures, and both have alternative
> explanations available. This is *suggestive*, not established. What the section
> claims is that `T/T_act` orders the evidence coherently where `v·T/L` does not
> — not that the threshold is known to three figures.

---

## 5. Recommendation

```
T_act = max(max_speed/max_accel, max_yaw_rate/max_delta_yaw_rate)

predict_time_obstacle  =  4 · T_act        (floor 1.8, ceiling from λ below)
time_decay_lambda      ≈  3 / predict_time_obstacle
```

`k = 4` sits above the floor of ~1.8 with roughly a factor of two in hand, below
`v20.7.8-2`'s 12.0 — which is generous to the point of being wasteful, since
compute is linear in `T` — and near AIS's 6.0 without inheriting its cost.

Applied:

| config | `T_act` | current `predT` | **`4·T_act`** | change |
|---|---|---|---|---|
| `v21.2` | 20.0 | 78.8 | **80.0** | +2% — essentially unchanged |
| `v20.7.8-2` | 2.5 | 30.0 | **10.0** | −67% |
| `ais_v1.1/ny_hist_366962130` | 20.0 | 120.0 | **80.0** | −33% |
| `ais_v1.1/ny_synth_00` | 54.6 | 120.0 | **218.4** | **+82%** |

Three things fall out of that table.

**`v21.2`'s horizon is accidentally right.** 78.76 s against a derived 80.0 s.
It was obtained by matching ship lengths — the wrong argument — and landed
within 2% of the right answer. Worth knowing it is not luck that needs
preserving: the two agree here only because `v21.2` also matches AIS on
`R/L = 2.20`, which ties the geometric and temporal framings together for this
particular vessel. It will not hold for a different hull.

**`v20.7.8-2` is 3× longer than it needs to be.** At `T_act = 2.5 s` the
wheeled-robot ego needs 10 s, not 30 s. Two thirds of every rollout in the
entire open-water suite's history has been computing a future the ego could have
reached three times over. Shortening it is the single cheapest speed-up
available — but it changes the reference case, so it is a proposal, not a patch.

**`ny_synth_00` needs a horizon 82% *longer*, not shorter.** A 147 m ship at
54.6 s of stopping time is being asked to plan 120 s ahead — 2.2 stopping times.
This is a concrete, independent prediction: **if the horizon is the cause of that
abort, raising `predict_time_obstacle` to ~218 s should change the outcome.**
That is a one-run experiment and it does not depend on any other backlog item.

---

## 6. What this does not settle

- **`predict_time_to_goal`.** A different question — it bounds the goal-seeking
  rollout, not the collision one, and is currently a redundant recomputation of
  a prefix of the obstacle rollout (both cost terms read only its final row).
  That is an efficiency defect, not a horizon question.
- **Whether `T_exec` is the right execution model.** `√(2·R·d)` is the
  small-offset approximation. For large course changes the exact advance/transfer
  geometry applies and `T_exec` grows.
- **The COLREGs direction.** "Early and substantial action" argues for a *longer*
  horizon than collision avoidance alone requires, and pushes against the
  `exp(−λ·t)` discount, which weights the near future more. `OPEN_PROBLEMS.md`
  §7.2 point 4 records that tension; nothing here resolves it.
- **The empirical threshold.** §4 shows failures at 2.0 and 2.2 and successes at
  3.9 and 12.0. Where between 2.2 and 3.9 the boundary lies is unmeasured. A
  `T/T_act` sweep on one scenario would settle it and has not been run.
