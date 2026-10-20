# Horizon weighting: does discounting beat uniform averaging?

The paper's third contribution is to aggregate threat severity over the
prediction horizon as an **exponentially discounted** expectation rather than a
**uniform** one, claiming it (a) gives "earlier and smoother evasion" and (b)
avoids "over-commitment to distant, low-confidence predictions" (`main.tex`,
lines 39, 119, 132). This folder tests both claims on v21.13.

**Answer, 2026-10-15, on single-threat encounters:** the discount's benefit
comes from its near-term emphasis -- the "urgency" it was designed to encode
(line 119) -- and on these scenes a gain matches it at equal cost. The
far-term down-weighting has no measurable effect, good or bad: the far
horizon barely moves the planner under any weighting. Neither "earlier and
smoother" nor "avoids over-commitment" shows here.

**What these scenes cannot test** is the shape's distinguishing claim: that
"an imminent hazard outweighs a distant one" (line 119) when the planner must
trade one against the other, "balancing multiple simultaneous threats" (line
132). With one target, every candidate's danger falls at about the same time,
so any time weight acts on all candidates alike -- exactly as a gain does.
The staggered-slalom suite (§10) was built for that test, and the shape did
not separate from the gain there either: how the ego divides its room
between an imminent and a later vessel is set by the geometry, not by the
weighting.

**Contents:** [1 Arms](#1-arms) · [2 Suites](#2-suites) ·
[3 Result: discount alone](#3-result-the-discount-alone) ·
[4 Why: normalised weights amplify the near term](#4-why-normalised-weights-amplify-the-near-term) ·
[5 Result: against a gain-matched uniform](#5-result-against-a-gain-matched-uniform) ·
[6 Result: when the forecast is wrong](#6-result-when-the-forecast-is-wrong) ·
[7 What this means for the paper](#7-what-this-means-for-the-paper) ·
[10 Two threats at different times](#10-two-threats-at-different-times-the-staggered-slalom) ·
[11 Why: what the weighting can and cannot reach](#11-why-what-the-weighting-can-and-cannot-reach) ·
[12 Mechanism tests: forecasts that drift, and a near-vs-far dilemma](#12-mechanism-tests-forecasts-that-drift-and-a-near-vs-far-dilemma) ·
[13 Mean vs max](#13-mean-vs-max-integrated-vs-extremum-aggregation) ·
[8 Caveats](#8-caveats) · [9 Reproducing](#9-reproducing)

---

## 1 Arms

Five configs, each differing from `v21.13.json` in one or two keys. All carry
v21.13's COLREGs layer, Stage 1 `braking_rollout_static`, and the `"mean"`
aggregation, so horizon weighting is the only thing that changes.

| arm | config | `time_decay_lambda` | λ·T | `side_cost_gain` |
|---|---|---|---|---|
| uniform | `v21.13/ablation_uniform.json` | 0 | 0 | 0.98 |
| uniform ×1.58 | `v21.13/ablation_uniform_gain1p58.json` | 0 | 0 | 1.5484 |
| **v21.13** | `v21.13.json` | 0.029958 | 1 | 0.98 |
| uniform ×3.15 | `v21.13/ablation_uniform_gain3p15.json` | 0 | 0 | 3.0821 |
| strong discount | `v21.13/ablation_discount_strong.json` | 0.089874 | 3 | 0.98 |

T = `predict_time_obstacle` = 33.38 s. λ·T = 3 is the ratio of the draft's
v20.7.8-2 (λ = 0.1, T = 30 s). The ×1.58 and ×3.15 factors are §4's
near-term weights.

## 2 Suites

- **Straight (16 scenes):** the 11 `owenc215_*` single-target encounters, the
  two combined open-water scenes and the three `v3crowd*_v2111` scenes. Every
  target holds constant velocity, so the forecast is exact.
- **Course-altering (6 scenes, `owturn215_*`):** a 45° alteration 20 s before
  the meeting, so the forecast is wrong. `_away` is a collision course that
  turns clear (does a planner over-commit to it?); `_in` passes clear and then
  turns onto a collision course (does it react too late?). See the
  "Course-altering targets" section of `../../README.md`.

v21.13's straight-suite runs are those of `runs/v21.13/port_turn_release6/`
(byte-identical config; `b44c3b4` re-verified bit-identical behaviour). Every
other run is under `runs/v21.13/<arm>/` and `runs/v21.13/encounters/`. 94 new
runs, all exit 0, all reaching the goal.

## 3 Result: the discount alone

| straight suite, 16 scenes | uniform | v21.13 | strong |
|---|---|---|---|
| reach the goal | 16 | 16 | 16 |
| lowest clearance | 0.396 | 0.539 | **0.707** |
| mean clearance | 0.987 | 1.059 | 1.127 |
| mean delay | 5.4 s | 5.7 s | 6.2 s |
| median lead time (leaves its track → CPA) | 20.5 s | 21.0 s | 21.1 s |
| mean extra heading change vs obstacle-free run | 120° | 125° | 138° |
| R14/R15 encounters on the wrong side | 0 | 0 | 0 |

Read alone, this looks like support: clearance rises with the discount in 15
of 16 scenes. But evasion is **not earlier** (lead time is flat to 0.6 s) and
**not smoother** (more turning, more delay). That is the signature of a
stronger avoidance term, not of better timing.

## 4 Why: normalised weights amplify the near term

The weights are normalised (`aggregate_compound_costs`, `"mean"`), so
discounting does not just shrink the far horizon -- it redistributes the same
total weight towards the near one:

| weight relative to uniform | t = 0 | t = 10 s | t = 20 s | t = T |
|---|---|---|---|---|
| λ·T = 1 (v21.13) | **1.580×** | 1.171× | 0.868× | 0.583× |
| λ·T = 3 (strong) | **3.145×** | 1.280× | 0.521× | 0.158× |

As two vessels close, the danger always arrives at small t, so at the moment
that decides the clearance a discounted arm is running a larger gain. The
control is therefore uniform averaging with `side_cost_gain` scaled by the
same factors.

## 5 Result: against a gain-matched uniform

| straight suite, 16 scenes | uniform | ×1.58 | v21.13 | ×3.15 | strong |
|---|---|---|---|---|---|
| lowest clearance | 0.396 | 0.603 | 0.539 | **0.915** | 0.707 |
| mean clearance | 0.987 | 1.122 | 1.059 | **1.314** | 1.127 |
| mean delay | 5.4 s | 5.7 s | 5.7 s | 6.1 s | 6.2 s |
| mean extra heading change | 120° | 132° | 125° | 145° | 138° |

Each gain-matched uniform arm buys more clearance than the discount it is
matched to, at more turning -- which is not "better": past some point a
stronger avoidance term only costs more. The meaningful comparison is at
**equal cost**. The discount is "strong near, weak far"; uniform ×k is
"strong near, strong far". If the weak far weight bought anything, uniform ×k
would pay for its strong far weight with earlier or larger manoeuvres per
unit of clearance. To compare at equal cost, the three uniform
gains trace a clearance-vs-cost curve per scene, and each discounted arm is
compared with that curve at its own cost (`compare.py --frontier`; both
suites, scenes without a manoeuvre excluded):

| discounted minus uniform clearance, at equal | v21.13 (λ·T = 1) | strong (λ·T = 3) |
|---|---|---|
| turning | **−0.012** u (better in 6/18 scenes, worse in 8) | **+0.002** u (8/18 better, 7 worse) |
| delay | **−0.023** u (6/20 better, 9 worse) | **−0.025** u (4/17 better, 10 worse) |

At equal turning or equal delay, discounting and uniform averaging are
indistinguishable: mean differences of 0.002–0.025 u against clearances
around 1 u, and the scenes split both ways. Lead time is flat too
(20.5–21.9 s in all five arms): the strong far weight of uniform ×3.15 does
not make the ego act earlier, so there is no over-commitment for the
discount to avoid.

## 6 Result: when the forecast is wrong

The course-altering suite is where "avoids over-committing to unreliable
forecasts" should show. It does not:

| | uniform | ×1.58 | v21.13 | ×3.15 | strong |
|---|---|---|---|---|---|
| `head_on_away`: deviation already committed when the threat vanished | 0.47 | 0.47 | 0.47 | 0.47 | 0.47 |
| `cross_stbd_090_away`: same | 0.85 | 0.85 | 0.87 | 0.82 | 0.86 |
| `_in`: seconds from the target's turn to the ego's first manoeuvre (head-on / stbd / port) | 6.6 / 5.1 / 10.4 | 6.5 / 5.0 / 10.1 | 6.6 / 5.1 / 10.4 | 6.1 / 5.0 / 9.8 | 6.6 / 5.1 / 10.4 |
| `_in` clearance (head-on / stbd / port) | 0.83 / 0.51 / 0.54 | 0.95 / 0.68 / 0.73 | 0.90 / 0.62 / 0.64 | 1.14 / 0.91 / 0.96 | 0.97 / 0.75 / 0.78 |

Commitment and reaction times are the same in every arm. Only clearance
differs, ordered by effective gain exactly as on the straight suite.
`cross_port_090_away` is omitted: the ego is the stand-on vessel and never
manoeuvres (deviation ≤ 0.07 in every arm).

## 7 What this means for the paper

- **On single threats, the discount works through its near-term emphasis**
  -- the urgency it was designed for -- and a tuned gain reaches the same
  trade-off. A reviewer asking "why not just tune the gain?" is not answered
  by these scenes.
- **The far-term down-weighting ("predictive confidence") shows no effect**,
  because the far horizon barely influences decisions in this planner under
  any weighting. "Earlier and smoother" (line 39) and "avoids over-commitment"
  (line 132) are not supported as written.
- **The claim that can still separate shape from gain** is ranking between
  threats at different times: "an imminent hazard outweighs a distant one".
  §10 tests it -- and finds no separation either.
- **Integrated vs extremum** (line 132's other comparison, `"max"`
  aggregation) remains untested and needs the same gain matching: the max of
  S is roughly an order of magnitude larger than its mean.

## 8 Caveats

- **One suite design, open water only.** No land, constant-speed targets,
  one 45° alteration at one time (20 s). A target that alters later, more,
  or repeatedly might still separate the arms; this suite found no sign of it.
- **`cross_stbd_090_away` is not the clean false alarm it was meant to be.**
  Its turn direction was chosen against a do-nothing ego; against an ego that
  correctly gives way (Rule 15, to starboard) the target's alteration to port
  turns it into the ego's new track, and every arm comes within 0.44–0.92 u of
  a target that would have passed 3.7 u off a do-nothing ego. It is a
  realistic Rule 17(c) violation by the other vessel, not a false alarm, and
  should be renamed and joined by a corrected case.
- **Three gains per curve.** The equal-cost comparison interpolates linearly
  between three points; 2–3 discounted points per metric fall outside the
  uniform curve's range and are excluded.
- **Deterministic runs, one per cell.** Differences of a few hundredths of a
  unit are within the scatter between neighbouring scenes.
- The `reversals` column of `compare.py` includes path-following wobble after
  the pass and is not used here.

## 9 Reproducing

```bash
# configs: PathPlanning/DWAT_v5_modular/configs/algorithm/v21.13/ablation_*.json
# suites:  configs/scenarios/encounters/owenc215_*, owturn215_*;
#          configs/scenarios/combined/ow_*215, v3crowd*_v2111
python -m PathPlanning.DWAT_v5_modular.main \
  --scenario PathPlanning/DWAT_v5_modular/configs/scenarios/encounters/owturn215_head_on_in.json \
  --algorithm-config PathPlanning/DWAT_v5_modular/configs/algorithm/v21.13/ablation_uniform.json \
  --run-config PathPlanning/DWAT_v5_modular/configs/run/default.json --frames-root /tmp/dwat-frames

python -m PathPlanning.DWAT_v5_modular.results.ablation_horizon_weighting.compare             # §3, §5 tables
python -m PathPlanning.DWAT_v5_modular.results.ablation_horizon_weighting.compare --turning   # §6
python -m PathPlanning.DWAT_v5_modular.results.ablation_horizon_weighting.compare --frontier  # §5, equal cost
```

## 10 Two threats at different times: the staggered slalom

`fig1_same_danger_different_time.png` shows the idea at the level of the cost,
computed with the planner's own barrier and aggregation at v21.13's
parameters: the same 0.3 u pass scored 2 s ahead and 10 s ahead.

| | uniform | v21.13 (λ·T = 1) | strong (λ·T = 3) |
|---|---|---|---|
| cost of the pass 2 s ahead ÷ the same pass 10 s ahead | 1.00× | 1.27× | 2.05× |

That ranking can only change behaviour when the candidates differ in *when*
they are in danger. `ow_slalom_{stbd,port}_d{05,10,20,30}` builds that choice:
two slow vessels ahead on the ego's course (0.2 against 0.5, overtaking, so
Rule 13 and no COLREGs cost), A met at 70 s offset 0.5 u to one side, B met Δ s
later offset 0.5 u to the other. A do-nothing ego touches both, so every
sidestep that clears one closes on the other. If the discount ranks the
imminent vessel higher, it should give A more of the room than uniform
averaging does -- and no single gain could reproduce that.

40 runs (5 arms × 8 scenes), all reaching the goal. Each mirrored pair
(`stbd`/`port`) came out identical to the last digit -- with no COLREGs cost the
planner is exactly mirror-symmetric -- so there are **four independent
scenes, one per Δ**.

**Clearance to A minus clearance to B** (positive = A got more of the room):

| Δ | uniform | ×1.58 | v21.13 | ×3.15 | strong |
|---|---|---|---|---|---|
| 5 s | +0.07 | +0.03 | +0.05 | −0.01 | +0.01 |
| 10 s | −0.13 | −0.17 | −0.14 | −0.20 | −0.14 |
| 20 s | +0.24 | +0.25 | +0.24 | +0.14 | +0.18 |
| 30 s | +0.09 | +0.10 | +0.10 | +0.10 | +0.09 |

The split between A and B changes with Δ -- with the geometry -- and hardly at
all with the weighting: each discounted arm is within 0.04 u of its
gain-matched uniform arm. The discount does not hand the imminent vessel more
room.

**The worse of the two clearances, at equal turning** (`compare.py --slalom`):
discounted minus uniform is −0.012 / −0.040 u at Δ = 5 s, +0.067 / +0.019 at
10 s, +0.033 / −0.051 at 20 s and −0.011 / −0.016 at 30 s (v21.13 / strong).
Signs flip from one Δ to the next, with one scene per Δ: no trend. (The
equal-delay comparison is not usable here: delay is 0.8–1.4 s in every run,
quantised at 0.1 s.)

**Reading.** The intuition holds for the cost -- figure 1 -- but not for the
closed-loop behaviour of this planner. A candidate is one (v, ω) setpoint held
for 33 s, so no candidate is "dodge A, then return for B": the sequencing is
done by re-planning every 0.1 s, under any weighting. A plausible
consequence is that the candidates never offer the near-vs-far trade-off the
weighting would arbitrate. That is a hypothesis; it can be checked
open-loop, by re-scoring the candidates at logged states under each
weighting and seeing whether the chosen candidate differs at all.

## 11 Why: what the weighting can and cannot reach

The question behind §10: two targets, one predicted to collide with the ego
in 1 s and one in 10 s. Under uniform weighting they cost the same -- which
is intuitively wrong. Why does no experiment show it? Three mechanisms, each
measured. Tool: `rescore.py`, which rebuilds the planner's exact inputs at
logged states and re-scores every candidate under each weighting (it
reproduces the logged control at 536/538 slalom states; 67-87% in scenes
with Rule 14/15 targets, whose COLREGs locks carry history one state cannot
rebuild).

**(1) A predicted collision is vetoed whenever it happens.** A rollout that
predicts contact anywhere in the 33 s horizon scores `inf` before any
weighting applies (`dwa_ramped.py`, `is_collision`). So the 1 s and 10 s
collisions of the thought experiment do cost the same in this planner -- not
because the weighting is uniform, but because the veto is: it is the one
part of the cost that ignores time entirely. The weighting only ever ranks
*near misses*. The veto binds often, and mostly on far predictions:

| v21.13 run | states with any vetoed candidate | most vetoed at one state | predicted contact < 5 s / 5–15 s / ≥ 15 s ahead |
|---|---|---|---|
| `owenc215_head_on` | 47% | 31% | 0% / 39% / 61% |
| `owenc215_cross_stbd_090` | 35% | 5% | 0% / 41% / 59% |
| `v3crowd5_v2111` | 56% | 33% | 0% / 44% / 56% |
| `v3crowd6_v2111` | 64% | 49% | 0% / 56% / 44% |
| `v3crowd7_v2111` | 67% | 33% | 0% / 28% / 72% |

Most vetoes are for contacts 15–30 s out -- the least certain predictions,
and the ones a re-planning ego can still avoid -- treated exactly like one
5 s out.

**But on these suites the veto almost never decides the choice.** Removing it
(`no_veto.py`: each vetoed candidate gets the finite cost the formula would
give, either the raw symmetric barrier or overlap clamped at maximum
severity) changes the cheapest candidate to a colliding one at **1 of 244**
states with a partial veto (head-on 1/28, starboard crossing 0/25, slalom
Δ = 20 s 0/85, v3crowd6 0/106), and that one contact is at 33.4 s, the last
step of the horizon. A candidate heading into a target also pays the
near-miss severity on the way in, and a clean alternative exists at every
state measured (no state had every candidate vetoed). So the veto removes
candidates that would lose anyway: it is not what hides the weighting here.

Removing it outright would still be unsafe where no clean alternative
exists: the barrier is symmetric in clearance, so a deep overlap scores
*lower* than a graze (−0.5 u and +0.5 u both give S = 1.375), and severity is
capped at 2.0 and averaged over 335 steps, so a 1.5 s contact adds at most
0.088 -- the size of an ordinary near miss.

**(2) A danger close in time is the same for every candidate.** The ship
cannot change what happens in the next few seconds: from cruise, all 495
candidates of one decision are within 0.033 u of each other 1 s ahead,
0.30 u at 3 s, 0.89 u at 5 s, 4.1 u at 10 s and 19 u at 30 s (ramped
rollout, v21.13 limits). DWA keeps only the cheapest candidate, so a term
that is the same for all of them cannot change the choice, whatever its
weight. Measured as the spread across candidates of each window's share of
the dynamic cost:

| window of the horizon | uniform | v21.13 | strong |
|---|---|---|---|
| 0–3 s | 0% | 0% | 0% |
| 3–8 s | 3% | 4% | 7% |
| 8–15 s | 15% | 18% | 22% |
| 15 s+ | **81%** | **77%** | **71%** |

(slalom, 386–395 states; the crowded and single-target runs give 79 / 75 /
68% for 15 s+.) The 1 s danger of the thought experiment adds the same amount
to every candidate: it has no say in the decision under any weighting. The
decision about it was made several seconds earlier, when it was the far
danger. What the weighting really trades is 8 s against 25 s, not 1 s against
10 s -- and even the strong discount leaves ~70% of the decision to the part
of the horizon beyond 15 s, where the candidates have spread furthest apart.

**(3) The weighting does change choices, but re-planning absorbs it.**
Re-scored open-loop, the discounted arms pick a different candidate from
their gain-matched uniform arms at 12–18% (v21.13) and 27–34% (strong) of
slalom states, and 12% / 26% of crowded and single-target states. The
differences are small: a median of 1 yaw-rate cell (0.0087 rad/s) for v21.13
and 5 for strong in the slalom, speed almost never. Each choice is executed
for 0.1 s and then re-planned, and in 0.1 s a ramped setpoint moves the yaw
rate by at most 0.0028 rad/s; the next decision corrects it. So the
executed paths converge, which is §10's outcome.

**Reading.** The intuition is right about *risk*: a close pass 2 s ahead is
worse than one 10 s ahead (fig1). It does not reach the *choice* in this
planner, for three measured reasons: real collisions are vetoed regardless
of time (though, measured, that rarely decides the choice); dangers too
close to change are equal for every candidate; and the ranking differences
that remain are small and re-planned away within 0.1 s.
The veto is a design decision, not a law -- a predicted contact at 25 s could
be a large finite, time-weighted cost -- but on these suites it is not the
binding constraint (above). Mechanisms (2) and (3) are.

## 12 Mechanism tests: forecasts that drift, and a near-vs-far dilemma

Both suites run the five arms with COLREGs off (`configs/algorithm/v21.13/mech_*.json`,
otherwise identical to the §1 arms), because the COLREGs terms are not
time-weighted and would fix the side. Analysis: `compare.py --arcs`,
`compare.py --dilemma`.

### 12.1 Forecasts that drift: steadily turning targets (`owarc215_*`)

A target turning at ω drifts from its constant-velocity forecast by about
½·v·ω·τ² at look-ahead τ -- 1.1 / 2.2 / 4.3 u at 25 s for 0.5 / 1 / 2 °/s.
That is the paper's "long-range predictions are less accurate", made a
controllable variable. *Phantom*: 25 s before the meeting the forecast says
collision, but the target turns clear. *Hidden*: a real collision that the
forecast 25 s earlier puts 1.1–4.2 u off. Straight control on each base.
85 runs, all reach the goal, **no contact**.

**The phantoms barely fool the planner -- less, the more the target turns.**
Wasted manoeuvre (max deviation from the obstacle-free path), uniform arm:

| base | straight (real threat) | phantom 0.5 °/s | 1 °/s | 2 °/s |
|---|---|---|---|---|
| head-on | 1.36 u | 0.42 | 0.13 | 0.07 |
| starboard crossing | 1.10 u | 0.50 | 0.22 | 0.07 |

*Corrected 2026-10-17.* An earlier version said the planner does not act on
forecasts beyond ~20 s. That misread §3's "lead time", which is when the
deviation first exceeds 0.25 u. The first non-zero yaw command comes **31–39 s
before the closest approach**, within a few seconds of the target entering
range, under every weighting (strong slightly later than uniform, by
0.4–2.6 s):

| scene | target in range | first yaw command: uniform / ×1.58 / v21.13 / ×3.15 / strong | deviation > 0.25 u |
|---|---|---|---|
| head-on | 38.7 s | 35.9 / 36.1 / 35.7 / 36.3 / 35.1 | 23.1 |
| starboard crossing 090 | 36.8 s | 32.7 / 32.8 / 32.3 / 32.9 / 31.1 | ~21.5 |
| port crossing 090 | 41.8 s | 37.7 / 38.2 / 37.6 / 39.0 / 36.6 | ~10.5 |
| overtaking | 44.9 s | 37.2 / 37.2 / 36.4 / 37.1 / 34.6 | ~4.3 |

So the planner does act on a phantom's far forecast -- but a far threat needs
only a small heading change (clearing 1.2 u at 30 s ahead takes ~4.6° at
0.5 u/s; at 10 s ahead, ~14°), and that small change is revised every 0.1 s
as the turning target's forecast updates. Acting on a wrong far forecast is
cheap here, under any weighting, so there is little over-commitment for a
discount to prevent.

**Hidden threats cost clearance as the turn sharpens** (head-on, uniform:
0.41 straight → 0.39 / 0.35 / 0.25 u; starboard crossing turning to
starboard: 0.54 → 0.46 / 0.39 / 0.26 u) -- in every arm, ordered by
effective gain.

At equal cost the discounted arms sit on the uniform curve: wasted deviation
at equal clearance −0.001 u (v21.13) and −0.004 u (strong) on the phantoms;
clearance at equal turning −0.007 / −0.009 u on the hidden threats.

### 12.2 A near-vs-far dilemma (`ow_dilemma_d10..d25`)

A, a slow vessel the ego overtakes, 1.2 u to starboard (a do-nothing ego
passes it 0.50 u clear); B head-on Δ s later, 0.5 u to port (a do-nothing ego
hits it), so starboard is the cheap side for B by geometry. Moving to
starboard while A is abeam would put the ego into A. 20 runs, all reach the
goal, no contact.

| Δ | arm | clearance A | clearance B | starboard move, s after A's CPA |
|---|---|---|---|---|
| 10 s | uniform / ×1.58 / v21.13 / ×3.15 / strong | 1.32 / 1.41 / 1.37 / 1.52 / 1.39 | 0.45 / 0.67 / 0.56 / 0.89 / 0.75 | +7.9 / +12.7 / +12.8 / +13.5 / +12.5 |
| 15 s | 〃 | 1.20 / 1.33 / 1.27 / 1.48 / 1.31 | 0.46 / 0.67 / 0.58 / 0.90 / 0.72 | +12.2 / +11.9 / +12.0 / +11.6 / +11.9 |
| 20 s | 〃 | 1.17 / 1.27 / 1.21 / 1.45 / 1.28 | 0.46 / 0.67 / 0.60 / 0.92 / 0.73 | +16.6 / +16.3 / +16.6 / +16.1 / +16.5 |
| 25 s | 〃 | 1.14 / 1.27 / 1.21 / 1.44 / 1.28 | 0.44 / 0.67 / 0.60 / 0.93 / 0.74 | +21.1 / +20.9 / +21.0 / +20.5 / +21.1 |

**No arm falls for the dilemma.** Every one, uniform included, holds clear of
A and moves for B 8–21 s *after* passing A; the move tracks B's timing, not
the weighting. While A is alongside, B is still far, and a far threat needs only a small
heading change -- which every arm defers until A is clear. At equal turning, the worse of the two clearances differs from
the uniform curve by −0.006 u (v21.13) and +0.018 u (strong).

### 12.3 Reading

Across every suite -- single threats, course-altering targets, two threats at
different times, drifting forecasts, a near-vs-far dilemma; 205 mechanism and
ablation runs, no contact -- the horizon weighting moves this planner along
the same clearance-vs-cost curve as the dynamic-obstacle gain. The reason is
structural and measured: a threat closer than ~4 s is the same for every
candidate (§11); a far threat moves the planner early (31–39 s ahead) but only
by small heading changes, revised every 0.1 s (§12.1); and where the weighting
does reorder candidates -- mid-encounter, where the discount factor varies by
15–57% across candidates -- the reordered choices differ by 1–5 yaw-rate cells
and are re-planned away (§11). The collision veto is the same for every arm;
removing it changes the choice at 0/216 states under uniform and 3/216 under
the strong discount (all head-on), so it does not drive the comparison.

## 13 Mean vs max: integrated vs extremum aggregation

The draft's other comparison: reduce each candidate's severity profile S(t)
by its time-weighted **mean** (integrated, duration-sensitive) or by its
**max** (extremum: the closest point of approach, duration-blind). 2 × 2 with
the time weighting: mean / max × uniform / v21.13's discount. Max has its
own gains (`v21.13/mech_max_{uniform,discount}_g{010,020,040}.json`);
mean + discount was already shown equal to mean + uniform at equal cost (§5)
and is not re-run. COLREGs off for the mechanism tests. `compare.py
--aggregation '<scene glob>'`.

**Why max can differ where the discount could not.** Open-loop at logged
v21.13 states, the mean/max spread ratio across candidates has a median of
0.13 (head-on) to 0.35 (being overtaken) and varies 20-fold between states
(10–90%: 0.02–0.65), so no single gain can turn max into mean; and max picks
a different candidate at 35–65% of states, a median of 1–15 yaw-rate cells
apart (90th percentile 16–36), against 1–5 for the discount.

### 13.1 Stage 0 (`owarc215_*_straight`, 12 runs)

All clean. Max gains 0.1 / 0.2 / 0.4 give clearances 0.42–1.12 u,
bracketing mean's 0.41–0.96; at equal cost all four families within
~0.05 u of the mean curve.

### 13.2 Target-speed sweep (`owspd2111_*`, 72 runs)

Same collision legs, target at 0.2 / 0.4 / 0.8 / 1.2 u/s, all meeting the
ego at 90 s. Under mean a fast target is close for less of the 33 s
horizon, so its danger is diluted; under max it is not.

**Head-on -- the prediction holds.** Clearance per gain (lowest / middle /
highest):

| target speed | mean uniform | max uniform | max discount |
|---|---|---|---|
| 0.2 | 0.52 / 0.73 / 0.99 | 0.49 / 0.84 / 1.09 | 0.33 / 0.76 / 1.03 |
| 0.4 | 0.41 / 0.62 / 0.89 | 0.53 / 0.87 / 1.11 | 0.42 / 0.78 / 1.06 |
| 0.8 | 0.16 / 0.44 / 0.75 | 0.57 / 0.88 / 1.13 | 0.49 / 0.82 / 1.07 |
| 1.2 | **0.02** / 0.27 / 0.64 | 0.59 / 0.87 / 1.12 | 0.46 / 0.83 / 1.07 |

At equal cost the families share one trade-off curve at every speed (max
minus mean: +0.01 to +0.07 u at equal turning or delay) -- max is not more
efficient. What differs is where a **fixed gain** lands on it: mean's effort
falls as the target speeds up (lowest gain: 75° → 70° → 50° of extra heading
change from 0.2 to 0.8 u/s), and at 1.2 u/s it reacts so late that it ends
with both the largest manoeuvre (2.14 u, 107°) and a near graze (0.02 u).
Max's clearance at a fixed gain stays within 0.49–0.59 u (lowest gain) from
0.2 to 1.2 u/s. A planner is deployed with one gain; with mean, that gain
cannot be right for slow and fast traffic alike. **Max is roughly invariant
to target speed; mean is not.**

**Starboard crossing at 0.8 and 1.2 u/s -- a COLREGs-off artefact.** Every
family loses its route there: the ego turns with the fast crosser (to port,
the way the target is going), runs alongside it and strays > 9 u off the
path, so the follower aborts (OPEN_PROBLEMS item 15). Two runs graze it
first, 0.003–0.011 u deep: mean uniform 0.98 at 1.2 u/s (t = 88.6 s) and max
discount 0.1 at 0.8 u/s (t = 101.2 s). With COLREGs on (v21.13) both scenes
reach the goal, passing astern as the give-way vessel at 0.81 / 0.61 u: the
side rule prevents a failure the cost alone allows, under either
aggregation. These two scenes are excluded from the mean-vs-max judgement.
At 0.2 and 0.4 u/s the crossings behave like head-on (all families on one
curve).

### 13.3 Long exposure (`owenc215_overtake*`, `owexp2111_converge_stbd_*`, 45 runs)

Overtaking, being overtaken, and a shallow convergence at the ego's own speed
(25° / 45°, closing 0.22 / 0.38 u/s): the target stays close for a long
time. `s<1u` is the time the ego spends within 1 u of a target.

| scene | mean uniform clearance (gains low / mid / high) | max uniform clearance | s<1u: mean vs max uniform |
|---|---|---|---|
| ego overtakes a slow vessel | 0.70 / 0.87 / 1.08 | **0.04** / 0.54 / 0.87 | 11 / 6 / 0 vs 14 / 13 / 10 |
| ego overtaken | 0.71 / 0.92 / 1.15 | **0.05 / 0.05** / 0.92 | 10 / 5 / 0 vs **18 / 19** / 8 |
| ego overtaken fast | 0.61 / 0.81 / 1.05 | **0.04** / 0.61 / 0.93 | 10 / 5 / 0 vs 8 / 14 / 8 |

Max discount is worse still: at 0.1 it **grazes** the vessel it overtakes
(t = 83.2 s, 0.0015 u deep); at 0.4 it spends 38 s within 1 u of the vessel
overtaking it. At equal delay, being overtaken, max discount sits 0.66 u
below the mean curve.

**This is the mirror image of §13.2.** Mean's cost grows with how long the
ego is close; max's depends only on the closest point. A brief, fast threat
is diluted under mean (head-on at 1.2 u/s: 0.02 u); a long, slow,
near-parallel one costs max nothing more once its closest point is set
(overtaking: 0.04–0.05 u, up to 19 s within 1 u). The max gains that matched
mean on head-on are too weak for overtaking; mean's, too weak for fast
traffic. **Complementary failure modes: no single gain fixes either
aggregation across both kinds of encounter.**

The convergences abort under every family with COLREGs off (the ego runs
alongside the target, heading −64° / −45° against the target's −65° / −45°,
and strays > 9 u off the path, item 15) 18–30 s after the closest approach.
The encounter phase, logged before the abort, shows the same pattern: max,
discounted above all, allows more time close (up to 11 s within 1 u against
0 for mean).

