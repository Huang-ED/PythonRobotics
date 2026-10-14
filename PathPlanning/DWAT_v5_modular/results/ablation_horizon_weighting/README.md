# Horizon weighting: does discounting beat uniform averaging?

The paper's third contribution is to aggregate threat severity over the
prediction horizon as an **exponentially discounted** expectation rather than a
**uniform** one, claiming it (a) gives "earlier and smoother evasion" and (b)
avoids "over-commitment to distant, low-confidence predictions" (`main.tex`,
lines 39, 119, 132). This folder tests both claims on v21.13.

**Answer, 2026-10-14: neither holds on these suites.** Once uniform averaging
is given a matched gain, discounting buys nothing at equal cost. What the
discount does is weight the near horizon more heavily, and since every
encounter ends in the near horizon, that acts as a larger gain on the dynamic
obstacle cost.

**Contents:** [1 Arms](#1-arms) · [2 Suites](#2-suites) ·
[3 Result: discount alone](#3-result-the-discount-alone) ·
[4 Why: normalised weights amplify the near term](#4-why-normalised-weights-amplify-the-near-term) ·
[5 Result: against a gain-matched uniform](#5-result-against-a-gain-matched-uniform) ·
[6 Result: when the forecast is wrong](#6-result-when-the-forecast-is-wrong) ·
[7 What this means for the paper](#7-what-this-means-for-the-paper) ·
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
matched to, at more turning. To compare at **equal cost**, the three uniform
gains trace a clearance-vs-cost curve per scene, and each discounted arm is
compared with that curve at its own cost (`compare.py --frontier`; both
suites, scenes without a manoeuvre excluded):

| discounted minus uniform clearance, at equal | v21.13 (λ·T = 1) | strong (λ·T = 3) |
|---|---|---|
| turning | **−0.012** u (better in 6/18 scenes, worse in 8) | **+0.002** u (8/18 better, 7 worse) |
| delay | **−0.023** u (6/20 better, 9 worse) | **−0.025** u (4/17 better, 10 worse) |

At equal turning or equal delay, discounting and uniform averaging are
indistinguishable: mean differences of 0.002–0.025 u against clearances
around 1 u, and the scenes split both ways.

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

- **"Discounted beats uniform" cannot be claimed.** On 22 scenes, five arms
  and 110 runs, the discount is equivalent to a gain change. The "earlier and
  smoother" wording (line 39) and the "over-commitment" argument (line 132)
  are not supported.
- **The time weight is still a legitimate design parameter**, just not a
  contribution: λ reshapes the cost towards the near term, which a reader can
  equally obtain with the gain.
- **What remains testable of contribution 3 is integrated vs extremum** --
  the draft's other comparison (line 132): an expectation over the horizon
  against `"max"` aggregation, i.e. the worst instant. That needs the same
  gain-matched design, because the max of S is roughly an order of magnitude
  larger than its mean when the threat fills a small part of the horizon.

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
