# PROMPT: Write my Slalom Challenge 2026 report

You are helping me write the final report for my individual assignment in the KAIST course **"Learning-Based Control for Mobility Systems" (2026)**: the **Slalom Challenge**. Everything below is my own project record: what I built, what I tested, what failed, what I measured and what I concluded. Use **only** these facts and numbers. Do **not** invent results, numbers, references or experiments. If something is missing, write a clearly marked placeholder like `[TODO: ...]`.

---

## 0. What you must produce

1. A report in **LaTeX** using the professor's **IEEE-style two-column template** (`report_template.tex` from the course repository), which I will compile in Overleaf into `report.pdf`.
2. The four **required sections** (from `SUBMISSION_AND_AI.md`):
   1. **Problem Statement**: state, actions, constraints, objective, and the relation of my training/planning objective to the official minimum-time task.
   2. **Proposed Approach**: course concepts implemented (DP, Bellman recursion, state augmentation, policy extraction, model-based design, tuning), model/data, offline computation, online action selection. **At least one representative figure is mandatory.**
   3. **Own Validation**: PyChrono outcomes and times, meaningful baseline comparison, supporting evidence, limitations and failures.
   4. **Conclusion**: result plus a brief **AI-use disclosure** (tool, model, main uses).
3. Length: the professor **recommends two pages; length is not graded**. I want to show **all my effort** (both modes, all tests, failures, fixes), so: keep the main text tight (about 2–3 pages) and put the large tuning tables and extra figures in an **Appendix**.
4. Tone: honest, technical, first person ("I"). Every design number must have a justification (physics, the professor's documents, or a PyChrono test). Clearly separate **(a) official rules**, **(b) the professor's suggested values**, and **(c) my own design choices**.
5. Suggest the figures (with the data source and a short plotting description) and draft the captions.
6. The report must be consistent with my code and results. I will be asked about it in a **5-minute individual Q&A on November 3, 2026**, so explanations must be ones I can defend.

Header placeholders: Name `[MY NAME]`, Student ID `[MY ID]`, email submission to fairytale@kaist.ac.kr, deadline **November 1, 2026, 23:59 KST**.

---

## 1. The challenge (official facts, from the course repository)

- Repository: `github.com/Kyunghwan-Choi/slalom-challenge-2026`, **commit `e05dace`** (2026-10-04); no upstream changes since.
- Plant: **PyChrono 10.0.0, BMW E90, TMeasy tires**, level rigid terrain, friction **μ = 0.9**. Physics step 0.001 s, control step **0.02 s**.
- Course: 8 cones (radius 0.25 m) at **X = 20 + 15j m, Y = 0** (j = 0…7). Pass alternately: **cone 0 on +Y (left), cone 1 on −Y (right), …** Finish at **X_REF = 145 m**. Road **−5.5 ≤ Y ≤ 5.5 m**. Timeout 60 s. Start: nominal REF (0,0), ψ = 0, then 0.8 s settling with brake (the measured initial state is about X = −0.015 m, vx = 0.17 m/s).
- **REF** point = 1.371 m ahead of the COM (approximately the front axle; lf = 1.345 m, wheelbase L = 2.776 m). Gates and finish use REF. Collision uses a **4.7 m × 1.9 m rectangle centred at COM**.
- Failure if: cone contact or footprint overlap; any part of the footprint leaves the road; wrong-side gate; vx > 12 or REF X-progress speed > 12 m/s; vx < −1; |ψ| > 1.3 rad; controller error; no finish in 60 s.
- **Official objective**: minimise finish time t_f subject to dynamics, bounds, collision/road constraints, ordered forward gate crossings with σ_j(Y_REF(t_j) − Y_j) > 0. **There is no required lateral gate distance and no required clearance margin; the only rule is clearance > 0.**
- Two modes:
  - **Easy**: submit `easy_plan.json`, a list of integer moves [m, n] on a fixed lattice (Δ = 0.5 s, h_X = 1 m, h_Y = 0.5 m, m ∈ {1..5}, n ∈ {−4..4}, ≤ 120 stages). The professor's fixed `EasyTracker` drives it. "Suggested" DP bounds in `easy_spec.json` (|Y| ≤ 3 m, V ≤ 11.5, |χ| ≤ 0.58 rad, accel ≤ 4, decel ≤ 6, |Δχ| ≤ 0.31 rad, V|Δχ|/Δ ≤ 6 m/s²) **may be tightened or relaxed** ("Relaxing a bound can shorten the DP plan while making it harder to drive").
  - **Full**: submit `controller.py` with `Controller.reset/act`, returning `{"steering" ∈ [−0.8, 0.8], "acceleration" ∈ [−7, 7] m/s²}` every 0.02 s. Input adapter: steering slew 0.04 per 0.02 s; acceleration converted open-loop to throttle/brake ≤ 0.6, clipped to the speed-dependent `acceleration_bounds`. The professor supplies a control-oriented model `dynamics.predict` (planar bicycle, fitted at μ = 0.9).
- Assessment: primarily finish time (a failed run gets the minimum performance score); course-based design and understanding (code, report, Q&A); a generic RL library as the solution gets zero credit (I used none).
- Reference points from the professor: the 5 m/s example Full controller takes **30.660 s**; the professor's own Easy DP results: 0.5 m gate margin → 31 stages, cone contact at 3.384 s; 2.5 m → 48 stages, success **24.074 s**, clearance 0.734 m. Example plans: `easy_success_73.json` → 36.548 s; `easy_failure_73.json` → cone contact at 5.341 s.

---

## 2. Environment, setup and practical difficulties (include briefly, they are part of my effort)

- Windows x64, VS Code, **Miniconda (conda 26.7.1)**, environment `slalom2026` from the professor's lock file: **Python 3.12.14, PyChrono 10.0.0, NumPy 2.5.3**. Installation checks passed (`test_contract.py` 8 tests OK, `test_easy_mode.py` 5 tests OK; example controller 30.660 s, success).
- Difficulties solved:
  1. `conda` not recognised in PowerShell/VS Code → `conda init powershell` + `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
  2. The repository ZIP extracted into a nested folder → ran everything from the inner folder.
  3. **Windows "Application Control policy" (Smart App Control) intermittently blocked DLLs** of NumPy (`_multiarray_umath`) and PyChrono (`_core`) ("DLL load failed … An Application Control policy has blocked this file"). Not a code problem; it resolved, and I noted the security setting as a risk.
  4. The base environment has **no Numba**, so `dynamics.predict` is slow (measured ≈ 0.31 ms per call, ≈ 0.16 ms per `step`, on the cross-check machine). This shaped my Full-mode design (see §5.1).
- **Cross-platform verification**: the same plans and controllers were also re-run on an independent Linux machine with PyChrono 10.0.0 (done with AI assistance). Results agreed to about 1 ms and about 0.5 mm, e.g. the 48-stage plan gave 24.04603 s / 0.73829 m (my Windows PC) vs 24.04545 s / 0.73776 m (Linux). This is my evidence that results are reproducible and that a small clearance buffer suffices.

---

## 3. Understanding phase (what I learned before designing)

- **Professor's Easy examples**: wide 2.5 m plan → success 36.548 s, min clearance 0.485 m, actual |Y_REF| at gates only 2.08–2.24 m (the tracker cuts corners); narrow 0.5 m plan → **cone contact at 5.341 s**, min clearance −0.002 m (the car stopped at X = 18.77 m, Y = 0.50 m). Lesson: **passing on the correct side is not enough; the body must clear the cone.**
- **Geometry of the minimum lateral distance (straight car)**: half width 0.95 m + cone radius 0.25 m = **1.2 m**. Clearance ≈ lateral REF offset − 1.2 m for a straight car (checked with the professor's `footprint_clearance`: offset 1.2 → 0.00; 1.5 → 0.30; 2.0 → 0.80; 2.5 → 1.30 m). Real clearance is smaller because (i) the tracker cuts corners and (ii) the car is turning, so its **rear corner swings toward the cone after REF has passed it**. Per-cone analysis of the 48-stage plan: the closest approach occurs when REF is about **2.5 m past** each cone (minimum 0.738 m at cones 3 and 5; per cone: 1.016, 0.914, 0.819, 0.738, 0.825, 0.738, 0.825, 0.932 m).
- **Model check (Full mode, `try_predict.py` on the example run, row 500 = 10 s)**: 1-step (0.02 s) errors X 2e-6 m, Y −9.3e-5 m, vx −0.004 m/s; 50-step (1 s) open-loop errors X −0.088 m, Y −0.026 m, vx −0.18 m/s (the known light-throttle speed bias around 5 m/s documented in `MODEL.md`). The professor's tables at 10–12 m/s: lateral RMSE 0.029 / 0.058 / 0.166 m at 0.5 / 1 / 2 s. Conclusion: the model is reliable for **short horizons with measured-state feedback**, which motivates keeping clearance buffers and feedback every 0.02 s.
- **Physical limits computed from the professor's model** (`contract.acceleration_bounds`): max forward acceleration ≈ 1.8–2.0 m/s² near standstill, ≈ 4.6–5.1 m/s² at 3–6 m/s, ≈ 2.4–2.6 m/s² near 10–11.5 m/s. Straight-line time floor (no cones, max acceleration): **≈ 15.8 s with a 10 m/s cap (Easy grid limit)**, **≈ 14.1 s with 11.5 m/s**.
- **Tire capacity**: the fitted model has `mu_scale = 1.159`, so the lateral capacity ≈ 0.9 × 1.159 × 9.81 ≈ **10.2 m/s²**. Measured peak lateral acceleration (vx·r + dvy/dt) in my best runs: **8.76 m/s² (Easy best), 9.24 m/s² (Full v1)** (99th percentiles 8.34 / 8.91). IMPORTANT: earlier I reasoned "μg = 8.8 m/s² is the grip limit". **That is not correct for this model and must NOT appear in the report.** The correct statement: bounds of 9–10 m/s² failed because the tracking controller could not follow such sharp references (trackability), not because of tire friction alone.

---

## 4. EASY MODE: my method and results

### 4.1 Formulation (finite-horizon / free-final-time DP on the published lattice), file `easy_dp.py`
- State z = (X index, Y index, previous move p) where p ∈ {45 moves} ∪ {START = at rest}. The previous move is needed because the bounds compare V_k, χ_k with V_{k−1}, χ_{k−1} (state augmentation; the professor's state is (X, Y, V_{k−1}, χ_{k−1})).
- Actions u = (m, n), 45 moves. V = √((m h_X)² + (n h_Y)²)/Δ, χ = atan2(n h_Y, m h_X).
- Constraints: `pair_allowed` (speed, angle, acceleration, deceleration, angle change, virtual lateral acceleration) and `position_allowed` (road strip |Y| ≤ max_abs_y; for every cone plane crossed by the move, interpolated Y_cross with λ = (X_j − X_k)/(X_{k+1} − X_k) and signed offset d_j = σ_j(Y_cross − Y_j) ≥ gate_margin; optional `body_margin` footprint check using the professor's `footprint_clearance` at 4 points along the segment).
- Cost: 1 per stage (plus a tiny tie-break 1e-4·(Δn)² that prefers smoother plans among equal-stage plans and can never add a stage); infeasible moves are excluded (equivalent to the professor's failure cost M > K_max); finish states (X ≥ 145) are absorbing with zero cost.
- Solver: **backward Bellman recursion over X** (X strictly increases, so the recursion is well ordered; this is equivalent to the professor's finite-horizon DP since the 120-stage horizon is never binding): J(ix, iy, p) = min_u [1 + J(next)], with policy = argmin; the plan is extracted by following the policy forward from (0, 0, START). Command-line interface for every bound.
- **Verification**: my DP reproduces the professor's published numbers exactly: **48 stages at 2.5 m margin and 31 stages at 0.5 m** with the suggested bounds. The 48-stage plan drove **24.046 s, clearance 0.738 m** (professor: 24.074 s). First moves of the 48-stage plan: [1,0], [2,0], [2,1], [2,2], [3,2], [4,1] (accelerate gently, shift left early for cone 0).
- Theoretical stage floor with the 4 m/s² bound: 1+2+3+4 = 10 m in 4 stages, then 27 stages at 5 m → **31 stages minimum** (29 without an acceleration bound, but physically irrelevant).

### 4.2 Tuning by simulation (outer loop: DP bounds → PyChrono → choose)
All results below are PyChrono runs of DP plans (other bounds = suggested values unless stated).

Gate-margin sweep:
| gate margin | stages | result | time (s) | min clearance (m) |
|---|---|---|---|---|
| 1.0 | 31 | cone contact | – | −0.002 |
| 1.5 | 32 | success | 16.477 | 0.008 |
| 1.75 | 37 | success | 19.121 | 0.246 |
| 2.0 | 38 | success | 19.655 | 0.328 |
| 2.25 | 46 | success | 23.121 | 0.428 |
| 2.5 | 48 | success | 24.046 | 0.738 |

Other tests: road strip 3.5/4.0 m → no change (19.126 / 19.59 s). Footprint (body_margin) constraint with gate 1.2: 0.3 → 34 stages, 17.585 s, 0.078 m; 0.5 → 37, 19.126 s, 0.246 m; 0.7 → 38, 19.657 s, 0.328 m.

Turning-bound relaxation (angle change / virtual lateral acceleration):
| gate | Δχ max | a_lat max | stages | result | time (s) | clearance (m) |
|---|---|---|---|---|---|---|
| 2.0 | 0.31 | 6 | 38 | success | 19.66 | 0.33 |
| 2.0 | 0.40 | 7 | 34 | success | 17.53 | 0.45 |
| 2.0 | 0.40 | 8 | 32 | success | 16.54 | 0.47 |
| **2.0** | **0.50** | **8** | **32** | **success** | **16.33** | **0.44** |
| 2.0 | 0.60 | 8 | 32 | success | 16.33 | 0.44 (same plan, Δχ no longer binding) |
| 1.75 | 0.50 | 8 | 32 | success | 16.51 | 0.47 |
| 2.25 | 0.50 | 8 | 35 | success | 18.02 | 0.72 |
| 2.25 | 0.50 | 9 | 34 | success | 17.58 | 0.70 |
| 2.5 | 0.50 | 8 | 36 | success | 18.53 | 0.57 |
| 2.5 | 0.60 | 8 | 36 | success | 18.56 | 0.63 |
| 2.5 | 0.50 | 10 | 34 | success | 17.59 | 0.92 |
| 2.5 | 0.60 | 10 | 34 | success | 17.62 | 0.59 |
| 2.5 | 0.50 | none | 33 | success | 17.18 | 0.93 |
| 2.5 | 0.60 | none | 32 | cone contact | – | – |
| 2.0 | 0.50 | 10 | 31 | cone contact | – | – |
| 2.0 | 0.50/0.60 | none | 31 | cone contact | – | – |
| 1.75 | 0.50 | 9 | 31 | cone contact | – | – |

Key findings: (1) the suggested turning bounds were conservative: relaxing them made the plan **faster and safer** (19.66 → 16.33 s, clearance 0.33 → 0.44 m), because the car reaches the side quickly instead of drifting near the cone; (2) **every 31-stage plan failed**: the professor's tracker cannot follow the sharpest weaving, so about 32 stages is the practical floor in Easy mode; (3) the DP model is not the truth: plans that are feasible on paper can crash, so PyChrono validation is essential.

Justification of the relaxed bounds: "Rule A" (Δχ ≤ const) and "Rule B" (V|Δχ|/Δ ≤ a_lat ⇒ Δχ ≤ a_lat·Δ/V) both limit turning per stage and the smaller one is active. With the suggested values (0.31, 6), the allowed turn was 0.30 rad at 10 m/s and a fixed 0.31 rad at 8/6/4 m/s; with (0.5, 8) it is 0.40 / 0.50 / 0.50 / 0.50 rad, which makes the speed-dependent lateral bound active at high speed. Physically plausible: below the fitted tire capacity (≈ 10.2 m/s²); the measured peak lateral acceleration was 8.76 m/s². Bounds of 9–10 failed in PyChrono (trackability).

### 4.3 Selected Easy plan (`plan_best.json`; submitted as `easy_plan.json` if Easy is evaluated)
- Bounds: gate_margin 2.0 m, max_angle_change 0.5 rad, max_lat_accel 8 m/s², the others suggested. **32 stages (16.0 s virtual) → PyChrono 16.333 s, 8 gates, min clearance 0.440 m.**
- Gate crossings every ≈ 1.51 s (≈ 9.9 m/s through the slalom); |Y_REF| at gates 1.85–1.90 m (planned 2.0 m: corner cutting ≈ 0.14 m); finishes at Y = −3.0 m, vx = 10.0 m/s (the DP does not waste time returning to the centre).
- **Selection rule (mine)**: the fastest PyChrono success with min clearance ≥ 0.3 m. It is also the fastest successful Easy run overall, so the choice holds for any threshold up to 0.44 m. Why 0.3 m: the cross-platform variation was ≈ 0.5 mm, and the `MODEL.md` lateral prediction RMSE is 0.03–0.17 m, so 0.3 m is a conservative buffer (about 2× the largest model error).

### 4.4 Easy robustness test (my own validation tool `robustness_test.py`)
Idea: keep the plan fixed and move **every cone toward the side the car passes** by a shift s (local course copies via `cone_centers_m`, as the professor allows for local experiments). The largest shift that still succeeds = real safety margin.
| shift (m) | Easy best (16.33 s) clearance | Easy 48-stage (24.05 s) clearance |
|---|---|---|
| 0 | 0.440 | 0.738 |
| 0.2 | 0.247 | – |
| 0.3 | 0.150 | – |
| 0.4 | 0.054 | 0.339 |
| 0.5 | **cone contact** | – |
| 0.6 | – | 0.140 |
| 0.8 | – | **cone contact** |
The clearance decreases about 1:1 with the shift and failure occurs at shift ≈ measured clearance, so `min_clearance_m` is a trustworthy safety measure. The finish time does not change (Easy plans are open-loop with respect to cone positions).

Other tool: `clearance_by_cone.py` (closest gap per cone from `trajectory.npz`, using the professor's `footprint_clearance`).

---

## 5. FULL MODE: my method and results

### 5.1 Design rationale
- Full mode removes the Easy limits (10 m/s grid cap; fixed generic tracker) and allows up to 12 m/s with my own controller.
- Architecture (allowed by `Q_AND_A.md`: "A learned planner may choose targets for your path or speed tracking controller"): **(1) offline backward-DP planner** producing a minimum-time REF reference path with speeds (`full_dp.py` → learned artifact `full_reference.npz`), and **(2) an online feedback tracking controller** every 0.02 s (`my_controller.py`, to be submitted as `controller.py`).
- Why not online rollout/MPC with `predict` now: about 0.3 ms per call without Numba; even 5 candidates × 50 steps = 75 ms per decision > the 20 ms control period. Rollout would need my own vectorised model; postponed (see Limitations / Future work).

### 5.2 Planner `full_dp.py` (same backward Bellman recursion as Easy, own lattice, model-based bounds)
- Final lattice (v2): Δ = 0.5 s, h_X = **0.25 m**, h_Y = 0.25 m, m = 1..23 (0.25–5.75 m per stage → speed resolution 0.5 m/s, so 11.5 m/s is reachable), n = −8..8 (±2 m per stage); **391 moves**; |Y| ≤ 4 m. (v1 used h_X = 0.5 m, m = 1..11, 187 moves, giving 1 m/s speed steps, so the planner could only use 11 m/s.)
- Bounds: max_speed 11.5 m/s (rule 12); max |χ| 0.6 rad; **acceleration from the vehicle model**: the time to accelerate from V_{k−1} to V_k, t = ∫ dv / a_max(v) with a_max from `contract.acceleration_bounds`, must be ≤ Δ (accel_scale 1.0); decel ≤ 6; virtual lateral acceleration ≤ 8 m/s²; gate margin 1.2 m (= touching distance of a straight car; only a floor); **footprint-to-cone clearance ≥ 0.45 m** (vectorised copy of the professor's `footprint_clearance` at 5 points along each segment, heading = segment angle, cones within 7 m); **footprint-to-road-edge clearance ≥ 0.8 m** (vectorised copy of `inside_road`).
- Objective: 1 per stage, except that **the stage crossing the finish line costs only the fraction needed to reach X = 145 m** (so the DP minimises the actual time to the line, not just a stage count); tie-break 1e-6·(Δn)².
- Implementation: all constraint tables computed with NumPy broadcasting (≈ 7.5 M position-move combinations); DP ≈ 4 s (v1) / ≈ 25 s (v2).
- Development difficulties: (a) evaluating the acceleration bound at the start speed (a_max(0) ≈ 1.98 m/s²) made the problem infeasible from rest, then the mid-speed version too; fixed by the **integral time-to-accelerate** formulation; (b) accel_scale 0.9 was infeasible from rest (reaching 1 m/s needs 0.459 s ≥ 0.9·0.5 s), so I used 1.0; (c) a Δ = 0.25 s lattice was infeasible from rest (the car cannot cover one 0.25 m cell in the first 0.25 s), so I kept Δ = 0.5 s.
- Result: v1 lattice: 30 stages (15.0 s virtual; 14.95 s with the fractional finish). **v2 lattice: 30 stages, 14.65 s virtual.**

### 5.3 Controller `my_controller.py` (online, every 0.02 s)
- `reset`: loads `full_reference.npz` and the fitted steering gain (sg = 0.5514, sv = −0.00042 from `model_parameters.json`); densifies the 31-node polyline to 0.1 m spacing.
- `act`: (1) nearest path point (forward-only search); (2) **pure pursuit** from the rear axle (REF ≈ front axle, so rear axle ≈ REF − L(cos ψ, sin ψ)) to a target point `look = LOOK_BASE + LOOK_GAIN·vx` ahead along the path: wheel angle δ = atan(2L·l_y / (l_x² + l_y²)); steering request = δ / (sg + sv·vx²), clipped to ±0.8; (3) **speed tracking**: target = planned segment speed 3 m ahead (SPEED_PREVIEW), capped at 11.6 m/s; acceleration request = 2.5·(target − vx), clipped to ±7. Returns plain Python floats.
- Computation: ≈ 0.09–0.10 ms median per call (budget 20 ms).
- SPEED_GAIN 2.5 and SPEED_PREVIEW 3 m were not tuned (they worked first time); say so honestly.

### 5.4 Full-mode tuning and failures (all PyChrono)
First attempt (v1 route, body 0.45, lookahead 2.0 + 0.35·v ≈ 5.5 m at 10 m/s): **cone contact at cone 2** (X = 48.65 m, 10.2 m/s); the long lookahead cut corners (gate 1 passed at Y = −1.31 m).

v1 sweep (lattice v1):
| body margin | a_lat | look base/gain | time (s) | clearance (m) | note |
|---|---|---|---|---|---|
| 0.45 | 8 | 1.0/0.2 | 15.215 | 0.267 | |
| 0.45 | 8 | 0.5/0.2 | 15.250 | 0.367 | |
| 0.45 | 8 | 1.0/0.1 | 15.283 | 0.452 | |
| **0.45** | **8** | **0.5/0.1** | **15.323** | **0.523** | **v1 chosen** |
| 0.45 | 9.5 | (same four) | 15.233–15.365 | 0.268–0.523 | all success |
| 0.6 | 8 | 0.5/0.1 … 2.0/0.2 | 15.633–15.818 | 0.209–0.740 | all success |
| 0.6 | 9.5 | 0.5/0.1 | – | – | road departure |
| 0.6 | 9.5 | 0.5/0.2 | – | – | cone contact |
| 0.6 | 9.5 | 1.0/0.1 | – | – | road departure |
| 0.6 | 9.5 | 1.0/0.2 | 15.306 | 0.539 | |
| 0.8 | 8 | 1.0/0.2 | 16.236 | 0.663 | |
| 1.0 | 8 | 1.0/0.2 | 16.735 | 0.719 | |
Findings: a shorter lookahead gives less corner cutting and more clearance; a larger body margin is slower; a_lat = 9.5 is unreliable, so 8 is kept (trackability).

**Full v1 result** (my Windows run): **15.324 s, 8 gates, clearance 0.525 m**; gates at 3.40, 4.79, 6.26, 7.74, 9.21, 10.68, 12.11, 13.51 s (≈ 1.46 s per gate ≈ 10.3 m/s); finish vx 11.05 m/s, Y −4.02 m. Gains vs Easy: ≈ 0.37 s at the start (model-based acceleration), ≈ 0.46 s in the slalom, ≈ 0.18 s on the final 20 m. Robustness: shift 0.3 → 0.225 m; 0.5 → 0.026 m; **0.6 → cone contact**.

v2 changes: finer speed lattice, fractional finish cost, road-edge check, then driver retuning. **Important failure**: with the faster v2 route (≈ 11 m/s through the cones) the short lookahead (0.5 + 0.1·v ≈ 1.6 m) became **unstable: steering oscillations grew, steering saturated at ±0.8, heading reached ≈ 0.6 rad, and the car left the road after cone 6** (also reproduced on my PC when I forgot to change LOOK_GAIN: road departure at 12.556 s, 7 gates). Tightening the road margin (0.8/1.2) did not help, which shows the cause was tracker stability, not the route. Lesson: **the lookahead must be tuned together with the reference speed.**

v2 lookahead sweep (body 0.45):
| look base/gain | time (s) | clearance (m) |
|---|---|---|
| 0.5/0.1 | – | road departure (unstable) |
| 0.5/0.15 | 15.101 | 0.414 |
| **0.5/0.2** | **14.987** | **0.380** |
| 0.5/0.25 | 14.929 | 0.320 |
| 0.5/0.3 | 14.877 | 0.187 |
| 1.0/0.15 | 14.994 | 0.385 |
| 1.0/0.2 | 14.934 | 0.334 |
| 1.0/0.25 | 14.881 | 0.201 |
| 1.0/0.3 | 14.833 | 0.054 |
(body 0.55: 15.142–15.327 s, clearance 0.277–0.613 m, all success.)
Choice of 0.5/0.2: its neighbours on both sides also succeed (not on a stability edge), the clearance stays at Easy level, and faster settings trade large clearance for a few hundredths of a second.

**Full v2 result (final candidate)** (my Windows run): **14.985 s, 8 gates, clearance 0.377 m** (Linux cross-check 14.987 s / 0.380 m). Gates at 3.371, 4.755, 6.162, 7.562, 8.978, 10.428, 11.856, 13.243 s (≈ 1.41 s per gate ≈ 10.6 m/s); |Y_REF| at gates 1.72–2.04 m; finish vx 11.53 m/s, Y −1.97 m; median action 0.09 ms.
Robustness (cone shift): 0 → 0.377; 0.2 → 0.177; 0.3 → 0.077; 0.35 → 0.027; **0.4 → cone contact**.

---

## 6. Overall comparison (use as the main results table)
| solution | mode | time (s) | min clearance (m) | survives cone shift up to |
|---|---|---|---|---|
| Professor's example controller | Full | 30.660 | 0.881 | – |
| Professor's example plan (73 stages) | Easy | 36.548 | 0.485 | – |
| Professor's DP (2.5 m, from RUNNING.md) | Easy | 24.074 | 0.734 | – |
| My DP, 2.5 m margin (48 stages) | Easy | 24.046 | 0.738 | ≈ 0.7 m |
| My tuned DP (32 stages) | Easy | 16.333 | 0.440 | ≈ 0.45 m |
| My DP + pure pursuit, v1 | Full | 15.324 | 0.525 | ≈ 0.5 m |
| **My DP + pure pursuit, v2** | **Full** | **14.985** | **0.377** | **≈ 0.35 m** |
Physical context: straight-road floor ≈ 15.8 s at 10 m/s (Easy cap) and ≈ 14.1 s at 11.5 m/s.

---

## 7. Course concepts used (map explicitly in the Approach section)
- **Problem formulation**: the constrained free-final-time minimum-time problem → a discrete stage problem; stage count (Easy) / time to the finish line (Full, fractional last stage) as the cost; constraints as infeasible transitions (failure cost M); absorbing finish.
- **Dynamic programming / Bellman equation**: backward recursion J(x) = min_u [g(x,u) + J(f(x,u))], ordered by the monotone X; **state augmentation** with the previous move (the bounds depend on V_{k−1}, χ_{k−1}); the **policy** as a state-feedback table, extracted by forward rollout of the policy.
- **Model-based design**: the professor's vehicle/input-adapter model gives the acceleration bound (time-to-accelerate integral); footprint and road geometry are exact copies of the evaluation checks.
- **Outer-loop tuning in simulation** (grid search of DP bounds and tracker gains against PyChrono; explicit selection rule).
- **Feedback control**: pure-pursuit path tracking plus proportional speed control as the online policy (can be described as a base policy).
- **Rollout / multistep lookahead**: analysed and motivated (with `predict` accuracy and timing measurements) but not implemented; proposed as the next step.
- No generic RL library, no neural networks.

---

## 8. Limitations (honest; put in Validation and/or Conclusion)
1. The Full controller **does not use the cone positions online**; it tracks a precomputed reference. On a different course the planner must be re-run (`full_dp.py` reads `course.json`). The finish time is identical under cone shifts, which shows this.
2. **Tracker stability depends on speed**: a lookahead tuned at 10.3 m/s became unstable at ≈ 11 m/s.
3. The DP uses approximations: segment (chord) headings for footprint checks, a virtual lateral acceleration over 0.5 s stages (the real peak lateral acceleration was 8.8–9.2 m/s² with a bound of 8), and an optimistic first stage from rest.
4. Easy is limited by the fixed grid (10 m/s) and the professor's tracker (corner cutting ≈ 0.14 m; all 31-stage plans crashed).
5. Speed-loop gains were not tuned; tuning was done by grid search on the official course plus cone-shift variants only (simulation deterministic, no noise).
6. `predict` is slow without Numba, which prevented online rollout in the available time.
7. Final clearance (0.377 m) is smaller than v1 (0.525 m): a deliberate speed/safety trade-off backed by the robustness test.

## 9. Possible future work
Online **rollout / MPC** using a vectorised copy of the model, checked against `dynamics.predict`, with real cone positions (adaptivity, safety); speed-adaptive lookahead or a yaw-rate-damped / Stanley-type tracker to allow tighter tracking at 11+ m/s; a finer time lattice with a special start-from-rest stage; tuning the speed loop; model check of my own Full run with `validate_requests.py` [TODO: run `python validate_requests.py --run runs\full_v2` and report the 0.5/1/2 s RMSE if time allows].

---

## 10. Figures to create (provide plotting code suggestions + captions)
1. **System diagram** (mandatory representative figure): offline DP planner (course.json, model bounds, footprint/road checks) → `full_reference.npz` → online controller (nearest point → pure pursuit → steering; speed target → P control → acceleration) → input adapter → PyChrono → measured state feedback.
2. **Top view of trajectories**: cones, road, Easy best plan nodes, Full v2 reference nodes, actual REF trajectories from `runs/best/trajectory.npz` and `runs/full_v2/trajectory.npz` (rows[:,0] = X, rows[:,1] = Y), with car rectangles at the closest-approach moments.
3. **Speed vs X**: actual vx (rows[:,3]) for the example controller, Easy best and Full v2.
4. **Trade-off scatter**: finish time vs min clearance for all successful tuning runs (Easy and Full), with failures marked.
5. **Robustness plot**: min clearance vs cone shift for Easy best, Full v1, Full v2 (lines cross zero at ≈ 0.45 / 0.5 / 0.35 m).

## 11. Submission facts to state or follow
- Evaluated result: **[TODO: confirm, recommended Full v2]**. In the email, give name, student ID and mode; if both modes are submitted, state which one to evaluate.
- ZIP: `report.pdf`; `controller.py` (= my_controller.py, LOOK_GAIN 0.2) + `full_reference.npz` + `full_dp.py`; Easy: `easy_plan.json` (= plan_best.json) + `easy_dp.py` and the exact command used (`--gate-margin 2.0 --max-angle-change 0.5 --max-lat-accel 8`); tools `robustness_test.py`, `clearance_by_cone.py`. Full DP command: `python full_dp.py` (defaults = final settings). No extra packages (NumPy only), so no requirements.txt is needed.
- [TODO: the controller currently reads `model_parameters.json` from its own folder; make it self-contained (store sg, sv in `full_reference.npz`) and re-verify the time before submission.]
- Record the repository commit `e05dace`.

## 12. AI-use disclosure (for the Conclusion)
I used **Claude (Anthropic), through the Claude app (configured model: claude-opus-5-5)** as a tutor and coding assistant: explaining the repository and concepts step by step, drafting and debugging code (easy_dp.py, full_dp.py, my_controller.py, robustness_test.py, clearance_by_cone.py), running cross-check simulations on a separate Linux setup, proposing tuning sweeps, and drafting/editing this report. I directed the work (mode order: Easy first, then Full; keeping benchmarks; adding the robustness test; requesting further optimisation; choosing the final trade-off), typed and ran all code on my own PC, reproduced every reported result myself, and checked the explanations (for example, I questioned and corrected the tire-grip justification). [Adjust the wording to my exact usage.]

---

Now, using all of the above, write the complete LaTeX report (main text plus appendix), the figure captions, and a short list of the figures/plots I still need to generate, with Python plotting code that uses my run folders.
