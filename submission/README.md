Slalom Challenge 2026 - Learning-Based Control for Mobility Systems
MD SYMUM HOSSAIN (ID: 20264789)

MODE TO EVALUATE: FULL
  Full : controller.py + full_reference.npz   -> 14.696 s, min clearance 0.414 m
  Easy : easy_plan.json (supporting work)     -> 16.333 s, min clearance 0.440 m
  (results from run_local.py on my PC, supplied slalom2026 environment)

FILES
  report.pdf           report (method, own validation, AI use)
  controller.py        Full-mode entry point (class Controller with reset/act)
  full_reference.npz   the only file controller.py loads, read from its own folder:
                       DP reference path (x, y), planned speeds (v), supplied fitted
                       steering map (steer_gain, steer_speed2), planner settings (bounds)
  full_dp.py           backward-DP planner that produces full_reference.npz
  easy_plan.json       Easy-mode plan, 32 stages (0.5 s each)
  easy_dp.py           backward-DP planner that produces easy_plan.json
  robustness_test.py   validation: drives a solution with all cones moved towards the car
  clearance_by_cone.py validation: closest body-to-cone gap per cone from a run

EVALUATION (no retraining; imports: math, pathlib, numpy only; no extra packages)
  python run_local.py --mode full --controller controller.py --output runs/full
  python run_local.py --mode easy --plan easy_plan.json --output runs/easy

HOW THE FILES WERE GENERATED
  Copy the .py files into the repository root and run them there
  (they read course.json, easy_spec.json, model_parameters.json, contract.py
  and run_local.py from the supplied repository).

  Easy plan (prints "stages: 32 (virtual time 16.0 s)"):
    python easy_dp.py --gate-margin 2.0 --max-angle-change 0.5 --max-lat-accel 8 --output easy_plan.json
    Other bounds = suggested values in easy_spec.json.

  Full reference (prints "stages: 29 (virtual time to the finish line 14.33 s)"):
    python full_dp.py --output full_reference.npz
    Default settings are the final ones: grid 0.25 x 0.25 m, 0.5 s stages,
    m <= 23, |n| <= 8; speed <= 11.8 m/s; acceleration limited by the input
    adapter's achievable a_max(v); lateral acceleration <= 8 m/s^2;
    gate offset >= 1.2 m; car body >= 0.4 m from every cone;
    body >= 0.8 m inside the road edge; |Y| <= 4 m.

  Validation:
    python robustness_test.py --controller controller.py --shifts 0 0.2 0.3 0.4 0.45
    python robustness_test.py --plan easy_plan.json --shifts 0 0.2 0.3 0.4 0.5
    python clearance_by_cone.py --run runs/full

CONTROLLER SETTINGS (constants inside controller.py)
  path smoothing 5 m moving average; pure pursuit from the rear axle,
  lookahead L_d = 0.5 + 0.1*v_x; speed control a_x = 2.5*(v_ref - v_x),
  v_ref = planned speed 3 m ahead, capped at 11.6 m/s.
  These values were chosen by PyChrono runs with the rule in Section III of
  the report (fastest run with clearance >= 0.3 m whose neighbouring settings
  also kept >= 0.3 m).

No training data, neural networks or RL libraries are used.

# Slalom Challenge


In this individual 2026 **Learning-Based Control for Mobility Systems** challenge, drive a PyChrono BMW E90 through eight alternating cones and finish as fast as possible. Stay on the road, avoid contact, respect vehicle limits, and justify your design using course methods.

Choose one mode; both use the same course and success rules.

| Mode | Your task | Supplied support |
|---|---|---|
| **Easy** | Plan a grid-based waypoint path with virtual 0.5 s steps; submit `easy_plan.json` | Simplified planning model and fixed path tracking controller |
| **Full** | Write a `Controller` that requests steering and acceleration every 0.02 s | Control-oriented vehicle model and 5 m/s example controller |

In Easy mode, the supplied path tracking controller drives the car every 0.02 s along your plan; actual waypoint arrival times may differ from the virtual steps. In Full mode, your controller chooses steering and acceleration every 0.02 s; it may combine a learned planner with your own tracking controller.

The **5 m/s example controller** checks installation and the interface; its path and speed are not required targets. Course configuration, the runner (`run_local.py`), model validation data, and interface checks are included. [Running](RUNNING.md#how-the-system-works) shows both control-system diagrams. Click the image to watch the drive.

[![Watch the 5 m/s example controller in PyChrono](assets/starter_drive_poster.png)](assets/starter_drive.mp4)

## Start

1. [Install](INSTALL.md) the environment.
2. [Run the 5 m/s example controller](RUNNING.md) and inspect `result.json` and `trajectory.npz`.
3. Use [Start here](START_HERE.md) to try the predictor and run data, then develop a method in your chosen mode.

## Guides

| Guide | Contents |
|---|---|
| [Assignment](ASSIGNMENT.md) | Rules, dates, scoring |
| [Installation](INSTALL.md) | Windows x64 setup |
| [Start here](START_HERE.md) | Predictor, result files, learned artifacts |
| [Running](RUNNING.md) | Both modes and local tests |
| [Interface](INTERFACE.md) | Inputs and outputs |
| [Model](MODEL.md) | Predictor and validation |
| [Submission and AI](SUBMISSION_AND_AI.md) | Files, report, AI disclosure |
| [Troubleshooting](TROUBLESHOOTING.md) | Technical issues |
| [Q&A](Q_AND_A.md) | Common questions |
| [Code style](DEVELOPMENT.md) | Optional Ruff formatting and linting |
