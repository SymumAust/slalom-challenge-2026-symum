"""Replay several finished runs side by side on one top-view course (2D, real time).

It only reads the trajectory.npz files written by run_local.py, so it does not
change or re-run the simulation. Needs Matplotlib (offline tool, not used in evaluation).

Usage:
    python race_replay.py runs/easy_best runs/full_v4 --labels Easy Full
    python race_replay.py runs/easy_best runs/full_v4 --labels Easy Full --speed 0.5
    python race_replay.py runs/easy_best runs/full_v4 --labels Easy Full --save race.gif
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore[reportMissingModuleSource]
import numpy as np
from matplotlib.animation import FuncAnimation  # type: ignore[reportMissingModuleSource]
from matplotlib.patches import Circle, Polygon  # type: ignore[reportMissingModuleSource]

from contract import scenario_from_config

ROOT = Path(__file__).resolve().parent
COURSE = json.loads((ROOT / "course.json").read_text(encoding="utf-8"))
SCEN = scenario_from_config(COURSE)
COLORS = ["tab:orange", "tab:blue", "tab:green", "tab:purple"]


def load(folder):
    rows = np.load(Path(folder) / "trajectory.npz")["rows"]
    res = Path(folder) / "result.json"
    if res.exists():  # official finish time and clearance for the legend
        r = json.loads(res.read_text(encoding="utf-8"))
        load.info[str(folder)] = f"{r['status']}, {r['finish_time_s']:.3f} s, {r['min_clearance_m']:.3f} m"
    # columns: X, Y, heading, ... , time (last column); X, Y are the REF point
    return rows[:, -1], rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]


load.info = {}


def body(x, y, psi):
    """Corners of the car footprint (COM-centred rectangle, as in contract.py)."""
    c, s = np.cos(psi), np.sin(psi)
    xc, yc = x - COURSE["ref_to_com_m"] * c, y - COURSE["ref_to_com_m"] * s
    hl, hw = COURSE["footprint_length_m"] / 2, COURSE["footprint_width_m"] / 2
    pts = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    return np.c_[xc + pts[:, 0] * c - pts[:, 1] * s, yc + pts[:, 0] * s + pts[:, 1] * c]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run folders that contain trajectory.npz")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--speed", type=float, default=1.0, help="1 = real time, 0.5 = slow motion")
    ap.add_argument("--save", help="save the animation (e.g. race.gif) instead of showing it")
    args = ap.parse_args()
    labels = args.labels or [Path(r).name for r in args.runs]
    data = [load(r) for r in args.runs]
    t_end = max(d[0][-1] for d in data)

    hw, fx = COURSE["road_half_width_m"], COURSE["finish_x_m"]
    fig, ax = plt.subplots(figsize=(14, 2.8))
    ax.set_xlim(-5, fx + 5)
    ax.set_ylim(-hw - 0.5, hw + 0.5)
    ax.set_aspect("equal")
    ax.axhline(hw, color="k"), ax.axhline(-hw, color="k")
    ax.axvline(fx, color="k", ls="--")
    for cone in SCEN["cones"]:
        ax.add_patch(Circle((cone["x"], cone["y"]), COURSE["cone_radius_m"], color="red"))
    cars, trails = [], []
    for i, (lab, d, run) in enumerate(zip(labels, data, args.runs)):
        col = COLORS[i % len(COLORS)]
        (trail,) = ax.plot([], [], color=col, lw=1.2, label=f"{lab} ({load.info.get(str(run), f'{d[0][-1]:.2f} s')})")
        car = Polygon(body(d[1][0], d[2][0], d[3][0]), closed=True, color=col, alpha=0.8)
        ax.add_patch(car)
        cars.append(car), trails.append(trail)
    title = ax.set_title("")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlabel("X (m)"), ax.set_ylabel("Y (m)")

    frame_dt = 0.04  # 25 frames per second of simulated time
    n_frames = int(t_end / frame_dt) + 30  # a short pause at the end

    def update(k):
        t = min(k * frame_dt, t_end)
        info = []
        for lab, d, car, trail in zip(labels, data, cars, trails):
            i = min(np.searchsorted(d[0], t), len(d[0]) - 1)
            car.set_xy(body(d[1][i], d[2][i], d[3][i]))
            trail.set_data(d[1][: i + 1], d[2][: i + 1])
            state = "FINISHED" if t >= d[0][-1] else f"{d[4][i]:.1f} m/s"
            info.append(f"{lab}: {state}")
        title.set_text(f"t = {t:5.2f} s    " + "    ".join(info))
        return cars + trails + [title]

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 * frame_dt / args.speed, blit=False, repeat=False)
    if args.save:
        anim.save(args.save, fps=int(round(args.speed / frame_dt)))
        print("saved", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
