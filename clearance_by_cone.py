"""Closest approach of the car body to each cone, from a run's trajectory.npz."""

import argparse
import json
from pathlib import Path

import numpy as np

from contract import footprint_clearance, scenario_from_config

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="run folder, e.g. runs/my_plan_2.5")
args = ap.parse_args()

root = Path(__file__).resolve().parent
course = json.loads((root / "course.json").read_text(encoding="utf-8"))
cones = scenario_from_config(course)["cones"]
with np.load(Path(args.run) / "trajectory.npz") as trace:
    rows = trace["rows"]

print("cone   X(m)  side   closest gap(m)  at time(s)  car X(m)  car Y(m)")
for j, cone in enumerate(cones):
    gaps = [footprint_clearance(row[:3], cone, course) for row in rows]
    k = int(np.argmin(gaps))
    side = "left " if cone["pass_sign"] > 0 else "right"
    print(f"{j:4d}  {cone['x']:5.0f}  {side}  {gaps[k]:14.3f}  {rows[k, 10]:10.2f}"
          f"  {rows[k, 0]:8.2f}  {rows[k, 1]:8.2f}")