
"""Robustness test: drive one solution on courses whose cones are moved toward the car.

Every cone is shifted by `shift` metres toward the side the car passes on, so the
course gets harder while the solution stays the same. The solution is safe up to
the largest shift that still succeeds.

Usage:
    Easy:  python robustness_test.py --plan plan_best.json --shifts 0 0.2 0.3 0.4 0.5
    Full:  python robustness_test.py --controller my_controller.py --shifts 0 0.2 0.3 0.4 0.5
"""

import argparse
import contextlib
import io
import json
from pathlib import Path

from run_local import run

ROOT = Path(__file__).resolve().parent


def shifted_course(shift, folder):
    """Copy course.json with every cone moved `shift` m toward its pass side."""
    course = json.loads((ROOT / "course.json").read_text(encoding="utf-8"))
    course["scenario_id"] = f"robustness-shift-{shift}"
    course["cone_centers_m"] = [
        [
            course["first_cone_x_m"] + j * course["cone_spacing_m"],
            shift if j % 2 == 0 else -shift,  # even cones: car passes left (+Y)
        ]
        for j in range(course["cone_count"])
    ]
    path = folder / f"course_shift_{shift}.json"
    path.write_text(json.dumps(course, indent=2), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan", help="Easy plan JSON to test")
    group.add_argument("--controller", help="Full-mode controller file to test")
    ap.add_argument("--shifts", type=float, nargs="+", default=[0.0, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--output", default="runs/robustness")
    args = ap.parse_args()

    mode = "easy" if args.plan else "full"
    tested = args.plan or args.controller
    folder = Path(args.output) / Path(tested).stem
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for shift in args.shifts:
        config = shifted_course(shift, folder)
        with contextlib.redirect_stdout(io.StringIO()):  # hide the runner's long printout
            result = run(args.controller, config, folder / f"run_shift_{shift}",
                         mode=mode, plan_path=args.plan)
        rows.append((shift, result["status"], result["finish_time_s"], result["min_clearance_m"]))
        print(f"shift {shift:4.2f} m: {result['status']}", flush=True)

    print(f"\n{mode} solution: {tested}")
    print("cone shift (m)  status         finish (s)  min clearance (m)")
    for shift, status, finish, clearance in rows:
        finish_text = f"{finish:.3f}" if finish is not None else "-"
        print(f"{shift:14.2f}  {status:13s}  {finish_text:>10s}  {clearance:17.3f}")
    (folder / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nsaved: {folder / 'summary.json'}")


if __name__ == "__main__":
    main()