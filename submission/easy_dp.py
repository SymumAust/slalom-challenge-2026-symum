"""Easy-mode planner: backward dynamic programming on the published waypoint lattice."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from contract import footprint_clearance, scenario_from_config

# ---------------------------------------------------------------------------
# PART 2: the planning world (grid, stages, actions)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SPEC = json.loads((ROOT / "easy_spec.json").read_text(encoding="utf-8"))
COURSE = json.loads((ROOT / "course.json").read_text(encoding="utf-8"))
SCENARIO = scenario_from_config(COURSE)

DT = SPEC["planning_dt_s"]  # 0.5 s per stage
HX = SPEC["grid_x_m"]  # 1.0 m per longitudinal grid step
HY = SPEC["grid_y_m"]  # 0.5 m per lateral grid step
K_MAX = SPEC["maximum_stages"]  # 120 stages at most
X_FINISH = COURSE["finish_x_m"]  # 145 m
NX = int(math.ceil(X_FINISH / HX))  # X index 145 = finish line

# Action set u = (m, n): move m*HX forward and n*HY sideways in one stage.
MOVES = [
    (m, n)
    for m in SPEC["longitudinal_moves"]
    for n in range(SPEC["lateral_move_min"], SPEC["lateral_move_max"] + 1)
]
START = len(MOVES)  # extra "previous move" index meaning "car at rest"


def segment_speed(m, n):
    """Virtual segment speed V_k in m/s."""
    return math.hypot(m * HX, n * HY) / DT


def segment_angle(m, n):
    """Virtual course angle chi_k in rad."""
    return math.atan2(n * HY, m * HX)

# ---------------------------------------------------------------------------
# PART 3: constraints (which moves are allowed)
# ---------------------------------------------------------------------------
def default_bounds():
    """Planning bounds; starting values come from easy_spec.json."""
    return {
        "gate_margin": 2.5,  # required signed offset d_j at every gate (m)
        "max_abs_y": SPEC["suggested_maximum_abs_y_m"],
        "max_speed": SPEC["suggested_maximum_segment_speed_m_s"],
        "max_angle": SPEC["suggested_maximum_abs_segment_course_angle_rad"],
        "max_accel": SPEC["suggested_maximum_acceleration_m_s2"],
        "max_decel": SPEC["suggested_maximum_deceleration_m_s2"],
        "max_angle_change": SPEC["suggested_maximum_course_angle_change_rad"],
        "max_lat_accel": SPEC["suggested_maximum_virtual_lateral_acceleration_m_s2"],
        "body_margin": None,  # optional footprint clearance along each segment (m)
    }


def pair_allowed(prev, move, b):
    """Speed/turn bounds that depend on the previous move and the new move."""
    m, n = MOVES[move]
    v, chi = segment_speed(m, n), segment_angle(m, n)
    if prev == START:
        v0, chi0 = 0.0, 0.0
    else:
        v0, chi0 = segment_speed(*MOVES[prev]), segment_angle(*MOVES[prev])
    return (
        v <= b["max_speed"]
        and abs(chi) <= b["max_angle"]
        and (v - v0) / DT <= b["max_accel"]
        and (v0 - v) / DT <= b["max_decel"]
        and abs(chi - chi0) <= b["max_angle_change"]
        and v * abs(chi - chi0) / DT <= b["max_lat_accel"]
    )


def position_allowed(ix, iy, move, b):
    """Gate, road-strip and optional footprint bounds for one move from (ix, iy)."""
    m, n = MOVES[move]
    x0, y0 = ix * HX, iy * HY
    x1, y1 = x0 + m * HX, y0 + n * HY
    if abs(y1) > b["max_abs_y"] + 1e-9:
        return False
    for cone in SCENARIO["cones"]:
        if x0 < cone["x"] <= x1:  # this move crosses cone j's X plane
            lam = (cone["x"] - x0) / (x1 - x0)
            d = cone["pass_sign"] * (y0 + lam * (y1 - y0) - cone["y"])
            if d < b["gate_margin"] - 1e-9:  # required: d_j > 0; chosen: d_j >= margin
                return False
    if b["body_margin"] is not None:
        psi = segment_angle(m, n)
        for t in (0.25, 0.5, 0.75, 1.0):  # sample REF poses along the segment
            state = [x0 + t * (x1 - x0), y0 + t * (y1 - y0), psi]
            for cone in SCENARIO["cones"]:
                if abs(cone["x"] - state[0]) < 6.0:
                    if footprint_clearance(state, cone, COURSE) < b["body_margin"]:
                        return False
    return True



# ---------------------------------------------------------------------------
# PART 4: backward DP and plan extraction
# ---------------------------------------------------------------------------
SMOOTH = 1e-4  # tiny tie-break: among equal-stage plans prefer smaller lateral jerks


def solve(b):
    """Backward DP. J[ix, iy, p] = minimum cost-to-go from X=ix, Y=iy, previous move p."""
    ny = int(round(b["max_abs_y"] / HY))
    n_y, n_p = 2 * ny + 1, len(MOVES) + 1
    J = np.full((NX + 6, n_y, n_p), np.inf)
    J[NX:, :, :] = 0.0  # finish states are absorbing with zero cost
    policy = np.full((NX, n_y, n_p), -1, dtype=int)
    pair_ok = np.array([[pair_allowed(p, u, b) for u in range(len(MOVES))] for p in range(n_p)])
    n_prev = np.array([n for _, n in MOVES] + [0])

    for ix in range(NX - 1, -1, -1):  # X always increases, so go from the finish backward
        for j in range(n_y):
            iy = j - ny
            best = np.full(n_p, np.inf)
            arg = np.full(n_p, -1)
            for u, (m, n) in enumerate(MOVES):
                j2 = j + n
                if not 0 <= j2 < n_y or not position_allowed(ix, iy, u, b):
                    continue
                cost = 1.0 + SMOOTH * (n - n_prev) ** 2 + J[ix + m, j2, u]
                cost = np.where(pair_ok[:, u], cost, np.inf)
                better = cost < best
                best[better], arg[better] = cost[better], u
            J[ix, j, :], policy[ix, j, :] = best, arg
    return J, policy, ny


def extract_plan(policy, ny):
    """Follow the optimal policy forward from the start state z0 = (0, 0, at rest)."""
    ix, j, p, moves = 0, ny, START, []
    while ix < NX:
        u = policy[ix, j, p]
        if u < 0:
            raise RuntimeError("No feasible plan with these bounds")
        m, n = MOVES[u]
        moves.append([m, n])
        ix, j, p = ix + m, j + n, u
        if len(moves) > K_MAX:
            raise RuntimeError("Plan exceeds the 120-stage horizon")
    return moves



# ---------------------------------------------------------------------------
# PART 5: command line
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="easy_plan.json")
    b = default_bounds()
    for key, value in b.items():
        ap.add_argument("--" + key.replace("_", "-"), type=float, default=value)
    args = ap.parse_args()
    for key in b:
        b[key] = getattr(args, key)

    J, policy, ny = solve(b)
    moves = extract_plan(policy, ny)
    Path(args.output).write_text(json.dumps({"moves": moves}), encoding="utf-8")
    print(f"bounds: {b}")
    print(f"stages: {len(moves)}  (virtual time {len(moves) * DT:.1f} s)")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()