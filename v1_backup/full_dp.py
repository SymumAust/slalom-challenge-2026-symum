"""Full-mode offline planner: backward DP for a minimum-time REF reference path.

Same idea as easy_dp.py, but on our own finer lattice and with bounds taken from
the vehicle model:
  * speeds up to 11.5 m/s (course limit 12 m/s),
  * forward acceleration limited by the input adapter's achievable range
    (contract.acceleration_bounds), so the reference is drivable,
  * a body-footprint clearance check against every cone (contract.footprint_clearance).

The result is saved as full_reference.npz (the learned artifact loaded by the controller).

Usage:
    python full_dp.py --output full_reference.npz [--body-margin 0.45 ...]
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from contract import acceleration_bounds, scenario_from_config

ROOT = Path(__file__).resolve().parent
COURSE = json.loads((ROOT / "course.json").read_text(encoding="utf-8"))
SCENARIO = scenario_from_config(COURSE)

# ---------------------------------------------------------------------------
# Lattice (our own choice in Full mode)
# ---------------------------------------------------------------------------
DT = 0.5  # stage length (s)
HX = 0.5  # longitudinal grid (m)
HY = 0.25  # lateral grid (m)
M_MOVES = np.arange(1, 12)  # 0.5 .. 5.5 m per stage -> 1 .. 11 m/s forward
N_MOVES = np.arange(-8, 9)  # -2 .. +2 m per stage
MOVES = [(int(m), int(n)) for m in M_MOVES for n in N_MOVES]
START = len(MOVES)
X_FINISH = COURSE["finish_x_m"]
NX = int(math.ceil(X_FINISH / HX))


def move_speed_angle(m, n):
    dx, dy = m * HX, n * HY
    return math.hypot(dx, dy) / DT, math.atan2(dy, dx)


SPEED = np.array([move_speed_angle(m, n)[0] for m, n in MOVES] + [0.0])
ANGLE = np.array([move_speed_angle(m, n)[1] for m, n in MOVES] + [0.0])


def default_bounds():
    return {
        "max_abs_y": 4.0,  # stay well inside the 5.5 m road half-width
        "max_speed": 11.5,  # course limit is 12 m/s
        "max_angle": 0.6,  # rad, keeps heading far from the 1.3 rad limit
        "accel_scale": 1.0,  # fraction of the model's achievable acceleration
        "max_decel": 6.0,
        "max_lat_accel": 8.0,  # tested safe in PyChrono
        "gate_margin": 1.2,  # REF offset at each gate (m)
        "body_margin": 0.45,  # footprint-to-cone clearance along each segment (m)
    }


def pair_table(b):
    """ok[p, u]: may move u follow previous move p (speed/acceleration/turn bounds)."""
    v0, c0 = SPEED[:, None], ANGLE[:, None]
    v1, c1 = SPEED[None, :-1], ANGLE[None, :-1]
    # Time the car needs to speed up from v0 to v1 at the model's achievable
    # acceleration: t = integral dv / (scale * a_max(v)); it must fit in one stage.
    grid = np.linspace(0.0, 12.0, 241)
    amax = np.array([acceleration_bounds(float(v), COURSE)[1] for v in grid])
    t_of_v = np.concatenate([[0.0], np.cumsum(np.diff(grid) * 2 / (amax[1:] + amax[:-1]))])
    t_needed = (np.interp(v1, grid, t_of_v) - np.interp(v0, grid, t_of_v)) / b["accel_scale"]
    dchi = np.abs(c1 - c0)
    return (
        (v1 <= b["max_speed"])
        & (np.abs(c1) <= b["max_angle"])
        & (t_needed <= DT + 1e-9)
        & ((v0 - v1) / DT <= b["max_decel"])
        & (v1 * dchi / DT <= b["max_lat_accel"])
    )


def footprint_clearance_np(x, y, psi, cone):
    """Vectorised copy of contract.footprint_clearance (REF pose -> COM rectangle)."""
    c, s = np.cos(psi), np.sin(psi)
    xc = x - COURSE["ref_to_com_m"] * c
    yc = y - COURSE["ref_to_com_m"] * s
    dx, dy = cone["x"] - xc, cone["y"] - yc
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return (
        np.hypot(
            np.maximum(np.abs(lx) - COURSE["footprint_length_m"] / 2, 0.0),
            np.maximum(np.abs(ly) - COURSE["footprint_width_m"] / 2, 0.0),
        )
        - COURSE["cone_radius_m"]
    )


def position_table(b, ny):
    """ok[ix, j, u]: road, gate-side/margin and footprint bounds for move u from node (ix, j)."""
    x0 = (np.arange(NX) * HX)[:, None, None]
    y0 = ((np.arange(2 * ny + 1) - ny) * HY)[None, :, None]
    m = np.array([mv[0] for mv in MOVES])[None, None, :]
    n = np.array([mv[1] for mv in MOVES])[None, None, :]
    shape = (NX, 2 * ny + 1, len(MOVES))
    x1, y1 = x0 + m * HX, y0 + n * HY
    psi = ANGLE[None, None, :-1]
    ok = np.broadcast_to(np.abs(y1) <= b["max_abs_y"] + 1e-9, shape).copy()
    for cone in SCENARIO["cones"]:
        crosses = (x0 < cone["x"]) & (cone["x"] <= x1)
        lam = (cone["x"] - x0) / (x1 - x0)
        d = cone["pass_sign"] * (y0 + lam * (y1 - y0) - cone["y"])
        ok &= ~crosses | (d >= b["gate_margin"] - 1e-9)
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):  # REF poses along the segment
            xs, ys = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            near = np.abs(xs - cone["x"]) < 7.0
            ok &= ~near | (footprint_clearance_np(xs, ys, psi, cone) >= b["body_margin"])
    return ok


def solve(b):
    ny = int(round(b["max_abs_y"] / HY))
    n_y, n_p = 2 * ny + 1, len(MOVES) + 1
    pair_ok = pair_table(b)
    pos_ok = position_table(b, ny)
    smooth = 1e-4 * (np.array([mv[1] for mv in MOVES])[None, :]
                     - np.array([mv[1] for mv in MOVES] + [0])[:, None]) ** 2
    J = np.full((NX + 12, n_y + 16, n_p), np.inf)  # padded so shifted look-ups stay in range
    J[NX:, :, :] = 0.0
    policy = np.full((NX, n_y, n_p), -1, dtype=np.int16)
    pad = 8
    for ix in range(NX - 1, -1, -1):
        best = np.full((n_y, n_p), np.inf)
        arg = np.full((n_y, n_p), -1, dtype=np.int16)
        for u, (m, n) in enumerate(MOVES):
            nxt = J[ix + m, pad + n : pad + n + n_y, u]  # cost-to-go after the move, per lateral node
            nxt = np.where(pos_ok[ix, :, u], nxt, np.inf)
            cost = 1.0 + smooth[None, :, u] + nxt[:, None]
            cost = np.where(pair_ok[None, :, u], cost, np.inf)
            better = cost < best
            best[better] = cost[better]
            arg[better] = u
        J[ix, pad : pad + n_y, :] = best
        policy[ix] = arg
    return J, policy, ny, pad


def extract(policy, ny):
    ix, j, p, nodes = 0, ny, START, [(0.0, 0.0, 0.0)]
    while ix < NX:
        u = int(policy[ix, j, p])
        if u < 0:
            raise RuntimeError("No feasible reference with these bounds")
        m, n = MOVES[u]
        ix, j, p = ix + m, j + n, u
        nodes.append((ix * HX, (j - ny) * HY, SPEED[u]))
    return np.array(nodes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="full_reference.npz")
    b = default_bounds()
    for key, value in b.items():
        ap.add_argument("--" + key.replace("_", "-"), type=float, default=value)
    args = ap.parse_args()
    for key in b:
        b[key] = getattr(args, key)
    J, policy, ny, pad = solve(b)
    nodes = extract(policy, ny)
    np.savez(args.output, x=nodes[:, 0], y=nodes[:, 1], v=nodes[:, 2],
             bounds=json.dumps(b))
    print(f"bounds: {b}")
    print(f"stages: {len(nodes) - 1}  (virtual time {(len(nodes) - 1) * DT:.1f} s)")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()