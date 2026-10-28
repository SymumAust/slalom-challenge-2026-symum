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
#-------------------------------------------------------------------------------------------------------


"""Full-mode offline planner: backward DP for a minimum-time REF reference path.

Same idea as easy_dp.py, but on our own finer lattice and with bounds taken from
the vehicle model:
  * speeds up to 11.5 m/s (course limit 12 m/s),
  * forward acceleration limited by the input adapter's achievable range
    (contract.acceleration_bounds), so the reference is drivable,
  * a body-footprint clearance check against every cone (contract.footprint_clearance)
    and against the road edges (contract.inside_road),
  * the move that crosses the finish line is charged only the fraction of a stage it uses.

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
# Lattice (our own choice in Full mode; adjustable from the command line)
# ---------------------------------------------------------------------------
X_FINISH = COURSE["finish_x_m"]


def set_lattice(dt=0.5, hx=0.25, hy=0.25, m_max=23, n_max=8):
    """Define the planning grid: stage time, cell size and the allowed moves."""
    global DT, HX, HY, MOVES, START, NX, SPEED, ANGLE, N_PAD
    DT, HX, HY = dt, hx, hy
    MOVES = [(m, n) for m in range(1, m_max + 1) for n in range(-n_max, n_max + 1)]
    START = len(MOVES)  # extra "previous move" index meaning "car at rest"
    NX = int(math.ceil(X_FINISH / HX))
    N_PAD = n_max
    SPEED = np.array([math.hypot(m * HX, n * HY) / DT for m, n in MOVES] + [0.0])
    ANGLE = np.array([math.atan2(n * HY, m * HX) for m, n in MOVES] + [0.0])


set_lattice()


def default_bounds():
    return {
        "max_abs_y": 4.0,  # stay well inside the 5.5 m road half-width
        "max_speed": 11.8,  # course limit is 12 m/s
        "max_angle": 0.6,  # rad, keeps heading far from the 1.3 rad limit
        "accel_scale": 1.0,  # fraction of the model's achievable acceleration
        "max_decel": 6.0,
        "max_lat_accel": 8.0,  # tested safe in PyChrono
        "gate_margin": 1.2,  # REF offset at each gate (m)
        "body_margin": 0.4,  # footprint-to-cone clearance along each segment (m)
        "road_margin": 0.8,  # footprint-to-road-edge clearance (m)
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


def road_clearance_np(y, psi):
    """Distance from the COM rectangle to the nearer road edge (contract.inside_road)."""
    yc = y - COURSE["ref_to_com_m"] * np.sin(psi)
    extent = (COURSE["footprint_length_m"] / 2 * np.abs(np.sin(psi))
              + COURSE["footprint_width_m"] / 2 * np.abs(np.cos(psi)))
    return COURSE["road_half_width_m"] - (np.abs(yc) + extent)


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
    for t in (0.5, 1.0):  # keep the whole car body away from the road edges
        ok &= road_clearance_np(y0 + t * (y1 - y0), psi) >= b["road_margin"]
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
    # tiny tie-break for smoother lateral motion (kept far below one stage in total)
    smooth = 1e-6 * (np.array([mv[1] for mv in MOVES])[None, :]
                     - np.array([mv[1] for mv in MOVES] + [0])[:, None]) ** 2
    pad = N_PAD
    m_max = max(m for m, _ in MOVES)
    J = np.full((NX + m_max + 1, n_y + 2 * pad, n_p), np.inf)  # padded for shifted look-ups
    J[NX:, :, :] = 0.0
    policy = np.full((NX, n_y, n_p), -1, dtype=np.int16)
    for ix in range(NX - 1, -1, -1):
        best = np.full((n_y, n_p), np.inf)
        arg = np.full((n_y, n_p), -1, dtype=np.int16)
        for u, (m, n) in enumerate(MOVES):
            nxt = J[ix + m, pad + n : pad + n + n_y, u]  # cost-to-go after the move, per lateral node
            nxt = np.where(pos_ok[ix, :, u], nxt, np.inf)
            # stage cost = 1 stage, except the move that crosses the finish line:
            # it only costs the fraction of the stage needed to reach X_FINISH
            x0, x1 = ix * HX, (ix + m) * HX
            stage = (X_FINISH - x0) / (x1 - x0) if x1 >= X_FINISH else 1.0
            cost = stage + smooth[None, :, u] + nxt[:, None]
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
    ap.add_argument("--dt", type=float, default=0.5, help="stage time (s)")
    ap.add_argument("--hx", type=float, default=0.25, help="longitudinal cell (m)")
    ap.add_argument("--hy", type=float, default=0.25, help="lateral cell (m)")
    ap.add_argument("--m-max", type=int, default=23, help="largest forward move (cells)")
    ap.add_argument("--n-max", type=int, default=8, help="largest lateral move (cells)")
    b = default_bounds()
    for key, value in b.items():
        ap.add_argument("--" + key.replace("_", "-"), type=float, default=value)
    args = ap.parse_args()
    for key in b:
        b[key] = getattr(args, key)
    set_lattice(args.dt, args.hx, args.hy, args.m_max, args.n_max)
    J, policy, ny, pad = solve(b)
    nodes = extract(policy, ny)
    # store the fitted steering map with the path, so the controller needs no other file
    fit = json.loads((ROOT / "model_parameters.json").read_text(encoding="utf-8"))
    fit = dict(zip(fit["parameter_names"], fit["parameters"]))
    np.savez(args.output, x=nodes[:, 0], y=nodes[:, 1], v=nodes[:, 2],
             steer_gain=fit["steer_gain"], steer_speed2=fit["steer_speed2"],
             bounds=json.dumps(b))
    print(f"lattice: dt={DT} hx={HX} hy={HY} moves={len(MOVES)}")
    print(f"bounds: {b}")
    t = J[0, pad + ny, START] * DT
    print(f"stages: {len(nodes) - 1}  (virtual time to the finish line {t:.2f} s)")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()