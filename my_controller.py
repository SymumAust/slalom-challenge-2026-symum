"""Full-mode controller: tracks the offline DP reference (full_reference.npz).

Layer 1 (offline, full_dp.py): backward DP gives a minimum-time REF path and speed.
Layer 2 (online, this file): pure-pursuit steering + speed tracking every 0.02 s.
"""

import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WHEELBASE = 2.776


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class Controller:
    # tracker gains (tuned in PyChrono)
    LOOK_BASE = 0.5  # lookahead distance at standstill (m)
    LOOK_GAIN = 0.2  # extra lookahead per m/s (s)
    SPEED_PREVIEW = 3.0  # read the speed target this far ahead (m)
    SPEED_GAIN = 2.5  # acceleration request per m/s of speed error (1/s)

    def reset(self, scenario, seed):
        self.scenario = scenario
        with np.load(HERE / "full_reference.npz", allow_pickle=False) as ref:
            self.px, self.py, self.pv = ref["x"].copy(), ref["y"].copy(), ref["v"].copy()
            # fitted steering map (wheel angle per unit request), stored by full_dp.py
            self.sg, self.sv = float(ref["steer_gain"]), float(ref["steer_speed2"])
        # densify the reference polyline so the lookahead search is smooth
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(self.px), np.diff(self.py)))])
        self.ds = np.linspace(0.0, s[-1] + 30.0, int((s[-1] + 30.0) / 0.1))
        ext_x = np.append(self.px, self.px[-1] + 30.0)
        ext_y = np.append(self.py, self.py[-1])
        ext_s = np.append(s, s[-1] + 30.0)
        self.dx = np.interp(self.ds, ext_s, ext_x)
        self.dy = np.interp(self.ds, ext_s, ext_y)
        self.last_i = 0

    def speed_target(self, x):
        # segment k (node k -> k+1) is driven at speed v[k+1]
        i = int(np.clip(np.searchsorted(self.px, x, side="right") - 1, 0, len(self.px) - 2))
        return float(self.pv[i + 1])

    def act(self, observation):
        x, y, psi, vx, vy, r, delta = observation["state"]
        c, s = math.cos(psi), math.sin(psi)
        rx, ry = x - WHEELBASE * c, y - WHEELBASE * s  # rear axle (REF is near the front axle)

        # nearest point on the path (search forward only)
        lo, hi = self.last_i, min(self.last_i + 400, len(self.dx))
        d2 = (self.dx[lo:hi] - x) ** 2 + (self.dy[lo:hi] - y) ** 2
        self.last_i = lo + int(np.argmin(d2))

        # pure pursuit from the rear axle to a point `look` ahead of REF along the path
        look = self.LOOK_BASE + self.LOOK_GAIN * max(vx, 0.0)
        j = min(self.last_i + int(look / 0.1), len(self.dx) - 1)
        gx, gy = self.dx[j] - rx, self.dy[j] - ry
        lx, ly = c * gx + s * gy, -s * gx + c * gy
        wheel = math.atan2(2.0 * WHEELBASE * ly, lx * lx + ly * ly)
        steering = clamp(wheel / (self.sg + self.sv * vx * vx), -0.8, 0.8)

        target = min(self.speed_target(x + self.SPEED_PREVIEW), 11.6)
        acceleration = clamp(self.SPEED_GAIN * (target - vx), -7.0, 7.0)
        return {"steering": float(steering), "acceleration": float(acceleration)}