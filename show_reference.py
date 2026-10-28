"""Print the contents of submission/full_reference.npz (viewing only, changes nothing)."""
import json

import numpy as np

d = np.load("submission/full_reference.npz", allow_pickle=False)
print("keys:", d.files)
print(f"points: {len(d['x'])}  (start point + {len(d['x']) - 1} stages)\n")
print(" i      x (m)    y (m)   v (m/s)")
for i, (x, y, v) in enumerate(zip(d["x"], d["y"], d["v"])):
    print(f"{i:2d}  {x:8.2f}  {y:7.2f}  {v:8.2f}")
print("\nsteer_gain   =", float(d["steer_gain"]))
print("steer_speed2 =", float(d["steer_speed2"]))
print("\nbounds used by the planner:")
for k, v in json.loads(str(d["bounds"])).items():
    print(f"  {k:14s} {v}")