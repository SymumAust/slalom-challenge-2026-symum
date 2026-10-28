"""Generate the report figures from the recorded PyChrono runs.

Usage: python make_figures.py --runs <folder with run subfolders> --repo <challenge repo> --out figures
Run folders used: example, easy_best, full_v1, full_v2 (each with trajectory.npz, result.json).
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--repo", required=True)
ap.add_argument("--out", default="figures")
a = ap.parse_args()
RUNS, REPO, OUT = Path(a.runs), Path(a.repo), Path(a.out)
OUT.mkdir(exist_ok=True)

C = {"easy": "#2a78d6", "v1": "#eb6834", "v2": "#1baf7a", "ex": "#888780", "ink": "#3d3d3a", "grid": "#e3e2dc", "cone": "#c0392b"}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": C["ink"], "axes.labelcolor": C["ink"],
                     "xtick.color": C["ink"], "ytick.color": C["ink"], "axes.linewidth": 0.6, "lines.linewidth": 1.3,
                     "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

def rows(name):
    with np.load(RUNS / name / "trajectory.npz") as d:
        return d["rows"]

def footprint(x, y, psi):
    c, s = math.cos(psi), math.sin(psi)
    xc, yc = x - 1.371 * c, y - 1.371 * s
    pts = []
    for lx, ly in ((2.35, 0.95), (2.35, -0.95), (-2.35, -0.95), (-2.35, 0.95)):
        pts.append((xc + c * lx - s * ly, yc + s * lx + c * ly))
    return pts

def clearance(x, y, psi, cx):
    c, s = math.cos(psi), math.sin(psi)
    xc, yc = x - 1.371 * c, y - 1.371 * s
    dx, dy = cx - xc, -yc
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return math.hypot(max(abs(lx) - 2.35, 0), max(abs(ly) - 0.95, 0)) - 0.25

cones = [20 + 15 * j for j in range(8)]

def course(ax):
    ax.axhline(5.5, color=C["ink"], lw=0.8); ax.axhline(-5.5, color=C["ink"], lw=0.8)
    ax.axhline(0, color=C["grid"], lw=0.6, ls=(0, (4, 4)))
    ax.axvline(145, color=C["ink"], lw=1.2)
    ax.text(144.2, 4.6, "finish", ha="right", fontsize=7, color=C["ink"])
    for j, x in enumerate(cones):
        ax.add_patch(plt.Circle((x, 0), 0.25, color=C["cone"], zorder=5))
        ax.text(x, 4.3 if j % 2 == 0 else -4.9, "pass +Y" if j % 2 == 0 else "pass $-$Y", ha="center", fontsize=6.3, color=C["ink"])

# ---------------- Fig: top view trajectories ----------------
fig, ax = plt.subplots(figsize=(7.1, 2.05))
course(ax)
plan = json.loads((RUNS / "plan_best.json").read_text())["moves"]
px, py = [0], [0]
for m, n in plan:
    px.append(px[-1] + m); py.append(py[-1] + 0.5 * n)
ref = np.load(RUNS / "full_v2" / "full_reference.npz")
r_e, r_2 = rows("easy_best"), rows("full_v2")
ax.plot(px, py, "o", ms=2.2, color=C["easy"], alpha=0.55)
ax.plot(ref["x"], ref["y"], "s", ms=2.0, color=C["v2"], alpha=0.55)
ax.plot(r_e[:, 0], r_e[:, 1], color=C["easy"], ls=(0, (5, 2)), lw=1.1, label="Easy best: REF path (16.33 s), dots = DP plan nodes")
ax.plot(r_2[:, 0], r_2[:, 1], color=C["v2"], lw=1.4, label="Full v2: REF path (14.99 s), squares = DP reference nodes")
k = min(range(len(r_2)), key=lambda i: min(clearance(*r_2[i, :3], cx) for cx in cones))
for i in (k,):
    ax.add_patch(Polygon(footprint(*r_2[i, :3]), closed=True, fill=False, ec=C["v2"], lw=0.9))
ax.annotate("closest body approach (0.38 m)", xy=(r_2[k, 0] - 1.4, r_2[k, 1]), xytext=(40, -3.75),
            fontsize=6.5, color=C["ink"], arrowprops=dict(arrowstyle="-", lw=0.5, color=C["ink"]))
ax.set_xlim(-2, 148); ax.set_ylim(-5.9, 5.9)
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=2, fontsize=7)
fig.savefig(OUT / "fig_trajectories.pdf"); plt.close(fig)

# ---------------- Fig: speed vs X ----------------
fig, ax = plt.subplots(figsize=(3.45, 1.9))
for name, key, lab, ls in (("example", "ex", "Example controller (30.66 s)", ":"), ("easy_best", "easy", "Easy best (16.33 s)", (0, (5, 2))),
                           ("full_v1", "v1", "Full v1 (15.32 s)", (0, (2, 1))), ("full_v2", "v2", "Full v2 (14.99 s)", "-")):
    r = rows(name)
    ax.plot(r[:, 0], r[:, 3], color=C[key], ls=ls, label=lab)
for x in cones:
    ax.axvline(x, color=C["grid"], lw=0.6, zorder=0)
ax.axhline(12, color=C["cone"], lw=0.8); ax.text(2, 12.15, "speed limit 12 m/s", fontsize=6.5, color=C["cone"])
ax.set_xlim(0, 146); ax.set_ylim(0, 13)
ax.set_xlabel("X (m)  (grey lines: cones)"); ax.set_ylabel("$v_x$ (m/s)")
ax.legend(loc="lower right", fontsize=6.0, ncol=2, columnspacing=0.8, handlelength=1.8)
fig.savefig(OUT / "fig_speed.pdf"); plt.close(fig)

# ---------------- Fig: robustness ----------------
rob = {  # cone shift (m) -> min clearance (m); None = cone contact
    "Easy 48-stage (24.05 s)": ("ex", [(0, .738), (.4, .339), (.6, .140), (.8, None)], "o"),
    "Easy best (16.33 s)": ("easy", [(0, .440), (.2, .247), (.3, .150), (.4, .054), (.5, None)], "s"),
    "Full v1 (15.32 s)": ("v1", [(0, .523), (.3, .225), (.5, .026), (.6, None)], "^"),
    "Full v2 (14.99 s)": ("v2", [(0, .377), (.2, .177), (.3, .077), (.35, .027), (.4, None)], "D"),
}
fig, ax = plt.subplots(figsize=(3.45, 1.95))
ax.axhline(0, color=C["ink"], lw=0.7)
for lab, (key, pts, mk) in rob.items():
    ok = [(s, c) for s, c in pts if c is not None]
    ax.plot([p[0] for p in ok], [p[1] for p in ok], color=C[key], marker=mk, ms=3.5, label=lab)
    fail = [s for s, c in pts if c is None][0]
    ax.plot([fail], [0], marker="x", ms=6, mew=1.4, color=C[key])
ax.text(0.81, 0.03, "x = cone contact", fontsize=6.3, ha="right", color=C["ink"])
ax.set_xlabel("all cones moved toward the car (m)"); ax.set_ylabel("min clearance (m)")
ax.set_xlim(-0.02, 0.85); ax.set_ylim(-0.05, 0.8)
ax.legend(fontsize=6.3, loc="upper right")
fig.savefig(OUT / "fig_robustness.pdf"); plt.close(fig)

# ---------------- Fig: trade-off ----------------
easy = [(16.477, .008), (19.121, .246), (19.655, .328), (23.121, .428), (24.046, .738), (19.126, .246), (19.59, .328), (17.585, .078), (19.657, .328),
        (17.53, .45), (16.536, .473), (16.331, .439), (16.514, .472), (18.016, .722), (17.577, .701), (18.53, .574), (18.557, .634), (17.588, .921), (17.618, .59), (17.178, .933)]
full = [(15.718, .468), (15.633, .209), (16.236, .663), (16.134, .39), (16.735, .719), (16.623, .486),
        (15.323, .523), (15.25, .367), (15.283, .452), (15.215, .267), (15.365, .523), (15.274, .369), (15.309, .454), (15.233, .268),
        (15.818, .74), (15.761, .572), (15.79, .671), (15.306, .539),
        (15.101, .414), (14.987, .38), (14.929, .32), (14.877, .187), (14.994, .385), (14.934, .334), (14.881, .201), (14.833, .054),
        (15.327, .613), (15.285, .546), (15.243, .463), (15.189, .365), (15.292, .561), (15.243, .472), (15.198, .372), (15.142, .277)]
fig, ax = plt.subplots(figsize=(3.45, 1.95))
ax.scatter(*zip(*easy), s=10, color=C["easy"], marker="o", label="Easy-mode DP plans", lw=0)
ax.scatter(*zip(*full), s=10, color=C["v2"], marker="s", label="Full-mode DP + tracker", lw=0)
for (t, c), txt, pos in (((16.333, .440), "Easy best", (17.2, 0.37)), ((15.323, .523), "Full v1", (14.35, 0.68)), ((14.987, .380), "Full v2", (17.0, 0.17))):
    ax.scatter([t], [c], s=42, facecolors="none", edgecolors=C["ink"], lw=0.9)
    ax.annotate(txt, (t, c), xytext=pos, textcoords="data", fontsize=6.5, color=C["ink"],
                arrowprops=dict(arrowstyle="-", lw=0.5, color=C["ink"], shrinkA=0, shrinkB=4))
ax.axhline(0.3, color=C["grid"], lw=0.8, ls="--"); ax.text(24.1, 0.31, "0.3 m buffer", fontsize=6.3, ha="right", color=C["ink"])
ax.set_xlim(14.2, 24.6); ax.set_xlabel("PyChrono finish time (s)"); ax.set_ylabel("min clearance (m)")
ax.legend(fontsize=6.3, loc="upper left")
fig.savefig(OUT / "fig_tradeoff.pdf"); plt.close(fig)
print("figures written to", OUT)
