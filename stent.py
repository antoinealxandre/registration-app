"""
braided_stent.py — Générateur paramétrique de stent tressé
Dépendances : numpy, matplotlib | trimesh (optionnel, export STL)
"""

import math
import numpy as np


# ── Géométrie ────────────────────────────────────────────────────────────────

def wire_centerlines(diameter=20., length=15., n_wires=16, braid_angle=45., n_pts=200):
    """Retourne deux listes de tableaux (n_pts, 3) : fils sens + et sens −."""
    R      = diameter / 2
    pitch  = math.pi * diameter / math.tan(math.radians(braid_angle))
    n_half = n_wires // 2
    t      = np.linspace(0, 1, n_pts)
    z      = (t - 0.5) * length

    def helix(i, d):
        phase = 2 * math.pi * i / n_half
        theta = d * 2 * math.pi * (length / pitch) * t + phase
        return np.column_stack([R * np.cos(theta), R * np.sin(theta), z])

    return {s: [helix(i, d) for i in range(n_half)]
            for s, d in (("plus", 1), ("minus", -1))}


def _tube_mesh(pts, radius, sides=8):
    """Mesh tubulaire autour d'une courbe 3D. Retourne (verts, faces)."""
    N   = len(pts)
    ang = np.linspace(0, 2 * math.pi, sides, endpoint=False)

    T = np.diff(pts, axis=0, append=pts[[-2]])
    T /= np.linalg.norm(T, axis=1, keepdims=True).clip(1e-12)

    arb = np.array([0., 0., 1.]) if abs(T[0, 2]) < .9 else np.array([1., 0., 0.])
    N0  = np.cross(T[0], arb); N0 /= np.linalg.norm(N0)
    Ns  = [N0]
    for i in range(1, N):
        b  = np.cross(T[i-1], T[i]); bn = np.linalg.norm(b)
        if bn < 1e-10:
            Ns.append(Ns[-1])
        else:
            b /= bn; th = math.acos(np.clip(T[i-1] @ T[i], -1, 1))
            c, s = math.cos(th), math.sin(th)
            x, y, z = b
            R_ = np.array([[c+x*x*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
                            [y*x*(1-c)+z*s, c+y*y*(1-c), y*z*(1-c)-x*s],
                            [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)]])
            Ns.append(R_ @ Ns[-1])
    Ns = np.array(Ns)
    Bs = np.cross(T, Ns)

    verts = (pts[:, None]
             + radius * np.cos(ang)[None, :, None] * Ns[:, None]
             + radius * np.sin(ang)[None, :, None] * Bs[:, None]).reshape(-1, 3)

    i = np.arange(N - 1); j = np.arange(sides)
    ii, jj = np.meshgrid(i, j, indexing="ij")
    a = (ii * sides + jj).ravel()
    b = (ii * sides + (jj + 1) % sides).ravel()
    c = ((ii + 1) * sides + jj).ravel()
    d = ((ii + 1) * sides + (jj + 1) % sides).ravel()

    return verts, np.concatenate([np.c_[a, b, d], np.c_[a, d, c]])


def stent_mesh(diameter=20., length=15., n_wires=16, braid_angle=45.,
               wire_radius=0.15, n_pts=200, tube_sides=8):
    """Retourne (vertices, faces, info) du mesh complet du stent."""
    wires = wire_centerlines(diameter, length, n_wires, braid_angle, n_pts)
    vl, fl, offset = [], [], 0
    for pts in wires["plus"] + wires["minus"]:
        v, f = _tube_mesh(pts, wire_radius, tube_sides)
        vl.append(v); fl.append(f + offset); offset += len(v)
    pitch = math.pi * diameter / math.tan(math.radians(braid_angle))
    info  = {"pitch_mm": round(pitch, 4), "n_turns": round(length / pitch, 3),
             "porosity": round(max(0, 1 - n_wires * 2 * wire_radius / (math.pi * diameter)), 4)}
    return np.concatenate(vl), np.concatenate(fl), info


# ── Visualiseur ───────────────────────────────────────────────────────────────

_CL_KEYS = ("diameter", "length", "n_wires", "braid_angle", "n_pts")

def show(diameter=20., length=15., n_wires=16, braid_angle=45.,
         wire_radius=0.15, n_pts=120):
    """Ouvre une fenêtre matplotlib 3D interactive avec sliders."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.widgets import Slider
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    fig = plt.figure(figsize=(11, 7), facecolor="#0d1117")
    fig.canvas.manager.set_window_title("Braided Stent Viewer")

    gs  = gridspec.GridSpec(6, 2, figure=fig,
                            left=0.05, right=0.98, top=0.96, bottom=0.05,
                            wspace=0.35, hspace=0.6)
    ax  = fig.add_subplot(gs[:, 0], projection="3d")
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#4a6fa5")
    for spine in ax.spines.values(): spine.set_color("#1e2d42")

    slider_cfg = [
        ("Diamètre (mm)",    "diameter",    3.,  30.,  diameter,   0.5),
        ("Longueur (mm)",    "length",      5., 100.,  length,     1.),
        ("Nb fils",          "n_wires",     6.,  32.,  n_wires,    2.),
        ("Angle tressage °", "braid_angle", 20., 80.,  braid_angle,1.),
        ("Rayon fil (mm)",   "wire_radius", 0.05, 0.8, wire_radius,0.05),
        ("Résolution",       "n_pts",       40., 200., n_pts,      10.),
    ]

    sliders = {}
    for row, (label, key, vmin, vmax, val, step) in enumerate(slider_cfg):
        sax = fig.add_subplot(gs[row, 1])
        sax.set_facecolor("#0d1117")
        sl  = Slider(sax, label, vmin, vmax, valinit=val, valstep=step,
                     color="#1e4a8a", track_color="#1e2d42")
        sl.label.set_color("#7a9ec0"); sl.valtext.set_color("#7a9ec0")
        sliders[key] = sl

    lines_plus  = []
    lines_minus = []
    info_text   = ax.text2D(0.02, 0.97, "", transform=ax.transAxes,
                            color="#7a9ec0", fontsize=8, va="top", family="monospace")

    def draw(diameter, length, n_wires, braid_angle, wire_radius, n_pts):
        nonlocal lines_plus, lines_minus
        for lc in lines_plus + lines_minus:
            lc.remove()
        lines_plus, lines_minus = [], []

        wires = wire_centerlines(diameter, length, n_wires, braid_angle, n_pts)
        pitch    = math.pi * diameter / math.tan(math.radians(braid_angle))
        n_turns  = length / pitch
        porosity = max(0, 1 - n_wires * 2 * wire_radius / (math.pi * diameter))

        for pts in wires["plus"]:
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            lc   = Line3DCollection(segs, colors="#4a9eff", linewidths=1.4, alpha=0.85)
            ax.add_collection3d(lc); lines_plus.append(lc)

        for pts in wires["minus"]:
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            lc   = Line3DCollection(segs, colors="#ff8c42", linewidths=1.4, alpha=0.85)
            ax.add_collection3d(lc); lines_minus.append(lc)

        R, L = diameter / 2, length / 2
        ax.set_xlim(-R*1.5, R*1.5); ax.set_ylim(-R*1.5, R*1.5); ax.set_zlim(-L*1.2, L*1.2)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z (mm)")
        ax.set_title(f"Stent tressé — ⌀{diameter} mm × {length} mm",
                     color="#7a9ec0", pad=8)
        info_text.set_text(
            f"pas={pitch:.2f} mm   tours={n_turns:.2f}   porosité={porosity:.2%}\n"
            f"● fils + (bleu)  ● fils − (orange)"
        )
        fig.canvas.draw_idle()

    def on_change(_):
        draw(
            diameter    = sliders["diameter"].val,
            length      = sliders["length"].val,
            n_wires     = int(sliders["n_wires"].val),
            braid_angle = sliders["braid_angle"].val,
            wire_radius = sliders["wire_radius"].val,
            n_pts       = int(sliders["n_pts"].val),
        )

    for sl in sliders.values():
        sl.on_changed(on_change)

    draw(diameter, length, n_wires, braid_angle, wire_radius, n_pts)
    plt.show()


# ── Exports ───────────────────────────────────────────────────────────────────

def to_csv(path="stent_wires.csv", **kw):
    import csv
    wires = wire_centerlines(**{k: v for k, v in kw.items() if k in _CL_KEYS})
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["wire_id", "family", "x", "y", "z"])
        for wid, (fam, pts) in enumerate((fam, pts)
                for fam in ("plus", "minus") for pts in wires[fam]):
            for x, y, z in pts:
                w.writerow([wid, fam, round(x,6), round(y,6), round(z,6)])
    print(f"[CSV] → {path}")


def to_obj(path="stent.obj", **kw):
    v, f, info = stent_mesh(**kw)
    with open(path, "w") as fp:
        fp.write("# braided_stent.py\n")
        for x, y, z in v: fp.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in f: fp.write(f"f {a+1} {b+1} {c+1}\n")
    print(f"[OBJ] {len(v)} verts, {len(f)} faces → {path} | {info}")


def to_stl(path="stent.stl", **kw):
    import trimesh
    v, f, info = stent_mesh(**kw)
    trimesh.Trimesh(vertices=v, faces=f, process=False).export(path)
    print(f"[STL] → {path} | {info}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    show()
    # to_csv("stent_wires.csv")
    # to_obj("stent.obj")
    # to_stl("stent.stl")   # pip install trimesh