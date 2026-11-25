"""Load the Fusion body export and attach each body to the simulated link that carries it.

sim/fusion_export/fusion_export.py writes the export: binary STL files in
millimetres in the Fusion root frame at the saved home pose, which is also the
frame of platform.example.json. Each body is assigned to one MuJoCo body from
where it touches the joints of the profile:

- a body reaching from the elbow axis to the spherical joint is that leg's rod,
- a body reaching from the motor axis to the elbow axis is that leg's crank,
- a body in the slab between the spherical joints and just above them is the plate,
- a body touching only the elbow axis follows the crank, and one touching only a
  spherical joint follows the rod if it hangs below the joint centre (rod-end
  housing) and the plate otherwise (the bolt clamping the ball to the bracket),
- everything else is fixed to the world.

An optional assign.json in the export directory maps body keys to
"world", "plate", "drive1".."drive3" or "rod1".."rod3" and overrides the rules.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import struct

import numpy as np

BODIES = ("world", "plate", *(f"{name}{i}" for name in ("drive", "rod") for i in (1, 2, 3)))
AXIS_RADIUS_MM = 25.
MOTOR_REACH_MM = 60.
ELBOW_REACH_MM = 40.
JOINT_RADIUS_MM = 20.
PLATE_BELOW_MM = 10.
PLATE_ABOVE_MM = 60.


@dataclass
class Part:
    key: str
    file: Path
    body: str
    reason: str
    rgba: tuple
    vertices: np.ndarray  # Unique vertices in millimetres, Fusion root frame.
    triangles: int
    material: str = ""    # Display material in mechanism.MATERIALS, from the Fusion appearance.


def material(appearance, key):
    """Display material for a Fusion appearance name and body key."""
    a, k = appearance.lower(), key.lower()
    if "moteus" in k:
        return "pcb"
    if k.endswith("/motor"):
        return "motor"
    for words, name in ((("glass", "acrylic", "clear"), "acrylic"), (("oak", "wood", "birch", "maple"), "oak"),
                        (("brass", "bronze"), "brass"), (("aluminum", "aluminium"), "aluminum"),
                        (("steel", "chrome", "nickel", "iron"), "steel"), (("black", "rubber"), "black"),
                        (("tough", "resin", "pla", "petg", "abs", "nylon", "epx", "plastic"), "printed")):
        if any(w in a for w in words):
            return name
    return ""


def read_stl(path):
    data = Path(path).read_bytes()
    if len(data) < 84:
        raise ValueError(f"{path} is not a binary STL")
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + 50*count:
        raise ValueError(f"{path} is not a binary STL with {count} triangles")
    record = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    triangles = np.frombuffer(data, record, count, 84)["v"].astype(float)
    if not np.isfinite(triangles).all():
        raise ValueError(f"{path} has non-finite vertices")
    return triangles


def _legs(cfg):
    legs = []
    for leg in cfg["legs"]:
        axis = np.array(leg["axis"], dtype=float)
        legs.append({"motor": np.array(leg["motor_mm"], dtype=float), "axis": axis/np.linalg.norm(axis),
                     "elbow": np.array(leg["elbow_mm"], dtype=float),
                     "tip": np.array(leg["ball_joint_mm"], dtype=float)})
    return legs


def _near_axis(v, point, axis, reach):
    along = (v-point) @ axis
    radial = np.linalg.norm((v-point) - along[:, None]*axis, axis=1)
    return bool(np.any((radial < AXIS_RADIUS_MM) & (np.abs(along) < reach)))


def classify(vertices, cfg):
    """Return (MuJoCo body name, reason) for one body's vertices in millimetres."""
    legs = _legs(cfg)
    tips = np.array([leg["tip"] for leg in legs])
    centre = tips.mean(axis=0)
    radius = np.linalg.norm(tips[:, :2]-centre[:2], axis=1).max()
    touch = []
    for leg in legs:
        touch.append((_near_axis(vertices, leg["motor"], leg["axis"], MOTOR_REACH_MM),
                      _near_axis(vertices, leg["elbow"], leg["axis"], ELBOW_REACH_MM),
                      bool(np.any(np.linalg.norm(vertices-leg["tip"], axis=1) < JOINT_RADIUS_MM))))
    for i, (_, elbow, tip) in enumerate(touch, 1):
        if elbow and tip:
            return f"rod{i}", "spans elbow and spherical joint"
    for i, (motor, elbow, _) in enumerate(touch, 1):
        if motor and elbow:
            return f"drive{i}", "spans motor and elbow axes"
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    horizontal = np.linalg.norm(vertices[:, :2]-centre[:2], axis=1).max()
    if (lo[2] > tips[:, 2].min()-PLATE_BELOW_MM and hi[2] < tips[:, 2].max()+PLATE_ABOVE_MM
            and horizontal < radius+80):
        return "plate", "in the plate slab"
    for i, (_, elbow, tip) in enumerate(touch, 1):
        if elbow:
            return f"drive{i}", "on the elbow axis"
        if tip:
            # A bolt through the rod-end ball is clamped to the plate; the rod-end housing hangs below.
            if (lo[2]+hi[2])/2 > legs[i-1]["tip"][2]:
                return "plate", "above a spherical joint"
            return f"rod{i}", "below a spherical joint"
    return "world", "touches no moving joint"


def load(directory, cfg):
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text())
    if manifest.get("units") != "mm":
        raise ValueError("The Fusion export must be in millimetres")
    overrides = {}
    if (directory/"assign.json").exists():
        overrides = json.loads((directory/"assign.json").read_text())
        unknown = {v for v in overrides.values()} - set(BODIES)
        if unknown:
            raise ValueError(f"assign.json names unknown bodies {sorted(unknown)}; use {BODIES}")
    parts = []
    for entry in manifest["bodies"]:
        path = directory/entry["file"]
        triangles = read_stl(path)
        vertices = np.unique(triangles.reshape(-1, 3), axis=0)
        if entry["key"] in overrides:
            body, reason = overrides[entry["key"]], "assign.json"
        else:
            body, reason = classify(vertices, cfg)
        rgba = tuple(float(np.clip(c, 0, 1)) for c in entry.get("rgba", (.62, .64, .67, 1)))
        parts.append(Part(entry["key"], path.resolve(), body, reason, rgba, vertices, len(triangles),
                          material(entry.get("appearance", ""), entry["key"])))
    if not parts:
        raise ValueError(f"No bodies in {directory/'manifest.json'}")
    return parts, manifest


def check(parts, manifest, cfg):
    """Return report lines and problems that make the export unusable for display."""
    lines, problems = [], []
    document = manifest.get("document", "?")
    version = manifest.get("version")
    lines.append(f"{document}" + (f" v{version}" if version is not None else "")
                 + f": {len(parts)} visible bodies, {sum(p.triangles for p in parts)} triangles, "
                 f"{len(manifest.get('hidden', []))} hidden, {len(manifest.get('failed', []))} failed")
    for failure in manifest.get("failed", []):
        problems.append(f"Fusion could not mesh {failure['key']}: {failure['error'].strip()}")
    placement = [b.get("placement_error_mm", 0) for b in manifest["bodies"]]
    if placement:
        lines.append(f"Largest mesh-to-B-rep centre difference in Fusion: {max(placement):.3f} mm")
    for name in BODIES:
        members = [p for p in parts if p.body == name]
        lines.append(f"  {name:7s} {len(members):3d} bodies")
        for p in members:
            lines.append(f"            {p.key}  ({p.reason})")
    for i in (1, 2, 3):
        for name in (f"drive{i}", f"rod{i}"):
            if not any(p.body == name and p.reason.startswith("spans") or p.body == name and p.reason == "assign.json"
                       for p in parts):
                problems.append(f"No exported body spans the joints of {name}. The export may be in another frame "
                                f"or pose than {name[:-1]} {i} of the platform profile.")
    legs = _legs(cfg)
    revolute = [j for j in manifest.get("joints", []) if j.get("type") == 1 and "origin_mm" in j and "axis" in j]
    if revolute:
        worst = 0.
        for joint in revolute:
            p, a = np.array(joint["origin_mm"]), np.array(joint["axis"], dtype=float)
            a /= np.linalg.norm(a)
            best = np.inf
            for leg in legs:
                for point in (leg["motor"], leg["elbow"]):
                    d = (p-point) - ((p-point) @ leg["axis"])*leg["axis"]
                    angle = np.degrees(np.arccos(min(1., abs(a @ leg["axis"]))))
                    best = min(best, np.linalg.norm(d) + angle)
            worst = max(worst, best)
        lines.append(f"{len(revolute)} revolute joints in the export; worst distance to a profile axis "
                     f"{worst:.3f} (mm + degrees). Joint data is informational.")
    return lines, problems


def plate_top_mm(parts):
    """Top face height of the widest plate body, normally the disk."""
    plate = [p for p in parts if p.body == "plate"]
    if not plate:
        return None
    widest = max(plate, key=lambda p: np.ptp(p.vertices[:, 0])*np.ptp(p.vertices[:, 1]))
    return float(widest.vertices[:, 2].max())
