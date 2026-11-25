"""Stereo tracking of two balls through the acrylic, on rendered images.

    python3 tests/test_two_balls.py build/ball-vision
"""
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"sim"))
from stereo import CameraPair, RIG, calibration, BALL_RADIUS


def main(binary):
    baseline, distance = .08, .17
    surface, normal = np.zeros(3), np.array([0., 0., 1.])
    eyes = [surface+[-baseline/2, 0, -distance], surface+[baseline/2, 0, -distance]]
    cams = CameraPair(np.eye(3), 420)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        calibration(tmp/"stereo.yml", baseline*1000)
        (tmp/"rig.json").write_text(json.dumps(RIG))
        settings = (ROOT/"tracker.yml").read_text().replace("balls: 1", "balls: 2")
        settings = settings.replace("ball_diameter_mm: 0.", f"ball_diameter_mm: {2000*BALL_RADIUS}")
        (tmp/"tracker.yml").write_text(settings)
        truth = {}
        with (tmp/"pairs.csv").open("w") as f:
            f.write("left_us,right_us,left,right\n")
            for i in range(40):
                stamp = 1_000_000+5000*i
                t = 5e-3*i
                # Two balls in separate columns, one rising and one falling.
                balls = np.array([[-.045, .004, BALL_RADIUS+.02+1.2*t], [.045, -.003, BALL_RADIUS+.25-1.0*t]])
                names = []
                for side, eye in (("left", eyes[0]), ("right", eyes[1])):
                    name = f"{side}-{i:05d}.png"
                    Image.fromarray(cams.image(eye, balls, surface, normal)).save(tmp/name)
                    names.append(name)
                f.write(f"{stamp},{stamp},{names[0]},{names[1]}\n")
                truth[stamp] = (balls-eyes[0])*1000   # Left optical frame equals the base frame here.
        out = subprocess.run([binary, "replay", str(tmp), str(tmp/"stereo.yml"), str(tmp/"tracker.yml")],
                             capture_output=True, text=True, check=True).stdout
    rows = list(csv.DictReader(io.StringIO(out)))
    found, errors = {}, []
    for r in rows:
        if int(r["valid"]):
            found.setdefault(int(r["left_us"]), []).append([float(r[k]) for k in ("x_mm", "y_mm", "z_mm")])
    for stamp, balls in truth.items():
        seen = np.array(found.get(stamp, []))
        assert len(seen) == 2, f"{len(seen)} balls at {stamp}"
        for b in balls:
            errors.append(np.min(np.linalg.norm(seen-b, axis=1)))
    errors = np.array(errors)
    assert errors.max() < 2.0, errors.max()
    print(f"Two balls found in all {len(truth)} pairs; position error median {np.median(errors):.2f} mm, "
          f"max {errors.max():.2f} mm")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT/"build"/"ball-vision"))
