"""Run with system python3, passing the built ball-vision binary as argv[1]."""
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(sys.argv[1]).resolve()
K = np.array([[470., 0, 320], [0, 472., 240], [0, 0, 1]])
BASELINE = 80.
SIZE = (640, 480)
RIG = dict(left_guid="0011223344556677", right_guid="0011223344556688")


def dataset(root, render, poses):
    root.mkdir()
    (root / "rig.json").write_text(json.dumps(RIG))
    with (root / "pairs.csv").open("w") as f:
        f.write("left_us,right_us,left,right\n")
        for i, pose in enumerate(poses):
            for side in (0, 1):
                assert cv.imwrite(str(root / f"{i}-{side}.pgm"), render(pose, side))
            f.write(f"{100000+i*5000},{100120+i*5000},{i}-0.pgm,{i}-1.pgm\n")


def test_calibration(root):
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
    if hasattr(cv.aruco, "CharucoDetector"):
        board = cv.aruco.CharucoBoard((12, 9), 100., 80., dictionary)
        image = board.generateImage((1200, 900))
    else:
        board = cv.aruco.CharucoBoard_create(12, 9, 100., 80., dictionary)
        image = board.draw((1200, 900))
    rng = np.random.default_rng(42)
    xyz = np.float32([[0, 0, 0], [1200, 0, 0], [1200, 900, 0], [0, 900, 0]])
    uv = xyz[:, :2].copy()
    poses = [(rng.uniform(-.45, .45, 3), rng.uniform([-650, -540, 870], [-450, -360, 1650]))
             for _ in range(20)]

    def render(pose, side):
        rot, t = pose
        corners = cv.projectPoints(xyz, rot, t - [BASELINE*side, 0, 0], K, None)[0]
        transform = cv.getPerspectiveTransform(uv, corners.reshape(-1, 2))
        return cv.warpPerspective(image, transform, SIZE, borderValue=175)

    directory = root / "poses"
    dataset(directory, render, poses)
    file = root / "calibration.yml"
    subprocess.run([sys.executable, str(ROOT / "calibrate.py"), str(directory), str(file)], check=True)
    fs = cv.FileStorage(str(file), cv.FILE_STORAGE_READ)
    baseline = np.linalg.norm(fs.getNode("T").mat())
    assert abs(baseline-BASELINE) < 2, baseline
    assert fs.getNode("stereo_rms_px").real() < 1
    fs.release()


def test_tracking(root):
    dist = np.array([-.19, .025, .001, -.001, .0])
    x, y = np.meshgrid(np.arange(SIZE[0]), np.arange(SIZE[1]))
    pixels = np.dstack((x, y)).astype(float)
    xy = cv.undistortPoints(pixels.reshape(-1, 1, 2), K, dist).reshape(480, 640, 2)
    rays = np.dstack((xy, np.ones((480, 640))))
    rays /= np.linalg.norm(rays, axis=2)[:, :, None]
    poses = [[-24, 5, 390], [-22, 5, 390], None, [145, -80, 430], [-80, -65, 380]]

    def render(pose, side):
        if pose is None and side == 1:
            return np.zeros((480, 640), np.uint8)
        ctr = np.array(pose if pose else [-20, 5, 390]) - [side * BASELINE, 0, 0]
        inside = rays @ ctr >= np.sqrt(ctr @ ctr - 15.**2)
        return np.where(inside, 223, 30).astype(np.uint8)

    directory = root / "balls"
    dataset(directory, render, poses)
    calib = root / "known.yml"
    fs = cv.FileStorage(str(calib), cv.FILE_STORAGE_WRITE)
    for key, val in dict(K1=K, D1=dist, K2=K, D2=dist, R=np.eye(3), T=np.array([[-BASELINE], [0], [0]]),
                         width=640, height=480, **RIG).items():
        fs.write(key, val)
    fs.release()
    air = root / "air.yml"
    air.write_text((ROOT / "tracker.yml").read_text().replace("acrylic_thickness_mm: 6.35", "acrylic_thickness_mm: 0."))
    command = [str(BINARY), "replay", str(directory), str(calib), str(air)]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    assert len(rows) == len(poses)
    for index, (pose, row) in enumerate(zip(poses, rows)):
        assert None not in row and row["processing_us"]
        assert bool(int(row["valid"])) == (pose is not None)
        if pose is not None:
            estimate = np.array([float(row[k]) for k in ("x_mm", "y_mm", "z_mm")])
            assert np.linalg.norm(estimate-pose) < 2., (estimate, pose)
        if index in (0, 2, 3):
            assert not row["vx_mm_s"]
    assert abs(float(rows[1]["vx_mm_s"])-400) < 75
    (directory / "rig.json").write_text(json.dumps(dict(left_guid=RIG["right_guid"], right_guid=RIG["left_guid"])))
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_acrylic(root):
    # Independent forward renderer: intersect the actual lower surface, refract
    # inside the slab, intersect the upper surface, then test sphere tangency.
    grid = np.indices((480, 640), dtype=float)
    rays = np.dstack(((grid[1]-320)/470, (grid[0]-240)/472, np.ones((480, 640))))
    rays /= np.linalg.norm(rays, axis=2, keepdims=True)
    normals = np.array([[0., 0., 1.], [.47, -.26, .85], [-.48, .32, .82]])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    centers = np.array([[0., 10., 270.], [79., -44., 255.], [-38., 30., 300.]])
    rotation = cv.Rodrigues(np.array([.02, .07, -.04]))[0]
    origin = np.array([80., 2., 1.])
    calib = root / "acrylic-calib.yml"
    fs = cv.FileStorage(str(calib), cv.FILE_STORAGE_WRITE)
    for key, value in dict(K1=K, D1=np.zeros(5), K2=K, D2=np.zeros(5), R=rotation,
                           T=(-rotation @ origin).reshape(3, 1), width=640, height=480, **RIG).items():
        fs.write(key, value)
    fs.release()

    def render(i, side):
        n = normals[i]
        d = rays @ rotation if side else rays
        eye = origin if side else np.zeros(3)
        cosine = d @ n
        entry = eye + d * ((150*n[2] - eye @ n)/cosine)[:, :, None]
        tangent = (d - cosine[:, :, None] * n)/1.49
        inward = np.sqrt(1 - (tangent*tangent).sum(axis=2))
        exit_point = entry + (tangent + inward[:, :, None]*n)*(6.35/inward)[:, :, None]
        relative = centers[i]-exit_point
        distance = np.linalg.norm(np.cross(relative, d), axis=2)
        return np.where(distance <= 18., 230, 30).astype(np.uint8)

    directory = root / "acrylic"
    dataset(directory, render, range(3))
    command = [str(BINARY), "replay", str(directory), str(calib)]
    pose = root / "normals.txt"
    pose.write_text("".join(f"{100000+i*5000} {n[0]} {n[1]} {n[2]}\n" for i, n in enumerate(normals)))

    def errors(args):
        output = subprocess.check_output(command + args, text=True)
        rows = list(csv.DictReader(io.StringIO(output)))
        xyz = np.array([[float(row[k]) for k in ("x_mm", "y_mm", "z_mm")] for row in rows])
        return np.linalg.norm(xyz-centers, axis=1)

    bare = errors([str(root / "air.yml")])
    corrected = errors([str(ROOT / "tracker.yml"), str(pose)])
    print("Acrylic synthetic Euclidean errors (mm): air model", bare, "corrected", corrected)
    assert np.all(corrected < bare) and max(corrected) < 1, corrected
    fixed = errors([str(ROOT / "tracker.yml")])
    assert abs(fixed[0]-corrected[0]) < 1e-7
    pose.write_text("100000 0 0 1\n")
    rows = list(csv.DictReader(io.StringIO(subprocess.check_output(
        command + [str(ROOT / "tracker.yml"), str(pose)], text=True))))
    assert rows[-1]["valid"] == "0"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        test_calibration(Path(tmp))
        test_tracking(Path(tmp))
        test_acrylic(Path(tmp))
    print("Stereo calibration, distorted sphere tracking, missing observation, ROI reacquisition and camera order checks passed.")
