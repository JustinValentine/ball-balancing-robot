"""Fit stereo calibration in millimetres from a ball-vision capture directory."""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

import cv2 as cv
import numpy as np


def board_detector(cfg):
    a = cv.aruco
    dims = (int(cfg["squares_x"]), int(cfg["squares_y"]))
    square, marker = cfg["square_mm"], cfg["marker_mm"]
    if min(dims) < 2 or not 0 < marker < square:
        raise ValueError("Check board dimensions and square/marker lengths")
    dictionary = a.getPredefinedDictionary(getattr(a, cfg["dictionary"]))
    if hasattr(a, "CharucoDetector"):
        board = a.CharucoBoard(dims, square, marker, dictionary)
        detector = a.CharucoDetector(board)

        def detect(image):
            pts, ids, _, _ = detector.detectBoard(image)
            return pts, ids

        return board.getChessboardCorners(), detect
    board = a.CharucoBoard_create(*dims, square, marker, dictionary)

    def detect(image):
        corners, ids, _ = a.detectMarkers(image, dictionary)
        if ids is None:
            return None, None
        _, pts, ids = a.interpolateCornersCharuco(corners, ids, image, board)
        return pts, ids

    return board.chessboardCorners, detect


def corners(image, detect):
    pts, ids = detect(image)
    return {} if ids is None else dict(zip(ids.ravel(), pts.reshape(-1, 2)))


def usable(ids, xyz):
    return len(ids) >= 6 and np.linalg.matrix_rank(xyz[ids, :2] - xyz[ids, :2].mean(0)) == 2


def calibrate(directory, config, output):
    xyz, detect = board_detector(config)
    mono = [([], []), ([], [])]
    obj, left, right = [], [], []
    size = None
    with (directory / "pairs.csv").open() as stream:
        for row in csv.DictReader(stream):
            views = []
            for index, side in enumerate(("left", "right")):
                image = cv.imread(str(directory / row[side]), cv.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"Unreadable image: {row[side]}")
                if size is None:
                    size = image.shape[::-1]
                if size != image.shape[::-1]:
                    raise ValueError("Mixed image sizes")
                pts = corners(image, detect)
                ids = sorted(pts)
                if usable(ids, xyz):
                    mono[index][0].append(xyz[ids])
                    mono[index][1].append(np.array([pts[i] for i in ids]))
                views.append(pts)
            common = sorted(views[0].keys() & views[1].keys())
            if usable(common, xyz):
                obj.append(xyz[common])
                left.append(np.array([views[0][i] for i in common]))
                right.append(np.array([views[1][i] for i in common]))
    if len(obj) < 8:
        raise ValueError(f"Only {len(obj)} usable stereo poses. Capture at least 8 varied poses with 6 shared noncollinear corners.")
    cameras = [cv.calibrateCamera(o, p, size, None, None)[:3] for o, p in mono]
    (rms1, k1, d1), (rms2, k2, d2) = cameras
    rms, _, _, _, _, rotation, translation, _, _ = cv.stereoCalibrate(
        obj, left, right, k1, d1, k2, d2, size, flags=cv.CALIB_FIX_INTRINSIC)
    if not np.isfinite([rms1, rms2, rms]).all() or not np.linalg.norm(translation) > 0:
        raise ValueError("Degenerate calibration. Check board geometry and pose diversity.")
    rig = json.loads((directory / "rig.json").read_text())
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".yml", delete=False) as tmp:
        name = tmp.name
    try:
        fs = cv.FileStorage(name, cv.FILE_STORAGE_WRITE)
        for key, val in dict(K1=k1, D1=d1, K2=k2, D2=d2, R=rotation, T=translation,
                             width=size[0], height=size[1], stereo_rms_px=rms,
                             left_guid=rig["left_guid"], right_guid=rig["right_guid"]).items():
            fs.write(key, val)
        fs.release()
        os.replace(name, output)
    finally:
        Path(name).unlink(missing_ok=True)
    print(f"{len(obj)} stereo poses. RMS pixels: left {rms1:.3f}, right {rms2:.3f}, stereo {rms:.3f}. Baseline {np.linalg.norm(translation):.2f} mm.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--board", type=Path, default=Path(__file__).with_name("board.json"))
    args = parser.parse_args()
    calibrate(args.directory, json.loads(args.board.read_text()), args.output)
