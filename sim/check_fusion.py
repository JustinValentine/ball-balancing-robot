"""Check the saved Fusion v12 assembly coordinates used by the simulator."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import zipfile

import numpy as np

# Occurrence indices and B-rep names here are specific to these two saved assets.
ASSETS = {
    "spherical-parallel-mechanism": "7f6adc8951824312ba114dafed8fc901318059cad3087ad7eb2fb93ed40f66d1",
    "ARM": "cc0d654700b5e6171a8d82efedf748a7eebbe7976afff00f7cbc5eb10f335c72"
}
PREFIX = "FusionAssetName[Active]/"


def read(blob, name, tmp):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        try:
            return z.read(PREFIX+name)
        except NotImplementedError:
            # The F3D archives use ZIP method 93 (Zstandard).
            path = tmp/"asset.zip"
            path.write_bytes(blob)
            return subprocess.check_output(["7z","x","-so","-spd",str(path),PREFIX+name])


def check(archive, profile):
    with zipfile.ZipFile(archive) as z:
        graph = json.loads(z.read("DesignDescription.json"))["designDescription"]["designGraphs"][0]
        blobs = {obj["displayName"]: z.read(obj["relativePath"]) for obj in graph["designObjects"]
                 if obj["displayName"] in ASSETS}
    for name, digest in ASSETS.items():
        if hashlib.sha256(blobs[name]).hexdigest() != digest:
            raise ValueError(f"Different Fusion revision of {name}; re-identify joint occurrences")
    with tempfile.TemporaryDirectory() as path:
        tmp = Path(path)
        scene = read(blobs["spherical-parallel-mechanism"],"OGS.BlobFolder/OGS/DefaultScene/world",tmp)
        rod = read(blobs["ARM"],"Breps.BlobParts/BREP.62425240-e860-4fa8-8881-9edc097b6a06.smb",tmp)
    sphere = re.search(b"sphere\x0d\x07surface.{27}\x13(.{24})\x06",rod,re.S)
    if sphere is None:
        raise ValueError("Spherical rod-end surface missing")
    local = np.array([*struct.unpack("<3d",sphere[1]),1])
    transforms = [np.frombuffer(scene,"<f4",count=16,offset=m.end()+5).reshape(4,4).astype(float)
                  for m in re.finditer("OverrideTransformAttribute".encode("utf-16-le"),scene)]
    max_mm = max_axis = 0.
    for leg, start in zip(profile["legs"], (13,1,7), strict=True):
        motor, rod_end, elbow = transforms[start:start+3]
        # Scene uses row-vector affine transforms and centimetres.
        actual = [motor[3,:3], elbow[3,:3], (local@rod_end)[:3]]
        for key, point in zip(("motor_mm","elbow_mm","ball_joint_mm"),actual):
            max_mm = max(max_mm,np.linalg.norm(point*10-leg[key]))
        max_axis = max(max_axis,np.linalg.norm(motor[2,:3]-leg["axis"]))
    print(f"Fusion v12 + ARM v10: nine joint points, maximum difference {max_mm:.8f} mm; "
          f"three axes, maximum vector difference {max_axis:.3g}.")
    if max_mm > 1e-4 or max_axis > 1e-7:
        raise ValueError("The simulation profile disagrees with the archived assembly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("--profile",type=Path,default=Path(__file__).resolve().parents[1]/"platform.example.json")
    args=parser.parse_args()
    check(args.archive,json.loads(args.profile.read_text()))
