"""Fusion export loading: link assignment, mesh placement and motion. Requires MuJoCo and NumPy."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"sim"))
import cad
from mechanism import make_model

FACES = [[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]]


def bar(a, b, width=12., height=6.):
    """Box from a to b in millimetres."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    x = (b-a)/np.linalg.norm(b-a)
    y = np.cross(x, [0, 0, 1.])
    y = y/np.linalg.norm(y) if np.linalg.norm(y) > 1e-6 else np.array([0, 1., 0])
    z = np.cross(x, y)
    return np.array([p + j*width/2*y + k*height/2*z for p in (a, b) for j in (-1, 1) for k in (-1, 1)])


def write_stl(path, corners):
    with open(path, "wb") as f:
        f.write(b"\0"*80 + struct.pack("<I", len(FACES)))
        for face in FACES:
            f.write(struct.pack("<12fH", 0, 0, 0, *corners[face].ravel(), 0))


class CadTest(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT/"platform.example.json").read_text())
        leg = self.cfg["legs"][0]
        m, e, t = (np.array(leg[k]) for k in ("motor_mm", "elbow_mm", "ball_joint_mm"))
        self.shapes = {"crank": bar(m, e), "rod": bar(e, t, 8, 5),
                       "pad": bar(t+[-15, 0, 30], t+[15, 0, 30], 30, 5),
                       "base": bar([-40, -40, -20], [40, 40, -20])}
        self.dir = tempfile.TemporaryDirectory()
        out = Path(self.dir.name)
        (out/"bodies").mkdir()
        entries = []
        for k, (name, corners) in enumerate(self.shapes.items()):
            write_stl(out/f"bodies/{k:03d}.stl", corners)
            entries.append({"file": f"bodies/{k:03d}.stl", "key": name, "rgba": [.5, .5, .5, 1]})
        (out/"manifest.json").write_text(json.dumps({"units": "mm", "bodies": entries}))

    def tearDown(self):
        self.dir.cleanup()

    def test_assignment_placement_and_motion(self):
        parts, manifest = cad.load(self.dir.name, self.cfg)
        self.assertEqual({p.key: p.body for p in parts},
                         {"crank": "drive1", "rod": "rod1", "pad": "plate", "base": "world"})
        lines, problems = cad.check(parts, manifest, self.cfg)
        self.assertEqual(len(problems), 4)  # Legs 2 and 3 have no links in this export.
        model = mujoco.MjModel.from_xml_string(make_model(self.cfg, 420, 170, 80, 55, parts, .0285))
        data = mujoco.MjData(model)

        def world_vertices(name):
            g = model.geom(name).id
            start, count = model.mesh_vertadr[model.geom_dataid[g]], model.mesh_vertnum[model.geom_dataid[g]]
            return data.geom_xpos[g] + model.mesh_vert[start:start+count] @ data.geom_xmat[g].reshape(3, 3).T

        mujoco.mj_forward(model, data)
        order = []
        for k, corners in enumerate(self.shapes.values()):
            distance = np.linalg.norm(world_vertices(f"cad{k}")[None]*1000-corners[:, None], axis=2)
            self.assertLess(distance.min(axis=1).max(), 1e-3)
            order.append(distance.argmin(axis=1))  # Mesh vertex of each box corner.
        data.ctrl[:] = [.25, -.15, .1]
        for _ in range(600):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        self.assertGreater(abs(data.joint("q1").qpos[0]), .1)
        crank, rod = world_vertices("cad0")[order[0]], world_vertices("cad1")[order[1]]
        elbow, tip = data.body("rod1").xpos, data.site("socket1").xpos
        # The far end of the crank and the near end of the rod stay on the moving elbow,
        # and the rod's far end follows the plate's socket.
        self.assertLess(np.linalg.norm(crank[4:].mean(axis=0)-elbow), 1e-6)
        self.assertLess(np.linalg.norm(rod[:4].mean(axis=0)-elbow), 1e-6)
        self.assertLess(np.linalg.norm(rod[4:].mean(axis=0)-tip), 1e-4)

    def test_override(self):
        (Path(self.dir.name)/"assign.json").write_text(json.dumps({"base": "plate"}))
        parts, _ = cad.load(self.dir.name, self.cfg)
        self.assertEqual({p.key: p.body for p in parts}["base"], "plate")


if __name__ == "__main__":
    unittest.main()
