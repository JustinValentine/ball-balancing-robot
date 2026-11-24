"""Mechanism and ball physics checks. Requires MuJoCo and NumPy."""
import json
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"sim"))
from mechanism import loop_error_mm, make_model, step_frame
from plate_pose import Platform


class SimulationTest(unittest.TestCase):
    def test_home_and_motion(self):
        cfg = json.loads((ROOT/"platform.example.json").read_text())
        xml = make_model(cfg,420,170,80,55)
        self.assertEqual(xml,(ROOT/"sim/fusion-platform.xml").read_text())
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        for i,leg in enumerate(cfg["legs"],1):
            for name, xyz in ((f"drive{i}",leg["motor_mm"]), (f"rod{i}",leg["elbow_mm"])):
                np.testing.assert_allclose(data.body(name).xpos*1000, xyz, atol=1e-6)
            np.testing.assert_allclose(data.xaxis[model.joint(f"q{i}").id], leg["axis"], atol=1e-7)
            np.testing.assert_allclose(data.site(f"socket{i}").xpos*1000, leg["ball_joint_mm"], atol=1e-6)
        fk = Platform(cfg)
        contact = rebound = False
        max_tilt = 0.
        for _ in range(200):
            step_frame(model,data)
            self.assertLess(loop_error_mm(data), .01)
            q = np.array([data.joint(f"q{i}").qpos[0] for i in range(1,4)])
            observed = fk.normal(fk.zero+(q-fk.zero_crank)/(2*np.pi*fk.scale))
            n = data.body("plate").xmat.reshape(3,3)[:,2]
            np.testing.assert_allclose(observed,fk.rotation@n,atol=1e-4)
            max_tilt = max(max_tilt,np.hypot(*n[:2]))
            p = {model.geom("acrylic").id,model.geom("ball_surface").id}
            contact |= any(set(c.geom)==p for c in data.contact)
            separation = n @ (data.body("ball").xpos-data.site("surface").xpos)
            rebound |= contact and separation > .019
        self.assertTrue(contact and rebound)
        self.assertGreater(max_tilt,.03)


if __name__ == "__main__":
    unittest.main()
