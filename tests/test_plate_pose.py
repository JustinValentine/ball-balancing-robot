import io
import json
import math
from pathlib import Path
import sys
import unittest
from contextlib import redirect_stdout

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plate_pose import Platform, convert


class PlatformTest(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / "platform.example.json").read_text())
        self.cfg["level_output_turns"] = [.2, -.11, .51]
        self.cfg["zero_crank_rad"] = [.84, .8, .86]   # Zeroed at a resting pose, not the CAD pose.
        self.cfg["crank_turns_per_output_turn"] = [-1, .5, 1]
        self.cfg["left_from_base_rotation"] = [[0,-1,0],[1,0,0],[0,0,1]]
        self.model = Platform(self.cfg)

    def measured_turns(self, tilt, direction, yaw):
        # Independent inverse fixture. Specify a rigid platform orientation,
        # solve its translation from the three fixed axial offsets, then solve
        # the two links in each actuator plane using the cosine rule.
        a, b, z = np.deg2rad([tilt, direction, yaw])
        axis = np.array([-math.sin(b), math.cos(b), 0])
        cross = np.cross(axis, np.eye(3)).T
        rot = np.eye(3) + np.sin(a)*cross + (1-np.cos(a))*cross@cross
        rot = rot@np.array([[np.cos(z),-np.sin(z),0], [np.sin(z),np.cos(z),0], [0,0,1]])
        motor = self.model.motor
        home = motor + self.model.crank + self.model.rod
        points = home @ rot.T
        axes = self.model.axis
        points += np.linalg.solve(axes, np.sum(axes*(home-points), axis=1))
        def planar(v):
            return v - axes * np.sum(v*axes, axis=1)[:, None]
        u, v, target = map(planar, (self.model.crank, self.model.rod, points-motor))
        crank, rod, reach = [np.linalg.norm(arr, axis=1) for arr in (u, v, target)]
        direction = np.arctan2(np.sum(axes*np.cross(u,target),axis=1), np.sum(u*target,axis=1))
        opening = np.arccos((reach**2+crank**2-rod**2)/(2*crank*reach))
        angles = direction - np.sign(np.sum(np.cross(u,v)*axes, axis=1))*opening
        turns = self.model.zero + (angles-self.model.zero_crank)/(2*np.pi*self.model.scale)
        return turns, self.model.rotation@rot[:, 2]

    def test_round_trip(self):
        for yaw in [-2,0,2]:
            for tilt in [0,4,8]:
                for direction in range(0, 360, 30):
                    q, expected = self.measured_turns(tilt,direction,yaw)
                    np.testing.assert_allclose(self.model.normal(q), expected, atol=1e-8)

    def test_stream(self):
        # Encoder readings of a level plate (the encoders were zeroed at another pose).
        level, _ = self.measured_turns(0, 0, 0)
        out = io.StringIO()
        with redirect_stdout(out):
            convert(self.model, io.StringIO("100000 " + " ".join(f"{t:.12f}" for t in level) + "\n"))
        np.testing.assert_allclose(np.fromstring(out.getvalue(), sep=" "), [100000,0,0,1], atol=1e-8)

    def test_invalid_sample(self):
        with self.assertRaises(ValueError):
            self.model.normal([float("nan"), 0, 0])
        with self.assertRaises(ValueError):
            self.model.normal([0, 0])


if __name__ == "__main__":
    unittest.main()
