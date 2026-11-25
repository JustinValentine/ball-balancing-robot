"""Closed-loop checks of the C++ controller on the simulated robot.

Requires MuJoCo, NumPy and the built library: cmake -S control -B control/build && cmake --build control/build
"""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"sim"))
from control_sim import ControlSim, Plant, apexes, impacts


def bounce_impacts(log, start):
    times, points = impacts(log, start)
    gap, t = log["gap"], log["t"]
    return points[[k for k, tk in enumerate(times) if gap[max(0, np.searchsorted(t, tk)-60)] > .02]]


class ControlTest(unittest.TestCase):
    def test_balance_recovers_and_holds(self):
        sim = ControlSim()
        sim.start(ball_offset=(.05, -.03))
        sim.mode("balance")
        sim.run(4)
        log = sim.log.arrays()
        error = np.linalg.norm(log["ball"][:, :2]-sim.centre[:2], axis=1)
        t = log["t"]-log["t"][0]
        self.assertLess(t[np.flatnonzero(error > .003)[-1]], 1.5)
        self.assertLess(np.sqrt(np.mean(error[t > 2]**2)), .001)

    def test_bounce_catch_and_robustness(self):
        for plant in (Plant(), Plant(restitution=.82, vision_latency=.010, seed=4)):
            with self.subTest(plant=plant):
                sim = ControlSim(plant=plant)
                sim.start()
                sim.apex(.10)
                sim.mode("bounce")
                start = sim.data.time
                sim.run(9)
                log = sim.log.arrays()
                peaks = apexes(log, start+4)[:, 1]
                self.assertGreater(len(peaks), 12)
                self.assertLess(abs(peaks.mean()-.10), .005)
                self.assertLess(peaks.std(), .006)
                points = bounce_impacts(log, start+4)
                self.assertLess(np.sqrt(np.mean(np.sum((points[:, :2]-sim.centre[:2])**2, axis=1))), .008)
                sim.mode("balance")
                sim.run(3)
                log = sim.log.arrays()
                error = np.linalg.norm(log["ball"][:, :2]-sim.centre[:2], axis=1)
                self.assertLess(error.max(), .2)          # Never fell off.
                self.assertLess(error[-1], .003)
                self.assertTrue(log["contact"][-200:].any())


    def test_gravity_calibration(self):
        payload = (.15, (.06, .02, .01))   # Unknown to the controller's CAD guess.
        cal = ControlSim(plant=Plant(payload=payload, torque_hysteresis=.04))
        cal.start()
        prior = cal.gravity()
        fitted = cal.calibrate(report=False)
        sag = {}
        for name, params in (("prior", prior), ("fitted", fitted)):
            sim = ControlSim(plant=Plant(kp=10, kd=.3, payload=payload, crank_friction=0))
            sim.gravity(params)
            sim.start()
            commands = []
            sim.mode("balance")
            sim.run(3, lambda s: commands.append(list(s.out.q)))
            log = sim.log.arrays()
            keep = log["t"]-log["t"][0] > 1.5
            sag[name] = np.abs(np.array(commands)[keep]-log["crank"][keep]).mean()
        self.assertLess(sag["fitted"], .001)
        self.assertLess(sag["fitted"], sag["prior"]/3)


    def test_circle(self):
        sim = ControlSim()
        sim.start()
        sim.mode("balance")
        sim.run(1)
        sim.circle(.08, 4)
        sim.run(8)
        log = sim.log.arrays()
        keep = log["t"] > log["t"][-1]-4
        radius = np.linalg.norm(log["ball"][keep, :2]-sim.centre[:2], axis=1)
        self.assertLess(abs(radius.mean()-.08), .003)
        self.assertLess(radius.std(), .002)

    def test_song_on_the_beat(self):
        sim = ControlSim()
        sim.start()
        sim.apex(.08)
        sim.mode("bounce")
        sim.song("twinkle")
        sim.run(16)
        notes = sim.log.notes
        self.assertGreater(len(notes), 40)
        times, _ = impacts(sim.log.arrays(), 0)
        error = np.array([times[np.argmin(np.abs(times-beat))]-beat for _, _, _, beat in notes[15:]])
        self.assertLess(np.sqrt(np.mean(error**2)), .012)


if __name__ == "__main__":
    unittest.main()
