"""Run the controller on many simulated robots whose hard-to-measure parameters are drawn at random.

    python3 sim/randomize.py [robots] [first seed]

Each robot balances, bounces at 200 mm and juggles at 400 mm. The ranges are my guesses at how far
the real robot could be from what the controller is told, not measurements.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from control_sim import ControlSim, Plant, apexes  # noqa: E402

RANGES = {
    # Ball and plate.
    "restitution": (.80, .93),
    "friction": (.2, .6),                 # Ball on acrylic; sets how much sideways surface motion it picks up.
    "ball_mass": (.0026, .0028),
    # Motors and drive.
    "torque_scale": (.85, 1.15),          # Torque constant error.
    "crank_friction": (0, .05),
    "torque_hysteresis": (0, .05),
    "rotor_inertia": (1.5e-4, 6e-4),
    "kp": (20, 30),
    "kd": (.35, .55),
    "command_delay": (.0005, .003),
    "payload_kg": (0, .08),               # Glue, screws, cables on the plate.
    "payload_offset": (-.03, .03),
    "crank_offset": (-.015, .015),        # Zeroing error, rad (about 0.9 deg).
    "geometry_error": (0, .001),          # Build error per joint point and axis, m.
    # Cameras.
    "camera_rotation": (-.007, .007),     # Stereo-to-base calibration error, rad (0.4 deg).
    "camera_offset": (-.002, .002),       # m.
    "vision_latency": (.004, .012),
    "timestamp_offset": (0, .006),
    "latency_jitter": (0, .002),
    "noise": (1, 2),                      # Times the default noise, growing with range.
    "outliers": (0, .01),
    "dropout": (0, .05),
}


def draw(seed):
    r = np.random.default_rng(10_000+seed)
    u = {k: r.uniform(*v) for k, v in RANGES.items() if k not in ("payload_offset", "crank_offset",
                                                                    "camera_rotation", "camera_offset")}
    vec = {k: tuple(r.uniform(*RANGES[k], 3)) for k in ("payload_offset", "crank_offset",
                                                          "camera_rotation", "camera_offset")}
    plant = Plant(seed=seed, restitution=u["restitution"], friction=u["friction"], ball_mass=u["ball_mass"],
                  torque_scale=u["torque_scale"], crank_friction=u["crank_friction"],
                  torque_hysteresis=u["torque_hysteresis"], rotor_inertia=u["rotor_inertia"], kp=u["kp"],
                  kd=u["kd"], command_delay=u["command_delay"], payload=(u["payload_kg"], vec["payload_offset"]),
                  crank_offset=vec["crank_offset"], geometry_error=u["geometry_error"],
                  camera_rotation=vec["camera_rotation"], camera_offset=vec["camera_offset"],
                  vision_latency=u["vision_latency"], timestamp_offset=u["timestamp_offset"],
                  latency_jitter=u["latency_jitter"], noise_xy=.0003*u["noise"], noise_depth=.0008*u["noise"],
                  noise_growth=True, outliers=u["outliers"], dropout=u["dropout"])
    return plant, u


def on_plate(sim, ball, after):
    """Seconds after `after` that the ball stayed on the plate."""
    t = np.array(sim.log.t)
    b = np.array(ball)
    off = ((np.linalg.norm(b[:, :2]-sim.centre[:2], axis=1) > .16) | (b[:, 2] < sim.centre[2]-.02)) & (t > after)
    return (t[np.argmax(off)]-after) if off.any() else t[-1]-after


def one(seed):
    plant, u = draw(seed)
    result = {"seed": seed, **u}
    # Balance: hold the centre, then hold 40 mm off it.
    s = ControlSim(plant=plant)
    s.start(ball_offset=(.03, -.02))
    s.mode("balance")
    s.run(5)
    s.target(.04, 0)
    s.run(4)
    L = s.log.arrays()
    t = L["t"]-L["t"][0]
    k = (t > 3) & (t < 5)
    result["balance_mm"] = np.linalg.norm(L["ball"][k, :2]-s.centre[:2], axis=1).mean()*1000
    k = t > 7.5
    result["offset_mm"] = np.linalg.norm(L["ball"][k, :2]-s.centre[:2]-[.04, 0], axis=1).mean()*1000
    result["balance_kept"] = on_plate(s, s.log.ball, 0) >= t[-1]-.05
    # Bounce at 200 mm.
    s = ControlSim(plant=plant)
    s.start()
    s.apex(.2)
    s.mode("bounce")
    s.run(15)
    L = s.log.arrays()
    a = apexes(L)
    late = a[a[:, 0]-L["t"][0] > 6, 1] if len(a) else np.array([])
    result["bounce_kept"] = on_plate(s, s.log.ball, 0) >= 14.9
    result["apex_err_mm"] = np.sqrt(np.mean((late-.2)**2))*1000 if len(late) else np.nan
    result["torque_sat"] = np.mean(np.abs(L["torque"]) >= plant.torque_scale*plant.torque_limit-1e-6)
    # Juggle at 400 mm.
    s = ControlSim(plant=plant, balls=2)
    s.start()
    s.apex(.4)
    s.mode("juggle")
    s.run(6.1+.05*(seed % 7))
    s.drop_second((.045, 0), .25)
    dropped = s.data.time
    s.run(20)
    result["juggle_s"] = min(on_plate(s, s.log.ball, dropped), on_plate(s, s.log.ball2, dropped))
    return result


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    first = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    with Pool() as pool:
        rows = pool.map(one, range(first, first+n))
    bal = np.array([r["balance_mm"] for r in rows])
    off = np.array([r["offset_mm"] for r in rows])
    apex = np.array([r["apex_err_mm"] for r in rows])
    jug = np.array([r["juggle_s"] for r in rows])
    print(f"{n} random robots")
    print(f"  balance: kept {sum(r['balance_kept'] for r in rows)}/{n}; distance from centre median "
          f"{np.median(bal):.1f} mm, worst {bal.max():.1f}; 40 mm target off by median {np.median(off):.1f}, "
          f"worst {off.max():.1f} mm")
    print(f"  bounce 200 mm: kept {sum(r['bounce_kept'] for r in rows)}/{n}; apex error rms median "
          f"{np.nanmedian(apex):.0f} mm, worst {np.nanmax(apex):.0f} mm; torque at the limit up to "
          f"{max(r['torque_sat'] for r in rows)*100:.1f}% of the time")
    print(f"  juggle 400 mm: kept both 20 s {(jug >= 19.8).sum()}/{n}, shortest {jug.min():.1f} s")
    bad = [r for r in rows if not r["balance_kept"] or not r["bounce_kept"] or r["juggle_s"] < 19.8]
    if bad:
        # Where in its range each parameter sat for the failures, 0 = low end, 1 = high end.
        print("  failures (parameters as 0 = low end of range, 1 = high end):")
        for r in bad:
            what = [w for w, ok in (("balance", r["balance_kept"]), ("bounce", r["bounce_kept"]),
                                    ("juggle", r["juggle_s"] >= 19.8)) if not ok]
            ends = {k: (r[k]-RANGES[k][0])/(RANGES[k][1]-RANGES[k][0]) for k in r if k in RANGES}
            extreme = ", ".join(f"{k} {v:.2f}" for k, v in sorted(ends.items(), key=lambda kv: -abs(kv[1]-.5))[:5])
            print(f"    seed {r['seed']}: {'/'.join(what)}; most extreme: {extreme}")


if __name__ == "__main__":
    main()
