"""Simulate the Fusion platform, check the Fusion body export, or record a 200 Hz stereo tracking benchmark."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "sim"))
import cad
from mechanism import SURFACE_OFFSET, loop_error_mm, make_model, step_frame
from stereo import CameraPair, RIG, calibration
from plate_pose import Platform


def record(args, cfg, xml):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    estimator = Platform(cfg)
    cameras = CameraPair(estimator.rotation,args.disk_mm)
    out = args.output
    out.mkdir(parents=True,exist_ok=False)
    (out/"model.xml").write_text(xml)
    (out/"platform.json").write_text(json.dumps(cfg,indent=2))
    (out/"rig.json").write_text(json.dumps(RIG))
    calibration(out/"stereo.yml",args.baseline_mm)
    peak_gap = peak_angle = 0.
    contacts = 0
    with ((out/"pairs.csv").open("w") as pairs_file, (out/"truth.csv").open("w") as truth_file,
          (out/"joint-positions.txt").open("w") as positions, (out/"normals.txt").open("w") as normals):
        pairs, truth = csv.writer(pairs_file,lineterminator="\n"), csv.writer(truth_file,lineterminator="\n")
        pairs.writerow(["left_us","right_us","left","right"])
        truth.writerow(["left_us","x_mm","y_mm","z_mm","nx","ny","nz","contact","full_ball_in_both","loop_gap_mm","normal_error_deg"])
        for i in range(args.frames):
            stamp = 1_000_000 + i*5000
            ball = data.body("ball").xpos
            n = data.body("plate").xmat.reshape(3,3)[:,2]
            surface = data.site("surface").xpos
            q = np.array([data.joint(f"q{j}").qpos[0] for j in range(1,4)])
            turns = estimator.zero + (q-estimator.zero_crank)/(2*np.pi*estimator.scale)
            estimated = estimator.normal(turns)
            print(stamp,*turns,file=positions)
            print(stamp,*estimated,file=normals)
            names = []
            full_in_both = True
            for side in ("left","right"):
                eye = data.cam(side).xpos
                img = cameras.image(eye,ball,surface,n)
                mask = img >= 200
                full_in_both &= mask.any() and not (mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())
                filename = f"{side}-{i:05d}.png"
                Image.fromarray(img).save(out/filename, compress_level=1)
                names.append(filename)
            xyz = estimator.rotation @ (ball-data.cam("left").xpos)*1000
            optical_n = estimator.rotation @ n
            gap = loop_error_mm(data)
            angle = np.degrees(np.arccos(np.clip(estimated@optical_n,-1,1)))
            contact = any(set(c.geom)=={model.geom("acrylic").id,model.geom("ball_surface").id} for c in data.contact)
            peak_gap, peak_angle = max(peak_gap,gap),max(peak_angle,angle)
            contacts += contact
            pairs.writerow([stamp,stamp,*names])
            truth.writerow([stamp,*xyz,*optical_n,int(contact),int(full_in_both),gap,angle])
            step_frame(model,data)
    print(f"{args.frames} stereo pairs, {contacts} contact frames. Maximum loop gap {peak_gap:.4f} mm; "
          f"maximum encoder-derived normal error {peak_angle:.5f} degrees.")


def load_cad(args, cfg, required=False):
    """Return (parts, surface offset in m) from the Fusion export, or (None, default)."""
    if args.no_cad:
        return None, SURFACE_OFFSET
    directory = args.cad or ROOT/"sim"/"cad"
    if not (directory/"manifest.json").exists():
        message = (f"No Fusion export in {directory}. Run sim/fusion_export/fusion_export.py "
                   "in Fusion with the platform design open.")
        if required or args.cad:
            raise SystemExit(message)
        print(message, "Showing the simplified shapes.")
        return None, SURFACE_OFFSET
    parts, manifest = cad.load(directory, cfg)
    lines, problems = cad.check(parts, manifest, cfg)
    print("\n".join(lines))
    for problem in problems:
        print("WARNING:", problem)
    top = cad.plate_top_mm(parts)
    offset = SURFACE_OFFSET
    if top is not None:
        tip_z = np.mean([leg["ball_joint_mm"][2] for leg in cfg["legs"]])
        offset = (top-tip_z)/1000
        print(f"CAD playing surface: {offset*1000:.2f} mm above the spherical-joint centres "
              f"(the simplified model assumes {SURFACE_OFFSET*1000:.0f} mm). The ball uses the CAD value.")
    return parts, offset


def snapshot(model, path):
    """Save the home pose and a driven pose from two directions."""
    from PIL import Image
    data = mujoco.MjData(model)
    model.vis.global_.offwidth, model.vis.global_.offheight = 1600, 1200
    renderer = mujoco.Renderer(model, 600, 800)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = model.stat.center
    camera.distance = 1.25*model.stat.extent
    tiles, gaps = [], []
    for drive in ((0, 0, 0), (.3, -.2, .1)):
        mujoco.mj_resetData(model, data)
        data.ctrl[:] = drive
        for _ in range(1200):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        gaps.append(loop_error_mm(data))
        for azimuth, elevation in ((115, -20), (205, -35)):
            camera.azimuth, camera.elevation = azimuth, elevation
            renderer.update_scene(data, camera)
            tiles.append(renderer.render())
    rows = [np.hstack(tiles[0:2]), np.hstack(tiles[2:4])]
    Image.fromarray(np.vstack(rows)).save(path)
    print(f"Wrote {path}: top row home pose, bottom row cranks at +0.3, -0.2, +0.1 rad "
          f"(loop gaps {gaps[0]:.3f} and {gaps[1]:.3f} mm).")


def control(args, cfg):
    """Run the C++ controller against the simulated robot, in the viewer or headless."""
    from control_sim import ControlSim, PHASES, Plant, apexes, impacts
    parts = None
    if args.view and not args.no_cad and (ROOT/"sim"/"cad"/"manifest.json").exists():
        parts, _ = cad.load(ROOT/"sim"/"cad", cfg)
    plant = Plant(restitution=args.restitution, vision_latency=args.latency_ms/1000, seed=args.seed,
                  kp=args.kp, kd=args.kd, payload=(args.payload_g/1000, (.06, .02, .01)))
    sim = ControlSim(args.config, plant, parts)
    sim.start(ball_offset=(args.ball_x_mm/1000, args.ball_y_mm/1000))
    if args.calibrate:
        print(f"Gravity calibration on the simulated robot (it carries a {plant.payload[0]*1000:.0f} g payload the "
              "controller's CAD guess does not know about):")
        sim.calibrate()
        sim.start(ball_offset=(args.ball_x_mm/1000, args.ball_y_mm/1000))
    sim.target(args.target_x_mm/1000, args.target_y_mm/1000)
    sim.apex(args.apex_mm/1000)
    sim.mode(args.mode)
    if args.view:
        import time
        import mujoco.viewer
        state = {"apex": args.apex_mm/1000, "x": args.target_x_mm/1000, "y": args.target_y_mm/1000}
        pending = []
        keys = {295: "hold", 296: "balance", 297: "bounce"}

        def key(code):
            if code in keys:
                pending.append(("mode", keys[code]))
            elif code in (266, 267):
                state["apex"] = max(.02, state["apex"]+(.01 if code == 266 else -.01))
                pending.append(("apex", state["apex"]))
            elif code in (260, 261, 268, 269):
                dx, dy = {260: (-.01, 0), 261: (.01, 0), 268: (0, .01), 269: (0, -.01)}[code]
                state["x"], state["y"] = state["x"]+dx, state["y"]+dy
                pending.append(("target", (state["x"], state["y"])))
        print("F6 hold, F7 balance, F8 bounce, Page Up/Down apex +-10 mm, Insert/Delete/Home/End move the "
              "target -x/+x/+y/-y by 10 mm. Key 3 shows the simplified physics shapes.")
        if sys.platform == "darwin" and "MJPYTHON_BIN" not in os.environ:
            raise SystemExit("On macOS the viewer needs mjpython: mjpython simulate.py control --view ...")
        with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=key) as viewer:
            start, sim_start = time.perf_counter(), sim.data.time
            last = -1
            while viewer.is_running():
                while pending:
                    what, value = pending.pop(0)
                    {"mode": sim.mode, "apex": sim.apex}.get(what, lambda v: sim.target(*v))(value)
                    print(what, value)
                sim.run(.01)
                viewer.sync()
                if PHASES[sim.out.phase] != last:
                    last = PHASES[sim.out.phase]
                    print(f"{sim.data.time-sim_start:7.2f} s  {last}")
                lag = (sim.data.time-sim_start)-(time.perf_counter()-start)
                if lag > 0:
                    time.sleep(lag)
        return
    sim.run(args.seconds)
    log = sim.log.arrays()
    t = log["t"]-log["t"][0]
    phases = log["phase"]
    changes = [(round(float(t[i]), 2), PHASES[phases[i]]) for i in np.flatnonzero(np.diff(phases))+1]
    print("Phases:", ", ".join(f"{p} at {tt} s" for tt, p in changes) or PHASES[phases[0]])
    error = np.linalg.norm(log["ball"][:, :2]-sim.centre[:2]-[args.target_x_mm/1000, args.target_y_mm/1000], axis=1)*1000
    last = t > t[-1]-2
    if args.mode == "balance":
        settled = t[np.flatnonzero(error > 3)[-1]] if (error > 3).any() else 0
        print(f"Balance: within 3 mm of the target after {settled:.2f} s; last 2 s RMS {np.sqrt((error[last]**2).mean()):.2f} mm")
    elif args.mode == "bounce":
        peaks = apexes(log, log["t"][0]+3)
        times, points = impacts(log, log["t"][0]+3)
        if len(peaks):
            print(f"Bounce after 3 s: apex {peaks[:, 1].mean()*1000:.1f} +- {peaks[:, 1].std()*1000:.1f} mm "
                  f"(target {args.apex_mm:.0f}) over {len(peaks)} bounces")
            gap, tt = log["gap"], log["t"]
            real = [k for k, tk in enumerate(times) if gap[max(0, np.searchsorted(tt, tk)-60)] > .02]
            if real:
                offset = (points[real, :2]-sim.centre[:2])*1000-[args.target_x_mm, args.target_y_mm]
                print(f"Impacts: mean offset {np.round(offset.mean(axis=0), 1)} mm, RMS {np.sqrt((offset**2).sum(axis=1).mean()):.1f} mm")
        else:
            print("No bounces")
    torque = np.abs(log["torque"])
    print(f"Peak torque {torque.max():.2f} N m, saturated {np.mean(torque >= sim.plant.torque_limit-1e-6)*100:.1f}% of the time; "
          f"cranks {np.round(log['crank'].min(axis=0), 2)} to {np.round(log['crank'].max(axis=0), 2)} rad")


def score(args):
    with (args.output/"truth.csv").open() as f:
        truth=list(csv.DictReader(f))
    with args.estimates.open() as f:
        measured=list(csv.DictReader(f))
    if not truth or [r["left_us"] for r in measured] != [r["left_us"] for r in truth]:
        raise SystemExit("Replay timestamps must match every simulated stereo pair in order")
    errors = []
    for a,b in zip(truth,measured):
        if int(b["valid"]):
            errors.append(np.linalg.norm([float(b[k])-float(a[k]) for k in ("x_mm","y_mm","z_mm")]))
    if not errors or not np.isfinite(errors).all():
        raise SystemExit("No finite tracking result")
    full = [(a,b) for a,b in zip(truth,measured) if int(a["full_ball_in_both"])]
    print(f"Full silhouettes in both cameras: {len(full)} pairs, {sum(int(b['valid']) for _,b in full)} detected.")
    print(f"Detected {len(errors)}/{len(truth)}. Position error in mm: median {np.median(errors):.3f}, "
          f"p95 {np.percentile(errors,95):.3f}, maximum {max(errors):.3f}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    view=sub.add_parser("view",help="Open the interactive physics viewer with motor sliders")
    export=sub.add_parser("model",help="Write a standalone MuJoCo XML model")
    inspect=sub.add_parser("cad",help="Report how the Fusion export attaches to the links and save snapshots")
    run=sub.add_parser("control",help="Run the C++ ball controller on the simulated robot")
    run.add_argument("--config",type=Path,default=ROOT/"control.example.json")
    run.add_argument("--mode",choices=("hold","balance","bounce"),default="balance")
    run.add_argument("--apex-mm",type=float,default=100)
    run.add_argument("--seconds",type=float,default=15,help="Headless run length")
    run.add_argument("--view",action="store_true",help="Open the viewer and run in real time")
    run.add_argument("--no-cad",action="store_true")
    run.add_argument("--ball-x-mm",type=float,default=40)
    run.add_argument("--ball-y-mm",type=float,default=-25)
    run.add_argument("--target-x-mm",type=float,default=0)
    run.add_argument("--target-y-mm",type=float,default=0)
    run.add_argument("--restitution",type=float,default=.88,help="Simulated ball, not the controller's value")
    run.add_argument("--latency-ms",type=float,default=6)
    run.add_argument("--seed",type=int,default=1)
    run.add_argument("--calibrate",action="store_true",help="Run the gravity calibration first")
    run.add_argument("--kp",type=float,default=25,help="Simulated servo stiffness, N m/rad")
    run.add_argument("--kd",type=float,default=.45,help="Simulated servo damping, N m s/rad")
    run.add_argument("--payload-g",type=float,default=0,help="Unmodelled mass on the plate, 60 mm off centre")
    capture=sub.add_parser("record",help="Record headless simulated images, encoders and truth")
    verify=sub.add_parser("score",help="Compare C++ replay results with physics ground truth")
    for p in (view,export,capture,inspect):
        p.add_argument("--platform",type=Path,default=ROOT/"platform.example.json")
        p.add_argument("--disk-mm",type=float,default=420,help="CAD diameter; use 330 for the stated physical disk")
        p.add_argument("--distance-mm",type=float,default=170,help="Home top surface to optical center")
        p.add_argument("--baseline-mm",type=float,default=80)
        p.add_argument("--drop-mm",type=float,default=55,help="Initial clearance between ball and disk")
    for p in (view,export,inspect):
        p.add_argument("--cad",type=Path,help="Fusion export directory (default sim/cad when present)")
        p.add_argument("--no-cad",action="store_true",help="Show the simplified shapes only")
    inspect.add_argument("--snapshot",type=Path,default=ROOT/"sim"/"cad-check.png")
    capture.add_argument("--frames",type=int,default=200)
    for p in (export,capture,verify):
        p.add_argument("output",type=Path)
    verify.add_argument("estimates",type=Path)
    args=parser.parse_args()
    if args.action=="score":
        score(args)
        return
    if args.action=="control":
        cfg=json.loads((args.config.parent/json.loads(args.config.read_text())["platform"]).read_text())
        control(args,cfg)
        return
    if not all(np.isfinite(x) and x>0 for x in (args.disk_mm,args.distance_mm,args.baseline_mm)) or not np.isfinite(args.drop_mm) or args.drop_mm<0:
        parser.error("Invalid geometric parameters")
    cfg=json.loads(args.platform.read_text())
    Platform(cfg)  # Validate the geometry before generating a model.
    parts, offset = (None, SURFACE_OFFSET) if args.action=="record" else load_cad(args,cfg,args.action=="cad")
    # Rest the floor under the lowest fixed CAD part.
    floor = min(p.vertices[:,2].min() for p in parts if p.body=="world")/1000 if parts and any(p.body=="world" for p in parts) else None
    xml=make_model(cfg,args.disk_mm,args.distance_mm,args.baseline_mm,args.drop_mm,parts,offset,floor)
    if args.action=="cad":
        snapshot(mujoco.MjModel.from_xml_string(xml),args.snapshot)
    elif args.action=="record":
        if args.frames<1:
            parser.error("Expected a positive frame count")
        record(args,cfg,xml)
    elif args.action=="model":
        with args.output.open("x") as f:
            f.write(xml)
    else:
        import mujoco.viewer as viewer
        viewer.launch(mujoco.MjModel.from_xml_string(xml))


if __name__=="__main__":
    main()
