# Ball vision

Linux stereo acquisition and ball tracking for two Point Grey Dragonfly Express cameras. The C++17 executable uses libdc1394 DMA capture and OpenCV on 640 × 480 MONO8 images. A separate Python command fits the calibration. Live capture runs headless.

## Build and identify the cameras

On Ubuntu 24.04, from this directory:

```sh
sudo apt install build-essential cmake pkg-config libdc1394-dev libopencv-dev python3-opencv
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
build/ball-vision list
```

Connect each camera to an independent S800 bus. The Linux user needs read/write access to the camera's /dev/fw* device. Copy the two printed GUIDs into camera.yml in physical left/right order. The list command also reports available Format7 modes, MONO8 support and IIDC trigger capabilities. Use the same camera order, ROI, focus, aperture and mounts for calibration and tracking.

Start with a supported full-resolution Format7 mode. The configuration requests S800 and the largest supported packet. If bandwidth allocation fails, verify the independent buses and use a supported packet size. For the 200 Hz target, 307,200 bytes must fit in 40 FireWire cycles per frame, requiring at least 7,680 image bytes per packet on average. The fixed VGA MONO8 option takes an IIDC rate index printed by list. The camera's sensor/readout mode can impose a lower triggered rate than the transport.

## Check acquisition

```sh
build/ball-vision probe camera.yml
```

Start the shared hardware trigger once both cameras report armed. The example selects external trigger mode 0, source 0, and retains the camera's existing polarity, shutter and gain. Confirm the electrical interface separately. To lock the shutter and gain, enter exposure_us and gain_raw. Mode 14 is selectable on cameras that advertise it. The Nucleo supplies the timing. trigger_period_us describes that timing to the pairing check.

Probe reports accepted pairs/s, cumulative frames dequeued per camera, corrupt/backlogged frames discarded, unmatched frames, and the latest compared timestamp difference. With the 200 Hz trigger enabled, look for sustained 200 pairs/s. A slow reader of tracking stdout or slow recording disk can cause drops. The capture loop drains DMA backlog to favor current observations.

Pairing uses Linux libdc1394 capture timestamps in microseconds and keeps a newer frame when its partner is missing. Verify on your rig that both timestamp streams refer to the same host time base and that the measured same-trigger difference is stable. Use an identifiable common optical event with deliberately spaced triggers for commissioning. Set right_offset_us to the measured right-minus-left difference for the same trigger, and keep the tolerance below half a period. Timing matches alone cannot prove exposure sync when one stream is displaced by a whole period. These timestamps describe capture. Absolute exposure-to-controller latency requires independent measurement.

## Capture board poses and calibrate

```sh
build/ball-vision record camera.yml board-poses 1.5
# Move the board between snapshots. Press Ctrl-C to finish.
python3 calibrate.py board-poses stereo.yml
```

Recording writes a pair of binary PGMs at the chosen interval in seconds, a CSV manifest containing both capture timestamps, and the two camera GUIDs. Choose a fresh output directory each time. Interval zero records every accepted pair that the disk can sustain.

board.json uses the nominal settings from the earlier stereo-calibration script: 12 squares across, 9 down, 100 mm square sides, 80 mm markers, DICT_6X6_250. Measure several squares on the physical print and divide by their count to verify scale. Edit the JSON or pass --board other.json to calibrate.py. Square length sets the scale of every reported position and velocity.

For the runtime acrylic model, remove the acrylic for calibration and face the printed side toward the cameras. Keep the camera mounts, lens focus and aperture fixed when reinstalling the plate. A calibration collected through the installed sheet can absorb its refraction and cause double correction. Aim for 20–40 distinct poses across the image and vary tilt about both axes and distance. With 100 mm squares, move the board farther away to get enough visible features. Each stereo pose needs six or more shared, noncollinear ChArUco corners. The calibrator uses all usable corners for each camera's intrinsics, then intersects corner IDs within each pair for the stereo fit. It reports monocular and stereo reprojection RMS in pixels and the recovered baseline in millimetres. Check baseline against the physical rig and use independent known ball positions across the intended working volume to assess accuracy. The camera model includes lens distortion. Accuracy through the acrylic at different platform tilts requires measurement.

## Runtime acrylic correction

The supplied tracker.yml enables a 6.35 mm plane-parallel sheet with refractive index 1.49, assuming air on both sides and both cameras viewing the ball through both faces. Set acrylic_thickness_mm to zero for bare-camera recordings. Older tracker settings without this key also use the bare-camera model.

The tracker undistorts the sampled silhouette rays, applies Snell’s law for the two parallel faces, and fits a sphere tangent to the resulting rays. The exiting direction equals the incoming direction, and the model computes the displacement of each ray. For a uniform parallel sheet, its distance along the ray cancels. The orientation, physical thickness and refractive index determine the line displacement. Rays that pass around the plate rim, surface warping and wedge-shaped sheets are outside this model.

For a stationary plate, set fixed_plate_normal in the **left camera optical frame**, pointing from the cameras toward the ball. The supplied [0, 0, 1] assumes the left optical axis is perpendicular to the plate. For a tilting plate, pass the robot’s orientation log as the final command argument:

```sh
build/ball-vision track camera.yml stereo.yml tracker.yml normals.txt > ball.csv
build/ball-vision replay ball-sequence stereo.yml tracker.yml normals.txt > replay.csv
```

Create normals.txt before launching. The robot can append one newline-terminated, space-separated row per pose as it runs. Save that same file alongside a recording for replay. Each row contains an increasing timestamp in microseconds and three components of the measured surface normal, for example:

```text
100000 0.0 0.0 1.0
105000 0.01 0.0 0.99995
```

Use the same clock domain as left_us. Align pose to exposure by measuring and setting capture_minus_exposure_us. The tracker interpolates between available nearby samples or holds the latest preceding sample for up to normal_max_age_us. Without a current sample it emits valid=0. A fixed normal is appropriate only while the plate holds that orientation. The robot or an external pose sensor must supply the changing normal.

## Measured plate pose from three moteus R4.11 controllers

plate_pose.py computes a plate normal from the three measured output positions in turns. It is independent of the balancing controller. Feed it the same position replies already collected by your control loop, or use its optional read-only CAN query mode for standalone vision commissioning.

**Geometry.** platform.example.json contains a reconstruction of the saved home assembly in spherical-parallel-mechanism.f3z, design v12 with ARM v10. Coordinates are millimetres in the Fusion assembly frame. Each leg specifies a point on the motor axis, its direction, a point on the parallel elbow axis, and the spherical joint center at home. These coordinates retain offsets along the revolute axes. Forward kinematics rotates the two links and solves three passive elbow angles subject to constant distances between all three spherical joints. It starts on the saved assembly branch and uses the previous solution for the next frame.

The CAD profile uses the saved occurrence transforms and the ARM subassembly geometry. Link1 has pivot axes at local x = 0 and 100 mm. The rod-end B-rep has its spherical center at local (0, 17, 0) mm. The assembled link lengths projected into the joint plane are 100 and 117 mm. The motor axes have vertical component 0.5 in magnitude, or 30° elevation. The inferred spherical-joint circle has a radius of approximately 201.7 mm. Check the approximately **349.3 mm** adjacent spherical-center spacing against the built mechanism, especially given the stated 330 mm disk. These are different physical dimensions. The archived camera components are Basler models.

The GitHub model uses a 93 mm driven crank, a 117 mm passive link and radial vertical leg planes. Its independently constructed attachment points also give triangle sides of 347.07, 347.07 and 345.07 mm at a 10° tilt toward +X, compared with 346.41 mm each at level. The code in this project enforces the rigid triangle measured in the Fusion assembly. [Reference inverse kinematics](https://github.com/DbugDiver/ball-balancing-robot/blob/7c24ffce7a4ded7bd3f4793e5c34792ae6d7f353/src/rob_kin.py)

Copy platform.example.json to platform.json and register the hardware before interpreting normals:

- Match moteus_ids to CAD legs in the listed order (+X, then counterclockwise as seen from above).
- Zero the encoders at the **resting pose** (sim/rest-pose.png): power off and let the plate settle with the elbows down, cranks on the base. Record the three output positions as level_output_turns. zero_crank_rad gives the crank angles of that pose measured from the CAD home, which the kinematics use; the example's -1.36 rad is where the CAD cranks reach the base. Check it by measuring the plate's top centre above the base's top face at rest: 238 mm at the CAD home, 217 at -0.54 rad (the -20° idle twist), 198 at -0.80, 180 at -1.00, 159 at -1.20, 147 at -1.30, 138 at -1.36. The CAD home is the plate's highest point and it can fall either way; the controller works on this (negative-twist) side, so let it settle this way every time. Set crank_turns_per_output_turn from the actual drive ratio and encoder sign relative to the listed right-hand axis. Defaults assume IDs 1–3, zero turns at rest, and +1 gain.
- Set left_from_base_rotation to the measured rotation mapping Fusion assembly vectors into the left Dragonfly optical frame. Identity is a placeholder. A single level pose establishes the vertical but leaves yaw ambiguous. Use at least two independent tilt directions to check yaw, sign and leg order. Fix the board parallel to the plate with the printed face visible and use its measured normal as a check at several held-out inclinations.

With an existing telemetry producer, use four whitespace-separated fields per line: host timestamp in microseconds, followed by the three measured positions in configured leg order. The pose log is compatible with live tracking and replay.

```sh
python3 plate_pose.py platform.json --positions joint-positions.txt > normals.txt
# Use --positions - to stream from another process.
```

For standalone PEAK PCIe acquisition on Linux, bring the SocketCAN FD interface up with arbitration/data bitrates matching the controllers. Install numpy, moteus and python-can in the interpreter that runs the reader. The code uses the moteus 1.0 Python transport API. Start the following in separate terminals, using a new normals file per capture:

```sh
python3 plate_pose.py platform.json --can can0 --hz 200 > normals.txt
build/ball-vision track camera.yml stereo.yml tracker.yml normals.txt > ball.csv
```

CAN mode sends one batch of three register queries per cycle. It timestamps positions at the host realtime midpoint of the batch, discards missing responses and drops batches spanning more than --max-span-us (default 4000). The three motors sample independently. Verify the FireWire timestamp clock and exposure offset on the target before using dynamic tilt compensation. A control loop that already owns CAN can supply its measured positions and timestamps through the stream interface.

Run the independent rigid-pose round-trip check with `python3 tests/test_plate_pose.py`.

## Track live or replay

```sh
build/ball-vision track camera.yml stereo.yml tracker.yml > ball.csv

# The commands without normals.txt assume a stationary, measured plate orientation.
# Record a short ball sequence for repeatable tuning, then replay offline
build/ball-vision record camera.yml ball-sequence 0
build/ball-vision replay ball-sequence stereo.yml tracker.yml > replay.csv
```

Tune threshold and bright_ball using the saved grayscale frames under both office lighting conditions. Set the pixel-radius limits using the farthest and nearest expected balls. Once the diameter is known, set ball_diameter_mm to reject stereo candidates inconsistent with it. The tracker searches around the previous ball observation and falls back to a full image search on loss. It filters outlines by size and circularity, undistorts sampled boundary rays, estimates the sphere-center direction, and triangulates compatible observations. It rejects clipped outlines, points behind a camera and pairs with excessive reprojection or diameter disagreement.

Each accepted stereo pair produces one CSV row. Position is in the **left camera optical frame**, in millimetres: +x right, +y down, +z forward from the lens. Velocity is a two-observation finite difference in mm/s using left capture times. It is blank for the first observation, after a missed detection, and across gaps longer than 50 ms. valid=0 means the pair produced no ball measurement. A missing stereo pair produces no row. Robot-side consumers should check both validity and age. Moving-platform coordinates require the platform pose at measurement time.

processing_us measures detection and reconstruction for that pair. It excludes acquisition, disk I/O, and output delivery. Use it alongside pair rate and drop counts when benchmarking on the target PC.

## Hardware-independent verification

```sh
python3 tests/test_pipeline.py build/ball-vision
```

The test renders stereo ChArUco poses with a known baseline and ball silhouettes with lens distortion. It exercises the calibration and replay commands, metric position and velocity, a missing observation, reacquisition, swapped camera IDs, tilted acrylic with non-parallel camera axes, and stale orientation data.

## Simulator with the Fusion geometry

`simulate.py view` opens the MuJoCo model of the platform. To show the real parts instead of the simplified shapes, export them from Fusion once:

1. Open the platform design at its home pose.
2. Utilities > Add-Ins > Scripts and Add-Ins, click + next to My Scripts, pick the folder `sim/fusion_export`, and run `fusion_export`.

The script writes `sim/cad/manifest.json` and one STL per visible body in millimetres in the Fusion root frame. Then:

```sh
python3 simulate.py cad      # assignment report and sim/cad-check.png
python3 simulate.py view     # uses sim/cad when it exists; --no-cad for the simplified shapes
```

Each body is attached to the world, a crank, a rod or the plate from the joints it touches in platform.example.json. `cad` lists every assignment, and the snapshot shows the home pose and a driven pose; a part that stays behind or flies off when the cranks move has the wrong link. Correct it with `sim/cad/assign.json`, for example `{"base:1/Body3": "drive2"}`, using keys from the report. Parts touching only the motor axis stay fixed. In the viewer the simplified physics shapes are in geom group 3 (key 3 toggles them). With the export, the ball rests on the top of the CAD disk instead of the provisional 18 mm offset.

Run `python3 tests/test_cad.py` to check assignment, mesh placement and motion with a synthetic export.

## Ball controller

control/ holds a C++17 controller for balancing and bouncing a ping-pong ball, with no dependencies beyond the moteus C++ headers for the hardware runtime. The same code runs in the simulator and on the robot.

The three cranks set the plate's roll, pitch and yaw. Turning them together twists the plate, and its height falls with the square of the twist, so the plate idles twisted (operation.idle_yaw_deg, 20°, about 21 mm below zero twist) and bounces by twisting further or back. The ball is kept near the plate centre while bouncing, where the twist barely moves the surface.

- Balancing: delay-compensated Kalman estimate of the ball and an LQR on tilt. The plate tilts about a point well away from its surface, so a fast tilt also slides the surface under the ball; the controller removes that term exactly. Camera noise is kept off the motors: the target moves along a jerk-limited path whose tilt goes straight to the plate, and only the LQR's correction passes a 60 ms filter. The correction fades in over 0.4 s after every start.
- Bouncing: each impact is planned from the predicted trajectory (gravity and air drag). The tilt and stroke speed at impact set the rebound direction and height. An outer loop corrects the apex from each observed flight, and the restitution is learned online. Launch from rest and catching back to balance are automatic.

```sh
cmake -S control -B control/build && cmake --build control/build
control/build/test-ballctl control.example.json
python3 simulate.py control --mode bounce --apex-mm 150          # headless report
mjpython simulate.py control --view --mode balance               # viewer (python3 on Linux); F6 hold, F7 balance, F8 bounce
python3 tests/test_control.py
```

On the robot (Linux, the build fetches the moteus headers):

```sh
build/ball-vision track camera.yml stereo.yml tracker.yml normals.txt |
  control/build/ball-control control.json --socketcan-iface can0 --normals normals.txt --max-torque 0.8 --log run.csv
```

It ramps to the idle posture over 3 s and then holds; press b to balance, j to bounce, w/a/s/d to move the target, +/- for the apex, q to stop. Every crank command passes a speed and acceleration cap (limits.crank_speed_max_rad_s, crank_accel_max_rad_s2; 10 rad/s and 800 rad/s², above the 6.8 rad/s and about 500 rad/s² bouncing uses) that approaches out-of-range requests without overshoot. It stops all motors on a controller fault, missing replies, or a crank far from its command, and each command carries a 50 ms moteus watchdog. It has not been run on the robot. The motors are EaglePower 8308s, direct drive. Balance first with a low --max-torque: holding the plate takes about 0.35 N m and balancing peaks near 0.8 N m, while launching and bouncing reach the 2.5 N m cap of the haptic demo (99% of the time below 1.4 N m; at 1.2 N m the simulated robot cannot launch). Before the first run, copy control.example.json and set:

- the crank limits (limits.crank_min_rad and crank_max_rad) from the real mechanism; the example allows -1.45 to 0.3 rad so the rest pose (-1.36) is inside, and balancing and bouncing use about -1.29 to 0.08 rad,
- left_origin_mm, the left camera's optical centre in the Fusion frame (the example matches the simulator's cameras),
- the rod ends' real swivel range. In the CAD, the bolt through each 59935K431 rod-end ball meets the housing at about 21.5° of twist; the -20° idle is just inside that, and bounce strokes twist ±11.5° further. The robot reaching its rest at 44° of twist suggests the real rod ends swivel further than the CAD models; if they bind, reduce the size of operation.idle_yaw_deg and limits.stroke_yaw_deg,
- the ball's restitution if measured, and the moteus position gains and torque limit on the controllers (below).

### Gravity calibration

The torque feedforward needs the weight of the plate, cranks and rods. Until calibrated it uses a CAD guess (plate.mass_kg and com_above_joints_mm). With the ball removed:

```sh
control/build/ball-control control.json --socketcan-iface can0 --max-torque 1.5 --gravity-cal gravity.json
```

The plate visits about 40 postures across twist and tilt (about 4 minutes). Each is approached twice along pure twist, once from each side, so every crank's friction cancels in the average, then held while the reported torques are averaged. A least-squares fit of 22 mass moments and motor offsets follows. The report gives the residual and the friction seen from the two approaches, and gravity.json holds the parameters and raw samples. Add `"gravity_file": "gravity.json"` to control.json to use it; the start-up ramp also fades this holding torque in. The individual parameters are not unique (some combinations give identical torques); the fitted torques are what matter. In simulation with an unmodelled 150 g payload 60 mm off centre, the fit predicts holding torque at unseen postures to 0.002 N m RMS, against 0.057 N m for the CAD guess.

Calibration is what allows softer servo gains, which pass less camera noise to the motors: without it the plate sags by torque/kp. The softest usable gain is then set by friction, which leaves a dead band of friction/kp; the calibration report measures it. In simulation, kp 10 N m/rad (63 N m/rev, as in the haptic demo) with kd 0.3 to 0.5 N m s/rad (2 to 3 N m s/rev) balances and bounces well; the demo's kd of 0.09 N m s/rev is too little damping to hold the plate. Try it with `python3 simulate.py control --calibrate --payload-g 150 --kp 10 --kd .3`.

The simulated plant parameters (servo gains, 8308 rotor inertia and friction, latencies, noise) are estimates in sim/control_sim.py. With them, balancing holds about 0.1 mm RMS with 0.02 N m of motor torque noise and recovers from 60 mm in about 1.1 s without overshoot, and bouncing holds a 100 mm apex to about ±2.5 mm with impacts within 3.5 mm of the target.
