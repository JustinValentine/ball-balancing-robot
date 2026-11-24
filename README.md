# Ball balancing robot

A three-motor plate that balances and bounces a ping-pong ball. Two Point Grey Dragonfly Express cameras track the ball in stereo at 200 Hz, three EaglePower 8308 motors on moteus r4 controllers drive the plate, and the controller runs in C++ on a Linux PC with a PEAK CAN card.

So far the controller has only been run in simulation.

## Layout

- `src/` stereo capture and ball tracking (C++17, libdc1394, OpenCV)
- `calibrate.py` stereo calibration from ChArUco board images
- `plate_pose.py` plate normal from the three motor positions
- `control/` balancing and bouncing controller, gravity calibration and the moteus runtime
- `sim/` MuJoCo model of the robot built from the Fusion assembly
- `simulate.py` viewer, CAD check, controller runs and tracker benchmarks
- `tests/`

## Vision

```sh
sudo apt install build-essential cmake pkg-config libdc1394-dev libopencv-dev python3-opencv
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
build/ball-vision list                # camera GUIDs, modes and trigger support
build/ball-vision probe camera.yml
```

Each camera needs its own S800 bus. Put the GUIDs from `list` into camera.yml in left/right order. The Nucleo drives the shared trigger, and with it on `probe` should show a steady 200 pairs/s.

To calibrate, take the acrylic off and face the printed side of the board toward the cameras:

```sh
build/ball-vision record camera.yml board-poses 1.5    # move the board between shots, Ctrl-C to stop
python3 calibrate.py board-poses stereo.yml
```

board.json is the 12 × 9 ChArUco board with 100 mm squares. I aim for 20 to 40 poses with the board tilted both ways and at different distances.

The tracker corrects for refraction through the 6.35 mm acrylic using the plate normal, which the controller writes to normals.txt while it runs:

```sh
build/ball-vision track camera.yml stereo.yml tracker.yml normals.txt > ball.csv
build/ball-vision record camera.yml ball-sequence 0
build/ball-vision replay ball-sequence stereo.yml tracker.yml normals.txt > replay.csv
```

Each stereo pair gives one CSV row with the ball position in mm in the left camera frame (+x right, +y down, +z forward). `python3 tests/test_pipeline.py build/ball-vision` checks the whole pipeline on rendered images.

Still to measure on the rig: the timestamp offset between the cameras (right_offset_us), the delay from exposure to timestamp, and the printed square size.

## Geometry and zeroing

platform.example.json has the joint geometry from the Fusion assembly (spherical-parallel-mechanism v12, ARM v10) in mm. `python3 sim/check_fusion.py spherical-parallel-mechanism.f3z` checks it against the archive.

The cranks set the plate's roll, pitch and yaw. Turning all three together twists the plate, and it drops as it twists, so the controller works at -20° of twist and bounces the ball by twisting further or back.

I zero the encoders at rest (sim/rest-pose.png). With the motors off the plate falls with the elbows down until the cranks sit on the base. Record the three positions as level_output_turns. zero_crank_rad is the crank angle at rest measured from the CAD pose, which is -1.36 rad in the CAD. To check it, measure the plate's top centre above the base's top face:

| Cranks (rad) | 0 (CAD) | -0.54 (idle) | -0.80 | -1.00 | -1.20 | -1.30 | -1.36 |
|---|---|---|---|---|---|---|---|
| Height (mm) | 238 | 217 | 198 | 180 | 159 | 147 | 138 |

From the CAD pose the plate can fall to either side, so let it settle on the elbows-down side every time. Also set moteus_ids (+X leg first, then counterclockwise seen from above), crank_turns_per_output_turn (drive ratio and sign) and left_from_base_rotation.

## Simulator

```sh
pip install mujoco numpy pillow
python3 simulate.py cad        # checks the CAD export and writes sim/cad-check.png
mjpython simulate.py view      # python3 on Linux
```

sim/cad holds the Fusion bodies, exported with sim/fusion_export (run it from Scripts and Add-Ins in Fusion with the design open). Each body goes on the base, a crank, a rod or the plate depending on which joints it touches. If a part lands on the wrong link, sim/cad/assign.json overrides it.

## Controller

```sh
cmake -S control -B control/build && cmake --build control/build
python3 simulate.py control --mode bounce --apex-mm 150
mjpython simulate.py control --view      # F6 hold, F7 balance, F8 bounce
python3 tests/test_control.py
```

Balancing uses a Kalman filter on the ball that makes up for the camera delay, and an LQR on the plate tilt. The plate tilts about a point well below its surface, so the controller also corrects for the surface sliding under the ball. For bouncing, each impact is planned from the predicted flight, and the tilt and stroke speed at impact set where the ball goes and how high. The apex is corrected after every bounce and the ball's restitution is learned as it runs.

In simulation it holds the ball to about 0.1 mm, recovers from 60 mm off centre in about 1 s, and bounces to a 100 mm apex within about 2.5 mm. The motor, friction and camera numbers in the simulator (sim/control_sim.py) are my estimates.

On the robot:

```sh
build/ball-vision track camera.yml stereo.yml tracker.yml normals.txt |
  control/build/ball-control control.json --socketcan-iface can0 --normals normals.txt --max-torque 0.8
```

It ramps from rest to the idle pose over 3 s and holds there. Keys are b to balance, j to bounce, wasd to move the target, +/- for the apex and q to quit. The motors stop on a moteus fault, missing replies or a large tracking error, every command has a 50 ms watchdog, and crank speed and acceleration are capped at 10 rad/s and 800 rad/s².

Before the first run, copy control.example.json and set the crank limits, left_origin_mm (the left camera position in the Fusion frame) and the moteus gains. In simulation kp 10 N m/rad (63 N m/rev) with kd 0.3 to 0.5 N m s/rad worked well. In the CAD the rod ends reach their swivel limit near -21.5° of twist, so check by hand that they don't bind between about -8° and -32°.

### Gravity calibration

With the ball off the plate:

```sh
control/build/ball-control control.json --socketcan-iface can0 --max-torque 1.5 --gravity-cal gravity.json
```

The plate moves through about 40 postures in 4 minutes and approaches each one from both sides so friction cancels. The weight of the plate, cranks and rods is then fitted to the measured torques. Add `"gravity_file": "gravity.json"` to control.json to use it. The report also gives the friction, which limits how soft the motor gains can be.
