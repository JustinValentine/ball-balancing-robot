"""Closed-loop MuJoCo simulation of the C++ ball controller.

The plant is the platform model from mechanism.py with a ping-pong ball (hollow shell, air drag,
bouncy contact), torque-limited servos that behave like moteus position mode, and a 200 Hz stereo
measurement with latency, noise and dropouts. The controller is control/build/libballctl, called
through its C interface at 1 kHz, the same code the hardware runtime uses.
"""
import ctypes
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"sim"))
from mechanism import make_model

PHASES = ("level", "balancing", "settling", "launching", "bouncing", "catching", "calibrating")


class Output(ctypes.Structure):
    _fields_ = [("q", ctypes.c_double*3), ("qd", ctypes.c_double*3), ("tau", ctypes.c_double*3),
                ("ok", ctypes.c_int), ("phase", ctypes.c_int), ("ball_valid", ctypes.c_int),
                ("contact", ctypes.c_int), ("ball", ctypes.c_double*3), ("ball_v", ctypes.c_double*3),
                ("yaw", ctypes.c_double), ("nx", ctypes.c_double), ("ny", ctypes.c_double),
                ("next_impact", ctypes.c_double), ("restitution", ctypes.c_double), ("apex", ctypes.c_double),
                ("plate_normal", ctypes.c_double*3), ("limited", ctypes.c_int),
                ("note", ctypes.c_int), ("note_index", ctypes.c_int), ("note_time", ctypes.c_double),
                ("note_beat", ctypes.c_double)]


def library():
    for name in ("libballctl.dylib", "libballctl.so"):
        path = ROOT/"control"/"build"/name
        if path.exists():
            lib = ctypes.CDLL(str(path))
            break
    else:
        raise SystemExit("Build the controller first: cmake -S control -B control/build && cmake --build control/build")
    d, p = ctypes.c_double, ctypes.POINTER(ctypes.c_double)
    lib.bc_create.restype = ctypes.c_void_p
    lib.bc_create.argtypes = [ctypes.c_char_p, d, ctypes.c_char_p, ctypes.c_int]
    lib.bc_destroy.argtypes = [ctypes.c_void_p]
    lib.bc_set_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.bc_set_target.argtypes = [ctypes.c_void_p, d, d]
    lib.bc_set_apex.argtypes = [ctypes.c_void_p, d]
    lib.bc_set_path.argtypes = [ctypes.c_void_p, p, p, p]
    lib.bc_measurement.argtypes = [ctypes.c_void_p, d, p]
    lib.bc_step.argtypes = [ctypes.c_void_p, d, p, p, ctypes.POINTER(Output)]
    lib.bc_info.argtypes = [ctypes.c_void_p, p, p]
    lib.bc_inverse.argtypes = [ctypes.c_void_p, d, d, d, p]
    lib.bc_inverse.restype = ctypes.c_int
    i = ctypes.POINTER(ctypes.c_int)
    lib.bc_calibration_start.argtypes = [ctypes.c_void_p, d, p]
    lib.bc_calibration_step.argtypes = [ctypes.c_void_p, d, p, p, p, ctypes.POINTER(Output)]
    lib.bc_calibration_step.restype = ctypes.c_int
    lib.bc_calibration_progress.argtypes = [ctypes.c_void_p, i, i]
    lib.bc_calibration_finish.argtypes = [ctypes.c_void_p, p, ctypes.c_char_p, ctypes.c_int]
    lib.bc_calibration_finish.restype = ctypes.c_int
    lib.bc_gravity_get.argtypes = [ctypes.c_void_p, p]
    lib.bc_gravity_set.argtypes = [ctypes.c_void_p, p]
    lib.bc_gravity_torque.argtypes = [ctypes.c_void_p, d, d, d, p]
    lib.bc_gravity_torque.restype = ctypes.c_int
    lib.bc_holding_torque.argtypes = [ctypes.c_void_p, p, p]
    lib.bc_holding_torque.restype = ctypes.c_int
    lib.bc_set_circle.argtypes = [ctypes.c_void_p, d, d]
    lib.bc_set_song.argtypes = [ctypes.c_void_p, ctypes.c_int, d]
    lib.bc_song_count.restype = ctypes.c_int
    lib.bc_song_name.argtypes = [ctypes.c_int]
    lib.bc_song_name.restype = ctypes.c_char_p
    lib.bc_song_notes.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), p, ctypes.c_int]
    lib.bc_song_notes.restype = ctypes.c_int
    lib.bc_song_beat.argtypes = [ctypes.c_int]
    lib.bc_song_beat.restype = d
    return lib


def song_names(lib=None):
    lib = lib or library()
    return [lib.bc_song_name(i).decode() for i in range(lib.bc_song_count())]


@dataclass
class Plant:
    """Physical parameters of the simulated robot. Defaults are estimates, not measurements.

    The motors are EaglePower 8308s on moteus r4 controllers, direct drive. The haptic demo caps
    them at 2.5 N m (peak roughly 5 N m); the rotor inertia is a guess for a motor this size."""
    ball_radius: float = .020          # ITTF ball, 40 mm
    ball_mass: float = .0027
    restitution: float = .88           # Normal coefficient the contact is tuned to give.
    friction: float = .35
    kp: float = 25.                    # Servo stiffness, N m/rad at the crank.
    kd: float = .45                    # Servo damping, N m s/rad.
    torque_limit: float = 2.5          # N m, as the haptic demo caps the 8308.
    rotor_inertia: float = 3e-4        # kg m^2 at the crank, estimate.
    crank_friction: float = .02        # Friction at each crank, N m (MuJoCo's soft model; it creeps).
    torque_hysteresis: float = .03     # Stick-slip seen in reported holding torque: the reading is off
                                       # by this much against the last direction of motion, N m.
    torque_noise: float = .005         # Noise on the torque the controllers report, N m.
    payload: tuple = (0., (0., 0., 0.))  # Unmodelled mass on the plate: kg and offset from the
                                         # joint centroid in the plate frame (m).
    command_delay: float = .001        # CAN round trip before a command takes effect.
    vision_period: float = .005
    vision_latency: float = .006       # Exposure to delivery at the controller.
    timestamp_offset: float = .003     # Exposure to the tracker's capture timestamp.
    noise_xy: float = .0003
    noise_depth: float = .0008
    dropout: float = .005
    crank_offset: tuple = (0., 0., 0.)  # Encoder zero error: true angle minus what the controller sees.
    torque_scale: float = 1.            # Torque delivered per torque the moteus reports (torque constant error).
    camera_rotation: tuple = (0., 0., 0.)  # Stereo extrinsic error: small rotation (rad) and offset (m) of the
    camera_offset: tuple = (0., 0., 0.)    # real left camera from where the controller's config puts it.
    noise_growth: bool = False          # Scale noise with distance: lateral with range, depth with range squared,
                                        # as for stereo; the base values then apply at the plate centre.
    outliers: float = 0.                # Fraction of measurements that are wrong by outlier_size (m).
    outlier_size: float = .01
    latency_jitter: float = 0.          # Extra delivery delay per measurement, uniform 0..this (s).
    geometry_error: float = 0.          # Build error: each motor, elbow and ball-joint point of the real
                                        # robot is off from the config by up to this much per axis (m).
    seed: int = 1


def build_xml(cfg, plant, cad=None, surface_offset=.02855):
    xml = make_model(cfg, 420, 170, 80, 55, cad, surface_offset)
    root = ET.fromstring(xml)
    option = root.find("option")
    option.set("density", "1.204")
    option.set("viscosity", "1.8e-5")
    ball = root.find(".//body[@name='ball']")
    geom = ball.find("geom")
    geom.set("size", str(plant.ball_radius))
    geom.attrib.pop("mass")
    geom.set("fluidshape", "ellipsoid")
    geom.set("friction", f"{plant.friction} .001 .0001")
    geom.set("condim", "4")
    r, m = plant.ball_radius, plant.ball_mass
    inertia = 2/3*m*r*r
    ET.SubElement(ball, "inertial", pos="0 0 0", mass=str(m), diaginertia=f"{inertia} {inertia} {inertia}")
    # A spring-damper contact (direct stiffness and damping per unit mass) with the damping ratio
    # for the restitution, e = exp(-pi z / sqrt(1 - z^2)). MuJoCo's time-constant form gains energy
    # at damping ratios this low. Contact lasts about 3 ms.
    zeta = -np.log(plant.restitution)/np.sqrt(np.pi**2+np.log(plant.restitution)**2)
    stiffness = 1e6
    pair = root.find(".//contact/pair")
    pair.set("solref", f"{-stiffness:.6g} {-2*zeta*np.sqrt(stiffness):.6g}")
    pair.set("solimp", ".97 .99 .0001")
    pair.set("condim", "4")
    pair.set("friction", f"{plant.friction} {plant.friction} .001 .0001 .0001")
    actuators = root.find("actuator")
    for act in list(actuators):
        actuators.remove(act)
    for i in (1, 2, 3):
        ET.SubElement(actuators, "motor", name=f"crank{i}", joint=f"q{i}", ctrllimited="true",
                      ctrlrange=f"{-plant.torque_limit} {plant.torque_limit}")
        joint = root.find(f".//joint[@name='q{i}']")
        joint.set("armature", str(plant.rotor_inertia))
        joint.set("frictionloss", str(plant.crank_friction))
    mass, offset = plant.payload
    if mass > 0:
        plate = root.find(".//body[@name='plate']")
        ET.SubElement(plate, "geom", type="sphere", size=".012", pos=" ".join(map(str, offset)), mass=str(mass),
                      contype="0", conaffinity="0", rgba=".9 .3 .2 1")
    return ET.tostring(root, encoding="unicode")


@dataclass
class Log:
    t: list = field(default_factory=list)
    ball: list = field(default_factory=list)       # True ball centre, base frame.
    estimate: list = field(default_factory=list)
    surface: list = field(default_factory=list)    # Plate surface centre.
    normal: list = field(default_factory=list)
    phase: list = field(default_factory=list)
    torque: list = field(default_factory=list)
    crank: list = field(default_factory=list)
    gap: list = field(default_factory=list)        # Ball surface above plate surface (m).
    contact: list = field(default_factory=list)    # True plate-ball contact.
    notes: list = field(default_factory=list)      # (planned impact, MIDI note, index, scheduled beat) per song impact.

    def arrays(self):
        return {k: np.array(v) for k, v in self.__dict__.items() if k != "notes"}


class ControlSim:
    def __init__(self, config=ROOT/"control.example.json", plant=None, cad=None, surface_offset=.02855):
        self.plant = plant or Plant()
        self.config_path = Path(config)
        control = json.loads(self.config_path.read_text())
        platform = json.loads((self.config_path.parent/control["platform"]).read_text())
        self.left_R = np.array(platform["left_from_base_rotation"], float)
        self.left_o = np.array(control["left_origin_mm"], float)/1000
        self.surface_offset = control["surface_offset_mm"]/1000
        built = json.loads(json.dumps(platform))
        if self.plant.geometry_error > 0:
            shift = np.random.default_rng(self.plant.seed+7919)
            for leg in built["legs"]:
                for key in ("motor_mm", "elbow_mm", "ball_joint_mm"):
                    leg[key] = list(np.array(leg[key])+shift.uniform(-1, 1, 3)*self.plant.geometry_error*1000)
        self.model = mujoco.MjModel.from_xml_string(build_xml(built, self.plant, cad, self.surface_offset))
        self.data = mujoco.MjData(self.model)
        self.lib = library()
        err = ctypes.create_string_buffer(512)
        self.dt = 1e-3
        self.handle = self.lib.bc_create(str(self.config_path).encode(), self.dt, err, 512)
        if not self.handle:
            raise RuntimeError(err.value.decode())
        centre, gains = (ctypes.c_double*3)(), (ctypes.c_double*5)()
        self.lib.bc_info(self.handle, centre, gains)
        self.centre = np.array(centre)
        self.gains = np.array(gains)
        self.rng = np.random.default_rng(self.plant.seed)
        rx, ry, rz = self.plant.camera_rotation
        skew = np.array([[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]])
        U, _, Vt = np.linalg.svd(np.eye(3)+skew)
        self.true_R = U@Vt@self.left_R
        self.true_o = self.left_o+np.array(self.plant.camera_offset)
        self.range0 = np.linalg.norm(self.left_R@(self.centre-self.left_o))
        self.out = Output()
        self.cmd = np.zeros(3)
        self.cmd_v = np.zeros(3)
        self.cmd_tau = np.zeros(3)
        self.queue = []          # Commands waiting out the transport delay.
        self.vision = []         # Measurements waiting out the latency.
        self.next_exposure = 0.
        self.next_control = 0.
        self.log = Log()
        self.steps_per_control = round(self.dt/self.model.opt.timestep)
        self.cranks = [self.model.joint(f"q{i}").qposadr[0] for i in (1, 2, 3)]
        self.crank_v = [self.model.joint(f"q{i}").dofadr[0] for i in (1, 2, 3)]
        self.ball_id = self.model.body("ball").id
        self.acrylic = self.model.geom("acrylic").id
        self.ball_geom = self.model.geom("ball_surface").id

    def close(self):
        if self.handle:
            self.lib.bc_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()

    # Commands --------------------------------------------------------------------------------
    def mode(self, name):
        self.lib.bc_set_mode(self.handle, {"hold": 0, "balance": 1, "bounce": 2}[name])

    def target(self, x, y):
        self.lib.bc_set_target(self.handle, x, y)

    def calibrate(self, report=True):
        """Run the gravity calibration on the simulated robot (no ball) and install the fit."""
        d, p = self.data, self.plant
        dbl3 = ctypes.c_double*3
        ball = self.model.joint(self.model.body("ball").jntadr[0])
        floor = min(self.model.geom_pos[g][2] for g in range(self.model.ngeom)
                    if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE)
        d.qpos[ball.qposadr[0]:ball.qposadr[0]+3] = [.8, .8, floor+p.ball_radius+.001]  # Off the robot.
        d.qpos[ball.qposadr[0]+3:ball.qposadr[0]+7] = [1, 0, 0, 0]
        d.qvel[ball.dofadr[0]:ball.dofadr[0]+6] = 0
        self.lib.bc_calibration_start(self.handle, d.time, dbl3(*d.qpos[self.cranks]))
        applied = np.zeros(3)
        moving = np.zeros(3)
        running = True
        while running:
            if d.time >= self.next_control-1e-9:
                q, qd = d.qpos[self.cranks]-np.array(p.crank_offset), d.qvel[self.crank_v]
                tau = applied+self.rng.normal(0, p.torque_noise, 3)-p.torque_hysteresis*moving
                running = bool(self.lib.bc_calibration_step(self.handle, d.time, dbl3(*q), dbl3(*qd), dbl3(*tau),
                                                            ctypes.byref(self.out)))
                self.queue.append((d.time+p.command_delay, np.array(self.out.q), np.array(self.out.qd),
                                   np.array(self.out.tau)))
                self.next_control += self.dt
            while self.queue and self.queue[0][0] <= d.time+1e-9:
                _, self.cmd, self.cmd_v, self.cmd_tau = self.queue.pop(0)
                self.cmd = self.cmd+np.array(p.crank_offset)
            q, qd = d.qpos[self.cranks], d.qvel[self.crank_v]
            d.ctrl[:] = p.torque_scale*np.clip(p.kp*(self.cmd-q)+p.kd*(self.cmd_v-qd)+self.cmd_tau,
                                               -p.torque_limit, p.torque_limit)
            applied = d.ctrl/p.torque_scale   # What the moteus reports.
            fast = np.abs(qd) > .01
            moving[fast] = np.sign(qd[fast])
            mujoco.mj_step(self.model, d)
        params = (ctypes.c_double*22)()
        text = ctypes.create_string_buffer(4096)
        ok = self.lib.bc_calibration_finish(self.handle, params, text, 4096)
        if report:
            print(text.value.decode())
        if not ok:
            raise RuntimeError(text.value.decode())
        return np.array(params)

    def gravity(self, params=None):
        """Read or set the controller's 22 gravity parameters."""
        buf = (ctypes.c_double*22)()
        if params is not None:
            self.lib.bc_gravity_set(self.handle, (ctypes.c_double*22)(*params))
        self.lib.bc_gravity_get(self.handle, buf)
        return np.array(buf)

    def path(self, x, v, a):
        """Balance along a path: position, velocity, acceleration relative to the plate centre."""
        d2 = ctypes.c_double*2
        self.lib.bc_set_path(self.handle, d2(*x), d2(*v), d2(*a))

    def circle(self, radius, period=4.):
        self.lib.bc_set_circle(self.handle, radius, period)

    def song(self, name, beat=0.):
        """Bounce to a song by name; None stops."""
        names = song_names(self.lib)
        self.lib.bc_set_song(self.handle, names.index(name) if name else -1, beat)

    def apex(self, h):
        self.lib.bc_set_apex(self.handle, h)

    # Setup -----------------------------------------------------------------------------------
    def start(self, ball_offset=(0., 0.), ball_height=None, settle=1.5):
        """Drive the plate to the idle posture with the ball held, then release the ball."""
        q = (ctypes.c_double*3)()
        if not self.lib.bc_inverse(self.handle, np.radians(json.loads(self.config_path.read_text())
                                   .get("operation", {}).get("idle_yaw_deg", -20)), 0, 0, q):
            raise RuntimeError("No idle posture")
        goal = np.array(q)
        d = self.data
        ball_adr = self.model.joint(self.model.body("ball").jntadr[0]).qposadr[0]
        hold = None
        t_end = d.time+settle
        while d.time < t_end:
            s = min(1., d.time/(.8*settle))
            s = s*s*(3-2*s)
            target = goal*s
            q_now, v_now = d.qpos[self.cranks], d.qvel[self.crank_v]
            # As the hardware runtime does: fade the gravity holding torque in during the ramp.
            hold = (ctypes.c_double*3)()
            self.lib.bc_holding_torque(self.handle, (ctypes.c_double*3)(*target), hold)
            ff = np.array(hold)*min(1., 3*d.time/settle)
            d.ctrl[:] = np.clip(self.plant.kp*(target-q_now)-self.plant.kd*v_now+ff, -self.plant.torque_limit,
                                self.plant.torque_limit)
            mujoco.mj_step(self.model, d)
            surface = d.site("surface").xpos.copy()
            n = d.body("plate").xmat.reshape(3, 3)[:, 2]
            hold = surface+n*(self.plant.ball_radius+(ball_height or 0))
            hold[:2] += ball_offset
            d.qpos[ball_adr:ball_adr+3] = hold+[0, 0, .0005 if ball_height is None else 0]
            d.qpos[ball_adr+3:ball_adr+7] = [1, 0, 0, 0]
            d.qvel[self.model.joint(self.model.body("ball").jntadr[0]).dofadr[0]:][:6] = 0
        self.cmd[:] = d.qpos[self.cranks]
        self.next_control = d.time
        self.next_exposure = d.time
        mujoco.mj_forward(self.model, d)

    # Simulation ------------------------------------------------------------------------------
    def run(self, seconds, callback=None):
        d, p = self.data, self.plant
        t_end = d.time+seconds
        R, o = self.left_R, self.left_o
        dbl3 = ctypes.c_double*3
        while d.time < t_end-1e-9:
            if d.time >= self.next_exposure-1e-9:
                if self.rng.random() >= p.dropout:
                    ball = d.xpos[self.ball_id]
                    left = self.true_R@(ball-self.true_o)
                    k = np.linalg.norm(left)/self.range0 if p.noise_growth else 1
                    left = left+self.rng.normal(0, [p.noise_xy*k, p.noise_xy*k, p.noise_depth*k*k])
                    if self.rng.random() < p.outliers:
                        u = self.rng.normal(size=3)
                        left = left+p.outlier_size*u/np.linalg.norm(u)
                    late = self.rng.uniform(0, p.latency_jitter) if p.latency_jitter > 0 else 0
                    self.vision.append((d.time+p.vision_latency+late, d.time+p.timestamp_offset, left))
                    self.vision.sort(key=lambda m: m[0])
                self.next_exposure += p.vision_period
            if d.time >= self.next_control-1e-9:
                while self.vision and self.vision[0][0] <= d.time+1e-9:
                    _, stamp, left = self.vision.pop(0)
                    self.lib.bc_measurement(self.handle, stamp, dbl3(*left))
                q, qd = d.qpos[self.cranks]-np.array(p.crank_offset), d.qvel[self.crank_v]
                self.lib.bc_step(self.handle, d.time, dbl3(*q), dbl3(*qd), ctypes.byref(self.out))
                self.queue.append((d.time+p.command_delay, np.array(self.out.q), np.array(self.out.qd),
                                   np.array(self.out.tau)))
                self.record()
                if callback:
                    callback(self)
                self.next_control += self.dt
            while self.queue and self.queue[0][0] <= d.time+1e-9:
                _, self.cmd, self.cmd_v, self.cmd_tau = self.queue.pop(0)
                self.cmd = self.cmd+np.array(p.crank_offset)
            q, qd = d.qpos[self.cranks], d.qvel[self.crank_v]
            d.ctrl[:] = p.torque_scale*np.clip(p.kp*(self.cmd-q)+p.kd*(self.cmd_v-qd)+self.cmd_tau,
                                               -p.torque_limit, p.torque_limit)
            mujoco.mj_step(self.model, d)

    def record(self):
        d, log = self.data, self.log
        surface = d.site("surface").xpos.copy()
        n = d.body("plate").xmat.reshape(3, 3)[:, 2].copy()
        ball = d.xpos[self.ball_id].copy()
        log.t.append(d.time)
        log.ball.append(ball)
        log.estimate.append(np.array(self.out.ball))
        log.surface.append(surface)
        log.normal.append(n)
        log.phase.append(self.out.phase)
        log.torque.append(d.ctrl.copy())
        log.crank.append(d.qpos[self.cranks].copy())
        log.gap.append(n@(ball-surface)-self.plant.ball_radius)
        log.contact.append(any(set(c.geom) == {self.acrylic, self.ball_geom}
                               for c in d.contact[:d.ncon]))
        if self.out.note:
            log.notes.append((self.out.note_time, self.out.note, self.out.note_index, self.out.note_beat))


def apexes(log, start=0.):
    """Ball apex heights above the plate surface between impacts, from the true trajectory."""
    t, gap = log["t"], log["gap"]
    keep = t >= start
    t, gap = t[keep], gap[keep]
    peaks = [(t[i], gap[i]) for i in range(1, len(gap)-1) if gap[i] >= gap[i-1] and gap[i] > gap[i+1] and gap[i] > .01]
    return np.array(peaks).reshape(-1, 2)


def impacts(log, start=0.):
    """Times and positions of true contact onsets."""
    t, contact, ball = log["t"], log["contact"].astype(bool), log["ball"]
    on = np.flatnonzero(contact[1:] & ~contact[:-1])+1
    on = on[t[on] >= start]
    return t[on], ball[on]
