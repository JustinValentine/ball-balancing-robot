"""Assemble the MuJoCo mechanism from the Fusion home coordinates in millimetres."""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

THICKNESS = .00635
BALL_RADIUS = .015
STEP = .0005
# Provisional distance from the rod-end plane to the top playing surface.
SURFACE_OFFSET = .018


def vec(values):
    return " ".join(f"{v:.12g}" for v in values)


def make_model(cfg, disk_mm, distance_mm, baseline_mm, drop_mm, cad=None, surface_offset=SURFACE_OFFSET, floor=None):
    """Return MJCF text. cad is a list of cad.Part; their meshes are then the visible
    geometry and the simplified bodies move to geom group 3 (toggle with key 3)."""
    xyz = lambda name: np.array([leg[name] for leg in cfg["legs"]])*0.001
    motors, elbows, tips = [xyz(key) for key in ("motor_mm", "elbow_mm", "ball_joint_mm")]
    axes = np.array([leg["axis"] for leg in cfg["legs"]])
    center = tips.mean(axis=0)
    surface = center + [0, 0, surface_offset]
    # Simplified shapes keep their physics. With CAD meshes shown, hide them in group 3.
    hide = {} if cad is None else {"group": "3"}
    model = ET.Element("mujoco", model="Fusion ball platform")
    add = ET.SubElement
    add(model, "compiler", angle="radian")
    add(model, "option", timestep=str(STEP), integrator="implicitfast", iterations="80",
        tolerance="1e-10", gravity="0 0 -9.81")
    add(model, "statistic", center=vec(center-[0,0,.07]), extent=".52")
    visual = add(model, "visual")
    add(visual, "global", azimuth="115", elevation="-20")
    add(visual,"headlight",ambient=".55 .55 .55",diffuse=".65 .65 .65")
    add(visual,"rgba",haze=".20 .24 .28 1")
    asset = add(model,"asset")
    add(asset,"texture",type="skybox",builtin="gradient",rgb1=".32 .37 .41",rgb2=".11 .14 .19",
        width="128",height="768")
    default = add(model, "default")
    add(default, "joint", damping=".01", armature=".00005")
    add(default, "geom", contype="0", conaffinity="0", rgba=".18 .24 .29 1")
    world = add(model, "worldbody")
    add(world, "light", pos="0 -.5 1", dir="0 .4 -1")
    add(world, "geom", type="plane", pos="0 0 -.11" if floor is None else vec([0,0,floor]), size="1 1 .01", rgba=".14 .16 .18 1",
        contype="1", conaffinity="1")
    # A narrow display ring connects the three fixed mounts. Its shape is illustrative.
    angles = np.linspace(0,2*np.pi,49)
    ring_r = np.linalg.norm(motors[:,:2]-center[:2],axis=1).mean()
    ring = np.column_stack((center[0]+ring_r*np.cos(angles),center[1]+ring_r*np.sin(angles),
                            np.full_like(angles,-.006)))
    for a,b in zip(ring[:-1],ring[1:]):
        add(world,"geom",type="capsule",fromto=vec([*a,*b]),size=".009",rgba=".55 .39 .23 1",**hide)
    plate = add(world, "body", name="plate", pos=vec(center))
    add(plate, "freejoint")
    # Cylinder determines contact and inertia. The joint brackets are visual.
    add(plate, "geom", name="acrylic", type="cylinder", size=vec([disk_mm/2000,THICKNESS/2]),
        pos=vec([0,0,surface_offset-THICKNESS/2]), density="1180", rgba=".45 .83 .9 .35",
        contype="1", conaffinity="1", friction=".55 .002 .0001", **hide)
    add(plate, "site", name="surface", pos=vec([0,0,surface_offset]), size=".001", rgba="0 0 0 0")
    ball = add(world, "body", name="ball", pos=vec(surface+[.008, .006, BALL_RADIUS+drop_mm/1000]))
    add(ball, "freejoint")
    add(ball, "geom", name="ball_surface", type="sphere", size=str(BALL_RADIUS), mass=".012",
        contype="1", conaffinity="1", rgba=".97 .94 .88 1")
    contacts = add(model,"contact")
    add(contacts,"pair",geom1="acrylic",geom2="ball_surface",solref=".003 .22",solimp=".95 .99 .0001")
    constraints, actuators = add(model, "equality"), add(model, "actuator")
    links = {"world": (world, np.zeros(3)), "plate": (plate, center)}
    for i, (base, elbow, tip, axis) in enumerate(zip(motors, elbows, tips, axes), 1):
        axis = axis / np.linalg.norm(axis)
        add(world, "geom", type="cylinder", fromto=vec([*(base-.033*axis), *base]), size=".034", **hide)
        foot = np.array([base[0],base[1],-.006])
        add(world,"geom",type="capsule",fromto=vec([*foot,*(base-.02*axis)]),size=".008",**hide)
        crank = add(world, "body", name=f"drive{i}", pos=vec(base))
        add(crank, "joint", name=f"q{i}", type="hinge", axis=vec(axis))
        add(crank, "geom", type="capsule", fromto=vec([0,0,0,*(elbow-base)]), size=".011", mass=".12", **hide)
        rod = add(crank, "body", name=f"rod{i}", pos=vec(elbow-base))
        add(rod, "joint", type="hinge", axis=vec(axis))
        add(rod, "geom", type="capsule", fromto=vec([0,0,0,*(tip-elbow)]), size=".008", mass=".06",
            rgba=".65 .69 .72 1", **hide)
        links[f"drive{i}"], links[f"rod{i}"] = (crank, base), (rod, elbow)
        add(rod, "site", name=f"tip{i}", pos=vec(tip-elbow), size=".005")
        add(plate, "site", name=f"socket{i}", pos=vec(tip-center), size=".004")
        # Draw a bracket from each CAD socket to the chosen disk perimeter.
        bracket_top = (tip-center).copy()
        bracket_top[:2] *= min(1., (disk_mm/2000-.008)/np.linalg.norm(bracket_top[:2]))
        bracket_top[2] = surface_offset-THICKNESS
        add(plate,"geom",type="capsule",fromto=vec([*(tip-center),*bracket_top]),size=".005",mass=".01",**hide)
        add(constraints, "connect", site1=f"tip{i}", site2=f"socket{i}", solref=".0012 1",
            solimp=".99 .999 .0001")
        add(actuators,"position", name=f"crank{i}",joint=f"q{i}",kp="45",kv=".55",
            ctrlrange="-.35 .35", forcerange="-4 4")
    # Synthetic cameras share the left optical frame orientation in the config.
    rotation = np.array(cfg["left_from_base_rotation"])
    for name, side in (("left",-1),("right",1)):
        origin = surface + rotation.T @ np.array([side*baseline_mm/2000, 0, -distance_mm/1000])
        # MuJoCo looks along -Z with +Y up. Optical coordinates look along +Z.
        add(world, "camera", name=name, pos=vec(origin), xyaxes=vec([*rotation[0], *-rotation[1]]),
            fovy="57.7")
        add(world, "geom", type="box", pos=vec(origin), size=".012 .01 .006", rgba=".16 .35 .4 1", **hide)
    for k, part in enumerate(cad or []):
        # STL vertices are millimetres in the Fusion frame. Every link frame keeps the
        # world orientation at home, so an offset places the mesh; MuJoCo keeps the
        # vertices where they are when it recentres the mesh.
        parent, origin = links[part.body]
        add(asset, "mesh", name=f"cad{k}", file=str(part.file), scale=".001 .001 .001", inertia="shell")
        add(parent, "geom", name=f"cad{k}", type="mesh", mesh=f"cad{k}", pos=vec(-origin),
            rgba=vec(part.rgba), density="0")
    ET.indent(model)
    return ET.tostring(model, encoding="unicode")


def loop_error_mm(data):
    return 1000*max(np.linalg.norm(data.site(f"tip{i}").xpos-data.site(f"socket{i}").xpos)
                    for i in range(1,4))


def exercise(data):
    # Bounded synthetic inputs to excite both tilt axes. Units are crank radians.
    phase = 2*np.pi*.8*data.time
    gain = min(1., data.time/.18)
    data.ctrl[:] = .12*gain*np.sin(phase+np.arange(3)*2*np.pi/3)


def step_frame(model, data, drive=True):
    for _ in range(10):  # 0.5 ms physics steps per 5 ms exposure interval
        if drive:
            exercise(data)
        mujoco.mj_step(model,data)
    # mj_step integrates after computing derived body/site positions.
    mujoco.mj_forward(model,data)
