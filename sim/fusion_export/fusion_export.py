"""Export the visible bodies of the open Fusion design for the ball-vision simulator.

Open the platform assembly at its home pose and run this script from
Utilities > Add-Ins > Scripts and Add-Ins. It writes sim/cad/manifest.json and
one binary STL per visible body, in millimetres in the root component frame,
which is the frame of platform.example.json.
"""
import json
import math
import os
import struct
import traceback

import adsk.core
import adsk.fusion

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cad")
MM = 10.0  # Fusion API lengths are centimetres.


def transform_points(coords, matrix):
    if matrix is None:
        return list(coords)
    m = matrix.asArray()  # Row-major, column-vector convention.
    out = []
    for i in range(0, len(coords), 3):
        x, y, z = coords[i:i+3]
        out += [m[0]*x+m[1]*y+m[2]*z+m[3], m[4]*x+m[5]*y+m[6]*z+m[7], m[8]*x+m[9]*y+m[10]*z+m[11]]
    return out


def bounds(coords):
    lo = [min(coords[i::3]) for i in range(3)]
    hi = [max(coords[i::3]) for i in range(3)]
    return lo, hi


def path_transforms(occurrence):
    # Compose each level's own placement. Both orders are offered because the
    # bounds check below decides which one reproduces the assembly position.
    chain = []
    while occurrence is not None:
        chain.append((occurrence.nativeObject or occurrence).transform2)
        occurrence = occurrence.assemblyContext
    results = []
    for order in (chain, chain[::-1]):
        total = adsk.core.Matrix3D.create()
        for step in order:
            total.transformBy(step)
        results.append(total)
    return results


def mesh_of(body):
    if isinstance(body, adsk.fusion.MeshBody):
        mesh = body.displayMesh
    else:
        calc = body.meshManager.createMeshCalculator()
        calc.setQuality(adsk.fusion.TriangleMeshQualityOptions.NormalQualityTriangleMesh)
        mesh = calc.calculate()
    return list(mesh.nodeCoordinatesAsFloat), list(mesh.nodeIndices)


def placed(body, occurrence, coords):
    """Pick the mesh frame whose bounds match the body's assembly-context bounds."""
    box = body.boundingBox
    lo = [box.minPoint.x, box.minPoint.y, box.minPoint.z]
    hi = [box.maxPoint.x, box.maxPoint.y, box.maxPoint.z]
    target = [(a+b)/2 for a, b in zip(lo, hi)]
    candidates = [("context", None)]
    if occurrence is not None:
        leaf_first, root_first = path_transforms(occurrence)
        candidates += [("transform2", occurrence.transform2), ("path", leaf_first), ("path_reversed", root_first)]
    best = None
    for label, matrix in candidates:
        points = transform_points(coords, matrix)
        mlo, mhi = bounds(points)
        error = math.dist(target, [(a+b)/2 for a, b in zip(mlo, mhi)])
        if best is None or error < best[0]:
            best = (error, label, points)
    return best


def colour(body):
    rgba = [0.62, 0.64, 0.67, 1.0]
    try:
        appearance = body.appearance
        props = appearance.appearanceProperties
        ranked = []
        for i in range(props.count):
            prop = adsk.core.ColorProperty.cast(props.item(i))
            if prop and prop.value:
                ident = prop.id.lower()
                rank = 0 if ("albedo" in ident or "diffuse" in ident) else 1
                ranked.append((rank, prop.value))
        if ranked:
            c = sorted(ranked, key=lambda r: r[0])[0][1]
            rgba = [c.red/255, c.green/255, c.blue/255, c.opacity/255 if c.opacity else 1.0]
        name = appearance.name.lower()
        if any(word in name for word in ("glass", "acrylic", "clear", "transparent", "polycarbonate")):
            rgba[3] = min(rgba[3], 0.35)
        return rgba, appearance.name
    except Exception:
        return rgba, ""


def write_stl(path, coords, indices):
    with open(path, "wb") as f:
        f.write(b"ball-vision Fusion export, millimetres".ljust(80, b" "))
        f.write(struct.pack("<I", len(indices)//3))
        for t in range(0, len(indices), 3):
            a, b, c = (coords[3*k:3*k+3] for k in indices[t:t+3])
            u = [b[i]-a[i] for i in range(3)]
            v = [c[i]-a[i] for i in range(3)]
            n = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
            s = math.sqrt(sum(x*x for x in n)) or 1.0
            f.write(struct.pack("<12fH", *(x/s for x in n), *a, *b, *c, 0))


def joint_record(joint, built):
    record = {"name": joint.name, "as_built": built}
    try:
        record["type"] = int(joint.jointMotion.jointType)
        for side in ("occurrenceOne", "occurrenceTwo"):
            occ = getattr(joint, side)
            record[side] = occ.fullPathName if occ else ""
        motion = adsk.fusion.RevoluteJointMotion.cast(joint.jointMotion)
        if motion and motion.rotationAxisVector:
            axis = motion.rotationAxisVector
            record["axis"] = [axis.x, axis.y, axis.z]
        geometry = joint.geometry if built else joint.geometryOrOriginOne
        if isinstance(geometry, adsk.fusion.JointOrigin):
            geometry = geometry.geometry
        if geometry and geometry.origin:
            p = geometry.origin
            record["origin_mm"] = [p.x*MM, p.y*MM, p.z*MM]
    except Exception as e:
        record["error"] = str(e)
    return record


def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            ui.messageBox("Open the platform design in the Design workspace first.")
            return
        root = design.rootComponent
        os.makedirs(os.path.join(OUT, "bodies"), exist_ok=True)
        for old in os.listdir(os.path.join(OUT, "bodies")):
            if old.endswith(".stl"):
                os.remove(os.path.join(OUT, "bodies", old))

        owners = [(None, root)] + [(occ, occ.component) for occ in root.allOccurrences]
        bodies, hidden, failed = [], [], []
        for occ, component in owners:
            for collection in (component.bRepBodies, component.meshBodies):
                for i in range(collection.count):
                    body = collection.item(i)
                    label = (occ.fullPathName + "/" if occ else "") + body.name
                    try:
                        if occ is not None:
                            body = body.createForAssemblyContext(occ)
                        if not body.isVisible or (occ is not None and not occ.isVisible):
                            hidden.append(label)
                            continue
                        coords, indices = mesh_of(body)
                        if not indices:
                            failed.append({"key": label, "error": "empty mesh"})
                            continue
                        error, frame, points = placed(body, occ, coords)
                    except Exception as e:
                        failed.append({"key": label, "error": str(e)})
                        continue
                    points = [x*MM for x in points]
                    filename = "bodies/%03d.stl" % len(bodies)
                    write_stl(os.path.join(OUT, filename), points, indices)
                    rgba, material = colour(body)
                    lo, hi = bounds(points)
                    bodies.append({
                        "file": filename, "key": label, "body": body.name,
                        "occurrence": occ.fullPathName if occ else "",
                        "component": component.name, "appearance": material, "rgba": rgba,
                        "triangles": len(indices)//3, "frame": frame,
                        "placement_error_mm": error*MM, "bounds_mm": [lo, hi]})

        occurrences = []
        for occ in root.allOccurrences:
            m = occ.transform2.asArray()
            occurrences.append({"path": occ.fullPathName, "component": occ.component.name,
                                "visible": occ.isVisible,
                                "transform_mm": [[m[4*r+c]*(MM if c == 3 else 1) for c in range(4)] for r in range(3)]})
        joints = []
        for collection, built in (("allJoints", False), ("allAsBuiltJoints", True)):
            try:
                joints += [joint_record(j, built) for j in getattr(root, collection)]
            except Exception:
                pass
        document = app.activeDocument
        version = None
        try:
            version = document.dataFile.versionNumber
        except Exception:
            pass
        manifest = {"format": 1, "units": "mm", "document": document.name, "version": version,
                    "bodies": bodies, "hidden": hidden, "failed": failed, "occurrences": occurrences, "joints": joints}
        with open(os.path.join(OUT, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
        worst = max([b["placement_error_mm"] for b in bodies] or [0])
        ui.messageBox("Exported %d visible bodies to\n%s\n\n%d hidden bodies skipped, %d failed.\n"
                      "Largest placement check: %.3f mm" % (len(bodies), OUT, len(hidden), len(failed), worst))
    except Exception:
        if ui:
            ui.messageBox("Export failed:\n" + traceback.format_exc())
