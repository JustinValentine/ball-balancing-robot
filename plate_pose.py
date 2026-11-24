"""Estimate plate normals from measured 3-RRS actuator positions."""
import argparse
import asyncio
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


class Platform:
    """Parallel revolute axes within each RRS leg, expressed in the CAD assembly frame."""

    def __init__(self, cfg):
        def vectors(key):
            values = np.array([leg[key] for leg in cfg["legs"]], dtype=float)
            if values.shape != (3, 3) or not np.isfinite(values).all():
                raise ValueError(f"Expected three finite CAD vectors for {key}")
            return values

        self.motor = vectors("motor_mm")
        elbow = vectors("elbow_mm")
        ball = vectors("ball_joint_mm")
        self.axis = vectors("axis")
        norms = np.linalg.norm(self.axis, axis=1)
        if min(norms) < 1e-8:
            raise ValueError("Zero joint axis")
        self.axis /= norms[:, None]
        self.crank = elbow - self.motor
        self.rod = ball - elbow
        self.edges = np.array([1, 2, 0])
        self.side = np.linalg.norm(ball-ball[self.edges], axis=1)
        self.branch = np.sign(np.sum(np.cross(self.crank, self.rod)*self.axis, axis=1))
        if min(self.side) <= 0 or not self.branch.all():
            raise ValueError("Degenerate home geometry")
        self.zero = np.array(cfg["level_output_turns"], dtype=float)
        # Crank angles (from the CAD pose) at which the encoders read level_output_turns; zero
        # when the encoders were zeroed at the CAD pose itself.
        self.zero_crank = np.array(cfg.get("zero_crank_rad", [0, 0, 0]), dtype=float)
        self.scale = np.array(cfg["crank_turns_per_output_turn"], dtype=float)
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in (self.zero, self.scale, self.zero_crank)) or (self.scale == 0).any():
            raise ValueError("Set three finite encoder zeros and three signed nonzero scales")
        self.rotation = np.array(cfg["left_from_base_rotation"], dtype=float)
        if (self.rotation.shape != (3, 3) or not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(self.rotation), 1)):
            raise ValueError("left_from_base_rotation must be a proper rotation")
        self.ids = cfg["moteus_ids"]
        if len(self.ids) != 3 or len(set(self.ids)) != 3 or any(type(i) is not int or not 1 <= i <= 126 for i in self.ids):
            raise ValueError("Specify three distinct moteus IDs in radial-leg order")
        self.beta = np.zeros(3)

    def rotated(self, vectors, angles):
        # Rodrigues rotation around each leg's own pair of parallel axes.
        c = np.cos(angles)[:, None]
        axial = self.axis*np.sum(self.axis*vectors, axis=1)[:, None]
        return axial + (vectors-axial)*c + np.cross(self.axis,vectors)*np.sin(angles)[:, None]

    def normal(self, positions):
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (3,) or not np.isfinite(positions).all():
            raise ValueError("Expected three finite output positions in turns")
        angles = self.zero_crank+2*np.pi*self.scale*(positions-self.zero)
        crank = self.rotated(self.crank, angles)
        elbows = self.motor + crank
        beta = self.beta.copy()
        for _ in range(24):
            rods = self.rotated(self.rod, angles + beta)
            points = elbows + rods
            edges = points - points[self.edges]
            lengths = np.linalg.norm(edges, axis=1)
            if min(lengths) < 1e-8:
                break
            residual = lengths-self.side
            tangent = np.cross(self.axis, rods)
            units = edges/lengths[:, None]
            jac = np.diag(np.sum(units*tangent, axis=1))
            jac[[0, 1, 2], [1, 2, 0]] = -np.sum(units*tangent[[1, 2, 0]], axis=1)
            if np.linalg.cond(jac) > 1e6:
                break
            if max(abs(residual)) < 1e-6:
                n = np.cross(edges[0], edges[1])
                hand = np.sum(np.cross(crank, rods)*self.axis, axis=1)
                if n[2] <= 0 or np.any(hand*self.branch <= 0):
                    break
                self.beta = beta
                return self.rotation @ (n/np.linalg.norm(n))
            step = np.linalg.solve(jac, residual)
            beta -= step / max(1., max(abs(step))/.25)
        raise ValueError("No well-conditioned solution on the upright operating branch")


def publish(model, timestamp, positions):
    n = model.normal(positions)
    print(timestamp, *(f"{v:.10f}" for v in n), flush=True)


def convert(model, stream):
    last = 0
    for i, line in enumerate(stream, 1):
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"Expected timestamp and three output positions on line {i}")
        timestamp = int(fields[0])
        if timestamp <= last:
            raise ValueError(f"Non-increasing timestamp on line {i}")
        publish(model, timestamp, [float(x) for x in fields[1:]])
        last = timestamp


async def query(model, interface, hz, max_span_us):
    import moteus
    period = 1/hz
    # The SDK encodes register reads and decodes replies. Each request is query-only.
    async with moteus.PythonCan(interface="socketcan", channel=interface, fd=True) as bus:
        commands = [moteus.Controller(id=i, transport=bus).make_query() for i in model.ids]
        rejected = 0
        last_timestamp = 0
        while True:
            start = time.monotonic_ns()
            wall_us = time.time_ns()//1000
            result = await asyncio.wait_for(bus.cycle(commands), timeout=1.)
            span_us = (time.monotonic_ns()-start)//1000
            positions = {r.id: r.values.get(moteus.Register.POSITION) for r in result}
            stamp = wall_us + span_us//2
            if stamp <= last_timestamp:
                raise ValueError("Host clock stepped backward")
            if (span_us <= max_span_us and len(result) == 3 and
                    all(positions.get(i) is not None for i in model.ids)):
                publish(model, stamp, [positions[i] for i in model.ids])
                last_timestamp = stamp
            else:
                rejected += 1
                if rejected == 1 or rejected % 200 == 0:
                    print(f"Discarded {rejected} incomplete or slow CAN batches", file=sys.stderr)
            await asyncio.sleep(max(0., period-(time.monotonic_ns()-start)*1e-9))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--positions", type=argparse.FileType("r"), help="Timestamped turns log, or - for stdin")
    sources.add_argument("--can", metavar="IFACE", help="Query the three controllers on a Linux SocketCAN FD interface")
    parser.add_argument("--hz", type=float, default=200.)
    parser.add_argument("--max-span-us", type=int, default=4000)
    args = parser.parse_args()
    if not math.isfinite(args.hz) or args.hz <= 0 or args.max_span_us <= 0:
        parser.error("Query frequency and maximum batch duration must be positive")
    model = Platform(json.loads(args.config.read_text()))
    if args.positions is not None:
        convert(model, args.positions)
    else:
        asyncio.run(query(model, args.can, args.hz, args.max_span_us))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
