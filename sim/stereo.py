"""Offline MONO8 camera model with a finite plane-parallel acrylic sheet."""
import numpy as np

from mechanism import BALL_RADIUS, THICKNESS

WIDTH, HEIGHT = 640, 480
# Nominal focal length / pixel pitch. Radial coefficients are synthetic.
FOCAL = 3.5/.0074
K1, K2 = -.18, .025
RIG = {"left_guid":"53494d0000000001", "right_guid":"53494d0000000002"}


def calibration(path, baseline):
    def matrix(rows, cols, values):
        return (f"!!opencv-matrix\n   rows: {rows}\n   cols: {cols}\n   dt: d\n   data: ["
                + ", ".join(str(v) for v in values) + "]\n")
    k = matrix(3,3,[FOCAL,0,319.5,0,FOCAL,239.5,0,0,1])
    d = matrix(1,5,[K1,K2,0,0,0])
    path.write_text("%YAML:1.0\n" + f'width: {WIDTH}\nheight: {HEIGHT}\n'
                    + "".join(f'{key}: "{value}"\n' for key,value in RIG.items())
                    + f"K1: {k}D1: {d}K2: {k}D2: {d}R: " + matrix(3,3,np.eye(3).ravel())
                    + "T: " + matrix(3,1,[-baseline,0,0]))


class CameraPair:
    def __init__(self, rotation, diameter_mm):
        self.rotation = rotation
        self.radius = diameter_mm/2000
        y,x = np.indices((HEIGHT,WIDTH),dtype=np.float32)
        distorted = np.dstack(((x-319.5)/FOCAL,(y-239.5)/FOCAL))
        xy = distorted.copy()
        for _ in range(12):
            r2 = (xy*xy).sum(axis=2)
            xy = distorted/(1+K1*r2+K2*r2*r2)[:,:,None]
        rays = np.dstack((xy,np.ones_like(x)))
        self.rays = (rays/np.linalg.norm(rays,axis=2,keepdims=True)) @ rotation.astype(np.float32)
        self.background = (40+10*x/WIDTH+7*y/HEIGHT).astype(np.uint8)

    def image(self, eye, ball, surface, normal):
        # Apply Snell's law at the physical lower/upper planes. Rays missing
        # the disk go directly to the ball. Edge-wall refraction is omitted.
        d = self.rays
        n = np.asarray(normal,dtype=np.float32)
        lower = surface - n*THICKNESS
        c = d @ n
        h = float((lower-eye)@n)
        first = d * (h/np.maximum(c,1e-6))[:,:,None]
        tang = (d-c[:,:,None]*n)/1.49
        cos_inside = np.sqrt(1-(tang*tang).sum(axis=2))
        last = first + THICKNESS*(n+tang/cos_inside[:,:,None])
        entry_radius2 = ((first-(lower-eye).astype(np.float32))**2).sum(axis=2)
        exit_radius2 = ((last-(surface-eye).astype(np.float32))**2).sum(axis=2)
        through = (c>0) & (h>0) & (entry_radius2<=self.radius**2) & (exit_radius2<=self.radius**2)
        origin = np.where(through[:,:,None],last,0)
        offset = (ball-eye).astype(np.float32)-origin
        forward = (offset*d).sum(axis=2)
        perp = offset-forward[:,:,None]*d
        r2 = (perp*perp).sum(axis=2)/BALL_RADIUS**2
        hit = (r2<=1) & (forward>0)
        image = self.background.copy()
        image[through] += 5
        rim = through & (exit_radius2>(self.radius-.0015)**2)
        image[rim] = 100
        image[hit] = 200+48*np.sqrt(1-r2[hit])
        return image
