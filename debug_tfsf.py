"""Debug TF/SF: look at field amplitudes at probe points inside vs outside
the contour, with E-only or H-only corrections active."""
from __future__ import annotations
import numpy as np
from fdtd2d import FDTD2D, Grid, Material
from source import PlaneWaveSource

GRID = Grid(nx=200, ny=200, dx=0.05, dy=0.05)
N_STEPS = 400
SRC = PlaneWaveSource(ia=40, ja=40, ib=160, jb=160,
                       angle_deg=0.0, wavelength=1.0, amplitude=1.0,
                       ramp_time=2.0)


def run(mode: str):
    sim = FDTD2D(grid=GRID, material=Material(), cfl=0.5)
    # Wrap source so we can disable E or H corrections.
    class Wrapped:
        def __init__(self, src, mode):
            self.src = src
            self.mode = mode
        def apply_E_corrections(self, sim, t):
            if self.mode in ('both', 'E_only'):
                self.src.apply_E_corrections(sim, t)
        def apply_H_corrections(self, sim, t):
            if self.mode in ('both', 'H_only'):
                self.src.apply_H_corrections(sim, t)
    sim.add_source(Wrapped(SRC, mode))
    # Probe inside TF (around middle of contour) and outside in SF.
    in_pt  = (100, 100)   # interior of contour
    out_pt = (10,  100)   # outside contour, to the left
    in_hist, out_hist = [], []
    for n in range(N_STEPS):
        sim.step()
        in_hist.append(sim.Ez[in_pt[0], in_pt[1]])
        out_hist.append(sim.Ez[out_pt[0], out_pt[1]])
    return np.array(in_hist), np.array(out_hist)


for mode in ['both', 'E_only', 'H_only']:
    in_h, out_h = run(mode)
    print(f'\n--- mode = {mode} ---')
    print(f'inside  TF, peak |Ez| = {np.max(np.abs(in_h)):.4f}')
    print(f'outside TF, peak |Ez| = {np.max(np.abs(out_h)):.4f}')
    print(f'last 5 inside : {in_h[-5:]}')
    print(f'last 5 outside: {out_h[-5:]}')
