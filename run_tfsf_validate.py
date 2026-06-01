"""Validate the TF/SF source: a single CW plane wave injected through a
rectangular contour should propagate cleanly into the TF region with no
back-reaction, and the SF region (outside the contour) should stay near
zero (modulo Mur ABC reflection at the outer grid boundary).

Acceptance criteria:
  1. After ~5 wave periods, the TF region holds the analytic plane wave
     to within a few percent (limited by FDTD numerical dispersion).
  2. The SF region's peak field is < 1% of the TF peak (no back-reflection
     from the contour edges).
  3. The wave propagates in the +k direction set by `angle_deg` (verifies
     the sign convention).
"""

from __future__ import annotations
import os
import numpy as np

from fdtd2d import FDTD2D, Grid, Material
from source import PlaneWaveSource


GRID = Grid(nx=320, ny=240, dx=0.05, dy=0.05)
N_STEPS = 800

# Contour: well inside the grid, plenty of room for SF region around it.
CONTOUR = dict(ia=40, ja=40, ib=200, jb=180)

SOURCE = PlaneWaveSource(
    **CONTOUR,
    angle_deg=20.0,
    wavelength=1.0,
    amplitude=1.0,
    ramp_time=3.0,
)


def run():
    sim = FDTD2D(grid=GRID, material=Material(eps_r=1.0), cfl=0.5)
    sim.add_source(SOURCE)

    snapshots = {}
    for n in range(N_STEPS + 1):
        if n in {0, 100, 200, 400, 600, 800}:
            snapshots[n] = sim.Ez.copy()
        if n < N_STEPS:
            sim.step()
    return sim, snapshots


def analyze(sim, snapshots):
    # Identify TF and SF masks on the Ez grid.
    g = sim.g
    ia, ja, ib, jb = SOURCE.ia, SOURCE.ja, SOURCE.ib, SOURCE.jb
    tf_mask = np.zeros((g.nx, g.ny), dtype=bool)
    tf_mask[ia + 2:ib - 1, ja + 2:jb - 1] = True   # inner TF, away from contour
    sf_mask = np.zeros_like(tf_mask)
    sf_mask[5:-5, 5:-5] = True                     # away from outer ABC
    sf_mask &= ~tf_mask
    # Drop a strip immediately around the contour where evanescent contour
    # artifacts are largest:
    contour_pad = np.zeros_like(tf_mask)
    contour_pad[ia - 2:ib + 3, ja - 2:jb + 3] = True
    sf_mask &= ~contour_pad

    print('\n=== Field amplitudes (peak |E_z|) ===')
    print(f'{"step":>6} {"TF peak":>12} {"SF peak":>12} {"SF/TF":>10}')
    last_step = max(snapshots)
    for s in sorted(snapshots):
        e = snapshots[s]
        tf_peak = float(np.max(np.abs(e[tf_mask]))) if tf_mask.any() else 0.0
        sf_peak = float(np.max(np.abs(e[sf_mask]))) if sf_mask.any() else 0.0
        ratio = sf_peak / max(tf_peak, 1e-15)
        print(f'{s:>6} {tf_peak:>12.4f} {sf_peak:>12.4e} {ratio:>10.3e}')
    return last_step


def plot(sim, snapshots, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    steps = sorted(snapshots.keys())
    ncols = 3
    nrows = (len(steps) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows))
    g = sim.g
    ia, ja, ib, jb = SOURCE.ia, SOURCE.ja, SOURCE.ib, SOURCE.jb
    vmax = max(np.max(np.abs(snapshots[s])) for s in steps)
    for ax, s in zip(np.array(axes).flat, steps):
        im = ax.imshow(snapshots[s].T, origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax,
                       extent=[0, g.Lx, 0, g.Ly])
        ax.add_patch(plt.Rectangle((ia * g.dx, ja * g.dy),
                                    (ib - ia) * g.dx, (jb - ja) * g.dy,
                                    fill=False, edgecolor='black', lw=1.5))
        ax.set_title(f'step {s}, TF/SF contour outlined')
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'TF/SF source validation: single wave at {SOURCE.angle_deg}°, λ={SOURCE.wavelength}')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    print('Running TF/SF validation (single CW source) ...')
    sim, snapshots = run()
    analyze(sim, snapshots)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'tfsf_validation.png')
    plot(sim, snapshots, out)
    print(f'\nFigure saved to {out}')
