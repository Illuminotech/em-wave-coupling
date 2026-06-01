"""Driver: two crossing Gaussian wave packets in linear vacuum.

Configure each wave's wavelength, amplitude, propagation angle, waist, and
pulse length. Runs the FDTD core, saves snapshots, and (optionally) animates.

Linearity test: also runs each wave alone and checks that the pair-run is
the sum of the singles within numerical noise. This is the *baseline* for
all subsequent nonlinear work — if it fails, no nonlinear claim is meaningful.
"""

from __future__ import annotations
import os
import numpy as np

from fdtd2d import FDTD2D, Grid, Material, add_gaussian_packet


# --------------------------------------------------------------------------
# Scenario configuration
# --------------------------------------------------------------------------

# Two waves, in natural units (c = 1).
WAVE_A = dict(wavelength=1.0,  angle_deg=20.0,  amplitude=1.0,
              waist=4.0, pulse_length=4.0, center=(8.0,  8.0))
WAVE_B = dict(wavelength=1.5,  angle_deg=-30.0, amplitude=0.7,
              waist=4.0, pulse_length=4.0, center=(8.0, 22.0))

GRID = Grid(nx=320, ny=240, dx=0.05, dy=0.05)
N_STEPS = 600
SNAPSHOT_EVERY = 50


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def make_sim() -> FDTD2D:
    return FDTD2D(grid=GRID, material=Material(eps_r=1.0, mu_r=1.0), cfl=0.7)


def run(sim: FDTD2D, n_steps: int, record_steps=None) -> dict:
    record_steps = set(record_steps or [])
    snapshots = {}
    energies = []
    for n in range(n_steps + 1):
        if n in record_steps:
            snapshots[n] = sim.Ez.copy()
        energies.append(sim.energy())
        if n < n_steps:
            sim.step()
    return {'snapshots': snapshots, 'energies': np.array(energies)}


def linearity_check():
    """Run A alone, B alone, and A+B together. Compare A+B vs sum(A, B)."""
    snaps_A = list(range(0, N_STEPS + 1, SNAPSHOT_EVERY))

    sim_A = make_sim()
    add_gaussian_packet(sim_A, **WAVE_A)
    out_A = run(sim_A, N_STEPS, record_steps=snaps_A)

    sim_B = make_sim()
    add_gaussian_packet(sim_B, **WAVE_B)
    out_B = run(sim_B, N_STEPS, record_steps=snaps_A)

    sim_AB = make_sim()
    add_gaussian_packet(sim_AB, **WAVE_A)
    add_gaussian_packet(sim_AB, **WAVE_B)
    out_AB = run(sim_AB, N_STEPS, record_steps=snaps_A)

    # Linearity residual: sup-norm of (Ez_AB - Ez_A - Ez_B) / max|Ez_AB|.
    residuals = []
    for k in snaps_A:
        diff = out_AB['snapshots'][k] - out_A['snapshots'][k] - out_B['snapshots'][k]
        denom = max(np.max(np.abs(out_AB['snapshots'][k])), 1e-12)
        residuals.append(np.max(np.abs(diff)) / denom)

    return {
        'A': out_A, 'B': out_B, 'AB': out_AB,
        'snap_steps': snaps_A,
        'residuals': np.array(residuals),
    }


def save_snapshot_grid(out, path: str):
    """Save 2x3 panel of Ez snapshots from the combined run."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    snaps = out['AB']['snapshots']
    steps = sorted(snaps.keys())
    pick = steps[:: max(1, len(steps) // 6)][:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    vmax = max(np.max(np.abs(snaps[s])) for s in pick)
    for ax, s in zip(axes.flat, pick):
        im = ax.imshow(snaps[s].T, origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax,
                       extent=[0, GRID.Lx, 0, GRID.Ly])
        ax.set_title(f'step {s}')
        ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.suptitle('E_z(x, y, t) — two crossing Gaussian packets, linear vacuum')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

if __name__ == '__main__':
    out = linearity_check()

    print('=== Linearity residual: ||E_AB - E_A - E_B||_inf / ||E_AB||_inf ===')
    for s, r in zip(out['snap_steps'], out['residuals']):
        print(f'  step {s:5d}    residual = {r:.3e}')

    e_AB = out['AB']['energies']
    e_sum = out['A']['energies'] + out['B']['energies']
    drift = (e_AB - e_sum) / np.maximum(e_sum, 1e-12)
    print('\n=== Energy: pair vs sum-of-singles (linear must give 0) ===')
    for k in [0, len(e_AB)//4, len(e_AB)//2, 3*len(e_AB)//4, len(e_AB)-1]:
        print(f'  t-step {k:5d}    E_AB = {e_AB[k]:.4f}    '
              f'E_A+E_B = {e_sum[k]:.4f}    rel diff = {drift[k]:+.3e}')

    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(out_dir, 'two_wave_linear.png')
    save_snapshot_grid(out, fig_path)
    print(f'\nSnapshots saved to {fig_path}')
