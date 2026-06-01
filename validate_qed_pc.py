"""Validate the predictor-corrector for the QED HE constitutive.

Theory: for a single plane wave |E| = |B|, so F = (E²-B²)/2 = 0 identically
and the QED nonlinearity does nothing. A Gaussian packet (not strictly plane)
gives a small envelope-derived F of order (1/(k·waist))² ~ 1e-3 for our setup,
so the maximum legitimate single-wave self-action is ~1e-3.

The previous (single-shot) solver had ‖A_qed − A_lin‖∞ / max|A_lin| ~ 0.5–0.8
at the end of a 600-step run — orders of magnitude above the physics limit.
That's the time-staggering artifact: the cubic D = E·(1+4κ(E²-B²)) was solved
with B at time n while D is at time n+1.

Predictor-corrector self-consistently iterates B² ↔ E^{n+1} via a one-step
Faraday prediction. Expected: self-action drops by orders of magnitude with
just 2-3 PC iterations.
"""

from __future__ import annotations
import os
import numpy as np

from fdtd2d import (FDTD2D, Grid, Material, HeisenbergEulerMaterial,
                    add_gaussian_packet)


WAVE = dict(wavelength=1.0, angle_deg=20.0, amplitude=1.0,
            waist=4.0, pulse_length=8.0, center=(10.0, 6.0))
GRID = Grid(nx=320, ny=240, dx=0.05, dy=0.05)
N_STEPS = 600
SNAPSHOT_EVERY = 100
KAPPA = 0.01


def run_single_wave(material_factory, label: str):
    sim = FDTD2D(grid=GRID, material=material_factory(), cfl=0.35,
                 boundary='mur')
    add_gaussian_packet(sim, **WAVE)
    snaps = {}
    for n in range(N_STEPS + 1):
        if n % SNAPSHOT_EVERY == 0:
            snaps[n] = sim.Ez.copy()
        if n < N_STEPS:
            sim.step()
    return snaps


def self_action_norm(snaps_qed, snaps_lin) -> dict:
    out = {}
    for k in snaps_qed:
        diff = snaps_qed[k] - snaps_lin[k]
        ref = max(np.max(np.abs(snaps_lin[k])), 1e-12)
        out[k] = float(np.max(np.abs(diff)) / ref)
    return out


def main():
    print('Running linear vacuum baseline ...')
    lin = run_single_wave(lambda: Material(eps_r=1.0), 'linear')

    print('Running QED without PC (single-shot cubic solve) ...')
    qed_nopc = run_single_wave(
        lambda: HeisenbergEulerMaterial(kappa=KAPPA, pc_max_iters=0),
        'qed_no_pc',
    )

    print('Running QED with PC (2 iterations) ...')
    qed_pc2 = run_single_wave(
        lambda: HeisenbergEulerMaterial(kappa=KAPPA, pc_max_iters=2),
        'qed_pc2',
    )

    print('Running QED with PC (5 iterations) ...')
    qed_pc5 = run_single_wave(
        lambda: HeisenbergEulerMaterial(kappa=KAPPA, pc_max_iters=5),
        'qed_pc5',
    )

    s_nopc = self_action_norm(qed_nopc, lin)
    s_pc2 = self_action_norm(qed_pc2, lin)
    s_pc5 = self_action_norm(qed_pc5, lin)

    print('\n=== Single-wave self-action ‖A_qed − A_lin‖∞ / max|A_lin| ===')
    print(f'{"step":>6} {"no PC":>12} {"PC=2":>12} {"PC=5":>12}  improvement (PC=2/no_PC)')
    for k in sorted(s_nopc.keys()):
        if k == 0:
            continue   # both start at 0
        improvement = s_nopc[k] / max(s_pc2[k], 1e-30)
        print(f'{k:>6} {s_nopc[k]:>12.3e} {s_pc2[k]:>12.3e} '
              f'{s_pc5[k]:>12.3e} {improvement:>14.2e}×')

    final_steps = sorted(s_nopc.keys())[-1]
    print(f'\nFinal-step summary (step {final_steps}):')
    print(f'  No PC :  self-action = {s_nopc[final_steps]:.3e}')
    print(f'  PC=2  :  self-action = {s_pc2[final_steps]:.3e}    '
          f'({s_nopc[final_steps]/max(s_pc2[final_steps],1e-30):.1e}× better)')
    print(f'  PC=5  :  self-action = {s_pc5[final_steps]:.3e}    '
          f'({s_nopc[final_steps]/max(s_pc5[final_steps],1e-30):.1e}× better)')

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        steps = sorted(s_nopc.keys())
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.semilogy(steps, [s_nopc[k] for k in steps], 'o-', label='no PC')
        ax.semilogy(steps, [s_pc2[k]  for k in steps], 's-', label='PC, 2 iters')
        ax.semilogy(steps, [s_pc5[k]  for k in steps], '^-', label='PC, 5 iters')
        ax.set_xlabel('time step')
        ax.set_ylabel('single-wave self-action  ‖A_qed − A_lin‖∞ / max|A_lin|')
        ax.set_title(f'Predictor-corrector eliminates QED time-staggering artifact (κ = {KAPPA})')
        ax.grid(True, which='both', alpha=0.3); ax.legend()
        fig.tight_layout()
        path = __file__.replace('.py', '_plot.png')
        fig.savefig(path, dpi=120)
        print(f'\nFigure saved to {path}')
    except ImportError:
        pass


if __name__ == '__main__':
    main()
