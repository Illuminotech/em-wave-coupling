"""Step 2: two crossing wave packets in a Kerr (chi^(3)) medium.

Compares linear vacuum vs nonlinear Kerr on the *same* two-wave scenario.

Key diagnostic: the linearity residual
    R(t) = ||Ez_AB(t) - Ez_A(t) - Ez_B(t)||_inf / ||Ez_AB(t)||_inf,
which in linear vacuum is at machine precision (Step 1) and which in a Kerr
medium grows as ~ chi3 * |E|^2. Nonzero R(t) is the signature of wave-wave
interaction (cross-phase modulation, four-wave mixing, third-harmonic).
"""

from __future__ import annotations
import os
import numpy as np

from fdtd2d import FDTD2D, Grid, Material, KerrMaterial, add_gaussian_packet


# Same scenario as Step 1 — keeps comparison clean.
WAVE_A = dict(wavelength=1.0,  angle_deg=20.0,  amplitude=1.0,
              waist=4.0, pulse_length=4.0, center=(8.0,  8.0))
WAVE_B = dict(wavelength=1.5,  angle_deg=-30.0, amplitude=0.7,
              waist=4.0, pulse_length=4.0, center=(8.0, 22.0))

GRID = Grid(nx=320, ny=240, dx=0.05, dy=0.05)
N_STEPS = 600
SNAPSHOT_EVERY = 50
CHI3 = 0.10            # Kerr coefficient in natural units; chi3 * |E|^2 ~ 0.1


def make_sim(material: Material) -> FDTD2D:
    return FDTD2D(grid=GRID, material=material, cfl=0.7)


def run(sim: FDTD2D, n_steps: int, record_steps) -> dict:
    record = set(record_steps)
    snapshots, energies = {}, []
    for n in range(n_steps + 1):
        if n in record:
            snapshots[n] = sim.Ez.copy()
        energies.append(sim.energy())
        if n < n_steps:
            sim.step()
    return {'snapshots': snapshots, 'energies': np.array(energies)}


def linearity_check(material_factory, label: str):
    """Run A, B, AB on `material_factory()` and compare."""
    snaps = list(range(0, N_STEPS + 1, SNAPSHOT_EVERY))

    sA = make_sim(material_factory());  add_gaussian_packet(sA, **WAVE_A)
    sB = make_sim(material_factory());  add_gaussian_packet(sB, **WAVE_B)
    sAB = make_sim(material_factory()); add_gaussian_packet(sAB, **WAVE_A); add_gaussian_packet(sAB, **WAVE_B)

    rA = run(sA, N_STEPS, snaps)
    rB = run(sB, N_STEPS, snaps)
    rAB = run(sAB, N_STEPS, snaps)

    residuals = []
    for k in snaps:
        diff = rAB['snapshots'][k] - rA['snapshots'][k] - rB['snapshots'][k]
        denom = max(np.max(np.abs(rAB['snapshots'][k])), 1e-12)
        residuals.append(np.max(np.abs(diff)) / denom)

    return dict(label=label, A=rA, B=rB, AB=rAB,
                snap_steps=snaps, residuals=np.array(residuals))


def comparison_plot(linear, kerr, path: str):
    """Side-by-side: linear AB vs Kerr AB at matched time steps; residual curve."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pick = linear['snap_steps'][:: max(1, len(linear['snap_steps']) // 4)][:4]
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    vmax = max(np.max(np.abs(linear['AB']['snapshots'][s])) for s in pick)

    for j, s in enumerate(pick):
        L = linear['AB']['snapshots'][s]
        K = kerr['AB']['snapshots'][s]
        diff = K - L
        for i, (img, title, vm) in enumerate([
            (L, f'linear, step {s}', vmax),
            (K, f'Kerr,   step {s}', vmax),
            (diff, f'diff (Kerr - linear), step {s}', max(np.max(np.abs(diff)), 1e-12)),
        ]):
            ax = axes[i, j]
            ax.imshow(img.T, origin='lower', cmap='RdBu_r',
                      vmin=-vm, vmax=vm,
                      extent=[0, GRID.Lx, 0, GRID.Ly])
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'Two-wave interaction: linear vacuum vs Kerr (chi3 = {CHI3})')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def residual_plot(linear, kerr, path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(linear['snap_steps'], np.maximum(linear['residuals'], 1e-17),
                'o-', label='linear vacuum')
    ax.semilogy(kerr['snap_steps'], np.maximum(kerr['residuals'], 1e-17),
                's-', label=f'Kerr, chi3 = {CHI3}')
    ax.set_xlabel('time step')
    ax.set_ylabel(r'$\|E_{AB} - E_A - E_B\|_\infty / \|E_{AB}\|_\infty$')
    ax.set_title('Linearity residual: nonzero ⇒ wave–wave interaction')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    print('Running linear vacuum baseline ...')
    lin = linearity_check(lambda: Material(eps_r=1.0, mu_r=1.0), 'linear')
    print('Running Kerr (chi3 = %.3f) ...' % CHI3)
    kerr = linearity_check(lambda: KerrMaterial(eps_r=1.0, chi3=CHI3), 'kerr')

    print('\n=== Linearity residual at sample steps ===')
    print(f'{"step":>6} {"linear":>14} {"Kerr":>14}   ratio Kerr/linear')
    for k, rl, rk in zip(lin['snap_steps'], lin['residuals'], kerr['residuals']):
        ratio = rk / max(rl, 1e-18)
        print(f'{k:>6} {rl:>14.3e} {rk:>14.3e}   {ratio:>10.2e}')

    out_dir = os.path.dirname(os.path.abspath(__file__))
    comparison_plot(lin, kerr, os.path.join(out_dir, 'two_wave_kerr_compare.png'))
    residual_plot(lin, kerr, os.path.join(out_dir, 'two_wave_kerr_residual.png'))
    print(f'\nFigures saved to {out_dir}/two_wave_kerr_compare.png and .../residual.png')
