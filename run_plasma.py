"""Step 3: two crossing wave packets in a relativistic cold electron plasma.

Couples a cold relativistic electron fluid to Maxwell. The wave-wave
interaction channels visible here are:
  - relativistic mass correction (γ-induced refractive-index modulation),
  - ponderomotive density depletion in the overlap region,
  - stimulated Raman scattering (pump → Stokes ω - ω_p / anti-Stokes ω + ω_p),
  - plasma-mediated four-wave mixing.

Diagnostic: the linearity residual ||E_AB - E_A - E_B||_inf / ||E_AB||_inf,
plus the density modulation Δn / n_0 in the overlap region.
"""

from __future__ import annotations
import os
import numpy as np

from fdtd2d import (FDTD2D, Grid, Material, RelativisticPlasmaMaterial,
                    add_gaussian_packet)


# Same scenario as Step 1 / Step 2 — keeps cross-step comparison consistent.
WAVE_A = dict(wavelength=1.0,  angle_deg=20.0,  amplitude=0.30,
              waist=4.0, pulse_length=4.0, center=(8.0,  8.0))
WAVE_B = dict(wavelength=1.5,  angle_deg=-30.0, amplitude=0.20,
              waist=4.0, pulse_length=4.0, center=(8.0, 22.0))

GRID = Grid(nx=320, ny=240, dx=0.05, dy=0.05)
N_STEPS = 600
SNAPSHOT_EVERY = 50

# Plasma density (n0 in natural units) sets ω_p² = n0. With our wavelengths
# (ω_A = 2π/1.0 ≈ 6.28, ω_B = 2π/1.5 ≈ 4.19), n0 = 4 ⇒ ω_p = 2 ⇒ underdense
# for both waves and SRS allowed (ω - ω_p = 4.28 > ω_p for wave A).
N0 = 4.0

# a_0 ≡ E_amp / ω. Wave A: a_0 ≈ 0.30/6.28 ≈ 0.048. Wave B: a_0 ≈ 0.20/4.19 ≈ 0.048.
# Perturbative regime, chosen so that δn/n0 (which scales as a_0^2 with a numerical
# prefactor of ~100 once envelope and packet effects are included — see
# probe_plasma_scaling.py) stays bounded for the visualization. For a higher-a_0
# study, switch to TF/SF source injection and a Boris pusher: see notes at end of
# RelativisticPlasmaMaterial in fdtd2d.py.


def make_sim(material: Material) -> FDTD2D:
    return FDTD2D(grid=GRID, material=material, cfl=0.5)   # tighter CFL for plasma


def run(sim: FDTD2D, n_steps: int, record_steps) -> dict:
    record = set(record_steps)
    snapshots, energies, density_mods = {}, [], []
    for n in range(n_steps + 1):
        if n in record:
            snapshots[n] = sim.Ez.copy()
        energies.append(sim.energy())
        # Track density excursion if plasma is present.
        if isinstance(sim.mat, RelativisticPlasmaMaterial):
            density_mods.append(np.max(np.abs(sim.mat.n - sim.mat.n0)) / sim.mat.n0)
        else:
            density_mods.append(0.0)
        if n < n_steps:
            sim.step()
    return {'snapshots': snapshots,
            'energies': np.array(energies),
            'density_mods': np.array(density_mods)}


def linearity_check(material_factory, label: str):
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
                snap_steps=snaps, residuals=np.array(residuals),
                sim_AB=sAB)   # keep reference for density plot


def comparison_plot(linear, plasma, path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pick = linear['snap_steps'][:: max(1, len(linear['snap_steps']) // 4)][:4]
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    vmax = max(np.max(np.abs(linear['AB']['snapshots'][s])) for s in pick)

    for j, s in enumerate(pick):
        L = linear['AB']['snapshots'][s]
        P = plasma['AB']['snapshots'][s]
        diff = P - L
        for i, (img, title, vm) in enumerate([
            (L, f'linear vacuum, step {s}', vmax),
            (P, f'plasma (n0={N0}), step {s}', vmax),
            (diff, f'plasma − linear, step {s}', max(np.max(np.abs(diff)), 1e-12)),
        ]):
            ax = axes[i, j]
            ax.imshow(img.T, origin='lower', cmap='RdBu_r',
                      vmin=-vm, vmax=vm,
                      extent=[0, GRID.Lx, 0, GRID.Ly])
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Two-wave interaction: linear vacuum vs relativistic cold plasma')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def density_plot(plasma, path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sim = plasma['sim_AB']
    delta_n = (sim.mat.n - sim.mat.n0) / sim.mat.n0
    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = max(np.max(np.abs(delta_n)), 1e-12)
    im = ax.imshow(delta_n.T, origin='lower', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax,
                   extent=[0, GRID.Lx, 0, GRID.Ly])
    fig.colorbar(im, ax=ax, label=r'$\Delta n / n_0$')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(r'Plasma density modulation at end of run')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def residual_plot(linear, plasma, path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(linear['snap_steps'], np.maximum(linear['residuals'], 1e-17),
                'o-', label='linear vacuum')
    ax.semilogy(plasma['snap_steps'], np.maximum(plasma['residuals'], 1e-17),
                's-', label=f'plasma, n0 = {N0}')
    ax.set_xlabel('time step')
    ax.set_ylabel(r'$\|E_{AB} - E_A - E_B\|_\infty / \|E_{AB}\|_\infty$')
    ax.set_title('Wave-wave coupling: linearity residual vs time')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    print('Running linear vacuum baseline ...')
    lin = linearity_check(lambda: Material(eps_r=1.0, mu_r=1.0), 'linear')
    print('Running relativistic cold plasma (n0 = %.2f) ...' % N0)
    pla = linearity_check(lambda: RelativisticPlasmaMaterial(n0=N0), 'plasma')

    print('\n=== Linearity residual at sample steps ===')
    print(f'{"step":>6} {"linear":>14} {"plasma":>14}   ratio plasma/linear')
    for k, rl, rp in zip(lin['snap_steps'], lin['residuals'], pla['residuals']):
        ratio = rp / max(rl, 1e-18)
        print(f'{k:>6} {rl:>14.3e} {rp:>14.3e}   {ratio:>10.2e}')

    print('\n=== Plasma density excursion |Δn|/n0 (plasma AB run) ===')
    dm = pla['AB']['density_mods']
    for k in [0, len(dm)//4, len(dm)//2, 3*len(dm)//4, len(dm)-1]:
        print(f'  step {k:4d}    max|Δn|/n0 = {dm[k]:.3e}')

    out_dir = os.path.dirname(os.path.abspath(__file__))
    comparison_plot(lin, pla, os.path.join(out_dir, 'two_wave_plasma_compare.png'))
    residual_plot(lin, pla, os.path.join(out_dir, 'two_wave_plasma_residual.png'))
    density_plot(pla, os.path.join(out_dir, 'two_wave_plasma_density.png'))
    print(f'\nFigures saved to {out_dir}/two_wave_plasma_*.png')
