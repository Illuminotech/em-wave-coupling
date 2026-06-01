"""Step 4: two crossing wave packets in QED-corrected vacuum (Heisenberg-Euler).

The F^2 channel of the one-loop effective Lagrangian gives an effective
constitutive relation
    D = (1 + chi) E,   H = (1 + chi) B,   chi = 4 kappa (E^2 - B^2).

For a single plane wave |E| = |B| ⇒ chi = 0 ⇒ no self-action. The wave-wave
coupling in this channel exists *purely* in the overlap region of multiple
waves and is identically zero for either wave alone.

This is a stronger statement than the linearity-residual ratios we saw for
Kerr or plasma: in those, A alone and B alone each saw nonlinear self-action
(self-phase modulation, ponderomotive density modulation), and the "AB - A - B"
residual measured the cross-coupling against a nonzero self-action background.
Here, A alone = A_linear, B alone = B_linear, and only AB is nonlinear at all.
The residual therefore directly measures the QED interaction with no
subtraction needed.
"""

from __future__ import annotations
import os
import numpy as np

from fdtd2d import (FDTD2D, Grid, Material, HeisenbergEulerMaterial,
                    add_gaussian_packet)


# Same scenario as Steps 1, 2, 3.
WAVE_A = dict(wavelength=1.0,  angle_deg=20.0,  amplitude=1.0,
              waist=4.0, pulse_length=4.0, center=(8.0,  8.0))
WAVE_B = dict(wavelength=1.5,  angle_deg=-30.0, amplitude=0.7,
              waist=4.0, pulse_length=4.0, center=(8.0, 22.0))

GRID = Grid(nx=320, ny=240, dx=0.05, dy=0.05)
N_STEPS = 600
SNAPSHOT_EVERY = 50

# QED prefactor in natural units. In real QED kappa = 2 alpha^2 / (45 m_e^4)
# and is utterly negligible until E approaches the Schwinger field
# E_S ≈ 1.32e18 V/m. Here we dial it up so the effect is visible at unit
# amplitudes — it's a knob to study the QED *channel*, not realistic field
# strengths. Adjust as needed.
KAPPA = 0.01


def make_sim(material: Material) -> FDTD2D:
    # Lower CFL than linear/Kerr/plasma drivers: in QED HE the cubic constitutive
    # is sensitive to the time-centering error of B² (~ ω·dt), which can drive a
    # spurious resonant self-action even though theory says F = 0 for a single
    # wave. Halving dt reduces the per-step artifact roughly quadratically.
    return FDTD2D(grid=GRID, material=material, cfl=0.35)


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


def single_wave_self_check(material_factory):
    """Verify the no-self-action property: A-alone in QED == A-alone in linear."""
    snaps = list(range(0, N_STEPS + 1, SNAPSHOT_EVERY))
    sA_lin = make_sim(Material(eps_r=1.0));         add_gaussian_packet(sA_lin, **WAVE_A)
    sA_qed = make_sim(material_factory());           add_gaussian_packet(sA_qed, **WAVE_A)
    rA_lin = run(sA_lin, N_STEPS, snaps)
    rA_qed = run(sA_qed, N_STEPS, snaps)
    deviations = []
    for k in snaps:
        diff = rA_qed['snapshots'][k] - rA_lin['snapshots'][k]
        denom = max(np.max(np.abs(rA_lin['snapshots'][k])), 1e-12)
        deviations.append(np.max(np.abs(diff)) / denom)
    return snaps, np.array(deviations)


def comparison_plot(linear, qed, path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pick = linear['snap_steps'][:: max(1, len(linear['snap_steps']) // 4)][:4]
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    vmax = max(np.max(np.abs(linear['AB']['snapshots'][s])) for s in pick)

    for j, s in enumerate(pick):
        L = linear['AB']['snapshots'][s]
        Q = qed['AB']['snapshots'][s]
        diff = Q - L
        for i, (img, title, vm) in enumerate([
            (L, f'linear vacuum, step {s}', vmax),
            (Q, f'HE QED (κ={KAPPA}), step {s}', vmax),
            (diff, f'QED − linear, step {s}', max(np.max(np.abs(diff)), 1e-12)),
        ]):
            ax = axes[i, j]
            ax.imshow(img.T, origin='lower', cmap='RdBu_r',
                      vmin=-vm, vmax=vm,
                      extent=[0, GRID.Lx, 0, GRID.Ly])
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Two-wave interaction: linear vacuum vs Heisenberg–Euler QED')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def residual_plot(linear, qed, single_devs, path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(linear['snap_steps'], np.maximum(linear['residuals'], 1e-17),
                'o-', label='linear vacuum: ||AB − A − B||')
    ax.semilogy(qed['snap_steps'], np.maximum(qed['residuals'], 1e-17),
                's-', label=f'QED HE κ={KAPPA}: ||AB − A − B||')
    ax.semilogy(single_devs[0], np.maximum(single_devs[1], 1e-17),
                '^--', alpha=0.7,
                label='QED single-wave self-action: ||A_QED − A_lin||')
    ax.set_xlabel('time step')
    ax.set_ylabel('relative residual')
    ax.set_title('QED nonlinearity: visible only when waves overlap')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def angle_scan_plot(material_factory, path: str):
    """Scan crossing angle Δθ. Two competing effects:
      - Local coupling strength: F = (E²-B²)/2 ∝ (1 - cos Δθ) E_A E_B
      - Overlap duration: small Δθ → packets travel together longer.
    For finite-size packets, the overlap time often dominates and the
    residual *decreases* with Δθ — opposite of plane-wave intuition.
    For beam-like geometries with the wave envelope width << wavelength
    of intersection scale, the (1 - cos Δθ) factor would re-emerge.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    angles = [10, 30, 60, 90, 120, 150]
    results = []
    for delta in angles:
        wa = dict(WAVE_A); wa['angle_deg'] = +delta / 2.0
        wb = dict(WAVE_B); wb['angle_deg'] = -delta / 2.0
        sA = make_sim(material_factory());  add_gaussian_packet(sA, **wa)
        sB = make_sim(material_factory());  add_gaussian_packet(sB, **wb)
        sAB = make_sim(material_factory()); add_gaussian_packet(sAB, **wa); add_gaussian_packet(sAB, **wb)
        rA = run(sA, 250, [250])
        rB = run(sB, 250, [250])
        rAB = run(sAB, 250, [250])
        diff = rAB['snapshots'][250] - rA['snapshots'][250] - rB['snapshots'][250]
        denom = max(np.max(np.abs(rAB['snapshots'][250])), 1e-12)
        results.append(np.max(np.abs(diff)) / denom)
        print(f'  Δθ = {delta:>4}°    residual = {results[-1]:.3e}')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(angles, results, 'o-', label='measured residual')
    theta_rad = np.deg2rad(np.array(angles))
    pred = (1 - np.cos(theta_rad))
    pred = pred * (results[-1] / pred[-1])  # rescale to align
    ax.plot(angles, pred, 'k--', alpha=0.5, label=r'$\propto (1 - \cos\Delta\theta)$')
    ax.set_xlabel(r'crossing angle $\Delta\theta$ (deg)')
    ax.set_ylabel('linearity residual at step 250')
    ax.set_title('QED wave–wave coupling: angular dependence')
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return angles, results


if __name__ == '__main__':
    print('Running linear vacuum baseline ...')
    lin = linearity_check(lambda: Material(eps_r=1.0, mu_r=1.0), 'linear')
    print(f'Running Heisenberg–Euler QED (κ = {KAPPA}) ...')
    qed = linearity_check(lambda: HeisenbergEulerMaterial(kappa=KAPPA), 'qed')

    print('Verifying no-self-action: single wave in QED vs linear ...')
    single_steps, single_devs = single_wave_self_check(lambda: HeisenbergEulerMaterial(kappa=KAPPA))

    print('\n=== Residuals ===')
    print(f'{"step":>6} {"||AB−A−B|| lin":>18} {"||AB−A−B|| QED":>18}'
          f' {"||A_qed−A_lin||":>18}')
    for k, rl, rq, sd in zip(lin['snap_steps'], lin['residuals'],
                              qed['residuals'], single_devs):
        print(f'{k:>6} {rl:>18.3e} {rq:>18.3e} {sd:>18.3e}')

    out_dir = os.path.dirname(os.path.abspath(__file__))
    comparison_plot(lin, qed, os.path.join(out_dir, 'two_wave_qed_compare.png'))
    residual_plot(lin, qed, (single_steps, single_devs),
                  os.path.join(out_dir, 'two_wave_qed_residual.png'))

    print('\nScanning crossing angle (key signature of HE wave–wave coupling) ...')
    angle_scan_plot(lambda: HeisenbergEulerMaterial(kappa=KAPPA),
                    os.path.join(out_dir, 'two_wave_qed_angle_scan.png'))
    print(f'\nFigures saved to {out_dir}/two_wave_qed_*.png')
