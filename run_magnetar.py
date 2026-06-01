"""Magnetar-binary scenario: wave-wave QED coupling in a strong static B background.

Physical setup, conceptual: two magnetars in close orbit. In the overlap
region between their dipole fields, the local B can approach ~B_Schwinger.
EM waves (e.g. radio emission from each pulsar, or beat-frequency
modulations) propagate through this strong-field region. The QED
Heisenberg-Euler nonlinearity gives wave-wave coupling that is *enhanced*
by the background field — schematically, the cross-coupling polarization
gains a term ~ -8 κ B_bg · (E_A B_B + E_B B_A) which is only second-order
in wave amplitudes but first-order in B_bg, so it dominates the bare
vacuum HE coupling (third-order in wave amplitudes) when E_wave ≪ B_bg.

This driver runs 4 conditions:
    - linear vacuum, no B_bg
    - linear vacuum, with B_bg
    - QED HE, no B_bg
    - QED HE, with B_bg
and compares the linearity residuals ||AB - A - B|| / ||AB||. Expectation:
the QED+B_bg residual exceeds QED-no-B_bg by roughly the factor
(B_bg / E_wave)² — a quantitative test of the background-enhancement.
"""

from __future__ import annotations
import os
import numpy as np

from fdtd2d import (FDTD2D, Grid, Material, HeisenbergEulerMaterial,
                    add_gaussian_packet, add_uniform_static_B)
from diagnostics import ProbeRecorder


# ---- Physical interpretation -----------------------------------------------
# We work in dimensionless natural units (c = ε₀ = μ₀ = m_e = e = 1).
# To re-interpret in SI for a magnetar context, fix:
#   - 1 grid length ~ 1 km    (so dx = 0.05 km = 50 m, λ = 1 km radio wave)
#   - 1 unit of B          ~ B_Schwinger ≈ 4.4 × 10⁹ T
# Then B_bg = 1 in our units = magnetar inner-magnetosphere field strength,
# and our wave amplitudes E_wave ~ 0.1 correspond to coherent radio waves
# with peak fields ~ 0.1 × m_e c ω / e — i.e. modest radio waves.
# The dimensionless QED coefficient κ encodes the (α²/m_e^4) prefactor and
# is dialed to ~0.01 here so the cubic constitutive stays well-conditioned.

# ---- Scenario --------------------------------------------------------------
# Two crossing waves both fully on-grid (corrected from earlier driver bug).
# Crossing computed: A_pos(t) = (4 + 0.94 t, 4 + 0.34 t),
# B_pos(t) = (4 + 0.87 t, 8 - 0.5 t) ⇒ y match at t ≈ 4.75, position (8.5, 5.6).
WAVE_A = dict(wavelength=1.0,  angle_deg=20.0,  amplitude=0.10,
              waist=4.0, pulse_length=8.0, center=(4.0, 4.0))
WAVE_B = dict(wavelength=1.5,  angle_deg=-30.0, amplitude=0.07,
              waist=4.0, pulse_length=8.0, center=(4.0, 8.0))

GRID = Grid(nx=320, ny=240, dx=0.05, dy=0.05)
N_STEPS = 1000
SNAPSHOT_EVERY = 100
KAPPA = 0.01            # QED HE coefficient
B_BG = (1.0, 0.0)       # Static background field in (x, y) plane


def make_sim(material_factory, with_bg: bool) -> tuple[FDTD2D, ProbeRecorder]:
    sim = FDTD2D(grid=GRID, material=material_factory(), cfl=0.35)
    if with_bg:
        add_uniform_static_B(sim, *B_BG)
    return sim


def run_pair_AB(material_factory, with_bg: bool):
    """Run A-alone, B-alone, AB and return the linearity residual time series."""
    snaps = list(range(0, N_STEPS + 1, SNAPSHOT_EVERY))

    def fresh():
        return make_sim(material_factory, with_bg)

    sA = fresh();  add_gaussian_packet(sA, **WAVE_A)
    sB = fresh();  add_gaussian_packet(sB, **WAVE_B)
    sAB = fresh(); add_gaussian_packet(sAB, **WAVE_A); add_gaussian_packet(sAB, **WAVE_B)

    snapshots_A, snapshots_B, snapshots_AB = {}, {}, {}
    for n in range(N_STEPS + 1):
        if n in set(snaps):
            snapshots_A[n] = sA.Ez.copy()
            snapshots_B[n] = sB.Ez.copy()
            snapshots_AB[n] = sAB.Ez.copy()
        if n < N_STEPS:
            sA.step(); sB.step(); sAB.step()

    residuals = []
    for k in snaps:
        diff = snapshots_AB[k] - snapshots_A[k] - snapshots_B[k]
        # Normalize by the peak deviation from background (subtract the
        # static B_bg's tiny contribution to E if any). For E_z this is 0
        # since B_bg is in (x,y) plane only and Ez stays at 0 outside waves.
        peak_AB = max(np.max(np.abs(snapshots_AB[k])), 1e-12)
        residuals.append(np.max(np.abs(diff)) / peak_AB)
    return np.array(snaps), np.array(residuals)


def residual_plot(results: dict, path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    style = {
        'linear, no B_bg':  ('o-',  'C0', 1.0),
        'linear, with B_bg': ('o--', 'C0', 0.5),
        'QED, no B_bg':     ('s-',  'C3', 1.0),
        'QED, with B_bg':   ('s-',  'C2', 1.0),
    }
    for label, (steps, resid) in results.items():
        marker, color, alpha = style[label]
        ax.semilogy(steps, np.maximum(resid, 1e-17),
                    marker, color=color, alpha=alpha, label=label)
    ax.set_xlabel('time step')
    ax.set_ylabel(r'$\|E_{AB} - E_A - E_B\|_\infty / \|E_{AB}\|_\infty$')
    ax.set_title('Wave–wave coupling: vacuum vs magnetar-scale B background')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def report(results: dict):
    print('\n=== Late-time linearity residuals (final step) ===')
    print(f'{"condition":>22}    final residual    enhancement')
    base_qed = results['QED, no B_bg'][1][-1]
    for label, (steps, resid) in results.items():
        ratio = resid[-1] / max(base_qed, 1e-30)
        print(f'{label:>22}    {resid[-1]:>14.3e}    {ratio:>10.2e}')

    rho_bg_over_no = (results['QED, with B_bg'][1][-1]
                      / max(results['QED, no B_bg'][1][-1], 1e-30))
    expected = (B_BG[0] ** 2 + B_BG[1] ** 2) / (WAVE_A['amplitude'] * WAVE_B['amplitude'])
    print(f'\nB_bg-enhancement (QED with-bg / QED no-bg): {rho_bg_over_no:.2f}')
    print(f'Theory expectation ~ (B_bg / E_A E_B)^(0..1) ≈ {expected:.1f} (B_bg dominant)'
          f'  /  1 (waves dominant)')


if __name__ == '__main__':
    print('Running 4 scenarios (linear/QED × no-bg/with-bg) ...')
    results = {
        'linear, no B_bg':
            run_pair_AB(lambda: Material(eps_r=1.0), with_bg=False),
        'linear, with B_bg':
            run_pair_AB(lambda: Material(eps_r=1.0), with_bg=True),
        'QED, no B_bg':
            run_pair_AB(lambda: HeisenbergEulerMaterial(kappa=KAPPA), with_bg=False),
        'QED, with B_bg':
            run_pair_AB(lambda: HeisenbergEulerMaterial(kappa=KAPPA), with_bg=True),
    }

    report(results)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'magnetar_residuals.png')
    residual_plot(results, out)
    print(f'\nFigure saved to {out}')
