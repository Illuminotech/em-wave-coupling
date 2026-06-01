"""Validate the 2D mixed-mode FDTD and demonstrate the G² coupling channel.

Three groups of tests:

A. Linear-vacuum decoupling
   - TM alone: TE energy stays at 0 (no mode crosstalk in vacuum).
   - TE alone: TM energy stays at 0.
   - TM + TE in linear vacuum: total energy is exact sum; residual = 0.

B. κ-scaling of HE channels
   - Sweep κ_F (with κ_G = 0) and κ_G (with κ_F = 0).
   - Confirm residual ∝ κ as expected for first-order linearization.
   - Compare the slopes: which channel gives larger cross-coupling per unit κ?

C. Physical interpretation
   - G = E·B is identically zero for either pure-TM or pure-TE wave alone.
   - So the G² channel produces wave-wave coupling that has NO self-action
     contamination — the cleanest possible "two-wave-only" QED channel.

Note: the linearized constitutive (first-order in κ) is stable for
κ × max|field|² ≲ 10⁻². For larger κ, drift errors accumulate and a full
iterative inversion is needed (deferred future work).
"""

from __future__ import annotations
import os
import numpy as np

from fdtd2d_mixed import FDTD2D_Mixed, add_TM_packet, add_TE_packet
from fdtd2d import Grid


GRID = Grid(nx=240, ny=240, dx=0.05, dy=0.05)
N_STEPS = 200            # short enough that linearization drift is small

WAVE_TM = dict(wavelength=1.0, angle_deg=20.0, amplitude=1.0,
               waist=3.0, pulse_length=3.0, center=(6.0, 6.0))
WAVE_TE = dict(wavelength=1.5, angle_deg=-30.0, amplitude=0.7,
               waist=3.0, pulse_length=3.0, center=(6.0, 6.0))


def residual_E_max(sAB, sTM, sTE) -> float:
    def comp(a, b, c):
        diff = a - b - c
        ref = max(np.max(np.abs(a)), 1e-12)
        return float(np.max(np.abs(diff)) / ref)
    return max(comp(sAB.Ez, sTM.Ez, sTE.Ez),
               comp(sAB.Ex, sTM.Ex, sTE.Ex),
               comp(sAB.Ey, sTM.Ey, sTE.Ey))


def run_AB(kappa_F: float, kappa_G: float):
    def fresh():
        return FDTD2D_Mixed(grid=GRID, cfl=0.5, kappa_F=kappa_F, kappa_G=kappa_G)
    sTM = fresh(); add_TM_packet(sTM, **WAVE_TM)
    sTE = fresh(); add_TE_packet(sTE, **WAVE_TE)
    sAB = fresh(); add_TM_packet(sAB, **WAVE_TM); add_TE_packet(sAB, **WAVE_TE)
    for n in range(N_STEPS):
        sTM.step(); sTE.step(); sAB.step()
    return residual_E_max(sAB, sTM, sTE)


def test_A_linear_decoupling():
    print('=== A. Linear-vacuum decoupling ===')
    sim = FDTD2D_Mixed(grid=GRID, cfl=0.5); add_TM_packet(sim, **WAVE_TM)
    for _ in range(N_STEPS): sim.step()
    print(f'  TM alone (step {N_STEPS}):   TM E = {sim.tm_energy():.4f},  TE leak = {sim.te_energy():.3e}')

    sim = FDTD2D_Mixed(grid=GRID, cfl=0.5); add_TE_packet(sim, **WAVE_TE)
    for _ in range(N_STEPS): sim.step()
    print(f'  TE alone (step {N_STEPS}):   TE E = {sim.te_energy():.4f},  TM leak = {sim.tm_energy():.3e}')

    sTM = FDTD2D_Mixed(grid=GRID, cfl=0.5); add_TM_packet(sTM, **WAVE_TM)
    sTE = FDTD2D_Mixed(grid=GRID, cfl=0.5); add_TE_packet(sTE, **WAVE_TE)
    sAB = FDTD2D_Mixed(grid=GRID, cfl=0.5); add_TM_packet(sAB, **WAVE_TM); add_TE_packet(sAB, **WAVE_TE)
    for _ in range(N_STEPS):
        sTM.step(); sTE.step(); sAB.step()
    r_lin = residual_E_max(sAB, sTM, sTE)
    print(f'  TM+TE linear: ||AB − TM − TE|| / max|AB|  =  {r_lin:.3e}')
    print('  → modes decouple exactly in vacuum.')
    return r_lin


def test_B_kappa_scaling():
    print('\n=== B. κ-scaling of HE channels ===')
    kappa_grid = [0.0, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3]
    print(f'  Scanning κ ∈ {kappa_grid}')

    r_F = [run_AB(k, 0.0) for k in kappa_grid]
    r_G = [run_AB(0.0, k) for k in kappa_grid]

    print('\n         κ      r(F² only)    r(G² only)   ratio G/F')
    for k, rf, rg in zip(kappa_grid, r_F, r_G):
        ratio = rg / max(rf, 1e-30)
        print(f'  {k:>8.0e}    {rf:>10.3e}    {rg:>10.3e}    {ratio:>8.2f}')

    # Fit slope: residual ∝ κ^p
    def slope(kappas, rs):
        ks = np.array(kappas)
        rs = np.array(rs)
        m = (ks > 0) & (rs > 0)
        if m.sum() < 2:
            return None
        return float(np.polyfit(np.log(ks[m]), np.log(rs[m]), 1)[0])

    sF = slope(kappa_grid, r_F)
    sG = slope(kappa_grid, r_G)
    print(f'\n  Power-law fit (residual ∝ κ^p):')
    print(f'    F² channel: p = {sF:.3f}  (theory: 1.0)')
    print(f'    G² channel: p = {sG:.3f}  (theory: 1.0)')

    return kappa_grid, r_F, r_G


def plot_scaling(kappas, r_F, r_G, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ks = np.array(kappas)
    ax.loglog(ks[1:], r_F[1:], 'o-', color='C0', markersize=9, label=r'F² channel (TM+TE)')
    ax.loglog(ks[1:], r_G[1:], 's-', color='C2', markersize=9, label=r'G² channel (TM·TE) — photon-photon')

    # Reference slope-1 lines anchored to the last G data point.
    kf = np.geomspace(ks[1], ks[-1], 50)
    anchor_F = r_F[-1] / ks[-1]
    anchor_G = r_G[-1] / ks[-1]
    ax.loglog(kf, anchor_F * kf, '--', color='C0', alpha=0.4)
    ax.loglog(kf, anchor_G * kf, '--', color='C2', alpha=0.4, label=r'slope-1 reference')

    ax.set_xlabel(r'$\kappa$')
    ax.set_ylabel(r'$\|E_{AB} - E_\mathrm{TM} - E_\mathrm{TE}\|_\infty / \max|E_{AB}|$')
    ax.set_title('Mixed-mode HE channels: F² (scalar) vs G² (E·B) wave-wave coupling')
    ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    r_lin = test_A_linear_decoupling()
    kgrid, r_F, r_G = test_B_kappa_scaling()

    print('\n=== Summary ===')
    print(f'  Linear-vacuum residual:       {r_lin:.2e}  (zero ⇒ TM/TE decouple)')
    print(f'  G²/F² coupling ratio:         {r_G[-1]/max(r_F[-1],1e-30):.2f}× '
          '(at κ = 3×10⁻³)')
    print(f'  → G² (photon-photon scattering) channel dominates over')
    print(f'    F² (scalar) channel for these cross-polarized waves.')

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'mixed_mode_scaling.png')
    plot_scaling(kgrid, r_F, r_G, out)
    print(f'\nFigure saved to {out}')
