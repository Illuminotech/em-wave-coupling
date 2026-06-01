"""Spectral diagnostics across all four physics regimes.

Runs the same two-wave scenario in:
  (1) linear vacuum
  (2) Kerr chi^(3) medium
  (3) cold relativistic plasma
  (4) Heisenberg-Euler QED (F^2 channel)

Records E_z(t) at probe points (one in the wave overlap region, one outside),
computes power spectra, and overlays the predicted mixing peaks for each
regime. Quantitative test for thesis-level claims of wave-wave coupling
channels — the spectrum either shows the predicted peaks at the predicted
frequencies, or the model is wrong.

Wave placement note: previous drivers used WAVE_B center=(8, 22) which is
*outside* the Ly=12 grid, so wave B was effectively absent. Centers here
are corrected so that both packets are fully on-grid and their crossing
point is inside the domain.
"""

from __future__ import annotations
import os
import numpy as np

from fdtd2d import (FDTD2D, Grid, Material, KerrMaterial,
                    RelativisticPlasmaMaterial, HeisenbergEulerMaterial,
                    add_gaussian_packet)
from diagnostics import ProbeRecorder


# ---- Scenario --------------------------------------------------------------
# Both centers inside the 16 x 12 grid. Crossing computed analytically:
#   A_pos(t) = (4 + 0.940 t, 4 + 0.342 t)
#   B_pos(t) = (4 + 0.866 t, 8 - 0.500 t)
#   ⇒ y match at t ≈ 4.75, position (8.5, 5.6).
WAVE_A = dict(wavelength=1.0,  angle_deg=20.0,  amplitude=0.6,
              waist=4.0, pulse_length=8.0, center=(4.0, 4.0))
WAVE_B = dict(wavelength=1.5,  angle_deg=-30.0, amplitude=0.4,
              waist=4.0, pulse_length=8.0, center=(4.0, 8.0))

GRID = Grid(nx=320, ny=240, dx=0.05, dy=0.05)
N_STEPS = 1200          # longer than other drivers — needed for Δω resolution.

# Probe positions:
PROBE_OVERLAP = (8.5, 5.6)         # in the wave-crossing region
PROBE_REFERENCE = (3.5, 1.5)       # outside the crossing — sees only one wave (A)

# Material parameters (chosen modest enough for stability):
KERR_CHI3 = 0.10
PLASMA_N0 = 4.0          # ω_p = 2 in natural units
QED_KAPPA = 0.01

# Carrier and predicted mixing frequencies (natural units, c=1):
OMEGA_A = 2.0 * np.pi / WAVE_A['wavelength']
OMEGA_B = 2.0 * np.pi / WAVE_B['wavelength']
OMEGA_P = np.sqrt(PLASMA_N0)


def make_sim(material_factory) -> tuple[FDTD2D, ProbeRecorder]:
    """Build a sim with two-wave packets and probes attached."""
    sim = FDTD2D(grid=GRID, material=material_factory(), cfl=0.5)
    add_gaussian_packet(sim, **WAVE_A)
    add_gaussian_packet(sim, **WAVE_B)
    probe = ProbeRecorder(positions=[PROBE_OVERLAP, PROBE_REFERENCE], grid=GRID)
    sim.add_probe(probe)
    return sim, probe


def run_regime(label: str, material_factory) -> ProbeRecorder:
    print(f'  running {label} ...')
    sim, probe = make_sim(material_factory)
    for _ in range(N_STEPS):
        sim.step()
    return probe


# ---- Mixing peak predictions ----------------------------------------------
def annotate_peaks(ax, regime: str, ymax: float):
    """Draw vertical lines at predicted spectral peaks for `regime`."""
    peaks = []
    # Carrier always present:
    peaks.append((OMEGA_A,           'ω_A',  'C0', 1.0))
    peaks.append((OMEGA_B,           'ω_B',  'C1', 1.0))

    if regime == 'kerr' or regime == 'qed':
        # Third-order: 3ω, 2ω±ω cross-mixing
        peaks += [
            (3 * OMEGA_A,            '3ω_A', 'C2', 0.6),
            (3 * OMEGA_B,            '3ω_B', 'C2', 0.6),
            (2 * OMEGA_A - OMEGA_B,  '2ω_A−ω_B', 'C3', 0.7),
            (2 * OMEGA_B - OMEGA_A,  '2ω_B−ω_A', 'C3', 0.7),
            (2 * OMEGA_A + OMEGA_B,  '2ω_A+ω_B', 'C4', 0.5),
            (2 * OMEGA_B + OMEGA_A,  '2ω_B+ω_A', 'C4', 0.5),
        ]
    if regime == 'plasma':
        # Stokes / anti-Stokes sidebands at ω±ω_p, plus difference ω_A − ω_B
        peaks += [
            (OMEGA_A - OMEGA_P,      'ω_A−ω_p (Stokes)',     'C2', 0.7),
            (OMEGA_A + OMEGA_P,      'ω_A+ω_p (anti-Stokes)','C2', 0.7),
            (OMEGA_B - OMEGA_P,      'ω_B−ω_p',              'C3', 0.6),
            (OMEGA_B + OMEGA_P,      'ω_B+ω_p',              'C3', 0.6),
            (OMEGA_A - OMEGA_B,      'ω_A−ω_B',              'C4', 0.6),
        ]

    seen = set()
    for omega, label, color, alpha in peaks:
        if omega < 0 or omega > 0:
            key = round(omega, 3)
            if key in seen:
                continue
            seen.add(key)
            ax.axvline(omega, color=color, ls='--', alpha=alpha, lw=0.8)
            ax.text(omega, ymax * 0.7, label, rotation=90,
                    fontsize=7, va='top', ha='right',
                    color=color, alpha=min(1.0, alpha + 0.2))


def plot_spectra(probes: dict, path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    omega_max = 4 * max(OMEGA_A, OMEGA_B)   # show through 4ω

    for ax, (label, regime_key, probe) in zip(axes.flat, [
        ('Linear vacuum',          'linear', probes['linear']),
        ('Kerr (χ³ = %.2f)' % KERR_CHI3, 'kerr',   probes['kerr']),
        ('Plasma (n₀ = %.1f, ω_p = %.2f)' % (PLASMA_N0, OMEGA_P), 'plasma', probes['plasma']),
        ('QED Heisenberg–Euler (κ = %.3f)' % QED_KAPPA, 'qed', probes['qed']),
    ]):
        omega, power = probe.spectrum(0)            # probe 0 = overlap region
        omega_ref, power_ref = probe.spectrum(1)    # probe 1 = single-wave reference
        # Normalize by linear-baseline carrier peak height for cross-comparison.
        norm = max(power.max(), 1e-30)
        ax.semilogy(omega, np.maximum(power / norm, 1e-14),
                    'C0-', label='overlap-region probe')
        ax.semilogy(omega_ref, np.maximum(power_ref / norm, 1e-14),
                    'C7-', alpha=0.5, label='reference probe')
        annotate_peaks(ax, regime_key, ymax=1.0)
        ax.set_xlim(0, omega_max)
        ax.set_ylim(1e-12, 5)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel(r'normalized $|E_z(\omega)|^2$')
        ax.grid(True, which='both', alpha=0.3)
        if ax is axes[0, 0]:
            ax.legend(loc='lower right', fontsize=8)
    for ax in axes[1, :]:
        ax.set_xlabel(r'angular frequency $\omega$')
    fig.suptitle(
        'E_z(ω) at the overlap probe — predicted nonlinear mixing channels marked',
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def report_peak_amplitudes(probes: dict):
    """Quantitative summary: how strong is each predicted peak relative to ω_A?"""
    def peak_at(probe, omega_target, half_width=0.4):
        omega, power = probe.spectrum(0)
        mask = (omega > omega_target - half_width) & (omega < omega_target + half_width)
        if not np.any(mask):
            return 0.0
        return float(power[mask].max())

    print('\n=== Power at predicted mixing peaks (relative to ω_A peak) ===')
    print(f"{'regime':>10}  {'ω_A':>10}  {'ω_B':>10}  {'3ω_A':>10}  "
          f"{'2ω_A−ω_B':>10}  {'ω_A−ω_p':>10}  {'2ω_A+ω_B':>10}")
    for label, probe in probes.items():
        pA = peak_at(probe, OMEGA_A)
        pB = peak_at(probe, OMEGA_B)
        p3A = peak_at(probe, 3 * OMEGA_A)
        pFWM = peak_at(probe, 2 * OMEGA_A - OMEGA_B)
        pStokes = peak_at(probe, OMEGA_A - OMEGA_P)
        pSum = peak_at(probe, 2 * OMEGA_A + OMEGA_B)
        ref = max(pA, 1e-30)
        print(f'{label:>10}  '
              f'{pA / ref:>10.2e}  {pB / ref:>10.2e}  {p3A / ref:>10.2e}  '
              f'{pFWM / ref:>10.2e}  {pStokes / ref:>10.2e}  {pSum / ref:>10.2e}')


if __name__ == '__main__':
    print('Running each physics regime with probe recording ...')
    probes = {
        'linear': run_regime('linear vacuum', lambda: Material(eps_r=1.0)),
        'kerr':   run_regime('Kerr',          lambda: KerrMaterial(chi3=KERR_CHI3)),
        'plasma': run_regime('plasma',        lambda: RelativisticPlasmaMaterial(n0=PLASMA_N0)),
        'qed':    run_regime('QED HE',        lambda: HeisenbergEulerMaterial(kappa=QED_KAPPA)),
    }

    print('\nFrequencies of interest (natural units, c=1):')
    print(f'  ω_A  = {OMEGA_A:.3f}    ω_B   = {OMEGA_B:.3f}    ω_p = {OMEGA_P:.3f}')
    print(f'  3ω_A = {3*OMEGA_A:.3f}  2ω_A−ω_B = {2*OMEGA_A - OMEGA_B:.3f}')
    print(f'  ω_A−ω_p = {OMEGA_A - OMEGA_P:.3f}  ω_A+ω_p = {OMEGA_A + OMEGA_P:.3f}')

    report_peak_amplitudes(probes)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spectra_2x2.png')
    plot_spectra(probes, out)
    print(f'\nFigure saved to {out}')
