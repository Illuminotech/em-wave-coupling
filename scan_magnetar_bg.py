"""Scan the magnetar background field strength: test B_bg^2 scaling of the
QED Heisenberg-Euler wave-wave coupling enhancement.

Theory: in the F^2 channel, the cross-coupling polarization between two
waves on a static B background has the form (after subtracting single-wave
self-action)
    ΔP_cross ~ 4 κ × [c2 · B_bg^2 + c1 · B_bg + c0] × (waves^2)
where c0 absorbs the bare wave-wave coupling (wave^4 in L_NL),
c1 the linear-in-bg term, and c2 the quadratic-in-bg term. For weak
waves and modest B_bg, c2 dominates: residual(B_bg) ≈ R0 + α · B_bg^2.

This script runs the linearity check at B_bg ∈ {0, 0.5, 1, 2, 3} and fits
(residual − R0) vs B_bg^2 in log-log to extract the scaling exponent.
A clean slope of 2 confirms the theoretical channel; deviations indicate
that other terms (wave^3 × B_bg, geometric) are non-negligible at our
parameter regime.
"""

from __future__ import annotations
import os
import numpy as np

from fdtd2d import (FDTD2D, Grid, Material, HeisenbergEulerMaterial,
                    add_gaussian_packet, add_uniform_static_B)


WAVE_A = dict(wavelength=1.0,  angle_deg=20.0,  amplitude=0.10,
              waist=4.0, pulse_length=8.0, center=(4.0, 4.0))
WAVE_B = dict(wavelength=1.5,  angle_deg=-30.0, amplitude=0.07,
              waist=4.0, pulse_length=8.0, center=(4.0, 8.0))

GRID = Grid(nx=320, ny=240, dx=0.05, dy=0.05)
N_STEPS = 1000
KAPPA = 0.01

# B_bg values to scan. Finer grid for channel-resolution (item c).
# Stay below 4*kappa*B_bg^2 = 1 (Newton monotonicity bound), i.e. B_bg < 5 for kappa=0.01.
BG_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]


def run_residual(material_factory, B_bg: float) -> float:
    """Run A, B, AB at given B_bg and return the late-time linearity residual."""
    def fresh():
        sim = FDTD2D(grid=GRID, material=material_factory(), cfl=0.35, boundary='mur')
        if B_bg != 0.0:
            add_uniform_static_B(sim, B_bg, 0.0)
        return sim

    sA = fresh();  add_gaussian_packet(sA, **WAVE_A)
    sB = fresh();  add_gaussian_packet(sB, **WAVE_B)
    sAB = fresh(); add_gaussian_packet(sAB, **WAVE_A); add_gaussian_packet(sAB, **WAVE_B)

    for _ in range(N_STEPS):
        sA.step(); sB.step(); sAB.step()

    diff = sAB.Ez - sA.Ez - sB.Ez
    peak = max(np.max(np.abs(sAB.Ez)), 1e-12)
    return float(np.max(np.abs(diff)) / peak)


def fit_power_law(B_bg_values, residuals, R0, *, min_bg=0.0):
    """Fit (residual - R0) ~ alpha * B_bg^p in log-log, return (slope, alpha)."""
    bgs = np.asarray(B_bg_values, dtype=float)
    rs  = np.asarray(residuals, dtype=float) - R0
    mask = (bgs > min_bg) & (rs > 0)
    if mask.sum() < 2:
        return None, None
    x = np.log(bgs[mask])
    y = np.log(rs[mask])
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0]), float(np.exp(coeffs[1]))


def fit_two_channel(B_bg_values, residuals, R0):
    """Fit (residual - R0) = α B_bg + β B_bg²."""
    bgs = np.asarray(B_bg_values, dtype=float)
    rs  = np.asarray(residuals, dtype=float) - R0
    mask = bgs > 0
    A = np.column_stack([bgs[mask], bgs[mask] ** 2])
    coeffs, residuals_, *_ = np.linalg.lstsq(A, rs[mask], rcond=None)
    pred = A @ coeffs
    rss = float(np.sum((rs[mask] - pred) ** 2))
    return float(coeffs[0]), float(coeffs[1]), rss


def fit_three_channel(B_bg_values, residuals, R0):
    """Fit (residual - R0) = α B_bg + β B_bg² + γ B_bg³."""
    bgs = np.asarray(B_bg_values, dtype=float)
    rs  = np.asarray(residuals, dtype=float) - R0
    mask = bgs > 0
    A = np.column_stack([bgs[mask], bgs[mask] ** 2, bgs[mask] ** 3])
    coeffs, *_ = np.linalg.lstsq(A, rs[mask], rcond=None)
    pred = A @ coeffs
    rss = float(np.sum((rs[mask] - pred) ** 2))
    return float(coeffs[0]), float(coeffs[1]), float(coeffs[2]), rss


def fit_quadratic_only(B_bg_values, residuals, R0):
    """Fit (residual - R0) = β B_bg² alone (one parameter)."""
    bgs = np.asarray(B_bg_values, dtype=float)
    rs  = np.asarray(residuals, dtype=float) - R0
    mask = bgs > 0
    A = (bgs[mask] ** 2)[:, None]
    coeffs, *_ = np.linalg.lstsq(A, rs[mask], rcond=None)
    pred = A @ coeffs
    rss = float(np.sum((rs[mask] - pred) ** 2))
    return float(coeffs[0]), rss


def plot_scan(B_bg_values, residuals, R0, fit_results, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    bgs = np.asarray(B_bg_values, dtype=float)
    rs  = np.asarray(residuals, dtype=float)
    excess = rs - R0

    slope_all  = fit_results['power_all']
    slope_high = fit_results['power_high']
    alpha2, beta2 = fit_results['two_channel']
    alpha3, beta3, gamma3 = fit_results['three_channel']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Linear-axes plot with both fits
    ax = axes[0]
    ax.plot(bgs, rs, 'o', color='C2', markersize=9, label='measured', zorder=5)
    bg_fine = np.linspace(0, bgs.max() * 1.05, 200)
    pred_two = R0 + alpha2 * bg_fine + beta2 * bg_fine ** 2
    ax.plot(bg_fine, pred_two, '--', color='C0', alpha=0.8,
            label=fr'2-ch: $R_0 {alpha2:+.2e}\,B + {beta2:.2e}\,B^2$')
    pred_three = R0 + alpha3 * bg_fine + beta3 * bg_fine ** 2 + gamma3 * bg_fine ** 3
    ax.plot(bg_fine, pred_three, '-', color='C3', alpha=0.8,
            label=fr'3-ch: $R_0 {alpha3:+.2e}\,B {beta3:+.2e}\,B^2 {gamma3:+.2e}\,B^3$')
    ax.set_xlabel(r'$B_\mathrm{bg}$')
    ax.set_ylabel(r'$\|E_{AB}-E_A-E_B\|_\infty / \|E_{AB}\|_\infty$')
    ax.set_title('HE wave-wave coupling vs $B_\\mathrm{bg}$: 2-channel vs 3-channel fit')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8.5)

    # Log-log of (residual − R0)
    ax = axes[1]
    mask = (bgs > 0) & (excess > 0)
    ax.loglog(bgs[mask], excess[mask], 'o', color='C2', markersize=10,
              label='measured (excess)')
    bg_fine_log = np.geomspace(max(bgs[mask].min() * 0.9, 0.1), bgs[mask].max() * 1.1, 200)
    # 3-channel prediction
    pred3_log = (alpha3 * bg_fine_log + beta3 * bg_fine_log ** 2 + gamma3 * bg_fine_log ** 3)
    pred3_log = np.where(pred3_log > 0, pred3_log, np.nan)
    ax.loglog(bg_fine_log, pred3_log, '-', color='C3', alpha=0.8,
              label='3-channel fit (excess only)')
    # Slope-2 reference anchored at last point
    anchor2 = excess[mask][-1] / bgs[mask][-1] ** 2
    ax.loglog(bg_fine_log, anchor2 * bg_fine_log ** 2, ':', color='gray', alpha=0.6,
              label=r'slope 2 ref')
    anchor3 = excess[mask][-1] / bgs[mask][-1] ** 3
    ax.loglog(bg_fine_log, anchor3 * bg_fine_log ** 3, ':', color='black', alpha=0.6,
              label=r'slope 3 ref')
    ax.set_xlabel(r'$B_\mathrm{bg}$')
    ax.set_ylabel(r'residual $-R_0$')
    ax.set_title(fr'log-log: all-pts slope $={slope_all:.2f}$, $B\geq 1$ slope $={slope_high:.2f}$')
    ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    print(f'Scanning B_bg over {BG_VALUES}, fixed κ = {KAPPA}, predictor-corrector ON (2 iters)')
    residuals = []
    for bg in BG_VALUES:
        r = run_residual(
            lambda: HeisenbergEulerMaterial(kappa=KAPPA, pc_max_iters=2), bg)
        print(f'  B_bg = {bg:5.2f}    residual = {r:.4e}')
        residuals.append(r)

    R0 = residuals[0]   # at B_bg = 0, the bare wave^4 baseline.
    print(f'\nBaseline R_0 (at B_bg=0): {R0:.4e}')

    # Drop any NaN runs (Newton failure at very high B_bg).
    valid = [(bg, r) for bg, r in zip(BG_VALUES, residuals)
             if not np.isnan(r) and bg > 0]
    bg_v = [b for b, _ in valid]
    r_v  = [r for _, r in valid]

    slope_all, _ = fit_power_law(bg_v, r_v, R0)
    slope_high, _ = fit_power_law(bg_v, r_v, R0, min_bg=0.99)
    alpha2, beta2, rss_2ch = fit_two_channel(bg_v, r_v, R0)
    alpha3, beta3, gamma3, rss_3ch = fit_three_channel(bg_v, r_v, R0)
    beta_only, rss_quad = fit_quadratic_only(bg_v, r_v, R0)

    fit_results = {
        'power_all':  slope_all,
        'power_high': slope_high,
        'two_channel':  (alpha2, beta2),
        'three_channel': (alpha3, beta3, gamma3),
    }

    print('\n--- Single power-law fits ---')
    print(f'  All B_bg > 0: slope = {slope_all:.3f}  (mixed regime)')
    print(f'  B_bg ≥ 1.0:   slope = {slope_high:.3f}  (high-bg asymptotic)')
    print('\n--- Channel decompositions (residual − R_0 = ...) ---')
    n_pts = sum(1 for b in bg_v if b > 0)
    n_eff_quad = max(n_pts - 1, 1)
    n_eff_2ch  = max(n_pts - 2, 1)
    n_eff_3ch  = max(n_pts - 3, 1)
    print(f'  1-channel  β B²       : β = {beta_only:.3e}                                 '
          f'σ_resid = {np.sqrt(rss_quad/n_eff_quad):.2e}')
    print(f'  2-channel  α B + β B² : α = {alpha2:+.3e}, β = {beta2:.3e}                  '
          f'σ_resid = {np.sqrt(rss_2ch/n_eff_2ch):.2e}')
    print(f'  3-channel  α B + β B² + γ B³ : α = {alpha3:+.3e}, β = {beta3:+.3e}, γ = {gamma3:+.3e}  '
          f'σ_resid = {np.sqrt(rss_3ch/n_eff_3ch):.2e}')

    print('\n--- Channel contributions at B_bg = 2 ---')
    print(f'  2-channel: α·B = {alpha2*2:+.3e},  β·B² = {beta2*4:+.3e}')
    print(f'  3-channel: α·B = {alpha3*2:+.3e},  β·B² = {beta3*4:+.3e},  γ·B³ = {gamma3*8:+.3e}')

    # Significance: how much does adding γ reduce RSS?
    rss_reduction = (rss_2ch - rss_3ch) / max(rss_2ch, 1e-30)
    print(f'\n--- 3-channel vs 2-channel ---')
    print(f'  RSS reduction from adding γ B³ term: {rss_reduction*100:.1f}%')
    if rss_reduction > 0.5:
        print('  → γ B³ channel is statistically important; adding it improves fit by >50%.')
    elif rss_reduction > 0.1:
        print('  → γ B³ channel improves fit by 10-50%; borderline significance.')
    else:
        print('  → γ B³ channel gives <10% RSS reduction; likely fitting noise.')

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'magnetar_scaling.png')
    plot_scan(BG_VALUES, residuals, R0, fit_results, out)
    print(f'\nFigure saved to {out}')
