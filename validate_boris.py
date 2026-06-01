"""Validate the Boris pusher: a single electron in a uniform static B field
should rotate at the cyclotron frequency with energy exactly conserved.

Compares Boris (current implementation) against forward Euler (saved as a
historical baseline below). For Boris, |p|² is preserved to floating-point
precision over arbitrary number of cyclotron periods. For forward Euler,
the kinetic energy grows as (1 + (ω_c Δt)²)^N — a few percent per period
at our typical Δt.

Setup: a single particle in a homogeneous test environment (no FDTD field
arrays — we just iterate the pusher directly).

Initial condition: p_z = 1, B_x = 1, B_y = 0. Lorentz force on this electron
makes it rotate in the (y, z) plane at ω_c = |B| / γ.
"""

from __future__ import annotations
import numpy as np


def boris_step(px, py, pz, Ez, Bx, By, dt):
    """One Boris pusher iteration on a scalar particle (electron, mass=1, q=-1)."""
    half_E = 0.5 * dt * Ez
    pmx = px
    pmy = py
    pmz = pz - half_E

    gm = np.sqrt(1.0 + pmx ** 2 + pmy ** 2 + pmz ** 2)
    factor = -dt / (2.0 * gm)
    tx = factor * Bx
    ty = factor * By
    denom = 1.0 + tx ** 2 + ty ** 2
    sx = 2.0 * tx / denom
    sy = 2.0 * ty / denom

    ppx = pmx + (-pmz * ty)
    ppy = pmy + (pmz * tx)
    ppz = pmz + (pmx * ty - pmy * tx)

    pplus_x = pmx + (-ppz * sy)
    pplus_y = pmy + (ppz * sx)
    pplus_z = pmz + (ppx * sy - ppy * sx)

    return pplus_x, pplus_y, pplus_z - half_E


def euler_step(px, py, pz, Ez, Bx, By, dt):
    """One forward-Euler iteration (the previous implementation)."""
    g = np.sqrt(1.0 + px ** 2 + py ** 2 + pz ** 2)
    vx, vy, vz = px / g, py / g, pz / g
    Fx = vz * By
    Fy = -vz * Bx
    Fz = -(Ez + vx * By - vy * Bx)
    return px + dt * Fx, py + dt * Fy, pz + dt * Fz


def run(stepper, p0, Ez, Bx, By, n_steps, dt):
    px, py, pz = p0
    energies = []
    for _ in range(n_steps):
        energies.append(np.sqrt(1.0 + px ** 2 + py ** 2 + pz ** 2) - 1.0)
        px, py, pz = stepper(px, py, pz, Ez, Bx, By, dt)
    return np.asarray(energies)


def main():
    Bx, By = 1.0, 0.0
    Ez = 0.0
    p0 = (0.0, 0.0, 1.0)            # initial p_z = 1, |p| = 1, γ = √2

    initial_KE = np.sqrt(1.0 + p0[0]**2 + p0[1]**2 + p0[2]**2) - 1.0   # γ−1

    omega_c = np.linalg.norm([Bx, By]) / np.sqrt(2.0)   # = 1/√2
    period = 2 * np.pi / omega_c
    dt = period / 50                                    # 50 steps per cyclotron period
    n_periods = 200
    n_steps = int(n_periods * period / dt)

    print(f'Cyclotron period: T_c = {period:.4f}')
    print(f'Δt = {dt:.4f}  →  {n_steps} steps over {n_periods} periods')
    print(f'Initial KE = γ−1 = {initial_KE:.6f}')

    KE_boris = run(boris_step, p0, Ez, Bx, By, n_steps, dt)
    KE_euler = run(euler_step, p0, Ez, Bx, By, n_steps, dt)

    print(f'\n{"":>20} {"final KE":>14} {"rel. drift":>14} {"theory":>10}')
    boris_drift = (KE_boris[-1] - initial_KE) / initial_KE
    euler_drift = (KE_euler[-1] - initial_KE) / initial_KE
    # Forward Euler kinetic-energy growth per step is (1 + (ω_c Δt)²) on the
    # rotating component. After N steps: (1 + (ω_c Δt)²)^N.
    euler_theory = (1.0 + (omega_c * dt) ** 2) ** n_steps - 1.0
    print(f'{"Boris":>20} {KE_boris[-1]:>14.6e} {boris_drift:>+14.3e} {0.0:>10.2e}')
    print(f'{"forward Euler":>20} {KE_euler[-1]:>14.6e} {euler_drift:>+14.3e} {euler_theory:>10.2e}')

    if abs(boris_drift) < 1e-10:
        print('\n✓ Boris preserves energy to floating-point precision.')
    else:
        print(f'\n⚠ Boris drift {boris_drift:.2e} larger than expected.')

    # Save plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        ts = np.arange(n_steps) * dt
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(ts / period, KE_boris / initial_KE - 1, label='Boris', color='C2')
        ax.plot(ts / period, KE_euler / initial_KE - 1, label='forward Euler', color='C3')
        ax.set_xlabel('cyclotron periods')
        ax.set_ylabel(r'$(KE - KE_0) / KE_0$')
        ax.set_title('Numerical heating: Boris vs forward Euler in static B field')
        ax.set_yscale('symlog', linthresh=1e-12)
        ax.grid(True, which='both', alpha=0.3); ax.legend()
        fig.tight_layout()
        path = __file__.replace('.py', '_plot.png')
        fig.savefig(path, dpi=120)
        print(f'\nFigure saved to {path}')
    except ImportError:
        pass


if __name__ == '__main__':
    main()
