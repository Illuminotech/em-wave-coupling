"""Controlled tests to determine whether the observed energy growth is:
  (1) a real FDTD core bug,
  (2) an artifact of our naive |Ez|² + |Hx|² + |Hy|² energy measure not capturing
      the conserved discrete Yee energy under E/H time staggering,
  (3) a boundary-handling issue separate from CPML,
  (4) a CPML-specific bug.

We run four progressively more demanding tests and compare against analytic
expectations.
"""

from __future__ import annotations
import numpy as np
from fdtd2d import FDTD2D, Grid, Material, add_gaussian_packet


def naive_energy(sim) -> float:
    """The same |Ez|² + |Hx|² + |Hy|² sum I've been using."""
    return 0.5 * (np.sum(sim.Ez ** 2) +
                  np.sum(sim.Hx ** 2) +
                  np.sum(sim.Hy ** 2)) * sim.g.dx * sim.g.dy


def yee_energy(sim) -> float:
    """Discrete Yee energy: interpolate Hx, Hy to Ez positions and sum at Ez locations.

    For 2D TMz, the spatially-collocated energy density is
        u(i,j) = 0.5 (Ez[i,j]² + <Hx²>[i,j] + <Hy²>[i,j])
    where <Hx²>[i,j] = 0.5 (Hx[i,j]² + Hx[i,j-1]²) interpolates Hx from
    (i, j±½) to (i, j) integer. This is the proper energy density in a
    Yee grid that's strictly bounded (not exactly conserved over a step
    because E^n and H^{n+½} are at different times, but the time-averaged
    quantity over a wave period is conserved).
    """
    nx, ny = sim.g.nx, sim.g.ny
    # Interpolate Hx² to integer (i, j):
    Hx2_interp = np.zeros((nx, ny))
    Hx2_interp[:, 1:-1] = 0.5 * (sim.Hx[:, :-1] ** 2 + sim.Hx[:, 1:] ** 2)
    # Interpolate Hy² to integer (i, j):
    Hy2_interp = np.zeros((nx, ny))
    Hy2_interp[1:-1, :] = 0.5 * (sim.Hy[:-1, :] ** 2 + sim.Hy[1:, :] ** 2)
    u = 0.5 * (sim.Ez ** 2 + Hx2_interp + Hy2_interp)
    return float(np.sum(u) * sim.g.dx * sim.g.dy)


def init_pec_eigenmode(sim, m: int, n: int):
    """Initialize an exact PEC-cavity eigenmode E_z = sin(k_x x) sin(k_y y) cos(ωt).

    With H staggered at t = -dt/2 for our IC convention.
    """
    g = sim.g
    kx = m * np.pi / g.Lx
    ky = n * np.pi / g.Ly
    omega = np.sqrt(kx ** 2 + ky ** 2)
    # E_z at t = 0
    x_E = np.arange(g.nx) * g.dx
    y_E = np.arange(g.ny) * g.dy
    X_E, Y_E = np.meshgrid(x_E, y_E, indexing='ij')
    sim.Ez += np.sin(kx * X_E) * np.sin(ky * Y_E)
    # H_x at (i, j+½), t = -dt/2:
    # ∂_t H_x = -∂_y E_z ⇒ H_x = -(k_y/ω) sin(k_x x) cos(k_y y) sin(ω t)
    y_H = (np.arange(g.ny - 1) + 0.5) * g.dy
    Xh, Yh = np.meshgrid(x_E, y_H, indexing='ij')
    sim.Hx += -(ky / omega) * np.sin(kx * Xh) * np.cos(ky * Yh) * np.sin(omega * (-sim.dt / 2))
    # H_y at (i+½, j), t = -dt/2:
    # H_y = +(k_x/ω) cos(k_x x) sin(k_y y) sin(ω t)
    x_H = (np.arange(g.nx - 1) + 0.5) * g.dx
    Xh2, Yh2 = np.meshgrid(x_H, y_E, indexing='ij')
    sim.Hy += (kx / omega) * np.cos(kx * Xh2) * np.sin(ky * Yh2) * np.sin(omega * (-sim.dt / 2))
    return omega


GRID = Grid(nx=200, ny=200, dx=0.05, dy=0.05)


# ---------------------------------------------------------------------------
def test1_zero_IC():
    print('\n--- Test 1: zero IC, Mur boundary. Energy must stay at 0.')
    sim = FDTD2D(grid=GRID, material=Material(eps_r=1.0), cfl=0.5, boundary='mur')
    for n in range(0, 401, 100):
        e_naive = naive_energy(sim)
        e_yee = yee_energy(sim)
        print(f'  step {n:4d}: naive E = {e_naive:.3e},  Yee E = {e_yee:.3e}')
        for _ in range(100):
            sim.step()


def test2_pec_eigenmode():
    print('\n--- Test 2: PEC eigenmode (m=4, n=3). Both energies should be bounded.')
    print('  Analytic total energy = Lx Ly / 8 = {:.3f}'.format(GRID.Lx * GRID.Ly / 8))
    sim = FDTD2D(grid=GRID, material=Material(eps_r=1.0), cfl=0.5, boundary='mur')
    omega = init_pec_eigenmode(sim, m=4, n=3)
    period_steps = int(2 * np.pi / omega / sim.dt)
    print(f'  cyclotron period = {period_steps} steps; running 5 periods')
    n_steps = 5 * period_steps
    snapshots = list(range(0, n_steps + 1, period_steps // 2))
    for n in range(n_steps + 1):
        if n in snapshots:
            e_naive = naive_energy(sim)
            e_yee = yee_energy(sim)
            print(f'  step {n:5d} (t={n*sim.dt:.3f}): naive E = {e_naive:.4f},  '
                  f'Yee E = {e_yee:.4f}')
        if n < n_steps:
            sim.step()


def test3_packet_no_boundary():
    print('\n--- Test 3: Gaussian packet, no boundary handling (implicit PEC at edges).')
    sim = FDTD2D(grid=GRID, material=Material(eps_r=1.0), cfl=0.5, boundary='cpml', n_pml=10)
    # Disable CPML by zeroing coefficients.
    for arr in [sim.pml.sigma_x_E, sim.pml.sigma_x_H, sim.pml.sigma_y_E, sim.pml.sigma_y_H,
                sim.pml.alpha_x_E, sim.pml.alpha_x_H, sim.pml.alpha_y_E, sim.pml.alpha_y_H]:
        arr[:] = 0.0
    for arr in [sim.pml.kappa_x_E, sim.pml.kappa_x_H, sim.pml.kappa_y_E, sim.pml.kappa_y_H]:
        arr[:] = 1.0
    sim.pml.b_x_E[:] = 1.0; sim.pml.a_x_E[:] = 0.0
    sim.pml.b_x_H[:] = 1.0; sim.pml.a_x_H[:] = 0.0
    sim.pml.b_y_E[:] = 1.0; sim.pml.a_y_E[:] = 0.0
    sim.pml.b_y_H[:] = 1.0; sim.pml.a_y_H[:] = 0.0

    add_gaussian_packet(sim, wavelength=1.0, angle_deg=0.0, amplitude=1.0,
                        waist=3.0, pulse_length=3.0, center=(5.0, 5.0))
    print(f'  initial: naive E = {naive_energy(sim):.4f}, Yee E = {yee_energy(sim):.4f}')
    n_steps = 600
    for n in range(n_steps + 1):
        if n in {0, 100, 200, 400, 600}:
            e_naive = naive_energy(sim)
            e_yee = yee_energy(sim)
            print(f'  step {n:4d}: naive E = {e_naive:.4f}, Yee E = {e_yee:.4f}, '
                  f'peak |Ez| = {np.max(np.abs(sim.Ez)):.4f}')
        if n < n_steps:
            sim.step()


def test4_packet_cpml_active():
    print('\n--- Test 4: Gaussian packet, ACTIVE CPML. Energy should decrease (absorbed).')
    sim = FDTD2D(grid=GRID, material=Material(eps_r=1.0), cfl=0.5, boundary='cpml', n_pml=10)
    add_gaussian_packet(sim, wavelength=1.0, angle_deg=0.0, amplitude=1.0,
                        waist=3.0, pulse_length=3.0, center=(5.0, 5.0))
    print(f'  initial: naive E = {naive_energy(sim):.4f}, Yee E = {yee_energy(sim):.4f}')
    n_steps = 600
    for n in range(n_steps + 1):
        if n in {0, 100, 200, 400, 600}:
            e_naive = naive_energy(sim)
            e_yee = yee_energy(sim)
            print(f'  step {n:4d}: naive E = {e_naive:.4f}, Yee E = {e_yee:.4f}, '
                  f'peak |Ez| = {np.max(np.abs(sim.Ez)):.4f}')
        if n < n_steps:
            sim.step()


def test5_packet_mur():
    print('\n--- Test 5: Gaussian packet, Mur ABC (control). Energy should decrease.')
    sim = FDTD2D(grid=GRID, material=Material(eps_r=1.0), cfl=0.5, boundary='mur')
    add_gaussian_packet(sim, wavelength=1.0, angle_deg=0.0, amplitude=1.0,
                        waist=3.0, pulse_length=3.0, center=(5.0, 5.0))
    print(f'  initial: naive E = {naive_energy(sim):.4f}, Yee E = {yee_energy(sim):.4f}')
    n_steps = 600
    for n in range(n_steps + 1):
        if n in {0, 100, 200, 400, 600}:
            e_naive = naive_energy(sim)
            e_yee = yee_energy(sim)
            print(f'  step {n:4d}: naive E = {e_naive:.4f}, Yee E = {e_yee:.4f}, '
                  f'peak |Ez| = {np.max(np.abs(sim.Ez)):.4f}')
        if n < n_steps:
            sim.step()


def test2b_pec_eigenmode_no_mur():
    """Re-do test 2 with no Mur ABC — the boundary cells stay at 0 by virtue
    of the Material's update_E only touching [1:-1, 1:-1]. This is closer to
    a real PEC cavity, so energy should be approximately conserved."""
    print('\n--- Test 2b: PEC eigenmode (m=4, n=3) with NO Mur (implicit PEC).')
    sim = FDTD2D(grid=GRID, material=Material(eps_r=1.0), cfl=0.5, boundary='cpml', n_pml=10)
    # Disable CPML so it acts like raw FDTD with implicit PEC.
    for arr in [sim.pml.sigma_x_E, sim.pml.sigma_x_H, sim.pml.sigma_y_E, sim.pml.sigma_y_H,
                sim.pml.alpha_x_E, sim.pml.alpha_x_H, sim.pml.alpha_y_E, sim.pml.alpha_y_H]:
        arr[:] = 0.0
    for arr in [sim.pml.kappa_x_E, sim.pml.kappa_x_H, sim.pml.kappa_y_E, sim.pml.kappa_y_H]:
        arr[:] = 1.0
    sim.pml.b_x_E[:] = 1.0; sim.pml.a_x_E[:] = 0.0
    sim.pml.b_x_H[:] = 1.0; sim.pml.a_x_H[:] = 0.0
    sim.pml.b_y_E[:] = 1.0; sim.pml.a_y_E[:] = 0.0
    sim.pml.b_y_H[:] = 1.0; sim.pml.a_y_H[:] = 0.0
    omega = init_pec_eigenmode(sim, m=4, n=3)
    period_steps = int(2 * np.pi / omega / sim.dt)
    n_steps = 5 * period_steps
    snapshots = list(range(0, n_steps + 1, period_steps // 2))
    for n in range(n_steps + 1):
        if n in snapshots:
            e_yee = yee_energy(sim)
            print(f'  step {n:5d} (t={n*sim.dt:.3f}): Yee E = {e_yee:.4f}')
        if n < n_steps:
            sim.step()


def test6_cpml_oblique():
    """Test CPML with oblique incidence (where TF/SF debugging showed problems)."""
    print('\n--- Test 6: Gaussian packet at 20° angle, CPML. Does CPML fail at oblique?')
    sim = FDTD2D(grid=GRID, material=Material(eps_r=1.0), cfl=0.5, boundary='cpml', n_pml=10)
    add_gaussian_packet(sim, wavelength=1.0, angle_deg=20.0, amplitude=1.0,
                        waist=3.0, pulse_length=3.0, center=(5.0, 5.0))
    print(f'  initial: Yee E = {yee_energy(sim):.4f}')
    n_steps = 600
    for n in range(n_steps + 1):
        if n in {0, 100, 200, 400, 600}:
            e_yee = yee_energy(sim)
            print(f'  step {n:4d}: Yee E = {e_yee:.4f}, peak |Ez| = {np.max(np.abs(sim.Ez)):.4f}')
        if n < n_steps:
            sim.step()


def test7_cpml_oblique_steeper():
    """Steeper angle to amplify the issue if one exists."""
    print('\n--- Test 7: Gaussian packet at 45° angle, CPML.')
    sim = FDTD2D(grid=GRID, material=Material(eps_r=1.0), cfl=0.5, boundary='cpml', n_pml=10)
    add_gaussian_packet(sim, wavelength=1.0, angle_deg=45.0, amplitude=1.0,
                        waist=3.0, pulse_length=3.0, center=(5.0, 5.0))
    print(f'  initial: Yee E = {yee_energy(sim):.4f}')
    n_steps = 600
    for n in range(n_steps + 1):
        if n in {0, 100, 200, 400, 600}:
            e_yee = yee_energy(sim)
            print(f'  step {n:4d}: Yee E = {e_yee:.4f}, peak |Ez| = {np.max(np.abs(sim.Ez)):.4f}')
        if n < n_steps:
            sim.step()


if __name__ == '__main__':
    test1_zero_IC()
    test2_pec_eigenmode()
    test2b_pec_eigenmode_no_mur()
    test3_packet_no_boundary()
    test4_packet_cpml_active()
    test5_packet_mur()
    test6_cpml_oblique()
    test7_cpml_oblique_steeper()
