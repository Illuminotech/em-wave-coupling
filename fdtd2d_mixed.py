"""2D mixed-polarization FDTD (TM + TE simultaneously).

Tracks all six field components on a 2D Yee grid:
    E_z   at (i, j)         shape (nx, ny)
    H_z   at (i+½, j+½)     shape (nx-1, ny-1)
    E_x   at (i+½, j)       shape (nx-1, ny)
    H_y   at (i+½, j)       shape (nx-1, ny)
    E_y   at (i, j+½)       shape (nx, ny-1)
    H_x   at (i, j+½)       shape (nx, ny-1)

In vacuum the TM mode (E_z, H_x, H_y) and TE mode (H_z, E_x, E_y) decouple
under ∂/∂z = 0. They couple only through nonlinear constitutive relations.

Key thesis use: the QED Heisenberg-Euler G² = (E·B)² channel is identically
zero for either pure-TM or pure-TE waves but nonzero whenever both polarizations
overlap — providing the cleanest possible 'wave-wave only, no self-action'
coupling channel in QED.

For now this module provides:
  - 2D mixed-mode Yee update in vacuum
  - Initial-condition helpers for TM and TE plane-wave packets
  - Linearized HE constitutive correction (F² and G² channels)
  - Diagnostics: total energy, single-component energy, probes

Boundary handling: simple PEC implicit (Ez=0 at the four E_z boundary cells,
analogous treatment for E_x at top/bottom edges etc.). Equivalent to closed
metal box. Adequate for short runs where the wave hasn't yet reached the wall.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from fdtd2d import Grid     # reuse the simple Grid dataclass


class FDTD2D_Mixed:
    """2D Yee FDTD with all 6 field components.

    Conventions (vacuum, ε₀=μ₀=c=1):
        ∂_t H_x = -∂_y E_z
        ∂_t H_y = +∂_x E_z
        ∂_t H_z = -(∂_x E_y - ∂_y E_x)
        ∂_t E_x = +∂_y H_z
        ∂_t E_y = -∂_x H_z
        ∂_t E_z = +∂_x H_y - ∂_y H_x
    """

    def __init__(self, grid: Grid, cfl: float = 0.5,
                 kappa_F: float = 0.0, kappa_G: float = 0.0):
        self.g = grid
        self.cfl = cfl
        self.dt = cfl / np.sqrt(1.0 / grid.dx ** 2 + 1.0 / grid.dy ** 2)
        self.t = 0.0
        # Nonlinear HE coefficients. With L = (1/2)(E²-B²) + κ_F F² + κ_G G²,
        # the linearized constitutive is
        #   D ≈ E (1 + 2κ_F F) + 2κ_G G B   (vector)
        #   H ≈ B (1 + 2κ_F F) - 2κ_G G E.
        self.kappa_F = kappa_F
        self.kappa_G = kappa_G

        nx, ny = grid.nx, grid.ny
        self.Ez = np.zeros((nx, ny))
        self.Hz = np.zeros((nx - 1, ny - 1))
        self.Ex = np.zeros((nx - 1, ny))
        self.Hy = np.zeros((nx - 1, ny))
        self.Ey = np.zeros((nx, ny - 1))
        self.Hx = np.zeros((nx, ny - 1))

    # ------------------------------------------------------------------
    # Core Yee updates (linear vacuum). Update H by Δt, then update E by Δt.
    # ------------------------------------------------------------------
    def _update_H_linear(self):
        dt, dx, dy = self.dt, self.g.dx, self.g.dy
        # H_x at (i, j+½): ∂_t H_x = -∂_y E_z.
        # Indexing: Hx[i, j] uses Ez[i, j+1] - Ez[i, j].
        self.Hx -= dt * (self.Ez[:, 1:] - self.Ez[:, :-1]) / dy
        # H_y at (i+½, j): ∂_t H_y = +∂_x E_z.
        self.Hy += dt * (self.Ez[1:, :] - self.Ez[:-1, :]) / dx
        # H_z at (i+½, j+½): ∂_t H_z = -(∂_x E_y - ∂_y E_x).
        # H_z[i, j] sits at (i+½, j+½); uses E_y[i+1, j] - E_y[i, j] (for ∂_x)
        # and E_x[i, j+1] - E_x[i, j] (for ∂_y).
        dEy_dx = (self.Ey[1:, :] - self.Ey[:-1, :]) / dx     # shape (nx-1, ny-1)
        dEx_dy = (self.Ex[:, 1:] - self.Ex[:, :-1]) / dy     # shape (nx-1, ny-1)
        self.Hz -= dt * (dEy_dx - dEx_dy)

    def _update_E_linear(self):
        dt, dx, dy = self.dt, self.g.dx, self.g.dy
        # E_x at (i+½, j): ∂_t E_x = +∂_y H_z. Only update interior (j ∈ [1, ny-2]).
        # Hz[i, j] at (i+½, j+½). For Ex[i, j] at (i+½, j), ∂_y H_z ≈ (Hz[i, j] - Hz[i, j-1])/dy.
        self.Ex[:, 1:-1] += dt * (self.Hz[:, 1:] - self.Hz[:, :-1]) / dy
        # E_y at (i, j+½): ∂_t E_y = -∂_x H_z. Only update interior (i ∈ [1, nx-2]).
        self.Ey[1:-1, :] -= dt * (self.Hz[1:, :] - self.Hz[:-1, :]) / dx
        # E_z at (i, j): ∂_t E_z = +∂_x H_y - ∂_y H_x. Same form as TM-only code.
        curl_H = (self.Hy[1:, 1:-1] - self.Hy[:-1, 1:-1]) / dx \
               - (self.Hx[1:-1, 1:] - self.Hx[1:-1, :-1]) / dy
        self.Ez[1:-1, 1:-1] += dt * curl_H

    # ------------------------------------------------------------------
    # Linearized HE constitutive correction (first order in κ).
    # ------------------------------------------------------------------
    def _interpolate_to_centers(self):
        """Return all six field components interpolated to integer (i, j) cell centers."""
        nx, ny = self.g.nx, self.g.ny
        Ez = self.Ez
        # Ex at (i+½, j): average neighbours in i to integer.
        Ex_cc = np.zeros((nx, ny))
        Ex_cc[1:-1, :] = 0.5 * (self.Ex[:-1, :] + self.Ex[1:, :])
        Ex_cc[0, :]    = self.Ex[0, :]
        Ex_cc[-1, :]   = self.Ex[-1, :]
        # Ey at (i, j+½): average neighbours in j.
        Ey_cc = np.zeros((nx, ny))
        Ey_cc[:, 1:-1] = 0.5 * (self.Ey[:, :-1] + self.Ey[:, 1:])
        Ey_cc[:, 0]    = self.Ey[:, 0]
        Ey_cc[:, -1]   = self.Ey[:, -1]
        # Hx at (i, j+½): same averaging as Ey.
        Hx_cc = np.zeros((nx, ny))
        Hx_cc[:, 1:-1] = 0.5 * (self.Hx[:, :-1] + self.Hx[:, 1:])
        Hx_cc[:, 0]    = self.Hx[:, 0]
        Hx_cc[:, -1]   = self.Hx[:, -1]
        # Hy at (i+½, j): same averaging as Ex.
        Hy_cc = np.zeros((nx, ny))
        Hy_cc[1:-1, :] = 0.5 * (self.Hy[:-1, :] + self.Hy[1:, :])
        Hy_cc[0, :]    = self.Hy[0, :]
        Hy_cc[-1, :]   = self.Hy[-1, :]
        # Hz at (i+½, j+½): average across both i and j.
        Hz_cc = np.zeros((nx, ny))
        # Interior 4-point average:
        Hz_cc[1:-1, 1:-1] = 0.25 * (self.Hz[:-1, :-1] + self.Hz[1:, :-1]
                                    + self.Hz[:-1, 1:] + self.Hz[1:, 1:])
        # Boundary: 2-point or copy.
        Hz_cc[0, 1:-1] = 0.5 * (self.Hz[0, :-1] + self.Hz[0, 1:])
        Hz_cc[-1, 1:-1] = 0.5 * (self.Hz[-1, :-1] + self.Hz[-1, 1:])
        Hz_cc[1:-1, 0] = 0.5 * (self.Hz[:-1, 0] + self.Hz[1:, 0])
        Hz_cc[1:-1, -1] = 0.5 * (self.Hz[:-1, -1] + self.Hz[1:, -1])
        return Ex_cc, Ey_cc, Ez, Hx_cc, Hy_cc, Hz_cc

    def _apply_HE_linearized(self):
        """Apply the linearized HE correction to E and H fields after the
        vacuum update. First-order in κ: δE = -2κ_F F E - 2κ_G G B (computed
        from current fields). Adequate for κ × max|field|² ≲ 0.1.
        """
        if self.kappa_F == 0.0 and self.kappa_G == 0.0:
            return
        Ex_cc, Ey_cc, Ez_cc, Bx_cc, By_cc, Bz_cc = self._interpolate_to_centers()
        E_sq = Ex_cc ** 2 + Ey_cc ** 2 + Ez_cc ** 2
        B_sq = Bx_cc ** 2 + By_cc ** 2 + Bz_cc ** 2
        F = 0.5 * (E_sq - B_sq)
        G = Ex_cc * Bx_cc + Ey_cc * By_cc + Ez_cc * Bz_cc
        # Corrections at cell centers; we apply them by interpolating back to
        # the Yee-staggered positions. For each component, the correction is
        # δE_i = 2κ_F F E_i + 2κ_G G B_i  (this is δD - δE_linear, but for our
        # linearized scheme we treat the correction as a small E-update term).
        # Sign: D = E + δD → E = D - δD, so the post-update E correction
        # subtracts δD: δE_total = -2κ_F F E_i - 2κ_G G B_i.
        twoF = 2.0 * self.kappa_F * F
        twoG = 2.0 * self.kappa_G * G
        dEx_cc = -(twoF * Ex_cc + twoG * Bx_cc)
        dEy_cc = -(twoF * Ey_cc + twoG * By_cc)
        dEz_cc = -(twoF * Ez_cc + twoG * Bz_cc)
        # Interpolate back to Yee positions and add.
        # Ex at (i+½, j): average of Ex_cc[i, j] and Ex_cc[i+1, j].
        self.Ex += 0.5 * (dEx_cc[:-1, :] + dEx_cc[1:, :])
        # Ey at (i, j+½): average of Ey_cc[i, j] and Ey_cc[i, j+1].
        self.Ey += 0.5 * (dEy_cc[:, :-1] + dEy_cc[:, 1:])
        # Ez at (i, j): use directly.
        self.Ez += dEz_cc

    def step(self):
        self._update_H_linear()
        self._update_E_linear()
        self._apply_HE_linearized()
        self.t += self.dt

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def total_energy(self) -> float:
        """Sum of squares of all 6 components × cell area."""
        return 0.5 * (np.sum(self.Ex ** 2) + np.sum(self.Ey ** 2) + np.sum(self.Ez ** 2)
                      + np.sum(self.Hx ** 2) + np.sum(self.Hy ** 2) + np.sum(self.Hz ** 2)) \
               * self.g.dx * self.g.dy

    def tm_energy(self) -> float:
        """Energy in the TM-mode components (Ez, Hx, Hy)."""
        return 0.5 * (np.sum(self.Ez ** 2)
                      + np.sum(self.Hx ** 2) + np.sum(self.Hy ** 2)) \
               * self.g.dx * self.g.dy

    def te_energy(self) -> float:
        """Energy in the TE-mode components (Ex, Ey, Hz)."""
        return 0.5 * (np.sum(self.Ex ** 2) + np.sum(self.Ey ** 2)
                      + np.sum(self.Hz ** 2)) * self.g.dx * self.g.dy


# ----------------------------------------------------------------------
# Initial-condition helpers
# ----------------------------------------------------------------------
def add_TM_packet(sim: FDTD2D_Mixed, *,
                  center, wavelength, angle_deg, amplitude,
                  waist, pulse_length):
    """Add a TM-polarized Gaussian wave packet (E_z, H_x, H_y components).

    Same sign conventions as fdtd2d.add_gaussian_packet:
       E_z = A · env · cos(k·r)
       H_x = +sin θ · A · env · cos(k·r + ω·Δt/2)
       H_y = -cos θ · A · env · cos(k·r + ω·Δt/2)
    """
    g = sim.g
    theta = np.deg2rad(angle_deg)
    k = 2.0 * np.pi / wavelength
    kx, ky = k * np.cos(theta), k * np.sin(theta)
    nx_, ny_ = np.cos(theta), np.sin(theta)
    tx_, ty_ = -np.sin(theta), np.cos(theta)
    cx, cy = center
    half_phase_shift = -0.5 * sim.dt * k

    def envelope(X, Y):
        rx, ry = X - cx, Y - cy
        s_long = rx * nx_ + ry * ny_
        s_trans = rx * tx_ + ry * ty_
        return amplitude * np.exp(-(s_long / pulse_length) ** 2) \
                         * np.exp(-(s_trans / waist) ** 2)

    # E_z at integer (i, j)
    x_E = np.arange(g.nx) * g.dx
    y_E = np.arange(g.ny) * g.dy
    XE, YE = np.meshgrid(x_E, y_E, indexing='ij')
    env_E = envelope(XE, YE)
    phase_E = kx * XE + ky * YE
    sim.Ez += env_E * np.cos(phase_E)

    # H_x at (i, j+½)
    y_Hx = (np.arange(g.ny - 1) + 0.5) * g.dy
    XHx, YHx = np.meshgrid(x_E, y_Hx, indexing='ij')
    env_Hx = envelope(XHx, YHx)
    phase_Hx = kx * XHx + ky * YHx
    sim.Hx += np.sin(theta) * env_Hx * np.cos(phase_Hx - half_phase_shift)

    # H_y at (i+½, j)
    x_Hy = (np.arange(g.nx - 1) + 0.5) * g.dx
    XHy, YHy = np.meshgrid(x_Hy, y_E, indexing='ij')
    env_Hy = envelope(XHy, YHy)
    phase_Hy = kx * XHy + ky * YHy
    sim.Hy += -np.cos(theta) * env_Hy * np.cos(phase_Hy - half_phase_shift)


def add_TE_packet(sim: FDTD2D_Mixed, *,
                  center, wavelength, angle_deg, amplitude,
                  waist, pulse_length):
    """Add a TE-polarized Gaussian wave packet (B_z, E_x, E_y components).

    For TE wave propagating in +k̂: B_z (single component, along ẑ).
    The consistent E components come from Faraday + Maxwell, mirroring
    the TM relation but for the dual fields. For B_z = A·env·cos(k·r - ωt):
       E_x = -sin θ · A · env · cos(k·r)   (mirror of H_x = +sin θ · E_z)
       E_y = +cos θ · A · env · cos(k·r)   (mirror of H_y = -cos θ · E_z)
    Derivation: ∂_t E_x = +∂_y H_z gives E_x = -sin θ · H_z for +k̂ propagation
    via the same logic that gave H_y = -cos θ · E_z for the TM wave.
    """
    g = sim.g
    theta = np.deg2rad(angle_deg)
    k = 2.0 * np.pi / wavelength
    kx, ky = k * np.cos(theta), k * np.sin(theta)
    nx_, ny_ = np.cos(theta), np.sin(theta)
    tx_, ty_ = -np.sin(theta), np.cos(theta)
    cx, cy = center
    half_phase_shift = -0.5 * sim.dt * k

    def envelope(X, Y):
        rx, ry = X - cx, Y - cy
        s_long = rx * nx_ + ry * ny_
        s_trans = rx * tx_ + ry * ty_
        return amplitude * np.exp(-(s_long / pulse_length) ** 2) \
                         * np.exp(-(s_trans / waist) ** 2)

    # H_z at (i+½, j+½): the carrier B field
    x_Hz = (np.arange(g.nx - 1) + 0.5) * g.dx
    y_Hz = (np.arange(g.ny - 1) + 0.5) * g.dy
    XHz, YHz = np.meshgrid(x_Hz, y_Hz, indexing='ij')
    env_Hz = envelope(XHz, YHz)
    phase_Hz = kx * XHz + ky * YHz
    sim.Hz += env_Hz * np.cos(phase_Hz - half_phase_shift)

    # E_x at (i+½, j): -sin θ · H_z analogue
    x_Ex = (np.arange(g.nx - 1) + 0.5) * g.dx
    y_Ex = np.arange(g.ny) * g.dy
    XEx, YEx = np.meshgrid(x_Ex, y_Ex, indexing='ij')
    env_Ex = envelope(XEx, YEx)
    phase_Ex = kx * XEx + ky * YEx
    sim.Ex += -np.sin(theta) * env_Ex * np.cos(phase_Ex)

    # E_y at (i, j+½): +cos θ · H_z analogue
    x_Ey = np.arange(g.nx) * g.dx
    y_Ey = (np.arange(g.ny - 1) + 0.5) * g.dy
    XEy, YEy = np.meshgrid(x_Ey, y_Ey, indexing='ij')
    env_Ey = envelope(XEy, YEy)
    phase_Ey = kx * XEy + ky * YEy
    sim.Ey += np.cos(theta) * env_Ey * np.cos(phase_Ey)
