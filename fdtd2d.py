"""
2D FDTD core for studying EM wave-wave interaction.

Mode: TM_z. Independent fields are E_z, H_x, H_y on a Yee grid.
Units: natural (c = eps0 = mu0 = 1). Lengths and times are in arbitrary units;
wavelengths are set in those same units. Convert to SI in post-processing.

The constitutive relation is dispatched through `Material.update_E`. The linear
vacuum/dielectric case is implemented here. Kerr, plasma, and QED variants
plug into the same interface (subclass Material; override update_E).

Yee staggering (2D, TMz):
    E_z[i, j]      at (i*dx,           j*dy)
    H_x[i, j]      at (i*dx,         (j+0.5)*dy)
    H_y[i, j]      at ((i+0.5)*dx,     j*dy)

Time staggering: E at integer steps n, H at half steps n+1/2.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# --------------------------------------------------------------------------
# Material interface
# --------------------------------------------------------------------------

class Material:
    """Base class. Subclass to add nonlinear constitutive behavior or auxiliary
    state (e.g. plasma fluid). Hooks:
      - init_state(grid)        : allocate aux fields once at simulation start
      - update_H(...)           : standard Faraday step (override only if H sees a polarization/magnetization rate)
      - step_state(...)         : advance aux state (e.g. plasma fluid) by dt
      - update_E(...)           : standard Ampere step with whatever currents/constitutive relation applies
    """

    def __init__(self, eps_r: float = 1.0, mu_r: float = 1.0):
        self.eps_r = eps_r
        self.mu_r = mu_r

    def init_state(self, grid):
        pass  # no-op for stateless materials

    def step_state(self, Ez, Hx, Hy, dt, dx, dy, *, Hx_prev=None, Hy_prev=None):
        pass  # no-op for stateless materials

    def update_E(self, Ez, Hx, Hy, dt, dx, dy, *, Hx_prev=None, Hy_prev=None):
        """Linear, isotropic, non-dispersive update. Hx_prev/Hy_prev are the
        H-field snapshots from before the H update of the current step
        (i.e., at time n-1/2). Linear materials ignore them; nonlinear
        materials whose constitutive relation depends on |B|² (e.g. QED HE)
        use them to time-center B² at integer step n via averaging."""
        curl_H = (Hy[1:, 1:-1] - Hy[:-1, 1:-1]) / dx \
               - (Hx[1:-1, 1:] - Hx[1:-1, :-1]) / dy
        Ez[1:-1, 1:-1] += (dt / self.eps_r) * curl_H

    def update_H(self, Ez, Hx, Hy, dt, dx, dy):
        Hx[:, :] -= (dt / self.mu_r) * (Ez[:, 1:] - Ez[:, :-1]) / dy
        Hy[:, :] += (dt / self.mu_r) * (Ez[1:, :] - Ez[:-1, :]) / dx


# --------------------------------------------------------------------------
# Grid + solver
# --------------------------------------------------------------------------

@dataclass
class Grid:
    nx: int
    ny: int
    dx: float
    dy: float

    @property
    def Lx(self) -> float: return self.nx * self.dx

    @property
    def Ly(self) -> float: return self.ny * self.dy


class FDTD2D:
    """2D TMz FDTD with Mur 1st-order absorbing boundaries."""

    def __init__(self, grid: Grid, material: Material, cfl: float = 0.7,
                 boundary: str = 'mur', n_pml: int = 10,
                 pml_R0: float = 1e-6):
        # NOTE: boundary='cpml' is correct but has a startup constraint:
        # the IC field MUST be zero in the PML region (see CPML2D docstring).
        # Default is 'mur' (1st-order ABC, ~10% reflection at grazing
        # incidence) which is robust for IC-based simulations regardless of
        # packet placement. Switch to 'cpml' when (a) using TF/SF sources
        # or (b) using IC packets placed ≥ 3 waists from every grid edge.
        self.g = grid
        self.mat = material
        self.boundary = boundary
        # CFL: c*dt <= 1/sqrt(1/dx^2 + 1/dy^2). c=1 in natural units.
        self.dt = cfl / np.sqrt(1.0 / grid.dx**2 + 1.0 / grid.dy**2)
        self.t = 0.0

        # Yee fields. Sizes chosen so curls land on the right grid points.
        self.Ez = np.zeros((grid.nx, grid.ny))
        self.Hx = np.zeros((grid.nx, grid.ny - 1))
        self.Hy = np.zeros((grid.nx - 1, grid.ny))

        # Let the material allocate its auxiliary state (e.g. plasma fluid).
        self.mat.init_state(grid)

        # Optional probes for spectral diagnostics; set via add_probe().
        self.probes = []

        # Optional TF/SF sources for boundary-driven wave injection.
        self.sources = []

        # Boundary handling.
        self.pml = None
        if boundary == 'cpml':
            self.pml = CPML2D(grid, dt=self.dt, n_pml=n_pml, R0=pml_R0)
        elif boundary == 'mur':
            pass    # Mur arrays initialized below
        else:
            raise ValueError(f"unknown boundary {boundary!r}; "
                             "use 'cpml' or 'mur'")

        # Mur 1st-order ABC: only used if boundary=='mur'.
        self._prev = {
            'x0_b': np.zeros(grid.ny), 'x0_i': np.zeros(grid.ny),
            'x1_b': np.zeros(grid.ny), 'x1_i': np.zeros(grid.ny),
            'y0_b': np.zeros(grid.nx), 'y0_i': np.zeros(grid.nx),
            'y1_b': np.zeros(grid.nx), 'y1_i': np.zeros(grid.nx),
        }

    def add_probe(self, probe):
        """Register a ProbeRecorder (or any object with .record(t, Ez))."""
        self.probes.append(probe)

    def add_source(self, source):
        """Register a TF/SF source. Multiple sources are superposed via the
        linearity of the contour-correction terms."""
        self.sources.append(source)

    def step(self):
        g = self.g
        # Snapshot edge values at time n before the E update for Mur ABC.
        snap = {
            'x0_b': self.Ez[0, :].copy(),  'x0_i': self.Ez[1, :].copy(),
            'x1_b': self.Ez[-1, :].copy(), 'x1_i': self.Ez[-2, :].copy(),
            'y0_b': self.Ez[:, 0].copy(),  'y0_i': self.Ez[:, 1].copy(),
            'y1_b': self.Ez[:, -1].copy(), 'y1_i': self.Ez[:, -2].copy(),
        }
        # Snapshot H at time n-1/2 (before update) for time-centering of B²
        # in nonlinear materials whose constitutive depends on |B|² (e.g. QED).
        Hx_prev = self.Hx.copy()
        Hy_prev = self.Hy.copy()
        # Leapfrog: advance H, push auxiliary state (plasma fluid, etc.), then E.
        self.mat.update_H(self.Ez, self.Hx, self.Hy, self.dt, g.dx, g.dy)
        # CPML correction to H. Uses E^n (still pre-update_E here) and updates
        # ψ_H based on E derivatives, then adjusts H.
        if self.pml is not None:
            self.pml.apply_H_correction(self)
        # TF/SF correction to H using E_inc at integer step t = n dt.
        for src in self.sources:
            src.apply_H_corrections(self, self.t)
        self.mat.step_state(self.Ez, self.Hx, self.Hy, self.dt, g.dx, g.dy,
                            Hx_prev=Hx_prev, Hy_prev=Hy_prev)
        self.mat.update_E(self.Ez, self.Hx, self.Hy, self.dt, g.dx, g.dy,
                          Hx_prev=Hx_prev, Hy_prev=Hy_prev)
        # CPML correction to E using H^{n+½}.
        if self.pml is not None:
            self.pml.apply_E_correction(self)
        # TF/SF correction to E using H_inc at half-step t = (n+1/2) dt.
        for src in self.sources:
            src.apply_E_corrections(self, self.t + 0.5 * self.dt)
        # Legacy Mur ABC on Ez boundaries (only if boundary='mur').
        if self.boundary == 'mur':
            self._apply_mur(snap)
        self.t += self.dt
        # Sample probes after the full step is complete and t is advanced.
        for p in self.probes:
            p.record(self.t, self.Ez)

    def _apply_mur(self, prev_n: dict):
        """Mur 1st order:
            E_b^{n+1} = E_i^n + (c dt - h) / (c dt + h) * (E_i^{n+1} - E_b^n),
        where 'b' is the boundary cell, 'i' is the cell one-in, h is dx or dy.
        """
        c, dx, dy = 1.0, self.g.dx, self.g.dy
        rx = (c * self.dt - dx) / (c * self.dt + dx)
        ry = (c * self.dt - dy) / (c * self.dt + dy)
        Ez = self.Ez
        Ez[0, :]  = prev_n['x0_i'] + rx * (Ez[1, :]  - prev_n['x0_b'])
        Ez[-1, :] = prev_n['x1_i'] + rx * (Ez[-2, :] - prev_n['x1_b'])
        Ez[:, 0]  = prev_n['y0_i'] + ry * (Ez[:, 1]  - prev_n['y0_b'])
        Ez[:, -1] = prev_n['y1_i'] + ry * (Ez[:, -2] - prev_n['y1_b'])

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def energy(self) -> float:
        """Total EM energy in the grid, natural units (eps0 = mu0 = 1)."""
        uE = 0.5 * np.sum(self.Ez ** 2)
        uH = 0.5 * (np.sum(self.Hx ** 2) + np.sum(self.Hy ** 2))
        return float((uE + uH) * self.g.dx * self.g.dy)


# --------------------------------------------------------------------------
# Initial conditions: Gaussian wave packet
# --------------------------------------------------------------------------

def add_gaussian_packet(
    sim: FDTD2D,
    *,
    center: tuple[float, float],
    wavelength: float,
    angle_deg: float,
    amplitude: float,
    waist: float,
    pulse_length: float,
):
    """Superpose a Gaussian-modulated plane-wave packet onto the existing fields.

    The packet is centered at `center`, propagates at `angle_deg` (measured from +x),
    has carrier wavelength `wavelength`, peak Ez amplitude `amplitude`, transverse
    (1/e) waist `waist`, and longitudinal (1/e) pulse length `pulse_length`.

    Linearity guarantees that calling this twice with different parameters injects
    two independent packets that will then evolve under the (linear or nonlinear)
    Maxwell update.
    """
    g = sim.g
    k = 2.0 * np.pi / wavelength
    theta = np.deg2rad(angle_deg)
    kx, ky = k * np.cos(theta), k * np.sin(theta)
    # Unit propagation and transverse vectors.
    nx, ny = np.cos(theta), np.sin(theta)
    tx, ty = -np.sin(theta), np.cos(theta)

    x = np.arange(g.nx) * g.dx
    y = np.arange(g.ny) * g.dy
    X, Y = np.meshgrid(x, y, indexing='ij')

    cx, cy = center
    rx, ry = X - cx, Y - cy
    s_long = rx * nx + ry * ny           # along propagation
    s_trans = rx * tx + ry * ty          # transverse to propagation

    envelope = (amplitude
                * np.exp(-(s_long / pulse_length) ** 2)
                * np.exp(-(s_trans / waist) ** 2))
    phase = kx * rx + ky * ry

    # E_z at integer time step.
    sim.Ez += envelope * np.cos(phase)

    # If the material carries a plasma fluid, seed v_z in its linear-response
    # steady state so the wave isn't switched on against a stationary fluid
    # (which would otherwise excite a transient free plasma oscillation of
    # amplitude comparable to the driven response and pollute the diagnostics).
    # Linear: dv_z/dt = -E_z ⇒ v_z = (A·env/ω) sin(phase). At low v, p_z ≈ v_z.
    mat = sim.mat
    if getattr(mat, 'pz', None) is not None:
        omega = k          # ω = c k = k in natural units
        mat.pz += (envelope / omega) * np.sin(phase)

    # For a wave with E = E_z ẑ propagating in +k̂ = (cos θ, sin θ), the
    # consistent solution of the FDTD update equations is
    #   H_x = +sin θ · E_z,    H_y = -cos θ · E_z
    # (verified by ∂_t H = -∇×E and Poynting S = E×H along +k̂.)
    # Earlier versions of this function used opposite signs; that produced
    # waves propagating at angle (θ + 180°). The correct signs below give
    # waves propagating at the requested angle.
    # Initialize H at t = -dt/2 so the FDTD2D pre/post H average lands at t = n·dt.
    half_phase_shift = -0.5 * sim.dt * k

    # Build H envelopes on their staggered locations.
    # H_x lives at (i*dx, (j+0.5)*dy):
    Yh = (np.arange(g.ny - 1) + 0.5) * g.dy
    Xh, Yh_g = np.meshgrid(x, Yh, indexing='ij')
    rxh, ryh = Xh - cx, Yh_g - cy
    sl = rxh * nx + ryh * ny
    st = rxh * tx + ryh * ty
    env_h = (amplitude
             * np.exp(-(sl / pulse_length) ** 2)
             * np.exp(-(st / waist) ** 2))
    sim.Hx += +np.sin(theta) * env_h * np.cos(kx * rxh + ky * ryh - half_phase_shift)

    # H_y lives at ((i+0.5)*dx, j*dy):
    Xh2 = (np.arange(g.nx - 1) + 0.5) * g.dx
    Xh2_g, Yh2 = np.meshgrid(Xh2, y, indexing='ij')
    rxh2, ryh2 = Xh2_g - cx, Yh2 - cy
    sl = rxh2 * nx + ryh2 * ny
    st = rxh2 * tx + ryh2 * ty
    env_h2 = (amplitude
              * np.exp(-(sl / pulse_length) ** 2)
              * np.exp(-(st / waist) ** 2))
    sim.Hy += -np.cos(theta) * env_h2 * np.cos(kx * rxh2 + ky * ryh2 - half_phase_shift)


class CPML2D:
    """Convolutional PML (Roden & Gedney 2000) for 2D TMz FDTD.

    USAGE WARNING — IC packets MUST NOT overlap the PML region.
    --------------------------------------------------------------------
    CPML uses memory variables ψ that are zero at t=0. For correct
    absorption, the field at t=0 must also be zero in the PML region;
    otherwise ψ and the field are "out of equilibrium," and the CPML
    correction *amplifies* the wave rather than absorbing it (the field
    grows ~ (E_initial_in_PML / E_total)^N for some integer N).
    Empirically, energy doubles or quintuples when IC overlap with PML > 0.1.

    Practical rules:
      1. Place IC Gaussian packets so the envelope amplitude < 1% in PML.
         For waist w and PML thickness n_pml*dx, keep the packet center at
         least 3w from each grid boundary.
      2. Or use a TF/SF source (see source.py) — waves are *injected* in
         the interior, so the PML never sees nonzero fields at t=0.
      3. Or initialize the ψ memory variables to match the IC (rarely done
         in practice).

    Replaces the Mur 1st-order ABC. Each spatial derivative in the PML region
    is replaced by (1/κ) ∂ + ψ where ψ is a memory variable that integrates
    the derivative with exponentially-decaying weight. In the interior
    (n_pml < i < nx-n_pml etc.), κ=1, σ=α=0, ψ stays at 0 ⇒ no-op.

    Profile (polynomial grading from PML inner edge ρ=0 to outer edge ρ=d_pml):
        σ(ρ) = σ_max (ρ/d_pml)^m
        κ(ρ) = 1 + (κ_max-1)(ρ/d_pml)^m
        α(ρ) = α_max (1 - ρ/d_pml)^m
    σ_max from target reflection R₀:  σ_max = -(m+1) ln R₀ / (2 η₀ d_pml).

    Update coefficients (precomputed since dt is fixed):
        b = exp(-(σ/κ + α) Δt)
        a = (σ / (σ κ + κ² α)) (b - 1)

    ψ updates (per FDTD step):
        ψ_E_zx[i,j]^{n+1/2} = b_x[i] · ψ_E_zx[i,j]^{n-1/2} + a_x[i] · ∂_x H_y^{n+1/2}
        ψ_E_zy similarly for ∂_y H_x.
        ψ_H_xy[i,j]^{n+1/2} = b_y[j+½] · ψ_H_xy[i,j]^{n-1/2} + a_y[j+½] · ∂_y E_z^n
        ψ_H_yx similarly for ∂_x E_z.

    Field corrections (added on top of the default Material updates):
        E_z += (Δt/ε) [((1/κ_x)-1) ∂_x H_y - ((1/κ_y)-1) ∂_y H_x + ψ_E_zx - ψ_E_zy]
        H_x += -(Δt/μ) [((1/κ_y)-1) ∂_y E_z + ψ_H_xy]
        H_y += +(Δt/μ) [((1/κ_x)-1) ∂_x E_z + ψ_H_yx]
    """

    def __init__(self, grid: Grid, dt: float,
                 n_pml: int = 10, m: int = 3,
                 R0: float = 1e-6, kappa_max: float = 5.0,
                 alpha_max: float = 0.05):
        self.grid = grid
        self.dt = dt
        self.n_pml = n_pml
        self.m = m

        nx, ny = grid.nx, grid.ny
        d_pml_x = n_pml * grid.dx
        d_pml_y = n_pml * grid.dy

        # σ_max from target reflection R₀.
        eta0 = 1.0
        self.sigma_max_x = -(m + 1) * np.log(R0) / (2.0 * eta0 * d_pml_x)
        self.sigma_max_y = -(m + 1) * np.log(R0) / (2.0 * eta0 * d_pml_y)

        # Build profiles at E positions (integer i, j) and H positions (half-integer).
        def profile(i, n_total, n_pml, sigma_max, kappa_max, alpha_max, m, half=False):
            """Return σ, κ, α at grid index `i` (with half-cell offset if `half`)."""
            x = i + 0.5 if half else float(i)
            sigma, kappa, alpha = 0.0, 1.0, 0.0
            # Distance from inner edge of left PML (in cells): ρ = n_pml - x
            if x < n_pml:
                rho = (n_pml - x) / n_pml
                sigma = sigma_max * rho ** m
                kappa = 1.0 + (kappa_max - 1.0) * rho ** m
                alpha = alpha_max * (1 - rho) ** m
            # Distance from inner edge of right PML: ρ = (x - (n_total-1-n_pml)) / n_pml
            elif x > n_total - 1 - n_pml:
                rho = (x - (n_total - 1 - n_pml)) / n_pml
                sigma = sigma_max * rho ** m
                kappa = 1.0 + (kappa_max - 1.0) * rho ** m
                alpha = alpha_max * (1 - rho) ** m
            return sigma, kappa, alpha

        def make_profile_arrays(n_total, n_pml, sigma_max, kappa_max, alpha_max, m, half):
            sigma = np.zeros(n_total - (1 if half else 0))
            kappa = np.ones_like(sigma)
            alpha = np.zeros_like(sigma)
            for i in range(len(sigma)):
                s, k, a = profile(i, n_total, n_pml,
                                   sigma_max, kappa_max, alpha_max, m, half)
                sigma[i], kappa[i], alpha[i] = s, k, a
            return sigma, kappa, alpha

        self.sigma_x_E, self.kappa_x_E, self.alpha_x_E = make_profile_arrays(
            nx, n_pml, self.sigma_max_x, kappa_max, alpha_max, m, half=False)
        self.sigma_x_H, self.kappa_x_H, self.alpha_x_H = make_profile_arrays(
            nx, n_pml, self.sigma_max_x, kappa_max, alpha_max, m, half=True)
        self.sigma_y_E, self.kappa_y_E, self.alpha_y_E = make_profile_arrays(
            ny, n_pml, self.sigma_max_y, kappa_max, alpha_max, m, half=False)
        self.sigma_y_H, self.kappa_y_H, self.alpha_y_H = make_profile_arrays(
            ny, n_pml, self.sigma_max_y, kappa_max, alpha_max, m, half=True)

        # Coefficients b, a (vectors).
        def coef(sigma, kappa, alpha, dt):
            denom_b = sigma / kappa + alpha
            b = np.exp(-denom_b * dt)
            denom_a = sigma * kappa + kappa ** 2 * alpha
            with np.errstate(divide='ignore', invalid='ignore'):
                a = np.where(np.abs(denom_a) > 1e-30,
                             sigma * (b - 1.0) / denom_a, 0.0)
            return b, a

        self.b_x_E, self.a_x_E = coef(self.sigma_x_E, self.kappa_x_E, self.alpha_x_E, dt)
        self.b_x_H, self.a_x_H = coef(self.sigma_x_H, self.kappa_x_H, self.alpha_x_H, dt)
        self.b_y_E, self.a_y_E = coef(self.sigma_y_E, self.kappa_y_E, self.alpha_y_E, dt)
        self.b_y_H, self.a_y_H = coef(self.sigma_y_H, self.kappa_y_H, self.alpha_y_H, dt)

        # ψ arrays — full-grid for clarity; non-PML cells stay at 0.
        self.psi_E_zx = np.zeros((nx, ny))    # at E position
        self.psi_E_zy = np.zeros((nx, ny))
        self.psi_H_yx = np.zeros((nx - 1, ny))  # at H_y position
        self.psi_H_xy = np.zeros((nx, ny - 1))  # at H_x position

    def apply_H_correction(self, sim):
        """Apply PML correction to H_x and H_y after the default H update.
        Uses E_z at time n; H storage was just updated to (n+1/2)."""
        dt, dx, dy = sim.dt, self.grid.dx, self.grid.dy

        # ∂_y E_z at H_x positions (i, j+½), shape = sim.Hx.shape = (nx, ny-1)
        dEz_dy = (sim.Ez[:, 1:] - sim.Ez[:, :-1]) / dy
        # Update ψ_H_xy in place. b_y_H, a_y_H are length ny-1.
        self.psi_H_xy *= self.b_y_H[None, :]
        self.psi_H_xy += self.a_y_H[None, :] * dEz_dy
        # PML correction to H_x: -dt * [((1/κ_y) - 1) ∂_y E_z + ψ_H_xy]
        sim.Hx += -dt * ((1.0 / self.kappa_y_H[None, :] - 1.0) * dEz_dy + self.psi_H_xy)

        # ∂_x E_z at H_y positions (i+½, j), shape = sim.Hy.shape = (nx-1, ny)
        dEz_dx = (sim.Ez[1:, :] - sim.Ez[:-1, :]) / dx
        self.psi_H_yx *= self.b_x_H[:, None]
        self.psi_H_yx += self.a_x_H[:, None] * dEz_dx
        sim.Hy += +dt * ((1.0 / self.kappa_x_H[:, None] - 1.0) * dEz_dx + self.psi_H_yx)

    def apply_E_correction(self, sim):
        """Apply PML correction to E_z after the default E update.
        Uses H_x, H_y at time (n+½); E storage was just updated to (n+1)."""
        dt, dx, dy = sim.dt, self.grid.dx, self.grid.dy

        # ∂_x H_y at E_z position (interior). Same indexing as default curl in update_E.
        dHy_dx = np.zeros_like(sim.Ez)
        dHy_dx[1:-1, 1:-1] = (sim.Hy[1:, 1:-1] - sim.Hy[:-1, 1:-1]) / dx
        # ∂_y H_x at E_z position.
        dHx_dy = np.zeros_like(sim.Ez)
        dHx_dy[1:-1, 1:-1] = (sim.Hx[1:-1, 1:] - sim.Hx[1:-1, :-1]) / dy

        # Update ψ_E_zx, ψ_E_zy.
        self.psi_E_zx *= self.b_x_E[:, None]
        self.psi_E_zx += self.a_x_E[:, None] * dHy_dx
        self.psi_E_zy *= self.b_y_E[None, :]
        self.psi_E_zy += self.a_y_E[None, :] * dHx_dy

        # PML correction to E_z.
        sim.Ez += dt * (
            (1.0 / self.kappa_x_E[:, None] - 1.0) * dHy_dx + self.psi_E_zx
            - (1.0 / self.kappa_y_E[None, :] - 1.0) * dHx_dy - self.psi_E_zy
        )


def add_uniform_static_B(sim, Bx: float, By: float):
    """Add a uniform static magnetic field (Bx, By) to the simulation.

    Use case: model wave-wave physics in a strong-field background, e.g. the
    overlap region between two magnetar dipole fields. The static field does
    not propagate (curl B_bg = 0), but it enters the QED Heisenberg-Euler
    constitutive D = (1 + 4κ(E² - |B|²)) E through |B|² = |B_bg + B_wave|²,
    which produces wave-wave cross-coupling enhanced by B_bg vs the bare
    vacuum HE coupling.

    Storage convention: HE material treats sim.Hx, sim.Hy as the B field, so
    a uniform addition here = uniform B background. For other materials the
    addition propagates as a static H, which in vacuum is equivalent.
    """
    sim.Hx += Bx
    sim.Hy += By


# --------------------------------------------------------------------------
# Hooks for the three nonlinear regimes (to be filled in subsequent steps)
# --------------------------------------------------------------------------

class KerrMaterial(Material):
    """Full nonlinear FDTD with instantaneous Kerr response (no dispersion):

        D = eps_r * E + chi3 * E^3.

    Update procedure each step (Joseph & Taflove 1991):
      1. From current E, recover D_old = eps_r * E + chi3 * E^3.
      2. Advance D via Ampere:  D_new = D_old + dt * curl H.
      3. Invert  chi3 * E^3 + eps_r * E - D_new = 0  for E (Newton).

    This is exact for the non-dispersive Kerr response and conserves D
    correctly across the time step. Newton typically converges in 2-3
    iterations starting from the previous E.

    Notes:
      - chi3 > 0 is self-focusing; chi3 < 0 is defocusing.
      - Stability requires chi3 * max|E|^2 < O(1) so that the cubic remains
        monotonic in E and Newton stays in its basin.
    """

    def __init__(self, eps_r: float = 1.0, chi3: float = 0.0,
                 newton_tol: float = 1e-12, newton_max_iter: int = 12):
        super().__init__(eps_r=eps_r)
        self.chi3 = chi3
        self.newton_tol = newton_tol
        self.newton_max_iter = newton_max_iter

    def update_E(self, Ez, Hx, Hy, dt, dx, dy, *, Hx_prev=None, Hy_prev=None):
        if self.chi3 == 0.0:
            return super().update_E(Ez, Hx, Hy, dt, dx, dy,
                                    Hx_prev=Hx_prev, Hy_prev=Hy_prev)

        curl_H = (Hy[1:, 1:-1] - Hy[:-1, 1:-1]) / dx \
               - (Hx[1:-1, 1:] - Hx[1:-1, :-1]) / dy

        E_old = Ez[1:-1, 1:-1]
        D_old = self.eps_r * E_old + self.chi3 * E_old ** 3
        D_new = D_old + dt * curl_H

        # Newton iterate to solve chi3 * E^3 + eps_r * E = D_new.
        E = E_old.copy()
        for _ in range(self.newton_max_iter):
            f = self.chi3 * E ** 3 + self.eps_r * E - D_new
            if np.max(np.abs(f)) < self.newton_tol:
                break
            fprime = 3.0 * self.chi3 * E ** 2 + self.eps_r
            E = E - f / fprime

        Ez[1:-1, 1:-1] = E


class RelativisticPlasmaMaterial(Material):
    """Maxwell + cold relativistic electron fluid (immobile neutralizing ions).

    Equations (natural units, c = eps0 = mu0 = m_e = e = 1):
        ∂_t n + ∇·(n v) = 0
        d_t p = -(E + v × B),     v = p / γ,   γ = √(1 + |p|^2)
        J = -n v                  (electron charge -1)
    Maxwell:
        ∂_t H = -∇ × E
        ∂_t E =  ∇ × H - J

    State (cell-centered, co-located with E_z):
        n  : electron number density,  shape (nx, ny), initial value n0
        px, py, pz : momentum density per electron, shape (nx, ny), initial 0

    Driver dimensionless parameters:
        omega_p^2 = n0   (plasma frequency squared, natural units)
        a_0       = E_amp / omega   (peak normalized vector potential)

    Time integration: forward Euler push for the fluid using E^n and B^{n+1/2}.
    Adequate for short runs; replace with Boris pusher if numerical heating
    becomes visible at long times.
    """

    def __init__(self, n0: float = 1.0):
        super().__init__(eps_r=1.0, mu_r=1.0)
        self.n0 = float(n0)
        # Allocated in init_state.
        self.n = None
        self.px = self.py = self.pz = None

    def init_state(self, grid):
        shape = (grid.nx, grid.ny)
        self.n = np.full(shape, self.n0)
        self.px = np.zeros(shape)
        self.py = np.zeros(shape)
        self.pz = np.zeros(shape)

    def _gamma(self):
        return np.sqrt(1.0 + self.px ** 2 + self.py ** 2 + self.pz ** 2)

    def _velocity(self):
        g = self._gamma()
        return self.px / g, self.py / g, self.pz / g

    def _B_at_centers(self, Hx, Hy):
        """Interpolate H from Yee positions to cell centers (where Ez lives)."""
        Bx = np.zeros((Hx.shape[0], Hx.shape[1] + 1))   # (nx, ny)
        By = np.zeros((Hy.shape[0] + 1, Hy.shape[1]))   # (nx, ny)
        # Hx[i, j] sits at (i, j+½); average neighbours along y.
        Bx[:, 1:-1] = 0.5 * (Hx[:, :-1] + Hx[:, 1:])
        Bx[:, 0]    = Hx[:, 0]
        Bx[:, -1]   = Hx[:, -1]
        # Hy[i, j] sits at (i+½, j); average neighbours along x.
        By[1:-1, :] = 0.5 * (Hy[:-1, :] + Hy[1:, :])
        By[0,    :] = Hy[0, :]
        By[-1,   :] = Hy[-1, :]
        return Bx, By

    def step_state(self, Ez, Hx, Hy, dt, dx, dy, *, Hx_prev=None, Hy_prev=None):
        """Advance fluid (p, n) by one time step using a relativistic Boris
        pusher for the momentum equation and upwind continuity for n.

        Boris algorithm (electron, charge -1, mass 1):
          1. Half-E kick:  p⁻ = p^n - (Δt/2) E
          2. Compute γ⁻   from |p⁻|.
          3. t = -B Δt / (2 γ⁻),  s = 2t / (1 + |t|²)
          4. p' = p⁻ + p⁻ × t
          5. p⁺ = p⁻ + p' × s         (rotation, exact energy conservation)
          6. p^{n+1} = p⁺ - (Δt/2) E

        The rotation step (4-5) preserves |p|² exactly (modulo floating point),
        eliminating the secular numerical heating that plagued the forward-Euler
        push. Time-centered B is used when H_prev is supplied: averaging H^{n-1/2}
        and H^{n+1/2} yields B^n, matching the time at which E^n is given to
        the pusher.
        """
        # Time-centered B (mid-step); fall back to post-update H if pre not given.
        if Hx_prev is not None and Hy_prev is not None:
            Hx_eff = 0.5 * (Hx_prev + Hx)
            Hy_eff = 0.5 * (Hy_prev + Hy)
        else:
            Hx_eff, Hy_eff = Hx, Hy
        Bx, By = self._B_at_centers(Hx_eff, Hy_eff)

        # Snapshot velocity at start-of-step (used only for continuity below).
        vx0, vy0, vz0 = self._velocity()
        n0 = self.n.copy()

        # ---- Boris push ----
        half_E = 0.5 * dt * Ez   # F_E_z = -E_z; E_x = E_y = 0 in TMz
        # 1. Half-E kick.
        pmx = self.px
        pmy = self.py
        pmz = self.pz - half_E

        # 2. γ⁻
        gamma_minus = np.sqrt(1.0 + pmx ** 2 + pmy ** 2 + pmz ** 2)

        # 3. t and s.
        factor = -dt / (2.0 * gamma_minus)
        tx = factor * Bx
        ty = factor * By
        # |t|² = tx² + ty² (tz = 0)
        denom = 1.0 + tx ** 2 + ty ** 2
        sx = 2.0 * tx / denom
        sy = 2.0 * ty / denom

        # 4. p' = p⁻ + p⁻ × t   (with t_z = 0)
        # (pm × t) = (-pmz·ty, pmz·tx, pmx·ty - pmy·tx)
        ppx = pmx + (-pmz * ty)
        ppy = pmy + (pmz * tx)
        ppz = pmz + (pmx * ty - pmy * tx)

        # 5. p⁺ = p⁻ + p' × s   (with s_z = 0)
        pplus_x = pmx + (-ppz * sy)
        pplus_y = pmy + (ppz * sx)
        pplus_z = pmz + (ppx * sy - ppy * sx)

        # 6. Final half-E kick.
        self.px = pplus_x
        self.py = pplus_y
        self.pz = pplus_z - half_E

        # Continuity: n^{n+1} = n^n - dt * ∇·(n v), first-order upwind donor-cell.
        # Centered differencing is unstable for pure advection (even-odd decoupling),
        # so we use the local sign of v to pick backward/forward stencils.
        # Use the *start-of-step* velocity (vx0, vy0) — consistent with how the
        # Boris push centers the rest of the dynamics on the half step.
        nvx = n0 * vx0
        nvy = n0 * vy0
        vx_i = vx0[1:-1, 1:-1]
        vy_i = vy0[1:-1, 1:-1]
        dx_back = (nvx[1:-1, 1:-1] - nvx[:-2, 1:-1]) / dx
        dx_fwd  = (nvx[2:,    1:-1] - nvx[1:-1, 1:-1]) / dx
        dy_back = (nvy[1:-1, 1:-1] - nvy[1:-1, :-2]) / dy
        dy_fwd  = (nvy[1:-1, 2:]   - nvy[1:-1, 1:-1]) / dy
        div_nv_int = (np.where(vx_i > 0.0, dx_back, dx_fwd)
                      + np.where(vy_i > 0.0, dy_back, dy_fwd))
        self.n[1:-1, 1:-1] -= dt * div_nv_int
        # Floor density to keep gamma finite if a wave evacuates a region.
        np.maximum(self.n, 1e-6 * self.n0, out=self.n)

    def update_E(self, Ez, Hx, Hy, dt, dx, dy, *, Hx_prev=None, Hy_prev=None):
        """Ampère with plasma current J_z = -n v_z (cell-centered)."""
        curl_H = (Hy[1:, 1:-1] - Hy[:-1, 1:-1]) / dx \
               - (Hx[1:-1, 1:] - Hx[1:-1, :-1]) / dy
        _, _, vz = self._velocity()
        Jz = -self.n * vz                       # full grid; we use the interior slice
        Ez[1:-1, 1:-1] += dt * (curl_H - Jz[1:-1, 1:-1]) / self.eps_r


# Backwards-compatible alias for the stub from Step 1.
ColdPlasmaMaterial = RelativisticPlasmaMaterial


class HeisenbergEulerMaterial(Material):
    """One-loop QED vacuum nonlinearity, F^2 channel only.

    For TMz (E along z, B in plane), the pseudoscalar invariant G = E·B
    vanishes identically, so we keep only the F^2 term:

        L = (1/2)(E^2 - B^2) + kappa * (E^2 - B^2)^2

    yielding the constitutive relations
        D = (1 + chi) E,    H = (1 + chi) B,    chi = 4 kappa (E^2 - B^2).

    Important physical fact: for a single plane wave |E| = |B|, so chi = 0
    and the wave does not see the QED correction at all. The nonlinearity
    only activates in regions where two or more waves overlap and locally
    break the |E| = |B| equality. This makes Heisenberg-Euler the cleanest
    possible "no-self-action, only wave-wave" coupling channel.

    `kappa` absorbs the QED prefactor (2 alpha^2 / 45 m_e^4 in real units).
    For the demo we treat it as a free parameter; in a thesis-quality run
    it should be set from the chosen field-strength normalization so that
    chi becomes O(1) only as |E| -> Schwinger field E_S.

    Storage convention: this class treats the simulation's `Hx, Hy` arrays
    as the *B* field. update_H is unchanged (Faraday's law has the same
    form whether the variable is called H or B in vacuum). update_E is the
    only place where H differs from B: locally compute H = (1 + chi) B,
    take its curl, advance D = (1 + chi) E by Ampere, then Newton-invert
    for E.
    """

    def __init__(self, kappa: float = 0.0,
                 newton_tol: float = 1e-12, newton_max_iter: int = 12,
                 pc_max_iters: int = 3, pc_tol: float = 1e-10):
        super().__init__(eps_r=1.0, mu_r=1.0)
        self.kappa = kappa
        self.newton_tol = newton_tol
        self.newton_max_iter = newton_max_iter
        # Predictor-corrector parameters: pc_max_iters > 0 enables iterative
        # update of B² (used in the constitutive cubic) toward its time-n+1
        # value via a one-step Faraday prediction of H at n+3/2. pc_max_iters=0
        # disables, recovering the previous single-shot solve.
        self.pc_max_iters = pc_max_iters
        self.pc_tol = pc_tol

    @staticmethod
    def _Bx_cc(Bx, shape):
        """Average B_x (at (i, j+½)) to cell centers (i, j integer)."""
        out = np.zeros(shape)
        out[:, 1:-1] = 0.5 * (Bx[:, :-1] + Bx[:, 1:])
        out[:, 0]    = Bx[:, 0]
        out[:, -1]   = Bx[:, -1]
        return out

    @staticmethod
    def _By_cc(By, shape):
        """Average B_y (at (i+½, j)) to cell centers (i, j integer)."""
        out = np.zeros(shape)
        out[1:-1, :] = 0.5 * (By[:-1, :] + By[1:, :])
        out[0,    :] = By[0, :]
        out[-1,   :] = By[-1, :]
        return out

    def update_E(self, Ez, Hx, Hy, dt, dx, dy, *, Hx_prev=None, Hy_prev=None):
        if self.kappa == 0.0:
            return super().update_E(Ez, Hx, Hy, dt, dx, dy,
                                    Hx_prev=Hx_prev, Hy_prev=Hy_prev)

        # Time-centering: use H at integer step n (== average of n-1/2 and n+1/2)
        # for B² in the constitutive relation. F = (E²-B²)/2 must vanish for a
        # single plane wave; that requires E^n and B^n at the SAME time.
        # Without this centering, E^n vs B^{n+1/2} produces F ~ ω·dt even for
        # a single wave, which then resonantly drives the QED nonlinearity and
        # accumulates over many steps. (storage convention: Hx, Hy carry B.)
        if Hx_prev is not None and Hy_prev is not None:
            Bx = 0.5 * (Hx_prev + Hx)
            By = 0.5 * (Hy_prev + Hy)
        else:
            Bx, By = Hx, Hy

        # |B|^2 at cell centers (co-located with E_z) for the constitutive relation.
        Bx_cc = self._Bx_cc(Bx, Ez.shape)
        By_cc = self._By_cc(By, Ez.shape)
        Bsq_cc = Bx_cc ** 2 + By_cc ** 2
        chi_cc = 4.0 * self.kappa * (Ez ** 2 - Bsq_cc)
        D_old = (1.0 + chi_cc) * Ez

        # ---- Compute H_eff = (1+chi) B at the Yee positions, for curl. ----
        # Use the post-update H (Hx, Hy) here so the curl_H entering Ampere
        # is at time n+1/2 (correct leapfrog timing).
        Bx_post, By_post = Hx, Hy

        Ez_at_Bx = 0.5 * (Ez[:, :-1] + Ez[:, 1:])
        By_x_int = np.zeros_like(Ez)
        By_x_int[1:-1, :] = 0.5 * (By_post[:-1, :] + By_post[1:, :])
        By_x_int[0,    :] = By_post[0,  :]
        By_x_int[-1,   :] = By_post[-1, :]
        By_at_Bx = 0.5 * (By_x_int[:, :-1] + By_x_int[:, 1:])
        Bsq_at_Bx = Bx_post ** 2 + By_at_Bx ** 2
        chi_at_Bx = 4.0 * self.kappa * (Ez_at_Bx ** 2 - Bsq_at_Bx)
        Hx_eff = (1.0 + chi_at_Bx) * Bx_post

        Ez_at_By = 0.5 * (Ez[:-1, :] + Ez[1:, :])
        Bx_y_int = np.zeros_like(Ez)
        Bx_y_int[:, 1:-1] = 0.5 * (Bx_post[:, :-1] + Bx_post[:, 1:])
        Bx_y_int[:, 0]    = Bx_post[:, 0]
        Bx_y_int[:, -1]   = Bx_post[:, -1]
        Bx_at_By = 0.5 * (Bx_y_int[:-1, :] + Bx_y_int[1:, :])
        Bsq_at_By = Bx_at_By ** 2 + By_post ** 2
        chi_at_By = 4.0 * self.kappa * (Ez_at_By ** 2 - Bsq_at_By)
        Hy_eff = (1.0 + chi_at_By) * By_post

        # ---- Ampere: D^{n+1} = D^n + dt * curl(H_eff^{n+1/2}). ----
        curl_H = (Hy_eff[1:, 1:-1] - Hy_eff[:-1, 1:-1]) / dx \
               - (Hx_eff[1:-1, 1:] - Hx_eff[1:-1, :-1]) / dy
        D_new_int = D_old[1:-1, 1:-1] + dt * curl_H

        # ---- Predictor-corrector inversion ----
        # Goal: solve D^{n+1} = E^{n+1} (1 + 4κ((E^{n+1})² - (B^{n+1})²)) for
        # E^{n+1} self-consistently with B^{n+1}. We don't have B^{n+1} (only
        # B^n from time-centering above and B^{n+1/2} from the post-update H);
        # the predictor-corrector estimates B^{n+1} by predicting H^{n+3/2}
        # via a one-step Faraday using the current E^{n+1} estimate, then
        # averaging with H^{n+1/2}. We iterate to convergence.
        #
        # Without this iteration (pc_max_iters=0), the cubic uses B^n
        # (off by Δt from where it should be), producing a small but secular
        # single-wave self-action artifact even when F = (E²-B²)/2 vanishes
        # analytically (e.g. for a single plane wave).
        a = 4.0 * self.kappa
        Bsq_int = Bsq_cc[1:-1, 1:-1]    # initial estimate: B² at time n
        E = Ez[1:-1, 1:-1].copy()
        for pc_iter in range(max(1, self.pc_max_iters)):
            E_old = E.copy()
            # Inner Newton solve on the cubic with current B² estimate.
            for _ in range(self.newton_max_iter):
                f = a * E ** 3 + (1.0 - a * Bsq_int) * E - D_new_int
                if np.max(np.abs(f)) < self.newton_tol:
                    break
                fprime = 3.0 * a * E ** 2 + (1.0 - a * Bsq_int)
                E = E - f / fprime
            if self.pc_max_iters == 0:
                break       # behave as the original non-iterative scheme
            # Check convergence.
            delta = np.max(np.abs(E - E_old))
            if pc_iter > 0 and delta < self.pc_tol:
                break
            # Update B² estimate to B^{n+1} via predicted H^{n+3/2}.
            # Need E^{n+1} on full grid (boundary unchanged for the prediction).
            Ez_full = Ez.copy()
            Ez_full[1:-1, 1:-1] = E
            # Predicted H at n+3/2 from Faraday:
            #   H_x^{n+3/2} = H_x^{n+1/2} - dt · ∂_y E^{n+1}
            #   H_y^{n+3/2} = H_y^{n+1/2} + dt · ∂_x E^{n+1}
            # Time-centered B at n+1 = 0.5·(H^{n+1/2} + H^{n+3/2}) =
            #   H^{n+1/2} ± 0.5·dt · ∂E^{n+1}/∂_perp.
            Bx_n1 = Hx - 0.5 * dt * (Ez_full[:, 1:] - Ez_full[:, :-1]) / dy
            By_n1 = Hy + 0.5 * dt * (Ez_full[1:, :] - Ez_full[:-1, :]) / dx
            Bx_cc_n1 = self._Bx_cc(Bx_n1, Ez.shape)
            By_cc_n1 = self._By_cc(By_n1, Ez.shape)
            Bsq_int = (Bx_cc_n1 ** 2 + By_cc_n1 ** 2)[1:-1, 1:-1]
        Ez[1:-1, 1:-1] = E
