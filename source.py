"""TF/SF (total-field / scattered-field) plane-wave source for FDTD2D.

Replaces the initial-condition Gaussian-packet seeding (`add_gaussian_packet`)
with a continuous source injected through a closed rectangular contour inside
the grid. Inside the contour: total field = incident + scattered. Outside:
scattered field only. Maxwell's equations are linear in the field, so the
"source" is implemented as small corrections at the four contour edges
where Yee curls cross the TF/SF boundary.

References: Taflove & Hagness, "Computational Electrodynamics" Ch. 5.

Sign convention (TMz):
    For a wave propagating in direction k̂ = (cos θ, sin θ) with
    E_z = A · ramp(t) · cos(k_x x + k_y y − ω t + φ):
        H_x_inc = −sin θ · A · ramp(t) · cos(...)
        H_y_inc = −cos θ · A · ramp(t) · cos(...)
    so that ∂_t H = −∇×E and Poynting points along +k̂.

Time staggering: E_inc evaluated at integer steps (t = n dt) is used for
the H-update corrections; H_inc evaluated at half-integer steps
(t = (n+1/2) dt) is used for the E-update corrections.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class PlaneWaveSource:
    """Plane-wave TF/SF source on a rectangular contour.

    Parameters
    ----------
    ia, ja, ib, jb : int
        Integer grid indices defining the TF region. ia < ib, ja < jb.
        The TF region is {(i, j) : ia ≤ i ≤ ib, ja ≤ j ≤ jb}; outside is SF.
        Place the contour at least 4–5 cells from the grid boundary so the
        Mur ABC has room to absorb the outgoing scattered field.
    angle_deg : float
        Propagation direction θ, measured from +x.
    wavelength : float
        Carrier wavelength in grid length units (natural units, c = 1).
    amplitude : float
        Peak E_z amplitude of the incident wave.
    ramp_time : float
        Half-period over which the wave is smoothly ramped from 0 to peak
        amplitude using a 1−cos profile. Avoids initial-condition shocks.
    phase : float
        Carrier phase offset (radians).
    """

    ia: int
    ja: int
    ib: int
    jb: int
    angle_deg: float
    wavelength: float
    amplitude: float
    ramp_time: float = 5.0
    phase: float = 0.0

    def __post_init__(self):
        self.theta = float(np.deg2rad(self.angle_deg))
        self.k = 2.0 * np.pi / self.wavelength
        self.kx = self.k * np.cos(self.theta)
        self.ky = self.k * np.sin(self.theta)
        self.omega = self.k                # vacuum dispersion, c = 1
        self.cos_theta = np.cos(self.theta)
        self.sin_theta = np.sin(self.theta)

    # -- Incident analytic fields ------------------------------------------
    def _ramp(self, t: float) -> float:
        if t <= 0:
            return 0.0
        if t >= self.ramp_time:
            return 1.0
        return 0.5 * (1.0 - np.cos(np.pi * t / self.ramp_time))

    def _carrier(self, x, y, t):
        return np.cos(self.kx * x + self.ky * y - self.omega * t + self.phase)

    def Ez_inc(self, x, y, t):
        return self.amplitude * self._ramp(t) * self._carrier(x, y, t)

    def Hx_inc(self, x, y, t):
        # Consistent solution of FDTD update for +k̂ propagation:
        #   ∂_t H_x = -∂_y E_z   ⇒  H_x = +sin θ · E_z
        #   ∂_t H_y = +∂_x E_z   ⇒  H_y = -cos θ · E_z
        # (Verified by computing Poynting S = E × H along k̂.)
        return +self.sin_theta * self.amplitude * self._ramp(t) * self._carrier(x, y, t)

    def Hy_inc(self, x, y, t):
        return -self.cos_theta * self.amplitude * self._ramp(t) * self._carrier(x, y, t)

    # -- TF/SF corrections to FDTD updates ---------------------------------
    # Standard Taflove TF/SF signs: TF storage inside contour, SF outside.
    def apply_E_corrections(self, sim, t_eval):
        g = sim.g
        dt = sim.dt
        ia, ja, ib, jb = self.ia, self.ja, self.ib, self.jb

        ys = (np.arange(ja, jb + 1)) * g.dy
        # Left edge: curl_TF needs Hy(ia-1/2)_TF = SF + inc, correction = -inc/dx.
        x_l = (ia - 0.5) * g.dx
        sim.Ez[ia, ja:jb + 1] -= (dt / g.dx) * self.Hy_inc(x_l, ys, t_eval)
        x_r = (ib + 0.5) * g.dx
        sim.Ez[ib, ja:jb + 1] += (dt / g.dx) * self.Hy_inc(x_r, ys, t_eval)

        xs = np.arange(ia, ib + 1) * g.dx
        y_b = (ja - 0.5) * g.dy
        sim.Ez[ia:ib + 1, ja] += (dt / g.dy) * self.Hx_inc(xs, y_b, t_eval)
        y_t = (jb + 0.5) * g.dy
        sim.Ez[ia:ib + 1, jb] -= (dt / g.dy) * self.Hx_inc(xs, y_t, t_eval)

    def apply_H_corrections(self, sim, t_eval):
        g = sim.g
        dt = sim.dt
        ia, ja, ib, jb = self.ia, self.ja, self.ib, self.jb

        ys = np.arange(ja, jb + 1) * g.dy
        x_at_Ez = ia * g.dx
        sim.Hy[ia - 1, ja:jb + 1] -= (dt / g.dx) * self.Ez_inc(x_at_Ez, ys, t_eval)
        x_at_Ez = ib * g.dx
        sim.Hy[ib, ja:jb + 1] += (dt / g.dx) * self.Ez_inc(x_at_Ez, ys, t_eval)

        xs = np.arange(ia, ib + 1) * g.dx
        y_at_Ez = ja * g.dy
        sim.Hx[ia:ib + 1, ja - 1] += (dt / g.dy) * self.Ez_inc(xs, y_at_Ez, t_eval)
        y_at_Ez = jb * g.dy
        sim.Hx[ia:ib + 1, jb] -= (dt / g.dy) * self.Ez_inc(xs, y_at_Ez, t_eval)
