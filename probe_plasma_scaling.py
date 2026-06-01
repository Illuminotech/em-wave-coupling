"""Probe: does δn/n0 scale like a_0^2 as expected from perturbative theory?

If yes: model is right, we just need to keep amplitudes in the perturbative window.
If no: bug — investigate fluid push, boundary, or time-step.
"""
from __future__ import annotations
import numpy as np
from fdtd2d import FDTD2D, Grid, RelativisticPlasmaMaterial, add_gaussian_packet


def run_amp(amp: float, n_steps: int = 200) -> dict:
    sim = FDTD2D(grid=Grid(nx=240, ny=200, dx=0.05, dy=0.05),
                 material=RelativisticPlasmaMaterial(n0=4.0), cfl=0.5)
    add_gaussian_packet(sim,
        wavelength=1.0, angle_deg=20.0, amplitude=amp,
        waist=4.0, pulse_length=4.0, center=(6.0, 5.0))
    history = []
    for n in range(n_steps + 1):
        history.append(np.max(np.abs(sim.mat.n - sim.mat.n0)) / sim.mat.n0)
        if n < n_steps:
            sim.step()
    return dict(amp=amp, dn_max=np.array(history))


if __name__ == '__main__':
    print('Probe: scan amplitude, look for dn ~ a_0^2 = (amp/omega)^2')
    print(f'{"amp":>8} {"a_0":>8} {"a_0^2":>10} {"max dn":>10} {"dn/a_0^2":>10}')
    for amp in [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]:
        out = run_amp(amp)
        a0 = amp / (2 * np.pi)        # ω = 2π for λ = 1
        a02 = a0 ** 2
        peak = float(out['dn_max'].max())
        print(f'{amp:>8.3f} {a0:>8.4f} {a02:>10.3e} {peak:>10.3e} {peak/a02:>10.2e}')
