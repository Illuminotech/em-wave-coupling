# em-wave-coupling

A 2D finite-difference time-domain (FDTD) framework for studying nonlinear electromagnetic wave-wave coupling across four physics regimes: linear vacuum, Kerr media, cold relativistic plasma, and Heisenberg-Euler QED vacuum.

Companion code for the paper:
> **Quantitative diagnostics of wave-wave electromagnetic coupling channels: a unified FDTD framework with Kerr, plasma, and Heisenberg-Euler regimes, with application to magnetar magnetospheres**
> Michael F. Golden (2026)

## Key idea

How do you cleanly measure two electromagnetic waves interacting with each other, separated from each wave's self-interaction with the medium?

This framework introduces a **linearity residual diagnostic**: run three simulations (wave A alone, wave B alone, both together) and measure the deviation from superposition:

```
R(t) = ‖E_AB(t) − E_A(t) − E_B(t)‖_∞ / ‖E_AB(t)‖_∞
```

In a linear medium, R ≈ 10⁻¹⁵ (machine precision). In a nonlinear medium, R isolates only the terms that depend on both waves simultaneously — pure wave-wave coupling with no self-action contamination.

## Requirements

- Python 3.10+
- NumPy
- Matplotlib

```bash
pip install numpy matplotlib
```

## Quick start

```bash
# 1. Verify linear superposition (baseline)
python run_two_wave.py

# 2. Observe nonlinear wave-wave coupling
python run_kerr.py       # Kerr χ⁽³⁾ medium
python run_plasma.py     # Cold relativistic plasma
python run_qed.py        # Heisenberg-Euler QED vacuum

# 3. Mixed-polarization QED (G² channel)
python run_mixed_demo.py

# 4. Magnetar magnetosphere scenario
python run_magnetar.py
```

Each script prints a residual table to the terminal and saves PNG figures to the working directory.

## Physics regimes

| Regime | Material class | Constitutive relation | Coupling mechanism |
|--------|---------------|----------------------|-------------------|
| Linear vacuum | `Material` | D = εE | None (validation baseline) |
| Kerr χ⁽³⁾ | `KerrMaterial` | D = εE + χ⁽³⁾E³ | Cross-phase modulation, third-harmonic generation |
| Cold plasma | `RelativisticPlasmaMaterial` | Maxwell + fluid equations | Ponderomotive density modulation, stimulated Raman scattering |
| QED (F²) | `HeisenbergEulerMaterial` | D = (1 + 4κF)E | Photon-photon scattering via virtual e⁺e⁻ pairs |
| QED (G²) | `FDTD2D_Mixed` | δE = −2κ_G(E·B)B | Cross-polarized photon-photon scattering |

All simulations use natural units (c = ε₀ = μ₀ = 1) on a 2D Yee staggered grid with leapfrog time integration.

## Project structure

```
em_sim/
│
│  Core library
├── fdtd2d.py               # FDTD engine, material classes, CPML boundaries
├── fdtd2d_mixed.py         # Mixed TM+TE polarization extension
├── source.py               # Total-field/scattered-field plane-wave source
├── diagnostics.py          # Probe recording and spectral (FFT) analysis
│
│  Simulation drivers
├── run_two_wave.py          # Linear vacuum baseline
├── run_kerr.py              # Kerr nonlinearity
├── run_plasma.py            # Relativistic cold plasma
├── run_qed.py               # Heisenberg-Euler QED (F² channel)
├── run_mixed_demo.py        # Mixed-mode TM+TE (G² channel)
├── run_magnetar.py          # QED with static background B field
├── run_spectra.py           # Spectral analysis of harmonic generation
├── scan_magnetar_bg.py      # Background field strength sweep
├── probe_plasma_scaling.py  # Plasma density scaling study
│
│  Validation
├── validate_boris.py        # Boris pusher energy conservation
├── validate_cpml.py         # Convolutional PML boundary absorption
├── validate_qed_pc.py       # Predictor-corrector for QED constitutive
└── run_tfsf_validate.py     # TF/SF source injection
```

## Core API

### Grid and solver

```python
from fdtd2d import Grid, FDTD2D, Material

grid = Grid(nx=320, ny=240, dx=0.05, dy=0.05)
sim = FDTD2D(grid, material=Material(), cfl=0.7, boundary='cpml')
```

### Adding wave packets

```python
from fdtd2d import add_gaussian_packet

add_gaussian_packet(
    sim,
    center=(8.0, 6.0),
    wavelength=1.0,
    angle_deg=20.0,
    amplitude=1.0,
    waist=4.0,
    pulse_length=4.0,
)
```

### Swapping physics

Each regime plugs into the same solver through the `Material` interface:

```python
from fdtd2d import KerrMaterial, RelativisticPlasmaMaterial, HeisenbergEulerMaterial

# Kerr medium
sim = FDTD2D(grid, material=KerrMaterial(chi3=0.1))

# Relativistic plasma
sim = FDTD2D(grid, material=RelativisticPlasmaMaterial(n0=4.0), cfl=0.5)

# QED vacuum
sim = FDTD2D(grid, material=HeisenbergEulerMaterial(kappa=0.01))
```

### Running and measuring

```python
import numpy as np

for step in range(n_steps):
    sim.step()

# Linearity residual
R = np.max(np.abs(Ez_AB - Ez_A - Ez_B)) / np.max(np.abs(Ez_AB))
```

### Spectral diagnostics

```python
from diagnostics import ProbeRecorder

probe = ProbeRecorder(positions=[(10.0, 6.0)], grid=grid)
sim.add_probe(probe)

# After simulation
omega, power = probe.spectrum(0)
```

## Numerical methods

**Boris pusher** — The plasma module uses a half-kick/rotation/half-kick split for the Lorentz force, conserving energy to machine precision (relative error ~10⁻¹⁴). The naive forward-Euler alternative produces ~40× energy growth over 200 cyclotron periods.

**Predictor-corrector** — The QED constitutive relation is implicit (B² at time step n requires the H field at n+½, which depends on E at n, which depends on B²). An iterative predictor-corrector reduces the resulting self-action artifact by 8.4×.

**CPML boundaries** — Convolutional perfectly-matched-layer absorbers achieve ~10⁻³ reflection at oblique incidence. Note: initial conditions must not overlap the PML region, or the auxiliary memory variables become destabilized.

## Reproducing paper figures

The simulation drivers generate all figures from the paper. Run them in sequence:

```bash
python validate_boris.py        # Fig: Boris vs Euler energy drift
python validate_cpml.py         # Fig: CPML vs Mur absorption
python validate_qed_pc.py       # Fig: Predictor-corrector improvement
python run_two_wave.py          # Fig: Linear baseline snapshots
python run_kerr.py              # Fig: Kerr coupling + residual
python run_plasma.py            # Fig: Plasma coupling + density
python run_qed.py               # Fig: QED coupling + angle scan
python run_mixed_demo.py        # Fig: F² vs G² channel scaling
python run_magnetar.py          # Fig: Magnetar background scan
python run_spectra.py           # Fig: Spectral 2×2 panel
```

Typical runtime is 1-5 minutes per script on a modern laptop.

## License

MIT
