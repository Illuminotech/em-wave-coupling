"""Time-domain probes and spectral diagnostics for the FDTD simulations.

A `ProbeRecorder` is registered with an `FDTD2D` simulation and samples
E_z(t) at one or more spatial locations every step. After the run, FFT
of each probe's time series yields a frequency-domain spectrum showing
mixing peaks (third-harmonic in Kerr, Stokes/anti-Stokes in plasma,
photon-photon mixing in QED) that are otherwise hidden in the field maps.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np

from fdtd2d import Grid


@dataclass
class ProbeRecorder:
    positions: Sequence[tuple[float, float]]
    grid: Grid
    times: list[float] = field(default_factory=list)
    values: list[list[float]] = field(default_factory=list)

    def __post_init__(self):
        self.indices = [
            (max(0, min(self.grid.nx - 1, int(round(x / self.grid.dx)))),
             max(0, min(self.grid.ny - 1, int(round(y / self.grid.dy)))))
            for x, y in self.positions
        ]
        self.values = [[] for _ in self.positions]

    def record(self, t: float, Ez: np.ndarray):
        self.times.append(float(t))
        for i, (ix, iy) in enumerate(self.indices):
            self.values[i].append(float(Ez[ix, iy]))

    def time_array(self) -> np.ndarray:
        return np.asarray(self.times)

    def signal(self, probe_idx: int) -> np.ndarray:
        return np.asarray(self.values[probe_idx])

    def spectrum(self, probe_idx: int, *, window: str = 'hann',
                 detrend: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Return (omega, |FFT|^2) for the probe's E_z(t) time series.

        - window: 'hann' tapers the time-domain signal to suppress
          spectral leakage from the abrupt start/end of the recorded interval.
        - detrend: subtract the time-domain mean before FFT.
        """
        sig = self.signal(probe_idx).astype(float)
        if detrend:
            sig = sig - sig.mean()
        if window == 'hann':
            sig = sig * np.hanning(len(sig))
        elif window is not None and window != 'rect':
            raise ValueError(f'unknown window {window!r}')

        # Sampling: assume uniform dt from the recorded times.
        ts = self.time_array()
        dt = (ts[-1] - ts[0]) / (len(ts) - 1) if len(ts) > 1 else 1.0
        spec = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(len(sig), d=dt)         # cycles per time unit
        omega = 2.0 * np.pi * freqs
        power = np.abs(spec) ** 2
        return omega, power
