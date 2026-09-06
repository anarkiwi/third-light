"""Spark audio: the channel's heat release as a thermoacoustic simple source.

Heat at rate P is the volume source V = (gamma - 1) E / (gamma p0), E its integral,
radiating p = rho0 Vddot(t - r/c) / (4 pi r); with c^2 = gamma p0 / rho0 the two
collapse to p(r, t) = (gamma - 1) Pdot(t - r/c) / (4 pi c^2 r). See §3.7 of design.
"""

import math
import wave

import numpy as np
from scipy.integrate import cumulative_trapezoid

GAMMA = 1.4
AMBIENT_PRESSURE = 101325.0
AMBIENT_DENSITY = 1.2041
SOUND_SPEED = math.sqrt(GAMMA * AMBIENT_PRESSURE / AMBIENT_DENSITY)
SAMPLE_RATE = 48000
DISTANCE = 1.0
FULL_SCALE = 32767


def thermoacoustic_volume(energy):
    """Volume a heat release displaces at ambient pressure, m^3.

    The constant-pressure expansion of an ideal gas: (gamma - 1) / gamma of the
    heat does the work p0 dV and the rest raises the internal energy.
    """
    return (GAMMA - 1.0) * np.asarray(energy, dtype=float) / (GAMMA * AMBIENT_PRESSURE)


def _mean_power(t, power, rate, energy=None):
    """Event-stepped channel power as its mean over each audio sample, W.

    The energy is the ledger's own and the samples are the differences of it
    interpolated, so no heat is created or lost by the resampling, and the
    boxcar's nulls at every multiple of the rate keep the carrier out of the band.
    """
    if energy is None:
        energy = cumulative_trapezoid(power, t, initial=0.0)
    count = max(int(round((t[-1] - t[0]) * rate)) + 1, 2)
    edges = t[0] + (np.arange(count + 1) - 0.5) / rate
    return np.diff(np.interp(edges, t, energy)) * rate


def bang(result, distance=DISTANCE, rate=SAMPLE_RATE):
    """Pressure signature of one run's channel dissipation at ``distance``, Pa.

    Sample zero is the run's first sample, emitted; the arrival sits
    ``distance / SOUND_SPEED`` later, quantised to the grid. Pdot is a centred
    difference, local where a spectral one imposes a periodicity a burst has not.
    """
    power = _mean_power(
        np.asarray(result.t, dtype=float),
        np.asarray(result.streamer_power, dtype=float),
        rate,
        np.concatenate([[0.0], np.cumsum(result.channel_energies)]),
    )
    gain = (GAMMA - 1.0) / (4.0 * math.pi * SOUND_SPEED**2 * distance)
    delay = int(round(distance / SOUND_SPEED * rate))
    return np.concatenate([np.zeros(delay), gain * np.gradient(power, 1.0 / rate)])


def render(result, schedule, duration, distance=DISTANCE, rate=SAMPLE_RATE):
    """Audio of an interrupter or ``Melody`` schedule, one bang per burst, Pa.

    The bangs land on the schedule's own edges that leave the gate enabled, by one
    scatter-add of the signature over every start at once: a MIDI program is 10^4
    bursts, and its offsets are as much an array as its edges are.
    """
    signature = bang(result, distance, rate)
    edges = np.asarray(schedule.edges(duration), dtype=float)
    starts = edges[schedule.active(np.nextafter(edges, math.inf))]
    index = np.rint(starts * rate).astype(np.intp)[:, None] + np.arange(signature.size)
    return np.bincount(
        index.ravel(),
        np.broadcast_to(signature, index.shape).ravel(),
        minlength=int(round(duration * rate)) + signature.size,
    )


def normalise(pressure, peak=1.0):
    """Waveform scaled so its largest excursion is ``peak``; silence stays silence."""
    pressure = np.asarray(pressure, dtype=float)
    largest = np.abs(pressure).max(initial=0.0)
    return pressure * (peak / largest) if largest > 0.0 else np.zeros_like(pressure)


def write_wav(path, pressure, rate=SAMPLE_RATE):
    """Write a normalised waveform to ``path`` as 16-bit mono PCM at ``rate``."""
    samples = np.rint(normalise(pressure) * FULL_SCALE).astype("<i2")
    with wave.Wave_write(str(path)) as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(rate))
        handle.writeframes(samples.tobytes())
