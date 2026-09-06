"""Thermoacoustic simple source, burst rendering and WAV output."""

import math
import wave
from dataclasses import dataclass

import numpy as np
import pytest

from thirdlight.acoustics import (
    AMBIENT_DENSITY,
    AMBIENT_PRESSURE,
    FULL_SCALE,
    GAMMA,
    SOUND_SPEED,
    _mean_power,  # pylint: disable=protected-access
    bang,
    normalise,
    render,
    thermoacoustic_volume,
    write_wav,
)
from thirdlight.circuit import Bridge, Switch, Tank, from_modes, tune
from thirdlight.control import Interrupter, Melody, note_frequency
from thirdlight.secondary import Modes
from thirdlight.solver import Result
from thirdlight.thermal.ledger import integrate

RATE = 48000
WIDTH = 1.0e-3
CENTRE = 6.0 * WIDTH
SPAN = 12.0 * WIDTH
AMPLITUDE = 1.0e4
DISTANCE = 100.0 * SOUND_SPEED / RATE
GAIN = (GAMMA - 1.0) / (4.0 * math.pi * SOUND_SPEED**2 * DISTANCE)
NOTES = (57, 60, 64)
NOTE_SPAN = 0.2


def gaussian(t, order=0):
    """``A exp(-(t - t0)^2 / 2 s^2)`` and its first three derivatives, closed form."""
    u = (np.asarray(t, dtype=float) - CENTRE) / WIDTH
    shape = (np.ones_like(u), -u, u * u - 1.0, -u * (u * u - 3.0))[order]
    return AMPLITUDE * shape * np.exp(-0.5 * u * u) / WIDTH**order


@dataclass(frozen=True)
class Pulse:
    """The acoustic source of a run alone: sample times and channel power."""

    t: np.ndarray
    streamer_power: np.ndarray


def pulse(rate=RATE):
    """The Gaussian power pulse sampled on the audio grid itself."""
    t = np.arange(round(SPAN * rate) + 1) / rate
    return Pulse(t=t, streamer_power=gaussian(t))


def signature(rate=RATE, distance=DISTANCE):
    """One bang with its propagation delay stripped, and the delay in samples."""
    delay = round(distance / SOUND_SPEED * rate)
    return bang(pulse(rate), distance, rate)[delay:], delay


def silent():
    """A real ``Result`` of a network with no streamer, at a nonzero state."""
    f, l_p, l_m = 1.0e5, 1.0e-4, 6.0e-2
    zero, inductance = np.zeros((1, 1)), np.array([l_m])
    modes = Modes(
        f=np.array([f]),
        v=zero,
        i=zero,
        l_m=inductance,
        c_m=1.0 / (inductance * (2.0 * math.pi * f) ** 2),
        z=zero,
    )
    bridge = Bridge(igbt=Switch(1.2, 0.012), diode=Switch(1.0, 0.010))
    net = from_modes(modes, [0.2], [400.0], l_p, Tank(tune(l_p, f)), bridge)
    count = 256
    empty = np.zeros(count)
    return Result(
        t=np.linspace(0.0, 1.0e-3, count),
        x=np.random.default_rng(0).normal(size=(count, net.size)),
        gate=np.ones(count, dtype=np.int8),
        state=np.zeros(count, dtype=np.int8),
        u=np.zeros((count, 3)),
        network=net,
        length=empty,
        channel=empty,
        loss=empty,
    )


def test_sound_speed_is_the_adiabatic_one():
    assert SOUND_SPEED**2 == pytest.approx(
        GAMMA * AMBIENT_PRESSURE / AMBIENT_DENSITY, rel=1e-15
    )


def test_thermoacoustic_volume_matches_constant_pressure_heating():
    heat, mass, temperature = 5.0, 1.0, 293.15
    gas_constant = AMBIENT_PRESSURE / (AMBIENT_DENSITY * temperature)
    heat_capacity = mass * GAMMA * gas_constant / (GAMMA - 1.0)
    expansion = mass * gas_constant * (heat / heat_capacity) / AMBIENT_PRESSURE
    assert thermoacoustic_volume(heat) == pytest.approx(expansion, rel=1e-12)


def test_pressure_falls_as_inverse_distance():
    near, delay = signature()
    far, twice = signature(distance=2.0 * DISTANCE)
    assert (delay, twice) == (100, 200)
    np.testing.assert_allclose(far, 0.5 * near, rtol=1e-12, atol=0.0)


def test_arrival_is_delayed_by_the_propagation_time():
    near = bang(pulse(), DISTANCE, RATE)
    far = bang(pulse(), 2.0 * DISTANCE, RATE)
    shift = round(DISTANCE / SOUND_SPEED * RATE)
    assert far.size - near.size == shift
    assert np.argmax(np.abs(far)) - np.argmax(np.abs(near)) == shift
    assert np.flatnonzero(far)[0] - np.flatnonzero(near)[0] == shift


def test_pressure_is_the_centred_difference_of_the_resampled_power():
    """The composite stencil, to 1e-12 of the terms it differences away."""
    power, step = gaussian(pulse().t), 1.0 / RATE
    stencil = power[4:] + 2.0 * power[3:-1] - 2.0 * power[1:-3] - power[:-4]
    scale = GAIN / (8.0 * step)
    np.testing.assert_allclose(
        signature()[0][2:-2],
        scale * stencil,
        rtol=0.0,
        atol=1e-12 * scale * np.abs(power).max(),
    )


def test_pressure_carries_the_truncation_error_it_should():
    """Deviation from the analytic derivative is the leading 5 h^2 P''' / 12."""
    t = pulse().t[2:-2]
    step = 1.0 / RATE
    error = np.abs(signature()[0][2:-2] - GAIN * gaussian(t, 1)).max()
    leading = (5.0 / 12.0) * step**2 * GAIN * np.abs(gaussian(t, 3)).max()
    assert error == pytest.approx(leading, rel=0.02)


def test_resampling_conserves_the_source_energy():
    source = pulse()
    coarse = _mean_power(source.t, source.streamer_power, RATE // 6)
    total = integrate(source.t, None, source.streamer_power)
    assert coarse.sum() / (RATE // 6) == pytest.approx(total, rel=1e-12)


def test_radiated_energy_matches_the_gaussian_closed_form():
    """Sphere integral of p^2 / (rho0 c) against 2 pi^1.5 K^2 A^2 / (rho0 c s)."""
    pressure = bang(pulse(), DISTANCE, RATE)
    area = 4.0 * math.pi * DISTANCE**2
    radiated = area * (pressure**2).sum() / (RATE * AMBIENT_DENSITY * SOUND_SPEED)
    strength = (GAMMA - 1.0) / (4.0 * math.pi * SOUND_SPEED**2)
    closed = 2.0 * math.pi**1.5 * strength**2 * AMPLITUDE**2
    closed /= AMBIENT_DENSITY * SOUND_SPEED * WIDTH
    assert radiated == pytest.approx(closed, rel=1.5 / (WIDTH * RATE) ** 2)


def test_pulse_train_spectrum_is_the_prf_comb():
    """An exact number of periods, folded, so the comb is exact rather than leaked."""
    prf, bursts = 300.0, 300
    period = round(RATE / prf)
    audio = render(pulse(), Interrupter(1.0e-3, prf), (bursts + 0.5) / prf, DISTANCE)
    length = bursts * period
    folded = audio[:length].copy()
    folded[: audio.size - length] += audio[length:]
    spectrum = np.abs(np.fft.rfft(folded))
    comb = np.zeros(spectrum.size, dtype=bool)
    comb[::bursts] = True
    assert spectrum[~comb].max() < 1e-12 * spectrum[comb].max()
    fundamental = np.flatnonzero(spectrum[1:] > 1e-9 * spectrum.max())[0] + 1
    assert fundamental * RATE / length == prf


def melody():
    """Three notes back to back, each held for the same span."""
    notes = tuple((i * NOTE_SPAN, NOTE_SPAN, note) for i, note in enumerate(NOTES))
    return Melody(notes=notes, on_time=1.0e-4)


def test_melody_notes_render_at_their_own_pitch():
    schedule = melody()
    audio = render(pulse(), schedule, len(NOTES) * NOTE_SPAN, DISTANCE)
    span = round(NOTE_SPAN * RATE)
    delay = round(DISTANCE / SOUND_SPEED * RATE)
    for index, note in enumerate(NOTES):
        note_audio = audio[delay + index * span : delay + (index + 1) * span]
        spectrum = np.abs(np.fft.rfft(note_audio))
        peak = int(np.argmax(spectrum[1:])) + 1
        assert peak == round(float(note_frequency(note)) * NOTE_SPAN)


@pytest.mark.parametrize("schedule", [Interrupter(1.0e-3, 137.0), melody()])
def test_scatter_add_placement_matches_a_per_burst_loop(schedule):
    duration = len(NOTES) * NOTE_SPAN
    shape, delay = signature()
    edges = schedule.edges(duration)
    starts = edges[schedule.active(np.nextafter(edges, math.inf))]
    assert starts.size > 1
    expected = np.zeros(round(duration * RATE) + shape.size + delay)
    for start in starts:
        first = int(np.rint(start * RATE)) + delay
        expected[first : first + shape.size] += shape
    got = render(pulse(), schedule, duration, DISTANCE)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0.0)


def test_wav_round_trip(tmp_path):
    audio = render(pulse(), Interrupter(1.0e-3, 137.0), 0.05, DISTANCE)
    path = tmp_path / "spark.wav"
    write_wav(path, audio, RATE)
    with wave.open(str(path), "rb") as handle:
        assert (handle.getnchannels(), handle.getsampwidth()) == (1, 2)
        assert handle.getframerate() == RATE
        raw = handle.readframes(handle.getnframes())
    samples = np.frombuffer(raw, dtype="<i2") / FULL_SCALE
    assert samples.size == audio.size
    np.testing.assert_allclose(
        samples, normalise(audio), rtol=0.0, atol=0.5 / FULL_SCALE
    )


def test_normalise_leaves_silence_alone():
    assert not np.any(normalise(np.zeros(8)))
    assert np.abs(normalise(np.array([-4.0, 1.0]))).max() == 1.0


def test_silent_run_renders_silence():
    result = silent()
    assert not np.any(result.streamer_power)
    assert not np.any(bang(result, DISTANCE, RATE))
    assert not np.any(render(result, Interrupter(1.0e-3, 137.0), 0.05, DISTANCE))
