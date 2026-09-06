"""Spark length against input power: operating points, Freau's law and the fit.

A burst is simulated; the gap between bursts is not, nothing happening there but
the channel cooling, which is one undriven step of the model. Iterating burst and
gap to their fixed point gives the settled length a photograph measures.
"""

from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares

FREAU_COEFFICIENT = 1.7 * 0.0254


def freau_length(power, coefficient=FREAU_COEFFICIENT):
    """Spark length from average input power, L = k sqrt(P); Freau's law at k = 1.7 in/sqrt(W)."""
    return coefficient * np.sqrt(np.asarray(power, dtype=float))


def inches_per_root_watt(power, length):
    """k = L / sqrt(P) of an operating point, in the in/sqrt(W) the coilers publish."""
    return np.asarray(length, dtype=float) / np.sqrt(np.asarray(power)) / 0.0254


def burst(machine, streamer, length=0.0, tail=5.0, rng=None):
    """One burst from a cold circuit and a channel seeded with ``length``.

    It runs ``tail`` resonant periods past the burst's end, far enough for the
    tank to ring down and return what it holds to the bus. ``rng`` is the
    randomness of a grown channel, and nothing to the scalar model.
    """
    return machine.run(span(machine, tail), streamer=streamer, length0=length, rng=rng)


def span(machine, tail=5.0):
    """Simulated part of an interrupter cycle: the burst and its ring-down."""
    return machine.driver.interrupter.on_time + tail / machine.frequency


def operating_point(machine, streamer, cycles=8, rtol=1e-3, rng=None):
    """Average input power and settled spark length of one interrupter cycle.

    What carries over is the model's own state, a length for the scalar model and
    the surviving tree for a grown one, cooled through the gap by one undriven
    step of the model itself.
    """
    interrupter = machine.driver.interrupter
    gap = interrupter.period - span(machine)
    state, length, result = 0.0, 0.0, None
    for _ in range(cycles):
        result = burst(machine, streamer, state, rng=rng)
        state = streamer.advance(result.channel_state, 0.0, 0.0, gap)
        seed = streamer.extent(state)
        settled = abs(seed - length) <= rtol * max(seed, streamer.resolution)
        length = seed
        if settled:
            break
    return result.input_energy * interrupter.frequency, float(result.length.max())


def sweep(machine, streamer, buses, frequencies=None, rng=None):
    """Operating points over bus voltages and, optionally, burst repetition rates.

    Neither the bus nor the interrupter enters the state space, so every point
    reuses the one network the machine was built with.
    """
    frequencies = frequencies or [machine.driver.interrupter.frequency]
    points = []
    for frequency in frequencies:
        gating = replace(machine.driver.interrupter, frequency=frequency)
        for bus in buses:
            driver = replace(machine.driver, bus=bus, interrupter=gating)
            points.append(
                operating_point(replace(machine, driver=driver), streamer, rng=rng)
            )
    return np.array(points).T


def residuals(machine, streamer, buses, frequencies, coefficient):
    """Log-space distance of a sweep's points from L = k sqrt(P)."""
    power, length = sweep(machine, streamer, buses, frequencies)
    return np.log(np.maximum(length, 1e-6)) - np.log(freau_length(power, coefficient))


def fit(
    machine, streamer, buses, frequencies=None, coefficient=FREAU_COEFFICIENT, **kwargs
):
    """Growth gain and cooling time placing a sweep on L = k sqrt(P).

    Fitted in logarithms, which keeps both constants positive and weights every
    operating point by its relative error rather than by its length.
    """

    def objective(log_pair):
        growth, cooling = np.exp(log_pair)
        return residuals(
            machine,
            replace(streamer, growth=growth, cooling=cooling),
            buses,
            frequencies,
            coefficient,
        )

    guess = np.log([streamer.growth, streamer.cooling])
    solution = least_squares(objective, guess, **kwargs)
    growth, cooling = np.exp(solution.x)
    return replace(streamer, growth=growth, cooling=cooling), solution
