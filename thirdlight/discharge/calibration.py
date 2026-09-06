"""Spark length against input power: operating points, Freau's law and the fit.

A burst is simulated; the gap between bursts is not, because nothing happens
there but the channel cooling, which is the analytic decay of the length ODE with
no drive. One burst plus that map is a whole interrupter cycle, and iterating it
to its fixed point gives the length the coil settles at, which is what a
photograph measures.
"""

import math
from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares

FREAU_COEFFICIENT = 1.7 * 0.0254


def freau_length(power, coefficient=FREAU_COEFFICIENT):
    """Spark length from average input power, L = k sqrt(P); Freau's law at k = 1.7 in/sqrt(W)."""
    return coefficient * np.sqrt(np.asarray(power, dtype=float))


def burst(machine, streamer, length=0.0, tail=5.0):
    """One interrupter burst from a cold circuit, ``tail`` resonant periods past its end."""
    duration = machine.driver.interrupter.on_time + tail / machine.frequency
    return machine.run(duration, streamer=streamer, length0=length)


def operating_point(machine, streamer, cycles=8, rtol=1e-3):
    """Average input power and settled spark length of one interrupter cycle.

    The burst is iterated from what the previous one left after cooling until the
    seed length stops moving, which is one iteration whenever the gap between
    bursts is long against the cooling time.
    """
    interrupter = machine.driver.interrupter
    gap = interrupter.period - interrupter.on_time
    length, result = 0.0, None
    for _ in range(cycles):
        result = burst(machine, streamer, length)
        seed = result.length[-1] * math.exp(-gap / streamer.cooling)
        settled = abs(seed - length) <= rtol * max(seed, streamer.minimum)
        length = seed
        if settled:
            break
    return result.input_energy * interrupter.frequency, float(result.length.max())


def sweep(machine, streamer, buses, frequencies=None):
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
            points.append(operating_point(replace(machine, driver=driver), streamer))
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
