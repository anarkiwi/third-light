"""Batched eigen-coordinate stepping: one kernel over a whole design space.

:func:`pack` flattens a sequence of machines into design-major constants -- the
eigendecomposition of every switch state, the two state-matrix rows the
comparator and the commutation test read, and a table of driver scalars -- and
:func:`run` steps them with no Python in the interval loop. The model is that of
:func:`~thirdlight.solver.stepping.simulate` less the streamer, the top-node load
and the history; see §3.3 of docs/design.md.

Every rule of :mod:`thirdlight.solver.kernels` holds: nothing is allocated inside
a kernel, complex data is carried as separate real and imaginary arrays, and the
design indexes the last axis.
"""

import functools
import math
from dataclasses import dataclass
from types import FunctionType

import numpy as np
import numba

from thirdlight import backend
from thirdlight.backend import kernel
from thirdlight.circuit.devices import STATES, polarity
from thirdlight.control.interrupter import Interrupter
from thirdlight.solver import kernels
from thirdlight.solver.propagator import Propagator

STEP, GAIN, TAU, DELAY, DEAD, BUS = 0, 1, 2, 3, 4, 5
ON_TIME, PERIOD, INITIAL, FINAL, RISE, GATED, RAMPED = 6, 7, 8, 9, 10, 11, 12
DRIVE = 13

POLARITY = polarity(np.arange(STATES))

_XTOL = 1e-14
_GATES = (-1, 0, 1)
_SIGNS = (-1.0, 0.0, 1.0)
_WORK = 11
_BLOCK = 64


@dataclass(frozen=True)
class Packed:  # pylint: disable=too-many-instance-attributes
    """Constants of a batch of designs, every array with the design on its last axis.

    ``drive`` is one table of the per-design driver scalars, rows :data:`STEP`
    through :data:`RAMPED`, rather than one array each: the parfor gufunc's
    compile time grows superlinearly in how many arrays it is handed.
    """

    lam_re: np.ndarray
    lam_im: np.ndarray
    basis_re: np.ndarray
    basis_im: np.ndarray
    inverse_re: np.ndarray
    inverse_im: np.ndarray
    inject_re: np.ndarray
    inject_im: np.ndarray
    a0: np.ndarray
    b0: np.ndarray
    resist: np.ndarray
    rows: np.ndarray
    offsets: np.ndarray
    drive: np.ndarray
    series: np.ndarray
    size: int
    loops: int
    inputs: int
    designs: int
    reservoir: bool

    @property
    def dtype(self):
        """Floating type the batch is packed in."""
        return self.a0.dtype


@dataclass(frozen=True)
class Batched:
    """Observables of a batch run: four totals and an interval count per design.

    ``state`` is the state each design finished at, design-major like everything
    the kernel was handed.
    """

    peak_current: np.ndarray
    peak_voltage: np.ndarray
    input_energy: np.ndarray
    dissipation: np.ndarray
    steps: np.ndarray
    state: np.ndarray


def _stack(values, dtype):
    """Design-major real array from one array per design."""
    return np.ascontiguousarray(np.stack(values, axis=-1), dtype=dtype)


def _split(values, dtype):
    """Design-major real and imaginary parts of one complex array per design."""
    stacked = np.stack(values, axis=-1)
    return (
        np.ascontiguousarray(stacked.real, dtype=dtype),
        np.ascontiguousarray(stacked.imag, dtype=dtype),
    )


def _reject(machines):
    """Raise on any design the batched model does not carry."""
    for d, machine in enumerate(machines):
        if machine.network.streamer is not None:
            raise ValueError(
                f"design {d} carries a streamer, whose channel capacitance re-levels "
                "mid-run and rebuilds the propagators"
            )
        gating = machine.driver.interrupter
        if gating is not None and not isinstance(gating, Interrupter):
            raise ValueError(
                f"design {d} is gated by a {type(gating).__name__}, whose note "
                "schedule is array data, not the scalars a plain Interrupter is"
            )
    shapes = {
        (m.network.size, m.network.b.shape[-1], m.network.modes) for m in machines
    }
    if len(shapes) > 1:
        raise ValueError(
            "batched designs must agree on state size, input count and mode count; "
            f"got {sorted(shapes)}"
        )


def _eigen(machines):
    """Every design's switch-state propagators, or a rejection of the Pade fallbacks."""
    built = [
        [Propagator.build(a, b, m.step) for a, b in zip(m.network.a, m.network.b)]
        for m in machines
    ]
    fallen = [d for d, props in enumerate(built) if not all(p.exact for p in props)]
    if fallen:
        raise ValueError(
            f"designs {fallen} fell back to Pade propagators, which carry no "
            "eigenbasis to pack"
        )
    return built


def _drive(machine, dtype):
    """One design's column of the driver-scalar table."""
    driver = machine.driver
    gating, ramp = driver.interrupter, driver.ramp
    column = np.zeros(DRIVE, dtype=dtype)
    column[STEP] = machine.step
    column[GAIN] = machine.network.bridge.gain
    column[TAU] = driver.lead.tau
    column[DELAY] = driver.delay
    column[DEAD] = driver.dead_time
    column[BUS] = driver.bus
    column[PERIOD] = 1.0 if gating is None else gating.period
    column[GATED] = gating is not None
    if gating is not None:
        column[ON_TIME] = gating.on_time
    column[RAMPED] = ramp is not None
    if ramp is not None:
        column[INITIAL], column[FINAL], column[RISE] = (
            ramp.initial,
            ramp.final,
            ramp.rise,
        )
    return column


def pack(machines, dtype=np.float64, load=None):
    """Batch constants of ``machines``, each stepped at its own :attr:`Machine.step`.

    ``load`` exists only to be refused: a top-node injection is an arbitrary
    Python callable and has no place in the kernel. Streamers, melodies, designs
    of disagreeing shape and inexact propagators are refused too.
    """
    if load is not None:
        raise ValueError(
            "the batched stepper carries no top-node load callback; use "
            "thirdlight.solver.simulate for one"
        )
    machines = list(machines)
    if not machines:
        raise ValueError("pack needs at least one machine")
    _reject(machines)
    props = _eigen(machines)
    nets = [m.network for m in machines]
    lam_re, lam_im = _split([[p.eigen.lam for p in ps] for ps in props], dtype)
    basis_re, basis_im = _split([[p.eigen.basis for p in ps] for ps in props], dtype)
    inv_re, inv_im = _split([[p.eigen.inverse for p in ps] for ps in props], dtype)
    inj_re, inj_im = _split([[p.eigen.inverse_b for p in ps] for ps in props], dtype)
    return Packed(
        lam_re=lam_re,
        lam_im=lam_im,
        basis_re=basis_re,
        basis_im=basis_im,
        inverse_re=inv_re,
        inverse_im=inv_im,
        inject_re=inj_re,
        inject_im=inj_im,
        a0=_stack([net.a[:, 0, :] for net in nets], dtype),
        b0=_stack([net.b[:, 0, :] for net in nets], dtype),
        resist=_stack([net.resistances for net in nets], dtype),
        rows=_stack(
            [[[net.index(g, s) for s in _SIGNS] for g in _GATES] for net in nets],
            np.int8,
        ),
        offsets=_stack(
            [[[net.offset(g, s) for s in _SIGNS] for g in _GATES] for net in nets],
            dtype,
        ),
        drive=_stack([_drive(m, dtype) for m in machines], dtype),
        series=np.asarray(kernels.SERIES, dtype),
        size=nets[0].size,
        loops=nets[0].loops,
        inputs=nets[0].b.shape[-1],
        designs=len(machines),
        reservoir=bool(nets[0].bus.reservoir),
    )


def _edge_time(edge, period, on_time, duration):
    """Time of interrupter edge ``edge``, or infinity once it passes ``duration``.

    Edges alternate off at k period + on_time and on at (k + 1) period, so the
    even ones end a burst and the odd ones start the next.
    """
    when = (edge // 2) * period + (on_time if edge % 2 == 0 else period)
    return when if when <= duration else math.inf


def _supply(t, burst_start, bus, initial, final, rise, ramped):
    """Supply voltage at ``t``, from the ramp measured since the burst start."""
    if not ramped:
        return bus
    if rise <= 0.0:
        return final
    return initial + (final - initial) * min(max((t - burst_start) / rise, 0.0), 1.0)


def _fire(t, gate, count, t0, g0, t1, g1):
    """Apply every queued gate transition due at or before ``t``."""
    while count > 0 and t0 <= t:
        gate = g0
        count -= 1
        t0, g0 = t1, g1
    return gate, count, t0, g0, t1, g1


def _queue(t, sign, gate, count, t0, g0, t1, g1, delay, dead):
    """Queue a comparator crossing to ``sign``, superseding any stale command.

    A crossing to the command already given or already queued is ignored;
    otherwise the queue is cleared, and a dead time inserts a zero ahead of it.
    """
    target = 1 if sign > 0.0 else -1
    last = gate
    if count == 2:
        last = g1
    elif count == 1:
        last = g0
    if target == last:
        return count, t0, g0, t1, g1
    if target == gate:
        return 0, t0, g0, t1, g1
    due = t + delay
    if gate != 0 and dead > 0.0:
        return 2, due, 0, due + dead, target
    return 1, due, target, math.inf, 0


def _conduction(a0, b0, rows, offsets, x, u, gate, size, inputs, d):
    """Polarity a blocked bridge starts to conduct with, or 0 while it stays blocked.

    Each candidate sign fixes the conducting devices and hence the loop equation,
    and is admissible when the resulting di_p/dt agrees with it.
    """
    for si in (2, 0):
        sign = 1.0 if si == 2 else -1.0
        r = rows[gate + 1, si, d]
        u[0, d] = offsets[gate + 1, si, d]
        rate = 0.0
        for j in range(size):
            rate += a0[r, j, d] * x[j, d]
        for j in range(inputs):
            rate += b0[r, j, d] * u[j, d]
        if rate * sign > 0.0:
            return sign
    return 0.0


def _step_design(  # pylint: disable=too-many-branches,too-many-statements,undefined-variable
    lam_re,
    lam_im,
    basis_re,
    basis_im,
    inverse_re,
    inverse_im,
    inject_re,
    inject_im,
    a0,
    b0,
    resist,
    rows,
    offsets,
    drive,
    sigma,
    series,
    size,
    loops,
    inputs,
    reservoir,
    duration,
    work,
    u,
    totals,
    steps,
    state,
    d,
):
    """Step design ``d`` for ``duration`` seconds, accumulating its observables.

    Mirrors :func:`~thirdlight.solver.stepping.simulate` without the streamer,
    the load injection or the history: every temporary is a column of the
    caller's ``work``, and the observables are integrated in flight with each
    interval closed under the weights of the sample that opened it.

    The driver scalars are read out in double whatever the batch is packed in:
    they set the clock, and a float32 time base would not resolve a step.
    """
    x, prev = work[0], work[1]
    y_re, y_im, ub_re, ub_im = work[2], work[3], work[4], work[5]
    w_re, w_im, out_re, out_im, lead = work[6], work[7], work[8], work[9], work[10]
    step, gain, tau = float(drive[STEP, d]), float(drive[GAIN, d]), float(drive[TAU, d])
    delay, dead = float(drive[DELAY, d]), float(drive[DEAD, d])
    bus, period = float(drive[BUS, d]), float(drive[PERIOD, d])
    on_time, rise = float(drive[ON_TIME, d]), float(drive[RISE, d])
    initial, final = float(drive[INITIAL, d]), float(drive[FINAL, d])
    gated, ramped = drive[GATED, d] != 0.0, drive[RAMPED, d] != 0.0
    for i in range(size):
        x[i, d] = 0.0
    gate, count, t0, g0, t1, g1 = 1, 0, math.inf, 0, math.inf, 0
    enabled, burst_start, edge = True, 0.0, 0
    sign_i, sign_fb, t = 0.0, 0.0, 0.0
    peak_i, peak_v, drawn, lost, taken = 0.0, 0.0, 0.0, 0.0, 0
    sampled = False
    left, drop_l, swing_l, switch_l = 0.0, 0.0, 0.0, 0
    while True:
        gate, count, t0, g0, t1, g1 = _fire(t, gate, count, t0, g0, t1, g1)
        due = math.inf
        if gated:
            due = _edge_time(edge, period, on_time, duration)
            while due <= t:
                count = 0
                enabled = edge % 2 == 1
                gate = 1 if enabled else 0
                if enabled:
                    burst_start = t
                edge += 1
                due = _edge_time(edge, period, on_time, duration)
        supply = _supply(t, burst_start, bus, initial, final, rise, ramped)
        u[0, d], u[1, d], u[2, d] = 0.0, 0.0, supply
        if sign_i == 0.0:
            sign_i = _conduction(a0, b0, rows, offsets, x, u, gate, size, inputs, d)
        si = int(sign_i) + 1
        switch = int(rows[gate + 1, si, d])
        u[0, d] = offsets[gate + 1, si, d]
        current = x[0, d]
        top = 0.0
        for i in range(loops + 1, 2 * loops):
            top += x[i, d]
        peak_i = max(peak_i, abs(current))
        peak_v = max(peak_v, abs(top))
        if sampled:
            width = t - left
            mean = 0.5 * (prev[0, d] + current)
            drawn += swing_l * mean * width
            lost -= drop_l * mean * width
            for k in range(loops):
                lost += (
                    resist[switch_l, k, d]
                    * 0.5
                    * (prev[k, d] * prev[k, d] + x[k, d] * x[k, d])
                    * width
                )
            taken += 1
        for i in range(size):
            prev[i, d] = x[i, d]
        sampled, left, drop_l, switch_l = True, t, u[0, d], switch
        swing_l = sigma[switch] * gain * (x[2 * loops, d] if reservoir else supply)
        if t >= duration:
            break
        limit = min(t + step, duration, due)
        if count > 0:
            limit = min(limit, t0)
        span = min(limit - t, step)
        if span <= 0.0:
            t = nextafter(t, math.inf)
            continue
        modal(inverse_re[switch], inverse_im[switch], x, y_re, y_im, size, d)
        inject(inject_re[switch], inject_im[switch], u, ub_re, ub_im, size, inputs, d)
        offset = 0.0
        for j in range(inputs):
            offset += b0[switch, j, d] * u[j, d]
        offset *= tau
        start = offset
        for i in range(size):
            lead[i, d] = tau * a0[switch, i, d] + (1.0 if i == 0 else 0.0)
            start += lead[i, d] * x[i, d]
        row(basis_re[switch], basis_im[switch], lead, w_re, w_im, size, d)
        tol = span * _XTOL
        hit_i = math.inf
        if sign_i != 0.0:
            hit_i = bisect(
                lam_re[switch],
                lam_im[switch],
                basis_re[switch, 0],
                basis_im[switch, 0],
                y_re,
                y_im,
                ub_re,
                ub_im,
                0.0,
                span,
                sign_i,
                current,
                tol,
                series,
                size,
                d,
            )
        if sign_fb != 0.0:
            hit_fb = bisect(
                lam_re[switch],
                lam_im[switch],
                w_re,
                w_im,
                y_re,
                y_im,
                ub_re,
                ub_im,
                offset,
                span,
                sign_fb,
                start,
                tol,
                series,
                size,
                d,
            )
            if hit_fb < min(span, hit_i):
                sign_fb = -sign_fb
                if enabled:
                    count, t0, g0, t1, g1 = _queue(
                        t + hit_fb, sign_fb, gate, count, t0, g0, t1, g1, delay, dead
                    )
                if count > 0:
                    span = min(span, t0 - t)
        first = min(span, hit_i)
        if first > 0.0:
            advance(
                lam_re[switch],
                lam_im[switch],
                y_re,
                y_im,
                ub_re,
                ub_im,
                first,
                series,
                out_re,
                out_im,
                size,
                d,
            )
            restore(basis_re[switch], basis_im[switch], out_re, out_im, x, size, d)
        t += first
        if first == hit_i:
            sign_i = 0.0
            x[0, d] = 0.0
        elif sign_fb == 0.0:
            value = offset
            for i in range(size):
                value += lead[i, d] * x[i, d]
            sign_fb = 0.0 if value == 0.0 else math.copysign(1.0, value)
    totals[0, d], totals[1, d] = peak_i, peak_v
    totals[2, d], totals[3, d] = drawn, lost
    steps[d] = taken
    for i in range(size):
        state[i, d] = x[i, d]


_SOURCES = (_edge_time, _supply, _fire, _queue, _conduction, _step_design)


def build(compile_):
    """Compile the stepper sources for one target, over that target's own kernels.

    Same rebinding as :func:`thirdlight.solver.kernels.build`, with the scope
    seeded from the kernel set built for the same target.
    """
    scope = kernels.build(compile_)
    scope.update(backend.primitives(compile_))
    scope.update(
        {
            "__name__": __name__,
            "__file__": __file__,
            "math": math,
            "STEP": STEP,
            "GAIN": GAIN,
            "TAU": TAU,
            "DELAY": DELAY,
            "DEAD": DEAD,
            "BUS": BUS,
            "ON_TIME": ON_TIME,
            "PERIOD": PERIOD,
            "INITIAL": INITIAL,
            "FINAL": FINAL,
            "RISE": RISE,
            "GATED": GATED,
            "RAMPED": RAMPED,
            "_XTOL": _XTOL,
        }
    )
    built = {}
    for func in _SOURCES:
        rebound = FunctionType(
            func.__code__, scope, func.__name__, func.__defaults__, func.__closure__
        )
        scope[func.__name__] = built[func.__name__] = compile_(rebound)
    return built


CPU = build(kernel)
_edge_time = CPU["_edge_time"]
_supply = CPU["_supply"]
_fire = CPU["_fire"]
_queue = CPU["_queue"]
_conduction = CPU["_conduction"]
_step_design = CPU["_step_design"]


@numba.njit(parallel=True, cache=True)
def _run_batch(
    lam_re,
    lam_im,
    basis_re,
    basis_im,
    inverse_re,
    inverse_im,
    inject_re,
    inject_im,
    a0,
    b0,
    resist,
    rows,
    offsets,
    drive,
    sigma,
    series,
    size,
    loops,
    inputs,
    reservoir,
    duration,
    work,
    u,
    totals,
    steps,
    state,
):
    """Step every design of the batch, one thread per design."""
    for d in numba.prange(work.shape[2]):  # pylint: disable=not-an-iterable
        _step_design(
            lam_re,
            lam_im,
            basis_re,
            basis_im,
            inverse_re,
            inverse_im,
            inject_re,
            inject_im,
            a0,
            b0,
            resist,
            rows,
            offsets,
            drive,
            sigma,
            series,
            size,
            loops,
            inputs,
            reservoir,
            duration,
            work,
            u,
            totals,
            steps,
            state,
            d,
        )


@functools.lru_cache(maxsize=1)
def _run_grid():
    """The CUDA launch kernel over the device build, one thread per design."""
    from numba import cuda  # pylint: disable=import-outside-toplevel

    step_design = build(backend.device)["_step_design"]

    @cuda.jit
    def grid(
        lam_re,
        lam_im,
        basis_re,
        basis_im,
        inverse_re,
        inverse_im,
        inject_re,
        inject_im,
        a0,
        b0,
        resist,
        rows,
        offsets,
        drive,
        sigma,
        series,
        size,
        loops,
        inputs,
        reservoir,
        duration,
        work,
        u,
        totals,
        steps,
        state,
    ):
        """Step the design this thread indexes, if the grid overhangs the batch."""
        d = cuda.grid(1)  # pylint: disable=no-value-for-parameter
        if d < work.shape[2]:
            step_design(
                lam_re,
                lam_im,
                basis_re,
                basis_im,
                inverse_re,
                inverse_im,
                inject_re,
                inject_im,
                a0,
                b0,
                resist,
                rows,
                offsets,
                drive,
                sigma,
                series,
                size,
                loops,
                inputs,
                reservoir,
                duration,
                work,
                u,
                totals,
                steps,
                state,
                d,
            )

    return grid


def _on_device(constants, scratch):
    """Run the batch on the GPU, returning host copies of the scratch buffers."""
    from numba import cuda  # pylint: disable=import-outside-toplevel

    moved = tuple(
        cuda.to_device(a) if isinstance(a, np.ndarray) else a for a in constants
    )
    buffers = tuple(cuda.to_device(a) for a in scratch)
    blocks = (scratch[0].shape[2] + _BLOCK - 1) // _BLOCK
    _run_grid()[blocks, _BLOCK](*moved, *buffers)
    return tuple(b.copy_to_host() for b in buffers)


def run(packed, duration):
    """Step every design of ``packed`` for ``duration`` seconds, on the active backend.

    The scratch is allocated once for the batch and indexed by design, so the
    kernel itself allocates nothing.
    """
    designs = packed.designs
    constants = (
        packed.lam_re,
        packed.lam_im,
        packed.basis_re,
        packed.basis_im,
        packed.inverse_re,
        packed.inverse_im,
        packed.inject_re,
        packed.inject_im,
        packed.a0,
        packed.b0,
        packed.resist,
        packed.rows,
        packed.offsets,
        packed.drive,
        POLARITY,
        packed.series,
        packed.size,
        packed.loops,
        packed.inputs,
        packed.reservoir,
        float(duration),
    )
    scratch = (
        np.zeros((_WORK, packed.size, designs), dtype=packed.dtype),
        np.zeros((packed.inputs, designs), dtype=packed.dtype),
        np.zeros((4, designs)),
        np.zeros(designs, dtype=np.int64),
        np.zeros((packed.size, designs)),
    )
    if backend.selected() == "cuda":
        scratch = _on_device(constants, scratch)
    else:
        _run_batch(*constants, *scratch)
    _, _, totals, steps, state = scratch
    return Batched(
        peak_current=totals[0],
        peak_voltage=totals[1],
        input_energy=totals[2],
        dissipation=totals[3],
        steps=steps,
        state=state,
    )
