"""Loss extraction from a run: conduction, commutation and the component ledger.

Conduction loss separates analytically out of the loop resistances the state
matrices carry. Switching energy is in no part of that state space and is
attributed to the commutation instants the stepper resolved; see §3.5 of design.
"""

from dataclasses import dataclass

import numpy as np

from thirdlight.circuit.devices import DIODE, IGBT, OPEN, STATES, polarity


def integrate(t, weight, values, sums=False):
    """Trapezoid of ``weight * values`` with the weight held over each interval.

    The bridge polarity, the conducting device's drop and the loop resistances
    all step at interval boundaries while the state runs on continuously, so each
    interval is closed with its own weight, a scalar one standing for every
    interval. ``sums`` reduces the trailing axis, for per-loop weights.
    """
    values = np.asarray(values)
    left, right = values[:-1], values[1:]
    if weight is not None:
        weight = np.asarray(weight)
        weight = weight if weight.ndim == 0 else weight[: len(values) - 1]
        left, right = weight * left, weight * right
    if sums:
        left, right = left.sum(axis=-1), right.sum(axis=-1)
    return float((0.5 * (left + right) * np.diff(t)).sum())


@dataclass(frozen=True)
class Switching:
    """Switching energy attributed to each commutation of a run, J.

    One entry per bridge polarity change, whether or not it cost anything.
    ``current`` is the primary current at the instant, signed, ``voltage`` the
    bus the devices commutate against, and each energy is for the whole bridge.
    """

    t: np.ndarray
    current: np.ndarray
    voltage: np.ndarray
    on: np.ndarray
    off: np.ndarray
    rr: np.ndarray

    def __len__(self):
        return len(self.t)

    @property
    def total(self):
        """Switching energy over the run, J."""
        return float(self.on.sum() + self.off.sum() + self.rr.sum())


@dataclass(frozen=True)
class Ledger:  # pylint: disable=too-many-instance-attributes
    """Energy dissipated per component over a run, J.

    :attr:`total` is the conduction ledger and equals ``Result.dissipation``,
    which the burst energy balance validates. :attr:`switching` is additive to
    it and not counted in it: no part of it appears in the state space.
    """

    igbt: float
    diode: float
    primary: float
    esr: float
    winding: np.ndarray
    former: float
    channel: float
    switching: Switching

    @property
    def conduction(self):
        """Semiconductor conduction loss, IGBTs and diodes together, J."""
        return self.igbt + self.diode

    @property
    def total(self):
        """Conduction ledger of the run, switching excluded, J."""
        return (
            self.conduction
            + self.primary
            + self.esr
            + float(np.sum(self.winding))
            + self.former
            + self.channel
        )


def commutations(result):
    """Samples at which the bridge polarity changes, and whether an IGBT is either side.

    A polarity change is the switching event: the IGBTs of the new polarity take
    the current off the diodes of the old one against the full bus. A kind change
    at one polarity is the handover to diodes gated across, at the current zero.
    """
    sigma = polarity(result.state)
    hit = np.flatnonzero(sigma[1:] != sigma[:-1]) + 1
    kind = np.asarray(result.state) // 2
    return hit, kind[hit - 1] == IGBT, kind[hit] == IGBT


def switching(result, tj=None):
    """Switching energy of every commutation of ``result`` at junction temperature ``tj``.

    An IGBT leaving conduction turns off the current it carried; one entering it
    turns on into that current and recovers the opposite leg's diode. ``tj``
    defaults to each fit's own test temperature, extrapolating nothing.
    """
    hit, off, on = commutations(result)
    current = result.primary_current[hit]
    voltage = np.asarray(result.bus_voltage)[hit]
    e_on, e_off, e_rr = result.network.bridge.commutation(current, tj, voltage)
    zero = np.zeros(len(hit))
    return Switching(
        t=result.t[hit],
        current=current,
        voltage=voltage,
        on=np.where(on, e_on, zero),
        off=np.where(off, e_off, zero),
        rr=np.where(on, e_rr, zero),
    )


def _device_resistance(bridge):
    """Differential resistance the conducting devices add to the loop, per state row."""
    return np.array(
        [
            0.0 if device is None else bridge.devices * device.r
            for device in map(bridge.conducting, np.arange(STATES) // 2)
        ]
    )


def ledger(result, tj=None):
    """Component-resolved energy ledger of a run; see :class:`Ledger`.

    Each row's primary loop resistance is tank.resistance + n_dev r and the tank's
    own is the loop plus the capacitor ESR, so the three separate by subtraction;
    the modal rows split the same way into winding and former dielectric.
    """
    net = result.network
    rows = np.arange(STATES)
    device = _device_resistance(net.bridge)
    esr = np.where(rows // 2 == OPEN, 0.0, net.esr)
    loop = net.resistances[:, 0] - device - esr
    state, t = result.state, result.t
    current = net.currents(result.x)
    square = current * current
    drop = -result.u[:, 0]
    dielectric = net.dielectric
    modal = net.resistances[state][:, 1:]
    conduction = [
        integrate(t, np.where(rows[state] // 2 == kind, drop, 0.0), current[:, 0])
        + integrate(
            t, np.where(rows[state] // 2 == kind, device[state], 0.0), square[:, 0]
        )
        for kind in (IGBT, DIODE)
    ]
    modes = range(net.modes)
    return Ledger(
        igbt=conduction[0],
        diode=conduction[1],
        primary=integrate(t, loop[state], square[:, 0]),
        esr=integrate(t, esr[state], square[:, 0]),
        winding=np.array(
            [integrate(t, modal[:, m] - dielectric[m], square[:, m + 1]) for m in modes]
        ),
        former=sum(integrate(t, dielectric[m], square[:, m + 1]) for m in modes),
        channel=integrate(t, None, result.streamer_power)
        + float(result.loss[-1] - result.loss[0]),
        switching=switching(result, tj),
    )
