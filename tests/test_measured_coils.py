"""Validation against published measurements of real air-cored coils.

Fixtures are the bare-coil table of Paul Nicholson's Tesla Secondary Simulation
Project, abelian.org/tssp/tests.html of 19 Jul 2008 read from the Internet Archive,
every coil base fed with no top load over a ground plane, and Denicolai's thesis.
"""

import numpy as np
import pytest

from thirdlight.em.inductance import solenoid_inductance
from thirdlight.geometry import Design, Primary, Solenoid
from thirdlight.secondary import resonance

# name, d, h/d, sr, b/h, turns, measured f1, f3 ... in kHz, and tssp's own f1 error.
TSSP = (
    ("mm2", 0.108, 9.97, 0.81, 0.31, 1700, (276.9, 711.8), -1.5),
    ("mm1", 0.091, 8.92, 0.76, 0.41, 1221, (455.5,), 1.9),
    ("sk16b55", 0.161, 8.71, 0.90, 0.39, 1976, (161.4, 386.4, 562.0, 710.3), -3.7),
    ("mm4", 0.114, 6.78, 0.83, 0.39, 1600, (237.0,), 2.1),
    ("tfsm1", 0.108, 6.14, 0.91, 0.03, 1176, (358.8, 883.1, 1265.5, 1602.5), -0.5),
    ("sk12b49", 0.121, 4.83, 0.92, 0.84, 894, (405.1,), 0.2),
    ("mm3", 0.221, 4.66, 0.93, 0.35, 2989, (61.9, 157.9, 229.7, 294.4, 355.6), 2.5),
    ("mwa1-4hd0", 0.168, 4.00, 0.92, 0.74, 1106, (224.0,), 0.1),
    ("mwa2-4hd0", 0.168, 4.00, 0.49, 0.74, 1106, (220.0,), 1.8),
    ("sk20b49", 0.205, 3.26, 0.90, 0.73, 943, (217.2, 497.8, 709.9), -5.0),
    (
        "tfltr",
        0.261,
        2.92,
        0.67,
        0.03,
        1000,
        (148.4, 353.4, 513.8, 666.4, 819.8, 977.4, 1133.1),
        -1.3,
    ),
    ("pn2", 0.580, 2.84, 0.88, 0.08, 725, (92.0, 213.0, 320.0), -0.9),
    ("pn1", 0.590, 1.36, 0.91, 0.05, 356, (150.7, 360.0, 543.0), 1.0),
)
SECTIONS = 240
# Thor, 939 turns of 1.45 mm wire over 1.575 m at radius 0.2 m; thesis and acmi thor.in.
THOR = Solenoid(radius=0.200, length=1.575, turns=939, wire_diameter=1.45e-3)
THOR_INDUCTANCE = 80.22e-3


def bare(d, length_ratio, spacing, base_ratio, turns):
    """A tssp bare coil, from the ratios its table prints.

    Only d, h/d, sr, b/h and turns are published, so the winding length is
    (h/d) d, the base height is (b/h) h and the wire diameter is sr h / turns,
    from the definitions at abelian.org/tssp/vsd/.
    """
    length = length_ratio * d
    coil = Solenoid(
        radius=0.5 * d,
        length=length,
        turns=turns,
        wire_diameter=spacing * length / turns,
        base=base_ratio * length,
    )
    return Design(
        secondary=coil,
        primary=Primary(inner_radius=d, turns=1.0),
        sections=min(SECTIONS, turns),
    )


@pytest.fixture(name="tssp", scope="module")
def tssp_errors():
    """Per cent error of every predicted tssp mode against its measurement."""
    return {
        row[0]: 100.0
        * (
            resonance(bare(*row[1:6]), modes=len(row[6])).f / (1e3 * np.array(row[6]))
            - 1.0
        )
        for row in TSSP
    }


@pytest.mark.slow
def test_the_overtones_of_the_measured_coils(tssp):
    """f3 and above land within 4 % on every coil that has them, and 2 % rms.

    The overtones carry their charge along the winding rather than at its ends, so
    they see little of the ground plane and the top fittings that f1 does.
    """
    overtones = np.concatenate([e[1:] for e in tssp.values() if len(e) > 1])
    assert overtones.size == 23
    assert np.abs(overtones).max() < 4.0
    assert np.sqrt((overtones**2).mean()) < 2.0


@pytest.mark.slow
def test_the_quarter_wave_resonance_of_the_measured_coils(tssp):
    """f1 lands within 4 % rms and is unbiased, against tssp's own 2.2 % rms.

    tssp images a ground plane of the coil's own height in radius where this model
    images an infinite one, and neither carries the formers, whose material none of
    these sources publishes; f1 is the mode those two omissions reach.
    """
    error = np.array([tssp[row[0]][0] for row in TSSP])
    assert np.abs(error.mean()) < 1.0
    assert np.sqrt((error**2).mean()) < 4.0
    assert np.abs(error).max() < 9.0


@pytest.mark.slow
def test_the_two_coils_tssp_itself_misses_are_missed_the_same_way(tssp):
    """sk16b55 and sk20b49 are tssp's worst bare coils, and this model repeats them.

    Agreeing to half a point with an independent model on the two coils that both
    miss places the residual in the measurements or the published geometry.
    """
    for row in TSSP:
        if row[0] in ("sk16b55", "sk20b49"):
            assert tssp[row[0]][0] == pytest.approx(row[7], abs=0.5)


def test_denicolais_measured_secondary_inductance():
    """Thor's 80.22 mH, measured at 1 kHz, from a filament sum over its 939 turns."""
    assert solenoid_inductance(THOR) == pytest.approx(THOR_INDUCTANCE, rel=0.015)
