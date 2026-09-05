"""Burst gating: fixed pulse width and PRF, or a MIDI note schedule.

A burst begins at every PRF period and lasts ``on_time``. Transition times are
generated as whole arrays, so a MIDI program of 10^4 bursts costs one arange
rather than a loop.
"""

from dataclasses import dataclass

import numpy as np

MIDI_REFERENCE_NOTE = 69
MIDI_REFERENCE_FREQUENCY = 440.0


def note_frequency(note):
    """Pitch of a MIDI note number, 440 Hz at note 69, twelve-tone equal temperament."""
    exponent = (np.asarray(note, dtype=float) - MIDI_REFERENCE_NOTE) / 12.0
    return MIDI_REFERENCE_FREQUENCY * np.exp2(exponent)


def _burst_edges(start, stop, period, on_time):
    """Alternating on/off times of bursts filling [start, stop), off clipped to stop."""
    on = start + period * np.arange(np.ceil((stop - start) / period))
    return np.stack([on, np.minimum(on + on_time, stop)], axis=1).ravel()


@dataclass(frozen=True)
class Interrupter:
    """Fixed burst gating: ``on_time`` seconds of enable at ``frequency`` bursts/s."""

    on_time: float
    frequency: float

    def __post_init__(self):
        if self.frequency <= 0.0 or self.on_time <= 0.0:
            raise ValueError("interrupter on time and frequency must be positive")
        if self.on_time >= self.period:
            raise ValueError(f"on time {self.on_time} exceeds period {self.period}")

    @classmethod
    def from_note(cls, note, on_time):
        """Gating at the PRF of a MIDI note number."""
        return cls(on_time=on_time, frequency=float(note_frequency(note)))

    @property
    def period(self):
        """Burst repetition period."""
        return 1.0 / self.frequency

    @property
    def duty(self):
        """Fraction of the period spent enabled."""
        return self.on_time * self.frequency

    def active(self, t):
        """Whether the gate is enabled at ``t``, bursts starting at t = 0."""
        return np.mod(t, self.period) < self.on_time

    def edges(self, duration):
        """Sorted transition times in (0, duration]."""
        edges = _burst_edges(0.0, duration + self.period, self.period, self.on_time)
        return edges[(edges > 0.0) & (edges <= duration)]


@dataclass(frozen=True)
class Melody:
    """MIDI note schedule: each note sets the interrupter PRF for its span."""

    notes: tuple
    on_time: float

    def __post_init__(self):
        _, _, periods = self.table
        if periods.size and self.on_time >= periods.min():
            raise ValueError(f"on time {self.on_time} exceeds the shortest note period")

    @property
    def table(self):
        """Note starts, ends and burst periods as three arrays, ordered by start."""
        notes = np.asarray(self.notes, dtype=float).reshape(-1, 3)
        notes = notes[np.argsort(notes[:, 0], kind="stable")]
        return notes[:, 0], notes[:, 0] + notes[:, 1], 1.0 / note_frequency(notes[:, 2])

    def active(self, t):
        """Whether the gate is enabled at ``t``, each note phased from its own start."""
        starts, stops, periods = self.table
        if not starts.size:
            return np.zeros(np.shape(t), dtype=bool)
        index = np.searchsorted(starts, t, side="right") - 1
        held = np.maximum(index, 0)
        phase = np.asarray(t, dtype=float) - starts[held]
        inside = (index >= 0) & (np.asarray(t) < stops[held])
        return inside & (np.mod(phase, periods[held]) < self.on_time)

    def edges(self, duration):
        """Sorted transition times in (0, duration], including note boundaries."""
        starts, stops, periods = self.table
        live = starts < duration
        parts = [
            _burst_edges(start, stop, period, self.on_time)
            for start, stop, period in zip(starts[live], stops[live], periods[live])
        ]
        edges = np.sort(np.concatenate(parts) if parts else np.zeros(0), kind="stable")
        return edges[(edges > 0.0) & (edges <= duration)]
