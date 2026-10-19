"""Parse the ESP32 firmware's serial output into raw recordings and device decisions.

Line formats (written by ``firmware/esp32/main/serial_output.cpp``)::

    SSVEP {"event":"raw_info","fs":250,"units":"uV","channels":[...],"source":..,"stream":..}
    RAW <seq> <v0> <v1> <v2> <v3>           one line per sample, microvolts, before filtering
    SSVEP {"event":"raw_gap","stream":..,"lost":N,"next_seq":S}
    SSVEP {"t":..,"status":..,"scores":[..],"winner":..,"margin":..,"decision":..,..}
    SSVEP {"event":"command","label":..,"t":..}
    anything else (ESP-IDF log lines, health JSON, ...) is kept only as JSON events or ignored

A ``raw_info`` line starts a new stream (the device resets its decoder there). ``RAW``
lines that arrive before any ``raw_info`` (recording started mid-stream) form a stream
whose fs/channels must be supplied by the caller. Within a stream, ``seq`` must increase
by exactly 1; a ``raw_gap`` event or a jump in ``seq`` marks the stream as gappy, and
:func:`RawStream.to_trial` refuses it unless asked to keep the longest contiguous run.

Like every loader: nothing is filtered or resampled; values are converted to volts.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from synapsys_decoder.types import Trial

PREFIX = "SSVEP "
RAW_PREFIX = "RAW "
_UNIT_TO_VOLTS = {"V": 1.0, "mV": 1e-3, "uV": 1e-6}


@dataclass
class RawStream:
    """Samples of one device stream (between two ``raw_info`` lines)."""

    info: dict[str, Any]
    seq: list[int] = field(default_factory=list)
    values: list[list[float]] = field(default_factory=list)  # device units, per sample
    gaps: list[dict[str, Any]] = field(default_factory=list)  # raw_gap events + seq jumps
    decisions: list[dict[str, Any]] = field(default_factory=list)  # device decision lines
    commands: list[dict[str, Any]] = field(default_factory=list)
    malformed: int = 0

    @property
    def n_samples(self) -> int:
        return len(self.seq)

    @property
    def contiguous(self) -> bool:
        return not self.gaps and self.malformed == 0

    def longest_contiguous_run(self) -> tuple[int, int]:
        """Index range ``[i0, i1)`` of the longest run with seq increasing by exactly 1."""
        if not self.seq:
            return (0, 0)
        best, start = (0, 1), 0
        for i in range(1, len(self.seq)):
            if self.seq[i] != self.seq[i - 1] + 1:
                start = i
            if i + 1 - start > best[1] - best[0]:
                best = (start, i + 1)
        return best

    def to_trial(
        self,
        *,
        fs: float | None = None,
        channels: Sequence[str] | None = None,
        allow_gaps: bool = False,
        **trial_fields: Any,
    ) -> Trial:
        """Convert to a :class:`Trial` in volts.

        ``fs`` / ``channels`` default to the ``raw_info`` line; they are required when the
        stream has none. Extra keyword arguments (``true_label``, ``subject_id``, ...)
        are passed to :class:`Trial`; ``metadata`` is merged with the provenance here.

        Raises:
            ValueError: gappy stream without ``allow_gaps``, unknown fs/channels/units,
                or a channel count that does not match the RAW columns.
        """
        if not self.seq:
            raise ValueError("stream has no RAW samples")
        i0, i1 = 0, len(self.seq)
        if not self.contiguous:
            if not allow_gaps:
                raise ValueError(
                    f"stream {self.info.get('stream')!r} is not contiguous "
                    f"({len(self.gaps)} gap(s), {self.malformed} malformed line(s)); "
                    "re-record, or pass allow_gaps=True to keep the longest contiguous run"
                )
            i0, i1 = self.longest_contiguous_run()
        fs_v = fs if fs is not None else self.info.get("fs")
        names = list(channels) if channels is not None else self.info.get("channels")
        units = self.info.get("units", "uV")
        if fs_v is None or names is None:
            raise ValueError("stream has no raw_info line: pass fs= and channels=")
        if units not in _UNIT_TO_VOLTS:
            raise ValueError(f"unknown units {units!r}")
        data = np.asarray(self.values[i0:i1], dtype=np.float64)
        if data.shape[1] != len(names):
            raise ValueError(f"RAW lines have {data.shape[1]} values but {len(names)} channels")
        meta = dict(trial_fields.pop("metadata", {}) or {})
        meta.update(
            {
                "device_source": self.info.get("source"),
                "device_stream": self.info.get("stream"),
                "first_seq": self.seq[i0],
                "last_seq": self.seq[i1 - 1],
                "kept_samples": i1 - i0,
                "total_samples": len(self.seq),
                "gaps": len(self.gaps),
                "units_source": units,
            }
        )
        return Trial(
            eeg=np.ascontiguousarray(data.T) * _UNIT_TO_VOLTS[units],
            fs=float(fs_v),
            channel_names=tuple(names),
            metadata=meta,
            **trial_fields,
        )


@dataclass
class SerialLog:
    streams: list[RawStream]
    events: list[dict[str, Any]]  # every SSVEP JSON line, in order (incl. health, summaries)


def parse_serial_lines(lines: Iterable[str]) -> SerialLog:
    """Parse firmware serial output (see module docstring). Never raises on bad lines."""
    streams: list[RawStream] = []
    events: list[dict[str, Any]] = []
    cur: RawStream | None = None

    def current() -> RawStream:
        nonlocal cur
        if cur is None:
            cur = RawStream(info={})
            streams.append(cur)
        return cur

    for raw_line in lines:
        line = raw_line.strip()
        # The firmware writes RAW at the start of a line; elsewhere (e.g. a boot log line
        # describing the format) it is not a sample.
        if line.startswith(RAW_PREFIX):
            parts = line[len(RAW_PREFIX) :].split()
            st = current()
            try:
                seq = int(parts[0])
                vals = [float(p) for p in parts[1:]]
                if not vals or (st.values and len(vals) != len(st.values[-1])):
                    raise ValueError
            except (ValueError, IndexError):
                st.malformed += 1
                continue
            if st.seq and seq != st.seq[-1] + 1:
                st.gaps.append({"event": "seq_jump", "after_seq": st.seq[-1], "next_seq": seq})
            st.seq.append(seq)
            st.values.append(vals)
            continue
        p = line.find(PREFIX)
        if p == -1:
            continue
        try:
            obj = json.loads(line[p + len(PREFIX) :])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        events.append(obj)
        ev = obj.get("event")
        if ev == "raw_info":
            cur = RawStream(info=obj)
            streams.append(cur)
        elif ev == "raw_gap":
            current().gaps.append(obj)
        elif ev == "command":
            _owner(streams, obj, current).commands.append(obj)
        elif ev is None and "scores" in obj and "t" in obj:
            _owner(streams, obj, current).decisions.append(obj)
    return SerialLog(streams=streams, events=events)


def _owner(streams: list[RawStream], obj: dict[str, Any], current: Any) -> RawStream:
    # Raw samples and decisions travel through different device queues, so a decision of
    # the previous replay stream may print after the next stream's raw_info. Replay lines
    # name their stream; attach to the most recent stream of that name.
    name = obj.get("stream")
    if name is not None:
        for st in reversed(streams):
            if st.info.get("stream") == name:
                return st
    return current()
