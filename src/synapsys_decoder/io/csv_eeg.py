"""CSV EEG trial format for own SynapSys recordings and regression fixtures.

Format specification (also the planned Python <-> ESP32 fixture format)::

    <name>.csv   header row of column names, then one row per sample.
                 Optional column ``time_s`` (seconds, must be strictly increasing;
                 never used as data). Every other column is one EEG channel, named
                 by its header. Values are plain decimals in the units given by the
                 sidecar, written with 10 significant digits (``%.10g``).
    <name>.json  sidecar with required ``fs`` (Hz) and ``units`` ("V"|"mV"|"uV"),
                 and optional ``subject_id``, ``block_id``, ``trial_id``,
                 ``true_label``, ``true_frequency`` (Hz), ``metadata`` (JSON object).

Loading converts to volts, float64, shape ``(n_channels, n_samples)``. Explicit
``fs`` / ``units`` arguments override the sidecar; if the sidecar is absent they
are required. Nothing is filtered or resampled. Round trips are exact to ~1e-10
relative (10 significant digits) of the stored unit.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from synapsys_decoder.types import Trial

TIME_COLUMN = "time_s"
_UNIT_TO_VOLTS = {"V": 1.0, "mV": 1e-3, "uV": 1e-6}
_SIDE_KEYS = ("subject_id", "block_id", "trial_id", "true_label", "true_frequency")


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(".json")


def load_csv_trial(
    path: str | Path,
    *,
    fs: float | None = None,
    units: str | None = None,
    channels: Sequence[str] | None = None,
) -> Trial:
    """Load a CSV (+ optional JSON sidecar) into a :class:`Trial` in volts.

    Args:
        path: Path to the ``.csv`` file; the sidecar is ``<same name>.json``.
        fs: Sample rate in Hz; overrides the sidecar.
        units: ``"V"``, ``"mV"`` or ``"uV"``; overrides the sidecar.
        channels: Channel names to keep, in this order (exact names).

    Raises:
        FileNotFoundError: CSV missing.
        ValueError: fs/units unknown, bad units, malformed CSV, non-monotonic time.
        KeyError: a requested channel is absent.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV trial file not found: {path}")
    side: dict = {}
    sp = _sidecar_path(path)
    if sp.is_file():
        side = json.loads(sp.read_text())
    fs_v = fs if fs is not None else side.get("fs")
    units_v = units if units is not None else side.get("units")
    if fs_v is None:
        raise ValueError(f"fs unknown for {path}: pass fs= or provide it in sidecar {sp}.")
    if units_v is None:
        raise ValueError(f"units unknown for {path}: pass units= or provide it in sidecar {sp}.")
    if units_v not in _UNIT_TO_VOLTS:
        raise ValueError(f"units must be one of {sorted(_UNIT_TO_VOLTS)}; got {units_v!r}")

    with path.open() as fh:
        header = [h.strip() for h in fh.readline().strip().split(",")]
    if not header or header == [""]:
        raise ValueError(f"{path} has an empty header row.")
    if len(set(header)) != len(header):
        raise ValueError(f"{path} has duplicate column names: {header}")
    table = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    if table.shape[1] != len(header):
        raise ValueError(f"{path}: {table.shape[1]} data columns but {len(header)} header names.")

    if TIME_COLUMN in header:
        ti = header.index(TIME_COLUMN)
        if table.shape[0] > 1 and not np.all(np.diff(table[:, ti]) > 0):
            raise ValueError(f"{path}: '{TIME_COLUMN}' column is not strictly increasing.")
        keep = [i for i in range(len(header)) if i != ti]
    else:
        keep = list(range(len(header)))
    names = [header[i] for i in keep]
    if channels is not None:
        missing = [c for c in channels if c not in names]
        if missing:
            raise KeyError(f"channels not in {path}: {missing}. Available: {names}")
        keep = [keep[names.index(c)] for c in channels]
        names = list(channels)
    if not names:
        raise ValueError(f"{path} has no channel columns.")

    eeg = np.ascontiguousarray(table[:, keep].T) * _UNIT_TO_VOLTS[units_v]
    tf = side.get("true_frequency")
    return Trial(
        eeg=eeg,
        fs=float(fs_v),
        channel_names=tuple(names),
        true_frequency=None if tf is None else float(tf),
        true_label=side.get("true_label"),
        subject_id=side.get("subject_id"),
        block_id=side.get("block_id"),
        trial_id=side.get("trial_id"),
        metadata={**side.get("metadata", {}), "source_file": str(path), "units_source": units_v},
    )


def save_csv_trial(
    trial: Trial,
    path: str | Path,
    *,
    units: str = "uV",
    include_time: bool = True,
) -> None:
    """Write ``trial`` to ``path`` (CSV) and ``path`` with ``.json`` suffix (sidecar).

    Values are converted from volts to ``units`` and written with ``%.10g``.
    Metadata must be JSON-serialisable (non-serialisable values are stringified).
    """
    if units not in _UNIT_TO_VOLTS:
        raise ValueError(f"units must be one of {sorted(_UNIT_TO_VOLTS)}; got {units!r}")
    if TIME_COLUMN in trial.channel_names:
        raise ValueError(f"channel name {TIME_COLUMN!r} is reserved for the time column.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = (trial.eeg / _UNIT_TO_VOLTS[units]).T
    header = list(trial.channel_names)
    if include_time:
        cols = np.column_stack([np.arange(trial.n_samples) / trial.fs, cols])
        header = [TIME_COLUMN, *header]
    np.savetxt(path, cols, delimiter=",", fmt="%.10g", header=",".join(header), comments="")
    side = {"fs": trial.fs, "units": units}
    for k in _SIDE_KEYS:
        v = getattr(trial, k)
        if v is not None:
            side[k] = v
    side["metadata"] = dict(trial.metadata)
    _sidecar_path(path).write_text(json.dumps(side, indent=2, default=str))
