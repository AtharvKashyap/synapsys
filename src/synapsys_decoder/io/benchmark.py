"""Loader for the Wang et al. 2017 Tsinghua SSVEP Benchmark dataset.

Expected layout (see ``data/README.md``)::

    <data_dir>/S1.mat ... S35.mat, Freq_Phase.mat, 64-channels.loc

Assumed file format (from the dataset paper / distribution notes):

* ``S{n}.mat``: MATLAB v5, variable ``data`` shaped
  ``[64 channels, 1500 samples, 40 targets, 6 blocks]``, 250 Hz, microvolts.
  Each epoch is 0.5 s pre-stimulus, 5 s stimulation, 0.5 s post-stimulus, so
  stimulus onset is at sample 125.
* ``Freq_Phase.mat``: ``freqs`` (40 values, 8.0-15.8 Hz, 0.2 Hz grid, target
  index order NOT sorted by frequency) and ``phases`` (radians).
* ``64-channels.loc``: EEGLAB loc text; columns index, theta, radius, label.

The authors' distributed data was already downsampled to 250 Hz and notch
filtered at 50 Hz. This loader performs NO filtering, resampling or cropping:
it only slices, converts microvolts to volts and attaches provenance.

NOT verified without the real files: the exact ``data`` variable layout and
dtype, the ``Freq_Phase.mat`` variable shapes, and the label spellings in the
real ``64-channels.loc`` (all are checked defensively and raise clear errors
if violated; the label normalisation rule is generic).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from synapsys_decoder.types import Trial

DATASET_NAME = "wang2017_tsinghua_benchmark"
FS_HZ = 250.0
N_CHANNELS = 64
N_SAMPLES = 1500
N_TARGETS = 40
N_BLOCKS = 6
N_SUBJECTS = 35
STIMULUS_ONSET_SAMPLE = 125
STIMULUS_ONSET_S = 0.5
STIMULATION_DURATION_S = 5.0
UPSTREAM_PROCESSING = (
    "Distributed by the dataset authors already downsampled to 250 Hz and "
    "notch-filtered at 50 Hz (mains); no further processing applied by this loader."
)
FREQ_PHASE_FILE = "Freq_Phase.mat"
LOC_FILE = "64-channels.loc"
_README_HINT = "See data/README.md for where to obtain and place the Wang 2017 Benchmark files."


def _subject_file(subject: int) -> str:
    return f"S{subject}.mat"


def benchmark_files_present(data_dir: str | Path, subject: int = 1) -> bool:
    """True if ``Freq_Phase.mat``, ``64-channels.loc`` and ``S{subject}.mat`` all exist."""
    d = Path(data_dir)
    return all((d / f).is_file() for f in (FREQ_PHASE_FILE, LOC_FILE, _subject_file(subject)))


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark file not found: {path}. {_README_HINT}")
    return path


def normalize_channel_name(raw: str) -> str:
    """Normalise an EEGLAB label to conventional 10-20 capitalisation.

    Trailing dots/whitespace are stripped, the label is uppercased, a trailing
    ``Z`` after the letter prefix becomes ``z`` and a leading ``FP`` becomes ``Fp``
    (``OZ``->``Oz``, ``POZ``->``POz``, ``FP1``->``Fp1``, ``CB1``->``CB1``).
    """
    name = raw.strip().rstrip(".").strip().upper()
    m = re.fullmatch(r"([A-Z]+?)(Z?)(\d*)", name)
    if m is None or not name:
        return name
    letters, z, digits = m.groups()
    # "Z" is only the midline marker when it is the last letter (before digits).
    if letters == "" and z:
        letters, z = z, ""
    out = letters + ("z" if z else "") + digits
    if out.startswith("FP"):
        out = "Fp" + out[2:]
    return out


def _read_loc_raw(loc_path: str | Path) -> list[str]:
    path = _require(Path(loc_path))
    labels: list[str] = []
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 4:
            labels.append(parts[3])
    if not labels:
        raise ValueError(
            f"No channel labels parsed from {path} (expected index theta radius label)."
        )
    return labels


def read_channel_labels_raw(loc_path: str | Path) -> tuple[str, ...]:
    """Raw labels exactly as written in the loc file (fourth column)."""
    return tuple(_read_loc_raw(loc_path))


def read_channel_names(loc_path: str | Path) -> tuple[str, ...]:
    """Normalised channel names in file (= data row) order. See :func:`normalize_channel_name`."""
    names = tuple(normalize_channel_name(r) for r in _read_loc_raw(loc_path))
    lowered = [n.lower() for n in names]
    if len(set(lowered)) != len(lowered):
        dups = sorted({n for n in names if lowered.count(n.lower()) > 1})
        raise ValueError(f"Duplicate channel names after normalisation in {loc_path}: {dups}")
    return names


def read_freq_phase(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read ``(freqs_hz, phases_rad)`` as 1-D float64 arrays of length 40, in target order."""
    path = _require(Path(path))
    mat = loadmat(str(path))
    out = []
    for key in ("freqs", "phases"):
        if key not in mat:
            raise KeyError(f"{path} has no variable '{key}' (found: {_user_keys(mat)}).")
        arr = np.asarray(mat[key], dtype=np.float64).ravel()
        if arr.size != N_TARGETS:
            raise ValueError(f"{path}: '{key}' has {arr.size} values; expected {N_TARGETS}.")
        out.append(arr)
    return out[0], out[1]


def _user_keys(mat: dict) -> list[str]:
    return [k for k in mat if not k.startswith("__")]


def _find_channel_indices(requested: Sequence[str], names: Sequence[str]) -> list[int]:
    lookup = {n.lower(): i for i, n in enumerate(names)}
    idx, missing = [], []
    for r in requested:
        key = normalize_channel_name(r).lower()
        if key in lookup:
            idx.append(lookup[key])
        elif r.strip().lower() in lookup:
            idx.append(lookup[r.strip().lower()])
        else:
            missing.append(r)
    if missing:
        raise KeyError(f"Channels not in 64-channels.loc: {missing}. Available: {list(names)}")
    return idx


def _find_target_indices(requested: Sequence[float], freqs: np.ndarray, tol: float) -> list[int]:
    idx: list[int] = []
    for f in requested:
        hits = np.flatnonzero(np.abs(freqs - float(f)) <= tol)
        if hits.size == 0:
            raise ValueError(
                f"Requested frequency {f} Hz does not exist in the Benchmark dataset. "
                f"Targets lie on a 0.2 Hz grid from {freqs.min():.1f} to {freqs.max():.1f} Hz "
                f"(tolerance {tol} Hz); e.g. SynapSys YES = 60/7 = 8.5714 Hz is NOT available. "
                "Do not substitute or relabel a nearby frequency."
            )
        idx.extend(int(h) for h in hits)
    return sorted(set(idx))


def load_benchmark_subject(
    data_dir: str | Path,
    subject: int,
    *,
    channels: Sequence[str] | None = None,
    frequencies: Sequence[float] | None = None,
    blocks: Sequence[int] | None = None,
    freq_tol_hz: float = 1e-3,
) -> list[Trial]:
    """Load Benchmark epochs for one subject as :class:`Trial` objects.

    Args:
        data_dir: Directory holding ``S{n}.mat``, ``Freq_Phase.mat``, ``64-channels.loc``.
        subject: 1-based subject number (1..35).
        channels: Channel names to keep, in the requested order (case-insensitive).
            ``None`` keeps all 64 in file order.
        frequencies: Keep only targets whose frequency (from ``Freq_Phase.mat``)
            matches within ``freq_tol_hz``. Raises if a frequency is not in the dataset.
        blocks: 0-based block indices (0..5). ``None`` keeps all.
        freq_tol_hz: Matching tolerance for ``frequencies``.

    Returns:
        Full 6 s epochs (no cropping), eeg ``(n_channels, 1500)`` float64 volts, ordered
        by block then target index. ``true_label`` is None (assigned by experiment
        scripts). Stimulus onset (sample 125 / 0.5 s) is in ``metadata``.
    """
    d = Path(data_dir)
    if not 1 <= int(subject) <= N_SUBJECTS:
        raise ValueError(f"subject must be in 1..{N_SUBJECTS}; got {subject}")
    mat_path = _require(d / _subject_file(int(subject)))
    freqs, phases = read_freq_phase(_require(d / FREQ_PHASE_FILE))
    raw_labels = read_channel_labels_raw(_require(d / LOC_FILE))
    names = read_channel_names(d / LOC_FILE)
    if len(names) != N_CHANNELS:
        raise ValueError(f"{LOC_FILE} has {len(names)} channels; expected {N_CHANNELS}.")

    mat = loadmat(str(mat_path))
    if "data" not in mat:
        raise KeyError(f"{mat_path} has no variable 'data' (found: {_user_keys(mat)}).")
    data = mat["data"]
    expected = (N_CHANNELS, N_SAMPLES, N_TARGETS, N_BLOCKS)
    if data.ndim != 4 or data.shape != expected:
        raise ValueError(
            f"{mat_path}: 'data' has shape {data.shape}; expected {expected} "
            "[channels, samples, targets, blocks]."
        )

    ch_idx = list(range(N_CHANNELS)) if channels is None else _find_channel_indices(channels, names)
    tgt_idx = (
        list(range(N_TARGETS))
        if frequencies is None
        else _find_target_indices(frequencies, freqs, freq_tol_hz)
    )
    if blocks is None:
        blk_idx = list(range(N_BLOCKS))
    else:
        bad = [b for b in blocks if not 0 <= int(b) < N_BLOCKS]
        if bad:
            raise ValueError(f"blocks are 0-based in 0..{N_BLOCKS - 1}; invalid: {bad}")
        blk_idx = [int(b) for b in blocks]

    sel_names = tuple(names[i] for i in ch_idx)
    sel_raw = tuple(raw_labels[i] for i in ch_idx)
    trials: list[Trial] = []
    for b in blk_idx:
        for t in tgt_idx:
            # Slice before converting so only the requested samples become float64.
            eeg = np.asarray(data[ch_idx, :, t, b], dtype=np.float64) * 1e-6
            trials.append(
                Trial(
                    eeg=eeg,
                    fs=FS_HZ,
                    channel_names=sel_names,
                    true_frequency=float(freqs[t]),
                    true_label=None,
                    subject_id=f"S{int(subject)}",
                    block_id=str(b),
                    trial_id=f"S{int(subject)}-b{b}-t{t}",
                    metadata={
                        "dataset": DATASET_NAME,
                        "source_file": str(mat_path),
                        "target_index": int(t),
                        "phase_rad": float(phases[t]),
                        "stimulus_onset_sample": STIMULUS_ONSET_SAMPLE,
                        "stimulus_onset_s": STIMULUS_ONSET_S,
                        "stimulation_duration_s": STIMULATION_DURATION_S,
                        "cropping": "none (full 6 s epoch: 0.5 s pre, 5 s stimulation, 0.5 s post)",
                        "units_source": "uV",
                        "upstream_processing": UPSTREAM_PROCESSING,
                        "channel_names_raw": sel_raw,
                    },
                )
            )
    return trials
