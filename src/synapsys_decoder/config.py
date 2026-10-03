"""Typed decoder configuration loaded from YAML.

All project-specific numbers (frequencies, sample rate, channels, window) live in
``configs/*.yaml`` and reach algorithm code only through these objects or as
explicit function arguments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

FilterMode = Literal["causal", "zero_phase"]


@dataclass(frozen=True)
class PreprocessingConfig:
    """Preprocessing stages. Any stage can be disabled for ablation.

    ``bandpass_low_hz`` / ``bandpass_high_hz`` set to ``None`` disable the
    corresponding edge (both ``None`` disables the bandpass). ``notch_hz=None``
    disables the notch. ``mode="zero_phase"`` is offline-only and not portable
    to the ESP32.
    """

    remove_dc: bool = True
    bandpass_low_hz: float | None = 0.5
    bandpass_high_hz: float | None = 45.0
    bandpass_order: int = 4
    notch_hz: float | None = 60.0
    notch_q: float = 30.0
    mode: FilterMode = "causal"

    def __post_init__(self) -> None:
        if self.mode not in ("causal", "zero_phase"):
            raise ValueError(f"mode must be 'causal' or 'zero_phase'; got {self.mode!r}")
        if self.bandpass_order < 1:
            raise ValueError("bandpass_order must be >= 1")
        lo, hi = self.bandpass_low_hz, self.bandpass_high_hz
        if lo is not None and lo <= 0:
            raise ValueError("bandpass_low_hz must be > 0")
        if hi is not None and hi <= 0:
            raise ValueError("bandpass_high_hz must be > 0")
        if lo is not None and hi is not None and lo >= hi:
            raise ValueError("bandpass_low_hz must be < bandpass_high_hz")
        if self.notch_hz is not None and self.notch_hz <= 0:
            raise ValueError("notch_hz must be > 0")
        if self.notch_q <= 0:
            raise ValueError("notch_q must be > 0")


@dataclass(frozen=True)
class DecisionConfig:
    """Rejection gating. ``None`` means "not configured" — never invent values."""

    confidence_threshold: float | None = None
    margin_threshold: float | None = None


@dataclass(frozen=True)
class DecoderConfig:
    """Full decoder configuration.

    ``targets`` maps class label -> stimulus frequency (Hz); its insertion order
    is the canonical class order.
    """

    sample_rate_hz: float
    channels: tuple[str, ...]
    targets: dict[str, float]
    window_seconds: float
    harmonics: int
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")
        if not self.channels:
            raise ValueError("at least one channel is required")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("channel names must be unique")
        if len(self.targets) < 2:
            raise ValueError("at least two targets are required")
        if any(f <= 0 for f in self.targets.values()):
            raise ValueError("target frequencies must be > 0")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if self.harmonics < 1:
            raise ValueError("harmonics must be >= 1")
        nyquist = self.sample_rate_hz / 2
        for label, f in self.targets.items():
            if f * self.harmonics >= nyquist:
                raise ValueError(
                    f"target {label} harmonic {self.harmonics} ({f * self.harmonics:.3f} Hz) "
                    f"is at/above Nyquist ({nyquist} Hz)"
                )

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.targets)

    @property
    def frequencies(self) -> tuple[float, ...]:
        return tuple(self.targets.values())

    @property
    def window_samples(self) -> int:
        """Analysis window length in samples (rounded to nearest integer)."""
        return round(self.window_seconds * self.sample_rate_hz)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form suitable for saving alongside results."""
        d = asdict(self)
        d["channels"] = list(self.channels)
        extra = d.pop("extra")
        d.update(extra)
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DecoderConfig:
        known = {
            "sample_rate_hz",
            "channels",
            "targets",
            "window_seconds",
            "harmonics",
            "preprocessing",
            "decision",
        }
        missing = {"sample_rate_hz", "channels", "targets", "window_seconds", "harmonics"} - set(
            raw
        )
        if missing:
            raise ValueError(f"config missing required keys: {sorted(missing)}")
        bad = [k for k in raw["targets"] if not isinstance(k, str)]
        if bad:
            raise ValueError(
                f"target labels must be strings; got {bad!r}. Quote labels in YAML "
                '(bare YES/NO parse as booleans): use "YES": 8.57'
            )
        return cls(
            sample_rate_hz=float(raw["sample_rate_hz"]),
            channels=tuple(raw["channels"]),
            targets={k: float(v) for k, v in raw["targets"].items()},
            window_seconds=float(raw["window_seconds"]),
            harmonics=int(raw["harmonics"]),
            preprocessing=PreprocessingConfig(**(raw.get("preprocessing") or {})),
            decision=DecisionConfig(**(raw.get("decision") or {})),
            extra={k: v for k, v in raw.items() if k not in known},
        )

    def replace(self, **changes: Any) -> DecoderConfig:
        """Return a copy with top-level fields replaced (for sweeps)."""
        from dataclasses import replace as _replace

        return _replace(self, **changes)


def load_config(path: str | Path) -> DecoderConfig:
    """Load a :class:`DecoderConfig` from a YAML file."""
    with Path(path).open() as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at top level")
    return DecoderConfig.from_dict(raw)
