"""Command re-arming: turn per-window decisions into discrete command events.

The decoder outputs one decision (a label or ``REJECT``) every hop (0.5 s =
125 samples). Staring at a target must not repeat the command, so this state
machine emits a label at most once per "gaze" and then waits to be re-armed.

Everything is integer logic: no floats, no wall clock. Time is the
``window_end_sample`` index passed with every decision. This module is the
specification for the C++ port; behaviour must match it exactly.

Parameters (:class:`RearmConfig`)
    K ``confirm_count``       same label on K consecutive decision points -> emit
    R ``refractory_samples``  min samples between an emit and the next possible re-arm
    N ``release_count``       consecutive "released" points needed to re-arm

A decision point is *released* with respect to label L if it is ``REJECT`` or a
label different from L.

State table (``update(decision, t)``; ``t`` = window_end_sample)
    ============  ============================  =====================================
    state         input                         next state / output
    ============  ============================  =====================================
    ARMED         REJECT                        ARMED, no output
    ARMED         label L, K > 1                CANDIDATE(L, 1), no output
    ARMED         label L, K == 1               REFRACTORY(L, t, 0), EMIT L
    CANDIDATE     same label L, k+1 < K         CANDIDATE(L, k+1), no output
    CANDIDATE     same label L, k+1 == K        REFRACTORY(L, t, 0), EMIT L
    CANDIDATE     different label M             CANDIDATE(M, 1), no output
    CANDIDATE     REJECT                        ARMED, no output
    REFRACTORY    label == L                    released = 0
    REFRACTORY    REJECT or label != L          released += 1
    REFRACTORY    (after the update above) if t - emitted_at >= R and released >= N:
                                                ARMED
    ============  ============================  =====================================
    REFRACTORY never emits (it blocks every label, including a different one,
    until re-armed). In all rows REFRACTORY output is None.

Decisions made explicit
    * The decision point that completes the release (moves REFRACTORY -> ARMED)
      does NOT itself start a candidate, even if it is a label; the next point does.
    * When a point is processed in REFRACTORY, the ``released`` update happens first,
      then the exit condition is checked on the updated value.
    * With N == 0 the release condition is always true, so only R gates re-arming.
    * ``emitted_at`` is the ``window_end_sample`` of the emitting point; the
      distance check is ``t - emitted_at >= R`` (inclusive).
    * ``window_end_sample`` must be non-decreasing across calls (equal is allowed);
      otherwise ValueError. ``reset()`` clears this memory too.

Known limitation
    Until the confidence/margin thresholds T/M are calibrated the decoder never
    outputs REJECT, so "looking away" is detectable only as a label change.
    R provides the time-based fallback for repeating the same command.
"""

from __future__ import annotations

from dataclasses import dataclass

from synapsys_decoder.types import REJECT

ARMED = "ARMED"
CANDIDATE = "CANDIDATE"
REFRACTORY = "REFRACTORY"


@dataclass(frozen=True)
class RearmConfig:
    confirm_count: int = 2
    refractory_samples: int = 625
    release_count: int = 2

    def __post_init__(self) -> None:
        for name in ("confirm_count", "refractory_samples", "release_count"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"{name} must be an int; got {v!r}")
        if self.confirm_count < 1:
            raise ValueError(f"confirm_count must be >= 1; got {self.confirm_count}")
        if self.refractory_samples < 0:
            raise ValueError(f"refractory_samples must be >= 0; got {self.refractory_samples}")
        if self.release_count < 0:
            raise ValueError(f"release_count must be >= 0; got {self.release_count}")


class RearmStateMachine:
    """Stateful decision-stream -> command-event converter (see module docstring)."""

    def __init__(self, cfg: RearmConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else RearmConfig()
        self.reset()

    def reset(self) -> None:
        self._state = ARMED
        self._label = ""
        self._count = 0  # candidate count
        self._emitted_at = 0
        self._released = 0
        self._last_t: int | None = None

    @property
    def state(self) -> str:
        return self._state

    def update(self, decision: str, window_end_sample: int) -> str | None:
        t = int(window_end_sample)
        if self._last_t is not None and t < self._last_t:
            raise ValueError(
                f"window_end_sample must be non-decreasing; got {t} after {self._last_t}"
            )
        self._last_t = t
        cfg = self.cfg
        is_reject = decision == REJECT

        if self._state == ARMED:
            if is_reject:
                return None
            return self._start_or_emit(decision, 1, t)

        if self._state == CANDIDATE:
            if is_reject:
                self._state = ARMED
                return None
            if decision == self._label:
                return self._start_or_emit(decision, self._count + 1, t)
            return self._start_or_emit(decision, 1, t)

        # REFRACTORY: never emits.
        if is_reject or decision != self._label:
            self._released += 1
        else:
            self._released = 0
        if t - self._emitted_at >= cfg.refractory_samples and self._released >= cfg.release_count:
            self._state = ARMED
        return None

    def _start_or_emit(self, label: str, count: int, t: int) -> str | None:
        if count >= self.cfg.confirm_count:
            self._state = REFRACTORY
            self._label = label
            self._emitted_at = t
            self._released = 0
            self._count = 0
            return label
        self._state = CANDIDATE
        self._label = label
        self._count = count
        return None
