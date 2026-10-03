"""Classification metrics for string-labelled decisions that may include REJECT.

Labels are strings; predictions may additionally be
:data:`synapsys_decoder.types.REJECT`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from synapsys_decoder.types import REJECT


def _check_pair(y_true: Sequence[str], y_pred: Sequence[str]) -> None:
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("inputs must be non-empty")
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_pred)}")


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Total command accuracy: correct / all trials. REJECT counts as wrong."""
    _check_pair(y_true, y_pred)
    return sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)


def accepted_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float | None:
    """Accuracy over non-REJECT predictions only; ``None`` if nothing was accepted."""
    _check_pair(y_true, y_pred)
    pairs = [(t, p) for t, p in zip(y_true, y_pred, strict=True) if p != REJECT]
    if not pairs:
        return None
    return sum(t == p for t, p in pairs) / len(pairs)


def rejection_rate(y_pred: Sequence[str]) -> float:
    """Fraction of predictions equal to REJECT."""
    if len(y_pred) == 0:
        raise ValueError("y_pred must be non-empty")
    return sum(p == REJECT for p in y_pred) / len(y_pred)


def confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    *,
    include_reject: bool = True,
) -> np.ndarray:
    """Counts matrix: rows = true (``labels`` order), cols = predicted.

    Columns are ``labels`` plus a final REJECT column when ``include_reject``.
    Unknown true labels, or unknown non-REJECT predictions, raise ValueError.
    If ``include_reject`` is False, a REJECT prediction also raises.
    """
    _check_pair(y_true, y_pred)
    row = {lab: i for i, lab in enumerate(labels)}
    col = dict(row)
    if include_reject:
        col[REJECT] = len(labels)
    cm = np.zeros((len(labels), len(col)), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        if t not in row:
            raise ValueError(f"unknown true label: {t!r}")
        if p not in col:
            raise ValueError(f"unknown predicted label: {p!r}")
        cm[row[t], col[p]] += 1
    return cm


def per_class_accuracy(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> dict[str, float | None]:
    """Recall per true class (REJECT counts wrong); ``None`` if the class has no trials."""
    _check_pair(y_true, y_pred)
    out: dict[str, float | None] = {}
    for lab in labels:
        pairs = [p for t, p in zip(y_true, y_pred, strict=True) if t == lab]
        out[lab] = (sum(p == lab for p in pairs) / len(pairs)) if pairs else None
    return out


def itr_bits_per_min(n_classes: int, accuracy: float, seconds_per_selection: float) -> float:
    """Wolpaw information transfer rate in bits/min.

    ``B = log2 N + P log2 P + (1-P) log2((1-P)/(N-1))`` bits per selection,
    times ``60 / T``. Edge cases: ``P == 1`` drops the zero-valued terms
    (``B = log2 N``); ``P <= 1/N`` returns 0 (the formula would go negative;
    chance level carries no information). ``0 * log 0`` is taken as 0.
    """
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2")
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError("accuracy must be in [0, 1]")
    if seconds_per_selection <= 0:
        raise ValueError("seconds_per_selection must be positive")
    n, p = n_classes, accuracy
    if p <= 1.0 / n:
        return 0.0
    bits = math.log2(n)
    if p < 1.0:
        bits += p * math.log2(p) + (1 - p) * math.log2((1 - p) / (n - 1))
    return bits * 60.0 / seconds_per_selection


def false_activation_rate(y_pred_on_rest: Sequence[str]) -> float:
    """Fraction of rest/no-command windows that were NOT rejected."""
    if len(y_pred_on_rest) == 0:
        raise ValueError("y_pred_on_rest must be non-empty")
    return sum(p != REJECT for p in y_pred_on_rest) / len(y_pred_on_rest)


def summarize(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> dict[str, Any]:
    """JSON-serializable summary of all metrics (lists, floats, ints, None)."""
    cm = confusion_matrix(y_true, y_pred, labels)
    return {
        "n_trials": len(y_true),
        "labels": list(labels),
        "accuracy": float(accuracy(y_true, y_pred)),
        "accepted_accuracy": accepted_accuracy(y_true, y_pred),
        "rejection_rate": float(rejection_rate(y_pred)),
        "per_class_accuracy": per_class_accuracy(y_true, y_pred, labels),
        "confusion_matrix": cm.tolist(),
        "confusion_columns": [*labels, REJECT],
    }
