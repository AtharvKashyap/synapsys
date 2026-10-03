import json
import math

import numpy as np
import pytest

from synapsys_decoder import metrics as m
from synapsys_decoder.types import REJECT

L = ["YES", "NO", "HELP"]
T = ["YES", "YES", "NO", "NO", "HELP", "HELP"]
P = ["YES", "NO", REJECT, "NO", "HELP", REJECT]


def test_accuracy_reject_wrong():
    assert m.accuracy(T, P) == pytest.approx(3 / 6)


def test_accepted_accuracy():
    assert m.accepted_accuracy(T, P) == pytest.approx(3 / 4)
    assert m.accepted_accuracy(["YES"], [REJECT]) is None


def test_rejection_rate():
    assert m.rejection_rate(P) == pytest.approx(2 / 6)


def test_confusion_matrix():
    cm = m.confusion_matrix(T, P, L)
    expected = np.array([[1, 1, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1]])
    np.testing.assert_array_equal(cm, expected)
    cm2 = m.confusion_matrix(["YES", "NO"], ["NO", "NO"], L, include_reject=False)
    np.testing.assert_array_equal(cm2, [[0, 1, 0], [0, 1, 0], [0, 0, 0]])


def test_confusion_unknown_raises():
    with pytest.raises(ValueError):
        m.confusion_matrix(["X"], ["YES"], L)
    with pytest.raises(ValueError):
        m.confusion_matrix(["YES"], ["X"], L)
    with pytest.raises(ValueError):
        m.confusion_matrix(["YES"], [REJECT], L, include_reject=False)


def test_per_class_accuracy():
    out = m.per_class_accuracy(T, P, L)
    assert out == {"YES": 0.5, "NO": 0.5, "HELP": 0.5}
    out2 = m.per_class_accuracy(["YES"], ["YES"], L)
    assert out2 == {"YES": 1.0, "NO": None, "HELP": None}


def test_itr_known_values():
    assert m.itr_bits_per_min(3, 1.0, 2.5) == pytest.approx(math.log2(3) * 60 / 2.5)
    assert m.itr_bits_per_min(3, 1 / 3, 2.5) == 0.0
    assert m.itr_bits_per_min(3, 0.1, 2.5) == 0.0
    p = 0.9
    bits = math.log2(3) + p * math.log2(p) + (1 - p) * math.log2((1 - p) / 2)
    assert m.itr_bits_per_min(3, p, 2.5) == pytest.approx(bits * 24)
    with pytest.raises(ValueError):
        m.itr_bits_per_min(1, 0.5, 1)
    with pytest.raises(ValueError):
        m.itr_bits_per_min(3, 1.5, 1)
    with pytest.raises(ValueError):
        m.itr_bits_per_min(3, 0.5, 0)


def test_false_activation_rate():
    assert m.false_activation_rate([REJECT, "YES", REJECT, REJECT]) == 0.25


def test_empty_inputs_raise():
    for fn in (m.accuracy, m.accepted_accuracy):
        with pytest.raises(ValueError):
            fn([], [])
    with pytest.raises(ValueError):
        m.rejection_rate([])
    with pytest.raises(ValueError):
        m.false_activation_rate([])
    with pytest.raises(ValueError):
        m.confusion_matrix([], [], L)
    with pytest.raises(ValueError):
        m.accuracy(["YES"], ["YES", "NO"])


def test_summarize_json():
    s = m.summarize(T, P, L)
    json.dumps(s)
    assert s["accuracy"] == 0.5
    assert s["confusion_columns"] == [*L, REJECT]
    assert s["n_trials"] == 6
    json.dumps(m.summarize(["YES"], [REJECT], L))
