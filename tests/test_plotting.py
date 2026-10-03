import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure

from synapsys_decoder import plotting as pl

FS = 250.0
CH = ["O1", "Oz"]
MARK = {"YES": 8.57, "NO": 12.0}
EEG = np.random.default_rng(0).normal(0, 1e-6, (2, 1000))


def test_psd():
    assert isinstance(pl.plot_psd(EEG, FS, CH, mark_frequencies=MARK), Figure)
    assert isinstance(pl.plot_psd(EEG, FS, CH), Figure)


def test_psd_before_after():
    assert isinstance(pl.plot_psd_before_after(EEG, EEG * 0.5, FS, CH, MARK), Figure)


def test_cca_scores():
    s = {"YES": 0.2, "NO": 0.6, "HELP": 0.3}
    assert isinstance(
        pl.plot_cca_scores(s, true_label="NO", frequencies={"YES": 8.57, "NO": 12, "HELP": 15}),
        Figure,
    )
    assert isinstance(pl.plot_cca_scores(s), Figure)


def test_confusion():
    cm = np.array([[3, 1, 0], [0, 4, 0]])
    for norm in (True, False):
        assert isinstance(
            pl.plot_confusion_matrix(cm, ["A", "B"], ["A", "B", "REJECT"], normalize=norm), Figure
        )
    assert isinstance(
        pl.plot_confusion_matrix(np.zeros((2, 3)), ["A", "B"], ["A", "B", "R"]), Figure
    )


def test_metric_curve():
    x = [-20, -10, 0]
    assert isinstance(
        pl.plot_metric_curve(x, [0.3, 0.6, 0.9], xlabel="SNR", ylabel="acc", chance_level=1 / 3),
        Figure,
    )
    f = pl.plot_metric_curve(
        x, xlabel="SNR", ylabel="acc", series={"a": [0.3, 0.6, 0.9], "b": [0.4, 0.5, 0.6]}
    )
    assert isinstance(f, Figure)


def test_score_distributions(tmp_path):
    d = {
        "YES": {"YES": [0.6, 0.7, 0.65], "NO": [0.2, 0.3]},
        "NO": {"YES": [0.2, 0.25], "NO": [0.7, 0.8, 0.75]},
    }
    fig = pl.plot_score_distributions(d)
    assert isinstance(fig, Figure)
    out = pl.save_figure(fig, tmp_path / "sub" / "dir" / "f.png")
    assert out.exists() and out.stat().st_size > 0


def test_filter_response_and_table_smoke(tmp_path):
    from synapsys_decoder.plotting import plot_filter_response, plot_table, save_figure

    f = np.linspace(0.1, 100, 50)
    fig = plot_filter_response(
        f,
        {"a": -f / 10, "b": -f / 5},
        keep_frequencies={"x": 12.0},
        reject_frequencies={"mains": 60.0},
    )
    assert save_figure(fig, tmp_path / "r.png").exists()
    fig = plot_table(["a", "b"], [["1", "2"], ["3", "4"]], title="t", highlight_rows=[1])
    assert save_figure(fig, tmp_path / "t.png").exists()
