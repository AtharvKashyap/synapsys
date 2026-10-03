"""Static report figures (matplotlib). Never call from algorithm modules.

Every function takes data, optionally an ``ax``, and returns a ``Figure``;
none calls ``plt.show``. Use :func:`save_figure` to write a PNG.

Design: a colour-blind-safe categorical palette (Okabe-Ito subset) assigned in
fixed order, thin lines, recessive grids, text in neutral ink (never series
colour), a legend whenever there are two or more series, and a highlight colour
reserved for "the true / winning" item against neutral grey for the rest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import signal as sps

#: Fixed-order categorical palette (Okabe-Ito subset, colour-blind safe).
PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000")
INK = "#222222"
MUTED = "#6b6b6b"
GRID = "#dddddd"
NEUTRAL = "#b8b8b8"
HIGHLIGHT = PALETTE[0]
TRUE_MARK = PALETTE[1]


def _style(ax: Axes, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _get_ax(ax: Axes | None, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    if ax is not None:
        return ax.figure, ax  # type: ignore[return-value]
    fig = Figure(figsize=figsize, layout="constrained")
    return fig, fig.add_subplot(111)


def save_figure(fig: Figure, path: str | Path) -> Path:
    """Save ``fig`` as PNG (dpi=150), creating parent directories, then close it."""
    import matplotlib.pyplot as plt

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150, format="png")
    plt.close(fig)
    return p


def _psd(eeg: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(eeg.shape[1], max(int(2 * fs), 8))
    return sps.welch(eeg, fs=fs, nperseg=nperseg, axis=-1)


def _draw_psd(
    ax: Axes,
    eeg: np.ndarray,
    fs: float,
    channel_names: Sequence[str],
    mark_frequencies: Mapping[str, float] | None,
    harmonics: int,
    fmax: float,
) -> None:
    f, pxx = _psd(eeg, fs)
    keep = f <= fmax
    for i, name in enumerate(channel_names):
        ax.semilogy(f[keep], pxx[i, keep], color=PALETTE[i % len(PALETTE)], lw=1.2, label=name)
    ymax = ax.get_ylim()[1]
    if mark_frequencies:
        for label, f0 in mark_frequencies.items():
            for h in range(1, harmonics + 1):
                if h * f0 > fmax:
                    continue
                ax.axvline(h * f0, color=MUTED, lw=0.8, ls="--" if h == 1 else ":")
                ax.text(
                    h * f0,
                    ymax,
                    label if h == 1 else f"{label} x{h}",
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=8,
                    color=MUTED,
                )
    ax.set_xlim(0, fmax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V$^2$/Hz)")
    _style(ax, "both")
    ax.legend(frameon=False, fontsize=8, ncols=min(len(channel_names), 4), loc="lower left")


def plot_psd(
    eeg: np.ndarray,
    fs: float,
    channel_names: Sequence[str],
    *,
    mark_frequencies: Mapping[str, float] | None = None,
    harmonics: int = 2,
    fmax: float = 50.0,
    title: str = "Power spectral density",
    ax: Axes | None = None,
) -> Figure:
    """Welch PSD (V^2/Hz, log y) per channel with target/harmonic markers."""
    fig, ax = _get_ax(ax, (7, 4))
    _draw_psd(ax, eeg, fs, channel_names, mark_frequencies, harmonics, fmax)
    ax.set_title(title, fontsize=11)
    return fig


def plot_psd_before_after(
    raw: np.ndarray,
    filtered: np.ndarray,
    fs: float,
    channel_names: Sequence[str],
    mark_frequencies: Mapping[str, float] | None = None,
    *,
    harmonics: int = 2,
    fmax: float = 50.0,
    title: str = "PSD before / after preprocessing",
) -> Figure:
    """Two side-by-side PSD panels with shared axes (raw vs filtered)."""
    fig = Figure(figsize=(11, 4), layout="constrained")
    ax1, ax2 = fig.subplots(1, 2, sharex=True, sharey=True)
    _draw_psd(ax1, raw, fs, channel_names, mark_frequencies, harmonics, fmax)
    _draw_psd(ax2, filtered, fs, channel_names, mark_frequencies, harmonics, fmax)
    ax1.set_title("Before", fontsize=10)
    ax2.set_title("After", fontsize=10)
    fig.suptitle(title, fontsize=11, color=INK)
    return fig


def plot_cca_scores(
    scores: Mapping[str, float],
    *,
    true_label: str | None = None,
    frequencies: Mapping[str, float] | None = None,
    title: str = "CCA scores",
    ax: Axes | None = None,
) -> Figure:
    """Bar chart of per-class scores; winner highlighted, true class outlined."""
    fig, ax = _get_ax(ax, (5, 3.5))
    labels = list(scores)
    vals = [scores[k] for k in labels]
    winner = labels[int(np.argmax(vals))]
    colors = [HIGHLIGHT if k == winner else NEUTRAL for k in labels]
    edges = [TRUE_MARK if k == true_label else "none" for k in labels]
    ax.bar(labels, vals, color=colors, edgecolor=edges, linewidth=2.5, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
    if frequencies:
        ax.set_xticks(
            range(len(labels)),
            [f"{k}\n{frequencies[k]:.2f} Hz" if k in frequencies else k for k in labels],
        )
    ax.set_ylabel("Max canonical correlation")
    ax.set_ylim(0, max(1.0, max(vals) * 1.1))
    handles = [Patch(color=HIGHLIGHT, label="winner")]
    if true_label is not None:
        handles.append(Patch(facecolor="white", edgecolor=TRUE_MARK, linewidth=2, label="true"))
    ax.legend(handles=handles, frameon=False, fontsize=8)
    ax.set_title(title, fontsize=11)
    _style(ax)
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    *,
    normalize: bool = True,
    title: str = "Confusion matrix",
    ax: Axes | None = None,
) -> Figure:
    """Heatmap (single-hue sequential) with counts annotated; rows normalised if asked."""
    cm = np.asarray(cm, dtype=float)
    fig, ax = _get_ax(ax, (5.5, 4.5))
    sums = cm.sum(axis=1, keepdims=True)
    shown = cm / np.where(sums == 0, 1, sums) if normalize else cm
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    thresh = (1 if normalize else shown.max()) * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{int(cm[i, j])}"
            if normalize:
                txt += f"\n({shown[i, j]:.0%})"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if shown[i, j] > thresh else INK,
            )
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=11)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    fig.colorbar(im, ax=ax, label="row fraction" if normalize else "count", shrink=0.8)
    return fig


def plot_metric_curve(
    x: Sequence[float],
    y: Sequence[float] | None = None,
    *,
    xlabel: str,
    ylabel: str,
    chance_level: float | None = None,
    title: str = "",
    series: Mapping[str, Sequence[float]] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Line plot of a metric vs x (e.g. accuracy vs SNR); optional chance line and series."""
    if y is None and not series:
        raise ValueError("provide y or series")
    fig, ax = _get_ax(ax, (6, 4))
    lines = dict(series) if series else {"": y}
    for i, (name, ys) in enumerate(lines.items()):
        ax.plot(
            x,
            ys,
            color=PALETTE[i % len(PALETTE)],
            lw=2,
            marker="o",
            ms=5,
            markeredgecolor="white",
            markeredgewidth=1,
            label=name or None,
        )
    if chance_level is not None:
        ax.axhline(chance_level, color=MUTED, lw=1, ls="--")
        ax.text(
            0.99,
            chance_level,
            "chance",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8,
            color=MUTED,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    if len(lines) >= 2:
        ax.legend(frameon=False, fontsize=9)
    _style(ax)
    return fig


def plot_score_distributions(
    scores_by_true_label: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    title: str = "CCA score distributions",
    ax: Axes | None = None,
) -> Figure:
    """Box + strip plot: x groups = true label, one box per scoring class.

    ``scores_by_true_label[true][scored_class]`` is a list of scores for trials
    whose true class is ``true``. Strip jitter is deterministic (no RNG).
    """
    fig, ax = _get_ax(ax, (7, 4))
    groups = list(scores_by_true_label)
    classes: list[str] = []
    for g in groups:
        for c in scores_by_true_label[g]:
            if c not in classes:
                classes.append(c)
    k = max(len(classes), 1)
    width = 0.8 / k
    for ci, c in enumerate(classes):
        color = PALETTE[ci % len(PALETTE)]
        for gi, g in enumerate(groups):
            vals = np.asarray(scores_by_true_label[g].get(c, []), dtype=float)
            if vals.size == 0:
                continue
            pos = gi - 0.4 + width * (ci + 0.5)
            ax.boxplot(
                [vals],
                positions=[pos],
                widths=width * 0.8,
                patch_artist=True,
                showfliers=False,
                boxprops={"facecolor": color, "alpha": 0.35, "edgecolor": color, "linewidth": 1},
                medianprops={"color": INK, "linewidth": 1.5},
                whiskerprops={"color": color, "linewidth": 1},
                capprops={"color": color, "linewidth": 1},
            )
            jit = np.linspace(-0.3, 0.3, vals.size) * width
            ax.scatter(
                pos + jit, vals, s=10, color=color, edgecolor="white", linewidth=0.4, zorder=3
            )
    ax.set_xticks(range(len(groups)), [f"true: {g}" for g in groups])
    ax.set_ylabel("Max canonical correlation")
    ax.set_title(title, fontsize=11)
    ax.legend(
        handles=[
            Line2D(
                [0], [0], marker="s", ls="", color=PALETTE[i % len(PALETTE)], label=f"{c} reference"
            )
            for i, c in enumerate(classes)
        ],
        frameon=False,
        fontsize=8,
    )
    _style(ax)
    return fig
