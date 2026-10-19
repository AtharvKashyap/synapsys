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
    """Box + strip plot: x groups = trial group (e.g. true label), one box per scoring class.

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
    ax.set_xticks(range(len(groups)), [g.replace(" ", "\n", 1) for g in groups], fontsize=9)
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


def plot_gating_scatter(
    points_by_kind: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    *,
    title: str = "Winner score vs margin",
    ax: Axes | None = None,
) -> Figure:
    """Scatter of (winner score, winner - runner-up margin) per trial kind.

    Used to inspect rejection mechanics: genuine target trials should sit
    up-and-right of noise / off-target / ambiguous trials. Draws no thresholds —
    thresholds are calibrated on own-system data, never chosen from this plot.
    """
    fig, ax = _get_ax(ax, (6, 4.5))
    for i, (kind, (conf, margin)) in enumerate(points_by_kind.items()):
        ax.scatter(
            conf,
            margin,
            s=14,
            alpha=0.6,
            color=PALETTE[i % len(PALETTE)],
            edgecolors="none",
            label=kind,
        )
    ax.set_xlabel("Winner CCA score (confidence)")
    ax.set_ylabel("Margin (winner − runner-up)")
    ax.set_title(title, fontsize=11)
    if len(points_by_kind) >= 2:
        ax.legend(frameon=False, fontsize=9)
    _style(ax, grid_axis="both")
    return fig


def plot_filter_response(
    freqs_hz: Sequence[float],
    response_db: Mapping[str, Sequence[float]],
    *,
    keep_frequencies: Mapping[str, float] | None = None,
    reject_frequencies: Mapping[str, float] | None = None,
    ylim_db: tuple[float, float] = (-80.0, 5.0),
    log_freq: bool = False,
    title: str = "Filter magnitude response",
    ax: Axes | None = None,
) -> Figure:
    """Magnitude response (dB) of one or more filter designs.

    ``keep_frequencies`` (e.g. SSVEP targets) are marked with dashed lines and
    ``reject_frequencies`` (drift, mains, out-of-band) with dotted lines.
    """
    fig, ax = _get_ax(ax, (7, 4))
    for i, (name, db) in enumerate(response_db.items()):
        ax.plot(
            freqs_hz,
            np.clip(db, ylim_db[0] - 10, None),
            color=PALETTE[i % len(PALETTE)],
            lw=2,
            label=name,
        )
    for marks, ls in ((keep_frequencies or {}, "--"), (reject_frequencies or {}, ":")):
        for name, f in marks.items():
            ax.axvline(f, color=MUTED, lw=1, ls=ls)
            ax.text(
                f,
                ylim_db[1],
                f" {name}",
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
                color=MUTED,
            )
    ax.set_ylim(*ylim_db)
    if log_freq:
        ax.set_xscale("log")
        ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain (dB)")
    ax.set_title(title, fontsize=11)
    if len(response_db) >= 2:
        ax.legend(frameon=False, fontsize=9, loc="lower left")
    _style(ax, grid_axis="both")
    return fig


def plot_table(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    title: str = "",
    highlight_rows: Sequence[int] = (),
    col_widths: Sequence[float] | None = None,
) -> Figure:
    """Render a small text table as a figure (for reports that need an image).

    Column widths default to the longest cell text per column, and the figure is
    sized to fit the table so nothing is clipped and no blank space is left.
    """
    cells = [list(header), *[list(r) for r in rows]]
    chars = [max(len(str(r[c])) for r in cells) + 2 for c in range(len(header))]
    if col_widths is None:
        col_widths = [n / sum(chars) for n in chars]
    width_in = max(4.0, 0.085 * sum(chars))
    height_in = 0.3 * (len(rows) + 1) + (0.45 if title else 0.1)
    fig = Figure(figsize=(width_in, height_in))
    title_frac = 0.4 / height_in if title else 0.02
    ax = fig.add_axes((0.005, 0.005, 0.99, 0.99 - title_frac))
    ax.axis("off")
    table = ax.table(
        cellText=cells[1:],
        colLabels=cells[0],
        cellLoc="center",
        colWidths=list(col_widths),
        bbox=(0, 0, 1, 1),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r, _c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(weight="bold", color=INK)
            cell.set_facecolor("#f2f2f2")
        elif (r - 1) in highlight_rows:
            cell.set_facecolor("#fde8d8")
    if title:
        fig.text(0.5, 1 - 0.2 / height_in, title, ha="center", va="center", fontsize=11, color=INK)
    return fig


def plot_timeseries(
    eeg: np.ndarray,
    fs: float,
    channel_names: Sequence[str],
    *,
    markers: Mapping[str, float] | None = None,
    shade: tuple[float, float] | None = None,
    shade_label: str = "analysis window",
    title: str = "EEG",
    ax: Axes | None = None,
) -> Figure:
    """Stacked per-channel traces (µV) with optional time markers and a shaded window."""
    fig, ax = _get_ax(ax, (9, 0.8 + 0.75 * len(channel_names)))
    t = np.arange(eeg.shape[1]) / fs
    uv = eeg * 1e6
    spacing = 4 * float(np.median(np.std(uv, axis=1))) or 1.0
    for i, name in enumerate(channel_names):
        off = -i * spacing
        ax.plot(t, uv[i] - np.mean(uv[i]) + off, color=PALETTE[i % len(PALETTE)], lw=0.8)
        ax.text(
            -0.01,
            off,
            name,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=9,
            color=INK,
        )
    if shade is not None:
        ax.axvspan(*shade, color=PALETTE[1], alpha=0.12, lw=0)
        ax.text(
            (shade[0] + shade[1]) / 2,
            1.0,
            shade_label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color=MUTED,
        )
    for name, x in (markers or {}).items():
        ax.axvline(x, color=MUTED, lw=1, ls="--")
        ax.text(
            x,
            1.0,
            f" {name}",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=8,
            color=MUTED,
        )
    ax.set_yticks([])
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time (s)")
    ax.set_title(title, fontsize=11, pad=14)
    ax.text(
        1.0,
        -0.12,
        f"trace spacing {spacing:.0f} µV",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color=MUTED,
    )
    _style(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    return fig


def plot_spectra(
    freqs_hz: np.ndarray,
    spectra: Mapping[str, np.ndarray],
    *,
    mark_frequencies: Mapping[str, float] | None = None,
    harmonics: int = 2,
    fmax: float = 40.0,
    ylabel: str = "PSD (V²/Hz)",
    title: str = "Spectra",
    ax: Axes | None = None,
) -> Figure:
    """One line per named spectrum (e.g. trial-averaged PSD per presented frequency)."""
    fig, ax = _get_ax(ax, (8, 4))
    sel = freqs_hz <= fmax
    for i, (name, p) in enumerate(spectra.items()):
        ax.semilogy(
            freqs_hz[sel], np.asarray(p)[sel], color=PALETTE[i % len(PALETTE)], lw=1.6, label=name
        )
    for i, (_name, f) in enumerate((mark_frequencies or {}).items()):
        for h in range(1, harmonics + 1):
            if f * h <= fmax:
                ax.axvline(
                    f * h,
                    color=PALETTE[i % len(PALETTE)],
                    lw=0.9,
                    ls="--" if h == 1 else ":",
                    alpha=0.7,
                )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    if len(spectra) >= 2:
        ax.legend(frameon=False, fontsize=9)
    _style(ax, grid_axis="both")
    return fig
