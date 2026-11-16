"""SynapSys senior design project timeline (Gantt chart).

Edit TASKS and MILESTONES below and re-run to regenerate the chart:

    python planning/gantt_chart.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator

OUTPUT_PATH = Path(__file__).parent / "gantt_chart.png"

# Owner -> (fill color, text color). Text color distinguishes "primary
# commitment" bars (bold black label) from lighter/flexible bars (gray
# label), matching fill alpha below.
OWNER_COLORS = {
    "Theodora": "#3b5b92",
    "Felipe": "#c8801f",
    "Atharv": "#6a9a3a",
    "All three": "#6a4c8c",
    "Fab + shipping": "#a6a6a6",
    "Slack": "#e6e6e6",
}

# Each task: (section, label, start, end, owner, alpha, critical_path)
# alpha < 1.0 renders as a lighter/flexible-looking bar with a gray label;
# alpha == 1.0 renders solid with a bold black label.
Task = tuple[str, str, date, date, str, float, bool]

TASKS: list[Task] = [
    # PCB
    (
        "PCB",
        "Analog design: schematic + layout",
        date(2026, 9, 21),
        date(2026, 10, 2),
        "Theodora",
        1.0,
        False,
    ),
    ("PCB", "Power / digital design", date(2026, 9, 24), date(2026, 10, 2), "Felipe", 1.0, False),
    (
        "PCB",
        "BOM, sourcing, DRC review",
        date(2026, 9, 21),
        date(2026, 10, 19),
        "Atharv",
        0.5,
        False,
    ),
    (
        "PCB",
        "v1 fab + shipping",
        date(2026, 10, 5),
        date(2026, 10, 23),
        "Fab + shipping",
        1.0,
        False,
    ),
    (
        "PCB",
        "v1 assembly + bring-up",
        date(2026, 10, 19),
        date(2026, 11, 2),
        "Theodora",
        1.0,
        False,
    ),
    ("PCB", "Slack: v1 debug", date(2026, 10, 28), date(2026, 11, 2), "Slack", 1.0, False),
    ("PCB", "v2 changes + order", date(2026, 11, 2), date(2026, 11, 6), "Felipe", 1.0, False),
    (
        "PCB",
        "v2 fab + shipping",
        date(2026, 11, 6),
        date(2026, 11, 20),
        "Fab + shipping",
        1.0,
        False,
    ),
    ("PCB", "v2 bring-up", date(2026, 11, 16), date(2026, 11, 23), "Theodora", 1.0, False),
    # Firmware and software
    (
        "Firmware and software",
        "SPI driver, streaming, latency",
        date(2026, 9, 21),
        date(2026, 11, 13),
        "Felipe",
        0.5,
        False,
    ),
    (
        "Firmware and software",
        "ADS1299 config, filters",
        date(2026, 9, 21),
        date(2026, 10, 26),
        "Theodora",
        0.5,
        False,
    ),
    (
        "Firmware and software",
        "CCA decoder, confidence gate",
        date(2026, 9, 21),
        date(2026, 10, 19),
        "Atharv",
        0.5,
        False,
    ),
    (
        "Firmware and software",
        "Stimulus renderer, phone app",
        date(2026, 10, 19),
        date(2026, 11, 13),
        "Atharv",
        0.5,
        False,
    ),
    (
        "Firmware and software",
        "BLE, on-device filtering",
        date(2026, 10, 26),
        date(2026, 11, 13),
        "Felipe",
        0.5,
        False,
    ),
    (
        "Firmware and software",
        "3-command classifier, tuning",
        date(2026, 11, 13),
        date(2026, 11, 23),
        "Atharv",
        0.5,
        False,
    ),
    # Integration and test
    (
        "Integration and test",
        "Alpha block validation",
        date(2026, 10, 23),
        date(2026, 10, 30),
        "Theodora",
        1.0,
        True,
    ),
    (
        "Integration and test",
        "First SSVEP peak",
        date(2026, 10, 23),
        date(2026, 10, 30),
        "Atharv",
        1.0,
        True,
    ),
    (
        "Integration and test",
        "End-to-end 2-command demo",
        date(2026, 10, 30),
        date(2026, 11, 13),
        "Felipe",
        0.5,
        False,
    ),
    (
        "Integration and test",
        "Slack: v2 integration",
        date(2026, 11, 20),
        date(2026, 11, 27),
        "Slack",
        1.0,
        False,
    ),
    (
        "Integration and test",
        "Full test matrix",
        date(2026, 11, 23),
        date(2026, 12, 4),
        "All three",
        1.0,
        True,
    ),
    # Deliverables
    (
        "Deliverables",
        "Design pres., document, ethics",
        date(2026, 9, 21),
        date(2026, 11, 13),
        "All three",
        1.0,
        False,
    ),
    (
        "Deliverables",
        "Interim demo",
        date(2026, 11, 13),
        date(2026, 11, 16),
        "All three",
        1.0,
        False,
    ),
    (
        "Deliverables",
        "Final presentation",
        date(2026, 11, 27),
        date(2026, 12, 2),
        "All three",
        0.5,
        False,
    ),
    ("Deliverables", "Slack: buffer", date(2026, 12, 2), date(2026, 12, 9), "Slack", 1.0, False),
    (
        "Deliverables",
        "Poster, video, report",
        date(2026, 12, 9),
        date(2026, 12, 21),
        "All three",
        1.0,
        False,
    ),
]

# (date, label) vertical milestone markers, drawn full-height with a label
# above the plot.
MILESTONES: list[tuple[date, str]] = [
    (date(2026, 10, 2), "Schematic freeze"),
    (date(2026, 10, 5), "v1 ordered"),
    (date(2026, 10, 30), "Pivot gate"),
    (date(2026, 11, 13), "Interim demo"),
    (date(2026, 11, 27), "Requirements validated"),
]

X_TICKS = [
    date(2026, 9, 21),
    date(2026, 10, 5),
    date(2026, 10, 19),
    date(2026, 11, 2),
    date(2026, 11, 16),
    date(2026, 11, 30),
    date(2026, 12, 14),
]


def build_rows(tasks: list[Task]) -> list[tuple[str | None, Task | None]]:
    """Interleave section-header rows in front of each section's first task."""
    rows: list[tuple[str | None, Task | None]] = []
    seen_sections: set[str] = set()
    for task in tasks:
        section = task[0]
        if section not in seen_sections:
            rows.append((section, None))
            seen_sections.add(section)
        rows.append((None, task))
    return rows


def plot_gantt(tasks: list[Task], milestones: list[tuple[date, str]]) -> None:
    rows = build_rows(tasks)
    n_rows = len(rows)

    fig, ax = plt.subplots(figsize=(14, 0.32 * n_rows + 1.2))

    y_labels = []
    for row_index, (section, task) in enumerate(rows):
        y = n_rows - row_index  # top-to-bottom order
        if section is not None:
            y_labels.append((y, section, "bold", "black"))
            continue

        _, label, start, end, owner, alpha, critical = task
        color = OWNER_COLORS[owner]
        edgecolor = "black" if critical else "none"
        linewidth = 1.6 if critical else 0
        ax.barh(
            y,
            width=(end - start).days,
            left=mdates.date2num(start),
            height=0.6,
            color=color,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=3,
        )
        weight = "normal"
        text_color = "#888888" if alpha < 1.0 else "black"
        y_labels.append((y, label, weight, text_color))

    for y, label, weight, color in y_labels:
        ax.text(
            mdates.date2num(TASKS[0][2]) - 1,
            y,
            label,
            ha="right",
            va="center",
            fontsize=9,
            fontweight=weight,
            color=color,
        )

    ax.set_ylim(0.3, n_rows + 0.7)
    ax.set_yticks([])

    ax.xaxis.set_major_locator(FixedLocator(mdates.date2num(X_TICKS)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    ax.set_xlim(
        mdates.date2num(X_TICKS[0]) - 3,
        mdates.date2num(date(2026, 12, 21)) + 2,
    )

    for index, (milestone_date, milestone_label) in enumerate(milestones):
        x = mdates.date2num(milestone_date)
        ax.axvline(x, color="#666666", linestyle="--", linewidth=0.8, zorder=1)
        # Stagger label height so closely spaced milestones don't overlap.
        label_y = n_rows + 1.3 + 0.8 * (index % 2)
        ax.annotate(
            milestone_label,
            xy=(x, n_rows + 0.7),
            xytext=(x, label_y),
            ha="center",
            va="bottom",
            fontsize=9,
            annotation_clip=False,
        )

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6, zorder=0)

    legend_handles = [Patch(facecolor=c, label=owner) for owner, c in OWNER_COLORS.items()]
    legend_handles.append(
        Patch(facecolor="white", edgecolor="black", linewidth=1.6, label="Critical path")
    )
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    plot_gantt(TASKS, MILESTONES)
