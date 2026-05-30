from __future__ import annotations

import csv
import glob
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "avida_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORK_DIR = os.path.join(REPO_ROOT, "work")
OUTPUT_DIR = os.path.join(REPO_ROOT, "scripts_ALI", "python_scripts", "outputs")

OVERLAPPING_RUNS = [
    "ecology_test_2_fast_run_*",
    "ecology_test_2_medium_run_*",
    "ecology_test_2_slow_run_*",
]

SETUP_LABELS = {
    "ecology_test_2_fast": "Overlapping 200",
    "ecology_test_2_medium": "Overlapping 500",
    "ecology_test_2_slow": "Overlapping 1000",
}

CSV_NAME = "overlapping_average_generation_over_time.csv"
PDF_NAME = "overlapping_average_generation_over_time.pdf"
SPREAD_CSV_NAME = "overlapping_generation_spread_over_time.csv"
SPREAD_PDF_NAME = "overlapping_generation_spread_over_time.pdf"
ENV_CHANGE_RE = re.compile(r"^[ug]\s+(\d+):(\d+):end\s+SetGradientResourceLeastPopulated\b")
DETAIL_RE = re.compile(r"detail-(-?\d+)\.spop$")
FOCUS_UPDATE_MAX = 10000


@dataclass(frozen=True)
class RunSeries:
    run_name: str
    group_name: str
    env_change_start: int
    env_change_interval: int
    points: list[tuple[int, float, float]]


@dataclass(frozen=True)
class SummaryPoint:
    group_name: str
    update: int
    run_count: int
    median_generation: float
    std_generation: float


def natural_key(text: str) -> list[object]:
    parts: list[object] = []
    current = ""
    in_number = False
    for char in text:
        char_is_number = char.isdigit()
        if current and char_is_number != in_number:
            parts.append(int(current) if in_number else current.lower())
            current = ""
        current += char
        in_number = char_is_number
    if current:
        parts.append(int(current) if in_number else current.lower())
    return parts


def group_name_for_run(run_name: str) -> str:
    return run_name.split("_run_", 1)[0] if "_run_" in run_name else run_name


def resolve_run_dirs() -> list[str]:
    paths: list[str] = []
    for pattern in OVERLAPPING_RUNS:
        paths.extend(glob.glob(os.path.join(WORK_DIR, pattern)))
    return sorted(set(path for path in paths if os.path.isdir(path)), key=lambda path: natural_key(os.path.basename(path)))


def read_environment_schedule(events_path: str) -> tuple[int, int]:
    with open(events_path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("#"):
                continue
            match = ENV_CHANGE_RE.match(line)
            if match:
                return int(match.group(1)), int(match.group(2))
    return 1000, 0


def detail_files(run_dir: str) -> list[tuple[int, str]]:
    details: list[tuple[int, str]] = []
    for path in glob.glob(os.path.join(run_dir, "data", "detail-*.spop")):
        match = DETAIL_RE.search(os.path.basename(path))
        if match:
            update = int(match.group(1))
            if update >= 0:
                details.append((update, path))
    return sorted(details)


def weighted_median(values: list[tuple[float, int]]) -> float:
    total_weight = sum(weight for _value, weight in values)
    if total_weight <= 0:
        return 0.0
    threshold = total_weight / 2
    cumulative = 0
    for value, weight in sorted(values, key=lambda item: item[0]):
        cumulative += weight
        if cumulative >= threshold:
            return value
    return values[-1][0]


def weighted_std(values: list[tuple[float, int]]) -> float:
    total_weight = sum(weight for _value, weight in values)
    if total_weight <= 0:
        return 0.0
    mean = sum(value * weight for value, weight in values) / total_weight
    variance = sum(weight * (value - mean) ** 2 for value, weight in values) / total_weight
    return variance**0.5


def read_generation_snapshot(detail_path: str) -> tuple[float, float] | None:
    values: list[tuple[float, int]] = []
    with open(detail_path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 17:
                continue
            living_count = int(float(fields[4]))
            if living_count <= 0:
                continue
            generation_born = float(fields[10])
            values.append((generation_born, living_count))
    if not values:
        return None
    return weighted_median(values), weighted_std(values)


def collect_runs() -> list[RunSeries]:
    series: list[RunSeries] = []
    for run_dir in resolve_run_dirs():
        run_name = os.path.basename(run_dir)
        events_path = os.path.join(run_dir, "events.cfg")
        if not os.path.exists(events_path):
            print(f"WARNING: skipping {run_name}; missing events.cfg")
            continue
        start, interval = read_environment_schedule(events_path)
        points: list[tuple[int, float, float]] = []
        for update, detail_path in detail_files(run_dir):
            if update > FOCUS_UPDATE_MAX:
                continue
            snapshot = read_generation_snapshot(detail_path)
            if snapshot is None:
                continue
            median_generation, std_generation = snapshot
            points.append((update, median_generation, std_generation))
        if not points:
            print(f"WARNING: skipping {run_name}; no population generation snapshots found")
            continue
        series.append(
            RunSeries(
                run_name=run_name,
                group_name=group_name_for_run(run_name),
                env_change_start=start,
                env_change_interval=interval,
                points=points,
            )
        )
    return series


def summarize_runs(series: list[RunSeries]) -> list[SummaryPoint]:
    values_by_group_update: dict[tuple[str, int], list[float]] = defaultdict(list)
    std_by_group_update: dict[tuple[str, int], list[float]] = defaultdict(list)
    for run in series:
        for update, generation, generation_std in run.points:
            values_by_group_update[(run.group_name, update)].append(generation)
            std_by_group_update[(run.group_name, update)].append(generation_std)

    summary: list[SummaryPoint] = []
    for (group_name, update), values in values_by_group_update.items():
        median = sorted(values)[len(values) // 2]
        if len(values) % 2 == 0:
            sorted_values = sorted(values)
            midpoint = len(sorted_values) // 2
            median = (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2
        std = sum(std_by_group_update[(group_name, update)]) / len(std_by_group_update[(group_name, update)])
        summary.append(
            SummaryPoint(
                group_name=group_name,
                update=update,
                run_count=len(values),
                median_generation=median,
                std_generation=std,
            )
        )
    return sorted(summary, key=lambda row: (natural_key(row.group_name), row.update))


def write_csv(summary: list[SummaryPoint], csv_path: str) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "update", "run_count", "median_generation", "std_generation"])
        for row in summary:
            writer.writerow(
                [
                    row.group_name,
                    row.update,
                    row.run_count,
                    row.median_generation,
                    row.std_generation,
                ]
            )


def write_spread_csv(summary: list[SummaryPoint], csv_path: str) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "update", "run_count", "std_generation"])
        for row in summary:
            writer.writerow([row.group_name, row.update, row.run_count, row.std_generation])


def add_environment_markers(ax, start: int, interval: int, max_update: int) -> None:
    if interval <= 0 or max_update < start:
        return
    update = start
    first = True
    while update <= max_update:
        ax.axvline(
            update,
            color="#c44e52",
            linewidth=0.75 if not first else 1.4,
            alpha=0.22 if not first else 0.75,
            zorder=0,
            label="Environment change" if first else None,
        )
        first = False
        update += interval


def save_plot(series: list[RunSeries], summary: list[SummaryPoint], pdf_path: str) -> None:
    series_by_group: dict[str, list[RunSeries]] = defaultdict(list)
    summary_by_group: dict[str, list[SummaryPoint]] = defaultdict(list)
    for run in series:
        series_by_group[run.group_name].append(run)
    for row in summary:
        summary_by_group[row.group_name].append(row)

    group_names = sorted(summary_by_group, key=natural_key)
    fig, axes = plt.subplots(len(group_names), 1, figsize=(13, 3.1 * len(group_names)), sharex=False)
    if len(group_names) == 1:
        axes = [axes]

    for ax, group_name in zip(axes, group_names):
        rows = [
            row
            for row in sorted(summary_by_group[group_name], key=lambda row: row.update)
            if row.update <= FOCUS_UPDATE_MAX
        ]
        if not rows:
            continue
        updates = [row.update for row in rows]
        medians = [row.median_generation for row in rows]
        stds = [row.std_generation for row in rows]
        low = [max(0.0, median - std) for median, std in zip(medians, stds)]
        high = [median + std for median, std in zip(medians, stds)]

        runs = series_by_group[group_name]
        start = runs[0].env_change_start
        interval = runs[0].env_change_interval
        add_environment_markers(ax, start, interval, FOCUS_UPDATE_MAX)

        ax.fill_between(updates, low, high, color="#2f6f95", alpha=0.14, linewidth=0, label="Median +/- SD")
        ax.plot(updates, medians, color="#2f6f95", linewidth=1.9, marker="o", markersize=3, label="Median generation")
        ax.set_title(
            f"{SETUP_LABELS.get(group_name, group_name)} - environment changes: {start}, then every {interval} updates",
            loc="left",
        )
        ax.set_ylabel("Generation")
        ax.set_xlim(0, FOCUS_UPDATE_MAX)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.ticklabel_format(axis="x", style="plain")

    axes[-1].set_xlabel("Update")
    axes[0].legend(loc="upper left", frameon=False, ncol=3)
    fig.suptitle("Overlapping tests: living-population median generation during first 10,000 updates", y=0.995)
    fig.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)


def save_spread_plot(series: list[RunSeries], summary: list[SummaryPoint], pdf_path: str) -> None:
    series_by_group: dict[str, list[RunSeries]] = defaultdict(list)
    summary_by_group: dict[str, list[SummaryPoint]] = defaultdict(list)
    for run in series:
        series_by_group[run.group_name].append(run)
    for row in summary:
        summary_by_group[row.group_name].append(row)

    group_names = sorted(summary_by_group, key=natural_key)
    fig, axes = plt.subplots(len(group_names), 1, figsize=(13, 3.1 * len(group_names)), sharex=False)
    if len(group_names) == 1:
        axes = [axes]

    for ax, group_name in zip(axes, group_names):
        rows = [
            row
            for row in sorted(summary_by_group[group_name], key=lambda row: row.update)
            if row.update <= FOCUS_UPDATE_MAX
        ]
        if not rows:
            continue

        updates = [row.update for row in rows]
        stds = [row.std_generation for row in rows]
        runs = series_by_group[group_name]
        start = runs[0].env_change_start
        interval = runs[0].env_change_interval
        add_environment_markers(ax, start, interval, FOCUS_UPDATE_MAX)

        ax.plot(
            updates,
            stds,
            color="#2f6f95",
            linewidth=1.9,
            marker="o",
            markersize=3,
            label="Generation SD",
        )
        ax.fill_between(updates, [0] * len(updates), stds, color="#2f6f95", alpha=0.12, linewidth=0)
        ax.set_title(
            f"{SETUP_LABELS.get(group_name, group_name)} - environment changes: {start}, then every {interval} updates",
            loc="left",
        )
        ax.set_ylabel("Generation SD")
        ax.set_xlim(0, FOCUS_UPDATE_MAX)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.ticklabel_format(axis="x", style="plain")

    axes[-1].set_xlabel("Update")
    axes[0].legend(loc="upper left", frameon=False, ncol=2)
    fig.suptitle("Overlapping tests: spread of living-generation structure during first 10,000 updates", y=0.995)
    fig.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)


def main() -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    series = collect_runs()
    if not series:
        raise SystemExit("No overlapping run data found.")
    summary = summarize_runs(series)

    csv_path = os.path.join(OUTPUT_DIR, CSV_NAME)
    pdf_path = os.path.join(OUTPUT_DIR, PDF_NAME)
    spread_csv_path = os.path.join(OUTPUT_DIR, SPREAD_CSV_NAME)
    spread_pdf_path = os.path.join(OUTPUT_DIR, SPREAD_PDF_NAME)
    write_csv(summary, csv_path)
    write_spread_csv(summary, spread_csv_path)
    save_plot(series, summary, pdf_path)
    save_spread_plot(series, summary, spread_pdf_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {spread_csv_path}")
    print(f"Wrote {spread_pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
