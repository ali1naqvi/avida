from __future__ import annotations

import csv
import glob
import os
import sys
from dataclasses import dataclass
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORK_DIR = os.path.join(REPO_ROOT, "work")

TEST_RUNS = [
    "ecology_test_relative_cost_fast_run_*",
    "ecology_test_relative_cost_medium_run_*",
    "ecology_test_relative_cost_slow_run_*",
    "ecology_test_1_fast_run_*",
    "ecology_test_1_medium_run_*",
    "ecology_test_1_slow_run_*",
    "ecology_test_2_fast_run_*",
    "ecology_test_2_medium_run_*",
    "ecology_test_2_slow_run_*",
]

# Optional display names for x-axis labels. The keys should match the group names
# created from the run folders, such as ecology_test_1_fast.
X_AXIS_LABELS = {
    "ecology_test_relative_cost_fast": "Non-overlapping 200 RC",
    "ecology_test_relative_cost_medium": "Non-overlapping 500 RC",
    "ecology_test_relative_cost_slow": "Non-overlapping 1000 RC",
    "ecology_test_1_fast": "Non-overlapping 200",
    "ecology_test_1_medium": "Non-overlapping 500",
    "ecology_test_1_slow": "Non-overlapping 1000",
    "ecology_test_2_fast": "Overlapping 200",
    "ecology_test_2_medium": "Overlapping 500",
    "ecology_test_2_slow": "Overlapping 1000",
}

COUNT_DAT_NAME = os.path.join("data", "count.dat")
OUTPUT_DIR = os.path.join(REPO_ROOT, "scripts_ALI", "python_scripts", "outputs")
CSV_NAME = "test_update_summary.csv"
PLOT_NAME = "test_update_summary.pdf"


@dataclass(frozen=True)
class RunSummary:
    run_name: str
    group_name: str
    count_path: str
    final_update: int
    final_individuals: int


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


def resolve_run_dirs(entries: Iterable[str]) -> list[str]:
    paths: list[str] = []
    for entry in entries:
        entry_path = entry if os.path.isabs(entry) else os.path.join(WORK_DIR, entry)
        matches = glob.glob(entry_path)
        if not matches and os.path.isdir(entry_path):
            matches = [entry_path]
        paths.extend(os.path.abspath(path) for path in matches if os.path.isdir(path))

    unique_paths = sorted(set(paths), key=lambda path: natural_key(os.path.basename(path)))
    return unique_paths


def group_name_for_run(run_name: str) -> str:
    marker = "_run_"
    if marker in run_name:
        return run_name.split(marker, 1)[0]
    return run_name


def read_final_count_row(count_path: str) -> tuple[int, int]:
    final_values: list[str] | None = None
    with open(count_path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            final_values = line.split()

    if final_values is None:
        raise ValueError("no numeric rows found")
    if len(final_values) < 3:
        raise ValueError("expected at least 3 columns: update, insts, organisms")

    final_update = int(float(final_values[0]))
    final_individuals = int(float(final_values[2]))
    return final_update, final_individuals


def collect_summaries() -> list[RunSummary]:
    summaries: list[RunSummary] = []
    for run_dir in resolve_run_dirs(TEST_RUNS):
        count_path = os.path.join(run_dir, COUNT_DAT_NAME)
        run_name = os.path.basename(run_dir)
        if not os.path.exists(count_path):
            print(f"WARNING: skipping {run_name}; missing {count_path}")
            continue
        try:
            final_update, final_individuals = read_final_count_row(count_path)
        except ValueError as err:
            print(f"WARNING: skipping {run_name}; {err}")
            continue
        summaries.append(
            RunSummary(
                run_name=run_name,
                group_name=group_name_for_run(run_name),
                count_path=count_path,
                final_update=final_update,
                final_individuals=final_individuals,
            )
        )
    return summaries


def write_csv(summaries: list[RunSummary], csv_path: str) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "group", "final_update", "final_individuals", "count_dat"])
        for row in summaries:
            writer.writerow(
                [
                    row.run_name,
                    row.group_name,
                    row.final_update,
                    row.final_individuals,
                    row.count_path,
                ]
            )


def save_box_plot(summaries: list[RunSummary], plot_path: str) -> None:
    grouped: dict[str, list[RunSummary]] = {}
    for summary in summaries:
        grouped.setdefault(summary.group_name, []).append(summary)

    group_names = sorted(grouped, key=natural_key)
    update_values = [[item.final_update for item in grouped[name]] for name in group_names]
    x_axis_labels = [X_AXIS_LABELS.get(name, name) for name in group_names]

    width = max(10, min(28, len(group_names) * 0.75))
    fig, ax = plt.subplots(figsize=(width, 6.5))
    ax.boxplot(update_values, labels=x_axis_labels, medianprops={"linewidth": 2.5})

    ax.set_title("Box plots of final updates per test groups")
    ax.set_ylabel("Final update")
    ax.set_xlabel("Experimental setup")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=45)

    ymax = max(max(values) for values in update_values) if update_values else 1
    ax.set_ylim(0, ymax * 1.12)

    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)



def main() -> int:
    summaries = collect_summaries()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    #csv_path = os.path.join(OUTPUT_DIR, CSV_NAME)
    plot_path = os.path.join(OUTPUT_DIR, PLOT_NAME)

    #write_csv(summaries, csv_path)
    save_box_plot(summaries, plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
