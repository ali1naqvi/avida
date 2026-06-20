from __future__ import annotations

import csv
import glob
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "avida_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORK_DIR = os.path.join(REPO_ROOT, "work")
OUTPUT_DIR = os.path.join(REPO_ROOT, "scripts_ALI", "python_scripts", "outputs")

TEST_RUNS = [
    "ecology_test_over_flash_run_*",
    "ecology_test_over_fast_run_*",
    "ecology_test_over_medium_run*",
    "ecology_test_over_slow_run*"
]

# Optional display names for x-axis labels. The keys should match the group names
# created from the run folders, such as ecology_test_1_fast.
X_AXIS_LABELS = {
    "ecology_test_over_flash": "Overlapping 20",
    "ecology_test_over_fast": "Overlapping 50",
    "ecology_test_over_medium": "Overlapping 200",
    "ecology_test_2_slow": "Overlapping 500",
}

CSV_NAME = "population_instruction_usage_largest_updates.csv"
PDF_NAME = "population_instruction_usage_largest_updates.pdf"
DETAIL_RE = re.compile(r"detail-(-?\d+)\.spop$")


@dataclass(frozen=True)
class RunUsage:
    run_name: str
    group_name: str
    detail_path: str
    update: int
    population: int
    genome_count: int
    total_weighted_instructions: int
    instruction_counts: dict[str, int]


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
        paths.extend(os.path.abspath(path) for path in glob.glob(entry_path) if os.path.isdir(path))
    return sorted(set(paths), key=lambda path: natural_key(os.path.basename(path)))


def group_name_for_run(run_name: str) -> str:
    return run_name.split("_run_", 1)[0] if "_run_" in run_name else run_name


def largest_detail_file(run_dir: str) -> tuple[int, str] | None:
    candidates: list[tuple[int, str]] = []
    for path in glob.glob(os.path.join(run_dir, "data", "detail-*.spop")):
        match = DETAIL_RE.search(os.path.basename(path))
        if match:
            update = int(match.group(1))
            if update >= 0:
                candidates.append((update, path))
    return max(candidates, key=lambda item: item[0]) if candidates else None


def read_instruction_names(instset_path: str) -> dict[str, str]:
    names: list[str] = []
    if os.path.exists(instset_path):
        with open(instset_path, encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                stripped = raw.strip()
                if not stripped.startswith("INST "):
                    continue
                parts = stripped.split()
                if len(parts) >= 2:
                    names.append(parts[1])

    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return {letter: name for letter, name in zip(alphabet, names)}


def read_detail_usage(detail_path: str) -> tuple[int, int, Counter[str]]:
    population = 0
    genome_count = 0
    counts: Counter[str] = Counter()
    with open(detail_path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 17:
                continue
            living_count = int(float(fields[4]))
            sequence = fields[16]
            population += living_count
            genome_count += 1
            for symbol in sequence:
                counts[symbol] += living_count
    return population, genome_count, counts


def collect_usage() -> tuple[list[RunUsage], dict[str, str]]:
    usages: list[RunUsage] = []
    all_instruction_names: dict[str, str] = {}
    for run_dir in resolve_run_dirs(TEST_RUNS):
        run_name = os.path.basename(run_dir)
        detail = largest_detail_file(run_dir)
        if detail is None:
            print(f"WARNING: skipping {run_name}; no non-negative detail snapshots found")
            continue

        update, detail_path = detail
        instruction_names = read_instruction_names(os.path.join(run_dir, "instset.cfg"))
        all_instruction_names.update(instruction_names)
        population, genome_count, counts = read_detail_usage(detail_path)
        usages.append(
            RunUsage(
                run_name=run_name,
                group_name=group_name_for_run(run_name),
                detail_path=detail_path,
                update=update,
                population=population,
                genome_count=genome_count,
                total_weighted_instructions=sum(counts.values()),
                instruction_counts=dict(counts),
            )
        )
    return usages, all_instruction_names


def write_csv(usages: list[RunUsage], instruction_names: dict[str, str], csv_path: str) -> None:
    symbols = sorted(
        {symbol for usage in usages for symbol in usage.instruction_counts},
        key=lambda symbol: natural_key(symbol),
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run",
                "group",
                "largest_update",
                "population",
                "genotypes",
                "total_weighted_instructions",
                "instruction_symbol",
                "instruction_name",
                "weighted_count",
                "proportion",
                "detail_spop",
            ]
        )
        for usage in usages:
            for symbol in symbols:
                weighted_count = usage.instruction_counts.get(symbol, 0)
                proportion = (
                    weighted_count / usage.total_weighted_instructions
                    if usage.total_weighted_instructions
                    else 0
                )
                writer.writerow(
                    [
                        usage.run_name,
                        usage.group_name,
                        usage.update,
                        usage.population,
                        usage.genome_count,
                        usage.total_weighted_instructions,
                        symbol,
                        instruction_names.get(symbol, f"unknown-{symbol}"),
                        weighted_count,
                        proportion,
                        usage.detail_path,
                    ]
                )


def save_pdf(usages: list[RunUsage], instruction_names: dict[str, str], pdf_path: str) -> None:
    grouped: dict[str, list[RunUsage]] = {}
    for usage in usages:
        grouped.setdefault(usage.group_name, []).append(usage)

    symbols = sorted(
        {symbol for usage in usages for symbol in usage.instruction_counts},
        key=lambda symbol: natural_key(symbol),
    )
    labels = [f"{symbol}\n{instruction_names.get(symbol, 'unknown')}" for symbol in symbols]
    group_names = sorted(grouped, key=natural_key)

    with PdfPages(pdf_path) as pdf:
        for group_name in group_names:
            runs = grouped[group_name]
            values = [
                [
                    (
                        run.instruction_counts.get(symbol, 0) / run.total_weighted_instructions
                        if run.total_weighted_instructions
                        else 0
                    )
                    for run in runs
                ]
                for symbol in symbols
            ]
            width = max(11, min(28, len(symbols) * 0.55))
            fig, ax = plt.subplots(figsize=(width, 7.5))
            ax.boxplot(
                values,
                tick_labels=labels,
                patch_artist=True,
                medianprops={"color": "#2f5f7f", "linewidth": 2.0},
                boxprops={"facecolor": "#2f6f95", "edgecolor": "#9a9a9a", "alpha": 0.95},
                whiskerprops={"color": "#9a9a9a", "linewidth": 1.25},
                capprops={"color": "#9a9a9a", "linewidth": 1.25},
                flierprops={
                    "marker": "o",
                    "markerfacecolor": "white",
                    "markeredgecolor": "#8a8a8a",
                    "markersize": 4,
                    "alpha": 0.9,
                },
            )
            ax.set_title(f"{X_AXIS_LABELS.get(group_name, group_name)} instruction usage")
            ax.set_ylabel("Proportion of living-population instructions")
            ax.set_xlabel("Instruction")
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            ax.set_ylim(bottom=0)
            ax.tick_params(axis="x", rotation=90, labelsize=8)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        total_runs = len(usages)
        lines = [
            "Population Instruction Usage Report",
            "",
            f"Runs included: {total_runs}",
            "Measurement: each genotype sequence in the largest detail-*.spop snapshot per run,",
            "weighted by the number of living organisms for that genotype, then normalized",
            "by total weighted instructions in that snapshot.",
            "",
            "Setups:",
        ]
        for group_name in group_names:
            runs = grouped[group_name]
            lines.append(
                f"- {X_AXIS_LABELS.get(group_name, group_name)}: {len(runs)} runs, "
                f"largest updates {min(run.update for run in runs):,}-{max(run.update for run in runs):,}"
            )
        ax.text(0.05, 0.95, "\n".join(lines), ha="left", va="top", fontsize=12, family="monospace")
        pdf.savefig(fig)
        plt.close(fig)


def main() -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    usages, instruction_names = collect_usage()
    if not usages:
        raise SystemExit("No usage data found.")

    csv_path = os.path.join(OUTPUT_DIR, CSV_NAME)
    pdf_path = os.path.join(OUTPUT_DIR, PDF_NAME)
    write_csv(usages, instruction_names, csv_path)
    save_pdf(usages, instruction_names, pdf_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
