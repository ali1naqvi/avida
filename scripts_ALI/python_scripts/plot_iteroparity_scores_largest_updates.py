from __future__ import annotations

import csv
import glob
import math
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "avida_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORK_DIR = os.path.join(REPO_ROOT, "work")
OUTPUT_DIR = os.path.join(REPO_ROOT, "scripts_ALI", "python_scripts", "outputs")

TEST_RUNS = [
    "ecology_test_1_fast_run_*",
    "ecology_test_1_medium_run_*",
    "ecology_test_1_slow_run_*",
    "ecology_test_2_fast_run_*",
    "ecology_test_2_medium_run_*",
    "ecology_test_2_slow_run_*",
]

X_AXIS_LABELS = {
    "ecology_test_1_fast": "Non-overlap 200",
    "ecology_test_1_medium": "Non-overlap 500",
    "ecology_test_1_slow": "Non-overlap 1000",
    "ecology_test_2_fast": "Overlap 200",
    "ecology_test_2_medium": "Overlap 500",
    "ecology_test_2_slow": "Overlap 1000",
}

CSV_NAME = "iteroparity_scores_largest_updates.csv"
PDF_NAME = "iteroparity_scores_largest_updates.pdf"
DETAIL_RE = re.compile(r"detail-(-?\d+)\.spop$")


@dataclass(frozen=True)
class SnapshotMedian:
    group_name: str
    update: int
    population: int
    genotype_count: int
    snapshot_count: int
    iteroparity_median: float
    iteroparity_std: float
    semelparity_median: float
    semelparity_std: float


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


def instruction_operand_to_symbol(operand: int) -> str:
    if operand == 255:
        return "_"
    offset = operand % 62
    prefix = operand // 62
    prefix_symbol = {1: "+", 2: "-", 3: "~", 4: "?"}.get(prefix, "")
    if offset < 26:
        symbol = chr(ord("a") + offset)
    elif offset < 52:
        symbol = chr(ord("A") + offset - 26)
    else:
        symbol = chr(ord("0") + offset - 52)
    return f"{prefix_symbol}{symbol}"


def read_instruction_symbols(instset_path: str) -> dict[str, str]:
    names: list[str] = []
    with open(instset_path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if line.startswith("INST "):
                parts = line.split()
                if len(parts) >= 2:
                    names.append(parts[1])
    return {name: instruction_operand_to_symbol(index) for index, name in enumerate(names)}


def tokenize_sequence(sequence: str) -> list[str]:
    tokens: list[str] = []
    prefixes = {"+", "-", "~", "?"}
    index = 0
    while index < len(sequence):
        char = sequence[index]
        if char == "_":
            index += 1
            continue
        if char in prefixes and index + 1 < len(sequence):
            tokens.append(sequence[index : index + 2])
            index += 2
            continue
        tokens.append(char)
        index += 1
    return tokens


def detail_files(run_dir: str) -> list[tuple[int, str]]:
    candidates: list[tuple[int, str]] = []
    for path in glob.glob(os.path.join(run_dir, "data", "detail-*.spop")):
        match = DETAIL_RE.search(os.path.basename(path))
        if match:
            update = int(match.group(1))
            if update >= 0:
                candidates.append((update, path))
    return sorted(candidates)


def normalized_repro_spacing(repro_positions: list[int], genome_length: int) -> float:
    if genome_length <= 0 or len(repro_positions) < 2:
        return 0.0
    positions = sorted(repro_positions)
    gaps = [
        positions[index + 1] - positions[index]
        for index in range(len(positions) - 1)
    ]
    gaps.append(genome_length - positions[-1] + positions[0])
    mean_gap = sum(gaps) / len(gaps)
    if mean_gap <= 0:
        return 0.0
    variance = sum((gap - mean_gap) ** 2 for gap in gaps) / len(gaps)
    cv = math.sqrt(variance) / mean_gap
    return max(0.0, min(1.0, 1.0 - cv))


def genotype_metrics(sequence: str, repro_symbol: str, semel_symbol: str) -> tuple[float, float, float, float]:
    tokens = tokenize_sequence(sequence)
    genome_length = len(tokens)
    if genome_length == 0:
        return 0.0, 0.0, 0.0, 0.0
    repro_positions = [index for index, symbol in enumerate(tokens) if symbol == repro_symbol]
    semel_count = sum(1 for symbol in tokens if symbol == semel_symbol)
    repro_frequency = len(repro_positions) / genome_length
    spacing = normalized_repro_spacing(repro_positions, genome_length)
    iteroparity_score = repro_frequency * spacing
    semelparity_frequency = semel_count / genome_length
    return iteroparity_score, repro_frequency, spacing, semelparity_frequency


def weighted_median(weighted_values: list[tuple[float, int]]) -> float:
    total_weight = sum(weight for _value, weight in weighted_values)
    if total_weight <= 0:
        return 0.0
    threshold = total_weight / 2
    cumulative = 0
    for value, weight in sorted(weighted_values, key=lambda item: item[0]):
        cumulative += weight
        if cumulative >= threshold:
            return value
    return weighted_values[-1][0]


def weighted_std(weighted_values: list[tuple[float, int]]) -> float:
    total_weight = sum(weight for _value, weight in weighted_values)
    if total_weight <= 0:
        return 0.0
    mean = sum(value * weight for value, weight in weighted_values) / total_weight
    variance = sum(weight * (value - mean) ** 2 for value, weight in weighted_values) / total_weight
    return math.sqrt(variance)


def read_snapshot_metrics(run_dir: str) -> list[tuple[str, int, int, int, list[tuple[float, int]], list[tuple[float, int]]]]:
    run_name = os.path.basename(run_dir)
    details = detail_files(run_dir)
    if not details:
        print(f"WARNING: skipping {run_name}; no non-negative detail snapshots found")
        return []

    symbols = read_instruction_symbols(os.path.join(run_dir, "instset.cfg"))
    repro_symbol = symbols.get("repro")
    semel_symbol = symbols.get("repro-semel")
    if repro_symbol is None or semel_symbol is None:
        print(f"WARNING: skipping {run_name}; missing repro or repro-semel in instset")
        return []

    rows = []
    for update, detail_path in details:
        population = 0
        genotype_count = 0
        iteroparity_values: list[tuple[float, int]] = []
        semelparity_values: list[tuple[float, int]] = []
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
                sequence = fields[16]
                iteroparity, _repro_frequency, _spacing, semel_frequency = genotype_metrics(
                    sequence,
                    repro_symbol,
                    semel_symbol,
                )
                population += living_count
                genotype_count += 1
                iteroparity_values.append((iteroparity, living_count))
                semelparity_values.append((semel_frequency, living_count))
        if population > 0:
            rows.append(
                (
                    group_name_for_run(run_name),
                    update,
                    population,
                    genotype_count,
                    iteroparity_values,
                    semelparity_values,
                )
            )
    return rows


def collect_medians() -> list[SnapshotMedian]:
    grouped_values: dict[tuple[str, int], dict[str, object]] = defaultdict(
        lambda: {
            "population": 0,
            "genotype_count": 0,
            "snapshot_count": 0,
            "iteroparity": [],
            "semelparity": [],
        }
    )
    for run_dir in resolve_run_dirs(TEST_RUNS):
        for group_name, update, population, genotype_count, iteroparity, semelparity in read_snapshot_metrics(run_dir):
            bucket = grouped_values[(group_name, update)]
            bucket["population"] = int(bucket["population"]) + population
            bucket["genotype_count"] = int(bucket["genotype_count"]) + genotype_count
            bucket["snapshot_count"] = int(bucket["snapshot_count"]) + 1
            bucket["iteroparity"].extend(iteroparity)  # type: ignore[union-attr]
            bucket["semelparity"].extend(semelparity)  # type: ignore[union-attr]

    medians: list[SnapshotMedian] = []
    for (group_name, update), bucket in grouped_values.items():
        iteroparity_values = bucket["iteroparity"]  # type: ignore[assignment]
        semelparity_values = bucket["semelparity"]  # type: ignore[assignment]
        medians.append(
            SnapshotMedian(
                group_name=group_name,
                update=update,
                population=int(bucket["population"]),
                genotype_count=int(bucket["genotype_count"]),
                snapshot_count=int(bucket["snapshot_count"]),
                iteroparity_median=weighted_median(iteroparity_values),
                iteroparity_std=weighted_std(iteroparity_values),
                semelparity_median=weighted_median(semelparity_values),
                semelparity_std=weighted_std(semelparity_values),
            )
        )
    return sorted(medians, key=lambda row: (natural_key(row.group_name), row.update))


def write_csv(scores: list[SnapshotMedian], csv_path: str) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "group",
                "update",
                "population",
                "genotypes",
                "snapshots",
                "iteroparity_median",
                "iteroparity_std",
                "semelparity_median",
                "semelparity_std",
            ]
        )
        for score in scores:
            writer.writerow(
                [
                    score.group_name,
                    score.update,
                    score.population,
                    score.genotype_count,
                    score.snapshot_count,
                    score.iteroparity_median,
                    score.iteroparity_std,
                    score.semelparity_median,
                    score.semelparity_std,
                ]
            )


def save_plot(scores: list[SnapshotMedian], pdf_path: str) -> None:
    grouped: dict[str, list[SnapshotMedian]] = {}
    for score in scores:
        grouped.setdefault(score.group_name, []).append(score)

    group_names = sorted(grouped, key=natural_key)
    fig, axes = plt.subplots(len(group_names), 1, figsize=(13, 2.8 * len(group_names)), sharey=True)
    if len(group_names) == 1:
        axes = [axes]

    for ax, group_name in zip(axes, group_names):
        rows = sorted(grouped[group_name], key=lambda row: row.update)
        updates = [row.update for row in rows]
        iteroparity = [row.iteroparity_median * 100 for row in rows]
        iteroparity_std = [row.iteroparity_std * 100 for row in rows]
        semelparity = [row.semelparity_median * 100 for row in rows]
        semelparity_std = [row.semelparity_std * 100 for row in rows]
        iteroparity_low = [max(0.0, value - std) for value, std in zip(iteroparity, iteroparity_std)]
        iteroparity_high = [value + std for value, std in zip(iteroparity, iteroparity_std)]
        semelparity_low = [max(0.0, value - std) for value, std in zip(semelparity, semelparity_std)]
        semelparity_high = [value + std for value, std in zip(semelparity, semelparity_std)]

        ax.plot(
            updates,
            iteroparity,
            color="#2f6f95",
            marker="o",
            markersize=3,
            linewidth=1.7,
            alpha=0.95,
            label="Iteroparity median",
        )
        ax.fill_between(
            updates,
            iteroparity_low,
            iteroparity_high,
            color="#2f6f95",
            alpha=0.16,
            linewidth=0,
            label="Iteroparity +/- SD",
        )
        ax.plot(
            updates,
            semelparity,
            color="#c44e52",
            marker="^",
            markersize=3,
            linewidth=1.7,
            alpha=0.9,
            label="Semelparity median",
        )
        ax.fill_between(
            updates,
            semelparity_low,
            semelparity_high,
            color="#c44e52",
            alpha=0.14,
            linewidth=0,
            label="Semelparity +/- SD",
        )
        ax.set_title(X_AXIS_LABELS.get(group_name, group_name), loc="left")
        ax.set_ylabel("Percent")
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", alpha=0.12)
        ax.set_axisbelow(True)
        ax.ticklabel_format(axis="x", style="plain")

    axes[-1].set_xlabel("Saved update")
    axes[0].legend(loc="upper right", frameon=False, ncol=4)
    fig.suptitle("Population median iteroparity and semelparity over saved updates", y=0.995)
    fig.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)


def main() -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scores = collect_medians()
    if not scores:
        raise SystemExit("No iteroparity scores found.")

    csv_path = os.path.join(OUTPUT_DIR, CSV_NAME)
    pdf_path = os.path.join(OUTPUT_DIR, PDF_NAME)
    write_csv(scores, csv_path)
    save_plot(scores, pdf_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
