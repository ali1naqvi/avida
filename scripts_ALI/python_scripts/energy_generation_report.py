#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import sys
import tempfile
from bisect import bisect_left
from dataclasses import dataclass
from typing import Iterable

_mpl_cache = os.path.join(tempfile.gettempdir(), "avida_matplotlib_cache")
os.makedirs(_mpl_cache, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cache)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Edit these values instead of passing a run folder as a command-line argument.
# WORK_RUN_DIR can be absolute, or relative to the repository root.
WORK_RUN_DIR = "avida-core/build/macos-arm64/work_experiments_with_nonoverlapping/ecology_test_over_fast"
DATA_SUBDIR = "data"
OUTPUT_PDF: str | None = None  # None writes data/energy_by_generation.pdf.

# Number of highest-energy organisms treated as elites at every saved snapshot.
# Set this to the same value as GENERATION_SELECTION_ELITES in avida.cfg.
ELITE_COUNT = 125


@dataclass(frozen=True)
class EnergySnapshot:
    update: float
    generation: float
    energies: tuple[float, ...]


def resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def parse_dat_columns(path: str) -> tuple[list[str], list[list[float]]]:
    """Read Avida's numbered-header numeric .dat format."""
    columns: list[str] = []
    rows: list[list[float]] = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                match = re.match(r"#\s*(\d+)\s*:\s*(.+)", line)
                if match:
                    index = int(match.group(1)) - 1
                    while len(columns) <= index:
                        columns.append("")
                    columns[index] = match.group(2).strip()
                continue
            row: list[float] = []
            for token in line.split():
                try:
                    row.append(float(token))
                except ValueError:
                    # Champion rows finish with a genotype name.  Preserve the
                    # numeric columns before it instead of discarding the row.
                    break
            if row:
                rows.append(row)
    return columns, rows


def column_index(columns: Iterable[str], *names: str) -> int | None:
    normalized = {name.lower(): i for i, name in enumerate(columns)}
    for name in names:
        index = normalized.get(name.lower())
        if index is not None:
            return index
    return None


def read_generation_series(average_path: str) -> list[tuple[float, float]]:
    columns, rows = parse_dat_columns(average_path)
    update_i = column_index(columns, "Update")
    generation_i = column_index(columns, "Generation")
    if update_i is None or generation_i is None:
        raise ValueError(f"{average_path} needs Update and Generation columns")
    out = [
        (row[update_i], row[generation_i])
        for row in rows
        if len(row) > max(update_i, generation_i)
        and math.isfinite(row[update_i])
        and math.isfinite(row[generation_i])
    ]
    if not out:
        raise ValueError(f"No Update/Generation rows in {average_path}")
    return sorted(out)


def generation_at(update: float, update_generations: list[tuple[float, float]]) -> float:
    """Linearly interpolate the mean generation logged around an update."""
    updates = [pair[0] for pair in update_generations]
    pos = bisect_left(updates, update)
    if pos == 0:
        return update_generations[0][1]
    if pos >= len(update_generations):
        return update_generations[-1][1]
    before_update, before_generation = update_generations[pos - 1]
    after_update, after_generation = update_generations[pos]
    if after_update == before_update:
        return after_generation
    fraction = (update - before_update) / (after_update - before_update)
    return before_generation + fraction * (after_generation - before_generation)


def read_org_loc_energies(path: str) -> list[float]:
    """Extract the final ``stored_energy`` CSV field from an org_loc dump."""
    energies: list[float] = []
    with open(path, encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                energy = float(row[-1].strip())
            except ValueError:
                continue
            if math.isfinite(energy):
                energies.append(energy)
    return energies


def org_loc_update(path: str) -> int | None:
    match = re.search(r"org_loc\.(\d+)\.dat$", os.path.basename(path))
    return int(match.group(1)) if match else None


def read_energy_snapshots(data_dir: str, update_generations: list[tuple[float, float]]) -> list[EnergySnapshot]:
    patterns = (
        os.path.join(data_dir, "grid_dumps", "org_loc.*.dat"),
        os.path.join(data_dir, "org_loc.*.dat"),
    )
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    snapshots: list[EnergySnapshot] = []
    for path in paths:
        update = org_loc_update(path)
        if update is None:
            continue
        energies = read_org_loc_energies(path)
        if energies:
            snapshots.append(EnergySnapshot(update, generation_at(update, update_generations), tuple(energies)))
    return sorted(snapshots, key=lambda snapshot: snapshot.update)


def read_best_from_extrema(
    extrema_path: str, update_generations: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    columns, rows = parse_dat_columns(extrema_path)
    update_i = column_index(columns, "Update", "update")
    generation_i = column_index(columns, "Generation", "generation")
    best_i = column_index(columns, "Maximum Stored Energy in Population", "max_fitness")
    death_best_i = column_index(
        columns, "Max stored energy among deaths (last update)", "max_fitness_dead"
    )
    death_count_i = column_index(
        columns, "Deaths counted toward ave/max_fitness_dead", "death_fitness_n"
    )
    if update_i is None or best_i is None:
        raise ValueError(f"{extrema_path} needs Update and Maximum Stored Energy in Population columns")
    best_by_generation: dict[int, float] = {}
    for row in rows:
        if len(row) <= max(update_i, best_i):
            continue
        update = row[update_i]
        if update < 0 or not math.isfinite(update):
            continue

        # New data include a generation column and are written every update, so
        # their live-population maximum can be grouped directly.  Older runs
        # wrote only at a generation boundary, after the live population had
        # reset; their ``max_fitness_dead`` value is the peak from the
        # generation that just ended.
        if generation_i is not None and len(row) > generation_i:
            energy = row[best_i]
            generation = row[generation_i]
        elif (
            death_best_i is not None
            and death_count_i is not None
            and len(row) > max(death_best_i, death_count_i)
            and row[death_count_i] > 0
        ):
            energy = row[death_best_i]
            generation = generation_at(update, update_generations) - 1
        else:
            energy = row[best_i]
            generation = generation_at(update, update_generations)

        if math.isfinite(energy) and math.isfinite(generation):
            generation_number = math.floor(generation)
            best_by_generation[generation_number] = max(best_by_generation.get(generation_number, -math.inf), energy)
    return [(float(generation), energy) for generation, energy in sorted(best_by_generation.items())]


def configure_axis(ax) -> None:
    ax.grid(True, color="#d7dce3", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Energy")


def add_best_page(pdf: PdfPages, points: list[tuple[float, float]], source: str) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27), layout="constrained")
    generations, energies = zip(*points)
    ax.plot(generations, energies, color="#146c94", linewidth=1.8, label="Best live energy")
    configure_axis(ax)
    ax.set_title("Best stored energy by generation", loc="left", weight="bold", fontsize=16, pad=20)
    ax.text(0, 1.01, source, transform=ax.transAxes, color="#50606f", fontsize=9, va="bottom")
    ax.legend(frameon=False, loc="best")
    fig.text(0.98, 0.015, "Page 1 of 2", ha="right", color="#50606f", fontsize=9)
    pdf.savefig(fig)
    plt.close(fig)


def add_elite_page(pdf: PdfPages, snapshots: list[EnergySnapshot], elite_count: int, note: str | None) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27), layout="constrained")
    configure_axis(ax)
    ax.set_title("Elite stored-energy distribution by generation", loc="left", weight="bold", fontsize=16, pad=20)
    if not snapshots:
        ax.set_axis_off()
        ax.text(
            0.5, 0.56, "Elite distribution unavailable", ha="center", va="center", fontsize=22, weight="bold", color="#34495e"
        )
        ax.text(
            0.5, 0.43,
            "No per-organism org_loc energy snapshots were found.\n"
            "Enable PrintOrgLocData env and rerun to create a true top-N distribution.",
            ha="center", va="center", fontsize=11, color="#50606f", linespacing=1.5,
        )
        if note:
            ax.text(0.5, 0.29, note, ha="center", va="center", fontsize=9, color="#7b8794")
    else:
        generations: list[float] = []
        medians: list[float] = []
        stds: list[float] = []
        elite_x: list[float] = []
        elite_y: list[float] = []
        for snapshot in snapshots:
            elite = sorted(snapshot.energies, reverse=True)[:elite_count]
            if not elite:
                continue
            median = float(sorted(elite)[len(elite) // 2])
            if len(elite) % 2 == 0:
                ordered = sorted(elite)
                median = (ordered[len(elite) // 2 - 1] + ordered[len(elite) // 2]) / 2
            mean = sum(elite) / len(elite)
            std = math.sqrt(sum((energy - mean) ** 2 for energy in elite) / len(elite))
            generations.append(snapshot.generation)
            medians.append(median)
            stds.append(std)
            elite_x.extend([snapshot.generation] * len(elite))
            elite_y.extend(elite)

        ax.scatter(elite_x, elite_y, s=13, color="#9aa5b1", alpha=0.35, linewidths=0, label="Elite organisms")
        lower = [median - std for median, std in zip(medians, stds)]
        upper = [median + std for median, std in zip(medians, stds)]
        ax.fill_between(generations, lower, upper, color="#f4a261", alpha=0.38, label="Median +/- 1 SD")
        ax.plot(generations, medians, color="#7b2cbf", linewidth=2.1, label="Elite median")
        ax.scatter(generations, medians, color="#7b2cbf", s=14, zorder=3)
        ax.text(
            0, 1.01,
            f"Top {elite_count} live organisms at each saved snapshot; standard deviation is population SD.",
            transform=ax.transAxes, color="#50606f", fontsize=9, va="bottom",
        )
        ax.legend(frameon=False, loc="best")
    fig.text(0.98, 0.015, "Page 2 of 2", ha="right", color="#50606f", fontsize=9)
    pdf.savefig(fig)
    plt.close(fig)


def build_report(run_dir: str, data_subdir: str, output_pdf: str, elite_count: int) -> tuple[str, int, str]:
    if elite_count < 1:
        raise ValueError("ELITE_COUNT must be at least 1")
    data_dir = os.path.join(run_dir, data_subdir)
    average_path = os.path.join(data_dir, "average.dat")
    extrema_path = os.path.join(data_dir, "fitness_extrema.dat")
    if not os.path.isfile(average_path):
        raise FileNotFoundError(f"Missing {average_path}")
    if not os.path.isfile(extrema_path):
        raise FileNotFoundError(f"Missing {extrema_path}")

    update_generations = read_generation_series(average_path)
    snapshots = read_energy_snapshots(data_dir, update_generations)
    best_points = read_best_from_extrema(extrema_path, update_generations)
    best_source = "Maximum live population energy observed during each generation from fitness_extrema.dat."
    if not best_points:
        raise ValueError("No usable stored-energy observations found")

    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        add_best_page(pdf, best_points, best_source)
        add_elite_page(pdf, snapshots, elite_count, "Page 1 uses fitness_extrema.dat.")
    return output_pdf, len(snapshots), best_source


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a two-page stored-energy PDF for an Avida run.")
    parser.add_argument("--run-dir", default=WORK_RUN_DIR, help="Defaults to WORK_RUN_DIR at the top of this file.")
    parser.add_argument("--data-dir", default=DATA_SUBDIR, help="Data directory relative to --run-dir.")
    parser.add_argument("--output", default=OUTPUT_PDF, help="PDF output path (default: DATA_DIR/energy_by_generation.pdf).")
    parser.add_argument("--elite-count", type=int, default=ELITE_COUNT, help="Top-N organisms in each elite distribution.")
    args = parser.parse_args()

    run_dir = os.path.abspath(resolve_path(args.run_dir))
    data_dir = os.path.join(run_dir, args.data_dir)
    output_pdf = os.path.abspath(args.output) if args.output else os.path.join(data_dir, "energy_by_generation.pdf")
    try:
        output, snapshot_count, source = build_report(run_dir, args.data_dir, output_pdf, args.elite_count)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {output}")
    print(f"Best-energy source: {source}")
    print(f"Elite snapshots used: {snapshot_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
