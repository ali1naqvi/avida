#!/usr/bin/env python3
"""Build one PDF with each diagnostic plot on its own page."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

_mpl_cache = os.path.join(tempfile.gettempdir(), "avida_matplotlib_cache")
os.makedirs(_mpl_cache, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cache)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from replay_living_champion import (  # noqa: E402
    find_latest_spop,
    parse_spop_rows,
    pick_spop_last_generation,
)


RUN_FOLDER = SCRIPT_DIR
DATA_SUBDIR = "data"
SPOP_PREFIX = "detail"


def instruction_operand_to_symbol(operand: int) -> str:
    if operand == 255:
        return "_"
    idx = 0
    offset = operand % 62
    sym = ["", ""]
    prefix = operand // 62
    if prefix == 1:
        idx = 1
        sym[0] = "+"
    elif prefix == 2:
        idx = 1
        sym[0] = "-"
    elif prefix == 3:
        idx = 1
        sym[0] = "~"
    elif prefix == 4:
        idx = 1
        sym[0] = "?"
    if offset < 26:
        sym[idx] = chr(ord("a") + offset)
    elif offset < 52:
        sym[idx] = chr(ord("A") + offset - 26)
    else:
        sym[idx] = chr(ord("0") + offset - 52)
    return "".join(sym)


def parse_inst_names_ordered(instset_path: str) -> list[str]:
    names: list[str] = []
    with open(instset_path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if line.startswith("INST "):
                parts = line.split()
                if len(parts) >= 2:
                    names.append(parts[1])
    return names


def name_to_symbol_map(names: list[str]) -> dict[str, str]:
    return {name: instruction_operand_to_symbol(i) for i, name in enumerate(names)}


def tokenize_instruction_sequence(sequence: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    prefixes = {"+", "-", "~", "?"}
    while i < len(sequence):
        ch = sequence[i]
        if ch == "_":
            i += 1
            continue
        if ch in prefixes and i + 1 < len(sequence):
            tokens.append(sequence[i : i + 2])
            i += 2
            continue
        tokens.append(ch)
        i += 1
    return tokens


def resolve_spop_path(data_dir: str, prefix: str, average_dat: str, use_last_generation: bool) -> tuple[str, str]:
    if use_last_generation:
        path, _upd, meta = pick_spop_last_generation(data_dir, prefix, average_dat)
        if path:
            return path, meta.get("snapshot_rule", "")
    path, upd = find_latest_spop(data_dir, prefix)
    if path:
        return path, f"latest update ({upd})"
    raise FileNotFoundError(f"No {prefix}-*.spop files under {data_dir}")


def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    w = weights.astype(float)
    if np.sum(w) <= 0:
        return float("nan"), float("nan")
    mean = float(np.sum(values * w) / np.sum(w))
    variance = float(np.sum(w * (values - mean) ** 2) / np.sum(w))
    return mean, variance**0.5


def parse_data_file(path: str) -> Dict[str, List[float]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")

    columns: List[str] = []
    rows: List[List[float]] = []
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                if ":" in line:
                    left, right = line.split(":", 1)
                    left = left.replace("#", "").strip()
                    if left.isdigit():
                        columns.append(right.strip())
                continue
            parsed_row: List[float] = []
            for token in line.split():
                try:
                    parsed_row.append(float(token))
                except ValueError:
                    break
            if parsed_row:
                rows.append(parsed_row)

    if not rows:
        raise ValueError(f"No numeric data rows found in: {path}")

    width = max(len(r) for r in rows)
    for row in rows:
        if len(row) < width:
            row.extend([float("nan")] * (width - len(row)))
    if len(columns) < width:
        columns.extend([f"Column {len(columns) + i + 1}" for i in range(width - len(columns))])

    data: Dict[str, List[float]] = {name: [] for name in columns[:width]}
    for row in rows:
        for i in range(width):
            data[columns[i]].append(row[i])
    return data


def parse_optional_data_file(path: str) -> Dict[str, List[float]] | None:
    return parse_data_file(path) if os.path.exists(path) else None


def parse_config_int(path: str, key: str, default: int) -> int:
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0] == key:
                try:
                    return int(float(parts[1]))
                except ValueError:
                    return default
    return default


def first_existing_key(data: Dict[str, List[float]], names: Sequence[str]) -> str | None:
    for name in names:
        if name in data:
            return name
    return None


def finite_peak_index(values: Sequence[float]) -> int | None:
    best_i: int | None = None
    best_v = -float("inf")
    for i, value in enumerate(values):
        v = float(value)
        if math.isnan(v):
            continue
        if best_i is None or v > best_v:
            best_i = i
            best_v = v
    return best_i


def annotate_peak_point(
    ax,
    xs: Sequence[float],
    ys: Sequence[float],
    label: str,
    color: str,
    xytext: tuple[int, int] = (8, 8),
) -> None:
    idx = finite_peak_index(ys)
    if idx is None or idx >= len(xs):
        return
    x = float(xs[idx])
    y = float(ys[idx])
    ax.scatter([x], [y], marker="X", s=80, color=color, edgecolors="black", linewidths=0.6, zorder=5)
    ax.annotate(
        f"peak {label}: {y:.4g}\n@ {x:.4g}",
        xy=(x, y),
        xytext=xytext,
        textcoords="offset points",
        fontsize=8,
        color=color,
        arrowprops={"arrowstyle": "->", "color": color, "linewidth": 0.8},
    )


def annotate_peak_bar(ax, x_positions: Sequence[float], values: Sequence[float], labels: Sequence[str], color: str) -> None:
    idx = finite_peak_index(values)
    if idx is None or idx >= len(x_positions) or idx >= len(labels):
        return
    x = float(x_positions[idx])
    y = float(values[idx])
    ax.scatter([x], [y], marker="X", s=70, color=color, edgecolors="black", linewidths=0.6, zorder=5)
    ax.annotate(
        f"peak: {labels[idx]}\n{y:.4g}",
        xy=(x, y),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=8,
        color=color,
        arrowprops={"arrowstyle": "->", "color": color, "linewidth": 0.8},
    )


def rates(values: Sequence[float], times: Sequence[float]) -> List[float]:
    out = [0.0]
    for i in range(1, len(values)):
        dt = times[i] - times[i - 1]
        out.append((values[i] - values[i - 1]) / dt if dt > 0 else float("nan"))
    return out


def _as_int_update(values: Sequence[float]) -> List[int]:
    return [-1 if v is None or (isinstance(v, float) and math.isnan(v)) else int(round(float(v))) for v in values]


def _norm_col(name: str) -> str:
    return "".join(c.lower() for c in name if c.isalnum())


def _extrema_update_key(extrema: Dict[str, List[float]]) -> str | None:
    for key in ("update", "Update"):
        if key in extrema:
            return key
    return None


def _extrema_ave_fitness_dead_key(extrema: Dict[str, List[float]]) -> str | None:
    for name in extrema:
        n = name.lower()
        if _norm_col(name) == "avefitnessdead":
            return name
        if "fitness" in n and ("dead" in n or "death" in n or "terminal" in n) and (
            n.startswith("ave") or n.startswith("mean") or "mean terminal" in n
        ):
            return name
    return None


def _extrema_max_fitness_dead_key(extrema: Dict[str, List[float]], allow_live_fallback: bool) -> str | None:
    fallback: List[str] = []
    for name in extrema:
        n = name.lower()
        if _norm_col(name) == "maxfitnessdead":
            return name
        if "fitness" in n and ("dead" in n or "death" in n or "terminal" in n) and (
            n.startswith("max") or "max terminal" in n or "maximum terminal" in n
        ):
            return name
        if "max" in n and "fitness" in n:
            fallback.append(name)
    if allow_live_fallback and "max_fitness" in extrema:
        return "max_fitness"
    return fallback[0] if allow_live_fallback and fallback else None


def _extrema_death_count_key(extrema: Dict[str, List[float]]) -> str | None:
    fallback = None
    for name in extrema:
        n = name.lower()
        if _norm_col(name) == "deathfitnessn":
            return name
        if fallback is None and (
            n.startswith("deaths counted") or ("death" in n and "fitness" in n and ("count" in n or "sample" in n))
        ):
            fallback = name
    return fallback


def _build_dead_extrema_lookup(extrema: Dict[str, List[float]] | None, value_key: str | None) -> List[Tuple[int, float]]:
    if extrema is None or not value_key or value_key not in extrema:
        return []
    key_u = _extrema_update_key(extrema)
    if not key_u:
        return []

    count_key = _extrema_death_count_key(extrema)
    updates = _as_int_update(extrema[key_u])
    vals = extrema[value_key]
    counts = extrema[count_key] if count_key and count_key in extrema else None
    pairs: List[Tuple[int, float]] = []
    for i, update in enumerate(updates):
        if update < 0 or i >= len(vals):
            continue
        if counts is not None and (i >= len(counts) or float(counts[i]) <= 0):
            continue
        value = float(vals[i])
        if not math.isnan(value):
            pairs.append((update, value))
    return sorted(pairs, key=lambda t: t[0])


def _lookup_value_forward(ext_pairs: Sequence[Tuple[int, float]], update: int) -> float:
    if not ext_pairs:
        return float("nan")
    best = float("nan")
    for eu, ev in ext_pairs:
        if eu > update:
            break
        best = ev
    return best


def align_on_update(
    average: Dict[str, List[float]],
    count: Dict[str, List[float]],
    extrema: Dict[str, List[float]] | None,
    allow_live_fitness_fallback: bool,
) -> Tuple[List[int], List[float], List[float], List[float], List[float], List[float], List[float]]:
    avg_u = _as_int_update(average["Update"])
    cnt_u = _as_int_update(count["update"])

    ave_dead_col = _extrema_ave_fitness_dead_key(extrema) if extrema else None
    max_dead_col = _extrema_max_fitness_dead_key(extrema, allow_live_fitness_fallback) if extrema else None
    ext_pairs_ave_dead = _build_dead_extrema_lookup(extrema, ave_dead_col)
    ext_pairs_max_dead = _build_dead_extrema_lookup(extrema, max_dead_col)

    if not allow_live_fitness_fallback and (not ext_pairs_ave_dead or not ext_pairs_max_dead):
        raise ValueError(
            "No death-based fitness data found. Re-run with fitness_extrema.dat columns "
            "ave_fitness_dead/max_fitness_dead, or pass --allow-live-fitness-fallback."
        )

    avg_i = {u: i for i, u in enumerate(avg_u) if u >= 0}
    cnt_i = {u: i for i, u in enumerate(cnt_u) if u >= 0}
    common = sorted(set(avg_i) & set(cnt_i))
    if not common:
        raise ValueError("No overlapping updates found across average/count.")

    updates: List[int] = []
    avg_gen: List[float] = []
    avg_fit: List[float] = []
    max_fit: List[float] = []
    pop_size: List[float] = []
    births: List[float] = []
    deaths: List[float] = []

    for update in common:
        ai = avg_i[update]
        ci = cnt_i[update]
        updates.append(update)
        avg_gen.append(float(average["Generation"][ai]))
        if ext_pairs_ave_dead:
            avg_fit.append(_lookup_value_forward(ext_pairs_ave_dead, update))
        elif allow_live_fitness_fallback:
            avg_fit.append(float(average["Fitness"][ai]))
        else:
            avg_fit.append(float("nan"))
        max_fit.append(_lookup_value_forward(ext_pairs_max_dead, update))
        pop_size.append(float(count["number of organisms"][ci]))
        births.append(float(count["number of births in this update"][ci]))
        deaths.append(float(count["number of deaths in this update"][ci]))
    return updates, avg_gen, avg_fit, max_fit, pop_size, births, deaths


def aggregate_per_generation(
    avg_gen: Sequence[float],
    avg_fit: Sequence[float],
    max_fit: Sequence[float],
    pop_size: Sequence[float],
    births: Sequence[float],
    deaths: Sequence[float],
) -> Tuple[List[int], List[float], List[float], List[float], List[float], List[float], List[float], List[float]]:
    n = min(len(avg_gen), len(avg_fit), len(max_fit), len(pop_size), len(births), len(deaths))
    avg_fit_sum: Dict[int, float] = defaultdict(float)
    avg_fit_cnt: Dict[int, int] = defaultdict(int)
    max_fit_max: Dict[int, float] = {}
    pop_sum: Dict[int, float] = defaultdict(float)
    pop_cnt: Dict[int, int] = defaultdict(int)
    birth_sum: Dict[int, float] = defaultdict(float)
    death_sum: Dict[int, float] = defaultdict(float)
    pop_first: Dict[int, float] = {}
    pop_last: Dict[int, float] = {}

    for i in range(n):
        gen = int(math.floor(float(avg_gen[i])))
        af = float(avg_fit[i])
        if not math.isnan(af):
            avg_fit_sum[gen] += af
            avg_fit_cnt[gen] += 1
        mf = float(max_fit[i])
        if not math.isnan(mf) and (gen not in max_fit_max or mf > max_fit_max[gen]):
            max_fit_max[gen] = mf
        pop_sum[gen] += float(pop_size[i])
        pop_cnt[gen] += 1
        birth_sum[gen] += float(births[i])
        death_sum[gen] += float(deaths[i])
        if gen not in pop_first:
            pop_first[gen] = float(pop_size[i])
        pop_last[gen] = float(pop_size[i])

    gens = sorted(pop_cnt)
    avg_fit_g = [avg_fit_sum[g] / avg_fit_cnt[g] if avg_fit_cnt[g] else float("nan") for g in gens]
    max_fit_g = [max_fit_max.get(g, float("nan")) for g in gens]
    pop_g = [pop_sum[g] / pop_cnt[g] for g in gens]
    births_g = [birth_sum[g] for g in gens]
    deaths_g = [death_sum[g] for g in gens]
    pop_first_g = [pop_first[g] for g in gens]
    pop_last_g = [pop_last[g] for g in gens]
    return gens, avg_fit_g, max_fit_g, pop_g, births_g, deaths_g, pop_first_g, pop_last_g


def add_repro_pages(
    pdf: PdfPages,
    run_dir: str,
    data_dir: str,
    inst_path: str,
    spop_prefix: str,
    use_last_generation_snapshot: bool,
) -> None:
    average_dat = os.path.join(data_dir, "average.dat")
    spop_path, rule = resolve_spop_path(data_dir, spop_prefix, average_dat, use_last_generation_snapshot)
    rows = parse_spop_rows(spop_path)
    alive = [r for r in rows if r["num_live"] > 0]
    if not alive:
        raise ValueError(f"No genotypes with num_live > 0 in {spop_path}")

    sym_for = name_to_symbol_map(parse_inst_names_ordered(inst_path))
    for need in ("repro", "repro-semel"):
        if need not in sym_for:
            raise ValueError(f"Instruction {need!r} not found in {inst_path}")

    sym_repro = sym_for["repro"]
    sym_semel = sym_for["repro-semel"]
    n_orgs = sum(r["num_live"] for r in alive)

    p_repro: List[float] = []
    p_semel: List[float] = []
    p_any: List[float] = []
    fitness: List[float] = []
    weights: List[int] = []
    for row in alive:
        seq = row["sequence"].replace("_", "")
        if not seq:
            continue
        nr = seq.count(sym_repro)
        ns = seq.count(sym_semel)
        p_repro.append(nr / len(seq))
        p_semel.append(ns / len(seq))
        p_any.append((nr + ns) / len(seq))
        fitness.append(row["fitness"])
        weights.append(row["num_live"])

    p_repro_a = np.array(p_repro, dtype=float)
    p_semel_a = np.array(p_semel, dtype=float)
    p_any_a = np.array(p_any, dtype=float)
    fit_a = np.array(fitness, dtype=float)
    w_a = np.array(weights, dtype=float)
    m_repro, sd_repro = weighted_mean_std(p_repro_a, w_a)
    m_semel, sd_semel = weighted_mean_std(p_semel_a, w_a)
    m_any, sd_any = weighted_mean_std(p_any_a, w_a)

    match = re.search(r"-(\d+)\.spop$", spop_path)
    spop_update = int(match.group(1)) if match else -1

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["repro", "repro-semel", "repro + repro-semel"]
    x = np.arange(len(labels))
    ax.bar(x, [m_repro, m_semel, m_any], yerr=[sd_repro, sd_semel, sd_any], capsize=6, color=["#4C72B0", "#55A868", "#8172B2"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("fraction of genome sites (org-weighted)")
    ax.set_title("Reproduction instructions in the population\n(mean +/- weighted SD over genotypes; weights = num_live)")
    ax.set_ylim(bottom=0)
    fig.text(
        0.5,
        0.01,
        f"Run: {run_dir}\nSnapshot: {os.path.basename(spop_path)} (update {spop_update})\n"
        f"Selection: {rule}\nLiving genotypes: {len(alive)}; organisms: {n_orgs}",
        ha="center",
        fontsize=8,
        color="gray",
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    pdf.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sizes = 18.0 * np.sqrt(w_a / max(1, np.min(w_a)))
    sc = ax.scatter(p_any_a, fit_a, s=sizes, c=np.log1p(w_a), cmap="viridis", alpha=0.65, edgecolors="k", linewidths=0.2)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("log(1 + num_live)")
    valid = w_a > 0
    if np.sum(valid) >= 2 and np.std(p_any_a[valid]) > 1e-12:
        coeff = np.polyfit(p_any_a[valid], fit_a[valid], 1, w=w_a[valid])
        xs = np.linspace(np.min(p_any_a), np.max(p_any_a), 50)
        ax.plot(xs, coeff[0] * xs + coeff[1], "r--", linewidth=1.5, label="weighted linear fit")
        ax.legend(loc="best")
    ax.set_xlabel("fraction of sites that are repro or repro-semel")
    ax.set_ylabel("fitness (genotype average from .spop)")
    ax.set_title("Fitness vs. reproduction-instruction density\n(one point per living genotype)")
    fig.text(
        0.5,
        0.01,
        f"Symbols: repro={sym_repro!r}, repro-semel={sym_semel!r} (from {os.path.basename(inst_path)})",
        ha="center",
        fontsize=8,
        color="gray",
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    pdf.savefig(fig)
    plt.close(fig)


def add_instruction_usage_pages(
    pdf: PdfPages,
    run_dir: str,
    data_dir: str,
    inst_path: str,
    spop_prefix: str,
    use_last_generation_snapshot: bool,
) -> None:
    average_dat = os.path.join(data_dir, "average.dat")
    spop_path, rule = resolve_spop_path(data_dir, spop_prefix, average_dat, use_last_generation_snapshot)
    rows = parse_spop_rows(spop_path)
    alive = [r for r in rows if r["num_live"] > 0]
    if not alive:
        raise ValueError(f"No genotypes with num_live > 0 in {spop_path}")

    names = parse_inst_names_ordered(inst_path)
    sym_for = name_to_symbol_map(names)
    name_for = {sym: name for name, sym in sym_for.items()}
    ordered_symbols = [sym_for[name] for name in names]

    weighted_counts: Counter[str] = Counter()
    genotype_counts: Counter[str] = Counter()
    total_weighted_sites = 0
    total_organisms = 0
    for row in alive:
        tokens = tokenize_instruction_sequence(row["sequence"])
        token_counts = Counter(tokens)
        num_live = int(row["num_live"])
        total_organisms += num_live
        total_weighted_sites += len(tokens) * num_live
        for sym, count in token_counts.items():
            weighted_counts[sym] += count * num_live
            genotype_counts[sym] += 1

    labels = [name_for[sym] for sym in ordered_symbols]
    weighted_values = [
        weighted_counts[sym] / total_weighted_sites if total_weighted_sites > 0 else float("nan")
        for sym in ordered_symbols
    ]
    genotype_values = [genotype_counts[sym] / len(alive) for sym in ordered_symbols]
    category_totals: Counter[str] = Counter()
    category_genotypes: Counter[str] = Counter()
    for row in alive:
        token_counts = Counter(tokenize_instruction_sequence(row["sequence"]))
        tokens = set(token_counts)
        num_live = int(row["num_live"])
        categories_present: set[str] = set()
        for name in names:
            sym = sym_for[name]
            if sym not in tokens:
                continue
            category = instruction_category(name)
            category_totals[category] += token_counts[sym] * num_live
            categories_present.add(category)
        for category in categories_present:
            category_genotypes[category] += 1
    category_order = ["reproduction", "movement", "sensing", "conditionals", "math/logic/stack", "no-op", "other"]
    category_labels = [c for c in category_order if c in category_totals or c in category_genotypes]
    category_weighted_values = [
        category_totals[c] / total_weighted_sites if total_weighted_sites > 0 else float("nan")
        for c in category_labels
    ]
    category_presence_values = [category_genotypes[c] / len(alive) for c in category_labels]

    match = re.search(r"-(\d+)\.spop$", spop_path)
    spop_update = int(match.group(1)) if match else -1
    footer = (
        f"Run: {run_dir}\nSnapshot: {os.path.basename(spop_path)} (update {spop_update})\n"
        f"Selection: {rule}\nLiving genotypes: {len(alive)}; organisms: {total_organisms}"
    )

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(labels))
    ax.bar(x, weighted_values, color="#4c78a8", alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("fraction of instruction sites (organism-weighted)")
    ax.set_title("Instruction usage in final population")
    ax.grid(axis="y", alpha=0.25)
    annotate_peak_bar(ax, x, weighted_values, labels, "#4c78a8")
    fig.text(0.5, 0.01, footer, ha="center", fontsize=8, color="gray")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.32)
    pdf.savefig(fig)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x_cat = np.arange(len(category_labels))
    width = 0.38
    bars1 = ax1.bar(
        x_cat - width / 2,
        category_weighted_values,
        width=width,
        color="#4c78a8",
        alpha=0.88,
        label="site fraction",
    )
    ax1.set_ylabel("fraction of instruction sites")
    ax1.set_ylim(bottom=0)
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        x_cat + width / 2,
        category_presence_values,
        width=width,
        color="#f58518",
        alpha=0.78,
        label="genotype presence",
    )
    ax2.set_ylabel("fraction of living genotypes")
    ax2.set_ylim(0, 1)
    ax1.set_xticks(x_cat)
    ax1.set_xticklabels(category_labels, rotation=25, ha="right")
    ax1.set_title("Instruction categories in final population")
    annotate_peak_bar(ax1, [xi - width / 2 for xi in x_cat], category_weighted_values, category_labels, "#4c78a8")
    annotate_peak_bar(ax2, [xi + width / 2 for xi in x_cat], category_presence_values, category_labels, "#f58518")
    ax1.legend(handles=[bars1, bars2], loc="upper right", fontsize=9)
    fig.text(0.5, 0.01, footer, ha="center", fontsize=8, color="gray")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.27)
    pdf.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x, genotype_values, color="#54a24b", alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of living genotypes containing instruction")
    ax.set_title("Instruction presence across final living genotypes")
    ax.grid(axis="y", alpha=0.25)
    annotate_peak_bar(ax, x, genotype_values, labels, "#54a24b")
    fig.text(0.5, 0.01, footer, ha="center", fontsize=8, color="gray")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.32)
    pdf.savefig(fig)
    plt.close(fig)


def instruction_category(name: str) -> str:
    if name.startswith("repro"):
        return "reproduction"
    if name == "move" or name.startswith("rotate"):
        return "movement"
    if name.startswith("sense"):
        return "sensing"
    if name.startswith("if-"):
        return "conditionals"
    if name in {"nop-A", "nop-B", "nop-C"}:
        return "no-op"
    if name in {"zero", "inc", "dec", "push", "pop", "swap-stk", "swap", "add", "sub", "nand"}:
        return "math/logic/stack"
    return "other"


def movement_series(data_dir: str) -> list[tuple[str, Dict[str, List[float]]]]:
    out: list[tuple[str, Dict[str, List[float]]]] = []
    for filename, label in (("champion.dat", "current champion"), ("lifetime_champion.dat", "lifetime champion")):
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            continue
        data = parse_data_file(path)
        if first_existing_key(data, ["Env X", "Env Y", "Easterly", "Northerly", "StepDisp"]):
            out.append((label, data))
    return out


def parse_update_from_grid_path(path: str) -> int | None:
    match = re.search(r"max_res_grid\.(\d+)\.dat$", os.path.basename(path))
    return int(match.group(1)) if match else None


def parse_update_from_dump_path(path: str, prefix: str) -> int | None:
    match = re.search(rf"{re.escape(prefix)}\.(\d+)\.dat$", os.path.basename(path))
    return int(match.group(1)) if match else None


def dump_paths(data_dir: str, prefix: str) -> list[tuple[int, str]]:
    grid_dir = os.path.join(data_dir, "grid_dumps")
    if not os.path.isdir(grid_dir):
        return []
    paths = []
    for filename in os.listdir(grid_dir):
        path = os.path.join(grid_dir, filename)
        update = parse_update_from_dump_path(path, prefix)
        if update is not None:
            paths.append((update, path))
    return sorted(paths, key=lambda row: row[0])


def nearest_dump_path(paths: Sequence[tuple[int, str]], update: int, prefer_before: bool) -> tuple[int, str] | None:
    if not paths:
        return None
    if prefer_before:
        candidates = [row for row in paths if row[0] <= update]
        return candidates[-1] if candidates else paths[0]
    candidates = [row for row in paths if row[0] >= update]
    return candidates[0] if candidates else paths[-1]


def read_resource_grid(path: str) -> np.ndarray:
    rows: list[list[float]] = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            vals: list[float] = []
            for token in raw.split():
                try:
                    vals.append(float(token))
                except ValueError:
                    vals = []
                    break
            if vals:
                rows.append(vals)
    if not rows:
        return np.zeros((1, 1))
    width = max(len(row) for row in rows)
    for row in rows:
        if len(row) < width:
            row.extend([float("nan")] * (width - len(row)))
    return np.array(rows, dtype=float)


def resource_grid_peak(path: str) -> tuple[float, float, float, int] | None:
    best = -float("inf")
    best_cells: list[tuple[int, int]] = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for y, raw in enumerate(handle):
            values: list[float] = []
            for token in raw.split():
                try:
                    values.append(float(token))
                except ValueError:
                    values = []
                    break
            for x, value in enumerate(values):
                if math.isnan(value):
                    continue
                if value > best:
                    best = value
                    best_cells = [(x, y)]
                elif value == best:
                    best_cells.append((x, y))
    if not best_cells:
        return None
    peak_x = sum(x for x, _y in best_cells) / len(best_cells)
    peak_y = sum(y for _x, y in best_cells) / len(best_cells)
    return peak_x, peak_y, best, len(best_cells)


def generation_at_update(timeline: Sequence[tuple[int, float]], update: int) -> float:
    generation = float("nan")
    for row_update, row_generation in timeline:
        if row_update > update:
            break
        generation = row_generation
    return generation


def average_generation_timeline(data_dir: str) -> list[tuple[int, float]]:
    path = os.path.join(data_dir, "average.dat")
    if not os.path.exists(path):
        return []
    average = parse_data_file(path)
    if "Update" not in average or "Generation" not in average:
        return []
    timeline = []
    for update, generation in zip(average["Update"], average["Generation"]):
        if math.isnan(update) or math.isnan(generation):
            continue
        timeline.append((int(round(update)), float(generation)))
    return sorted(timeline, key=lambda row: row[0])


def write_peak_location_file(
    data_dir: str,
    run_dir: str,
    out_path: str | None = None,
) -> str:
    out_path = out_path or os.path.join(data_dir, "peak_location.dat")
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "SetGradientResourceLeastPopulated" in line:
                    return out_path

    grid_dir = os.path.join(data_dir, "grid_dumps")
    timeline = average_generation_timeline(data_dir)
    rows: list[tuple[int, float, float, float, float, int, str]] = []
    if os.path.isdir(grid_dir):
        paths = []
        for filename in os.listdir(grid_dir):
            if re.match(r"max_res_grid\.\d+\.dat$", filename):
                path = os.path.join(grid_dir, filename)
                update = parse_update_from_grid_path(path)
                if update is not None:
                    paths.append((update, path))
        for update, path in sorted(paths):
            peak = resource_grid_peak(path)
            if peak is None:
                continue
            peak_x, peak_y, peak_value, tie_count = peak
            rows.append((update, generation_at_update(timeline, update), peak_x, peak_y, peak_value, tie_count, "grid_dump"))

    if not rows:
        width = parse_config_int(os.path.join(run_dir, "avida.cfg"), "WORLD_X", 50)
        height = parse_config_int(os.path.join(run_dir, "avida.cfg"), "WORLD_Y", 50)
        rows.append((0, 0.0, width / 2.0, height / 2.0, float("nan"), 1, "center_fallback"))

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("#  1: Update\n")
        handle.write("#  2: Generation\n")
        handle.write("#  3: Peak X\n")
        handle.write("#  4: Peak Y\n")
        handle.write("#  5: Peak resource value\n")
        handle.write("#  6: Tied peak cells\n")
        handle.write("#  7: Source\n")
        for row in rows:
            update, generation, peak_x, peak_y, peak_value, tie_count, source = row
            handle.write(f"{update} {generation:.10g} {peak_x:.10g} {peak_y:.10g} {peak_value:.10g} {tie_count} {source}\n")
    return out_path


def parse_peak_location_file(path: str) -> Dict[str, List[float]]:
    data: Dict[str, List[float]] = {
        "Update": [],
        "Generation": [],
        "Peak X": [],
        "Peak Y": [],
        "Peak resource value": [],
        "Tied peak cells": [],
    }
    with open(path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            for key, token in zip(data, parts[:6]):
                try:
                    data[key].append(float(token))
                except ValueError:
                    data[key].append(float("nan"))
    return data


def parse_org_loc_file(path: str) -> dict[int, tuple[float, float]]:
    orgs: dict[int, tuple[float, float]] = {}
    if not os.path.exists(path):
        return orgs
    with open(path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                orgs[int(parts[0])] = (float(parts[1]), float(parts[2]))
            except ValueError:
                continue
    return orgs


def plot_gradient_panel(
    ax,
    grid: np.ndarray,
    title: str,
    orgs: dict[int, tuple[float, float]] | None = None,
    survivor_ids: set[int] | None = None,
    peak_xy: tuple[float, float] | None = None,
):
    image = ax.imshow(grid, origin="lower", cmap="viridis", interpolation="nearest")
    if orgs:
        xs = [xy[0] for xy in orgs.values()]
        ys = [xy[1] for xy in orgs.values()]
        ax.scatter(xs, ys, s=7, color="white", edgecolors="black", linewidths=0.15, alpha=0.55, label="organisms")
        if survivor_ids:
            sx = [orgs[oid][0] for oid in survivor_ids if oid in orgs]
            sy = [orgs[oid][1] for oid in survivor_ids if oid in orgs]
            if sx:
                ax.scatter(sx, sy, s=28, facecolors="none", edgecolors="#ff4da6", linewidths=1.2, label="survivors")
    if peak_xy:
        ax.scatter([peak_xy[0]], [peak_xy[1]], marker="X", s=100, color="red", edgecolors="black", linewidths=0.7, label="peak")
    ax.set_xlabel("Env X")
    ax.set_ylabel("Env Y")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7)
    return image


def peak_xy_at_or_before(peak: Dict[str, List[float]], update: int) -> tuple[float, float] | None:
    best: tuple[float, float] | None = None
    for row_update, x, y in zip(peak["Update"], peak["Peak X"], peak["Peak Y"]):
        if row_update > update:
            break
        best = (x, y)
    return best


def first_peak_change(peak: Dict[str, List[float]]) -> tuple[int, float, float, float] | None:
    last_xy: tuple[float, float] | None = None
    for update, generation, x, y in zip(peak["Update"], peak["Generation"], peak["Peak X"], peak["Peak Y"]):
        xy = (x, y)
        if last_xy is not None and xy != last_xy:
            return int(round(update)), generation, x, y
        last_xy = xy
    return None


def add_gradient_pages(pdf: PdfPages, run_dir: str, data_dir: str) -> None:
    peak_path = write_peak_location_file(data_dir, run_dir)
    peak = parse_peak_location_file(peak_path)
    max_paths = dump_paths(data_dir, "max_res_grid")
    org_paths = dump_paths(data_dir, "org_loc")
    if not max_paths:
        return

    final_update, final_grid_path = max_paths[-1]
    average = parse_optional_data_file(os.path.join(data_dir, "average.dat"))
    run_final_update = average["Update"][-1] if average and "Update" in average else final_update
    final_grid = read_resource_grid(final_grid_path)
    final_org_path = nearest_dump_path(org_paths, final_update, prefer_before=True)
    final_orgs = parse_org_loc_file(final_org_path[1]) if final_org_path else {}
    final_peak = peak_xy_at_or_before(peak, final_update)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = plot_gradient_panel(
        ax,
        final_grid,
        f"Latest recorded resource gradient\nupdate {final_update}",
        orgs=final_orgs,
        peak_xy=final_peak,
    )
    if run_final_update > final_update:
        ax.text(
            0.02,
            0.02,
            f"Run continues to update {run_final_update:.0f}; no later grid dump is present.",
            transform=ax.transAxes,
            fontsize=8,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.75, "pad": 4},
        )
    fig.colorbar(image, ax=ax, label="resource value")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    change = first_peak_change(peak)
    if change:
        change_update, change_generation, _x, _y = change
        before = nearest_dump_path(max_paths, change_update - 1, prefer_before=True)
        after = nearest_dump_path(max_paths, change_update, prefer_before=False)
        title = f"Gradient transition around peak change\nupdate {change_update}, generation {change_generation:.4g}"
    else:
        before = max_paths[0]
        after = max_paths[-1]
        title = "No recorded peak shift in this run\nshowing initial vs final gradient"
    if before is None or after is None:
        return

    before_update, before_grid_path = before
    after_update, after_grid_path = after
    before_org_path = nearest_dump_path(org_paths, before_update, prefer_before=True)
    after_org_path = nearest_dump_path(org_paths, after_update, prefer_before=False)
    before_orgs = parse_org_loc_file(before_org_path[1]) if before_org_path else {}
    after_orgs = parse_org_loc_file(after_org_path[1]) if after_org_path else {}
    survivors = set(before_orgs) & set(after_orgs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    image = plot_gradient_panel(
        axes[0],
        read_resource_grid(before_grid_path),
        f"Before\nupdate {before_update}",
        orgs=before_orgs,
        survivor_ids=survivors,
        peak_xy=peak_xy_at_or_before(peak, before_update),
    )
    plot_gradient_panel(
        axes[1],
        read_resource_grid(after_grid_path),
        f"After\nupdate {after_update}",
        orgs=after_orgs,
        survivor_ids=survivors,
        peak_xy=peak_xy_at_or_before(peak, after_update),
    )
    fig.suptitle(f"{title}\nshared organism IDs across shown dumps: {len(survivors)}", fontsize=12)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="resource value", shrink=0.82)
    pdf.savefig(fig)
    plt.close(fig)


def add_movement_behavior_pages(pdf: PdfPages, run_dir: str, data_dir: str) -> None:
    series = movement_series(data_dir)
    peak_path = write_peak_location_file(data_dir, run_dir)
    peak = parse_peak_location_file(peak_path)
    if not series and not peak["Peak X"]:
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    for label, data in series:
        x_key = first_existing_key(data, ["Env X"])
        y_key = first_existing_key(data, ["Env Y"])
        if not x_key or not y_key:
            continue
        x = data[x_key]
        y = data[y_key]
        if not x or not y:
            continue
        ax.plot(x, y, marker="o", linewidth=1.7, markersize=4, label=label)
        ax.scatter([x[0]], [y[0]], marker="s", s=55)
        ax.scatter([x[-1]], [y[-1]], marker="*", s=110)
    if peak["Peak X"] and peak["Peak Y"]:
        unique_peak_positions = set(zip(peak["Peak X"], peak["Peak Y"]))
        ax.plot(
            peak["Peak X"],
            peak["Peak Y"],
            color="black",
            linewidth=2.0,
            linestyle=":",
            marker=".",
            markersize=3,
            alpha=0.8,
            label="resource peak",
        )
        ax.scatter([peak["Peak X"][0]], [peak["Peak Y"][0]], marker="s", s=70, color="black", zorder=6)
        ax.scatter([peak["Peak X"][-1]], [peak["Peak Y"][-1]], marker="*", s=140, color="black", zorder=6)
        ax.annotate(
            f"peak start\nu={peak['Update'][0]:.0f}, g={peak['Generation'][0]:.4g}",
            xy=(peak["Peak X"][0], peak["Peak Y"][0]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            color="black",
            arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 0.8},
        )
        ax.annotate(
            f"peak end\nu={peak['Update'][-1]:.0f}, g={peak['Generation'][-1]:.4g}",
            xy=(peak["Peak X"][-1], peak["Peak Y"][-1]),
            xytext=(8, -24),
            textcoords="offset points",
            fontsize=8,
            color="black",
            arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 0.8},
        )
        if len(unique_peak_positions) == 1:
            ax.text(
                0.02,
                0.02,
                "Recorded peak location is static.\nFor this run, grid dumps only show the center peak.",
                transform=ax.transAxes,
                fontsize=8,
                color="black",
                bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.75, "pad": 4},
            )
    ax.set_xlabel("Env X")
    ax.set_ylabel("Env Y")
    ax.set_title("Champion movement paths with resource peak trajectory")
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(11, 6))
    colors = ["#4c78a8", "#f58518"]
    handles = []
    for idx, (label, data) in enumerate(series):
        update_key = first_existing_key(data, ["Update", "update"])
        step_key = first_existing_key(data, ["StepDisp"])
        if not update_key or not step_key:
            continue
        color = colors[idx % len(colors)]
        line, = ax1.plot(data[update_key], data[step_key], marker="o", linewidth=1.8, color=color, label=f"{label} StepDisp")
        annotate_peak_point(ax1, data[update_key], data[step_key], f"{label} StepDisp", color)
        handles.append(line)
    ax1.set_xlabel("Update")
    ax1.set_ylabel("StepDisp")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    for idx, (label, data) in enumerate(series):
        update_key = first_existing_key(data, ["Update", "update"])
        east_key = first_existing_key(data, ["Easterly"])
        north_key = first_existing_key(data, ["Northerly"])
        if not update_key or not east_key or not north_key:
            continue
        net_disp = [math.hypot(e, n) for e, n in zip(data[east_key], data[north_key])]
        color = colors[idx % len(colors)]
        line, = ax2.plot(
            data[update_key],
            net_disp,
            linestyle="--",
            linewidth=1.4,
            color=color,
            alpha=0.8,
            label=f"{label} net displacement",
        )
        annotate_peak_point(ax2, data[update_key], net_disp, f"{label} net", color, xytext=(8, -28))
        handles.append(line)
    ax2.set_ylabel("sqrt(Easterly^2 + Northerly^2)")
    ax1.set_title("Champion displacement over time")
    ax1.legend(handles=handles, loc="best", fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def add_non_overlap_pages(pdf: PdfPages, data_dir: str, allow_live_fitness_fallback: bool) -> None:
    average = parse_data_file(os.path.join(data_dir, "average.dat"))
    count = parse_data_file(os.path.join(data_dir, "count.dat"))
    extrema = parse_optional_data_file(os.path.join(data_dir, "fitness_extrema.dat"))

    updates, avg_gen, avg_fit, max_fit, pop_size, births, deaths = align_on_update(
        average, count, extrema, allow_live_fitness_fallback
    )
    gen_rate = rates(avg_gen, updates)
    gens, avg_fit_g, max_fit_g, pop_g, births_g, deaths_g, pop_first_g, pop_last_g = aggregate_per_generation(
        avg_gen, avg_fit, max_fit, pop_size, births, deaths
    )

    net_g = [b - d for b, d in zip(births_g, deaths_g)]
    net_repro_rate = [n / p if p > 0 else float("nan") for n, p in zip(net_g, pop_g)]
    delta_n_bin = [last - first for first, last in zip(pop_first_g, pop_last_g)]
    max_err = max((abs(net_g[i] - delta_n_bin[i]) for i in range(len(net_g))), default=0.0)

    title_prefix = (
        "Non-overlapping diagnostic\n"
        f"max |sum(B-D) on logged rows - (N_last-N_first)| = {max_err:.4g}"
    )

    fig, ax1 = plt.subplots(figsize=(11, 6), constrained_layout=True)
    l1, = ax1.plot(updates, avg_gen, color="#4c78a8", linewidth=2.0, label="Average generation")
    ax1.set_xlabel("Update")
    ax1.set_ylabel("Average generation", color="#4c78a8")
    ax1.tick_params(axis="y", labelcolor="#4c78a8")
    ax1.grid(alpha=0.25)
    ax1b = ax1.twinx()
    l2, = ax1b.plot(updates, gen_rate, color="#f58518", linewidth=1.4, alpha=0.9, label="Generation rate")
    ax1b.set_ylabel("Generation rate (gen/update)", color="#f58518")
    ax1b.tick_params(axis="y", labelcolor="#f58518")
    ax1.legend(handles=[l1, l2], loc="upper left")
    ax1.set_title(f"{title_prefix}\nAverage generation and generation rate")
    pdf.savefig(fig)
    plt.close(fig)

    fig, ax2 = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax2.plot(gens, net_repro_rate, color="#7f3c8d", linewidth=2.0, label="Net reproductive rate")
    ax2.axhline(0.0, color="#999999", linewidth=1.0, alpha=0.6)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Per-capita (births - deaths) / mean N")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_title(f"{title_prefix}\nNet reproductive rate")
    pdf.savefig(fig)
    plt.close(fig)

    fig, ax3 = plt.subplots(figsize=(11, 6), constrained_layout=True)
    l4, = ax3.plot(gens, max_fit_g, color="#e45756", linewidth=2.0, label="Highest fitness")
    l4b, = ax3.plot(gens, avg_fit_g, color="#4c78a8", linewidth=1.6, alpha=0.9, label="Mean fitness")
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Fitness")
    ax3.grid(alpha=0.25)
    ax3.legend(handles=[l4, l4b], loc="upper left", fontsize=9)
    ax3.set_title(f"{title_prefix}\nFitness by generation")
    pdf.savefig(fig)
    plt.close(fig)

    fig, ax4 = plt.subplots(figsize=(11, 6), constrained_layout=True)
    x = list(range(len(gens)))
    bar_w = 0.45
    b = ax4.bar([xi - bar_w / 2 for xi in x], births_g, width=bar_w, color="#54a24b", alpha=0.8, label="Births/gen")
    d = ax4.bar([xi + bar_w / 2 for xi in x], deaths_g, width=bar_w, color="#e45756", alpha=0.8, label="Deaths/gen")
    ax4.set_xlabel("Generation")
    ax4.set_ylabel("Births / deaths")
    ax4.grid(alpha=0.25)
    tick_step = max(1, len(x) // 12)
    tick_positions = x[::tick_step]
    ax4.set_xticks(tick_positions)
    ax4.set_xticklabels([str(gens[i]) for i in tick_positions], rotation=0)
    ax4.legend(handles=[b, d], loc="upper right")
    ax4.set_title(f"{title_prefix}\nBirths and deaths by generation")

    pdf.savefig(fig)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate one PDF with each ecology_test diagnostic plot on its own page.")
    parser.add_argument("--run-dir", default=RUN_FOLDER, help="Run directory containing instset.cfg and data/")
    parser.add_argument("--data-subdir", default=DATA_SUBDIR, help="Data subdirectory under --run-dir")
    parser.add_argument("--spop-prefix", default=SPOP_PREFIX, help="SavePopulation prefix")
    parser.add_argument("--instset", default=None, help="Instruction set path; default: RUN_DIR/instset.cfg")
    parser.add_argument("--save", default=None, help="Output PDF; default: DATA_DIR/plots.pdf")
    parser.add_argument("--latest-spop", action="store_true", help="Use latest .spop instead of last-generation snapshot")
    parser.add_argument("--allow-live-fitness-fallback", action="store_true", help="Use live fitness if death-based extrema are unavailable")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    data_dir = os.path.join(run_dir, args.data_subdir)
    inst_path = os.path.abspath(args.instset) if args.instset else os.path.join(run_dir, "instset.cfg")
    out_pdf = os.path.abspath(args.save) if args.save else os.path.join(data_dir, "plots.pdf")

    if not os.path.isdir(data_dir):
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        return 1
    if not os.path.isfile(inst_path):
        print(f"Error: instset not found: {inst_path}", file=sys.stderr)
        return 1

    with PdfPages(out_pdf) as pdf:
        add_repro_pages(pdf, run_dir, data_dir, inst_path, args.spop_prefix, not args.latest_spop)
        add_instruction_usage_pages(pdf, run_dir, data_dir, inst_path, args.spop_prefix, not args.latest_spop)
        add_movement_behavior_pages(pdf, run_dir, data_dir)
        add_gradient_pages(pdf, run_dir, data_dir)
        add_non_overlap_pages(pdf, data_dir, args.allow_live_fitness_fallback)

    print(f"Wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
