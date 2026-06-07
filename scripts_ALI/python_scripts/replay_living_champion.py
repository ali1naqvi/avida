# python3 replay_living_champion.py

from __future__ import annotations

# Edit these values instead of passing the run folder as a command-line arg.
# WORK_RUN_DIR can be absolute, or relative to the repo root.
WORK_RUN_DIR = "work/ecology_test_relative_cost_medium_run_1"

# "persistence" = replay a representative genotype from the lineage label that
#                 appears across the most saved population snapshots.
#                 "persistance" is accepted as a spelling alias.
# "energy"      = replay the genotype with the highest Stored Energy in
#                 lifetime_champion.dat; falls back to highest .spop merit.
REPLAY_SELECTION = "persistence"

import argparse
import glob
import math
import os
import re
import shutil
import sys
import tempfile

# Keep matplotlib from trying to write into ~/.matplotlib when this is run from
# the Codex/app sandbox or a machine with a read-only home cache directory.
_mpl_cache = os.path.join(tempfile.gettempdir(), "avida_matplotlib_cache")
os.makedirs(_mpl_cache, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cache)

import visualize_highest_fitness as vhf


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


MOVE_INSTRUCTIONS = {
    "move",
    "rotate-left-one",
    "rotate-right-one",
    "rotate-uphill",
    "rotate-to-unoccupied-cell",
}


def clean_header_key(key: str) -> str:
    """Convert Avida's padded header labels to stable names."""
    return " ".join(key.replace(".", " ").split())


def read_org_header(org_path: str) -> dict:
    """Parse ``# key: value`` header lines from a champion .org file."""
    meta: dict = {}
    with open(org_path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip()
            if not line.startswith("#"):
                if line:
                    break
                continue
            stripped = line.lstrip("#").strip()
            if ":" in stripped:
                key, val = stripped.split(":", 1)
                meta[clean_header_key(key.strip())] = val.strip()
    return meta


def read_champion_dat_last_row(dat_path: str) -> dict:
    """Return the final row of champion.dat as ``{column_name: value}``."""
    if not os.path.exists(dat_path):
        return {}
    columns, rows = vhf.parse_dominant_dat(dat_path)
    if not rows:
        return {}
    last = rows[-1]
    return {col: last[i] for i, col in enumerate(columns) if i < len(last)}


def read_champion_dat_rows(dat_path: str) -> list[dict]:
    """Return champion/lifetime rows as ``{column_name: value}`` dictionaries."""
    if not os.path.exists(dat_path):
        return []
    columns, rows = vhf.parse_dominant_dat(dat_path)
    out: list[dict] = []
    for row in rows:
        out.append({col: row[i] for i, col in enumerate(columns) if i < len(row)})
    return out


def pick_highest_stored_energy(data_dir: str) -> tuple[str | None, dict]:
    """Pick the highest Stored Energy record from lifetime_champion.dat."""
    lifetime_dat = os.path.join(data_dir, "lifetime_champion.dat")
    rows = read_champion_dat_rows(lifetime_dat)
    if not rows:
        return None, {}

    def stored_energy(row: dict) -> float:
        try:
            return float(row.get("Stored Energy", "nan"))
        except ValueError:
            return float("nan")

    viable = [row for row in rows if not math.isnan(stored_energy(row))]
    if not viable:
        return None, {}
    best = max(
        viable,
        key=lambda row: (
            stored_energy(row),
            int(float(row.get("Update", 0))),
            int(float(row.get("Lifetime Fitness", 0))),
        ),
    )
    genotype_name = best.get("Genotype Name", "")
    candidates = []
    if genotype_name:
        candidates.append(os.path.join(data_dir, "archive", f"{genotype_name}.org"))
    # If the chosen row is the final saved champion, lifetime_champion.org is exact.
    if best is rows[-1]:
        candidates.append(os.path.join(data_dir, "lifetime_champion.org"))
    candidates.extend([
        os.path.join(data_dir, "champion.org"),
        os.path.join(data_dir, "lifetime_champion.org"),
    ])
    org_path = next((path for path in candidates if path and os.path.exists(path)), None)
    meta = {
        "source": "highest Stored Energy from lifetime_champion.dat",
        "stored_energy": stored_energy(best),
        "update": best.get("Update"),
        "lifetime_fitness": best.get("Lifetime Fitness"),
        "genotype_name": genotype_name,
        "genotype_id": best.get("Genotype ID"),
        "num_divides": best.get("Num Divides"),
    }
    if org_path and genotype_name and os.path.basename(org_path) != f"{genotype_name}.org":
        meta["org_resolution_note"] = (
            f"archive/{genotype_name}.org not found; using {os.path.basename(org_path)}"
        )
    return org_path, meta


def _glob_spop_updates(data_dir: str, prefix: str) -> list[tuple[str, int]]:
    pattern = os.path.join(data_dir, f"{prefix}-*.spop")
    out: list[tuple[str, int]] = []
    for path in glob.glob(pattern):
        m = re.search(r"-(-?\d+)\.spop$", path)
        if m:
            out.append((path, int(m.group(1))))
    return out


def find_latest_spop(data_dir: str, prefix: str) -> tuple[str | None, int]:
    """Return ``(path, update)`` for the highest-update ``prefix-<u>.spop`` under data_dir."""
    spops = _glob_spop_updates(data_dir, prefix)
    if not spops:
        return None, -1
    best_path, best_u = max(spops, key=lambda t: t[1])
    return best_path, best_u


def pick_spop_last_generation(
    data_dir: str,
    prefix: str,
    average_dat_path: str,
) -> tuple[str | None, int, dict]:
    """Prefer the snapshot with the greatest mean ``Generation`` (from average.dat), then latest update.

    Only considers ``.spop`` updates **≤** the final ``Update`` in ``average.dat``
    when that file exists, so we stay aligned with the logged end of the run.
    """
    meta: dict = {}
    spops = _glob_spop_updates(data_dir, prefix)
    if not spops:
        return None, -1, meta

    meta["spop_candidates"] = len(spops)

    if not os.path.exists(average_dat_path):
        best_path, best_u = max(spops, key=lambda t: t[1])
        meta["snapshot_rule"] = "latest_update (no average.dat)"
        return best_path, best_u, meta

    columns, rows = vhf.parse_dominant_dat(average_dat_path)
    idx_u = vhf._dominant_col_index(columns, "Update")
    idx_g = vhf._dominant_col_index(columns, "Generation")
    if idx_u is None or idx_g is None or not rows:
        best_path, best_u = max(spops, key=lambda t: t[1])
        meta["snapshot_rule"] = "latest_update (average.dat missing Update/Generation)"
        return best_path, best_u, meta

    def row_pair(parts: list[str]) -> tuple[int, float] | None:
        if idx_u >= len(parts) or idx_g >= len(parts):
            return None
        try:
            return int(float(parts[idx_u])), float(parts[idx_g])
        except ValueError:
            return None

    last_pair = row_pair(rows[-1])
    if last_pair:
        meta["average_dat_final_update"], meta["average_dat_final_generation"] = last_pair
    u_final = last_pair[0] if last_pair else None

    timeline: list[tuple[int, float]] = []
    for parts in rows:
        p = row_pair(parts)
        if p:
            timeline.append(p)
    timeline.sort(key=lambda t: t[0])

    def mean_generation_at_or_before(u_bound: int) -> float | None:
        best_g: float | None = None
        for uu, gg in timeline:
            if uu <= u_bound:
                best_g = gg
            else:
                break
        return best_g

    capped = [(p, u) for p, u in spops if u_final is None or u <= u_final]
    if not capped:
        capped = spops
        meta["snapshot_note"] = (
            "all .spop updates are after final average.dat Update; using full spop list"
        )

    scored: list[tuple[float, int, str]] = []
    for path, u_s in capped:
        g_s = mean_generation_at_or_before(u_s)
        if g_s is not None:
            scored.append((g_s, u_s, path))

    if not scored:
        best_path, best_u = max(spops, key=lambda t: t[1])
        meta["snapshot_rule"] = "latest_update (could not match spop updates to average.dat)"
        return best_path, best_u, meta

    best_g, best_u, best_path = max(scored, key=lambda t: (t[0], t[1]))
    meta["snapshot_rule"] = (
        "last_generation (max mean Generation ≤ final average.dat Update, then max spop Update)"
    )
    meta["mean_generation_at_snapshot"] = best_g
    return best_path, best_u, meta


def _inst_set_token_index(line_parts: list[str]) -> int | None:
    for k, p in enumerate(line_parts):
        if p.endswith(".cfg") or p == "(default)":
            return k
        # INSTSET name may be a bare token (e.g. ``instset``) after hw_type.
        if k >= 10 and p == "instset":
            return k
    return None


def parse_spop_rows(spop_path: str) -> list[dict]:
    """Parse data rows from a structured ``.spop`` file."""
    out: list[dict] = []
    with open(spop_path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 15:
                continue
            try:
                gid = int(parts[0])
                num_live = int(parts[4])
                total_units = int(parts[5])
                length = int(parts[6])
                merit = float(parts[7])
                gest = float(parts[8])
                fitness = float(parts[9])
                update_born = int(float(parts[11]))
                depth = int(float(parts[13]))
            except (ValueError, IndexError):
                continue
            i = _inst_set_token_index(parts)
            if i is None or i + 3 >= len(parts):
                continue
            sequence = parts[i + 1]
            cells = parts[i + 2]
            gest_off = parts[i + 3]
            lineage = parts[i + 4] if i + 4 < len(parts) else ""
            out.append({
                "id": gid,
                "num_live": num_live,
                "total_units": total_units,
                "length": length,
                "merit": merit,
                "gest_time": gest,
                "fitness": fitness,
                "update_born": update_born,
                "depth": depth,
                "sequence": sequence,
                "cells": cells,
                "gest_offset": gest_off,
                "lineage": lineage,
            })
    return out


def lineage_labels(row: dict) -> list[str]:
    """Return lineage labels represented by a row, preserving per-cell labels."""
    raw = str(row.get("lineage", "")).strip()
    labels = [label for label in raw.split(",") if label != ""]
    if labels:
        return labels
    return ["unknown"]


def parse_instset_symbols(instset_path: str) -> dict[str, str]:
    """Map Avida genome symbols (a, b, c, ...) to instruction names."""
    names: list[str] = []
    with open(instset_path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "INST":
                names.append(parts[1])
    symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return {symbols[i]: name for i, name in enumerate(names) if i < len(symbols)}


def write_org_from_sequence(sequence: str, instset_path: str, org_path: str, header: list[str]) -> None:
    symbol_to_inst = parse_instset_symbols(instset_path)
    missing = sorted({symbol for symbol in sequence if symbol not in symbol_to_inst})
    if missing:
        raise ValueError(
            f"Cannot decode genome symbols {missing}; {instset_path} defines "
            f"{len(symbol_to_inst)} instructions."
        )
    with open(org_path, "w", encoding="utf-8") as handle:
        for line in header:
            handle.write(f"# {line}\n")
        handle.write("\n")
        for symbol in sequence:
            handle.write(f"{symbol_to_inst[symbol]}\n")


def safe_filename_token(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def pick_best_spop_row(rows: list[dict]) -> dict | None:
    """Highest fitness among genotypes with at least one living organism."""
    alive = [r for r in rows if r["num_live"] > 0]
    if not alive:
        return None
    return max(alive, key=lambda r: (r["fitness"], r["num_live"], r["id"]))


def pick_persistent_lineage_spop(
    data_dir: str,
    prefix: str,
) -> tuple[dict | None, dict]:
    """Pick a replay row from the lineage label present across the most snapshots."""
    spops = _glob_spop_updates(data_dir, prefix)
    if not spops:
        return None, {}

    lineage_stats: dict[str, dict] = {}
    rows_by_update: list[tuple[int, dict]] = []
    for spop_path, update in spops:
        if update < 0:
            continue
        rows = parse_spop_rows(spop_path)
        seen_this_snapshot: set[str] = set()
        for row in rows:
            if row["num_live"] <= 0:
                continue
            rows_by_update.append((update, row))
            labels = lineage_labels(row)
            per_label_live = max(1, row["num_live"] // max(1, len(labels)))
            for label in labels:
                stat = lineage_stats.setdefault(
                    label,
                    {"snapshots": 0, "total_live": 0, "first_update": update, "last_update": update},
                )
                stat["total_live"] += per_label_live
                stat["first_update"] = min(stat["first_update"], update)
                stat["last_update"] = max(stat["last_update"], update)
                seen_this_snapshot.add(label)
        for label in seen_this_snapshot:
            lineage_stats[label]["snapshots"] += 1

    if not lineage_stats:
        return None, {}

    best_label, best_stat = max(
        lineage_stats.items(),
        key=lambda item: (
            item[1]["snapshots"],
            item[1]["last_update"] - item[1]["first_update"],
            item[1]["total_live"],
            item[1]["last_update"],
            item[0],
        ),
    )
    candidates = [
        (update, row)
        for update, row in rows_by_update
        if best_label in lineage_labels(row)
    ]
    if not candidates:
        return None, {}
    best_update, best_row = max(
        candidates,
        key=lambda item: (item[0], item[1]["num_live"], item[1]["fitness"], item[1]["id"]),
    )
    return best_row, {
        "source": "most persistent lineage from detail-*.spop",
        "lineage_label": best_label,
        "lineage_snapshots": best_stat["snapshots"],
        "lineage_total_live": best_stat["total_live"],
        "lineage_first_update": best_stat["first_update"],
        "lineage_last_update": best_stat["last_update"],
        "spop_update": best_update,
        "genotype_id": best_row["id"],
        "num_live": best_row["num_live"],
        "avg_live_fitness": best_row["fitness"],
        "avg_merit": best_row["merit"],
        "genome_length": best_row["length"],
    }


def pick_highest_merit_spop(data_dir: str, prefix: str) -> tuple[dict | None, dict]:
    """Fallback energy proxy: highest average merit among living .spop genotypes."""
    spops = _glob_spop_updates(data_dir, prefix)
    best: tuple[float, int, int, dict] | None = None
    for spop_path, update in spops:
        if update < 0:
            continue
        for row in parse_spop_rows(spop_path):
            if row["num_live"] <= 0:
                continue
            candidate = (row["merit"], update, row["num_live"], row)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    if best is None:
        return None, {}
    merit, update, _num_live, row = best
    return row, {
        "source": "highest .spop average merit (Stored Energy fallback)",
        "spop_update": update,
        "genotype_id": row["id"],
        "num_live": row["num_live"],
        "avg_merit": merit,
        "avg_live_fitness": row["fitness"],
        "genome_length": row["length"],
    }


def resolve_org_by_genome_length(archive_dir: str, target_len: int) -> tuple[str | None, list[str]]:
    """Return ``(path, candidates)`` — unique ``.org`` with matching instruction count, else ambiguous."""
    candidates: list[str] = []
    if not os.path.isdir(archive_dir):
        return None, []
    for name in os.listdir(archive_dir):
        if not name.endswith(".org"):
            continue
        path = os.path.join(archive_dir, name)
        try:
            glen = len(vhf.read_org_genome(path))
        except OSError:
            continue
        if glen == target_len:
            candidates.append(path)
    candidates.sort()
    if len(candidates) == 1:
        return candidates[0], candidates
    return None, candidates


def find_avida_binary(user_path: str | None) -> str | None:
    candidates = [user_path] if user_path else []
    candidates += [
        os.path.join(REPO_ROOT, "cbuild", "bin", "avida"),
        os.path.join(REPO_ROOT, "cbuild", "work", "avida"),
        os.path.join(REPO_ROOT, "work", "avida"),
        "../../bin/avida",
        "../avida",
        "avida",
    ]
    for path in candidates:
        if not path:
            continue
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and os.access(abs_path, os.X_OK):
            return abs_path
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a genotype from the configured WORK_RUN_DIR. By default, "
            "choose either the most persistent lineage or highest-energy record "
            "using REPLAY_SELECTION at the top of this file."
        )
    )
    parser.add_argument(
        "--run-dir",
        default=WORK_RUN_DIR,
        help="Avida run directory. Defaults to WORK_RUN_DIR at the top of this file.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("persistence", "persistance", "energy", "live_fitness"),
        default=REPLAY_SELECTION,
        help="Replay target selector. Defaults to REPLAY_SELECTION at the top of this file.",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--spop-prefix",
        default="detail",
        help="SavePopulation base name (files: DATA_DIR/PREFIX-<update>.spop).",
    )
    parser.add_argument(
        "--spop",
        default=None,
        help="Use this .spop file instead of auto-selecting from PREFIX-*.spop.",
    )
    parser.add_argument(
        "--average-dat",
        default=None,
        help="average.dat path (default: DATA_DIR/average.dat); used to pick .spop by mean Generation.",
    )
    parser.add_argument(
        "--latest-update-snapshot",
        action="store_true",
        help="Pick PREFIX-*.spop with the highest <update> in the filename (ignore average.dat).",
    )
    parser.add_argument(
        "--org",
        default=None,
        help="Explicit .org to replay (skips .spop / champion resolution).",
    )
    parser.add_argument("--updates", type=int, default=None,
                        help="Replay length in updates (default: one lifespan).")
    parser.add_argument(
        "--reproduction",
        choices=("off", "on"),
        default="on",
        help=(
            "Whether replay should allow reproduction. 'on' keeps semel offspring "
            "alive after the focal individual dies; mutations remain disabled. "
            "Use 'off' for the old one-organism trajectory view."
        ),
    )
    parser.add_argument("--save", default=None, help="Save animation as .gif or .mp4.")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--avida", default=None, help="Path to avida binary.")
    parser.add_argument("--summary-only", action="store_true",
                        help="Run the replay and print a path summary; do not open an animation window.")
    args = parser.parse_args()
    if args.selection_mode == "persistance":
        args.selection_mode = "persistence"

    run_dir = args.run_dir
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(REPO_ROOT, run_dir)
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        print(f"ERROR: run directory not found: {run_dir}", file=sys.stderr)
        return 1
    os.chdir(run_dir)

    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(run_dir, args.data_dir)
    args.data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(args.data_dir):
        print(f"ERROR: data directory not found: {args.data_dir}", file=sys.stderr)
        return 1

    snapshot_meta: dict = {}
    source = ""
    generated_org_path: str | None = None

    if args.org:
        champion_org = args.org
        if not os.path.isabs(champion_org) and not os.path.exists(champion_org):
            champion_org = os.path.join(args.data_dir, champion_org)
        if not os.path.exists(champion_org):
            print(f"ERROR: {champion_org} not found.", file=sys.stderr)
            return 1
        source = "explicit --org"
    elif args.selection_mode == "energy":
        champion_org, snapshot_meta = pick_highest_stored_energy(args.data_dir)
        if champion_org is None:
            row, snapshot_meta = pick_highest_merit_spop(args.data_dir, args.spop_prefix)
            if row is None:
                print("ERROR: no Stored Energy records or living .spop rows found.", file=sys.stderr)
                return 1
            generated_org_path = os.path.join(tempfile.gettempdir(), f"avida_energy_{row['id']}.org")
            write_org_from_sequence(
                row["sequence"],
                os.path.join(run_dir, "instset.cfg"),
                generated_org_path,
                [
                    "Generated by replay_living_champion.py",
                    f"Selection: {snapshot_meta.get('source')}",
                    f"Genotype ID: {row['id']}",
                ],
            )
            champion_org = generated_org_path
        source = snapshot_meta.get("source", "energy selection")
    elif args.selection_mode == "persistence":
        row, snapshot_meta = pick_persistent_lineage_spop(args.data_dir, args.spop_prefix)
        if row is None:
            print("ERROR: no living lineage rows found in detail-*.spop.", file=sys.stderr)
            return 1
        generated_org_path = os.path.join(
            tempfile.gettempdir(),
            f"avida_persistent_lineage_{safe_filename_token(snapshot_meta.get('lineage_label', 'unknown'))}_{row['id']}.org",
        )
        write_org_from_sequence(
            row["sequence"],
            os.path.join(run_dir, "instset.cfg"),
            generated_org_path,
            [
                "Generated by replay_living_champion.py",
                f"Selection: {snapshot_meta.get('source')}",
                f"Lineage Label: {snapshot_meta.get('lineage_label')}",
                f"Genotype ID: {row['id']}",
                f"Snapshot Update: {snapshot_meta.get('spop_update')}",
            ],
        )
        champion_org = generated_org_path
        source = snapshot_meta.get("source", "persistence selection")
    else:
        spop_path = args.spop
        spop_update = -1
        spop_meta_extra: dict = {}
        if spop_path:
            if not os.path.isabs(spop_path) and not os.path.exists(spop_path):
                spop_path = os.path.join(args.data_dir, spop_path)
        else:
            if args.latest_update_snapshot:
                spop_path, spop_update = find_latest_spop(args.data_dir, args.spop_prefix)
            else:
                avg_path = args.average_dat or os.path.join(args.data_dir, "average.dat")
                if args.average_dat and not os.path.isabs(avg_path) and not os.path.exists(avg_path):
                    avg_path = os.path.join(args.data_dir, avg_path)
                spop_path, spop_update, spop_meta_extra = pick_spop_last_generation(
                    args.data_dir, args.spop_prefix, avg_path,
                )

        champion_fallback = os.path.join(args.data_dir, "champion.org")
        champion_dat = os.path.join(args.data_dir, "champion.dat")
        archive_dir = os.path.join(args.data_dir, "archive")

        champion_org: str | None = None

        if spop_path and os.path.exists(spop_path):
            m_u = re.search(r"-(\d+)\.spop$", spop_path)
            inferred_u = int(m_u.group(1)) if m_u else None
            rows = parse_spop_rows(spop_path)
            best = pick_best_spop_row(rows)
            if best:
                resolved, cands = resolve_org_by_genome_length(archive_dir, best["length"])
                snapshot_meta = {
                    **spop_meta_extra,
                    "spop_file": spop_path,
                    "spop_update": spop_update if spop_update >= 0 else inferred_u,
                    "genotype_id": str(best["id"]),
                    "num_live": best["num_live"],
                    "avg_live_fitness": best["fitness"],
                    "genome_length": best["length"],
                    "ambiguous_matches": len(cands),
                }
                if resolved:
                    champion_org = resolved
                    source = (
                        f"spop (max avg live fitness @ generation-picked snapshot) → "
                        f"{os.path.basename(spop_path)}"
                    )
                elif len(cands) > 1:
                    print(
                        f"WARNING: {len(cands)} archive/*.org files have length {best['length']}; "
                        "cannot pick unambiguously. Falling back to champion.org if present.",
                        file=sys.stderr,
                    )
                elif len(cands) == 0:
                    print(
                        f"WARNING: no archive/*.org with genome length {best['length']}. "
                        "Falling back to champion.org if present.",
                        file=sys.stderr,
                    )

        if champion_org is None and os.path.exists(champion_fallback):
            champion_org = champion_fallback
            source = "fallback champion.org (PrintChampionGenotype)"
            dat_row = read_champion_dat_last_row(champion_dat)
            if dat_row:
                snapshot_meta = {
                    "champion_update": dat_row.get("Update"),
                    "champion_fitness": dat_row.get("Champion Fitness"),
                    "genotype_name": dat_row.get("Genotype Name"),
                    "num_divides": dat_row.get("Num Divides"),
                }

        if champion_org is None:
            print(
                "ERROR: could not resolve a genome .org (no unambiguous spop match and no champion.org).",
                file=sys.stderr,
            )
            print(
                "Ensure SavePopulation produced .spop files and archive genotypes, "
                "or add champion.org via PrintChampionGenotype.",
                file=sys.stderr,
            )
            return 1

    header = read_org_header(champion_org)
    genome = vhf.read_org_genome(champion_org)
    move_count = sum(1 for inst in genome if inst in MOVE_INSTRUCTIONS)

    print(f"Replaying {champion_org}")
    print(f"  run directory        : {run_dir}")
    print(f"  selection mode       : {args.selection_mode}")
    print(f"  source               : {source}")
    if snapshot_meta:
        for k, v in snapshot_meta.items():
            if k == "source":
                continue
            if v not in (None, ""):
                print(f"  {k:20s} : {v}")
    if "Update Output" in header:
        print(f"  saved at update      : {header['Update Output']}")
    if "Fitness" in header:
        print(f"  TestCPU fitness      : {header['Fitness']}  (empty-world recompute)")
    print(f"  genome length        : {len(genome)}")
    print(f"  movement insts       : {move_count}")
    if move_count == 0:
        print("  NOTE: no movement instructions in the genome; any positional change you")
        print("        see in the replay is offspring placement, not active chemotaxis.")
    print(f"  reproduction replay : {args.reproduction}")

    world_x, world_y = vhf.read_world_size()
    env_world_x, env_world_y = vhf.read_env_world_size()
    if args.updates is not None:
        num_updates = max(1, args.updates)
    else:
        num_updates = vhf.replay_updates_for_lifespan(os.path.abspath("avida.cfg"), len(genome))
    print(f"  replay updates       : {num_updates}")

    avida_bin = find_avida_binary(args.avida)
    if not avida_bin:
        print("ERROR: cannot find avida binary. Try --avida ../../bin/avida", file=sys.stderr)
        return 1

    replay_dir = os.path.join(os.getcwd(), "_replay_living_champion")
    if os.path.exists(replay_dir):
        shutil.rmtree(replay_dir)
    vhf.setup_replay(
        champion_org,
        replay_dir,
        world_x,
        world_y,
        num_updates,
        allow_reproduction=(args.reproduction == "on"),
    )
    vhf.run_avida(replay_dir, avida_bin, timeout_sec=max(120.0, 0.05 * num_updates + 30.0))

    trajectory = vhf.parse_org_loc_files(replay_dir)
    if not trajectory:
        print("ERROR: replay produced no trajectory.", file=sys.stderr)
        shutil.rmtree(replay_dir, ignore_errors=True)
        if generated_org_path and os.path.exists(generated_org_path):
            os.remove(generated_org_path)
        return 1

    max_live_orgs = max((len(orgs) for _, orgs in trajectory), default=0)
    print(f"  max live orgs   : {max_live_orgs}")

    if args.reproduction == "on":
        focal_id = None
        for _, orgs in trajectory:
            if orgs:
                focal_id = orgs[0][0]
                break
        if focal_id is not None:
            truncated = []
            reproduction_frame = None
            for update, orgs in trajectory:
                truncated.append((update, orgs))
                ids = {org[0] for org in orgs}
                if focal_id not in ids and orgs:
                    reproduction_frame = update
                    break
            if reproduction_frame is not None:
                trajectory = truncated
                print(
                    f"  reproduction frame  : update={reproduction_frame} "
                    "(parent gone; offspring shown; later offspring execution hidden)"
                )
            else:
                print("  reproduction frame  : not observed in this replay")

    res_grid = vhf.load_resource_grid(replay_dir, env_world_x, env_world_y)

    first = None
    for u, orgs in trajectory:
        if orgs:
            first = (u, orgs[0])
            break
    last = None
    for u, orgs in reversed(trajectory):
        if orgs:
            last = (u, orgs[-1])
            break
    if first is not None and last is not None:
        _, fx, fy = first[1]
        _, lx, ly = last[1]
        print(f"  first frame     : update={first[0]} pos=({fx}, {fy})")
        print(f"  last  frame     : update={last[0]} pos=({lx}, {ly})")

    org_display = (
        snapshot_meta.get("genotype_name")
        or snapshot_meta.get("genotype_id")
        or os.path.basename(champion_org).replace(".org", "")
    )
    if not args.summary_only:
        vhf.visualize(
            trajectory, res_grid, env_world_x, env_world_y,
            f"highest live fitness (last-gen snapshot) — {org_display}",
            save_path=args.save, fps=args.fps,
            show_gradient_layer_labels=True,
        )

    shutil.rmtree(replay_dir, ignore_errors=True)
    if generated_org_path and os.path.exists(generated_org_path):
        os.remove(generated_org_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
