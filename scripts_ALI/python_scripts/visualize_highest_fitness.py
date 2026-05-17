#!/usr/bin/env python3
"""
Visualize a high-fitness organism's movement through the environment

This script:
1. Picks a genotype from dominant.dat — by default the row with the highest
   recorded fitness (max of average vs max-fitness columns for that snapshot).
   Use --selection last to use the final dominant instead.
2. Places it alone in a fresh copy of the environment (same death rules as
   avida.cfg: no artificial AGE_LIMIT extension).
3. Runs avida for a number of updates (see --updates / --strict-lifespan below).
4. Animates its path overlaid on the food gradient

Why runs can look "weird":
  - DEATH_METHOD 2 + AGE_LIMIT 1 ⇒ only ~genome_length total instructions of life
    (often ~100 updates). By default this script now replays only that true
    lifespan; use --min-observation-updates or --updates to force a longer clip.
  - BIRTH_METHOD 4 (mass action) + reproduction ⇒ brief multi-org states; replay
    forces local empty-cell birth + ALLOW_PARENT 0 to keep a single moving body.

Usage:
    cd chemotaxis
    python3 visualize_dominant.py
    python3 visualize_dominant.py --save path.gif
    python3 visualize_dominant.py --org data/archive/125-aabev.org
    python3 visualize_dominant.py --selection last --updates 800
"""

from __future__ import annotations

import os
import sys
import glob
import shutil
import argparse
import subprocess
import tempfile
import numpy as np
import math
from typing import List

try:
    import champion_utils as cu
except ImportError:
    cu = None  # Champion selection will be disabled if the helper isn't on path.

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Circle
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    print("ERROR: matplotlib is required.  pip3 install matplotlib")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Step 1: Choose organism by highest fitness (not dominance)
# ---------------------------------------------------------------------------

def _strip_cfg_line(raw: str) -> str:
    if "#" in raw:
        raw = raw.split("#", 1)[0]
    return raw.strip()


def parse_dominant_dat(dom_file: str):
    """Parse dominant.dat header comments and numeric rows."""
    columns: List[str] = []
    rows: List[List[str]] = []
    if not os.path.exists(dom_file):
        return columns, rows
    with open(dom_file, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#") and ":" in line:
                left, right = line.split(":", 1)
                left = left.replace("#", "").strip()
                if left.isdigit():
                    columns.append(right.strip())
                continue
            if line.startswith("#"):
                continue
            parts = line.split()
            if parts:
                rows.append(parts)
    return columns, rows


def _dominant_col_index(columns: List[str], needle: str) -> int | None:
    nl = needle.lower()
    for i, name in enumerate(columns):
        if nl in name.lower():
            return i
    return None


def find_org_from_dominant_dat(selection: str = "max_fitness", dom_file: str = "data/dominant.dat"):
    """Return (org_path or None, meta dict).

    selection:
      - "max_fitness": row with highest max(ave_fitness, max_fitness); tie → later Update.
      - "last": final data row (legacy behavior).
    """
    columns, rows = parse_dominant_dat(dom_file)
    if not rows:
        return None, {}

    idx_u = _dominant_col_index(columns, "Update")
    idx_af = _dominant_col_index(columns, "Average Fitness of the Dominant")
    idx_mf = _dominant_col_index(columns, "Max Fitness")
    idx_nm = _dominant_col_index(columns, "Name of the Dominant")
    if idx_u is None:
        idx_u = 0
    if idx_af is None:
        idx_af = 3
    if idx_mf is None:
        idx_mf = 13
    if idx_nm is None:
        idx_nm = len(rows[0]) - 1

    def row_meta(parts: List[str]):
        name = parts[idx_nm] if idx_nm < len(parts) else parts[-1]
        try:
            u = int(float(parts[idx_u]))
            af = float(parts[idx_af])
            mf = float(parts[idx_mf])
        except (ValueError, IndexError):
            return None
        return {"update": u, "ave_fitness": af, "max_fitness": mf, "genotype_name": name}

    if selection == "last":
        parts = rows[-1]
        meta = row_meta(parts)
        if not meta:
            return None, {}
        name = meta["genotype_name"]
        path = f"data/archive/{name}.org"
        return (path if os.path.exists(path) else None), meta

    # max_fitness: best score = max(ave, maxfit) per row; tie-break by higher Update.
    best: tuple[float, int, List[str]] | None = None
    for parts in rows:
        m = row_meta(parts)
        if not m:
            continue
        score = max(m["ave_fitness"], m["max_fitness"])
        cand = (score, m["update"], parts)
        if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] >= best[1]):
            best = cand

    if best is None:
        return None, {}
    parts = best[2]
    meta = row_meta(parts)
    name = meta["genotype_name"]
    path = f"data/archive/{name}.org"
    return (path if os.path.exists(path) else None), meta


def parse_fitness_extrema(path: str) -> List[dict]:
    """Parse fitness_extrema.dat from PrintData.

    Expected columns: update, ave_fitness, max_fitness
    """
    if not os.path.exists(path):
        return []
    cols = []
    rows = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                if ":" in line:
                    left, right = line.split(":", 1)
                    left = left.replace("#", "").strip()
                    if left.isdigit():
                        cols.append(right.strip())
                continue
            parts = line.split()
            if parts:
                rows.append(parts)
    if not rows:
        return []
    # PrintData uses comma by default, but can also be whitespace depending on format.
    # If we see commas, split again.
    if len(rows[0]) == 1 and "," in rows[0][0]:
        rows = [r[0].split(",") for r in rows]
    out = []
    for r in rows:
        try:
            u = int(float(r[0]))
            ave = float(r[1])
            mx = float(r[2])
        except (ValueError, IndexError):
            continue
        out.append({"update": u, "ave_fitness": ave, "max_fitness": mx})
    return out


def parse_champion_dat(
    path: str,
    *,
    min_organism_generation: int | None = None,
    min_update: int | None = None,
) -> dict:
    """Return an exact champion record written by PrintChampionGenotype.

    By default this is the final row in champion.dat (global all-time live champion).

    If ``min_update`` is set and positive, only rows whose ``Update`` column is
    greater than or equal to that value are eligible before any generation
    filter. If no row qualifies, all rows are used and
    ``champion_update_filter_relaxed`` is set True.

    If min_organism_generation is set, only rows whose ``Generation`` field
    (organism lineage depth from cPhenotype) is *greater than* that value are
    eligible. The last eligible row is used (best fitness among those records,
    since champion.dat rows are chronological new highs). If no row qualifies,
    the final row is still used and ``champion_generation_filter_relaxed`` is
    set True in the returned dict.
    """
    columns, rows = parse_dominant_dat(path)
    if not rows:
        return {}

    def find_col(needle: str) -> int | None:
        nl = needle.lower()
        for i, name in enumerate(columns):
            if nl in name.lower():
                return i
        return None

    idx_update = find_col("Update") or 0
    idx_fitness = find_col("Champion Fitness")
    idx_org = find_col("Organism ID")
    idx_cell = find_col("Cell ID")
    idx_generation = find_col("Generation")
    idx_age = find_col("Age")
    idx_divides = find_col("Num Divides")
    idx_merit = find_col("Merit")
    idx_gestation = find_col("Gestation Time")
    idx_genotype = find_col("Genotype ID")
    idx_name = find_col("Genotype Name")

    def row_update(parts_row: List[str]) -> int | None:
        if idx_update is None or idx_update >= len(parts_row):
            return None
        try:
            return int(float(parts_row[idx_update]))
        except ValueError:
            return None

    def row_generation(parts_row: List[str]) -> int | None:
        if idx_generation is None or idx_generation >= len(parts_row):
            return None
        try:
            return int(float(parts_row[idx_generation]))
        except ValueError:
            return None

    row_candidates: List[List[str]] = list(rows)
    update_relaxed = False
    mu = 0 if min_update is None else int(min_update)
    if mu > 0:
        filtered_u: List[List[str]] = []
        for r in row_candidates:
            u = row_update(r)
            if u is not None and u >= mu:
                filtered_u.append(r)
        if filtered_u:
            row_candidates = filtered_u
        else:
            update_relaxed = True

    parts = row_candidates[-1]
    filter_relaxed = False
    if min_organism_generation is not None and idx_generation is not None:
        filtered: List[List[str]] = []
        for r in row_candidates:
            g = row_generation(r)
            if g is not None and g > min_organism_generation:
                filtered.append(r)
        if filtered:
            parts = filtered[-1]
        else:
            filter_relaxed = True

    out: dict = {"source": "champion.org (exact all-time champion from PrintChampionGenotype)"}
    if update_relaxed:
        out["champion_update_filter_relaxed"] = True
    if filter_relaxed:
        out["champion_generation_filter_relaxed"] = True

    def get_float(idx: int | None):
        if idx is None or idx >= len(parts):
            return None
        try:
            return float(parts[idx])
        except ValueError:
            return None

    def get_int(idx: int | None):
        v = get_float(idx)
        return None if v is None else int(v)

    out["update"] = get_int(idx_update)
    out["max_fitness"] = get_float(idx_fitness)
    out["ave_fitness"] = out["max_fitness"]
    out["organism_id"] = get_int(idx_org)
    out["cell_id"] = get_int(idx_cell)
    out["generation"] = get_int(idx_generation)
    out["age"] = get_int(idx_age)
    out["num_divides"] = get_int(idx_divides)
    out["merit"] = get_float(idx_merit)
    out["gestation_time"] = get_float(idx_gestation)
    out["genotype_id"] = get_int(idx_genotype)
    if idx_name is not None and idx_name < len(parts):
        out["genotype_name"] = parts[idx_name]
    return {k: v for k, v in out.items() if v is not None}


def find_org_champion(data_dir: str = "data", **_unused):
    """Return (champion .org path, meta dict) using ``data/champion.org``.

    Avida's ``PrintChampionGenotype`` action overwrites ``champion.org`` whenever
    a new all-time highest LIVE fitness is observed, so the file on disk *is*
    the best individual ever observed. If it is missing, the run did not
    produce one and we just report that.
    """
    champion_path = os.path.join(data_dir, "champion.org")
    if not os.path.exists(champion_path):
        return None, {"error": f"{champion_path} not found; ensure events.cfg has PrintChampionGenotype."}

    meta_from_header = cu.read_champion_org_meta(champion_path) if cu is not None else {}
    exact_meta = parse_champion_dat(os.path.join(data_dir, "champion.dat"))
    meta: dict = {
        "source": "champion.org (live highest-fitness organism)",
        "genotype_name": os.path.basename(champion_path).replace(".org", ""),
    }
    meta.update(exact_meta)
    if "Fitness (LRO, FITNESS_METHOD 3)" in meta_from_header:
        try:
            meta["max_fitness"] = float(meta_from_header["Fitness (LRO, FITNESS_METHOD 3)"])
        except ValueError:
            pass
    for key in ("Update Output", "Genotype ID"):
        if key in meta_from_header:
            try:
                meta[key.lower().replace(" ", "_")] = int(meta_from_header[key])
            except ValueError:
                meta[key.lower().replace(" ", "_")] = meta_from_header[key]
    meta.setdefault("update", meta.get("update_output"))
    meta.setdefault("ave_fitness", meta.get("max_fitness"))
    return champion_path, meta


def find_org_by_population_max_fitness(
    extrema_file: str = "data/fitness_extrema.dat",
    dom_file: str = "data/dominant.dat",
):
    """Pick genotype snapshot at the update where *population* max_fitness peaked.

    Note: max_fitness is tracked in cStats, but does not record which genotype achieved it.
    We therefore choose the genotype snapshot recorded by dominant.dat at that same update
    as the replay target (best-available proxy with existing logs).
    """
    extrema = parse_fitness_extrema(extrema_file)
    if not extrema:
        return None, {}
    best = max(extrema, key=lambda r: (r["max_fitness"], r["update"]))
    target_update = best["update"]

    columns, rows = parse_dominant_dat(dom_file)
    if not rows:
        return None, {}
    idx_u = _dominant_col_index(columns, "Update") or 0
    idx_nm = _dominant_col_index(columns, "Name of the Dominant") or (len(rows[0]) - 1)
    # Find the closest row by update (dominant.dat cadence may differ).
    best_row = None
    best_du = None
    for parts in rows:
        try:
            u = int(float(parts[idx_u]))
        except (ValueError, IndexError):
            continue
        du = abs(u - target_update)
        if best_du is None or du < best_du:
            best_du = du
            best_row = parts
    if best_row is None:
        return None, {}
    name = best_row[idx_nm] if idx_nm < len(best_row) else best_row[-1]
    org_path = f"data/archive/{name}.org"
    meta = {
        "update": target_update,
        "ave_fitness": best["ave_fitness"],
        "max_fitness": best["max_fitness"],
        "genotype_name": name,
        "note": "Chosen by peak population max_fitness; genotype proxied via dominant.dat at same update.",
    }
    return (org_path if os.path.exists(org_path) else None), meta


def read_lifespan_instruction_cap(cfg_path: str, genome_len: int) -> int | None:
    """Maximum total instructions before age death, from avida.cfg.

    DEATH_METHOD 0 → None (no age limit in config).
    1 (CONST) → AGE_LIMIT instructions.
    2 (MULTIPLE) → AGE_LIMIT * genome_len (Avida eDEATH_METHOD_MULTIPLE).
    """
    death_method = 0
    age_limit = 20
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = _strip_cfg_line(raw)
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key, val = parts[0], parts[1]
            if key == "DEATH_METHOD":
                death_method = int(float(val))
            elif key == "AGE_LIMIT":
                age_limit = int(float(val))
    if death_method <= 0:
        return None
    if death_method == 1:
        return max(1, age_limit)
    # DEATH_METHOD 2 == genome_length * AGE_LIMIT
    return max(1, int(genome_len * age_limit))


def read_ave_time_slice(cfg_path: str) -> int:
    v = 1
    if not os.path.exists(cfg_path):
        return v
    with open(cfg_path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = _strip_cfg_line(raw)
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "AVE_TIME_SLICE":
                try:
                    v = max(1, int(float(parts[1])))
                except ValueError:
                    pass
                break
    return v


def replay_updates_for_lifespan(cfg_path: str, genome_len: int, fallback: int = 500) -> int:
    """Approximate number of updates for one org to reach instruction cap (solo merit ~ constant)."""
    cap = read_lifespan_instruction_cap(cfg_path, genome_len)
    if cap is None:
        return fallback
    ts = read_ave_time_slice(cfg_path)
    # One organism: roughly ts instructions executed per update (constant slicing).
    n = int(math.ceil(float(cap) / float(ts)))
    return max(20, min(500_000, n))


def read_org_genome(org_path):
    """Read instructions from a .org file (skip comment/blank lines)."""
    lines = []
    with open(org_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
    return lines


def read_world_size():
    """Read WORLD_X and WORLD_Y from avida.cfg."""
    wx, wy = 120, 120
    if os.path.exists("avida.cfg"):
        with open("avida.cfg") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    if parts[0] == "WORLD_X":
                        wx = int(parts[1])
                    elif parts[0] == "WORLD_Y":
                        wy = int(parts[1])
    return wx, wy


def read_env_world_size():
    """Read ENV_WORLD_X/Y from avida.cfg, falling back to WORLD_X/Y."""
    wx, wy = read_world_size()
    ex, ey = wx, wy
    if os.path.exists("avida.cfg"):
        with open("avida.cfg") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    if parts[0] == "ENV_WORLD_X":
                        val = int(parts[1])
                        if val > 0:
                            ex = val
                    elif parts[0] == "ENV_WORLD_Y":
                        val = int(parts[1])
                        if val > 0:
                            ey = val
    return ex, ey


# ---------------------------------------------------------------------------
# Step 2: Set up a solo replay run
# ---------------------------------------------------------------------------

def setup_replay(org_path, replay_dir, world_x, world_y, num_updates=500, allow_reproduction=False):
    """Copy the real experiment configs and patch only what's needed for a solo replay."""
    os.makedirs(replay_dir, exist_ok=True)

    # Copy the organism file
    shutil.copy(org_path, os.path.join(replay_dir, "dominant.org"))

    # Copy environment and instset verbatim from the main experiment
    for fname in ["environment.cfg", "instset.cfg"]:
        src = os.path.abspath(fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(replay_dir, fname))

    # --- Patch avida.cfg: start from the real one and override specific keys ---
    # Settings we override for the solo replay:
    overrides = {
        "RANDOM_SEED":          "42",
        # Must match main chemotaxis runs: speculative execution bundles many
        # instructions into one scheduler step (jumps on org_loc per update).
        "SPECULATIVE":          "0",
        # Solo replay: avoid generation gate blocking the single organism.
        "GENERATION_LOCK":    "0",
        # Zero out ALL mutations so we watch the exact genome
        "COPY_MUT_PROB":        "0.0",
        "COPY_INS_PROB":        "0.0",
        "COPY_DEL_PROB":        "0.0",
        "COPY_UNIFORM_PROB":    "0.0",
        "COPY_SLIP_PROB":       "0.0",
        "DIV_MUT_PROB":         "0.0",
        "DIV_INS_PROB":         "0.0",
        "DIV_DEL_PROB":         "0.0",
        "DIV_UNIFORM_PROB":     "0.0",
        "DIV_SLIP_PROB":        "0.0",
        "DIVIDE_MUT_PROB":      "0.0",
        "DIVIDE_INS_PROB":      "0.0",
        "DIVIDE_DEL_PROB":      "0.0",
        "DIVIDE_UNIFORM_PROB":  "0.0",
        "DIVIDE_SLIP_PROB":     "0.0",
        "POINT_MUT_PROB":       "0.0",
        "INJECT_MUT_PROB":      "0.0",
        "INJECT_INS_PROB":      "0.0",
        "INJECT_DEL_PROB":      "0.0",
        "PARENT_MUT_PROB":      "0.0",
        # Population: default replay keeps only the focal lineage alive at a
        # time. Reproduction-enabled replay overrides this below so offspring
        # remain visible after semel reproduction.
        "POP_CAP_ELDEST":       "1",
        "POPULATION_CAP":       "0",
        # Solo movement: avoid mass-action birth (BIRTH_METHOD 4) which briefly
        # creates two cells and confuses trajectory tracking; parent replaced by
        # offspring keeps a single org id lineage when divide succeeds.
        "BIRTH_METHOD":         "3",
        "PREFER_EMPTY":         "1",
        "ALLOW_PARENT":         "0",
        # Replay should show active movement behavior, not the experiment's
        # OFFSPRING_WORLD_POS=1 birth re-randomization jumps. Inherit parent
        # world position so a path discontinuity is a real `move`, not a new
        # offspring trial being placed elsewhere.
        "OFFSPRING_WORLD_POS":   "0",
        # Do not override AGE_LIMIT / DEATH_METHOD — replay uses same lifespan rules
        # as the main experiment (see replay_updates_for_lifespan).
        # Match the experiment's bounded/blocked edge behavior. Using open
        # boundaries here can make replay paths diverge near the world edge.
        "DEADLY_BOUNDARIES":    "2",
        # Keep REQUIRED_TASK as original (-1 = no required task)
        "REQUIRED_TASK":        "-1",
    }

    if allow_reproduction:
        overrides.update({
            # Let semel offspring remain in the replay instead of enforcing a
            # one-organism trajectory. Mutations are still zeroed above.
            "POP_CAP_ELDEST":   "0",
            "POPULATION_CAP":   "0",
            # Use the experiment's empty-neighbor placement for offspring.
            "BIRTH_METHOD":     "3",
            "PREFER_EMPTY":     "1",
            "ALLOW_PARENT":     "0",
        })

    src_cfg = os.path.abspath("avida.cfg")
    patched_lines = []
    applied = set()

    with open(src_cfg) as f:
        for line in f:
            stripped = line.strip()
            # Skip blank/comment lines — pass through
            if not stripped or stripped.startswith("#"):
                patched_lines.append(line)
                continue

            parts = stripped.split()
            key = parts[0]

            if key in overrides:
                # Replace the value, keep trailing comment if any
                comment_idx = stripped.find("#")
                comment = ""
                if comment_idx > 0:
                    comment = "  " + stripped[comment_idx:]
                patched_lines.append(f"{key} {overrides[key]}{comment}\n")
                applied.add(key)
            else:
                patched_lines.append(line)

    # Append any overrides that weren't already in the file
    for key, val in overrides.items():
        if key not in applied:
            patched_lines.append(f"{key} {val}\n")

    with open(os.path.join(replay_dir, "avida.cfg"), "w") as f:
        f.writelines(patched_lines)

    # --- Write events.cfg: inject one organism using the same decoupled
    # away-from-peak world-position path as evolution, then track the env
    # coordinates that food and move/sense actually use.
    with open(os.path.join(replay_dir, "events.cfg"), "w") as f:
        f.write(f"""\
# Inject dominant organism away from the peak region with the same minimum
# distance rule as evolution.
u begin InjectAwayFromGradientPeak dominant.org 1 20

# Track organism WORLD/ENV position every update (not population-grid slot).
u 0:1:end PrintOrgLocData env

# Resource grid every update (matches org_loc cadence for a static heatmap)
u 0:1:end DumpMaxResGrid

# Exit
u {num_updates} Exit
""")

    # Write empty analyze.cfg
    with open(os.path.join(replay_dir, "analyze.cfg"), "w") as f:
        f.write("# empty\n")


# ---------------------------------------------------------------------------
# Step 3: Run avida
# ---------------------------------------------------------------------------

def run_avida(replay_dir, avida_bin, timeout_sec: float | None = None):
    """Run avida in the replay directory."""
    print(f"Running avida in {replay_dir} ...")
    if timeout_sec is None:
        timeout_sec = 120.0
    result = subprocess.run(
        [avida_bin],
        cwd=replay_dir,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    if result.returncode != 0:
        print("avida stderr:", result.stderr[-2000:] if result.stderr else "(none)")
        print("avida stdout:", result.stdout[-2000:] if result.stdout else "(none)")
        sys.exit(1)
    print("avida finished successfully.")


# ---------------------------------------------------------------------------
# Step 4: Parse results
# ---------------------------------------------------------------------------

def parse_org_loc_files(replay_dir):
    """Parse all org_loc files. Returns list of (update, [(org_id, x, y), ...])."""
    pattern = os.path.join(replay_dir, "data", "grid_dumps", "org_loc.*.dat")
    files = glob.glob(pattern)

    results = []
    for fpath in files:
        basename = os.path.basename(fpath)
        parts = basename.split(".")
        try:
            update = int(parts[-2])
        except (ValueError, IndexError):
            continue

        organisms = []
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                cols = line.split(",")
                if len(cols) >= 3:
                    org_id = int(cols[0])
                    x = int(cols[1])
                    y = int(cols[2])
                    organisms.append((org_id, x, y))
        results.append((update, organisms))

    results.sort(key=lambda x: x[0])
    return results


def parse_res_grid(filepath, wx, wy):
    """Parse a max_res_grid file into a 2D array."""
    grid = np.zeros((wy, wx))
    with open(filepath) as f:
        row = 0
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            for col, v in enumerate(vals):
                if col < wx and row < wy:
                    grid[row][col] = float(v)
            row += 1
    return grid


def res_grid_file_dimensions(filepath):
    """Return (cols, rows) for a whitespace resource-grid dump."""
    rows = 0
    cols = 0
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            rows += 1
            cols = max(cols, len(vals))
    return cols, rows


def _strip_cfg_comment(raw: str) -> str:
    if "#" in raw:
        raw = raw.split("#", 1)[0]
    return raw.strip()


def _read_gradient_from_environment_cfg(env_path: str, wx: int, wy: int):
    """Parse GRADIENT_RESOURCE params from environment.cfg.

    Returns (cx, cy, spread, plateau, height) or None if not present.
    """
    if not os.path.exists(env_path):
        return None

    gradient_line = None
    with open(env_path) as f:
        for raw in f:
            line = _strip_cfg_comment(raw)
            if not line:
                continue
            if line.startswith("GRADIENT_RESOURCE"):
                gradient_line = line
                break

    if not gradient_line:
        return None

    parts = gradient_line.split(None, 1)
    if len(parts) < 2:
        return None

    kv = {}
    for tok in parts[1].split(":"):
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k.strip()] = v.strip()

    def as_float(key):
        try:
            return float(kv[key])
        except Exception:
            return None

    def as_int(key):
        try:
            return int(float(kv[key]))
        except Exception:
            return None

    spread = as_float("spread")
    plateau = as_float("plateau")
    height = as_float("height")
    if spread is None or plateau is None or height is None:
        return None

    min_x = as_int("min_x")
    max_x = as_int("max_x")
    min_y = as_int("min_y")
    max_y = as_int("max_y")

    peakx = as_int("peakx")
    peaky = as_int("peaky")
    if peakx is not None and peaky is not None:
        cx = max(0, min(peakx, wx - 1))
        cy = max(0, min(peaky, wy - 1))
    elif None not in (min_x, max_x, min_y, max_y):
        min_x = max(0, min(min_x, wx - 1))
        max_x = max(0, min(max_x, wx - 1))
        min_y = max(0, min(min_y, wy - 1))
        max_y = max(0, min(max_y, wy - 1))
        cx = int(round((min_x + max_x) / 2.0))
        cy = int(round((min_y + max_y) / 2.0))
    else:
        cx = wx // 2
        cy = wy // 2

    return cx, cy, spread, plateau, height


def _make_synthetic_gradient(
    wx: int, wy: int, cx: int, cy: int, spread: float, plateau: float, height: float = 1.0,
) -> np.ndarray:
    """Approximate Avida's radial gradient: height / (distance + 1)."""
    grid = np.zeros((wy, wx))
    if spread <= 0:
        return grid
    for y in range(wy):
        for x in range(wx):
            d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if d <= spread:
                value = height / (d + 1.0)
                if plateau >= 0.0 and value >= 1.0:
                    value = plateau
                grid[y, x] = max(0.0, value)
    return grid


def load_resource_grid(replay_dir, wx, wy):
    """Load the first available resource grid."""
    pattern = os.path.join(replay_dir, "data", "grid_dumps", "max_res_grid.*.dat")
    files = sorted(glob.glob(pattern))
    if files:
        gx, gy = res_grid_file_dimensions(files[0])
        if (gx, gy) == (wx, wy):
            return parse_res_grid(files[0], wx, wy)
        print(
            f"Note: resource dump is {gx}x{gy}, but replay is plotting {wx}x{wy} "
            "env coordinates; synthesizing gradient from environment.cfg instead."
        )
    # Fallback: try from the main experiment
    main_pattern = os.path.join("data", "grid_dumps", "max_res_grid.*.dat")
    main_files = sorted(glob.glob(main_pattern))
    if main_files:
        gx, gy = res_grid_file_dimensions(main_files[0])
        if (gx, gy) == (wx, wy):
            return parse_res_grid(main_files[0], wx, wy)

    # Final fallback: synthesize from environment.cfg so the replay always shows the gradient.
    parsed = _read_gradient_from_environment_cfg(os.path.join(replay_dir, "environment.cfg"), wx, wy)
    if parsed is None:
        parsed = _read_gradient_from_environment_cfg("environment.cfg", wx, wy)
    if parsed is not None:
        cx, cy, spread, plateau, height = parsed
        return _make_synthetic_gradient(wx, wy, cx, cy, spread, plateau, height)

    return np.zeros((wy, wx))


# ---------------------------------------------------------------------------
# Step 5: Visualize
# ---------------------------------------------------------------------------

def _format_layer_value(value: float) -> str:
    """Format a resource-layer value without noisy trailing zeros."""
    if abs(value - round(value)) < 0.005:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _positive_resource_layers(res_grid: np.ndarray) -> np.ndarray:
    """Return positive resource values, rounded to avoid float dump noise."""
    if not res_grid.size:
        return np.array([])
    return np.unique(np.round(res_grid[res_grid > 0.0], 2))


def _food_scale_colorbar(fig, ax, mappable, res_grid: np.ndarray) -> None:
    """Add a slim colorbar: low / high numeric ticks only; mid shown as a swatch."""
    lo = float(np.nanmin(res_grid)) if res_grid.size else 0.0
    hi = float(np.nanmax(res_grid)) if res_grid.size else 0.0

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.2%", pad=0.08)
    cb = fig.colorbar(mappable, cax=cax)
    cb.ax.tick_params(labelsize=8, colors="white", length=0)
    # Label under the bar so it cannot collide with the top tick value.
    cb.ax.set_xlabel("food", color="#aaaaaa", fontsize=9, labelpad=4)
    cb.ax.set_ylabel("")
    cb.outline.set_edgecolor("#444444")
    for spine in cb.ax.spines.values():
        spine.set_edgecolor("#444444")

    if hi <= lo:
        cb.set_ticks([lo])
        cb.set_ticklabels([_format_layer_value(lo)])
        return

    cb.set_ticks([lo, hi])
    cb.set_ticklabels([_format_layer_value(lo), _format_layer_value(hi)])

    mid = 0.5 * (lo + hi)
    rgba = np.asarray(mappable.cmap(mappable.norm(mid)))
    cb.ax.scatter(
        [0.5],
        [mid],
        s=64,
        c=[rgba],
        clip_on=True,
        zorder=10,
        linewidths=1.2,
        edgecolors="#ffffffaa",
    )


def visualize(
    trajectory, res_grid, world_x, world_y, org_name,
    save_path=None, fps=10, show_gradient_layer_labels=False,
):
    """Animate the organism's path on the resource grid.

    If *show_gradient_layer_labels* is True, the food field uses the discrete
    layer colormap and a right-hand colorbar: min/max labels only, a mid-scale
    colour swatch, and the word "food" below the bar (avoids top overlap).
    """

    # Build one (x,y) per logged update using *spatial continuity*, not raw org_id.
    #
    # Avida can reuse org_id after death/rebirth. Preferring "same id" then
    # "closest" causes end-of-run jumps when several organisms exist briefly.
    # We only trust positions within a Chebyshev "movement budget" from the last
    # point (scaled if org_loc skips updates); otherwise we take the global
    # closest point and print a one-line note.
    max_step = 4  # max cells per update (Chebyshev); generous for move + grid quirks
    tracked_id: int | None = None
    last_xy: tuple[float, float] | None = None
    last_update_num: int | None = None
    path_points: list[tuple[int, int, int, int]] = []
    orgs_by_update: dict[int, list[tuple[int, int, int]]] = {}
    original_parent_id: int | None = None
    discontinuity_notes = 0

    def cheb(ax: int, ay: int, bx: float, by: float) -> int:
        return max(abs(ax - int(bx)), abs(ay - int(by)))

    for update, organisms in trajectory:
        if not organisms:
            continue
        orgs_by_update[update] = organisms

        chosen: tuple[int, int, int] | None = None

        if last_xy is None:
            # First frame: replay injects near world center; prefer that over org_id.
            cx0, cy0 = world_x // 2, world_y // 2
            if len(organisms) == 1:
                chosen = organisms[0]
            else:
                chosen = min(
                    organisms,
                    key=lambda o: ((o[1] - cx0) ** 2 + (o[2] - cy0) ** 2, o[0]),
                )
            tracked_id = chosen[0]
            original_parent_id = chosen[0]
        else:
            lx, ly = last_xy
            gap = max(1, update - last_update_num) if last_update_num is not None else 1
            budget = max_step * gap

            near = [o for o in organisms if cheb(o[1], o[2], lx, ly) <= budget]
            if near:
                same = [o for o in near if tracked_id is not None and o[0] == tracked_id]
                if same:
                    chosen = same[0]
                else:
                    chosen = min(
                        near,
                        key=lambda o: ((o[1] - lx) ** 2 + (o[2] - ly) ** 2, o[0]),
                    )
            else:
                chosen = min(
                    organisms,
                    key=lambda o: ((o[1] - lx) ** 2 + (o[2] - ly) ** 2, o[0]),
                )
                if cheb(chosen[1], chosen[2], lx, ly) > budget and discontinuity_notes < 5:
                    print(
                        f"Note: no organism within {budget} cells of last position at "
                        f"update {update}; following closest at ({chosen[1]},{chosen[2]}) "
                        f"(from ({int(lx)},{int(ly)}))."
                    )
                    discontinuity_notes += 1

            tracked_id = chosen[0]

        if chosen is None:
            continue

        chosen_id, x, y = chosen
        last_xy = (float(x), float(y))
        last_update_num = update
        path_points.append((update, chosen_id, x, y))

    if not path_points:
        print("ERROR: No position data found. The organism may have died immediately.")
        sys.exit(1)

    updates = [p[0] for p in path_points]
    tracked_ids = [p[1] for p in path_points]
    xs = [p[2] for p in path_points]
    ys = [p[3] for p in path_points]
    has_nontracked_orgs = any(
        len([o for o in orgs_by_update.get(update, []) if o[0] != tracked_id_at_frame]) > 0
        for update, tracked_id_at_frame in zip(updates, tracked_ids)
    )

    print(f"Tracked {len(path_points)} updates, from update {updates[0]} to {updates[-1]}")
    print(f"Start position: ({xs[0]}, {ys[0]})")
    print(f"End position:   ({xs[-1]}, {ys[-1]})")

    # Find the food peak location.
    # max_res_grid often contains a flat plateau (many cells at the max), so argmax()
    # can pick an arbitrary corner. Use the centroid of max-valued cells instead.
    max_val = float(np.max(res_grid)) if res_grid.size else 0.0
    if max_val > 0.0:
        mask = res_grid >= (max_val - 1e-12)
        coords = np.argwhere(mask)  # rows (y), cols (x)
        if coords.size:
            peak_y = float(np.mean(coords[:, 0]))
            peak_x = float(np.mean(coords[:, 1]))
        else:
            peak_y, peak_x = np.unravel_index(np.argmax(res_grid), res_grid.shape)
            peak_x = float(peak_x)
            peak_y = float(peak_y)
    else:
        peak_x = world_x / 2.0
        peak_y = world_y / 2.0
    print(f"Food peak at:   ({peak_x:.2f}, {peak_y:.2f})")

    # --- Figure setup ---
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    fig.patch.set_facecolor("#0f0f1a")

    if show_gradient_layer_labels:
        layer_values = _positive_resource_layers(res_grid)
        color_steps = max(3, len(layer_values) + 1)
        food_cmap = ListedColormap(
            LinearSegmentedColormap.from_list("food_layers", [
                "#0f0f1a",
                "#17315f",
                "#1f7a68",
                "#82c94f",
                "#fff07a",
            ])(np.linspace(0, 1, color_steps))
        )
        if len(layer_values):
            zero_upper = min(0.001, float(layer_values[0]) / 2.0)
            bounds = np.concatenate((
                [-0.001, zero_upper],
                (layer_values[:-1] + layer_values[1:]) / 2.0,
                [layer_values[-1] + 0.01],
            ))
            food_norm = BoundaryNorm(bounds, food_cmap.N)
        else:
            food_norm = None
    else:
        food_cmap = LinearSegmentedColormap.from_list("food", [
            (0.0, "#0f0f1a"),
            (0.15, "#0a2a1a"),
            (0.4, "#1a5c3c"),
            (1.0, "#33ff33"),
        ])
        food_norm = None

    # Resource heatmap
    if food_norm is not None:
        im_food = ax.imshow(
            res_grid, cmap=food_cmap, norm=food_norm, origin="upper",
            extent=[0, world_x, world_y, 0],
            alpha=0.85, interpolation="nearest",
        )
        _food_scale_colorbar(fig, ax, im_food, res_grid)
    else:
        ax.imshow(res_grid, cmap=food_cmap, origin="upper",
                  extent=[0, world_x, world_y, 0],
                  vmin=0, vmax=max(res_grid.max(), 0.01),
                  alpha=0.85, interpolation="bilinear")

    # Food peak marker
    food_circle = Circle((peak_x, peak_y), 3,
                         fill=False, edgecolor="#33ff33", linewidth=1.5,
                         linestyle="--", alpha=0.6)
    ax.add_patch(food_circle)
    ax.text(peak_x, peak_y - 5, "FOOD", ha="center", va="bottom",
            fontsize=9, color="#33ff33", fontweight="bold", alpha=0.7)

    # Trail line (grows over time)
    trail_line, = ax.plot([], [], color="#ff6b6b", linewidth=1.0, alpha=0.4)

    # Other live organisms in the same frame, usually offspring during replay.
    offspring_dots = ax.scatter(
        [], [], s=52, c="#ffd166", edgecolors="#111111", linewidths=0.8,
        alpha=0.95, zorder=9
    )

    # Current position dot
    current_dot, = ax.plot([], [], "o", color="#ffffff", markersize=10,
                           markeredgecolor="#ff6b6b", markeredgewidth=2, zorder=10)

    # Start marker
    ax.plot(xs[0], ys[0], "s", color="#4488ff", markersize=10,
            markeredgecolor="white", markeredgewidth=1.5, zorder=9)
    ax.text(xs[0] + 2, ys[0] - 2, "START", fontsize=8, color="#4488ff",
            fontweight="bold")

    # Title and labels
    title = ax.set_title("", fontsize=14, color="white", fontweight="bold", pad=10)
    ax.set_xlabel("X", color="white", fontsize=10)
    ax.set_ylabel("Y", color="white", fontsize=10)
    ax.set_xlim(0, world_x)
    ax.set_ylim(world_y, 0)
    ax.set_facecolor("#0f0f1a")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333333")

    # Info text
    info_text = ax.text(0.02, 0.02, "", transform=ax.transAxes,
                        fontsize=9, color="#aaaaaa", va="bottom",
                        fontfamily="monospace")

    # Genome name
    ax.text(0.98, 0.98, org_name, transform=ax.transAxes, fontsize=9,
            color="#888888", ha="right", va="top", fontfamily="monospace")

    def animate(frame_idx):
        # Show trail up to this point
        trail_x = xs[:frame_idx + 1]
        trail_y = ys[:frame_idx + 1]
        trail_line.set_data(trail_x, trail_y)

        # Current position
        cx, cy = xs[frame_idx], ys[frame_idx]
        current_tracked_id = tracked_ids[frame_idx]
        current_dot.set_data([cx], [cy])
        if original_parent_id is not None and current_tracked_id != original_parent_id:
            current_dot.set_markerfacecolor("#ffd166")
            current_dot.set_markeredgecolor("#111111")
        else:
            current_dot.set_markerfacecolor("#ffffff")
            current_dot.set_markeredgecolor("#ff6b6b")

        other_orgs = [
            (ox, oy)
            for oid, ox, oy in orgs_by_update.get(updates[frame_idx], [])
            if oid != current_tracked_id
        ]
        if other_orgs:
            offspring_dots.set_offsets(np.asarray(other_orgs, dtype=float))
        else:
            offspring_dots.set_offsets(np.empty((0, 2)))

        # Distance to food
        dist = np.sqrt((cx - peak_x)**2 + (cy - peak_y)**2)

        subject = "Offspring" if (
            original_parent_id is not None and current_tracked_id != original_parent_id
        ) else "Dominant Organism"
        title.set_text(f"{subject} — Update {updates[frame_idx]}")
        offspring_msg = f"  |  Other org dots: {len(other_orgs)}" if has_nontracked_orgs else ""
        info_text.set_text(
            f"Position: ({cx}, {cy})  |  "
            f"Distance to food: {dist:.1f}  |  "
            f"Steps: {frame_idx}/{len(path_points)-1}"
            f"{offspring_msg}"
        )

        return trail_line, offspring_dots, current_dot, title, info_text

    anim = animation.FuncAnimation(
        fig, animate, frames=len(path_points),
        interval=50, blit=False, repeat=True,
    )

    if save_path:
        print(f"Saving to {save_path} ...")
        if save_path.endswith(".gif"):
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=120)
        print(f"Saved: {save_path}")
    else:
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize the highest-fitness organism navigating the environment"
    )
    parser.add_argument("--org", type=str, default=None,
                        help="Path to .org file (default: pick from dominant.dat)")
    parser.add_argument(
        "--selection",
        choices=("champion", "pop_max_fitness", "dominant_max_fitness", "last"),
        default="champion",
        help=(
            "How to pick organism for replay.\n"
            "- champion (default): the highest-fitness organism saved live by "
            "PrintChampionGenotype (data/champion.org).\n"
            "- pop_max_fitness: use data/fitness_extrema.dat peak max_fitness update "
            "(proxy genotype via dominant.dat)\n"
            "- dominant_max_fitness: legacy dominant.dat-based max selection\n"
            "- last: final dominant"
        ),
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory with Avida outputs (default: data).",
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=None,
        help="Replay length in updates (overrides lifespan / observation floor)",
    )
    parser.add_argument(
        "--strict-lifespan",
        action="store_true",
        help="Run only for DEATH_METHOD/AGE_LIMIT-derived updates (no observation floor)",
    )
    parser.add_argument(
        "--min-observation-updates",
        type=int,
        default=0,
        metavar="N",
        help="When not using --updates, extend replay to at least N updates (default 0).\n"
             "Set this if you want a longer movement clip than the organism's true lifespan.",
    )
    parser.add_argument("--save", type=str, default=None,
                        help="Save animation (e.g. dominant.gif or dominant.mp4)")
    parser.add_argument("--fps", type=int, default=15,
                        help="Frames per second for saved animation (default: 15)")
    parser.add_argument("--avida", type=str, default="../avida",
                        help="Path to avida binary (default: ../avida)")
    args = parser.parse_args()

    # Find the organism
    pick_meta = {}
    if args.org:
        org_path = args.org
    else:
        if args.selection == "champion":
            org_path, pick_meta = find_org_champion(data_dir=args.data_dir)
            if not org_path and pick_meta.get("error"):
                print(f"champion selection failed: {pick_meta['error']}")
                print("Falling back to pop_max_fitness selection.")
                org_path, pick_meta = find_org_by_population_max_fitness(
                    extrema_file=os.path.join(args.data_dir, "fitness_extrema.dat"),
                    dom_file=os.path.join(args.data_dir, "dominant.dat"),
                )
        elif args.selection == "pop_max_fitness":
            org_path, pick_meta = find_org_by_population_max_fitness(
                extrema_file=os.path.join(args.data_dir, "fitness_extrema.dat"),
                dom_file=os.path.join(args.data_dir, "dominant.dat"),
            )
        else:
            # Map to the old dominant.dat selection modes.
            sel = "max_fitness" if args.selection == "dominant_max_fitness" else "last"
            org_path, pick_meta = find_org_from_dominant_dat(
                selection=sel,
                dom_file=os.path.join(args.data_dir, "dominant.dat"),
            )

    if not org_path or not os.path.exists(org_path):
        print("ERROR: Could not find organism .org file.")
        print("Either run the evolution experiment first, or specify --org path/to/file.org")
        if pick_meta.get("genotype_name"):
            print(f"(Expected archive: data/archive/{pick_meta['genotype_name']}.org)")
        sys.exit(1)

    org_name = os.path.basename(org_path).replace(".org", "")
    genome = read_org_genome(org_path)
    if pick_meta:
        print(
            f"Selected ({args.selection}): {org_name}  |  "
            f"update={pick_meta.get('update')}  "
            f"ave_fitness={pick_meta.get('ave_fitness')}  "
            f"max_fitness={pick_meta.get('max_fitness')}"
        )
        if pick_meta.get("note"):
            print(f"Note: {pick_meta['note']}")
    print(f"Genome: {len(genome)} instructions")

    # Check for movement instructions
    move_instrs = [g for g in genome if g in ("move", "rotate-uphill", "rotate-left-one",
                                                "rotate-right-one", "rotate-to-unoccupied-cell")]
    print(f"Movement instructions: {len(move_instrs)} "
          f"({', '.join(set(move_instrs)) if move_instrs else 'NONE — organism may not move!'})")

    # Population grid and environment grid can differ. Movement/food now use
    # the env grid, so visualization should be scaled to ENV_WORLD_X/Y.
    world_x, world_y = read_world_size()
    env_world_x, env_world_y = read_env_world_size()
    print(f"Population grid: {world_x} x {world_y}")
    print(f"Environment grid: {env_world_x} x {env_world_y}")

    cfg_abs = os.path.abspath("avida.cfg")
    if args.updates is not None:
        num_updates = max(1, args.updates)
        print(f"Replay updates (manual): {num_updates}")
    else:
        life_u = replay_updates_for_lifespan(cfg_abs, len(genome), fallback=500)
        cap = read_lifespan_instruction_cap(cfg_abs, len(genome))
        ts = read_ave_time_slice(cfg_abs)

        # Default behavior: replay the organism's true age (lifespan-derived).
        num_updates = life_u

        # Optional extension: force a longer clip for visualization.
        if not args.strict_lifespan:
            floor = max(0, int(args.min_observation_updates))
            if floor > 0:
                num_updates = max(life_u, floor)

        if args.strict_lifespan:
            print(
                f"Replay updates (--strict-lifespan): {num_updates}  "
                f"(instruction cap={cap}, AVE_TIME_SLICE={ts})"
            )
        elif num_updates > life_u:
            print(
                f"Note: lifespan from avida.cfg implies only ~{life_u} updates "
                f"(cap={cap} instr, AVE_TIME_SLICE={ts}); running {num_updates} "
                f"for a longer movement clip. Use --strict-lifespan to disable extension, "
                f"or --updates N to set exactly N."
            )
        else:
            print(
                f"Replay updates: {num_updates}  "
                f"(lifespan-derived, cap={cap} instr, AVE_TIME_SLICE={ts})"
            )

    # Setup replay
    replay_dir = os.path.join(os.getcwd(), "_replay_dominant")
    if os.path.exists(replay_dir):
        shutil.rmtree(replay_dir)
    setup_replay(org_path, replay_dir, world_x, world_y, num_updates)

    # Find avida binary
    avida_bin = os.path.abspath(args.avida)
    if not os.path.exists(avida_bin):
        print(f"ERROR: avida binary not found at {avida_bin}")
        sys.exit(1)

    # Run (scale timeout with replay length)
    run_avida(replay_dir, avida_bin, timeout_sec=max(120.0, 0.05 * float(num_updates) + 30.0))

    # Parse trajectory
    trajectory = parse_org_loc_files(replay_dir)
    if not trajectory:
        print("ERROR: No position data generated. Check avida output.")
        sys.exit(1)

    # Load resource grid
    res_grid = load_resource_grid(replay_dir, env_world_x, env_world_y)

    # Visualize
    visualize(trajectory, res_grid, env_world_x, env_world_y, org_name,
              save_path=args.save, fps=args.fps)

    # Clean up
    shutil.rmtree(replay_dir, ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()
