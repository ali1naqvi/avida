#!/usr/bin/env python3
"""Replay ``data/lifetime_champion.org`` -- the all-time R0 champion.

The genome at ``data/lifetime_champion.org`` is overwritten by the live
``PrintLifetimeFitnessChampion`` event (see ``events.cfg``) every time a new
all-time highest lifetime reproductive output is observed.
``data/lifetime_champion.dat`` records the same events with extra metadata
(cell, easterly/northerly, num_divides, etc.).

This script just runs that genome through Avida in a fresh world and animates
where it goes -- there is no post-hoc snapshot scan, no genome rebuilding,
and no champion selection logic. If ``data/lifetime_champion.org`` is missing,
fix the events file and rerun the simulation.

Notes on what "fitness" means here (FITNESS_METHOD 3 = lifetime reproductive
output, R0):

* The number printed below as ``recorded R0`` is the cumulative successful
  divides the champion achieved during evolution. R0 = 2 means the lineage
  reproduced twice; R0 = 0 means it never divided. It does *not* measure
  motion -- with ``ENERGY_GIVEN_ON_INJECT 60`` and
  ``FLAT_ENERGY_COST_PER_INST 0.00005`` a sit-still founder still has budget
  to execute ``repro`` once or twice before starving.
* The replay below is a behavioral assay: it injects this genome into a
  fresh world with the same ``environment.cfg`` and watches the trajectory.
  Its R0 in the replay can differ from the recorded R0 because environment,
  neighbors, and starting cell are different.
"""

from __future__ import annotations

import argparse
import tempfile
import os
import shutil
import sys

# Keep matplotlib from trying to write into ~/.matplotlib when this is run from
# the Codex/app sandbox or a machine with a read-only home cache directory.
_mpl_cache = os.path.join(tempfile.gettempdir(), "avida_matplotlib_cache")
os.makedirs(_mpl_cache, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cache)

import visualize_highest_fitness as vhf


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
    """Return the final row of champion data as ``{column_name: value}``."""
    if not os.path.exists(dat_path):
        return {}
    columns, rows = vhf.parse_dominant_dat(dat_path)
    if not rows:
        return {}
    last = rows[-1]
    return {col: last[i] for i, col in enumerate(columns) if i < len(last)}


def find_avida_binary(user_path: str | None) -> str | None:
    candidates = [user_path] if user_path else []
    candidates += ["../../bin/avida", "../avida", "avida"]
    for path in candidates:
        if not path:
            continue
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and os.access(abs_path, os.X_OK):
            return abs_path
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay data/lifetime_champion.org (highest-R0 organism saved by PrintLifetimeFitnessChampion)."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--org", default=None,
                        help="Champion .org filename or path (default: data/lifetime_champion.org).")
    parser.add_argument("--dat", default=None,
                        help="Champion .dat filename or path (default: data/lifetime_champion.dat).")
    parser.add_argument("--legacy-champion", action="store_true",
                        help="Replay data/champion.org and data/champion.dat instead.")
    parser.add_argument("--updates", type=int, default=None,
                        help="Replay length in updates (default: one lifespan).")
    parser.add_argument("--save", default=None, help="Save animation as .gif or .mp4.")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--avida", default=None, help="Path to avida binary.")
    parser.add_argument("--summary-only", action="store_true",
                        help="Run the replay and print a path summary; do not open an animation window.")
    args = parser.parse_args()

    default_org_name = "champion.org" if args.legacy_champion else "lifetime_champion.org"
    default_dat_name = "champion.dat" if args.legacy_champion else "lifetime_champion.dat"

    champion_org = args.org or os.path.join(args.data_dir, default_org_name)
    champion_dat = args.dat or os.path.join(args.data_dir, default_dat_name)
    if args.org and not os.path.isabs(champion_org) and not os.path.exists(champion_org):
        champion_org = os.path.join(args.data_dir, champion_org)
    if args.dat and not os.path.isabs(champion_dat) and not os.path.exists(champion_dat):
        champion_dat = os.path.join(args.data_dir, champion_dat)

    if not os.path.exists(champion_org):
        print(f"ERROR: {champion_org} not found.", file=sys.stderr)
        print("Make sure events.cfg has 'PrintLifetimeFitnessChampion lifetime_champion.org lifetime_champion.dat'", file=sys.stderr)
        print("and the run produced at least one organism.", file=sys.stderr)
        if not args.legacy_champion and os.path.exists(os.path.join(args.data_dir, "champion.org")):
            print("Legacy data/champion.org exists; rerun with --legacy-champion to replay it.", file=sys.stderr)
        return 1

    header = read_org_header(champion_org)
    dat_row = read_champion_dat_last_row(champion_dat)
    genome = vhf.read_org_genome(champion_org)
    move_count = sum(1 for inst in genome if inst in MOVE_INSTRUCTIONS)

    print(f"Replaying {champion_org}")
    if "Update Output" in header:
        print(f"  saved at update : {header['Update Output']}")
    if dat_row:
        if "Update" in dat_row:
            print(f"  champion update : {dat_row['Update']}")
        recorded_r0 = dat_row.get("Lifetime Fitness", dat_row.get("Champion Fitness"))
        if recorded_r0 is not None:
            print(f"  recorded R0     : {recorded_r0}  (lifetime offspring during evolution)")
        for key in ("Genotype Name", "Genotype ID", "Generation", "Num Divides",
                    "Pop Cell ID", "Env Cell ID", "Env X", "Env Y",
                    "Easterly", "Northerly", "StepDisp"):
            if dat_row.get(key) not in (None, ""):
                print(f"  {key.lower():15s} : {dat_row[key]}")
    if "Fitness" in header:
        print(f"  TestCPU fitness : {header['Fitness']}  (recomputed in an empty world; expected to be small for chemotaxis-dependent orgs)")
    print(f"  genome length   : {len(genome)}")
    print(f"  movement insts  : {move_count}")
    if move_count == 0:
        print("  NOTE: no movement instructions in the genome; any positional change you")
        print("        see in the replay is offspring placement, not active chemotaxis.")

    world_x, world_y = vhf.read_world_size()
    env_world_x, env_world_y = vhf.read_env_world_size()
    if args.updates is not None:
        num_updates = max(1, args.updates)
    else:
        num_updates = vhf.replay_updates_for_lifespan(os.path.abspath("avida.cfg"), len(genome))
    print(f"  replay updates  : {num_updates}")

    avida_bin = find_avida_binary(args.avida)
    if not avida_bin:
        print("ERROR: cannot find avida binary. Try --avida ../../bin/avida", file=sys.stderr)
        return 1

    replay_dir = os.path.join(os.getcwd(), "_replay_champion")
    if os.path.exists(replay_dir):
        shutil.rmtree(replay_dir)
    vhf.setup_replay(champion_org, replay_dir, world_x, world_y, num_updates)
    vhf.run_avida(replay_dir, avida_bin, timeout_sec=max(120.0, 0.05 * num_updates + 30.0))

    trajectory = vhf.parse_org_loc_files(replay_dir)
    if not trajectory:
        print("ERROR: replay produced no trajectory.", file=sys.stderr)
        shutil.rmtree(replay_dir, ignore_errors=True)
        return 1
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

    if not args.summary_only:
        vhf.visualize(
            trajectory, res_grid, env_world_x, env_world_y,
            "champion replay", save_path=args.save, fps=args.fps,
            show_gradient_layer_labels=True,
        )

    shutil.rmtree(replay_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
