#!/usr/bin/env python3
"""
Visualize Avida's GRADIENT_RESOURCE as an ASCII heatmap and/or image.

This reproduces the *initial* spatial profile computed by Avida's
`cGradientCount::fillinResourceValues()` for the common non-halo case,
using the "just reset" branch (i.e., plateau cells set to `plateau` and
non-plateau cells set to `height/(dist+1)` with optional `floor`).

It is meant for sanity-checking how a gradient looks given the parameters
in an `environment.cfg`.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class GradientSpec:
    name: str
    peakx: int = -1
    peaky: int = -1
    height: int = 0
    spread: int = 0
    plateau: float = -1.0
    floor: float = 0.0
    min_x: int = 0
    max_x: int = 0
    min_y: int = 0
    max_y: int = 0


def _parse_cfg_kv_lines(path: str) -> Dict[str, str]:
    """
    Parse avida.cfg-like lines: KEY VALUE [# comment]
    Returns lowercase keys -> raw value tokens string.
    """
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0].strip().lower()
            val = " ".join(parts[1:]).strip()
            out[key] = val
    return out


def _parse_gradient_resource_line(line: str) -> Optional[GradientSpec]:
    """
    Parse a single line like:
      GRADIENT_RESOURCE food0:height=20:spread=120:plateau=1:...
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if not line.upper().startswith("GRADIENT_RESOURCE "):
        return None

    rest = line.split(None, 1)[1].strip()
    # resource chunk ends at whitespace (env files can contain multiple resources on one line)
    chunk = rest.split(None, 1)[0].strip()
    name, *opts = chunk.split(":")
    opts_map: Dict[str, str] = {}
    for opt in opts:
        if not opt:
            continue
        if "=" not in opt:
            continue
        k, v = opt.split("=", 1)
        opts_map[k.strip().lower()] = v.strip()

    def get_int(key: str, default: int) -> int:
        if key not in opts_map:
            return default
        return int(float(opts_map[key]))

    def get_float(key: str, default: float) -> float:
        if key not in opts_map:
            return default
        return float(opts_map[key])

    return GradientSpec(
        name=name,
        peakx=get_int("peakx", -1),
        peaky=get_int("peaky", -1),
        height=get_int("height", 0),
        spread=get_int("spread", 0),
        plateau=get_float("plateau", -1.0),
        floor=get_float("floor", 0.0),
        min_x=get_int("min_x", 0),
        max_x=get_int("max_x", 0),
        min_y=get_int("min_y", 0),
        max_y=get_int("max_y", 0),
    )


def load_gradient_specs(env_path: str) -> List[GradientSpec]:
    specs: List[GradientSpec] = []
    with open(env_path, "r", encoding="utf-8") as f:
        for raw in f:
            # strip inline comments too
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            spec = _parse_gradient_resource_line(line)
            if spec is not None:
                specs.append(spec)
    return specs


def _choose_default_peak(spec: GradientSpec) -> Tuple[int, int]:
    """
    Avida picks a random location when peakx/peaky == -1. For visualization,
    choose the center of the allowed bounding box, clamped so the edible
    radius stays within bounds (matching the intent of Avida's constraints).
    """
    edible_radius = 1 if spec.plateau < 0 else max(0, spec.height)

    # If bounds aren't set (0s), just return (0,0) and let caller override.
    if spec.min_x == 0 and spec.max_x == 0 and spec.min_y == 0 and spec.max_y == 0:
        return (0, 0)

    cx = (spec.min_x + spec.max_x) // 2
    cy = (spec.min_y + spec.max_y) // 2

    # clamp into [min+edible_radius, max-edible_radius]
    lo_x = spec.min_x + edible_radius
    hi_x = spec.max_x - edible_radius
    lo_y = spec.min_y + edible_radius
    hi_y = spec.max_y - edible_radius

    if lo_x > hi_x:
        lo_x, hi_x = spec.min_x, spec.max_x
    if lo_y > hi_y:
        lo_y, hi_y = spec.min_y, spec.max_y

    cx = min(max(cx, lo_x), hi_x)
    cy = min(max(cy, lo_y), hi_y)
    return (cx, cy)


def compute_initial_gradient_grid(
    *,
    world_x: int,
    world_y: int,
    spec: GradientSpec,
    peakx: int,
    peaky: int,
) -> List[List[float]]:
    """
    Compute the initial gradient grid following cGradientCount::fillinResourceValues()
    in the "just reset" branch.
    """
    grid: List[List[float]] = [[0.0 for _ in range(world_x)] for _ in range(world_y)]

    for y in range(world_y):
        for x in range(world_x):
            dist = math.sqrt((peakx - x) * (peakx - x) + (peaky - y) * (peaky - y))
            if dist > spec.spread:
                grid[y][x] = 0.0
                continue

            # theoretical slope value
            val = float(spec.height) / (dist + 1.0)
            if val < spec.floor:
                val = spec.floor

            is_plat_cell = (float(spec.height) / (dist + 1.0)) >= 1.0
            if is_plat_cell and spec.plateau >= 0.0:
                val = float(spec.plateau)

            if val < 0.0:
                val = 0.0

            grid[y][x] = val

    return grid


def render_ascii(grid: List[List[float]], *, charset: str = " .:-=+*#%@") -> str:
    max_val = max((v for row in grid for v in row), default=0.0)
    if max_val <= 0:
        max_val = 1.0

    # Print with y increasing upward (so the top row is y=world_y-1)
    lines: List[str] = []
    for row in reversed(grid):
        chars: List[str] = []
        for v in row:
            t = max(0.0, min(1.0, v / max_val))
            idx = int(round(t * (len(charset) - 1)))
            chars.append(charset[idx])
        lines.append("".join(chars))
    return "\n".join(lines)


def write_csv(path: str, grid: List[List[float]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in grid:
            f.write(",".join(f"{v:.6g}" for v in row))
            f.write("\n")


def write_pgm(path: str, grid: List[List[float]]) -> None:
    """
    Write a plain (ASCII) PGM image (P2). No external deps needed.
    """
    h = len(grid)
    w = len(grid[0]) if h else 0
    max_val = max((v for row in grid for v in row), default=0.0)
    if max_val <= 0:
        max_val = 1.0
    max_gray = 255

    with open(path, "w", encoding="ascii") as f:
        f.write("P2\n")
        f.write(f"{w} {h}\n")
        f.write(f"{max_gray}\n")
        # Flip vertically so that y=world_y-1 is top in the image too.
        for row in reversed(grid):
            vals = []
            for v in row:
                t = max(0.0, min(1.0, v / max_val))
                vals.append(str(int(round(t * max_gray))))
            f.write(" ".join(vals))
            f.write("\n")


def write_png(path: str, grid: List[List[float]]) -> None:
    """
    Write a grayscale PNG (no required dependencies).

    Tries Pillow first, then matplotlib. If neither are available, raises.
    """
    h = len(grid)
    w = len(grid[0]) if h else 0
    max_val = max((v for row in grid for v in row), default=0.0)
    if max_val <= 0:
        max_val = 1.0

    # Build 8-bit grayscale pixels, flipped vertically (so y=world_y-1 is top).
    pixels: List[int] = []
    for row in reversed(grid):
        for v in row:
            t = max(0.0, min(1.0, v / max_val))
            pixels.append(int(round(t * 255)))

    try:
        from PIL import Image  # type: ignore

        img = Image.new("L", (w, h))
        img.putdata(pixels)
        img.save(path)
        return
    except Exception:
        pass

    try:
        import matplotlib.pyplot as plt  # type: ignore

        # reshape pixels into (h, w)
        arr = [pixels[i * w : (i + 1) * w] for i in range(h)]
        plt.imsave(path, arr, cmap="gray", vmin=0, vmax=255, format="png")
        return
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PNG output requires either Pillow (PIL) or matplotlib. "
            "Install one of them, or use --pgm."
        ) from e


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Visualize an Avida GRADIENT_RESOURCE as ASCII / CSV / PGM / PNG."
    )
    ap.add_argument("--env", required=True, help="Path to environment.cfg")
    ap.add_argument("--avida", default=None, help="Optional path to avida.cfg to read WORLD_X/WORLD_Y")
    ap.add_argument("--resource", required=True, help="Gradient resource name (e.g., food0)")
    ap.add_argument("--world-x", type=int, default=None, help="Override WORLD_X")
    ap.add_argument("--world-y", type=int, default=None, help="Override WORLD_Y")
    ap.add_argument("--peakx", type=int, default=None, help="Override peak x")
    ap.add_argument("--peaky", type=int, default=None, help="Override peak y")
    ap.add_argument("--ascii", action="store_true", help="Print ASCII heatmap to stdout")
    ap.add_argument("--csv", default=None, help="Write CSV grid to this path")
    ap.add_argument("--pgm", default=None, help="Write grayscale PGM (P2) image to this path")
    ap.add_argument("--png", default=None, help="Write grayscale PNG image to this path")
    args = ap.parse_args()

    specs = load_gradient_specs(args.env)
    spec = next((s for s in specs if s.name == args.resource), None)
    if spec is None:
        known = ", ".join(s.name for s in specs) if specs else "(none found)"
        raise SystemExit(f"Resource '{args.resource}' not found. Known gradient resources: {known}")

    world_x = args.world_x
    world_y = args.world_y
    if (world_x is None or world_y is None) and args.avida:
        cfg = _parse_cfg_kv_lines(args.avida)
        if world_x is None and "world_x" in cfg:
            world_x = int(float(cfg["world_x"]))
        if world_y is None and "world_y" in cfg:
            world_y = int(float(cfg["world_y"]))
    if world_x is None:
        world_x = 120
    if world_y is None:
        world_y = 120

    peakx = args.peakx if args.peakx is not None else (spec.peakx if spec.peakx != -1 else _choose_default_peak(spec)[0])
    peaky = args.peaky if args.peaky is not None else (spec.peaky if spec.peaky != -1 else _choose_default_peak(spec)[1])

    if not (0 <= peakx < world_x and 0 <= peaky < world_y):
        raise SystemExit(f"Peak ({peakx},{peaky}) must be within world (0..{world_x-1}, 0..{world_y-1})")

    grid = compute_initial_gradient_grid(world_x=world_x, world_y=world_y, spec=spec, peakx=peakx, peaky=peaky)

    if args.ascii:
        print(render_ascii(grid))

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        write_csv(args.csv, grid)

    if args.pgm:
        os.makedirs(os.path.dirname(os.path.abspath(args.pgm)) or ".", exist_ok=True)
        write_pgm(args.pgm, grid)

    if args.png:
        os.makedirs(os.path.dirname(os.path.abspath(args.png)) or ".", exist_ok=True)
        write_png(args.png, grid)

    if not args.ascii and not args.csv and not args.pgm and not args.png:
        # default behavior: ASCII
        print(render_ascii(grid))

    # Print a short numeric summary to stderr-like stdout (kept minimal)
    max_val = max((v for row in grid for v in row), default=0.0)
    min_val = min((v for row in grid for v in row), default=0.0)
    nonzero = sum(1 for row in grid for v in row if v > 0)
    print(f"\n(resource={spec.name} peak=({peakx},{peaky}) world={world_x}x{world_y} min={min_val:.6g} max={max_val:.6g} nonzero={nonzero})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

