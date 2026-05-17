"""Helpers for reading the champion files written by Avida at runtime.

The C++ ``PrintChampionGenotype`` action overwrites ``data/champion.org``
(genome) and appends to ``data/champion.dat`` (record) every time a new
all-time highest LIVE-organism fitness is observed. This module just exposes
a small reader for the ``.org`` header so analysis scripts can show the
recorded fitness/genotype id without re-deriving anything from snapshots.
"""

from __future__ import annotations

import os
from typing import Dict


def read_champion_org_meta(path: str) -> Dict[str, str]:
    """Return ``# key: value`` metadata parsed from a champion ``.org`` header."""
    meta: Dict[str, str] = {}
    if not os.path.exists(path):
        return meta
    with open(path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip()
            if not line:
                break
            if not line.startswith("#"):
                break
            stripped = line[1:].strip()
            if ":" in stripped:
                key, val = stripped.split(":", 1)
                meta[key.strip()] = val.strip()
    return meta
