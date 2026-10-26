"""Nested data pools: which ids of the pool split each experiment may use.

Each seed shuffles the split's ids once and every pool is a prefix of that order,
so smaller pools are subsets of larger ones. Files keep the shuffled order and are
the source of truth; they are committed, so later runs never re-sample.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from lrmt.config import POOL_DIR, data_spec


def pool_path(seed: int, n: int) -> Path:
    return POOL_DIR / f"seed{seed}_N{n}.json"


def pool_grid() -> list[tuple[int, int]]:
    """All (seed, n) combinations; the full split is used with the first seed only."""
    spec = data_spec()
    full = spec.splits[spec.pools.split]
    return [
        (seed, n)
        for n in spec.pools.sizes
        for seed in (spec.pools.seeds[:1] if n == full else spec.pools.seeds)
    ]


def build_pools() -> dict[tuple[int, int], dict]:
    """Sample every pool of the grid; returns (seed, n) -> file payload."""
    spec = data_spec()
    total = spec.splits[spec.pools.split]
    pools = {}
    for seed, n in pool_grid():
        order = list(range(total))
        random.Random(seed).shuffle(order)
        pools[seed, n] = {"split": spec.pools.split, "seed": seed, "n": n, "ids": order[:n]}
    return pools


def load_pool(seed: int, n: int) -> list[int]:
    """Ids of one pool in sampling order; raise ValueError if the file is inconsistent."""
    if (seed, n) not in pool_grid():
        raise ValueError(f"(seed={seed}, N={n}) is not in the configured pool grid")
    path = pool_path(seed, n)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run scripts/make_pools.py first")
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    spec = data_spec()
    ids = payload["ids"]
    if (payload["split"], payload["seed"], payload["n"]) != (spec.pools.split, seed, n):
        raise ValueError(f"{path}: header does not match split/seed/N of its name")
    if len(ids) != n or len(set(ids)) != n:
        raise ValueError(f"{path}: expected {n} unique ids, got {len(set(ids))} unique of {len(ids)}")
    if not all(isinstance(i, int) and 0 <= i < spec.splits[spec.pools.split] for i in ids):
        raise ValueError(f"{path}: ids out of range")
    return ids
