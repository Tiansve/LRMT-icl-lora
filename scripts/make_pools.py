"""Write the nested data pools pools/seed{s}_N{n}.json (see configs/data.yaml).

Pools only contain sentence ids and are committed. Existing files are never changed
silently: identical ones are skipped, differing ones abort unless --force is given.

    python scripts/make_pools.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from lrmt import pools

log = logging.getLogger("make_pools")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="overwrite pools that differ from a fresh sample")
    args = parser.parse_args()

    built = pools.build_pools()
    conflicts = []
    for (seed, n), payload in built.items():
        path = pools.pool_path(seed, n)
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                if json.load(f) == payload:
                    log.info("unchanged %s", path.name)
                    continue
            if not args.force:
                conflicts.append(path.name)
                continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f)
            f.write("\n")
        log.info("wrote %s", path.name)
    if conflicts:
        log.error("existing pools differ from a fresh sample: %s; rerun with --force only if intended",
                  ", ".join(conflicts))
        return 1

    # Re-read what is on disk and check nesting within each seed.
    loaded = {key: pools.load_pool(*key) for key in built}
    for (seed, n), ids in loaded.items():
        for (seed2, n2), ids2 in loaded.items():
            if seed2 == seed and n2 > n and ids2[:n] != ids:
                log.error("seed %d: N%d is not a prefix of N%d", seed, n, n2)
                return 1
    log.info("%d pools valid and nested", len(loaded))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)
    sys.exit(main())
