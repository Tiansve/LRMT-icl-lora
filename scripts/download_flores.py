"""Download the FLORES+ subsets listed in configs/languages.yaml and validate them.

Requires access to the gated dataset: `hf auth login` once, or set HF_TOKEN.

    python scripts/download_flores.py
"""

from __future__ import annotations

import argparse
import logging
import sys

from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

from lrmt import config, data

log = logging.getLogger("download_flores")


def main() -> int:
    all_codes = [lang.code for lang in config.languages()]
    splits = list(config.data_spec().splits)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--langs", nargs="+", default=all_codes, help="FLORES+ codes (default: all configured)")
    parser.add_argument("--splits", nargs="+", default=splits, choices=splits)
    args = parser.parse_args()

    try:
        data.download(args.langs, args.splits)
    except GatedRepoError:
        log.error("no access to %s: accept its terms on the Hub, then `hf auth login` or set HF_TOKEN",
                  config.data_spec().repo_id)
        return 1
    except HfHubHTTPError as e:
        log.error("download failed: %s", e)
        return 1

    pivot = config.pivot_code()
    codes = [pivot] + [c for c in args.langs if c != pivot]
    for split in args.splits:
        try:
            texts = data.load_parallel(codes, split)
        except data.DataError as e:
            log.error("validation failed: %s", e)
            return 1
        log.info("%s: %d languages x %d sentences, NFC-normalized and aligned with %s",
                 split, len(texts), len(texts[pivot]), pivot)
    log.info("data in %s (revision %s)", config.DATA_DIR, config.data_spec().revision)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)
    sys.exit(main())
