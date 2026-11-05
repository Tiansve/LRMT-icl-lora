"""Score translation output jsonl files and write summary CSV tables.

Scans outputs/icl/*.jsonl and outputs/lora/*.jsonl, writes all results to
scores/scores.csv, and extracts full-devtest k=0 runs to indicators/zeroshot.csv.

    python scripts/score.py
    python scripts/score.py --files outputs/icl/qwen3-8b_swh_Latn_cmn_Hans_k0.jsonl
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from lrmt import config, scoring

log = logging.getLogger("score")

SCORE_FIELDS = [
    "strategy", "model", "src", "tgt", "seed", "n", "k", "n_sents",
    "chrf", "bleu", "chrfpp", "off_target_rate", "rep_rate", "empty_rate",
    "mean_prompt_tokens", "max_prompt_tokens", "file",
]

ZEROSHOT_FIELDS = [
    "model", "src", "name", "role", "tgt", "n_sents",
    "chrf", "bleu", "chrfpp", "off_target_rate", "rep_rate",
    "mean_prompt_tokens", "max_prompt_tokens",
]


def discover_files() -> list[Path]:
    files: list[Path] = []
    for subdir in ("icl", "lora"):
        folder = config.OUTPUT_DIR / subdir
        if folder.is_dir():
            files.extend(sorted(folder.glob("*.jsonl")))
    return files


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--files", nargs="*", type=Path, help="specific jsonl files (default: all in outputs/)")
    parser.add_argument("--out", type=Path, default=config.SCORE_DIR / "scores.csv")
    parser.add_argument("--zeroshot-out", type=Path, default=None, help="default: indicators/zeroshot.csv on full scan")
    args = parser.parse_args()

    full_scan = args.files is None
    files = discover_files() if full_scan else args.files
    if not files:
        log.warning("no output jsonl files found to score")
        return 0

    score_rows: list[dict] = []
    zeroshot_rows: list[dict] = []
    for path in files:
        try:
            row = scoring.score_file(path)
        except Exception as e:
            log.error("failed to score %s: %s", path, e)
            return 1
        score_rows.append(row)
        log.info(
            "%-8s %-11s %s->%s seed=%-2s N=%-4s k=%-2s n=%-4d chrF=%6.2f BLEU=%6.2f chrF++=%6.2f off_tgt=%.3f",
            row["strategy"], row["model"], row["src"], row["tgt"],
            row["seed"] or "-", row["n"] or "-", row["k"] if row["k"] != "" else "-",
            row["n_sents"], row["chrf"], row["bleu"], row["chrfpp"], row["off_target_rate"],
        )

        if row["strategy"] == "zeroshot":
            lang = config.language(str(row["src"]))
            zeroshot_rows.append({
                "model": row["model"],
                "src": row["src"],
                "name": lang.name,
                "role": lang.role,
                "tgt": row["tgt"],
                "n_sents": row["n_sents"],
                "chrf": row["chrf"],
                "bleu": row["bleu"],
                "chrfpp": row["chrfpp"],
                "off_target_rate": row["off_target_rate"],
                "rep_rate": row["rep_rate"],
                "mean_prompt_tokens": row["mean_prompt_tokens"],
                "max_prompt_tokens": row["max_prompt_tokens"],
            })

    write_csv(args.out, SCORE_FIELDS, score_rows)
    log.info("wrote %d rows to %s", len(score_rows), args.out)

    zs_out = args.zeroshot_out or (config.INDICATOR_DIR / "zeroshot.csv" if full_scan else None)
    if zeroshot_rows and zs_out is not None:
        write_csv(zs_out, ZEROSHOT_FIELDS, zeroshot_rows)
        log.info("wrote %d zero-shot rows to %s", len(zeroshot_rows), zs_out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)
    sys.exit(main())
