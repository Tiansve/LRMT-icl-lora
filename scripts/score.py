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

BOOTSTRAP_FIELDS = [
    "model", "src", "tgt", "seed", "n", "k_mode", "icl_k", "n_sents",
    "lora_chrf", "icl_chrf", "delta_chrf", "ci_low", "ci_high",
    "p_value", "significant", "direction",
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
    parser.add_argument("--bootstrap-out", type=Path, default=None, help="default: scores/paired_bootstrap.csv on full scan")
    parser.add_argument("--n-boot", type=int, default=1000, help="paired bootstrap resamples (default: 1000)")
    args = parser.parse_args()

    full_scan = args.files is None
    files = discover_files() if full_scan else args.files
    if not files:
        log.warning("no output jsonl files found to score")
        return 0

    score_rows: list[dict] = []
    zeroshot_rows: list[dict] = []
    path_by_name: dict[str, Path] = {p.name: p for p in files}
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

    # Compute paired bootstrap tests whenever matching LoRA and ICL files exist.
    boot_out = args.bootstrap_out or (config.SCORE_DIR / "paired_bootstrap.csv" if full_scan else None)
    lora_by_key = {
        (r["model"], r["src"], r["tgt"], int(r["seed"]), int(r["n"])): r
        for r in score_rows
        if r["strategy"] == "lora"
    }
    icl_by_key: dict[tuple, list[dict]] = {}
    for r in score_rows:
        if r["strategy"] == "icl":
            key = (r["model"], r["src"], r["tgt"], int(r["seed"]), int(r["n"]))
            icl_by_key.setdefault(key, []).append(r)

    boot_rows: list[dict] = []
    if boot_out is not None and lora_by_key and icl_by_key:
        for key, lora_row in sorted(lora_by_key.items()):
            icl_candidates = icl_by_key.get(key)
            if not icl_candidates:
                continue
            model, src, tgt, seed, n = key
            lora_data = scoring.read_output_jsonl(path_by_name[str(lora_row["file"])])
            lora_hyps = [r["translation"] for r in lora_data]
            refs = [r["reference"] for r in lora_data]

            # 1. Primary analysis: best k on test set (highest chrF, tie-break by smaller k).
            best_icl = max(icl_candidates, key=lambda r: (float(r["chrf"]), -int(r["k"])))
            modes: list[tuple[str, dict]] = [("best", best_icl)]
            # 2. Robustness check: fixed k = 20.
            k20_matches = [r for r in icl_candidates if int(r["k"]) == 20]
            if k20_matches:
                modes.append(("k20", k20_matches[0]))

            for k_mode, icl_row in modes:
                icl_data = scoring.read_output_jsonl(path_by_name[str(icl_row["file"])])
                if len(icl_data) != len(lora_data):
                    continue
                icl_hyps = [r["translation"] for r in icl_data]
                res = scoring.paired_bootstrap_chrf(
                    lora_hyps, icl_hyps, refs, n_boot=args.n_boot, seed=seed
                )
                boot_rows.append({
                    "model": model, "src": src, "tgt": tgt, "seed": seed, "n": n,
                    "k_mode": k_mode, "icl_k": int(icl_row["k"]),
                    **res,
                })

        if boot_rows:
            write_csv(boot_out, BOOTSTRAP_FIELDS, boot_rows)
            log.info("wrote %d paired bootstrap rows to %s", len(boot_rows), boot_out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)
    sys.exit(main())
