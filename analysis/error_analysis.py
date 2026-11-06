"""Error Analysis: Full-dataset automatic error statistics and 50-sentence manual annotation sampling.

Addresses Error Analysis (Stage 6 of the work plan):
1. Summarizes automatic error indicators (`off_target_rate` via Han ratio < 50%,
   `rep_rate` via 6-gram repetition, `empty_rate`) across all languages, N, and strategies.
2. Samples 50 sentences per language for 4 representative languages spanning familiarity
   tiers (default: swh_Latn, kaz_Cyrl, amh_Ethi, sat_Olck) at model = qwen3-8b, N = 200,
   seed = 1, comparing Best ICL vs QLoRA.
3. Exports `analysis/tables/error_annotation_sheet.csv` with sentence-level chrF,
   automatic diagnostic flags, and annotation columns for the 5 error categories:
   `omission`, `named_entity`, `wrong_language`, `hallucination`, `repetition`.

Usage:
    python analysis/error_analysis.py
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from lrmt import config, scoring

log = logging.getLogger("error_analysis")

ANNOTATION_FIELDS = [
    "model", "src", "lang_name", "tgt", "n", "seed", "best_k", "sent_id",
    "source_text", "reference_text",
    "icl_translation", "lora_translation",
    "icl_chrf", "lora_chrf", "delta_chrf",
    "auto_icl_off_target", "auto_lora_off_target",
    "auto_icl_repetition", "auto_lora_repetition",
    "icl_omission", "icl_named_entity", "icl_wrong_language", "icl_hallucination", "icl_repetition",
    "lora_omission", "lora_named_entity", "lora_wrong_language", "lora_hallucination", "lora_repetition",
    "notes",
]


def summarize_auto_errors(scores_df: pd.DataFrame, *, out_dir: Path) -> pd.DataFrame:
    """Aggregate automatic error rates (off_target_rate, rep_rate, empty_rate) by strategy and N."""
    if scores_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (model, tgt, strategy, src), grp in scores_df.groupby(["model", "tgt", "strategy", "src"], sort=True):
        if strategy == "zeroshot":
            r = grp.iloc[0]
            rows.append({
                "model": model, "tgt": tgt, "src": src, "strategy": "zeroshot", "n": 0,
                "chrf_mean": round(float(r["chrf"]), 4),
                "off_target_rate": round(float(r["off_target_rate"]), 4),
                "rep_rate": round(float(r["rep_rate"]), 4),
                "empty_rate": round(float(r["empty_rate"]), 4),
            })
        elif strategy == "lora":
            for n_val, n_grp in grp.groupby("n"):
                rows.append({
                    "model": model, "tgt": tgt, "src": src, "strategy": "lora", "n": int(n_val),
                    "chrf_mean": round(float(n_grp["chrf"].mean()), 4),
                    "off_target_rate": round(float(n_grp["off_target_rate"].mean()), 4),
                    "rep_rate": round(float(n_grp["rep_rate"].mean()), 4),
                    "empty_rate": round(float(n_grp["empty_rate"].mean()), 4),
                })
        elif strategy == "icl":
            # Take best k per (seed, n) before averaging across seeds.
            best_k_grp = (
                grp.sort_values(["chrf", "k"], ascending=[False, True])
                .groupby(["seed", "n"], as_index=False)
                .first()
            )
            for n_val, n_grp in best_k_grp.groupby("n"):
                rows.append({
                    "model": model, "tgt": tgt, "src": src, "strategy": "icl_best", "n": int(n_val),
                    "chrf_mean": round(float(n_grp["chrf"].mean()), 4),
                    "off_target_rate": round(float(n_grp["off_target_rate"].mean()), 4),
                    "rep_rate": round(float(n_grp["rep_rate"].mean()), 4),
                    "empty_rate": round(float(n_grp["empty_rate"].mean()), 4),
                })

    auto_df = pd.DataFrame(rows)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tables_dir / "error_auto_summary.csv"
    auto_df.to_csv(out_csv, index=False)
    log.info("wrote %d automatic error summary rows to %s", len(auto_df), out_csv)
    return auto_df


def sample_manual_annotation_sheet(
    scores_df: pd.DataFrame,
    *,
    outputs_dir: Path,
    out_dir: Path,
    model_key: str = "qwen3-8b",
    tgt_code: str = "cmn_Hans",
    langs: list[str] | None = None,
    n_pool: int = 200,
    seed: int = 1,
    n_samples: int = 50,
) -> Path | None:
    """Sample 50 paired (Best ICL vs LoRA) outputs per language for manual error coding."""
    target_langs = langs or config.pilot_codes()
    sub_icl = scores_df[
        (scores_df["model"] == model_key)
        & (scores_df["tgt"] == tgt_code)
        & (scores_df["strategy"] == "icl")
        & (scores_df["n"] == n_pool)
        & (scores_df["seed"] == seed)
    ]

    sampled_rows: list[dict[str, Any]] = []
    for src_code in target_langs:
        lang_icl = sub_icl[sub_icl["src"] == src_code]
        if lang_icl.empty:
            continue
        best_row = lang_icl.sort_values(["chrf", "k"], ascending=[False, True]).iloc[0]
        best_k = int(best_row["k"])

        icl_path = scoring.icl_output_path(
            model_key, src_code, tgt_code, k=best_k, seed=seed, n=n_pool, out_dir=outputs_dir / "icl"
        )
        lora_path = scoring.lora_output_path(
            model_key, src_code, tgt_code, seed=seed, n=n_pool, out_dir=outputs_dir / "lora"
        )
        if not (icl_path.is_file() and lora_path.is_file()):
            log.info("output jsonl files for %s (N=%d, seed=%d) not present locally; skipping sheet export", src_code, n_pool, seed)
            continue

        icl_records = scoring.read_output_jsonl(icl_path)
        lora_records = scoring.read_output_jsonl(lora_path)
        if len(icl_records) != len(lora_records):
            continue

        total_sents = len(icl_records)
        rng = random.Random(f"error_sample:{src_code}:seed={seed}")
        chosen_ids = sorted(rng.sample(range(total_sents), k=min(n_samples, total_sents)))
        lang_name = config.language(src_code).name

        for sid in chosen_ids:
            rec_i = icl_records[sid]
            rec_l = lora_records[sid]
            hyp_i = str(rec_i["translation"])
            hyp_l = str(rec_l["translation"])
            ref_t = str(rec_i["reference"])
            chrf_i = scoring.sentence_chrf(hyp_i, ref_t)
            chrf_l = scoring.sentence_chrf(hyp_l, ref_t)

            sampled_rows.append({
                "model": model_key,
                "src": src_code,
                "lang_name": lang_name,
                "tgt": tgt_code,
                "n": n_pool,
                "seed": seed,
                "best_k": best_k,
                "sent_id": sid,
                "source_text": rec_i["src"],
                "reference_text": ref_t,
                "icl_translation": hyp_i,
                "lora_translation": hyp_l,
                "icl_chrf": chrf_i,
                "lora_chrf": chrf_l,
                "delta_chrf": round(chrf_l - chrf_i, 4),
                "auto_icl_off_target": int(scoring.is_off_target(hyp_i, tgt_code)),
                "auto_lora_off_target": int(scoring.is_off_target(hyp_l, tgt_code)),
                "auto_icl_repetition": int(scoring.has_repetition(hyp_i)),
                "auto_lora_repetition": int(scoring.has_repetition(hyp_l)),
                "icl_omission": "", "icl_named_entity": "", "icl_wrong_language": "", "icl_hallucination": "", "icl_repetition": "",
                "lora_omission": "", "lora_named_entity": "", "lora_wrong_language": "", "lora_hallucination": "", "lora_repetition": "",
                "notes": "",
            })

    if not sampled_rows:
        return None

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = tables_dir / "error_annotation_sheet.csv"
    with open(sheet_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows(sampled_rows)
    log.info("wrote %d manual error annotation rows to %s", len(sampled_rows), sheet_path)
    return sheet_path


def run_error_analysis(
    *,
    scores_path: Path,
    outputs_dir: Path,
    out_dir: Path,
    model_key: str = "qwen3-8b",
    tgt_code: str = "cmn_Hans",
    langs: list[str] | None = None,
    n_pool: int = 200,
    seed: int = 1,
    n_samples: int = 50,
) -> pd.DataFrame:
    if not scores_path.is_file():
        raise FileNotFoundError(f"{scores_path} not found; run scripts/score.py first")

    scores_df = pd.read_csv(scores_path)
    auto_df = summarize_auto_errors(scores_df, out_dir=out_dir)
    sample_manual_annotation_sheet(
        scores_df,
        outputs_dir=outputs_dir,
        out_dir=out_dir,
        model_key=model_key,
        tgt_code=tgt_code,
        langs=langs,
        n_pool=n_pool,
        seed=seed,
        n_samples=n_samples,
    )
    return auto_df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, default=config.SCORE_DIR / "scores.csv")
    parser.add_argument("--outputs", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=config.ANALYSIS_DIR)
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--tgt", default="cmn_Hans")
    parser.add_argument("--langs", nargs="*", default=None, help="default: 4 pilot languages")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=50)
    args = parser.parse_args()

    try:
        run_error_analysis(
            scores_path=args.scores,
            outputs_dir=args.outputs,
            out_dir=args.out_dir,
            model_key=args.model,
            tgt_code=args.tgt,
            langs=args.langs,
            n_pool=args.n,
            seed=args.seed,
            n_samples=args.n_samples,
        )
    except Exception as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)
    sys.exit(main())
