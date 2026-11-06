"""Master runner for Stage 6 statistical analysis and plots (RQ1, RQ2, RQ3, Error Analysis).

Also provides `--self-test` to synthesize a realistic multi-language dataset in a
temporary directory and verify the entire RQ1-RQ3 + plotting + error analysis pipeline
end-to-end before cluster runs finish.

Usage:
    python analysis/run_all.py
    python analysis/run_all.py --self-test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import error_analysis  # noqa: E402
import rq1  # noqa: E402
import rq2  # noqa: E402
import rq3  # noqa: E402
from lrmt import config, pools  # noqa: E402

log = logging.getLogger("run_all")


def _generate_synthetic_fixture(base_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Create a realistic synthetic fixture (indicators, scores, bootstrap, sample jsonls) for self-test."""
    ind_dir = base_dir / "indicators"
    score_dir = base_dir / "scores"
    out_dir = base_dir / "outputs"
    ind_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    # Copy real tokenizer_parity.csv into fixture.
    shutil.copy2(config.INDICATOR_DIR / "tokenizer_parity.csv", ind_dir / "tokenizer_parity.csv")
    tok_df = pd.read_csv(ind_dir / "tokenizer_parity.csv")

    rng = np.random.default_rng(42)
    info_rows = []
    zs_rows = []
    score_rows = []
    boot_rows = []

    for _, row in tok_df.iterrows():
        if row["lang"] == "cmn_Hans":
            continue
        model = str(row["model"])
        src = str(row["lang"])
        tp = float(row["parity"])
        ip = round(min(0.85, max(0.08, tp * 1.15 + float(rng.normal(0, 0.02)))), 4)
        ip_zh = round(ip * 1.08, 4)
        info_rows.append({
            "model": model, "split": "dev", "lang": src, "name": row["name"], "role": row["role"],
            "n_sents": 997, "info_parity": ip, "info_parity_zh": ip_zh,
            "info_parity_corpus": ip, "info_parity_zh_corpus": ip_zh,
            "src_nll_mean": round(120.0 / ip, 2), "eng_nll_mean": 120.0, "zh_nll_mean": 129.6,
        })

        for tgt in ("cmn_Hans", "eng_Latn"):
            if model != "qwen3-8b" and tgt == "eng_Latn":
                continue
            zs_chrf = round(max(1.5, 42.0 * (ip ** 1.3) + (6.0 if tgt == "eng_Latn" else 0.0) + float(rng.normal(0, 1.0))), 4)
            off_tgt = round(max(0.0, 0.55 - ip), 4)
            zs_rows.append({
                "model": model, "src": src, "name": row["name"], "role": row["role"], "tgt": tgt,
                "n_sents": 1012, "chrf": zs_chrf, "bleu": round(zs_chrf * 0.35, 4),
                "chrfpp": round(zs_chrf * 0.95, 4), "off_target_rate": off_tgt, "rep_rate": 0.02,
                "mean_prompt_tokens": 120.0, "max_prompt_tokens": 220,
            })
            score_rows.append({
                "strategy": "zeroshot", "model": model, "src": src, "tgt": tgt,
                "seed": "", "n": "", "k": 0, "n_sents": 1012,
                "chrf": zs_chrf, "bleu": round(zs_chrf * 0.35, 4), "chrfpp": round(zs_chrf * 0.95, 4),
                "off_target_rate": off_tgt, "rep_rate": 0.02, "empty_rate": 0.0,
                "mean_prompt_tokens": 120.0, "max_prompt_tokens": 220,
                "file": f"{model}_{src}_{tgt}_k0.jsonl",
            })

            if row["role"] != "source":
                continue

            for seed, n in pools.pool_grid():
                log_n = math.log10(n)
                # Realistic ground-truth relation: low IP -> ICL wins; high IP & larger N -> LoRA wins.
                icl_base = zs_chrf + 6.5 + 2.2 * log_n
                true_delta = -6.0 + 14.0 * (ip - 0.28) + 3.2 * (log_n - 2.0) + 4.5 * (ip - 0.28) * (log_n - 2.0)
                noise = float(rng.normal(0, 0.45))
                lora_chrf = round(max(2.0, icl_base + true_delta + noise), 4)

                score_rows.append({
                    "strategy": "lora", "model": model, "src": src, "tgt": tgt,
                    "seed": seed, "n": n, "k": "", "n_sents": 1012,
                    "chrf": lora_chrf, "bleu": round(lora_chrf * 0.38, 4), "chrfpp": round(lora_chrf * 0.96, 4),
                    "off_target_rate": round(off_tgt * 0.2, 4), "rep_rate": 0.01, "empty_rate": 0.0,
                    "mean_prompt_tokens": 120.0, "max_prompt_tokens": 220,
                    "file": f"{model}_{src}_{tgt}_seed{seed}_N{n}.jsonl",
                })

                best_icl_chrf = -1.0
                for k in (5, 20, 50):
                    k_bonus = {5: -1.1, 20: 0.0, 50: 0.4}[k]
                    icl_val = round(max(2.0, icl_base + k_bonus + float(rng.normal(0, 0.2))), 4)
                    best_icl_chrf = max(best_icl_chrf, icl_val)
                    score_rows.append({
                        "strategy": "icl", "model": model, "src": src, "tgt": tgt,
                        "seed": seed, "n": n, "k": k, "n_sents": 1012,
                        "chrf": icl_val, "bleu": round(icl_val * 0.36, 4), "chrfpp": round(icl_val * 0.95, 4),
                        "off_target_rate": round(off_tgt * 0.1, 4), "rep_rate": 0.01, "empty_rate": 0.0,
                        "mean_prompt_tokens": float(k * 110), "max_prompt_tokens": k * 160,
                        "file": f"{model}_{src}_{tgt}_seed{seed}_N{n}_k{k}.jsonl",
                    })

                delta = round(lora_chrf - best_icl_chrf, 4)
                sig = abs(delta) >= 0.85
                direction = (1 if delta > 0 else -1) if sig else 0
                for k_mode, k_used in [("best", 50), ("k20", 20)]:
                    boot_rows.append({
                        "model": model, "src": src, "tgt": tgt, "seed": seed, "n": n,
                        "k_mode": k_mode, "icl_k": k_used, "n_sents": 1012,
                        "lora_chrf": lora_chrf, "icl_chrf": best_icl_chrf, "delta_chrf": delta,
                        "ci_low": round(delta - 0.7, 4), "ci_high": round(delta + 0.7, 4),
                        "p_value": 0.004 if sig else 0.32, "significant": sig, "direction": direction,
                    })

    pd.DataFrame(info_rows).to_csv(ind_dir / "info_parity.csv", index=False)
    pd.DataFrame(zs_rows).to_csv(ind_dir / "zeroshot.csv", index=False)
    scores_csv = score_dir / "scores.csv"
    boot_csv = score_dir / "paired_bootstrap.csv"
    pd.DataFrame(score_rows).to_csv(scores_csv, index=False)
    pd.DataFrame(boot_rows).to_csv(boot_csv, index=False)

    # Create dummy jsonls for pilot languages at N=200 seed=1 to test error_analysis sheet sampling.
    (out_dir / "icl").mkdir(parents=True, exist_ok=True)
    (out_dir / "lora").mkdir(parents=True, exist_ok=True)
    for src in config.pilot_codes():
        icl_p = out_dir / "icl" / f"qwen3-8b_{src}_cmn_Hans_seed1_N200_k50.jsonl"
        lora_p = out_dir / "lora" / f"qwen3-8b_{src}_cmn_Hans_seed1_N200.jsonl"
        with open(icl_p, "w", encoding="utf-8") as fi, open(lora_p, "w", encoding="utf-8") as fl:
            for sid in range(60):
                rec_i = {"id": sid, "src": f"src {sid}", "translation": f"ICL译文第{sid}句", "reference": f"参考译文第{sid}句", "prompt_tokens": 500}
                rec_l = {"id": sid, "src": f"src {sid}", "translation": f"LoRA译文第{sid}句", "reference": f"参考译文第{sid}句", "prompt_tokens": 100}
                fi.write(json.dumps(rec_i, ensure_ascii=False) + "\n")
                fl.write(json.dumps(rec_l, ensure_ascii=False) + "\n")

    return scores_csv, boot_csv, ind_dir, out_dir


def run_pipeline(
    *,
    scores_path: Path,
    boot_path: Path,
    indicator_dir: Path,
    outputs_dir: Path,
    out_dir: Path,
) -> None:
    log.info("=== Running RQ1 (Chinese main target) ===")
    rq1.run_rq1(
        scores_path=scores_path,
        boot_path=boot_path,
        indicator_dir=indicator_dir,
        tgt_code="cmn_Hans",
        out_dir=out_dir,
    )

    log.info("=== Running RQ2 (Regression, VIF & 12-fold LOLO CV) ===")
    rq2.run_rq2(
        scores_path=scores_path,
        boot_path=boot_path,
        indicator_dir=indicator_dir,
        tgt_code="cmn_Hans",
        out_dir=out_dir,
    )

    log.info("=== Running RQ3 (English control & cross-target transfer) ===")
    rq3.run_rq3(
        scores_path=scores_path,
        boot_path=boot_path,
        indicator_dir=indicator_dir,
        out_dir=out_dir,
    )

    log.info("=== Running Error Analysis (Auto summary & 50-sentence sampling) ===")
    error_analysis.run_error_analysis(
        scores_path=scores_path,
        outputs_dir=outputs_dir,
        out_dir=out_dir,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, default=config.SCORE_DIR / "scores.csv")
    parser.add_argument("--bootstrap", type=Path, default=config.SCORE_DIR / "paired_bootstrap.csv")
    parser.add_argument("--indicators", type=Path, default=config.INDICATOR_DIR)
    parser.add_argument("--outputs", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=config.ANALYSIS_DIR)
    parser.add_argument("--self-test", action="store_true", help="run end-to-end validation on synthetic data")
    args = parser.parse_args()

    if args.self_test:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            scores_csv, boot_csv, ind_dir, out_dir_sim = _generate_synthetic_fixture(tmp_path)
            analysis_out = tmp_path / "analysis"
            run_pipeline(
                scores_path=scores_csv,
                boot_path=boot_csv,
                indicator_dir=ind_dir,
                outputs_dir=out_dir_sim,
                out_dir=analysis_out,
            )
            generated_tables = sorted((analysis_out / "tables").glob("*.csv"))
            generated_figs = sorted((analysis_out / "figures").glob("*.*"))
            log.info("Self-test succeeded! Generated %d tables and %d figure files:", len(generated_tables), len(generated_figs))
            for p in generated_tables + generated_figs:
                log.info("  %s (%d bytes)", p.name, p.stat().st_size)
        return 0

    try:
        run_pipeline(
            scores_path=args.scores,
            boot_path=args.bootstrap,
            indicator_dir=args.indicators,
            outputs_dir=args.outputs,
            out_dir=args.out_dir,
        )
    except Exception as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)
    sys.exit(main())
