"""RQ3 Analysis: Comparing Chinese main target (`cmn_Hans`) against English control (`eng_Latn`).

Addresses RQ3 (Stage 6 of the work plan):
1. Repeats RQ1 and RQ2 analyses on Qwen3-8B English target results (`eng_Latn`).
2. Compares the direction and strength of indicator-Delta correlations and standardized
   regression coefficients between Chinese and English targets.
3. Evaluates optional cross-target transfer: fitting the standardized OLS regression
   on Chinese (`cmn_Hans`) and directly predicting English (`eng_Latn`) Delta.

Usage:
    python analysis/rq3.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq1  # noqa: E402
import rq2  # noqa: E402
from lrmt import config, stats  # noqa: E402

log = logging.getLogger("rq3")


def plot_zh_vs_en_coefficients(
    comp_df: pd.DataFrame,
    *,
    out_dir: Path,
) -> list[Path]:
    """Side-by-side horizontal bar chart of standardized OLS coefficients for Chinese vs English."""
    sub = comp_df[comp_df["term"] != "const"].copy()
    if sub.empty:
        return []

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    y = np.arange(len(sub))
    height = 0.35

    ax.axvline(0, color="#888888", linestyle="--", linewidth=1.0)
    ax.barh(y - height / 2, sub["coef_cmn_Hans"], height=height, color="#d62728", alpha=0.85, label="Chinese (cmn_Hans)")
    ax.barh(y + height / 2, sub["coef_eng_Latn"], height=height, color="#1f77b4", alpha=0.85, label="English (eng_Latn)")

    ax.set_yticks(y)
    ax.set_yticklabels(sub["term"].tolist(), fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Standardized OLS Coefficient")
    ax.set_title("RQ3: Predictor Direction & Strength (Chinese vs. English Target)", fontsize=10.5)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for ext in ("pdf", "png"):
        p = out_dir / f"rq3_zh_vs_en_coefficients.{ext}"
        fig.savefig(p, dpi=220, bbox_inches="tight")
        saved.append(p)
    plt.close(fig)
    return saved


def run_rq3(
    *,
    scores_path: Path,
    boot_path: Path | None,
    indicator_dir: Path,
    out_dir: Path,
    model_key: str = "qwen3-8b",
) -> pd.DataFrame:
    if not scores_path.is_file():
        raise FileNotFoundError(f"{scores_path} not found; run scripts/score.py first")

    scores_df = pd.read_csv(scores_path)
    if "eng_Latn" not in set(scores_df["tgt"].unique()):
        log.warning("no eng_Latn rows in %s; skipping RQ3 comparison", scores_path)
        return pd.DataFrame()

    # Run RQ1 and RQ2 on eng_Latn first.
    rq1.run_rq1(
        scores_path=scores_path,
        boot_path=boot_path,
        indicator_dir=indicator_dir,
        tgt_code="eng_Latn",
        out_dir=out_dir,
    )
    rq2.run_rq2(
        scores_path=scores_path,
        boot_path=boot_path,
        indicator_dir=indicator_dir,
        tgt_code="eng_Latn",
        out_dir=out_dir,
    )

    boot_df = pd.read_csv(boot_path) if (boot_path and boot_path.is_file()) else None
    obs_zh = stats.aggregate_rq1_observations(scores_df, boot_df, tgt_code="cmn_Hans", k_mode="best")
    obs_en = stats.aggregate_rq1_observations(scores_df, boot_df, tgt_code="eng_Latn", k_mode="best")
    obs_zh = obs_zh[obs_zh["model"] == model_key]
    obs_en = obs_en[obs_en["model"] == model_key]
    if obs_zh.empty or obs_en.empty:
        log.warning("missing %s observations for cmn_Hans or eng_Latn", model_key)
        return pd.DataFrame()

    ind_zh = stats.load_indicators(indicator_dir, tgt_code="cmn_Hans", include_heldout=False)
    ind_en = stats.load_indicators(indicator_dir, tgt_code="eng_Latn", include_heldout=False)

    feat_zh = stats.prepare_rq2_features(stats.build_rq2_dataset(obs_zh, ind_zh), pivot="eng")
    feat_en = stats.prepare_rq2_features(stats.build_rq2_dataset(obs_en, ind_en), pivot="eng")
    if len(feat_zh) < 6 or len(feat_en) < 6:
        log.warning("insufficient rows for RQ3 comparison (zh=%d, en=%d)", len(feat_zh), len(feat_en))
        return pd.DataFrame()

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    comp_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []

    for include_zs in (True, False):
        coef_zh, meta_zh = stats.fit_rq2_full_models(feat_zh, include_zeroshot=include_zs, interaction_mode="info_only")
        coef_en, meta_en = stats.fit_rq2_full_models(feat_en, include_zeroshot=include_zs, interaction_mode="info_only")

        merged_coef = coef_zh.merge(coef_en, on="term", suffixes=("_cmn_Hans", "_eng_Latn"))
        for _, r in merged_coef.iterrows():
            c_zh = float(r["ols_coef_cmn_Hans"])
            c_en = float(r["ols_coef_eng_Latn"])
            comp_rows.append({
                "include_zeroshot": include_zs,
                "term": r["term"],
                "coef_cmn_Hans": c_zh,
                "p_cmn_Hans": float(r["ols_p_cmn_Hans"]),
                "coef_eng_Latn": c_en,
                "p_eng_Latn": float(r["ols_p_eng_Latn"]),
                "same_direction": bool((c_zh >= 0 and c_en >= 0) or (c_zh < 0 and c_en < 0)),
                "r2_cmn_Hans": meta_zh["r2"],
                "r2_eng_Latn": meta_en["r2"],
            })

        # Optional cross-target transfer: fit on cmn_Hans, predict eng_Latn.
        _, trans_sum = stats.predict_cross_target(
            feat_zh, feat_en, include_zeroshot=include_zs, interaction_mode="info_only"
        )
        transfer_rows.append(trans_sum)
        log.info(
            "RQ3 cross-target transfer (zh->en, include_zeroshot=%s): sign_acc=%.3f r=%.3f rho=%.3f rmse=%.2f",
            include_zs, trans_sum["sign_acc"], trans_sum["pearson_r"], trans_sum["spearman_rho"], trans_sum["rmse"],
        )

    comp_df = pd.DataFrame(comp_rows)
    trans_df = pd.DataFrame(transfer_rows)
    comp_df.to_csv(tables_dir / "rq3_target_comparison.csv", index=False)
    trans_df.to_csv(tables_dir / "rq3_cross_target_transfer.csv", index=False)
    log.info("wrote RQ3 comparison table to %s", tables_dir / "rq3_target_comparison.csv")

    primary_comp = comp_df[comp_df["include_zeroshot"] == True]  # noqa: E712
    fig_paths = plot_zh_vs_en_coefficients(primary_comp, out_dir=out_dir / "figures")
    if fig_paths:
        log.info("saved RQ3 comparison plot: %s", fig_paths[0])

    return comp_df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, default=config.SCORE_DIR / "scores.csv")
    parser.add_argument("--bootstrap", type=Path, default=config.SCORE_DIR / "paired_bootstrap.csv")
    parser.add_argument("--indicators", type=Path, default=config.INDICATOR_DIR)
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--out-dir", type=Path, default=config.ANALYSIS_DIR)
    args = parser.parse_args()

    try:
        run_rq3(
            scores_path=args.scores,
            boot_path=args.bootstrap,
            indicator_dir=args.indicators,
            out_dir=args.out_dir,
            model_key=args.model,
        )
    except Exception as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)
    sys.exit(main())
