"""RQ2 Analysis: Predicting Delta = chrF(LoRA) - chrF(ICL) from training-free indicators.

Addresses RQ2 (Stage 6 of the work plan):
1. Treats each (language, model, N) combination as one observation (36 points for
   Qwen3-8B, 72 points with Llama-3.1-8B).
2. Reports pairwise correlations & VIF among the 3 indicators, plus Spearman rank
   correlations between English-pivot and Chinese-pivot parity indicators.
3. Fits standardized OLS & Ridge regressions (with and without zero-shot chrF, and
   with English vs Chinese reference pivots). When n_obs <= 36, keeps only the
   `info_parity x log_n` interaction term.
4. Evaluates 12-fold Leave-One-Language-Out (LOLO) cross-validation against both
   majority-class and log(N)-only baselines: sign accuracy, Pearson r, Spearman rho,
   RMSE, and trajectory pattern & crossover interval match rates.

Usage:
    python analysis/rq2.py
    python analysis/rq2.py --tgt eng_Latn
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

from lrmt import config, stats

log = logging.getLogger("rq2")


def plot_rq2_diagnostics(
    lolo_df: pd.DataFrame,
    coef_df: pd.DataFrame,
    lolo_meta: dict[str, Any],
    *,
    tgt_code: str,
    out_dir: Path,
) -> list[Path]:
    """Plot Panel A (LOLO Predicted vs Observed Delta) and Panel B (Standardized OLS Coefficients)."""
    if lolo_df.empty or coef_df.empty:
        return []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # Panel A: LOLO Predicted vs Observed Delta
    obs = lolo_df["obs_delta"].to_numpy(dtype=np.float64)
    pred = lolo_df["pred_delta"].to_numpy(dtype=np.float64)
    lim = max(float(np.max(np.abs(np.concatenate([obs, pred])))) * 1.15, 2.0)

    ax1.axhline(0, color="#888888", linestyle="--", linewidth=1.0)
    ax1.axvline(0, color="#888888", linestyle="--", linewidth=1.0)
    ax1.plot([-lim, lim], [-lim, lim], color="#444444", linestyle=":", linewidth=1.2, label="Ideal (y = x)")

    markers = {50: "o", 200: "s", 997: "^"}
    colors = {50: "#1f77b8", 200: "#ff7f0e", 997: "#2ca02c"}
    for n_val, grp in lolo_df.groupby("n"):
        n_int = int(n_val)
        ax1.scatter(
            grp["obs_delta"], grp["pred_delta"],
            marker=markers.get(n_int, "o"), color=colors.get(n_int, "#1f77b4"),
            s=48, alpha=0.85, label=f"N = {n_int}",
        )

    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_xlabel("Observed $\\Delta$ = chrF(LoRA) - chrF(ICL)")
    ax1.set_ylabel("LOLO Predicted $\\hat{\\Delta}$")
    ax1.set_title(
        f"A. LOLO Cross-Validation ({tgt_code})\n"
        f"Sign Acc={lolo_meta.get('sign_acc', 0):.1%} (Maj={lolo_meta.get('majority_sign_acc', 0):.1%}), "
        f"r={lolo_meta.get('pearson_r', 0):.2f}, PatMatch={lolo_meta.get('pattern_match_rate', 0):.1%}",
        fontsize=10,
    )
    ax1.legend(loc="upper left", frameon=True, fontsize=8.5)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Panel B: Standardized OLS Coefficients with 95% CIs
    sub_coef = coef_df[coef_df["term"] != "const"].copy()
    y_pos = np.arange(len(sub_coef))
    coefs = sub_coef["ols_coef"].to_numpy(dtype=np.float64)
    ci_low = sub_coef["ci_low"].to_numpy(dtype=np.float64)
    ci_high = sub_coef["ci_high"].to_numpy(dtype=np.float64)
    err_left = coefs - ci_low
    err_right = ci_high - coefs

    ax2.axvline(0, color="#888888", linestyle="--", linewidth=1.0)
    ax2.errorbar(
        coefs, y_pos, xerr=[err_left, err_right],
        fmt="o", color="#d62728", markersize=6, linewidth=1.8, capsize=4, label="OLS (95% CI)",
    )
    ax2.scatter(sub_coef["ridge_coef"], y_pos, marker="x", color="#1f77b4", s=50, label="Ridge")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sub_coef["term"].tolist(), fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Standardized Coefficient")
    ax2.set_title(f"B. Standardized Predictors of $\\Delta$ ({tgt_code})", fontsize=10)
    ax2.legend(loc="best", frameon=True, fontsize=8.5)
    ax2.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for ext in ("pdf", "png"):
        p = out_dir / f"rq2_diagnostics_{tgt_code}.{ext}"
        fig.savefig(p, dpi=220, bbox_inches="tight")
        saved.append(p)
    plt.close(fig)
    return saved


def run_rq2(
    *,
    scores_path: Path,
    boot_path: Path | None,
    indicator_dir: Path,
    tgt_code: str,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not scores_path.is_file():
        raise FileNotFoundError(f"{scores_path} not found; run scripts/score.py first")

    scores_df = pd.read_csv(scores_path)
    boot_df = pd.read_csv(boot_path) if (boot_path and boot_path.is_file()) else None
    ind_df = stats.load_indicators(indicator_dir, tgt_code=tgt_code, include_heldout=False)

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pivot Spearman correlation check (English vs Chinese reference pivot).
    spearman_df = stats.pivot_spearman_summary(ind_df)
    if not spearman_df.empty:
        sp_path = tables_dir / "rq2_pivot_spearman.csv"
        spearman_df.to_csv(sp_path, index=False)
        log.info("wrote pivot Spearman correlations to %s", sp_path)

    # 2. VIF & indicator correlation table on unique (model, src) rows.
    vif_df = stats.compute_vif_table(ind_df, ["tok_parity", "info_parity", "zeroshot_chrf"])
    if not vif_df.empty:
        vif_path = tables_dir / f"rq2_vif_{tgt_code}.csv"
        vif_df.to_csv(vif_path, index=False)
        log.info("wrote VIF & correlation table to %s", vif_path)

    obs_df = stats.aggregate_rq1_observations(scores_df, boot_df, tgt_code=tgt_code, k_mode="best")
    if obs_df.empty:
        log.warning("no observations available for RQ2 (%s)", tgt_code)
        return pd.DataFrame(), pd.DataFrame()

    rq2_raw = stats.build_rq2_dataset(obs_df, ind_df)

    all_coefs: list[pd.DataFrame] = []
    lolo_summaries: list[dict[str, Any]] = []
    all_lolo_preds: list[pd.DataFrame] = []

    primary_lolo_df = pd.DataFrame()
    primary_coef_df = pd.DataFrame()
    primary_meta: dict[str, Any] = {}

    for pivot in ("eng", "zh"):
        feat_df = stats.prepare_rq2_features(rq2_raw, pivot=pivot)
        if len(feat_df) < 6:
            log.warning("insufficient non-null rows (%d) for RQ2 pivot=%s", len(feat_df), pivot)
            continue

        for include_zs in (True, False):
            coef_df, fit_meta = stats.fit_rq2_full_models(feat_df, include_zeroshot=include_zs)
            coef_df.insert(0, "pivot", pivot)
            coef_df.insert(1, "include_zeroshot", include_zs)
            coef_df["r2"] = fit_meta["r2"]
            coef_df["adj_r2"] = fit_meta["adj_r2"]
            all_coefs.append(coef_df)

            for estimator in ("ols", "ridge"):
                pred_df, lolo_sum = stats.run_lolo_cv(
                    feat_df, include_zeroshot=include_zs, estimator=estimator
                )
                if not lolo_sum:
                    continue
                lolo_row = {
                    "tgt": tgt_code,
                    "pivot": pivot,
                    **lolo_sum,
                    "full_r2": fit_meta["r2"],
                    "full_adj_r2": fit_meta["adj_r2"],
                }
                lolo_summaries.append(lolo_row)
                pred_df.insert(0, "pivot", pivot)
                pred_df.insert(1, "include_zeroshot", include_zs)
                pred_df.insert(2, "estimator", estimator)
                all_lolo_preds.append(pred_df)

                log.info(
                    "RQ2 LOLO [tgt=%s pivot=%-3s zs=%-5s %-5s]: sign_acc=%.3f (maj=%.3f logN=%.3f) r=%.3f pat_match=%.3f",
                    tgt_code, pivot, str(include_zs), estimator,
                    lolo_sum["sign_acc"], lolo_sum["majority_sign_acc"], lolo_sum["logn_only_sign_acc"],
                    lolo_sum["pearson_r"], lolo_sum["pattern_match_rate"],
                )

                if pivot == "eng" and include_zs and estimator == "ols":
                    primary_lolo_df = pred_df
                    primary_coef_df = coef_df
                    primary_meta = lolo_sum

    if not all_coefs:
        return pd.DataFrame(), pd.DataFrame()

    coefs_out = pd.concat(all_coefs, ignore_index=True)
    lolo_sum_out = pd.DataFrame(lolo_summaries)
    lolo_preds_out = pd.concat(all_lolo_preds, ignore_index=True)

    coefs_out.to_csv(tables_dir / f"rq2_coefficients_{tgt_code}.csv", index=False)
    lolo_sum_out.to_csv(tables_dir / f"rq2_lolo_summary_{tgt_code}.csv", index=False)
    lolo_preds_out.to_csv(tables_dir / f"rq2_lolo_predictions_{tgt_code}.csv", index=False)

    if not primary_lolo_df.empty:
        fig_paths = plot_rq2_diagnostics(
            primary_lolo_df, primary_coef_df, primary_meta,
            tgt_code=tgt_code, out_dir=out_dir / "figures",
        )
        if fig_paths:
            log.info("saved RQ2 diagnostic plot: %s", fig_paths[0])

    return coefs_out, lolo_sum_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, default=config.SCORE_DIR / "scores.csv")
    parser.add_argument("--bootstrap", type=Path, default=config.SCORE_DIR / "paired_bootstrap.csv")
    parser.add_argument("--indicators", type=Path, default=config.INDICATOR_DIR)
    parser.add_argument("--tgt", default="cmn_Hans", help="target language code (default: cmn_Hans)")
    parser.add_argument("--out-dir", type=Path, default=config.ANALYSIS_DIR)
    args = parser.parse_args()

    try:
        run_rq2(
            scores_path=args.scores,
            boot_path=args.bootstrap,
            indicator_dir=args.indicators,
            tgt_code=args.tgt,
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
