"""RQ1 Analysis: LoRA vs ICL across data sizes N, pattern classification, and faceted plots.

Addresses RQ1 (Stage 6 of the work plan):
1. Computes Delta = chrF(LoRA) - chrF(ICL) for each (model, language, N), reporting
   mean +/- SD across the 3 seeds for N <= 200.
2. Uses paired bootstrap resampling (1,000 samples per seed; >= 2/3 seeds significant
   in the same direction).
3. Classifies each language into 4 patterns:
   - lora_always (LoRA performs better at all N)
   - icl_always (ICL performs better at all N)
   - crossover (better strategy switches between two tested N values; records interval)
   - no_sig_diff (no statistically significant difference)
4. Generates both the primary analysis (best k) and robustness check (fixed k = 20),
   plus the 12-panel faceted curve figure ordered by Information Parity.

Usage:
    python analysis/rq1.py
    python analysis/rq1.py --tgt eng_Latn
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lrmt import config, stats

log = logging.getLogger("rq1")

PATTERN_LABELS = {
    "lora_always": "LoRA > ICL",
    "icl_always": "ICL > LoRA",
    "crossover": "Crossover",
    "no_sig_diff": "No sig. diff.",
}


def plot_rq1_facets(
    obs_df: pd.DataFrame,
    pat_df: pd.DataFrame,
    ind_df: pd.DataFrame,
    *,
    model_key: str,
    tgt_code: str,
    k_mode: str,
    out_dir: Path,
) -> list[Path]:
    """Render a 3x4 grid of chrF vs N curves ordered by Information Parity."""
    m_obs = obs_df[(obs_df["model"] == model_key) & (obs_df["tgt"] == tgt_code) & (obs_df["k_mode"] == k_mode)]
    m_pat = pat_df[(pat_df["model"] == model_key) & (pat_df["tgt"] == tgt_code) & (pat_df["k_mode"] == k_mode)]
    m_ind = ind_df[ind_df["model"] == model_key].set_index("src")
    if m_obs.empty or m_pat.empty:
        return []

    langs_ordered = m_pat["src"].tolist()
    n_langs = len(langs_ordered)
    ncols = 4
    nrows = int(np.ceil(n_langs / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 3.1 * nrows), sharex=True, sharey=False)
    axes_flat = np.atleast_1d(axes).ravel()

    for idx, src_code in enumerate(langs_ordered):
        ax = axes_flat[idx]
        sub = m_obs[m_obs["src"] == src_code].sort_values("n")
        prow = m_pat[m_pat["src"] == src_code].iloc[0]

        ns = sub["n"].to_numpy(dtype=np.float64)
        lora_m = sub["lora_chrf_mean"].to_numpy(dtype=np.float64)
        lora_sd = sub["lora_chrf_sd"].to_numpy(dtype=np.float64)
        icl_m = sub["icl_chrf_mean"].to_numpy(dtype=np.float64)
        icl_sd = sub["icl_chrf_sd"].to_numpy(dtype=np.float64)
        sdirs = sub["sig_dir"].to_numpy(dtype=int)

        ax.errorbar(
            ns, icl_m, yerr=icl_sd, fmt="s--", color="#1f77b4",
            linewidth=1.6, markersize=5, capsize=3, label="ICL (best k)" if k_mode == "best" else "ICL (k=20)",
        )
        ax.errorbar(
            ns, lora_m, yerr=lora_sd, fmt="o-", color="#d62728",
            linewidth=1.6, markersize=5, capsize=3, label="QLoRA",
        )

        if src_code in m_ind.index and pd.notna(m_ind.loc[src_code, "zeroshot_chrf"]):
            zs_val = float(m_ind.loc[src_code, "zeroshot_chrf"])
            ax.axhline(zs_val, color="#7f7f7f", linestyle=":", linewidth=1.2, label="Zero-shot (k=0)")

        # Mark statistically significant N points with an asterisk above the higher curve.
        y_span = max(float(np.max(np.maximum(lora_m + lora_sd, icl_m + icl_sd))) - float(np.min(np.minimum(lora_m, icl_m))), 2.0)
        for n_val, lm, im, sd_flag in zip(ns, lora_m, icl_m, sdirs):
            if sd_flag != 0:
                y_top = max(lm, im) + 0.06 * y_span
                ax.text(n_val, y_top, "*", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#222222")

        pat_key = str(prow["pattern"])
        pat_str = PATTERN_LABELS.get(pat_key, pat_key)
        if pat_key == "crossover" and prow["crossover_interval"]:
            pat_str = f"Cross ({prow['crossover_interval']})"

        ip_val = prow.get("info_parity", np.nan)
        tp_val = prow.get("tok_parity", np.nan)
        metric_tag = f"IP={ip_val:.2f}" if pd.notna(ip_val) else (f"TP={tp_val:.2f}" if pd.notna(tp_val) else "")
        ax.set_title(f"{prow['name']} ({src_code.split('_')[0]})\n[{pat_str}] {metric_tag}", fontsize=9.5)

        ax.set_xscale("log")
        ax.set_xticks([50, 200, 997])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(True, linestyle="--", alpha=0.35)
        if idx % ncols == 0:
            ax.set_ylabel("chrF")
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Pool size N (log scale)")

    for extra_ax in axes_flat[n_langs:]:
        extra_ax.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"RQ1: QLoRA vs ICL across Data Sizes ({model_key} -> {tgt_code}, {k_mode})", fontsize=12, y=1.04)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for ext in ("pdf", "png"):
        p = out_dir / f"rq1_curves_{model_key}_{tgt_code}_{k_mode}.{ext}"
        fig.savefig(p, dpi=220, bbox_inches="tight")
        saved.append(p)
    plt.close(fig)
    return saved


def run_rq1(
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

    all_obs: list[pd.DataFrame] = []
    all_pats: list[pd.DataFrame] = []

    for k_mode in ("best", "k20"):
        obs_df = stats.aggregate_rq1_observations(scores_df, boot_df, tgt_code=tgt_code, k_mode=k_mode)
        if obs_df.empty:
            continue
        source_codes = set(ind_df["src"].unique())
        obs_df = obs_df[obs_df["src"].isin(source_codes)].reset_index(drop=True)
        pat_df = stats.build_rq1_pattern_table(obs_df, ind_df)
        all_obs.append(obs_df)
        all_pats.append(pat_df)

        for model_key in pat_df["model"].unique():
            fig_paths = plot_rq1_facets(
                obs_df, pat_df, ind_df,
                model_key=str(model_key), tgt_code=tgt_code, k_mode=k_mode,
                out_dir=out_dir / "figures",
            )
            if fig_paths:
                log.info("saved RQ1 plot: %s", fig_paths[0])

    if not all_obs:
        log.warning("no matching LoRA + ICL observations found for target %s", tgt_code)
        return pd.DataFrame(), pd.DataFrame()

    full_obs = pd.concat(all_obs, ignore_index=True)
    full_pats = pd.concat(all_pats, ignore_index=True)

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    obs_csv = tables_dir / f"rq1_observations_{tgt_code}.csv"
    pat_csv = tables_dir / f"rq1_patterns_{tgt_code}.csv"
    full_obs.to_csv(obs_csv, index=False)
    full_pats.to_csv(pat_csv, index=False)
    log.info("wrote %d RQ1 observation rows to %s", len(full_obs), obs_csv)
    log.info("wrote %d RQ1 pattern rows to %s", len(full_pats), pat_csv)

    for (model_key, k_mode), grp in full_pats.groupby(["model", "k_mode"]):
        counts = grp["pattern"].value_counts().to_dict()
        log.info("RQ1 patterns [%s, %s, %s]: %s", model_key, tgt_code, k_mode, counts)

    return full_obs, full_pats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, default=config.SCORE_DIR / "scores.csv")
    parser.add_argument("--bootstrap", type=Path, default=config.SCORE_DIR / "paired_bootstrap.csv")
    parser.add_argument("--indicators", type=Path, default=config.INDICATOR_DIR)
    parser.add_argument("--tgt", default="cmn_Hans", help="target language code (default: cmn_Hans)")
    parser.add_argument("--out-dir", type=Path, default=config.ANALYSIS_DIR)
    args = parser.parse_args()

    try:
        run_rq1(
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
