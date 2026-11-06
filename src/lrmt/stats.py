"""Statistical analysis, pattern classification, and LOLO regression for RQ1-RQ3.

Implements Stage 6 of the work plan:
- Indicator table assembly with both English pivot (`tok_parity`, `info_parity`) and
  Chinese pivot (`tok_parity_zh`, `info_parity_zh`) + Spearman rank correlation check.
- RQ1 aggregation across seeds (mean +/- SD for N <= 200), >= 2/3 seed significance
  rule from paired bootstrap resampling, and 4-way trajectory classification
  (lora_always / icl_always / crossover with interval / no_sig_diff).
- RQ2 standardized OLS & Ridge regressions (with/without zero-shot, English/Chinese
  pivot), VIF diagnostics, and 12-fold Leave-One-Language-Out (LOLO) cross-validation
  evaluated against majority-class and log(N)-only baselines.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from lrmt import config


def load_indicators(
    indicator_dir: Path | None = None,
    *,
    tgt_code: str = "cmn_Hans",
    include_heldout: bool = False,
) -> pd.DataFrame:
    """Merge tokenizer_parity, info_parity and zeroshot CSVs for source languages.

    Computes `tok_parity_zh` directly from the `cmn_Hans` row in `tokenizer_parity.csv`
    (Stage 2: Chinese total tokens / source total tokens).
    """
    ind_dir = indicator_dir or config.INDICATOR_DIR
    tok_path = ind_dir / "tokenizer_parity.csv"
    if not tok_path.is_file():
        raise FileNotFoundError(f"{tok_path} not found; run scripts/tokenizer_parity.py first")

    tok_df = pd.read_csv(tok_path)
    zh_rows = tok_df[tok_df["lang"] == "cmn_Hans"].set_index("model")["src_tokens"].to_dict()

    allowed_roles = {"source", "heldout"} if include_heldout else {"source"}
    df = tok_df[tok_df["role"].isin(allowed_roles)].copy()
    df = df.rename(columns={"parity": "tok_parity"})
    df["tok_parity_zh"] = [
        round(float(zh_rows[row["model"]]) / float(row["src_tokens"]), 4)
        if row["model"] in zh_rows else np.nan
        for _, row in df.iterrows()
    ]
    keep_cols = ["model", "lang", "name", "role", "tok_parity", "tok_parity_zh", "src_tokens_per_sent"]
    df = df[keep_cols].rename(columns={"lang": "src"})

    info_path = ind_dir / "info_parity.csv"
    if info_path.is_file():
        info_df = pd.read_csv(info_path)
        info_sub = info_df[["model", "lang", "info_parity", "info_parity_zh"]].rename(columns={"lang": "src"})
        df = df.merge(info_sub, on=["model", "src"], how="left")
    else:
        df["info_parity"] = np.nan
        df["info_parity_zh"] = np.nan

    zs_path = ind_dir / "zeroshot.csv"
    if zs_path.is_file():
        zs_df = pd.read_csv(zs_path)
        zs_sub = zs_df[zs_df["tgt"] == tgt_code][["model", "src", "chrf", "off_target_rate"]].rename(
            columns={"chrf": "zeroshot_chrf", "off_target_rate": "zeroshot_off_target"}
        )
        df = df.merge(zs_sub, on=["model", "src"], how="left")
    else:
        df["zeroshot_chrf"] = np.nan
        df["zeroshot_off_target"] = np.nan

    return df.reset_index(drop=True)


def pivot_spearman_summary(ind_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman rank correlations between English-pivot and Chinese-pivot indicators."""
    rows: list[dict[str, Any]] = []
    for model_key, grp in ind_df.groupby("model", sort=False):
        for col_en, col_zh, label in [
            ("tok_parity", "tok_parity_zh", "tokenizer_parity"),
            ("info_parity", "info_parity_zh", "info_parity"),
        ]:
            sub = grp[[col_en, col_zh]].dropna()
            if len(sub) >= 3:
                res = sp_stats.spearmanr(sub[col_en], sub[col_zh])
                rows.append({
                    "model": model_key,
                    "indicator": label,
                    "n_langs": len(sub),
                    "spearman_rho": round(float(res.statistic), 4),
                    "p_value": round(float(res.pvalue), 6),
                })
    return pd.DataFrame(rows)


def aggregate_rq1_observations(
    scores_df: pd.DataFrame,
    boot_df: pd.DataFrame | None = None,
    *,
    tgt_code: str = "cmn_Hans",
    k_mode: str = "best",
) -> pd.DataFrame:
    """Aggregate seed-level LoRA vs ICL scores into (model, src, tgt, n) observations.

    `k_mode`:
      - "best": at each (model, src, tgt, seed, n), select the k with highest chrF.
      - "k20": select fixed k = 20 (robustness check for test-set k selection).
    """
    if k_mode not in ("best", "k20"):
        raise ValueError(f"k_mode must be 'best' or 'k20', got {k_mode!r}")

    sub = scores_df[scores_df["tgt"] == tgt_code].copy()
    lora_df = sub[sub["strategy"] == "lora"].copy()
    icl_df = sub[sub["strategy"] == "icl"].copy()
    if lora_df.empty or icl_df.empty:
        return pd.DataFrame()

    lora_df["seed"] = lora_df["seed"].astype(int)
    lora_df["n"] = lora_df["n"].astype(int)
    icl_df["seed"] = icl_df["seed"].astype(int)
    icl_df["n"] = icl_df["n"].astype(int)
    icl_df["k"] = icl_df["k"].astype(int)

    if k_mode == "k20":
        icl_sel = icl_df[icl_df["k"] == 20].copy()
        if icl_sel.empty:
            icl_sel = icl_df.sort_values(["chrf", "k"], ascending=[False, True]).groupby(
                ["model", "src", "tgt", "seed", "n"], as_index=False
            ).first()
    else:
        icl_sel = (
            icl_df.sort_values(["chrf", "k"], ascending=[False, True])
            .groupby(["model", "src", "tgt", "seed", "n"], as_index=False)
            .first()
        )

    merged = lora_df.merge(
        icl_sel,
        on=["model", "src", "tgt", "seed", "n"],
        suffixes=("_lora", "_icl"),
    )
    merged["delta_chrf"] = merged["chrf_lora"] - merged["chrf_icl"]

    # Attach seed-level paired bootstrap direction (+1 / -1 / 0) when available.
    boot_lookup: dict[tuple[str, str, str, int, int], int] = {}
    if boot_df is not None and not boot_df.empty:
        b_sub = boot_df[(boot_df["tgt"] == tgt_code) & (boot_df["k_mode"] == k_mode)]
        for _, brow in b_sub.iterrows():
            key = (str(brow["model"]), str(brow["src"]), str(brow["tgt"]), int(brow["seed"]), int(brow["n"]))
            boot_lookup[key] = int(brow["direction"])

    merged["boot_dir"] = [
        boot_lookup.get((r["model"], r["src"], r["tgt"], int(r["seed"]), int(r["n"])), 0)
        for _, r in merged.iterrows()
    ]
    has_boot = bool(boot_lookup)

    obs_rows: list[dict[str, Any]] = []
    for (model, src, tgt, n), grp in merged.groupby(["model", "src", "tgt", "n"], sort=True):
        n_seeds = len(grp)
        lora_vals = grp["chrf_lora"].to_numpy(dtype=np.float64)
        icl_vals = grp["chrf_icl"].to_numpy(dtype=np.float64)
        delta_vals = grp["delta_chrf"].to_numpy(dtype=np.float64)
        k_vals = grp["k_icl"].astype(int).tolist()

        # Majority significance criterion: >= 2/3 of seeds significant in the same direction.
        req_seeds = math.ceil(2.0 * n_seeds / 3.0)
        pos_sig = int(np.sum(grp["boot_dir"] > 0))
        neg_sig = int(np.sum(grp["boot_dir"] < 0))
        if has_boot:
            if pos_sig >= req_seeds:
                sig_dir = 1
            elif neg_sig >= req_seeds:
                sig_dir = -1
            else:
                sig_dir = 0
        else:
            # Fallback when bootstrap table is absent: use >= 2/3 seed sign consistency & |mean delta| >= 1.0.
            pos_raw = int(np.sum(delta_vals > 0.5))
            neg_raw = int(np.sum(delta_vals < -0.5))
            sig_dir = 1 if pos_raw >= req_seeds else (-1 if neg_raw >= req_seeds else 0)

        obs_rows.append({
            "model": model,
            "src": src,
            "tgt": tgt,
            "n": int(n),
            "log_n": round(math.log10(int(n)), 4),
            "k_mode": k_mode,
            "n_seeds": n_seeds,
            "selected_k": max(set(k_vals), key=k_vals.count),
            "lora_chrf_mean": round(float(np.mean(lora_vals)), 4),
            "lora_chrf_sd": round(float(np.std(lora_vals, ddof=1)) if n_seeds > 1 else 0.0, 4),
            "icl_chrf_mean": round(float(np.mean(icl_vals)), 4),
            "icl_chrf_sd": round(float(np.std(icl_vals, ddof=1)) if n_seeds > 1 else 0.0, 4),
            "delta_chrf": round(float(np.mean(delta_vals)), 4),
            "delta_chrf_sd": round(float(np.std(delta_vals, ddof=1)) if n_seeds > 1 else 0.0, 4),
            "pos_sig_seeds": pos_sig,
            "neg_sig_seeds": neg_sig,
            "sig_dir": sig_dir,
        })

    return pd.DataFrame(obs_rows)


def _raw_trajectory_pattern(deltas_by_n: Mapping[int, float]) -> tuple[str, str, float | None]:
    """Classify continuous Delta(N) trajectory by sign and locate any crossover interval."""
    ns = sorted(deltas_by_n)
    vals = [float(deltas_by_n[n]) for n in ns]
    if all(v >= 0.0 for v in vals) and any(v > 0.0 for v in vals):
        return "lora_always", "", None
    if all(v <= 0.0 for v in vals) and any(v < 0.0 for v in vals):
        return "icl_always", "", None

    # Find the first adjacent interval where Delta changes sign.
    for n1, n2 in zip(ns[:-1], ns[1:]):
        v1, v2 = float(deltas_by_n[n1]), float(deltas_by_n[n2])
        if (v1 < 0.0 < v2) or (v1 > 0.0 > v2):
            interval = f"{n1}-{n2}"
            log_n1, log_n2 = math.log10(n1), math.log10(n2)
            frac = abs(v1) / (abs(v1) + abs(v2))
            cross_n = 10.0 ** (log_n1 + frac * (log_n2 - log_n1))
            return "crossover", interval, round(cross_n, 1)

    return "no_sig_diff", "", None


def classify_trajectory(
    deltas_by_n: Mapping[int, float],
    sig_dirs_by_n: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Classify a language's LoRA-vs-ICL trajectory across N into the 4 RQ1 patterns."""
    raw_pat, raw_int, cross_n = _raw_trajectory_pattern(deltas_by_n)
    if sig_dirs_by_n is None:
        return {
            "pattern": raw_pat,
            "crossover_interval": raw_int,
            "crossover_n_est": cross_n or "",
            "raw_pattern": raw_pat,
            "raw_interval": raw_int,
        }

    ns = sorted(deltas_by_n)
    sdirs = [int(sig_dirs_by_n.get(n, 0)) for n in ns]
    nonzero = [d for d in sdirs if d != 0]

    if not nonzero:
        sig_pat, sig_int = "no_sig_diff", ""
    elif all(d > 0 for d in nonzero) and raw_pat == "lora_always":
        sig_pat, sig_int = "lora_always", ""
    elif all(d < 0 for d in nonzero) and raw_pat == "icl_always":
        sig_pat, sig_int = "icl_always", ""
    elif raw_pat == "crossover":
        sig_pat, sig_int = "crossover", raw_int
    elif all(d > 0 for d in nonzero):
        sig_pat, sig_int = "lora_always", ""
    elif all(d < 0 for d in nonzero):
        sig_pat, sig_int = "icl_always", ""
    else:
        sig_pat, sig_int = "crossover", raw_int

    return {
        "pattern": sig_pat,
        "crossover_interval": sig_int,
        "crossover_n_est": cross_n if sig_pat == "crossover" and cross_n is not None else "",
        "raw_pattern": raw_pat,
        "raw_interval": raw_int,
    }


def build_rq1_pattern_table(obs_df: pd.DataFrame, ind_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Summarize each (model, src, tgt, k_mode) trajectory across N into one pattern row."""
    rows: list[dict[str, Any]] = []
    for (model, src, tgt, k_mode), grp in obs_df.groupby(["model", "src", "tgt", "k_mode"], sort=True):
        deltas = {int(r["n"]): float(r["delta_chrf"]) for _, r in grp.iterrows()}
        sdirs = {int(r["n"]): int(r["sig_dir"]) for _, r in grp.iterrows()}
        classified = classify_trajectory(deltas, sdirs)
        lang_obj = config.language(str(src))
        row: dict[str, Any] = {
            "model": model,
            "src": src,
            "name": lang_obj.name,
            "tgt": tgt,
            "k_mode": k_mode,
        }
        for n in sorted(deltas):
            sub_n = grp[grp["n"] == n].iloc[0]
            row[f"delta_N{n}"] = deltas[n]
            row[f"delta_sd_N{n}"] = float(sub_n["delta_chrf_sd"])
            row[f"sig_N{n}"] = sdirs[n]
        row.update(classified)
        rows.append(row)

    pat_df = pd.DataFrame(rows)
    if ind_df is not None and not ind_df.empty and not pat_df.empty:
        pat_df = pat_df.merge(
            ind_df[["model", "src", "tok_parity", "info_parity", "zeroshot_chrf"]],
            on=["model", "src"],
            how="left",
        )
        sort_col = "info_parity" if pat_df["info_parity"].notna().any() else "tok_parity"
        pat_df = pat_df.sort_values(["model", sort_col, "src"], ascending=[True, True, True]).reset_index(drop=True)
    return pat_df


def build_rq2_dataset(obs_df: pd.DataFrame, ind_df: pd.DataFrame) -> pd.DataFrame:
    """Merge RQ1 seed-averaged observations with source-language familiarity indicators."""
    source_codes = {lang.code for lang in config.source_languages(include_heldout=False)}
    df = obs_df[obs_df["src"].isin(source_codes)].merge(
        ind_df,
        on=["model", "src"],
        how="inner",
    )
    return df.reset_index(drop=True)


def compute_vif_table(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Compute Variance Inflation Factors (VIF) and pairwise correlations for predictors."""
    sub = df[list(cols)].dropna()
    if len(sub) < len(cols) + 1:
        return pd.DataFrame()
    scaled = (sub - sub.mean()) / sub.std(ddof=0).replace(0, 1.0)
    x_mat = sm.add_constant(scaled)
    rows: list[dict[str, Any]] = []
    for idx, col in enumerate(cols, start=1):
        vif_val = float(variance_inflation_factor(x_mat.values, idx))
        row: dict[str, Any] = {"variable": col, "vif": round(vif_val, 3)}
        for other in cols:
            r_val = float(sub[col].corr(sub[other], method="pearson"))
            row[f"r_{other}"] = round(r_val, 3)
        rows.append(row)
    return pd.DataFrame(rows)


def _design_matrix_from_scaled(
    scaled_base: pd.DataFrame,
    *,
    include_zeroshot: bool,
    interaction_mode: str,
) -> pd.DataFrame:
    """Construct predictor matrix including interaction(s) with log_n from z-scored base columns."""
    base_cols = ["tok_parity", "info_parity"] + (["zeroshot"] if include_zeroshot else []) + ["log_n"]
    x = scaled_base[base_cols].copy()
    if interaction_mode == "info_only":
        x["info_parity_x_log_n"] = x["info_parity"] * x["log_n"]
    elif interaction_mode == "all":
        x["tok_parity_x_log_n"] = x["tok_parity"] * x["log_n"]
        x["info_parity_x_log_n"] = x["info_parity"] * x["log_n"]
        if include_zeroshot:
            x["zeroshot_x_log_n"] = x["zeroshot"] * x["log_n"]
    else:
        raise ValueError(f"unknown interaction_mode {interaction_mode!r}")
    return x


def prepare_rq2_features(
    df: pd.DataFrame,
    *,
    pivot: str = "eng",
) -> pd.DataFrame:
    """Extract canonical predictor columns (`tok_parity`, `info_parity`, `zeroshot`, `log_n`)."""
    if pivot == "eng":
        tok_col, info_col = "tok_parity", "info_parity"
    elif pivot == "zh":
        tok_col, info_col = "tok_parity_zh", "info_parity_zh"
    else:
        raise ValueError(f"pivot must be 'eng' or 'zh', got {pivot!r}")

    out = pd.DataFrame({
        "model": df["model"],
        "src": df["src"],
        "tgt": df["tgt"],
        "n": df["n"].astype(int),
        "delta_chrf": df["delta_chrf"].astype(float),
        "tok_parity": df[tok_col].astype(float),
        "info_parity": df[info_col].astype(float),
        "zeroshot": df["zeroshot_chrf"].astype(float),
        "log_n": df["log_n"].astype(float),
    })
    return out.dropna().reset_index(drop=True)


def fit_rq2_full_models(
    feat_df: pd.DataFrame,
    *,
    include_zeroshot: bool = True,
    interaction_mode: str = "auto",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit full-sample standardized OLS and Ridge models and return coefficient tables."""
    if feat_df.empty:
        raise ValueError("feat_df is empty; cannot fit RQ2 regression")

    if interaction_mode == "auto":
        # Work plan Stage 6 rule: with 36 observations, retain only info_parity x log_n.
        interaction_mode = "info_only" if len(feat_df) <= 36 else "all"

    base_cols = ["tok_parity", "info_parity", "zeroshot", "log_n"]
    scaler = StandardScaler()
    scaled_arr = scaler.fit_transform(feat_df[base_cols])
    scaled_base = pd.DataFrame(scaled_arr, columns=base_cols, index=feat_df.index)

    x_df = _design_matrix_from_scaled(
        scaled_base, include_zeroshot=include_zeroshot, interaction_mode=interaction_mode
    )
    y = feat_df["delta_chrf"].to_numpy(dtype=np.float64)

    x_ols = sm.add_constant(x_df)
    ols_res = sm.OLS(y, x_ols).fit()

    ridge = RidgeCV(alphas=np.logspace(-2, 3, 50))
    ridge.fit(x_df.to_numpy(dtype=np.float64), y)
    ridge_coefs = {"const": float(ridge.intercept_), **dict(zip(x_df.columns, ridge.coef_))}

    coef_rows: list[dict[str, Any]] = []
    conf_int = ols_res.conf_int()
    for term in x_ols.columns:
        coef_rows.append({
            "term": term,
            "ols_coef": round(float(ols_res.params[term]), 4),
            "ols_se": round(float(ols_res.bse[term]), 4),
            "ols_t": round(float(ols_res.tvalues[term]), 3),
            "ols_p": round(float(ols_res.pvalues[term]), 4),
            "ci_low": round(float(conf_int.loc[term, 0]), 4),
            "ci_high": round(float(conf_int.loc[term, 1]), 4),
            "ridge_coef": round(float(ridge_coefs[term]), 4),
        })

    meta = {
        "n_obs": len(feat_df),
        "n_langs": feat_df["src"].nunique(),
        "include_zeroshot": include_zeroshot,
        "interaction_mode": interaction_mode,
        "r2": round(float(ols_res.rsquared), 4),
        "adj_r2": round(float(ols_res.rsquared_adj), 4),
        "f_pvalue": round(float(ols_res.f_pvalue), 6) if not np.isnan(ols_res.f_pvalue) else np.nan,
        "ridge_alpha": round(float(ridge.alpha_), 4),
        "scaler_mean": dict(zip(base_cols, scaler.mean_.tolist())),
        "scaler_scale": dict(zip(base_cols, scaler.scale_.tolist())),
        "ols_params": {k: float(v) for k, v in ols_res.params.items()},
    }
    return pd.DataFrame(coef_rows), meta


def run_lolo_cv(
    feat_df: pd.DataFrame,
    *,
    include_zeroshot: bool = True,
    estimator: str = "ols",
    interaction_mode: str = "auto",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run Leave-One-Language-Out (LOLO) cross-validation across all source languages."""
    if estimator not in ("ols", "ridge"):
        raise ValueError(f"estimator must be 'ols' or 'ridge', got {estimator!r}")
    if interaction_mode == "auto":
        interaction_mode = "info_only" if len(feat_df) <= 36 else "all"

    base_cols = ["tok_parity", "info_parity", "zeroshot", "log_n"]
    pred_rows: list[dict[str, Any]] = []

    for heldout_src in sorted(feat_df["src"].unique()):
        train_df = feat_df[feat_df["src"] != heldout_src].copy()
        test_df = feat_df[feat_df["src"] == heldout_src].copy()
        if train_df.empty or test_df.empty:
            continue

        scaler = StandardScaler()
        train_scaled = pd.DataFrame(
            scaler.fit_transform(train_df[base_cols]), columns=base_cols, index=train_df.index
        )
        test_scaled = pd.DataFrame(
            scaler.transform(test_df[base_cols]), columns=base_cols, index=test_df.index
        )

        x_train = _design_matrix_from_scaled(
            train_scaled, include_zeroshot=include_zeroshot, interaction_mode=interaction_mode
        )
        x_test = _design_matrix_from_scaled(
            test_scaled, include_zeroshot=include_zeroshot, interaction_mode=interaction_mode
        )
        y_train = train_df["delta_chrf"].to_numpy(dtype=np.float64)

        if estimator == "ols":
            ols_fit = sm.OLS(y_train, sm.add_constant(x_train, has_constant="add")).fit()
            y_pred = ols_fit.predict(sm.add_constant(x_test, has_constant="add")).to_numpy(dtype=np.float64)
        else:
            ridge = RidgeCV(alphas=np.logspace(-2, 3, 50))
            ridge.fit(x_train.to_numpy(dtype=np.float64), y_train)
            y_pred = ridge.predict(x_test.to_numpy(dtype=np.float64))

        # Baseline 1: majority class sign in training fold.
        pos_frac = float(np.mean(y_train >= 0.0))
        maj_sign = 1 if pos_frac >= 0.5 else -1

        # Baseline 2: log(N)-only OLS in training fold.
        logn_fit = sm.OLS(y_train, sm.add_constant(train_scaled[["log_n"]], has_constant="add")).fit()
        y_pred_logn = logn_fit.predict(
            sm.add_constant(test_scaled[["log_n"]], has_constant="add")
        ).to_numpy(dtype=np.float64)

        for idx, (_, row) in enumerate(test_df.iterrows()):
            obs_val = float(row["delta_chrf"])
            pred_val = float(y_pred[idx])
            logn_val = float(y_pred_logn[idx])
            obs_sign = 1 if obs_val >= 0.0 else -1
            pred_sign = 1 if pred_val >= 0.0 else -1
            logn_sign = 1 if logn_val >= 0.0 else -1

            pred_rows.append({
                "model": row["model"],
                "src": row["src"],
                "tgt": row["tgt"],
                "n": int(row["n"]),
                "obs_delta": round(obs_val, 4),
                "pred_delta": round(pred_val, 4),
                "pred_logn_only": round(logn_val, 4),
                "obs_sign": obs_sign,
                "pred_sign": pred_sign,
                "maj_sign": maj_sign,
                "logn_sign": logn_sign,
                "sign_correct": int(pred_sign == obs_sign),
                "maj_correct": int(maj_sign == obs_sign),
                "logn_correct": int(logn_sign == obs_sign),
            })

    pred_df = pd.DataFrame(pred_rows)
    if pred_df.empty:
        return pred_df, {}

    obs_arr = pred_df["obs_delta"].to_numpy(dtype=np.float64)
    pred_arr = pred_df["pred_delta"].to_numpy(dtype=np.float64)
    pearson_r = float(sp_stats.pearsonr(obs_arr, pred_arr).statistic) if len(obs_arr) >= 3 else np.nan
    spearman_rho = float(sp_stats.spearmanr(obs_arr, pred_arr).statistic) if len(obs_arr) >= 3 else np.nan
    rmse = float(np.sqrt(np.mean((obs_arr - pred_arr) ** 2)))
    mae = float(np.mean(np.abs(obs_arr - pred_arr)))

    # Pattern & crossover interval match rate across (model, src) trajectories.
    pat_matches = 0
    int_matches = 0
    n_trajectories = 0
    for (_, _), grp in pred_df.groupby(["model", "src"]):
        obs_map = {int(r["n"]): float(r["obs_delta"]) for _, r in grp.iterrows()}
        pred_map = {int(r["n"]): float(r["pred_delta"]) for _, r in grp.iterrows()}
        obs_pat, obs_int, _ = _raw_trajectory_pattern(obs_map)
        pred_pat, pred_int, _ = _raw_trajectory_pattern(pred_map)
        n_trajectories += 1
        if pred_pat == obs_pat:
            pat_matches += 1
        if (pred_pat == obs_pat) and (pred_int == obs_int):
            int_matches += 1

    summary = {
        "estimator": estimator,
        "include_zeroshot": include_zeroshot,
        "interaction_mode": interaction_mode,
        "n_obs": len(pred_df),
        "n_trajectories": n_trajectories,
        "sign_acc": round(float(pred_df["sign_correct"].mean()), 4),
        "majority_sign_acc": round(float(pred_df["maj_correct"].mean()), 4),
        "logn_only_sign_acc": round(float(pred_df["logn_correct"].mean()), 4),
        "pearson_r": round(pearson_r, 4),
        "spearman_rho": round(spearman_rho, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "pattern_match_rate": round(pat_matches / max(1, n_trajectories), 4),
        "pattern_and_interval_match_rate": round(int_matches / max(1, n_trajectories), 4),
    }
    return pred_df, summary


def predict_cross_target(
    train_feat_df: pd.DataFrame,
    test_feat_df: pd.DataFrame,
    *,
    include_zeroshot: bool = True,
    interaction_mode: str = "auto",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """RQ3 optional transfer: fit standardized OLS on Chinese target and predict English target Delta."""
    if interaction_mode == "auto":
        interaction_mode = "info_only" if len(train_feat_df) <= 36 else "all"

    base_cols = ["tok_parity", "info_parity", "zeroshot", "log_n"]
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_feat_df[base_cols]), columns=base_cols, index=train_feat_df.index
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_feat_df[base_cols]), columns=base_cols, index=test_feat_df.index
    )

    x_train = sm.add_constant(
        _design_matrix_from_scaled(train_scaled, include_zeroshot=include_zeroshot, interaction_mode=interaction_mode),
        has_constant="add",
    )
    x_test = sm.add_constant(
        _design_matrix_from_scaled(test_scaled, include_zeroshot=include_zeroshot, interaction_mode=interaction_mode),
        has_constant="add",
    )
    y_train = train_feat_df["delta_chrf"].to_numpy(dtype=np.float64)
    ols_fit = sm.OLS(y_train, x_train).fit()
    y_pred = ols_fit.predict(x_test).to_numpy(dtype=np.float64)

    out_df = test_feat_df[["model", "src", "tgt", "n", "delta_chrf"]].copy()
    out_df = out_df.rename(columns={"delta_chrf": "obs_delta"})
    out_df["pred_delta"] = np.round(y_pred, 4)
    out_df["sign_correct"] = (
        np.where(out_df["pred_delta"] >= 0, 1, -1) == np.where(out_df["obs_delta"] >= 0, 1, -1)
    ).astype(int)

    obs_arr = out_df["obs_delta"].to_numpy(dtype=np.float64)
    pearson_r = float(sp_stats.pearsonr(obs_arr, y_pred).statistic) if len(obs_arr) >= 3 else np.nan
    spearman_rho = float(sp_stats.spearmanr(obs_arr, y_pred).statistic) if len(obs_arr) >= 3 else np.nan
    summary = {
        "include_zeroshot": include_zeroshot,
        "interaction_mode": interaction_mode,
        "n_obs": len(out_df),
        "sign_acc": round(float(out_df["sign_correct"].mean()), 4),
        "pearson_r": round(pearson_r, 4),
        "spearman_rho": round(spearman_rho, 4),
        "rmse": round(float(np.sqrt(np.mean((obs_arr - y_pred) ** 2))), 4),
    }
    return out_df, summary
