"""Translation quality metrics and output file scoring.

Metrics (Stage 2 & Stage 6 of the work plan):
- chrf: primary metric, character n-gram F-score with word_order=0 (sacreBLEU).
- bleu: secondary metric, sacreBLEU with tokenize="zh" for Chinese and "13a" for English.
- chrfpp: secondary metric, chrF++ (word_order=2) after jieba segmentation for Chinese
  and standard whitespace tokenization for English.
- off_target_rate: proportion of outputs in the wrong target language (for Chinese,
  sentences where Han characters make up < 50% of alphabetic/ideographic characters).
- rep_rate: proportion of outputs with degenerate repetition (any character 6-gram
  repeated >= 4 times).
"""

from __future__ import annotations

import json
import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import jieba
import numpy as np
from sacrebleu.metrics import BLEU, CHRF

from lrmt import config

jieba.setLogLevel(60)  # suppress noisy jieba startup logs

_CHRF = CHRF(word_order=0)
_CHRFPP = CHRF(word_order=2)
_BLEU_ZH = BLEU(tokenize="zh")
_BLEU_13A = BLEU(tokenize="13a")

_ICL_K0_RE = re.compile(
    r"^(?P<model>[^_]+)_(?P<src>[a-z]{3}_[A-Za-z]{4})_(?P<tgt>[a-z]{3}_[A-Za-z]{4})_k0\.jsonl$"
)
_ICL_RE = re.compile(
    r"^(?P<model>[^_]+)_(?P<src>[a-z]{3}_[A-Za-z]{4})_(?P<tgt>[a-z]{3}_[A-Za-z]{4})"
    r"_seed(?P<seed>\d+)_N(?P<n>\d+)_k(?P<k>\d+)\.jsonl$"
)
_LORA_RE = re.compile(
    r"^(?P<model>[^_]+)_(?P<src>[a-z]{3}_[A-Za-z]{4})_(?P<tgt>[a-z]{3}_[A-Za-z]{4})"
    r"_seed(?P<seed>\d+)_N(?P<n>\d+)\.jsonl$"
)


@dataclass(frozen=True)
class RunMeta:
    strategy: str  # "zeroshot" | "icl" | "lora"
    model: str
    src: str
    tgt: str
    seed: int | None
    n: int | None
    k: int | None
    path: Path


def icl_output_path(
    model: str,
    src: str,
    tgt: str,
    *,
    k: int,
    seed: int | None = None,
    n: int | None = None,
    out_dir: Path | None = None,
) -> Path:
    base = out_dir or (config.OUTPUT_DIR / "icl")
    if k == 0:
        return base / f"{model}_{src}_{tgt}_k0.jsonl"
    if seed is None or n is None:
        raise ValueError("seed and n are required when k > 0")
    return base / f"{model}_{src}_{tgt}_seed{seed}_N{n}_k{k}.jsonl"


def lora_output_path(
    model: str,
    src: str,
    tgt: str,
    *,
    seed: int,
    n: int,
    out_dir: Path | None = None,
) -> Path:
    base = out_dir or (config.OUTPUT_DIR / "lora")
    return base / f"{model}_{src}_{tgt}_seed{seed}_N{n}.jsonl"


def parse_run_path(path: Path) -> RunMeta:
    """Extract run metadata from a standard output jsonl filename."""
    name = path.name
    if m := _ICL_K0_RE.match(name):
        return RunMeta("zeroshot", m["model"], m["src"], m["tgt"], None, None, 0, path)
    if m := _ICL_RE.match(name):
        return RunMeta("icl", m["model"], m["src"], m["tgt"], int(m["seed"]), int(m["n"]), int(m["k"]), path)
    if m := _LORA_RE.match(name):
        return RunMeta("lora", m["model"], m["src"], m["tgt"], int(m["seed"]), int(m["n"]), None, path)
    raise ValueError(f"unrecognized output filename format: {name}")


def is_han(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0xF900 <= cp <= 0xFAFF
        or 0x20000 <= cp <= 0x2FA1F
    )


def is_latin(ch: str) -> bool:
    return ("a" <= ch.lower() <= "z") or unicodedata.name(ch, "").startswith("LATIN")


def han_ratio(text: str) -> float:
    """Fraction of Han characters among all alphabetic/ideographic characters in text."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if is_han(c)) / len(alpha)


def latin_ratio(text: str) -> float:
    """Fraction of Latin letters among all alphabetic/ideographic characters in text."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if is_latin(c)) / len(alpha)


def is_off_target(text: str, tgt_code: str) -> bool:
    """True when target script ratio is below 50% (Stage 2: Han ratio < 50% for cmn_Hans)."""
    if tgt_code == "cmn_Hans":
        return han_ratio(text) < 0.5
    if tgt_code == "eng_Latn":
        return latin_ratio(text) < 0.5
    return not text.strip()


def has_repetition(text: str, *, ngram_size: int = 6, min_count: int = 4) -> bool:
    """Detect degenerate repetition where any character n-gram appears >= min_count times."""
    cleaned = " ".join(text.split())
    if len(cleaned) < ngram_size * min_count:
        return False
    counts = Counter(cleaned[i : i + ngram_size] for i in range(len(cleaned) - ngram_size + 1))
    return counts.most_common(1)[0][1] >= min_count


def _jieba_segment(texts: Sequence[str]) -> list[str]:
    return [" ".join(tok for tok in jieba.lcut(t) if tok.strip()) for t in texts]


def compute_metrics(
    hyps: Sequence[str],
    refs: Sequence[str],
    *,
    tgt_code: str = "cmn_Hans",
    prompt_tokens: Sequence[int] | None = None,
) -> dict[str, float | int]:
    """Compute corpus-level MT metrics and diagnostic error rates."""
    if len(hyps) != len(refs):
        raise ValueError(f"len(hyps)={len(hyps)} != len(refs)={len(refs)}")
    if not hyps:
        raise ValueError("cannot score an empty list of translations")

    hyp_list = list(hyps)
    ref_list = list(refs)
    n_sents = len(hyp_list)

    chrf = float(_CHRF.corpus_score(hyp_list, [ref_list]).score)
    bleu_metric = _BLEU_ZH if tgt_code == "cmn_Hans" else _BLEU_13A
    bleu = float(bleu_metric.corpus_score(hyp_list, [ref_list]).score)

    if tgt_code == "cmn_Hans":
        seg_hyps = _jieba_segment(hyp_list)
        seg_refs = _jieba_segment(ref_list)
        chrfpp = float(_CHRFPP.corpus_score(seg_hyps, [seg_refs]).score)
    else:
        chrfpp = float(_CHRFPP.corpus_score(hyp_list, [ref_list]).score)

    off_target = sum(1 for h in hyp_list if is_off_target(h, tgt_code)) / n_sents
    rep = sum(1 for h in hyp_list if has_repetition(h)) / n_sents
    empty = sum(1 for h in hyp_list if not h.strip()) / n_sents

    result: dict[str, float | int] = {
        "n_sents": n_sents,
        "chrf": round(chrf, 4),
        "bleu": round(bleu, 4),
        "chrfpp": round(chrfpp, 4),
        "off_target_rate": round(off_target, 4),
        "rep_rate": round(rep, 4),
        "empty_rate": round(empty, 4),
    }
    if prompt_tokens is not None and len(prompt_tokens) == n_sents:
        result["mean_prompt_tokens"] = round(statistics.fmean(prompt_tokens), 1)
        result["max_prompt_tokens"] = int(max(prompt_tokens))
    else:
        result["mean_prompt_tokens"] = 0.0
        result["max_prompt_tokens"] = 0
    return result


def read_output_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows.sort(key=lambda r: r["id"])
    return rows


def score_file(path: Path) -> dict[str, str | int | float]:
    """Read one output jsonl file and return a complete summary row."""
    meta = parse_run_path(path)
    rows = read_output_jsonl(path)
    hyps = [r["translation"] for r in rows]
    refs = [r["reference"] for r in rows]
    ptoks = [int(r["prompt_tokens"]) for r in rows if "prompt_tokens" in r]
    metrics = compute_metrics(
        hyps,
        refs,
        tgt_code=meta.tgt,
        prompt_tokens=ptoks if len(ptoks) == len(rows) else None,
    )
    return {
        "strategy": meta.strategy,
        "model": meta.model,
        "src": meta.src,
        "tgt": meta.tgt,
        "seed": "" if meta.seed is None else meta.seed,
        "n": "" if meta.n is None else meta.n,
        "k": "" if meta.k is None else meta.k,
        **metrics,
        "file": path.name,
    }


def sentence_chrf(hyp: str, ref: str) -> float:
    """Compute sentence-level chrF (word_order=0) for error analysis inspection."""
    return round(float(_CHRF.sentence_score(hyp, [ref]).score), 4)


def _chrf_segment_stats(hyps: Sequence[str], refs: Sequence[str]) -> np.ndarray:
    """Extract the (n_sents, 18) sufficient-statistics matrix for sacreBLEU chrF."""
    metric = CHRF(word_order=0, references=[list(refs)])
    raw_stats = metric._extract_corpus_statistics(list(hyps), None)
    return np.asarray(raw_stats, dtype=np.float64)


def paired_bootstrap_chrf(
    lora_hyps: Sequence[str],
    icl_hyps: Sequence[str],
    refs: Sequence[str],
    *,
    n_boot: int = 1000,
    seed: int = 1,
    alpha: float = 0.05,
) -> dict[str, float | int | bool]:
    """Paired bootstrap resampling (Koehn, 2004) for Delta = chrF(LoRA) - chrF(ICL)."""
    if not (len(lora_hyps) == len(icl_hyps) == len(refs)) or not refs:
        raise ValueError("lora_hyps, icl_hyps and refs must be non-empty and have equal length")

    m = len(refs)
    stats_lora = _chrf_segment_stats(lora_hyps, refs)
    stats_icl = _chrf_segment_stats(icl_hyps, refs)

    obs_lora = float(_CHRF._compute_f_score(stats_lora.sum(axis=0).tolist()))
    obs_icl = float(_CHRF._compute_f_score(stats_icl.sum(axis=0).tolist()))
    obs_delta = obs_lora - obs_icl

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, m, size=(n_boot, m))
    boot_lora_sums = stats_lora[boot_indices].sum(axis=1)
    boot_icl_sums = stats_icl[boot_indices].sum(axis=1)

    boot_deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        s_lora = float(_CHRF._compute_f_score(boot_lora_sums[b].tolist()))
        s_icl = float(_CHRF._compute_f_score(boot_icl_sums[b].tolist()))
        boot_deltas[b] = s_lora - s_icl

    # Two-sided paired bootstrap p-value: proportion of resamples where the sign flips or is 0.
    le_zero = int(np.sum(boot_deltas <= 0.0))
    ge_zero = int(np.sum(boot_deltas >= 0.0))
    p_one = min((le_zero + 1) / (n_boot + 1), (ge_zero + 1) / (n_boot + 1))
    p_value = min(1.0, 2.0 * p_one)

    ci_low, ci_high = np.percentile(boot_deltas, [100 * (alpha / 2), 100 * (1.0 - alpha / 2)])
    sig = bool(p_value < alpha and obs_delta != 0.0)
    direction = (1 if obs_delta > 0 else -1) if sig else 0

    return {
        "n_sents": m,
        "lora_chrf": round(obs_lora, 4),
        "icl_chrf": round(obs_icl, 4),
        "delta_chrf": round(obs_delta, 4),
        "ci_low": round(float(ci_low), 4),
        "ci_high": round(float(ci_high), 4),
        "p_value": round(float(p_value), 4),
        "significant": sig,
        "direction": direction,
    }

