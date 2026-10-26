"""Tokenizer parity (Petrov et al., 2023) of every configured language against English.

parity = total English tokens / total tokens of the translation, over all sentence
pairs of the split; values below 1 mean the language is split into more tokens than
English. The mean of the per-sentence ratios is reported as a robustness check.
Only tokenizers are downloaded, so this runs on a CPU.

    python scripts/tokenizer_parity.py                    # main models, dev split
    python scripts/tokenizer_parity.py --models qwen3-8b
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from lrmt import config, data
from lrmt.config import Language

log = logging.getLogger("tokenizer_parity")

FIELDS = [
    "model", "split", "lang", "name", "role", "n_sents",
    "src_tokens", "pivot_tokens", "parity", "parity_sent_mean",
    "src_tokens_per_sent", "pivot_tokens_per_sent",
]


def count_tokens(tokenizer: PreTrainedTokenizerBase, texts: list[str]) -> list[int]:
    counts = [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]
    if min(counts) == 0:
        raise ValueError("a non-empty sentence was encoded to zero tokens")
    return counts


def parity_row(model: str, split: str, lang: Language, src: list[int], pivot: list[int]) -> dict:
    n = len(src)
    return {
        "model": model,
        "split": split,
        "lang": lang.code,
        "name": lang.name,
        "role": lang.role,
        "n_sents": n,
        "src_tokens": sum(src),
        "pivot_tokens": sum(pivot),
        "parity": round(sum(pivot) / sum(src), 4),
        "parity_sent_mean": round(statistics.fmean(p / s for p, s in zip(pivot, src)), 4),
        "src_tokens_per_sent": round(sum(src) / n, 2),
        "pivot_tokens_per_sent": round(sum(pivot) / n, 2),
    }


def main() -> int:
    splits = list(config.data_spec().splits)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=config.main_model_keys(), help="keys in configs/models.yaml")
    parser.add_argument("--split", default="dev", choices=splits)
    parser.add_argument("--out", type=Path, default=config.INDICATOR_DIR / "tokenizer_parity.csv")
    args = parser.parse_args()

    try:
        specs = [config.model_spec(key) for key in args.models]
    except KeyError as e:
        parser.error(str(e))

    # Load all tokenizers before computing anything, so a missing one fails fast.
    tokenizers = {}
    for spec in specs:
        try:
            tokenizers[spec.key] = AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.revision)
        except OSError as e:  # includes gated / missing repos
            log.error("cannot load tokenizer of %s (%s): %s", spec.key, spec.hf_id, str(e).splitlines()[0])
            log.error("request access on the Hub if the repo is gated, or restrict --models")
            return 1
        if not tokenizers[spec.key].is_fast:
            log.warning("%s: slow tokenizer in use", spec.key)

    pivot = config.pivot_code()
    langs = [lang for lang in config.languages() if lang.code != pivot]
    try:
        texts = data.load_parallel([pivot] + [lang.code for lang in langs], args.split)
    except data.DataError as e:
        log.error("%s", e)
        return 1

    rows = []
    for key, tokenizer in tokenizers.items():
        pivot_counts = count_tokens(tokenizer, texts[pivot])
        for lang in langs:
            rows.append(parity_row(key, args.split, lang, count_tokens(tokenizer, texts[lang.code]), pivot_counts))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    for key in tokenizers:
        log.info("%s (%s, %d sentences), sorted by parity:", key, args.split, len(texts[pivot]))
        for row in sorted((r for r in rows if r["model"] == key), key=lambda r: r["parity"]):
            log.info("  %-9s %-8s parity=%.3f  sent_mean=%.3f  tokens/sent=%.1f",
                     row["lang"], row["role"], row["parity"], row["parity_sent_mean"], row["src_tokens_per_sent"])
    log.info("wrote %d rows to %s", len(rows), args.out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)
    sys.exit(main())
