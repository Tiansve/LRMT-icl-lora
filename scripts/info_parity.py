"""Compute Information Parity (Tsvetkov & Kipnis, 2024) on the dev split.

For each sentence pair i in dev (997 sentences):
- Sentence NLL = sum of negative log-probabilities of all tokens of the bare sentence
  conditioned on a leading BOS / document-start token (no chat template, no instruction,
  bf16 weights without quantization).
- info_parity = mean_i(NLL(eng_i) / NLL(src_i))      [main indicator, English pivot]
- info_parity_zh = mean_i(NLL(cmn_i) / NLL(src_i))   [robustness check, Chinese pivot]
Corpus-level ratios (sum NLL_pivot / sum NLL_src) are also recorded alongside mean NLLs.

Examples:
    # UPPMAX Pelle (both main models, full dev split):
    python scripts/info_parity.py

    # Single model or pilot subset:
    python scripts/info_parity.py --models qwen3-8b --langs pilot

    # Local CPU smoke test with Qwen3-0.6B on 10 sentences:
    python scripts/info_parity.py --models qwen3-0.6b --langs swh_Latn sat_Olck --limit 10 --out indicators/smoke_info_parity.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

from lrmt import config, data, generate
from lrmt.config import Language, ModelSpec

log = logging.getLogger("info_parity")

ZH_CODE = "cmn_Hans"
FIELDS = [
    "model", "split", "lang", "name", "role", "n_sents",
    "info_parity", "info_parity_zh",
    "info_parity_corpus", "info_parity_zh_corpus",
    "src_nll_mean", "eng_nll_mean", "zh_nll_mean",
]


def prefix_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    """Return the BOS token ID, or `<|endoftext|>` for tokenizers without explicit BOS (e.g. Qwen3)."""
    if tokenizer.bos_token_id is not None:
        return int(tokenizer.bos_token_id)
    eot_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if isinstance(eot_id, int) and eot_id != tokenizer.unk_token_id:
        return eot_id
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    raise ValueError("tokenizer has no bos_token_id, <|endoftext|>, or eos_token_id")


def encode_bare_with_bos(tokenizer: PreTrainedTokenizerBase, texts: Sequence[str]) -> list[list[int]]:
    bos_id = prefix_token_id(tokenizer)
    encoded: list[list[int]] = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError("a non-empty sentence encoded to zero tokens")
        encoded.append([bos_id] + ids)
    return encoded


@torch.inference_mode()
def compute_nlls_transformers(
    spec: ModelSpec,
    tokenizer: PreTrainedTokenizerBase,
    texts_by_lang: dict[str, list[str]],
    *,
    batch_size: int | None = None,
) -> dict[str, list[float]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    bs = batch_size or (8 if device == "cuda" else 1)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id, revision=spec.revision, dtype=torch_dtype, local_files_only=True
        )
    except OSError:
        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id, revision=spec.revision, dtype=torch_dtype
        )
    model = model.to(device)
    model.eval()

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    nlls_by_lang: dict[str, list[float]] = {}

    for code, texts in texts_by_lang.items():
        seqs = encode_bare_with_bos(tokenizer, texts)
        lang_nlls: list[float] = []
        for start in range(0, len(seqs), bs):
            chunk = seqs[start : start + bs]
            max_len = max(len(s) for s in chunk)
            input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros((len(chunk), max_len), dtype=torch.long, device=device)
            for row_idx, s in enumerate(chunk):
                input_ids[row_idx, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
                attention_mask[row_idx, : len(s)] = 1

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            # Shift so position t predicts token t+1; mask out padding positions.
            shift_logits = logits[:, :-1, :].float()
            shift_labels = input_ids[:, 1:]
            shift_mask = attention_mask[:, 1:].float()

            token_nll = F.cross_entropy(
                shift_logits.transpose(1, 2), shift_labels, reduction="none"
            )
            sent_nll = (token_nll * shift_mask).sum(dim=1).cpu().tolist()
            lang_nlls.extend(float(v) for v in sent_nll)
        nlls_by_lang[code] = lang_nlls
        log.info("  %s %-9s mean_NLL=%.2f", spec.key, code, statistics.fmean(lang_nlls))

    return nlls_by_lang


def compute_nlls_vllm(
    spec: ModelSpec,
    tokenizer: PreTrainedTokenizerBase,
    texts_by_lang: dict[str, list[str]],
) -> dict[str, list[float]]:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=spec.hf_id,
        revision=spec.revision,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    nlls_by_lang: dict[str, list[float]] = {}

    for code, texts in texts_by_lang.items():
        seqs = encode_bare_with_bos(tokenizer, texts)
        prompts = [TokensPrompt(prompt_token_ids=s) for s in seqs]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        lang_nlls: list[float] = []
        for s, out in zip(seqs, outputs):
            plogprobs: Any = out.prompt_logprobs
            if plogprobs is None or len(plogprobs) != len(s):
                raise RuntimeError(f"vLLM did not return full prompt_logprobs for {code}")
            nll = 0.0
            for pos in range(1, len(s)):
                tok_id = s[pos]
                entry = plogprobs[pos]
                if entry is None or tok_id not in entry:
                    raise RuntimeError(f"missing logprob for token {tok_id} at pos {pos}")
                nll -= float(entry[tok_id].logprob)
            lang_nlls.append(nll)
        nlls_by_lang[code] = lang_nlls
        log.info("  %s %-9s mean_NLL=%.2f", spec.key, code, statistics.fmean(lang_nlls))

    return nlls_by_lang


def build_row(
    model_key: str,
    split: str,
    lang: Language,
    src_nlls: list[float],
    eng_nlls: list[float],
    zh_nlls: list[float],
) -> dict[str, str | int | float]:
    if min(src_nlls) <= 0:
        raise ValueError(f"{lang.code}: encountered non-positive sentence NLL")
    n = len(src_nlls)
    sent_ratios_eng = [e / s for e, s in zip(eng_nlls, src_nlls)]
    sent_ratios_zh = [z / s for z, s in zip(zh_nlls, src_nlls)]
    return {
        "model": model_key,
        "split": split,
        "lang": lang.code,
        "name": lang.name,
        "role": lang.role,
        "n_sents": n,
        "info_parity": round(statistics.fmean(sent_ratios_eng), 4),
        "info_parity_zh": round(statistics.fmean(sent_ratios_zh), 4),
        "info_parity_corpus": round(sum(eng_nlls) / sum(src_nlls), 4),
        "info_parity_zh_corpus": round(sum(zh_nlls) / sum(src_nlls), 4),
        "src_nll_mean": round(statistics.fmean(src_nlls), 2),
        "eng_nll_mean": round(statistics.fmean(eng_nlls), 2),
        "zh_nll_mean": round(statistics.fmean(zh_nlls), 2),
    }


def resolve_target_langs(tokens: list[str]) -> list[Language]:
    pivot = config.pivot_code()
    if tokens == ["all"]:
        return [lang for lang in config.languages() if lang.code != pivot]
    if tokens == ["pilot"]:
        return [config.language(c) for c in config.pilot_codes()]
    return [config.language(c) for c in tokens if c != pivot]


def main() -> int:
    splits = list(config.data_spec().splits)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=config.main_model_keys(), help="keys in configs/models.yaml")
    parser.add_argument("--langs", nargs="+", default=["all"], help="language codes, 'all' or 'pilot'")
    parser.add_argument("--split", default="dev", choices=splits)
    parser.add_argument("--limit", type=int, default=None, help="limit to first LIMIT sentences (for smoke tests)")
    parser.add_argument("--backend", choices=["auto", "vllm", "transformers"], default="auto")
    parser.add_argument("--out", type=Path, default=config.INDICATOR_DIR / "info_parity.csv")
    args = parser.parse_args()

    try:
        specs = [config.model_spec(k) for k in args.models]
        langs = resolve_target_langs(args.langs)
    except KeyError as e:
        parser.error(str(e))

    pivot = config.pivot_code()
    required_codes = list(dict.fromkeys([pivot, ZH_CODE] + [lang.code for lang in langs]))
    texts_all = data.load_parallel(required_codes, args.split)
    n_total = len(texts_all[pivot])
    n_eval = min(args.limit, n_total) if args.limit is not None else n_total
    texts = {code: sents[:n_eval] for code, sents in texts_all.items()}

    backend = args.backend
    if backend == "auto":
        try:
            import vllm  # noqa: F401

            backend = "vllm" if torch.cuda.is_available() else "transformers"
        except ImportError:
            backend = "transformers"

    rows: list[dict[str, str | int | float]] = []
    for spec in specs:
        log.info("computing NLLs for %s (%s, %d sentences, backend=%s)...", spec.key, args.split, n_eval, backend)
        tokenizer = generate.load_tokenizer(spec)
        if backend == "vllm":
            nlls = compute_nlls_vllm(spec, tokenizer, texts)
        else:
            nlls = compute_nlls_transformers(spec, tokenizer, texts)

        eng_nlls = nlls[pivot]
        zh_nlls = nlls[ZH_CODE]
        for lang in langs:
            rows.append(build_row(spec.key, args.split, lang, nlls[lang.code], eng_nlls, zh_nlls))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    log.info("wrote %d rows to %s", len(rows), args.out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)
    sys.exit(main())
