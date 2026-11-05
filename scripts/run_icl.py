"""Run zero-shot (k=0) and In-Context Learning (k>0) translation experiments.

Loads the model once and iterates over the requested (src, pool, k) combinations,
skipping runs whose output jsonl file is already complete unless --force is given.
Use --dry-run-tokens to measure exact prompt token lengths on CPU using only the
tokenizer and BM25 index (no model weights loaded).

Examples:
    # Zero-shot indicator for all 13 languages (12 sources + held-out Zulu):
    python scripts/run_icl.py --model qwen3-8b --k 0

    # Pilot 4 languages, all 7 pools, k = 5 / 20 / 50:
    python scripts/run_icl.py --model qwen3-8b --langs pilot --k 5 20 50

    # Local CPU smoke test (Qwen3-0.6B, 10 sentences, k = 0 and 5):
    python scripts/run_icl.py --model qwen3-0.6b --langs swh_Latn sat_Olck --pools 1:50 --k 0 5 --limit 10 --out-dir outputs/smoke

    # Check exact k=50 prompt token lengths on full devtest (CPU, tokenizer only):
    python scripts/run_icl.py --model qwen3-8b --langs sat_Olck dzo_Tibt --pools 1:997 --k 50 --dry-run-tokens
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

from lrmt import config, data, generate, pools, prompt, retrieval, scoring

log = logging.getLogger("run_icl")


def resolve_langs(tokens: list[str], *, include_heldout_default: bool) -> list[str]:
    if tokens == ["all"]:
        return [lang.code for lang in config.source_languages(include_heldout=include_heldout_default)]
    if tokens == ["all-with-heldout"]:
        return [lang.code for lang in config.source_languages(include_heldout=True)]
    if tokens == ["sources"]:
        return [lang.code for lang in config.source_languages(include_heldout=False)]
    if tokens == ["pilot"]:
        return config.pilot_codes()
    for code in tokens:
        config.language(code)
    return tokens


def parse_pools(tokens: list[str] | None) -> list[tuple[int, int]]:
    if not tokens:
        return pools.pool_grid()
    result: list[tuple[int, int]] = []
    grid = set(pools.pool_grid())
    for item in tokens:
        try:
            seed_s, n_s = item.split(":", 1)
            pair = (int(seed_s), int(n_s))
        except ValueError as e:
            raise ValueError(f"invalid pool spec {item!r}; expected SEED:N, e.g. 1:200") from e
        if pair not in grid:
            raise ValueError(f"pool {pair} is not in configured pool grid {sorted(grid)}")
        result.append(pair)
    return result


def is_complete(path: Path, expected_rows: int) -> bool:
    if not path.is_file():
        return False
    try:
        rows = scoring.read_output_jsonl(path)
    except Exception:
        return False
    return len(rows) == expected_rows


def build_batch_messages(
    src_code: str,
    tgt_code: str,
    test_src: list[str],
    *,
    k: int,
    seed: int | None = None,
    retriever: retrieval.BM25Retriever | None = None,
) -> list[list[dict[str, str]]]:
    messages_list: list[list[dict[str, str]]] = []
    for sent_id, src_text in enumerate(test_src):
        if k == 0:
            examples: list[tuple[str, str]] = []
        else:
            assert retriever is not None and seed is not None
            examples = retriever.retrieve_pairs(src_text, k, seed=seed, query_id=sent_id)
        messages_list.append(prompt.build_messages(src_code, tgt_code, src_text, examples))
    return messages_list


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model key in configs/models.yaml")
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["all"],
        help="source language codes, or 'all' / 'sources' / 'all-with-heldout' / 'pilot'",
    )
    parser.add_argument("--tgt", default="cmn_Hans", help="target language code (default: cmn_Hans)")
    parser.add_argument("--k", nargs="+", type=int, default=[5, 20, 50], help="shot counts (use 0 for zero-shot)")
    parser.add_argument("--pools", nargs="*", help="subset of SEED:N pools (default: all 7 configured pools)")
    parser.add_argument("--limit", type=int, default=None, help="only run on the first LIMIT devtest sentences")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--backend", choices=["auto", "vllm", "transformers"], default="auto")
    parser.add_argument("--out-dir", type=Path, default=None, help="override output directory (default: outputs/icl)")
    parser.add_argument("--dry-run-tokens", action="store_true", help="only compute prompt token lengths (CPU)")
    parser.add_argument("--force", action="store_true", help="overwrite existing complete output files")
    args = parser.parse_args()

    try:
        spec = config.model_spec(args.model)
        config.language(args.tgt)
        only_zeroshot = all(k == 0 for k in args.k)
        src_codes = resolve_langs(args.langs, include_heldout_default=only_zeroshot)
        pool_list = parse_pools(args.pools) if any(k > 0 for k in args.k) else []
    except (KeyError, ValueError) as e:
        parser.error(str(e))

    if any(k < 0 for k in args.k):
        parser.error("--k values must be >= 0")

    # Load parallel data upfront so any missing file fails before loading the model.
    all_codes = [args.tgt] + [c for c in src_codes if c != args.tgt]
    devtest_texts = data.load_parallel(all_codes, "devtest")
    dev_texts = data.load_parallel(all_codes, "dev") if pool_list else {}

    n_total = len(devtest_texts[args.tgt])
    n_eval = min(args.limit, n_total) if args.limit is not None else n_total
    test_refs = devtest_texts[args.tgt][:n_eval]

    # Build the schedule of (src_code, seed, n, k, out_path) runs.
    # Group by (src_code, seed, n) so the BM25 index is built once across k = 5, 20, 50.
    schedule: list[tuple[str, int | None, int | None, int, Path]] = []
    k_zero = [k for k in args.k if k == 0]
    k_pos = [k for k in args.k if k > 0]
    for src_code in src_codes:
        for _ in k_zero:
            out_path = scoring.icl_output_path(spec.key, src_code, args.tgt, k=0, out_dir=args.out_dir)
            schedule.append((src_code, None, None, 0, out_path))
        for seed, n in pool_list:
            for k in k_pos:
                out_path = scoring.icl_output_path(
                    spec.key, src_code, args.tgt, k=k, seed=seed, n=n, out_dir=args.out_dir
                )
                schedule.append((src_code, seed, n, k, out_path))

    if not args.dry_run_tokens and not args.force:
        pending = [item for item in schedule if not is_complete(item[4], n_eval)]
        skipped = len(schedule) - len(pending)
        if skipped:
            log.info("skipping %d already-complete output file(s)", skipped)
        schedule = pending

    if not schedule:
        log.info("all requested runs are already complete")
        return 0

    if args.dry_run_tokens:
        tokenizer = generate.load_tokenizer(spec)
        for src_code, seed, n, k, _ in schedule:
            retriever = None
            if k > 0:
                assert seed is not None and n is not None
                pids = pools.load_pool(seed, n)
                retriever = retrieval.BM25Retriever(
                    pids,
                    [dev_texts[src_code][i] for i in pids],
                    [dev_texts[args.tgt][i] for i in pids],
                )
            batch_msgs = build_batch_messages(
                src_code, args.tgt, devtest_texts[src_code][:n_eval], k=k, seed=seed, retriever=retriever
            )
            lengths = [len(ids) for ids in generate.prepare_prompt_ids_batch(tokenizer, spec, batch_msgs)]
            log.info(
                "dry-run %-11s %s->%s seed=%-2s N=%-4s k=%-2d n=%-4d mean_prompt_tokens=%7.1f max_prompt_tokens=%d",
                spec.key, src_code, args.tgt, seed or "-", n or "-", k, n_eval,
                statistics.fmean(lengths), max(lengths),
            )
        return 0

    gen = generate.create_generator(spec, backend=args.backend)

    # Cache BM25 retrievers per (src_code, seed, n) to avoid rebuilding across k = 5, 20, 50.
    retriever_cache: dict[tuple[str, int, int], retrieval.BM25Retriever] = {}

    for idx, (src_code, seed, n, k, out_path) in enumerate(schedule, 1):
        retriever = None
        if k > 0:
            assert seed is not None and n is not None
            cache_key = (src_code, seed, n)
            if cache_key not in retriever_cache:
                pids = pools.load_pool(seed, n)
                retriever_cache[cache_key] = retrieval.BM25Retriever(
                    pids,
                    [dev_texts[src_code][i] for i in pids],
                    [dev_texts[args.tgt][i] for i in pids],
                )
            retriever = retriever_cache[cache_key]

        test_src = devtest_texts[src_code][:n_eval]
        batch_msgs = build_batch_messages(src_code, args.tgt, test_src, k=k, seed=seed, retriever=retriever)
        gen_results = gen.generate_batch(batch_msgs, max_new_tokens=args.max_new_tokens)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".jsonl.tmp")
        hyps: list[str] = []
        ptoks: list[int] = []
        n_trunc = 0

        with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
            for sent_id, (src_text, ref_text, res) in enumerate(zip(test_src, test_refs, gen_results)):
                hyp = prompt.clean_translation(res.raw_output, src_code, args.tgt)
                hyps.append(hyp)
                ptoks.append(res.prompt_tokens)
                n_trunc += int(res.truncated)
                record = {
                    "id": sent_id,
                    "src": src_text,
                    "translation": hyp,
                    "reference": ref_text,
                    "prompt_tokens": res.prompt_tokens,
                    "output_tokens": res.output_tokens,
                    "truncated": res.truncated,
                    "raw_output": res.raw_output,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        tmp_path.replace(out_path)

        metrics = scoring.compute_metrics(hyps, test_refs, tgt_code=args.tgt, prompt_tokens=ptoks)
        log.info(
            "[%d/%d] %s %s->%s seed=%s N=%s k=%d: chrF=%.2f BLEU=%.2f max_ptok=%d trunc=%d -> %s",
            idx, len(schedule), spec.key, src_code, args.tgt,
            seed or "-", n or "-", k,
            metrics["chrf"], metrics["bleu"], metrics["max_prompt_tokens"], n_trunc, out_path.name,
        )

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)
    sys.exit(main())
