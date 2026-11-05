"""Run greedy translation inference with trained LoRA adapters (Stage 5).

Loads the bf16 base model once (vLLM with enable_lora=True on Pelle, or transformers
+ peft locally) and evaluates all requested adapters on devtest (1,012 sentences)
using the shared zero-shot prompt.

 Also supports --grid-search to evaluate held-out tuning adapters in adapters/tuning/
and print a ranked summary table of chrF scores for hyperparameter selection.

Examples:
    # Evaluate Zulu hyperparameter search grid and rank by chrF:
    python scripts/run_lora.py --model qwen3-8b --langs zul_Latn --pools 1:50 1:200 1:997 --grid-search

    # Evaluate all 7 trained adapters for one language:
    python scripts/run_lora.py --model qwen3-8b --langs swh_Latn

    # Evaluate all 84 adapters for Qwen3-8B Chinese:
    python scripts/run_lora.py --model qwen3-8b --langs sources
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from pathlib import Path

from lrmt import config, data, generate, lora, pools, prompt, scoring

log = logging.getLogger("run_lora")


def resolve_langs(tokens: list[str]) -> list[str]:
    if tokens in (["all"], ["sources"]):
        return [lang.code for lang in config.source_languages(include_heldout=False)]
    if tokens == ["all-with-heldout"]:
        return [lang.code for lang in config.source_languages(include_heldout=True)]
    if tokens == ["pilot"]:
        return config.pilot_codes()
    if tokens == ["heldout"]:
        return [config.lora_spec().heldout_lang]
    for code in tokens:
        config.language(code)
    return tokens


def parse_pools(tokens: list[str] | None) -> list[tuple[int, int]]:
    if not tokens:
        return pools.pool_grid()
    grid = set(pools.pool_grid())
    result: list[tuple[int, int]] = []
    for item in tokens:
        try:
            seed_s, n_s = item.split(":", 1)
            pair = (int(seed_s), int(n_s))
        except ValueError as e:
            raise ValueError(f"invalid pool {item!r}; expected SEED:N, e.g. 1:50") from e
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model key in configs/models.yaml")
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["sources"],
        help="source language codes, or 'sources' / 'pilot' / 'heldout' / 'all-with-heldout'",
    )
    parser.add_argument("--tgt", default="cmn_Hans", help="target language code (default: cmn_Hans)")
    parser.add_argument("--pools", nargs="*", help="subset of SEED:N pools (default: all 7 pools)")
    parser.add_argument("--limit", type=int, default=None, help="limit to first LIMIT devtest sentences")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--backend", choices=["auto", "vllm", "transformers"], default="auto")
    parser.add_argument("--adapter-dir", type=Path, default=None, help="override base adapter directory")
    parser.add_argument("--out-dir", type=Path, default=None, help="override output directory (default: outputs/lora)")
    parser.add_argument("--grid-search", action="store_true", help="evaluate tuning grid adapters in adapters/tuning/")
    parser.add_argument("--force", action="store_true", help="overwrite existing complete output jsonl files")
    args = parser.parse_args()

    try:
        m_spec = config.model_spec(args.model)
        l_spec = config.lora_spec()
        config.language(args.tgt)
        src_codes = resolve_langs(args.langs)
        pool_list = parse_pools(args.pools)
    except (KeyError, ValueError) as e:
        parser.error(str(e))

    all_codes = [args.tgt] + [c for c in src_codes if c != args.tgt]
    devtest_texts = data.load_parallel(all_codes, "devtest")
    n_total = len(devtest_texts[args.tgt])
    n_eval = min(args.limit, n_total) if args.limit is not None else n_total
    test_refs = devtest_texts[args.tgt][:n_eval]

    if args.grid_search:
        hp_grid = [(r, lr) for r in l_spec.search_ranks for lr in l_spec.search_learning_rates]
        max_rank = max(l_spec.search_ranks)
    else:
        hp_grid = [(l_spec.rank, l_spec.learning_rate)]
        max_rank = l_spec.rank

    # Schedule entries: (src_code, seed, n, rank, lr, adapter_dir, out_path)
    schedule: list[tuple[str, int, int, int, float, Path, Path]] = []
    missing_adapters: list[str] = []
    out_base = args.out_dir or (
        (config.OUTPUT_DIR / "lora_tuning") if args.grid_search else (config.OUTPUT_DIR / "lora")
    )

    for src_code in src_codes:
        for seed, n in pool_list:
            for r, lr in hp_grid:
                if args.grid_search:
                    adir = lora.tuning_adapter_path(
                        m_spec.key, src_code, args.tgt,
                        seed=seed, n=n, rank=r, learning_rate=lr, base_dir=args.adapter_dir,
                    )
                    lr_tag = f"{lr:.0e}".replace("-0", "-")
                    opath = out_base / f"{m_spec.key}_{src_code}_{args.tgt}_seed{seed}_N{n}_r{r}_lr{lr_tag}.jsonl"
                else:
                    adir = lora.adapter_path(
                        m_spec.key, src_code, args.tgt, seed=seed, n=n, base_dir=args.adapter_dir
                    )
                    opath = scoring.lora_output_path(
                        m_spec.key, src_code, args.tgt, seed=seed, n=n, out_dir=out_base
                    )
                if not lora.is_adapter_complete(adir):
                    missing_adapters.append(str(adir))
                schedule.append((src_code, seed, n, r, lr, adir, opath))

    if missing_adapters:
        log.error("missing or incomplete adapter(s): %s; run scripts/train_lora.py first", ", ".join(missing_adapters[:5]))
        return 1

    to_run = [item for item in schedule if args.force or not is_complete(item[6], n_eval)]
    if len(to_run) < len(schedule):
        log.info("skipping %d already-complete LoRA output file(s)", len(schedule) - len(to_run))

    if to_run:
        gen_kwargs = {"enable_lora": True, "max_lora_rank": max(max_rank, 16)} if args.backend != "transformers" else {}
        try:
            gen = generate.create_generator(m_spec, backend=args.backend, **gen_kwargs)
        except TypeError:
            gen = generate.create_generator(m_spec, backend=args.backend)

        for idx, (src_code, seed, n, r, lr, adir, opath) in enumerate(to_run, 1):
            test_src = devtest_texts[src_code][:n_eval]
            batch_msgs = [prompt.build_messages(src_code, args.tgt, s) for s in test_src]
            gen_results = gen.generate_batch(batch_msgs, max_new_tokens=args.max_new_tokens, lora_path=adir)

            opath.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = opath.with_suffix(".jsonl.tmp")
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
            tmp_path.replace(opath)

            metrics = scoring.compute_metrics(hyps, test_refs, tgt_code=args.tgt, prompt_tokens=ptoks)
            log.info(
                "[%d/%d] %s %s->%s seed=%d N=%d r=%d lr=%.0e: chrF=%.2f BLEU=%.2f trunc=%d -> %s",
                idx, len(to_run), m_spec.key, src_code, args.tgt, seed, n, r, lr,
                metrics["chrf"], metrics["bleu"], n_trunc, opath.name,
            )

    if args.grid_search:
        summary_rows: list[dict] = []
        by_combo: dict[tuple[int, float], list[float]] = {}
        for src_code, seed, n, r, lr, _, opath in schedule:
            rows = scoring.read_output_jsonl(opath)
            metrics = scoring.compute_metrics(
                [row["translation"] for row in rows],
                [row["reference"] for row in rows],
                tgt_code=args.tgt,
            )
            chrf = float(metrics["chrf"])
            by_combo.setdefault((r, lr), []).append(chrf)
            summary_rows.append({
                "model": m_spec.key, "src": src_code, "tgt": args.tgt,
                "seed": seed, "n": n, "rank": r, "learning_rate": lr,
                "chrf": chrf, "bleu": metrics["bleu"], "chrfpp": metrics["chrfpp"],
                "off_target_rate": metrics["off_target_rate"],
            })

        tuning_csv = config.SCORE_DIR / "lora_tuning.csv"
        tuning_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(tuning_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

        log.info("Held-out hyperparameter grid summary (mean chrF across pools):")
        ranked = sorted(by_combo.items(), key=lambda kv: (-statistics.fmean(kv[1]), kv[0][0], kv[0][1]))
        for (r, lr), vals in ranked:
            log.info("  rank=%-2d lr=%.0e  mean_chrF=%6.2f  per_pool=%s", r, lr, statistics.fmean(vals), vals)
        best_r, best_lr = ranked[0][0]
        log.info("Selected best hyperparameters: rank=%d, learning_rate=%.0e (update configs/lora.yaml)", best_r, best_lr)

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)
    sys.exit(main())
