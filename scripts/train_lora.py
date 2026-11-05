"""Train QLoRA adapters on the nested FLORES+ dev pools (Stage 5).

Loads and 4-bit quantizes the base model once per invocation, then trains each
requested (src, seed, n) adapter sequentially by attaching, saving and unloading
the LoRA layers. Completed adapters in adapters/ are skipped automatically unless
--force is passed.

Also supports hyperparameter grid search on the held-out language Zulu (--grid-search),
which trains adapters for every (learning_rate, rank) pair configured in configs/lora.yaml.

Examples:
    # Held-out hyperparameter search on Zulu (Stage 5, Nov 9-10):
    python scripts/train_lora.py --model qwen3-8b --langs zul_Latn --pools 1:50 1:200 1:997 --grid-search

    # First batch check: all 12 source languages, N=50, seed=1:
    python scripts/train_lora.py --model qwen3-8b --langs sources --pools 1:50

    # Train all 7 pools for one language (used by Slurm job arrays):
    python scripts/train_lora.py --model qwen3-8b --langs swh_Latn
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lrmt import config, data, generate, lora, pools

log = logging.getLogger("train_lora")


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
    parser.add_argument("--rank", type=int, default=None, help="override LoRA rank from configs/lora.yaml")
    parser.add_argument("--lr", type=float, default=None, help="override learning rate from configs/lora.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override epoch count")
    parser.add_argument("--grid-search", action="store_true", help="run (lr x rank) search grid from configs/lora.yaml")
    parser.add_argument("--adapter-dir", type=Path, default=None, help="override base directory (default: adapters/)")
    parser.add_argument("--no-4bit", action="store_true", help="disable 4-bit quantization (for CPU smoke tests)")
    parser.add_argument("--force", action="store_true", help="overwrite already-trained adapters")
    args = parser.parse_args()

    try:
        m_spec = config.model_spec(args.model)
        l_spec = config.lora_spec()
        config.language(args.tgt)
        src_codes = resolve_langs(args.langs)
        pool_list = parse_pools(args.pools)
    except (KeyError, ValueError) as e:
        parser.error(str(e))

    if args.grid_search:
        hp_grid = [(r, lr) for r in l_spec.search_ranks for lr in l_spec.search_learning_rates]
    else:
        hp_grid = [
            (
                args.rank if args.rank is not None else l_spec.rank,
                args.lr if args.lr is not None else l_spec.learning_rate,
            )
        ]

    # Build schedule: (src_code, seed, n, rank, lr, out_dir)
    schedule: list[tuple[str, int, int, int, float, Path]] = []
    for src_code in src_codes:
        for seed, n in pool_list:
            for r, lr in hp_grid:
                if args.grid_search:
                    out_dir = lora.tuning_adapter_path(
                        m_spec.key, src_code, args.tgt,
                        seed=seed, n=n, rank=r, learning_rate=lr, base_dir=args.adapter_dir,
                    )
                else:
                    out_dir = lora.adapter_path(
                        m_spec.key, src_code, args.tgt, seed=seed, n=n, base_dir=args.adapter_dir
                    )
                schedule.append((src_code, seed, n, r, lr, out_dir))

    if not args.force:
        pending = [item for item in schedule if not lora.is_adapter_complete(item[5])]
        skipped = len(schedule) - len(pending)
        if skipped:
            log.info("skipping %d already-complete adapter(s)", skipped)
        schedule = pending

    if not schedule:
        log.info("all requested adapters are already trained")
        return 0

    all_codes = [args.tgt] + [c for c in src_codes if c != args.tgt]
    dev_texts = data.load_parallel(all_codes, "dev")

    tokenizer = generate.load_tokenizer(m_spec)
    base_model = lora.load_quantized_base_model(m_spec, l_spec, use_4bit=not args.no_4bit)

    for idx, (src_code, seed, n, r, lr, out_dir) in enumerate(schedule, 1):
        pids = pools.load_pool(seed, n)
        pool_src = [dev_texts[src_code][i] for i in pids]
        pool_tgt = [dev_texts[args.tgt][i] for i in pids]

        log.info(
            "[%d/%d] training %s %s->%s seed=%d N=%d (rank=%d, lr=%.0e) -> %s",
            idx, len(schedule), m_spec.key, src_code, args.tgt, seed, n, r, lr, out_dir,
        )
        base_model, res = lora.train_single_adapter(
            base_model,
            tokenizer,
            m_spec,
            l_spec,
            src_code=src_code,
            tgt_code=args.tgt,
            src_texts=pool_src,
            tgt_texts=pool_tgt,
            seed=seed,
            n=n,
            out_dir=out_dir,
            rank=r,
            learning_rate=lr,
            epochs=args.epochs,
        )
        log.info(
            "[%d/%d] done %s: loss %.4f -> %.4f in %.1fs (steps=%d, trunc=%d)",
            idx, len(schedule), out_dir.name, res.initial_loss, res.final_loss,
            res.elapsed_sec, res.total_steps, res.truncated_samples,
        )

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)
    sys.exit(main())
