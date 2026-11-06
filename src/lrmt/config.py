"""Project paths and typed access to the YAML files in configs/."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data" / "flores_plus"
POOL_DIR = ROOT / "pools"
INDICATOR_DIR = ROOT / "indicators"
OUTPUT_DIR = ROOT / "outputs"
ADAPTER_DIR = ROOT / "adapters"
SCORE_DIR = ROOT / "scores"
ANALYSIS_DIR = ROOT / "analysis"


@dataclass(frozen=True)
class Language:
    code: str  # FLORES+ subset, e.g. "swh_Latn"
    name: str  # name used in prompts
    role: str  # "source" | "heldout" | "target"

    @property
    def script(self) -> str:
        return self.code.split("_", 1)[1]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    revision: str
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PoolSpec:
    split: str
    sizes: tuple[int, ...]
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class DataSpec:
    repo_id: str
    revision: str
    splits: dict[str, int]  # split name -> expected number of sentences
    pools: PoolSpec


@dataclass(frozen=True)
class PromptSpec:
    instruction: str
    pair_template: str
    query_template: str
    block_sep: str
    instruction_names: dict[str, str] = field(default_factory=dict)

    def instruction_name(self, lang: Language) -> str:
        return self.instruction_names.get(lang.code, lang.name)


@dataclass(frozen=True)
class LoraSpec:
    rank: int
    alpha_ratio: int
    dropout: float
    learning_rate: float
    target_modules: tuple[str, ...]
    quant_type: str
    double_quant: bool
    compute_dtype: str
    max_seq_length: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    lr_scheduler_type: str
    weight_decay: float
    epochs_by_n: dict[int, int]
    heldout_lang: str
    search_learning_rates: tuple[float, ...]
    search_ranks: tuple[int, ...]

    @property
    def lora_alpha(self) -> int:
        return self.alpha_ratio * self.rank

    def epochs_for(self, n: int) -> int:
        if n not in self.epochs_by_n:
            raise KeyError(f"no epoch count configured for pool size N={n} in lora.yaml")
        return self.epochs_by_n[n]


def load_yaml(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


@cache
def data_spec() -> DataSpec:
    cfg = load_yaml("data.yaml")
    splits = dict(cfg["splits"])
    p = cfg["pools"]
    pools = PoolSpec(p["split"], tuple(sorted(p["sizes"])), tuple(p["seeds"]))
    if pools.split not in splits:
        raise ValueError(f"pool split {pools.split!r} is not a configured split")
    if not pools.seeds or len(set(pools.seeds)) != len(pools.seeds):
        raise ValueError(f"pool seeds must be non-empty and unique: {pools.seeds}")
    if (not pools.sizes or len(set(pools.sizes)) != len(pools.sizes)
            or pools.sizes[0] < 1 or pools.sizes[-1] > splits[pools.split]):
        raise ValueError(f"pool sizes must lie in 1..{splits[pools.split]}: {pools.sizes}")
    return DataSpec(cfg["repo_id"], cfg["revision"], splits, pools)


_ROLES = {"sources": "source", "heldout": "heldout", "targets": "target"}


@cache
def languages() -> tuple[Language, ...]:
    """All languages in config order: sources, held-out, targets."""
    cfg = load_yaml("languages.yaml")
    langs = tuple(
        Language(entry["code"], entry["name"], role)
        for section, role in _ROLES.items()
        for entry in cfg[section]
    )
    codes = [lang.code for lang in langs]
    if len(set(codes)) != len(codes):
        raise ValueError(f"duplicate language codes in languages.yaml: {codes}")
    return langs


def language(code: str) -> Language:
    for lang in languages():
        if lang.code == code:
            return lang
    raise KeyError(f"unknown language code: {code}")


def pivot_code() -> str:
    code = load_yaml("languages.yaml")["pivot"]
    language(code)  # must be one of the configured languages
    return code


def source_languages(include_heldout: bool = False) -> list[Language]:
    roles = {"source", "heldout"} if include_heldout else {"source"}
    return [lang for lang in languages() if lang.role in roles]


def pilot_codes() -> list[str]:
    codes = list(load_yaml("languages.yaml")["pilot"])
    for code in codes:
        language(code)
    return codes


@cache
def prompt_spec() -> PromptSpec:
    cfg = load_yaml("prompt.yaml")
    return PromptSpec(
        instruction=cfg["instruction"],
        pair_template=cfg["pair_template"],
        query_template=cfg["query_template"],
        block_sep=cfg["block_sep"],
        instruction_names=dict(cfg.get("instruction_names") or {}),
    )


@cache
def lora_spec() -> LoraSpec:
    cfg = load_yaml("lora.yaml")
    q = cfg["quantization"]
    t = cfg["training"]
    g = cfg["search_grid"]
    return LoraSpec(
        rank=int(cfg["rank"]),
        alpha_ratio=int(cfg["alpha_ratio"]),
        dropout=float(cfg["dropout"]),
        learning_rate=float(cfg["learning_rate"]),
        target_modules=tuple(cfg["target_modules"]),
        quant_type=str(q["quant_type"]),
        double_quant=bool(q["double_quant"]),
        compute_dtype=str(q["compute_dtype"]),
        max_seq_length=int(t["max_seq_length"]),
        per_device_train_batch_size=int(t["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
        warmup_ratio=float(t["warmup_ratio"]),
        lr_scheduler_type=str(t["lr_scheduler_type"]),
        weight_decay=float(t["weight_decay"]),
        epochs_by_n={int(k): int(v) for k, v in t["epochs_by_n"].items()},
        heldout_lang=str(g["heldout_lang"]),
        search_learning_rates=tuple(float(x) for x in g["learning_rates"]),
        search_ranks=tuple(int(x) for x in g["ranks"]),
    )


def main_model_keys() -> list[str]:
    return list(load_yaml("models.yaml")["main"])


def model_spec(key: str) -> ModelSpec:
    models = load_yaml("models.yaml")["models"]
    if key not in models:
        raise KeyError(f"unknown model key: {key} (known: {', '.join(models)})")
    entry = models[key]
    return ModelSpec(key, entry["hf_id"], entry["revision"], dict(entry.get("chat_template_kwargs") or {}))
