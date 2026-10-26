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
class DataSpec:
    repo_id: str
    revision: str
    splits: dict[str, int]  # split name -> expected number of sentences


def load_yaml(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


@cache
def data_spec() -> DataSpec:
    cfg = load_yaml("data.yaml")
    return DataSpec(cfg["repo_id"], cfg["revision"], dict(cfg["splits"]))


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


def main_model_keys() -> list[str]:
    return list(load_yaml("models.yaml")["main"])


def model_spec(key: str) -> ModelSpec:
    models = load_yaml("models.yaml")["models"]
    if key not in models:
        raise KeyError(f"unknown model key: {key} (known: {', '.join(models)})")
    entry = models[key]
    return ModelSpec(key, entry["hf_id"], entry["revision"], dict(entry.get("chat_template_kwargs") or {}))
