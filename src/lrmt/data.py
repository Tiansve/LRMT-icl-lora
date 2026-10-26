"""FLORES+ download, loading and validation.

Files are mirrored from the Hub as data/flores_plus/{split}/{code}.jsonl. Every
load is validated and NFC-normalized, so downstream code can rely on complete,
id-ordered, non-empty sentences.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from lrmt.config import DATA_DIR, data_spec


class DataError(ValueError):
    """A FLORES+ file is missing, incomplete or inconsistent."""


@dataclass(frozen=True)
class Sentence:
    id: int
    text: str
    url: str


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def flores_path(code: str, split: str) -> Path:
    return DATA_DIR / split / f"{code}.jsonl"


def download(codes: list[str], splits: list[str] | None = None) -> list[Path]:
    """Fetch the pinned revision of each (split, code) file; unchanged files are skipped."""
    spec = data_spec()
    paths = []
    for split in splits or list(spec.splits):
        for code in codes:
            hf_hub_download(
                repo_id=spec.repo_id,
                repo_type="dataset",
                filename=f"{split}/{code}.jsonl",
                revision=spec.revision,
                local_dir=DATA_DIR,
            )
            paths.append(flores_path(code, split))
    return paths


def read_split(code: str, split: str) -> list[Sentence]:
    """Load one split of one language, ordered by id; raise DataError on any inconsistency."""
    splits = data_spec().splits
    if split not in splits:
        raise DataError(f"unknown split {split!r} (known: {', '.join(splits)})")
    expected = splits[split]
    path = flores_path(code, split)
    if not path.is_file():
        raise DataError(f"{path} not found; run scripts/download_flores.py first")

    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    iso, script = code.split("_", 1)
    problems = []
    if len(rows) != expected:
        problems.append(f"{len(rows)} rows, expected {expected}")
    ids = sorted(row["id"] for row in rows)
    if ids != list(range(len(ids))):
        problems.append("ids are not exactly 0..n-1 (duplicates or gaps)")
    wrong_lang = [row["id"] for row in rows if (row["iso_639_3"], row["iso_15924"]) != (iso, script)]
    if wrong_lang:
        problems.append(f"{len(wrong_lang)} rows with another language/script, e.g. id {wrong_lang[0]}")
    empty = [row["id"] for row in rows if not normalize(row["text"])]
    if empty:
        problems.append(f"{len(empty)} empty sentences, e.g. id {empty[0]}")
    if problems:
        raise DataError(f"{path}: " + "; ".join(problems))

    rows.sort(key=lambda row: row["id"])
    return [Sentence(row["id"], normalize(row["text"]), row["url"]) for row in rows]


def load_parallel(codes: list[str], split: str) -> dict[str, list[str]]:
    """Load several languages and check they are sentence-aligned (same source URL per id)."""
    if not codes:
        raise ValueError("no language codes given")
    loaded = {code: read_split(code, split) for code in codes}
    ref_code, ref = codes[0], loaded[codes[0]]
    for code in codes[1:]:
        mismatched = [a.id for a, b in zip(ref, loaded[code]) if a.url != b.url]
        if mismatched:
            raise DataError(
                f"{split}: {code} is not aligned with {ref_code} "
                f"({len(mismatched)} ids with a different source URL, e.g. id {mismatched[0]})"
            )
    return {code: [s.text for s in sents] for code, sents in loaded.items()}
