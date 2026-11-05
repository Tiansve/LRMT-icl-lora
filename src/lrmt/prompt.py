"""Unified translation prompt construction and output post-processing.

Zero-shot (k=0), ICL (k>0) and LoRA share the same instruction and query format
defined in configs/prompt.yaml. ICL places the k retrieved parallel examples in
the same user message above the test sentence, ordered from least to most similar
so the closest example immediately precedes the query.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from transformers import PreTrainedTokenizerBase

from lrmt import config
from lrmt.config import Language

_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", flags=re.DOTALL)


def _as_lang(lang: Language | str) -> Language:
    return config.language(lang) if isinstance(lang, str) else lang


def build_user_content(
    src: Language | str,
    tgt: Language | str,
    src_text: str,
    examples: Sequence[tuple[str, str]] = (),
) -> str:
    """Format the user message content for one translation query."""
    src_lang, tgt_lang = _as_lang(src), _as_lang(tgt)
    spec = config.prompt_spec()
    header = spec.instruction.format(
        src_name=src_lang.name,
        tgt_name=tgt_lang.name,
        tgt_instr_name=spec.instruction_name(tgt_lang),
    )
    blocks = [header]
    for ex_src, ex_tgt in examples:
        blocks.append(
            spec.pair_template.format(
                src_name=src_lang.name,
                tgt_name=tgt_lang.name,
                src_text=ex_src,
                tgt_text=ex_tgt,
            )
        )
    blocks.append(
        spec.query_template.format(
            src_name=src_lang.name,
            tgt_name=tgt_lang.name,
            src_text=src_text,
        )
    )
    return spec.block_sep.join(blocks)


def build_messages(
    src: Language | str,
    tgt: Language | str,
    src_text: str,
    examples: Sequence[tuple[str, str]] = (),
    tgt_text: str | None = None,
) -> list[dict[str, str]]:
    """Build chat messages for inference (tgt_text=None) or LoRA training (tgt_text given)."""
    messages = [{"role": "user", "content": build_user_content(src, tgt, src_text, examples)}]
    if tgt_text is not None:
        messages.append({"role": "assistant", "content": tgt_text})
    return messages


def render_chat_prompt(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> str:
    """Apply the tokenizer's chat template (including model-specific kwargs like enable_thinking)."""
    kwargs = dict(chat_template_kwargs or {})
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )
    if not isinstance(rendered, str):
        raise TypeError(f"expected str from apply_chat_template, got {type(rendered).__name__}")
    return rendered


def clean_translation(
    raw_text: str,
    src: Language | str | None = None,
    tgt: Language | str | None = None,
) -> str:
    """Extract the single-sentence translation from raw model output.

    FLORES+ sentences have no internal newlines. If a model echoes the target
    language label (e.g. 'Chinese: ...') or continues generating extra lines/examples,
    strip the label prefix and keep the first non-empty line.
    """
    text = _THINK_RE.sub("", raw_text).strip()
    if not text:
        return ""

    # Take the first non-empty line (prevents trailing commentary or hallucinated examples).
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if not line:
        return ""

    prefixes: list[str] = []
    if tgt is not None:
        tgt_lang = _as_lang(tgt)
        spec = config.prompt_spec()
        prefixes.extend([f"{spec.instruction_name(tgt_lang)}:", f"{tgt_lang.name}:"])
    if src is not None:
        src_lang = _as_lang(src)
        prefixes.append(f"{src_lang.name}:")

    lower = line.lower()
    for prefix in prefixes:
        if lower.startswith(prefix.lower()):
            line = line[len(prefix) :].strip()
            break
    return line
