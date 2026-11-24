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
    *,
    multi_turn: bool = False,
) -> list[dict[str, str]]:
    """Build chat messages for inference (tgt_text=None) or LoRA training (tgt_text given).

    When `multi_turn=True` and `examples` is non-empty (used for instruction models like
    Llama-3.1-8B-Instruct that otherwise echo all in-context examples when packed into a
    single user turn), each demonstration is formatted as a (user, assistant) turn pair
    using the shared instruction and query_template.
    """
    if multi_turn and examples:
        src_lang, tgt_lang = _as_lang(src), _as_lang(tgt)
        spec = config.prompt_spec()
        header = spec.instruction.format(
            src_name=src_lang.name,
            tgt_name=tgt_lang.name,
            tgt_instr_name=spec.instruction_name(tgt_lang),
        )
        messages: list[dict[str, str]] = []
        for idx, (ex_src, ex_tgt) in enumerate(examples):
            q_str = spec.query_template.format(
                src_name=src_lang.name,
                tgt_name=tgt_lang.name,
                src_text=ex_src,
            )
            user_text = f"{header}{spec.block_sep}{q_str}" if idx == 0 else q_str
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": ex_tgt})

        final_q = spec.query_template.format(
            src_name=src_lang.name,
            tgt_name=tgt_lang.name,
            src_text=src_text,
        )
        messages.append({"role": "user", "content": final_q})
        if tgt_text is not None:
            messages.append({"role": "assistant", "content": tgt_text})
        return messages

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


def _strip_lang_prefix(line: str, prefixes: Sequence[str]) -> str:
    lower = line.lower()
    for prefix in prefixes:
        if lower.startswith(prefix.lower()):
            return line[len(prefix) :].strip()
    return line


def clean_translation(
    raw_text: str,
    src: Language | str | None = None,
    tgt: Language | str | None = None,
    examples: Sequence[tuple[str, str]] = (),
) -> str:
    """Extract the single-sentence translation from raw model output.

    FLORES+ sentences have no internal newlines. If a model echoes the target
    language label (e.g. 'Chinese: ...'), echoes the in-context example targets
    before translating the query, or continues generating extra lines, strip the
    label prefix, skip echoed demonstration lines, and keep the first translation line.
    """
    text = _THINK_RE.sub("", raw_text).strip()
    if not text:
        return ""

    prefixes: list[str] = []
    if tgt is not None:
        tgt_lang = _as_lang(tgt)
        spec = config.prompt_spec()
        prefixes.extend([f"{spec.instruction_name(tgt_lang)}:", f"{tgt_lang.name}:"])
    if src is not None:
        src_lang = _as_lang(src)
        prefixes.append(f"{src_lang.name}:")

    lines = [_strip_lang_prefix(ln.strip(), prefixes) for ln in text.splitlines() if ln.strip()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""

    if len(lines) > 1 and examples:
        ex_tgts = [et.strip() for _, et in examples if et.strip()]
        filtered = [
            ln for ln in lines
            if not any(ln == et or (len(ln) >= 12 and len(et) >= 12 and (ln[:12] == et[:12])) for et in ex_tgts)
        ]
        if filtered:
            return filtered[0]

    return lines[0]
