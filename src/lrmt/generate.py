"""Unified generation interface with `transformers` (local) and `vllm` (Pelle) backends.

Both backends share identical prompt rendering and tokenization (`add_special_tokens=False`
after `apply_chat_template` so BOS is never duplicated), use greedy decoding with
`max_new_tokens=256`, and return `GenerationResult` records.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from lrmt import prompt
from lrmt.config import ModelSpec

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationResult:
    raw_output: str
    prompt_tokens: int
    output_tokens: int
    truncated: bool  # True if prompt exceeded max_model_len or output hit max_new_tokens


def load_tokenizer(spec: ModelSpec) -> PreTrainedTokenizerBase:
    """Load the pinned tokenizer, checking the local HF cache first to avoid network hangs."""
    try:
        tok = AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.revision, local_files_only=True)
    except OSError:
        tok = AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.revision)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def prepare_prompt_ids(
    tokenizer: PreTrainedTokenizerBase,
    spec: ModelSpec,
    messages: list[dict[str, str]],
) -> list[int]:
    """Render chat messages and encode without adding duplicate special tokens."""
    rendered = prompt.render_chat_prompt(
        tokenizer,
        messages,
        add_generation_prompt=True,
        chat_template_kwargs=spec.chat_template_kwargs,
    )
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    if not ids:
        raise ValueError("rendered prompt encoded to zero tokens")
    return ids


def prepare_prompt_ids_batch(
    tokenizer: PreTrainedTokenizerBase,
    spec: ModelSpec,
    batch_messages: Sequence[list[dict[str, str]]],
) -> list[list[int]]:
    """Batch-render and fast-tokenize chat messages without duplicate special tokens."""
    if not batch_messages:
        return []
    rendered = [
        prompt.render_chat_prompt(
            tokenizer,
            msgs,
            add_generation_prompt=True,
            chat_template_kwargs=spec.chat_template_kwargs,
        )
        for msgs in batch_messages
    ]
    all_ids: list[list[int]] = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if any(len(ids) == 0 for ids in all_ids):
        raise ValueError("a rendered prompt encoded to zero tokens")
    return all_ids


class TransformersGenerator:
    """Hugging Face `transformers` backend for local CPU smoke tests and small-scale debugging."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        device: str | None = None,
        dtype: str = "auto",
        batch_size: int | None = None,
    ) -> None:
        self.spec = spec
        self.tokenizer = load_tokenizer(spec)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size or (8 if self.device.startswith("cuda") else 1)

        torch_dtype: Any = "auto"
        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "float32":
            torch_dtype = torch.float32
        elif self.device == "cpu":
            torch_dtype = torch.float32

        try:
            model = AutoModelForCausalLM.from_pretrained(
                spec.hf_id,
                revision=spec.revision,
                dtype=torch_dtype,
                local_files_only=True,
            )
        except OSError:
            model = AutoModelForCausalLM.from_pretrained(
                spec.hf_id,
                revision=spec.revision,
                dtype=torch_dtype,
            )
        self.base_model = model.to(self.device)
        self.base_model.eval()
        self.max_model_len = int(getattr(self.base_model.config, "max_position_embeddings", 32768))

    @torch.inference_mode()
    def generate_batch(
        self,
        batch_messages: Sequence[list[dict[str, str]]],
        *,
        max_new_tokens: int = 256,
        lora_path: Path | None = None,
    ) -> list[GenerationResult]:
        model = self.base_model
        if lora_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(self.base_model, str(lora_path))
            model.eval()

        all_prompt_ids = prepare_prompt_ids_batch(self.tokenizer, self.spec, batch_messages)
        results: list[GenerationResult] = []
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        for start in range(0, len(all_prompt_ids), self.batch_size):
            chunk = all_prompt_ids[start : start + self.batch_size]
            truncated_flags = [len(ids) + max_new_tokens > self.max_model_len for ids in chunk]
            max_len = max(len(ids) for ids in chunk)

            input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long, device=self.device)
            attention_mask = torch.zeros((len(chunk), max_len), dtype=torch.long, device=self.device)
            for row_idx, ids in enumerate(chunk):
                offset = max_len - len(ids)
                input_ids[row_idx, offset:] = torch.tensor(ids, dtype=torch.long, device=self.device)
                attention_mask[row_idx, offset:] = 1

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=pad_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            for row_idx, ids in enumerate(chunk):
                gen_ids = outputs[row_idx, max_len:].tolist()
                # Strip trailing EOS/PAD tokens if present.
                eos_id = self.tokenizer.eos_token_id
                if eos_id is not None and eos_id in gen_ids:
                    gen_ids = gen_ids[: gen_ids.index(eos_id)]
                raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                hit_limit = len(gen_ids) >= max_new_tokens
                results.append(
                    GenerationResult(
                        raw_output=raw_text,
                        prompt_tokens=len(ids),
                        output_tokens=len(gen_ids),
                        truncated=truncated_flags[row_idx] or hit_limit,
                    )
                )

        if lora_path is not None and hasattr(model, "unload"):
            model.unload()
        return results


class VLLMGenerator:
    """vLLM backend for high-throughput greedy inference (base and LoRA) on UPPMAX Pelle."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        max_model_len: int | None = None,
        enable_lora: bool = False,
        max_lora_rank: int = 16,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "bfloat16",
    ) -> None:
        from vllm import LLM

        self.spec = spec
        self.tokenizer = load_tokenizer(spec)
        self.enable_lora = enable_lora
        self._lora_seq = 0

        llm_kwargs: dict[str, Any] = {
            "model": spec.hf_id,
            "revision": spec.revision,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enable_lora": enable_lora,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        if enable_lora:
            llm_kwargs["max_lora_rank"] = max_lora_rank

        self.llm = LLM(**llm_kwargs)
        self.max_model_len = int(self.llm.llm_engine.model_config.max_model_len)

    def generate_batch(
        self,
        batch_messages: Sequence[list[dict[str, str]]],
        *,
        max_new_tokens: int = 256,
        lora_path: Path | None = None,
    ) -> list[GenerationResult]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        from vllm.lora.request import LoRARequest

        all_prompt_ids = prepare_prompt_ids_batch(self.tokenizer, self.spec, batch_messages)
        prompts: list[TokensPrompt] = []
        prompt_truncated: list[bool] = []
        budget = self.max_model_len - max_new_tokens

        for ids in all_prompt_ids:
            if len(ids) > budget:
                prompt_truncated.append(True)
                prompts.append(TokensPrompt(prompt_token_ids=ids[:budget]))
            else:
                prompt_truncated.append(False)
                prompts.append(TokensPrompt(prompt_token_ids=ids))

        if any(prompt_truncated):
            log.warning(
                "%d / %d prompts exceeded max_model_len - max_new_tokens (%d)",
                sum(prompt_truncated),
                len(prompt_truncated),
                budget,
            )

        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        lora_request = None
        if lora_path is not None:
            if not self.enable_lora:
                raise ValueError("VLLMGenerator was initialized with enable_lora=False")
            self._lora_seq += 1
            lora_request = LoRARequest(
                lora_name=f"{lora_path.name}_{self._lora_seq}",
                lora_int_id=self._lora_seq,
                lora_path=str(lora_path.resolve()),
            )

        outputs = self.llm.generate(prompts, sampling_params, lora_request=lora_request, use_tqdm=False)
        results: list[GenerationResult] = []
        for orig_ids, p_trunc, out in zip(all_prompt_ids, prompt_truncated, outputs):
            comp = out.outputs[0]
            hit_limit = comp.finish_reason == "length" or len(comp.token_ids) >= max_new_tokens
            results.append(
                GenerationResult(
                    raw_output=comp.text,
                    prompt_tokens=len(orig_ids),
                    output_tokens=len(comp.token_ids),
                    truncated=p_trunc or hit_limit,
                )
            )
        return results


def create_generator(
    spec: ModelSpec,
    *,
    backend: str = "auto",
    **kwargs: Any,
) -> TransformersGenerator | VLLMGenerator:
    """Create a generator backend ('vllm', 'transformers', or 'auto')."""
    if backend == "auto":
        try:
            import vllm  # noqa: F401

            backend = "vllm" if torch.cuda.is_available() else "transformers"
        except ImportError:
            backend = "transformers"

    if backend == "vllm":
        return VLLMGenerator(spec, **kwargs)
    if backend == "transformers":
        return TransformersGenerator(spec, **kwargs)
    raise ValueError(f"unknown backend {backend!r} (expected 'auto', 'vllm' or 'transformers')")
