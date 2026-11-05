"""QLoRA dataset preparation, prompt-masked loss collation, and adapter training.

Implements Stage 5 of the work plan:
- 4-bit QLoRA (NF4, double quantization, bf16 compute) via bitsandbytes + peft.
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj;
  lora_alpha = 2 * rank; dropout = 0.05; no early stopping.
- Same zero-shot chat prompt as inference; loss is computed strictly on the
  assistant translation + EOS tokens (`labels = -100` on all prompt tokens).
- Seeds simultaneously control pool sampling, LoRA initialization and batch shuffling.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    get_scheduler,
    set_seed,
)

from lrmt import config, generate, prompt
from lrmt.config import LoraSpec, ModelSpec

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainResult:
    model: str
    src: str
    tgt: str
    seed: int
    n: int
    rank: int
    lora_alpha: int
    learning_rate: float
    epochs: int
    total_steps: int
    initial_loss: float
    final_loss: float
    elapsed_sec: float
    truncated_samples: int
    adapter_dir: str


def adapter_path(
    model: str,
    src: str,
    tgt: str,
    *,
    seed: int,
    n: int,
    base_dir: Path | None = None,
) -> Path:
    root = base_dir or config.ADAPTER_DIR
    return root / f"{model}_{src}_{tgt}_seed{seed}_N{n}"


def tuning_adapter_path(
    model: str,
    src: str,
    tgt: str,
    *,
    seed: int,
    n: int,
    rank: int,
    learning_rate: float,
    base_dir: Path | None = None,
) -> Path:
    root = (base_dir or config.ADAPTER_DIR) / "tuning"
    lr_tag = f"{learning_rate:.0e}".replace("-0", "-")
    return root / f"{model}_{src}_{tgt}_seed{seed}_N{n}_r{rank}_lr{lr_tag}"


def is_adapter_complete(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "adapter_config.json").is_file()
        and (path / "adapter_model.safetensors").is_file()
        and (path / "train_meta.json").is_file()
    )


class PromptMaskedDataset(Dataset):
    """Tokenized parallel dataset where prompt tokens are masked with -100 in labels."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model_spec: ModelSpec,
        src_code: str,
        tgt_code: str,
        src_texts: Sequence[str],
        tgt_texts: Sequence[str],
        *,
        max_seq_length: int = 512,
    ) -> None:
        if len(src_texts) != len(tgt_texts):
            raise ValueError("src_texts and tgt_texts must have the same length")
        if not src_texts:
            raise ValueError("cannot build PromptMaskedDataset from empty lists")
        if tokenizer.eos_token_id is None:
            raise ValueError(f"tokenizer for {model_spec.key} has no eos_token_id")

        eos_id = int(tokenizer.eos_token_id)
        user_batches = [prompt.build_messages(src_code, tgt_code, s) for s in src_texts]
        prompt_id_batches = generate.prepare_prompt_ids_batch(tokenizer, model_spec, user_batches)
        resp_id_batches: list[list[int]] = tokenizer(list(tgt_texts), add_special_tokens=False)["input_ids"]

        self.examples: list[tuple[list[int], list[int]]] = []
        self.truncated_count = 0

        for idx, (p_ids, r_ids) in enumerate(zip(prompt_id_batches, resp_id_batches)):
            full_resp = list(r_ids) + [eos_id]
            if len(p_ids) + len(full_resp) > max_seq_length:
                self.truncated_count += 1
                # Preserve the target translation (and the chat template's assistant header at
                # the tail of p_ids) by trimming the middle of the source prompt first.
                resp_keep = min(len(full_resp), max(16, max_seq_length // 3))
                kept_resp = full_resp[:resp_keep]
                prompt_budget = max_seq_length - len(kept_resp)
                tail_keep = min(16, prompt_budget)
                head_keep = prompt_budget - tail_keep
                kept_prompt = p_ids[:head_keep] + (p_ids[-tail_keep:] if tail_keep > 0 else [])
                input_ids = kept_prompt + kept_resp
                labels = [-100] * len(kept_prompt) + kept_resp
            else:
                input_ids = p_ids + full_resp
                labels = [-100] * len(p_ids) + full_resp

            if not any(lbl != -100 for lbl in labels):
                raise ValueError(f"sample {idx} ({src_code}->{tgt_code}) has no target tokens for loss")
            self.examples.append((input_ids, labels))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.examples[idx]


class PromptMaskedCollator:
    """Right-pad input_ids with pad_token_id and labels with -100."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[tuple[list[int], list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(inp) for inp, _ in batch)
        bsz = len(batch)
        input_ids = torch.full((bsz, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        labels = torch.full((bsz, max_len), -100, dtype=torch.long)

        for row, (inp, lbl) in enumerate(batch):
            length = len(inp)
            input_ids[row, :length] = torch.tensor(inp, dtype=torch.long)
            attention_mask[row, :length] = 1
            labels[row, :length] = torch.tensor(lbl, dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def resolve_torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported compute_dtype: {name!r}")


def load_quantized_base_model(
    model_spec: ModelSpec,
    lora_cfg: LoraSpec,
    *,
    use_4bit: bool = True,
) -> Any:
    """Load the base CausalLM (4-bit NF4 on CUDA, or fp32 on CPU for smoke tests)."""
    compute_dtype = resolve_torch_dtype(lora_cfg.compute_dtype)
    if use_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit QLoRA requires a CUDA GPU (use --no-4bit for CPU smoke tests)")
        from peft import prepare_model_for_kbit_training

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=lora_cfg.quant_type,
            bnb_4bit_use_double_quant=lora_cfg.double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_spec.hf_id,
                revision=model_spec.revision,
                quantization_config=bnb_config,
                dtype=compute_dtype,
                device_map="auto",
                local_files_only=True,
            )
        except OSError:
            model = AutoModelForCausalLM.from_pretrained(
                model_spec.hf_id,
                revision=model_spec.revision,
                quantization_config=bnb_config,
                dtype=compute_dtype,
                device_map="auto",
            )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        return model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = compute_dtype if device == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_spec.hf_id, revision=model_spec.revision, dtype=dtype, local_files_only=True
        )
    except OSError:
        model = AutoModelForCausalLM.from_pretrained(
            model_spec.hf_id, revision=model_spec.revision, dtype=dtype
        )
    model.config.use_cache = False
    return model.to(device)


def train_single_adapter(
    base_model: Any,
    tokenizer: PreTrainedTokenizerBase,
    model_spec: ModelSpec,
    lora_cfg: LoraSpec,
    *,
    src_code: str,
    tgt_code: str,
    src_texts: Sequence[str],
    tgt_texts: Sequence[str],
    seed: int,
    n: int,
    out_dir: Path,
    rank: int | None = None,
    learning_rate: float | None = None,
    epochs: int | None = None,
    max_seq_length: int | None = None,
) -> tuple[Any, TrainResult]:
    """Attach a fresh LoRA adapter to base_model, train on the N pairs, save and unload."""
    from peft import LoraConfig, get_peft_model

    set_seed(seed)
    r = rank if rank is not None else lora_cfg.rank
    alpha = lora_cfg.alpha_ratio * r
    lr = learning_rate if learning_rate is not None else lora_cfg.learning_rate
    num_epochs = epochs if epochs is not None else lora_cfg.epochs_for(n)
    seq_len = max_seq_length if max_seq_length is not None else lora_cfg.max_seq_length

    dataset = PromptMaskedDataset(
        tokenizer,
        model_spec,
        src_code,
        tgt_code,
        src_texts,
        tgt_texts,
        max_seq_length=seq_len,
    )
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    collator = PromptMaskedCollator(int(pad_id))

    gen = torch.Generator()
    gen.manual_seed(seed)
    batch_size = min(lora_cfg.per_device_train_batch_size, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=gen,
    )

    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=list(lora_cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, peft_config)
    peft_model.train()

    trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=lora_cfg.weight_decay,
    )

    grad_accum = lora_cfg.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(loader) / grad_accum)
    total_steps = max(1, steps_per_epoch * num_epochs)
    warmup_steps = int(round(total_steps * lora_cfg.warmup_ratio))
    scheduler = get_scheduler(
        lora_cfg.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    device = next(peft_model.parameters()).device
    use_amp = device.type == "cuda"
    amp_dtype = resolve_torch_dtype(lora_cfg.compute_dtype)

    t0 = time.perf_counter()
    epoch_losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        n_batches = len(loader)
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    outputs = peft_model(**batch)
                    raw_loss = outputs.loss
            else:
                outputs = peft_model(**batch)
                raw_loss = outputs.loss

            loss_val = float(raw_loss.detach().cpu().item())
            if not math.isfinite(loss_val):
                raise RuntimeError(
                    f"non-finite training loss ({loss_val}) at epoch {epoch} step {step} "
                    f"for {model_spec.key} {src_code}->{tgt_code} seed={seed} N={n}"
                )
            running_loss += loss_val
            scaled_loss = raw_loss / grad_accum
            scaled_loss.backward()

            if step % grad_accum == 0 or step == n_batches:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        epoch_loss = running_loss / n_batches
        epoch_losses.append(epoch_loss)
        log.info(
            "  [%s %s->%s seed=%d N=%d r=%d lr=%.0e] epoch %d/%d loss=%.4f",
            model_spec.key, src_code, tgt_code, seed, n, r, lr, epoch, num_epochs, epoch_loss,
        )

    elapsed = round(time.perf_counter() - t0, 2)
    result = TrainResult(
        model=model_spec.key,
        src=src_code,
        tgt=tgt_code,
        seed=seed,
        n=n,
        rank=r,
        lora_alpha=alpha,
        learning_rate=lr,
        epochs=num_epochs,
        total_steps=total_steps,
        initial_loss=round(epoch_losses[0], 4),
        final_loss=round(epoch_losses[-1], 4),
        elapsed_sec=elapsed,
        truncated_samples=dataset.truncated_count,
        adapter_dir=str(out_dir),
    )

    # Save atomically via a temporary directory so interrupted jobs never leave partial adapters.
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    peft_model.save_pretrained(str(tmp_dir))
    tokenizer.save_pretrained(str(tmp_dir))
    with open(tmp_dir / "train_meta.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(asdict(result), f, indent=2)
        f.write("\n")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.replace(out_dir)

    unloaded_base = peft_model.unload()
    return unloaded_base, result
