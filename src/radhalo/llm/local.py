"""Open-weights models run locally: Hugging Face transformers (`local`) and vLLM (`vllm`)."""

import os
import re
from collections.abc import Mapping
from functools import partial

THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def strip_think(text: str) -> str:
    """Remove `<think>...</think>` blocks (Qwen 3 thinking mode)."""
    return THINK_BLOCK.sub("", text)


def thinking_enabled(cfg: Mapping) -> bool:
    thinking = cfg.get("thinking")
    return thinking is not None and thinking.get("enable", False)


def hf_generation_kwargs(cfg: Mapping) -> dict:
    """Sampling arguments passed to `model.generate`."""
    kwargs = {
        "do_sample": True,
        "temperature": cfg.get("temperature"),
        "top_p": cfg.get("top_p"),
        "num_return_sequences": 1,
    }
    for key in ("top_k", "min_p"):
        if cfg.get(key) is not None:
            kwargs[key] = cfg[key]
    return kwargs


def load_hf_model(path: str):
    """Load a causal LM in bfloat16 and its left-padding, left-truncating tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        path, padding_side="left", truncation_side="left", token=token, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, device_map="auto", torch_dtype=torch.bfloat16, token=token, trust_remote_code=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True
    return model, tokenizer


def hf_generate(
    conversations: list[list[dict]],
    *,
    model,
    tokenizer,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    enable_thinking: bool,
    **generation_kwargs,
) -> list[str]:
    """Generate one response per conversation; inputs longer than `max_length` tokens are cut on the left."""
    import torch

    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if "eos_token_id" not in generation_kwargs and tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

    results = []
    for start in range(0, len(conversations), batch_size):
        prompts = [
            tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            for conversation in conversations[start : start + batch_size]
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=max_new_tokens,
                **generation_kwargs,
            )
        new_tokens = outputs[:, inputs["input_ids"].shape[1] :]
        for text in tokenizer.batch_decode(new_tokens, skip_special_tokens=True):
            results.append((strip_think(text) if enable_thinking else text).strip())
    return results


def build_hf(cfg: Mapping, json_schema: dict | None = None):
    """Batch prediction function for a `local` config."""
    if json_schema is not None:
        raise NotImplementedError("JSON schema enforcement is not supported for local models")
    model, tokenizer = load_hf_model(cfg["path"])
    return partial(
        hf_generate,
        model=model,
        tokenizer=tokenizer,
        batch_size=cfg["batch_size"],
        max_length=cfg["max_length"],
        max_new_tokens=cfg["max_new_tokens"],
        enable_thinking=thinking_enabled(cfg),
        **hf_generation_kwargs(cfg),
    )


def vllm_sampling_kwargs(cfg: Mapping) -> dict:
    """Arguments of vLLM `SamplingParams` (top_k is not passed)."""
    return {
        "temperature": cfg.get("temperature"),
        "top_p": cfg.get("top_p"),
        "max_tokens": cfg.get("max_new_tokens"),
    }


def build_vllm(cfg: Mapping, json_schema: dict | None = None):
    """Batch prediction function for a `vllm` config (used for models transformers cannot run)."""
    if json_schema is not None:
        raise NotImplementedError("JSON schema enforcement is not supported for vllm models")
    from vllm import LLM, SamplingParams

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    llm = LLM(model=cfg["path"], dtype="bfloat16", trust_remote_code=True, max_model_len=cfg["max_length"])

    def predict(conversations: list[list[dict]]) -> list[str]:
        outputs = llm.chat(
            messages=conversations, sampling_params=SamplingParams(**vllm_sampling_kwargs(cfg))
        )
        return [output.outputs[0].text.strip() for output in outputs]

    return predict
