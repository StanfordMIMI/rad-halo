"""RadCheck inference (Sec. 4.2, App. C.3): one classification per candidate sentence.

Each findings sentence is scored for omission and each impression sentence for hallucination, with the
template used in training (`prompts/radcheck.txt`). The score is the softmax probability of class 1.
"""

import json
from pathlib import Path

from radhalo.data.segmentation import clean_whitespace, impression_text, split_impression

OMISSION_TASK = "DETECT_OMISSION"
HALLUCINATION_TASK = "DETECT_HALLUCINATION"
MAX_LENGTH = 1536


def report_inputs(report: dict) -> tuple[str, str, str]:
    """Context, findings text and impression text shared by all candidates of a report.

    The findings text is the raw text after "FINDINGS", and the impression is re-split and renumbered,
    as in the synthetic training data.
    """
    text = report["background_and_findings_text"]
    context = clean_whitespace(text.split("FINDINGS")[0])
    parts = text.split("FINDINGS")
    if len(parts) > 1:
        findings = parts[1].lstrip(": \n")
    else:
        findings = " ".join(clean_whitespace(s["text"]) for s in report["findings_sentences"])
    impression = impression_text(split_impression(report["impression_text"]))
    return context, clean_whitespace(findings), impression


def candidate_prompts(report: dict, template: str) -> tuple[list[str], list[str]]:
    """Prompts for the findings sentences (omission) and the impression sentences (hallucination)."""
    context, findings, impression = report_inputs(report)

    def render(task: str, sentence: dict) -> str:
        return template.format(
            task=task,
            context=context,
            findings_text=findings,
            impression_text=impression,
            candidate_sentence=clean_whitespace(sentence["text"]),
        )

    return (
        [render(OMISSION_TASK, s) for s in report["findings_sentences"]],
        [render(HALLUCINATION_TASK, s) for s in report["impression_sentences"]],
    )


def load_tokenizer(checkpoint: Path):
    """Tokenizer saved with the adapter, left padding and truncation, EOS as pad token if none is set."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left", truncation_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_classifier(checkpoint: Path, device: str):
    """Base model named in `adapter_config.json` (bf16, no quantization) with the LoRA adapter."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification

    checkpoint = Path(checkpoint)
    base_path = json.loads((checkpoint / "adapter_config.json").read_text())["base_model_name_or_path"]
    tokenizer = load_tokenizer(checkpoint)
    base = AutoModelForSequenceClassification.from_pretrained(
        base_path, num_labels=2, torch_dtype=torch.bfloat16, device_map=device
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, str(checkpoint))
    model.eval()
    return tokenizer, model


def tokenize(tokenizer, prompts: list[str]):
    """Pad to the longest prompt of the batch; truncate at `MAX_LENGTH` tokens."""
    return tokenizer(prompts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")


def score_prompts(prompts: list[str], tokenizer, model, batch_size: int, device: str) -> list[float]:
    import torch

    scores = []
    for start in range(0, len(prompts), batch_size):
        inputs = tokenize(tokenizer, prompts[start : start + batch_size]).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)
        scores.extend(probs[:, 1].cpu().tolist())
    return scores


def score_report(report: dict, template: str, tokenizer, model, batch_size: int, device: str) -> dict:
    """Scores of one report; findings and impression candidates are batched separately."""
    findings_prompts, impression_prompts = candidate_prompts(report, template)
    return {
        "findings_scores": score_prompts(findings_prompts, tokenizer, model, batch_size, device),
        "impression_scores": score_prompts(impression_prompts, tokenizer, model, batch_size, device),
    }
