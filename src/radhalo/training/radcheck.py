"""Fine-tune RadCheck: QLoRA on Qwen3-Embedding-8B with a binary classification head (App. C.4).

The synthetic samples are split 95/5 by sample (`datasets` `train_test_split`, seed 42), padded and
truncated on the left to 1,536 tokens, and trained for two epochs. The checkpoint with the highest
validation F1 (evaluated every 25 steps) is kept and saved with the tokenizer and the resolved config.
"""

import os
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf


def load_splits(cfg: DictConfig):
    """Train and validation `datasets.Dataset` from the synthetic JSONL."""
    from datasets import load_dataset
    from hydra.utils import to_absolute_path

    dataset = load_dataset("json", data_files=to_absolute_path(cfg.dataset.input_file), split="train")
    split = dataset.train_test_split(test_size=cfg.dataset.validation_split, seed=cfg.seed)
    return split["train"], split["test"]


def load_tokenizer(cfg: DictConfig):
    """Left padding and truncation; EOS as pad token if the tokenizer has none."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.path, token=os.environ.get("HF_TOKEN"), padding_side="left", truncation_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_function(tokenizer, cfg: DictConfig):
    """Batched `Dataset.map` function: pad every sample to `max_length`, copy the labels."""

    def tokenize(examples: dict) -> dict:
        tokenized = tokenizer(
            examples[cfg.dataset.text_column],
            padding="max_length",
            truncation=True,
            max_length=cfg.dataset.max_length,
        )
        tokenized["labels"] = examples[cfg.dataset.label_column]
        return tokenized

    return tokenize


def load_model(cfg: DictConfig, tokenizer):
    """4-bit NF4 base model with gradient checkpointing and LoRA adapters.

    `prepare_model_for_kbit_training` is not applied (as in the paper run).
    """
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=cfg.quantization.load_in_4bit,
        bnb_4bit_compute_dtype=getattr(torch, cfg.quantization.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.path,
        token=os.environ.get("HF_TOKEN"),
        num_labels=cfg.model.num_labels,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=torch.bfloat16 if cfg.training.bf16 else torch.float16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
    )
    return get_peft_model(model, lora)


def compute_metrics(eval_pred) -> dict[str, float]:
    """Accuracy, binary precision/recall/F1 and confusion counts of the argmax predictions."""
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def training_arguments(cfg: DictConfig):
    from transformers import TrainingArguments

    t = cfg.training
    return TrainingArguments(
        output_dir=cfg.output.model_dir,
        num_train_epochs=t.num_epochs,
        per_device_train_batch_size=cfg.model.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.model.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        weight_decay=t.weight_decay,
        warmup_ratio=t.warmup_ratio,
        eval_strategy=t.eval_strategy,
        eval_steps=t.eval_steps,
        save_strategy=t.save_strategy,
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        load_best_model_at_end=t.load_best_model_at_end,
        metric_for_best_model=t.metric_for_best_model,
        greater_is_better=t.greater_is_better,
        logging_dir="./logs",
        logging_steps=t.logging_steps,
        report_to="wandb" if cfg.wandb.enabled else "none",
        fp16=False,
        bf16=t.bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        push_to_hub=False,
        seed=cfg.seed,
    )


def init_wandb(cfg: DictConfig) -> None:
    import wandb

    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name
        or f"{cfg.model.name}_ep{cfg.training.num_epochs}_lr{cfg.training.learning_rate}_r{cfg.lora.r}",
        group=cfg.model.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[*cfg.wandb.tags, cfg.model.name, "lora", "4bit"],
    )


def train(cfg: DictConfig) -> dict:
    """Train, evaluate the selected checkpoint and save it to `output.final_model_dir`."""
    from transformers import Trainer

    if cfg.wandb.enabled:
        init_wandb(cfg)
    train_set, eval_set = load_splits(cfg)
    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg, tokenizer)
    tokenize = tokenize_function(tokenizer, cfg)
    drop = [c for c in train_set.column_names if c not in ("input_ids", "attention_mask", "labels")]
    trainer = Trainer(
        model=model,
        args=training_arguments(cfg),
        train_dataset=train_set.map(tokenize, batched=True, remove_columns=drop, desc="Tokenizing train"),
        eval_dataset=eval_set.map(tokenize, batched=True, remove_columns=drop, desc="Tokenizing eval"),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    result = trainer.evaluate()

    final_dir = cfg.output.final_model_dir
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    OmegaConf.save(cfg, Path(final_dir) / "training_config.yaml")
    if cfg.wandb.enabled:
        import wandb

        wandb.summary.update({f"final_{k}": v for k, v in result.items()})
        wandb.finish()
    return result
