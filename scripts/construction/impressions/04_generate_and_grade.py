"""Generate impressions and grade them with MedVAL until the risk-level targets are met (Sec. 3.3-3.4).

    python scripts/construction/impressions/04_generate_and_grade.py generator=gemini-2.5-flash
    python scripts/construction/impressions/04_generate_and_grade.py generator=reference \
        attribution_examples=<private yaml>

The eligible reports are shuffled with a per-generator seed and consumed in batches. Generators write an
impression with the strict prompt; the reference source keeps the MIMIC-IV impression after removing
recommendation and administrative sentences without attributed findings (attribution with Gemini 2.5
Flash Lite, as in `synthetic/01_attribute.py`). Every impression is graded by MedVAL (Gemini 2.5 Flash on
Vertex AI: GCP_PROJECT_ID, GCP_LOCATION); a failed grading is logged and the report skipped.
`<out_dir>/<generator>.jsonl` is rewritten after every batch. Needs the `medval` extra; dspy caches
MedVAL responses on disk.
"""

import json
import os
from pathlib import Path

import hydra
import yaml
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from radhalo.data.mimic import process_report
from radhalo.generation import medval
from radhalo.generation.loop import generate_and_grade, reference_impression, write_records
from radhalo.generation.prompts import build_messages, find_rows, load_prompt, mimic_examples
from radhalo.generation.sampling import generation_order
from radhalo.llm import build_model
from radhalo.synthetic import attribution

GENERATORS = Path(__file__).resolve().parents[3] / "configs" / "generators"


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def reference_impressions(cfg: DictConfig):
    """Batch function for the reference source."""
    if cfg.attribution_examples is None:
        raise ValueError("the reference source needs attribution_examples=<yaml>")
    note_ids = attribution.example_note_ids(Path(cfg.note_ids))
    rows = find_rows(Path(cfg.eligible), note_ids.values())
    examples = {n: attribution.example_blocks(process_report(rows[i])) for n, i in note_ids.items()}
    examples.update(
        {int(k): v for k, v in yaml.safe_load(Path(cfg.attribution_examples).read_text()).items()}
    )
    system_prompt, user_template = attribution.load_prompt(Path(cfg.attribution_prompt), examples)
    generate = attribution.vertexai_model(system_prompt, os.environ.get("GCP_PROJECT_ID"))
    return lambda batch: [reference_impression(row, user_template, generate) for row in batch]


def generated_impressions(cfg: DictConfig, generator: dict):
    """Batch function for an LLM generator."""
    examples = mimic_examples(
        attribution.example_note_ids(Path(cfg.note_ids), "generation"), Path(cfg.eligible)
    )
    system_prompt, user_template = load_prompt(Path(cfg.prompt), examples)
    predict = build_model(generator)
    return lambda batch: predict(build_messages(batch, system_prompt, user_template))


@hydra.main(config_path="../../../configs", config_name="generate", version_base=None)
def main(cfg: DictConfig) -> None:
    generator = OmegaConf.to_container(OmegaConf.load(GENERATORS / f"{cfg.generator}.yaml"))
    config = {**OmegaConf.to_container(cfg, resolve=True), "generation_model": generator}
    out = Path(cfg.out_dir) / f"{cfg.generator}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = generation_order(
        read_jsonl(Path(cfg.eligible)), cfg.random_seed, generator["path"], generator["type"]
    )
    if generator["type"] == "none":
        impressions = reference_impressions(cfg)
    else:
        impressions = generated_impressions(cfg, generator)
    grade = medval.build_grader(
        os.environ.get("GCP_PROJECT_ID"), os.environ.get("GCP_LOCATION"), cfg.medval_model
    )

    buckets = generate_and_grade(
        rows,
        generator["batch_size"],
        impressions,
        grade,
        targets=cfg.targets,
        max_generations=cfg.max_generations,
        on_batch=lambda b: write_records(b, config, out),
    )
    counts = {level: len(samples) for level, samples in buckets.items()}
    print(f"{cfg.generator}: risk-level counts {counts} (targets {dict(cfg.targets)}) -> {out}")


if __name__ == "__main__":
    load_dotenv()
    main()
