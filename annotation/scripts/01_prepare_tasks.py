"""Select the 300 impressions, assign Rad-Halo ids and write Label Studio tasks with pre-highlights.

Input: one generation log per impression source (`scripts/construction/impressions/`; JSONL with
`risk_level, config, radiology_report` (MIMIC row), `generated_impression`). From each log (sorted by
file name) the first 5/5/5/35 impressions of MedVAL risk levels 1-4 are kept. The source reports are
segmented and their reference impressions attributed with Gemini 2.5 Flash Lite (as in
`scripts/construction/synthetic/01_attribute.py`). The 300 items are shuffled with seed 0; the position
becomes the Rad-Halo id (`generated_impression_id`, = `radhalo_id`).

Each task shows the report without its impression section and the evaluated impression. Every impression
sentence is highlighted; findings sentences are highlighted if attribution links them to a sentence of
the reference impression, ignoring reference sentences linked to more than 6 findings sentences
(summaries of normal findings). Outputs: `processed_reports.jsonl` and `tasks.json` in `--out-dir`.
"""

import argparse
import json
import os
import random
from pathlib import Path

import yaml
from dotenv import load_dotenv
from labelstudio import find_span, make_task, write_json

from radhalo.data.mimic import process_report
from radhalo.data.report import Report
from radhalo.data.segmentation import remove_impression_section, split_impression
from radhalo.synthetic.attribution import (
    MODEL,
    attribute,
    example_note_ids,
    load_prompt,
    mimic_examples,
    vertexai_model,
)

REPO = Path(__file__).resolve().parents[2]
TARGETS = {1: 5, 2: 5, 3: 5, 4: 35}
MAX_ATTRIBUTED_FINDINGS = 6
SHUFFLE_SEED = 0
MODEL_VERSION = "v1_init"


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def select_impressions(log_dir: Path, targets: dict[int, int] = TARGETS) -> list[dict]:
    """First N items per risk level of each log, logs in file-name order, risk levels 1-4."""
    selected = []
    for path in sorted(Path(log_dir).glob("*.jsonl")):
        items = read_jsonl(path)
        for level, n in targets.items():
            selected.extend([item for item in items if item["risk_level"] == level][:n])
    return selected


def assign_ids(items: list[dict], seed: int = SHUFFLE_SEED) -> list[dict]:
    """Shuffle in place with `random.seed(seed)` and number the items (`generated_impression_id`)."""
    random.seed(seed)
    random.shuffle(items)
    for index, item in enumerate(items):
        item["generated_impression_id"] = index
    return items


def prehighlighted_findings(report: Report, max_attributed: int = MAX_ATTRIBUTED_FINDINGS) -> list[str]:
    """Findings sentences attributed to the reference impression (deduplicated, in order of first link)."""
    spans = [
        span
        for sentence in report.impression_sentences
        if len(sentence.attributed_findings_spans) <= max_attributed
        for span in sentence.attributed_findings_spans
    ]
    return list(dict.fromkeys(spans))


def build_task(report: Report, impression: str, rid: int) -> dict:
    """Label Studio task; sentences that cannot be located in the text are not highlighted."""
    findings_text = remove_impression_section(report.full_text)
    findings_spans = [find_span(findings_text, s) for s in prehighlighted_findings(report)]
    impression_spans = [find_span(impression, s.text) for s in split_impression(impression)]
    task = make_task(
        findings_text,
        impression,
        [s for s in findings_spans if s is not None],
        [s for s in impression_spans if s is not None],
        MODEL_VERSION,
    )
    task["data"]["generated_impression_id"] = rid
    return task


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs", type=Path, required=True, help="directory with one generation log per source"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--examples", type=Path, required=True, help="attribution worked examples 1 and 2 (YAML)"
    )
    parser.add_argument("--mimic-note-dir", type=Path, default=os.environ.get("MIMIC_NOTE_DIR"))
    parser.add_argument("--prompt", type=Path, default=REPO / "prompts" / "attribution.yaml")
    parser.add_argument("--note-ids", type=Path, default=REPO / "derived" / "few_shot_note_ids.csv")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    if args.mimic_note_dir is None:
        parser.error("set --mimic-note-dir or MIMIC_NOTE_DIR")

    examples = mimic_examples(
        example_note_ids(args.note_ids), Path(args.mimic_note_dir) / "note" / "radiology.csv"
    )
    examples.update({int(k): v for k, v in yaml.safe_load(args.examples.read_text()).items()})
    system_prompt, user_template = load_prompt(args.prompt, examples)
    generate = vertexai_model(system_prompt, os.environ.get("GCP_PROJECT_ID"), args.model)

    items = select_impressions(args.logs)
    reports = [process_report(item["radiology_report"]) for item in items]
    for report in reports:
        attribute(report, user_template, generate)
    for item, report in zip(items, reports):
        item["radiology_report"] = report
    assign_ids(items)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "processed_reports.jsonl", "w") as f:
        for item in items:
            f.write(json.dumps({**item, "radiology_report": item["radiology_report"].to_dict()}) + "\n")
    tasks = [
        build_task(i["radiology_report"], i["generated_impression"], i["generated_impression_id"])
        for i in items
    ]
    write_json(tasks, args.out_dir / "tasks.json", ensure_ascii=False)
    print(f"{len(tasks)} tasks -> {args.out_dir / 'tasks.json'}")


if __name__ == "__main__":
    main()
