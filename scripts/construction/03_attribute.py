"""Link impression sentences to supporting findings sentences with Gemini 2.5 Flash Lite (App. D.1).

Input: the report pool of `02_preprocess.py`. Calls Vertex AI once per report (GCP_PROJECT_ID; region
us-central1) and writes the reports with sentence types and attributed findings.

Worked example 3 of the prompt is rendered from MIMIC-IV-Note (`derived/few_shot_note_ids.csv`);
examples 1 and 2 are not reports in MIMIC-IV-Note and are read from `--examples`, a YAML file
{1: {findings: ..., impression: ...}, 2: {...}} with the numbered lines.
"""

import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from radhalo.data.report import read_reports, write_reports
from radhalo.synthetic.attribution import (
    MODEL,
    attribute,
    example_note_ids,
    load_prompt,
    mimic_examples,
    vertexai_model,
)

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="report pool (JSONL from 02_preprocess.py)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True, help="worked examples not in MIMIC-IV-Note")
    parser.add_argument(
        "--mimic-note-dir",
        type=Path,
        default=os.environ.get("MIMIC_NOTE_DIR"),
        help="MIMIC-IV-Note directory containing note/radiology.csv (default: $MIMIC_NOTE_DIR)",
    )
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

    reports = list(read_reports(args.input))
    for report in tqdm(reports, desc="Attribution"):
        attribute(report, user_template, generate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_reports(reports, args.out)
    print(f"{len(reports):,} attributed reports -> {args.out}")


if __name__ == "__main__":
    main()
