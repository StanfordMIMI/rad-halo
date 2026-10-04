"""Attribution linking with Gemini 2.5 Flash Lite (App. D.1).

The model gets the numbered findings sentences (headers removed) and the numbered impression sentences
and returns, per impression sentence, its type and the indices of supporting findings sentences.

The prompt contains three worked examples. Their findings and impression lines are derived from MIMIC-IV
reports and are not part of this repository: `prompts/attribution.yaml` holds placeholders such as
`{example_3_findings}`, filled by `fill_examples`. Examples listed in `derived/few_shot_note_ids.csv`
are rendered from the reader's MIMIC-IV-Note copy; the others are read from a separate file.
"""

import copy
import csv
import json
from collections.abc import Callable, Mapping
from pathlib import Path

import yaml

from radhalo.data.mimic import eligible_chunks, process_report
from radhalo.data.report import Report
from radhalo.data.segmentation import is_header
from radhalo.synthetic.perturbation import MAX_SPECIFIC_SPANS

MODEL = "gemini-2.5-flash-lite"
LOCATION = "us-central1"
GENERATION_CONFIG = {"temperature": 0, "max_output_tokens": 8192, "response_mime_type": "application/json"}
EXAMPLE_PARTS = ("findings", "impression")


def findings_without_headers(report: Report) -> list[str]:
    return [s for s in report.findings_sentences if not is_header(s)]


def numbered_findings(report: Report) -> str:
    return "\n".join(f"{i}: {s}" for i, s in enumerate(findings_without_headers(report)))


def numbered_impression(report: Report) -> str:
    return "\n".join(f"{i}: {s.text}" for i, s in enumerate(report.impression_sentences))


def example_blocks(report: Report) -> dict[str, str]:
    """Findings and impression lines of a worked example, rendered like the model input."""
    return {"findings": numbered_findings(report), "impression": numbered_impression(report)}


def example_note_ids(path: Path, prompt: str = "attribution") -> dict[int, str]:
    """Example number -> MIMIC note id, from `derived/few_shot_note_ids.csv`."""
    with open(path) as f:
        return {
            int(row["example"]): row["mimic_note_id"] for row in csv.DictReader(f) if row["prompt"] == prompt
        }


def mimic_examples(note_ids: Mapping[int, str], radiology_csv: Path) -> dict[int, dict[str, str]]:
    """Render the worked examples from their MIMIC-IV-Note reports."""
    rows = {}
    for chunk in eligible_chunks(radiology_csv):
        for row in chunk[chunk["note_id"].isin(set(note_ids.values()))].to_dict("records"):
            rows[row["note_id"]] = row
    return {number: example_blocks(process_report(rows[note_id])) for number, note_id in note_ids.items()}


def fill_examples(user_template: str, examples: Mapping[int, Mapping[str, str]]) -> str:
    """Insert the worked examples ({example number: {"findings": ..., "impression": ...}})."""
    for number, blocks in examples.items():
        for part in EXAMPLE_PARTS:
            user_template = user_template.replace(f"{{example_{number}_{part}}}", blocks[part])
    if "{example_" in user_template:
        raise ValueError("attribution prompt has unfilled worked examples")
    return user_template


def load_prompt(path: Path, examples: Mapping[int, Mapping[str, str]]) -> tuple[str, str]:
    """System prompt and user template (with `{findings_text}`, `{impression_sentences}`)."""
    prompt = yaml.safe_load(Path(path).read_text())
    return prompt["system"], fill_examples(prompt["user"], examples)


def build_prompt(user_template: str, report: Report) -> str:
    return user_template.format(
        findings_text=numbered_findings(report), impression_sentences=numbered_impression(report)
    )


def prune_attribution_map(attribution_map: dict) -> dict:
    """Remove findings linked to a specific sentence (<= 3 links) from blanket sentences (> 3 links).

    Counts are taken on the raw response, before out-of-range indices are dropped.
    """
    pruned = copy.deepcopy(attribution_map)
    specific = [k for k, v in pruned.items() if len(v.get("findings_sentences", [])) <= MAX_SPECIFIC_SPANS]
    blanket = [k for k, v in pruned.items() if len(v.get("findings_sentences", [])) > MAX_SPECIFIC_SPANS]
    covered = {f for k in specific for f in pruned[k].get("findings_sentences", [])}
    for k in blanket:
        pruned[k]["findings_sentences"] = [
            f for f in pruned[k].get("findings_sentences", []) if f not in covered
        ]
    return pruned


def apply_response(report: Report, response: str) -> None:
    """Parse a model response and set type flags and attributed findings on the impression sentences.

    Sentences missing from the response are left unchanged; out-of-range indices are dropped.
    """
    attribution_map = prune_attribution_map(
        json.loads(response.replace("```json", "").replace("```", "").strip())
    )
    findings = findings_without_headers(report)
    for i, sentence in enumerate(report.impression_sentences):
        if str(i) not in attribution_map:
            continue
        entry = attribution_map[str(i)]
        sentence.is_admin_note = entry["type"] == "admin"
        sentence.is_recommendation = entry["type"] == "recommendation"
        indices = entry.get("findings_sentences", [])
        sentence.attributed_findings_spans = [findings[j] for j in indices if 0 <= j < len(findings)]


def vertexai_model(system_prompt: str, project: str | None, model: str = MODEL) -> Callable[[str], str]:
    """Prompt -> response text with the `vertexai` SDK, as in the paper runs (no retries)."""
    import vertexai
    from vertexai.generative_models import GenerationConfig, GenerativeModel

    vertexai.init(project=project, location=LOCATION)
    client = GenerativeModel(model, system_instruction=system_prompt)
    config = GenerationConfig(**GENERATION_CONFIG)

    def generate(prompt: str) -> str:
        return client.generate_content(prompt, generation_config=config).text

    return generate


def attribute(report: Report, user_template: str, generate: Callable[[str], str]) -> None:
    """Attribute one report in place."""
    apply_response(report, generate(build_prompt(user_template, report)))
