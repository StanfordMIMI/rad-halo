"""Load the Rad-Halo PhysioNet release and detector predictions."""

import json
from pathlib import Path

DATASET_FILE = "rad_halo.json"

# error type -> sentence list key
ERROR_TYPES = {
    "hallucination": "impression_sentences",
    "omission": "findings_sentences",
}

IMPRESSION_SOURCES = {
    "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "mistralai/Ministral-3-3B-Instruct-2512": "Ministral 3 3B",
    "Qwen/Qwen3-1.7B": "Qwen 3 1.7B",
    "original_radiologist": "MIMIC-IV reference",
}


def load_reports(data_dir: Path) -> list[dict]:
    """Return the 300 reports of `rad_halo.json`, ordered by radhalo_id."""
    path = Path(data_dir) / DATASET_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; set RADHALO_DATA to the Rad-Halo PhysioNet release")
    reports = json.loads(path.read_text())
    return sorted(reports, key=lambda r: r["radhalo_id"])


def sentences(report: dict, error_type: str) -> list[dict]:
    return report[ERROR_TYPES[error_type]]


def resident_labels(sentence: dict) -> dict[str, bool]:
    """Labels of the residents who labeled the report (the adjudicator's label is `adjudicator_label`)."""
    return sentence["annotator_labels"]


def load_predictions(path: Path, reports: list[dict]) -> dict[int, dict[str, list[float]]]:
    """Read a prediction file and check that every report has one score per sentence."""
    by_id = {r["radhalo_id"]: r for r in reports}
    predictions = {}
    for line in Path(path).read_text().splitlines():
        item = json.loads(line)
        rid = item["radhalo_id"]
        scores = {
            "hallucination": item["impression_scores"],
            "omission": item["findings_scores"],
        }
        for error_type, values in scores.items():
            if len(values) != len(sentences(by_id[rid], error_type)):
                raise ValueError(f"{path}: radhalo_id {rid} has a wrong number of {error_type} scores")
        predictions[rid] = scores
    return predictions
