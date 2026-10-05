"""Load the Rad-Halo PhysioNet release and detector predictions."""

import json
from pathlib import Path

ADJUDICATOR = "adjudicator"

# error type -> (sentence list key, per-annotator label key)
ERROR_TYPES = {
    "hallucination": ("impression_sentences", "is_hallucination"),
    "omission": ("findings_sentences", "is_omission"),
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
    """Return the 300 reports of `rad_halo_dataset.json`, ordered by radhalo_id."""
    path = Path(data_dir) / "rad_halo_dataset.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; set RADHALO_DATA to the Rad-Halo PhysioNet release")
    reports = json.loads(path.read_text())
    return sorted(reports, key=lambda r: r["radhalo_id"])


def sentences(report: dict, error_type: str) -> list[dict]:
    return report[ERROR_TYPES[error_type][0]]


def annotations(sentence: dict, error_type: str) -> dict[str, bool]:
    """Labels of the residents and, for disputed sentences, the adjudicator."""
    return sentence[ERROR_TYPES[error_type][1]]


def resident_labels(sentence: dict, error_type: str) -> dict[str, bool]:
    return {k: v for k, v in annotations(sentence, error_type).items() if k != ADJUDICATOR}


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
