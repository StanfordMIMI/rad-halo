"""Label Studio task format and export helpers shared by the annotation scripts.

Tasks hold the report without its impression section (`findings_text`) and the evaluated impression
(`impression_text`). Sentences are text spans labelled with `label_findings` / `label_impression`.
`generated_impression_id` in the task data is the Rad-Halo report id (`radhalo_id`).
"""

import json
import re
from pathlib import Path

import yaml

FINDINGS = ("label_findings", "text_findings")
IMPRESSION = ("label_impression", "text_impression")
TO_REVIEW = "To Review"
ANNOTATORS = ("annotator_1", "annotator_2", "annotator_3", "annotator_4")


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path: Path, ensure_ascii: bool = True) -> None:
    """Indented JSON, as all pipeline files were written."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=ensure_ascii)


def find_span(text: str, sentence: str) -> tuple[int, int, str] | None:
    """First occurrence of `sentence` in `text`, where each space may match any run of whitespace."""
    match = re.search(re.escape(sentence).replace(r"\ ", r"\s+"), text)
    if match is None:
        return None
    return match.start(), match.end(), text[match.start() : match.end()]


def span_result(names: tuple[str, str], start: int, end: int, text: str, label: str = TO_REVIEW) -> dict:
    return {
        "from_name": names[0],
        "to_name": names[1],
        "type": "labels",
        "value": {"start": start, "end": end, "text": text, "labels": [label]},
    }


def make_task(
    findings_text: str,
    impression_text: str,
    findings_spans: list[tuple[int, int, str]],
    impression_spans: list[tuple[int, int, str]],
    model_version: str,
) -> dict:
    """A task whose highlighted sentences are "To Review" predictions."""
    results = [span_result(FINDINGS, *s) for s in findings_spans]
    results += [span_result(IMPRESSION, *s) for s in impression_spans]
    return {
        "data": {"findings_text": findings_text, "impression_text": impression_text, "metadata": None},
        "predictions": [{"model_version": model_version, "result": results}],
    }


def annotations_to_predictions(tasks: list[dict], model_version: str) -> list[dict]:
    """Turn the first annotation of each exported task into a prediction for re-import.

    Tasks without annotations keep the result of their first prediction.
    """
    out = []
    for task in tasks:
        if task.get("annotations"):
            results = task["annotations"][0].get("result", [])
        elif task.get("predictions"):
            results = task["predictions"][0].get("result", [])
        else:
            results = []
        out.append(
            {
                "data": task.get("data", {}),
                "predictions": [{"model_version": model_version, "result": results}],
            }
        )
    return out


def normalize_newlines(tasks: list[dict]) -> list[dict]:
    """Fix span texts of an export in place.

    Exports may store newlines as the two characters "\\n". These are converted, and leading/trailing
    newlines are stripped with start/end moved accordingly (offsets refer to the task text).
    """
    for task in tasks:
        for annotation in task.get("annotations", []):
            for result in annotation.get("result", []):
                value = result.get("value", {})
                if "text" not in value:
                    continue
                text = value["text"].replace("\\n", "\n")
                value["start"] += len(text) - len(text.lstrip("\n"))
                value["end"] -= len(text) - len(text.rstrip("\n"))
                value["text"] = text.strip("\n")
    return tasks


def read_exports(directory: Path) -> list[dict]:
    """All tasks of the Label Studio JSON exports in a directory, newlines normalised."""
    tasks = []
    for path in sorted(Path(directory).glob("*.json")):
        tasks.extend(normalize_newlines(read_json(path)))
    return tasks


def read_user_map(path: Path) -> dict[str, int]:
    """Rad-Halo role -> Label Studio user id (`completed_by`); see `user_map.example.yaml`."""
    user_map = yaml.safe_load(Path(path).read_text())
    missing = [role for role in ("anchor", *ANNOTATORS) if not isinstance(user_map.get(role), int)]
    if missing:
        raise ValueError(f"{path}: no Label Studio user id for {', '.join(missing)}")
    return user_map
