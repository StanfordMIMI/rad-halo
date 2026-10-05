"""Write Label Studio tasks for the thematic analysis of errors in reference impressions (App. E.3).

Input: `rad-halo_dataset_imaging_technique.json` (resident labels, before adjudication). One task per
sentence that at least one resident labelled as an error, restricted to one impression source
(default: the MIMIC-IV reference impressions), with the sentence highlighted read-only as "Error" and
`annotators_agree` in the task data. Tasks are shuffled with seed 0 (hallucinations, then omissions,
from one random state). Coded with `label_studio/thematic_coding.xml`.
"""

import argparse
import random
from pathlib import Path

from annotated_dataset import ERROR_TYPES
from labelstudio import read_json, write_json

ERROR_LABEL = "Error"
MODEL_VERSION = "radhalo_error"
SEED = 0


def error_task(report: dict, sentence: dict, labels: dict[str, bool], error_type: str) -> dict:
    """Task with the error sentence highlighted at its first exact occurrence (no highlight if absent)."""
    names = ERROR_TYPES[error_type][2]
    text = report["impression_text"] if error_type == "hallucination" else report["report_text"]
    start = text.find(sentence["text"])
    result = []
    if start >= 0:
        value = {
            "start": start,
            "end": start + len(sentence["text"]),
            "text": sentence["text"],
            "labels": [ERROR_LABEL],
        }
        result.append(
            {"from_name": names[0], "to_name": names[1], "type": "labels", "value": value, "readonly": True}
        )
    values = list(labels.values())
    return {
        "data": {
            "radhalo_id": report["radhalo_id"],
            "medval_risk_level": report["medval_risk_level"],
            "annotators_agree": all(values) or not any(values),
            "findings_text": report["report_text"],
            "impression_text": report["impression_text"],
        },
        "predictions": [{"model_version": MODEL_VERSION, "result": result}],
    }


def error_tasks(reports: list[dict], error_type: str) -> list[dict]:
    """Tasks for sentences labelled as an error by any resident (union rule)."""
    sentence_key, label_key, _, _ = ERROR_TYPES[error_type]
    return [
        error_task(report, sentence, sentence[label_key], error_type)
        for report in reports
        for sentence in report[sentence_key]
        if any(sentence[label_key].values())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="rad-halo_dataset_imaging_technique.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source", default="original_radiologist", help="impression source to analyse")
    args = parser.parse_args()

    reports = [r for r in read_json(args.dataset) if r["generation_model"] == args.source]
    hallucinations, omissions = error_tasks(reports, "hallucination"), error_tasks(reports, "omission")
    random.seed(SEED)
    random.shuffle(hallucinations)
    random.shuffle(omissions)
    for name, tasks in (("hallucinations", hallucinations), ("omissions", omissions)):
        out = args.out_dir / f"{name}_{args.source}.json"
        write_json(tasks, out, ensure_ascii=False)
        print(f"{len(tasks)} tasks -> {out}")


if __name__ == "__main__":
    main()
