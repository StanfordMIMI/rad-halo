"""The annotated Rad-Halo dataset: build from Label Studio exports, adjudicate, finalise, release.

A report is a dict with `mimic_notes_id, radhalo_id, medval_risk_level, generation_model, report_text,
impression_text, findings_sentences, impression_sentences` (+ `imaging_technique, anatomy_location`
after the imaging step). Sentences hold `start, end, text` and per-annotator labels in `is_omission`
(findings) or `is_hallucination` (impression); `adjudicator` is added for disputed sentences.
"""

import uuid

from labelstudio import ANNOTATORS, FINDINGS, IMPRESSION, TO_REVIEW, span_result

from radhalo.generation.sampling import source_id

ADJUDICATOR = "adjudicator"
# error type -> (sentence list, label dict, Label Studio names, positive and negative resident label)
ERROR_TYPES = {
    "hallucination": ("impression_sentences", "is_hallucination", IMPRESSION, "HALLUCINATION"),
    "omission": ("findings_sentences", "is_omission", FINDINGS, "OMISSION"),
}
# Adjudication labels (label_studio/labeling_config_{hallucinations,omissions}.xml).
ADJUDICATION_LABELS = {
    "hallucination": ("Hallucination", "Not a hallucination"),
    "omission": ("Omission", "Not an omission"),
}
ADJUDICATION_MODEL_VERSION = "adjudication_v1"


# --- Build the dataset from resident annotations ---


def anchor_report(anchor_task: dict, processed: dict, anchor_user: int) -> dict:
    """Report with the anchor segmentation (sentence spans of the anchor user's annotation), no labels."""
    matching = [a for a in anchor_task["annotations"] if a["completed_by"] == anchor_user]
    rid = processed["generated_impression_id"]
    if len(matching) != 1:
        raise ValueError(f"radhalo_id {rid}: expected one anchor annotation, found {len(matching)}")
    results = matching[0]["result"]

    def spans(names: tuple[str, str], label_key: str) -> list[dict]:
        return [
            {
                label_key: {},
                "start": r["value"]["start"],
                "end": r["value"]["end"],
                "text": r["value"]["text"],
            }
            for r in results
            if r["from_name"] == names[0]
        ]

    return {
        "mimic_notes_id": processed["radiology_report"]["id"],
        "radhalo_id": rid,
        "medval_risk_level": processed["risk_level"],
        "generation_model": source_id(processed["config"]["generation_model"]),
        "report_text": anchor_task["data"]["findings_text"],
        "impression_text": anchor_task["data"]["impression_text"],
        "findings_sentences": spans(FINDINGS, "is_omission"),
        "impression_sentences": spans(IMPRESSION, "is_hallucination"),
    }


def build_reports(anchor_tasks: list[dict], processed: list[dict], anchor_user: int) -> list[dict]:
    """One unlabelled report per id present in both the anchor export and the processed data."""
    anchors = {}
    for task in anchor_tasks:
        rid = task["data"]["generated_impression_id"]
        if rid in anchors:
            raise ValueError(f"radhalo_id {rid}: several anchor tasks")
        anchors[rid] = task
    by_id = {}
    for item in processed:
        rid = item["generated_impression_id"]
        if rid in by_id:
            raise ValueError(f"radhalo_id {rid}: several processed entries")
        by_id[rid] = item
    return [
        anchor_report(anchors[rid], by_id[rid], anchor_user) for rid in sorted(anchors.keys() & by_id.keys())
    ]


def annotations_to_use(task: dict, annotator_user: int, anchor_user: int) -> list[dict]:
    """The anchor user's annotations of a task if present, else the annotator's own.

    The anchor user copied each resident annotation onto the anchor segmentation; those copies carry
    the resident's labels and are preferred.
    """
    by_anchor = [a for a in task["annotations"] if a["completed_by"] == anchor_user]
    return by_anchor or [a for a in task["annotations"] if a["completed_by"] == annotator_user]


def _same_span(result: dict, sentence: dict) -> bool:
    value = result["value"]
    return (
        value["text"] == sentence["text"]
        and value["start"] == sentence["start"]
        and value["end"] == sentence["end"]
    )


def _sentences_for(report: dict, from_name: str) -> list[dict]:
    if from_name == FINDINGS[0]:
        return report["findings_sentences"]
    if from_name == IMPRESSION[0]:
        return report["impression_sentences"]
    raise ValueError(f"unknown from_name {from_name}")


def _labelled_results(exports: dict[str, list[dict]], user_map: dict[str, int]):
    """Yield (annotator, radhalo_id, result) for every sentence result the annotators' labels come from."""
    for annotator, tasks in exports.items():
        for task in tasks:
            for annotation in annotations_to_use(task, user_map[annotator], user_map["anchor"]):
                for result in annotation["result"]:
                    if result["from_name"] in (FINDINGS[0], IMPRESSION[0]):
                        yield annotator, task["data"]["generated_impression_id"], result


def validation_problems(
    exports: dict[str, list[dict]], reports: list[dict], user_map: dict[str, int]
) -> list[tuple[str, int, str, int, int, str]]:
    """Spans outside the anchor segmentation and spans still labelled "To Review" (ids and offsets only)."""
    by_id = {r["radhalo_id"]: r for r in reports}
    problems = []
    for annotator, rid, result in _labelled_results(exports, user_map):
        value = result["value"]
        where = (annotator, rid, result["from_name"], value["start"], value["end"])
        if not any(_same_span(result, s) for s in _sentences_for(by_id[rid], result["from_name"])):
            problems.append((*where, "segmentation"))
        if TO_REVIEW in value["labels"]:
            problems.append((*where, TO_REVIEW))
    return problems


def add_resident_labels(
    exports: dict[str, list[dict]], reports: list[dict], user_map: dict[str, int]
) -> None:
    """Set each annotator's label on every sentence of the reports they annotated.

    Sentences without an explicit "HALLUCINATION"/"OMISSION" label are negative. `exports` maps
    annotator ids (`annotator_1`..`annotator_4`) to their exported tasks.
    """
    by_id = {r["radhalo_id"]: r for r in reports}
    for annotator in sorted(exports):
        for task in exports[annotator]:
            report = by_id[task["data"]["generated_impression_id"]]
            for sentence in report["impression_sentences"]:
                sentence["is_hallucination"][annotator] = False
            for sentence in report["findings_sentences"]:
                sentence["is_omission"][annotator] = False
            for annotation in annotations_to_use(task, user_map[annotator], user_map["anchor"]):
                for result in annotation["result"]:
                    _set_label(report, result, annotator)


def _set_label(report: dict, result: dict, annotator: str) -> None:
    for error_type in ERROR_TYPES.values():
        sentence_key, label_key, names, positive = error_type
        if result["from_name"] != names[0]:
            continue
        for sentence in report[sentence_key]:
            if _same_span(result, sentence):
                sentence[label_key][annotator] = positive in result["value"]["labels"]
                break


# --- Adjudication ---


def describe_labels(labels: dict[str, bool], error_type: str) -> str:
    """Resident labels shown to the adjudicator, e.g. "Annotator 1: Omission | Annotator 3: Not Omission"."""
    positive = "Hallucination" if error_type == "hallucination" else "Omission"
    return " | ".join(
        f"{a.replace('annotator_', 'Annotator ')}: {positive if labels[a] else 'Not ' + positive}"
        for a in sorted(labels)
    )


def adjudication_tasks(reports: list[dict], error_type: str) -> list[dict]:
    """One task per sentence on which the residents disagree, with the sentence pre-highlighted."""
    sentence_key, label_key, names, _ = ERROR_TYPES[error_type]
    tasks = []
    for report in reports:
        for index, sentence in enumerate(report[sentence_key]):
            labels = sentence[label_key]
            if len(set(labels.values())) <= 1:
                continue
            result = {
                "id": str(uuid.uuid4())[:8],
                **span_result(names, sentence["start"], sentence["end"], sentence["text"]),
            }
            tasks.append(
                {
                    "data": {
                        "radhalo_id": report["radhalo_id"],
                        "sentence_index": index,
                        "generation_model": report["generation_model"],
                        "medval_risk_level": report["medval_risk_level"],
                        "findings_text": report["report_text"],
                        "impression_text": report["impression_text"],
                        "annotator_labels": describe_labels(labels, error_type),
                    },
                    "predictions": [{"model_version": ADJUDICATION_MODEL_VERSION, "result": [result]}],
                }
            )
    return tasks


def adjudicator_decision(task: dict, error_type: str) -> bool | None:
    """Decision in the first annotation of an exported adjudication task; None if not adjudicated."""
    names = ERROR_TYPES[error_type][2]
    positive, negative = ADJUDICATION_LABELS[error_type]
    if not task.get("annotations"):
        return None
    for result in task["annotations"][0].get("result", []):
        if result.get("from_name") != names[0]:
            continue
        labels = result.get("value", {}).get("labels", [])
        if positive in labels:
            return True
        if negative in labels:
            return False
    return None


def add_adjudications(
    reports: list[dict], tasks: list[dict], error_type: str
) -> tuple[int, list[tuple[int, int]]]:
    """Add the adjudicator's label to the disputed sentences; returns (integrated, skipped (id, index))."""
    sentence_key, label_key, _, _ = ERROR_TYPES[error_type]
    by_id = {r["radhalo_id"]: r for r in reports}
    integrated, skipped = 0, []
    for task in tasks:
        rid, index = task["data"]["radhalo_id"], task["data"]["sentence_index"]
        decision = adjudicator_decision(task, error_type)
        if decision is None or rid not in by_id:
            skipped.append((rid, index))
            continue
        by_id[rid][sentence_key][index][label_key][ADJUDICATOR] = decision
        integrated += 1
    return integrated, skipped


def adjudicated_label(labels: dict[str, bool]) -> bool:
    """The adjudicator's label for disputed sentences, else the unanimous resident label."""
    if ADJUDICATOR in labels:
        return labels[ADJUDICATOR]
    residents = [v for k, v in labels.items() if k != ADJUDICATOR]
    assert len(set(residents)) == 1, f"no adjudicator label but residents disagree ({len(residents)} labels)"
    return residents[0]


def finalize(reports: list[dict]) -> None:
    """Add `adjudicated_label` to every sentence in place."""
    for report in reports:
        for sentence_key, label_key, _, _ in ERROR_TYPES.values():
            for sentence in report[sentence_key]:
                sentence["adjudicated_label"] = adjudicated_label(sentence[label_key])


# --- PhysioNet release ---

RELEASE_RENAMES = {"report_text": "background_and_findings_text", "generation_model": "impression_source"}
REPORT_COLUMNS = [
    "radhalo_id",
    "mimic_notes_id",
    "impression_source",
    "medval_risk_level",
    "imaging_technique",
    "anatomy_location",
    "background_and_findings_text",
    "impression_text",
]
LABEL_COLUMNS = [*(f"{a}_label" for a in ANNOTATORS), f"{ADJUDICATOR}_label"]
SENTENCE_COLUMNS = ["radhalo_id", "sentence_id", "start", "end", "text", *LABEL_COLUMNS, "adjudicated_label"]


def release_reports(reports: list[dict]) -> list[dict]:
    """Final reports with the release field names (key order kept)."""
    return [{RELEASE_RENAMES.get(k, k): v for k, v in report.items()} for report in reports]


def _csv_bool(value: bool | None) -> str:
    return "" if value is None else str(value).lower()


def sentence_rows(reports: list[dict], error_type: str) -> list[list]:
    """Flattened sentences; label cells are "true", "false" or empty if that rater did not label."""
    sentence_key, label_key, _, _ = ERROR_TYPES[error_type]
    rows = []
    for report in reports:
        for index, sentence in enumerate(report[sentence_key]):
            labels = sentence[label_key]
            rows.append(
                [report["radhalo_id"], index, sentence["start"], sentence["end"], sentence["text"]]
                + [_csv_bool(labels.get(rater)) for rater in (*ANNOTATORS, ADJUDICATOR)]
                + [_csv_bool(sentence["adjudicated_label"])]
            )
    return rows
