"""Annotation provenance code (`annotation/scripts/`) on small fictional inputs."""

import csv
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from radhalo.data.report import ImpressionSentence, Report

SCRIPTS = Path(__file__).resolve().parents[1] / "annotation" / "scripts"
FINDINGS_TEXT = "FINDINGS: Lungs are clear.\nNo  effusion. Heart is normal."
IMPRESSION_TEXT = "1. No acute process.\n2. Normal heart."
USER_MAP = {"anchor": 10, "annotator_1": 11, "annotator_2": 12, "annotator_3": 13, "annotator_4": 14}


@pytest.fixture(scope="module")
def m():
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str(SCRIPTS))
        names = [
            "labelstudio",
            "annotated_dataset",
            "01_prepare_tasks",
            "03_split_partitions",
            "06_merge_imaging",
            "thematic_export",
            "thematic_extract",
            "build_physionet_release",
            "assignments",
        ]
        yield SimpleNamespace(**{n.lstrip("0123456789_"): importlib.import_module(n) for n in names})


def span(m, names, text, sentence, label):
    start = text.index(sentence)
    return m.labelstudio.span_result(names, start, start + len(sentence), sentence, label)


def annotation(user, results):
    return {"completed_by": user, "result": results}


def exported_task(rid, annotations):
    return {
        "data": {
            "generated_impression_id": rid,
            "findings_text": FINDINGS_TEXT,
            "impression_text": IMPRESSION_TEXT,
        },
        "annotations": annotations,
    }


def processed(rid, path="none", risk=2):
    return {
        "generated_impression_id": rid,
        "risk_level": risk,
        "radiology_report": {"id": f"note-{rid}"},
        "config": {"generation_model": {"type": "none" if path == "none" else "api", "path": path}},
    }


def anchor_results(m, label_findings="To Review", label_impression="To Review"):
    ls = m.labelstudio
    return [
        span(m, ls.FINDINGS, FINDINGS_TEXT, "Lungs are clear.", label_findings),
        span(m, ls.FINDINGS, FINDINGS_TEXT, "No  effusion.", label_findings),
        span(m, ls.IMPRESSION, IMPRESSION_TEXT, "No acute process.", label_impression),
        span(m, ls.IMPRESSION, IMPRESSION_TEXT, "Normal heart.", label_impression),
    ]


def test_find_span_allows_any_whitespace(m):
    assert m.labelstudio.find_span(FINDINGS_TEXT, "No effusion.") == (27, 40, "No  effusion.")
    assert m.labelstudio.find_span("a\nb c", "a b") == (0, 3, "a\nb")
    assert m.labelstudio.find_span(FINDINGS_TEXT, "Pneumothorax.") is None


def test_annotations_to_predictions_falls_back_to_predictions(m):
    tasks = [
        {
            "data": {"x": 1},
            "annotations": [{"result": ["a"]}, {"result": ["b"]}],
            "predictions": [{"result": ["p"]}],
        },
        {"data": {"x": 2}, "annotations": [], "predictions": [{"result": ["p"]}]},
        {"data": {"x": 3}},
    ]
    out = m.labelstudio.annotations_to_predictions(tasks, "v2")
    assert [t["predictions"][0]["result"] for t in out] == [["a"], ["p"], []]
    assert all(t["predictions"][0]["model_version"] == "v2" for t in out)


def test_normalize_newlines_moves_offsets(m):
    task = {"annotations": [{"result": [{"value": {"text": "\\nLungs.\\n\\n", "start": 3, "end": 16}}]}]}
    value = m.labelstudio.normalize_newlines([task])[0]["annotations"][0]["result"][0]["value"]
    assert value == {"text": "Lungs.", "start": 4, "end": 14}


def test_read_user_map_requires_all_roles(m, tmp_path):
    path = tmp_path / "map.yaml"
    path.write_text("anchor: 1\nannotator_1: 2\n")
    with pytest.raises(ValueError, match="annotator_2"):
        m.labelstudio.read_user_map(path)
    example = SCRIPTS.parent / "user_map.example.yaml"
    assert set(m.labelstudio.read_user_map(example)) == set(USER_MAP)


def test_prehighlighted_findings_skip_long_summaries(m):
    many = [f"f{i}" for i in range(7)]
    report = Report(
        "id",
        "",
        "",
        "",
        [],
        [
            ImpressionSentence("a", "1.", attributed_findings_spans=["f2", "f1"]),
            ImpressionSentence("b", "2.", attributed_findings_spans=many),
            ImpressionSentence("c", "3.", attributed_findings_spans=["f1", "f3"]),
        ],
    )
    assert m.prepare_tasks.prehighlighted_findings(report) == ["f2", "f1", "f3"]


def test_assign_ids_is_seeded_shuffle(m):
    items = m.prepare_tasks.assign_ids([{"k": i} for i in range(5)])
    assert [i["generated_impression_id"] for i in items] == list(range(5))
    again = m.prepare_tasks.assign_ids([{"k": i} for i in range(5)])
    assert [i["k"] for i in items] == [i["k"] for i in again] != list(range(5))


def test_split_partitions(m):
    assert m.split_partitions.split(list(range(6)), (1, 2, 3)) == [[0], [1, 2], [3, 4, 5]]
    with pytest.raises(ValueError):
        m.split_partitions.split(list(range(6)), (1, 2))


def build(m, exports):
    ds = m.annotated_dataset
    anchors = [exported_task(7, [annotation(USER_MAP["anchor"], anchor_results(m))])]
    reports = ds.build_reports(anchors, [processed(7), processed(8)], USER_MAP["anchor"])
    ds.add_resident_labels(exports, reports, USER_MAP)
    return reports


def test_build_dataset_maps_users_to_annotator_ids(m):
    ls = m.labelstudio
    # annotator_2 labelled directly (no anchor copy); annotator_1's labels come from the anchor user's copy.
    own = [span(m, ls.IMPRESSION, IMPRESSION_TEXT, "Normal heart.", "HALLUCINATION")]
    copy = [span(m, ls.FINDINGS, FINDINGS_TEXT, "No  effusion.", "OMISSION")]
    exports = {
        "annotator_2": [exported_task(7, [annotation(12, own)])],
        "annotator_1": [exported_task(7, [annotation(11, own), annotation(10, copy)])],
    }
    (report,) = build(m, exports)
    assert report["radhalo_id"] == 7 and report["generation_model"] == "original_radiologist"
    assert report["mimic_notes_id"] == "note-7" and report["medval_risk_level"] == 2
    assert [s["is_omission"] for s in report["findings_sentences"]] == [
        {"annotator_1": False, "annotator_2": False},
        {"annotator_1": True, "annotator_2": False},
    ]
    assert [s["is_hallucination"] for s in report["impression_sentences"]] == [
        {"annotator_1": False, "annotator_2": False},
        {"annotator_1": False, "annotator_2": True},
    ]
    assert m.annotated_dataset.validation_problems(exports, [report], USER_MAP) == []


def test_validation_reports_offsets_only(m):
    ls = m.labelstudio
    bad = span(m, ls.FINDINGS, FINDINGS_TEXT, "Lungs are clear", "OMISSION")  # boundary differs from anchor
    review = span(m, ls.IMPRESSION, IMPRESSION_TEXT, "Normal heart.", "To Review")
    exports = {"annotator_3": [exported_task(7, [annotation(13, [bad, review])])]}
    (report,) = build(m, {})
    problems = m.annotated_dataset.validation_problems(exports, [report], USER_MAP)
    assert problems == [
        ("annotator_3", 7, "label_findings", 10, 25, "segmentation"),
        ("annotator_3", 7, "label_impression", 24, 37, "To Review"),
    ]


def labelled_report(hall, omis):
    return {
        "radhalo_id": 3,
        "generation_model": "gemini-2.5-flash",
        "medval_risk_level": 4,
        "report_text": FINDINGS_TEXT,
        "impression_text": IMPRESSION_TEXT,
        "findings_sentences": [{"is_omission": omis, "start": 10, "end": 26, "text": "Lungs are clear."}],
        "impression_sentences": [
            {"is_hallucination": hall, "start": 3, "end": 20, "text": "No acute process."}
        ],
    }


def test_adjudication_tasks_show_anonymised_labels(m):
    report = labelled_report(
        {"annotator_3": False, "annotator_1": True}, {"annotator_1": True, "annotator_3": True}
    )
    ds = m.annotated_dataset
    (task,) = ds.adjudication_tasks([report], "hallucination")
    assert task["data"]["annotator_labels"] == "Annotator 1: Hallucination | Annotator 3: Not Hallucination"
    assert task["data"]["sentence_index"] == 0
    result = task["predictions"][0]["result"][0]
    assert result["value"]["labels"] == ["To Review"] and result["from_name"] == "label_impression"
    assert ds.adjudication_tasks([report], "omission") == []


def test_adjudications_and_final_label(m):
    ds = m.annotated_dataset
    report = labelled_report(
        {"annotator_1": True, "annotator_3": False}, {"annotator_1": False, "annotator_3": False}
    )
    decided = {
        "data": {"radhalo_id": 3, "sentence_index": 0},
        "annotations": [
            {
                "result": [
                    {"from_name": "label_impression", "value": {"labels": ["To Review"]}},
                    {"from_name": "label_impression", "value": {"labels": ["Not a hallucination"]}},
                ]
            }
        ],
    }
    undecided = {"data": {"radhalo_id": 3, "sentence_index": 0}, "annotations": []}
    assert ds.add_adjudications([report], [undecided, decided], "hallucination") == (1, [(3, 0)])
    assert report["impression_sentences"][0]["is_hallucination"]["adjudicator"] is False
    ds.finalize([report])
    assert report["impression_sentences"][0]["adjudicated_label"] is False
    assert report["findings_sentences"][0]["adjudicated_label"] is False
    assert ds.adjudicated_label({"annotator_1": True, "annotator_2": True}) is True
    with pytest.raises(AssertionError):
        ds.adjudicated_label({"annotator_1": True, "annotator_2": False})


def test_release_renames_and_flattening(m, tmp_path):
    ds = m.annotated_dataset
    report = labelled_report(
        {"annotator_1": True, "annotator_3": False, "adjudicator": True}, {"annotator_2": False}
    )
    report = {"mimic_notes_id": "note-3", **report, "imaging_technique": "CT", "anatomy_location": "Chest"}
    ds.finalize([report])
    (released,) = ds.release_reports([report])
    assert list(released) == [
        "mimic_notes_id",
        "radhalo_id",
        "impression_source",
        "medval_risk_level",
        "background_and_findings_text",
        "impression_text",
        "findings_sentences",
        "impression_sentences",
        "imaging_technique",
        "anatomy_location",
    ]
    assert released["impression_source"] == "gemini-2.5-flash"
    assert ds.sentence_rows([released], "hallucination") == [
        [3, 0, 3, 20, "No acute process.", "true", "", "false", "", "true", "true"]
    ]
    m.build_physionet_release.build_release([report], tmp_path)
    rows = list(csv.reader(open(tmp_path / "data" / "findings_sentences.csv", newline="")))
    assert rows[0] == ds.SENTENCE_COLUMNS and rows[1][5:] == ["", "false", "", "", "", "false"]
    assert json.loads((tmp_path / "data" / "rad_halo_dataset.json").read_text()) == [released]
    assert (tmp_path / "data" / "reports.csv").read_bytes().startswith(b"radhalo_id,mimic_notes_id,")
    assert json.loads((tmp_path / "metadata" / "schema.json").read_text())["type"] == "array"
    assert (tmp_path / "code" / "example_usage.py").exists()


def test_merge_imaging_marks_missing_unknown(m):
    reports = [{"radhalo_id": 0}, {"radhalo_id": 1}]
    missing = m.merge_imaging.merge(
        reports, [{"radhalo_id": 1.0, "imaging_technique": "MRI", "anatomy_location": "Knee"}]
    )
    assert missing == [0]
    assert reports == [
        {"radhalo_id": 0, "imaging_technique": "Unknown", "anatomy_location": "Unknown"},
        {"radhalo_id": 1, "imaging_technique": "MRI", "anatomy_location": "Knee"},
    ]


def test_thematic_extract_union_rule_and_span(m):
    report = labelled_report(
        {"annotator_1": True, "annotator_2": False}, {"annotator_1": False, "annotator_2": False}
    )
    (task,) = m.thematic_extract.error_tasks([report], "hallucination")
    assert task["data"]["annotators_agree"] is False
    assert task["predictions"][0]["result"][0]["value"]["start"] == 3
    assert m.thematic_extract.error_tasks([report], "omission") == []


def test_thematic_clusters(m):
    def task(theme):
        result = {"from_name": "theme", "type": "textarea", "value": {"text": [theme]}}
        return {"annotations": [{"result": [result]}]}

    tasks = [task("comparison [external]"), task("prior [external]"), task("advice [judgment]"), task("")]
    clusters = m.thematic_export.coded_clusters(tasks)
    assert clusters == ["external", "external", "judgment"]
    assert m.thematic_export.cluster("no cluster") == ""
    assert m.thematic_export.cluster_rows({"hallucination": clusters}) == [
        {"error_type": "hallucination", "cluster": "external", "count": 2, "percentage": 66.7},
        {"error_type": "hallucination", "cluster": "judgment", "count": 1, "percentage": 33.3},
    ]


def test_assignment_partitions(m):
    a = m.assignments
    assert a.partition(10, "annotator_1", ["annotator_1", "annotator_2"]) == "1"
    assert a.partition(60, "annotator_1", ["annotator_1", "annotator_3"]) == "2"
    assert a.partition(200, "annotator_2", ["annotator_2", "annotator_4"]) == "3"
    assert a.partition(60, "annotator_4", ["annotator_1", "annotator_3", "annotator_4"]) == "extra"
    assert a.partition(60, "annotator_4", ["annotator_1", "annotator_4"]) == "5"
