import numpy as np
import pytest

from radhalo.evaluation.agreement import agreement, annotator_ids
from radhalo.evaluation.bootstrap import percentile_ci, resample_report_ids
from radhalo.evaluation.metrics import detection_metrics, f1_optimal_threshold
from radhalo.evaluation.trend import cochran_armitage


def test_judge_metrics_use_fixed_threshold():
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([1.0, 0.0, 1.0, 0.0])
    m = detection_metrics(y_true, y_score, is_classifier=False)
    assert (m["precision"], m["recall"], m["f1"]) == (0.5, 0.5, 0.5)
    assert m["auroc"] is None and m["threshold"] is None


def test_classifier_threshold_maximises_f1():
    y_true = np.array([0, 0, 1, 1, 0])
    y_score = np.array([0.1, 0.2, 0.7, 0.8, 0.75])
    assert f1_optimal_threshold(y_true, y_score) == pytest.approx(0.7)
    m = detection_metrics(y_true, y_score, is_classifier=True)
    assert m["f1"] == 0.8 and m["auroc"] is not None


def test_single_class_falls_back_to_half():
    assert f1_optimal_threshold(np.zeros(3), np.array([0.2, 0.6, 0.9])) == 0.5


def test_bootstrap_is_seeded_and_report_level():
    a = resample_report_ids([10, 20, 30], n_bootstrap=5, seed=0)
    b = resample_report_ids([10, 20, 30], n_bootstrap=5, seed=0)
    assert all((x == y).all() for x, y in zip(a, b))
    assert all(set(x) <= {10, 20, 30} and len(x) == 3 for x in a)


def test_percentile_ci_ignores_nan():
    assert percentile_ci(np.array([np.nan, 1.0, 1.0])) == (1.0, 1.0)
    assert percentile_ci(np.array([np.nan])) == (None, None)


def test_cochran_armitage_detects_trend():
    z, p = cochran_armitage([1, 2, 3, 4], [1, 2, 5, 100], [30, 30, 30, 210])
    assert z > 0 and p < 0.05
    z_flat, _ = cochran_armitage([1, 2, 3, 4], [3, 3, 3, 21], [30, 30, 30, 210])
    assert z_flat == pytest.approx(0.0, abs=1e-9)


def _report(labels):
    return {
        "impression_sentences": [{"is_hallucination": lab} for lab in labels],
        "findings_sentences": [],
    }


def test_agreement_excludes_adjudicator_and_pools_pairs():
    reports = [
        _report([{"annotator_1": True, "annotator_2": True}, {"annotator_1": False, "annotator_2": False}]),
        _report([{"annotator_1": True, "annotator_2": False, "adjudicator": True}]),
    ]
    ids = annotator_ids(reports)
    assert ids == ["annotator_1", "annotator_2"]
    result = agreement(reports, "hallucination", ids)
    assert result["label_pairs"] == 3
    assert result["pairwise_f1"] == pytest.approx(round(2 / 3, 4))
