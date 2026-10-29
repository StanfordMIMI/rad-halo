"""Detector metrics against adjudicated labels with report-level bootstrap CIs (Sec. 5.3, Fig. 3, App. E.4),
and per-source flag rates of the LLM judges (Limitations)."""

import numpy as np

from radhalo.data.release import load_predictions, load_reports, sentences
from radhalo.detectors import DETECTORS
from radhalo.evaluation.bootstrap import percentile_ci, resample_report_ids
from radhalo.evaluation.metrics import detection_metrics
from radhalo.reporting.io import REPO, parse_args, write_csv

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 0
ERROR_TYPES = {"hallucination": "hall", "omission": "omis"}


def collect(report_ids, by_id: dict, predictions: dict, error_type: str):
    """Adjudicated labels and scores of all sentences of the given reports (repeats allowed)."""
    y_true, y_score = [], []
    for rid in report_ids:
        y_true += [int(s["label"]) for s in sentences(by_id[int(rid)], error_type)]
        y_score += predictions[int(rid)][error_type]
    return np.array(y_true), np.array(y_score)


def bootstrap_f1_ci(replicates, by_id, predictions, error_type, is_classifier) -> tuple:
    values = []
    for ids in replicates:
        y_true, y_score = collect(ids, by_id, predictions, error_type)
        if len(set(y_true)) < 2:
            values.append(np.nan)
        else:
            values.append(detection_metrics(y_true, y_score, is_classifier)["f1"])
    lower, upper = percentile_ci(np.array(values, dtype=float))
    return round(lower, 4), round(upper, 4)


def judge_flags(reports: list[dict], by_id: dict, predictions: dict, judge: str) -> list[dict]:
    rows = []
    for source in sorted({r["impression_source"] for r in reports}):
        ids = [r["radhalo_id"] for r in reports if r["impression_source"] == source]
        for error_type in ERROR_TYPES:
            y_true, y_score = collect(ids, by_id, predictions, error_type)
            flagged = (y_score >= 0.5).astype(int)
            rows.append(
                {
                    "judge": judge,
                    "impression_source": source,
                    "error_type": error_type,
                    "positives": int(y_true.sum()),
                    "recovered": int(((y_true == 1) & (flagged == 1)).sum()),
                    "flag_rate": round(float(flagged.mean()), 4),
                }
            )
    return rows


def main() -> None:
    args = parse_args(__doc__)
    reports = load_reports(args.data)
    by_id = {r["radhalo_id"]: r for r in reports}
    report_ids = sorted(by_id)
    replicates = resample_report_ids(report_ids, N_BOOTSTRAP, BOOTSTRAP_SEED)

    metric_rows, flag_rows = [], []
    for detector in DETECTORS:
        predictions = load_predictions(REPO / "predictions" / f"{detector.key}.jsonl", reports)
        row = {"detector": detector.key}
        for error_type, prefix in ERROR_TYPES.items():
            y_true, y_score = collect(report_ids, by_id, predictions, error_type)
            metrics = detection_metrics(y_true, y_score, detector.is_classifier)
            for name in ("precision", "recall", "f1", "auroc", "auprc", "threshold"):
                row[f"{prefix}_{name}"] = metrics[name]
            lower, upper = bootstrap_f1_ci(replicates, by_id, predictions, error_type, detector.is_classifier)
            row[f"{prefix}_f1_ci_lower"] = lower
            row[f"{prefix}_f1_ci_upper"] = upper
        metric_rows.append(row)
        if not detector.is_classifier:
            flag_rows += judge_flags(reports, by_id, predictions, detector.key)

    write_csv(metric_rows, args.results / "csv" / "detector_metrics.csv")
    write_csv(flag_rows, args.results / "csv" / "judge_flags_by_source.csv")


if __name__ == "__main__":
    main()
