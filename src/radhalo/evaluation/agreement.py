"""Inter-annotator agreement among the residents (adjudicator excluded)."""

from itertools import combinations

import krippendorff
import numpy as np
from sklearn.metrics import f1_score

from radhalo.data.release import resident_labels, sentences


def annotator_ids(reports: list[dict]) -> list[str]:
    ids = set()
    for report in reports:
        for error_type in ("hallucination", "omission"):
            for s in sentences(report, error_type):
                ids.update(resident_labels(s, error_type))
    return sorted(ids)


def krippendorff_alpha(reports: list[dict], error_type: str) -> float:
    """Nominal alpha over all sentences; annotators who did not label a sentence are missing."""
    ids = annotator_ids(reports)
    row = {a: i for i, a in enumerate(ids)}
    items = [resident_labels(s, error_type) for r in reports for s in sentences(r, error_type)]
    matrix = np.full((len(ids), len(items)), np.nan)
    for j, labels in enumerate(items):
        for annotator, label in labels.items():
            matrix[row[annotator], j] = int(label)
    return krippendorff.alpha(reliability_data=matrix, level_of_measurement="nominal")


def pooled_pairwise_labels(
    reports: list[dict], error_type: str, ids: list[str]
) -> tuple[list[int], list[int]]:
    """Concatenate the labels of every annotator pair on the sentences both labeled."""
    first, second = [], []
    for a, b in combinations(ids, 2):
        for report in reports:
            for s in sentences(report, error_type):
                labels = resident_labels(s, error_type)
                if a in labels and b in labels:
                    first.append(int(labels[a]))
                    second.append(int(labels[b]))
    return first, second


def agreement(reports: list[dict], error_type: str, ids: list[str]) -> dict:
    """Krippendorff's alpha and pooled pairwise F1, rounded to 4 decimals.

    `ids` fixes the annotator pairs; alpha uses the annotators present in `reports`.
    """
    first, second = pooled_pairwise_labels(reports, error_type, ids)
    return {
        "alpha": round(krippendorff_alpha(reports, error_type), 4),
        "pairwise_f1": round(f1_score(first, second, zero_division=0), 4),
        "label_pairs": len(first),
    }
