"""Dataset statistics with adjudicated labels (Section 5.1, Table 1-2, Appendix B, E.2)."""

import csv
from collections import defaultdict

from radhalo.data.release import ADJUDICATOR, annotations, load_reports, resident_labels, sentences
from radhalo.evaluation.trend import cochran_armitage
from radhalo.reporting.io import REPO, parse_args, write_csv, write_metric_values

RISK_LEVELS = [1, 2, 3, 4]


def count_errors(report: dict, error_type: str) -> int:
    return sum(s["adjudicated_label"] for s in sentences(report, error_type))


def pct(numerator: int, denominator: int) -> float:
    return round(numerator / denominator * 100, 2) if denominator else 0


def error_rates(reports: list[dict]) -> dict:
    n_impression = sum(len(r["impression_sentences"]) for r in reports)
    n_findings = sum(len(r["findings_sentences"]) for r in reports)
    hall = [count_errors(r, "hallucination") for r in reports]
    omis = [count_errors(r, "omission") for r in reports]
    return {
        "num_reports": len(reports),
        "impression_sentences": n_impression,
        "findings_sentences": n_findings,
        "hallucinations": sum(hall),
        "omissions": sum(omis),
        "hallucination_rate": pct(sum(hall), n_impression),
        "omission_rate": pct(sum(omis), n_findings),
        "pct_reports_with_hallucination": pct(sum(h > 0 for h in hall), len(reports)),
        "pct_reports_with_omission": pct(sum(o > 0 for o in omis), len(reports)),
    }


def rates_by(reports: list[dict], key: str, column: str) -> list[dict]:
    groups = defaultdict(list)
    for report in reports:
        groups[report[key]].append(report)
    return [{column: value, **error_rates(group)} for value, group in sorted(groups.items())]


def annotator_error_rates(reports: list[dict]) -> list[dict]:
    counts = defaultdict(lambda: defaultdict(int))
    for report in reports:
        for error_type in ("hallucination", "omission"):
            for s in sentences(report, error_type):
                for annotator, label in resident_labels(s, error_type).items():
                    counts[annotator][f"{error_type}_sentences"] += 1
                    counts[annotator][f"{error_type}_errors"] += label
    return [
        {
            "annotator": annotator,
            "impression_sentences": c["hallucination_sentences"],
            "hallucinations": c["hallucination_errors"],
            "hallucination_rate": pct(c["hallucination_errors"], c["hallucination_sentences"]),
            "findings_sentences": c["omission_sentences"],
            "omissions": c["omission_errors"],
            "omission_rate": pct(c["omission_errors"], c["omission_sentences"]),
        }
        for annotator, c in sorted(counts.items())
    ]


def adjudication(reports: list[dict]) -> tuple[dict, list[dict]]:
    """Adjudication outcomes and how often the adjudicator sided with each resident."""
    summary = {}
    agree = defaultdict(lambda: defaultdict(int))
    for prefix, error_type in (("hall", "hallucination"), ("omis", "omission")):
        disputed = positive = 0
        for report in reports:
            for s in sentences(report, error_type):
                labels = annotations(s, error_type)
                if ADJUDICATOR not in labels:
                    continue
                disputed += 1
                positive += labels[ADJUDICATOR]
                for annotator, label in resident_labels(s, error_type).items():
                    agree[annotator][f"{prefix}_disputed"] += 1
                    agree[annotator][f"{prefix}_agrees"] += label == labels[ADJUDICATOR]
        summary[f"{prefix}_disputed_sentences"] = disputed
        summary[f"{prefix}_adjudicated_as_error"] = positive
        summary[f"{prefix}_adjudicated_as_not_error"] = disputed - positive
    for outcome in ("disputed_sentences", "adjudicated_as_error", "adjudicated_as_not_error"):
        summary[f"total_{outcome}"] = summary[f"hall_{outcome}"] + summary[f"omis_{outcome}"]

    def rate(agrees, total):
        return round(agrees / total * 100, 4) if total else 0

    rows = []
    for annotator, c in sorted(agree.items()):
        total = c["hall_disputed"] + c["omis_disputed"]
        rows.append(
            {
                "annotator": annotator,
                "hall_disputed": c["hall_disputed"],
                "hall_agreement_rate": rate(c["hall_agrees"], c["hall_disputed"]),
                "omis_disputed": c["omis_disputed"],
                "omis_agreement_rate": rate(c["omis_agrees"], c["omis_disputed"]),
                "overall_agreement_rate": rate(c["hall_agrees"] + c["omis_agrees"], total),
            }
        )
    return summary, rows


def modality_anatomy_counts(reports: list[dict]) -> list[dict]:
    counts = defaultdict(int)
    for report in reports:
        counts[f"{report['imaging_technique']} {report['anatomy_location']}"] += 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [{"modality_anatomy": k, "num_reports": n} for k, n in ranked]


def risk_trend(reports: list[dict]) -> list[dict]:
    """Reports with at least one adjudicated error per MedVAL risk level (report-level outcome)."""
    rows = []
    for error_type in ("hallucination", "omission"):
        totals = [sum(r["medval_risk_level"] == lv for r in reports) for lv in RISK_LEVELS]
        positives = [
            sum(r["medval_risk_level"] == lv and count_errors(r, error_type) > 0 for r in reports)
            for lv in RISK_LEVELS
        ]
        z, p = cochran_armitage(RISK_LEVELS, positives, totals)
        for lv, total, pos in zip(RISK_LEVELS, totals, positives):
            rows.append({"error_type": error_type, "metric": f"reports_level_{lv}", "value": total})
            rows.append({"error_type": error_type, "metric": f"reports_with_error_level_{lv}", "value": pos})
        rows.append({"error_type": error_type, "metric": "cochran_armitage_z", "value": round(z, 3)})
        rows.append({"error_type": error_type, "metric": "cochran_armitage_p", "value": f"{p:.3e}"})
    return rows


def prehighlight_coverage(reports: list[dict]) -> dict:
    """Share of findings sentences pre-highlighted for omission review (Appendix B.3)."""
    with open(REPO / "derived" / "prehighlighted_findings.csv") as f:
        highlighted = {(int(r["radhalo_id"]), int(r["sentence_id"])) for r in csv.DictReader(f)}
    total = inside = omissions = omissions_inside = 0
    for report in reports:
        for i, s in enumerate(report["findings_sentences"]):
            is_highlighted = (report["radhalo_id"], i) in highlighted
            total += 1
            inside += is_highlighted
            omissions += s["adjudicated_label"]
            omissions_inside += s["adjudicated_label"] and is_highlighted
    return {
        "findings_sentences": total,
        "highlighted_findings_sentences": inside,
        "pct_findings_sentences_highlighted": pct(inside, total),
        "omissions_inside_highlighted": omissions_inside,
        "omissions_outside_highlighted": omissions - omissions_inside,
    }


def main() -> None:
    args = parse_args(__doc__)
    reports = load_reports(args.data)
    out = args.results / "csv"

    write_metric_values(error_rates(reports), out / "error_rates_overall.csv")
    write_csv(rates_by(reports, "impression_source", "impression_source"), out / "error_rates_by_source.csv")
    write_csv(rates_by(reports, "medval_risk_level", "risk_level"), out / "error_rates_by_risk_level.csv")
    write_csv(annotator_error_rates(reports), out / "annotator_error_rates.csv")
    summary, agreement_rows = adjudication(reports)
    write_metric_values(summary, out / "adjudication_summary.csv")
    write_csv(agreement_rows, out / "adjudicator_agreement.csv")
    write_csv(modality_anatomy_counts(reports), out / "modality_anatomy_counts.csv")
    write_csv(risk_trend(reports), out / "risk_trend.csv")
    write_metric_values(prehighlight_coverage(reports), out / "prehighlight_coverage.csv")


if __name__ == "__main__":
    main()
