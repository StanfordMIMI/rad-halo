"""Inter-annotator agreement among residents, overall and by source and risk level (Sec. 5.2, App. E.1)."""

from collections import defaultdict

from radhalo.data.release import load_reports
from radhalo.evaluation.agreement import agreement, annotator_ids
from radhalo.reporting.io import parse_args, write_csv


def by_group(reports: list[dict], key: str, column: str, ids: list[str]) -> list[dict]:
    groups = defaultdict(list)
    for report in reports:
        groups[report[key]].append(report)
    rows = []
    for value, group in sorted(groups.items()):
        row = {column: value, "num_reports": len(group)}
        for prefix, error_type in (("hall", "hallucination"), ("omis", "omission")):
            result = agreement(group, error_type, ids)
            row[f"{prefix}_alpha"] = result["alpha"]
            row[f"{prefix}_f1"] = result["pairwise_f1"]
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args(__doc__)
    reports = load_reports(args.data)
    ids = annotator_ids(reports)
    out = args.results / "csv"

    overall = [
        {"error_type": error_type, **agreement(reports, error_type, ids)}
        for error_type in ("hallucination", "omission")
    ]
    write_csv(overall, out / "agreement_overall.csv")
    write_csv(
        by_group(reports, "impression_source", "impression_source", ids), out / "agreement_by_source.csv"
    )
    write_csv(by_group(reports, "medval_risk_level", "risk_level", ids), out / "agreement_by_risk_level.csv")


if __name__ == "__main__":
    main()
