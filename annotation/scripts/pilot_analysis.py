"""Error statistics of the pilot annotation (Sec. 3.4, App. A.5).

Inputs: the Label Studio export of the pilot project (labels normalised to "Hallucination" for impression
sentences and "True" for omitted findings sentences) and the processed data of its tasks
(`processed_reports.jsonl` format). Keeps the first 10 reports per MedVAL risk level by id (40 reports)
and writes `descriptive_stats.csv` and `stats_by_risk_level.csv`.

Sentence counts come from the stored source report: findings sentences, and the sentences of the
*reference* impression, which is also the denominator of the hallucination rate (as in the paper run).
"""

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from labelstudio import read_json

HALLUCINATION = "Hallucination"
OMISSION = "True"
KNOWN_LABELS = {HALLUCINATION, OMISSION, "To Review"}
N_PER_RISK_LEVEL = 10
RISK_COLUMNS = [
    "risk_level",
    "num_reports",
    "total_findings_sentences",
    "total_impression_sentences",
    "total_hallucinations",
    "total_omissions",
    "hallucination_rate",
    "omission_rate",
    "pct_reports_with_hallucination",
    "pct_reports_with_omission",
]


@dataclass
class PilotReport:
    radhalo_id: int
    risk_level: int
    findings_sentences: int
    impression_sentences: int
    hallucinations: int
    omissions: int


def count_errors(task: dict) -> tuple[int, int]:
    """Hallucination and omission spans in the first annotation of a task."""
    if not task.get("annotations"):
        return 0, 0
    hallucinations = omissions = 0
    for result in task["annotations"][0].get("result", []):
        labels = result.get("value", {}).get("labels")
        if not labels:
            continue
        if labels[0] not in KNOWN_LABELS:
            raise ValueError(f"unknown label in task {task['data'].get('generated_impression_id')}")
        hallucinations += labels[0] == HALLUCINATION and "impression" in result.get("from_name", "")
        omissions += labels[0] == OMISSION and "findings" in result.get("from_name", "")
    return hallucinations, omissions


def pilot_reports(tasks: list[dict], processed: list[dict]) -> list[PilotReport]:
    by_id = {item["generated_impression_id"]: item for item in processed}
    reports = []
    for task in tasks:
        item = by_id.get(task.get("data", {}).get("generated_impression_id"))
        if item is None:
            continue
        hallucinations, omissions = count_errors(task)
        source = item["radiology_report"]
        reports.append(
            PilotReport(
                radhalo_id=item["generated_impression_id"],
                risk_level=item.get("risk_level", 0),
                findings_sentences=len(source["findings_sentences"]),
                impression_sentences=len(source["impression_sentences"]),
                hallucinations=hallucinations,
                omissions=omissions,
            )
        )
    return reports


def first_n_per_risk_level(reports: list[PilotReport], n: int = N_PER_RISK_LEVEL) -> list[PilotReport]:
    by_level = defaultdict(list)
    for report in sorted(reports, key=lambda r: r.radhalo_id):
        by_level[report.risk_level].append(report)
    return [r for level in sorted(by_level) for r in by_level[level][:n]]


def _rate(count: int, total: int) -> float:
    return round(count / total * 100, 2) if total > 0 else 0


def descriptive_stats(reports: list[PilotReport]) -> dict:
    n = len(reports)
    findings = sum(r.findings_sentences for r in reports)
    impression = sum(r.impression_sentences for r in reports)
    hallucinations = sum(r.hallucinations for r in reports)
    omissions = sum(r.omissions for r in reports)
    return {
        "total_reports": n,
        "total_findings_sentences": findings,
        "total_impression_sentences": impression,
        "total_hallucinations": hallucinations,
        "total_omissions": omissions,
        "hallucination_rate": _rate(hallucinations, impression),
        "omission_rate": _rate(omissions, findings),
        "avg_findings_sentences_per_report": round(np.mean([r.findings_sentences for r in reports]), 2),
        "avg_impression_sentences_per_report": round(np.mean([r.impression_sentences for r in reports]), 2),
        "avg_hallucinations_per_report": round(np.mean([r.hallucinations for r in reports]), 2),
        "avg_omissions_per_report": round(np.mean([r.omissions for r in reports]), 2),
        "pct_reports_with_any_hallucination": round(sum(r.hallucinations > 0 for r in reports) / n * 100, 2),
        "pct_reports_with_any_omission": round(sum(r.omissions > 0 for r in reports) / n * 100, 2),
        "pct_reports_with_any_error": round(
            sum(r.hallucinations + r.omissions > 0 for r in reports) / n * 100, 2
        ),
    }


def stats_by_risk_level(reports: list[PilotReport]) -> list[dict]:
    by_level = defaultdict(list)
    for report in reports:
        by_level[report.risk_level].append(report)
    rows = []
    for level in sorted(by_level):
        group = by_level[level]
        findings = sum(r.findings_sentences for r in group)
        impression = sum(r.impression_sentences for r in group)
        hallucinations = sum(r.hallucinations for r in group)
        omissions = sum(r.omissions for r in group)
        rows.append(
            {
                "risk_level": level,
                "num_reports": len(group),
                "total_findings_sentences": findings,
                "total_impression_sentences": impression,
                "total_hallucinations": hallucinations,
                "total_omissions": omissions,
                "hallucination_rate": _rate(hallucinations, impression),
                "omission_rate": _rate(omissions, findings),
                "pct_reports_with_hallucination": round(
                    sum(r.hallucinations > 0 for r in group) / len(group) * 100, 2
                ),
                "pct_reports_with_omission": round(sum(r.omissions > 0 for r in group) / len(group) * 100, 2),
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True, help="Label Studio export of the pilot project")
    parser.add_argument("--processed", type=Path, required=True, help="processed data of the pilot tasks")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    with open(args.processed) as f:
        processed = [json.loads(line) for line in f if line.strip()]
    reports = first_n_per_risk_level(pilot_reports(read_json(args.export), processed))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = descriptive_stats(reports)
    write_csv(
        [{"metric": k, "value": v} for k, v in stats.items()],
        args.out_dir / "descriptive_stats.csv",
        ["metric", "value"],
    )
    write_csv(stats_by_risk_level(reports), args.out_dir / "stats_by_risk_level.csv", RISK_COLUMNS)
    print(f"{stats['total_reports']} reports -> {args.out_dir}")


if __name__ == "__main__":
    main()
