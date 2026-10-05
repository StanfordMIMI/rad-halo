#!/usr/bin/env python3
"""Minimal example usage for Rad-Halo.

This script demonstrates how to load the canonical JSON file and the flattened
CSV views, count final labels, and join sentence rows back to report metadata.
It uses only the Python standard library.
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of dictionaries."""
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_bool(value: str) -> bool | None:
    """Convert release CSV boolean strings to Python booleans."""
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def main() -> None:
    # Load canonical nested JSON.
    with (DATA / "rad_halo_dataset.json").open(encoding="utf-8") as f:
        reports_json = json.load(f)

    # Load flattened CSV views.
    reports = read_csv(DATA / "reports.csv")
    findings = read_csv(DATA / "findings_sentences.csv")
    impressions = read_csv(DATA / "impression_sentences.csv")

    print(f"Reports in JSON: {len(reports_json)}")
    print(f"Reports in reports.csv: {len(reports)}")
    print(f"Findings sentences: {len(findings)}")
    print(f"Impression sentences: {len(impressions)}")

    n_omissions = sum(as_bool(row["adjudicated_label"]) is True for row in findings)
    n_hallucinations = sum(as_bool(row["adjudicated_label"]) is True for row in impressions)

    print(f"Adjudicated omissions: {n_omissions}")
    print(f"Adjudicated hallucinations: {n_hallucinations}")

    # Join sentence rows to report metadata/context with radhalo_id.
    reports_by_id = {row["radhalo_id"]: row for row in reports}

    # Example hallucination benchmark item.
    hall_example = next(row for row in impressions if as_bool(row["adjudicated_label"]) is True)
    hall_report = reports_by_id[hall_example["radhalo_id"]]

    print("\nExample hallucination item")
    print(f"radhalo_id: {hall_example['radhalo_id']}")
    print(f"impression_source: {hall_report['impression_source']}")
    print(f"candidate_sentence: {hall_example['text']}")
    print(f"label: {hall_example['adjudicated_label']}")

    # Example omission benchmark item.
    omis_example = next(row for row in findings if as_bool(row["adjudicated_label"]) is True)
    omis_report = reports_by_id[omis_example["radhalo_id"]]

    print("\nExample omission item")
    print(f"radhalo_id: {omis_example['radhalo_id']}")
    print(f"impression_source: {omis_report['impression_source']}")
    print(f"candidate_sentence: {omis_example['text']}")
    print(f"label: {omis_example['adjudicated_label']}")


if __name__ == "__main__":
    main()
