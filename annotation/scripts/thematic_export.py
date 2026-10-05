"""Count thematic clusters of the coded errors (App. E.3, `derived/thematic_clusters.csv`).

Input: Label Studio exports of the thematic coding projects. Each coded task has a `theme` text of the
form "theme name [cluster name]"; only the cluster is exported. Free-text notes and initial codes stay
private. Percentages are relative to all coded tasks of the error type.
"""

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

from labelstudio import read_json

THEME = re.compile(r"^(.+?)\s*\[(.+?)\]\s*$")


def cluster(theme: str) -> str:
    """Cluster name of a "theme [cluster]" text; empty if there is no bracketed cluster."""
    match = THEME.match(theme)
    return match.group(2).strip() if match else ""


def coded_clusters(tasks: list[dict]) -> list[str]:
    """One cluster per non-empty `theme` text area of the exported tasks."""
    return [
        cluster(result["value"]["text"][0])
        for task in tasks
        for annotation in task.get("annotations", [])
        for result in annotation.get("result", [])
        if result.get("from_name") == "theme"
        and result.get("type") == "textarea"
        and result["value"]["text"][0]
    ]


def cluster_rows(clusters_by_type: dict[str, list[str]]) -> list[dict]:
    """Rows sorted by error type, then by decreasing count (ties in order of first appearance)."""
    rows = []
    for error_type, clusters in clusters_by_type.items():
        for name, count in Counter(clusters).items():
            rows.append(
                {
                    "error_type": error_type,
                    "cluster": name,
                    "count": count,
                    "percentage": round(count / len(clusters) * 100, 1),
                }
            )
    return sorted(rows, key=lambda r: (r["error_type"], -r["count"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hallucinations", type=Path, help="export of the hallucination coding project")
    parser.add_argument("--omissions", type=Path, help="export of the omission coding project")
    parser.add_argument("--out", type=Path, default=Path("derived/thematic_clusters.csv"))
    args = parser.parse_args()

    exports = {"hallucination": args.hallucinations, "omission": args.omissions}
    clusters = {t: coded_clusters(read_json(path)) for t, path in exports.items() if path is not None}
    rows = cluster_rows({t: c for t, c in clusters.items() if c})
    if not rows:
        sys.exit("no coded themes found")
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["error_type", "cluster", "count", "percentage"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} clusters -> {args.out}")


if __name__ == "__main__":
    main()
