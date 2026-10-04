"""Segmented radiology reports and their JSONL format."""

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ImpressionSentence:
    """One impression sentence; `number_label` is "1.", "2.", ... or "pre" for text before item 1."""

    text: str
    number_label: str
    is_recommendation: bool | None = None
    is_admin_note: bool | None = None
    attributed_findings_spans: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sentence_text": self.text,
            "number_label": self.number_label,
            "is_recommendation": self.is_recommendation,
            "is_admin_note": self.is_admin_note,
            "attributed_findings_spans": self.attributed_findings_spans,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ImpressionSentence":
        return cls(
            text=d["sentence_text"],
            number_label=d.get("number_label", ""),
            is_recommendation=d.get("is_recommendation"),
            is_admin_note=d.get("is_admin_note"),
            attributed_findings_spans=d.get("attributed_findings_spans") or [],
        )


@dataclass
class Report:
    """A report with its findings and impression sections split into sentences.

    `findings_sentences` keeps section headers (e.g. "LUNGS:") as separate items.
    """

    id: str
    full_text: str
    findings_text: str
    impression_text: str
    findings_sentences: list[str]
    impression_sentences: list[ImpressionSentence]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "full_text": self.full_text,
            "findings_text": self.findings_text,
            "impression_text": self.impression_text,
            "findings_sentences": self.findings_sentences,
            "impression_sentences": [s.to_dict() for s in self.impression_sentences],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Report":
        """Read a report dict; extra keys of older files (dataset_id, modality, ...) are ignored."""
        return cls(
            id=d["id"],
            full_text=d["full_text"],
            findings_text=d["findings_text"],
            impression_text=d["impression_text"],
            findings_sentences=d["findings_sentences"],
            impression_sentences=[ImpressionSentence.from_dict(s) for s in d["impression_sentences"]],
        )


def read_reports(path: Path) -> Iterator[Report]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield Report.from_dict(json.loads(line))


def write_reports(reports: Iterable[Report], path: Path) -> None:
    with open(path, "w") as f:
        for report in reports:
            f.write(json.dumps(report.to_dict()) + "\n")
