"""MIMIC-IV-Note radiology reports: eligibility and section extraction (Section 3.2, Appendix A.1)."""

import re
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from radhalo.data.report import Report
from radhalo.data.segmentation import split_findings, split_impression

# Case-sensitive; the first match is used. Findings end at "IMPRESSION:", the impression at the next
# line starting with six capital letters (the next section header) or at the end of the report.
FINDINGS = re.compile(r"FINDINGS:\s*(.*?)(?=\Z|IMPRESSION:)", re.DOTALL)
IMPRESSION = re.compile(r"IMPRESSION:\s*(.*?)(?=\Z|\n[A-Z]{6})", re.DOTALL)
SINGLE_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")
MAX_FINDINGS_SENTENCE_CHARS = 400


def _extract(pattern: re.Pattern, text: str) -> str | None:
    match = pattern.search(text)
    if match is None:
        return None
    section = SINGLE_NEWLINE.sub(" ", match.group(1).strip())
    return section or None


def extract_findings(text: str) -> str | None:
    """Findings section with single newlines replaced by spaces; None if absent or empty."""
    return _extract(FINDINGS, text)


def extract_impression(text: str) -> str | None:
    """Impression section with single newlines replaced by spaces; None if absent or empty."""
    return _extract(IMPRESSION, text)


def is_eligible(text: str) -> bool:
    """Not a procedure note (no "procedure" anywhere, any case) and has both sections."""
    if not isinstance(text, str) or "procedure" in text.lower():
        return False
    return extract_findings(text) is not None and extract_impression(text) is not None


def eligible_chunks(radiology_csv: Path, chunk_size: int = 100_000) -> Iterator[pd.DataFrame]:
    """Yield the eligible rows of MIMIC-IV-Note `radiology.csv`, chunk by chunk, in file order."""
    with pd.read_csv(radiology_csv, chunksize=chunk_size) as reader:
        for chunk in reader:
            yield chunk[chunk["text"].map(is_eligible).astype(bool)]


def process_report(row: dict) -> Report:
    """Extract and segment the sections of one eligible MIMIC row (keys `note_id`, `text`)."""
    text = row["text"]
    findings, impression = extract_findings(text), extract_impression(text)
    return Report(
        id=row["note_id"],
        full_text=text,
        findings_text=findings,
        impression_text=impression,
        findings_sentences=split_findings(findings),
        impression_sentences=split_impression(impression),
    )


def has_short_findings_sentences(report: Report) -> bool:
    """True if no findings item (headers included) exceeds 400 characters; used for the synthetic pool."""
    return max(len(s) for s in report.findings_sentences) <= MAX_FINDINGS_SENTENCE_CHARS
