"""Deletion perturbations of an attributed report (App. D.2).

Impression sentences are compared by value (dataclass equality), and findings sentences by string
equality, so identical sentences are treated as one.
"""

from dataclasses import dataclass

import numpy as np

from radhalo.data.report import ImpressionSentence, Report

MAX_SPECIFIC_SPANS = 3  # more attributed findings than this make a blanket summary
BLANKET_DROP_PROBABILITY = 0.33
SECONDARY_MIN_SUPPORT = 0.5  # a preserved sentence keeping less than this share becomes a hallucination


@dataclass
class Perturbation:
    impression_sentences: list[ImpressionSentence]  # final impression, original order
    findings_sentences: list[str]  # final findings, headers included
    omitted_findings: list[str]
    hallucinated: list[ImpressionSentence]


def drop_unsupported_notes(sentences: list[ImpressionSentence]) -> list[ImpressionSentence]:
    """Remove recommendations and administrative notes without attributed findings."""
    return [
        s
        for s in sentences
        if not ((s.is_recommendation or s.is_admin_note) and len(s.attributed_findings_spans) == 0)
    ]


def assign_roles(
    sentences: list[ImpressionSentence],
    min_hallucinated: int,
    min_omitted: int,
    min_preserved: int,
    seed: int,
) -> tuple[list, list, list] | None:
    """Shuffle the sentences and split them into (hallucinated, omitted, preserved).

    Each role gets its minimum count; every further sentence gets a uniformly random role.
    None if there are fewer sentences than the minimums require.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(sentences))
    rng.shuffle(indices)
    n_extra = len(sentences) - (min_hallucinated + min_omitted + min_preserved)
    if n_extra < 0:
        return None
    n_hallucinated, n_omitted = min_hallucinated, min_omitted
    if n_extra > 0:
        roles = rng.integers(0, 3, size=n_extra)
        n_hallucinated += np.sum(roles == 0)
        n_omitted += np.sum(roles == 1)
    order = [sentences[i] for i in indices]
    return (
        order[:n_hallucinated],
        order[n_hallucinated : n_hallucinated + n_omitted],
        order[n_hallucinated + n_omitted :],
    )


def keep_blanket_summaries(sentences: list[ImpressionSentence], seed: int) -> list[ImpressionSentence]:
    """Keep each blanket summary with probability 1 - 0.33."""
    if not sentences:
        return []
    keep = np.random.default_rng(seed).random(len(sentences)) > BLANKET_DROP_PROBABILITY
    return [s for s, k in zip(sentences, keep) if k]


def secondary_hallucinations(
    preserved: list[ImpressionSentence], findings: list[str]
) -> tuple[list[ImpressionSentence], list[ImpressionSentence]]:
    """Split preserved sentences into (still preserved, newly hallucinated) after findings were removed.

    One pass; a sentence is reclassified if strictly less than half of its findings remain.
    Sentences without attributed findings stay preserved.
    """
    kept, hallucinated = [], []
    for sentence in preserved:
        spans = sentence.attributed_findings_spans
        if spans and sum(f in findings for f in spans) / len(spans) < SECONDARY_MIN_SUPPORT:
            hallucinated.append(sentence)
        else:
            kept.append(sentence)
    return kept, hallucinated


def is_omitted(finding: str, omitted: list[ImpressionSentence], preserved: list[ImpressionSentence]) -> bool:
    """Supports an omitted impression sentence and no preserved one."""
    return any(finding in s.attributed_findings_spans for s in omitted) and not any(
        finding in s.attributed_findings_spans for s in preserved
    )


def perturb(
    report: Report, min_hallucinated: int, min_omitted: int, min_preserved: int, seed: int
) -> Perturbation | None:
    """Assign roles to the specific impression sentences and apply the deletions.

    None if the report has too few specific sentences for the role minimums.
    """
    relevant = drop_unsupported_notes(report.impression_sentences)
    specific = [s for s in relevant if len(s.attributed_findings_spans) <= MAX_SPECIFIC_SPANS]
    blanket = [s for s in relevant if len(s.attributed_findings_spans) > MAX_SPECIFIC_SPANS]
    roles = assign_roles(specific, min_hallucinated, min_omitted, min_preserved, seed)
    if roles is None:
        return None
    hallucinated, omitted, preserved = roles
    preserved = preserved + keep_blanket_summaries(blanket, seed)

    removed = [f for s in hallucinated for f in s.attributed_findings_spans]
    findings = [f for f in report.findings_sentences if f not in removed]
    preserved, secondary = secondary_hallucinations(preserved, findings)
    hallucinated = hallucinated + secondary

    kept = hallucinated + preserved
    return Perturbation(
        impression_sentences=[s for s in report.impression_sentences if s in kept],
        findings_sentences=findings,
        omitted_findings=[f for f in findings if is_omitted(f, omitted, preserved)],
        hallucinated=hallucinated,
    )
