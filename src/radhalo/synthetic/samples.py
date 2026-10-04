"""Training samples for RadCheck from perturbed reports (App. D.2).

Per report and base seed s, four samples are attempted, each from its own perturbation:
omission positive (seed s), omission negative (s + 1), hallucination positive (s + 100) and
hallucination negative (s + 101). The same seed drives role assignment, blanket dropping and the
choice of the candidate sentence.
"""

import numpy as np

from radhalo.data.report import Report
from radhalo.data.segmentation import clean_whitespace, impression_text, is_header
from radhalo.synthetic.perturbation import Perturbation, perturb

OMISSION_TASK = "DETECT_OMISSION"
HALLUCINATION_TASK = "DETECT_HALLUCINATION"


def background(report: Report) -> str:
    """Report text before the first "FINDINGS"."""
    return clean_whitespace(report.full_text.split("FINDINGS")[0].strip())


def render(template: str, task: str, report: Report, perturbation: Perturbation, candidate: str) -> str:
    return template.format(
        task=task,
        context=background(report),
        findings_text=" ".join(perturbation.findings_sentences),
        impression_text=impression_text(perturbation.impression_sentences),
        candidate_sentence=candidate,
    )


def omission_positive(report: Report, seed: int, template: str) -> str | None:
    """At least one omitted impression sentence; the candidate is one of the omitted findings."""
    p = perturb(report, min_hallucinated=0, min_omitted=1, min_preserved=0, seed=seed)
    if p is None or not p.omitted_findings:
        return None
    candidate = np.random.default_rng(seed).choice(p.omitted_findings)
    return render(template, OMISSION_TASK, report, p, candidate)


def omission_negative(report: Report, seed: int, template: str) -> str | None:
    """Random roles; the candidate is a remaining findings sentence that is neither omitted nor a header."""
    p = perturb(report, min_hallucinated=0, min_omitted=0, min_preserved=0, seed=seed)
    if p is None:
        return None
    present = [f for f in p.findings_sentences if f not in p.omitted_findings and not is_header(f)]
    if not present:
        return None
    candidate = np.random.default_rng(seed).choice(present)
    return render(template, OMISSION_TASK, report, p, candidate)


def hallucination_positive(report: Report, seed: int, template: str) -> str | None:
    """At least one hallucinated impression sentence; the candidate is one of the hallucinations."""
    p = perturb(report, min_hallucinated=1, min_omitted=0, min_preserved=0, seed=seed)
    if p is None or not p.hallucinated:
        return None
    candidate = np.random.default_rng(seed).choice(p.hallucinated)
    return render(template, HALLUCINATION_TASK, report, p, candidate.text)


def hallucination_negative(report: Report, seed: int, template: str) -> str | None:
    """At least one preserved sentence; the candidate is a remaining non-hallucinated impression sentence."""
    p = perturb(report, min_hallucinated=0, min_omitted=0, min_preserved=1, seed=seed)
    if p is None:
        return None
    supported = [s for s in p.impression_sentences if s not in p.hallucinated]
    if not supported:
        return None
    candidate = np.random.default_rng(seed).choice(supported)
    return render(template, HALLUCINATION_TASK, report, p, candidate.text)


BUILDERS = [  # (builder, seed offset, label, error type), in output order
    (omission_positive, 0, 1, "omission"),
    (omission_negative, 1, 0, "omission"),
    (hallucination_positive, 100, 1, "hallucination"),
    (hallucination_negative, 101, 0, "hallucination"),
]


def report_samples(report: Report, seed: int, template: str) -> list[dict]:
    """Up to four samples (`report_id, features, label, error_type`) of one report."""
    samples = []
    for build, offset, label, error_type in BUILDERS:
        features = build(report, seed + offset, template)
        if features is not None:
            samples.append(
                {"report_id": report.id, "features": features, "label": label, "error_type": error_type}
            )
    return samples
