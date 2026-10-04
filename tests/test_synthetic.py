import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from radhalo.data.report import ImpressionSentence, Report
from radhalo.synthetic.attribution import (
    apply_response,
    build_prompt,
    fill_examples,
    load_prompt,
    prune_attribution_map,
)
from radhalo.synthetic.perturbation import keep_blanket_summaries, perturb, secondary_hallucinations
from radhalo.synthetic.samples import background, report_samples
from radhalo.training.radcheck import compute_metrics

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = (REPO / "prompts" / "radcheck.txt").read_text()


def sentence(text, label="1.", spans=(), rec=None, admin=None):
    return ImpressionSentence(
        text=text,
        number_label=label,
        is_recommendation=rec,
        is_admin_note=admin,
        attributed_findings_spans=list(spans),
    )


def report(findings, impression):
    """Fictional report."""
    return Report(
        id="r1",
        full_text="EXAMINATION: Chest.\n\nINDICATION:  Cough.\n\nFINDINGS: ...\n\nIMPRESSION: ...",
        findings_text="",
        impression_text="",
        findings_sentences=findings,
        impression_sentences=impression,
    )


# Attribution --------------------------------------------------------------------------------------


def test_prune_removes_specifically_covered_findings_from_blanket_sentences():
    raw = {
        "0": {"type": "fact", "findings_sentences": [1]},
        "1": {"type": "fact", "findings_sentences": [0, 1, 2, 3]},
        "2": {"type": "admin"},
    }
    pruned = prune_attribution_map(raw)
    assert pruned["1"]["findings_sentences"] == [0, 2, 3]
    assert pruned["0"]["findings_sentences"] == [1] and "findings_sentences" not in pruned["2"]
    assert raw["1"]["findings_sentences"] == [0, 1, 2, 3]  # input unchanged


def test_prune_uses_raw_counts():
    # Four links, two out of range: still a blanket sentence for pruning.
    pruned = prune_attribution_map(
        {
            "0": {"type": "fact", "findings_sentences": [0]},
            "1": {"type": "fact", "findings_sentences": [0, 1, 8, 9]},
        }
    )
    assert pruned["1"]["findings_sentences"] == [1, 8, 9]


def test_apply_response_indexes_findings_without_headers():
    r = report(
        ["LUNGS:", "Lungs clear.", "HEART:", "Heart normal.", "No effusion."],
        [sentence("Normal chest."), sentence("Follow up.", "2."), sentence("Untouched.", "3.")],
    )
    response = (
        "```json\n"
        + json.dumps(
            {"0": {"type": "fact", "findings_sentences": [0, 1, 5, -1]}, "1": {"type": "recommendation"}}
        )
        + "\n```"
    )
    apply_response(r, response)
    first, second, third = r.impression_sentences
    assert first.attributed_findings_spans == ["Lungs clear.", "Heart normal."]
    assert (first.is_admin_note, first.is_recommendation) == (False, False)
    assert (second.is_recommendation, second.attributed_findings_spans) == (True, [])
    assert third.is_recommendation is None and third.attributed_findings_spans == []


def test_prompt_numbering_and_examples():
    system, template = load_prompt(
        REPO / "prompts" / "attribution.yaml",
        {n: {"findings": f"0: finding {n}", "impression": f"0: impression {n}"} for n in (1, 2, 3)},
    )
    assert system == "You are an expert radiologist annotator."
    prompt = build_prompt(
        template, report(["LUNGS:", "Clear.", "Normal."], [sentence("Fine."), sentence("Ok.")])
    )
    assert prompt.endswith("[FINDINGS]\n0: Clear.\n1: Normal.\n\n[IMPRESSION]\n0: Fine.\n1: Ok.\n\n[OUTPUT]")
    assert "0: finding 2" in prompt and '"0": {"type": "fact"' in prompt
    with pytest.raises(ValueError):
        fill_examples("{example_1_findings}", {})


# Perturbations -------------------------------------------------------------------------------------


def test_notes_without_findings_are_removed_and_blanket_sentences_are_not_perturbed():
    blanket = sentence("Otherwise normal.", "3.", spans=["a.", "b.", "c.", "d."])
    r = report(
        ["a.", "b.", "c.", "d.", "e."],
        [sentence("Nodule.", spans=["e."]), sentence("Recommend CT.", "2.", rec=True), blanket],
    )
    p = perturb(r, min_hallucinated=1, min_omitted=0, min_preserved=0, seed=0)
    assert [s.text for s in p.hallucinated] == ["Nodule."]
    assert p.findings_sentences == ["a.", "b.", "c.", "d."]
    assert "Recommend CT." not in [s.text for s in p.impression_sentences]
    kept = keep_blanket_summaries([blanket], 0)
    assert [s.text for s in p.impression_sentences] == ["Nodule."] + [s.text for s in kept]
    # Not enough specific sentences for the minimums.
    assert perturb(r, min_hallucinated=1, min_omitted=1, min_preserved=0, seed=0) is None


def test_blanket_summaries_kept_with_probability_067():
    sentences = [sentence(f"s{i}", spans=["a", "b", "c", "d"]) for i in range(10)]
    expected = np.random.default_rng(5).random(10) > 0.33
    assert [s.text for s in keep_blanket_summaries(sentences, 5)] == [
        f"s{i}" for i in range(10) if expected[i]
    ]


def test_secondary_hallucination_needs_less_than_half_support():
    half = sentence("Half.", spans=["a.", "b."])
    less = sentence("Less.", spans=["a.", "b.", "c."])
    none = sentence("No support.")
    kept, extra = secondary_hallucinations([half, less, none], findings=["b.", "x."])
    assert kept == [half, none] and extra == [less]


def test_omitted_findings_exclude_findings_of_preserved_sentences():
    r = report(
        ["a.", "b.", "c."], [sentence("One.", spans=["a.", "b."]), sentence("Two.", "2.", spans=["b.", "c."])]
    )
    for seed in range(50):
        p = perturb(r, min_hallucinated=0, min_omitted=1, min_preserved=1, seed=seed)
        omitted = [s for s in r.impression_sentences if s not in p.impression_sentences]
        assert len(omitted) == 1
        assert p.omitted_findings == (["a."] if omitted[0].text == "One." else ["c."])


# Samples -------------------------------------------------------------------------------------------


def test_report_samples_format():
    r = report(
        ["LUNGS:", "Small nodule.", "No effusion.", "HEART:", "Normal heart."],
        [
            sentence("Nodule.", spans=["Small nodule."]),
            sentence("No effusion.", "2.", spans=["No effusion."]),
            sentence("Normal heart.", "3.", spans=["Normal heart."]),
        ],
    )
    samples = report_samples(r, 0, TEMPLATE)
    assert [(s["error_type"], s["label"]) for s in samples] == [
        ("omission", 1),
        ("omission", 0),
        ("hallucination", 1),
        ("hallucination", 0),
    ]
    assert all(set(s) == {"report_id", "features", "label", "error_type"} for s in samples)
    assert background(r) == "EXAMINATION: Chest. INDICATION: Cough."
    features = samples[0]["features"]
    assert features.startswith(
        "### TASK: DETECT_OMISSION\n\n[CONTEXT]\nEXAMINATION: Chest. INDICATION: Cough."
    )
    impression = features.split("[IMPRESSION]\n")[1].split("\n\n")[0]
    assert impression.startswith("1. ") and "3. " not in impression  # renumbered after the deletion
    assert report_samples(r, 0, TEMPLATE) == samples


# Training ------------------------------------------------------------------------------------------


def test_compute_metrics():
    logits = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    metrics = compute_metrics((logits, np.array([1, 1, 0, 0])))
    assert metrics["f1"] == 0.5 and metrics["accuracy"] == 0.5
    assert (metrics["true_positives"], metrics["false_negatives"]) == (1, 1)


def test_training_config_values():
    cfg = OmegaConf.load(REPO / "configs" / "train_radcheck.yaml")
    assert cfg.training.learning_rate == pytest.approx(2e-4)
    assert (cfg.lora.r, cfg.lora.alpha, cfg.dataset.max_length, cfg.seed) == (16, 32, 1536, 42)
    assert cfg.wandb.enabled is False
