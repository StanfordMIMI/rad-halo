import json

import pandas as pd
import pytest

from radhalo.data.release import load_predictions
from radhalo.reporting.latex import Macros, fmt, lookup, thousands


def test_fmt_rounds_half_up_and_keeps_zeros():
    assert fmt(2.35, 1) == "2.4"
    assert fmt(55.05, 1) == "55.1"
    assert fmt(0.3, 3) == "0.300"


def test_thousands_uses_latex_separator():
    assert thousands(35151) == "35{,}151"


def test_lookup_requires_exactly_one_row():
    df = pd.DataFrame({"k": ["a", "a", "b"], "v": [1, 2, 3]})
    assert lookup(df, "k", "b", "v") == 3
    with pytest.raises(KeyError):
        lookup(df, "k", "a", "v")


def test_macro_names_are_letters_and_unique():
    m = Macros()
    m.add("totalreports", 300)
    with pytest.raises(ValueError):
        m.add("totalreports", 1)
    with pytest.raises(ValueError):
        m.add("f1score", 1)


def test_predictions_must_match_sentence_counts(tmp_path):
    reports = [{"radhalo_id": 0, "impression_sentences": [{}], "findings_sentences": [{}, {}]}]
    good = tmp_path / "good.jsonl"
    good.write_text(json.dumps({"radhalo_id": 0, "impression_scores": [1.0], "findings_scores": [0.0, 1.0]}))
    assert load_predictions(good, reports)[0]["omission"] == [0.0, 1.0]
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"radhalo_id": 0, "impression_scores": [1.0], "findings_scores": [0.0]}))
    with pytest.raises(ValueError):
        load_predictions(bad, reports)
