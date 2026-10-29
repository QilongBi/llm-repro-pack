from __future__ import annotations

import pytest

from compare import cohens_kappa, compare_labels, krippendorff_alpha_nominal, macro_f1
from extract import INVALID, extract
import render_inputs as ri


def test_cohens_kappa_known_value():
    assert cohens_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)
    assert cohens_kappa(["a", "b"], ["a", "b"]) == 1.0


def test_krippendorff_alpha_known_value():
    units = [[1, 1], [1, 0], [0, 0], [0, 0]]
    assert krippendorff_alpha_nominal(units) == pytest.approx(1 - 14 / 30)
    assert krippendorff_alpha_nominal([["a", "a"], ["b", "b"]]) == 1.0
    # Missing values are skipped.
    assert krippendorff_alpha_nominal([["a", None], ["a", "a"], ["b", "b"]]) == 1.0


def test_macro_f1():
    assert macro_f1(["a", "a", "b"], ["a", "b", "b"]) == pytest.approx((2 / 3 + 2 / 3) / 2)


def test_compare_single_and_multi():
    a = {"1": "x", "2": "y", "3": "x"}
    b = {"1": "x", "2": "x", "4": "y"}
    r = compare_labels(a, b)
    assert r["n_compared"] == 2 and r["missing_in_b"] == ["3"] and r["exact_match_rate"] == 0.5
    m = compare_labels({"1": ("a", "b"), "2": ("a",)}, {"1": ("a", "b"), "2": ("b",)}, multi_label=True)
    assert m["exact_match_rate"] == 0.5
    assert m["mean_jaccard"] == pytest.approx(0.5)


def test_extract_rules():
    labels = ["positive", "negative"]
    assert extract(" Positive.\n", {"type": "exact"}, labels, ["1"]) == {"1": "positive"}
    assert extract("I think positive", {"type": "exact"}, labels, ["1"]) == {"1": INVALID}
    js = '{"predictions": [{"id": "a", "label": "negative"}, {"id": "b", "label": "maybe"}]}'
    assert extract(js, {"type": "json_predictions"}, labels, ["a", "b", "c"]) == \
        {"a": "negative", "b": INVALID, "c": INVALID}
    harmony = ("<|channel|>analysis<|message|>The answer is negative maybe<|end|>"
               "<|start|>assistant<|channel|>final<|message|>positive<|return|>")
    assert extract(harmony, {"type": "harmony_final", "then": {"type": "exact"}}, labels, ["1"]) == {"1": "positive"}
    multi = extract("positive; negative", {"type": "exact", "multi_label": True}, labels, ["1"])
    assert multi == {"1": ("negative", "positive")}


def test_render_from_spec_is_deterministic():
    spec = {"system": "Labels: {label_list}. JSON like {\"a\": 1}", "user_template": "Text: {text}",
            "labels": [{"name": "p", "definition": "d"}], "few_shot": [{"input": {"text": "x"}, "output": "p"}]}
    recs = [{"id": "1", "text": "hello {name}"}]
    a = ri.render_from_spec(spec, recs, "id")
    b = ri.render_from_spec(spec, recs, "id")
    assert a == b
    msgs = a[0]["messages"]
    assert msgs[0]["content"] == 'Labels: p. JSON like {"a": 1}'
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[-1]["content"] == "Text: hello {name}"
