import numpy as np
import pytest

from qfc.squad import exact_match, f1_score, normalize_answer, per_example_scores, postprocess


def test_official_normalization_and_scores():
    assert normalize_answer("The  Eiffel-Tower!") == "eiffeltower"
    assert exact_match("the Eiffel Tower", "Eiffel Tower.") == 1.0
    assert f1_score("Eiffel Tower in Paris", "the Eiffel Tower") == pytest.approx(2 * (2 / 4) * 1.0 / (2 / 4 + 1.0))
    assert f1_score("Berlin", "Paris") == 0.0


def test_postprocess_picks_best_valid_span_across_windows():
    ex = {"context": ["Paris is the capital of France."], "answers": [{"text": ["Paris"]}]}
    # two windows of the same example; offsets None = question/special tokens
    feats = {"example_index": [0, 0],
             "offset_mapping": [[None, (0, 5), (6, 8), (9, 12)], [None, (13, 20), (24, 30)]]}
    starts = [np.array([0.0, 5.0, 1.0, 0.0]), np.array([9.0, 1.0, 0.0])]
    ends = [np.array([0.0, 5.0, 0.0, 0.0]), np.array([9.0, 0.0, 1.0])]
    preds = postprocess(ex, feats, starts, ends)
    assert preds == ["Paris"]  # window 2's high scores sit on a None (question) token and are ignored
    f1, em = per_example_scores(ex, preds)
    assert f1 == [1.0] and em == [1.0]
