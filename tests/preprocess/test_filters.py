import copy

import pytest

from ragagent.preprocess.filters import effective_filters
from ragagent.retrieval.filters import build_expr

A, B = "0123456789abcdef", "aaaaaaaaaaaaaaaa"


@pytest.mark.parametrize("hard,soft,expected", [
    (None, None, None),
    ({"file_types": ["pdf"]}, None, {"file_types": ["pdf"]}),
    (None, {"doc_ids": [A]}, {"doc_ids": [A]}),
    ({"doc_ids": [A, B]}, {"doc_ids": [A]}, {"doc_ids": [A]}),
    ({"file_types": ["pdf"]}, {"doc_ids": [A]}, {"file_types": ["pdf"], "doc_ids": [A]}),
    ({"file_types": ["pdf"]}, {"file_types": ["md"], "doc_ids": [A]}, {"file_types": ["pdf"]}),
    ({"doc_ids": [A], "file_types": ["pdf", "md"]},
     {"doc_ids": [B], "file_types": ["pdf"]}, {"doc_ids": [A], "file_types": ["md", "pdf"]}),
])
def test_hard_soft_merge_always_passes_security_boundary(hard, soft, expected):
    before = copy.deepcopy((hard, soft))
    assert effective_filters(hard, soft) == expected
    assert build_expr(effective_filters(hard, soft)).startswith("searchable == true")
    assert (hard, soft) == before


@pytest.mark.parametrize("bad", [{"expr": "or true"}, {"doc_ids": ['bad" or true']},
                                  {"file_types": ["pdf"] * 101}, {"file_types": "pdf"}])
def test_merge_cannot_hide_invalid_values(bad):
    with pytest.raises(ValueError):
        effective_filters({"file_types": ["pdf"]}, bad)
    with pytest.raises(ValueError):
        effective_filters(bad, {"file_types": ["md"]})
