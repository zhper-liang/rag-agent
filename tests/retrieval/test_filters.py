import copy

import pytest

from ragagent.retrieval.filters import FILE_TYPES, build_expr


@pytest.mark.parametrize("bad", [
    {"expr": 'true or searchable == false'},
    {"file_type": ["pdf"]},
    {"doc_ids": ['0123456789abcdef" or true']},
    {"doc_ids": ["(0123456789abcdef)"]},
    {"doc_ids": ["or"]},
    {"doc_ids": ["A123456789abcdef"]},
    {"doc_ids": ["0123456789abcde"]},
    {"doc_ids": ["0123456789abcdef\n"]},
    {"doc_ids": ["0123456789abcdef"] * 101},
    {"doc_ids": "0123456789abcdef"},
    {"doc_ids": [None]},
    {"doc_ids": [1]},
    {"doc_ids": None},
    {"file_types": ['pdf" or (true)']},
    {"file_types": ["PDF"]},
    {"file_types": ["exe"]},
    {"file_types": "pdf"},
    {"file_types": {"pdf": True}},
    {"file_types": ["pdf"] * 101},
    {"file_types": [False]},
    [], "", 0, False,
])
def test_rejects_all_invalid_input_without_building_expression(bad):
    # 原样字符串永远到不了调用者，更到不了 Milvus。
    with pytest.raises(ValueError):
        build_expr(bad)


def test_only_validated_values_are_in_expression_and_input_is_unchanged():
    filters = {"file_types": ["pdf", "md", "pdf"],
               "doc_ids": ["0123456789abcdef", "aaaaaaaaaaaaaaaa"]}
    before = copy.deepcopy(filters)
    assert build_expr(filters) == ('searchable == true and file_type in ["md","pdf"] '
                                   'and doc_id in ["0123456789abcdef","aaaaaaaaaaaaaaaa"]')
    assert filters == before


@pytest.mark.parametrize("filters", [None, {}, {"doc_ids": []}, {"file_types": []}])
def test_empty_filters(filters):
    assert build_expr(filters) == "searchable == true"


def test_all_file_types_and_maximum_doc_ids():
    expr = build_expr({"file_types": list(FILE_TYPES), "doc_ids": [f"{i:016x}" for i in range(100)]})
    assert expr.count('"') == 220
