import pytest

from reverse_index.inverted_index import InvertedIndex
from reverse_index.search import BooleanSearcher, QueryParser


def make_index() -> InvertedIndex:
    """Minimal index with predictable terms for search tests."""
    idx = InvertedIndex()
    idx.add_document(0, ["python", "code"], metadata={"title": "doc0"})
    idx.add_document(1, ["python", "test"], metadata={"title": "doc1"})
    idx.add_document(2, ["java", "code"], metadata={"title": "doc2"})
    idx.add_document(3, ["java", "test"], metadata={"title": "doc3"})
    return idx


class TestQueryParser:
    def test_single_term_parsed_as_and(self):
        assert QueryParser.parse("hello") == [("AND", "hello")]

    def test_two_terms_both_and(self):
        result = QueryParser.parse("hello world")
        assert ("AND", "hello") in result
        assert ("AND", "world") in result

    def test_or_keyword_produces_or_group(self):
        # "hello or world" -> [("AND", "hello"), ("OR_GROUP", ["world"])]
        result = QueryParser.parse("hello or world")
        or_groups = [v for op, v in result if op == "OR_GROUP"]
        assert any("world" in g for g in or_groups)

    def test_slash_as_or_operator(self):
        # "hello / world" with spaces -> OR_GROUP
        result = QueryParser.parse("hello / world")
        or_groups = [v for op, v in result if op == "OR_GROUP"]
        assert any("world" in g for g in or_groups)

    def test_and_keyword_is_ignored(self):
        # "hello AND world" -> both parsed as AND terms
        result = QueryParser.parse("hello AND world")
        assert ("AND", "hello") in result
        assert ("AND", "world") in result

    def test_empty_string_returns_empty(self):
        assert QueryParser.parse("") == []

    def test_grouped_terms_produce_or_group(self):
        result = QueryParser.parse("(alpha beta)")
        assert any(op == "OR_GROUP" for op, _ in result)

    def test_result_is_list_of_tuples(self):
        result = QueryParser.parse("term")
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)


class TestBooleanSearcherIntersect:
    def test_basic_intersection(self):
        result = BooleanSearcher._intersect([[1, 2, 3], [2, 3, 4]])
        assert result == [2, 3]

    def test_no_overlap(self):
        assert BooleanSearcher._intersect([[1, 2], [3, 4]]) == []

    def test_empty_lists_input(self):
        assert BooleanSearcher._intersect([]) == []

    def test_single_list_returned_as_is(self):
        assert BooleanSearcher._intersect([[1, 2, 3]]) == [1, 2, 3]

    def test_one_empty_list_yields_empty(self):
        assert BooleanSearcher._intersect([[1, 2, 3], []]) == []

    def test_three_lists(self):
        result = BooleanSearcher._intersect([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
        assert result == [3, 4]

    def test_identical_lists(self):
        assert BooleanSearcher._intersect([[1, 2, 3], [1, 2, 3]]) == [1, 2, 3]


class TestBooleanSearcherUnion:
    def test_disjoint_lists_merged_sorted(self):
        result = BooleanSearcher._union([[1, 3], [2, 4]])
        assert result == [1, 2, 3, 4]

    def test_empty_input(self):
        assert BooleanSearcher._union([]) == []

    def test_single_list_returned_as_is(self):
        assert BooleanSearcher._union([[1, 2, 3]]) == [1, 2, 3]

    def test_one_empty_list(self):
        result = BooleanSearcher._union([[], [1, 2]])
        assert 1 in result and 2 in result

    def test_preserves_order(self):
        result = BooleanSearcher._union([[1, 5], [2, 4], [3]])
        assert result == sorted(result)


class TestBooleanSearcherSearch:
    def test_single_term_finds_matching_docs(self):
        searcher = BooleanSearcher(make_index())
        assert searcher.search("python") == [0, 1]

    def test_and_query_intersects_results(self):
        searcher = BooleanSearcher(make_index())
        assert searcher.search("python code") == [0]

    def test_missing_term_returns_empty(self):
        searcher = BooleanSearcher(make_index())
        assert searcher.search("ruby") == []

    def test_empty_query_returns_empty(self):
        searcher = BooleanSearcher(make_index())
        assert searcher.search("") == []

    def test_result_is_sorted(self):
        searcher = BooleanSearcher(make_index())
        result = searcher.search("code")
        assert result == sorted(result)

    def test_and_with_all_missing(self):
        searcher = BooleanSearcher(make_index())
        assert searcher.search("python ruby") == []

    def test_search_with_metadata_returns_dicts(self):
        searcher = BooleanSearcher(make_index())
        results = searcher.search_with_metadata("python")
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)

    def test_get_query_stats_structure(self):
        searcher = BooleanSearcher(make_index())
        stats = searcher.get_query_stats("python")
        assert "parsed_query" in stats
        assert "terms_found" in stats
        assert "posting_list_sizes" in stats

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("java", [2, 3]),
            ("code", [0, 2]),
            ("test", [1, 3]),
            ("java code", [2]),
        ],
    )
    def test_parametrized_queries(self, query: str, expected: list[int]):
        searcher = BooleanSearcher(make_index())
        assert searcher.search(query) == expected
