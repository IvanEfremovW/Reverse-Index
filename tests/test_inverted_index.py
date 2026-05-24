import pytest

from reverse_index.inverted_index import InvertedIndex


def make_index(*doc_tokens: list[str]) -> InvertedIndex:
    """Helper: build an InvertedIndex from positional token lists (doc_id = position)."""
    idx = InvertedIndex()
    for doc_id, tokens in enumerate(doc_tokens):
        idx.add_document(doc_id, tokens)
    return idx


class TestAddDocument:
    def test_creates_vocabulary_entries(self):
        idx = make_index(["hello", "world"])
        assert "hello" in idx.vocabulary
        assert "world" in idx.vocabulary

    def test_posting_list_contains_doc_id(self):
        idx = make_index(["word"])
        assert idx.get_posting_list("word") == [0]

    def test_multiple_docs_same_term(self):
        idx = make_index(["word"], ["word"], ["word"])
        assert idx.get_posting_list("word") == [0, 1, 2]

    def test_deduplicates_term_within_doc(self):
        idx = make_index(["word", "word", "word"])
        plist = idx.get_posting_list("word")
        assert plist is not None
        assert plist.count(0) == 1

    def test_stores_metadata(self):
        idx = InvertedIndex()
        idx.add_document(0, ["hello"], metadata={"channel": "test"})
        assert idx.doc_metadata[0]["channel"] == "test"

    def test_no_metadata_does_not_store(self):
        idx = InvertedIndex()
        idx.add_document(0, ["hello"])
        assert 0 not in idx.doc_metadata

    def test_returns_copy_of_posting_list(self):
        idx = make_index(["word"])
        plist = idx.get_posting_list("word")
        assert plist is not None
        plist.append(99)
        assert 99 not in idx.get_posting_list("word")  # type: ignore[operator]

    def test_posting_list_sorted_after_build(self):
        idx = InvertedIndex()
        # Build processes docs in order; docs with same term but added in order
        docs = [
            {"doc_id": 0, "tokens": ["x"]},
            {"doc_id": 2, "tokens": ["x"]},
            {"doc_id": 1, "tokens": ["x"]},
        ]
        idx.build(docs)
        plist = idx.get_posting_list("x")
        assert plist is not None
        assert plist == sorted(plist)


class TestGetPostingList:
    def test_missing_term_returns_none(self):
        idx = make_index(["hello"])
        assert idx.get_posting_list("missing") is None

    def test_existing_term_returns_list(self):
        idx = make_index(["hello"])
        result = idx.get_posting_list("hello")
        assert isinstance(result, list)


class TestBuild:
    def test_builds_from_documents(self):
        docs = [
            {"doc_id": 0, "tokens": ["alpha", "beta"]},
            {"doc_id": 1, "tokens": ["beta", "gamma"]},
        ]
        idx = InvertedIndex()
        idx.build(docs)

        assert idx.get_posting_list("alpha") == [0]
        assert idx.get_posting_list("beta") == [0, 1]
        assert idx.get_posting_list("gamma") == [1]

    def test_stores_extra_fields_as_metadata(self):
        docs = [{"doc_id": 0, "tokens": ["word"], "channel": "news", "views": 100}]
        idx = InvertedIndex()
        idx.build(docs)

        meta = idx.doc_metadata[0]
        assert meta["channel"] == "news"
        assert meta["views"] == 100

    def test_deduplicates_posting_list(self):
        docs = [
            {"doc_id": 0, "tokens": ["dup", "dup"]},
            {"doc_id": 0, "tokens": ["dup"]},  # same doc_id again
        ]
        idx = InvertedIndex()
        idx.build(docs)
        plist = idx.get_posting_list("dup")
        assert plist is not None
        assert len(plist) == len(set(plist))


class TestGetTermsAndDocCount:
    def test_get_terms_returns_all_unique(self):
        idx = make_index(["a", "b", "c"], ["b", "d"])
        assert set(idx.get_terms()) == {"a", "b", "c", "d"}

    def test_get_doc_count_with_metadata(self):
        idx = InvertedIndex()
        idx.add_document(0, ["word"], metadata={"x": 1})
        idx.add_document(1, ["word"], metadata={"x": 2})
        assert idx.get_doc_count() == 2

    def test_get_doc_count_without_metadata(self):
        idx = make_index(["a"], ["b"])
        # add_document without metadata doesn't populate doc_metadata
        assert idx.get_doc_count() == 0

    def test_len_equals_vocabulary_size(self):
        idx = make_index(["a", "b", "c"])
        assert len(idx) == 3

    def test_len_empty_index(self):
        assert len(InvertedIndex()) == 0


class TestSerialisation:
    def test_to_dict_from_dict_roundtrip(self):
        idx = make_index(["hello", "world"], ["hello"])
        idx.add_document(0, ["hello", "world"], metadata={"tag": "test"})

        restored = InvertedIndex.from_dict(idx.to_dict())

        assert restored.get_posting_list("hello") == idx.get_posting_list("hello")
        assert restored.get_posting_list("world") == idx.get_posting_list("world")

    def test_from_dict_preserves_compressed_flag(self):
        idx = InvertedIndex(compressed=True)
        data = idx.to_dict()
        assert data["compressed"] is True
        restored = InvertedIndex.from_dict(data)
        assert restored.compressed is True

    def test_from_dict_default_compressed_false(self):
        data = {"vocabulary": {}, "posting_lists": []}
        restored = InvertedIndex.from_dict(data)
        assert restored.compressed is False

    @pytest.mark.parametrize("n_docs", [1, 10, 100])
    def test_roundtrip_various_sizes(self, n_docs: int):
        idx = InvertedIndex()
        for i in range(n_docs):
            idx.add_document(i, [f"term{i}", "shared"])
        restored = InvertedIndex.from_dict(idx.to_dict())
        assert restored.get_posting_list("shared") == list(range(n_docs))
