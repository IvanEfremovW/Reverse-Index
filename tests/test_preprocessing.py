import pytest

from reverse_index.preprocessing import (
    clean_text,
    filter_by_document_frequency,
    normalize_text,
    tokenize,
)


class TestNormalizeText:
    def test_lowercases(self):
        assert normalize_text("Hello World") == "hello world"

    def test_strips_whitespace(self):
        assert normalize_text("  hello  ") == "hello"

    def test_collapses_multiple_spaces(self):
        assert normalize_text("hello   world") == "hello world"

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_already_normalized(self):
        assert normalize_text("hello world") == "hello world"


class TestCleanText:
    def test_removes_http_url(self):
        result = clean_text("visit https://example.com now")
        assert "https" not in result
        assert "example" not in result

    def test_removes_www_url(self):
        result = clean_text("go to www.example.com please")
        assert "www" not in result

    def test_removes_mention(self):
        result = clean_text("hello @username world")
        assert "@username" not in result

    def test_removes_hashtag(self):
        result = clean_text("trending #python today")
        assert "#python" not in result

    def test_removes_digits(self):
        result = clean_text("year 2024 report")
        assert "2024" not in result

    def test_removes_punctuation(self):
        result = clean_text("hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_keeps_cyrillic_letters(self):
        result = clean_text("привет мир")
        assert "привет" in result
        assert "мир" in result

    def test_keeps_latin_letters(self):
        result = clean_text("hello world")
        assert "hello" in result
        assert "world" in result

    def test_lowercases_output(self):
        result = clean_text("HELLO WORLD")
        assert result == result.lower()

    def test_empty_string(self):
        assert clean_text("") == ""


class TestTokenize:
    def test_filters_single_char_tokens(self):
        tokens = tokenize("a b c word")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens

    def test_filters_english_stopwords(self):
        tokens = tokenize("the cat sat on the mat")
        assert "the" not in tokens
        assert "on" not in tokens

    def test_filters_russian_stopwords(self):
        tokens = tokenize("это и так далее")
        # "и" is in Russian stopwords
        assert "и" not in tokens

    def test_returns_list(self):
        result = tokenize("hello world")
        assert isinstance(result, list)

    def test_empty_string_returns_empty(self):
        assert tokenize("") == []

    def test_content_word_survives(self):
        tokens = tokenize("программирование")
        assert len(tokens) > 0


class TestFilterByDocumentFrequency:
    def test_removes_terms_below_threshold(self):
        docs = [
            {"tokens": ["common", "rare"]},
            {"tokens": ["common"]},
            {"tokens": ["common"]},
        ]
        result = filter_by_document_frequency(docs, min_doc_freq=2)
        for doc in result:
            assert "rare" not in doc["tokens"]

    def test_keeps_terms_at_threshold(self):
        docs = [
            {"tokens": ["word", "other"]},
            {"tokens": ["word", "other"]},
        ]
        result = filter_by_document_frequency(docs, min_doc_freq=2)
        assert "word" in result[0]["tokens"]
        assert "other" in result[0]["tokens"]

    def test_min_doc_freq_one_keeps_all(self):
        docs = [{"tokens": ["unique", "singular"]}]
        result = filter_by_document_frequency(docs, min_doc_freq=1)
        assert "unique" in result[0]["tokens"]
        assert "singular" in result[0]["tokens"]

    def test_returns_same_list_object(self):
        docs = [{"tokens": ["word"]}]
        result = filter_by_document_frequency(docs, min_doc_freq=1)
        assert result is docs

    def test_empty_documents(self):
        result = filter_by_document_frequency([], min_doc_freq=1)
        assert result == []

    def test_term_appears_in_multiple_docs(self):
        docs = [{"tokens": ["x", "y"]}, {"tokens": ["x"]}, {"tokens": ["x"]}]
        result = filter_by_document_frequency(docs, min_doc_freq=3)
        for doc in result:
            assert "x" in doc["tokens"]
            assert "y" not in doc["tokens"]

    @pytest.mark.parametrize("min_freq", [1, 2, 3])
    def test_parametrized_threshold(self, min_freq: int):
        docs = [{"tokens": ["a", "b"]} for _ in range(min_freq)]
        result = filter_by_document_frequency(docs, min_doc_freq=min_freq)
        assert all("a" in doc["tokens"] for doc in result)
