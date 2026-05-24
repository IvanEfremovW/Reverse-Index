import json
from pathlib import Path

import pytest

from reverse_index.compression import CompressedInvertedIndex
from reverse_index.inverted_index import InvertedIndex
from reverse_index.storage import (
    METADATA_FILE,
    compare_index_sizes,
    format_size,
    load_index,
    save_index,
)


def make_index(n_docs: int = 3) -> InvertedIndex:
    idx = InvertedIndex()
    for i in range(n_docs):
        idx.add_document(i, [f"term{i}", "shared"], metadata={"doc": i})
    return idx


class TestFormatSize:
    def test_bytes(self):
        assert format_size(500) == "500.00 B"

    def test_exactly_one_kb(self):
        assert format_size(1024) == "1.00 KB"

    def test_two_kb(self):
        assert format_size(2048) == "2.00 KB"

    def test_one_mb(self):
        assert format_size(1024 * 1024) == "1.00 MB"

    def test_one_gb(self):
        assert format_size(1024 * 1024 * 1024) == "1.00 GB"

    def test_zero_bytes(self):
        assert format_size(0) == "0.00 B"


class TestSaveLoadUncompressed:
    def test_roundtrip_posting_lists(self, tmp_path: Path):
        idx = make_index()
        save_index(idx, tmp_path, compressed=False)
        loaded = load_index(tmp_path, compressed=False)

        assert loaded.get_posting_list("shared") == list(range(3))
        assert loaded.get_posting_list("term0") == [0]

    def test_returns_inverted_index_instance(self, tmp_path: Path):
        save_index(make_index(), tmp_path, compressed=False)
        loaded = load_index(tmp_path, compressed=False)
        assert isinstance(loaded, InvertedIndex)

    def test_saves_index_file(self, tmp_path: Path):
        save_index(make_index(), tmp_path, compressed=False)
        assert (tmp_path / "index_uncompressed.pkl").exists()

    def test_save_result_contains_size(self, tmp_path: Path):
        result = save_index(make_index(), tmp_path, compressed=False)
        assert "uncompressed_size" in result
        assert result["uncompressed_size"] > 0

    def test_save_result_contains_human_size(self, tmp_path: Path):
        result = save_index(make_index(), tmp_path, compressed=False)
        assert "uncompressed_size_human" in result
        assert isinstance(result["uncompressed_size_human"], str)

    def test_creates_output_dir(self, tmp_path: Path):
        nested = tmp_path / "nested" / "dir"
        save_index(make_index(), nested, compressed=False)
        assert nested.exists()


class TestSaveLoadCompressed:
    def test_roundtrip_posting_lists(self, tmp_path: Path):
        idx = make_index()
        save_index(idx, tmp_path, compressed=True)
        loaded = load_index(tmp_path, compressed=True)

        assert loaded.get_posting_list("shared") == list(range(3))
        assert loaded.get_posting_list("term1") == [1]

    def test_returns_compressed_index_instance(self, tmp_path: Path):
        save_index(make_index(), tmp_path, compressed=True)
        loaded = load_index(tmp_path, compressed=True)
        assert isinstance(loaded, CompressedInvertedIndex)

    def test_saves_compressed_file(self, tmp_path: Path):
        save_index(make_index(), tmp_path, compressed=True)
        assert (tmp_path / "index_compressed.pkl").exists()

    def test_save_result_contains_size(self, tmp_path: Path):
        result = save_index(make_index(), tmp_path, compressed=True)
        assert "compressed_size" in result
        assert result["compressed_size"] > 0

    def test_accepts_already_compressed_index(self, tmp_path: Path):
        idx = make_index()
        compressed = CompressedInvertedIndex(idx)
        result = save_index(compressed, tmp_path, compressed=True)
        assert result["compressed_size"] > 0


class TestMetadataFile:
    def test_metadata_file_created_by_default(self, tmp_path: Path):
        save_index(make_index(), tmp_path, compressed=False)
        assert (tmp_path / METADATA_FILE).exists()

    def test_metadata_file_skipped_when_disabled(self, tmp_path: Path):
        save_index(make_index(), tmp_path, compressed=False, include_metadata=False)
        assert not (tmp_path / METADATA_FILE).exists()

    def test_metadata_contains_doc_count(self, tmp_path: Path):
        save_index(make_index(5), tmp_path, compressed=False)
        with open(tmp_path / METADATA_FILE, encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["doc_count"] == 5

    def test_metadata_contains_vocabulary_size(self, tmp_path: Path):
        idx = make_index(3)
        save_index(idx, tmp_path, compressed=False)
        with open(tmp_path / METADATA_FILE, encoding="utf-8") as f:
            meta = json.load(f)
        # 3 term{i} + 1 shared = 4 unique terms
        assert meta["vocabulary_size"] == 4


class TestCompareIndexSizes:
    def test_returns_expected_keys(self, tmp_path: Path):
        idx = make_index(10)
        save_index(idx, tmp_path, compressed=False)
        save_index(idx, tmp_path, compressed=True)

        result = compare_index_sizes(
            tmp_path / "index_uncompressed.pkl",
            tmp_path / "index_compressed.pkl",
        )

        assert "compression_ratio" in result
        assert "space_savings_percent" in result
        assert "uncompressed_bytes" in result
        assert "compressed_bytes" in result

    def test_compression_ratio_positive(self, tmp_path: Path):
        idx = make_index(10)
        save_index(idx, tmp_path, compressed=False)
        save_index(idx, tmp_path, compressed=True)

        result = compare_index_sizes(
            tmp_path / "index_uncompressed.pkl",
            tmp_path / "index_compressed.pkl",
        )
        assert result["compression_ratio"] > 0

    def test_human_readable_sizes_present(self, tmp_path: Path):
        idx = make_index(5)
        save_index(idx, tmp_path, compressed=False)
        save_index(idx, tmp_path, compressed=True)

        result = compare_index_sizes(
            tmp_path / "index_uncompressed.pkl",
            tmp_path / "index_compressed.pkl",
        )
        assert isinstance(result["uncompressed_human"], str)
        assert isinstance(result["compressed_human"], str)

    @pytest.mark.parametrize("n_docs", [5, 20, 50])
    def test_various_index_sizes(self, tmp_path: Path, n_docs: int):
        idx = make_index(n_docs)
        save_index(idx, tmp_path, compressed=False)
        save_index(idx, tmp_path, compressed=True)

        result = compare_index_sizes(
            tmp_path / "index_uncompressed.pkl",
            tmp_path / "index_compressed.pkl",
        )
        assert result["compression_ratio"] > 0
        assert result["space_savings_percent"] <= 100
