import pytest
from bitarray import bitarray

from reverse_index.compression import (
    BitWriter,
    CompressedInvertedIndex,
    DeltaGammaCompressor,
    EliasGammaCoder,
)
from reverse_index.inverted_index import InvertedIndex


class TestBitWriter:
    def test_write_bits_stores_msb_first(self):
        writer = BitWriter()
        writer.write_bits(0b101, 3)
        assert list(writer.bits) == [1, 0, 1]

    def test_write_bits_all_zeros(self):
        writer = BitWriter()
        writer.write_bits(0, 4)
        assert list(writer.bits) == [0, 0, 0, 0]

    def test_write_bits_padding_to_requested_length(self):
        writer = BitWriter()
        writer.write_bits(1, 4)  # 0001
        assert list(writer.bits) == [0, 0, 0, 1]

    def test_to_bytes_from_bytes_roundtrip(self):
        writer = BitWriter()
        writer.write_bits(0b10110, 5)
        original_bits = bitarray(writer.bits)

        data = writer.to_bytes()
        restored, _ = BitWriter.from_bytes(data)

        # Original bits are preserved at the start; the tail is alignment padding
        assert restored.bits[: len(original_bits)] == original_bits

    def test_from_bytes_returns_correct_padding(self):
        writer = BitWriter()
        writer.write_bits(1, 1)  # 1 bit -> needs 7 bits of padding to reach 8
        data = writer.to_bytes()
        _, padding = BitWriter.from_bytes(data)
        assert padding == 7


class TestEliasGammaCoder:
    def test_encode_one_is_single_bit(self):
        assert EliasGammaCoder.encode(1) == bitarray("1")

    def test_encode_two(self):
        # k=1, unary "01", binary last 1 bit of 2=10 -> "0"
        assert EliasGammaCoder.encode(2) == bitarray("010")

    def test_encode_three(self):
        # k=1, unary "01", binary last 1 bit of 3=11 -> "1"
        assert EliasGammaCoder.encode(3) == bitarray("011")

    def test_encode_four(self):
        # k=2, unary "001", binary last 2 bits of 4=100 -> "00"
        assert EliasGammaCoder.encode(4) == bitarray("00100")

    def test_encode_rejects_zero(self):
        with pytest.raises(ValueError):
            EliasGammaCoder.encode(0)

    def test_encode_rejects_negative(self):
        with pytest.raises(ValueError):
            EliasGammaCoder.encode(-5)

    def test_decode_one(self):
        value, pos = EliasGammaCoder.decode(bitarray("1"), 0)
        assert value == 1
        assert pos == 1

    def test_decode_two(self):
        value, pos = EliasGammaCoder.decode(bitarray("010"), 0)
        assert value == 2
        assert pos == 3

    def test_decode_advances_position(self):
        # Two encoded values concatenated: encode(1)="1", encode(2)="010"
        bits = bitarray("1010")
        v1, pos = EliasGammaCoder.decode(bits, 0)
        v2, _ = EliasGammaCoder.decode(bits, pos)
        assert v1 == 1
        assert v2 == 2

    def test_decode_invalid_missing_separator(self):
        with pytest.raises(ValueError):
            EliasGammaCoder.decode(bitarray("000"), 0)

    def test_decode_invalid_truncated(self):
        # encode(4) = "00100"; truncate to "001" — missing binary part
        with pytest.raises(ValueError):
            EliasGammaCoder.decode(bitarray("001"), 0)

    @pytest.mark.parametrize("x", [1, 2, 3, 4, 5, 7, 8, 15, 16, 63, 64, 127, 255])
    def test_roundtrip(self, x: int):
        encoded = EliasGammaCoder.encode(x)
        decoded, _ = EliasGammaCoder.decode(encoded, 0)
        assert decoded == x


class TestDeltaGammaCompressor:
    def test_empty_list_encodes_and_decodes(self):
        data = DeltaGammaCompressor.encode([])
        assert DeltaGammaCompressor.decode(data) == []

    def test_decode_truncated_data_returns_empty(self):
        assert DeltaGammaCompressor.decode(b"\x00\x01") == []

    def test_decode_zero_length_header_returns_empty(self):
        assert DeltaGammaCompressor.decode(b"\x00\x00\x00\x00") == []

    def test_single_element(self):
        for doc_id in [0, 1, 5, 100, 10000]:
            encoded = DeltaGammaCompressor.encode([doc_id])
            assert DeltaGammaCompressor.decode(encoded) == [doc_id]

    def test_sequential_ids(self):
        original = list(range(50))
        assert (
            DeltaGammaCompressor.decode(DeltaGammaCompressor.encode(original))
            == original
        )

    def test_sparse_ids(self):
        original = [0, 100, 500, 1000, 9999]
        assert (
            DeltaGammaCompressor.decode(DeltaGammaCompressor.encode(original))
            == original
        )

    def test_two_element_list(self):
        original = [3, 7]
        assert (
            DeltaGammaCompressor.decode(DeltaGammaCompressor.encode(original))
            == original
        )

    def test_compressed_is_bytes(self):
        data = DeltaGammaCompressor.encode([1, 2, 3])
        assert isinstance(data, bytes)

    def test_compression_reduces_size_for_sequential(self):
        ids = list(range(1000))
        compressed = DeltaGammaCompressor.encode(ids)
        # 1000 ints * 4 bytes = 4000 bytes uncompressed (raw)
        assert len(compressed) < 4000


class TestCompressedInvertedIndex:
    def _make_base(self) -> InvertedIndex:
        idx = InvertedIndex()
        idx.add_document(0, ["alpha", "beta"], metadata={"id": 0})
        idx.add_document(1, ["alpha", "gamma"], metadata={"id": 1})
        idx.add_document(2, ["beta", "gamma"], metadata={"id": 2})
        return idx

    def test_vocabulary_matches_base(self):
        base = self._make_base()
        compressed = CompressedInvertedIndex(base)
        assert set(compressed.get_terms()) == {"alpha", "beta", "gamma"}

    def test_get_posting_list_correct(self):
        base = self._make_base()
        compressed = CompressedInvertedIndex(base)
        assert compressed.get_posting_list("alpha") == [0, 1]
        assert compressed.get_posting_list("beta") == [0, 2]
        assert compressed.get_posting_list("gamma") == [1, 2]

    def test_get_posting_list_missing_returns_none(self):
        compressed = CompressedInvertedIndex(self._make_base())
        assert compressed.get_posting_list("missing") is None

    def test_get_compressed_size_positive(self):
        compressed = CompressedInvertedIndex(self._make_base())
        assert compressed.get_compressed_size("alpha") > 0

    def test_get_compressed_size_missing_returns_zero(self):
        compressed = CompressedInvertedIndex(self._make_base())
        assert compressed.get_compressed_size("missing") == 0

    def test_doc_count(self):
        compressed = CompressedInvertedIndex(self._make_base())
        assert compressed.get_doc_count() == 3

    def test_to_dict_from_dict_roundtrip(self):
        base = self._make_base()
        compressed = CompressedInvertedIndex(base)
        restored = CompressedInvertedIndex.from_dict(compressed.to_dict())

        assert restored.get_posting_list("alpha") == [0, 1]
        assert restored.get_posting_list("gamma") == [1, 2]
        assert restored.get_doc_count() == 3
