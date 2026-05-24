import pickle
import json
from pathlib import Path
from typing import Union, Dict, Any

from .inverted_index import InvertedIndex
from .compression import CompressedInvertedIndex


INDEX_FILE_UNCOMPRESSED = "index_uncompressed.pkl"
INDEX_FILE_COMPRESSED = "index_compressed.pkl"
METADATA_FILE = "metadata.json"


def get_file_size(filepath: Union[str, Path]) -> int:
    """Получение размера файла в байтах."""
    return Path(filepath).stat().st_size


def format_size(bytes_: int) -> str:
    """Человекочитаемое представление размера."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_ < 1024:
            return f"{bytes_:.2f} {unit}"

        bytes_ //= 1024
    return f"{bytes_:.2f} TB"


def save_index(
    index: Union[InvertedIndex, CompressedInvertedIndex],
    output_dir: Union[str, Path],
    compressed: bool = False,
    include_metadata: bool = True,
) -> Dict[str, Any]:
    """
    Сохранение индекса на диск.

    Args:
        index: Экземпляр индекса для сохранения
        output_dir: Директория для сохранения
        compressed: Сохранять ли в сжатом формате
        include_metadata: Включать ли метаданные документов

    Returns:
        Dict с информацией о сохранённых файлах
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {}

    if compressed:
        # Для сжатого индекса
        if not isinstance(index, CompressedInvertedIndex):
            index = CompressedInvertedIndex(index)

        filepath = output_dir / INDEX_FILE_COMPRESSED
        data = index.to_dict()

        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        result["compressed_file"] = str(filepath)
        result["compressed_size"] = get_file_size(filepath)
        result["compressed_size_human"] = format_size(result["compressed_size"])

    else:
        # Для несжатого индекса
        filepath = output_dir / INDEX_FILE_UNCOMPRESSED
        data = index.to_dict()

        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        result["uncompressed_file"] = str(filepath)
        result["uncompressed_size"] = get_file_size(filepath)
        result["uncompressed_size_human"] = format_size(result["uncompressed_size"])

    # Сохранение метаданных
    if include_metadata:
        metadata = {
            "doc_count": index.get_doc_count(),
            "vocabulary_size": len(index.get_terms()),
            "compressed": compressed,
        }
        meta_path = output_dir / METADATA_FILE
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        result["metadata_file"] = str(meta_path)

    return result


def load_index(
    input_dir: Union[str, Path], compressed: bool = False
) -> Union[InvertedIndex, CompressedInvertedIndex]:
    """
    Загрузка индекса с диска.

    Args:
        input_dir: Директория с сохранённым индексом
        compressed: Загружать ли сжатый формат

    Returns:
        Экземпляр загруженного индекса
    """
    input_dir = Path(input_dir)

    if compressed:
        filepath = input_dir / INDEX_FILE_COMPRESSED
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return CompressedInvertedIndex.from_dict(data)
    else:
        filepath = input_dir / INDEX_FILE_UNCOMPRESSED
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return InvertedIndex.from_dict(data)


def compare_index_sizes(
    uncompressed_path: Union[str, Path], compressed_path: Union[str, Path]
) -> Dict[str, Any]:
    """
    Сравнение размеров сжатого и несжатого индексов.

    Returns:
        Dict с метриками сравнения
    """
    uncompressed_size = get_file_size(uncompressed_path)
    compressed_size = get_file_size(compressed_path)

    ratio = uncompressed_size / compressed_size if compressed_size > 0 else float("inf")
    savings = (
        (1 - compressed_size / uncompressed_size) * 100 if uncompressed_size > 0 else 0
    )

    return {
        "uncompressed_bytes": uncompressed_size,
        "compressed_bytes": compressed_size,
        "uncompressed_human": format_size(uncompressed_size),
        "compressed_human": format_size(compressed_size),
        "compression_ratio": round(ratio, 2),
        "space_savings_percent": round(savings, 2),
    }
