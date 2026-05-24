import time
import statistics
import tracemalloc
import json
from pathlib import Path
from typing import List, Dict, Any
from contextlib import contextmanager

from .preprocessing import preprocess_documents
from .inverted_index import InvertedIndex
from .compression import CompressedInvertedIndex
from .storage import save_index, load_index, compare_index_sizes, format_size
from .search import BooleanSearcher


@contextmanager
def timer(label: str = ""):
    """Контекстный менеджер для замера времени выполнения."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"[TIMER] {label}: {elapsed:.3f} сек" if label else f"{elapsed:.3f} сек")


@contextmanager
def memory_tracker():
    """Контекстный менеджер для отслеживания пикового потребления RAM."""
    tracemalloc.start()
    snapshot_start = tracemalloc.get_traced_memory()
    yield
    snapshot_end = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Вычисляем разницу (дельту) потребления памяти внутри блока
    peak_increase = snapshot_end[1] - snapshot_start[1]
    return max(0, peak_increase)


class BenchmarkRunner:
    """
    Оркестратор бенчмарков для инвертированного индекса.
    """

    def __init__(self, data_path: str | Path, output_dir: str | Path = "results"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Any] = {}

    def load_documents(self, limit: int | None = None) -> List[Dict]:
        """Загрузка и предобработка документов."""
        docs = preprocess_documents(self.data_path)
        if limit:
            docs = docs[:limit]
        return docs

    def benchmark_indexing(
        self, documents: List[Dict], compressed: bool = False, runs: int = 3
    ) -> Dict[str, Any]:
        times = []

        # 1. Замер времени построения (CPU bound)
        for _ in range(runs):
            start = time.perf_counter()
            temp_idx = InvertedIndex(compressed=compressed)
            temp_idx.build(documents)
            if compressed:
                temp_idx = CompressedInvertedIndex(temp_idx)
            times.append(time.perf_counter() - start)

        # 2. Построение финального индекса для сохранения и замеров
        base_index = InvertedIndex(compressed=compressed)
        base_index.build(documents)

        # Расчет размера несжатых списков (до применения компрессора)
        pl_uncompressed_size = sum(len(pl) * 4 for pl in base_index.posting_lists)

        # Явная типизация для type checker
        index: InvertedIndex | CompressedInvertedIndex
        if compressed:
            index = CompressedInvertedIndex(base_index)
            pl_compressed_size = sum(len(v) for v in index._compressed_plists.values())
        else:
            index = base_index
            pl_compressed_size = 0

        # 3. Сохранение на диск (IO bound)
        save_path = (
            self.output_dir / f"{'compressed' if compressed else 'uncompressed'}"
        )
        with timer(f"Saving (compressed={compressed})"):
            save_result = save_index(index, save_path, compressed=compressed)

        file_size = save_result.get(
            f"{'compressed' if compressed else 'uncompressed'}_size", 0
        )

        return {
            "doc_count": len(documents),
            "vocabulary_size": len(index.vocabulary),
            "avg_build_time_sec": round(statistics.mean(times), 3),
            "file_size_bytes": file_size,
            "file_size_human": format_size(file_size),
            "pl_uncompressed_bytes": pl_uncompressed_size,
            "pl_compressed_bytes": pl_compressed_size
            if compressed
            else pl_uncompressed_size,
            "compressed": compressed,
        }

    def compare_plist_compression(
        self, uncomp_metrics: Dict, comp_metrics: Dict
    ) -> Dict[str, Any]:
        """
        Сравнение эффективности сжатия исключительно инвертированных списков.
        Это позволяет убрать влияние размера pickle-файла и словаря.
        """
        uncomp_size = uncomp_metrics["pl_uncompressed_bytes"]
        comp_size = comp_metrics["pl_compressed_bytes"]

        if comp_size == 0 or comp_size == uncomp_size:
            ratio = 1.0
            savings = 0.0
        else:
            ratio = uncomp_size / comp_size
            savings = (1 - comp_size / uncomp_size) * 100

        return {
            "uncompressed_pl_bytes": uncomp_size,
            "compressed_pl_bytes": comp_size,
            "uncompressed_pl_human": format_size(uncomp_size),
            "compressed_pl_human": format_size(comp_size),
            "plist_compression_ratio": round(ratio, 2),
            "plist_space_savings_percent": round(savings, 2),
        }

    def benchmark_search(
        self, query: str, compressed: bool = False, runs: int = 50
    ) -> Dict[str, Any]:
        """
        Бенчмарк поискового запроса.
        """
        index_dir = self.output_dir / ("compressed" if compressed else "uncompressed")
        index = load_index(index_dir, compressed=compressed)
        searcher = BooleanSearcher(index)

        latencies = []
        results_count = []

        for _ in range(runs):
            start = time.perf_counter()
            results = searcher.search(query)
            elapsed = (time.perf_counter() - start) * 1000  # мс

            latencies.append(elapsed)
            results_count.append(len(results))

        return {
            "query": query,
            "compressed": compressed,
            "results_count": {
                "mean": round(statistics.mean(results_count), 1),
                "median": statistics.median(results_count),
            },
            "latency_ms": {
                "mean": round(statistics.mean(latencies), 3),
                "median": round(statistics.median(latencies), 3),
                "p95": round(sorted(latencies)[int(len(latencies) * 0.95)], 3)
                if len(latencies) >= 20
                else None,
                "min": round(min(latencies), 3),
                "max": round(max(latencies), 3),
            },
            "query_stats": searcher.get_query_stats(query),
        }

    def run_full_benchmark(
        self, query: str = "ректор спбгу мгу", doc_limit: int | None = None
    ):
        """Полный цикл бенчмаркинга."""
        print("Загрузка документов...")
        documents = self.load_documents(limit=doc_limit)
        print(f"Загружено {len(documents)} документов")

        print("\nБенчмарк индексирования (несжатый)...")
        uncomp_metrics = self.benchmark_indexing(documents, compressed=False)

        print("\nБенчмарк индексирования (сжатый)...")
        comp_metrics = self.benchmark_indexing(documents, compressed=True)

        print("\n Сравнение размеров...")
        # Сравнение полных файлов (для справки)
        file_comparison = compare_index_sizes(
            self.output_dir / "uncompressed" / "index_uncompressed.pkl",
            self.output_dir / "compressed" / "index_compressed.pkl",
        )
        # Сравнение эффективности алгоритма (только списки)
        plist_comparison = self.compare_plist_compression(uncomp_metrics, comp_metrics)

        print(f"\nБенчмарк поиска: '{query}'")
        search_uncomp = self.benchmark_search(query, compressed=False)
        search_comp = self.benchmark_search(query, compressed=True)

        report = {
            "documents_indexed": len(documents),
            "indexing": {
                "uncompressed": uncomp_metrics,
                "compressed": comp_metrics,
            },
            "storage_comparison": file_comparison,  # Размер pickle файлов
            "plist_compression": plist_comparison,  # Эффективность алгоритма
            "search": {
                "query": query,
                "uncompressed": search_uncomp,
                "compressed": search_comp,
            },
            "summary": {
                "file_compression_ratio": file_comparison["compression_ratio"],
                "plist_compression_ratio": plist_comparison["plist_compression_ratio"],
                "plist_space_saved": plist_comparison["plist_space_savings_percent"],
                "search_overhead_ms": round(
                    search_comp["latency_ms"]["median"]
                    - search_uncomp["latency_ms"]["median"],
                    3,
                ),
            },
        }

        report_path = self.output_dir / "benchmark_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        self._print_summary(report)

        return report

    def _print_summary(self, report: Dict):
        """Красивый вывод сводки бенчмарка."""
        print("\n" + "=" * 60)
        print("СВОДНЫЙ ОТЧЁТ")
        print("=" * 60)

        s = report["summary"]
        print("\n️ Сжатие:")
        print(
            f"   - Полный файл (Pickle): {s['file_compression_ratio']}x (включает словарь)"
        )
        print(
            f"   - Только Posting Lists: {s['plist_compression_ratio']}x (экономия: {s['plist_space_saved']}%)"
        )
        print(f"   - Накладные расходы на поиск: {s['search_overhead_ms']} мс")

        print("\n Индекс:")
        print(f"   - Документов: {report['documents_indexed']:,}")
        print(
            f"   - Уникальных терминов: {report['indexing']['compressed']['vocabulary_size']:,}"
        )

        print(f"\n Поиск '{report['search']['query']}':")
        for mode in ["uncompressed", "compressed"]:
            lat = report["search"][mode]["latency_ms"]
            print(
                f"   - {mode:12s}: медиана {lat['median']} мс, "
                f"p95 {lat.get('p95', 'N/A')} мс, "
                f"результатов ~{report['search'][mode]['results_count']['median']:.0f}"
            )

        print(
            "\nДетальный отчёт сохранён в:",
            self.output_dir / "benchmark_report.json",
        )
        print("=" * 60)
