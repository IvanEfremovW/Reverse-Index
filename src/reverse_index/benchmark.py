import time
import statistics
import tracemalloc
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
    yield
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return current, peak


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
        """
        Бенчмарк процесса индексирования.

        Returns:
            Метрики: время, размер, потребление памяти
        """
        times = []
        sizes = []

        for run in range(runs):
            with timer(f"Indexing (run {run + 1}, compressed={compressed})"):
                index = InvertedIndex(compressed=compressed)
                index.build(documents)

                if compressed:
                    index = CompressedInvertedIndex(index)

            # Сохранение и замер на диске
            with timer(f"Saving (compressed={compressed})"):
                save_result = save_index(
                    index,
                    self.output_dir
                    / f"{'compressed' if compressed else 'uncompressed'}",
                    compressed=compressed,
                )

            times.append(
                save_result.get(
                    f"{'compressed' if compressed else 'uncompressed'}_size", 0
                )
            )
            sizes.append(
                save_result.get(
                    f"{'compressed' if compressed else 'uncompressed'}_size", 0
                )
            )

        return {
            "doc_count": len(documents),
            "vocabulary_size": len(index.vocabulary),
            "avg_time_sec": round(statistics.mean(times) / 1000 if times else 0, 3),
            "avg_size_bytes": int(statistics.mean(sizes)) if sizes else 0,
            "avg_size_human": format_size(int(statistics.mean(sizes)))
            if sizes
            else "N/A",
            "compressed": compressed,
        }

    def benchmark_search(
        self, query: str, compressed: bool = False, runs: int = 50
    ) -> Dict[str, Any]:
        """
        Бенчмарк поискового запроса.

        Args:
            query: Поисковый запрос (например, "Ректор СПбГУ/МГУ")
            compressed: Использовать ли сжатый индекс
            runs: Количество прогонов для статистики

        Returns:
            Метрики поиска: латентность, количество результатов
        """
        # Загрузка индекса
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
        """
        Полный цикл бенчмаркинга.

        1. Индексирование (сжатое / несжатое)
        2. Сравнение размеров
        3. Поиск по тестовому запросу
        4. Формирование отчёта
        """
        print("Загрузка документов...")
        documents = self.load_documents(limit=doc_limit)
        print(f"Загружено {len(documents)} документов")

        # Бенчмарк индексирования
        print("\nБенчмарк индексирования (несжатый)...")
        uncomp_metrics = self.benchmark_indexing(documents, compressed=False)

        print("\nБенчмарк индексирования (сжатый)...")
        comp_metrics = self.benchmark_indexing(documents, compressed=True)

        # Сравнение размеров
        print("\nСравнение размеров...")
        size_comparison = compare_index_sizes(
            self.output_dir / "uncompressed" / "index_uncompressed.pkl",
            self.output_dir / "compressed" / "index_compressed.pkl",
        )

        # Бенчмарк поиска
        print(f"\nБенчмарк поиска: '{query}'")
        search_uncomp = self.benchmark_search(query, compressed=False)
        search_comp = self.benchmark_search(query, compressed=True)

        # Формирование отчёта
        report = {
            "documents_indexed": len(documents),
            "indexing": {
                "uncompressed": uncomp_metrics,
                "compressed": comp_metrics,
            },
            "storage_comparison": size_comparison,
            "search": {
                "query": query,
                "uncompressed": search_uncomp,
                "compressed": search_comp,
            },
            "summary": {
                "compression_ratio": size_comparison["compression_ratio"],
                "space_saved": size_comparison["space_savings_percent"],
                "search_overhead_ms": round(
                    search_comp["latency_ms"]["median"]
                    - search_uncomp["latency_ms"]["median"],
                    3,
                ),
            },
        }

        # Сохранение отчёта
        import json

        report_path = self.output_dir / "benchmark_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        # Печать сводки
        self._print_summary(report)

        return report

    def _print_summary(self, report: Dict):
        """Красивый вывод сводки бенчмарка."""
        print("\n" + "=" * 60)
        print("СВОДНЫЙ ОТЧЁТ")
        print("=" * 60)

        s = report["summary"]
        print("\nСжатие:")
        print(f"   - Коэффициент сжатия: {s['compression_ratio']}x")
        print(f"   - Экономия места: {s['space_saved']}%")
        print(f"   - Накладные расходы на поиск: {s['search_overhead_ms']} мс")

        print("\nИндекс:")
        print(f"   - Документов: {report['documents_indexed']:,}")
        print(
            f"   - Уникальных терминов: {report['indexing']['compressed']['vocabulary_size']:,}"
        )

        print(f"\nПоиск '{report['search']['query']}':")
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
