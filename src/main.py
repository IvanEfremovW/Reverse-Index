import argparse

from reverse_index.preprocessing import preprocess_documents
from reverse_index.inverted_index import InvertedIndex
from reverse_index.compression import CompressedInvertedIndex
from reverse_index.storage import save_index, load_index
from reverse_index.search import BooleanSearcher
from reverse_index.benchmark import BenchmarkRunner


def build_index_cli(args):
    """CLI: построение индекса."""
    print(f"Обработка {args.input}...")
    docs = preprocess_documents(args.input)
    print(f"Загружено {len(docs)} документов")

    print("Построение индекса...")
    index = InvertedIndex(compressed=args.compress)
    index.build(docs)

    if args.compress:
        index = CompressedInvertedIndex(index)
        print("Применение сжатия (дельта + гамма)...")

    print("💾 Сохранение...")
    result = save_index(index, args.output, compressed=args.compress)

    print("\nГотово!")
    print(f"   - Терминов: {len(index.vocabulary):,}")
    print(
        f"   - Размер на диске: {result.get('compressed_size_human' if args.compress else 'uncompressed_size_human', 'N/A')}"
    )


def search_cli(args):
    """CLI: выполнение поиска."""
    print(f"Загрузка индекса из {args.index_dir}...")
    index = load_index(args.index_dir, compressed=args.compress)
    searcher = BooleanSearcher(index)

    print(f"Поиск: '{args.query}'")
    results = searcher.search_with_metadata(args.query)

    print(f"\nНайдено {len(results)} документов:")
    for i, meta in enumerate(results[: args.limit], 1):
        channel = meta.get("channel", "N/A")
        date = meta.get("date", "N/A")
        text = (
            meta.get("original_text", "")[:100] + "..."
            if len(meta.get("original_text", "")) > 100
            else meta.get("original_text", "")
        )
        print(f"{i}. [{date}] {channel}: {text}")


def benchmark_cli(args):
    """CLI: запуск бенчмарков."""
    runner = BenchmarkRunner(args.input, output_dir=args.output)
    runner.run_full_benchmark(query=args.query, doc_limit=args.limit)


def main():
    parser = argparse.ArgumentParser(
        description="Inverted Index with Elias Gamma Compression"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build
    build_p = subparsers.add_parser("build", help="Построить индекс")
    build_p.add_argument("-i", "--input", required=True, help="Путь к CSV с данными")
    build_p.add_argument(
        "-o", "--output", default="index_output", help="Директория для сохранения"
    )
    build_p.add_argument(
        "-c", "--compress", action="store_true", help="Применить сжатие"
    )
    build_p.set_defaults(func=build_index_cli)

    # search
    search_p = subparsers.add_parser("search", help="Выполнить поиск")
    search_p.add_argument(
        "-d", "--index-dir", required=True, help="Директория с индексом"
    )
    search_p.add_argument("-q", "--query", required=True, help="Поисковый запрос")
    search_p.add_argument("-c", "--compress", action="store_true", help="Индекс сжатый")
    search_p.add_argument(
        "-l", "--limit", type=int, default=10, help="Макс. результатов"
    )
    search_p.set_defaults(func=search_cli)

    # benchmark
    bench_p = subparsers.add_parser("benchmark", help="Запустить бенчмарки")
    bench_p.add_argument("-i", "--input", required=True, help="Путь к CSV с данными")
    bench_p.add_argument(
        "-o", "--output", default="benchmark_results", help="Директория для отчётов"
    )
    bench_p.add_argument(
        "-q", "--query", default="ректор спбгу мгу", help="Тестовый запрос"
    )
    bench_p.add_argument("-l", "--limit", type=int, help="Ограничить кол-во документов")
    bench_p.set_defaults(func=benchmark_cli)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
