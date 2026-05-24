# diagnostic.py
import sys
import statistics

from reverse_index.inverted_index import InvertedIndex
from reverse_index.preprocessing import preprocess_documents

# Загрузка и построение индекса
docs = preprocess_documents("data/telegram_results.csv")
index = InvertedIndex(compressed=False)
index.build(docs)

# 1. Статистика posting lists
plist_lengths = [len(pl) for pl in index.posting_lists]
print(f"Всего терминов: {len(plist_lengths)}")
print(f"Средняя длина posting list: {statistics.mean(plist_lengths):.2f}")
print(f"Медиана длины: {statistics.median(plist_lengths):.2f}")
print(
    f"Доля списков длиной 1: {sum(1 for l in plist_lengths if l == 1) / len(plist_lengths) * 100:.1f}%"
)
print(
    f"Доля списков длиной >100: {sum(1 for l in plist_lengths if l > 100) / len(plist_lengths) * 100:.1f}%"
)

# 2. Анализ дельт
large_deltas = 0
total_deltas = 0
for pl in index.posting_lists:
    if len(pl) < 2:
        continue
    prev = 0
    for doc_id in pl:
        delta = doc_id - prev + 1
        total_deltas += 1
        if delta > 100:
            large_deltas += 1
        prev = doc_id

print(f"\nДоля больших дельт (>100): {large_deltas / total_deltas * 100:.1f}%")

# 3. Структура размера индекса
vocab_size = sys.getsizeof(index.vocabulary)
plist_raw_size = sum(sys.getsizeof(pl) for pl in index.posting_lists)
metadata_size = sys.getsizeof(index.doc_metadata)
total_uncompressed = vocab_size + plist_raw_size + metadata_size

print("\nРаспределение размера (несжатый индекс):")
print(f"  Словарь: {vocab_size / total_uncompressed * 100:.1f}%")
print(f"  Posting lists: {plist_raw_size / total_uncompressed * 100:.1f}%")
print(f"  Метаданные: {metadata_size / total_uncompressed * 100:.1f}%")
