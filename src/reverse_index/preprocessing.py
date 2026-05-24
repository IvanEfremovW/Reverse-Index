import re
import csv
from typing import Iterator, Dict, List, Any
from pathlib import Path
import pymorphy3

import nltk
from nltk.corpus import stopwords

MORPHY = pymorphy3.MorphAnalyzer()
USE_MORPHY = True

# Загрузка стоп-слов при первом импорте
try:
    STOPWORDS_RU = set(stopwords.words("russian"))
    STOPWORDS_EN = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOPWORDS_RU = set(stopwords.words("russian"))
    STOPWORDS_EN = set(stopwords.words("english"))

STOPWORDS = STOPWORDS_RU | STOPWORDS_EN

# Регулярки для очистки
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+", re.IGNORECASE)
HASHTAG_RE = re.compile(r"#\w+", re.IGNORECASE)
NON_LETTER_RE = re.compile(r"[^а-яА-ЯёЁa-zA-Z\s]")


def normalize_text(text: str) -> str:
    """Приведение текста к нижнему регистру и удаление лишних пробелов."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_text(text: str) -> str:
    """Удаление URL, упоминаний, хештегов, пунктуации и цифр."""
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = NON_LETTER_RE.sub(" ", text)
    return normalize_text(text)


def tokenize(text: str) -> List[str]:
    """
    Токенизация текста

    Args:
        text: Исходный текст

    Returns:
        Список токенов (лемм или слов)
    """
    tokens = text.split()
    result = []

    for token in tokens:
        if len(token) < 2 or token in STOPWORDS:
            continue

        if USE_MORPHY:
            token = MORPHY.parse(token)[0].normal_form

        result.append(token)

    return result


def load_telegram_csv(filepath: Path | str) -> Iterator[Dict[str, Any]]:
    """
    Загрузка данных из CSV в формате Telegram export.

    Expected columns: Дата,Канал,Текст,Просмотры,Запрос,Ссылка

    Yields:
        Dict with doc_id, channel, text, tokens, metadata
    """
    filepath = Path(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")

        for idx, row in enumerate(reader):
            raw_text = row.get("Текст", "")
            if not raw_text:
                continue

            cleaned = clean_text(raw_text)
            tokens = tokenize(cleaned)

            yield {
                "doc_id": idx,  # Уникальный числовой идентификатор
                "channel": row.get("Канал", ""),
                "date": row.get("Дата", ""),
                "views": int(row.get("Просмотры", 0) or 0),
                "query_tag": row.get("Запрос", ""),
                "url": row.get("Ссылка", ""),
                "original_text": raw_text,
                "tokens": tokens,
            }


def filter_by_document_frequency(
    documents: List[Dict[str, Any]], min_doc_freq: int = 3
) -> List[Dict[str, Any]]:
    """
    Удаляет термины, встречающиеся менее чем в min_doc_freq документах.

    Args:
        documents: Список документов с ключом 'tokens'
        min_doc_freq: Минимальная частота по документам

    Returns:
        Отфильтрованный список документов (in-place модификация)
    """
    from collections import Counter

    # Подсчёт: в скольких документах встречается каждый термин
    term_doc_freq = Counter(term for doc in documents for term in set(doc["tokens"]))

    # Фильтрация токенов в каждом документе
    for doc in documents:
        doc["tokens"] = [t for t in doc["tokens"] if term_doc_freq[t] >= min_doc_freq]

    return documents


def preprocess_documents(
    source: Path | str,
    min_tokens: int = 1,
    min_doc_freq: int = 3,
) -> List[Dict[str, Any]]:
    """
    Полный пайплайн предобработки документов.

    Args:
        source: Путь к CSV файлу
        min_tokens: Минимальное количество токенов в документе
        min_doc_freq: Минимальная частота по документам

    Returns:
        Список обработанных документов
    """
    documents = []

    for doc in load_telegram_csv(source):
        if len(doc["tokens"]) >= min_tokens:
            documents.append(doc)

    # Фильтрация редких терминов
    if min_doc_freq > 1:
        documents = filter_by_document_frequency(documents, min_doc_freq)

    return documents
