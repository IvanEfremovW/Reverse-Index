from typing import Dict, List, Optional, Iterable, Any


class InvertedIndex:
    """
    Инвертированный индекс с поддержкой опционального сжатия.

    Posting list хранится как отсортированный список уникальных doc_id.
    """

    def __init__(self, compressed: bool = False):
        """
        Args:
            compressed: Если True, posting lists будут сжиматься при сохранении
        """
        self.compressed = compressed
        self.vocabulary: Dict[str, int] = {}  # term -> index in posting_lists
        self.posting_lists: List[List[int]] = []  # List of sorted doc_id lists
        self.doc_metadata: Dict[int, Dict] = {}  # doc_id -> metadata

    def add_document(
        self, doc_id: int, tokens: List[str], metadata: Optional[Dict] = None
    ):
        """
        Добавление одного документа в индекс.

        Args:
            doc_id: Уникальный числовой идентификатор документа
            tokens: Список токенов документа
            metadata: Дополнительные метаданные для хранения
        """
        if metadata:
            self.doc_metadata[doc_id] = metadata

        # Группируем термины по документу (убираем дубликаты внутри документа)
        unique_terms = set(tokens)

        for term in unique_terms:
            if term not in self.vocabulary:
                # Создаём новый posting list
                self.vocabulary[term] = len(self.posting_lists)
                self.posting_lists.append([doc_id])
            else:
                # Добавляем doc_id в существующий список
                plist_idx = self.vocabulary[term]
                # Поддерживаем сортировку: добавляем только если doc_id больше последнего
                if (
                    not self.posting_lists[plist_idx]
                    or self.posting_lists[plist_idx][-1] < doc_id
                ):
                    self.posting_lists[plist_idx].append(doc_id)

    def build(self, documents: Iterable[Dict[str, Any]]):
        """
        Построение индекса из итератора документов.

        Args:
            documents: Итератор с ключами 'doc_id', 'tokens', опционально 'metadata'
        """
        for doc in documents:
            self.add_document(
                doc_id=doc["doc_id"],
                tokens=doc["tokens"],
                metadata={
                    k: v for k, v in doc.items() if k not in ("doc_id", "tokens")
                },
            )

        # Финальная сортировка и дедупликация всех posting lists
        for i in range(len(self.posting_lists)):
            self.posting_lists[i] = sorted(set(self.posting_lists[i]))

    def get_posting_list(self, term: str) -> Optional[List[int]]:
        """Получение posting list для термина."""
        if term not in self.vocabulary:
            return None
        idx = self.vocabulary[term]
        return self.posting_lists[idx].copy()  # Возвращаем копию для безопасности

    def get_terms(self) -> List[str]:
        """Список всех терминов в словаре."""
        return list(self.vocabulary.keys())

    def get_doc_count(self) -> int:
        """Общее количество проиндексированных документов."""
        return len(self.doc_metadata)

    def to_dict(self) -> Dict:
        """Сериализация индекса в словарь (для pickle/json)."""
        return {
            "compressed": self.compressed,
            "vocabulary": self.vocabulary,
            "posting_lists": self.posting_lists,
            "doc_metadata": self.doc_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "InvertedIndex":
        """Десериализация индекса из словаря."""
        idx = cls(compressed=data.get("compressed", False))
        idx.vocabulary = data["vocabulary"]
        idx.posting_lists = data["posting_lists"]
        idx.doc_metadata = data.get("doc_metadata", {})
        return idx

    def __len__(self) -> int:
        """Размер словаря (количество уникальных терминов)."""
        return len(self.vocabulary)
