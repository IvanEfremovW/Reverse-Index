import re
from typing import List, Tuple, Dict, Any

from .preprocessing import clean_text
from .inverted_index import InvertedIndex
from .compression import CompressedInvertedIndex


class QueryParser:
    """
    Парсер поисковых запросов.

    Поддерживает:
    - Простые термины: "ректор"
    - Логическое И (пробел или AND): "ректор мгу"
    - Логическое ИЛИ (слэш или OR): "спбгу/мгу"
    - Группировка: "ректор (спбгу/мгу)"
    """

    # Токенизация запроса с учётом операторов
    TOKEN_PATTERN = re.compile(r'(\(|\)|AND|OR|/|\+|-|"[^"]+"|\S+)')

    @classmethod
    def parse(cls, query: str) -> List[Tuple]:
        """
        Парсинг запроса в список (оператор, термин).

        Returns:
            Список кортежей: [('AND', 'ректор'), ('OR_GROUP', ['спбгу', 'мгу'])]
        """
        tokens = cls.TOKEN_PATTERN.findall(query.lower())
        result = []
        i = 0

        while i < len(tokens):
            token = tokens[i].strip('"')

            if token in ("and", "&&", ""):
                i += 1
                continue  # AND по умолчанию

            elif token == "or" or token == "/":
                # Собираем группу OR
                or_terms = []
                i += 1
                while i < len(tokens):
                    next_tok = tokens[i].strip('"')
                    if next_tok in (")", "and", "or"):
                        break
                    if next_tok and next_tok not in ("/", "|"):
                        or_terms.append(next_tok)
                    i += 1
                if or_terms:
                    result.append(("OR_GROUP", or_terms))
                continue

            elif token == "(":
                # Рекурсивный парсинг группы (упрощённо)
                i += 1
                group_terms = []
                while i < len(tokens) and tokens[i] != ")":
                    t = tokens[i].strip('"')
                    if t and t not in ("/", "or", "and"):
                        group_terms.append(t)
                    elif t == "/":
                        # Вложенная OR-логика
                        pass
                    i += 1
                if group_terms:
                    result.append(("OR_GROUP", group_terms))
                i += 1  # Пропускаем ')'
                continue

            elif token and token not in (")", "-", "+"):
                result.append(("AND", token))

            i += 1

        return result


class BooleanSearcher:
    """
    Булев поисковик по инвертированному индексу.

    Поддерживает операции AND, OR через слияние отсортированных списков.
    """

    def __init__(self, index: InvertedIndex | CompressedInvertedIndex):
        self.index = index

    def _get_posting_list(self, term: str) -> List[int]:
        """Получение posting list с обработкой отсутствующих терминов."""
        plist = self.index.get_posting_list(term)
        return plist if plist is not None else []

    @staticmethod
    def _intersect(lists: List[List[int]]) -> List[int]:
        """Пересечение отсортированных списков (операция AND)."""
        if not lists:
            return []
        if len(lists) == 1:
            return lists[0]

        result = lists[0]
        for other in lists[1:]:
            if not result or not other:
                return []
            # Двухуказательное слияние
            i = j = 0
            new_result = []
            while i < len(result) and j < len(other):
                if result[i] == other[j]:
                    new_result.append(result[i])
                    i += 1
                    j += 1
                elif result[i] < other[j]:
                    i += 1
                else:
                    j += 1
            result = new_result
        return result

    @staticmethod
    def _union(lists: List[List[int]]) -> List[int]:
        """Объединение отсортированных списков (операция OR)."""
        if not lists:
            return []
        if len(lists) == 1:
            return lists[0]

        # Слияние с удалением дубликатов
        import heapq

        return list(heapq.merge(*lists))

    def search(self, query: str) -> List[int]:
        """
        Выполнение поискового запроса.

        Логика для "Ректор СПбГУ/МГУ":
        - "ректор" AND ("спбгу" OR "мгу")

        Returns:
            Отсортированный список doc_id, удовлетворяющих запросу
        """
        # Предобработка запроса
        cleaned = clean_text(query)
        parsed = QueryParser.parse(cleaned)

        if not parsed:
            return []

        and_lists = []

        for op, value in parsed:
            if op == "AND":
                plist = self._get_posting_list(value)
                and_lists.append(plist)
            elif op == "OR_GROUP":
                or_plists = [self._get_posting_list(t) for t in value]
                or_result = self._union(or_plists)
                and_lists.append(or_result)

        return self._intersect(and_lists)

    def search_with_metadata(self, query: str) -> List[Dict]:
        """
        Поиск с возвратом метаданных документов.

        Returns:
            Список метаданных найденных документов
        """
        doc_ids = self.search(query)
        return [
            self.index.doc_metadata.get(doc_id, {"doc_id": doc_id})
            for doc_id in doc_ids
        ]

    def get_query_stats(self, query: str) -> Dict[str, Any]:
        """Статистика по запросу (для отладки и бенчмаркинга)."""
        cleaned = clean_text(query)
        parsed = QueryParser.parse(cleaned)

        stats = {
            "parsed_query": parsed,
            "terms_found": {},
            "posting_list_sizes": {},
        }

        for op, value in parsed:
            if op == "AND":
                plist = self._get_posting_list(value)
                stats["terms_found"][value] = value in self.index.vocabulary
                stats["posting_list_sizes"][value] = len(plist)
            elif op == "OR_GROUP":
                for term in value:
                    plist = self._get_posting_list(term)
                    stats["terms_found"][term] = term in self.index.vocabulary
                    stats["posting_list_sizes"][term] = len(plist)

        return stats
