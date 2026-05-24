from typing import List, Optional, Dict
from bitarray import bitarray


class BitWriter:
    """Побитовая запись в битовый массив."""

    def __init__(self):
        self.bits = bitarray()

    def write_bits(self, value: int, length: int):
        """Запись 'length' бит числа 'value' (MSB first)."""
        for i in range(length - 1, -1, -1):
            self.bits.append((value >> i) & 1)

    def to_bytes(self) -> bytes:
        """Конвертация в bytes с дополнением до байта."""
        # Дополняем до кратности 8
        padding = (8 - len(self.bits) % 8) % 8
        self.bits.extend([0] * padding)
        # Сохраняем информацию о паддинге в первых 3 битах
        header = bitarray(f"{padding:03b}")
        return (header + self.bits).tobytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> tuple["BitWriter", int]:
        """Восстановление из bytes. Возвращает (BitWriter, padding)."""
        bits = bitarray()
        bits.frombytes(data)
        padding = int(bits[:3].to01(), 2)
        bits = bits[3:]  # Убираем заголовок
        writer = cls()
        writer.bits = bits
        return writer, padding


class EliasGammaCoder:
    """
    Гамма-кодирование Элиаса для положительных целых чисел (x >= 1).

    Алгоритм:
    1. k = floor(log2(x))
    2. Записать k нулей, затем 1, затем (k) бит двоичного представления x без старшей 1
    """

    @staticmethod
    def encode(x: int) -> bitarray:
        if x < 1:
            raise ValueError(f"Elias gamma requires x >= 1, got {x}")

        if x == 1:
            return bitarray("1")

        k = x.bit_length() - 1  # floor(log2(x))
        result = bitarray()

        # Унарная часть: k нулей + 1
        result.extend([0] * k)
        result.append(1)

        # Бинарная часть: младшие k бит числа x
        for i in range(k - 1, -1, -1):
            result.append((x >> i) & 1)

        return result

    @staticmethod
    def decode(bits: bitarray, pos: int) -> tuple[int, int]:
        """
        Декодирование одного числа из битового массива.

        Returns:
            (decoded_value, new_position)
        """
        # Считаем количество нулей до первой единицы
        k = 0
        while pos < len(bits) and bits[pos] == 0:
            k += 1
            pos += 1

        if pos >= len(bits) or bits[pos] != 1:
            raise ValueError("Invalid gamma code: missing separator bit")
        pos += 1  # Пропускаем разделительную 1

        # Читаем k бит для восстановления значения
        value = 1  # Старшая единица подразумевается
        for i in range(k):
            if pos + i < len(bits):
                value = (value << 1) | bits[pos + i]
            else:
                raise ValueError("Invalid gamma code: unexpected end of bits")

        return value, pos + k


class DeltaGammaCompressor:
    """
    Компрессор для posting lists: дельта-кодирование + гамма-кодирование.

    Вход: отсортированный список уникальных doc_id [d1, d2, d3, ...]
    Выход: байтовое представление сжатых дельт
    """

    @staticmethod
    def encode(posting_list: List[int]) -> bytes:
        """
        Кодирование posting list.

        Args:
            posting_list: Отсортированный список уникальных doc_id (>= 0)

        Returns:
            Сжатые данные в виде bytes
        """
        if not posting_list:
            # Специальный маркер для пустого списка
            return b"\x00\x00\x00\x00"  # 4 нулевых байта

        # Дельта-кодирование
        deltas = []
        prev = 0
        for doc_id in posting_list:
            delta = (
                doc_id - prev + 1
            )  # +1 чтобы дельты были >= 1 (требование гамма-кода)
            deltas.append(delta)
            prev = doc_id

        # Гамма-кодирование каждой дельты
        writer = BitWriter()
        for delta in deltas:
            encoded = EliasGammaCoder.encode(delta)
            writer.bits.extend(encoded)

        # Сохраняем длину исходного списка для декодирования
        length_bytes = len(posting_list).to_bytes(4, byteorder="big")
        return length_bytes + writer.to_bytes()

    @staticmethod
    def decode(data: bytes) -> List[int]:
        """
        Декодирование posting list.

        Args:
            data: Сжатые данные от encode()

        Returns:
            Восстановленный список doc_id
        """
        if len(data) < 4:
            return []

        # Читаем длину списка
        length = int.from_bytes(data[:4], byteorder="big")
        if length == 0:
            return []

        # Восстанавливаем битовый поток
        reader, padding = BitWriter.from_bytes(data[4:])
        bits = reader.bits

        # Декодируем дельты
        deltas = []
        pos = 0
        for _ in range(length):
            delta, pos = EliasGammaCoder.decode(bits, pos)
            deltas.append(delta)

        # Обратное дельта-кодирование
        result = []
        prev = 0
        for delta in deltas:
            doc_id = prev + delta - 1  # Компенсируем +1 из encode
            result.append(doc_id)
            prev = doc_id

        return result


class CompressedInvertedIndex:
    """
    Обёртка над InvertedIndex для работы со сжатыми posting lists.

    Posting lists хранятся в сжатом виде и распаковываются по требованию.
    """

    def __init__(self, base_index):
        """
        Args:
            base_index: Экземпляр InvertedIndex с построенным индексом
        """
        self.vocabulary = base_index.vocabulary.copy()
        self.doc_metadata = base_index.doc_metadata.copy()
        self._compressed_plists: Dict[str, bytes] = {}

        # Сжимаем все posting lists
        for term, plist_idx in base_index.vocabulary.items():
            posting_list = base_index.posting_lists[plist_idx]
            self._compressed_plists[term] = DeltaGammaCompressor.encode(posting_list)

    def get_posting_list(self, term: str) -> Optional[List[int]]:
        """Получение и распаковка posting list для термина."""
        if term not in self._compressed_plists:
            return None
        return DeltaGammaCompressor.decode(self._compressed_plists[term])

    def get_compressed_size(self, term: str) -> int:
        """Размер сжатого представления термина в байтах."""
        if term not in self._compressed_plists:
            return 0
        return len(self._compressed_plists[term])

    def get_terms(self) -> List[str]:
        return list(self.vocabulary.keys())

    def get_doc_count(self) -> int:
        return len(self.doc_metadata)

    def to_dict(self) -> Dict:
        """Сериализация для сохранения."""
        return {
            "vocabulary": self.vocabulary,
            "compressed_plists": self._compressed_plists,
            "doc_metadata": self.doc_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CompressedInvertedIndex":
        """Десериализация."""
        obj = cls.__new__(cls)
        obj.vocabulary = data["vocabulary"]
        obj._compressed_plists = data["compressed_plists"]
        obj.doc_metadata = data.get("doc_metadata", {})
        return obj
