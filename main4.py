import argparse
import sys
from typing import List, Tuple, Set, Dict
import math


class GrayCode:
    """Генератор кодов Грея"""

    @staticmethod
    def generate(n: int) -> List[int]:
        """Генерирует коды Грея для n бит"""
        if n == 0:
            return [0]

        result = []
        for i in range(1 << n):
            result.append(i ^ (i >> 1))
        return result

    @staticmethod
    def to_binary(gray: int, n: int) -> int:
        """Преобразует код Грея в двоичный"""
        binary = gray
        gray >>= 1
        while gray:
            binary ^= gray
            gray >>= 1
        return binary


class GrayMatrix:
    """Матрица Грея для булевой функции n переменных"""

    def __init__(self, function_values: str):
        """
        Инициализация матрицы Грея
        function_values: строка из 0 и 1, например "10111100"
        """
        self.function_values = function_values
        self.n_vars = int(math.log2(len(function_values)))

        if 2 ** self.n_vars != len(function_values):
            raise ValueError(f"Длина вектора должна быть степенью 2")

        if self.n_vars < 2:
            raise ValueError("Минимум 2 переменные")

        # Разделяем переменные на строки и столбцы
        self.n_row_vars = self.n_vars // 2
        self.n_col_vars = self.n_vars - self.n_row_vars

        self.n_rows = 2 ** self.n_row_vars
        self.n_cols = 2 ** self.n_col_vars

        # Генерируем коды Грея
        self.row_gray_codes = GrayCode.generate(self.n_row_vars)
        self.col_gray_codes = GrayCode.generate(self.n_col_vars)

        # Создаём матрицу
        self.matrix = [[0] * self.n_cols for _ in range(self.n_rows)]

        # Маппинги
        self.coord_to_vars = {}
        self.coord_to_gray = {}  # (row_idx, col_idx) -> (row_gray, col_gray)

        # Заполняем матрицу
        for row_idx, row_gray in enumerate(self.row_gray_codes):
            for col_idx, col_gray in enumerate(self.col_gray_codes):
                row_binary = GrayCode.to_binary(row_gray, self.n_row_vars)
                col_binary = GrayCode.to_binary(col_gray, self.n_col_vars)

                binary_index = (row_binary << self.n_col_vars) | col_binary

                self.matrix[row_idx][col_idx] = int(function_values[binary_index])

                # Сохраняем маппинг на Gray-коды
                self.coord_to_gray[(row_idx, col_idx)] = (row_gray, col_gray)

                # Сохраняем маппинг на переменные
                vars_list = []
                for i in range(self.n_row_vars):
                    vars_list.append((row_binary >> (self.n_row_vars - 1 - i)) & 1)
                for i in range(self.n_col_vars):
                    vars_list.append((col_binary >> (self.n_col_vars - 1 - i)) & 1)

                self.coord_to_vars[(row_idx, col_idx)] = tuple(vars_list)

    def get_minterms(self) -> Set[Tuple[int, int]]:
        """Возвращает координаты всех единиц в матрице"""
        minterms = set()
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                if self.matrix[row][col] == 1:
                    minterms.add((row, col))
        return minterms

    def get_minterm_indices(self) -> List[int]:
        """Возвращает индексы минтермов"""
        indices = []
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                if self.matrix[row][col] == 1:
                    row_gray = self.row_gray_codes[row]
                    col_gray = self.col_gray_codes[col]
                    row_binary = GrayCode.to_binary(row_gray, self.n_row_vars)
                    col_binary = GrayCode.to_binary(col_gray, self.n_col_vars)
                    binary_index = (row_binary << self.n_col_vars) | col_binary
                    indices.append(binary_index)
        return sorted(indices)

    def print_matrix(self):
        """Печать матрицы Грея"""
        print(f"\nМатрица Грея ({self.n_vars} переменных):")

        col_header = " " * (self.n_row_vars * 2 + 3) + "|"
        for col_idx in range(self.n_cols):
            col_gray = self.col_gray_codes[col_idx]
            col_binary = GrayCode.to_binary(col_gray, self.n_col_vars)
            col_str = format(col_binary, f'0{self.n_col_vars}b')
            col_header += f" {col_str}"
        print(col_header)
        print("-" * len(col_header))

        for row_idx in range(self.n_rows):
            row_gray = self.row_gray_codes[row_idx]
            row_binary = GrayCode.to_binary(row_gray, self.n_row_vars)
            row_str = format(row_binary, f'0{self.n_row_vars}b')

            line = f"{row_str} |"
            for col_idx in range(self.n_cols):
                line += f"  {self.matrix[row_idx][col_idx]}"
            print(line)
        print()


class DNFMinimizer:
    """Минимизатор ДНФ - поиск булевых кубов"""

    def __init__(self, gray_matrix: GrayMatrix):
        self.matrix = gray_matrix
        self.minterms = gray_matrix.get_minterms()
        self.covered = set()
        self.solution = []

    def find_all_cubes_for_minterm(self, minterm: Tuple[int, int]) -> List[Set[Tuple[int, int]]]:
        """
        Находит все возможные кубы, содержащие данный минтерм.
        Куб строится путём фиксирования подмножества битов Gray-кодов.
        """
        row_idx, col_idx = minterm
        row_gray, col_gray = self.matrix.coord_to_gray[minterm]

        cubes = []

        # Перебираем все возможные маски фиксированных битов
        # Для row-переменных: какие биты фиксируем
        for row_mask in range(1 << self.matrix.n_row_vars):
            # Для col-переменных: какие биты фиксируем
            for col_mask in range(1 << self.matrix.n_col_vars):
                cube = self._build_cube(row_gray, col_gray, row_mask, col_mask)

                # Проверяем: все ли точки куба - минтермы
                if cube and all(point in self.minterms for point in cube):
                    cubes.append(cube)

        return cubes

    def _build_cube(self, base_row_gray: int, base_col_gray: int,
                    row_mask: int, col_mask: int) -> Set[Tuple[int, int]]:
        """
        Строит куб по базовому Gray-коду и маскам фиксированных битов.
        row_mask: биты, которые фиксируются в row_gray (1 = фиксирован, 0 = свободен)
        col_mask: биты, которые фиксируются в col_gray
        """
        cube = set()

        # Перебираем все возможные значения свободных битов
        n_free_row_bits = bin(~row_mask & ((1 << self.matrix.n_row_vars) - 1)).count('1')
        n_free_col_bits = bin(~col_mask & ((1 << self.matrix.n_col_vars) - 1)).count('1')

        for row_var in range(1 << n_free_row_bits):
            for col_var in range(1 << n_free_col_bits):
                # Строим Gray-код с учётом маски
                row_gray = self._apply_mask(base_row_gray, row_mask, row_var, self.matrix.n_row_vars)
                col_gray = self._apply_mask(base_col_gray, col_mask, col_var, self.matrix.n_col_vars)

                # Находим соответствующие индексы в матрице
                try:
                    row_idx = self.matrix.row_gray_codes.index(row_gray)
                    col_idx = self.matrix.col_gray_codes.index(col_gray)
                    cube.add((row_idx, col_idx))
                except ValueError:
                    # Такого Gray-кода нет в нашей последовательности
                    continue

        return cube

    def _apply_mask(self, base_gray: int, mask: int, free_vals: int, n_bits: int) -> int:
        """
        Применяет маску к Gray-коду:
        - фиксированные биты (mask=1) берутся из base_gray
        - свободные биты (mask=0) берутся из free_vals
        """
        result = 0
        free_bit_idx = 0

        for bit_pos in range(n_bits):
            bit_mask = 1 << bit_pos

            if mask & bit_mask:
                # Фиксированный бит - берём из base
                if base_gray & bit_mask:
                    result |= bit_mask
            else:
                # Свободный бит - берём из free_vals
                if free_vals & (1 << free_bit_idx):
                    result |= bit_mask
                free_bit_idx += 1

        return result

    def find_max_cube(self, uncovered: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """
        Находит куб, покрывающий максимальное количество непокрытых точек.
        Это жадный выбор для приближённой минимизации.
        """
        best_cube = None
        best_uncovered_count = 0

        # Проверяем кубы для каждой непокрытой точки
        for minterm in uncovered:
            cubes = self.find_all_cubes_for_minterm(minterm)

            for cube in cubes:
                uncovered_count = len(cube & uncovered)

                # Выбираем куб с максимальным покрытием непокрытых точек
                # При равенстве - выбираем больший куб
                if (uncovered_count > best_uncovered_count or
                        (uncovered_count == best_uncovered_count and
                         best_cube is not None and len(cube) > len(best_cube))):
                    best_cube = cube
                    best_uncovered_count = uncovered_count

        return best_cube if best_cube else set()

    def minimize(self) -> List[Set[Tuple[int, int]]]:
        """Основной алгоритм минимизации"""
        uncovered = self.minterms.copy()

        while uncovered:
            # Находим куб с максимальным покрытием
            cube = self.find_max_cube(uncovered)

            if not cube:
                # Не удалось найти куб - берём одиночную точку
                point = uncovered.pop()
                cube = {point}

            # Добавляем куб в решение
            self.solution.append(cube)

            # Отмечаем покрытые точки
            uncovered -= cube
            self.covered.update(cube)

        return self.solution

    def cube_to_term(self, cube: Set[Tuple[int, int]]) -> str:
        """
        Преобразует куб в литерал ДНФ.
        Куб определяется фиксированными битами Gray-кодов.
        """
        # Получаем значения переменных для всех точек куба
        vars_values = [self.matrix.coord_to_vars[point] for point in cube]

        # Определяем фиксированные переменные
        n_vars = len(vars_values[0])
        var_names = ['x', 'y', 'z', 'w', 'v', 'u'][:n_vars]

        term = []
        for i in range(n_vars):
            values = set(v[i] for v in vars_values)
            if len(values) == 1:
                val = list(values)[0]
                if val == 1:
                    term.append(var_names[i])
                else:
                    term.append(var_names[i] + "'")

        return ''.join(term) if term else '1'

    def get_dnf(self) -> str:
        """Возвращает ДНФ в виде строки"""
        if not self.solution:
            return '0'

        terms = [self.cube_to_term(cube) for cube in self.solution]
        return ' + '.join(terms)


def parse_input(input_file: str) -> str:
    """Парсит входной файл с вектором значений функции"""
    try:
        with open(input_file, 'r') as f:
            content = f.read().strip()

            content = content.replace(' ', '').replace(',', '').replace(';', '').replace('\n', '')

            if not all(c in '01' for c in content):
                raise ValueError("Вектор должен содержать только 0 и 1")

            return content

    except FileNotFoundError:
        print(f"Ошибка: файл '{input_file}' не найден", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Минимизация ДНФ через матрицу Грея (корректный алгоритм)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python dnf_minimizer.py input.txt
  python dnf_minimizer.py input.txt -o output.txt
  python dnf_minimizer.py input.txt -v

Формат входного файла:
  Вектор из 0 и 1 (длина должна быть степенью 2)
  Например: 10111100

  Это означает:
  - 3 переменные (2^3 = 8 значений)
  - Минтермы: 0, 2, 3, 4, 5, 6
        """
    )
    parser.add_argument('input', help='Входной файл с вектором значений функции')
    parser.add_argument('-o', '--output', help='Выходной файл')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Подробный вывод')

    args = parser.parse_args()

    # Читаем входные данные
    function_vector = parse_input(args.input)

    # Создаём матрицу Грея
    try:
        gray_matrix = GrayMatrix(function_vector)
    except ValueError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)

    # Формируем вывод
    output_lines = []

    output_lines.append(f"Вектор значений: {function_vector}")
    output_lines.append(f"Количество переменных: {gray_matrix.n_vars}")

    minterm_indices = gray_matrix.get_minterm_indices()
    output_lines.append(f"Минтермы: {', '.join(map(str, minterm_indices))}")
    output_lines.append("")

    if args.verbose:
        gray_matrix.print_matrix()

    # Минимизируем ДНФ
    minimizer = DNFMinimizer(gray_matrix)
    solution = minimizer.minimize()

    if args.verbose:
        output_lines.append("Найденные кубы:")
        for i, cube in enumerate(solution, 1):
            term = minimizer.cube_to_term(cube)
            # Показываем индексы минтермов в двоичной нумерации
            indices = []
            for point in sorted(cube):
                row_gray, col_gray = gray_matrix.coord_to_gray[point]
                row_bin = GrayCode.to_binary(row_gray, gray_matrix.n_row_vars)
                col_bin = GrayCode.to_binary(col_gray, gray_matrix.n_col_vars)
                idx = (row_bin << gray_matrix.n_col_vars) | col_bin
                indices.append(idx)
            indices_str = ', '.join(map(str, sorted(indices)))
            output_lines.append(f"  {i}. {term:15} покрывает минтермы: {indices_str}")
        output_lines.append("")

    dnf = minimizer.get_dnf()
    output_lines.append(f"ДНФ: {dnf}")
    output_lines.append(f"Количество термов: {len(solution)}")

    # Выводим результат
    result = '\n'.join(output_lines)

    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(result + '\n')
            print(f"Результат записан в '{args.output}'")
        except Exception as e:
            print(f"Ошибка при записи в файл: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(result)


if __name__ == '__main__':
    main()