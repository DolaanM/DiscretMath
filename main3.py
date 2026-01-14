import argparse
import re
import sys
from itertools import product
from typing import List, Dict, Tuple


class BooleanParser:
    """Парсер и вычислитель булевых выражений"""

    # Символы операций
    OPERATORS = {
        '~': 'NOT', '!': 'NOT', '¬': 'NOT',
        '&': 'AND', '∧': 'AND',
        '|': 'OR', '∨': 'OR',
        '^': 'XOR', '⊕': 'XOR',
        '↓': 'NOR',
        '/': 'NAND', '|/': 'NAND',  # Штрих Шеффера
        '->': 'IMP', '→': 'IMP',
        '<-': 'RIMP', '←': 'RIMP',
        '<->': 'EQ', '↔': 'EQ', '~': 'EQV',  # Эквивалентность
    }

    def __init__(self, mode=1):
        """
        mode=1: NOT > AND > остальные слева направо
        mode=2: NOT > AND > (OR, XOR) > IMP > EQ, NAND/NOR раскрываются
        """
        self.mode = mode

    def tokenize(self, formula: str) -> List[str]:
        """Разбивает формулу на токены"""
        # Замена многосимвольных операторов
        formula = formula.replace('<->', '↔').replace('->', '→').replace('<-', '←')
        formula = formula.replace('|/', '/')  # NAND

        tokens = []
        i = 0
        while i < len(formula):
            if formula[i].isspace():
                i += 1
                continue

            # Многосимвольные операторы
            if i < len(formula) - 1:
                two_char = formula[i:i + 2]
                if two_char in ['<-', '->']:
                    tokens.append(two_char)
                    i += 2
                    continue
            if i < len(formula) - 2:
                three_char = formula[i:i + 3]
                if three_char == '<->':
                    tokens.append(three_char)
                    i += 3
                    continue

            # Однобуквенные переменные или операторы
            if formula[i].isalpha() or formula[i].isdigit() or formula[i] == '_':
                # Переменная (x1, x2, var_name и т.д.)
                j = i
                while j < len(formula) and (formula[j].isalnum() or formula[j] == '_'):
                    j += 1
                tokens.append(formula[i:j])
                i = j
            elif formula[i] in '()':
                tokens.append(formula[i])
                i += 1
            elif formula[i] in self.OPERATORS or formula[i] in '~!¬&∧|∨^⊕↓/→←↔':
                tokens.append(formula[i])
                i += 1
            else:
                i += 1

        return tokens

    def get_variables(self, formula: str) -> List[str]:
        """Извлекает все переменные из формулы"""
        tokens = self.tokenize(formula)
        variables = []
        for token in tokens:
            if token not in self.OPERATORS and token not in '()' and not token in ['~', '!', '¬', '&', '∧', '|', '∨',
                                                                                   '^', '⊕', '↓', '/', '→', '←', '↔']:
                if token not in variables and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token):
                    variables.append(token)

        # Сортировка переменных естественным образом (x1, x2, ..., x10)
        def natural_key(s):
            return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]

        return sorted(variables, key=natural_key)

    def to_postfix(self, tokens: List[str]) -> List[str]:
        """Преобразует инфиксную нотацию в постфиксную (обратная польская нотация)"""
        # Приоритеты операций
        if self.mode == 1:
            precedence = {
                '~': 4, '!': 4, '¬': 4,
                '&': 3, '∧': 3,
                '|': 2, '∨': 2, '^': 2, '⊕': 2, '↓': 2, '/': 2,
                '→': 2, '←': 2, '↔': 2,
            }
        else:  # mode == 2
            precedence = {
                '~': 6, '!': 6, '¬': 6,
                '&': 5, '∧': 5,
                '|': 4, '∨': 4, '^': 4, '⊕': 4,
                '→': 3, '←': 3,
                '↔': 2,
                '↓': 1, '/': 1,  # Будут раскрыты позже
            }

        output = []
        stack = []

        for token in tokens:
            if token not in self.OPERATORS and token not in ['~', '!', '¬', '&', '∧', '|', '∨', '^', '⊕', '↓', '/', '→',
                                                             '←', '↔', '(', ')']:
                # Переменная
                output.append(token)
            elif token in ['~', '!', '¬', '&', '∧', '|', '∨', '^', '⊕', '↓', '/', '→', '←', '↔']:
                # Оператор
                while (stack and stack[-1] != '(' and
                       stack[-1] in precedence and
                       precedence.get(stack[-1], 0) >= precedence.get(token, 0)):
                    output.append(stack.pop())
                stack.append(token)
            elif token == '(':
                stack.append(token)
            elif token == ')':
                while stack and stack[-1] != '(':
                    output.append(stack.pop())
                if stack:
                    stack.pop()  # Удаляем '('

        while stack:
            output.append(stack.pop())

        return output

    def evaluate(self, postfix: List[str], values: Dict[str, int]) -> int:
        """Вычисляет значение выражения в постфиксной нотации"""
        stack = []

        for token in postfix:
            if token in values:
                stack.append(values[token])
            elif token in ['~', '!', '¬']:
                # Унарная операция NOT
                if stack:
                    a = stack.pop()
                    stack.append(1 - a)
            elif token in ['&', '∧']:
                # AND
                if len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(a & b)
            elif token in ['|', '∨']:
                # OR
                if len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(a | b)
            elif token in ['^', '⊕']:
                # XOR
                if len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(a ^ b)
            elif token == '↓':
                # NOR
                if len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    if self.mode == 2:
                        # Раскрываем как ¬(a ∨ b)
                        stack.append(1 - (a | b))
                    else:
                        stack.append(1 - (a | b))
            elif token == '/':
                # NAND (Штрих Шеффера)
                if len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    if self.mode == 2:
                        # Раскрываем как ¬(a ∧ b)
                        stack.append(1 - (a & b))
                    else:
                        stack.append(1 - (a & b))
            elif token in ['→']:
                # Импликация: a → b = ¬a ∨ b
                if len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append((1 - a) | b)
            elif token in ['←']:
                # Обратная импликация: a ← b = a ∨ ¬b
                if len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(a | (1 - b))
            elif token in ['↔']:
                # Эквивалентность: a ↔ b = (a → b) ∧ (b → a)
                if len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(1 if a == b else 0)

        return stack[0] if stack else 0

    def compute_truth_table(self, formula: str) -> Tuple[List[str], List[List[int]]]:
        """Генерирует таблицу истинности"""
        # 1. Определить все переменные
        variables = self.get_variables(formula)

        if not variables:
            raise ValueError("В формуле не найдено переменных")

        # 2. Преобразовать в постфикс
        tokens = self.tokenize(formula)
        postfix = self.to_postfix(tokens)

        # 3. Сгенерировать 2^n наборов значений
        n = len(variables)
        rows = []

        for values_tuple in product([0, 1], repeat=n):
            values = dict(zip(variables, values_tuple))
            result = self.evaluate(postfix, values)
            rows.append(list(values_tuple) + [result])

        return variables, rows


def format_table(variables: List[str], rows: List[List[int]], cell_width: int = 4) -> str:
    """Форматирует таблицу истинности с псевдографикой"""
    headers = variables + ['f']
    n_cols = len(headers)

    # Псевдографические символы
    TL, TR, BL, BR = '┌', '┐', '└', '┘'
    H, V = '─', '│'
    TJ, BJ, LJ, RJ, CROSS = '┬', '┴', '├', '┤', '┼'

    # Ширина колонок
    col_widths = [max(len(h), cell_width) for h in headers]

    # Верхняя граница
    line = TL + TJ.join(H * (w + 2) for w in col_widths) + TR
    result = [line]

    # Заголовок
    header_line = V + V.join(f' {h:^{w}} ' for h, w in zip(headers, col_widths)) + V
    result.append(header_line)

    # Разделитель после заголовка
    line = LJ + CROSS.join(H * (w + 2) for w in col_widths) + RJ
    result.append(line)

    # Строки данных
    for row in rows:
        row_line = V + V.join(f' {val:^{w}} ' for val, w in zip(row, col_widths)) + V
        result.append(row_line)

    # Нижняя граница
    line = BL + BJ.join(H * (w + 2) for w in col_widths) + BR
    result.append(line)

    return '\n'.join(result)


def main():
    arg_parser = argparse.ArgumentParser(
        description='Генератор таблиц истинности для булевых функций',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Поддерживаемые операции:
  ~ ! ¬     - отрицание (NOT)
  & ∧       - конъюнкция (AND)
  | ∨       - дизъюнкция (OR)
  ^ ⊕       - исключающее ИЛИ (XOR)
  ↓         - стрелка Пирса (NOR)
  /         - штрих Шеффера (NAND)
  -> →      - импликация
  <- ←      - обратная импликация
  <-> ↔     - эквивалентность

Примеры использования:
  %(prog)s formula.txt output.txt
  %(prog)s formula.txt -c
  %(prog)s formula.txt output.txt --mode 2 --width 6

Алгоритм работы:
  1. Определяет все переменные в формуле
  2. Генерирует 2^n наборов значений
  3. Преобразует формулу в постфиксную нотацию (алгоритм Дейкстры)
  4. Подставляет значения и вычисляет результат
  5. Форматирует таблицу истинности с псевдографикой
        """
    )

    arg_parser.add_argument('input', help='Файл с булевой формулой')
    arg_parser.add_argument('output', nargs='?', help='Файл для сохранения таблицы истинности')
    arg_parser.add_argument('-c', '--console', action='store_true',
                            help='Вывести таблицу в консоль (по умолчанию: нет)')
    arg_parser.add_argument('-n', '--no-file', action='store_true',
                            help='Не сохранять в файл (по умолчанию: нет)')
    arg_parser.add_argument('-w', '--width', type=int, default=4,
                            help='Ширина ячейки таблицы (по умолчанию: 4)')
    arg_parser.add_argument('-f', '--force', action='store_true', default=True,
                            help='Перезаписать выходной файл, если существует (по умолчанию: да)')
    arg_parser.add_argument('-m', '--mode', type=int, choices=[1, 2], default=1,
                            help='Режим приоритета операций: 1 или 2 (по умолчанию: 1)')

    args = arg_parser.parse_args()

    # Чтение формулы из файла
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            formula = f.read().strip()
    except FileNotFoundError:
        print(f"Ошибка: файл '{args.input}' не найден", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}", file=sys.stderr)
        return 1

    if not formula:
        print("Ошибка: формула пустая", file=sys.stderr)
        return 1

    # Генерация таблицы истинности
    try:
        bool_parser = BooleanParser(mode=args.mode)
        variables, rows = bool_parser.compute_truth_table(formula)
        table = format_table(variables, rows, cell_width=args.width)
    except Exception as e:
        print(f"Ошибка при обработке формулы: {e}", file=sys.stderr)
        return 1

    # Вывод в консоль
    if args.console:
        print(f"\nФормула: {formula}")
        print(f"Режим: {args.mode}")
        print(table)

    # Сохранение в файл
    if not args.no_file:
        if not args.output:
            print("Ошибка: не указан выходной файл", file=sys.stderr)
            return 1

        try:
            mode = 'w' if args.force else 'x'
            with open(args.output, mode, encoding='utf-8') as f:
                f.write(f"Формула: {formula}\n")
                f.write(f"Режим: {args.mode}\n\n")
                f.write(table)
                f.write("\n")

            if not args.console:
                print(f"Таблица истинности сохранена в '{args.output}'")
        except FileExistsError:
            print(f"Ошибка: файл '{args.output}' уже существует. Используйте -f для перезаписи", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Ошибка при записи в файл: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())