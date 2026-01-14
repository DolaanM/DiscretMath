#!/usr/bin/env python3
"""
Тесты для генератора таблиц истинности
Запуск: python test_bool_truth_table.py
"""

import unittest
import tempfile
import os
from typing import List


# Импортируем основной код (предполагается, что он в файле bool_truth_table.py)
# Для тестирования скопируйте класс BooleanParser сюда или импортируйте
class BooleanParser:
    """Парсер и вычислитель булевых выражений"""

    # Символы операций (поддержка ASCII и Unicode)
    OPERATORS = {
        '~': 'NOT', '!': 'NOT', '¬': 'NOT',
        '&': 'AND', '∧': 'AND',
        '|': 'OR', '∨': 'OR',
        '^': 'XOR', '⊕': 'XOR',
        '↓': 'NOR',
        '/': 'NAND', '|/': 'NAND',
        '->': 'IMP', '→': 'IMP',
        '<-': 'RIMP', '←': 'RIMP',
        '<->': 'EQ', '↔': 'EQ', '~': 'EQV',
    }

    def __init__(self, mode=1):
        self.mode = mode

    def tokenize(self, formula: str) -> List[str]:
        """Разбивает формулу на токены"""
        import re
        formula = formula.replace('<->', '↔').replace('->', '→').replace('<-', '←')
        formula = formula.replace('|/', '/')

        tokens = []
        i = 0
        while i < len(formula):
            if formula[i].isspace():
                i += 1
                continue

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

            if formula[i].isalpha() or formula[i].isdigit() or formula[i] == '_':
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
        import re
        tokens = self.tokenize(formula)
        variables = []
        for token in tokens:
            if token not in self.OPERATORS and token not in '()' and not token in ['~', '!', '¬', '&', '∧', '|', '∨',
                                                                                   '^', '⊕', '↓', '/', '→', '←', '↔']:
                if token not in variables and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token):
                    variables.append(token)

        def natural_key(s):
            return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]

        return sorted(variables, key=natural_key)

    def to_postfix(self, tokens: List[str]) -> List[str]:
        """Преобразует инфиксную нотацию в постфиксную"""
        if self.mode == 1:
            precedence = {
                '~': 4, '!': 4, '¬': 4,
                '&': 3, '∧': 3,
                '|': 2, '∨': 2, '^': 2, '⊕': 2, '↓': 2, '/': 2,
                '→': 2, '←': 2, '↔': 2,
            }
        else:
            precedence = {
                '~': 6, '!': 6, '¬': 6,
                '&': 5, '∧': 5,
                '|': 4, '∨': 4, '^': 4, '⊕': 4,
                '→': 3, '←': 3,
                '↔': 2,
                '↓': 1, '/': 1,
            }

        output = []
        stack = []

        for token in tokens:
            if token not in self.OPERATORS and token not in ['~', '!', '¬', '&', '∧', '|', '∨', '^', '⊕', '↓', '/', '→',
                                                             '←', '↔', '(', ')']:
                output.append(token)
            elif token in ['~', '!', '¬', '&', '∧', '|', '∨', '^', '⊕', '↓', '/', '→', '←', '↔']:
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
                    stack.pop()

        while stack:
            output.append(stack.pop())

        return output

    def evaluate(self, postfix: List[str], values: dict) -> int:
        """Вычисляет значение выражения в постфиксной нотации"""
        stack = []

        for token in postfix:
            if token in values:
                stack.append(values[token])
            elif token in ['~', '!', '¬']:
                if stack:
                    a = stack.pop()
                    stack.append(1 - a)
            elif token in ['&', '∧']:
                if len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(a & b)
            elif token in ['|', '∨']:
                if len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(a | b)
            elif token in ['^', '⊕']:
                if len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(a ^ b)
            elif token == '↓':
                if len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(1 - (a | b))
            elif token == '/':
                if len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(1 - (a & b))
            elif token in ['→']:
                if len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append((1 - a) | b)
            elif token in ['←']:
                if len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(a | (1 - b))
            elif token in ['↔']:
                if len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(1 if a == b else 0)

        return stack[0] if stack else 0

    def compute_truth_table(self, formula: str):
        """Генерирует таблицу истинности"""
        from itertools import product

        variables = self.get_variables(formula)

        if not variables:
            raise ValueError("В формуле не найдено переменных")

        tokens = self.tokenize(formula)
        postfix = self.to_postfix(tokens)

        n = len(variables)
        rows = []

        for values_tuple in product([0, 1], repeat=n):
            values = dict(zip(variables, values_tuple))
            result = self.evaluate(postfix, values)
            rows.append(list(values_tuple) + [result])

        return variables, rows


class TestBooleanParser(unittest.TestCase):
    """Тесты для класса BooleanParser"""

    def setUp(self):
        """Инициализация перед каждым тестом"""
        self.parser_mode1 = BooleanParser(mode=1)
        self.parser_mode2 = BooleanParser(mode=2)

    # === Тесты токенизации ===

    def test_tokenize_simple(self):
        """Тест простой токенизации"""
        tokens = self.parser_mode1.tokenize("x1 & x2")
        self.assertEqual(tokens, ['x1', '&', 'x2'])

    def test_tokenize_complex(self):
        """Тест сложной токенизации"""
        tokens = self.parser_mode1.tokenize("(x1 | x2) & ~x3")
        self.assertEqual(tokens, ['(', 'x1', '|', 'x2', ')', '&', '~', 'x3'])

    def test_tokenize_unicode(self):
        """Тест токенизации с Unicode символами"""
        tokens = self.parser_mode1.tokenize("x1 ∧ x2 ∨ ¬x3")
        self.assertEqual(tokens, ['x1', '∧', 'x2', '∨', '¬', 'x3'])

    def test_tokenize_implication(self):
        """Тест токенизации импликации"""
        tokens = self.parser_mode1.tokenize("x1 -> x2")
        self.assertEqual(tokens, ['x1', '→', 'x2'])

    def test_tokenize_equivalence(self):
        """Тест токенизации эквивалентности"""
        tokens = self.parser_mode1.tokenize("x1 <-> x2")
        self.assertEqual(tokens, ['x1', '↔', 'x2'])

    # === Тесты извлечения переменных ===

    def test_get_variables_simple(self):
        """Тест извлечения переменных"""
        vars = self.parser_mode1.get_variables("x1 & x2")
        self.assertEqual(vars, ['x1', 'x2'])

    def test_get_variables_ordering(self):
        """Тест естественной сортировки переменных"""
        vars = self.parser_mode1.get_variables("x10 & x2 & x1")
        self.assertEqual(vars, ['x1', 'x2', 'x10'])

    def test_get_variables_duplicates(self):
        """Тест удаления дубликатов переменных"""
        vars = self.parser_mode1.get_variables("x1 & x1 | x2")
        self.assertEqual(vars, ['x1', 'x2'])

    # === Тесты преобразования в постфикс ===

    def test_postfix_simple_and(self):
        """Тест постфикса для простого AND"""
        tokens = self.parser_mode1.tokenize("x1 & x2")
        postfix = self.parser_mode1.to_postfix(tokens)
        self.assertEqual(postfix, ['x1', 'x2', '&'])

    def test_postfix_with_not(self):
        """Тест постфикса с NOT"""
        tokens = self.parser_mode1.tokenize("~x1 & x2")
        postfix = self.parser_mode1.to_postfix(tokens)
        self.assertEqual(postfix, ['x1', '~', 'x2', '&'])

    def test_postfix_with_parentheses(self):
        """Тест постфикса со скобками"""
        tokens = self.parser_mode1.tokenize("(x1 | x2) & x3")
        postfix = self.parser_mode1.to_postfix(tokens)
        self.assertEqual(postfix, ['x1', 'x2', '|', 'x3', '&'])

    def test_postfix_nested_parentheses(self):
        """Тест постфикса с вложенными скобками"""
        tokens = self.parser_mode1.tokenize("((x1 & x2) | x3)")
        postfix = self.parser_mode1.to_postfix(tokens)
        self.assertEqual(postfix, ['x1', 'x2', '&', 'x3', '|'])

    # === Тесты вычисления логических операций ===

    def test_evaluate_and(self):
        """Тест вычисления AND"""
        postfix = ['x1', 'x2', '&']
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 0, 'x2': 0}), 0)
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 0, 'x2': 1}), 0)
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 1, 'x2': 0}), 0)
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 1, 'x2': 1}), 1)

    def test_evaluate_or(self):
        """Тест вычисления OR"""
        postfix = ['x1', 'x2', '|']
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 0, 'x2': 0}), 0)
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 0, 'x2': 1}), 1)
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 1, 'x2': 0}), 1)
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 1, 'x2': 1}), 1)

    def test_evaluate_not(self):
        """Тест вычисления NOT"""
        postfix = ['x1', '~']
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 0}), 1)
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 1}), 0)

    def test_evaluate_xor(self):
        """Тест вычисления XOR"""
        postfix = ['x1', 'x2', '^']
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 0, 'x2': 0}), 0)
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 0, 'x2': 1}), 1)
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 1, 'x2': 0}), 1)
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 1, 'x2': 1}), 0)

    def test_evaluate_nand(self):
        """Тест вычисления NAND (штрих Шеффера)"""
        postfix = ['x1', 'x2', '/']
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 0, 'x2': 0}), 1)
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 0, 'x2': 1}), 1)
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 1, 'x2': 0}), 1)
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 1, 'x2': 1}), 0)

    def test_evaluate_nor(self):
        """Тест вычисления NOR (стрелка Пирса)"""
        postfix = ['x1', 'x2', '↓']
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 0, 'x2': 0}), 1)
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 0, 'x2': 1}), 0)
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 1, 'x2': 0}), 0)
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 1, 'x2': 1}), 0)

    def test_evaluate_implication(self):
        """Тест вычисления импликации"""
        postfix = ['x1', 'x2', '→']
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 0, 'x2': 0}), 1)
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 0, 'x2': 1}), 1)
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 1, 'x2': 0}), 0)
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 1, 'x2': 1}), 1)

    def test_evaluate_reverse_implication(self):
        """Тест вычисления обратной импликации"""
        postfix = ['x1', 'x2', '←']
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 0, 'x2': 0}), 1)
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 0, 'x2': 1}), 0)
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 1, 'x2': 0}), 1)
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 1, 'x2': 1}), 1)

    def test_evaluate_equivalence(self):
        """Тест вычисления эквивалентности"""
        postfix = ['x1', 'x2', '↔']
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 0, 'x2': 0}), 1)
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 0, 'x2': 1}), 0)
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 1, 'x2': 0}), 0)
        self.assertEqual(self.parser_mode1.evaluate(postfix, {'x1': 1, 'x2': 1}), 1)

    # === Тесты таблиц истинности ===

    def test_truth_table_and(self):
        """Тест таблицы истинности для AND"""
        vars, rows = self.parser_mode1.compute_truth_table("x1 & x2")
        self.assertEqual(vars, ['x1', 'x2'])
        expected = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]]
        self.assertEqual(rows, expected)

    def test_truth_table_or(self):
        """Тест таблицы истинности для OR"""
        vars, rows = self.parser_mode1.compute_truth_table("x1 | x2")
        expected = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
        self.assertEqual(rows, expected)

    def test_truth_table_complex(self):
        """Тест таблицы истинности для сложного выражения"""
        vars, rows = self.parser_mode1.compute_truth_table("(x1 & x2) | ~x3")
        self.assertEqual(vars, ['x1', 'x2', 'x3'])
        # x1=0, x2=0, x3=0: (0&0)|~0 = 0|1 = 1
        # x1=0, x2=0, x3=1: (0&0)|~1 = 0|0 = 0
        # ...
        expected = [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [1, 1, 0, 1],
            [1, 1, 1, 1]
        ]
        self.assertEqual(rows, expected)

    # === Тесты приоритетов ===

    def test_priority_mode1(self):
        """Тест приоритетов в режиме 1: NOT > AND > остальные"""
        # x1 | x2 & x3 должно быть x1 | (x2 & x3)
        vars, rows = self.parser_mode1.compute_truth_table("x1 | x2 & x3")
        # При x1=0, x2=1, x3=1: 0 | (1&1) = 0 | 1 = 1
        self.assertEqual(rows[3], [0, 1, 1, 1])

    def test_priority_mode2_or_xor(self):
        """Тест приоритетов в режиме 2: OR и XOR одинаковы"""
        # В режиме 2: x1 ^ x2 | x3 обрабатывается слева направо
        vars, rows = self.parser_mode2.compute_truth_table("x1 ^ x2 | x3")
        # При x1=1, x2=0, x3=1: (1^0)|1 = 1|1 = 1
        self.assertEqual(rows[5], [1, 0, 1, 1])

    # === Тесты граничных случаев ===

    def test_single_variable(self):
        """Тест с одной переменной"""
        vars, rows = self.parser_mode1.compute_truth_table("x1")
        self.assertEqual(vars, ['x1'])
        self.assertEqual(rows, [[0, 0], [1, 1]])

    def test_double_negation(self):
        """Тест двойного отрицания"""
        vars, rows = self.parser_mode1.compute_truth_table("~(~x1)")
        # ~(~0) = ~1 = 0, ~(~1) = ~0 = 1
        self.assertEqual(rows, [[0, 0], [1, 1]])

    def test_single_negation(self):
        """Тест одиночного отрицания"""
        vars, rows = self.parser_mode1.compute_truth_table("~x1")
        # ~0 = 1, ~1 = 0
        self.assertEqual(rows, [[0, 1], [1, 0]])

    def test_complex_parentheses(self):
        """Тест сложных вложенных скобок"""
        vars, rows = self.parser_mode1.compute_truth_table("((x1 & x2) | (x3 & x4))")
        self.assertEqual(len(vars), 4)
        self.assertEqual(len(rows), 16)  # 2^4


class TestFileOperations(unittest.TestCase):
    """Тесты работы с файлами"""

    def test_read_formula_from_file(self):
        """Тест чтения формулы из файла"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("x1 & x2 | x3")
            temp_file = f.name

        try:
            with open(temp_file, 'r') as f:
                formula = f.read().strip()
            self.assertEqual(formula, "x1 & x2 | x3")
        finally:
            os.unlink(temp_file)

    def test_write_table_to_file(self):
        """Тест записи таблицы в файл"""
        parser = BooleanParser(mode=1)
        vars, rows = parser.compute_truth_table("x1 & x2")

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Test table\n")
            for row in rows:
                f.write(' '.join(map(str, row)) + '\n')
            temp_file = f.name

        try:
            with open(temp_file, 'r') as f:
                content = f.read()
            self.assertIn("Test table", content)
            self.assertIn("0 0 0", content)
            self.assertIn("1 1 1", content)
        finally:
            os.unlink(temp_file)


class TestSpecialCases(unittest.TestCase):
    """Тесты специальных и граничных случаев"""

    def test_tautology(self):
        """Тест тавтологии (всегда истина)"""
        parser = BooleanParser(mode=1)
        # x1 | ~x1 всегда истина
        vars, rows = parser.compute_truth_table("x1 | ~x1")
        self.assertTrue(all(row[-1] == 1 for row in rows))

    def test_contradiction(self):
        """Тест противоречия (всегда ложь)"""
        parser = BooleanParser(mode=1)
        # x1 & ~x1 всегда ложь
        vars, rows = parser.compute_truth_table("x1 & ~x1")
        self.assertTrue(all(row[-1] == 0 for row in rows))

    def test_de_morgan_law(self):
        """Тест закона де Моргана: ~(x1 & x2) = ~x1 | ~x2"""
        parser = BooleanParser(mode=1)
        vars1, rows1 = parser.compute_truth_table("~(x1 & x2)")
        vars2, rows2 = parser.compute_truth_table("~x1 | ~x2")

        # Результаты должны совпадать
        self.assertEqual([r[-1] for r in rows1], [r[-1] for r in rows2])


def run_tests():
    """Запуск всех тестов"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestBooleanParser))
    suite.addTests(loader.loadTestsFromTestCase(TestFileOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestSpecialCases))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\n{'=' * 70}")
    print(f"Тестов запущено: {result.testsRun}")
    print(f"Успешно: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Провалено: {len(result.failures)}")
    print(f"Ошибок: {len(result.errors)}")
    print(f"{'=' * 70}")

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)