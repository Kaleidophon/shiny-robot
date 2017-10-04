# -*- coding: utf-8 -*-
"""
Module comprising a SAT solver to solve sudokus using the efficient encodings.
"""

# STD
import os.path
import subprocess

# EXT
import pycosat
import numpy as np

# PROJECT
import terminal_py_solve

# CONST
TERMINAL_SOLVE_PATH = os.path.abspath(terminal_py_solve.__file__)
CLAUSES_NP_FILE_PATH = os.path.abspath(terminal_py_solve.__file__.replace("terminal_py_solve.py", "/"))


class SudokuSolver:
    """
    SAT solver to solve sudokus.
    """
    clauses = []

    @staticmethod
    def v(i, j, d):
        return 81 * (i - 1) + 9 * (j - 1) + d

    @property
    def sudoku_clauses(self):
        """
        Generate logic clauses for sudoku. Use cache if they were already created.
        """
        if len(self.clauses) == 0:
            self.clauses = self._sudoku_clauses()
        return list(self.clauses)

    def _sudoku_clauses(self):
        """
        Efficient encoding for sudokus.
        """
        res = []
        # for all cells, ensure that the each cell:
        for i in range(1, 10):
            for j in range(1, 10):
                # denotes (at least) one of the 9 digits (1 clause)
                res.append([self.v(i, j, d) for d in range(1, 10)])
                # does not denote two different digits at once (36 clauses)
                for d in range(1, 10):
                    for dp in range(d + 1, 10):
                        res.append([-self.v(i, j, d), -self.v(i, j, dp)])

        def valid(cells):
            for i, xi in enumerate(cells):
                for j, xj in enumerate(cells):
                    if i < j:
                        for d in range(1, 10):
                            res.append([-self.v(xi[0], xi[1], d), -self.v(xj[0], xj[1], d)])

        # ensure rows and columns have distinct values
        for i in range(1, 10):
            valid([(i, j) for j in range(1, 10)])
            valid([(j, i) for j in range(1, 10)])

        # ensure 3x3 sub-grids "regions" have distinct values
        for i in 1, 4, 7:
            for j in 1, 4, 7:
                valid([(i + k % 3, j + k // 3) for k in range(9)])

        assert len(res) == 81 * (1 + 36) + 27 * 324
        return res

    def create_and_save_clauses(self, grid):
        clauses = self.sudoku_clauses
        # print(len(clauses))
        for i in range(1, 10):
            for j in range(1, 10):
                d = grid[i - 1][j - 1]
                # For each digit already known, a clause (with one literal).
                if d:
                    clauses.append([self.v(i, j, d)])

        # Print number SAT clause
        numclause = len(clauses)
        # print("P CNF " + str(numclause) + "(number of clauses)")

        np.save(CLAUSES_NP_FILE_PATH + "/CNF_clauses.npy", clauses)
        return clauses

    def is_proper(self, sudoku):
        clauses = self.create_and_save_clauses(sudoku.list_representation)
        return len(list(pycosat.itersolve(clauses))) == 1

    def solve_sudoku(self, sudoku):
        clauses = self.create_and_save_clauses(sudoku.list_representation)

        # solve the SAT problem
        proc = subprocess.Popen(["python3", TERMINAL_SOLVE_PATH],
        stdout=subprocess.PIPE)
        out = proc.communicate()[0]

        return self.parse_statistics(out)

    def parse_statistics(self, output):
        # Parse output
        statistics = np.array(
            list(filter(
                lambda x: x != "" and self.is_number(x), output.decode().split(" ")
            ))[-10:]
        ).astype(np.float)

        return dict(zip(
            ["seconds", "level", "variables", "used", "original", "conflicts", "learned", "limit", "agility", "MB"],
            statistics[:10]
        ))

    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
