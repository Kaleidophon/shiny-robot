import pycosat
import time
import subprocess
import numpy as np
import copy
import os
import os.path

import terminal_py_solve

TERMINAL_SOLVE_PATH = os.path.abspath(terminal_py_solve.__file__)
CLAUSES_NP_FILE_PATH = os.path.abspath(terminal_py_solve.__file__.replace("terminal_py_solve.py", "/"))

class SudokuSolver:

    def solve_sudoku(self, sudoku):
        return self._solve(sudoku.list_representation)

    def _solve(self, problemset):
        #print('Problem:')
        #pprint(problemset)
        all_statistics = self.solve(problemset)
        #print('Answer:')
        #pprint(problemset)

        #print the statistics
        print(all_statistics)

        return all_statistics

    def is_proper(self,problemset):
        #find unique solution
        if not os.path.isfile(CLAUSES_NP_FILE_PATH):
            self.create_and_save_claues(problemset)

        unsolved_sudoku = copy.deepcopy(problemset)

        for i in range(1,10):
            self.solve(problemset)
            solution = copy.deepcopy(problemset)
            if i == 1:
                initial_solution = copy.deepcopy(solution)
            else:
                if initial_solution != solution:
                    print("Found mult solutions , improper sudoku")
                    print("INITIAL SOLUTION",initial_solution)
                    print("ANOTHER SOLUTION",solution)
                    return 0

            problemset = copy.deepcopy(unsolved_sudoku)
        return 1

    @staticmethod
    def v(i, j, d):
        return 81 * (i - 1) + 9 * (j - 1) + d

    # Reduces Sudoku problem to a SAT clauses
    def sudoku_clauses(self):
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

    def create_and_save_claues(self, grid):
        clauses = self.sudoku_clauses()
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

        c = CLAUSES_NP_FILE_PATH
        np.save(CLAUSES_NP_FILE_PATH + "/CNF_clauses.npy", clauses)

    def solve(self,grid):
        # solve a Sudoku problem
        self.create_and_save_claues(grid)
        #sol = set(pycosat.solve(clauses, verbose=1))


        # solve the SAT problem
        start = time.time()
        proc = subprocess.Popen(["python3", TERMINAL_SOLVE_PATH],
        stdout=subprocess.PIPE)
        out = proc.communicate()[0]

        all_statistics = self.parse_statistics(out)

        out = 5

        end = time.time()
        #print("Time: " + str(end - start))


        # sol = set(pycosat.solve(clauses))
        # def read_cell(i, j):
        #     # return the digit of cell i, j according to the solution
        #     for d in range(1, 10):
        #         if self.v(i, j, d) in sol:
        #             return d
        #
        # for i in range(1, 10):
        #     for j in range(1, 10):
        #         grid[i - 1][j - 1] = read_cell(i, j)

        return all_statistics

    def parse_statistics(self,output):
        statistics = np.array(list(filter(lambda x: x != "" and self.is_number(x), output.decode().split(" ")))[-10:]).astype(
            np.float)
        all_statistics = {"seconds": statistics[0],
                           "level": statistics[1],
                           "variables": statistics[2],
                           "used": statistics[3],
                           "original": statistics[4],
                           "conflicts": statistics[5],
                           "learned": statistics[6],
                           "limit": statistics[7],
                           "agility": statistics[8],
                           "MB": statistics[9]}
        return all_statistics

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False


if __name__ == '__main__':
    from pprint import pprint

    hard = [[0, 2, 0, 0, 0, 0, 0, 3, 0],
            [0, 0, 0, 6, 0, 1, 0, 0, 0],
            [0, 6, 8, 2, 0, 0, 0, 0, 5],
            [0, 0, 9, 0, 0, 8, 3, 0, 0],
            [0, 4, 6, 0, 0, 0, 7, 5, 0],
            [0, 0, 1, 3, 0, 0, 4, 0, 0],
            [9, 0, 0, 0, 0, 7, 5, 1, 0],
            [0, 0, 0, 1, 0, 4, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 9, 0]]

    evil = [[0, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 6, 0, 0, 0, 0, 3],
            [0, 7, 4, 0, 8, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 3, 0, 0, 2],
            [0, 8, 0, 0, 4, 0, 0, 1, 0],
            [6, 0, 0, 5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 7, 8, 0],
            [5, 0, 0, 0, 0, 9, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 4, 0]]

    easy = [[0, 0, 0, 1, 0, 9, 4, 2, 7],
            [1, 0, 9, 8, 0, 0, 0, 0, 6],
            [0, 0, 7, 0, 5, 0, 1, 0, 8],
            [0, 5, 6, 0, 0, 0, 0, 8, 2],
            [0, 0, 0, 0, 2, 0, 0, 0, 0],
            [9, 4, 0, 0, 0, 0, 6, 1, 0],
            [7, 0, 4, 0, 6, 0, 9, 0, 0],
            [6, 0, 0, 0, 0, 8, 2, 0, 5],
            [2, 9, 5, 3, 0, 1, 0, 0, 0]]

    solver = SudokuSolver()
    solver._solve(evil)
    #solver.is_proper(evil)
