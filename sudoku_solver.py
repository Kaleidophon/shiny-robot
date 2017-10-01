import codecs
import hashlib
import operator
import itertools
import pycosat
import sys, getopt
import time
import subprocess
import numpy as np



class Sudoku_solver:
    def __solve__(self,problemset):
        unsolved_sudoku = problemset
        print('Problem:')
        pprint(problemset)
        conflicts = self.solve(problemset)
        print('Answer:')
        pprint(problemset)
        solution = problemset
        #unique = self.is_proper(unsolved_sudoku,solution)
        return conflicts

    def is_proper(self,problemset, solution):
        #find unique solution
        unsolved_sudoku = problemset
        for i in range(1,10):
            self.solve(problemset)
            if solution != problemset:
                print("Found mult solutions , improper sudoku")
                return 0

            problemset = unsolved_sudoku
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

    def solve(self,grid):
        # solve a Sudoku problem
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
        print("P CNF " + str(numclause) + "(number of clauses)")

        np.save("CNF_clauses", clauses)

        # solve the SAT problem
        start = time.time()
        proc = subprocess.Popen(["python", "terminal_py_solve.py"],
        stdout=subprocess.PIPE)
        out = proc.communicate()[0]
        print(out)

        conflicts = self.parse_statistics(out)

        #sol = set(pycosat.solve(clauses, verbose=1))
        out = 5

        end = time.time()
        print("Time: " + str(end - start))


        sol = set(pycosat.solve(clauses))
        def read_cell(i, j):
            # return the digit of cell i, j according to the solution
            for d in range(1, 10):
                if self.v(i, j, d) in sol:
                    return d

        for i in range(1, 10):
            for j in range(1, 10):
                grid[i - 1][j - 1] = read_cell(i, j)

        return conflicts

    def parse_statistics(self,output):
        output_utf8 = output.decode("utf-8")
        rows_output = output_utf8.split('\n')
        print("ROOOOOOOOOOW:", rows_output)
        conflicts = rows_output[9].split(' ')
        print("CONFILCTS",conflicts[19])

        return conflicts





if __name__ == '__main__':
    from pprint import pprint

    evil = [[0, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 6, 0, 0, 0, 0, 3],
            [0, 7, 4, 0, 8, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 3, 0, 0, 2],
            [0, 8, 0, 0, 4, 0, 0, 1, 0],
            [6, 0, 0, 5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 7, 8, 0],
            [5, 0, 0, 0, 0, 9, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 4, 0]]

    solver = Sudoku_solver()
    solver.__solve__(evil)
