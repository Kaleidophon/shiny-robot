# -*- coding: utf-8 -*-
"""
See if sudoku dispersion and number of conflicts is correlated.
"""

# EXT
import matplotlib.pyplot as plt

# PROJECT
from experiments.entropy import EntropySudoku, EntropySudokuCollection
from experiments.distances import CentroidSudoku, SpatialAnalysisSudokuCollection
from sudoku_solver import SudokuSolver
from general import read_line_sudoku_file


if __name__ == "__main__":
    sudoku_path = "../data/100_solved.txt"
    sudokus = read_line_sudoku_file(sudoku_path, sudoku_class=CentroidSudoku)
    sasc = SpatialAnalysisSudokuCollection(sudokus, precision=2)
    solver = SudokuSolver()

    dispersions, conflicts = [], []
    counter = 0
    for _, sudoku in sasc:
        print(_)
        counter += 1
        dispersions.append(sudoku.metric)
        conflicts.append(solver.solve_sudoku(sudoku)["conflicts"])
        print("\rEvaluating {} {}...".format(counter, sudoku.__class__.__name__ + "s"), end="", flush=True)

    print(dispersions, conflicts)
    plt.scatter(dispersions, conflicts)
    plt.show()
