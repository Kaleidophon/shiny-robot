# -*- coding: utf-8 -*-
"""
See if sudoku dispersion and number of conflicts is correlated.
"""

# STD
import random

# EXT
import matplotlib.pyplot as plt

# PROJECT
from experiments.entropy import EntropySudoku, EntropySudokuCollection
from experiments.distances import CentroidSudoku, SpatialAnalysisSudokuCollection
from sudoku_solver import SudokuSolver
from general import read_line_sudoku_file


def get_data(sudoku, solver):
    return sudoku.metric, solver.solve_sudoku(sudoku)["conflicts"]

if __name__ == "__main__":
    sudoku_path = "../data/100_25.txt"
    sample_size = 10

    #centroid_sudokus = read_line_sudoku_file(sudoku_path, sudoku_class=CentroidSudoku)
    #sasc = SpatialAnalysisSudokuCollection(centroid_sudokus, precision=2)

    entropy_sudokus = read_line_sudoku_file(sudoku_path)
    esc = EntropySudokuCollection(entropy_sudokus, precision=2)

    eval_sudokus = random.sample(list(esc.entropy_sudokus.values()), sample_size)
    solver = SudokuSolver()

    dispersions, conflicts = [], []
    counter = 0

    for sudoku in eval_sudokus:
        counter += 1
        dispersions.append(sudoku.metric)
        conflicts.append(solver.solve_sudoku(sudoku)["conflicts"])

        print("\rEvaluating {} {}...".format(counter, sudoku.__class__.__name__ + "s"), end="", flush=True)

    plt.scatter(dispersions, conflicts, marker=".")
    plt.show()
