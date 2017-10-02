# -*- coding: utf-8 -*-
"""
Functions used to create Sudokus which given numbers are either spread out within the sudoku or as closely together as
possible.
"""

# STD
import copy
import random

# EXT
import numpy

# PROJECT
from experiments.distances import SpatialAnalysisSudokuCollection, SpatialAnalysisSudoku
from general import read_line_sudoku_file
from sudoku_solver import SudokuSolver


def create_extreme_sudoku(sudoku, goal=17, eliminate_randomly=50, objective=lambda sudoku: sudoku.metric):
    assert goal >= 17  # Minimum number for solvable sudokus
    assert eliminate_randomly < 64
    given_coordinates = sudoku.given_coordinates

    # Step 1: Remove n random numbers from the sudoku
    coordinates_to_eliminate = random.sample(given_coordinates, eliminate_randomly)
    for x, y in coordinates_to_eliminate:
        sudoku[x][y] = 0

    # Step 2: Try to optimize sudoku according to the objective function
    given_coordinates = sudoku.given_coordinates
    while len(given_coordinates) > goal:
        deltas = {}  # Changes in value for objective function given the removal of coordinates of index i

        for x, y in given_coordinates:
            temp_sudoku = copy.copy(sudoku)
            old = objective(temp_sudoku)
            temp_sudoku[x][y] = 0
            temp_sudoku.update()
            new = objective(temp_sudoku)
            delta = new - old

            deltas[(x, y)] = delta

        # TODO: Check whether sudokus are still proper

        max_delta_index = numpy.argmax(list(deltas.values()))
        x_max, y_max = given_coordinates[max_delta_index]
        print("Eliminating given at ({}, {})".format(x_max, y_max))
        given_coordinates.pop(max_delta_index)
        sudoku[x_max][y_max] = 0
        sudoku.update()
        print(str(sudoku))

        solver = SudokuSolver()
        print(solver.is_proper(sudoku.list_representation))

    sudoku.update()  # Save changes internally
    return sudoku

if __name__ == "__main__":

    sudoku_path = "../data/100_solved.txt"
    sudokus = read_line_sudoku_file(sudoku_path, sudoku_class=SpatialAnalysisSudoku)
    sasc = SpatialAnalysisSudokuCollection(sudokus, precision=2)

    for _, sudoku in sudokus.items():
        print(str(sudoku))

        # Maximizing average distance -> Spread out sudokus
        #print(str(create_extreme_sudoku(sudoku, objective=lambda sudoku: sudoku.average_distance)))

        # Maximizing negative average distance -> Minimizing average distance -> dense sudokus
        print(str(create_extreme_sudoku(sudoku, objective=lambda sudoku: -sudoku.metric)))

