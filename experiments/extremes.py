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
from experiments.distances import SpatialAnalysisSudokuCollection, SpatialAnalysisSudoku, CentroidSudoku
from general import read_line_sudoku_file
from sudoku_solver import SudokuSolver


def create_extreme_sudoku(sudoku, goal=17, eliminate_randomly=50, objective=lambda sudoku: sudoku.metric):
    #assert goal >= 17  # Minimum number for solvable sudokus
    assert eliminate_randomly < 64
    given_coordinates = sudoku.given_coordinates
    solver = SudokuSolver()

    # Step 1: Remove n random numbers from the sudoku
    coordinates_to_eliminate = random.sample(given_coordinates, eliminate_randomly)
    for x, y in coordinates_to_eliminate:
        sudoku[x][y] = 0

    # Step 2: Try to optimize sudoku according to the objective function
    given_coordinates = sudoku.given_coordinates
    while len(given_coordinates) > goal:

        if len(given_coordinates) % 2 == 0:
            x, y = random.sample(given_coordinates, 1)[0]
            sudoku[x][y] = 0
            sudoku.update()

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

        max_delta_indices = list(reversed(sorted(zip(range(len(deltas)), list(deltas.values())), key=lambda x: x[1])))

        # Iterate through all the values, starting with the max, to see whether deleting the associated given number
        # results in a proper sudoku or not
        for max_delta_index, max_delta in max_delta_indices:
            if max_delta_index == len(max_delta_indices)-1:
                print("Fail at {} given numbers".format(len(given_coordinates)))
                import time
                time.sleep(4)
                return sudoku

            x_max, y_max = given_coordinates[max_delta_index]
            temp_sudoku = copy.copy(sudoku)
            temp_sudoku[x][y] = 0
            temp_sudoku.update()

            if solver.is_proper(temp_sudoku):
                print("Eliminating given at ({}, {}), {} numbers left.".format(x_max, y_max, len(given_coordinates)-1))
                given_coordinates.pop(max_delta_index)
                sudoku[x_max][y_max] = 0
                sudoku.update()
                print(str(sudoku))
                break

    sudoku.update()  # Save changes internally
    return sudoku

if __name__ == "__main__":

    sudoku_path = "../data/100_solved.txt"
    sudokus = read_line_sudoku_file(sudoku_path, sudoku_class=CentroidSudoku)
    sasc = SpatialAnalysisSudokuCollection(sudokus, precision=2)

    for _, sudoku in sudokus.items():
        print(str(sudoku))

        # Maximizing average distance -> Spread out sudokus
        #print(str(create_extreme_sudoku(sudoku, objective=lambda sudoku: sudoku.average_distance)))

        # Maximizing negative average distance -> Minimizing average distance -> dense sudokus
        print(str(create_extreme_sudoku(sudoku, goal=17, eliminate_randomly=10, objective=lambda sudoku: -sudoku.metric)))

