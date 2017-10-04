# -*- coding: utf-8 -*-
"""
Functions used to create Sudokus which given numbers are either spread out within the sudoku or as closely together as
possible.

########################################################################################################################
WARNING: THIS CODE IS COMPLICATED AND DOESN'T WORK.
Apparently, removing numbers that maximize / minimize the dispersion metric lead to improper sudokus quickly. The
algorithm usually gets stuck between 45 - 60 given numbers. Following heuristics were tried out:
- When looking for a number to remove, only takes ones, twos .. etc.
- When looking for a number to remove, only take on number (1-9) chosen randomly

It still doesn't work.
########################################################################################################################
"""

# STD
import copy
import random

# PROJECT
from experiments.distances import SpatialAnalysisSudokuCollection, CentroidSudoku
from general import read_line_sudoku_file
from sudoku_solver import SudokuSolver


def create_extreme_sudoku(sudoku, goal=17, eliminate_randomly=50, objective=lambda sudoku: sudoku.metric):
    assert goal >= 17  # Minimum number for solvable sudokus
    assert eliminate_randomly < 64
    given_coordinates = sudoku.given_coordinates
    solver = SudokuSolver()

    # Step 1: Remove n random numbers from the sudoku
    coordinates_to_eliminate = random.sample(given_coordinates, eliminate_randomly)
    for x, y in coordinates_to_eliminate:
        sudoku[x][y] = 0

    # Step 2: Try to optimize sudoku according to the objective function
    given_coordinates = sudoku.given_coordinates

    dead_numbers = set()
    resample = lambda dead_numbers: random.sample(set(range(1, 10)) - dead_numbers, 1)[0]
    while len(given_coordinates) > goal:
        current_number_to_remove = resample(dead_numbers)
        deltas = {}  # Changes in value for objective function given the removal of coordinates of index i

        if len(dead_numbers) == 9:
            print("Fail at {} given numbers".format(len(given_coordinates)))
            import time
            time.sleep(4)
            return sudoku

        for x, y in given_coordinates:
            temp_sudoku = copy.copy(sudoku)
            old = objective(temp_sudoku)
            temp_sudoku[x][y] = 0
            temp_sudoku.update()
            new = objective(temp_sudoku)
            delta = new - old

            deltas[(x, y)] = (delta, sudoku[x][y])

        max_delta_indices = list(reversed(
            sorted(zip(range(len(deltas)), list(deltas.values())), key=lambda x: x[1][0])
        ))
        max_delta_indices = list(filter(
            lambda delta_tuple: delta_tuple[1][1] == current_number_to_remove, max_delta_indices
        ))

        print(current_number_to_remove)
        print("Indices", max_delta_indices)
        if len(max_delta_indices) == 0:
            current_number_to_remove = resample(dead_numbers)
            continue

        # Iterate through all the values, starting with the max, to see whether deleting the associated given number
        # results in a proper sudoku or not
        removed_number = False
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

                removed_number = True
                dead_numbers = set()
                break

        if not removed_number:
            dead_numbers.add(current_number_to_remove)

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
        print(str(
            create_extreme_sudoku(sudoku, goal=17, eliminate_randomly=0, objective=lambda sudoku: -sudoku.metric)
        ))
