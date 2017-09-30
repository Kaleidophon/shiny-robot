# -*- coding: utf-8 -*-
"""
Measure the dispersion of sudokus based on the shannon entropy of their distance matrix.
"""

# STD
import math

# EXT
import numpy

# PROJECT
from general import Sudoku, read_line_sudoku_file
from experiments.distances import SpatialAnalysisSudokuCollection


class EntropySudokuCollection(SpatialAnalysisSudokuCollection):
    """
    Collection that contains multiple EntropySudokus
    """
    position_probabilities = numpy.zeros(shape=(9, 9))
    entropy_sudokus = None

    def __init__(self, sudokus, precision=3):
        super().__init__(sudokus, precision)
        self._build_probabilities()
        self._convert_sudokus()

    def __iter__(self):
        for sudoku_uid, sudoku in self.entropy_sudokus.items():
            yield sudoku_uid, sudoku

    @property
    def sudoku_cls(self):
        return EntropySudoku

    def _build_probabilities(self):
        for _, sudoku in self.sudokus.items():
            for x in range(9):
                for y in range(9):
                    if sudoku[x][y] != 0:
                        self.position_probabilities[x][y] += 1

        self.position_probabilities /= len(self.sudokus)

    def _convert_sudokus(self):
        self.entropy_sudokus = {
            sudoku_id: EntropySudoku(sudoku.raw, self.position_probabilities)
            for sudoku_id, sudoku in self.sudokus.items()
        }

    def plot_average_and_variance(self):
        raise NotImplementedError


class EntropySudoku(Sudoku):
    """
    Class to represent a Sudoku whose degree of dispersion is measure using shannon entropy.
    """
    def __init__(self, raw_data, position_probabilities):
        super().__init__(raw_data)
        self.position_probabilities = position_probabilities

    @property
    def metric(self):
        """
        Shannon entropy of sudoku.
        """
        entropy = 0

        for x in range(9):
            for y in range(9):
                if self[x][y] != 0:
                    entropy += self.position_probabilities[x][y] * math.log(self.position_probabilities[x][y], 2)
                else:
                    entropy += (1 - self.position_probabilities[x][y]) * math.log(
                        (1 - self.position_probabilities[x][y]), 2
                    )

        return -1 * entropy


if __name__ == "__main__":
    sudoku_path = "../data/49k_17.txt"
    sudokus = read_line_sudoku_file(sudoku_path)

    esc = EntropySudokuCollection(sudokus)
    esc.calculate_metric_distribution()

    highest = esc.get_n_highest(3)
    lowest = esc.get_n_lowest(3)

    print("Sudokus with highest average entropy...")
    for _, sudoku in highest.items():
        print(str(sudoku))

    print("Sudokus with lowest average entropy...")
    for _, sudoku in lowest.items():
        print(str(sudoku))

    esc.plot_average_metric_distribution()
