
# STD
from collections import defaultdict
import math

# EXT
import numpy
import matplotlib.pyplot as plt

# PROJECT
from sudoku_io import Sudoku, read_line_sudoku_file


class DistributivenessSudoku(Sudoku):
    """
    Sudoku class with also measures the degree of spatial distribution of given numbers within the sudoku.
    """
    given_coordinates = []

    def __init__(self, raw_data):
        super().__init__(raw_data)
        self.given_coordinates, self.distance_matrix = self._build_distance_matrix(self.list_representation)
        # self.print_matrix(self.distance_matrix)

    def _build_distance_matrix(self, list_representation):
        given_coordinates = []

        # Get coordinates of given numbers
        for x in range(len(list_representation)):
            for y in range(len(list_representation[x])):
                if list_representation[x][y] != 0:
                    given_coordinates.append((x, y))

        # Calculate distance matrix
        distance_matrix = numpy.zeros(shape=(len(given_coordinates), len(given_coordinates)))
        for i in range(len(given_coordinates)):
            for j in range(len(given_coordinates)):
                distance_matrix[i][j] += self.numbers_distance(given_coordinates[i], given_coordinates[j])

        return given_coordinates, distance_matrix

    @property
    def average_distance(self):
        shape = self.distance_matrix.shape
        return sum([sum(row) for row in self.distance_matrix]) / (shape[0] * shape[1])

    @staticmethod
    def numbers_distance(coordinates1, coordinates2):
        return numpy.linalg.norm(
            numpy.array(
                [coordinates1[0] - coordinates2[0], coordinates1[1] - coordinates2[1]]
            )
        )

    @staticmethod
    def print_matrix(matrix):
        numpy.set_printoptions(precision=2, suppress=True, linewidth=220)
        print(matrix)


def calculate_distance_distribution(average_distances, precision=2):
    frequencies = defaultdict(int)

    for dist in average_distances:
        frequencies[math.ceil(dist*10**precision)/10**precision] += 1

    return frequencies


def plot_distribution(frequencies):
    plt.bar(list(frequencies.keys()), list(frequencies.values()), align="center")
    plt.show()

if __name__ == "__main__":
    sudoku_path = "../data/10k_25.txt"
    sudokus = read_line_sudoku_file(sudoku_path, sudoku_class=DistributivenessSudoku)
    average_distances = [sudoku.average_distance for sudoku in sudokus]
    frequencies = calculate_distance_distribution(average_distances, precision=2)
    plot_distribution(frequencies)

    #for sudoku in sudokus:
    #    print(sudoku)
