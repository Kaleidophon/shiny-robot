
# STD
from collections import defaultdict
import math

# EXT
import numpy
import matplotlib.pyplot as plt

# PROJECT
from sudoku import Sudoku, SudokuCollection, read_line_sudoku_file


class SpatialAnalysisSudoku(Sudoku):
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


class SpatialAnalysisSudokuCollection(SudokuCollection):
    """
    Collection that contains multiple distributiveness sudokus.
    """
    average_distances = None
    sorted_average_distances = None
    distances_frequencies = None
    sorted_distances_frequencies = None

    def __init__(self, sudokus, precision=2):
        assert precision in range(2, 4)
        super().__init__(sudokus)
        self.precision = precision

    def calculate_distance_distribution(self):
        self.average_distances = {
            sudoku_uid: sudoku.average_distance for sudoku_uid, sudoku in self
        }
        self.distances_frequencies = defaultdict(int)

        for _, dist in self.average_distances.items():
            self.distances_frequencies[math.ceil(dist*10**self.precision)/10**self.precision] += 1

        # Sort average distances - sorted by value
        self.sorted_average_distances = sorted(self.average_distances.items(), key=lambda x: x[1])

        # Sort average distances frequencies - sorted by key
        self.sorted_distances_frequencies = sorted(self.distances_frequencies.items(), key=lambda x: x[0])

    def get_n_highest(self, n):
        """
        Get the n sudokus with the highest average distance.
        """
        return {
            sudoku_uid: self.sudokus[sudoku_uid]
            for sudoku_uid, _ in self.sorted_average_distances[len(self.sorted_average_distances)-n:]
        }

    def get_n_lowest(self, n):
        """
        Get the n sudokus with the lowest average distance.
        """
        return {
            sudoku_uid: self.sudokus[sudoku_uid]
            for sudoku_uid, _ in self.sorted_average_distances[:n]
        }

    def plot_distribution(self):
        assert self.average_distances is not None and self.distances_frequencies is not None, \
            "Please run calculate_distance_distribution() first"
        number_of_sudokus = len(self.sudokus)
        plot_min = self.sorted_distances_frequencies[0][0]
        plot_max = self.sorted_distances_frequencies[len(self.sorted_distances_frequencies)-1][0]
        plot_range = plot_max - plot_min

        fig, ax = plt.subplots()
        ax.bar(
            list(self.distances_frequencies.keys()),
            list(self.distances_frequencies.values()),
            width=plot_range/len(self.distances_frequencies)/self.precision,
            align='center', edgecolor="black", facecolor="grey"
        )
        ax.set_title(
            "Distribution of average distances of given numbers among {} sudokus".format(number_of_sudokus),
            fontsize=10
        )
        print("Showing plot...")
        plt.show()

if __name__ == "__main__":
    sudoku_path = "../data/100_25.txt"
    sudokus = read_line_sudoku_file(sudoku_path, sudoku_class=SpatialAnalysisSudoku)
    sasc = SpatialAnalysisSudokuCollection(sudokus, precision=2)
    sasc.calculate_distance_distribution()

    highest = sasc.get_n_highest(3)
    lowest = sasc.get_n_lowest(3)

    print("Sudokus with highest average distances...")
    for _, sudoku in highest.items():
        print(str(sudoku))

    print("Sudokus with lowest average distances...")
    for _, sudoku in lowest.items():
        print(str(sudoku))

    sasc.plot_distribution()


