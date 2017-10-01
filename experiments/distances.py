# -*- coding: utf-8 -*-
"""
Functions to analyze the distances between given number inside a sudoku.
"""

# STD
from collections import defaultdict
import math
import functools

# EXT
import numpy
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# PROJECT
from general import Sudoku, SudokuCollection, read_line_sudoku_file


# https://en.wikipedia.org/wiki/Spatial_descriptive_statistics


class SpatialAnalysisSudoku(Sudoku):
    """
    Sudoku class with also measures the degree of spatial distribution of given numbers within the sudoku.
    """
    given_coordinates = []

    def __init__(self, raw_data):
        super().__init__(raw_data)
        self._build_distance_matrix()
        # self.print_matrix(self.distance_matrix)

    def _build_distance_matrix(self):
        given_coordinates = self.given_coordinates
        n_givens = len(given_coordinates)

        # Calculate distance matrix
        self.distance_matrix = numpy.zeros(shape=(n_givens, n_givens))
        for i in range(n_givens):
            for j in range(n_givens):
                self.distance_matrix[i][j] += self.numbers_distance(given_coordinates[i], given_coordinates[j])

    @property
    def metric(self):
        shape = self.distance_matrix.shape
        return sum([sum(row) for row in self.distance_matrix]) / (shape[0] * shape[1])

    @property
    def distance_variance(self):
        average_distance = self.metric
        shape = self.distance_matrix.shape
        return sum([
            sum([
                (dist - average_distance)**2 for dist in row
            ])
            for row in self.distance_matrix
        ]) / (shape[0] * shape[1])

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

    def update(self):
        super().update()
        self._build_distance_matrix()


class DeterminantSudoku(SpatialAnalysisSudoku):
    """
    Use the determinant of the distance matrix as a measure of the dispersion of the given numbers.
    """
    @property
    def metric(self):
        return numpy.linalg.det(self.distance_matrix)

    @property
    def distance_variance(self):
        raise NotImplementedError


class EigenvalueSudoku(SpatialAnalysisSudoku):
    """
    Use the biggest eigenvalue of the distance matrix as a measure of the dispersion of the given numbers.
    """
    @property
    def metric(self):
        return eigh(self.distance_matrix, eigvals_only=True)[0]

    @property
    def distance_variance(self):
        raise NotImplementedError


class CentroidSudoku(SpatialAnalysisSudoku):
    """
    Create a centroid based on the distribution of the given numbers and measure the average distance to it.
    """
    @property
    def centroid(self):
        given_coordinates = self.given_coordinates
        x_coordinates, y_coordinates = zip(*given_coordinates)
        return (
            sum(x_coordinates)/len(x_coordinates),  # x coordinate for centroid
            sum(y_coordinates)/len(y_coordinates)  # y coordinate for centroid
        )

    @property
    def metric(self):
        distance_matrix = self.distance_matrix
        return sum(distance_matrix) / len(distance_matrix)

    @property
    def distance_variance(self):
        distance_matrix = self.distance_matrix
        average_distance = self.metric
        return functools.reduce(
            lambda sum_, dist: sum_ + (dist - average_distance)**2, distance_matrix
        ) / len(distance_matrix)

    def _build_distance_matrix(self):
        given_coordinates = self.given_coordinates
        centroid = self.centroid
        n_givens = len(given_coordinates)

        # Calculate distance matrix
        self.distance_matrix = numpy.zeros(shape=(n_givens, 1))

        for coordinates, i in zip(given_coordinates, range(n_givens)):
            self.distance_matrix[i] = self.numbers_distance(centroid, coordinates)


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

    @property
    def sudoku_cls(self):
        return type(list(self.sudokus.values())[0])

    def calculate_metric_distribution(self):
        self.average_distances = {
            sudoku_uid: sudoku.metric for sudoku_uid, sudoku in self
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

    def plot_average_metric_distribution(self, title=True):
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
        if title:
            ax.set_title(
                "Distribution of average distances of given numbers among {} sudokus".format(number_of_sudokus),
                fontsize=10
            )
        print("Showing plot...")
        plt.show()

    def plot_average_and_variance(self, title=True):
        averages_and_variances = {
            sudoku_uid: (sudoku.metric, sudoku.distance_variance)
            for sudoku_uid, sudoku in self
        }
        number_of_sudokus = len(self.sudokus)

        fig, ax = plt.subplots()
        x, y = zip(*list(averages_and_variances.values()))

        ax.scatter(x, y)
        if title:
            ax.set_title(
                "Distribution of {} Sudokus given their distance mean and average".format(number_of_sudokus),
                fontsize=10
            )
        plt.xlabel("Average distance")
        plt.ylabel("Distance variance")

        plt.show()

    def plot_heatmap(self, n=0):
        """
        Generate heat map of given numbers for n sudokus. If n is 0, a heat map for all the sudokus in this collection
        will be created. If n is positive, a heat map with the n highest scoring sudokus according to the metric of the
        sudokus in this collection will be created, the same accordingly for the lowest ones when n is negative.
        """
        heatmap_sudokus = None
        if n == 0:
            heatmap_sudokus = self.sudokus
        if n > 0:
            heatmap_sudokus = self.get_n_highest(n)
        if n < 0:
            heatmap_sudokus = self.get_n_lowest(-n)

        heatmap_data = numpy.zeros(shape=(9, 9))

        for _, sudoku in heatmap_sudokus.items():
            for x, y in sudoku.given_coordinates:
                heatmap_data[x][y] += 1

        plt.imshow(heatmap_data, cmap="seismic", interpolation='nearest')
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    sudoku_path = "../data/49k_17.txt"
    sudokus = read_line_sudoku_file(sudoku_path, sudoku_class=CentroidSudoku)
    sasc = SpatialAnalysisSudokuCollection(sudokus, precision=2)
    sasc.calculate_metric_distribution()

    highest = sasc.get_n_highest(3)
    lowest = sasc.get_n_lowest(3)

    print("Sudokus with highest average distances...")
    for _, sudoku in highest.items():
        print(str(sudoku))

    print("Sudokus with lowest average distances...")
    for _, sudoku in lowest.items():
        print(str(sudoku))

    #sasc.plot_average_metric_distribution(title=False)
    #sasc.plot_average_and_variance()

    #sasc.plot_heatmap()
    #sasc.plot_heatmap(1000)
    #sasc.plot_heatmap(-1000)
