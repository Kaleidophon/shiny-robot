# -*- coding: utf-8 -*-
"""
Measure the distribution of values for a set of sudokus given a dispersion metric and see how the different metrics are
correlated.
"""

# EXT
import numpy
from scipy.stats import pearsonr

# PROJECT
from general import Sudoku
from experiments.distances import (
    SpatialAnalysisSudokuCollection,
    SpatialAnalysisSudoku,
    DeterminantSudoku,
    EigenvalueSudoku,
    CentroidSudoku,
    read_line_sudoku_file
)
from experiments.entropy import EntropySudokuCollection


class MetricComparison:
    """
    Class that compares the different metrics for measuring the dispersion of given numbers in Sudokus.
    """
    frequency_distributions = None
    correlations = None

    def __init__(self, sudoku_path, sudoku_collections, precision=3):
        assert len(sudoku_collections) > 1, "You need to compare at least two different metrics."
        self.sudoku_path = sudoku_path
        self.precision = precision

        self.sudoku_collections = {
            sudoku_collection.sudoku_cls: sudoku_collection
            for sudoku_collection in sudoku_collections
        }

    def gather_distributions(self):
        self.frequency_distributions = {}

        for sudoku_class, sudoku_collection in self.sudoku_collections.items():
            sudoku_collection.calculate_metric_distribution()
            distances_frequencies = sudoku_collection.distances_frequencies

            # Normalize and extend
            distances_frequencies = self.normalize_distribution_range(distances_frequencies)
            distances_frequencies = self.extend_distribution(distances_frequencies)

            self.frequency_distributions[sudoku_class] = distances_frequencies

    def normalize_distribution_range(self, distance_distribution):
        """
        Normalize the values of the dispersion metric between 0 and 1.
        """
        keys = numpy.array(list(distance_distribution.keys()))
        min_value = keys.min()
        max_value = keys.max()
        new_keys = numpy.array(list(map(
            lambda value: round((value-min_value)/(max_value-min_value), self.precision), keys)
        ))
        return dict(zip(new_keys, distance_distribution.values()))

    def extend_distribution(self, distance_distribution):
        """
        Extend the distribution entries to also accommodate values that didn't appear in the dataset (improves
        comparability).
        """
        extended_keys = numpy.linspace(0, 1, (10**self.precision))
        extended_keys = [numpy.floor(key*10**self.precision)/10**self.precision for key in extended_keys]
        return {
            key: 0 if key not in distance_distribution else distance_distribution[key]
            for key in extended_keys
        }

    def compute_metrics_correlations(self):
        assert self.frequency_distributions is not None, "run gather_distributions() first"

        print("\nCorrelation between dispersion metrics for the {} data set".format(self.sudoku_path))
        print("\n{pad}Sudoku class A |{pad}Sudoku class B | Pearson's rho".format(pad=9*" "))
        print("{pad}+{pad}+{pad}".format(pad=24*"-"))
        combs = set()
        for sudoku_class_a, distribution_a in self.frequency_distributions.items():
            for sudoku_class_b, distribution_b in self.frequency_distributions.items():
                if sudoku_class_a != sudoku_class_b and (sudoku_class_a, sudoku_class_b) not in combs:
                    combs.add((sudoku_class_a, sudoku_class_b))
                    combs.add((sudoku_class_b, sudoku_class_a))
                    pearsons_rho, _ = pearsonr(list(distribution_a.values()), list(distribution_b.values()))
                    print("{:>23} |{:>23} | {:.2f}".format(sudoku_class_a.__name__, sudoku_class_b.__name__, pearsons_rho))

if __name__ == "__main__":
    sudoku_path = "../data/10k_25.txt"
    precision = 2

    sudoku_collections = [
        SpatialAnalysisSudokuCollection(
            sudokus=read_line_sudoku_file(sudoku_path, sudoku_class=sudoku_class),
            precision=precision
        )
        for sudoku_class in [SpatialAnalysisSudoku, DeterminantSudoku, EigenvalueSudoku, CentroidSudoku]
    ]
    sudoku_collections.append(
        EntropySudokuCollection(
            sudokus=read_line_sudoku_file(sudoku_path, sudoku_class=Sudoku),
            precision=precision
        )
    )

    mc = MetricComparison(sudoku_path, sudoku_collections, precision=precision)
    mc.gather_distributions()
    mc.compute_metrics_correlations()
