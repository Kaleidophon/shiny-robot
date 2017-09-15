# -*- coding: utf-8 -*-
"""
Functions concerning the input and output of Sudokus.
"""

# STD
import codecs


class Sudoku:
    """
    Class to represent a sudoku.
    """
    def __init__(self, raw_data):
        self.raw_data = raw_data

    def __repr__(self):
        divider = "{}+{}+{}\n".format(6*"-", 6*"-", 6*"-")

        representation = ""
        for i, n in zip(range(len(self.raw_data)), self.raw_data):
            representation += n + " " if int(n) != 0 else ". "
            if (i+1) % 3 == 0 and (i+1) % 9 != 0:
                representation += "|"
            elif (i+1) % 9 == 0:
                representation += "\n"

            if (i+1) % 27 == 0 and i+1 != 81:
                representation += divider
        return representation


def read_line_sudoku_file(path):
    sudokus = []

    with codecs.open(path, "rb", "utf-8") as sudoku_file:
        for line in sudoku_file.readlines():
            line = line.strip()
            sudokus.append(Sudoku(line))

    return sudokus


if __name__ == "__main__":
    sudoku_path = "./data/10k_25.txt"
    sudokus = read_line_sudoku_file(sudoku_path)

    for sudoku in sudokus:
        print(sudoku)
