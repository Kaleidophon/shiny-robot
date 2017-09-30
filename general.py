# -*- coding: utf-8 -*-
"""
Functions concerning the input and output of Sudokus.
"""

# STD
import codecs


class SudokuCollection:
    """
    Collection that contains multiple sudokus.
    """
    def __init__(self, sudokus):
        self.sudokus = sudokus

    def __iter__(self):
        for sudoku_uid, sudoku in self.sudokus.items():
            yield sudoku_uid, sudoku


class Sudoku:
    """
    Class to represent a sudoku.
    """
    def __init__(self, raw_data):
        self.uid = id(raw_data)
        self.raw_data = raw_data
        self.list_representation = self._to_lists(self.raw_data)

    def __getitem__(self, index):
        return self.list_representation[index]

    def __repr__(self):
        return "<Sudoku with id={}>".format(self.uid)

    def __str__(self):
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

    @staticmethod
    def _to_lists(raw_data):
        list_representation, current_row = [], []

        for i, n in zip(range(len(raw_data)), raw_data):
            current_row.append(int(n))

            if (i+1) % 9 == 0:
                list_representation.append(current_row)
                current_row = []

        return list_representation

    @property
    def raw(self):
        return self.raw_data

    def _to_raw(self):
        return "".join(["".join([str(cell) for cell in row]) for row in self.list_representation])

    def update(self):
        self.raw_data = self._to_raw()

    def __copy__(self):
        return self.__class__(self.raw_data)


def read_line_sudoku_file(path, sudoku_class=Sudoku):
    sudokus = {}
    i = 0

    with codecs.open(path, "rb", "utf-8") as sudoku_file:
        for line in sudoku_file.readlines():
            line = line.strip()
            sudoku = sudoku_class(line)
            sudokus[sudoku.uid] = sudoku
            i += 1
            print("\rProcessing {} {}...".format(i, sudoku_class.__name__ + "s"), end="", flush=True)

    print("")

    return sudokus


if __name__ == "__main__":
    sudoku_path = "./data/100_25.txt"
    sudokus = read_line_sudoku_file(sudoku_path)


    #for sudoku in sudokus:
    #    print(sudoku)
