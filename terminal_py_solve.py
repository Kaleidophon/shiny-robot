#!/usr/bin/python3

import pycosat
import sys

import numpy

if __name__ == '__main__':
    CLAUSES_NP_FILE_PATH = sys.modules[__name__].__file__.replace("terminal_py_solve.py", "/")
    clauses = numpy.load(CLAUSES_NP_FILE_PATH + "/CNF_clauses.npy")
    pycosat.solve(clauses, verbose=1)
