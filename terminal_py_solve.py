import pycosat

import numpy as np

if __name__ == '__main__':
    from pprint import pprint

clauses = np.load("CNF_clauses.npy")
pycosat.solve(clauses,verbose=1)