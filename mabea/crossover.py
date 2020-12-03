import numpy as np


class CrossOver:
    def __init__(self):
        pass

    def run(self, energies, neighbours_inds):
        max_ind_abs = neighbours_inds[np.argmax(energies)]
        return max_ind_abs
