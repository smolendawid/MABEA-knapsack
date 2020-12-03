import copy

from tabulate import tabulate
import numpy as np

from mabea.agent import Agent


class Lattice:
    def __init__(self, profits, weights, capacity, size=4):
        self.size = size
        self.n_agents = size * size
        self.grid = [Agent(profits, weights, capacity) for _ in range(self.n_agents)]

        self.ind2agent = {}
        i = 0
        for i_row in range(self.size):
            for i_col in range(self.size):
                self.ind2agent[(i_row, i_col)] = i
                i += 1
        self.agent2ind = {v: k for k, v in self.ind2agent.items()}
        self.print()

    def print(self):
        print(tabulate(self.get_energies()))

    def get_energies(self):
        energies = np.zeros((self.size, self.size))
        for i, (i_row, i_col) in self.agent2ind.items():
            energies[i_row, i_col] = self.grid[i].energy
        return energies

    def _get_neighbours(self, i_row, i_col):

        col_bw = i_col-1
        col_fw = i_col+1
        row_bw = i_row-1
        row_fw = i_row+1

        if row_bw < 0:
            row_bw = 0
        if row_fw == self.size:
            row_fw = self.size - 1
        if col_bw < 0:
            col_bw = 0
        if col_fw == self.size:
            col_fw = self.size - 1

        neighbours = []
        for i_row in range(row_bw, row_fw+1):
            for i_col in range(col_bw, col_fw+1):
                neighbours.append(self.ind2agent[(i_row, i_col)])

        return neighbours

    def crossover(self, energies, neighbours_inds):

        max_ind_abs = neighbours_inds[np.argmax(energies)]
        return copy.deepcopy(self.grid[max_ind_abs])

    def mutate(self, offspring, profits, weights, capacity,):
        prob = np.random.rand()
        if prob > 0.66:
            offspring.mutate(profits, weights, capacity)
        elif prob > 0.33:
            offspring.mutate_remove(profits, weights)
        elif prob > 0.0:
            offspring.mutate_add(profits, weights, capacity)

        return offspring

    def selection(self, profits, weights, capacity, mutation_probability):

        energies = np.array(self.get_energies())
        new_grid = []

        for i, (i_row, i_col) in self.agent2ind.items():
            neighbours_inds = self._get_neighbours(i_row, i_col)
            neighbours_energies = energies.ravel()[neighbours_inds]

            if self.grid[i].energy == np.amax(neighbours_energies):
                alpha = copy.deepcopy(self.grid[i])
                offspring = alpha
            else:
                offspring = self.crossover(neighbours_energies, neighbours_inds)

            if np.random.rand() < mutation_probability:
                new_grid.append(self.mutate(offspring, profits, weights, capacity))
            else:
                new_grid.append(offspring)
        self.grid = new_grid
