import copy

from tabulate import tabulate
import numpy as np

from mabea.agent import Agent
from mabea.crossover import CrossOver


class Lattice:
    def __init__(self, profits, weights, capacity, size=4):
        self.size = size
        self.n_agents = size * size
        self.grid = [Agent(profits, weights, capacity) for _ in range(self.n_agents)]

        self.crossover = CrossOver()
        self.ind2agent = {}
        i = 0
        for i_row in range(self.size):
            for i_col in range(self.size):
                self.ind2agent[(i_row, i_col)] = i
                i += 1
        self.agent2ind = {v: k for k, v in self.ind2agent.items()}
        self.print()

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

    def print(self):
        print(tabulate(self.get_energies_lattice()))

    def get_energies_lattice(self):
        energies = np.zeros((self.size, self.size))
        for i, (i_row, i_col) in self.agent2ind.items():
            energies[i_row, i_col] = self.grid[i].energy
        return energies

    def distribute_resources(self, inds, energies):
        """ add resources in neighbourhood according to local ranking"""
        inds_sorted = np.argsort(energies, )
        inds = np.array(inds)[inds_sorted]

        # todo on edges scores are too high or too less because neighbourhood is smaller
        for i, ind in enumerate(inds):
            self.grid[ind].collect_resources(i)

    def get_resources(self):
        return [agent.resources for agent in self.grid]

    def get_diversity(self):
        return [agent.resources for agent in self.grid]

    def selection(self, profits, weights, capacity, mutation_probability):

        energies = np.array(self.get_energies_lattice())
        new_grid = []

        for i, (i_row, i_col) in self.agent2ind.items():
            neighbours_inds = self._get_neighbours(i_row, i_col)
            neighbours_energies = energies.ravel()[neighbours_inds]

            self.distribute_resources(neighbours_inds, neighbours_energies)

            if self.grid[i].energy == np.amax(neighbours_energies):
                alpha = copy.deepcopy(self.grid[i])
                offspring = alpha
            else:
                max_ind_abs = self.crossover.run(neighbours_energies, neighbours_inds)
                offspring = copy.deepcopy(self.grid[max_ind_abs])

            if np.random.rand() < mutation_probability:
                offspring.mutate(profits, weights, capacity)
                new_grid.append(offspring)
            else:
                new_grid.append(offspring)
        self.grid = new_grid

    def diversity(self):

        genotypes = []
        for agent in self.grid:
            genotypes.append(agent.genotype)

        from sklearn.neighbors import NearestNeighbors
        nrst_neigh = NearestNeighbors(n_neighbors=len(self.grid), algorithm='ball_tree')
        nrst_neigh.fit(np.array(genotypes))

        distances, indices = nrst_neigh.kneighbors(np.array(genotypes))

        return distances
