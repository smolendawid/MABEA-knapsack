
from tabulate import tabulate
import numpy as np
from sklearn.neighbors import NearestNeighbors

from mabea.agent import Agent
from mabea.crossover import CrossOver


class Lattice:
    def __init__(self, profits, weights, capacity, size=4):
        self.size = size
        self.n_agents = size * size
        self.grid = [Agent(profits, weights, capacity) for _ in range(self.n_agents)]

        self.crossover = CrossOver()

        # Prepare index to lattice mapping
        self.ind2agent = {}
        i = 0
        for i_row in range(self.size):
            for i_col in range(self.size):
                self.ind2agent[(i_row, i_col)] = i
                i += 1
        self.agent2ind = {v: k for k, v in self.ind2agent.items()}

        # Precompute neighbourhood indices
        self.neighbourhood_inds = []
        for i, (i_row, i_col) in self.agent2ind.items():
            self.neighbourhood_inds.append(self._get_neighbourhood(i_row, i_col))

        self.print()

    def _get_neighbourhood(self, i_row, i_col):

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

    def selection(self):

        energies = np.array(self.get_energies_lattice())
        reproducing = []

        for i, (i_row, i_col) in self.agent2ind.items():
            neighbourhood_inds = self.neighbourhood_inds[i]
            neighbours_energies = energies.ravel()[neighbourhood_inds]
            self.distribute_resources(neighbourhood_inds, neighbours_energies)

            is_reproducing = self.crossover.is_ready_to_reproduce(i, neighbourhood_inds, self.grid)
            reproducing.append(is_reproducing)

        new_grid_cands = [[] for _ in range(len(self.grid))]
        for i in range(len(self.grid)):
            if reproducing[i]:
                self.grid[i].resources = 0
                neighbourhood_inds = self.neighbourhood_inds[i]
                neighbours_energies = energies.ravel()[neighbourhood_inds]
                min_ind = neighbourhood_inds[np.argmin(neighbours_energies)]
                new_grid_cands[min_ind].append(i)

        new_grid = self.crossover.run(self.grid, new_grid_cands)
        self.grid = new_grid

    def mutation(self, profits, weights, capacity, mutation_probability):
        for i, (i_row, i_col) in self.agent2ind.items():
            if np.random.rand() < mutation_probability:
                self.grid[i].mutate(profits, weights, capacity)

    def diversity(self):

        genotypes = []
        for agent in self.grid:
            genotypes.append(agent.genotype)

        nrst_neigh = NearestNeighbors(n_neighbors=len(self.grid), metric='cosine')
        nrst_neigh.fit(np.array(genotypes))

        distances, indices = nrst_neigh.kneighbors(np.array(genotypes))

        return distances.mean()
