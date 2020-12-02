from datetime import datetime
import yaml
import numpy as np
from tabulate import tabulate
import matplotlib.pylab as plt


values =  [1, 6, 10, 16, 17, 18, 20, 31]
weights = [1, 2, 3,  5,  5,  6,  7,  11]
weight_limit = 20


class Agent:
    def __init__(self, values, weights, weight_limit):
        self.energy = 0
        self.weight_limit = weight_limit
        self.values = []
        self.weights = []

        self._random_init(values, weights, weight_limit)

    def _random_init(self, values, weights, weight_limit):
        inds = np.arange(len(values))
        np.random.shuffle(inds)

        total_weight = 0
        total_value = 0

        for i, (value, weight) in enumerate(zip(np.array(values)[inds], np.array(weights)[inds])):
            if total_weight + weight > weight_limit:
                break
            total_value += value
            total_weight += weight
            self.values.append(value)
            self.weights.append(weight)

        self.energy = total_value

    def mutate(self, values, weights):
        inds = np.arange(len(values))

        weights_sum = np.sum(self.weights)
        while True:
            to_remove = np.random.choice(inds, 1)
            to_add = np.random.choice(inds, 1)
            if weights_sum - weights[to_remove] + weights[to_add] < weight_limit:
                self.values.pop(to_remove)
                self.weights.pop(to_remove)

                self.values.append(weights[to_add])
                self.weights.append(weights[to_add])
                self.energy = weights_sum - weights[to_remove] + weights[to_add]
                break


class Lattice:
    def __init__(self, size=4):
        self.size = size
        self.n_agents = size * size
        self.grid = [Agent(values, weights, weight_limit) for _ in range(self.n_agents)]

        self.ind2agent = {}
        i = 0
        for i_row in range(self.size):
            for i_col in range(self.size):
                self.ind2agent[(i_row, i_col)] = i
                i += 1
        self.agent2ind = {v: k for k, v in self.ind2agent.items()}

    def print(self):
        print(tabulate(self._get_energies()))

    def get_energies(self):
        energies = [[] for _ in range(self.size)]
        for i, (i_row, i_col) in self.agent2ind.items():
            curr_ind = (i_row * self.size) + i_col
            energies[i_row].append(self.grid[curr_ind].energy)
        return energies

    def _get_neighbours(self, energies, i_row, i_col):

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

    def selection(self):
        mutation_probability = 1

        energies = np.array(self.get_energies())
        new_grid = []

        for i, (i_row, i_col) in self.agent2ind.items():
            neighbours = self._get_neighbours(energies, i_row, i_col)
            for neighbour in neighbours:
                if self.grid[neighbour].energy == np.amax(energies):
                    if np.random.rand() < mutation_probability:
                        self.grid[i].mutate()
                    new_grid.append(self.grid[i])
                else:
                    crossover()
                    new_grid.append(i)


if __name__ == '__main__':
    config = {
        'seed': 1000,
        'num_iters': 1000,
    }
    np.random.seed(config['seed'])
    timestamp = datetime.now().strftime("%d-%b-%Y_(%H:%M:%S.%f)")

    latt = Lattice()
    latt.print()

    means = []
    maxes = []

    for i in range(config['num_iters']):
        latt.selection()
        means.append(np.mean(latt.get_energies()))
        maxes.append(np.amax(latt.get_energies()))

    plt.figure()
    plt.plot(means)
    plt.plot(maxes, 'r')
    plt.savefig(f'results/{timestamp}.png')

    with open(f'results/{timestamp}.yml') as f:
        f.write(yaml.dump(config))
