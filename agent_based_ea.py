import copy
from datetime import datetime
import yaml
import numpy as np
from tabulate import tabulate
import matplotlib.pylab as plt

from generate_data import create_knapsack_data


class Agent:
    def __init__(self, profits, weights, weight_limit):
        self.energy = 0
        self.weight_limit = weight_limit
        self.chosen_profits = []
        self.chosen_weights = []

        self._random_init(profits, weights, weight_limit)

    def _random_init(self, profits, weights, weight_limit):
        inds = np.arange(len(profits))

        total_weight = 0
        total_value = 0

        while True:
            ind = np.random.choice(inds, 1)[0]
            if total_weight + weights[ind] > weight_limit:
                break
            total_value += profits[ind]
            total_weight += weights[ind]
            self.chosen_profits.append(profits[ind])
            self.chosen_weights.append(weights[ind])

        self.energy = total_value

    def mutate(self, profits, weights, capacity):
        inds = np.arange(len(self.chosen_profits))

        weights_sum = np.sum(self.chosen_weights)
        profits_sum = np.sum(self.chosen_profits)
        while True:
            to_remove = np.random.choice(inds, 1)[0]
            to_add = np.random.choice(inds, 1)[0]
            if weights_sum - weights[to_remove] + weights[to_add] < capacity:
                self.chosen_profits.pop(to_remove)
                self.chosen_weights.pop(to_remove)

                self.chosen_profits.append(profits[to_add])
                self.chosen_weights.append(weights[to_add])
                self.energy = profits_sum - profits[to_remove] + profits[to_add]
                break

    # def mutate_remove(self, profits, weights):
    #     inds = np.arange(len(self.chosen_profits))
    #
    #     profits_sum = np.sum(self.chosen_profits)
    #     while True:
    #         to_remove = np.random.choice(inds, 1)[0]
    #         self.chosen_profits.pop(to_remove)
    #         self.chosen_weights.pop(to_remove)
    #         self.energy = profits_sum - profits[to_remove]
    #         break


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

    def selection(self, profits, weights, capacity, mutation_probability):

        energies = np.array(self.get_energies())
        new_grid = []

        for i, (i_row, i_col) in self.agent2ind.items():
            neighbours_inds = self._get_neighbours(i_row, i_col)
            neighbours_energies = energies.ravel()[neighbours_inds]

            if self.grid[i].energy == np.amax(neighbours_energies):
                if np.random.rand() < mutation_probability:
                    mutated = copy.deepcopy(self.grid[i])
                    mutated.mutate(profits, weights, capacity)
                    new_grid.append(mutated)
                else:
                    new_grid.append(copy.deepcopy(self.grid[i]))
            else:
                new_grid.append(self.crossover(neighbours_energies, neighbours_inds))
        self.grid = new_grid


if __name__ == '__main__':
    config = {
        'seed': 42,
        'print_interval': 60,
        'n_generations': 10000,
        'early_stopping': 1000,
        'mutation_probability': 0.2,
        'lattice_size': 12,
    }

    # profits = [1, 6, 10, 16, 17, 18, 20, 31]
    # weights = [1, 2, 3,  5,  5,  6,  7,  11]
    # weight_limit = 20
    np.random.seed(config['seed'])

    profits, weights, capacity = create_knapsack_data(item_count=20)

    timestamp = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")

    latt = Lattice(profits, weights, capacity, size=config['lattice_size'])

    means = []
    maxes = []

    es_it = 0
    prev_maximum = 0

    for i in range(config['n_generations']):
        latt.selection(profits, weights, capacity, mutation_probability=config['mutation_probability'])

        # Log
        if i % config['print_interval'] == 0:
            print(i)
            latt.print()

        means.append(np.mean(latt.get_energies()))
        maximum = np.amax(latt.get_energies())
        maxes.append(maximum)

        if maximum <= prev_maximum:
            es_it += 1
        else:
            prev_maximum = maximum
            es_it = 0
        if es_it == config['early_stopping']:
            break

    print(f'Best result: {np.amax(maxes)}')

    plt.figure()
    plt.plot(means)
    plt.plot(maxes, 'r')
    plt.savefig(f'results/{timestamp}.png')

    with open(f'results/{timestamp}.yml', 'w') as f:
        f.write(yaml.dump(config))
