from datetime import datetime
import yaml
import os

import numpy as np
import matplotlib.pylab as plt

from generate_data import create_knapsack_data
from mabea.lattice import Lattice


if __name__ == '__main__':
    config = {
        'seed': 42,
        'print_interval': 1,
        'n_generations': 10000,
        'early_stopping': 200,
        'mutation_probability': 0.2,
        'lattice_size': 12,
    }

    # profits = [1, 6, 10, 16, 17, 18, 20, 31]
    # weights = [1, 2, 3,  5,  5,  6,  7,  11]
    # capacity = 20
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
            os.system('clear')
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
