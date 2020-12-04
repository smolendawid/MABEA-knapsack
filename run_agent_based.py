import json
from datetime import datetime
import yaml
import os

import numpy as np
import matplotlib.pylab as plt

from generate_data import create_knapsack_data
from mabea.lattice import Lattice


def log_genotype(latt, timestamp, it):

    genotypes = []
    for agent in latt.grid:
        genotypes.append((agent.genotype.tolist(), agent.energy))

    with open(os.path.join('results', timestamp, f'genotypes_{it}.json'), 'w') as f:
        for genotype in genotypes:
            f.write(f"{genotype[0]}, {genotype[1]} \n")


if __name__ == '__main__':
    config = {
        'data_seed': 42,
        'exp_seed': 42,
        'log_diversity': True,
        'log_genotypes': True,
        'print_interval': 10,
        'n_generations': 10000,
        'early_stopping': 200,
        'mutation_probability': 0.2,
        'lattice_size': 12,
    }

    # profits = [1, 6, 10, 16, 17, 18, 20, 31]
    # weights = [1, 2, 3,  5,  5,  6,  7,  11]
    # capacity = 20
    np.random.seed(config['data_seed'])

    profits, weights, capacity = create_knapsack_data(item_count=20)

    np.random.seed(config['exp_seed'])

    timestamp = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
    os.makedirs(os.path.join('results', timestamp))

    latt = Lattice(profits, weights, capacity, size=config['lattice_size'])

    means = []
    maxes = []
    rsc_mean = []
    rsc_std = []
    diversities = []

    es_it = 0
    prev_maximum = 0

    for i in range(config['n_generations']):
        latt.selection()
        latt.mutation(profits, weights, capacity, mutation_probability=config['mutation_probability'])

        if i % config['print_interval'] == 0:
            os.system('clear')
            print(i)
            latt.print()
            if config['log_genotypes']:
                log_genotype(latt, timestamp, i)
            if config['log_diversity']:
                diversities.append((i, latt.diversity()))

        means.append(np.mean(latt.get_energies_lattice()))
        maximum = np.amax(latt.get_energies_lattice())
        maxes.append(maximum)
        rsc_mean.append(np.mean(latt.get_resources()))
        rsc_std.append(np.mean(latt.get_resources()))

        if maximum <= prev_maximum:
            es_it += 1
        else:
            prev_maximum = maximum
            es_it = 0
        if es_it == config['early_stopping']:
            break

    # Remember the resutls
    print(f'Best result: {np.amax(maxes)}')

    plt.figure()
    plt.plot(means)
    plt.plot(maxes, 'r')
    plt.savefig(os.path.join('results', timestamp, 'energies.png'))

    plt.figure()
    plt.plot(rsc_mean)
    plt.plot(rsc_std, 'r')
    plt.savefig(os.path.join('results', timestamp, 'resources.png'))

    plt.figure()
    plt.plot([d[0] for d in diversities], [d[1] for d in diversities])
    plt.savefig(os.path.join('results', timestamp, 'diversity.png'))

    log_genotype(latt, timestamp, it='final')

    with open(os.path.join('results', timestamp, 'config.yml'), 'w') as f:
        f.write(yaml.dump(config))
