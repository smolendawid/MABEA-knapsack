from datetime import datetime
import yaml
import os

import numpy as np
import matplotlib.pylab as plt

from generate_data import create_knapsack_data, create_knapsack_correlated
from mabea.lattice import Lattice


def log_genotype(latt, timestamp, it):

    genotypes = []
    for agent in latt.grid:
        genotypes.append((agent.genotype.tolist(), agent.energy))

    with open(os.path.join('results', timestamp, f'genotypes_{it}.json'), 'w') as f:
        for genotype in genotypes:
            f.write(f"{genotype[0]}, {genotype[1]} \n")


if __name__ == '__main__':

    with open('config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    np.random.seed(config['data_seed'])

    if config['dataset'] == 'random':
        profits, weights, capacity = create_knapsack_data(item_count=config['item_count'])
    elif config['dataset'] == 'correlated':
        profits, weights, capacity = create_knapsack_correlated(item_count=config['item_count'])
    else:
        raise ValueError("Config param 'dataset' can be 'random' or 'correlated'")

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
    plt.title('Fitness')
    plt.plot(means)
    plt.plot(maxes, 'r')
    plt.legend(['mean', 'max'])
    plt.savefig(os.path.join('results', timestamp, 'energies.png'))

    plt.figure()
    plt.title('Available resources')
    plt.plot(rsc_mean)
    plt.plot(rsc_std, 'r')
    plt.savefig(os.path.join('results', timestamp, 'resources.png'))

    plt.figure()
    plt.title('Diversity during experiment')
    plt.plot([d[0] for d in diversities], [d[1] for d in diversities])
    plt.savefig(os.path.join('results', timestamp, 'diversity.png'))

    log_genotype(latt, timestamp, it='final')

    with open(os.path.join('results', timestamp, 'config.yml'), 'w') as f:
        f.write(yaml.dump(config))
