import numpy as np
from collections import namedtuple

Product = namedtuple('Product', ['i', 'weight', 'profit'])


class Agent:
    def __init__(self, profits, weights, capacity):
        self.energy = 0
        self.capacity = capacity
        self.chosen_products = []
        self.genotype = np.zeros((len(weights), ), dtype=int)
        self.resources = 0

        self._random_init(profits, weights, capacity)

    def _random_init(self, profits, weights, capacity):
        inds = np.arange(len(profits))

        total_weight = 0
        total_value = 0

        while True:
            ind = np.random.choice(inds, 1)[0]
            if total_weight + weights[ind] > capacity:
                break
            total_value += profits[ind]
            total_weight += weights[ind]

            self.chosen_products.append(Product(ind, weights[ind], profits[ind]))
            self.genotype[ind] += 1

        self.energy = total_value

    def _mutate_both(self, profits, weights, capacity):
        inds_inner = np.arange(len(self.chosen_products))
        inds_outer = np.arange(len(profits))

        weights_sum = np.sum([p.weight for p in self.chosen_products])
        profits_sum = np.sum([p.profit for p in self.chosen_products])
        while True:
            to_remove = np.random.choice(inds_inner, 1)[0]
            to_add = np.random.choice(inds_outer, 1)[0]
            if weights_sum - self.chosen_products[to_remove].weight + weights[to_add] < capacity:
                self.energy = profits_sum - self.chosen_products[to_remove].profit + profits[to_add]

                self.genotype[self.chosen_products[to_remove].i] -= 1
                self.genotype[to_add] += 1

                self.chosen_products.pop(to_remove)
                self.chosen_products.append(Product(to_add, weights[to_add], profits[to_add]))
                break

    def _mutate_remove(self):
        inds = np.arange(len(self.chosen_products))

        profits_sum = np.sum([p.profit for p in self.chosen_products])
        to_remove = np.random.choice(inds, 1)[0]

        self.genotype[self.chosen_products[to_remove].i] -= 1

        self.energy = profits_sum - self.chosen_products[to_remove].profit
        self.chosen_products.pop(to_remove)

    def _mutate_add(self, profits, weights, capacity):
        inds = np.arange(len(profits))

        weights_sum = np.sum([p.weight for p in self.chosen_products])
        profits_sum = np.sum([p.profit for p in self.chosen_products])
        for i in range(10):
            to_add = np.random.choice(inds, 1)[0]
            if weights_sum + weights[to_add] < capacity:
                self.genotype[to_add] += 1
                self.chosen_products.append(Product(to_add, weights[to_add], profits[to_add]))
                self.energy = profits_sum + profits[to_add]
                break

    def mutate(self, profits, weights, capacity):

        prob = np.random.rand()
        if prob > 0.66:
            self._mutate_both(profits, weights, capacity)
        elif prob > 0.33:
            self._mutate_remove()
        elif prob > 0.0:
            self._mutate_add(profits, weights, capacity)

    def collect_resources(self, resources):
        self.resources += resources
