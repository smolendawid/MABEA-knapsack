import numpy as np


class Agent:
    def __init__(self, profits, weights, capacity):
        self.energy = 0
        self.capacity = capacity
        self.chosen_profits = []
        self.chosen_weights = []

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
            self.chosen_profits.append(profits[ind])
            self.chosen_weights.append(weights[ind])

        self.energy = total_value

    def mutate(self, profits, weights, capacity):
        inds_inner = np.arange(len(self.chosen_profits))
        inds_outer = np.arange(len(profits))

        weights_sum = np.sum(self.chosen_weights)
        profits_sum = np.sum(self.chosen_profits)
        while True:
            to_remove = np.random.choice(inds_inner, 1)[0]
            to_add = np.random.choice(inds_outer, 1)[0]
            if weights_sum - self.chosen_weights[to_remove] + weights[to_add] < capacity:
                self.energy = profits_sum - self.chosen_profits[to_remove] + profits[to_add]
                self.chosen_profits.pop(to_remove)
                self.chosen_weights.pop(to_remove)

                self.chosen_profits.append(profits[to_add])
                self.chosen_weights.append(weights[to_add])
                break

    def mutate_remove(self, profits, weights):
        inds = np.arange(len(self.chosen_profits))

        profits_sum = np.sum(self.chosen_profits)
        while True:
            to_remove = np.random.choice(inds, 1)[0]
            self.energy = profits_sum - self.chosen_profits[to_remove]
            self.chosen_profits.pop(to_remove)
            self.chosen_weights.pop(to_remove)
            break

    def mutate_add(self, profits, weights, capacity):
        inds = np.arange(len(profits))

        weights_sum = np.sum(self.chosen_weights)
        profits_sum = np.sum(self.chosen_profits)
        for i in range(10):
            to_add = np.random.choice(inds, 1)[0]
            if weights_sum + weights[to_add] < capacity:
                self.chosen_profits.append(profits[to_add])
                self.chosen_weights.append(weights[to_add])
                self.energy = profits_sum + profits[to_add]
                break

