import numpy as np


def create_knapsack_data(item_count=5):
    """ Generate random dataset """

    weights = np.random.randint(1, 45, item_count)
    profits = np.random.randint(1, 99, item_count)
    capacity = np.random.randint(50, 99)
    return profits, weights, capacity
