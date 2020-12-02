""" Reference code from
https://towardsdatascience.com/neural-knapsack-8edd737bdc15
No rights claimed
"""

import numpy as np


def brute_force_knapsack(x_weights, x_prices, x_capacity):
    item_count = x_weights.shape[0]
    picks_space = 2 ** item_count
    best_price = -1
    best_picks = np.zeros(item_count)
    for p in range(picks_space):
        picks = [int(c) for c in f"{p:0{item_count}b}"]
        price = np.dot(x_prices, picks)
        weight = np.dot(x_weights, picks)
        if weight <= x_capacity and price > best_price:
            best_price = price
            best_picks = picks
    return best_picks


def create_knapsack(item_count=5):

    x_weights = np.random.randint(1, 45, item_count)
    x_prices = np.random.randint(1, 99, item_count)
    x_capacity = np.random.randint(50, 99)
    return x_weights, x_prices, x_capacity


if __name__ == '__main__':
    np.random.seed(42)
    x_weights, x_prices, x_capacity = create_knapsack(item_count=20)
    best_picks = brute_force_knapsack(x_weights, x_prices, x_capacity)

    print(x_weights)
    print(x_prices)
    print(x_capacity)
    print(best_picks)
    print(np.sum(x_prices[best_picks]))
