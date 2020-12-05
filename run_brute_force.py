""" Reference code from
https://towardsdatascience.com/neural-knapsack-8edd737bdc15
No rights claimed
"""

import numpy as np
from generate_data import create_knapsack_data, create_knapsack_correlated


def brute_force_knapsack(x_prices, x_weights, x_capacity):
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


def unboundedKnapsack(W, n, val, wt):
    # dp[i] is going to store maximum
    # value with knapsack capacity i.
    dp = [0 for i in range(W + 1)]

    # Fill dp[] using above recursive formula
    for i in range(W + 1):
        for j in range(n):
            if wt[j] <= i:
                if dp[i] > dp[i - wt[j]] + val[j]:
                    pass
                else:
                    dp[i] = dp[i - wt[j]] + val[j]

    return dp[W]


if __name__ == '__main__':
    np.random.seed(42)
    # profits, weights, capacity = create_knapsack_data(item_count=20)
    profits, weights, capacity = create_knapsack_correlated(item_count=20)

    # Unbounded
    print("Unbounded")
    n = len(profits)
    print(unboundedKnapsack(capacity, n, profits, weights))

    print("Bounded")
    best_picks = brute_force_knapsack(profits, weights, capacity)
    print(profits)
    print(weights)
    print(capacity)
    best_picks = [i for i, val in enumerate(best_picks) if val == 1]
    print(best_picks)

    print(f"Weight sum: {np.sum(weights[best_picks])}")
    print(f"Best score: {np.sum(profits[best_picks])}")

