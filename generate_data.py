import numpy as np
import matplotlib.pylab as plt


def create_knapsack_data(item_count=5):
    """ Generate random dataset """

    weights = np.random.randint(1, 45, item_count)
    profits = np.random.randint(1, 99, item_count)
    capacity = np.random.randint(50, 99)
    return profits, weights, capacity


def create_knapsack_correlated(item_count=5):
    """ Generate random correlated dataset """

    density = 3
    yy = np.array([1, 45])
    xx = np.array([1, 99])
    means = [xx.mean(), yy.mean()]
    stds = [xx.std() / density, yy.std() / density]
    corr = 0.99  # correlation
    covs = [[stds[0] ** 2, stds[0] * stds[1] * corr],
            [stds[0] * stds[1] * corr, stds[1] ** 2]]

    m = np.random.multivariate_normal(means, covs, item_count).T

    m = np.round(m).astype(int)
    capacity = 400
    return m[0], m[1], capacity


if __name__ == '__main__':
    np.random.seed(42)
    profits, weights, _ = create_knapsack_data(item_count=20)

    plt.Figure()
    plt.scatter(weights, profits)
    plt.xlabel("weights")
    plt.ylabel("profits")

    np.random.seed(42)
    profits, weights, _ = create_knapsack_correlated(item_count=20)

    plt.scatter(weights, profits)
    plt.legend(['random', 'correlated'])
    plt.show()
