import numpy as np
import pandas as pd


def get_data():
    """
    Reads the course project data file
    :return: np.array
    """

    data = pd.read_csv('../data/ecommerce_data.csv').values
    X = data[:, 0:-1]
    Y = data[:, -1]

    # Normalize numeric columns
    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()  # num products
    X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()  # duration

    # time of day is categorial value (0-3) we are going to one-hot encode it
    tod_col = 4  # time of day column idx
    N, D = X.shape
    tod = np.zeros((N, 4))
    tod[np.arange(N), X[:, tod_col].astype(np.int32)] = 1

    # remove the time-of-day column and insert the new columns
    X = np.delete(X, tod_col, axis=1)
    X = np.insert(X, [tod_col], tod, axis=1)

    return X, Y


def get_first_two_classes():
    """
    Reads the course project data file and return samples only where the actual result category is either 0 or 1
    :return:
    """
    X, Y = get_data()
    X = X[Y <= 1]
    Y = Y[Y <= 1]

    return X, Y
