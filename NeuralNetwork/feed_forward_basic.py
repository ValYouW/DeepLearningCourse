import numpy as np
import matplotlib.pyplot as plt


def main():
    """
    This is basic feed forward implementation demo
    It will implement a single forward calculation of nn with 1 hidden layer
    using random weights
    """

    nPerK = 500

    # create random data for 3 different classes (normally distributed numbers around some avg)
    XK1 = np.random.randn(nPerK, 2) + [0, -2]
    XK2 = np.random.randn(nPerK, 2) + [2, 2]
    XK3 = np.random.randn(nPerK, 2) + [-2, 2]

    # set the "actual" classes (0, 1, 2)
    Y = np.array([0] * nPerK + [1] * nPerK + [2] * nPerK)

    # join all classes to one dataset
    X = np.vstack([XK1, XK2, XK3])

    D = X.shape[1]  # number of "features"
    K = 3  # number of classes
    M = 3  # number of activation units in the first hidden layer

    # weights and bias for the first hidden layer
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)

    # weights and bias for the output layer
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    Y_hat = forward(X, W1, b1, W2, b2)
    class_rate = classification_rate(Y, Y_hat)
    print("Classification rate: %s" % class_rate)

    # by c=Y we assign each point in the scatter a colormap value of its class (Y)
    # plt.scatter(X[:, 0], X[:, 1], c=Y, alpha=0.5)
    # plt.show()


def forward(X, W1, b1, W2, b2):
    """
    Feed forward X in a network with a single hidden layer
    :param X: Input dataset
    :param W1: Weights of first hidden layer
    :param b1: bias for first hidden layer
    :param W2: weights for the output layer (aka V)
    :param b2: bias for the output layer
    :return: Matrix with probabilities of each sample belong to each k (group)
    """

    # First hidden layer, each row has M values per each activation unit (will use sigmoid)
    Z = X.dot(W1) + b1
    Z = 1 / (1 + np.exp(-1 * Z))  # sigmoid

    # Second layer (output layer), each row has K values per each class prediction probability (we'll use softmax)
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)

    return Y


def classification_rate(Y_act, Y_hat):
    """
    Calculate the classification error
    :param Y_act: The actual class of each sample (nX1)
    :param Y_hat: The probability of each sample belongs to each k (nXk)
    :return:
    """

    # find the index of the maximum value per sample
    Y_hat = np.argmax(Y_hat, axis=1)
    class_rate = np.sum(Y_hat == Y_act)
    return class_rate / Y_act.size


if __name__ == '__main__':
    main()

