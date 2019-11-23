import numpy as np
import matplotlib.pyplot as plt


def main():
    """
    This is back propagation on a simple NN (1 hidden layer)
    """

    nPerK = 500

    D = 2  # number of "features"
    K = 3  # number of classes
    M = 3  # number of activation units in the first hidden layer
    N = nPerK * K

    # create random data for 3 different classes (normally distributed numbers around some avg)
    XK1 = np.random.randn(nPerK, D) + [0, -2]
    XK2 = np.random.randn(nPerK, D) + [2, 2]
    XK3 = np.random.randn(nPerK, D) + [-2, 2]

    # join all classes to one dataset
    X = np.vstack([XK1, XK2, XK3])

    # set the "actual" classes (0, 1, 2)
    Y = np.array([0] * nPerK + [1] * nPerK + [2] * nPerK)

    # 1-hot encode the T matrix (remember that classes in Y are 0 based, so the first class is 0 etc)
    T = np.zeros((N, K))
    for i in range(N):
        T[i, Y[i]] = 1


    # weights and bias for the first hidden layer
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)

    # weights and bias for the output layer
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    costs, class_rate = learn(50000, 10e-7, X, T, W1, b1, W2, b2)
    plt.plot(costs)
    plt.show()


def forward(X, W1, b1, W2, b2):
    """
    Feed forward X in a network with a single hidden layer
    :param X: Input dataset (N*D)
    :param W1: Weights of first hidden layer - DxM
    :param b1: bias for first hidden layer - 1xM
    :param W2: weights for the output layer (aka V) - MxK
    :param b2: bias for the output layer - 1xK
    :return: Matrix with probabilities of each sample belong to each k (NxK) and the hidden layer values (Z)
    """

    # First hidden layer, each row has M values per each activation unit (will use sigmoid)
    Z = X.dot(W1) + b1
    Z = 1 / (1 + np.exp(-1 * Z))  # sigmoid

    # Second layer (output layer), each row has K values per each class prediction probability (we'll use softmax)
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)

    return Y, Z


def learn(epochs, alpha, X, T, W1, b1, W2, b2):
    """
    Learn the NN weights
    :param epochs: Number of iterations
    :param alpha: Learning rate
    :param X: Input dataset - NxD
    :param T: The actual Y value - 1xN
    :param W1: Weights of first hidden layer - DxM
    :param b1: bias for first hidden layer - 1xM
    :param W2: weights for the output layer (aka V) - MxK
    :param b2: bias for the output layer - 1xK
    :return:
    """

    costs = []
    class_rate = 0
    for i in range(epochs):
        # Recall dimensions:
        # Y - NxK
        # Z - NxM
        Y, Z = forward(X, W1, b1, W2, b2)

        if i % 100 == 0:
            class_rate = classification_rate(T, Y)
            cost = (T * np.log(Y)).sum()
            costs.append(cost)
            print("epoch %s classification rate: %s" % (i, class_rate))

        # grads of W2 and b2
        W2Grads = Z.T.dot(T - Y)
        b2Grads = np.sum(T - Y, axis=0)

        W2 = W2 + alpha * W2Grads
        b2 = b2 + alpha * b2Grads

        # grads of W1 and b1
        temp = ((T - Y).dot(W2.T)) * Z * (1 - Z)  # NxM
        W1Grads = X.T.dot(temp)  # Result: DxM
        b1Grads = np.sum(temp, axis=0)

        W1 = W1 + alpha * W1Grads
        b1 = b1 + alpha * b1Grads

    return costs, class_rate


def classification_rate(T, Y):
    """
    Calculate the classification error
    :param T: The actual class of each sample 1-hot encoded (nXk)
    :param Y: The probability of each sample belongs to each k (nXk)
    :return:
    """

    # find the index of the maximum value per sample
    P = np.argmax(Y, axis=1)
    K = np.argmax(T, axis=1)
    class_rate = np.sum(P == K)
    return class_rate / T.shape[0]


if __name__ == '__main__':
    main()
