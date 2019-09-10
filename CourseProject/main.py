import numpy as np
import utils


def main():
    # read the data
    X, Y = utils.get_data()

    M = 5  # hidden layers
    D = X.shape[1]  # num input features
    K = len(set(Y))  # num of classes

    W1 = np.random.rand(D, M)
    b1 = np.zeros(M)

    W2 = np.random.rand(M, K)
    b2 = np.zeros(K)

    probs = forward(X, W1, b1,W2, b2)
    Y_hat = np.argmax(probs, axis=1)
    correct = np.sum(Y_hat == Y)
    class_rate = correct / Y.size
    print("Classification rate for dummy weights: %s" % class_rate)


def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)


def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    A = Z.dot(W2) + b2
    return softmax(A)






if __name__ == '__main__':
    main()
