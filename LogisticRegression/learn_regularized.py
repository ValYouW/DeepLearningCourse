import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils


def plot_data(x_mat, y):
    plt.figure()
    admitted = (y == 1).flatten()
    rejected = (y == 0).flatten()

    # plot admitted
    plt.scatter(x_mat[admitted, 0], x_mat[admitted, 1], color='blue', marker='+')

    # plot rejected
    plt.scatter(x_mat[rejected, 0], x_mat[rejected, 1], edgecolors='red', facecolors='none', marker='o')

    plt.xlabel('test 1 score')
    plt.ylabel('test 2 score')
    plt.legend(['admitted', 'rejected'])


def run(x_norm, y, x_mean, x_std, _lambda):
    # Learn model
    print('starting to learn with lambda=%s...' % _lambda)
    (loss, reg_loss, theta) = utils.learn(x_norm, y, 5000, 0.1, _lambda)
    print('Final loss %s' % loss[-1])
    print('Final theta \n%s' % theta)

    utils.plot_loss(loss, reg_loss, 'lambda=' + str(_lambda))

    # Create the decision boundary, we create a plane filled with 100 points ranging from -1 to 1.5 on both axes
    # then for each point we calculate the model value (the "z" that goes to the sigmoid), then the decision
    # boundary is the contour where values are changed from negative to positive (e.g like edge detection)
    print('Visualizing decision boundary')
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    plane = np.zeros((u.size, v. size))
    for i in range(u.size):
        for j in range(v.size):
            feats = utils.map_features(u[i:i + 1], v[j:j + 1], 6, False)
            feats_norm = (feats - x_mean) / x_std
            feats_norm = np.insert(feats_norm, 0, 1, axis=1)
            plane[i, j] = feats_norm.dot(theta)

    return plane


def main():
    print('Loading dataset...')
    # data is: microchip test 1 score, microchip test 2 score, microchip accepted/rejected
    frame = pd.read_csv('ex2data2.csv', header=None)
    data = frame.values
    x_mat = data[:, 0:2]  # test scores
    y = data[:, 2:3]  # accepted/rejected

    # augment features with polynomials
    x_feats = utils.map_features(x_mat[:, 0], x_mat[:, 1], 6, False)

    # normalize input (input has large values which causes sigmoid to always be 1 or 0)
    x_mean = np.mean(x_feats, axis=0)
    x_std = np.std(x_feats, axis=0)
    x_norm = (x_feats - x_mean) / x_std

    # add back intercept
    x_norm = np.insert(x_norm, 0, 1, axis=1)

    # Learn model
    plane_lambda_1 = run(x_norm, y, x_mean, x_std, 1)
    plane_lambda_100 = run(x_norm, y, x_mean, x_std, 100)

    # res, img = cv2.threshold(plane, 0, 1, cv2.THRESH_BINARY)
    # cv2.imshow("XXX", img)
    # cv2.waitKey(0)

    plot_data(x_mat, y)
    plt.title('Lambda = 1')
    u = np.linspace(-1, 1.5, 50)
    plt.contour(u, u, plane_lambda_1.T, 0)

    plot_data(x_mat, y)
    plt.title('Lambda = 100')
    u = np.linspace(-1, 1.5, 50)
    plt.contour(u, u, plane_lambda_100.T, 0)

    plt.show()


if __name__ == '__main__':
    main()
