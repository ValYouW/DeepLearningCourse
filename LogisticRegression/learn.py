import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils


def plot_data(x_mat, y, db_x, db_y):
    plt.figure()
    plt.title('Data')
    admitted = (y == 1).flatten()
    rejected = (y == 0).flatten()

    # plot decision boundary
    plt.plot(db_x, db_y)

    # plot admitted
    plt.scatter(x_mat[admitted, 0], x_mat[admitted, 1], color='blue', marker='+')

    # plot rejected
    plt.scatter(x_mat[rejected, 0], x_mat[rejected, 1], edgecolors='red', facecolors='none', marker='o')

    plt.xlabel('exam 1 score')
    plt.ylabel('exam 2 score')
    plt.legend(['boundary', 'admitted', 'rejected'])


def main():
    print('Loading dataset...')
    # data is: exam 1 score, exam 2 score, bool whether admitted
    frame = pd.read_csv('ex2data1.csv', header=None)
    data = frame.values
    x_mat = data[:, 0:2]  # exam scores
    y = data[:, 2:3]  # admitted or not

    # normalize input (input has large values which causes sigmoid to always be 1 or 0)
    x_mean = np.mean(x_mat, axis=0)
    x_std = np.std(x_mat, axis=0)
    x_norm = (x_mat - x_mean) / x_std

    # add intercept
    x_norm = np.insert(x_norm, 0, 1, axis=1)

    # Learn model
    print('starting to learn...')
    (loss, reg_loss, theta) = utils.learn(x_norm, y, 5000, 0.1)
    print('Final loss %s' % loss[-1])
    print('Final theta \n%s' % theta)

    # predict for student
    joe = np.array([[45, 85]])
    joe_norm = (joe - x_mean) / x_std
    joe_norm = np.insert(joe_norm, 0, 1, axis=1)
    p = utils.sigmoid(joe_norm.dot(theta))
    print('Student with grades %s and %s has admission probability: %s' % (45, 85, p[0, 0]))

    # Predict on train set
    prediction = (utils.sigmoid(x_norm.dot(theta)) >= 0.5)
    actual = (y == 1)
    predict_success = np.sum(prediction == actual)
    print('Model evaluation on training set has success of %s/%s' % (predict_success, y.shape[0]))

    # calc decision boundary
    # The decision boundary is the threshold line that separates true/false predictions,
    # this means that on this line the prediction is exactly 0.5, meaning:
    # p = sigmoid(x_mat.dot(theta)) = 0.5 ====> x_mat.dot(theta) = 0
    # so our line equation is: theta0 + theta1*x1 + theta2*x2 = 0
    # x2 = -theta0 / theta2 - (theta1/theta2)*x1
    theta = theta.flatten()

    # calc 2 points on the line
    plot_x = np.array([np.min(x_norm[:, 1]), np.max(x_norm[:, 1])])
    plot_y = -1 * (theta[0] / theta[2]) - (theta[1] / theta[2]) * plot_x

    # denormalize the points
    plot_x = plot_x * x_std[0] + x_mean[0]
    plot_y = plot_y * x_std[1] + x_mean[1]

    plot_data(x_mat, y, plot_x, plot_y)
    utils.plot_loss(loss)

    plt.show()


if __name__ == '__main__':
    main()
