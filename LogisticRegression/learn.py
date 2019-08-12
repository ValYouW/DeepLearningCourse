import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def plotData(X, y):
	admitted = (y == 1).flatten()
	rejected = (y == 0).flatten()

	# plot admitted
	plt.scatter(X[admitted, 0], X[admitted, 1], color='blue', marker='+')

	# plot rejected
	plt.scatter(X[rejected, 0], X[rejected, 1], edgecolors='red', facecolors='none', marker='o')

	plt.xlabel('exam 1 score')
	plt.ylabel('exam 2 score')
	plt.legend(['admitted', 'rejected'])

	plt.show()

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def calcCost(X, y, theta):
	'''
	Calculate the cost and gradients per given theta array.
	X shape is (m,n)
	y shape is (m,1)
	theta shape is (n,1)
	'''
	
	# h(i) = sigmoid(theta0*x0 + theta1*x1 + theta2*x2 + ... thetaN*xN)
	# Cost = J = -1/m * sum[ y(i) * log(h(i)) + (1 - y(i)) * log(1 - h(i)) ]
	m = X.shape[0]
	hi = sigmoid(X.dot(theta))

	# We break the component in the Sum of the cost function into 2 parts: left and right.
	left = y * np.log(hi);
	right = (1 - y) * (np.log(1 - hi));

	# Finally compute J.
	J = -1 * (1/m) * np.sum(left + right);

	# Calculate the partial derivative of cost per each theta (j)
	# dJ/dTheta(j) = 1/m * sum[ (h(i) - y(i)) * x(j,i) ]
	# basically we sum the prediction error per sample multiplied by the j's feature, and then divide by m
	# this calculate what is the "contribution" of each feature to the total error
	grad = (1 / m) * X.T.dot(hi - y)

	return (J, grad)

print('Loading and visualize dataset...')
# data is: exam 1 score, exam 2 score, bool whether admitted
frame = pd.read_csv('ex2data1.txt', header=None)
data = frame.values
X = data[:, 0:2] # exam scores
y = data[:, 2:3] # admitted or not

## plotData(X, y)

# add intercept
(m, n) = X.shape
X = np.insert(X, 0, 1, axis = 1)
theta = np.zeros((n + 1, 1))
(cost, grad) = calcCost(X, y, theta)

print('cost %s' % cost)
print('grad \n%s' % grad)
