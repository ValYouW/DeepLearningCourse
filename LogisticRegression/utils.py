import numpy as np
import matplotlib.pyplot as plt

def plotLoss(loss):
	plt.figure()
	plt.title('Learning loss')
	plt.plot(loss)
	plt.xlabel('iter')
	plt.ylabel('loss')

def sigmoid(z):
	return 1 / (1 + np.exp(-1 * z))

def calcLoss(X, y, theta):
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
	grads = (1 / m) * X.T.dot(hi - y)

	return (J, grads)

def learn(X, y, epochs, alpha):
	costs = np.zeros(epochs)
	(m, n) = X.shape
	theta = np.zeros((n, 1))
	for e in range(epochs):
		(cost, grads) = calcLoss(X, y, theta)
		costs[e] = cost
		theta = theta - (alpha * grads);

	return (costs, theta)

