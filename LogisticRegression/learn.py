import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def plotData(X, y, descisionBoundary):
	plt.figure()
	plt.title('Data')
	admitted = (y == 1).flatten()
	rejected = (y == 0).flatten()

	# plot descision boundary
	plt.plot([descisionBoundary[0], descisionBoundary[2]], [descisionBoundary[1], descisionBoundary[3]])

	# plot admitted
	plt.scatter(X[admitted, 0], X[admitted, 1], color='blue', marker='+')

	# plot rejected
	plt.scatter(X[rejected, 0], X[rejected, 1], edgecolors='red', facecolors='none', marker='o')

	plt.xlabel('exam 1 score')
	plt.ylabel('exam 2 score')
	plt.legend(['boundary', 'admitted', 'rejected'])

def plotCosts(costs):
	plt.figure()
	plt.title('Learning loss')
	plt.plot(costs)
	plt.xlabel('iter')
	plt.ylabel('loss')

def sigmoid(z):
	return 1 / (1 + np.exp(-1 * z))

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
	grads = (1 / m) * X.T.dot(hi - y)

	return (J, grads)

def learn(X, y, epochs, alpha):
	costs = np.zeros(epochs)
	(m, n) = X.shape
	theta = np.zeros((n, 1))
	for e in range(epochs):
		(cost, grads) = calcCost(X, y, theta)
		costs[e] = cost
		theta = theta - (alpha * grads);

	return (costs, theta)

print('Loading and visualize dataset...')
# data is: exam 1 score, exam 2 score, bool whether admitted
frame = pd.read_csv('ex2data1.csv', header=None)
data = frame.values
X = data[:, 0:2] # exam scores
y = data[:, 2:3] # admitted or not

# normalize input (input has large values which causes sigmoid to always be 1 or 0)
x_mean = np.mean(X, axis = 0)
x_std = np.std(X, axis=0)
Xnorm = (X - x_mean) / x_std

# add intercept
Xnorm = np.insert(Xnorm, 0, 1, axis = 1)

# Learn model
print('starting to learn...')
(costs, theta) = learn(Xnorm, y, 5000, 0.1)
print('Final cost %s' % costs[-1])
print('Final theta \n%s' % theta)

# predict for student
joe = np.array([[45, 85]])
joeNorm = (joe - x_mean) / x_std
joeNorm = np.insert(joeNorm, 0, 1, axis = 1)
p = sigmoid(joeNorm.dot(theta))
print('Student with grades %s and %s has admission probability: %s' % (45, 85, p[0, 0]))

# Predict on train set
XpredictAdmit = (sigmoid(Xnorm.dot(theta)) >= 0.5)
yActualAdmit = (y == 1)
predictSuccess = np.sum(XpredictAdmit == yActualAdmit)
print('Model evaluation on training set has success of %s/%s' % (predictSuccess, y.shape[0]))

# calc descision boundary
# The descision boundary is the threshold line that seperates true/false predictions,
# this means that on this line the prediction is exactly 0.5, meaning:
# p = sigmoid(X.dot(theta)) = 0.5 ====> X.dot(theta) = 0
# so our line equation is: theta0 + theta1*x1 + theta2*x2 = 0
# x2 = -theta0 / theta2 - (theta1/theta2)*x1
theta = theta.flatten()

# calc 2 points on the line
minX = np.min(Xnorm[:,1])
maxX = np.max(Xnorm[:,1])
y_minX = -1 * (theta[0] / theta[2]) - (theta[1] / theta[2]) * minX # y value when x = minX
y_maxX = -1 * (theta[0] / theta[2]) - (theta[1] / theta[2]) * maxX

# denormalize the points
minX = minX * x_std[0] + x_mean[0]
maxX = maxX * x_std[0] + x_mean[0]
y_minX = y_minX * x_std[1] + x_mean[1]
y_maxX = y_maxX * x_std[1] + x_mean[1]

plotData(X, y, [minX, y_minX, maxX, y_maxX])
plotCosts(costs)

plt.show()