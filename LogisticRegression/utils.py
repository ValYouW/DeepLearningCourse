import numpy as np
import matplotlib.pyplot as plt


def plot_loss(loss, reg_loss=np.array([]), title='Loss'):
	if reg_loss.size == 0:
		reg_loss = np.zeros(loss.shape)

	fig, ax1 = plt.subplots()
	plt.title(title)

	color = 'tab:red'
	ax1.set_xlabel('epoch')
	ax1.set_ylabel('loss', color=color)
	ax1.plot(loss - reg_loss, color=color)
	ax1.tick_params(axis='y', labelcolor=color)

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	color = 'tab:blue'
	ax2.set_ylabel('reg loss', color=color)  # we already handled the x-label with ax1
	ax2.plot(reg_loss, color=color)
	ax2.tick_params(axis='y', labelcolor=color)

	fig.tight_layout()  # otherwise the right y-label is slightly clipped


def map_features(x1, x2, deg, add_intercept=True):
	"""
	Return a matrix with x1 and x2 and all combination of x1 and x2 up to the "deg" degree
	e/g: X1, X2, X1^2, X2^2, X1*X2, X1*X2^2, etc..
	:param x1: Array of features
	:param x2: Array of features
	:param deg: The polynomial degree to map the features to
	:param add_intercept: whether to add the intercept column to the result
	:return: Matrix
	"""
	x1 = x1.reshape((x1.shape[0], 1))
	x2 = x2.reshape((x2.shape[0], 1))
	out = np.ones((x1.shape[0], 1))
	for i in range(1, deg + 1):
		for j in range(0, i + 1):
			new_col = (x1 ** (i - j)) * (x2 ** j)
			out = np.insert(out, [out.shape[1]], new_col, axis=1)

	if not add_intercept:
		return out[:, 1:]

	return out


def sigmoid(z):
	return 1 / (1 + np.exp(-1 * z))


def calc_loss(x_mat, y, theta, _lambda=0):
	"""
	Calculate the loss and gradients per given theta array and regularization factor "lambda"
	:param x_mat: shape is (m,n)
	:param y: shape is (m,1)
	:param theta: shape is (n,1) where n is number of features
	:param _lambda: regularization term
	:return: loss, grads
	"""
	# h(i) = sigmoid(theta0*x0 + theta1*x1 + theta2*x2 + ... thetaN*xN)
	# Loss = J = -1/m * sum[ y(i) * log(h(i)) + (1 - y(i)) * log(1 - h(i)) ] + (lambda/2m) * sum[theta(j)^2]
	# Note in the above that theta(0) is not regularized
	m = x_mat.shape[0]
	hi = sigmoid(x_mat.dot(theta))

	# We break the component in the Sum of the loss function into 2 parts: left and right.
	left = y * np.log(hi)
	right = (1 - y) * (np.log(1 - hi))

	# Finally compute J.
	j = -1 * (1/m) * np.sum(left + right)

	reg_loss = 0
	if _lambda > 0:
		reg_loss = ((_lambda / 2 * m) * np.sum(theta[1:] ** 2))  # theta[0] should not regularized

	j += reg_loss

	# Calculate the partial derivative of loss per each theta (j)
	# dJ/dTheta(j) = 1/m * sum[ (h(i) - y(i)) * x(j,i) ]
	# basically we sum the prediction error per sample multiplied by the j's feature, and then divide by m
	# this calculate what is the "contribution" of each feature to the total error
	# grads shape is (n, 1)
	grads = (1 / m) * x_mat.T.dot(hi - y)

	if _lambda > 0:
		grads += ((_lambda / m) * theta)
		grads[0] -= ((_lambda / m) * theta[0])  # theta[0] is not regularized so we undo it only for it

	return j, reg_loss, grads


def learn(x_mat, y, epochs, alpha, _lambda=0):
	lost_arr = np.zeros(epochs)
	reg_loss_arr = np.zeros(epochs)
	(m, n) = x_mat.shape
	theta = np.zeros((n, 1))
	for e in range(epochs):
		(lost, reg_loss, grads) = calc_loss(x_mat, y, theta, _lambda)
		lost_arr[e] = lost
		reg_loss_arr[e] = reg_loss
		theta = theta - (alpha * grads)

	return lost_arr, reg_loss_arr, theta

