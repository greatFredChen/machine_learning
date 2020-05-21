import numpy as np
from scipy.io import loadmat

data = loadmat('ex3data1.mat')
theta_matrix = loadmat('ex3weights.mat')
X = data['X']
y = data['y']
theta1 = theta_matrix['Theta1']
theta2 = theta_matrix['Theta2']

y = y.flatten()
print(np.unique(y))
X = np.insert(X, 0, 1, axis=1)
print(X.shape, y.shape)
print(theta1.shape, theta2.shape)


# sigmoid
def sigmoid(z):
    return 1. / (1. + np.exp(-z))


# forward propagation
a1 = X
z2 = a1 @ theta1.T
# print(z2.shape)
a2 = sigmoid(z2)
a2 = np.insert(a2, 0, 1, axis=1)
# print(a2.shape)
z3 = a2 @ theta2.T
a3 = sigmoid(z3)   # a3 is output
# print(a3.shape)

# prediction
lmax = np.argmax(a3, axis=1)
y_predict = lmax + 1
accuracy = sum([1 if a == b else 0 for a, b in zip(y_predict, y)]) / len(X)
print(accuracy)
