import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# extract the data
data_path = 'ex1data1.txt'
data = pd.read_csv(data_path, header=None, names=['Population', 'Profit'])
# check the data
print(data.head())
# draw the plot
data.plot(kind='scatter', x='Population', y='Profit', figsize=(10, 6), use_index=True)
plt.show()

# Gradient Descent
# insert x0 feature
data.insert(0, 'Ones', 1)
theta = np.matrix([0, 0])
alpha = 0.01
iterations = 1500

# from data extract X and y
col = data.shape[1]
X = data.iloc[:, 0:col - 1]
X = np.matrix(X.values)
y = data.iloc[:, col - 1:col]
y = np.matrix(y.values)


# compute cost function
def costfunction(X, y, theta):
    return np.sum(np.power((X * theta.T) - y, 2)) / float(2 * len(X))


# compute the cost with initial theta
print("The cost size for initial theta is", costfunction(X, y, theta))


def gradient_descent(X, y, theta, alpha, iterations):
    # gradient descent
    cost = np.zeros(iterations)  # 记录每次迭代时cost function的值
    m = len(X)

    # theta(j) = theta(j) - a/m * sum(hx - y)x(j) j对应列 i对应行(对行求和)
    for i in range(iterations):
        theta = theta - alpha / m * (X * theta.T - y).T * X  # 向量化 对行求和
        cost[i] = costfunction(X, y, theta)
    return theta, cost


# final result
f_theta, f_cost = gradient_descent(X, y, theta, alpha, iterations)
print("The theta vector is", f_theta)
predict1 = np.matrix([1., 3.5]) * f_theta.T
predict2 = np.matrix([1., 7.]) * f_theta.T
print("predict1:", predict1)
print("predict2:", predict2)

# plot the line
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = f_theta[0, 0] + (f_theta[0, 1] * x)  # f代表y值

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data['Population'], data['Profit'], label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('scatter and line')
plt.show()

# visualizing cost
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.arange(iterations), f_cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Cost and Iterations')
plt.show()
