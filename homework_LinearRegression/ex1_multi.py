# multiple variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# extract the data
path = "ex1data2.txt"
data2 = pd.read_csv(path, names=['Size', 'Bedrooms', 'Price'])
print(data2.head())  # check the data

# feature scaling (feature normalization)
data_scaling = (data2 - data2.mean()) / data2.std()
print(data_scaling.head())  # check the data

#
data_scaling.insert(0, 'Ones', 1)
col = data_scaling.shape[1]
X = data_scaling.iloc[:, 0:col-1]
y = data_scaling.iloc[:, col-1:col]
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix([0., 0., 0.])
alpha = 0.01
iterations = 1500


# compute cost function
def costfunction(X, y, theta):
    return np.sum(np.power((X * theta.T - y), 2)) / float(2 * len(X))


# compute the cost with initial theta
print("The cost size for initial theta is", costfunction(X, y, theta))


# gradient descent
def gradient_descent(X, y, theta, alpha, iterations):
    # gradient descent
    cost = np.zeros(iterations)  # 记录每次迭代时cost function的值
    m = len(X)

    # theta(j) = theta(j) - a/m * sum(hx - y)x(j) j对应列 i对应行(对行求和)
    for i in range(iterations):
        theta = theta - alpha / m * (X * theta.T - y).T * X  # 向量化 对行求和
        cost[i] = costfunction(X, y, theta)
    return theta, cost


f_theta, f_cost = gradient_descent(X, y, theta, alpha, iterations)
print("final theta is", f_theta)

# plot the cost
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(np.arange(iterations), f_cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Cost and iterations')
plt.show()

# predicted price
size = (1650. - data2.mean()['Size']) / data2.std()['Size']
bedrooms = (3 - data2.mean()['Bedrooms']) / data2.std()['Bedrooms']
inputs = np.matrix([1., size, bedrooms])
prices = (inputs * f_theta.T) * data2.std()['Price'] + data2.mean()['Price']
print("The predicted price is", prices)