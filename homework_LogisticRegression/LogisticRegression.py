import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import classification_report

# extract the data
path = 'ex2data1.txt'
data = pd.read_csv(path, names=['first', 'second', 'admission'])
# print(data.head())

# plot the data
positive = data[data['admission'].isin(['1'])]
negative = data[data['admission'].isin(['0'])]
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(positive['first'], positive['second'], c='black', marker='+', label='Admitted')
ax.scatter(negative['first'], negative['second'], s=50, c='yellow', marker='o', label='Not Admitted')
ax.set_xlabel('first score')
ax.set_ylabel('second score')
# 设置图例
plt.legend()
plt.show()

# X y
data.insert(0, 'Ones', 1)  # theta0
col = data.shape[1]
X = np.matrix(data.iloc[:, 0:col - 1].values)
y = np.matrix(data.iloc[:, col - 1:col].values)
theta = np.zeros(X.shape[1])
print('shape', X.shape, theta.shape, y.shape)


# sigmoid function
def sigmoid(z):
    return 1. / (1. + np.exp(-z))


# print(sigmoid(0), sigmoid(10), sigmoid(-10))  # test

# cost function
def cost_function(theta, X, y):
    m = len(X)
    hx = sigmoid(X @ theta).reshape(m, -1)
    fp = (-y).T @ np.log(hx)
    sp = (1 - y).T @ np.log(1 - hx)
    return 1. / m * (fp - sp)


# check the cost_function
print('The value of cost function when using initial theta is', cost_function(theta, X, y))


# gradient
def gradient(theta, X, y):
    m = len(X)
    f1 = sigmoid(X @ theta).reshape(m, -1) - y
    return (1./m) * (X.T @ f1)


# check the gradient function
print('check the gradient:', gradient(theta, X, y))


# using fminunc to calculate
result = opt.minimize(fun=cost_function, x0=theta, args=(X, y), method='TNC', jac=gradient)
print('fminunc result:')
print(result)

# cost using final theta
final_theta = result.x
print('The final cost is', cost_function(final_theta, X, y))

# plot the decision boundary
x1 = np.arange(data['first'].min() - 10, data['first'].max() + 10, step=0.1)
x2 = (-final_theta[0] - final_theta[1] * x1) / final_theta[2]
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(positive['first'], positive['second'], c='black', marker='+', label='Admitted')
ax.scatter(negative['first'], negative['second'], s=50, c='yellow', marker='o', label='Not Admitted')
ax.set_xlabel('first score')
ax.set_ylabel('second score')
ax.set_title('Decision Boundary')
ax.plot(x1, x2)
plt.show()

# predict(evaluate)
x_score = np.matrix([1, 45, 85])
hx = sigmoid(x_score @ final_theta)
print('The admission probability of (45, 85) is', hx)


# 两种精准度测算办法(第一种，数组比对)
def predict(theta, X, y):
    # g(z) >= 0.5 ===> y = 1  or  z(X @ theta) >=0 ===> y = 1
    z = (X @ theta).reshape(len(X), -1)
    predict1 = [1 if a >= 0 else 0 for a in z]
    accuracy = sum([1 if a == b else 0 for a, b in zip(predict1, y)]) / len(X)
    return accuracy, predict1


accuracy, predictions = predict(final_theta, X, y)
print('The accuracy of the model is', accuracy)

# 第二种方法,使用sklearn
print(classification_report(predictions, y))
