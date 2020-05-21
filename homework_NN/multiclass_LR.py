import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt

# extract the data
data = loadmat('ex3data1.mat')
X = data['X']
y = data['y']


# visualizing the data
def plotdigit(X):
    # choose row randomly
    random_row = np.random.choice(5000, 100, False)
    random_X = X[random_row, :]
    fig, ax = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True, figsize=(8, 8))
    for r in np.arange(10):
        for c in np.arange(10):
            ax[r, c].matshow(random_X[10 * r + c].reshape(20, 20), cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()


plotdigit(X)

# 处理X和y
y = y.reshape(len(X), -1)
print(np.unique(y))
X = np.insert(X, 0, 1, axis=1)  # 插入第0列，值为1
print(X.shape, y.shape)


# sigmoid
def sigmoid(z):
    return 1. / (1. + np.exp(-z))


# cost function with regularization
def cost_function(theta, X, y, r):
    m = len(X)
    thetaReg = theta[1:].reshape(X.shape[1] - 1, 1)  # 正则化项不需要theta0
    hx = sigmoid(X @ theta).reshape(m, -1)
    fp = (-y).T @ np.log(hx)
    sp = (1 - y).T @ np.log(1 - hx)
    reg = (thetaReg.T @ thetaReg) * r / (2 * m)  # reg = (r/2m) * reg^2
    # print(fp - sp + reg)
    return (fp - sp) / m + reg


# gradient with regularization
def gradient(theta, X, y, r):
    thetaReg = theta[1:]
    m = len(X)
    f1 = sigmoid(X @ theta).reshape(m, -1) - y
    # thetaReg (400, 1)  reg (401, 1) 记得拼接一个值为0的维度，取消theta0的惩罚
    # 利用np.concatenate进行拼接，默认axis=0(行),即0在第一行
    reg = np.concatenate([np.array([0]), (r / m) * thetaReg], axis=0).reshape(-1, 1)
    return (X.T @ f1) / m + reg


# one-vs-all classification  k means number of labels
def one_vs_all(X, y, r, k):
    final_theta = np.zeros((k, X.shape[1]))
    for i in range(1, k + 1):
        theta = np.zeros(X.shape[1])
        # label class k to y=1 ,others =0
        _y = np.array([1 if a == i else 0 for a in y]).reshape(len(X), 1)
        result = opt.minimize(fun=cost_function, x0=theta, args=(X, _y, r), method='TNC',
                              jac=gradient, options={'disp': True})
        # print(result)
        final_theta[i-1, :] = result.x
    return final_theta


# get final theta
final_theta = one_vs_all(X, y, 1, 10)
print(final_theta)  # (10, 401)

# prediction
def predict(X, theta):
    hy = sigmoid(X @ theta.T)  # (5000, 10)
    # 比较找出每行最大的数的索引，该索引+1即为我们所需的label
    # 使用np.argmax()来寻找最大数值的索引  axis=0 列最大   axis=1 行最大
    lmax = np.argmax(hy, axis=1)  # 此时返回的是label - 1
    return lmax + 1


# calculate the accuracy
y_predict = predict(X, final_theta)
accuracy1 = sum([1 if a == b else 0 for a, b in zip(y_predict, y)]) / len(X)
print('accuracy:', accuracy1)
