import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm

data1 = loadmat('ex6data1.mat')
X = data1['X']
y = data1['y']


# 画散点图
def plot(X, y):
    positive = X[[i for i in range(len(y)) if y[i] == 1]]
    negative = X[[i for i in range(len(y)) if y[i] == 0]]
    plt.figure(figsize=(8, 6))
    plt.scatter(positive[:, 0], positive[:, 1], c='black', marker='+', label='positive')
    plt.scatter(negative[:, 0], negative[:, 1], c='yellow', marker='o', label='negative')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()


# 画边界
def plot_boundary(model, X, title):
    minx, maxx = X[:, 0].min() * 1.1, X[:, 1].max() * 1.2
    miny, maxy = X[:, 1].min() * 1.1, X[:, 1].max() * 1.2
    plotx, ploty = np.meshgrid(np.linspace(minx, maxx, 500),
                               np.linspace(miny, maxy, 500))
    z = model.predict(X=np.c_[plotx.flatten(), ploty.flatten()])
    z = z.reshape(plotx.shape)
    plt.contour(plotx, ploty, z)
    plt.title(title)
    plt.show()


# 散点图
plot(X, y)
plt.title('scatter for linear model')
plt.show()
# 模型构造
models = [svm.SVC(C, kernel='linear') for C in [1, 100]]
res = [model.fit(X, y.flatten()) for model in models]
title = ['SVM for C = {}'.format(C) for C in [1, 100]]
for model, title in zip(models, title):
    plot(X, y)
    plot_boundary(model, X, title)


# 高斯核函数
def gauss_kernel(x1, x2, sigma):
    return np.exp(-((x1 - x2) ** 2).sum() / (2 * (sigma ** 2)))


# 检验高斯核函数
print(gauss_kernel(np.array([1, 2, 1]),np.array([0, 4, -1]), 2.))

# 非线性边界SVM
data2 = loadmat('ex6data2.mat')
X2 = data2['X']
y2 = data2['y']
plot(X2, y2)
plt.title('scatter for non-linear model')
plt.show()
# 非线性边界SVM模型
sigma = 0.1
# 高斯核函数: np.exp(-((x1 - x2) ** 2).sum() * gamma)
gamma = np.power(sigma, -2.) / 2.
model = svm.SVC(C=1, kernel='rbf', gamma=gamma)
fit_model = model.fit(X2, y2.flatten())
plot(X2, y2.flatten())
plot_boundary(fit_model, X2, 'SVM non-linear boundary model')

# C和sigma选择
data3 = loadmat('ex6data3.mat')
X3 = data3['X']
y3 = data3['y']
Xval = data3['Xval']
yval = data3['yval']
plot(X3, y3)
plt.title('dataset3 scatter')
plt.show()

Carray = [0.01, 0.03, 0.1, 0.3, 1., 3., 10., 30.]
sigmaarray = Carray
best_pair, best_score = (0., 0.), 0.

for C in Carray:
    for sigma in sigmaarray:
        gamma = np.power(sigma, -2.) / 2.
        model = svm.SVC(C=C, gamma=gamma, kernel='rbf')
        model.fit(X3, y3.flatten())
        score = model.score(Xval, yval)
        if score > best_score:
            best_score = score
            best_pair = (C, sigma)
print('best pair is {}, best score is {}'.format(best_pair, best_score))
best_gamma = np.power(best_pair[1], -2.) / 2.
model = svm.SVC(C=best_pair[0], gamma=best_gamma, kernel='rbf')
model.fit(X3, y3.flatten())
plot(X3, y3)
plot_boundary(model, X3, 'SVM boundary for dataset3')
