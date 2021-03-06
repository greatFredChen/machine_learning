import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.optimize as opt
from copy import deepcopy


class Data:
    def __init__(self, path):
        self.path = path
        self.data = loadmat(self.path)
        self.rawX = self.data['X']
        self.y = self.data['y']
        self.rawXtest = self.data['Xtest']
        self.ytest = self.data['ytest']
        self.rawXval = self.data['Xval']
        self.yval = self.data['yval']
        # 处理生成数据集X Xval
        self.X = np.insert(self.rawX, 0, 1, axis=1)  # 往第0列插入1
        self.Xval = np.insert(self.rawXval, 0, 1, axis=1)
        self.Xtest = np.insert(self.rawXtest, 0, 1, axis=1)

    def print_data(self):
        print(self.X)


class PlotData:
    def __init__(self):
        self.LR_Model = LinearRegressionMethod()

    # 数据可视化 原始散点图
    def plot_scatter(self, X, y):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(X, y, c='red', marker='x')
        ax.set_xlabel('Change in water level(x)')
        ax.set_ylabel('Water flowing out of the dam(y)')

    def plot_line(self, X, fit_theta):
        fit_theta = fit_theta.reshape(-1, 1)
        plt.plot(X[:, 1], X @ fit_theta)

    def print_learning_curves(self, training_set_error, cv_set_error):
        plt.figure(figsize=(7, 6))
        # 两个数组的长度等于len(X)
        axis_x = range(1, len(cv_set_error) + 1)
        plt.plot(axis_x, training_set_error, label='Train')
        plt.plot(axis_x, cv_set_error, label='Cross Validation')
        plt.legend()  # 生成图例
        plt.xlabel('Number of training examples')
        plt.ylabel('Error')

    def plot_polynomial(self, poly_theta, mean, std, power):
        x = np.linspace(-75, 55, 50)  # 创建等差数列 [-75, ..., 50]
        axis_x = x.reshape(-1, 1)
        axis_x = np.insert(axis_x, 0, 1, axis=1)
        axis_x_nor = self.LR_Model.Normalization(self.LR_Model.GenPolynomial(axis_x, power),
                                             mean,
                                             std)
        poly_theta = poly_theta.reshape(-1, 1)
        # plot the curve
        plt.plot(x, axis_x_nor @ poly_theta, 'b--')

    def print_learning_curve_lambda(self, lambda_set, train_error_set, cv_error_set):
        plt.figure(figsize=(7, 6))
        # 两个数组的长度等于len(X)
        plt.plot(lambda_set, train_error_set, label='Train')
        plt.plot(lambda_set, cv_error_set, label='Cross Validation')
        plt.legend()  # 生成图例
        plt.xlabel('lambda')
        plt.ylabel('Error')


class LinearRegressionMethod:
    def cost_function(self, theta, X, y):
        m = len(X)
        hx = (X @ theta).flatten()
        return np.sum(np.power(hx - y, 2)) / (2 * m)

    def regularized_cost_function(self, theta, X, y, l=1.0):
        m = len(X)
        cost = self.cost_function(theta, X, y)
        _theta = theta[1:]
        penalty = l * np.sum(_theta * _theta) / (2 * m)
        return cost + penalty

    def gradient(self, theta, X, y):
        m = len(X)
        hx = (X @ theta).flatten()
        return ((hx - y) @ X) / m

    def regularized_gradient(self, theta, X, y, l=1.0):
        m = len(X)
        gradient = self.gradient(theta, X, y)  # (1, 2)
        penalty = l * theta / m
        penalty[0] = 0
        return gradient + penalty

    def get_linear_model(self, theta, X, y, l=0.0):
        res = opt.fmin_cg(f=self.regularized_cost_function, x0=theta,
                          fprime=self.regularized_gradient,
                          args=(X, y, l),
                          disp=False)
        return res

    # training set error and cross validation set error
    def train_error(self, X, y, cvX, cvy, l=0.0):
        # training set use subset
        # cross validation set use entire set
        m = len(X)
        training_error_set, cv_error_set = [], []
        for i in range(1, m + 1):
            tmp_theta = np.ones((X.shape[1],), dtype=float)
            subX = X[:i, :]  # 取前i个样本
            suby = y[:i]
            tmp_fit_theta = self.get_linear_model(tmp_theta, subX, suby.flatten(), l)
            # compute training set error
            training_error = self.cost_function(tmp_fit_theta, subX, suby.flatten())
            training_error_set.append(training_error)
            # compute cross validation set error
            cv_error = self.cost_function(tmp_fit_theta, cvX, cvy.flatten())
            cv_error_set.append(cv_error)
        return training_error_set, cv_error_set

    def random_train_error(self, X, y, cvX, cvy, l=0.0):
        # 交叉验证用全集
        m = len(X)
        random_train_error, random_cv_error = [], []
        for i in range(1, m + 1):
            tmp_train, tmp_cv = [], []
            for times in range(50):
                tmp_theta = np.ones((X.shape[1],), dtype=float)
                random_row = np.random.choice(m, i, replace=False)
                subX = X[random_row]
                suby = y[random_row]
                tmp_fit_theta = self.get_linear_model(tmp_theta, subX, suby.flatten(), l)
                # compute training set error
                training_error = self.cost_function(tmp_fit_theta, subX, suby.flatten())
                tmp_train.append(training_error)
                # compute cross validation set error
                cv_error = self.cost_function(tmp_fit_theta, cvX, cvy.flatten())
                tmp_cv.append(cv_error)
            random_train_error.append(np.mean(tmp_train))
            random_cv_error.append(np.mean(tmp_cv))
        return random_train_error, random_cv_error

    def GenPolynomial(self, X, power):
        polyX = deepcopy(X)
        for i in range(2, power + 1):
            polyX = np.insert(polyX, i, np.power(X[:, 1], i), axis=1)  # insert生成副本..
        return polyX  # 返回多项式X

    # 注意:样本标准差！
    def mean_std(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0, ddof=1)  # ddof = 1 才是样本标准差!默认为总体标准差
        return mean, std

    def Normalization(self, X, mean, std):
        norX = deepcopy(X)
        norX[:, 1:] = norX[:, 1:] - mean[1:]
        norX[:, 1:] = norX[:, 1:] / std[1:]
        return norX


data = Data('ex5data1.mat')  # 数据类实例
plot_data = PlotData()  # 画图类实例
LR_Model = LinearRegressionMethod()  # 方法类实例
# 画散点图
plot_data.plot_scatter(data.rawX, data.y.flatten())
plt.show()
# 测试数据
# data.print_data()
theta = np.ones((data.X.shape[1],), dtype=float)
print('The regularized cost of initial theta is {}'.format(
    LR_Model.regularized_cost_function(
        theta, data.X, data.y.flatten(), 1)))
print('The regularized gradient of initial theta is {}'.format(
    LR_Model.regularized_gradient(
        theta, data.X, data.y.flatten(), 1)))
fit_theta = LR_Model.get_linear_model(theta, data.X, data.y.flatten(), 0)
print('linear mode theta: {}'.format(fit_theta))
# plot the line for linear model
plot_data.plot_scatter(data.rawX, data.y)
plot_data.plot_line(data.X, fit_theta)
plt.title('Linear regression(lambda = 0.000000)')
plt.show()
# pdf中没说在这步的lambda到底用什么值..只是说会作为参数传递给绘制函数..
# 所以这里的lambda统一用0 本来就underfit了，用0
training_error_set, cv_error_set = LR_Model.train_error(data.X,
                                                        data.y.flatten(),
                                                        data.Xval,
                                                        data.yval.flatten(),
                                                        0)
print('linear model training set error: ', training_error_set)
print('linear model cross validation set error: ', cv_error_set)
# print the learning curves
plot_data.print_learning_curves(training_error_set, cv_error_set)
plt.title('Learning curve for linear regression(lambda = 0.000000)')
plt.show()

# 多项式回归 论文使用power=8 scipy和octave的优化算法不同
power = 6
mean, std = LR_Model.mean_std(LR_Model.GenPolynomial(data.X, power))
norX = LR_Model.Normalization(
    LR_Model.GenPolynomial(data.X, power),
    mean,
    std
)
norXval = LR_Model.Normalization(
    LR_Model.GenPolynomial(data.Xval, power),
    mean,
    std
)
norXtest = LR_Model.Normalization(
    LR_Model.GenPolynomial(data.Xtest, power),
    mean,
    std
)
# train the model for polynomial
theta = np.ones((norX.shape[1],), dtype=float)
poly_theta = LR_Model.get_linear_model(theta, norX, data.y.flatten(), 0)
print('polynomial theta is {}'.format(poly_theta))
# plot the line for lambda=0 polynomial model
plot_data.plot_scatter(data.rawX, data.y)
plot_data.plot_polynomial(poly_theta, mean, std, power)
plt.title('Polynomial linear regression(lambda = 0.000000)')
plt.show()
# plot the learning curve for polynomial
training_error_set, cv_error_set = LR_Model.train_error(norX,
                                                        data.y.flatten(),
                                                        norXval,
                                                        data.yval.flatten(),
                                                        0)
plot_data.print_learning_curves(training_error_set, cv_error_set)
plt.title('Polynomial Regression Learning Curve (lambda = 0.000000)')
plt.show()

# adjust different regularization parameter
lambda_set = [1, 100]
theta = np.ones((norX.shape[1],), dtype=float)
for l in lambda_set:
    poly_theta = LR_Model.get_linear_model(theta, norX,
                                           data.y.flatten(),
                                           l)
    # curves for polynomial
    plot_data.plot_scatter(data.rawX, data.y)
    plot_data.plot_polynomial(poly_theta, mean, std, power)
    plt.title('Polynomial linear regression(lambda = {}.000000)'.format(l))
    plt.show()
    # learning curves
    error_train, error_cv = LR_Model.train_error(norX, data.y.flatten(),
                                                 norXval, data.yval.flatten(),
                                                 l)
    plot_data.print_learning_curves(error_train, error_cv)
    plt.title('Polynomial Regression Learning Curve (lambda = {}.000000)'.format(l))
    plt.show()

# Selecting lambda using a cross validation set
lambda_set = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
theta = np.ones((norX.shape[1],), dtype=float)  # 初始theta
train_error_i_set, cv_error_i_set = [], []
for i in range(len(lambda_set)):
    l = lambda_set[i]
    theta_i = LR_Model.get_linear_model(theta, norX, data.y.flatten(), l)
    train_error_i = LR_Model.cost_function(theta_i, norX, data.y.flatten())
    cv_error_i = LR_Model.cost_function(theta_i, norXval, data.yval.flatten())
    train_error_i_set.append(train_error_i)
    cv_error_i_set.append(cv_error_i)
# print learning curve of lambda
plot_data.print_learning_curve_lambda(lambda_set, train_error_i_set, cv_error_i_set)
plt.title('Selecting lambda using a cross validation set')
plt.show()
# choose the best lambda
best_index = cv_error_i_set.index(min(cv_error_i_set))  # 查找最小的cv error对应的索引
# best_index = np.argmin(cv_error_i_set)
best_lambda = lambda_set[best_index]
print('The best lambda for polynomial is {}'.format(best_lambda))
# computing test set error
test_theta = LR_Model.get_linear_model(theta, norX, data.y.flatten(), best_lambda)
test_error = LR_Model.regularized_cost_function(
    test_theta,
    norXtest,
    data.ytest.flatten(),
    best_lambda
)
print('The test error using the best value of lambda is {}'.format(test_error))
print('When using the same theta '
      'with lambda = 0, the cost is {}'.format(
    LR_Model.cost_function(test_theta, norXtest,
                           data.ytest.flatten())
))
# plotting learning curves with randomly selected examples with lambda = 0.01
training_error_set, cv_error_set = LR_Model.random_train_error(norX,
                            data.y.flatten(),
                            norXval,
                            data.yval.flatten(),
                            0.01)
plot_data.print_learning_curves(training_error_set, cv_error_set)
plt.title('Polynomial Regression Learning Curve (lambda = 0.010000)')
plt.show()
