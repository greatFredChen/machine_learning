from copy import deepcopy
import numpy as np
import scipy.optimize as opt


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
