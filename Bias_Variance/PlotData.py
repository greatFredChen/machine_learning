from LinearRegressionMethod import LinearRegressionMethod
import matplotlib.pyplot as plt
import numpy as np


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
