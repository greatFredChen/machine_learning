from PlotData import PlotData
from LinearRegressionMethod import LinearRegressionMethod
from Data import Data
import numpy as np
import matplotlib.pyplot as plt

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