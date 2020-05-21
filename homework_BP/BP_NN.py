import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from copy import deepcopy


def loaddata(path):
    data = loadmat(path)
    X = data['X']
    y = data['y'].reshape(-1)
    return X, y


# extract data
X, y = loaddata('ex4data1.mat')
raw_y = deepcopy(y)  # 保留原始的y
print(X.shape, y.shape)
print(np.unique(y))


# visualizing data
def displayData(X):
    random_row = np.random.choice(5000, 100, False)
    random_X = X[random_row, :]
    fig, ax = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True, figsize=(8, 8))
    # (1, 400)需要reshape为(20, 20)
    for r in np.arange(10):
        for c in np.arange(10):
            ax[r, c].matshow(random_X[r * 10 + c].reshape(20, 20), cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()


displayData(X)
data_theta = loadmat('ex4weights.mat')
Theta1 = data_theta['Theta1']
Theta2 = data_theta['Theta2']
print(Theta1.shape, Theta2.shape)


# deal with y  transfer y to (5000, 10)
# 1 = [1 0 0 ...]  2 = [0 1 0 ...]  3 = [0 0 1 0 ..]
def deal_with_y(y):
    expand_y = np.zeros((5000, 10))
    for line in np.arange(len(y)):
        expand_y[line, y[line] - 1] = 1
    return expand_y


# sigmoid
def sigmoid(z):
    return 1. / (1. + np.exp(-z))


# 序列化  一维化之后用np.r_进行拼接(默认'r')
def serialize(theta1, theta2):
    return np.r_[theta1.flatten(), theta2.flatten()]


# 反序列化 计算出该取出的元素重新reshape  401*25=10025  26*10=260
def deserialize(theta):
    return theta[:25*401].reshape(25, 401), theta[25*401:].reshape(10, 26)


# cost function里需要用到hx,hx维度为(5000, 10),此时hx需要通过前向传播算出来
# a1 = x ===> hx
def feedforward(theta, X):
    # 此时的theta为压缩过的theta,高级优化方法只能用一个theta
    t1, t2 = deserialize(theta)
    a1 = X
    z2 = a1 @ t1.T
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, 1, axis=1)
    z3 = a2 @ t2.T
    a3 = sigmoid(z3)
    # print(a3.shape)  # (5000, 10)
    return a1, z2, a2, z3, a3   # 计算delta时需要这5个参数


# cost function
def cost_function(theta, X, y):
    m = len(X)
    a1, z2, a2, z3, hx = feedforward(theta, X)
    # cost计算
    fp = (-y) * np.log(hx)
    sp = (1 - y) * np.log(1 - hx)
    return 1. / m * np.sum(fp - sp)


def regularized_cost_function(theta, X, y, r):
    m = len(X)
    cost = cost_function(theta, X, y)
    t1, t2 = deserialize(theta)
    _t1, _t2 = t1[:, 1:], t2[:, 1:]
    regularized = r / (2 * m) * (np.sum(_t1 * _t1) + np.sum(_t2 * _t2))
    return cost + regularized


# BackPropagation Part
# sigmoid gradient
def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


# random initialization
# numpy.random.uniform   返回ndarray  直接生成t1+t2的序列化形式
# 这样可以省去分别初始化t1 t2以及将它们序列化成一个theta的时间！  einit = 0.12
def random_initialization(theta_size):
    return np.random.uniform(-0.12, 0.12, theta_size)


# gradient
def gradient(theta, X, y):
    a1, z2, a2, z3, a3 = feedforward(theta, X)  # hx = a3
    t1, t2 = deserialize(theta)  # (25, 401) (10, 26)
    m = len(X)
    d3 = a3 - y  # (5000, 10)
    # t2 @ d3 (5000, 26)  g'(z2) (5000, 25)  因此不需要把bias算进去
    d2 = d3 @ t2[:, 1:] * sigmoid_gradient(z2)  # (5000, 25)
    D2 = (d3.T @ a2) / m  # (10, 26)
    D1 = (d2.T @ a1) / m  # (25, 401)
    return serialize(D1, D2)  # 返回序列化之后的theta


# regularized gradient
def regularized_gradient(theta, X, y, r):
    m = len(X)
    D1, D2 = deserialize(gradient(theta, X, y))  # (25, 401) (10, 26)
    t1, t2 = deserialize(theta)  # (25, 401) (10, 26)
    # 不惩罚第一列 bias 第一列全部设0
    t1[:, 0], t2[:, 0] = 0, 0
    r_D1 = D1 + r / m * t1  # (25, 401)
    r_D2 = D2 + r / m * t2  # (10, 26)
    return serialize(r_D1, r_D2)


# 用f'(theta) 与 gradient(theta)相比较
# gradient为准确的梯度(导数)，而f'为近似，当e->0时，将得到gradient
# 当e比较小时，比较f'与gradient,两者相差小于一定数值时说明gradient计算正确
# 这就是gradient checking的意义所在，即检测梯度是否计算正确
def gradient_checking(theta, X, y, r, e):
    def f(theta_plus, theta_minus, r):
        return (regularized_cost_function(theta_plus, X, y, r)
                - regularized_cost_function(theta_minus, X, y, r)) \
               / (2 * e)
    f_gradient = []
    for i in np.arange(len(theta)):
        t1 = deepcopy(theta)
        t2 = deepcopy(theta)
        # print(t1.shape, t2.shape)
        t1[i] = t1[i] + e
        t2[i] = t2[i] - e
        f_i = f(t1, t2, r)  # f'
        f_gradient.append(f_i)  # 已经序列化 (10285, )
    # now check the gradient
    f_gradient = np.array(f_gradient)
    r_gradient = regularized_gradient(theta, X, y, r)
    r_minus = np.abs(f_gradient - r_gradient)
    # 当r_minus中所有元素的值都小于1e-9时，gradient check通过
    # 只需要取最大值比较即可
    if np.ma.max(r_minus.flatten()) < 1e-9:
        print("Pass the gradient check!")
    else:
        print("Doesn't pass the gradient check!")


# train the data
def bp_nn(X, y, r):
    theta = random_initialization(10285)
    # start training!
    result = opt.minimize(fun=regularized_cost_function,
                          x0=theta,
                          args=(X, y, r),
                          method='TNC',
                          jac=regularized_gradient,
                          options={'maxiter': 400})
    return result


def accuracy(theta, X, y):
    a1, z2, a2, z3, hx = feedforward(theta, X)
    hy = np.argmax(hx, axis=1) + 1
    return sum([1 if a == b else 0 for a, b in zip(hy, y)]) / len(X)


def decide_check(theta, X, y, r, e):
    cmd = input('Do you want to execute the gradient check? (y / n):\n')
    while cmd != 'y' and cmd != 'n':
        print('Your input is not right!')
        cmd = input('Do you want to execute the gradient check? (y / n):\n')
    if cmd == 'y':
        gradient_checking(theta, X, y, r, e)  # gradient check
        print('gradient check is over!')
    else:
        print('gradient check is banned!')


# visualizing the hidden layer
def visualize_hidden(theta):
    # using theta1 (25, 401) to plot
    t1, t2 = deserialize(theta)  # (25, 401) (10, 26)
    _t1 = t1[:, 1:]  # 去掉bias  (25, 400)
    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True, figsize=(6, 6))
    for r in np.arange(5):
        for c in np.arange(5):
            ax[r, c].matshow(_t1[r * 5 + c].reshape(20, 20), cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()


y = y.flatten()
X = np.insert(X, 0, 1, axis=1)
y = deal_with_y(y)
print(X.shape, y.shape)
test_t = serialize(Theta1, Theta2)
# test
# cost function
print("test cost function:", cost_function(test_t, X, y))
# cost function with regularization
print("test regularized cost function:", regularized_cost_function(test_t, X, y, 1))
# sigmoid gradient test
print("test sigmoid gradient:", sigmoid_gradient(np.array([0, 0, 0])))
# gradient checking
decide_check(test_t, X, y, 1, 1e-4)  # r=1
# train the model
result = bp_nn(X, y, 1)
print("Neural Network Model:", result)
# calculate the accuracy
print("The accuracy of this model:", accuracy(result.x, X, raw_y))
# plot the hidden layer
visualize_hidden(test_t)
