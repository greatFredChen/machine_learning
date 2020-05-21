import numpy as np
from scipy.io import loadmat


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
