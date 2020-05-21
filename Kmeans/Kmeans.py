import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from skimage import io

data = loadmat('ex7data2.mat')
# print(data)
X = data['X']


# mark the x(i) with the index of the closest centroid
def findClosestCentroids(X, centroids):
    idx = []
    for i in range(len(X)):
        min_dist = 1000000
        min_j = -1
        for j in range(len(centroids)):
            xi, centroid = X[i], centroids[j]
            dist = 0
            for xij, centroidj in zip(xi, centroid):
                dist = dist + (xij - centroidj) ** 2
            dist = np.sqrt(dist)
            if dist < min_dist:
                min_dist = dist
                min_j = j
        idx.append(min_j)
    return np.array(idx)


init_centroids = np.array([[3, 3], [6, 2], [8, 5]])
idx = findClosestCentroids(X, init_centroids)
print('The k index of first three examples: {}'
      .format(findClosestCentroids(X[0:3], init_centroids)))


def computeCentroids(X, idx):
    # print(idx)
    centroids = []
    K = np.unique(idx)
    # print(K)
    for index in K:
        miu_k = np.mean(X[idx == index], axis=0)
        centroids.append(miu_k)
    return np.array(centroids)


miu = computeCentroids(X, idx)
print('The location of centroids after first step: {}'
      .format(miu))


def runKmeans(X, centroids, **kwargs):
    centroids_iter = [centroids]
    _centroids = centroids
    if kwargs['iterations']:
        iterations = kwargs['iterations']
        for i in range(iterations):
            idx = findClosestCentroids(X, _centroids)
            _centroids = computeCentroids(X, idx)  # update _centroids
            centroids_iter.append(_centroids)
        last_idx = findClosestCentroids(X, centroids_iter[-1])  # 用最终生成的中心点进行分类
        return np.array(centroids_iter), last_idx
    else:
        pass


def plotData(X, centroids_iter, last_idx):
    # 点只展示最终分类，不展示中间过程的分类
    # 中心点一次展示所有点，并画出移动轨迹
    K = len(centroids_iter[0])
    colors = ['r', 'g', 'b', 'gold', 'darkorange', 'salmon', 'olivedrab',
              'maroon', 'navy', 'sienna', 'tomato', 'lightgray', 'aliceblue']
    assert K <= len(colors), 'colors are not enough for K'
    subX = []  # K分类下X的子集
    for index in np.unique(last_idx):
        subX.append(X[last_idx == index])
    subX = np.array(subX)
    # plot the scatter
    plt.figure(figsize=(8, 6))
    for i in range(len(subX)):
        xk = subX[i]
        plt.scatter(xk[:, 0], xk[:, 1], c=colors[i], marker='o',
                    label='Cluster {}'.format(i))
    plt.legend()
    plt.xlabel('x1')
    plt.ylabel('x2')
    # plot the tracks of the centroids
    track_x, track_y = [], []
    for centroids in centroids_iter:
        track_x.append(centroids[:, 0])
        track_y.append(centroids[:, 1])
    plt.plot(track_x, track_y, color='black', marker='x', linewidth=1,
             markersize=4)


iterations = 10
# print(init_centroids)
centroids_iter, last_idx = runKmeans(X, init_centroids, iterations=iterations)
# print(centroids_iter)
plotData(X, centroids_iter, last_idx)
plt.title('Iteration number {}'.format(iterations))
plt.show()


# random initialization
def randomInitialization(X, k):
    # 从X中随机不重复地取出k个样本作为中心点(centroids)
    centroids_idx = np.random.choice(len(X), size=k, replace=False)
    # print(centroids_idx)
    return X[centroids_idx]


# print(randomInitialization(X, len(init_centroids)))
# Image compression with K-means
image = io.imread('bird_small.png')
print('image shape: {}'.format(image.shape))
# show the origin picture
plt.imshow(image)
plt.show()
# rescale the RGB value
image = image / 255.0
imageX = image.reshape(-1, 3)  # flatten the image
K = 16
init_centroids = randomInitialization(imageX, K)
centroids_iter, last_idx = runKmeans(imageX, init_centroids, iterations=10)
last_centroid = centroids_iter[-1]
process_img = np.zeros(imageX.shape)  # flatten image shape
for index in np.unique(last_idx):
    process_img[last_idx == index] = last_centroid[index]
# show the processed image
process_img = process_img.reshape((128, 128, 3))
# print(process_img)  # 所有数据都在0-1之间
plt.imshow(process_img)
plt.show()
