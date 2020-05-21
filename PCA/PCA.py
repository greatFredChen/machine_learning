import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
data = loadmat('ex7data1.mat')
# print(data)
X = data['X']
plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors='b')
plt.show()


# Normalize the X will be a normal distribution
# mean = 0, std = 1 normal distribution
def featureNormalize(X):
    std = np.std(X, axis=0, ddof=1)
    mean = np.mean(X, axis=0)
    norX = (X - mean) / std
    return norX, mean, std


norX, mean, std = featureNormalize(X)


# get covariance matrix
def pca(X):
    m = float(len(X))
    coMatrix = (X.T @ X) / m
    U, S, V = np.linalg.svd(coMatrix)
    return U, S, V


U, S, V = pca(norX)
print('2.2 result: ', U[:, 0])
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors='b')
plt.plot([mean[0], mean[0] + 1.5*S[0]*U[0, 0]],
         [mean[1], mean[1] + 1.5*S[0]*U[0, 1]],
         c='black', label='FPC')  # first principle component
plt.plot([mean[0], mean[0] + 1.5*S[0]*U[1, 0]],
         [mean[1], mean[1] + 1.5*S[0]*U[1, 1]],
         c='r', label='SPC')  # second principle component 与 第一个正交
plt.legend()
plt.axis('equal')
plt.show()


# Dimensionality Reduction with PCA
# project => a ` b / |b| |b|为常数
def projectData(X, U, K):
    project = X @ U[:, 0: K]
    return project


# test
Z = projectData(norX, U, 1)
print('2.3.1 result: ', Z[0])


# recover data to high dimension
def recoverData(Z, U, K):
    X_rec = Z @ U[:, :K].T
    return X_rec


X_rec = recoverData(Z, U, 1)
print('2.3.2 result: ', X_rec[0])

# visualize the projection
plt.figure(figsize=(7, 5))
plt.axis('equal')
plt.scatter(norX[:, 0], norX[:, 1], facecolors='none',
            edgecolors='b', label='原始数据点')
plt.scatter(X_rec[:, 0], X_rec[:, 1], facecolors='none',
            edgecolors='r', label='投影数据点')
for x in range(norX.shape[0]):
    plt.plot([norX[:, 0], X_rec[:, 0]], [norX[:, 1], X_rec[:, 1]], 'k--')
plt.legend()
plt.show()

# Face Image Dataset
face_data = loadmat('ex7faces.mat')
faceX = face_data['X']
print('The shape of face images: ', faceX.shape)


# visualize the first 100 faces
def displayFaces(faceX, row, col):
    fig, ax = plt.subplots(row, col, figsize=(7, 7))
    for r in range(row):
        for c in range(col):
            ax[r][c].imshow(faceX[r * col + c].reshape(32, 32).T, cmap='Greys_r')
            ax[r][c].set_xticks([])
            ax[r][c].set_yticks([])


displayFaces(faceX, 10, 10)
plt.show()

# PCA on the face dataset
norFaceX, faceMean, faceStd = featureNormalize(faceX)
U, S, V = pca(norFaceX)

# display the first 36 principal components
# remember to transpose before displaying the faces
displayFaces(U[:, :36].T, 6, 6)
plt.show()

# Dimensionality reduction
faceZ = projectData(norFaceX, U, 100)
faceX_rec = recoverData(faceZ, U, 100)
displayFaces(faceX, 10, 10)
plt.title('Original images of faces')
displayFaces(faceX_rec, 10, 10)
plt.title('reconstructed images of faces')
plt.show()
