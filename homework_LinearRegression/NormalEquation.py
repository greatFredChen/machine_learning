import numpy as np
import pandas as pd

# extract the data
path = "ex1data2.txt"
data3 = pd.read_csv(path, names=['Size', 'Bedrooms', 'Price'])
print(data3.head())

# normal equation
data3.insert(0, 'Ones', 1)
theta = np.matrix([0., 0., 0.])
col = data3.shape[1]
X = data3.iloc[:, 0:col-1]
y = data3.iloc[:, col-1:col]
X = np.matrix(X.values)
y = np.matrix(y.values)

theta = np.linalg.inv(X.T@X)@X.T@y
print("theta is", theta)

# prediction
predicted_price = np.matrix([1., 1650., 3.]) * theta
print("The predicted price is", predicted_price)