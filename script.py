import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

# Task 1
print(df.head)

# Task 2
prod_per_year = df.groupby('year').totalprod.mean().reset_index()
print(prod_per_year)

# Task 3
X = prod_per_year['year']
X = X.values.reshape(-1, 1)

# Task 4
y = prod_per_year['totalprod']

# Task 5
plt.scatter(X, y)

# Task 6
regr = linear_model.LinearRegression()

# Task 7
regr.fit(X, y)

# Task 8
print(regr.coef_)
print(regr.intercept_)

# Task 9
y_predict = regr.predict(X)

# Task 10
plt.plot(X, y_predict)
#plt.show()

# Task 11
X_future = np.array(range(2013, 2050))
#print(X_future)
X_future = X_future.reshape(-1, 1)
#print(X_future)

# Task 12
future_predict = regr.predict(X_future)

# Task 13
plt.plot(X_future, future_predict)
plt.show()
