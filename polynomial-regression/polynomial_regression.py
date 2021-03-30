# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Training the Polynomial Regression model on the whole dataset

# Visualizing the Linear Regression results

# Visualizing the Polynomial Regression results

# Visualizing the Polynomial Regression results (for higher resolution and smoother curve)

# Predicting a new result with Linear Regression

# Predicting a new result with Polynomial Regression