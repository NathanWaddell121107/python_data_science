# -*- coding: utf-8 -*-

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split into training / test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

# Training the simple linear regression model on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)

# Visualize the Training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualize the Test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Predict salary after 12 years
print(regressor.predict([[12]])) #[137605.23485427]

# Get the final linear regression equation with values of the coefficients
print(regressor.coef_) #[9332.94473799]
print(regressor.intercept_) #25609.89799835482

# Final equation would be Salary = 9332.94 x YearsExperience + 25609.90