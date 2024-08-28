#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate some sample data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 3 + 2 * X + np.random.randn(100, 1)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Print coefficients
print("Coefficients: ", model.coef_)
print("Intercept: ", model.intercept_)

# Plot the data and regression line
plt.scatter(X, y)
plt.plot(X, y_pred, color='red')
plt.show()


# In[ ]:




