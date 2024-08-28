#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Generate some sample data
np.random.seed(0)
x = np.random.rand(100, 1)
y = 2 + 3*x + 4*x**2 + np.random.randn(100, 1)

# Create a polynomial regression model with degree 2
model = make_pipeline(PolynomialFeatures(2), LinearRegression())

# Train the model
model.fit(x, y)

# Make predictions
y_pred = model.predict(x)

# Plot the data and the fitted curve
plt.scatter(x, y, label='Data')
plt.plot(x, y_pred, color='red', label='Polynomial Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.legend()
plt.show()


# In[ ]:




