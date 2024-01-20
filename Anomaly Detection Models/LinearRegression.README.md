#Linear Regression:<br/>

Predicting housing prices based on square footage:
```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Example data - Square footage as input (X) and housing prices as output (y)
X = np.array([1400, 1600, 1700, 1875, 1100]).reshape(-1, 1)
y = np.array([245000, 312000, 279000, 308000, 199000])

# Creating and training the linear regression model
model = LinearRegression()
model.fit(X, y)

# Making predictions
square_footage_to_predict = np.array([1500, 1800]).reshape(-1, 1)
predictions = model.predict(square_footage_to_predict)

print(f'Predictions for housing prices: {predictions}')
```
Estimating the salary based on years of experience:
```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Example data - Years of experience as input (X) and salary as output (y)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([50000, 60000, 75000, 90000, 110000])

# Creating and training the linear regression model
model = LinearRegression()
model.fit(X, y)

# Making predictions
experience_to_predict = np.array([2.5, 4.5]).reshape(-1, 1)
predictions = model.predict(experience_to_predict)

print(f'Estimated salaries based on years of experience: {predictions}')
```
Forecasting sales volume for a product based on advertising spend:
```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Example data - Advertising spend as input (X) and sales volume as output (y)
X = np.array([2000, 3000, 4000, 4500, 5000]).reshape(-1, 1)
y = np.array([50000, 70000, 90000, 95000, 100000])

# Creating and training the linear regression model
model = LinearRegression()
model.fit(X, y)

# Making predictions
advertising_spend_to_predict = np.array([3500, 4800]).reshape(-1, 1)
predictions = model.predict(advertising_spend_to_predict)

print(f'Forecasted sales volume based on advertising spend: {predictions}')
```
