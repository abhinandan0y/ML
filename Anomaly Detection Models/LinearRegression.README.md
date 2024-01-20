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
Predicting the temperature based on time of day:
```python
# Import necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate random data for time of day and temperature
time_of_day = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
temperature = np.array([15, 18, 20, 22, 25, 28, 30, 26, 23, 20])

# Reshape the data to fit the model
time_of_day = time_of_day.reshape(-1, 1)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(time_of_day, temperature)

# Make predictions for a new set of time_of_day
new_time_of_day = np.array([11, 12, 13]).reshape(-1, 1)
temperature_predictions = model.predict(new_time_of_day)

# Plot the data and the regression line
plt.scatter(time_of_day, temperature, color='blue')
plt.plot(time_of_day, model.predict(time_of_day), color='red', linewidth=2)
plt.xlabel('Time of Day')
plt.ylabel('Temperature')
plt.title('Temperature Prediction based on Time of Day')
plt.show()

# Output temperature predictions for new_time_of_day
print("Temperature Predictions for New Time of Day:", temperature_predictions)
```

Projecting student performance based on study hours:
```python
# Generate random data for study hours and student performance
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
student_performance = np.array([50, 60, 70, 80, 85, 90, 92, 95, 98, 100])

# Reshape the data to fit the model
study_hours = study_hours.reshape(-1, 1)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(study_hours, student_performance)

# Make predictions for a new set of study_hours
new_study_hours = np.array([11, 12, 13]).reshape(-1, 1)
performance_predictions = model.predict(new_study_hours)

# Plot the data and the regression line
plt.scatter(study_hours, student_performance, color='blue')
plt.plot(study_hours, model.predict(study_hours), color='red', linewidth=2)
plt.xlabel('Study Hours')
plt.ylabel('Student Performance')
plt.title('Student Performance Projection based on Study Hours')
plt.show()

# Output performance predictions for new_study_hours
print("Performance Predictions for New Study Hours:", performance_predictions)
```
