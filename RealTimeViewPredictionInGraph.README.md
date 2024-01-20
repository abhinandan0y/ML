Q:How to view real time machine learning predictions in a line graph?

Real-time live graph using plotly:
 ```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
import time

# Initialize figure
fig = make_subplots(rows=1, cols=1, subplot_titles=['Real-time Temperature Prediction'])
fig.update_layout(title_text="Real-time Temperature Prediction", showlegend=True)

# Initialize model and data
model = LinearRegression()
time_of_day = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
temperature = np.array([15, 18, 20, 22, 25, 28, 30, 26, 23, 20])
time_of_day = time_of_day.reshape(-1, 1)

model.fit(time_of_day, temperature)

# Create initial trace
trace = go.Scatter(x=time_of_day.flatten(), y=temperature, mode='markers', name='Actual Temperature')
fig.add_trace(trace)

# Update layout
fig.update_xaxes(title_text="Time of Day")
fig.update_yaxes(title_text="Temperature")

# Display the plot
fig.show()

# Real-time update loop
for new_time in range(11, 21):
    temperature_prediction = model.predict(np.array([new_time]).reshape(-1, 1))

    # Update trace
    trace.x = np.append(trace.x, new_time)
    trace.y = np.append(trace.y, temperature_prediction[0])

    # Update layout
    fig.update_layout(title_text="Real-time Temperature Prediction", showlegend=True)

    # Update the figure
    fig.update_traces(marker=dict(size=8))
    time.sleep(1)
```

<div style="width: 100%;">
  <img src="https://raw.githubusercontent.com/abhinandan0y/ML/main/plotly_liveGraph.png" style="width: 100%;" alt="bioinformatics_lab.png">
</div>

Real-time stock price prediction

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time

# Function to update and plot real-time predictions
def update_plot(model, time_points, prices, ax):
    ax.clear()
    ax.scatter(time_points, prices, color='blue', label='Actual Prices')
    ax.plot(time_points, model.predict(time_points.reshape(-1, 1)), color='red', label='Regression Line')
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock Price')
    ax.set_title('Real-time Stock Price Prediction')
    ax.legend()
    plt.pause(0.1)

# Generate initial data for time and stock prices
time_points = np.arange(1, 11)
prices = np.array([50, 55, 60, 65, 70, 75, 80, 85, 90, 95])

# Create a linear regression model
model = LinearRegression()
model.fit(time_points.reshape(-1, 1), prices)

# Create the plot
fig, ax = plt.subplots()

# Run the real-time update loop
for new_time in range(11, 21):
    price_prediction = model.predict(np.array([new_time]).reshape(-1, 1))
    prices = np.append(prices, price_prediction)
    update_plot(model, np.arange(1, len(prices) + 1), prices, ax)  # Fix here
    time.sleep(1)

plt.show()
```
<div style="width: 100%;">
  <img src="https://raw.githubusercontent.com/abhinandan0y/ML/main/plotly_liveGraph.png" style="width: 100%;" alt="bioinformatics_lab.png">
</div>

Temperature prediction example:

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time

# Function to update and plot real-time predictions
def update_plot(model, time_of_day, temperature, ax):
    ax.clear()
    ax.scatter(time_of_day, temperature, color='blue', label='Actual Temperature')
    ax.plot(time_of_day, model.predict(time_of_day.reshape(-1, 1)), color='red', label='Regression Line')
    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Temperature')
    ax.set_title('Real-time Temperature Prediction')
    ax.legend()
    plt.pause(0.1)

# Generate initial data for time of day and temperature
time_of_day = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
temperature = np.array([15, 18, 20, 22, 25, 28, 30, 26, 23, 20])

# Reshape the data to fit the model
time_of_day = time_of_day.reshape(-1, 1)

# Create a linear regression model
model = LinearRegression()
model.fit(time_of_day, temperature)

# Create the plot
fig, ax = plt.subplots()

# Run the real-time update loop
for new_time in range(11, 21):
    temperature_prediction = model.predict(np.array([new_time]).reshape(-1, 1))
    temperature = np.append(temperature, temperature_prediction)
    update_plot(model, np.arange(1, len(temperature) + 1), temperature, ax)  # Fix here
    time.sleep(1)

plt.show()
```
<div style="width: 100%;">
  <img src="https://raw.githubusercontent.com/abhinandan0y/ML/main/plotly_liveGraph.png" style="width: 100%;" alt="bioinformatics_lab.png">
</div>
