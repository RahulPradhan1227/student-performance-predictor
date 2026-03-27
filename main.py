import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("dataset.csv")

# Inputs and output
X = data[['study_hours', 'sleep_hours', 'attendance']]
y = data['marks']

# Train model (no split → simpler)
model = LinearRegression()
model.fit(X, y)

# Show basic info
print("Model trained successfully!")

# User input
study = float(input("Study hours: "))
sleep = float(input("Sleep hours: "))
attendance = float(input("Attendance (%): "))

# Prediction
result = model.predict([[study, sleep, attendance]])
print("Predicted Marks:", round(result[0], 2))

# Simple graph
plt.scatter(data['study_hours'], data['marks'])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study vs Marks")
plt.show()
