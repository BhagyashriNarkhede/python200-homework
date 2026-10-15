# --- scikit-learn API ---

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Q1: Create, fit, and predict

temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])

model = LinearRegression()
model.fit(temp_min, temp_max)

prediction_6 = model.predict(np.array([[6]]))
prediction_18 = model.predict(np.array([[18]]))

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted high for a low of 6 degrees:", prediction_6[0])
print("Predicted high for a low of 18 degrees:", prediction_18[0])

# Q2: Reshape a 1D array into a 2D array

x = np.array([10, 20, 30, 40, 50])

print("Original shape:", x.shape)

x_2d = x.reshape(-1, 1)

print("New shape:", x_2d.shape)

# scikit-learn needs X to be 2D because rows represent samples
# and columns represent features, even when there is only one feature.


# --- Linear Regression ---

rng = np.random.default_rng(42)
num_days  = 120
temp_min  = rng.uniform(-5, 22, num_days)
is_rainy  = rng.integers(0, 2, num_days).astype(float)
temp_max  = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)


# Q1: Plot daily high versus low

plt.figure()
plt.scatter(temp_min, temp_max, c=is_rainy, cmap="coolwarm")
plt.title("Daily High vs Low")
plt.xlabel("Daily Low Temperature")
plt.ylabel("Daily High Temperature")
plt.savefig("assignments_02/outputs/high_vs_low.png")
plt.close()

# The scatter plot shows two visible groups because rainy and dry
# days tend to have different high temperatures at similar low temperatures.
# This suggests that the is_rainy variable affects the daily high.


# Q2: Split the data into training and test sets

X = temp_min.reshape(-1, 1)
y = temp_max

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Comment: The original array is one-dimensional with shape (5,).
# Scikit-learn expects the input features X to be two-dimensional,
# with rows representing samples and columns representing features.
# Reshaping produces a (5, 1) array with five samples and one feature.

# Q3: Fit a model using temp_min only

model = LinearRegression()
model.fit(X_train, y_train)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

y_pred = model.predict(X_test)

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² on the test set:", model.score(X_test, y_test))

# Comment: The baseline model has a slope of approximately 0.944,
# so each one-degree increase in minimum temperature is associated
# with about a 0.944-degree increase in predicted maximum temperature.
# The test RMSE is approximately 2.48 degrees, and the test R²
# of 0.894 indicates that the model explains about 89.4% of the
# variation in maximum temperatures in the test set.


# Q4: Fit a model using temp_min and is_rainy

X_full = np.column_stack([temp_min, is_rainy])

X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_full_train, y_full_train)

y_full_pred = model_full.predict(X_full_test)

r2_temp_min_only = model.score(X_test, y_test)
r2_with_rain = model_full.score(X_full_test, y_full_test)

print("Test R² with temp_min only:", r2_temp_min_only)
print("Test R² with temp_min and is_rainy:", r2_with_rain)

print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])

## Comment: Adding is_rainy improves test R² from approximately
# 0.894 to 0.945. The temp_min coefficient is approximately
# +0.976, while the is_rainy coefficient is approximately -2.919.
# Holding minimum temperature constant, the model estimates
# that rainy days have highs about 2.92 degrees lower.

# Q5: Plot predicted versus actual values using the two-feature model

plt.figure()
plt.scatter(y_full_pred, y_full_test)

# Add a diagonal reference line where predicted equals actual.
line_min = min(y_full_pred.min(), y_full_test.min())
line_max = max(y_full_pred.max(), y_full_test.max())
plt.plot([line_min, line_max], [line_min, line_max], linestyle="--")

plt.title("Predicted vs Actual")
plt.xlabel("Predicted High Temperature")
plt.ylabel("Actual High Temperature")
plt.savefig("assignments_02/outputs/predicted_vs_actual_warmup.png")
plt.close()


# Comment: The diagonal line represents perfect predictions.
# Points above the line indicate that the model predicted a
# temperature higher than the actual value. Points below the
# line indicate that the model predicted a temperature lower
# than the actual value. Points closer to the line have smaller
# prediction errors.