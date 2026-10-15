
# --- Mini-Project: Predicting the Daily High ---

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Set the output directory relative to this Python file.
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- Task 1: Fetch the Data ---

# City chosen: Morrisville, North Carolina.
url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 35.82,
    "longitude": -78.83,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}

response = requests.get(url, params=params, timeout=30)
response.raise_for_status()

df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

print("--- Task 1: Dataset ---")
print("Dataset shape:", df.shape)
print("First five rows:")
print(df.head())

# Comment: This project uses daily weather data for Morrisville,
# North Carolina, from January 1 through December 31, 2023.
# The dataset contains 365 daily observations and five columns,
# including the date, daily high and low temperatures,
# precipitation, and maximum wind speed.

# --- Task 2: Look at the Features ---

print("\n--- Task 2: Descriptive Statistics ---")
print(df.describe())

print("\nMissing values in each column:")
print(df.isnull().sum())

numeric_cols = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]
df = df.dropna(subset=numeric_cols).copy()

print("\nDataset shape after handling missing values:", df.shape)
print("Total missing values after handling:")
print(df.isnull().sum().sum())

print("\nCorrelation with daily high, sorted:")
correlations = (
    df.select_dtypes(include="number")
    .corr()["temperature_2m_max"]
    .sort_values(ascending=False)
)
print(correlations)

# The correlation closest to +1, excluding the target itself,
# indicates the strongest positive linear relationship with daily high.
plt.figure()
plt.hist(df["temperature_2m_max"], bins=20, edgecolor="black")
plt.title("Distribution of Daily High Temperatures")
plt.xlabel("Daily High Temperature (°C)")
plt.ylabel("Number of Days")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "daily_high_distribution.png")
plt.close()

# The histogram shows how daily high temperatures are distributed
# throughout the year and which temperature ranges occur most often.
# Comment: The dataset contains no missing values, so no rows needed to be removed. Minimum temperature has the strongest
# correlation with daily high temperature at approximately 0.921. Precipitation and maximum wind speed have weak negative
# correlations with daily high temperature. The plot helps visualize the distribution of daily high temperatures.

# --- Task 3: Engineer a Feature ---

df["is_summer"] = df["date"].dt.month.isin([6, 7, 8]).astype(int)

summer_counts = df["is_summer"].value_counts().sort_index()

print("\n--- Task 3: Summer and Non-Summer Days ---")
print("Non-summer days (0):", summer_counts.get(0, 0))
print("Summer days (1):", summer_counts.get(1, 0))

# Comment: The dataset contains 92 summer days and 273 non-summer
# days. Summer is defined as June, July, and August. I expect summer to be associated with higher daily maximum temperatures in Morrisville because these months are generally warmer.

# --- Task 4: Baseline Model ---

X = df[["temperature_2m_min"]].values
y = df["temperature_2m_max"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
baseline_r2 = model.score(X_test, y_test)

print("\n--- Task 4: Baseline Model ---")
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("Slope:", model.coef_[0])
print("RMSE:", baseline_rmse)
print("Test R²:", baseline_r2)

# Comment: The baseline model uses minimum temperature alone
# to predict daily maximum temperature. Its test RMSE is about3.21 degrees Celsius, meaning the typical prediction error is approximately 3.21 degrees Celsius in magnitude.
# The test R² of 0.857 indicates that the model explains about 85.7% of the variation in test-set daily highs.
# RMSE is measured in degrees Celsius. A typical prediction error is roughly the size of the RMSE, although individual errors vary.


# --- Task 5: Full Model ---

feature_cols = [
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "is_summer",
]

X = df[feature_cols].values
y = df["temperature_2m_max"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train, y_train)

y_full_pred = model_full.predict(X_test)

train_r2 = model_full.score(X_train, y_train)
full_r2 = model_full.score(X_test, y_test)
full_rmse = np.sqrt(mean_squared_error(y_test, y_full_pred))

print("\n--- Task 5: Full Model ---")
print("Training R²:", train_r2)
print("Test R²:", full_r2)
print("Test RMSE:", full_rmse)

print("\nBaseline test R²:", baseline_r2)
print("Full model test R²:", full_r2)
print("Change in test R²:", full_r2 - baseline_r2)

print("\nFeature coefficients:")
for name, coef in zip(feature_cols, model_full.coef_):
    print(f"{name:22s}: {coef:+.3f}")

# Comment: The full model improves test R² from approximately 0.857 to 0.895 and reduces test RMSE from about 3.21 to 2.74 degrees Celsius. Minimum temperature has the largest
# coefficient by absolute value at +0.927, indicating a positive
# relationship with predicted daily highs when other features are held constant. Precipitation and wind speed have negative coefficients, while is_summer has a positive coefficient.
# Training R² (0.869) is lower than test R² (0.895), so there is no training score higher than the test score in this split.
# The small gap does not, by itself, indicate overfitting.



# --- Task 6: Evaluate and Summarize ---

plt.figure()
plt.scatter(y_full_pred, y_test)

# Add a diagonal line where predicted temperatures equal actual temperatures.
line_min = min(y_full_pred.min(), y_test.min())
line_max = max(y_full_pred.max(), y_test.max())
plt.plot(
    [line_min, line_max],
    [line_min, line_max],
    linestyle="--",
)

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Daily High (°C)")
plt.ylabel("Actual Daily High (°C)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "predicted_vs_actual_high.png")
plt.close()

# Points above the diagonal have actual temperatures higher than predicted.
# Points below the diagonal have actual temperatures lower than predicted.
# Inspect the plot to see whether errors are larger at the high or low end.

print("\n--- Task 6: Summary ---")
print("City: Morrisville, North Carolina")
print("Dataset size:", len(df), "days")
print("Test set size:", len(y_test), "days")
print(f"Full model test RMSE: {full_rmse:.2f} °C")
print(f"Full model test R²: {full_r2:.3f}")

largest_effect_index = np.argmax(np.abs(model_full.coef_))
largest_effect_name = feature_cols[largest_effect_index]
largest_effect_coef = model_full.coef_[largest_effect_index]

direction = "raises" if largest_effect_coef > 0 else "lowers"

print(
    f"Largest coefficient by absolute value: {largest_effect_name} "
    f"({largest_effect_coef:+.3f}), which {direction} the predicted high "
    "when the other features are held constant."
)

print("Baseline test R²:", round(baseline_r2, 3))
print("Full model test R²:", round(full_r2, 3))

## Comment: Points closer to the diagonal line have more accurate
# predictions. Points above the line represent overpredictions,
# while points below the line represent underpredictions.
# The distance from the line indicates the size of the error.

# --- Written Summary ---
#
# The dataset contains 365 days of weather data for Morrisville,North Carolina. The test set contains 73 days.
# The full model has an RMSE of about 2.74 degrees Celsius and a test R² of about 0.895. Predictions typically differ from
# actual daily highs by roughly 2.74 degrees Celsius, although individual errors can be larger or smaller.
# The feature with the largest coefficient by absolute value is temperature_2m_min, with a positive coefficient of about 0.927.
# Holding other features constant, higher daily low temperatures are associated with higher predicted daily high temperatures.
# Adding precipitation, wind speed, and is_summer improved the test R² from about 0.857 to 0.895.
# One result that surprised me was that is_summer had a positive
# coefficient of about 0.891, showing an additional summer effect after accounting for the other features.
# I will inspect the predicted-versus-actual plot to determine whether errors are larger at the high end, the low end, or roughly even across the range.



# ============================================================
# OPTIONAL EXTENSION A: Compare a second city


print("\nOPTIONAL EXTENSION A: SECOND CITY")

# Keep Morrisville, NC as the original city and compare Phoenix, AZ.
# Both cities use the same year and the same full-model features.

def get_city_weather(latitude, longitude, start_date, end_date, timezone):
    """Download daily weather data for another location."""
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": (
            "temperature_2m_max,temperature_2m_min,"
            "precipitation_sum,wind_speed_10m_max"
        ),
        "timezone": timezone,
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    city_df = pd.DataFrame(response.json()["daily"])
    city_df["date"] = pd.to_datetime(city_df["time"])
    city_df = city_df.drop(columns=["time"])
    city_df = city_df.dropna().copy()
    city_df["is_summer"] = (
        city_df["date"].dt.month.isin([6, 7, 8]).astype(int)
    )

    return city_df


def evaluate_full_model(data):
    """Fit the full model and return test RMSE and R²."""
    features = [
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
        "is_summer",
    ]

    X = data[features]
    y = data["temperature_2m_max"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        "rows": len(data),
        "test_rows": len(y_test),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "r2": r2_score(y_test, predictions),
    }

morrisville_extension = df.copy()

phoenix_extension = get_city_weather(
    latitude=33.4484,
    longitude=-112.0740,
    start_date="2023-01-01",
    end_date="2023-12-31",
    timezone="America/Phoenix",
)

morrisville_results = evaluate_full_model(morrisville_extension)
phoenix_results = evaluate_full_model(phoenix_extension)

city_comparison = pd.DataFrame([
    {
        "City": "Morrisville, NC",
        "Rows": morrisville_results["rows"],
        "Test rows": morrisville_results["test_rows"],
        "RMSE (C)": morrisville_results["rmse"],
        "Test R2": morrisville_results["r2"],
    },
    {
        "City": "Phoenix, AZ",
        "Rows": phoenix_results["rows"],
        "Test rows": phoenix_results["test_rows"],
        "RMSE (C)": phoenix_results["rmse"],
        "Test R2": phoenix_results["r2"],
    },
])

print(city_comparison.to_string(index=False))

## Found: Phoenix had a higher test R² of 0.921 compared with Morrisville's 0.895, meaning the model explained a larger  proportion of variation in Phoenix's daily high temperatures.
# However, Morrisville had a slightly lower RMSE of 2.74°C compared with Phoenix's 2.83°C, so its predictions had smaller errors on average. The different results may be related to
# differences in climate and weather patterns between the cities.


# ============================================================
# OPTIONAL EXTENSION B: Use three years instead of one

print("\nOPTIONAL EXTENSION B: MORE YEARS")

weather_three_years = get_city_weather(
    latitude=35.82,
    longitude=-78.83,
    start_date="2021-01-01",
    end_date="2023-12-31",
    timezone="America/New_York",
)

sone_year_results = evaluate_full_model(df)
three_year_results = evaluate_full_model(weather_three_years)

years_comparison = pd.DataFrame([
    {
        "Period": "2023 only",
        "Rows": one_year_results["rows"],
        "Test rows": one_year_results["test_rows"],
        "RMSE (C)": one_year_results["rmse"],
        "Test R2": one_year_results["r2"],
    },
    {
        "Period": "2021-2023",
        "Rows": three_year_results["rows"],
        "Test rows": three_year_results["test_rows"],
        "RMSE (C)": three_year_results["rmse"],
        "Test R2": three_year_results["r2"],
    },
])

print(years_comparison.to_string(index=False))

rmse_change = (
    three_year_results["rmse"] - one_year_results["rmse"]
)
r2_change = three_year_results["r2"] - one_year_results["r2"]

print(f"Change in RMSE: {rmse_change:.3f} degrees C")
print(f"Change in test R2: {r2_change:.3f}")

# Found: Using three years of data increased the dataset from 365 to 1,095 days. However, test RMSE increased from 2.74°C to 3.07°C, and test R² decreased from 0.895 to 0.879.
# This is a small decline in model performance on this split.
# Adding more data did not automatically improve the predictions, possibly because weather patterns varied across the three years.
# Both models use a random 80/20 split, so this comparison does not measure performance exclusively on a future year.


# ============================================================
# OPTIONAL EXTENSION C: Compare seasonality representations


print("\nOPTIONAL EXTENSION C: SEASONALITY")

seasonal_data = df.copy()
seasonal_data["month"] = seasonal_data["date"].dt.month

common_features = [
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

# Model 1: The original is_summer feature.
summer_features = common_features + ["is_summer"]
summer_results = evaluate_full_model(seasonal_data)

# Model 2: Numeric month, where month 1 through 12 is one number.
numeric_month_data = seasonal_data.copy()
numeric_month_features = common_features + ["month"]


def evaluate_with_features(data, features):
    """Evaluate a model with a selected set of features."""
    X = data[features]
    y = data["temperature_2m_max"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        "rows": len(data),
        "test_rows": len(y_test),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "r2": r2_score(y_test, predictions),
    }


numeric_month_results = evaluate_with_features(
    numeric_month_data, numeric_month_features
)

# Model 3: One-hot encoded month features.
one_hot_data = pd.get_dummies(
    seasonal_data,
    columns=["month"],
    prefix="month",
    drop_first=True,
    dtype=int,
)

month_features = [
    column for column in one_hot_data.columns
    if column.startswith("month_")
]

one_hot_features = common_features + month_features

one_hot_results = evaluate_with_features(
    one_hot_data, one_hot_features
)

seasonality_comparison = pd.DataFrame([
    {
        "Representation": "is_summer",
        "RMSE (C)": summer_results["rmse"],
        "Test R2": summer_results["r2"],
    },
    {
        "Representation": "Numeric month",
        "RMSE (C)": numeric_month_results["rmse"],
        "Test R2": numeric_month_results["r2"],
    },
    {
        "Representation": "One-hot months",
        "RMSE (C)": one_hot_results["rmse"],
        "Test R2": one_hot_results["r2"],
    },
])

print(seasonality_comparison.to_string(index=False))

best_seasonality = seasonality_comparison.loc[
    seasonality_comparison["RMSE (C)"].idxmin()
]

print(
    f"Lowest test RMSE: {best_seasonality['Representation']} "
    f"({best_seasonality['RMSE (C)']:.3f} degrees C)"
)

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

city_comparison.to_csv(
    OUTPUT_DIR / "extension_a_city_comparison.csv", index=False
)
years_comparison.to_csv(
    OUTPUT_DIR / "extension_b_years_comparison.csv", index=False
)
seasonality_comparison.to_csv(
    OUTPUT_DIR / "extension_c_seasonality_comparison.csv", index=False
)

print("\nAll optional extension comparisons have finished.")

# Found: One-hot encoded months performed best, with an RMSE of 2.55°C and test R² of 0.910. Numeric month performed slightly better than is_summer, with RMSE of 2.72°C and test R² of 0.897.
# The is_summer model had an RMSE of 2.74°C and test R² of 0.895.
# One-hot encoding allows each month to have its own effect, while numeric month assumes a linear relationship between the month number and temperature. This flexibility may explain why
# one-hot encoding predicted daily high temperatures better.
