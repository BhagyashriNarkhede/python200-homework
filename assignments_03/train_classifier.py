
import sys
import json
from pathlib import Path

import requests
import pandas as pd
import sklearn
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# --- Task 1: Fetch the Data ---

# City chosen: Morrisville, North Carolina.
# These coordinates are approximate coordinates for Morrisville.
city = "Morrisville, North Carolina"

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 35.8235,
    "longitude": -78.8256,
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
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

# Extension A: Create a season feature from the month.
df["season"] = df["date"].dt.month.map({
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "fall",
    10: "fall",
    11: "fall",
})

print("Task 1 - Weather data shape:")
print(df.shape)
print("\nFirst five rows:")
print(df[["date", "season"]].head())


# --- Task 2: Engineer the Label ---

# A good running day has:
# high temperature between 7 and 26 degrees Celsius,
# low temperature of at least 0 degrees Celsius,
# precipitation under 3 mm,
# maximum wind speed under 30 km/h.
#
# These are the assignment's original thresholds.

df["good_for_running"] = (
    df["temperature_2m_max"].between(7, 26)
    & (df["temperature_2m_min"] >= 0)
    & (df["precipitation_sum"] < 3)
    & (df["wind_speed_10m_max"] < 30)
).astype(int)

good_days = int(df["good_for_running"].sum())
skip_days = int((df["good_for_running"] == 0).sum())
good_fraction = good_days / len(df)

print("\nTask 2 - Class balance:")
print(f"Good days (1): {good_days}")
print(f"Skip days (0): {skip_days}")
print(f"Fraction of days that are good: {good_fraction:.2%}")

# Check the class balance before evaluating the model.
# If the classes are imbalanced, precision, recall, and F1 are
# important alongside accuracy because accuracy alone can be misleading.
# The fraction of good days helps us judge whether the labels seem
# reasonable for Morrisville's climate.


# --- Prepare the Features and Split the Data ---

features = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

X = df[features]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# About 34.79% of days were labeled as good running days.
# This seems reasonable because the rules exclude days that are
# too cold, too hot, rainy, or very windy.

# --- Task 3: Build and Evaluate a KNN Classifier ---

k_values = [3, 5, 7, 9, 11, 15, 21]
best_k = None
best_cv_accuracy = -1

print("\nTask 3 - KNN cross-validation:")

for k in k_values:
    knn_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ]
    )

    scores = cross_val_score(
        knn_pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
    )

    mean_accuracy = scores.mean()
    print(f"k = {k}, mean CV accuracy = {mean_accuracy:.4f}")

    if mean_accuracy > best_cv_accuracy:
        best_cv_accuracy = mean_accuracy
        best_k = k

print(f"\nBest k: {best_k}")
print(f"Best mean CV accuracy: {best_cv_accuracy:.4f}")

knn_model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=best_k)),
    ]
)

knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_pred)
knn_f1 = f1_score(y_test, knn_pred, pos_label=1)

print("\nKNN test accuracy:")
print(knn_accuracy)

print("\nKNN classification report:")
print(
    classification_report(
        y_test,
        knn_pred,
        labels=[0, 1],
        target_names=["skip", "good"],
        zero_division=0,
    )
)


# --- Task 4: Build and Evaluate Logistic Regression ---

logreg_model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000)),
    ]
)

logreg_model.fit(X_train, y_train)
logreg_pred = logreg_model.predict(X_test)

logreg_accuracy = accuracy_score(y_test, logreg_pred)
logreg_precision = precision_score(
    y_test, logreg_pred, pos_label=1, zero_division=0
)
logreg_recall = recall_score(
    y_test, logreg_pred, pos_label=1, zero_division=0
)
logreg_f1 = f1_score(
    y_test, logreg_pred, pos_label=1, zero_division=0
)

print("\nTask 4 - Logistic Regression results:")
print(f"Accuracy:  {logreg_accuracy:.4f}")
print(f"Precision: {logreg_precision:.4f}")
print(f"Recall:    {logreg_recall:.4f}")
print(f"F1-score:  {logreg_f1:.4f}")

print("\nLogistic Regression classification report:")
print(
    classification_report(
        y_test,
        logreg_pred,
        labels=[0, 1],
        target_names=["skip", "good"],
        zero_division=0,
    )
)

coefficients = logreg_model.named_steps["logreg"].coef_[0]

print("\nLogistic Regression feature coefficients:")
for feature, coefficient in zip(features, coefficients):
    print(f"{feature}: {coefficient:.4f}")

# Interpretation:
# precipitation_sum has the most negative coefficient (-2.5428).
# More precipitation pushes the prediction toward "skip",
# holding the other weather features constant.
# This makes sense because rainy weather is less suitable for running.
    
# --- Extension A: Add Season ---

numeric_features = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

X_season = df[numeric_features + ["season"]]
y_season = df["good_for_running"]

X_train_season, X_test_season, y_train_season, y_test_season = (
    train_test_split(
        X_season,
        y_season,
        test_size=0.2,
        random_state=42,
        stratify=y_season,
    )
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        ("season", OneHotEncoder(handle_unknown="ignore"), ["season"]),
    ]
)

season_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("knn", KNeighborsClassifier(n_neighbors=best_k)),
    ]
)

season_pipeline.fit(X_train_season, y_train_season)
season_pred = season_pipeline.predict(X_test_season)

season_accuracy = accuracy_score(y_test_season, season_pred)
season_f1 = f1_score(
    y_test_season, season_pred, pos_label=1, zero_division=0
)

print("\nExtension A - KNN with season:")
print(f"Accuracy: {season_accuracy:.4f}")
print(f"F1-score: {season_f1:.4f}")

print(f"Original KNN accuracy: {knn_accuracy:.4f}")
print(f"KNN accuracy with season: {season_accuracy:.4f}")

# Compare the results after running the script.
# Season may help because running conditions vary by time of year.
# It may also add little benefit because temperature and precipitation
# already capture some seasonal weather patterns.


# Negative coefficients push predictions toward skip (class 0).
# Positive coefficients push predictions toward good (class 1).
# The most negative coefficients identify conditions that most strongly
# push the prediction toward skip, holding the other features constant.


# --- Task 5: Compare and Choose ---

print("\nTask 5 - Model comparison:")
print(f"KNN accuracy:                 {knn_accuracy:.4f}")
print(f"Logistic Regression accuracy: {logreg_accuracy:.4f}")
print(f"KNN F1-score:                 {knn_f1:.4f}")
print(f"Logistic Regression F1-score: {logreg_f1:.4f}")

# Select the model with higher test accuracy.
# If accuracy is tied, select the model with higher F1-score.
if (knn_accuracy, knn_f1) >= (logreg_accuracy, logreg_f1):
    chosen_model = knn_model
    chosen_name = "KNN"
    chosen_accuracy = knn_accuracy
    chosen_f1 = knn_f1
else:
    chosen_model = logreg_model
    chosen_name = "Logistic Regression"
    chosen_accuracy = logreg_accuracy
    chosen_f1 = logreg_f1

print(f"\nSelected model for deployment: {chosen_name}")

# Accuracy is useful, but it may not be the most important metric.
# A false positive means recommending a run on a day labeled skip.
# A false negative means missing a day labeled good.
# For a running app, false positives may be more concerning, so precision
# for the good class deserves attention. Recall matters too because it
# measures how many good days the model identifies.
#
# The selected model is chosen by test accuracy and F1 as a simple rule.
# KNN is straightforward but stores training examples; Logistic Regression
# is compact and easier to interpret. Both pipelines support probabilities.
# Consider false positives and the evaluation results before deployment.


# --- Task 6: Save the Model and Metadata ---

models_dir = Path("models")
models_dir.mkdir(parents=True, exist_ok=True)

model_path = models_dir / "weather_classifier.pkl"
metadata_path = models_dir / "weather_classifier_metadata.json"

joblib.dump(chosen_model, model_path)

metadata = {
    "city": city,
    "python_version": sys.version.split()[0],
    "scikit_learn_version": sklearn.__version__,
    "features": features,
    "label_rule": (
        "good=1 when maximum temperature is between 7 and 26 degrees C "
        "(inclusive), minimum temperature is at least 0 degrees C, "
        "precipitation is under 3 mm, and maximum wind speed is under "
        "30 km/h; otherwise skip=0"
    ),
    "selected_model": chosen_name,
    "test_accuracy": float(chosen_accuracy),
    "test_f1": float(chosen_f1),
}

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print(f"\nSaved model to: {model_path}")
print(f"Saved metadata to: {metadata_path}")
print("Model and metadata saved successfully.")
