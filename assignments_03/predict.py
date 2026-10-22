
import json
from pathlib import Path

import joblib
import pandas as pd


# --- Task 7: Load the Saved Model and Metadata ---

model_path = Path("models/weather_classifier.pkl")
metadata_path = Path("models/weather_classifier_metadata.json")

# Give a helpful error if the training script has not been run.
if not model_path.exists() or not metadata_path.exists():
    raise FileNotFoundError(
        "The trained model or metadata was not found. "
        "Run 'python train_classifier.py' from the assignments_03 "
        "folder before running predict.py."
    )

model = joblib.load(model_path)

with open(metadata_path, "r", encoding="utf-8") as file:
    metadata = json.load(file)

print("Weather classifier metadata")
print(f"City: {metadata['city']}")
print(f"Features: {metadata['features']}")
print(f"Test accuracy: {metadata['test_accuracy']:.4f}")
print(f"Selected model: {metadata['selected_model']}")


# --- Create Hypothetical Weather Days ---

# The columns come from metadata so their names and order match training.
features = metadata["features"]

hypothetical_days = pd.DataFrame(
    [
        # Clearly good: mild, dry, and low wind.
        [20.0, 12.0, 0.0, 10.0],

        # Clearly bad: heavy rain and strong wind.
        [18.0, 10.0, 12.0, 40.0],

        # Borderline: temperature and wind are acceptable,
        # but precipitation is close to the 3 mm threshold.
        [20.0, 10.0, 2.8, 15.0],

        # Cold day: minimum temperature is below freezing.
        [5.0, -3.0, 0.0, 8.0],
    ],
    columns=features,
    index=["clearly_good", "rainy_windy", "borderline", "cold"],
)

predictions = model.predict(hypothetical_days)
probabilities = model.predict_proba(hypothetical_days)

# Find which probability column represents class 1 (good).
good_index = list(model.classes_).index(1)

print("\nTask 7 - Predictions:")

for i, day_name in enumerate(hypothetical_days.index):
    predicted_label = "good" if predictions[i] == 1 else "skip"
    good_probability = probabilities[i, good_index]

    print(f"\nDay: {day_name}")
    print(hypothetical_days.iloc[i].to_dict())
    print(f"Predicted label: {predicted_label}")
    print(f"Probability of good: {good_probability:.2%}")


# --- Task 8: Reflection ---

#
# --- Task 8: Reflection ---

# 1. Borderline day:
# My borderline day was predicted as good with a probability of 100%.
# This is the KNN model's estimate based on its nearest neighbors.
# It does not guarantee that the weather is safe for running.
# If the model predicts a probability of 0.52, I would consider it
# uncertain and check the actual weather before making a decision.

# 2. Running predict.py before train_classifier.py:
# The saved model and metadata files would not exist, so prediction
# would fail. The FileNotFoundError message explains that I need to
# run train_classifier.py first.

# 3. Week 10 cloud pipeline:
# My current script uses four hard-coded hypothetical days.
# A cloud pipeline would need to receive new daily weather data automatically, validate the input, and make predictions for it.

