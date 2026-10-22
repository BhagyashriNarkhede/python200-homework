
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import joblib

df = pd.read_csv("weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# --- Preprocessing ---
# Q1

scaler = StandardScaler()

# Fit only on training data to avoid leaking information from the test data.
X_train_scaled = scaler.fit_transform(X_train)

# Use the same scaler to transform the test data.
X_test_scaled = scaler.transform(X_test)

# Print the mean of each scaled training column.
print("Q1 - Scaled training column means:")
print(X_train_scaled.mean(axis=0))

# Fit only on training data to avoid leaking information from the test data.

# Q2

encoder = OneHotEncoder(sparse_output=False)

season_encoded = encoder.fit_transform(df[["season"]])

print("Q2 - Encoded season array shape:")
print(season_encoded.shape)

print("Q2 - Category names:")
print(encoder.get_feature_names_out(["season"]))

# The season column becomes one column per unique season because
# one-hot encoding creates a separate binary feature for each category.


# Q3

# Include season along with the four numeric features.
X = df[numeric + ["season"]]

# Use the same train/test split settings as the instructor's setup.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply the correct preprocessing to each group of columns.
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric),
        ("season", OneHotEncoder(sparse_output=False), ["season"]),
    ]
)

# Combine preprocessing and classification into one pipeline.
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression()),
    ]
)

# Train the complete pipeline.
pipeline.fit(X_train, y_train)

# Predict the target for the test data.
y_pred = pipeline.predict(X_test)

# Print test accuracy.
print("Q3 - Pipeline test accuracy:")
print(accuracy_score(y_test, y_pred))

# The pipeline automatically applies the same fitted preprocessing
# to new data before making predictions, so I do not have to do it by hand.

# KNN uses the four numeric columns from the original setup.
X = df[numeric]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- KNN ---

# Q1

# Build a pipeline that scales the data and then runs KNN.
knn_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ]
)

# Train the model using the training data.
knn_pipeline.fit(X_train, y_train)

# Predict the target for the test data.
knn_pred = knn_pipeline.predict(X_test)

# Print test accuracy.
print("KNN Q1 - Test accuracy:")
print(accuracy_score(y_test, knn_pred))

# Print the full classification report.
print("KNN Q1 - Classification report:")
print(classification_report(y_test, knn_pred))


# Q2

k_values = [1, 3, 5, 7, 9, 11, 15, 21]

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

    print(f"k = {k}, Mean CV accuracy = {scores.mean():.4f}")

# I would choose the k with the highest mean cross-validation accuracy
# because it performed best across the five validation folds.


# Q3

# KNN without scaling.
knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_raw.fit(X_train, y_train)

raw_pred = knn_raw.predict(X_test)
raw_accuracy = accuracy_score(y_test, raw_pred)

print("KNN Q3 - Accuracy without scaling:")
print(raw_accuracy)

# KNN with scaling.
knn_scaled = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ]
)

knn_scaled.fit(X_train, y_train)

scaled_pred = knn_scaled.predict(X_test)
scaled_accuracy = accuracy_score(y_test, scaled_pred)

print("KNN Q3 - Accuracy with scaling:")
print(scaled_accuracy)

# Scaling helps when features use different ranges because KNN relies
# on distances between data points. Compare the two accuracies to see
# whether scaling helped on this dataset.


# --- Logistic Regression ---

# Q1

logreg_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000)),
    ]
)

logreg_pipeline.fit(X_train, y_train)

logreg_pred = logreg_pipeline.predict(X_test)

print("Logistic Regression Q1 - Accuracy:")
print(accuracy_score(y_test, logreg_pred))

print("Logistic Regression Q1 - Precision:")
print(precision_score(y_test, logreg_pred, pos_label=1))

print("Logistic Regression Q1 - Recall:")
print(recall_score(y_test, logreg_pred, pos_label=1))

print("Logistic Regression Q1 - F1-score:")
print(f1_score(y_test, logreg_pred, pos_label=1))


# Q2

feature_names = numeric
coefficients = logreg_pipeline.named_steps["logreg"].coef_[0]

print("Logistic Regression Q2 - Feature coefficients:")

for feature, coefficient in zip(feature_names, coefficients):
    print(f"{feature}: {coefficient:.4f}")

# Negative coefficients push predictions toward class 0 (skip),
# while positive coefficients push predictions toward class 1 (good).
# After checking the output, identify the most negative feature and
# consider whether its effect matches your intuition about running weather.


# Q3

probabilities = logreg_pipeline.predict_proba(X_test)

# Find the probability column corresponding to class 1 (good).
good_index = list(logreg_pipeline.named_steps["logreg"].classes_).index(1)
good_probabilities = probabilities[:, good_index]

print("Logistic Regression Q3 - Probability of good for first five test days:")

for probability in good_probabilities[:5]:
    print(round(probability, 2))

# A probability near 0.5 means the model is less certain which class
# to predict because the probability is close to its decision threshold.


# --- Evaluation ---

# Q1

cm = confusion_matrix(y_test, logreg_pred, labels=[0, 1])

print("Evaluation Q1 - Confusion matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["skip", "good"],
)

display.plot()
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/logreg_confusion_matrix.png")
plt.show()

false_positives = cm[0, 1]
false_negatives = cm[1, 0]

print("False positives:", false_positives)
print("False negatives:", false_negatives)

# A false positive means predicting good when the day is actually skip.
# A false negative means predicting skip when the day is actually good.
# For a running app, false positives may be more concerning if they
# encourage someone to run in unsuitable weather day.


# --- Evaluation ---

# Q1

cm = confusion_matrix(y_test, logreg_pred, labels=[0, 1])

print("Evaluation Q1 - Confusion matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["skip", "good"],
)

display.plot()
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/logreg_confusion_matrix.png")
plt.show()

false_positives = cm[0, 1]
false_negatives = cm[1, 0]

print("False positives:", false_positives)
print("False negatives:", false_negatives)

# A false positive means predicting good when the day is actually skip.
# A false negative means predicting skip when the day is actually good.
# For a running app, false positives may be more concerning if they
# encourage someone to run in unsuitable weather.

# --- joblib ---

# Q1

# Save the entire fitted pipeline, including the scaler and classifier.
joblib.dump(logreg_pipeline, "models/warmup_model.pkl")

# Load the saved pipeline.
loaded = joblib.load("models/warmup_model.pkl")

# Confirm that the loaded model makes identical predictions.
assert (loaded.predict(X_test) == logreg_pipeline.predict(X_test)).all()

print("Predictions match.")

# If I saved only LogisticRegression without the scaler, raw data would
# not be scaled as during training, which could change the predictions.
