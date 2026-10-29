
"""Tests for the reusable weather classifier."""

import joblib
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from weather_model import Prediction, WeatherClassifier


@pytest.fixture
def classifier(tmp_path):
    """Build and save a tiny test model in a temporary directory."""
    features = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]

    # Four good-weather examples followed by four bad-weather examples.
    training_data = pd.DataFrame(
        [
            [25, 15, 0, 10],
            [28, 18, 0, 8],
            [23, 13, 1, 12],
            [20, 12, 0, 15],
            [2, -5, 25, 40],
            [5, 0, 18, 35],
            [0, -8, 30, 50],
            [8, 1, 10, 25],
        ],
        columns=features,
    )

    labels = [1, 1, 1, 1, 0, 0, 0, 0]

    model = DecisionTreeClassifier(random_state=42)
    model.fit(training_data, labels)

    model_path = tmp_path / "test_weather_model.pkl"
    joblib.dump(model, model_path)

    return WeatherClassifier(model_path)


GOOD_DAY = {
    "temperature_2m_max": 25,
    "temperature_2m_min": 15,
    "precipitation_sum": 0,
    "wind_speed_10m_max": 10,
}

BAD_DAY = {
    "temperature_2m_max": 2,
    "temperature_2m_min": -5,
    "precipitation_sum": 25,
    "wind_speed_10m_max": 40,
}


# Tests 1 and 2: good and bad days receive the correct labels.
@pytest.mark.parametrize(
    ("day", "expected_label"),
    [
        (GOOD_DAY, "good"),
        (BAD_DAY, "skip"),
    ],
)
def test_prediction_labels(classifier, day, expected_label):
    """Check labels for clearly good and bad weather."""
    result = classifier.predict([day])

    assert result[0].label == expected_label


# Test 3: probability stays within the valid range.
def test_probability_is_between_zero_and_one(classifier):
    """Check that the probability is between 0 and 1."""
    result = classifier.predict([GOOD_DAY])

    assert 0.0 <= result[0].probability <= 1.0


# Test 4: multiple days return predictions in the same order.
def test_multiple_days_preserve_order(classifier):
    """Check the number, types, and order of predictions."""
    days = [GOOD_DAY, BAD_DAY, GOOD_DAY]

    result = classifier.predict(days)

    assert len(result) == 3
    assert all(isinstance(item, Prediction) for item in result)
    assert [item.label for item in result] == ["good", "skip", "good"]


# Test 5: an empty list returns an empty list.
def test_empty_days_returns_empty_list(classifier):
    """Check that no input days produce no predictions."""
    assert classifier.predict([]) == []


# Test 6: missing features raise a helpful ValueError.
def test_missing_feature_raises_value_error(classifier):
    """Check that missing required features are identified."""
    incomplete_day = {
        "temperature_2m_max": 25,
        "temperature_2m_min": 15,
        "precipitation_sum": 0,
        # wind_speed_10m_max is missing.
    }

    with pytest.raises(ValueError, match="wind_speed_10m_max"):
        classifier.predict([incomplete_day])


# Test 7: a missing model file raises FileNotFoundError.
def test_missing_model_file_raises_file_not_found_error(tmp_path):
    """Check that a missing saved model produces a clear error."""
    missing_path = tmp_path / "does_not_exist.pkl"

    with pytest.raises(FileNotFoundError, match="Weather model file not found"):
        WeatherClassifier(missing_path)


# Mutation-testing note:
# If test_prediction_labels always received "skip" as its prediction,
# the GOOD_DAY parameter case would fail. Restore the correct implementation
# after deliberately testing that failure.
