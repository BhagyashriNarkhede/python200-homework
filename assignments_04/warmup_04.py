
import json
from dataclasses import dataclass

import pytest


# Question 1: ModelInfo

class ModelInfo:
    """Load model metadata from a JSON file."""

    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as file:
            self.data = json.load(file)

    def feature_count(self) -> int:
        """Return the number of model features."""
        return len(self.data["features"])


# Create a small JSON file and test ModelInfo.
metadata_path = "sample_metadata.json"

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(
        {"features": ["temperature", "rain", "wind"]},
        file,
    )

model_info = ModelInfo(metadata_path)
print("Feature count:", model_info.feature_count())

# Loading JSON in __init__ is better because the file is read only
# once, even when feature_count() is called many times.


# Question 2: DayForecast

@dataclass
class DayForecast:
    """Store a simple daily weather forecast."""

    date: str
    high_c: float
    will_rain: bool

    def summary(self) -> str:
        """Return a readable forecast sentence."""
        if self.will_rain:
            rain_message = "rain expected"
        else:
            rain_message = "no rain expected"

        return f"{self.date}: high {self.high_c:.1f}C, {rain_message}"


forecast1 = DayForecast("2023-06-15", 24.1, True)
forecast2 = DayForecast("2023-06-16", 27.5, False)

print(forecast1.summary())
print(forecast2.summary())

# @dataclass automatically creates an __init__ method for us.


# Question 3: parse_temperature

def parse_temperature(text: str) -> float:
    """Convert temperature text to a float."""
    try:
        return float(text)
    except ValueError as error:
        raise ValueError(
            f"Invalid temperature: {text}"
        ) from error


def test_parse_temperature_valid():
    """Test that valid temperature text becomes a float."""
    assert parse_temperature("24.1") == pytest.approx(24.1)


def test_parse_temperature_bad_input_raises():
    """Test that invalid temperature text raises ValueError."""
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")


# pytest.raises(ValueError) without match would not check that
# the error message contains the required word "temperature".
