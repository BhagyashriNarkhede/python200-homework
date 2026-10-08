import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from weatherkit import WeatherResponse


def test_valid_response_loads():
    # A plain relative path depends on the directory pytest was started from.
    # Building the path from this test file works regardless of the working directory.
    path = Path(__file__).parent.parent / "weather_raw.json"

    with open(path, "r") as file:
        data = json.load(file)

    response = WeatherResponse.model_validate(data)

    assert len(response.hourly.time) == 168


def test_invalid_latitude_raises_validation_error():
    data = {
        "latitude": 200.0,
        "longitude": -80.0,
        "timezone": "GMT",
        "elevation": 100.0,
        "hourly": {
            "time": ["2026-04-08T00:00"],
            "temperature_2m": [16.8],
            "precipitation": [0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)


def test_mismatched_list_lengths_raise_validation_error():
    data = {
        "latitude": 35.0,
        "longitude": -80.0,
        "timezone": "GMT",
        "elevation": 100.0,
        "hourly": {
            "time": [
                "2026-04-08T00:00",
                "2026-04-08T01:00",
                "2026-04-08T02:00",
            ],
            "temperature_2m": [16.8, 16.5],
            "precipitation": [0.0, 0.0, 0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)


def test_null_temperature_raises_validation_error():
    data = {
        "latitude": 35.0,
        "longitude": -80.0,
        "timezone": "GMT",
        "elevation": 100.0,
        "hourly": {
            "time": ["2026-04-08T00:00"],
            "temperature_2m": [None],
            "precipitation": [0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)