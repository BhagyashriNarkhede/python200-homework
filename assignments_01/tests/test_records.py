import json
from pathlib import Path

from weatherkit import HourlyReading, WeatherResponse, to_readings


def load_response() -> WeatherResponse:
    """Load and validate the real weather response."""
    path = Path(__file__).parent.parent / "weather_raw.json"

    with open(path, "r") as file:
        data = json.load(file)

    return WeatherResponse.model_validate(data)


def test_to_readings_returns_one_reading_per_hour_in_order():
    response = load_response()

    readings = to_readings(response)

    assert len(readings) == 168
    assert readings[0].timestamp == response.hourly.time[0]
    assert readings[-1].timestamp == response.hourly.time[-1]


def test_reading_values_match_input_indices():
    response = load_response()

    readings = to_readings(response)

    for i in [0, 50, 100, 167]:
        assert readings[i].timestamp == response.hourly.time[i]
        assert readings[i].temperature_c == response.hourly.temperature_2m[i]
        assert readings[i].precipitation_mm == response.hourly.precipitation[i]


def test_hourly_readings_with_identical_fields_compare_equal():
    reading1 = HourlyReading(
        timestamp="2026-04-08T00:00",
        temperature_c=16.8,
        precipitation_mm=0.0,
    )

    reading2 = HourlyReading(
        timestamp="2026-04-08T00:00",
        temperature_c=16.8,
        precipitation_mm=0.0,
    )

    assert reading1 == reading2

    # Deliberate-break check: changing timestamp=hourly.time[i] to
# timestamp=hourly.time[0] caused the order and index-matching tests to fail.