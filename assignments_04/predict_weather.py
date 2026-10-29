
"""Run weather predictions using the saved classifier."""

from pathlib import Path

from weather_model import WeatherClassifier


def main() -> None:
    """Load the saved model and predict four example days."""
    model_path = Path(__file__).parent / "models" / "weather_classifier.pkl"
    classifier = WeatherClassifier(model_path)

    examples = [
        (
            "Clearly good",
            {
                "temperature_2m_max": 25,
                "temperature_2m_min": 15,
                "precipitation_sum": 0,
                "wind_speed_10m_max": 10,
            },
        ),
        (
            "Clearly bad",
            {
                "temperature_2m_max": 2,
                "temperature_2m_min": -5,
                "precipitation_sum": 25,
                "wind_speed_10m_max": 40,
            },
        ),
        (
            "Borderline",
            {
                "temperature_2m_max": 18,
                "temperature_2m_min": 10,
                "precipitation_sum": 2,
                "wind_speed_10m_max": 18,
            },
        ),
        (
            "Cold day",
            {
                "temperature_2m_max": 5,
                "temperature_2m_min": -2,
                "precipitation_sum": 0,
                "wind_speed_10m_max": 12,
            },
        ),
    ]

    for description, day in examples:
        prediction = classifier.predict([day])[0]

        print(
            f"{description}: inputs={day}, "
            f"label={prediction.label}, "
            f"probability_good={prediction.probability:.1%}"
        )


if __name__ == "__main__":
    main()

# Without the main guard, importing this module would run the predictions
# automatically, even if another program only wanted to reuse a helper.


# Reflection
#
# 1. Fixed feature order prevents a column-order bug. For example, wind speed
#    could accidentally be passed where the model expects maximum temperature.
#
# 2. A tiny test model makes tests independent of the real model file and
#    allows the tests to create predictable examples in a temporary directory.
#
# 3. Readiness for Week 10: the classifier has a reusable interface and
#    validates missing features. Before production, I would add input-value
#    validation, logging, and monitoring for prediction quality.
