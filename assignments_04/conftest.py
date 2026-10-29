
"""Reusable weather classification component."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd


@dataclass
class Prediction:
    """Store a running-weather prediction and its probability."""

    label: str
    probability: float


class WeatherClassifier:
    """Load a saved weather model and predict running conditions."""

    FEATURES = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]

    def __init__(self, model_path: str | Path) -> None:
        """Load a saved model pipeline.

        Args:
            model_path: Path to the saved .pkl model file.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        path = Path(model_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Weather model file not found: {path}"
            )

        self.model = joblib.load(path)

    def predict(self, days: list[dict[str, float]]) -> list[Prediction]:
        """Predict whether each day is good for running.

        Args:
            days: Weather records containing the four required features.

        Returns:
            One Prediction object per input day, in the same order.

        Raises:
            ValueError: If one or more required features are missing.
        """
        if not days:
            return []

        missing = [
            feature
            for feature in self.FEATURES
            if any(feature not in day for day in days)
        ]

        if missing:
            raise ValueError(
                f"Missing required weather features: {', '.join(missing)}"
            )

        # Always use the same feature order that the model was trained on.
        data = pd.DataFrame(
            [
                {feature: day[feature] for feature in self.FEATURES}
                for day in days
            ],
            columns=self.FEATURES,
        )

        labels = self.model.predict(data)
        probabilities = self.model.predict_proba(data)[:, 1]

        predictions = []

        for label, probability in zip(labels, probabilities):
            predictions.append(
                Prediction(
                    label="good" if label == 1 else "skip",
                    probability=float(probability),
                )
            )

        return predictions
