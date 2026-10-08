from dataclasses import dataclass

from .records import HourlyReading


@dataclass
class DailySummary:
    """Summary of hourly weather observations for one calendar day.

    Attributes:
        date: Calendar date in YYYY-MM-DD format.
        temp_max: Maximum temperature in degrees Celsius.
        temp_min: Minimum temperature in degrees Celsius.
        precipitation_sum: Total precipitation in millimeters.
        hours_observed: Number of hourly observations included.
    """

    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int

    def temp_range(self) -> float:
        """Return the difference between the daily maximum and minimum temperature."""
        return self.temp_max - self.temp_min

class DailyAggregator:
    """Aggregate hourly weather readings into daily summaries."""

    def __init__(self, min_hours: int = 24) -> None:
        """Initialize the aggregator.

        Args:
            min_hours: Minimum number of hourly observations required
                for a day to be included in the summaries.
        """
        self.min_hours = min_hours

    def summarize(
        self, readings: list[HourlyReading]
    ) -> list[DailySummary]:
        """Group hourly readings into daily summaries.

        Args:
            readings: Hourly weather readings to aggregate.

        Returns:
            Daily summaries sorted by date. Days with fewer than
            min_hours observations are excluded.
        """
        grouped: dict[str, list[HourlyReading]] = {}

        for reading in readings:
            date = reading.timestamp[:10]
            grouped.setdefault(date, []).append(reading)

        summaries = []

        for date, day_readings in grouped.items():
            if len(day_readings) < self.min_hours:
                continue

            temperatures = [reading.temperature_c for reading in day_readings]
            precipitation = [
                reading.precipitation_mm for reading in day_readings
            ]

            summaries.append(
                DailySummary(
                    date=date,
                    temp_max=max(temperatures),
                    temp_min=min(temperatures),
                    precipitation_sum=sum(precipitation),
                    hours_observed=len(day_readings),
                )
            )

        return sorted(summaries, key=lambda summary: summary.date)
    def incomplete_days(self, readings: list[HourlyReading]) -> list[str]:
        """Return dates that have fewer than min_hours observations.

        Args:
            readings: Hourly weather readings to inspect.

        Returns:
            Dates with fewer than min_hours observations, sorted by date.
        """
        counts: dict[str, int] = {}

        for reading in readings:
            date = reading.timestamp[:10]
            counts[date] = counts.get(date, 0) + 1

        return sorted(
            date
            for date, count in counts.items()
            if count < self.min_hours
        )