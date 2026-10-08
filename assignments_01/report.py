
import json

from weatherkit import DailyAggregator, WeatherResponse, to_readings


def main() -> None:
    """Load, validate, transform, and summarize weather data."""
    with open("weather_raw.json", "r") as file:
        data = json.load(file)

    response = WeatherResponse.model_validate(data)
#conversion
    readings = to_readings(response)
#aggregation
    aggregator = DailyAggregator()

    summaries = aggregator.summarize(readings)
    incomplete = aggregator.incomplete_days(readings)

    print(
        f"{'Date':<12}"
        f"{'High (°C)':>12}"
        f"{'Low (°C)':>12}"
        f"{'Precip (mm)':>15}"
        f"{'Range (°C)':>13}"
    )

    print("-" * 64)

    for summary in summaries:
        print(
            f"{summary.date:<12}"
            f"{summary.temp_max:>12.1f}"
            f"{summary.temp_min:>12.1f}"
            f"{summary.precipitation_sum:>15.1f}"
            f"{summary.temp_range():>13.1f}"
        )

    if incomplete:
        print(
            "WARNING: Incomplete days dropped: "
            + ", ".join(incomplete)
        )
    else:
        print("WARNING: Incomplete days dropped: none")


# Without this guard, importing report.py would immediately run the
# weather-loading and reporting code instead of only making its functions
# available for reuse.
if __name__ == "__main__":
    main()


# Reflection
#
# 1. WeatherResponse rejects the whole file when one temperature is null.
# This is useful when the pipeline requires complete, trustworthy observations
# and a missing value means the source response is invalid. However, a weather
# pipeline might reasonably tolerate an occasional missing temperature and
# continue processing the other observations. To tolerate the gap, the
# temperature_2m field could be changed to list[float | None], and the
# downstream code would need to decide how to handle None values.
#
# 2. DailyAggregator defaults to 24 hours, so if a pipeline runs at noon,
# the current day may only contain about half a day's observations. That day
# would be dropped rather than treated as a complete day. incomplete_days()
# makes that decision visible by reporting which dates were dropped, rather
# than silently losing them.
#
# 3. A package makes it easier in Week 10 to import the validated schemas,
# conversion functions, and aggregation classes from one reusable module
# instead of copying the weather-processing code into another pipeline.