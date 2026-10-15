import pytest

from weatherkit import DailyAggregator, HourlyReading


@pytest.fixture
def sample_readings():
    """Provide shared hourly readings for aggregation tests."""
    return [
        HourlyReading("2026-04-08T00:00", 10.0, 0.5),
        HourlyReading("2026-04-08T01:00", 15.0, 1.0),
        HourlyReading("2026-04-08T02:00", 12.0, 0.5),
        HourlyReading("2026-04-09T00:00", 8.0, 0.0),
        HourlyReading("2026-04-09T01:00", 18.0, 2.0),
        HourlyReading("2026-04-09T02:00", 14.0, 1.0),
    ]


def test_grouping_produces_two_daily_summaries(sample_readings):
    aggregator = DailyAggregator(min_hours=3)

    summaries = aggregator.summarize(sample_readings)

    assert len(summaries) == 2
    assert summaries[0].date == "2026-04-08"
    assert summaries[1].date == "2026-04-09"


def test_temperature_max_and_min_are_correct(sample_readings):
    aggregator = DailyAggregator(min_hours=3)

    summaries = aggregator.summarize(sample_readings)

    assert summaries[0].temp_max == 15.0
    assert summaries[0].temp_min == 10.0
    assert summaries[1].temp_max == 18.0
    assert summaries[1].temp_min == 8.0


def test_precipitation_sum_is_correct(sample_readings):
    aggregator = DailyAggregator(min_hours=3)

    summaries = aggregator.summarize(sample_readings)

    assert summaries[0].precipitation_sum == pytest.approx(2.0)
    assert summaries[1].precipitation_sum == pytest.approx(3.0)


def test_incomplete_day_is_dropped_and_reported(sample_readings):
    aggregator = DailyAggregator(min_hours=4)

    summaries = aggregator.summarize(sample_readings)
    incomplete = aggregator.incomplete_days(sample_readings)

    assert len(summaries) == 0
    assert incomplete == ["2026-04-08", "2026-04-09"]


@pytest.mark.parametrize("min_hours, expected_count", [
    (2, 2),
    (3, 2),
    (4, 0),
])
def test_min_hours_controls_which_days_are_kept(
    sample_readings, min_hours, expected_count
):
    aggregator = DailyAggregator(min_hours=min_hours)

    summaries = aggregator.summarize(sample_readings)

    assert len(summaries) == expected_count