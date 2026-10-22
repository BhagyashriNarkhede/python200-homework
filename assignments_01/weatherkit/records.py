from dataclasses import dataclass

from .schemas import WeatherResponse


@dataclass
class HourlyReading:
    """A single hourly weather reading.

    Attributes:
        timestamp: Observation timestamp as a string.
        temperature_c: Temperature in degrees Celsius.
        precipitation_mm: Precipitation in millimeters.
    """

    timestamp: str
    temperature_c: float
    precipitation_mm: float

def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Convert the columnar hourly response into individual readings.

    Args:
        response: A validated Open-Meteo weather response.

    Returns:
        A list of HourlyReading objects in the same order as the API data.
    """
    hourly = response.hourly

    return [
        HourlyReading(
            timestamp=hourly.time[i],
            temperature_c=hourly.temperature_2m[i],
            precipitation_mm=hourly.precipitation[i],
        )
        for i in range(len(hourly.time))
    ]

# WeatherResponse is a Pydantic model because it sits at the API boundary
# and must validate untrusted external data. HourlyReading is a dataclass
# because it is an internal representation created after that validation,
# so it does not need to validate the API input again.