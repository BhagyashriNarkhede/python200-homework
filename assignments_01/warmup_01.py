# --- Classes ---
# Q1

from pydantic import BaseModel, Field, ValidationError, model_validator

import pytest


class Thermometer:
    """Store temperature readings in Celsius for a location."""

    def __init__(self, location, readings=None):
        self.location = location

        if readings is None:
            self.readings = []
        else:
            self.readings = readings

    def add(self, reading):
        self.readings.append(reading)

    def average(self):
        if not self.readings:
            return None

        return sum(self.readings) / len(self.readings)

    def hottest(self):
        if not self.readings:
            return None

        return max(self.readings)
 #Q2 
 # Classes Question 2   
    def __repr__(self):
        return (
            f"Thermometer("
            f"location={self.location!r}, "
            f"n_readings={len(self.readings)}, "
            f"average={self.average()})"
        )

my_thermometer = Thermometer("Cary")

my_thermometer.add(20.0)
my_thermometer.add(22.5)
my_thermometer.add(25.0)
my_thermometer.add(21.5)

print("Average:", my_thermometer.average())
print("Hottest:", my_thermometer.hottest())
print(my_thermometer)
another_thermometer = Thermometer("Raleigh", [18.0, 20.0, 22.0, 19.0])

print([my_thermometer, another_thermometer])

# Without __repr__, Python displays a default object representation
# containing the class name and memory address. That is not very
# helpful for debugging because it does not show the object's useful data.


# average() needs to check whether the readings list is empty.
# Without this check, len(self.readings) would be 0 and Python would
# raise a ZeroDivisionError when calculating the average.
# --- Classes ---


# Q3 Classes Q3 — TemperatureAlert
class TemperatureAlert:
    def __init__(self, threshold=30.0):
        self.threshold = threshold

    def breaches(self, thermometer):
        hottest = thermometer.hottest()

        if hottest is None:
            return False

        return hottest > self.threshold


alert1 = TemperatureAlert(23.0)
alert2 = TemperatureAlert(30.0)

print("Alert 1 breaches:", alert1.breaches(my_thermometer))
print("Alert 2 breaches:", alert2.breaches(my_thermometer))

# The threshold is stored on the TemperatureAlert object so each alert
# can have its own threshold and be reused with different thermometers.

# --- Dataclasses ---
# Q1

from dataclasses import dataclass


@dataclass(frozen=True)
class Station:
    """Represent a weather station."""

    name: str
    region: str


station1 = Station("Cary", "North Carolina")
station2 = Station("Cary", "North Carolina")

print("Station 1:", station1)
print("Station 2:", station2)
print("Stations equal:", station1 == station2)

# Q2

from dataclasses import FrozenInstanceError


try:
    station1.name = "Raleigh"
except FrozenInstanceError as error:
    print("FrozenInstanceError:", error)


station3 = Station("Raleigh", "North Carolina")

stations = {station1, station2, station3}

print("Number of unique stations:", len(stations))

# A frozen dataclass prevents accidental changes after an object is created.
# It also allows Station objects to be used in sets because they are hashable.

# A dataclass automatically provides useful methods such as __init__
# and __eq__, so we do not have to write them ourselves.

# Q3

from dataclasses import field


@dataclass
class StationBatch:
    """Store a batch of weather stations for a region."""

    region: str
    stations: list[Station] = field(default_factory=list)
    #stations: list[Station] = []

    def add(self, station: Station) -> None:
        """Add a station to the batch."""
        self.stations.append(station)

    def highest(self) -> Station | None:
        """Return the station with the highest name, or None if empty."""
        if not self.stations:
            return None

        return max(self.stations, key=lambda station: station.name)


batch = StationBatch("North Carolina")

batch.add(station1)
batch.add(station3)

print("Station batch:", batch)
print("Highest station:", batch.highest())

empty_batch = StationBatch("Empty Region")
print("Highest empty batch:", empty_batch.highest())

# A mutable default such as stations: list[Station] = [] should not be used.
# default_factory=list creates a new empty list for each StationBatch object.

# --- Pydantic ---
# Q1


class Reading(BaseModel):
    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def check_extreme_conditions(self):
        """Reject impossible extreme temperature and humidity combination."""
        if self.humidity == 0 and self.temperature_c < -40:
            raise ValueError(
                "humidity cannot be 0 when temperature is below -40°C"
            )

        return self


reading = Reading(
    station_id="CAR",
    timestamp="2026-10-06T12:00:00",
    temperature_c=21.5,
    humidity=55.0,
)

print("Reading:", reading)

# Q2

#from pydantic import ValidationError


try:
    Reading(
        station_id="CA",
        timestamp="2026-10-06T12:00:00",
        temperature_c=21.5,
        humidity=55.0,
    )
except ValidationError as error:
    print("Validation error 1:", error)


try:
    Reading(
        station_id="CAR",
        timestamp="2026-10-06T12:00:00",
        temperature_c=70.0,
        humidity=55.0,
    )
except ValidationError as error:
    print("Validation error 2:", error)


try:
    Reading(
        station_id="CAR",
        timestamp="2026-10-06T12:00:00",
        temperature_c=21.5,
        humidity=120.0,
    )
except ValidationError as error:
    print("Validation error 3:", error)


coerced_reading = Reading(
    station_id="CAR",
    timestamp="2026-10-06T12:00:00",
    temperature_c="21.5",
    humidity=40,
)

print("Coerced temperature:", coerced_reading.temperature_c)
print("Coerced humidity:", coerced_reading.humidity)

# Pydantic can safely convert compatible input types, such as the string
# "21.5" to a float and the integer 40 to a float.

# Q3

try:
    Reading(
        station_id="CA",
        timestamp="2026-10-06T12:00:00",
        temperature_c=70.0,
        humidity=120.0,
    )
except ValidationError as error:
    print("Multiple validation errors:")

    for item in error.errors():
        print(item["loc"], "-", item["msg"])

# Pydantic reports all of the validation errors it finds in one model
# construction, which makes it easier to identify multiple bad inputs
# at the same time.

# Q4

valid_reading = Reading(
    station_id="CAR",
    timestamp="2026-10-06T12:00:00",
    temperature_c=-30.0,
    humidity=0,
)

print("Valid extreme reading:", valid_reading)


try:
    Reading(
        station_id="CAR",
        timestamp="2026-10-06T12:00:00",
        temperature_c=-50.0,
        humidity=0,
    )
except ValidationError as error:
    print("Invalid extreme reading:", error)

# Field constraints can validate individual fields, but they cannot
# express a rule that depends on the combination of two fields.
# model_validator is used for this kind of cross-field validation.

# --- pytest ---
# Question 1


def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a temperature from Celsius to Fahrenheit."""
    return (celsius * 9 / 5) + 32

def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)

    # pytest.approx is needed for 37°C because floating-point calculations
# can produce tiny rounding differences instead of exactly 98.6.

def mean(values: list[float]) -> float:
    """Return the arithmetic mean of a list of values."""
    if not values:
        raise ValueError("values cannot be empty")

    return sum(values) / len(values)

def test_mean_of_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        mean([])

        # pytest.raises(ValueError) alone checks that a ValueError happened,
# but match= also checks that the error message contains the expected word.
@pytest.mark.parametrize(
    "values, expected",
    [
        ([10], 10),
        ([10, 20], 15),
        ([-10, 10], 0),
        ([1, 2, 3, 4], 2.5),
    ],
)
def test_mean_values(values, expected):
    assert mean(values) == expected
    # Test summary: 6 passed in 0.36s
# Parametrize is useful because it lets one test function check multiple
# input/output cases without writing several nearly identical test functions.

# Deliberate failure: changing 9 / 5 to 9 / 4 caused the 100°C test
# to return 257.0 instead of the expected 212°F. Pytest showed the
# specific actual and expected values, which is more useful than only
# saying that an assertion failed.
