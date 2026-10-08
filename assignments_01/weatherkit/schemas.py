from pydantic import BaseModel, Field, model_validator


class HourlyBlock(BaseModel):
    """Columnar hourly weather measurements from the API response."""

    time: list[str]
    temperature_2m: list[float]
    precipitation: list[float]

    @model_validator(mode="after")
    def validate_lengths(self) -> "HourlyBlock":
        """Reject the block if the hourly lists have different lengths."""
        if not (
            len(self.time)
            == len(self.temperature_2m)
            == len(self.precipitation)
        ):
            raise ValueError(
                "time, temperature_2m, and precipitation must have the same length"
            )

        return self


class WeatherResponse(BaseModel):
    """Validated Open-Meteo weather response containing hourly observations."""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str
    elevation: float
    hourly: HourlyBlock