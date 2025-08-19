import os
from dataclasses import dataclass


DEFAULT_SPORTSDATA_API_KEY = "3d8b7ea85cf946f682ff8bed1279f0c1"


@dataclass(frozen=True)
class Settings:
    sportsdata_api_key: str
    sportsdata_base_url: str = "https://api.sportsdata.io/api/mlb"


def get_settings() -> Settings:
    api_key = os.getenv("SPORTSDATA_API_KEY", DEFAULT_SPORTSDATA_API_KEY).strip()
    return Settings(sportsdata_api_key=api_key) 