from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional, Tuple

import requests


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_hourly_forecast(latitude: float, longitude: float, date: dt.date, timezone: str = "America/New_York") -> Dict[str, Any]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,precipitation_probability,windspeed_10m",
        "start_date": date.isoformat(),
        "end_date": date.isoformat(),
        "timezone": timezone,
    }
    response = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def select_hour_weather(forecast_json: Dict[str, Any], target_hour: int) -> Dict[str, Optional[float]]:
    hourly = forecast_json.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    precips = hourly.get("precipitation_probability", [])
    winds = hourly.get("windspeed_10m", [])

    selected = {"temperature_2m": None, "precipitation_probability": None, "windspeed_10m": None}

    for idx, time_str in enumerate(times):
        try:
            hour = dt.datetime.fromisoformat(time_str).hour
        except Exception:
            continue
        if hour == target_hour:
            selected["temperature_2m"] = float(temps[idx]) if idx < len(temps) else None
            selected["precipitation_probability"] = float(precips[idx]) if idx < len(precips) else None
            selected["windspeed_10m"] = float(winds[idx]) if idx < len(winds) else None
            break

    return selected


def map_stadium_coordinates(stadium: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Extract coordinates from a SportsDataIO stadium record."""
    lat = stadium.get("GeoLat") or stadium.get("Latitude")
    lon = stadium.get("GeoLong") or stadium.get("Longitude")
    if lat is None or lon is None:
        return None
    try:
        return float(lat), float(lon)
    except Exception:
        return None 