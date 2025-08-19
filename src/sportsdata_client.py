from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

import requests

from .config import get_settings


class SportsDataClient:
    """Minimal SportsDataIO MLB client.

    This client supports the subset of endpoints needed for this project and
    handles both header-based and query-param API key auth styles.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.sportsdata_api_key
        self.base_url = (base_url or settings.sportsdata_base_url).rstrip("/")
        self.session = requests.Session()

    # --- Internal helpers -------------------------------------------------
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        final_params = dict(params or {})
        # Many SportsDataIO endpoints accept the key via query string as `key`.
        final_params.setdefault("key", self.api_key)
        headers = {
            # Some products use this header; harmless if not required.
            "Ocp-Apim-Subscription-Key": self.api_key,
        }
        response = self.session.get(url, params=final_params, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()

    # --- Public methods ---------------------------------------------------
    def get_stadiums(self) -> List[Dict[str, Any]]:
        return self._get("scores/json/Stadiums")

    def get_team_season_stats(self, season: int) -> List[Dict[str, Any]]:
        return self._get(f"stats/json/TeamSeasonStats/{season}")

    def get_games_by_date(self, date: dt.date) -> List[Dict[str, Any]]:
        return self._get(f"scores/json/GamesByDate/{date.isoformat()}")

    def get_schedule(self, season: int) -> List[Dict[str, Any]]:
        return self._get(f"scores/json/Games/{season}")

    def get_standings(self, season: int) -> List[Dict[str, Any]]:
        return self._get(f"scores/json/Standings/{season}") 