from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from .sportsdata_client import SportsDataClient


@dataclass
class GameFeatureRow:
    game_id: str
    season: int
    date: dt.date
    home_team: str
    away_team: str
    # Engineered features
    home_win_pct: float
    away_win_pct: float
    win_pct_diff: float
    home_run_diff_per_game: float
    away_run_diff_per_game: float
    run_diff_gap: float
    # Optional weather (used at predict time)
    temp_c: Optional[float] = None
    precip_prob: Optional[float] = None
    windspeed: Optional[float] = None
    # Label
    home_won: Optional[int] = None


def _extract_team_aggregates(stats_record: Dict[str, Any]) -> Dict[str, float]:
    wins = float(stats_record.get("Wins", 0.0))
    losses = float(stats_record.get("Losses", 0.0))
    games = float(stats_record.get("Games", wins + losses)) or 1.0
    runs_scored = float(stats_record.get("RunsScored", stats_record.get("Runs", 0.0)))
    runs_allowed = float(stats_record.get("RunsAllowed", stats_record.get("RunsAgainst", 0.0)))

    win_pct = wins / max(wins + losses, 1.0)
    run_diff_per_game = (runs_scored - runs_allowed) / games

    return {
        "win_pct": win_pct,
        "run_diff_per_game": run_diff_per_game,
    }


def _index_team_stats_by_key(stats: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    index: Dict[str, Dict[str, float]] = {}
    for rec in stats:
        key = rec.get("Team") or rec.get("Key") or rec.get("Name")
        if not key:
            continue
        index[str(key)] = _extract_team_aggregates(rec)
    return index


def build_training_frame(seasons: List[int], client: Optional[SportsDataClient] = None) -> pd.DataFrame:
    client = client or SportsDataClient()
    rows: List[GameFeatureRow] = []

    for season in seasons:
        try:
            team_stats = client.get_team_season_stats(season)
        except requests.HTTPError:
            team_stats = client.get_standings(season)
        team_index = _index_team_stats_by_key(team_stats)
        schedule = client.get_schedule(season)

        for game in schedule:
            home = str(game.get("HomeTeam"))
            away = str(game.get("AwayTeam"))
            if not home or not away:
                continue
            game_id = str(game.get("GameID") or game.get("GameId") or f"{season}-{home}-{away}-{game.get('Day')}")

            home_aggs = team_index.get(home, {"win_pct": 0.5, "run_diff_per_game": 0.0})
            away_aggs = team_index.get(away, {"win_pct": 0.5, "run_diff_per_game": 0.0})

            label = None
            if "HomeTeamRuns" in game and "AwayTeamRuns" in game:
                try:
                    label = 1 if int(game["HomeTeamRuns"]) > int(game["AwayTeamRuns"]) else 0
                except Exception:
                    label = None

            date_str = (game.get("Day") or game.get("DateTime") or game.get("Date"))
            game_date = dt.date.today()
            if date_str:
                try:
                    game_date = dt.datetime.fromisoformat(str(date_str).replace("Z", "+00:00")).date()
                except Exception:
                    pass

            row = GameFeatureRow(
                game_id=game_id,
                season=int(season),
                date=game_date,
                home_team=home,
                away_team=away,
                home_win_pct=float(home_aggs["win_pct"]),
                away_win_pct=float(away_aggs["win_pct"]),
                win_pct_diff=float(home_aggs["win_pct"]) - float(away_aggs["win_pct"]),
                home_run_diff_per_game=float(home_aggs["run_diff_per_game"]),
                away_run_diff_per_game=float(away_aggs["run_diff_per_game"]),
                run_diff_gap=float(home_aggs["run_diff_per_game"]) - float(away_aggs["run_diff_per_game"]),
                home_won=label,
            )
            rows.append(row)

    df = pd.DataFrame([r.__dict__ for r in rows])
    # Keep only rows with labels for training
    df = df[df["home_won"].notna()].reset_index(drop=True)
    return df


def build_prediction_frame_for_date(target_date: dt.date, client: Optional[SportsDataClient] = None) -> pd.DataFrame:
    client = client or SportsDataClient()
    # Use current season context
    season = target_date.year
    try:
        team_stats = client.get_team_season_stats(season)
    except requests.HTTPError:
        team_stats = client.get_standings(season)
    team_index = _index_team_stats_by_key(team_stats)
    games = client.get_games_by_date(target_date)

    rows: List[GameFeatureRow] = []
    for game in games:
        home = str(game.get("HomeTeam"))
        away = str(game.get("AwayTeam"))
        if not home or not away:
            continue
        game_id = str(game.get("GameID") or game.get("GameId") or f"{season}-{home}-{away}-{target_date.isoformat()}")

        home_aggs = team_index.get(home, {"win_pct": 0.5, "run_diff_per_game": 0.0})
        away_aggs = team_index.get(away, {"win_pct": 0.5, "run_diff_per_game": 0.0})

        date_time_str = game.get("DateTime") or game.get("Day") or f"{target_date.isoformat()}T19:00:00"
        hour = 19
        try:
            hour = int(pd.to_datetime(date_time_str).hour)
        except Exception:
            pass

        row = GameFeatureRow(
            game_id=game_id,
            season=int(season),
            date=target_date,
            home_team=home,
            away_team=away,
            home_win_pct=float(home_aggs["win_pct"]),
            away_win_pct=float(away_aggs["win_pct"]),
            win_pct_diff=float(home_aggs["win_pct"]) - float(away_aggs["win_pct"]),
            home_run_diff_per_game=float(home_aggs["run_diff_per_game"]),
            away_run_diff_per_game=float(away_aggs["run_diff_per_game"]),
            run_diff_gap=float(home_aggs["run_diff_per_game"]) - float(away_aggs["run_diff_per_game"]),
        )
        # Weather will be filled by CLI using `weather.py` to avoid per-call dependencies here.
        rows.append(row)

    return pd.DataFrame([r.__dict__ for r in rows]) 