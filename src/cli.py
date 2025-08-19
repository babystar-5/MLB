from __future__ import annotations

import argparse
import datetime as dt
import os
from typing import List

import pandas as pd
from rich.console import Console
from rich.table import Table

from .features import build_training_frame, build_prediction_frame_for_date
from .model import FEATURE_COLUMNS, MODEL_DIR, TrainResult, load_model, predict_proba, train_and_save
from .odds import format_moneyline, prob_to_american_odds
from .sportsdata_client import SportsDataClient
from .weather import fetch_hourly_forecast, map_stadium_coordinates, select_hour_weather


console = Console()


def cmd_train(args: argparse.Namespace) -> None:
    seasons: List[int] = [int(s) for s in (args.seasons or [])] or [2021, 2022, 2023, 2024]
    console.print(f"Building training data for seasons: {seasons}")
    df = build_training_frame(seasons)
    result: TrainResult = train_and_save(df)
    console.print(f"Saved model to: {result.model_path}")
    console.print({"rows": result.num_rows, **result.metrics})


def cmd_predict_today(args: argparse.Namespace) -> None:
    date = dt.date.today()
    client = SportsDataClient()
    df = build_prediction_frame_for_date(date, client)

    # Weather enrichment per game
    stadiums = {s.get("StadiumID"): s for s in client.get_stadiums()}
    schedule = client.get_games_by_date(date)
    gameid_to_weather = {}
    for game in schedule:
        game_id = str(game.get("GameID") or game.get("GameId"))
        stadium_id = game.get("StadiumID") or game.get("StadiumId")
        coords = None
        if stadium_id in stadiums:
            coords = map_stadium_coordinates(stadiums[stadium_id])
        if coords is None:
            continue
        # Determine local start hour
        date_time_str = game.get("DateTime") or game.get("Day") or f"{date.isoformat()}T19:00:00"
        hour = 19
        try:
            hour = int(pd.to_datetime(date_time_str).hour)
        except Exception:
            pass
        forecast = fetch_hourly_forecast(coords[0], coords[1], date)
        w = select_hour_weather(forecast, hour)
        gameid_to_weather[game_id] = w

    # Attach weather to frame
    df = df.copy()
    df["temp_c"] = df["game_id"].map(lambda gid: (gameid_to_weather.get(gid) or {}).get("temperature_2m"))
    df["precip_prob"] = df["game_id"].map(lambda gid: (gameid_to_weather.get(gid) or {}).get("precipitation_probability"))
    df["windspeed"] = df["game_id"].map(lambda gid: (gameid_to_weather.get(gid) or {}).get("windspeed_10m"))

    model = load_model()
    probs = predict_proba(model, df, FEATURE_COLUMNS)

    table = Table(title=f"MLB Odds for {date.isoformat()}")
    table.add_column("Home", justify="left")
    table.add_column("Away", justify="left")
    table.add_column("Home Win %", justify="right")
    table.add_column("Home ML", justify="right")

    for _, row in df.iterrows():
        gid = row["game_id"]
        p = float(probs[_])
        odds = prob_to_american_odds(p)
        table.add_row(str(row["home_team"]), str(row["away_team"]), f"{p*100:.1f}%", format_moneyline(odds))

    console.print(table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MLB Odds Generation CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--seasons", nargs="*", type=int, help="Seasons to include e.g. 2021 2022 2023 2024")
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict-today", help="Predict odds for today's games")
    p_pred.set_defaults(func=cmd_predict_today)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main() 