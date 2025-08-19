from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_DIR = os.path.join("MLB Odds Generation", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "mlb_home_win_model.joblib")
MODEL_META_PATH = os.path.join(MODEL_DIR, "mlb_home_win_model.meta.json")


FEATURE_COLUMNS = [
    "home_win_pct",
    "away_win_pct",
    "win_pct_diff",
    "home_run_diff_per_game",
    "away_run_diff_per_game",
    "run_diff_gap",
    # optional weather
    "temp_c",
    "precip_prob",
    "windspeed",
]
TARGET_COLUMN = "home_won"


@dataclass
class TrainResult:
    model_path: str
    num_rows: int
    metrics: Dict[str, float]


def _build_pipeline(feature_columns: List[str]) -> Pipeline:
    numeric_features = feature_columns
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(with_mean=True, with_std=True), numeric_features),
    ], remainder="drop")

    clf = LogisticRegression(max_iter=1000, solver="lbfgs")

    return Pipeline([
        ("pre", preprocessor),
        ("clf", clf),
    ])


def train_and_save(df: pd.DataFrame, feature_columns: List[str] = FEATURE_COLUMNS, target_column: str = TARGET_COLUMN) -> TrainResult:
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Fill missing optional features with neutral values
    df = df.copy()
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_columns + [target_column]].dropna()

    X = df[feature_columns]
    y = df[target_column].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pipeline = _build_pipeline(feature_columns)
    pipeline.fit(X_train, y_train)

    val_pred_proba = pipeline.predict_proba(X_val)[:, 1]
    metrics = {
        "brier": float(brier_score_loss(y_val, val_pred_proba)),
        "log_loss": float(log_loss(y_val, val_pred_proba, labels=[0, 1])),
        "num_val": int(len(y_val)),
    }

    joblib.dump(pipeline, MODEL_PATH)
    with open(MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump({"feature_columns": feature_columns, "metrics": metrics}, f, indent=2)

    return TrainResult(model_path=MODEL_PATH, num_rows=len(df), metrics=metrics)


def load_model() -> Pipeline:
    return joblib.load(MODEL_PATH)


def predict_proba(model: Pipeline, df: pd.DataFrame, feature_columns: List[str] = FEATURE_COLUMNS) -> np.ndarray:
    data = df.copy()
    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0.0
    return model.predict_proba(data[feature_columns])[:, 1] 