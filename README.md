# MLB Odds Generation

This project builds a simple MLB odds generation pipeline using SportsDataIO for baseball data and Open-Meteo for weather. It trains a logistic regression model to estimate home win probabilities and converts them into moneyline odds.

## Quickstart

1. Install dependencies
```bash
pip install -r "MLB Odds Generation/requirements.txt"
```

2. Provide API key (from discoverylab.sportsdata.io). Either set environment variable or edit `src/config.py` default.
```bash
$env:SPORTSDATA_API_KEY="YOUR_KEY"   # PowerShell
# or
export SPORTSDATA_API_KEY="YOUR_KEY"  # Bash
```

3. Train model (uses seasons 2021-2024 by default)
```bash
python -m "MLB Odds Generation.src.cli" train --seasons 2021 2022 2023 2024
```

4. Predict today odds
```bash
python -m "MLB Odds Generation.src.cli" predict-today
```

Artifacts will be saved under `MLB Odds Generation/models/` and intermediate data under `MLB Odds Generation/data/`.

## Notes
- This starter focuses on team-level signals (records, run differentials, rates) and augments predictions with forecasted weather for today.
- Extend `features.py` to incorporate player-level stats (e.g., starting pitchers) if your subscription permits those endpoints. 