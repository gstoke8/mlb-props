# MLB Props Betting Model

A fully automated MLB player props betting model that identifies edges in Home Run, Strikeout, and Hits markets using Statcast data, synthetic Batter vs Pitcher matchups, and Closing Line Value tracking.

## Models

| Model | Algorithm | Key Features |
|-------|-----------|-------------|
| **Home Runs** | Logistic Regression (binary) | Barrel rate, xISO, synthetic BvP (pitch mix × HR/FB), park factors, weather |
| **Strikeouts** | Poisson GLM | CSW%, Stuff+, umpire K-factor, lineup whiff matchup, opener detection |
| **Hits** | Logistic Regression (binary) | Contact rate, BABIP, line drive %, pitcher BABIP allowed |

All models blend model output with market implied probability (70/30) and size bets using fractional Kelly (0.25×).

## Architecture

```
mlb-props/
├── Foundation
│   ├── db.py                    # SQLite schema (10 tables, 18 methods)
│   ├── mlb_api.py               # MLB Stats API client
│   ├── weather.py               # Visual Crossing weather client
│   ├── odds.py                  # The Odds API per-event client
│   ├── umpires.py               # UmpScorecards + Retrosheet K-factors
│   └── notifier.py              # Resend email notifications
├── Data
│   ├── statcast_nightly.py      # pybaseball Statcast batch fetcher
│   └── closing_lines.py         # Closing line capture + CLV computation
├── Execution
│   ├── sizing.py                # Fractional Kelly sizing, edge tiers
│   ├── risk.py                  # Pre-flight checks (lineup, injury, weather)
│   ├── scheduler.py             # Daily schedule builder
│   └── daily_runner.py          # Main orchestrator (--mode analysis|results|statcast)
├── HR Model
│   ├── hr_features.py           # Feature engineering
│   ├── pitch_type_matchup.py    # Synthetic BvP (pitch mix × barrel%)
│   ├── hr_model.py              # Binary classifier + market blend
│   ├── hr_calibrate.py          # Platt scaling calibration
│   └── hr_backtest.py           # Walk-forward backtest
├── K Model
│   ├── k_features.py            # Feature engineering
│   ├── lineup_whiff_matchup.py  # Lineup whiff% matchup
│   ├── k_model.py               # Poisson GLM + market blend
│   ├── k_calibrate.py           # Calibration
│   └── k_backtest.py            # Walk-forward backtest
├── Hits Model
│   ├── hits_features.py         # Feature engineering
│   ├── hits_model.py            # Binary classifier + market blend
│   └── hits_backtest.py         # Walk-forward backtest
└── Dashboard
    ├── dashboard.py             # Flask API + web server (port 5050)
    └── templates/dashboard.html # Dark-themed P/L dashboard
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

Add to `~/.zshrc` or `~/.bashrc`:

```bash
export ODDS_API_KEY="your_key"              # The Odds API ($30/mo)
export RESEND_API_KEY="your_key"            # Resend (free tier)
export VISUAL_CROSSING_API_KEY="your_key"  # Visual Crossing (free tier)
export RESEND_FROM="Props <your@email.com>"
export MLB_BANKROLL="1000"                  # Starting bankroll in USD
```

### 3. Configure launchd (macOS)

Copy the example plists and fill in your API keys:

```bash
cp com.mlb-props.daily.plist.example com.mlb-props.daily.plist
cp com.mlb-props.results.plist.example com.mlb-props.results.plist
cp com.mlb-props.statcast.plist.example com.mlb-props.statcast.plist
# Edit each file and replace YOUR_*_API_KEY placeholders

mkdir -p ~/mlb-props/logs
launchctl load ~/mlb-props/com.mlb-props.statcast.plist   # 6 AM ET
launchctl load ~/mlb-props/com.mlb-props.daily.plist      # 11 AM ET
launchctl load ~/mlb-props/com.mlb-props.results.plist    # 11:30 PM ET
```

### 4. Run manually

```bash
# Analysis run (fetch props, generate picks, send email)
python3 daily_runner.py --mode analysis --dry-run

# Results run (resolve bets, compute CLV)
python3 daily_runner.py --mode results

# Statcast nightly update
python3 daily_runner.py --mode statcast
```

### 5. Start dashboard

```bash
python3 dashboard.py
# Open http://localhost:5050
```

## Dashboard

The web dashboard at `http://localhost:5050` shows:
- **Today's picks** — player, prop, line, book, odds, edge, confidence, units, outcome
- **7-day history** — bets grouped by day with daily P/L subtotals
- **P/L chart** — dual-axis: daily P/L bars + cumulative line (last 30 days)
- **CLV summary** — average CLV, CLV+ %, breakdown by confidence tier
- Auto-refreshes every 60 seconds

## Go-Live Criteria

Do not bet real money until:
- CLV ≥ +1.5% over 300+ paper bets
- Hit rate ≥ 55% on MEDIUM+ confidence
- Calibration passes (ECE < 0.05)
- Monte Carlo: bankroll survives -10u drawdown in 95% of simulations

## Edge Tiers

| Tier | Edge | Units |
|------|------|-------|
| HIGH | > 6% | 1.5u |
| MEDIUM | 3–6% | 0.75u |
| LOW | 1.5–3% | 0.25u |
| Skip | < 1.5% | — |

Sizing uses fractional Kelly (0.25×), capped at 2% of bankroll per bet.

## Data Sources

| Source | Usage | Cost |
|--------|-------|------|
| MLB Stats API | Schedule, lineups, umpires, splits | Free |
| pybaseball / Baseball Savant | Statcast pitch-level data | Free |
| The Odds API | Player prop lines (DK, FD, MGM) | $30/mo |
| Visual Crossing | Game-time weather | Free tier |
| UmpScorecards | Historical umpire K-factors | Free |
| Retrosheet | Historical umpire assignments | Free |
