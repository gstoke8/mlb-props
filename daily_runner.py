from __future__ import annotations
"""daily_runner.py — Main entry point for the MLB props betting pipeline.

Runs in two modes triggered by launchd:
  analysis  (~90 min before first game): fetch data, run models, send picks email
  results   (~30 min after last game):   resolve bets, compute CLV, send results email
  statcast  (nightly): refresh Statcast rolling metrics for yesterday
"""

import argparse
import logging
import math
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from scipy.stats import binom, poisson

from db import get_db
from mlb_api import get_client
from weather import get_weather_client
from odds import get_odds_client
import umpires
import notifier
import statcast_nightly
import closing_lines
import sizing
import risk
import matchup_context
import k_model as k_model_module
import hits_model as hits_model_module
import hr_model as hr_model_module

# ---------------------------------------------------------------------------
# Config — all values from environment
# ---------------------------------------------------------------------------

DRY_RUN = os.getenv("DRY_RUN", "0") == "1"
# is_live controls whether bets are marked for real-money tracking.
# Set to 1 to track results as live; set to 0 for pure paper record-keeping.
IS_LIVE = int(os.getenv("MLB_IS_LIVE", "1"))
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
VISUAL_CROSSING_KEY = os.getenv("VISUAL_CROSSING_API_KEY", "")
BANKROLL = float(os.getenv("MLB_BANKROLL", "1000"))
BOOKS = [
    "draftkings", "fanduel", "betmgm", "espnbet",
]
# DraftKings is the preferred book for tiebreaking when edges are equal.
_PREFERRED_BOOK = "draftkings"
MARKETS = [
    "batter_hits",
    "pitcher_strikeouts",
    "batter_home_runs",
]
# Excluded markets (lineup/teammate dependent — no individual player signal):
# - batter_rbis: depends on who's on base, not the batter's skill alone
# - batter_runs_scored: depends on teammates hitting behind you
# - batter_hits_runs_rbis: all three combined — compounded noise
# All three use flat league-average Poisson with no player adjustment.
# - singles, doubles: λ so low that Under bets always look positive vs vig
#   but extreme negative juice (-1000+) makes payouts not worth the risk.

# Odds floor: don't bet when the American odds are worse than this threshold.
# Prevents juiced lock-style Under bets (e.g. -1250) where absolute return is
# negligible and model-uncertainty risk is high.
MAX_NEGATIVE_ODDS = -300

# Odds ceiling: don't bet extreme longshots beyond this threshold.
# Our league-average models (esp. HR Poisson) assign false edge to
# non-power-hitters priced at +3000–+6000; capping at +600 removes them.
MAX_POSITIVE_ODDS = 600

# MEDIUM confidence enabled. Tier thresholds will be recalibrated once 300+ bets
# accumulate under the consistent de-vigged edge formula (post-May-2026 data).
# To temporarily disable: set env var MLB_ALLOW_MEDIUM=0.
ALLOW_MEDIUM = os.getenv("MLB_ALLOW_MEDIUM", "1") == "1"

# K model line cap: Under bets on strikeout lines >= 5.5 have 36-46% win rates
# despite 20-30%+ claimed edge — model systematically underestimates elite pitchers.
# Cap = max line (inclusive) allowed for strikeout Under bets.
# Default 5.0 blocks lines 5.5+. Set MLB_K_LINE_CAP=99 to disable entirely.
K_LINE_CAP = float(os.getenv("MLB_K_LINE_CAP", "5.0"))

# Hits Under line cap: Hits Under 1.5 HIGH bets lost -16.18u on 68 bets (57.4% win at avg -201
# odds) due to class_weight="balanced" miscalibration in hits-v6; model over-assigns ~88% prob
# to Under on contact hitters. Under 0.5 is profitable (+3.57u). Block Until retrained.
# Default 0.5 allows only Hits Under 0.5. Set MLB_HITS_UNDER_MAX_LINE=1.5 to re-enable 1.5.
HITS_UNDER_MAX_LINE = float(os.getenv("MLB_HITS_UNDER_MAX_LINE", "0.5"))

# Phase 1 stub model version identifier
_MODEL_VERSION = "baseline-v1"

# Prop-availability retry: if the Odds API has no lines for a target game
# (books haven't posted yet), retry every 10 minutes until game starts.
_PROP_RETRY_INTERVAL_SEC = 600   # 10 minutes between retries

# Slot scheduling constants — must match scheduler.py
_SLOT_LEAD_MINUTES = 60      # how far ahead of first pitch the slot fires
_SLOT_GROUPING_MINUTES = 30  # games within 30 min of the slot anchor go together

# ---------------------------------------------------------------------------
# Statistical baseline model parameters (2023-24 MLB averages)
# ---------------------------------------------------------------------------

_MLB_BA = 0.255          # league batting average
_MLB_PA_PER_GAME = 4     # approx plate appearances per game per batter (int for binom n)

# Expected plate appearances per game by batting order position (2023-24 MLB averages).
# Used to give lineup-adjusted n in the Binomial hits conversion.
_PA_BY_LINEUP_SPOT: dict[int, float] = {
    1: 4.65, 2: 4.55, 3: 4.35, 4: 4.20, 5: 4.05,
    6: 3.90, 7: 3.70, 8: 3.55, 9: 3.35,
}


def _expected_pa(lineup_spot: float) -> float:
    """Return expected PA for a given lineup spot (1–9), defaulting to 4.0."""
    return _PA_BY_LINEUP_SPOT.get(int(round(lineup_spot)), 4.0)


def _binom_over(line: float, n_pa: float, p: float) -> float:
    """P(hits > line) via Binomial, interpolating fractional n_pa across floor/ceil."""
    lo = int(n_pa)
    hi = lo + 1
    frac = n_pa - lo
    p_lo = float(1.0 - binom.cdf(int(line), n=lo, p=p))
    p_hi = float(1.0 - binom.cdf(int(line), n=hi, p=p))
    return p_lo + frac * (p_hi - p_lo)
_MLB_K_LAMBDA = 5.75     # expected Ks per start: K/9=9.0 × 5.75 IP avg
_MLB_AVG_IP = 5.75       # average SP innings per start
_MLB_HR_LAMBDA = 0.12    # expected HR per game: ~3.2% HR/AB × 3.8 AB/game


_MLB_RBI_LAMBDA = 0.50          # team R/G (≈4.5) ÷ 9 batters
_MLB_R_LAMBDA = 0.50            # similar to RBI rate
_MLB_1B_LAMBDA = 0.19           # singles per game: roughly (BA - XBH_rate) × PA
_MLB_2B_LAMBDA = 0.048          # doubles per game: ~5% of PA
_MLB_HRR_LAMBDA = 2.02          # H+R+RBI combined: 1.02 + 0.50 + 0.50


def _hits_model_prob_over(line: float, lineup_spot: float = 5.0) -> float:
    """P(hits > line) via Binomial with lineup-adjusted n, p=0.255."""
    return _binom_over(line, _expected_pa(lineup_spot), _MLB_BA)


def _k_model_prob_over(line: float) -> float:
    """P(Ks > line) via Poisson(lambda=5.75)."""
    return float(1.0 - poisson.cdf(int(line), mu=_MLB_K_LAMBDA))


def _hr_model_prob_over(line: float) -> float:
    """P(HR > line) via Poisson(lambda=0.12)."""
    return float(1.0 - poisson.cdf(int(line), mu=_MLB_HR_LAMBDA))


def _rbis_model_prob_over(line: float) -> float:
    """P(RBI > line) via Poisson(lambda=0.50)."""
    return float(1.0 - poisson.cdf(int(line), mu=_MLB_RBI_LAMBDA))


def _runs_model_prob_over(line: float) -> float:
    """P(R > line) via Poisson(lambda=0.50)."""
    return float(1.0 - poisson.cdf(int(line), mu=_MLB_R_LAMBDA))


def _singles_model_prob_over(line: float) -> float:
    """P(1B > line) via Poisson(lambda=0.19)."""
    return float(1.0 - poisson.cdf(int(line), mu=_MLB_1B_LAMBDA))


def _doubles_model_prob_over(line: float) -> float:
    """P(2B > line) via Poisson(lambda=0.048)."""
    return float(1.0 - poisson.cdf(int(line), mu=_MLB_2B_LAMBDA))


def _hrr_model_prob_over(line: float) -> float:
    """P(H+R+RBI > line) via Poisson(lambda=2.02)."""
    return float(1.0 - poisson.cdf(int(line), mu=_MLB_HRR_LAMBDA))

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prop_type_from_market(market_key: str) -> str:
    """Map an Odds API market key to an internal prop_type string."""
    mapping = {
        "batter_hits": "hits",
        "pitcher_strikeouts": "strikeouts",
        "batter_home_runs": "home_runs",
        "batter_rbis": "rbis",
        "batter_runs_scored": "runs_scored",
        "batter_singles": "singles",
        "batter_doubles": "doubles",
        "batter_hits_runs_rbis": "hits_runs_rbis",
    }
    return mapping.get(market_key, market_key)


def _resolve_outcome(pick: str, line: float, actual: float) -> str:
    """Determine WIN/LOSS/PUSH from pick direction and actual stat."""
    if pick.lower() == "over":
        if actual > line:
            return "WIN"
        if actual < line:
            return "LOSS"
        return "PUSH"
    if pick.lower() == "under":
        if actual < line:
            return "WIN"
        if actual > line:
            return "LOSS"
        return "PUSH"
    return "NO_ACTION"


def _compute_pl_units(outcome: str, odds: int, units: float) -> float:
    """Compute profit/loss in units given outcome, American odds, and stake."""
    if outcome == "WIN":
        if odds > 0:
            return units * (odds / 100.0)
        return units * (100.0 / abs(odds))
    if outcome == "LOSS":
        return -units
    return 0.0


def _group_stat_key(prop_type: str) -> str:
    """Return the MLB Stats API stat group for a prop type."""
    if prop_type in ("strikeouts", "pitcher_strikeouts"):
        return "pitching"
    return "hitting"


def _extract_stat_value(game_log_entry: dict, prop_type: str) -> float | None:
    """Pull the relevant stat from a game log entry dict."""
    stat_map = {
        "hits": "hits",
        "strikeouts": "strikeOuts",
        "home_runs": "homeRuns",
        "rbis": "rbi",
        "runs_scored": "runs",
        "singles": None,   # computed: hits - (doubles + triples + homeRuns)
        "doubles": "doubles",
        "hits_runs_rbis": None,  # computed: hits + runs + rbi
    }
    # Computed stats
    if prop_type == "singles":
        try:
            h = float(game_log_entry.get("hits") or 0)
            d = float(game_log_entry.get("doubles") or 0)
            t = float(game_log_entry.get("triples") or 0)
            hr = float(game_log_entry.get("homeRuns") or 0)
            return max(0.0, h - d - t - hr)
        except (TypeError, ValueError):
            return None
    if prop_type == "hits_runs_rbis":
        try:
            h = float(game_log_entry.get("hits") or 0)
            r = float(game_log_entry.get("runs") or 0)
            rbi = float(game_log_entry.get("rbi") or 0)
            return h + r + rbi
        except (TypeError, ValueError):
            return None

    key = stat_map.get(prop_type)
    if key is None:
        return None
    val = game_log_entry.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _no_vig_edge(over_implied: float, under_implied: float, bet_implied: float) -> float:
    """Compute the no-vig edge for a bet.

    Removes the bookmaker's vig from a 2-way market to derive the consensus
    true probability, then returns edge = true_prob - bet_implied_prob.

    A positive edge means the bet is priced better than the true probability.

    Parameters
    ----------
    over_implied:   Market implied probability for the Over (includes vig).
    under_implied:  Market implied probability for the Under (includes vig).
    bet_implied:    The specific book's implied probability for the chosen side.

    Returns
    -------
    float
        Edge as a fraction (e.g. 0.025 = 2.5%).
    """
    vig_sum = over_implied + under_implied
    if vig_sum <= 0:
        return 0.0
    true_prob = over_implied / vig_sum   # no-vig true probability for the Over
    return true_prob - bet_implied


# ---------------------------------------------------------------------------
# run_analysis
# ---------------------------------------------------------------------------

# Game statuses that are safe to bet — game hasn't started yet
_BETTABLE_STATUSES = {
    "scheduled", "pre-game", "pregame", "warmup",
    "delayed start", "delayed", "",
}


def _filter_games_by_pks(schedule: list[dict], game_pks_str: str) -> list[dict]:
    """Return only games whose game_pk is in the comma-separated *game_pks_str*."""
    try:
        allowed = {int(pk.strip()) for pk in game_pks_str.split(",") if pk.strip()}
    except ValueError:
        log.warning("Invalid --game-pks %r — processing all games.", game_pks_str)
        return schedule

    filtered = [g for g in schedule if g.get("game_pk") in allowed]
    log.info("game-pks filter: %s → %d/%d games", game_pks_str, len(filtered), len(schedule))
    return filtered


def _group_games_into_slots(games: list[dict]) -> list[list[dict]]:
    """Cluster games into start-time slots (mirrors scheduler.group_games_by_slot)."""
    sorted_games = sorted(games, key=lambda g: g["game_time_utc"])
    slots: list[list[dict]] = []
    current: list[dict] = []
    anchor: datetime | None = None
    threshold = _SLOT_GROUPING_MINUTES * 60

    for game in sorted_games:
        t = game["game_time_utc"]
        if anchor is None or (t - anchor).total_seconds() > threshold:
            if current:
                slots.append(current)
            current = [game]
            anchor = t
        else:
            current.append(game)

    if current:
        slots.append(current)

    return slots


def _infer_slot_game_pks(schedule: list[dict]) -> str | None:
    """Infer the current slot's game PKs from the current time.

    Groups today's games into start-time slots and returns the PKs for the
    slot whose scheduled run time (slot_utc − SLOT_LEAD_MINUTES) is within
    ±SLOT_GROUPING_MINUTES of now.  Returns None when no slot matches (e.g.
    if called outside any slot window) — callers should treat None as
    "nothing to process right now".
    """
    now_utc = datetime.now(timezone.utc)
    slots = _group_games_into_slots(schedule)
    tolerance_secs = _SLOT_GROUPING_MINUTES * 60

    best_slot: list[dict] | None = None
    best_delta: float | None = None

    for slot in slots:
        slot_utc = slot[0]["game_time_utc"]
        run_time = slot_utc - timedelta(minutes=_SLOT_LEAD_MINUTES)
        delta = abs((now_utc - run_time).total_seconds())
        if delta <= tolerance_secs:
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_slot = slot

    if best_slot is None:
        return None

    pks = [str(g["game_pk"]) for g in best_slot if g.get("game_pk")]
    return ",".join(pks) if pks else None


def _filter_pregame_only(schedule: list[dict]) -> list[dict]:
    """Drop any games that are already in progress, final, or postponed."""
    pregame = []
    skipped = []
    for game in schedule:
        status = str(game.get("status") or "").lower()
        if status in _BETTABLE_STATUSES:
            pregame.append(game)
        else:
            skipped.append((game.get("game_pk"), status))

    if skipped:
        log.info(
            "Skipping %d game(s) already in progress or finished: %s",
            len(skipped),
            [(pk, s) for pk, s in skipped],
        )
    return pregame


def _build_confirmed_lineup_sets(
    mlb_client,
    game_pks: list[int],
) -> tuple[dict[int, set[str]], dict[int, dict]]:
    """Fetch confirmed batting lineups for each game.

    Returns:
        names_by_game   — game_pk -> set of lowercased player names
                          (used for prop candidate filtering)
        lineups_by_game — game_pk -> {
                            "home": [{player_id, name, batting_order, position}],
                            "away": [...],
                            "name_to_id": {name_lower: player_id},
                          }

    If a game's lineup hasn't been posted yet, that game_pk is absent from
    both dicts (no filtering applied for that game).
    """
    names_by_game: dict[int, set[str]] = {}
    lineups_by_game: dict[int, dict] = {}

    for game_pk in game_pks:
        try:
            lineup = mlb_client.get_confirmed_lineup(game_pk)
        except Exception:
            log.warning("Could not fetch lineup for game_pk=%d — skipping lineup filter.", game_pk)
            continue

        names: set[str] = set()
        name_to_id: dict[str, int] = {}
        for side in ("home", "away"):
            for player in lineup.get(side, []):
                name = player.get("name")
                pid = player.get("player_id")
                if name:
                    names.add(name.lower())
                    if pid:
                        name_to_id[name.lower()] = pid

        if names:
            names_by_game[game_pk] = names
            lineups_by_game[game_pk] = {
                "home": lineup.get("home", []),
                "away": lineup.get("away", []),
                "name_to_id": name_to_id,
            }
            log.info("Confirmed lineup for game_pk=%d: %d players", game_pk, len(names))
        else:
            log.info(
                "No confirmed lineup yet for game_pk=%d — all prop candidates accepted.", game_pk
            )

    return names_by_game, lineups_by_game


def _schedule_lines_retry(game_date: date, game_pks: list[int]) -> None:
    """Create and load a one-shot launchd plist that retries prop-line fetching.

    Fires ``daily_runner.py --mode analysis --date <date> --game-pks <pks>``
    in _PROP_RETRY_INTERVAL_SEC seconds.  If lines are still missing at that
    time, this function is called again — each retry is a fresh plist.
    """
    gk_str = ",".join(str(gk) for gk in sorted(game_pks))
    # Use a timestamp in the label so repeated retries don't collide.
    ts = datetime.now().strftime("%H%M%S")
    label = f"com.mlb-props.lines-retry-{gk_str.replace(',', '-')}-{ts}"

    fire_dt = datetime.now(timezone.utc) + timedelta(seconds=_PROP_RETRY_INTERVAL_SEC)

    env_keys = [
        "ODDS_API_KEY", "RESEND_API_KEY", "VISUAL_CROSSING_API_KEY",
        "MLB_BANKROLL", "MLB_IS_LIVE", "DATABASE_URL",
    ]
    env_xml = "\n".join(
        f"        <key>{k}</key>\n        <string>{os.environ.get(k, '')}</string>"
        for k in env_keys
        if os.environ.get(k)
    )

    log_dir = Path.home() / "mlb-props" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = Path.home() / "mlb-props" / "daily_runner.py"

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>

    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>{runner}</string>
        <string>--mode</string>
        <string>analysis</string>
        <string>--date</string>
        <string>{game_date.isoformat()}</string>
        <string>--game-pks</string>
        <string>{gk_str}</string>
    </array>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{fire_dt.hour}</integer>
        <key>Minute</key>
        <integer>{fire_dt.minute}</integer>
    </dict>

    <key>EnvironmentVariables</key>
    <dict>
{env_xml}
    </dict>

    <key>WorkingDirectory</key>
    <string>{Path.home() / 'mlb-props'}</string>

    <key>StandardOutPath</key>
    <string>{log_dir / f'lines-retry-{gk_str}.log'}</string>

    <key>StandardErrorPath</key>
    <string>{log_dir / f'lines-retry-{gk_str}_err.log'}</string>

    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>"""

    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    plist_path.write_text(plist_content)

    uid = os.getuid()
    subprocess.run(
        ["launchctl", "bootstrap", f"gui/{uid}", str(plist_path)],
        check=False, capture_output=True,
    )
    log.info(
        "Lines-retry plist scheduled: fires at %02d:%02d UTC for game_pk(s) %s",
        fire_dt.hour, fire_dt.minute, game_pks,
    )


def run_analysis(
    game_date: date,
    db,
    mlb_client,
    weather_client,
    odds_client,
    game_pks: str | None = None,
) -> list[dict]:
    """Fetch schedule + props, run preflight, return qualifying picks list.

    Phase 1 (stub): edge is always 0.0; should_bet() will return False for all
    props until a real model is plugged in.  The full pipeline still executes
    end-to-end so downstream components can be validated.

    Returns
    -------
    list[dict]
        Each entry is a qualifying prop dict (may be empty when no games today
        or all props fail preflight).
    """
    date_str = game_date.isoformat()
    log.info("=== Analysis run for %s (dry_run=%s) ===", date_str, DRY_RUN)
    # Per-run cache for name→player_id lookups (avoids repeat API calls for the same player)
    _player_id_cache: dict[str, int] = {}

    # 1. Schedule
    schedule = mlb_client.get_schedule(date_str)
    if not schedule:
        log.info("No games found for %s — skipping analysis.", date_str)
        return []

    # Determine which slot to process.
    # If --game-pks was supplied (normal scheduler path), use it directly.
    # Otherwise, infer the current slot from the wall-clock time so analysis
    # never accidentally fires for the entire day's schedule.
    if not game_pks:
        game_pks = _infer_slot_game_pks(schedule)
        if game_pks:
            log.info("No --game-pks supplied; inferred slot from current time: %s", game_pks)
        else:
            log.warning(
                "No --game-pks supplied and current time does not fall within any "
                "slot window (±%d min of a T-%d run time). "
                "Pass --game-pks explicitly to override.",
                _SLOT_GROUPING_MINUTES,
                _SLOT_LEAD_MINUTES,
            )
            return []

    # Filter schedule to the exact game_pks for this slot
    schedule = _filter_games_by_pks(schedule, game_pks)
    if not schedule:
        log.info("No matching games for --game-pks %s — skipping analysis.", game_pks)
        return []

    # Drop any games already in progress or finished
    schedule = _filter_pregame_only(schedule)
    if not schedule:
        log.info("All games for %s are already in progress or finished — skipping analysis.", date_str)
        return []

    game_pk_list = [g["game_pk"] for g in schedule if g.get("game_pk")]
    log.info("Found %d pre-game games: %s", len(schedule), game_pk_list)

    # Build game-info lookup: game_pk → {away_team, home_team, game_time_et}
    _ET_OFFSET = timedelta(hours=4)  # EDT (Mar–Nov); close enough for display
    game_info: dict[int, dict] = {}
    for g in schedule:
        gk = g.get("game_pk")
        if not gk:
            continue
        gt_utc = g.get("game_time_utc", "")
        game_time_et = ""
        if gt_utc:
            try:
                dt_utc = datetime.strptime(gt_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                dt_et = dt_utc - _ET_OFFSET
                game_time_et = dt_et.strftime("%-I:%M %p ET")
            except ValueError:
                pass
        game_info[gk] = {
            "away_team": g.get("away_team", ""),
            "home_team": g.get("home_team", ""),
            "game_time_et": game_time_et,
            "venue_name": g.get("venue_name", ""),
        }

    # 1b. Confirmed lineups — filter props to starting players only.
    # At T-30, lineups are usually posted. If not yet posted for a game,
    # we process all prop candidates for that game (no filtering).
    confirmed_lineups, lineups_full = _build_confirmed_lineup_sets(mlb_client, game_pk_list)
    current_season = game_date.year

    # 2. Umpires
    try:
        umpires.refresh_umpire_factors(db)
        todays_umpires = umpires.get_todays_umpires(mlb_client, game_pk_list)
        log.info("Umpire data loaded for %d games.", len(todays_umpires))
    except Exception:
        log.exception("Umpire refresh failed — continuing with neutral factors.")
        todays_umpires = {}

    # 3. Probable pitchers (keyed by game_pk)
    try:
        probable_pitchers = mlb_client.get_probable_pitchers(date_str)
    except Exception:
        log.exception("Failed to fetch probable pitchers.")
        probable_pitchers = {}

    # 4. Odds events + props
    # Games starting after midnight UTC (e.g. 9:40 PM ET = 01:40 UTC) are listed
    # under the next calendar date in the Odds API. Fetch both today and tomorrow
    # so late-night slots don't miss their events.
    try:
        tomorrow_str = (game_date + timedelta(days=1)).isoformat()
        events_today = odds_client.get_mlb_events(date_str)
        events_tomorrow = odds_client.get_mlb_events(tomorrow_str)
        # Deduplicate by event_id
        seen: set[str] = set()
        events: list[dict] = []
        for e in events_today + events_tomorrow:
            if e["event_id"] not in seen:
                seen.add(e["event_id"])
                events.append(e)
        log.info("Fetched %d Odds API events (%d today + %d tomorrow).",
                 len(events), len(events_today), len(events_tomorrow))
        all_props = odds_client.get_all_games_props(events, markets=MARKETS)

        # Fetch game totals (over/under) for each event — used for team_implied_runs
        game_totals_by_event: dict[str, float | None] = {}
        for _event in events:
            _eid = _event["event_id"]
            game_totals_by_event[_eid] = odds_client.get_game_total(_eid)
    except Exception:
        log.exception("Failed to fetch odds/props — aborting analysis.")
        raise

    # Map event_id -> game_pk using schedule
    event_to_game_pk: dict[str, int | None] = {}
    for event in events:
        gk = odds_client.match_event_to_game_pk(event, schedule)
        event_to_game_pk[event["event_id"]] = gk

    # 5. Per-game weather (cache by game_pk)
    venue_weather: dict[int, dict] = {}
    for game in schedule:
        gk = game.get("game_pk")
        if gk is None:
            continue
        try:
            venue_id = game.get("venue_id")
            venue_info = mlb_client.get_venue_info(venue_id) if venue_id else {}
            lat = venue_info.get("lat") or 0.0
            lon = venue_info.get("lon") or 0.0
            game_time_utc = game.get("game_time_utc", "")
            venue_name = game.get("venue_name", "")
            wx = weather_client.get_game_weather(
                lat=float(lat),
                lon=float(lon),
                game_time_utc=str(game_time_utc),
                venue_name=str(venue_name),
            )
            venue_weather[gk] = wx
        except Exception:
            log.exception("Weather fetch failed for game_pk=%s; using neutral.", gk)
            venue_weather[gk] = {}

    # 6. Collect all prop lines across every book for each (player, prop_type).
    #    Structure: {(player_name, prop_type): {line: {book: {side: outcome_dict}}}}
    #    Keeping context (game_pk, pitchers, weather) alongside each line.
    today_str = date_str

    PlayerLineKey = tuple  # (player_name, prop_type)

    def _build_catalog(props_by_event: dict) -> dict:
        """Build prop_catalog from a fresh Odds API props payload."""
        catalog: dict[PlayerLineKey, dict] = {}
        for ev_id, props_data in props_by_event.items():
            gk = event_to_game_pk.get(ev_id)
            if game_pks and gk is None:
                log.debug("Skipping event %s — not in target game-pks list.", ev_id)
                continue
            wx = venue_weather.get(gk, {}) if gk else {}
            pitchers_ctx = probable_pitchers.get(gk, {}) if gk else {}

            for book_key, book_markets in props_data.get("props", {}).items():
                if book_key not in BOOKS:
                    continue

                for market_key, all_outcomes in book_markets.items():
                    prop_type = _prop_type_from_market(market_key)

                    for o in all_outcomes:
                        player_name = o.get("player_name", "")
                        line = o.get("line")
                        side = o.get("pick", "")   # "Over" or "Under"
                        if not player_name or line is None or not side:
                            continue

                        # Skip batters not in the confirmed starting lineup.
                        # Pitcher props (strikeouts) are exempt — pitchers are never
                        # listed in the batting order but their props are valid.
                        is_pitcher_market = market_key == "pitcher_strikeouts"
                        if not is_pitcher_market and gk in confirmed_lineups:
                            if player_name.lower() not in confirmed_lineups[gk]:
                                log.debug(
                                    "Skip %s — not in confirmed lineup for game_pk=%s",
                                    player_name, gk,
                                )
                                continue

                        pk = (player_name, prop_type)
                        if pk not in catalog:
                            catalog[pk] = {
                                "game_pk": gk,
                                "event_id": ev_id,
                                "pitchers": pitchers_ctx,
                                "weather": wx,
                                "game_total": game_totals_by_event.get(ev_id),
                                "lines": {},
                            }
                        cat_lines = catalog[pk]["lines"]
                        if line not in cat_lines:
                            cat_lines[line] = {}
                        if book_key not in cat_lines[line]:
                            cat_lines[line][book_key] = {}
                        cat_lines[line][book_key][side] = o
        return catalog

    prop_catalog = _build_catalog(all_props)

    # Per-game-pk no-lines checker: identify which target games have no Odds API
    # props at all (books haven't posted yet).  For those games, retry every
    # _PROP_RETRY_INTERVAL_SEC seconds until lines appear or the game starts.
    # Games that already have lines are unaffected — their picks flow through normally.
    _game_start_utc: dict[int, datetime] = {}
    for _g in schedule:
        _gk = _g.get("game_pk")
        _raw = _g.get("game_time_utc", "")
        if _gk and _raw:
            try:
                _game_start_utc[_gk] = datetime.strptime(
                    _raw, "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=timezone.utc)
            except ValueError:
                pass

    _pks_with_lines = {v["game_pk"] for v in prop_catalog.values() if v.get("game_pk") is not None}
    _no_lines_pks = [gk for gk in game_pk_list if gk not in _pks_with_lines]

    if _no_lines_pks:
        _now = datetime.now(timezone.utc)
        _still_pending = [
            gk for gk in _no_lines_pks
            if _game_start_utc.get(gk, _now) > _now
        ]
        _already_started = [gk for gk in _no_lines_pks if gk not in _still_pending]

        if _already_started:
            log.info(
                "No lines for game_pk(s) %s and game(s) have already started — skipping.",
                _already_started,
            )

        if _still_pending:
            log.info(
                "No Odds API prop lines yet for game_pk(s) %s — "
                "processing other games now and scheduling retry in %d min.",
                _still_pending, _PROP_RETRY_INTERVAL_SEC // 60,
            )
            _schedule_lines_retry(game_date, _still_pending)

        # Exclude no-lines games from this run so other games aren't blocked
        _no_lines_set = set(_no_lines_pks)
        prop_catalog = {k: v for k, v in prop_catalog.items()
                        if v.get("game_pk") not in _no_lines_set}

    # 7. For each player+prop, find the single best-edge Over bet across all
    #    books and lines using the no-vig consensus method, then apply the
    #    same thresholds as the live model.
    picks: list[dict] = []

    # Track per-game bet counts in memory so the correlation cap works before
    # any bets are flushed to the DB.
    game_bets_in_flight: dict[int, int] = {}

    for (player_name, prop_type), catalog in prop_catalog.items():
        game_pk = catalog["game_pk"]
        pitchers = catalog["pitchers"]
        weather = catalog["weather"]
        event_id = catalog["event_id"]

        # --- Resolve player IDs from lineup / probable pitcher data ---
        lineup_ctx = lineups_full.get(game_pk, {}) if game_pk else {}
        name_to_id = lineup_ctx.get("name_to_id", {})
        player_name_lower = player_name.lower()

        is_pitcher_prop = prop_type == "strikeouts"

        if is_pitcher_prop:
            # K prop: player_name IS the pitcher. Match against probable_pitchers.
            game_pitchers = pitchers  # {home_pitcher_id/name, away_pitcher_id/name}
            player_id = 0
            for side in ("home", "away"):
                pid = game_pitchers.get(f"{side}_pitcher_id")
                pname = (game_pitchers.get(f"{side}_pitcher_name") or "").lower()
                if pid and (player_name_lower in pname or pname in player_name_lower):
                    player_id = pid
                    break
            # Opposing lineup: the side NOT pitching
            opposing_side = None
            if game_pitchers.get("home_pitcher_id") == player_id:
                opposing_side = "away"
            elif game_pitchers.get("away_pitcher_id") == player_id:
                opposing_side = "home"
            opposing_lineup = [
                {"batter_id": p["player_id"], "lineup_spot": p.get("batting_order", 9)}
                for p in lineup_ctx.get(opposing_side or "away", [])
                if p.get("player_id")
            ]
            pitcher_id = player_id
        else:
            # Batter prop: resolve batter_id from confirmed lineup name map
            player_id = name_to_id.get(player_name_lower, 0)
            # Opposing pitcher: the pitcher on the OTHER team from this batter
            # Determine batter's team side by checking which lineup they appear in
            batter_side = None
            for side in ("home", "away"):
                if any(
                    (p.get("name") or "").lower() == player_name_lower
                    for p in lineup_ctx.get(side, [])
                ):
                    batter_side = side
                    break
            opp_side = "away" if batter_side == "home" else "home"
            pitcher_id = pitchers.get(f"{opp_side}_pitcher_id") or 0

        # Fallback: if still unresolved, search by name via MLB API (cached per run)
        if not player_id:
            if player_name not in _player_id_cache:
                try:
                    results = mlb_client.search_player(player_name)
                    _player_id_cache[player_name] = (results[0].get("id") or results[0].get("player_id") or 0) if results else 0
                except Exception:
                    _player_id_cache[player_name] = 0
            player_id = _player_id_cache[player_name]
            if player_id:
                log.debug("Resolved player_id=%d for %s via name search", player_id, player_name)

        if player_id:
            log.debug("Resolved player_id=%d for %s (%s)", player_id, player_name, prop_type)
        else:
            log.debug("No player_id resolved for %s (%s) — using league average", player_name, prop_type)

        # Resolve team / opponent / game_time from game_info lookup
        ginfo = game_info.get(game_pk, {}) if game_pk else {}
        if is_pitcher_prop:
            # Determine pitcher's team side
            if game_pitchers.get("home_pitcher_id") == player_id:
                player_team = ginfo.get("home_team", "")
                opp_team = ginfo.get("away_team", "")
            elif game_pitchers.get("away_pitcher_id") == player_id:
                player_team = ginfo.get("away_team", "")
                opp_team = ginfo.get("home_team", "")
            else:
                player_team = ""
                opp_team = ""
        else:
            if batter_side == "home":
                player_team = ginfo.get("home_team", "")
                opp_team = ginfo.get("away_team", "")
            elif batter_side == "away":
                player_team = ginfo.get("away_team", "")
                opp_team = ginfo.get("home_team", "")
            else:
                player_team = ""
                opp_team = ""
        game_time_et = ginfo.get("game_time_et", "")

        best: dict | None = None  # best-edge candidate for this player+prop

        for line, book_sides in catalog["lines"].items():
            # Gather market implied probs across books at this line
            over_implieds: list[float] = []
            under_implieds: list[float] = []
            for sides in book_sides.values():
                if "Over" in sides:
                    over_implieds.append(sides["Over"]["implied_prob"])
                if "Under" in sides:
                    under_implieds.append(sides["Under"]["implied_prob"])

            if not over_implieds:
                continue

            avg_over_implied = sum(over_implieds) / len(over_implieds)
            avg_under_implied = (
                sum(under_implieds) / len(under_implieds)
                if under_implieds
                else 1.0 - avg_over_implied
            )

            # --- Model probabilities ---
            # Priority: trained ML model (pkl) > matchup-context adjustment > league average
            # All three trained models blend their output 70% model / 30% market at prediction.
            umpire_factor = (todays_umpires.get(game_pk) or {}).get("k_factor", 1.0) if game_pk else 1.0

            # Compute no-vig probability first so the model blend receives a true probability,
            # not the vig-inflated implied probability.
            vig_sum_cur = avg_over_implied + avg_under_implied
            current_no_vig = avg_over_implied / vig_sum_cur if vig_sum_cur > 0 else 0.5
            avg_market_implied = current_no_vig  # de-vigged; used for 30% market blend

            # Line movement: compare current no-vig implied vs morning opening snapshot
            opening_no_vig = db.get_opening_snapshot(today_str, player_name, prop_type, line)
            line_movement = round(current_no_vig - opening_no_vig, 4) if opening_no_vig else 0.0

            # Opposing lineup xwOBA (for pitcher models)
            opp_xwoba_list = [p.get("player_id") for p in lineup_ctx.get(
                "away" if (not is_pitcher_prop or game_pitchers.get("home_pitcher_id") == player_id) else "home",
                []
            ) if p.get("player_id")]
            opp_lineup_xwoba = 0.320  # MLB avg xwOBA default
            if opp_xwoba_list and player_id:
                from k_features import _opp_lineup_xwoba as _k_opp_xwoba
                computed = _k_opp_xwoba(opp_xwoba_list, db)
                if computed is not None:
                    opp_lineup_xwoba = computed

            prop_features: dict | None = None  # captured per-prop for feature_snapshot

            if prop_type == "strikeouts":
                km = k_model_module.get_model()
                if km.is_trained and player_id:
                    # Read live pitcher swstr stats from DB
                    swstr_rows = db.get_player_stats(player_id, "pitcher_rolling_swstr_rate", days=2)
                    swstr_rate = float(swstr_rows[-1]["value"]) if swstr_rows else 0.0
                    ff_whiff_rows = db.get_player_stats(player_id, "pitcher_rolling_ff_whiff", days=2)
                    ff_whiff = float(ff_whiff_rows[-1]["value"]) if ff_whiff_rows else 0.0
                    sl_whiff_rows = db.get_player_stats(player_id, "pitcher_rolling_sl_whiff", days=2)
                    sl_whiff = float(sl_whiff_rows[-1]["value"]) if sl_whiff_rows else 0.0
                    ch_whiff_rows = db.get_player_stats(player_id, "pitcher_rolling_ch_whiff", days=2)
                    ch_whiff = float(ch_whiff_rows[-1]["value"]) if ch_whiff_rows else 0.0

                    # Pitcher-specific K rate + game context from season stats
                    _LEAGUE_K_RATE = 0.224  # K/BF league average
                    k_rate_season = _LEAGUE_K_RATE
                    avg_ip_30d = _MLB_AVG_IP
                    pstats: dict = {}
                    try:
                        pstats = mlb_client.get_player_season_stats(player_id, current_season, group="pitching")
                        bf = float(pstats.get("battersFaced") or 0)
                        if bf > 0:
                            k_rate_season = float(pstats.get("strikeOuts") or 0) / bf
                        gs = float(pstats.get("gamesStarted") or 0)
                        ip = float(pstats.get("inningsPitched") or 0)
                        if gs > 0:
                            avg_ip_30d = ip / gs
                    except Exception:
                        pass

                    # Days rest from pitcher's most recent prior start
                    days_rest_k = 5.0
                    try:
                        p_game_log = mlb_client.get_player_game_log(player_id, current_season, group="pitching")
                        starts_log = [
                            g for g in p_game_log
                            if g.get("gamesStarted", 0) == 1
                        ]
                        # game_log is most-recent-first; find first start before today
                        today_dt = game_date
                        for sg in starts_log:
                            sg_date_str = (sg.get("date") or "")[:10]
                            if sg_date_str and sg_date_str < today_dt.isoformat():
                                from datetime import datetime as _dt
                                sg_date = _dt.strptime(sg_date_str, "%Y-%m-%d").date()
                                days_rest_k = float((today_dt - sg_date).days)
                                break
                    except Exception:
                        pass

                    # is_home from probable_pitchers
                    is_home_k = 1.0 if game_pitchers.get("home_pitcher_id") == player_id else 0.0

                    # weather_k_factor from pre-fetched venue weather
                    weather_k_factor = float(weather.get("k_weather_factor", 1.0))

                    # park_k_factor from DB
                    venue_name_k = ginfo.get("venue_name", "")
                    park_row_k = db.get_park_factor(venue_name_k, "neutral") if venue_name_k else None
                    park_k_factor = float(park_row_k["K_factor"]) if park_row_k and park_row_k.get("K_factor") else 1.0

                    # opp_k_rate_season and lineup_handedness_split from opposing lineup
                    opp_k_rate = 0.224
                    lineup_handedness_split = 0.5
                    opp_batter_ids = [e["batter_id"] for e in opposing_lineup if e.get("batter_id")]
                    if opp_batter_ids:
                        from k_features import _opponent_k_stats as _k_opp_stats
                        ok_stats = _k_opp_stats(opp_batter_ids, db)
                        if ok_stats.get("opp_k_rate_season") is not None:
                            opp_k_rate = ok_stats["opp_k_rate_season"]
                        # Compute fraction of left-handed batters in opposing lineup
                        left_count = 0
                        resolved = 0
                        for bid_opp in opp_batter_ids:
                            try:
                                hand = mlb_client.get_player_handedness(bid_opp)
                                side = hand.get("bat_side")
                                if side:
                                    resolved += 1
                                    if side == "L":
                                        left_count += 1
                            except Exception:
                                pass
                        if resolved > 0:
                            lineup_handedness_split = left_count / resolved

                    # Pitch-type BvP matchup factor: pitcher mix × lineup whiff rates
                    k_matchup_factor = 1.0
                    if opposing_lineup:
                        try:
                            from lineup_whiff_matchup import compute_lineup_k_matchup
                            k_matchup_result = compute_lineup_k_matchup(player_id, opposing_lineup, db)
                            k_matchup_factor = k_matchup_result.get("matchup_factor", 1.0)
                            log.debug(
                                "K BvP matchup: pitcher_id=%d factor=%.3f quality=%s",
                                player_id, k_matchup_factor, k_matchup_result.get("data_quality"),
                            )
                        except Exception:
                            pass

                    features = {
                        "csw_rate_30d":            swstr_rate,   # swstr proxies CSW
                        "k_rate_30d":              k_rate_season,
                        "k_rate_season":           k_rate_season,
                        "whiff_rate_30d":          swstr_rate,
                        "swstr_rate_30d":          swstr_rate,
                        "ff_whiff_rate_30d":       ff_whiff,
                        "sl_whiff_rate_30d":       sl_whiff,
                        "ch_whiff_rate_30d":       ch_whiff,
                        # Opposing lineup features (v4 model)
                        "opp_k_rate_season":       opp_k_rate,
                        "opp_lineup_whiff_factor": k_matchup_factor,
                        "lineup_lhb_pct":          lineup_handedness_split,
                        # Legacy keys (used by k-v3 pkl; kept for backward compat)
                        "opp_k_rate_30d":          opp_k_rate,
                        "lineup_handedness_split": lineup_handedness_split,
                        "opp_lineup_xwoba":        opp_lineup_xwoba,
                        "umpire_k_factor":         umpire_factor,
                        "weather_k_factor":        weather_k_factor,
                        "park_k_factor":           park_k_factor,
                        "is_home":                 is_home_k,
                        "days_rest":               days_rest_k,
                        "avg_ip_30d":              avg_ip_30d,
                        "is_opener_risk":          0.0,
                        "matchup_factor":          k_matchup_factor,
                        "market_implied_over":     avg_market_implied,
                        "line_movement":           line_movement,
                    }
                    prop_features = features
                    try:
                        result = km.predict_with_blend(features, float(line), avg_market_implied)
                        # Apply batter lineup matchup factor post-prediction.
                        # The K model is trained on pitcher-only features; k_matchup_factor
                        # scales the model's lambda by the opposing lineup's relative
                        # strikeout difficulty vs the league-average lineup.
                        if k_matchup_factor != 1.0:
                            adj_lambda = result["lambda"] * k_matchup_factor
                            adj_model_prob_over = float(
                                1.0 - poisson.cdf(math.floor(float(line)), adj_lambda)
                            )
                            model_prob_over = (
                                adj_model_prob_over * (1.0 - k_model_module.MARKET_BLEND)
                                + avg_market_implied * k_model_module.MARKET_BLEND
                            )
                        else:
                            model_prob_over = result["final_prob_over"]
                    except Exception:
                        model_prob_over = None

                    if model_prob_over is None:
                        # Fallback: matchup-context Poisson
                        adj_lambda = matchup_context.adjusted_k_lambda(
                            player_id, opposing_lineup, umpire_factor, db, mlb_client, current_season
                        )
                        model_prob_over = float(1.0 - poisson.cdf(int(line), mu=adj_lambda))
                else:
                    adj_lambda = (
                        matchup_context.adjusted_k_lambda(
                            player_id, opposing_lineup, umpire_factor, db, mlb_client, current_season
                        ) if player_id
                        else _MLB_K_LAMBDA * umpire_factor
                    )
                    model_prob_over = float(1.0 - poisson.cdf(int(line), mu=adj_lambda))

            elif prop_type == "hits":
                hm = hits_model_module.get_model()
                if hm.is_trained and player_id and pitcher_id:
                    # Read live batter discipline stats from DB
                    chase_rows = db.get_player_stats(player_id, "batter_rolling_chase_rate", days=2)
                    chase_rate = float(chase_rows[-1]["value"]) if chase_rows else 0.0
                    speed_rows = db.get_player_stats(player_id, "batter_sprint_speed", days=180)
                    sprint_speed = float(speed_rows[-1]["value"]) if speed_rows else 27.0

                    # Batter season stats — K rate, walk rate, batting average, HR rate
                    batter_k_rate = 0.224
                    batter_walk_rate = 0.084
                    hit_rate_season = _MLB_BA
                    batter_hr_rate = 0.030
                    try:
                        bstats = mlb_client.get_player_season_stats(player_id, current_season, group="hitting")
                        bpa = float(bstats.get("plateAppearances") or 0)
                        if bpa > 0:
                            batter_k_rate = float(bstats.get("strikeOuts") or 0) / bpa
                            batter_walk_rate = float(bstats.get("baseOnBalls") or 0) / bpa
                            batter_hr_rate = float(bstats.get("homeRuns") or 0) / bpa
                        raw_avg = bstats.get("avg")
                        if raw_avg:
                            hit_rate_season = float(raw_avg)
                    except Exception:
                        pass

                    # Pitcher stats: hit rate, K rate, BABIP allowed, and GB rate
                    pitcher_hit_rate = 0.243
                    pitcher_k_rate_for_hits = 0.224
                    pitcher_babip = 0.295
                    pitcher_gb_pct = 0.44   # league-average GB rate
                    if pitcher_id:
                        try:
                            h_pstats = mlb_client.get_player_season_stats(pitcher_id, current_season, group="pitching")
                            h_bf = float(h_pstats.get("battersFaced") or 0)
                            if h_bf >= 10:
                                h_hits = float(h_pstats.get("hits") or 0)
                                h_ks   = float(h_pstats.get("strikeOuts") or 0)
                                h_hrs  = float(h_pstats.get("homeRuns") or 0)
                                h_bbs  = float(h_pstats.get("baseOnBalls") or 0)
                                pitcher_hit_rate = h_hits / h_bf
                                pitcher_k_rate_for_hits = h_ks / h_bf
                                bip = h_bf - h_ks - h_hrs - h_bbs
                                if bip > 0:
                                    pitcher_babip = (h_hits - h_hrs) / bip
                                    pitcher_babip = min(max(pitcher_babip, 0.200), 0.400)
                                go = float(h_pstats.get("groundOuts") or 0)
                                ao = float(h_pstats.get("airOuts") or 0)
                                if go + ao > 0:
                                    pitcher_gb_pct = go / (go + ao)
                        except Exception:
                            pass

                    # Pitch-type BvP contact factor: 1 - (pitcher_mix × batter_whiff_rates)
                    hits_bvp_contact = 0.776  # league avg contact rate fallback
                    if player_id and pitcher_id:
                        try:
                            from pitch_type_matchup import get_pitcher_pitch_mix
                            from lineup_whiff_matchup import get_batter_whiff_rates
                            pm = get_pitcher_pitch_mix(pitcher_id, db)
                            if pm:
                                wr = get_batter_whiff_rates(player_id, db)
                                weighted_whiff = sum(
                                    frac * wr.get(pt, 0.240)
                                    for pt, frac in pm.items() if frac > 0.0
                                )
                                hits_bvp_contact = min(max(1.0 - weighted_whiff, 0.50), 0.95)
                        except Exception:
                            pass

                    # Batter lineup spot from confirmed lineup
                    batter_lineup_spot = 4.0
                    if batter_side:
                        for lp in lineup_ctx.get(batter_side, []):
                            if (lp.get("name") or "").lower() == player_name_lower:
                                batter_lineup_spot = float(lp.get("batting_order") or 4)
                                break

                    # Hard-hit rate, exit velocity, and launch angle from DB rolling stats
                    hh_rows = db.get_player_stats(player_id, "batter_rolling_hard_hit_rate", days=2)
                    hard_hit_30d = float(hh_rows[-1]["value"]) if hh_rows else 0.38

                    ev_rows = db.get_player_stats(player_id, "batter_rolling_avg_exit_velocity", days=2)
                    avg_exit_velo_30d = float(ev_rows[-1]["value"]) if ev_rows else 88.0

                    la_rows = db.get_player_stats(player_id, "batter_rolling_avg_launch_angle", days=2)
                    avg_launch_angle_30d = float(la_rows[-1]["value"]) if la_rows else 10.0

                    # contact_rate_30d ≈ 1 - K rate (swings making contact)
                    contact_rate_30d = min(0.95, max(0.50, 1.0 - batter_k_rate))

                    # babip_30d = (H - HR) / (PA - K - HR - BB) using season totals
                    _bip = 1.0 - batter_k_rate - batter_hr_rate - batter_walk_rate
                    _hit_bip = hit_rate_season - batter_hr_rate
                    babip_30d = min(0.500, max(0.150, _hit_bip / max(_bip, 0.30))) if _bip > 0 else 0.295

                    # Park factor for hits from venue
                    hits_venue = ginfo.get("venue_name", "")
                    park_factor_hits = 1.0
                    if hits_venue:
                        pf_row = db.get_park_factor(hits_venue, "neutral")
                        if pf_row and pf_row.get("H_factor"):
                            park_factor_hits = float(pf_row["H_factor"])

                    # Platoon advantage: 1 if batter and pitcher throw from opposite sides
                    is_platoon_hits = 0.0
                    if player_id and pitcher_id:
                        try:
                            b_hand = mlb_client.get_player_handedness(player_id)
                            p_hand = mlb_client.get_player_handedness(pitcher_id)
                            bat = b_hand.get("bat_side")
                            pitch = p_hand.get("pitch_hand")
                            if bat and pitch and bat != "S":
                                is_platoon_hits = 1.0 if bat != pitch else 0.0
                        except Exception:
                            pass

                    _hits_pa_by_spot: dict[int, float] = {
                        1: 3.20, 2: 3.10, 3: 3.00, 4: 2.90, 5: 2.80,
                        6: 2.70, 7: 2.65, 8: 2.55, 9: 2.40,
                    }
                    _hits_expected_pa = _hits_pa_by_spot.get(int(batter_lineup_spot), 2.85)

                    _game_total = catalog.get("game_total")
                    _team_implied_runs = round(_game_total / 2, 2) if _game_total else 4.5

                    features = {
                        "contact_rate_30d":          contact_rate_30d,
                        "babip_30d":                 babip_30d,
                        "avg_exit_velo_30d":         avg_exit_velo_30d,
                        "hard_hit_rate_30d":         hard_hit_30d,
                        "hit_rate_season":           hit_rate_season,
                        "avg_launch_angle_30d":      avg_launch_angle_30d,
                        "line_drive_rate_30d":       0.20,  # no rolling stat stored; minor feature
                        "pitcher_babip_allowed_30d":       pitcher_babip,
                        "pitcher_babip_allowed_season":    pitcher_babip,
                        "pitcher_hit_rate_allowed_season": pitcher_hit_rate,
                        "pitcher_k_rate_season":           pitcher_k_rate_for_hits,
                        "pitcher_gb_pct":                  pitcher_gb_pct,
                        "pitcher_bvp_contact_factor":      hits_bvp_contact,
                        "expected_pa":               _hits_expected_pa,
                        "team_implied_runs":         _team_implied_runs,
                        "park_factor_hits_h":        park_factor_hits,
                        "lineup_spot":               batter_lineup_spot,
                        "is_platoon_advantage":      is_platoon_hits,
                        "batter_k_rate_season":      batter_k_rate,
                        "batter_walk_rate_season":   batter_walk_rate,
                        "chase_rate_30d":            chase_rate,
                        "sprint_speed":              sprint_speed,
                        "opp_lineup_xwoba":          opp_lineup_xwoba,
                        "line_movement":             line_movement,
                        "market_implied_prob":       avg_market_implied,
                    }
                    prop_features = features
                    try:
                        result = hm.predict_with_blend(features, avg_market_implied)
                        # HitsModel is a binary classifier for P(1+ hit in a game).
                        # For line=0.5 that equals model_prob directly.
                        # For line=1.5 / 2.5 we must infer the per-PA hit probability
                        # from the model's P(1+ hit) output and reapply the Binomial.
                        p1plus = result["final_prob"]
                        n_pa = _expected_pa(batter_lineup_spot)
                        # Infer per-PA probability: P(0 hits) = (1-p_pa)^n = 1 - P(1+)
                        p_pa = 1.0 - max(0.0, 1.0 - p1plus) ** (1.0 / n_pa)
                        p_pa = min(max(p_pa, 0.001), 0.999)
                        model_prob_over = _binom_over(line, n_pa, p_pa)
                    except Exception:
                        model_prob_over = None

                    if model_prob_over is None:
                        n_pa = _expected_pa(batter_lineup_spot)
                        adj_p = matchup_context.adjusted_hit_prob(
                            player_id, pitcher_id, db, mlb_client, current_season
                        )
                        model_prob_over = _binom_over(line, n_pa, adj_p)
                else:
                    n_pa = _expected_pa(batter_lineup_spot)
                    adj_p = (
                        matchup_context.adjusted_hit_prob(
                            player_id, pitcher_id, db, mlb_client, current_season
                        ) if player_id and pitcher_id
                        else _MLB_BA
                    )
                    model_prob_over = _binom_over(line, n_pa, adj_p)

            elif prop_type == "home_runs":
                hrm = hr_model_module.get_model()
                if hrm.is_trained and player_id:
                    # Resolve batter lineup spot from confirmed lineup
                    hr_batter_side = batter_side if not is_pitcher_prop else None
                    hr_lineup_spot = 4.0
                    if hr_batter_side:
                        for lp in lineup_ctx.get(hr_batter_side, []):
                            if (lp.get("name") or "").lower() == player_name_lower:
                                hr_lineup_spot = float(lp.get("batting_order") or 4)
                                break

                    features = matchup_context.batter_hr_features(
                        player_id, current_season, mlb_client,
                        lineup_spot=hr_lineup_spot,
                    )
                    features["market_implied_prob"] = avg_market_implied
                    features["opp_lineup_xwoba"] = opp_lineup_xwoba
                    features["line_movement"] = line_movement

                    # Weather HR multiplier from pre-fetched venue weather
                    features["weather_hr_multiplier"] = float(weather.get("hr_weather_multiplier", 1.0))

                    # Park HR factor from DB
                    venue_name_hr = ginfo.get("venue_name", "")
                    park_row_hr = db.get_park_factor(venue_name_hr, "neutral") if venue_name_hr else None
                    if park_row_hr and park_row_hr.get("HR_factor"):
                        features["park_factor_h"] = float(park_row_hr["HR_factor"])

                    # Pitcher HR rate, GB%, and platoon advantage
                    if pitcher_id:
                        try:
                            hr_pstats = mlb_client.get_player_season_stats(pitcher_id, current_season, group="pitching")
                            hr_bf = float(hr_pstats.get("battersFaced") or 0)
                            if hr_bf > 0:
                                features["pitcher_hr_rate_season"] = float(hr_pstats.get("homeRuns") or 0) / hr_bf
                            go = float(hr_pstats.get("groundOuts") or 0)
                            ao = float(hr_pstats.get("airOuts") or 0)
                            if go + ao > 0:
                                features["pitcher_gb_pct"] = go / (go + ao)
                        except Exception:
                            pass
                        try:
                            b_hand_hr = mlb_client.get_player_handedness(player_id)
                            p_hand_hr = mlb_client.get_player_handedness(pitcher_id)
                            bat_hr = b_hand_hr.get("bat_side")
                            pitch_hr = p_hand_hr.get("pitch_hand")
                            if bat_hr and pitch_hr and bat_hr != "S":
                                features["is_platoon_advantage"] = 1.0 if bat_hr != pitch_hr else 0.0
                                features["batter_hand_vs_pitcher"] = 1.0 if bat_hr != pitch_hr else 0.0
                        except Exception:
                            pass

                    # Barrel rate from DB rolling stats
                    barrel_rows = db.get_player_stats(player_id, "batter_rolling_barrel_rate", days=2)
                    if barrel_rows:
                        br = float(barrel_rows[-1]["value"])
                        features["barrel_rate_30d"] = br
                        features["barrel_rate_60d"] = br

                    if pitcher_id:
                        features["bvp_factor"] = matchup_context.adjusted_hr_lambda(
                            player_id, pitcher_id, db
                        ) / _MLB_HR_LAMBDA
                    prop_features = features
                    try:
                        result = hrm.predict_with_blend(features, avg_market_implied)
                        model_prob_over = result["final_prob"]
                    except Exception:
                        model_prob_over = None

                    if model_prob_over is None:
                        adj_lambda = matchup_context.adjusted_hr_lambda(player_id, pitcher_id, db) if pitcher_id else _MLB_HR_LAMBDA
                        model_prob_over = float(1.0 - poisson.cdf(int(line), mu=adj_lambda))
                else:
                    adj_lambda = (
                        matchup_context.adjusted_hr_lambda(player_id, pitcher_id, db)
                        if player_id and pitcher_id
                        else _MLB_HR_LAMBDA
                    )
                    model_prob_over = float(1.0 - poisson.cdf(int(line), mu=adj_lambda))

            elif prop_type == "rbis":
                model_prob_over = _rbis_model_prob_over(line)
            elif prop_type == "runs_scored":
                model_prob_over = _runs_model_prob_over(line)
            elif prop_type == "singles":
                model_prob_over = _singles_model_prob_over(line)
            elif prop_type == "doubles":
                model_prob_over = _doubles_model_prob_over(line)
            elif prop_type == "hits_runs_rbis":
                model_prob_over = _hrr_model_prob_over(line)
            else:
                # Unknown props: no-vig market consensus
                vig_sum = avg_over_implied + avg_under_implied
                model_prob_over = avg_over_implied / vig_sum if vig_sum > 0 else 0.5

            model_prob_under = 1.0 - model_prob_over

            # Evaluate Over and Under for every book; pick best edge across both
            for pick_side, model_prob_side in [
                ("Over", model_prob_over),
                ("Under", model_prob_under),
            ]:
                for book_key, sides in book_sides.items():
                    if pick_side not in sides:
                        continue
                    outcome = sides[pick_side]
                    bet_implied = outcome["implied_prob"]
                    odds_american = outcome["odds"]

                    # Edge vs de-vigged market consensus (fair comparison; model blend
                    # already uses avg_market_implied which is the no-vig probability).
                    no_vig_side = avg_market_implied if pick_side == "Over" else (1.0 - avg_market_implied)
                    edge = model_prob_side - no_vig_side

                    is_better = (
                        best is None
                        or edge > best["edge"]
                        or (edge == best["edge"] and book_key == _PREFERRED_BOOK)
                    )
                    if is_better:
                        # Projection = model probability for the chosen side as a %
                        # This varies per line/direction and explains why the pick has edge
                        # e.g. "73.0" for Under 1.5 hits means model says 73% chance Under
                        display_projection = round(model_prob_side * 100, 1)

                        # No-vig market consensus probability for the chosen side
                        vig_sum = avg_over_implied + avg_under_implied
                        if vig_sum > 0:
                            no_vig_over = avg_over_implied / vig_sum
                        else:
                            no_vig_over = 0.5
                        no_vig_prob = no_vig_over if pick_side == "Over" else (1.0 - no_vig_over)

                        best = {
                            "game_date": today_str,
                            "game_pk": game_pk,
                            "event_id": event_id,
                            "player_name": player_name,
                            "player_id": player_id,
                            "team": player_team,
                            "opponent": opp_team,
                            "game_time": game_time_et,
                            "prop_type": prop_type,
                            "line": line,
                            "pick": pick_side,
                            "book": book_key,
                            "odds": odds_american,
                            "implied_prob": bet_implied,
                            "no_vig_prob": no_vig_prob,
                            "model_prob": model_prob_side,
                            "model_projection": display_projection,
                            "edge": edge,
                            "model_version": _MODEL_VERSION,
                            "features": prop_features,
                        }

        if best is None:
            continue

        # Run preflight with the best candidate's context
        try:
            preflight = risk.run_preflight(
                game_pk=game_pk or 0,
                player_id=player_id,
                pitcher_id=pitcher_id,
                prop_type=prop_type,
                opening_line=float(best["odds"]),
                current_line=float(best["odds"]),
                weather=weather,
                mlb_client=mlb_client,
                db=db,
            )
        except Exception:
            log.exception("Preflight error for %s %s — skipping.", player_name, prop_type)
            continue

        # Weather is the only critical check we can reliably evaluate without
        # real player IDs (lineup/injury stubs always pass with player_id=0).
        weather_check = next(
            (c for c in preflight.get("checks", []) if c["name"] == "weather_playable"),
            None,
        )
        if weather_check and not weather_check["passed"]:
            log.debug("Skip %s %s — weather: %s", player_name, prop_type, weather_check["reason"])
            continue

        # Edge threshold gate — require minimum edge from the statistical model
        edge = best["edge"]
        if edge < sizing.MIN_EDGE:
            log.info(
                "Below threshold: %s %s %s %.1f @ %s  model=%.1f%%  market=%.1f%%  edge=%.2f%%  (min=%.1f%%)",
                player_name, prop_type, best["pick"], best["line"], best["book"],
                best["model_prob"] * 100, best.get("no_vig_prob", best["implied_prob"]) * 100,
                edge * 100, sizing.MIN_EDGE * 100,
            )
            continue

        # K model line cap: Under bets at high lines lose despite large claimed edges.
        # Elite strikeout pitchers (Ohtani, Wheeler, Glasnow, etc.) consistently go over
        # 5.5-K lines; the model underestimates them and labels it high-edge.
        if prop_type == "strikeouts" and best["pick"] == "Under" and best["line"] > K_LINE_CAP:
            log.info(
                "K-line cap: %s strikeouts Under %.1f skipped (max allowed=%.1f)",
                player_name, best["line"], K_LINE_CAP,
            )
            continue

        # Hits Under line cap: Hits Under ≥1.5 lose badly due to hits-v6 class_weight
        # miscalibration. Under 0.5 is profitable. Block until model is retrained.
        if prop_type == "hits" and best["pick"] == "Under" and best["line"] > HITS_UNDER_MAX_LINE:
            log.info(
                "Hits-Under cap: %s hits Under %.1f skipped (max allowed=%.1f)",
                player_name, best["line"], HITS_UNDER_MAX_LINE,
            )
            continue

        # Odds floor/ceiling: skip extreme juice and extreme longshots
        if best["odds"] < MAX_NEGATIVE_ODDS:
            log.debug(
                "Skip %s %s %s %.1f — odds %+d below MAX_NEGATIVE_ODDS %+d",
                player_name, prop_type, best["pick"], best["line"],
                best["odds"], MAX_NEGATIVE_ODDS,
            )
            continue
        if best["odds"] > MAX_POSITIVE_ODDS:
            log.debug(
                "Skip %s %s %s %.1f — odds %+d above MAX_POSITIVE_ODDS %+d (longshot filter)",
                player_name, prop_type, best["pick"], best["line"],
                best["odds"], MAX_POSITIVE_ODDS,
            )
            continue

        # Enforce correlation cap in-memory (bets not yet written to DB)
        if game_bets_in_flight.get(game_pk or 0, 0) >= sizing.CORRELATION_CAP:
            reason = f"correlation cap ({sizing.CORRELATION_CAP}) reached for game_pk={game_pk}"
            log.debug("Skip %s %s — %s", player_name, prop_type, reason)
            bet_ok, units, confidence = False, 0.0, None
        else:
            units = sizing.compute_units(edge, best["odds"], BANKROLL, method="tier")
            confidence = sizing.classify_confidence(edge)
            if confidence == "TRACKED":
                # Edge > 30%: record in DB for analysis but do not suggest
                bet_ok = False
                reason = "edge > 30% — tracked only, not suggested"
                units = 0.0
            elif confidence == "MEDIUM" and not ALLOW_MEDIUM:
                # MEDIUM bets (-8.4% ROI) disabled until CLV validates this tier.
                # Set env var MLB_ALLOW_MEDIUM=1 to re-enable.
                bet_ok = False
                reason = "MEDIUM confidence disabled (MLB_ALLOW_MEDIUM=0)"
                units = 0.0
            else:
                bet_ok = True
                reason = "OK"
                if units == 0.0:
                    # Edge above MIN_EDGE but tier returned 0 — use MIN_UNITS as floor
                    units = sizing.MIN_UNITS
                    confidence = "LOW"
                game_bets_in_flight[game_pk or 0] = game_bets_in_flight.get(game_pk or 0, 0) + 1

        pick_dict = {
            **best,
            "confidence": confidence,
            "units": units,
            "bet_ok": bet_ok,
            "skip_reason": None if bet_ok else reason,
            "preflight": preflight,
            "is_live": IS_LIVE,
            "stake_usd": round(units * BANKROLL, 2) if bet_ok else 0.0,
        }
        picks.append(pick_dict)
        log.debug(
            "Candidate: %s %s Over %.1f @ %s (edge=%.4f bet_ok=%s)",
            player_name, prop_type, best["line"], best["book"], best["edge"], bet_ok,
        )

    bettable = [p for p in picks if p["bet_ok"]]

    # ── Dedup rule: at most 1 bet per player per day ──────────────────────────
    # Step 1 — within this slot: keep only the highest-edge bet per player.
    best_per_player: dict[str, dict] = {}
    for p in bettable:
        pname = p["player_name"]
        if pname not in best_per_player or p["edge"] > best_per_player[pname]["edge"]:
            best_per_player[pname] = p

    dropped_within = len(bettable) - len(best_per_player)
    if dropped_within:
        log.info(
            "Dedup (within slot): dropped %d lower-edge duplicate(s) — keeping best per player.",
            dropped_within,
        )

    # Step 2 — across slots: skip players already bet on today from a prior slot.
    already_bet = db.get_bet_player_names(game_date.isoformat())
    bettable = []
    for pname, p in best_per_player.items():
        if pname in already_bet:
            log.info(
                "Dedup (cross-slot): skipping %s %s — already have a bet for this player today.",
                pname, p["prop_type"],
            )
        else:
            bettable.append(p)

    log.info(
        "Analysis complete — %d candidates evaluated, %d pass thresholds, %d after dedup.",
        len(picks), sum(1 for p in picks if p["bet_ok"]), len(bettable),
    )

    # 7. Persist bets (skip in dry-run)
    if not DRY_RUN:
        today_full = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        tracked_picks = [p for p in picks if p.get("confidence") == "TRACKED"]
        medium_disabled = [
            p for p in picks
            if p.get("confidence") == "MEDIUM" and not p.get("bet_ok")
            and p.get("skip_reason", "").startswith("MEDIUM confidence disabled")
        ]
        for p in bettable + tracked_picks + medium_disabled:
            try:
                is_tracked = p.get("confidence") == "TRACKED"
                bet_row = {
                    "bet_date": today_full,
                    "game_date": p["game_date"],
                    "game_time": p.get("game_time"),
                    "game_pk": p["game_pk"],
                    "player_id": p["player_id"],
                    "player_name": p["player_name"],
                    "team": p.get("team", ""),
                    "opponent": p.get("opponent", ""),
                    "prop_type": p["prop_type"],
                    "line": p["line"],
                    "pick": p["pick"],
                    "book": p["book"],
                    "odds": p["odds"],
                    "implied_prob": p["implied_prob"],
                    "no_vig_prob": p.get("no_vig_prob"),
                    "model_projection": p["model_projection"],
                    "model_prob": p["model_prob"],
                    "edge": p["edge"],
                    "confidence": p["confidence"],
                    "model_version": p["model_version"],
                    "feature_snapshot": p.get("features"),
                    "units": p["units"],
                    "stake_usd": p["stake_usd"],
                    "open_line": p["odds"],
                    "line_at_open": p["line"],
                    "is_live": p["is_live"],
                    "notes": (
                        "tracked-only: edge > 30%" if is_tracked
                        else "medium-disabled: pending CLV validation" if p.get("confidence") == "MEDIUM" and not p.get("bet_ok")
                        else None
                    ),
                }
                db.save_bet(bet_row)
            except Exception:
                log.exception("Failed to save bet for %s %s.", p["player_name"], p["prop_type"])

    # 8. Capture closing lines — run at analysis time (~T-60min) so lines are
    # available before games start. This is earlier than ideal (T-5min), but
    # captures a valid pre-game line for CLV computation. Per-slot captures
    # mean each game slot gets its own snapshot; run_results then computes CLV.
    if not DRY_RUN:
        try:
            closing_lines.run_pregame_capture(db, odds_client, mlb_client)
        except Exception:
            log.exception("Closing line capture failed (non-fatal — CLV will be empty for this slot).")

    return picks


# ---------------------------------------------------------------------------
# run_morning_snapshot
# ---------------------------------------------------------------------------

def run_morning_snapshot(
    game_date: date,
    db,
    mlb_client,
    odds_client,
) -> int:
    """Capture opening prop lines for today's games (~9 AM).

    Stores no-vig implied probabilities per player/prop/line so that
    run_analysis() can compute line_movement = current_implied - opening_implied.

    Returns
    -------
    int
        Number of opening snapshots stored.
    """
    date_str = game_date.isoformat()
    log.info("=== Morning snapshot for %s (dry_run=%s) ===", date_str, DRY_RUN)

    try:
        schedule = mlb_client.get_schedule(date_str)
        if not schedule:
            log.info("No games on %s — skipping morning snapshot.", date_str)
            return 0

        tomorrow_str = (game_date + timedelta(days=1)).isoformat()
        events_today = odds_client.get_mlb_events(date_str)
        events_tomorrow = odds_client.get_mlb_events(tomorrow_str)
        seen: set[str] = set()
        events: list[dict] = []
        for e in events_today + events_tomorrow:
            if e["event_id"] not in seen:
                seen.add(e["event_id"])
                events.append(e)

        all_props = odds_client.get_all_games_props(events, markets=MARKETS)
    except Exception:
        log.exception("Morning snapshot: failed to fetch props.")
        return 0

    # Per-player/prop_type/line: collect over implied across books, compute no-vig
    PlayerLineKey = tuple  # (player_name, prop_type, line)
    snapshot_data: dict[PlayerLineKey, dict[str, list[float]]] = {}

    for props_data in all_props.values():
        for book_markets in props_data.get("props", {}).values():
            for market_key, outcomes in book_markets.items():
                prop_type = _prop_type_from_market(market_key)
                for o in outcomes:
                    player_name = o.get("player_name", "")
                    line = o.get("line")
                    side = o.get("pick", "")
                    if not player_name or line is None or not side:
                        continue
                    key = (player_name, prop_type, line)
                    if key not in snapshot_data:
                        snapshot_data[key] = {"Over": [], "Under": []}
                    snapshot_data[key][side].append(o["implied_prob"])

    if DRY_RUN:
        log.info("DRY RUN — would store %d opening snapshots.", len(snapshot_data))
        return len(snapshot_data)

    count = 0
    for (player_name, prop_type, line), sides in snapshot_data.items():
        over_implieds = sides.get("Over", [])
        under_implieds = sides.get("Under", [])
        if not over_implieds:
            continue
        avg_over = sum(over_implieds) / len(over_implieds)
        avg_under = sum(under_implieds) / len(under_implieds) if under_implieds else 1.0 - avg_over
        vig_sum = avg_over + avg_under
        no_vig = avg_over / vig_sum if vig_sum > 0 else 0.5
        try:
            db.save_opening_snapshot(date_str, player_name, prop_type, line, no_vig)
            count += 1
        except Exception as exc:
            log.debug("Failed to save opening snapshot %s %s %.1f: %s", player_name, prop_type, line, exc)

    log.info("Morning snapshot complete — %d opening lines stored for %s.", count, date_str)
    return count


# ---------------------------------------------------------------------------
# run_results
# ---------------------------------------------------------------------------

def run_results(
    game_date: date,
    db,
    mlb_client,
    game_pks: str | None = None,
) -> dict:
    """Resolve open bets and compute CLV for the given game date.

    Parameters
    ----------
    game_pks:
        Optional comma-separated game_pk list. When provided, only bets for
        those specific games are resolved. Used by per-game incremental runs.

    Returns
    -------
    dict
        {
          'resolved': int,
          'wins': int,
          'losses': int,
          'pushes': int,
          'daily_pl': float,
          'clv_summary': dict,
          'results': list[dict],
        }
    """
    date_str = game_date.isoformat()
    log.info("=== Results run for %s game_pks=%s (dry_run=%s) ===", date_str, game_pks, DRY_RUN)

    unresolved = db.get_unresolved_bets()

    if game_pks:
        # When specific game_pks are provided, skip the date filter — bets for that
        # game may have been stored on a different calendar date (e.g. doubleheader
        # game 1 placed the day before, game 2 placed day-of).
        try:
            allowed_pks = {int(pk.strip()) for pk in game_pks.split(",") if pk.strip()}
            target_bets = [b for b in unresolved if b.get("game_pk") in allowed_pks]
            log.info("game-pks filter: %s → %d unresolved bets", game_pks, len(target_bets))
        except ValueError:
            log.warning("Invalid game_pks %r — processing all unresolved bets for %s.", game_pks, date_str)
            target_bets = [b for b in unresolved if b.get("game_date", "")[:10] == date_str]
    else:
        target_bets = [b for b in unresolved if b.get("game_date", "")[:10] == date_str]
        log.info("Found %d unresolved bets for %s.", len(target_bets), date_str)

    season = game_date.year
    results: list[dict] = []
    wins = losses = pushes = 0
    daily_pl = 0.0

    for bet in target_bets:
        bet_id = bet["id"]
        player_id = bet.get("player_id")
        prop_type = bet.get("prop_type", "")
        line = float(bet.get("line") or 0.0)
        pick = bet.get("pick", "over")
        odds_american = int(bet.get("odds") or -110)
        units = float(bet.get("units") or 0.0)

        if not player_id:
            log.warning("Bet id=%d has no player_id — skipping resolution.", bet_id)
            continue

        stat_group = _group_stat_key(prop_type)
        try:
            game_log = mlb_client.get_player_game_log(
                player_id=int(player_id),
                season=season,
                group=stat_group,
            )
        except Exception:
            log.exception("Failed to fetch game log for player_id=%s.", player_id)
            continue

        # Find entry matching the game date
        actual_stat: float | None = None
        for entry in game_log:
            if str(entry.get("date", ""))[:10] == date_str:
                actual_stat = _extract_stat_value(entry, prop_type)
                break

        if actual_stat is None:
            log.warning(
                "No game log entry found for player_id=%s on %s — skipping.",
                player_id, date_str,
            )
            continue

        outcome = _resolve_outcome(pick, line, actual_stat)
        pl_units = _compute_pl_units(outcome, odds_american, units)
        pl_usd = round(pl_units * BANKROLL, 2)

        if not DRY_RUN:
            try:
                db.update_result(
                    bet_id=bet_id,
                    actual_stat=actual_stat,
                    outcome=outcome,
                    pl_units=pl_units,
                    pl_usd=pl_usd,
                )
            except Exception:
                log.exception("Failed to update result for bet id=%d.", bet_id)
                continue

        if outcome == "WIN":
            wins += 1
        elif outcome == "LOSS":
            losses += 1
        elif outcome == "PUSH":
            pushes += 1

        daily_pl += pl_units
        results.append({**bet, "actual_stat": actual_stat, "outcome": outcome,
                        "pl_units": pl_units, "pl_usd": pl_usd})

    log.info(
        "Resolution complete — %d resolved (%d W / %d L / %d P), daily P/L=%.2f u",
        len(results), wins, losses, pushes, daily_pl,
    )

    # CLV computation (closing lines must already be captured by run_analysis)
    clv_summary: dict = {}
    try:
        closing_lines.run_clv_computation(game_date, db)
        clv_summary = closing_lines.get_daily_clv_summary(game_date, db)
        if clv_summary.get("missing_lines", 0) > 0:
            log.warning(
                "CLV: %d bets missing closing lines — pregame capture may have been skipped.",
                clv_summary["missing_lines"],
            )
        log.info("CLV summary: %s", clv_summary)
    except Exception:
        log.exception("CLV computation failed.")

    # Sync to cloud if DATABASE_URL is configured
    if os.getenv("DATABASE_URL"):
        try:
            from sync_to_cloud import run_full_sync
            sync_result = run_full_sync(since_days=1)
            log.info("Cloud sync: %s", sync_result)
        except Exception as e:
            log.warning("Cloud sync failed (non-fatal): %s", e)

    running_pl = db.get_running_pl()

    return {
        "resolved": len(results),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "daily_pl": daily_pl,
        "running_pl": running_pl,
        "clv_summary": clv_summary,
        "results": results,
    }


# ---------------------------------------------------------------------------
# run_statcast_update
# ---------------------------------------------------------------------------

def run_statcast_update(db) -> dict:
    """Refresh Statcast rolling metrics for yesterday's slate.

    Pulls the active player roster from recent bets so only relevant players
    are fetched.

    Returns
    -------
    dict
        Result dict from statcast_nightly.run_full_nightly().
    """
    log.info("=== Statcast nightly update ===")

    # Collect player IDs from recent bets as a lightweight roster proxy
    try:
        recent = db.get_unresolved_bets()
        batter_ids: list[int] = []
        pitcher_ids: list[int] = []
        for bet in recent:
            pid = bet.get("player_id")
            if not pid:
                continue
            pid = int(pid)
            prop = bet.get("prop_type", "")
            if "strikeout" in prop.lower():
                pitcher_ids.append(pid)
            else:
                batter_ids.append(pid)
        # De-duplicate
        batter_ids = list(dict.fromkeys(batter_ids))
        pitcher_ids = list(dict.fromkeys(pitcher_ids))
    except Exception:
        log.exception("Failed to build player ID list for Statcast — using empty lists.")
        batter_ids = []
        pitcher_ids = []

    log.info(
        "Running Statcast for %d batters, %d pitchers.",
        len(batter_ids), len(pitcher_ids),
    )

    try:
        result = statcast_nightly.run_full_nightly(
            batter_ids=batter_ids,
            pitcher_ids=pitcher_ids,
        )
        log.info("Statcast nightly complete: %s", result)
        return result
    except Exception:
        log.exception("statcast_nightly.run_full_nightly() failed.")
        raise


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLB player props daily pipeline runner."
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date to process (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--mode",
        choices=["analysis", "results", "statcast", "morning"],
        default="analysis",
        help="Pipeline mode to run.",
    )
    parser.add_argument(
        "--game-pks",
        default=None,
        metavar="PKS",
        help=(
            "Comma-separated game_pk list to process (e.g. '745123,745124'). "
            "Set by the scheduler for per-slot runs. When omitted, the current "
            "time slot is inferred automatically; analysis is skipped if no slot "
            "window matches."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Skip all writes (bets, DB updates, emails) but run everything else.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    global DRY_RUN
    if args.dry_run:
        DRY_RUN = True
    if DRY_RUN:
        log.info("DRY RUN mode — no writes or emails will be sent.")

    # Resolve target date
    if args.date:
        try:
            game_date = date.fromisoformat(args.date)
        except ValueError:
            log.error("Invalid --date value %r. Expected YYYY-MM-DD.", args.date)
            sys.exit(1)
    else:
        game_date = date.today()

    log.info("Mode=%s  Date=%s  GamePKs=%s  DryRun=%s", args.mode, game_date, args.game_pks, DRY_RUN)

    try:
        db = get_db()
    except Exception:
        log.exception("Failed to initialise database.")
        sys.exit(1)

    try:
        if args.mode == "analysis":
            mlb_client = get_client()
            weather_client = get_weather_client()
            odds_client = get_odds_client()

            picks = run_analysis(
                game_date=game_date,
                db=db,
                mlb_client=mlb_client,
                weather_client=weather_client,
                odds_client=odds_client,
                game_pks=args.game_pks,
            )

            running_pl = db.get_running_pl()
            bettable = [p for p in picks if p["bet_ok"]]

            if not DRY_RUN:
                if bettable:
                    notifier.send_picks_email(
                        picks=bettable,
                        running_pl=running_pl,
                        date=game_date.isoformat(),
                    )
                    log.info("Picks email sent (%d picks).", len(bettable))
                else:
                    # Count only the slot's games (from --game-pks), not the full-day schedule.
                    slot_game_count = (
                        len([pk for pk in args.game_pks.split(",") if pk.strip()])
                        if args.game_pks
                        else 0
                    )
                    log.info(
                        "No qualifying bets for this slot (%d game(s), %d candidate(s)). "
                        "No email sent.",
                        slot_game_count,
                        len(picks),
                    )
            else:
                log.info("DRY RUN — suppressed picks email. Bettable: %d / %d candidates", len(bettable), len(picks))

        elif args.mode == "results":
            mlb_client = get_client()

            summary = run_results(
                game_date=game_date,
                db=db,
                mlb_client=mlb_client,
                game_pks=args.game_pks,
            )

            if not DRY_RUN and not args.game_pks:
                # Only send the results email on full-day runs (no game_pks filter).
                # Per-game incremental runs resolve and record to DB silently.
                notifier.send_results_email(
                    date=game_date.isoformat(),
                    results=summary["results"],
                    daily_pl=summary["daily_pl"],
                    running_pl=summary["running_pl"],
                )
                log.info("Results email sent.")
            elif not DRY_RUN:
                log.info(
                    "Per-game results resolved (game_pks=%s): %s/%s/%s pl=%.2fu — no email.",
                    args.game_pks, summary["wins"], summary["losses"], summary["pushes"],
                    summary["daily_pl"],
                )
            else:
                log.info(
                    "DRY RUN — suppressed results email. Summary: %s/%s/%s pl=%.2fu",
                    summary["wins"], summary["losses"], summary["pushes"],
                    summary["daily_pl"],
                )

        elif args.mode == "statcast":
            run_statcast_update(db)

        elif args.mode == "morning":
            mlb_client = get_client()
            odds_client = get_odds_client()
            n = run_morning_snapshot(
                game_date=game_date,
                db=db,
                mlb_client=mlb_client,
                odds_client=odds_client,
            )
            log.info("Morning snapshot stored %d opening lines.", n)

    except Exception as exc:
        log.exception("Pipeline failed with unhandled exception: %s", exc)
        if not DRY_RUN:
            try:
                notifier.send_alert_email(
                    subject=f"Pipeline failure: {args.mode} {game_date}",
                    message=str(exc),
                    level="CRITICAL",
                )
            except Exception:
                log.exception("Failed to send alert email after pipeline failure.")
        sys.exit(1)


if __name__ == "__main__":
    main()
