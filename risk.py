"""
Pre-flight risk management checks for MLB player props betting model.

All check functions return (bool, str) — (passed, reason).
run_preflight() returns a structured dict with full diagnostics.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from db import get_db
    from mlb_api import MLBClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

PRECIP_POSTPONE_THRESHOLD: float = 0.5       # inches
OPENER_IP_THRESHOLD: float = 4.0             # innings pitched cutoff
OPENER_LOOKBACK_STARTS: int = 3              # number of recent starts to average
LINE_MOVEMENT_THRESHOLD: float = 0.05        # default implied prob shift (5%)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def implied_prob_from_american(odds: float) -> float:
    """Convert American odds to implied probability (raw, no vig removal).

    Args:
        odds: American odds line, e.g. -110, +150.

    Returns:
        Implied probability as a float in [0, 1].
    """
    if odds < 0:
        return (-odds) / (-odds + 100)
    return 100 / (odds + 100)


# ---------------------------------------------------------------------------
# Individual pre-flight checks
# ---------------------------------------------------------------------------

def check_lineup_confirmed(
    game_pk: int,
    player_id: int,
    mlb_client: "MLBClient",
) -> tuple[bool, str]:
    """Check whether the player appears in the confirmed batting lineup.

    Returns:
        (True, 'lineup confirmed') if found in batting order.
        (False, 'lineup not yet posted') if lineup is empty.
        (False, 'player not in confirmed lineup') if lineup exists but player absent.
    """
    try:
        lineup = mlb_client.get_confirmed_lineup(game_pk)
    except Exception as exc:
        logger.error("check_lineup_confirmed failed for game_pk=%s: %s", game_pk, exc)
        return False, "lineup check error"

    home_players = lineup.get("home", [])
    away_players = lineup.get("away", [])
    all_players = home_players + away_players

    if not all_players:
        return False, "lineup not yet posted"

    player_ids = {p["player_id"] for p in all_players}
    if player_id in player_ids:
        return True, "lineup confirmed"

    return False, "player not in confirmed lineup"


def check_pitcher_confirmed(
    game_pk: int,
    pitcher_id: int,
    mlb_client: "MLBClient",
) -> tuple[bool, str]:
    """Check whether the pitcher is confirmed as the starting pitcher.

    Uses the live feed boxscore first; falls back to the probable pitcher
    schedule endpoint keyed by today's date.

    Returns:
        (True, 'pitcher confirmed as starter') if confirmed.
        (False, 'probable pitcher not yet confirmed') if only probable or absent.
    """
    try:
        live = mlb_client.get_live_feed(game_pk)
    except Exception as exc:
        logger.error("check_pitcher_confirmed live feed failed for game_pk=%s: %s", game_pk, exc)
        live = {}

    # Check live boxscore pitchers (confirmed once game is close/started)
    teams = (
        live.get("liveData", {})
        .get("boxscore", {})
        .get("teams", {})
    )
    for side in ("home", "away"):
        side_pitchers = (
            teams.get(side, {})
            .get("players", {})
        )
        for key, player_data in side_pitchers.items():
            pid = player_data.get("person", {}).get("id")
            position = player_data.get("position", {}).get("abbreviation", "")
            if pid == pitcher_id and position == "P":
                game_status = (
                    live.get("gameData", {})
                    .get("status", {})
                    .get("detailedState", "")
                )
                # If lineup card has been exchanged or game is underway, pitcher is confirmed
                if game_status not in ("Scheduled", "Pre-Game", "Warmup"):
                    return True, "pitcher confirmed as starter"

    # Fall back to probable pitcher schedule for today's date
    try:
        today_str = date.today().isoformat()
        probables = mlb_client.get_probable_pitchers(today_str)
    except Exception as exc:
        logger.error("check_pitcher_confirmed probables failed: %s", exc)
        return False, "probable pitcher not yet confirmed"

    game_entry = probables.get(game_pk, {})
    probable_ids = {
        game_entry.get("home_pitcher_id"),
        game_entry.get("away_pitcher_id"),
    } - {None}

    if pitcher_id in probable_ids:
        return False, "probable pitcher not yet confirmed"

    return False, "probable pitcher not yet confirmed"


def check_injury_status(
    player_id: int,
    mlb_client: "MLBClient",
) -> tuple[bool, str]:
    """Check whether the player is on the Injured List.

    Uses the player's current team to fetch the IL roster.

    Returns:
        (False, 'player on IL') if on the injured list.
        (True, 'active') otherwise.
    """
    try:
        player_info = mlb_client.get_player_info(player_id)
        team_id: int | None = player_info.get("currentTeamId")
        if team_id is None:
            logger.warning("check_injury_status: no team found for player_id=%s", player_id)
            return True, "active"

        il_players = mlb_client.get_il_players(team_id)
    except Exception as exc:
        logger.error("check_injury_status failed for player_id=%s: %s", player_id, exc)
        return True, "active"

    il_ids = {p["player_id"] for p in il_players}
    if player_id in il_ids:
        return False, "player on IL"

    return True, "active"


def check_opener_bulk(
    game_pk: int,
    pitcher_id: int,
    mlb_client: "MLBClient",
    db: object,
) -> tuple[bool, str]:
    """Soft filter: check whether the pitcher is likely an opener/bulk guy.

    Looks at the pitcher's last OPENER_LOOKBACK_STARTS starts from the
    MLB game log. If average IP < OPENER_IP_THRESHOLD, flags as opener risk.

    This is a soft filter — the calling code may still place the bet
    with reduced confidence.

    Returns:
        (False, 'opener/bulk risk — K bet not recommended') if flagged.
        (True, 'starter expected') otherwise.
    """
    try:
        current_season = date.today().year
        game_log = mlb_client.get_player_game_log(
            pitcher_id,
            season=current_season,
            group="pitching",
        )
    except Exception as exc:
        logger.error(
            "check_opener_bulk: failed to fetch game log for pitcher_id=%s: %s",
            pitcher_id,
            exc,
        )
        return True, "starter expected"

    # Filter to starts only (games where pitcher entered as starter)
    starts = [g for g in game_log if g.get("gamesStarted", 0) >= 1]
    recent = starts[:OPENER_LOOKBACK_STARTS]

    if not recent:
        return True, "starter expected"

    total_ip = sum(float(g.get("inningsPitched", 0.0)) for g in recent)
    avg_ip = total_ip / len(recent)

    if avg_ip < OPENER_IP_THRESHOLD:
        return False, "opener/bulk risk — K bet not recommended"

    return True, "starter expected"


def check_weather_playable(weather_dict: dict) -> tuple[bool, str]:
    """Check whether conditions are suitable for the game to be played.

    Args:
        weather_dict: Output from WeatherClient.get_game_weather().

    Returns:
        (False, 'game likely postponed — precip > 0.5in') if precip > threshold.
        (False, 'game likely postponed — thunderstorm') if thunderstorm detected.
        (True, 'weather OK') otherwise.
    """
    precip: float = float(weather_dict.get("precip_prob", 0.0))
    conditions: str = str(weather_dict.get("conditions", "")).lower()

    if "thunder" in conditions:
        return False, "game likely postponed — thunderstorm"

    if precip > PRECIP_POSTPONE_THRESHOLD:
        return False, f"game likely postponed — precip > {PRECIP_POSTPONE_THRESHOLD}in"

    return True, "weather OK"


def check_line_movement(
    opening_line: float,
    current_line: float,
    threshold: float = LINE_MOVEMENT_THRESHOLD,
) -> tuple[bool, str]:
    """Check for adverse sharp line movement against our pick.

    Compares the implied probability shift from opening to current line.
    Movement away from our pick (i.e. line shortening = worse price) above
    the threshold is flagged as sharp action.

    Args:
        opening_line: Opening American odds, e.g. -110.
        current_line: Current American odds at time of check.
        threshold: Maximum allowable implied prob shift (default 0.05 = 5%).

    Returns:
        (False, 'sharp line movement against pick') if adverse movement exceeds threshold.
        (True, 'line stable') otherwise.
    """
    try:
        opening_prob = implied_prob_from_american(opening_line)
        current_prob = implied_prob_from_american(current_line)
    except (ZeroDivisionError, ValueError) as exc:
        logger.error("check_line_movement: invalid odds values: %s", exc)
        return True, "line stable"

    # Adverse movement = implied prob increased (line shortened / worse price for bettor)
    movement = current_prob - opening_prob

    if movement > threshold:
        return False, "sharp line movement against pick"

    return True, "line stable"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_preflight(
    game_pk: int,
    player_id: int,
    pitcher_id: int,
    prop_type: str,
    opening_line: float,
    current_line: float,
    weather: dict,
    mlb_client: "MLBClient",
    db: object,
) -> dict:
    """Run all applicable pre-flight checks and return full diagnostics.

    All checks are always run (no short-circuit) to provide complete visibility.
    K props additionally run the opener/bulk check.

    Args:
        game_pk: MLB game primary key.
        player_id: MLB player ID for the batter/prop subject.
        pitcher_id: MLB player ID for the starting pitcher.
        prop_type: Prop type string, e.g. 'strikeouts', 'hits', 'home_runs'.
        opening_line: Opening American odds.
        current_line: Current American odds.
        weather: Weather dict from WeatherClient.get_game_weather().
        mlb_client: MLBClient instance.
        db: Database connection/handle (passed to opener_bulk check).

    Returns:
        {
            'passed': bool,  # True only if ALL checks passed
            'checks': [
                {'name': str, 'passed': bool, 'reason': str},
                ...
            ]
        }
    """
    check_results: list[dict] = []

    def _record(name: str, result: tuple[bool, str]) -> None:
        passed, reason = result
        check_results.append({"name": name, "passed": passed, "reason": reason})

    # 1. Injury status
    _record("injury_status", check_injury_status(player_id, mlb_client))

    # 2. Lineup confirmed
    _record("lineup_confirmed", check_lineup_confirmed(game_pk, player_id, mlb_client))

    # 3. Pitcher confirmed
    _record("pitcher_confirmed", check_pitcher_confirmed(game_pk, pitcher_id, mlb_client))

    # 4. Weather playable
    _record("weather_playable", check_weather_playable(weather))

    # 5. Line movement
    _record("line_movement", check_line_movement(opening_line, current_line))

    # 6. Opener/bulk — K props only
    is_k_prop = "strikeout" in prop_type.lower() or prop_type.lower() in ("k", "ks", "so")
    if is_k_prop:
        _record("opener_bulk", check_opener_bulk(game_pk, pitcher_id, mlb_client, db))

    overall_passed = all(c["passed"] for c in check_results)

    return {
        "passed": overall_passed,
        "checks": check_results,
    }
