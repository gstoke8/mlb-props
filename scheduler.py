#!/usr/bin/env python3
"""
MLB Props Scheduler

Runs once each morning. Fetches today's game schedule and computes three
run windows based on actual first-pitch times:

  1. ANALYSIS RUN  — PRE_GAME_LEAD_MINUTES before the first first-pitch of the day
  2. CLOSING LINES — CLOSING_LINE_LEAD_MINUTES before each individual first pitch
  3. RESULTS RUN   — RESULTS_DELAY_MINUTES after the expected last game end
                     (last game start + 3.5 hours)

All schedule times are computed in UTC; launchd plist uses ET Hour/Minute.

Usage:
    python3 scheduler.py           # normal run
    python3 scheduler.py --dry-run # print schedule without writing plist
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytz

from mlb_api import get_client

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ET = pytz.timezone("America/New_York")

PRE_GAME_LEAD_MINUTES = 90
CLOSING_LINE_LEAD_MINUTES = 10
RESULTS_DELAY_MINUTES = 30
GAME_DURATION_HOURS = 3.5

DRY_RUN = "--dry-run" in sys.argv

PLIST_LABEL = "com.mlb-props.daily"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{PLIST_LABEL}.plist"
RUNNER_PATH = Path.home() / "mlb-props" / "daily_runner.py"
LOG_DIR = Path.home() / "mlb-props" / "logs"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "scheduler.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Team timezone map
# ---------------------------------------------------------------------------

TEAM_TZ: dict[str, str] = {
    # AL East
    "Yankees": "America/New_York",
    "Red Sox": "America/New_York",
    "Blue Jays": "America/Toronto",
    "Rays": "America/New_York",
    "Orioles": "America/New_York",
    # AL Central
    "White Sox": "America/Chicago",
    "Indians": "America/New_York",   # historical name kept for data compatibility
    "Guardians": "America/New_York",
    "Tigers": "America/Detroit",
    "Royals": "America/Chicago",
    "Twins": "America/Chicago",
    # AL West
    "Astros": "America/Chicago",
    "Angels": "America/Los_Angeles",
    "Athletics": "America/Los_Angeles",
    "Mariners": "America/Los_Angeles",
    "Rangers": "America/Chicago",
    # NL East
    "Mets": "America/New_York",
    "Braves": "America/New_York",
    "Phillies": "America/New_York",
    "Marlins": "America/New_York",
    "Nationals": "America/New_York",
    # NL Central
    "Cubs": "America/Chicago",
    "Cardinals": "America/Chicago",
    "Brewers": "America/Chicago",
    "Reds": "America/New_York",
    "Pirates": "America/New_York",
    # NL West
    "Dodgers": "America/Los_Angeles",
    "Giants": "America/Los_Angeles",
    "Padres": "America/Los_Angeles",
    "Rockies": "America/Denver",
    "Diamondbacks": "America/Phoenix",
}

# ---------------------------------------------------------------------------
# Game fetching
# ---------------------------------------------------------------------------


def get_todays_games(game_date: date, mlb_client: Any) -> list[dict]:
    """Return a list of game dicts for *game_date* from the MLB API client.

    Each dict contains:
        game_pk        – int, unique MLB game identifier
        home_team      – str
        away_team      – str
        venue          – str
        game_time_utc  – datetime (UTC-aware)
        status         – str  (e.g. 'Scheduled', 'In Progress', 'Final')
    """
    raw_games = mlb_client.get_schedule(game_date)

    def _parse(g: dict) -> dict:
        raw_time = g.get("game_time_utc") or g.get("gameDate") or g.get("game_date")
        if isinstance(raw_time, datetime):
            game_time_utc = (
                raw_time
                if raw_time.tzinfo is not None
                else pytz.utc.localize(raw_time)
            )
        else:
            game_time_utc = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
            if game_time_utc.tzinfo is None:
                game_time_utc = pytz.utc.localize(game_time_utc)

        return {
            "game_pk": g.get("game_pk") or g.get("gamePk"),
            "home_team": g.get("home_team") or g.get("homeTeam"),
            "away_team": g.get("away_team") or g.get("awayTeam"),
            "venue": g.get("venue") or g.get("venueName", ""),
            "game_time_utc": game_time_utc,
            "status": g.get("status") or g.get("abstractGameState", "Scheduled"),
        }

    return [_parse(g) for g in raw_games]


# ---------------------------------------------------------------------------
# Schedule computation
# ---------------------------------------------------------------------------


def compute_run_times(games: list[dict]) -> dict:
    """Derive analysis, closing-line, and results run times from *games*.

    Returns a dict with:
        analysis_time       – datetime (UTC) — first pitch minus PRE_GAME_LEAD_MINUTES
        closing_line_times  – list of (game_pk, datetime UTC)
        results_time        – datetime (UTC) — after last game's expected end
    """
    if not games:
        raise ValueError("Cannot compute run times from an empty games list.")

    sorted_games = sorted(games, key=lambda g: g["game_time_utc"])

    first_pitch: datetime = sorted_games[0]["game_time_utc"]
    last_pitch: datetime = sorted_games[-1]["game_time_utc"]

    analysis_time = first_pitch - timedelta(minutes=PRE_GAME_LEAD_MINUTES)

    closing_line_times = [
        (g["game_pk"], g["game_time_utc"] - timedelta(minutes=CLOSING_LINE_LEAD_MINUTES))
        for g in sorted_games
    ]

    expected_last_end = last_pitch + timedelta(hours=GAME_DURATION_HOURS)
    results_time = expected_last_end + timedelta(minutes=RESULTS_DELAY_MINUTES)

    return {
        "analysis_time": analysis_time,
        "closing_line_times": closing_line_times,
        "results_time": results_time,
    }


# ---------------------------------------------------------------------------
# Schedule builder
# ---------------------------------------------------------------------------


def build_daily_schedule(game_date: date, mlb_client: Any) -> dict | None:
    """Fetch games and compute the full daily schedule.

    Returns None if there are no games on *game_date*.

    Returns a dict with:
        date                – str (ISO)
        game_count          – int
        analysis_time       – datetime (UTC)
        closing_line_times  – list of (game_pk, datetime UTC)
        results_time        – datetime (UTC)
        games               – list of game dicts
    """
    games = get_todays_games(game_date, mlb_client)

    if not games:
        logger.info("No MLB games found for %s.", game_date.isoformat())
        return None

    run_times = compute_run_times(games)

    return {
        "date": game_date.isoformat(),
        "game_count": len(games),
        "analysis_time": run_times["analysis_time"],
        "closing_line_times": run_times["closing_line_times"],
        "results_time": run_times["results_time"],
        "games": games,
    }


# ---------------------------------------------------------------------------
# launchd plist helpers
# ---------------------------------------------------------------------------

_PLIST_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>

    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>{runner}</string>
        <string>--date</string>
        <string>TODAY</string>
    </array>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{hour}</integer>
        <key>Minute</key>
        <integer>{minute}</integer>
    </dict>

    <key>EnvironmentVariables</key>
    <dict>
        <key>ODDS_API_KEY</key>
        <string>{odds_api_key}</string>
        <key>RESEND_API_KEY</key>
        <string>{resend_api_key}</string>
        <key>VISUAL_CROSSING_API_KEY</key>
        <string>{visual_crossing_api_key}</string>
    </dict>

    <key>StandardOutPath</key>
    <string>{log_out}</string>

    <key>StandardErrorPath</key>
    <string>{log_err}</string>

    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
"""


def write_launchd_plist(schedule: dict, plist_path: Path = PLIST_PATH) -> Path:
    """Write a launchd plist that fires daily_runner.py at *schedule['analysis_time']*.

    The Hour/Minute in the plist use Eastern Time so macOS fires the job
    at the right wall-clock moment regardless of the system's local timezone.

    Returns the path of the written plist.
    """
    analysis_utc: datetime = schedule["analysis_time"]
    analysis_et = analysis_utc.astimezone(ET)

    odds_api_key = os.environ.get("ODDS_API_KEY", "")
    resend_api_key = os.environ.get("RESEND_API_KEY", "")
    visual_crossing_api_key = os.environ.get("VISUAL_CROSSING_API_KEY", "")

    if not odds_api_key:
        logger.warning("ODDS_API_KEY is not set in the environment.")
    if not resend_api_key:
        logger.warning("RESEND_API_KEY is not set in the environment.")
    if not visual_crossing_api_key:
        logger.warning("VISUAL_CROSSING_API_KEY is not set in the environment.")

    content = _PLIST_TEMPLATE.format(
        label=PLIST_LABEL,
        runner=str(RUNNER_PATH),
        hour=analysis_et.hour,
        minute=analysis_et.minute,
        odds_api_key=odds_api_key,
        resend_api_key=resend_api_key,
        visual_crossing_api_key=visual_crossing_api_key,
        log_out=str(LOG_DIR / "daily.log"),
        log_err=str(LOG_DIR / "daily_err.log"),
    )

    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(content, encoding="utf-8")
    logger.info("Plist written to %s (fires at %s ET).", plist_path, analysis_et.strftime("%H:%M"))
    return plist_path


def install_launchd(plist_path: Path = PLIST_PATH) -> tuple[bool, str]:
    """Load *plist_path* with launchctl.

    Returns (True, stdout) on success, (False, stderr) on failure.
    """
    try:
        result = subprocess.run(
            ["launchctl", "load", str(plist_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            logger.info("launchd job loaded: %s", PLIST_LABEL)
            return (True, result.stdout)
        logger.error("launchctl load failed: %s", result.stderr)
        return (False, result.stderr)
    except OSError as exc:
        logger.error("Failed to invoke launchctl: %s", exc)
        return (False, str(exc))


def uninstall_launchd(label: str = PLIST_LABEL) -> tuple[bool, str]:
    """Unload *label* with launchctl.

    Returns (True, stdout) on success, (False, stderr) on failure.
    """
    try:
        result = subprocess.run(
            ["launchctl", "unload", str(PLIST_PATH)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            logger.info("launchd job unloaded: %s", label)
            return (True, result.stdout)
        logger.error("launchctl unload failed: %s", result.stderr)
        return (False, result.stderr)
    except OSError as exc:
        logger.error("Failed to invoke launchctl: %s", exc)
        return (False, str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    today = date.today()
    mlb_client = get_client()

    schedule = build_daily_schedule(today, mlb_client)
    if schedule is None:
        logger.info("No games today (%s). Nothing scheduled.", today.isoformat())
        return

    logger.info(
        "Built schedule for %s: %d game(s) | analysis=%s UTC | results=%s UTC",
        schedule["date"],
        schedule["game_count"],
        schedule["analysis_time"].strftime("%H:%M"),
        schedule["results_time"].strftime("%H:%M"),
    )

    if DRY_RUN:
        logger.info("Dry-run mode — skipping plist write and launchctl load.")
        for game in schedule["games"]:
            logger.info(
                "  game_pk=%s  %s vs %s  first_pitch=%s UTC",
                game["game_pk"],
                game["away_team"],
                game["home_team"],
                game["game_time_utc"].strftime("%H:%M"),
            )
        return

    plist_path = write_launchd_plist(schedule)
    success, output = install_launchd(plist_path)
    if not success:
        raise RuntimeError(f"Failed to install launchd job: {output}")


if __name__ == "__main__":
    main()
