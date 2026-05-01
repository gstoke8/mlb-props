#!/usr/bin/env python3
"""
MLB Props Scheduler

Runs once each morning (6 AM via launchd). Fetches today's game schedule,
groups games into start-time slots, and creates one launchd plist per slot
so the model fires at T-30 before each distinct wave of first pitches:

  Slot plists  — SLOT_LEAD_MINUTES before each cluster of first pitches
                 Each fires daily_runner.py --mode analysis --game-window HH:MM
                 where HH:MM is the slot's UTC start time (used to filter games)
  RESULTS RUN  — GAME_RESULTS_DELAY_HOURS after each individual game start

All schedule times are computed in UTC; launchd plist uses ET Hour/Minute.

Usage:
    python3 scheduler.py           # normal run
    python3 scheduler.py --dry-run # print schedule without writing plists
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

# Games within this many minutes of the FIRST game in the group are placed in
# the same slot.  E.g. with 30 min: games at 7:10, 7:30, and 8:10 produce two
# slots — [7:10, 7:30] and [8:10].
SLOT_GROUPING_MINUTES = 30
# Fire the model this many minutes before the slot's first pitch.
# 60 min: close enough for all prop markets (hits/HR/TB posted ~60 min out)
# while still giving confirmed lineups time to post (~T-60 as well).
SLOT_LEAD_MINUTES = 60
# Fire the per-game results runner this many hours after each game's start time.
# 4 hours covers a typical 3-hour game + 1 hour margin for extras/delays.
GAME_RESULTS_DELAY_HOURS = 4
GAME_DURATION_HOURS = 3.5

DRY_RUN = "--dry-run" in sys.argv

# Slot plists are named com.mlb-props.slot-HHMM (e.g. com.mlb-props.slot-1840)
SLOT_PLIST_PREFIX = "com.mlb-props.slot"
RESULTS_PLIST_LABEL = "com.mlb-props.results"
MORNING_PLIST_LABEL = "com.mlb-props.morning"
MORNING_SNAPSHOT_HOUR_ET = 9   # 9:00 AM ET daily — captures opening prop lines
LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
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
# Slot grouping
# ---------------------------------------------------------------------------


def group_games_by_slot(games: list[dict], threshold_minutes: int = SLOT_GROUPING_MINUTES) -> list[list[dict]]:
    """Cluster games into start-time slots.

    Games whose first pitches are within *threshold_minutes* of the FIRST game
    in the current group are placed in the same slot.  Returns a list of slots,
    where each slot is a list of game dicts sorted by game_time_utc.

    Example (threshold=30): games at 7:10, 7:40, 8:10, 9:20
    → three slots: [7:10/7:40], [8:10], [9:20]
    Run times: 6:10, 7:10, 8:20 (each T-60 before earliest in slot)
    """
    sorted_games = sorted(games, key=lambda g: g["game_time_utc"])
    slots: list[list[dict]] = []
    current_slot: list[dict] = []
    slot_anchor: datetime | None = None

    for game in sorted_games:
        t = game["game_time_utc"]
        if slot_anchor is None or (t - slot_anchor).total_seconds() > threshold_minutes * 60:
            if current_slot:
                slots.append(current_slot)
            current_slot = [game]
            slot_anchor = t
        else:
            current_slot.append(game)

    if current_slot:
        slots.append(current_slot)

    return slots


# ---------------------------------------------------------------------------
# Schedule computation
# ---------------------------------------------------------------------------


def compute_run_times(games: list[dict]) -> dict:
    """Derive per-slot run times and results time from *games*.

    Returns a dict with:
        slots        – list of {slot_utc, run_time, games}
                       slot_utc  – UTC datetime of the earliest game in the slot
                       run_time  – UTC datetime T-SLOT_LEAD_MINUTES before slot_utc
                       games     – list of game dicts in this slot
    """
    if not games:
        raise ValueError("Cannot compute run times from an empty games list.")

    slots_raw = group_games_by_slot(games)
    slot_entries = []
    for slot_games in slots_raw:
        slot_utc: datetime = slot_games[0]["game_time_utc"]
        run_time = slot_utc - timedelta(minutes=SLOT_LEAD_MINUTES)
        slot_entries.append({
            "slot_utc": slot_utc,
            "run_time": run_time,
            "games": slot_games,
        })

    return {
        "slots": slot_entries,
    }


# ---------------------------------------------------------------------------
# Schedule builder
# ---------------------------------------------------------------------------


def build_daily_schedule(game_date: date, mlb_client: Any) -> dict | None:
    """Fetch games and compute the full daily schedule.

    Returns None if there are no games on *game_date*.

    Returns a dict with:
        date         – str (ISO)
        game_count   – int
        slots        – list of {slot_utc, run_time, games}
        games        – flat list of all game dicts
    """
    games = get_todays_games(game_date, mlb_client)

    if not games:
        logger.info("No MLB games found for %s.", game_date.isoformat())
        return None

    run_times = compute_run_times(games)

    return {
        "date": game_date.isoformat(),
        "game_count": len(games),
        "slots": run_times["slots"],
        "games": games,
    }


# ---------------------------------------------------------------------------
# launchd plist helpers
# ---------------------------------------------------------------------------

def _plist_content(
    label: str,
    hour: int,
    minute: int,
    extra_args: list[str],
    log_suffix: str,
) -> str:
    """Return a launchd plist XML string for a single calendar-interval job."""
    odds_api_key = os.environ.get("ODDS_API_KEY", "")
    resend_api_key = os.environ.get("RESEND_API_KEY", "")
    visual_crossing_api_key = os.environ.get("VISUAL_CROSSING_API_KEY", "")
    mlb_bankroll = os.environ.get("MLB_BANKROLL", "1000")
    mlb_is_live = os.environ.get("MLB_IS_LIVE", "1")
    database_url = os.environ.get("DATABASE_URL", "")

    extra_str = "\n".join(f"        <string>{a}</string>" for a in extra_args)
    return f"""\
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
        <string>{RUNNER_PATH}</string>
{extra_str}
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
        <key>MLB_BANKROLL</key>
        <string>{mlb_bankroll}</string>
        <key>MLB_IS_LIVE</key>
        <string>{mlb_is_live}</string>
        <key>DATABASE_URL</key>
        <string>{database_url}</string>
    </dict>

    <key>WorkingDirectory</key>
    <string>{Path.home() / "mlb-props"}</string>

    <key>StandardOutPath</key>
    <string>{LOG_DIR / f"{log_suffix}.log"}</string>

    <key>StandardErrorPath</key>
    <string>{LOG_DIR / f"{log_suffix}_err.log"}</string>

    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
"""


def _launchctl(action: str, plist_path: Path) -> tuple[bool, str]:
    """Run launchctl load/unload; return (success, output)."""
    try:
        result = subprocess.run(
            ["launchctl", action, str(plist_path)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0:
            return (True, result.stdout)
        return (False, result.stderr.strip())
    except OSError as exc:
        return (False, str(exc))


def _launchctl_remove(label: str) -> None:
    """Force-remove a launchd job by label (fallback when file-based unload fails)."""
    try:
        subprocess.run(
            ["launchctl", "remove", label],
            capture_output=True, text=True, check=False,
        )
    except OSError:
        pass


def cleanup_old_slot_plists() -> None:
    """Unload and delete any slot plists left over from a previous run.

    If the file-based ``launchctl unload`` fails (e.g. the label is already
    loaded from a previous session whose plist was deleted), we fall back to
    ``launchctl remove <label>`` which removes the job from launchd by name
    regardless of whether the backing file still exists.  This prevents stale
    StartCalendarInterval jobs from firing on subsequent days.
    """
    LAUNCH_AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    for p in LAUNCH_AGENTS_DIR.glob(f"{SLOT_PLIST_PREFIX}-*.plist"):
        ok, msg = _launchctl("unload", p)
        if not ok:
            label = p.stem  # filename without .plist extension = label
            logger.warning(
                "unload failed for %s (%s) — falling back to launchctl remove %s",
                p.name, msg, label,
            )
            _launchctl_remove(label)
        p.unlink(missing_ok=True)
        logger.info("Removed old slot plist: %s", p.name)


def cleanup_old_results_plists() -> None:
    """Unload and delete all results and lines-retry plists from previous runs.

    Handles:
    - Legacy single results plist (com.mlb-props.results.plist)
    - Per-game results plists (com.mlb-props.results-{game_pk}.plist)
    - Lines-retry plists (com.mlb-props.lines-retry-*.plist)
    """
    LAUNCH_AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    candidates = (
        list(LAUNCH_AGENTS_DIR.glob(f"{RESULTS_PLIST_LABEL}*.plist"))
        + list(LAUNCH_AGENTS_DIR.glob("com.mlb-props.lines-retry-*.plist"))
    )
    for plist_path in candidates:
        label = plist_path.stem
        ok, msg = _launchctl("unload", plist_path)
        if not ok:
            logger.warning(
                "unload failed for %s (%s) — falling back to launchctl remove %s",
                plist_path.name, msg, label,
            )
            _launchctl_remove(label)
        plist_path.unlink(missing_ok=True)
        logger.info("Removed old results/retry plist: %s", plist_path.name)


def write_and_load_slot_plist(slot: dict) -> Path:
    """Write and load a launchd plist for one game slot.

    Fires daily_runner.py --mode analysis --game-window HH:MM
    where HH:MM is the slot's UTC start time.
    """
    run_et: datetime = slot["run_time"].astimezone(ET)
    slot_utc: datetime = slot["slot_utc"]
    slot_label = slot_utc.strftime("%H%M")
    window_str = slot_utc.strftime("%H:%M")

    label = f"{SLOT_PLIST_PREFIX}-{slot_label}"
    plist_path = LAUNCH_AGENTS_DIR / f"{label}.plist"

    game_pks_str = ",".join(str(g["game_pk"]) for g in slot["games"] if g.get("game_pk"))
    extra_args = ["--mode", "analysis", "--game-pks", game_pks_str]
    plist_path.write_text(
        _plist_content(label, run_et.hour, run_et.minute, extra_args, f"slot-{slot_label}"),
        encoding="utf-8",
    )

    ok, msg = _launchctl("load", plist_path)
    game_list = ", ".join(f"{g['away_team']} @ {g['home_team']}" for g in slot["games"])
    if ok:
        logger.info(
            "Loaded %s — fires %s ET | game_pks: %s | games: %s",
            label, run_et.strftime("%H:%M"), game_pks_str, game_list,
        )
    else:
        logger.error("Failed to load %s: %s", label, msg)

    return plist_path


def write_and_load_morning_plist() -> Path:
    """Write and load a launchd plist for the 9 AM morning snapshot.

    Fires daily_runner.py --mode morning at 9:00 AM ET every day.
    Captures opening prop lines so line_movement can be computed at analysis time.
    """
    label = MORNING_PLIST_LABEL
    plist_path = LAUNCH_AGENTS_DIR / f"{label}.plist"

    extra_args = ["--mode", "morning"]
    plist_path.write_text(
        _plist_content(label, MORNING_SNAPSHOT_HOUR_ET, 0, extra_args, "morning"),
        encoding="utf-8",
    )

    ok, msg = _launchctl("load", plist_path)
    if ok:
        logger.info("Loaded %s — fires %02d:00 ET (morning opening lines snapshot)", label, MORNING_SNAPSHOT_HOUR_ET)
    else:
        logger.error("Failed to load %s: %s", label, msg)

    return plist_path


def write_and_load_per_game_results_plist(game: dict) -> Path:
    """Write and load a launchd plist to resolve results for one game.

    Fires daily_runner.py --mode results --game-pks <game_pk>
    at game_time_utc + GAME_RESULTS_DELAY_HOURS.  One plist per game
    so results arrive incrementally throughout the evening.
    """
    game_pk: int = game["game_pk"]
    fire_time_utc: datetime = game["game_time_utc"] + timedelta(hours=GAME_RESULTS_DELAY_HOURS)
    fire_time_et: datetime = fire_time_utc.astimezone(ET)
    matchup = f"{game.get('away_team', '?')} @ {game.get('home_team', '?')}"

    label = f"{RESULTS_PLIST_LABEL}-{game_pk}"
    plist_path = LAUNCH_AGENTS_DIR / f"{label}.plist"

    extra_args = ["--mode", "results", "--game-pks", str(game_pk)]
    plist_path.write_text(
        _plist_content(
            label,
            fire_time_et.hour,
            fire_time_et.minute,
            extra_args,
            f"results-{game_pk}",
        ),
        encoding="utf-8",
    )

    ok, msg = _launchctl("load", plist_path)
    if ok:
        logger.info(
            "Loaded %s — fires %s ET | game_pk=%s | %s",
            label, fire_time_et.strftime("%H:%M"), game_pk, matchup,
        )
    else:
        logger.error("Failed to load %s: %s", label, msg)

    return plist_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _run_previous_night_results(yesterday: date) -> None:
    """Run the results runner for yesterday inline at scheduler startup.

    This ensures last night's late games (whose dynamic results plist may have
    already fired, or may have been missed) are always resolved before today's
    slots are written.  Idempotent — already-resolved bets are skipped.
    """
    import subprocess as _sp

    yesterday_str = yesterday.isoformat()
    logger.info("Running results for previous night (%s) …", yesterday_str)
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "daily_runner.py"),
        "--mode", "results",
        "--date", yesterday_str,
    ]
    try:
        result = _sp.run(cmd, capture_output=True, text=True, timeout=120)
        # Echo the last few meaningful log lines so they appear in scheduler.log
        for line in (result.stdout + result.stderr).splitlines():
            if any(kw in line for kw in ("resolved", "P/L", "Results email", "unresolved", "ERROR")):
                logger.info("  [results] %s", line.strip())
        if result.returncode != 0:
            logger.warning("Results runner exited %d for %s", result.returncode, yesterday_str)
    except Exception as exc:
        logger.warning("Could not run results for %s: %s", yesterday_str, exc)


def main() -> None:
    today = date.today()
    yesterday = today - timedelta(days=1)
    mlb_client = get_client()

    # Pull last night's results before scheduling today's slots
    if not DRY_RUN:
        _run_previous_night_results(yesterday)

    schedule = build_daily_schedule(today, mlb_client)
    if schedule is None:
        logger.info("No games today (%s). Nothing scheduled.", today.isoformat())
        return

    slots = schedule["slots"]

    logger.info(
        "Schedule for %s: %d game(s) in %d slot(s)",
        schedule["date"],
        schedule["game_count"],
        len(slots),
    )
    for slot in slots:
        run_et = slot["run_time"].astimezone(ET)
        logger.info(
            "  Slot %s UTC → run at %s ET | %d game(s): %s",
            slot["slot_utc"].strftime("%H:%M"),
            run_et.strftime("%H:%M"),
            len(slot["games"]),
            ", ".join(f"{g['away_team']} @ {g['home_team']}" for g in slot["games"]),
        )

    if DRY_RUN:
        logger.info("Dry-run mode — skipping plist writes.")
        return

    # Clean up yesterday's slot and results plists, then write fresh ones.
    cleanup_old_slot_plists()
    cleanup_old_results_plists()

    for slot in slots:
        write_and_load_slot_plist(slot)

    # Per-game results plists: one per game, fires GAME_RESULTS_DELAY_HOURS after first pitch.
    games_with_pk = [g for g in schedule["games"] if g.get("game_pk")]
    for game in games_with_pk:
        write_and_load_per_game_results_plist(game)

    # Morning snapshot plist: always rewrite so env vars (API keys) stay current.
    morning_plist = LAUNCH_AGENTS_DIR / f"{MORNING_PLIST_LABEL}.plist"
    if morning_plist.exists():
        ok, msg = _launchctl("unload", morning_plist)
        if not ok:
            _launchctl_remove(MORNING_PLIST_LABEL)
        morning_plist.unlink(missing_ok=True)
    write_and_load_morning_plist()
    logger.info("Morning snapshot plist written/reloaded (9:00 AM ET).")

    logger.info(
        "Scheduled %d slot run(s) and %d per-game results run(s) for today.",
        len(slots),
        len(games_with_pk),
    )


if __name__ == "__main__":
    main()
