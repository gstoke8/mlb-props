"""
umpires.py — Umpire data module for MLB player props model.

Responsibilities:
  1. Historical K-factor table — built from UmpScorecards API, cached in SQLite.
  2. Day-of umpire assignment — MLB Stats API primary, Action Network fallback.
  3. K-factor lookup with name normalization.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import statistics
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Generator

import requests

from db import get_db
from mlb_api import get_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_UMPSCORECARDS_API = "https://umpscorecards.com/api/umpires"
_ACTION_NETWORK_URL = "https://www.actionnetwork.com/mlb/referee-assignments"

_CACHE_DIR = Path.home() / "mlb-props" / ".cache"
_CACHE_FILE = _CACHE_DIR / "umpire_cache.json"

_CACHE_TTL_DAYS = 7

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
_SCRAPE_HEADERS = {"User-Agent": _USER_AGENT}

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

_CREATE_FACTORS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS umpire_factors (
    umpire_name          TEXT PRIMARY KEY,
    k_factor             REAL NOT NULL,
    games                INTEGER NOT NULL,
    accuracy_wmean       REAL,
    wrong_calls_per_game REAL,
    updated_at           TEXT NOT NULL
)
"""

_CREATE_META_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS umpire_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
"""

_UPSERT_FACTOR_SQL = """
INSERT INTO umpire_factors
    (umpire_name, k_factor, games, accuracy_wmean, wrong_calls_per_game, updated_at)
VALUES
    (:umpire_name, :k_factor, :games, :accuracy_wmean, :wrong_calls_per_game, :updated_at)
ON CONFLICT(umpire_name) DO UPDATE SET
    k_factor             = excluded.k_factor,
    games                = excluded.games,
    accuracy_wmean       = excluded.accuracy_wmean,
    wrong_calls_per_game = excluded.wrong_calls_per_game,
    updated_at           = excluded.updated_at
"""

_UPSERT_META_SQL = """
INSERT INTO umpire_meta (key, value)
VALUES (:key, :value)
ON CONFLICT(key) DO UPDATE SET value = excluded.value
"""


@contextmanager
def _open_db(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """
    Open a WAL-mode SQLite connection, commit on success, rollback on error.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _ensure_schema(db_path: Path) -> None:
    """Create umpire tables if they don't exist."""
    with _open_db(db_path) as conn:
        conn.execute(_CREATE_FACTORS_TABLE_SQL)
        conn.execute(_CREATE_META_TABLE_SQL)


def _get_db_path() -> Path:
    """Return the SQLite file path from the db module singleton."""
    return get_db().db_path


# ---------------------------------------------------------------------------
# K-factor derivation from UmpScorecards data
# ---------------------------------------------------------------------------

def _derive_k_factor(row: dict[str, Any], league_wrong_per_game: float) -> float:
    """
    Derive a K-factor from UmpScorecards aggregate stats.

    The K-factor represents how much this umpire's missed-call rate deviates
    from the league average.  Missed calls (balls called strikes / vice versa)
    correlate with expanded/contracted strike zones, affecting strikeout and
    walk rates.  A value of 1.15 means ~15 % more missed calls than average.

    Formula:
        wrong_per_game = called_wrong_sum / n
        k_factor       = wrong_per_game / league_avg_wrong_per_game

    Clamped to [0.5, 2.0] to prevent extreme values from small sample sizes.
    """
    n = row.get("n") or 0
    if n == 0:
        return 1.0

    wrong = row.get("called_wrong_sum") or 0
    wrong_per_game = wrong / n

    if league_wrong_per_game <= 0:
        return 1.0

    raw = wrong_per_game / league_wrong_per_game
    return max(0.5, min(2.0, raw))


def _fetch_umpscorecards() -> list[dict[str, Any]]:
    """
    Fetch umpire aggregate stats from UmpScorecards API.

    Returns list of raw row dicts.
    Raises requests.RequestException on HTTP failure.
    Raises ValueError on unexpected response shape.
    """
    response = requests.get(_UMPSCORECARDS_API, headers=_SCRAPE_HEADERS, timeout=20)
    response.raise_for_status()
    payload: Any = response.json()

    # The endpoint returns {"rows": [...]}
    rows: list[dict[str, Any]] = (
        payload.get("rows", payload) if isinstance(payload, dict) else payload
    )
    if not isinstance(rows, list):
        raise ValueError(
            f"Unexpected UmpScorecards response shape: {type(payload)}"
        )
    return rows


def _compute_league_average(rows: list[dict[str, Any]]) -> float:
    """Return the league-average wrong-calls-per-game across all umpires."""
    per_game = [
        row["called_wrong_sum"] / row["n"]
        for row in rows
        if (row.get("n") or 0) > 0 and row.get("called_wrong_sum") is not None
    ]
    if not per_game:
        return 1.0
    return statistics.mean(per_game)


# ---------------------------------------------------------------------------
# Part 1: refresh_umpire_factors
# ---------------------------------------------------------------------------

def refresh_umpire_factors(db: Any, force: bool = False) -> int:
    """
    Scrape umpscorecards.com and update umpire_factors table.

    Returns number of umpires updated.
    Only re-scrapes if cache is >7 days old or force=True.

    Args:
        db:    MLBPropsDB instance returned by get_db().
        force: Skip the age check and always re-scrape.

    Returns:
        Number of umpire rows written to the database (0 if cache was fresh).
    """
    db_path = db.db_path
    _ensure_schema(db_path)

    if not force:
        with _open_db(db_path) as conn:
            row = conn.execute(
                "SELECT value FROM umpire_meta WHERE key = 'last_scraped'"
            ).fetchone()
        if row:
            last_scraped = datetime.fromisoformat(row[0])
            age_days = (datetime.now(timezone.utc) - last_scraped).days
            if age_days < _CACHE_TTL_DAYS:
                logger.debug(
                    "umpire_factors cache is %d days old — skipping re-scrape",
                    age_days,
                )
                return 0

    try:
        rows = _fetch_umpscorecards()
    except requests.RequestException as exc:
        logger.warning(
            "UmpScorecards unreachable (%s) — continuing with cached data", exc
        )
        return 0

    league_avg = _compute_league_average(rows)
    now_iso = datetime.now(timezone.utc).isoformat()
    updated = 0

    with _open_db(db_path) as conn:
        for row in rows:
            name = (row.get("umpire") or "").strip()
            if not name:
                continue

            n = row.get("n") or 0
            k_factor = _derive_k_factor(row, league_avg)
            wrong_per_game = (
                (row.get("called_wrong_sum") or 0) / n if n > 0 else None
            )
            accuracy = row.get("overall_accuracy_wmean")

            conn.execute(
                _UPSERT_FACTOR_SQL,
                {
                    "umpire_name": name,
                    "k_factor": k_factor,
                    "games": n,
                    "accuracy_wmean": accuracy,
                    "wrong_calls_per_game": wrong_per_game,
                    "updated_at": now_iso,
                },
            )
            updated += 1

        conn.execute(
            _UPSERT_META_SQL, {"key": "last_scraped", "value": now_iso}
        )

    logger.info("Refreshed %d umpire factors from UmpScorecards", updated)
    return updated


# ---------------------------------------------------------------------------
# Part 3: Name normalization + K-factor lookup
# ---------------------------------------------------------------------------

def normalize_umpire_name(name: str) -> str:
    """
    Normalize an umpire name for fuzzy matching.

    Transformations applied:
      - Lowercase and strip surrounding whitespace
      - Remove name suffixes: Jr, Sr, II, III, IV
      - Remove middle initials (single letter + period between words)
      - Collapse dots (e.g. "C.B." -> "CB")
      - Collapse multiple spaces

    Examples::

        normalize_umpire_name("Angel Hernandez Jr.") -> "angel hernandez"
        normalize_umpire_name("C.B. Bucknor")        -> "cb bucknor"
        normalize_umpire_name("Joe W. West")          -> "joe west"
    """
    if not name:
        return ""

    normalized = name.lower().strip()

    # Remove trailing suffixes: Jr, Sr, II, III, IV (with optional period)
    normalized = re.sub(r"\b(jr\.?|sr\.?|ii|iii|iv)\b\.?", "", normalized)

    # Remove middle initials: single letter followed by period, surrounded by spaces
    normalized = re.sub(r"(?<=\s)[a-z]\.", "", normalized)

    # Collapse remaining dots (handles "c.b." -> "cb")
    normalized = re.sub(r"\.", "", normalized)

    # Collapse multiple whitespace characters
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def get_k_factor(db: Any, umpire_name: str) -> float:
    """
    Look up umpire K-factor from the umpire_factors table.

    Tries exact match first, then falls back to normalized-name matching.

    Args:
        db:           MLBPropsDB instance returned by get_db().
        umpire_name:  Umpire name as received (any casing/formatting).

    Returns:
        Float k_factor, e.g. 1.15 = 15 % more missed calls than average.
        Returns 1.0 (neutral) if the umpire is not found in the database.
    """
    db_path = db.db_path
    _ensure_schema(db_path)

    with _open_db(db_path) as conn:
        # 1. Exact match
        row = conn.execute(
            "SELECT k_factor FROM umpire_factors WHERE umpire_name = ?",
            (umpire_name,),
        ).fetchone()
        if row:
            return float(row[0])

        # 2. Normalize and scan all stored names
        target = normalize_umpire_name(umpire_name)
        if not target:
            return 1.0

        all_rows = conn.execute(
            "SELECT umpire_name, k_factor FROM umpire_factors"
        ).fetchall()

    for stored_name, k_factor in all_rows:
        if normalize_umpire_name(stored_name) == target:
            return float(k_factor)

    logger.debug(
        "Umpire '%s' not found in umpire_factors — using neutral 1.0", umpire_name
    )
    return 1.0


# ---------------------------------------------------------------------------
# Part 2a: MLB Stats API umpire lookup
# ---------------------------------------------------------------------------

def _get_umpire_from_mlb_client(
    mlb_client: Any, game_pk: int
) -> dict[str, Any] | None:
    """
    Fetch home-plate umpire for a game via the MLBClient wrapper.

    Uses ``mlb_client.get_home_plate_umpire(game_pk)`` which queries the
    live feed endpoint (``/game/{gamePk}/feed/live``).

    Returns:
        Dict with keys ``'umpire_id'`` and ``'name'``, or ``None`` if
        the umpire has not yet been assigned.
    """
    try:
        result = mlb_client.get_home_plate_umpire(game_pk)
    except Exception as exc:
        logger.warning(
            "MLBClient.get_home_plate_umpire failed for game %d: %s", game_pk, exc
        )
        return None

    if not result:
        return None

    uid = result.get("umpire_id")
    name = result.get("name", "")
    if not name:
        return None

    return {"umpire_id": uid, "name": name}


# ---------------------------------------------------------------------------
# Part 2b: Action Network fallback
# ---------------------------------------------------------------------------

def _fetch_action_network_games() -> list[dict[str, Any]]:
    """
    Scrape Action Network referee-assignments page.

    Action Network embeds all data in a Next.js ``__NEXT_DATA__`` JSON blob.
    Each game object contains an ``'officials'`` list with position and name.

    Returns:
        List of game dicts, each containing ``'teams'`` and ``'officials'``.

    Raises:
        requests.RequestException: On HTTP failure.
        ValueError: If the ``__NEXT_DATA__`` blob is absent or malformed.
    """
    response = requests.get(
        _ACTION_NETWORK_URL, headers=_SCRAPE_HEADERS, timeout=20
    )
    response.raise_for_status()

    match = re.search(
        r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
        response.text,
        re.S,
    )
    if not match:
        raise ValueError(
            "Could not locate __NEXT_DATA__ in Action Network page"
        )

    try:
        next_data: dict[str, Any] = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse __NEXT_DATA__ JSON: {exc}"
        ) from exc

    games: list[dict[str, Any]] = (
        next_data.get("props", {})
        .get("pageProps", {})
        .get("officials", [])
    )
    return games


def _build_team_abbr_to_game_pk(
    mlb_client: Any, game_pks: list[int], target_date: date
) -> dict[str, int]:
    """
    Build a map of team abbreviation (lower) -> game_pk for the given date.

    Uses the MLB Stats API schedule endpoint via the mlb_client.
    Returns an empty dict if the request fails.
    """
    schedule = mlb_client.get_schedule(target_date.isoformat())
    pk_set = set(game_pks)
    mapping: dict[str, int] = {}
    for game in schedule:
        pk = game.get("game_pk")
        if pk not in pk_set:
            continue
        # game dict keys: away_team, away_id, home_team, home_id
        # Team abbreviations aren't in the schedule response; use IDs to
        # build the map via the teams endpoint.
        away_id = game.get("away_id")
        home_id = game.get("home_id")
        for team_id in (away_id, home_id):
            if team_id:
                mapping[str(team_id)] = pk
    return mapping


def _get_umpires_from_action_network(
    mlb_client: Any,
    game_pks: list[int],
    target_date: date,
) -> dict[int, dict[str, Any]]:
    """
    Scrape Action Network for home-plate umpire assignments.

    Matches Action Network games (which carry team abbreviations) to game_pks
    by cross-referencing with the MLB Stats API teams list.

    Returns:
        Dict mapping ``game_pk`` -> ``{'umpire_id': int|None, 'name': str}``.
        Partial dict — only games successfully matched are included.
    """
    try:
        an_games = _fetch_action_network_games()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Action Network scrape failed: %s", exc)
        return {}

    # Build abbr (lower) -> game_pk using the MLB teams list + schedule
    schedule = mlb_client.get_schedule(target_date.isoformat())
    pk_set = set(game_pks)

    # Map away_id/home_id -> game_pk from the schedule
    mlb_id_to_pk: dict[int, int] = {}
    for game in schedule:
        pk = game.get("game_pk")
        if pk not in pk_set:
            continue
        for key in ("away_id", "home_id"):
            tid = game.get(key)
            if tid:
                mlb_id_to_pk[tid] = pk

    # Map team abbreviation (lower) -> MLB team id from the teams list
    all_teams = mlb_client.get_teams()
    abbr_to_mlb_id: dict[str, int] = {
        t["abbreviation"].lower(): t["id"]
        for t in all_teams
        if t.get("abbreviation") and t.get("id")
    }

    results: dict[int, dict[str, Any]] = {}

    for an_game in an_games:
        teams: list[dict[str, Any]] = an_game.get("teams", [])
        officials: list[dict[str, Any]] = an_game.get("officials", [])

        hp_umpire: dict[str, Any] | None = next(
            (o for o in officials if o.get("position") == "HP"), None
        )
        if not hp_umpire:
            continue

        matched_pk: int | None = None
        for team in teams:
            abbr = (team.get("abbr") or "").lower()
            mlb_id = abbr_to_mlb_id.get(abbr)
            if mlb_id and mlb_id in mlb_id_to_pk:
                matched_pk = mlb_id_to_pk[mlb_id]
                break

        if matched_pk is not None and matched_pk not in results:
            results[matched_pk] = {
                "umpire_id": hp_umpire.get("id"),
                "name": hp_umpire.get("full_name") or "",
            }

    return results


# ---------------------------------------------------------------------------
# Part 2: get_todays_umpires
# ---------------------------------------------------------------------------

def get_todays_umpires(
    mlb_client: Any,
    game_pks: list[int],
) -> dict[int, dict[str, Any]]:
    """
    Get home-plate umpire for each game_pk, with K-factor attached.

    Primary source: MLB Stats API live feed via ``MLBClient``
    (umpire populated ~T-90 min to T-2 hrs before first pitch).

    Fallback: Action Network ``referee-assignments`` page.

    Args:
        mlb_client: ``MLBClient`` instance returned by ``get_client()``.
        game_pks:   List of MLB game primary keys.

    Returns:
        Dict mapping ``game_pk`` -> ::

            {
                'umpire_id': int | None,
                'name': str,
                'k_factor': float,   # 1.0 = neutral; >1.0 = more missed calls
            }

        All requested ``game_pks`` are guaranteed to appear in the output
        (fallback value: ``{'umpire_id': None, 'name': '', 'k_factor': 1.0}``).
    """
    db = get_db()
    target_date = datetime.now(timezone.utc).date()

    results: dict[int, dict[str, Any]] = {}
    missing_pks: list[int] = []

    # --- Primary: MLB Stats API ---
    for pk in game_pks:
        ump = _get_umpire_from_mlb_client(mlb_client, pk)
        if ump and ump.get("name"):
            results[pk] = {
                "umpire_id": ump["umpire_id"],
                "name": ump["name"],
                "k_factor": get_k_factor(db, ump["name"]),
            }
        else:
            missing_pks.append(pk)

    if not missing_pks:
        return results

    logger.info(
        "MLB API missing umpires for %d game(s) — trying Action Network fallback",
        len(missing_pks),
    )

    # --- Fallback: Action Network ---
    an_results = _get_umpires_from_action_network(
        mlb_client, missing_pks, target_date
    )

    for pk in missing_pks:
        an = an_results.get(pk)
        if an and an.get("name"):
            results[pk] = {
                "umpire_id": an.get("umpire_id"),
                "name": an["name"],
                "k_factor": get_k_factor(db, an["name"]),
            }
        else:
            logger.warning(
                "No umpire found for game_pk=%d from any source — using neutral", pk
            )
            results[pk] = {
                "umpire_id": None,
                "name": "",
                "k_factor": 1.0,
            }

    return results


# ---------------------------------------------------------------------------
# JSON cache helpers
# ---------------------------------------------------------------------------

def _load_json_cache() -> dict[str, Any]:
    """Load the JSON cache file; return empty dict if absent or corrupt."""
    if not _CACHE_FILE.exists():
        return {}
    try:
        return json.loads(_CACHE_FILE.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read umpire cache file: %s", exc)
        return {}


def _save_json_cache(data: dict[str, Any]) -> None:
    """Write data to the JSON cache file, creating parent dirs as needed."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(data, indent=2))
    except OSError as exc:
        logger.warning("Could not write umpire cache file: %s", exc)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_umpire_data(
    game_pks: list[int],
    target_date: date | None = None,
) -> dict[int, dict[str, Any]]:
    """
    Main entry point: return enriched umpire data for each game on the given date.

    Workflow:
      1. Ensure umpire K-factors are up-to-date (refresh if >7 days old).
      2. Fetch today's home-plate umpire for each ``game_pk``.
      3. Attach K-factor to each result.
      4. Persist results to the JSON cache file.

    Args:
        game_pks:    List of MLB game primary keys for the day.
        target_date: Date the games are played (defaults to today UTC).
                     Used for cache keying and schedule lookups.

    Returns:
        Dict mapping ``game_pk`` -> ::

            {
                'umpire_id': int | None,
                'name': str,
                'k_factor': float,
            }
    """
    resolved_date = target_date or datetime.now(timezone.utc).date()
    logger.info(
        "Fetching umpire data for %d game(s) on %s", len(game_pks), resolved_date
    )

    db = get_db()
    mlb_client = get_client()

    # Refresh the K-factor table if stale (silently tolerates failure)
    try:
        refresh_umpire_factors(db)
    except Exception as exc:
        logger.warning("refresh_umpire_factors raised unexpectedly: %s", exc)

    umpire_data = get_todays_umpires(mlb_client, game_pks)

    # Persist a date-keyed snapshot to the JSON cache for lightweight access
    cache = _load_json_cache()
    cache[resolved_date.isoformat()] = {
        str(pk): info for pk, info in umpire_data.items()
    }
    _save_json_cache(cache)

    return umpire_data
