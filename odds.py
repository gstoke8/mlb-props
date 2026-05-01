"""
Odds API client for MLB player props betting model.

Wraps https://api.the-odds-api.com/v4 with:
- Credit-aware logging
- 5-minute event cache
- 2 calls/second rate limiter
- American-odds storage convention
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "baseball_mlb"
TARGET_BOOKS = [
    "draftkings", "fanduel", "betmgm", "espnbet",
]
DEFAULT_MARKETS = ["pitcher_strikeouts", "batter_home_runs"]
ALL_MARKETS = [
    "pitcher_strikeouts",
    "batter_home_runs",
    "batter_hits",
    "batter_total_bases",
]

_EVENT_CACHE_TTL_SECONDS = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Utility functions (module-level, no state)
# ---------------------------------------------------------------------------


def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal odds."""
    if odds > 0:
        return (odds / 100) + 1.0
    return (100 / abs(odds)) + 1.0


def decimal_to_american(odds: float) -> int:
    """Convert decimal odds to American odds."""
    if odds >= 2.0:
        return int(round((odds - 1) * 100))
    if odds <= 1.0:
        return -10000  # guard against invalid/zero-profit prices
    return int(round(-100 / (odds - 1)))


def implied_prob(odds: int) -> float:
    """No-vig implied probability from American odds (0–1 range)."""
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


# ---------------------------------------------------------------------------
# OddsClient
# ---------------------------------------------------------------------------


class OddsClient:
    """Client for The Odds API v4 player props endpoints."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("ODDS_API_KEY not configured")

        self._api_key = api_key
        self._session = requests.Session()
        self._credits_remaining: int | None = None

        # Rate limiting: max 2 calls/second
        self._min_interval = 0.5  # seconds between calls
        self._last_call_time: float = 0.0

        # Event list cache
        self._event_cache: list[dict] | None = None
        self._event_cache_expires_at: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        """Block until enough time has passed to respect the rate limit."""
        elapsed = time.monotonic() - self._last_call_time
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """
        Execute a GET request, handle rate limiting, log credits,
        and raise on 4xx/5xx errors.
        """
        self._throttle()
        url = f"{BASE_URL}{path}"
        base_params: dict[str, Any] = {"apiKey": self._api_key}
        if params:
            base_params.update(params)

        try:
            response = self._session.get(url, params=base_params, timeout=30)
        except requests.RequestException as exc:
            logger.error("HTTP request failed: %s", exc)
            raise
        finally:
            self._last_call_time = time.monotonic()

        self._parse_credits_header(response)

        if not response.ok:
            logger.error(
                "Odds API error %s for %s: %s",
                response.status_code,
                url,
                response.text,
            )
            response.raise_for_status()

        return response.json()

    def _parse_credits_header(self, response: requests.Response) -> None:
        """Extract and cache remaining credits from response headers."""
        raw = response.headers.get("x-requests-remaining")
        if raw is not None:
            try:
                self._credits_remaining = int(raw)
                logger.info("Odds API credits remaining: %d", self._credits_remaining)
            except ValueError:
                logger.warning("Could not parse x-requests-remaining: %r", raw)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_credits_remaining(self) -> int | None:
        """Return cached value of x-requests-remaining from the last response."""
        return self._credits_remaining

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def get_mlb_events(self, date_str: str | None = None) -> list[dict]:
        """
        Fetch today's MLB games from the Odds API.

        Args:
            date_str: Optional ISO date string 'YYYY-MM-DD'. When provided,
                      only events commencing on that date (UTC) are returned.

        Returns:
            List of dicts: {event_id, game_pk_hint, away_team, home_team,
                            commence_time_utc}
        """
        now = time.monotonic()
        if self._event_cache is not None and now < self._event_cache_expires_at:
            logger.debug("Returning cached MLB event list")
            events = self._event_cache
        else:
            data = self._get(f"/sports/{SPORT}/events")
            events = [self._parse_event(e) for e in data]
            self._event_cache = events
            self._event_cache_expires_at = now + _EVENT_CACHE_TTL_SECONDS
            logger.info("Fetched %d MLB events from Odds API", len(events))

        if date_str is None:
            return list(events)

        return [e for e in events if e["commence_time_utc"].startswith(date_str)]

    def _parse_event(self, raw: dict) -> dict:
        """Normalise a raw Odds API event object."""
        return {
            "event_id": raw["id"],
            "game_pk_hint": None,  # filled in by match_event_to_game_pk
            "away_team": raw.get("away_team", ""),
            "home_team": raw.get("home_team", ""),
            "commence_time_utc": raw.get("commence_time", ""),
        }

    def match_event_to_game_pk(
        self, event: dict, schedule: list[dict]
    ) -> int | None:
        """
        Match an Odds API event to an MLB Stats API game by team names and date.

        Args:
            event:    A dict returned by get_mlb_events().
            schedule: List of game dicts from the MLB Stats API schedule
                      endpoint. Each dict must contain at minimum:
                      {game_pk, away_team_name, home_team_name, game_date}

        Returns:
            game_pk integer if a match is found, else None.
        """
        event_date = event["commence_time_utc"][:10]  # 'YYYY-MM-DD' in UTC
        away = event["away_team"].lower()
        home = event["home_team"].lower()

        # Late-night games (≥8 PM ET) start after midnight UTC, so the Odds API
        # records them as the next UTC calendar day while the MLB schedule API uses
        # the ET calendar day.  Build the "ET-equivalent" date to check both.
        try:
            prev_date = (
                datetime.strptime(event_date, "%Y-%m-%d") - timedelta(days=1)
            ).strftime("%Y-%m-%d")
        except ValueError:
            prev_date = None

        for game in schedule:
            game_dt = (game.get("game_date") or game.get("game_time_utc", ""))[:10]
            if game_dt != event_date and game_dt != prev_date:
                continue
            sched_away = (game.get("away_team_name") or game.get("away_team", "")).lower()
            sched_home = (game.get("home_team_name") or game.get("home_team", "")).lower()
            if _team_names_match(away, sched_away) and _team_names_match(
                home, sched_home
            ):
                return game["game_pk"]

        logger.warning(
            "No schedule match for Odds event %s (%s @ %s on %s)",
            event["event_id"],
            away,
            home,
            event_date,
        )
        return None

    # ------------------------------------------------------------------
    # Props
    # ------------------------------------------------------------------

    def get_event_props(
        self,
        event_id: str,
        markets: list[str] | None = None,
    ) -> dict:
        """
        Fetch player prop lines for a single event (1 credit per call).

        Args:
            event_id: Odds API event identifier.
            markets:  List of market keys to request. Defaults to
                      ['pitcher_strikeouts', 'batter_home_runs'].

        Returns:
            {
                event_id,
                away_team,
                home_team,
                props: {
                    book_key: {
                        market_key: [
                            {player_name, pick, line, odds, implied_prob}
                        ]
                    }
                }
            }
        """
        if markets is None:
            markets = list(DEFAULT_MARKETS)

        markets_param = ",".join(markets)
        bookmakers_param = ",".join(TARGET_BOOKS)

        data = self._get(
            f"/sports/{SPORT}/events/{event_id}/odds",
            params={
                "regions": "us,us2",
                "markets": markets_param,
                "bookmakers": bookmakers_param,
                "oddsFormat": "decimal",
            },
        )

        return self._parse_props_response(data)

    def _parse_props_response(self, data: dict) -> dict:
        """
        Transform the raw Odds API odds object into the canonical props dict.
        Decimal odds are converted to American for storage.
        """
        result: dict = {
            "event_id": data.get("id", ""),
            "away_team": data.get("away_team", ""),
            "home_team": data.get("home_team", ""),
            "props": {},
        }

        for bookmaker in data.get("bookmakers", []):
            book_key = bookmaker["key"]
            if book_key not in TARGET_BOOKS:
                continue

            book_markets: dict[str, list[dict]] = {}

            for market in bookmaker.get("markets", []):
                market_key = market["key"]
                outcomes: list[dict] = []

                for outcome in market.get("outcomes", []):
                    decimal_price = outcome.get("price", 1.0)
                    american_odds = decimal_to_american(decimal_price)
                    outcomes.append(
                        {
                            "player_name": outcome.get("description", ""),
                            "pick": outcome.get("name", ""),
                            "line": outcome.get("point"),
                            "odds": american_odds,
                            "implied_prob": implied_prob(american_odds),
                        }
                    )

                book_markets[market_key] = outcomes

            result["props"][book_key] = book_markets

        return result

    def get_game_total(self, event_id: str) -> float | None:
        """Fetch the consensus game over/under total for an event.

        Returns the Over point value averaged across available books, or None.
        Uses market key 'totals'. Costs 1 API credit per call.
        """
        try:
            props = self.get_event_props(event_id, markets=["totals"])
        except Exception as exc:
            logger.warning("get_game_total: failed to fetch totals for %s: %s", event_id, exc)
            return None

        over_points: list[float] = []
        for book_markets in props.get("props", {}).values():
            for outcome in book_markets.get("totals", []):
                if outcome.get("pick") == "Over":
                    point = outcome.get("line")
                    if point is not None:
                        over_points.append(float(point))

        if not over_points:
            logger.debug("get_game_total: no totals market found for event %s", event_id)
            return None

        total = round(sum(over_points) / len(over_points), 2)
        logger.info("Game total for event %s: %.2f (avg of %d books)", event_id, total, len(over_points))
        return total

    def get_all_games_props(
        self,
        events: list[dict],
        markets: list[str] | None = None,
    ) -> dict[str, dict]:
        """
        Batch-fetch props for all events.

        Args:
            events:  List of event dicts returned by get_mlb_events().
            markets: Market keys to request for every event.

        Returns:
            Dict keyed by event_id, each value is the result of
            get_event_props() for that event.
        """
        results: dict[str, dict] = {}

        for idx, event in enumerate(events):
            event_id = event["event_id"]
            logger.info(
                "Fetching props for event %s (%d/%d): %s @ %s",
                event_id,
                idx + 1,
                len(events),
                event.get("away_team"),
                event.get("home_team"),
            )

            try:
                props = self.get_event_props(event_id, markets=markets)
                results[event_id] = props
                logger.info(
                    "Credits remaining after event %s: %s",
                    event_id,
                    self._credits_remaining,
                )
            except requests.HTTPError as exc:
                logger.error("Failed to fetch props for event %s: %s", event_id, exc)
                raise

        return results

    # ------------------------------------------------------------------
    # Line tracking
    # ------------------------------------------------------------------

    def get_opening_lines(
        self,
        event_id: str,
        markets: list[str] | None = None,
    ) -> dict:
        """
        Snapshot of current lines, labeled as the opening snapshot.

        Returns the same structure as get_event_props().
        """
        snapshot = self.get_event_props(event_id, markets=markets)
        return {**snapshot, "snapshot_type": "opening"}

    def capture_closing_lines(
        self,
        event_id: str,
        game_pk: int,
        markets: list[str] | None = None,
        db=None,
    ) -> int:
        """
        Fetch props and persist them to the closing_lines table.

        Args:
            event_id: Odds API event identifier.
            game_pk:  MLB Stats API game_pk for cross-referencing.
            markets:  Market keys to request.
            db:       Database connection/session object. Must expose an
                      execute(sql, params) interface. If None, lines are
                      fetched but not saved and a warning is logged.

        Returns:
            Number of individual lines saved.
        """
        snapshot = self.get_event_props(event_id, markets=markets)
        captured_at = datetime.now(timezone.utc).isoformat()

        if db is None:
            logger.warning(
                "capture_closing_lines called without db; %d books fetched but not saved",
                len(snapshot.get("props", {})),
            )
            return 0

        rows_saved = 0

        for book_key, book_data in snapshot.get("props", {}).items():
            for market_key, outcomes in book_data.items():
                for outcome in outcomes:
                    try:
                        db.execute(
                            """
                            INSERT INTO closing_lines
                                (event_id, game_pk, book, market, player_name,
                                 pick, line, odds, implied_prob, captured_at)
                            VALUES
                                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                event_id,
                                game_pk,
                                book_key,
                                market_key,
                                outcome["player_name"],
                                outcome["pick"],
                                outcome["line"],
                                outcome["odds"],
                                outcome["implied_prob"],
                                captured_at,
                            ),
                        )
                        rows_saved += 1
                    except Exception as exc:
                        logger.error(
                            "Failed to insert closing line for %s/%s/%s: %s",
                            book_key,
                            market_key,
                            outcome.get("player_name"),
                            exc,
                        )
                        raise

        logger.info(
            "Saved %d closing lines for event %s (game_pk=%d)",
            rows_saved,
            event_id,
            game_pk,
        )
        return rows_saved


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _team_names_match(a: str, b: str) -> bool:
    """
    Fuzzy team name comparison: returns True if one name contains the other
    (handles differences like 'New York Yankees' vs 'Yankees').
    """
    return a in b or b in a


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_client: OddsClient | None = None


def get_odds_client() -> OddsClient:
    """Return (and lazily create) the module-level OddsClient singleton."""
    global _client
    if _client is None:
        api_key = os.getenv("ODDS_API_KEY", "")
        _client = OddsClient(api_key)
    return _client
