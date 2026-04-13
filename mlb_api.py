"""
MLB Stats API client for player props betting model.

Wraps https://statsapi.mlb.com/api/v1 — no API key required.
All methods return empty dicts/lists on 404 or missing data (never raise).
"""
from __future__ import annotations

import time
import logging
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

BASE_URL = "https://statsapi.mlb.com/api/v1"
REQUEST_TIMEOUT = 10
BATCH_DELAY = 0.5
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.0


def _build_session() -> requests.Session:
    """Build a requests.Session with retry logic on 5xx errors."""
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class MLBClient:
    """Client for the MLB Stats API.

    Usage::

        client = get_client()
        games = client.get_schedule("2025-04-13")
    """

    def __init__(self) -> None:
        self._session = _build_session()
        self._venue_cache: dict[int, dict] = {}
        self._teams_cache: list[dict] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        """Perform a GET request, returning parsed JSON or {} on error."""
        url = f"{BASE_URL}{path}"
        try:
            response = self._session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 404:
                logger.debug("404 for %s params=%s", url, params)
                return {}
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as exc:
            logger.warning("HTTP error for %s: %s", url, exc)
            return {}
        except requests.exceptions.RequestException as exc:
            logger.error("Request failed for %s: %s", url, exc)
            return {}
        except ValueError as exc:
            logger.error("JSON decode error for %s: %s", url, exc)
            return {}

    # ------------------------------------------------------------------
    # Schedule & Games
    # ------------------------------------------------------------------

    def get_schedule(self, date: str) -> list[dict]:
        """Return a list of games for the given date (YYYY-MM-DD).

        Each game dict contains:
            game_pk, game_time_utc, away_team, away_id, home_team, home_id,
            venue_name, venue_id, status
        """
        data = self._get(
            "/schedule",
            params={
                "sportId": 1,
                "date": date,
                "hydrate": "team,linescore,probablePitcher,venue(location)",
            },
        )
        games: list[dict] = []
        for date_block in data.get("dates", []):
            for game in date_block.get("games", []):
                teams = game.get("teams", {})
                away = teams.get("away", {})
                home = teams.get("home", {})
                venue = game.get("venue", {})
                games.append(
                    {
                        "game_pk": game.get("gamePk"),
                        "game_time_utc": game.get("gameDate"),
                        "away_team": away.get("team", {}).get("name"),
                        "away_id": away.get("team", {}).get("id"),
                        "home_team": home.get("team", {}).get("name"),
                        "home_id": home.get("team", {}).get("id"),
                        "venue_name": venue.get("name"),
                        "venue_id": venue.get("id"),
                        "status": game.get("status", {}).get("detailedState"),
                    }
                )
        return games

    def get_probable_pitchers(self, date: str) -> dict[int, dict]:
        """Return probable pitchers keyed by game_pk.

        Each value dict contains:
            home_pitcher_id, home_pitcher_name,
            away_pitcher_id, away_pitcher_name
        """
        data = self._get(
            "/schedule",
            params={
                "sportId": 1,
                "date": date,
                "hydrate": "probablePitcher",
            },
        )
        result: dict[int, dict] = {}
        for date_block in data.get("dates", []):
            for game in date_block.get("games", []):
                game_pk: int = game.get("gamePk")
                if game_pk is None:
                    continue
                teams = game.get("teams", {})
                away_pitcher = teams.get("away", {}).get("probablePitcher", {})
                home_pitcher = teams.get("home", {}).get("probablePitcher", {})
                result[game_pk] = {
                    "home_pitcher_id": home_pitcher.get("id"),
                    "home_pitcher_name": home_pitcher.get("fullName"),
                    "away_pitcher_id": away_pitcher.get("id"),
                    "away_pitcher_name": away_pitcher.get("fullName"),
                }
        return result

    def get_venue_info(self, venue_id: int) -> dict:
        """Return venue metadata including lat/lon for weather API calls.

        Returns:
            {venue_id, name, lat, lon, roof_type, surface, capacity}
        Cached in memory after first fetch.
        """
        if venue_id in self._venue_cache:
            return self._venue_cache[venue_id]

        data = self._get(f"/venues/{venue_id}", params={"hydrate": "location,fieldInfo"})
        venue = data.get("venues", [{}])[0] if data.get("venues") else {}
        if not venue:
            return {}

        location = venue.get("location", {})
        field_info = venue.get("fieldInfo", {})
        info = {
            "venue_id": venue.get("id"),
            "name": venue.get("name"),
            "lat": location.get("defaultCoordinates", {}).get("latitude"),
            "lon": location.get("defaultCoordinates", {}).get("longitude"),
            "roof_type": field_info.get("roofType"),
            "surface": field_info.get("turfType"),
            "capacity": field_info.get("capacity"),
        }
        self._venue_cache[venue_id] = info
        return info

    # ------------------------------------------------------------------
    # Live game data
    # ------------------------------------------------------------------

    def get_live_feed(self, game_pk: int) -> dict:
        """Return the raw live feed for a game.

        Endpoint: GET /game/{gamePk}/feed/live
        """
        return self._get(f"/game/{game_pk}/feed/live")

    def get_confirmed_lineup(self, game_pk: int) -> dict:
        """Return confirmed batting lineups for a game.

        Returns:
            {
                home: [{player_id, name, batting_order, position}],
                away: [...]
            }
        Returns empty lists if the lineup has not yet been posted.
        """
        data = self.get_live_feed(game_pk)
        boxscore = (
            data.get("liveData", {}).get("boxscore", {}).get("teams", {})
        )
        if not boxscore:
            return {"home": [], "away": []}

        def _parse_side(side_data: dict) -> list[dict]:
            batting_order: list[int] = side_data.get("battingOrder", [])
            players: dict = side_data.get("players", {})
            lineup: list[dict] = []
            for order_idx, player_id in enumerate(batting_order, start=1):
                key = f"ID{player_id}"
                player_data = players.get(key, {})
                person = player_data.get("person", {})
                position = player_data.get("position", {})
                lineup.append(
                    {
                        "player_id": player_id,
                        "name": person.get("fullName"),
                        "batting_order": order_idx,
                        "position": position.get("abbreviation"),
                    }
                )
            return lineup

        return {
            "home": _parse_side(boxscore.get("home", {})),
            "away": _parse_side(boxscore.get("away", {})),
        }

    def get_home_plate_umpire(self, game_pk: int) -> dict | None:
        """Return the home plate umpire for a game.

        Returns:
            {umpire_id, name} or None if not yet assigned.
        """
        data = self.get_live_feed(game_pk)
        officials = (
            data.get("liveData", {})
            .get("linescore", {})
            .get("officials", [])
        )
        for official in officials:
            if official.get("officialType") == "Home Plate":
                person = official.get("official", {})
                return {
                    "umpire_id": person.get("id"),
                    "name": person.get("fullName"),
                }
        return None

    # ------------------------------------------------------------------
    # Player stats
    # ------------------------------------------------------------------

    def get_player_season_stats(
        self,
        player_id: int,
        season: int,
        group: str = "pitching",
    ) -> dict:
        """Return season-level aggregate stats for a player.

        Args:
            player_id: MLB player ID.
            season: Four-digit season year.
            group: 'pitching' or 'hitting'.

        Returns:
            Stat dict from the MLB API (keys vary by group).
        """
        data = self._get(
            f"/people/{player_id}/stats",
            params={
                "stats": "season",
                "group": group,
                "season": season,
            },
        )
        stats_list = data.get("stats", [])
        if not stats_list:
            return {}
        splits = stats_list[0].get("splits", [])
        if not splits:
            return {}
        return splits[0].get("stat", {})

    def get_player_game_log(
        self,
        player_id: int,
        season: int,
        group: str = "pitching",
    ) -> list[dict]:
        """Return game-level stats for a player, most recent first.

        Args:
            player_id: MLB player ID.
            season: Four-digit season year.
            group: 'pitching' or 'hitting'.

        Returns:
            List of per-game stat dicts, each including 'date' and 'stat' keys
            merged at the top level for convenience.
        """
        data = self._get(
            f"/people/{player_id}/stats",
            params={
                "stats": "gameLog",
                "group": group,
                "season": season,
            },
        )
        stats_list = data.get("stats", [])
        if not stats_list:
            return []
        splits = stats_list[0].get("splits", [])
        result: list[dict] = []
        for split in splits:
            game_entry = {
                "date": split.get("date"),
                "game_pk": split.get("game", {}).get("gamePk"),
                "team": split.get("team", {}).get("name"),
                "opponent": split.get("opponent", {}).get("name"),
                "is_home": split.get("isHome"),
                **split.get("stat", {}),
            }
            result.append(game_entry)
        # API returns chronological; reverse for most-recent-first
        result.reverse()
        return result

    def get_player_splits(
        self,
        player_id: int,
        season: int,
        group: str = "pitching",
        sit_codes: list[str] | None = None,
    ) -> list[dict]:
        """Return situational splits for a player.

        Args:
            player_id: MLB player ID.
            season: Four-digit season year.
            group: 'pitching' or 'hitting'.
            sit_codes: Situation codes such as ['vl', 'vr', 'h', 'a'].
                       If None, returns all available splits.

        Returns:
            List of split dicts, each with 'split' and merged stat keys.
        """
        params: dict = {
            "stats": "statSplits",
            "group": group,
            "season": season,
        }
        if sit_codes:
            params["sitCodes"] = ",".join(sit_codes)

        data = self._get(f"/people/{player_id}/stats", params=params)
        stats_list = data.get("stats", [])
        if not stats_list:
            return []
        splits = stats_list[0].get("splits", [])
        result: list[dict] = []
        for split in splits:
            entry = {
                "split": split.get("split", {}).get("description"),
                "split_code": split.get("split", {}).get("code"),
                **split.get("stat", {}),
            }
            result.append(entry)
        return result

    def get_player_info(self, player_id: int) -> dict:
        """Return biographical info for a player.

        Returns:
            {id, fullName, position, currentTeam, batSide, pitchHand}
        """
        data = self._get(
            f"/people/{player_id}",
            params={"hydrate": "currentTeam"},
        )
        people = data.get("people", [])
        if not people:
            return {}
        person = people[0]
        return {
            "id": person.get("id"),
            "fullName": person.get("fullName"),
            "position": person.get("primaryPosition", {}).get("abbreviation"),
            "currentTeam": person.get("currentTeam", {}).get("name"),
            "currentTeamId": person.get("currentTeam", {}).get("id"),
            "batSide": person.get("batSide", {}).get("code"),
            "pitchHand": person.get("pitchHand", {}).get("code"),
        }

    # ------------------------------------------------------------------
    # Injuries / roster
    # ------------------------------------------------------------------

    def get_il_players(self, team_id: int) -> list[dict]:
        """Return players currently on the Injured List for a team.

        Returns:
            List of {player_id, name, injury_description, date_placed, expected_return}.
        """
        data = self._get(
            f"/teams/{team_id}/roster",
            params={"rosterType": "injuredList"},
        )
        roster = data.get("roster", [])
        result: list[dict] = []
        for entry in roster:
            person = entry.get("person", {})
            result.append(
                {
                    "player_id": person.get("id"),
                    "name": person.get("fullName"),
                    "injury_description": entry.get("note"),
                    "date_placed": entry.get("injuryDate"),
                    "expected_return": entry.get("expectedActivationDate"),
                    "status": entry.get("status"),
                }
            )
        return result

    def get_transactions(self, player_id: int) -> list[dict]:
        """Return IL placements and activations for a player.

        Returns:
            List of transaction dicts with {date, type_code, description}.
        """
        data = self._get(
            "/transactions",
            params={"playerId": player_id},
        )
        transactions = data.get("transactions", [])
        result: list[dict] = []
        for txn in transactions:
            result.append(
                {
                    "date": txn.get("date"),
                    "effective_date": txn.get("effectiveDate"),
                    "type_code": txn.get("typeCode"),
                    "description": txn.get("description"),
                    "team": txn.get("team", {}).get("name"),
                }
            )
        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def search_player(self, name: str) -> list[dict]:
        """Search for players by name.

        Returns:
            List of {id, fullName, currentTeam}.
        """
        data = self._get(
            "/people/search",
            params={"names": name, "hydrate": "currentTeam"},
        )
        people = data.get("people", [])
        return [
            {
                "id": p.get("id"),
                "fullName": p.get("fullName"),
                "currentTeam": p.get("currentTeam", {}).get("name"),
                "currentTeamId": p.get("currentTeam", {}).get("id"),
            }
            for p in people
        ]

    def get_teams(self) -> list[dict]:
        """Return all active MLB teams.

        Returns:
            List of {id, name, abbreviation, venue_id}.
        Cached in memory after first fetch.
        """
        if self._teams_cache is not None:
            return self._teams_cache

        data = self._get("/teams", params={"sportId": 1, "activeStatus": "Y"})
        teams: list[dict] = []
        for team in data.get("teams", []):
            teams.append(
                {
                    "id": team.get("id"),
                    "name": team.get("name"),
                    "abbreviation": team.get("abbreviation"),
                    "venue_id": team.get("venue", {}).get("id"),
                }
            )
        self._teams_cache = teams
        return teams

    def batch_delay(self) -> None:
        """Sleep for the polite batch delay between consecutive API calls."""
        time.sleep(BATCH_DELAY)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_client_instance: MLBClient | None = None


def get_client() -> MLBClient:
    """Return the module-level MLBClient singleton.

    Creates it on first call; subsequent calls return the same instance.
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = MLBClient()
    return _client_instance
