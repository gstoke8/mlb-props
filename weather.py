from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stadium centerfield bearings (degrees from home plate toward CF)
# Meteorological: 0=N, 90=E, 180=S, 270=W
# ---------------------------------------------------------------------------
STADIUM_CF_BEARING: dict[str, float] = {
    # AL East
    "Yankee Stadium": 5,
    "Fenway Park": 45,
    "Camden Yards": 10,          # Oriole Park at Camden Yards
    "Oriole Park at Camden Yards": 10,
    "Rogers Centre": 330,
    "Tropicana Field": 20,
    # AL Central
    "Guaranteed Rate Field": 0,  # Comiskey / U.S. Cellular
    "Progressive Field": 15,
    "Comerica Park": 5,
    "Kauffman Stadium": 355,
    "Target Field": 320,
    # AL West
    "Minute Maid Park": 5,
    "Angel Stadium": 290,
    "T-Mobile Park": 5,
    "Oakland Coliseum": 320,
    "Oakland-Alameda County Coliseum": 320,
    "Globe Life Field": 20,
    # NL East
    "Truist Park": 10,
    "Citi Field": 350,
    "Citizens Bank Park": 350,
    "Nationals Park": 0,
    "loanDepot park": 350,
    "Marlins Park": 350,
    # NL Central
    "Wrigley Field": 90,         # Lake Michigan / prevailing east wind matters
    "American Family Field": 350,
    "Busch Stadium": 10,
    "Great American Ball Park": 15,
    "PNC Park": 355,
    # NL West
    "Coors Field": 352,
    "Dodger Stadium": 0,
    "Chase Field": 340,
    "Oracle Park": 320,
    "Petco Park": 290,
}

# Neutral weather result (all factors at 1.0, no adjustment)
_NEUTRAL_WEATHER: dict = {
    "temp_f": 72.0,
    "humidity": 50.0,
    "wind_speed_mph": 0.0,
    "wind_dir_deg": 0.0,
    "wind_cf_component": 0.0,
    "precip_prob": 0.0,
    "conditions": "Unknown",
    "hr_weather_multiplier": 1.0,
    "k_weather_factor": 1.0,
    "data_fresh": False,
}

_VISUAL_CROSSING_BASE = (
    "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
)


def wind_cf_component(wind_speed: float, wind_dir_deg: float, venue_name: str) -> float:
    """Return wind component blowing out toward CF in mph.

    Positive = blowing out (favors HRs), negative = blowing in (suppresses HRs).

    Args:
        wind_speed: Wind speed in mph.
        wind_dir_deg: Meteorological wind direction — the direction the wind
            comes *from* (0=N, 90=E, 180=S, 270=W).
        venue_name: Stadium name used to look up CF bearing.
    """
    cf_bearing = STADIUM_CF_BEARING.get(venue_name, 0.0)
    # Convert "wind comes from" to "wind travels toward"
    wind_to_deg = (wind_dir_deg + 180) % 360
    angle_diff = abs(wind_to_deg - cf_bearing)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    return wind_speed * math.cos(math.radians(angle_diff))


def _compute_hr_multiplier(
    wind_cf: float,
    temp_f: float,
) -> float:
    """Compute HR weather multiplier from model spec formula."""
    wind_out = max(0.0, wind_cf - 5.0)
    multiplier = 1.0 + (wind_out * 0.025) + ((temp_f - 70.0) * 0.006)
    return min(1.40, max(0.70, multiplier))


def _compute_k_factor(wind_cf: float, humidity: float) -> float:
    """Compute strikeout weather factor from model spec formula."""
    humidity_above_60 = max(0.0, humidity - 60.0)
    return 1.0 - (max(0.0, -wind_cf) * 0.01) + (humidity_above_60 * 0.005)


def _parse_game_time_utc(game_time_utc: str) -> datetime:
    """Parse ISO-8601 game time string to an aware UTC datetime."""
    # Accept both '2024-07-04T19:10:00Z' and '2024-07-04T19:10:00+00:00'
    cleaned = game_time_utc.rstrip("Z")
    if "+" not in cleaned and cleaned.count("-") <= 2:
        cleaned += "+00:00"
    return datetime.fromisoformat(cleaned).astimezone(timezone.utc)


def _find_closest_hour(hours: list[dict], target: datetime) -> Optional[dict]:
    """Return the hourly record whose datetime is closest to *target*."""
    if not hours:
        return None
    best: Optional[dict] = None
    best_delta = float("inf")
    for hour in hours:
        dt_str = hour.get("datetimeEpoch")
        if dt_str is None:
            continue
        try:
            hour_dt = datetime.fromtimestamp(int(dt_str), tz=timezone.utc)
        except (ValueError, TypeError):
            continue
        delta = abs((hour_dt - target).total_seconds())
        if delta < best_delta:
            best_delta = delta
            best = hour
    return best


def _extract_weather_fields(
    hour: dict,
    game_time: datetime,
    venue_name: str,
) -> dict:
    """Extract and compute all weather fields from a Visual Crossing hourly record."""
    temp_f: float = float(hour.get("temp", 72.0))
    humidity: float = float(hour.get("humidity", 50.0))
    wind_speed: float = float(hour.get("windspeed", 0.0))
    wind_dir: float = float(hour.get("winddir", 0.0))
    precip_prob: float = float(hour.get("precipprob", 0.0))
    conditions: str = str(hour.get("conditions", "Unknown"))

    cf_wind = wind_cf_component(wind_speed, wind_dir, venue_name)

    # Determine freshness: forecast epoch vs game time
    epoch = hour.get("datetimeEpoch")
    data_fresh = False
    if epoch is not None:
        try:
            hour_dt = datetime.fromtimestamp(int(epoch), tz=timezone.utc)
            data_fresh = abs((hour_dt - game_time).total_seconds()) <= 6 * 3600
        except (ValueError, TypeError):
            pass

    return {
        "temp_f": temp_f,
        "humidity": humidity,
        "wind_speed_mph": wind_speed,
        "wind_dir_deg": wind_dir,
        "wind_cf_component": cf_wind,
        "precip_prob": precip_prob,
        "conditions": conditions,
        "hr_weather_multiplier": _compute_hr_multiplier(cf_wind, temp_f),
        "k_weather_factor": _compute_k_factor(cf_wind, humidity),
        "data_fresh": data_fresh,
    }


def _cache_key(lat: float, lon: float, game_time: datetime) -> tuple:
    lat_r = round(lat, 2)
    lon_r = round(lon, 2)
    return (lat_r, lon_r, game_time.date().isoformat(), game_time.hour)


class WeatherClient:
    """Fetches and caches Visual Crossing weather data for MLB game venues."""

    def __init__(self) -> None:
        self._api_key: str = os.getenv("VISUAL_CROSSING_API_KEY", "")
        self._cache: dict[tuple, dict] = {}
        if not self._api_key:
            logger.warning(
                "VISUAL_CROSSING_API_KEY is not set — weather calls will return "
                "neutral defaults (all multipliers = 1.0)."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_game_weather(
        self,
        lat: float,
        lon: float,
        game_time_utc: str,
        venue_name: str,
    ) -> dict:
        """Fetch weather for a single game.

        Args:
            lat: Venue latitude.
            lon: Venue longitude.
            game_time_utc: ISO-8601 UTC game start time, e.g. ``'2024-07-04T19:10:00Z'``.
            venue_name: Stadium name (used for CF-bearing lookup).

        Returns:
            Dict with weather fields and pre-computed HR/K multipliers.
            Falls back to neutral defaults on any error.
        """
        if not self._api_key:
            return dict(_NEUTRAL_WEATHER)

        try:
            game_time = _parse_game_time_utc(game_time_utc)
        except (ValueError, AttributeError) as exc:
            logger.error("Invalid game_time_utc '%s': %s", game_time_utc, exc)
            return dict(_NEUTRAL_WEATHER)

        key = _cache_key(lat, lon, game_time)
        if key in self._cache:
            return dict(self._cache[key])

        result = self._fetch_and_parse(lat, lon, game_time, venue_name)
        self._cache[key] = result
        return dict(result)

    def get_batch_weather(self, games: list[dict]) -> dict[int, dict]:
        """Fetch weather for multiple games, reusing cached results.

        Args:
            games: List of dicts, each containing:
                ``game_pk``, ``lat``, ``lon``, ``game_time_utc``, ``venue_name``.

        Returns:
            Dict keyed by ``game_pk`` mapping to weather result dicts.
        """
        results: dict[int, dict] = {}

        for game in games:
            game_pk: int = game["game_pk"]
            try:
                weather = self.get_game_weather(
                    lat=float(game["lat"]),
                    lon=float(game["lon"]),
                    game_time_utc=str(game["game_time_utc"]),
                    venue_name=str(game["venue_name"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.error(
                    "Skipping game_pk=%s due to bad input: %s", game_pk, exc
                )
                weather = dict(_NEUTRAL_WEATHER)
            results[game_pk] = weather

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_and_parse(
        self,
        lat: float,
        lon: float,
        game_time: datetime,
        venue_name: str,
    ) -> dict:
        """Call the Visual Crossing API and parse the result."""
        # Format: YYYY-MM-DD for single-day timeline endpoint
        date_str = game_time.strftime("%Y-%m-%d")
        location = f"{lat},{lon}"
        url = f"{_VISUAL_CROSSING_BASE}{location}/{date_str}"

        params = {
            "key": self._api_key,
            "unitGroup": "us",          # Fahrenheit, mph
            "include": "hours",
            "contentType": "json",
            "elements": (
                "datetimeEpoch,temp,humidity,windspeed,winddir,"
                "precipprob,conditions"
            ),
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            logger.error(
                "Visual Crossing request failed for (%s, %s) on %s: %s",
                lat,
                lon,
                date_str,
                exc,
            )
            return dict(_NEUTRAL_WEATHER)
        except ValueError as exc:
            logger.error(
                "Failed to parse Visual Crossing JSON for (%s, %s): %s",
                lat,
                lon,
                exc,
            )
            return dict(_NEUTRAL_WEATHER)

        try:
            days: list[dict] = payload.get("days", [])
            if not days:
                logger.warning(
                    "No day data returned from Visual Crossing for (%s, %s) on %s.",
                    lat,
                    lon,
                    date_str,
                )
                return dict(_NEUTRAL_WEATHER)

            hours: list[dict] = days[0].get("hours", [])
            closest = _find_closest_hour(hours, game_time)
            if closest is None:
                logger.warning(
                    "No hourly data found for (%s, %s) on %s.",
                    lat,
                    lon,
                    date_str,
                )
                return dict(_NEUTRAL_WEATHER)

            return _extract_weather_fields(closest, game_time, venue_name)

        except (KeyError, IndexError, TypeError) as exc:
            logger.error(
                "Unexpected Visual Crossing response structure: %s", exc
            )
            return dict(_NEUTRAL_WEATHER)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_client: Optional[WeatherClient] = None


def get_weather_client() -> WeatherClient:
    """Return the process-lifetime singleton WeatherClient."""
    global _client
    if _client is None:
        _client = WeatherClient()
    return _client
