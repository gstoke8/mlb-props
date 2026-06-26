"""Feature engineering for the Home Run binary classification model."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from db import MLBPropsDB, get_db
from mlb_api import MLBClient, get_client
from weather import WeatherClient, get_weather_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOOKBACK_30 = 30
LOOKBACK_60 = 60
DEFAULT_RATE = 0.0
DEFAULT_MULTIPLIER = 1.0
DEFAULT_PARK_FACTOR = 1.0
PROP_TYPE = "hr"
XWOBA_ISO_SCALE = 1.8   # proxy: xiso = xwOBA * scale

# Pitcher HR rate regression constants
# FanGraphs research: pitcher HR/FB YoY r=0.29 for starters, 0.07 for relievers.
# Heavy regression toward league mean is required for any pitcher under ~400 career FB.
LEAGUE_HR_PER_9 = 1.35
HR_RATE_PRIOR_IP = 400.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else DEFAULT_RATE


def _safe_rate(num: float, denom: float, default: float = DEFAULT_RATE) -> float:
    return num / denom if denom else default


def _current_season() -> int:
    return datetime.now(timezone.utc).year


def _rolling_mean(rows: list[dict], key: str = "value") -> float:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return _safe_mean(vals)


# ---------------------------------------------------------------------------
# Feature sub-builders
# ---------------------------------------------------------------------------

def _db_stat(db: MLBPropsDB, player_id: int, stat: str, default: float) -> float:
    """Read the most recent rolling stat from the DB with a safe fallback."""
    try:
        rows = db.get_player_stats(player_id, stat, days=2)
        return float(rows[-1]["value"]) if rows else default
    except Exception:
        return default


# Empirical Bayes HR rate constants
_LEAGUE_HR_RATE = 0.033   # league average HR / PA
_HR_EB_PRIOR_PA = 300.0   # prior pseudo-PA for regression


def _batter_power_features(
    batter_id: int, season: int, db: MLBPropsDB, mlb: MLBClient
) -> dict[str, float]:
    barrel_30 = db.get_player_stats(batter_id, "barrel_rate", LOOKBACK_30)
    barrel_60 = db.get_player_stats(batter_id, "barrel_rate", LOOKBACK_60)
    hard_hit = db.get_player_stats(batter_id, "hard_hit_rate", LOOKBACK_30)
    xwoba_30 = db.get_player_stats(batter_id, "xwoba", LOOKBACK_30)
    launch = db.get_player_stats(batter_id, "launch_angle", LOOKBACK_30)
    pull = db.get_player_stats(batter_id, "pull_pct", LOOKBACK_30)
    oppo = db.get_player_stats(batter_id, "oppo_pct", LOOKBACK_30)

    if not barrel_30:
        logger.warning("No barrel_rate (30d) for batter_id=%s", batter_id)
    if not barrel_60:
        logger.warning("No barrel_rate (60d) for batter_id=%s", batter_id)

    season_stats = mlb.get_player_season_stats(batter_id, season, group="hitting")
    pa = float(season_stats.get("plateAppearances") or 0)
    hrs = float(season_stats.get("homeRuns") or 0)
    hr_rate_season = _safe_rate(hrs, pa)

    # Empirical Bayes HR rate: regress season rate toward league mean
    hr_rate_eb_30d = (hr_rate_season * pa + _LEAGUE_HR_RATE * _HR_EB_PRIOR_PA) / max(pa + _HR_EB_PRIOR_PA, 1.0)

    barrel_30_mean = _rolling_mean(barrel_30)
    barrel_60_mean = _rolling_mean(barrel_60)
    if not barrel_30 and barrel_60:
        barrel_30_mean = barrel_60_mean * 0.95

    # Batted ball quality from nightly DB (populated by statcast_nightly)
    fly_ball_rate = _db_stat(db, batter_id, "batter_rolling_fly_ball_rate", 0.35)
    sweet_spot_pct = _db_stat(db, batter_id, "batter_rolling_sweet_spot_pct", 0.34)

    return {
        "barrel_rate_30d": barrel_30_mean,
        "barrel_rate_60d": barrel_60_mean,
        "hard_hit_rate_30d": _rolling_mean(hard_hit),
        "xiso_30d": _rolling_mean(xwoba_30) * XWOBA_ISO_SCALE,
        "avg_launch_angle_30d": _rolling_mean(launch),
        "hr_rate_season": hr_rate_season,
        "hr_rate_eb_30d": hr_rate_eb_30d,
        "pull_pct_30d": _rolling_mean(pull),
        "oppo_pct_30d": _rolling_mean(oppo),
        "fly_ball_rate_30d": fly_ball_rate,
        "sweet_spot_pct_30d": sweet_spot_pct,
    }


def _park_factor_features(
    venue_id: int, batter_hand: str, db: MLBPropsDB, mlb: MLBClient
) -> dict[str, float]:
    park_name = mlb.get_venue_info(venue_id).get("name", "")
    if not park_name:
        logger.warning("Cannot resolve venue name for venue_id=%s", venue_id)
        return {"park_factor_h": DEFAULT_PARK_FACTOR}

    row = db.get_park_factor(park_name, batter_hand)
    if not row or row.get("HR_factor") is None:
        logger.warning(
            "No HR park factor for park='%s' hand='%s'; using 1.0", park_name, batter_hand
        )
        return {"park_factor_h": DEFAULT_PARK_FACTOR}

    return {"park_factor_h": float(row["HR_factor"])}


def _weather_features(
    venue_id: int, game_time_utc: str, mlb: MLBClient, wc: WeatherClient
) -> dict[str, float]:
    _neutral = {
        "weather_hr_multiplier": DEFAULT_MULTIPLIER,
        "game_temp_f": 72.0,
        "wind_hr_factor": DEFAULT_MULTIPLIER,
        "wind_mph": 0.0,
    }
    info = mlb.get_venue_info(venue_id)
    lat, lon = info.get("lat"), info.get("lon")

    if lat is None or lon is None:
        logger.warning("Missing lat/lon for venue_id=%s; using neutral weather", venue_id)
        return _neutral

    try:
        w = wc.get_game_weather(
            lat=float(lat),
            lon=float(lon),
            game_time_utc=game_time_utc,
            venue_name=info.get("name", ""),
        )
    except Exception as exc:
        logger.error("Weather fetch failed for venue_id=%s: %s", venue_id, exc)
        return _neutral

    hr_mult = float(w.get("hr_weather_multiplier", DEFAULT_MULTIPLIER))
    return {
        "weather_hr_multiplier": hr_mult,
        "game_temp_f": float(w.get("temp_f", 72.0)),
        "wind_hr_factor": hr_mult,
        "wind_mph": float(w.get("wind_speed_mph", 0.0)),
    }


def _pitcher_features(pitcher_id: int, season: int, mlb: MLBClient, db: MLBPropsDB) -> dict[str, float]:
    stats = mlb.get_player_season_stats(pitcher_id, season, group="pitching")
    if not stats:
        logger.warning("No pitching stats for pitcher_id=%s season=%s", pitcher_id, season)

    ip = float(stats.get("inningsPitched") or 0)
    hr_allowed = float(stats.get("homeRuns") or 0)
    ks = float(stats.get("strikeOuts") or 0)
    raw_hr_rate = _safe_rate(hr_allowed * 9.0, ip)
    pitcher_hr_rate = (raw_hr_rate * ip + LEAGUE_HR_PER_9 * HR_RATE_PRIOR_IP) / (ip + HR_RATE_PRIOR_IP)
    pitcher_k_rate = _safe_rate(ks * 9.0, ip)

    splits = mlb.get_player_splits(pitcher_id, season, group="pitching")
    gb_pct = DEFAULT_RATE
    for split in splits:
        code = (split.get("split_code") or "").lower()
        if code in ("gb", "groundball"):
            raw = split.get("groundOuts") or split.get("groundBallPct") or split.get("gb_pct")
            if raw is not None:
                try:
                    val = float(raw)
                    gb_pct = val / 100.0 if val > 1.0 else val
                except (TypeError, ValueError):
                    pass
            break

    # Contact quality allowed — from statcast nightly rolling stats
    barrel_allowed   = _db_stat(db, pitcher_id, "pitcher_rolling_barrel_rate_allowed",   0.07)
    hard_hit_allowed = _db_stat(db, pitcher_id, "pitcher_rolling_hard_hit_pct_allowed",  0.38)

    return {
        "pitcher_hr_rate_season": pitcher_hr_rate,
        "pitcher_k_rate": pitcher_k_rate,
        "pitcher_gb_pct": gb_pct,
        "pitcher_barrel_rate_allowed": barrel_allowed,
        "pitcher_hard_hit_pct_allowed": hard_hit_allowed,
    }


def _handedness_features(
    batter_hand: str, pitcher_hand: str, pull_pct_30d: float = 0.40
) -> dict[str, float]:
    same = batter_hand.upper() == pitcher_hand.upper()
    batter_hand_vs_pitcher = 0.0 if same else 1.0
    is_platoon = 0.0 if same else 1.0
    return {
        "batter_hand_vs_pitcher": batter_hand_vs_pitcher,
        "is_platoon_advantage": is_platoon,
        "pull_x_platoon": pull_pct_30d * is_platoon,
    }


def _market_features(market_line: float, market_odds: int) -> dict[str, float]:
    try:
        odds = int(market_odds)
        implied = 100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        logger.warning("Cannot parse market_odds=%s: %s", market_odds, exc)
        implied = DEFAULT_RATE

    return {"market_line": float(market_line), "market_implied_prob": implied}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_hr_features(
    batter_id: int,
    pitcher_id: int,
    game_pk: int,
    venue_id: int,
    game_time_utc: str,
    batter_hand: str,
    pitcher_hand: str,
    lineup_spot: int,
    market_line: float,
    market_odds: int,
    db: Optional[MLBPropsDB] = None,
    mlb_client: Optional[MLBClient] = None,
    weather_client: Optional[WeatherClient] = None,
) -> dict[str, Any]:
    """Compute all HR features for a batter-game matchup and persist them.

    Missing sub-feature groups are filled with sensible defaults (0.0 for
    rates, 1.0 for multipliers) and a warning is logged.

    Returns
    -------
    dict
        Flat feature dictionary; also saved via ``db.save_features()``.
    """
    _db = db or get_db()
    _mlb = mlb_client or get_client()
    _wc = weather_client or get_weather_client()
    season = _current_season()

    _power_defaults: dict[str, float] = {
        "barrel_rate_30d": DEFAULT_RATE, "barrel_rate_60d": DEFAULT_RATE,
        "hard_hit_rate_30d": DEFAULT_RATE, "xiso_30d": DEFAULT_RATE,
        "avg_launch_angle_30d": DEFAULT_RATE, "hr_rate_season": DEFAULT_RATE,
        "hr_rate_eb_30d": _LEAGUE_HR_RATE, "pull_pct_30d": 0.40,
        "oppo_pct_30d": DEFAULT_RATE, "fly_ball_rate_30d": 0.35,
        "sweet_spot_pct_30d": 0.34,
    }
    _pitcher_defaults: dict[str, float] = {
        "pitcher_hr_rate_season": DEFAULT_RATE,
        "pitcher_k_rate": DEFAULT_RATE,
        "pitcher_gb_pct": 0.44,
        "pitcher_barrel_rate_allowed": 0.07,
        "pitcher_hard_hit_pct_allowed": 0.38,
    }

    try:
        power = _batter_power_features(batter_id, season, _db, _mlb)
    except Exception as exc:
        logger.error("Batter power features failed batter_id=%s: %s", batter_id, exc)
        power = _power_defaults

    try:
        park = _park_factor_features(venue_id, batter_hand, _db, _mlb)
    except Exception as exc:
        logger.error("Park factor features failed venue_id=%s: %s", venue_id, exc)
        park = {"park_factor_h": DEFAULT_PARK_FACTOR}

    try:
        weather = _weather_features(venue_id, game_time_utc, _mlb, _wc)
    except Exception as exc:
        logger.error("Weather features failed venue_id=%s: %s", venue_id, exc)
        weather = {
            "weather_hr_multiplier": DEFAULT_MULTIPLIER,
            "game_temp_f": 72.0,
            "wind_hr_factor": DEFAULT_MULTIPLIER,
            "wind_mph": 0.0,
        }

    try:
        pitcher = _pitcher_features(pitcher_id, season, _mlb, _db)
    except Exception as exc:
        logger.error("Pitcher features failed pitcher_id=%s: %s", pitcher_id, exc)
        pitcher = _pitcher_defaults

    pull_pct = power.get("pull_pct_30d", 0.40)
    handedness = _handedness_features(batter_hand, pitcher_hand, pull_pct_30d=pull_pct)

    features: dict[str, Any] = {
        **power,
        **park,
        **weather,
        **pitcher,
        **handedness,
        "lineup_spot": int(lineup_spot),
        **_market_features(market_line, market_odds),
    }

    feat_date = game_time_utc[:10]
    try:
        _db.save_features(batter_id, feat_date, PROP_TYPE, features)
    except Exception as exc:
        logger.error(
            "save_features failed batter_id=%s game_pk=%s: %s", batter_id, game_pk, exc
        )

    return features


def get_hr_feature_matrix(
    game_pks: list[int],
    db: Optional[MLBPropsDB] = None,
    matchups: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Load saved HR features for a list of game PKs.

    The features table is keyed by ``(player_id, feat_date, prop_type)``.
    Pass *matchups* (list of dicts with ``game_pk``, ``batter_id``,
    ``game_time_utc``) to enable individual lookups via
    :meth:`~db.MLBPropsDB.get_features`.

    Returns
    -------
    list[dict]
        One feature dict per batter-game resolved from the features table.
    """
    _db = db or get_db()
    game_pk_set = set(game_pks)
    results: list[dict[str, Any]] = []

    if not matchups:
        logger.warning(
            "get_hr_feature_matrix: no matchups provided; cannot resolve "
            "(player_id, feat_date) from game_pks alone — returning empty list."
        )
        return results

    for matchup in matchups:
        if matchup.get("game_pk") not in game_pk_set:
            continue
        batter_id = matchup.get("batter_id")
        feat_date = (matchup.get("game_time_utc") or "")[:10]
        if not batter_id or len(feat_date) < 10:
            logger.warning("Skipping matchup with missing batter_id/game_time_utc: %s", matchup)
            continue
        try:
            feat = _db.get_features(batter_id, feat_date, PROP_TYPE)
        except Exception as exc:
            logger.error(
                "get_features failed batter_id=%s feat_date=%s: %s", batter_id, feat_date, exc
            )
            feat = None
        if feat is not None:
            results.append({**feat, "game_pk": matchup["game_pk"], "batter_id": batter_id})
        else:
            logger.warning(
                "No saved features for batter_id=%s feat_date=%s game_pk=%s",
                batter_id, feat_date, matchup.get("game_pk"),
            )

    return results


def build_training_features(
    season_games: list[dict[str, Any]],
    db: Optional[MLBPropsDB] = None,
    mlb_client: Optional[MLBClient] = None,
) -> list[dict[str, Any]]:
    """Build historical training features with HR labels attached.

    Each element of *season_games* must contain:
    ``game_pk``, ``batter_id``, ``pitcher_id``, ``venue_id``,
    ``game_time_utc``, ``batter_hand``, ``pitcher_hand``,
    ``lineup_spot``, ``market_line``, ``market_odds``, ``actual_hr``.

    Returns
    -------
    list[dict]
        Feature dicts with ``actual_hr`` label; error rows are skipped.
    """
    _db = db or get_db()
    _mlb = mlb_client or get_client()
    _wc = get_weather_client()
    rows: list[dict[str, Any]] = []

    for idx, game in enumerate(season_games):
        try:
            features = compute_hr_features(
                batter_id=game["batter_id"],
                pitcher_id=game["pitcher_id"],
                game_pk=game["game_pk"],
                venue_id=game["venue_id"],
                game_time_utc=game["game_time_utc"],
                batter_hand=game["batter_hand"],
                pitcher_hand=game["pitcher_hand"],
                lineup_spot=game["lineup_spot"],
                market_line=game["market_line"],
                market_odds=game["market_odds"],
                db=_db,
                mlb_client=_mlb,
                weather_client=_wc,
            )
        except KeyError as exc:
            logger.warning(
                "Skipping game index %d (game_pk=%s): missing field %s",
                idx, game.get("game_pk"), exc,
            )
            continue
        except Exception as exc:
            logger.error(
                "Unexpected error game index %d (game_pk=%s): %s",
                idx, game.get("game_pk"), exc,
            )
            continue

        rows.append({**features, "actual_hr": int(game["actual_hr"])})

    logger.info(
        "build_training_features: %d rows produced from %d input games",
        len(rows), len(season_games),
    )
    return rows
