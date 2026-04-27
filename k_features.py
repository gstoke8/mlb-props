"""k_features.py — Feature engineering for the Strikeout (K) Poisson GLM."""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from typing import Any, Optional

import umpires as umpires_module
from db import get_db
from mlb_api import get_client
from weather import get_weather_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

K_LOOKBACK_DAYS = 30
PARK_K_DEFAULT = 1.0
STUFF_PLUS_DEFAULT = 100.0
OPENER_IP_THRESHOLD = 4.5

_PITCHES_PER_INNING = 15
_RECENT_STARTS = 3
_PROP_TYPE = "k"

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _safe_mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _stat_values(rows: list[dict], key: str = "stat_value") -> list[float]:
    return [r[key] for r in rows if r.get(key) is not None]


def _rolling_pitcher_stats(pitcher_id: int, db: Any) -> dict[str, Any]:
    """CSW, K/PA, whiff rates and Stuff+ from DB player_stats (last 30 days)."""
    csw = _safe_mean(_stat_values(db.get_player_stats(pitcher_id, "csw", K_LOOKBACK_DAYS)))
    k_pa = _safe_mean(_stat_values(db.get_player_stats(pitcher_id, "k_pa", K_LOOKBACK_DAYS)))
    whiff = _safe_mean(_stat_values(db.get_player_stats(pitcher_id, "whiff", K_LOOKBACK_DAYS)))
    stuff_vals = _stat_values(db.get_player_stats(pitcher_id, "stuff_plus", K_LOOKBACK_DAYS))
    stuff = _safe_mean(stuff_vals) if stuff_vals else STUFF_PLUS_DEFAULT
    return {"csw_rate_30d": csw, "k_rate_30d": k_pa, "whiff_rate_30d": whiff, "stuff_plus": stuff}


def _pitcher_swstr_features(pitcher_id: int, db: Any) -> dict[str, Optional[float]]:
    """SwStr% overall and per-pitch-type whiff rates from nightly Statcast computation.

    Uses days=2 to pick up stats stored today or yesterday.
    Returns None for each metric when insufficient data exists.
    """
    def _latest(stat_type: str) -> Optional[float]:
        rows = db.get_player_stats(pitcher_id, stat_type, days=2)
        vals = [r["value"] for r in rows if r.get("value") is not None]
        return vals[-1] if vals else None

    return {
        "swstr_rate_30d":    _latest("pitcher_rolling_swstr_rate"),
        "ff_whiff_rate_30d": _latest("pitcher_rolling_ff_whiff"),
        "sl_whiff_rate_30d": _latest("pitcher_rolling_sl_whiff"),
        "ch_whiff_rate_30d": _latest("pitcher_rolling_ch_whiff"),
        "si_whiff_rate_30d": _latest("pitcher_rolling_si_whiff"),
    }


def _opp_lineup_xwoba(lineup_ids: list[int], db: Any) -> Optional[float]:
    """Mean xwOBA_30d across the opposing lineup from stored Statcast rolling stats."""
    vals: list[float] = []
    for pid in lineup_ids:
        rows = db.get_player_stats(pid, "batter_rolling_xwOBA_30d", days=2)
        if rows:
            v = rows[-1].get("value")
            if v is not None:
                vals.append(float(v))
    return _safe_mean(vals) if vals else None


def _season_k_rate(pitcher_id: int, season: int, mlb_client: Any) -> Optional[float]:
    """K/9 this season; tries strikeoutsPer9Inn first, then computes from raw counts."""
    stats = mlb_client.get_player_season_stats(pitcher_id, season, group="pitching")
    if not stats:
        return None
    try:
        return float(stats["strikeoutsPer9Inn"])
    except (KeyError, TypeError, ValueError):
        pass
    ks, ip = stats.get("strikeOuts"), stats.get("inningsPitched")
    try:
        ip_f = float(ip)
        return (float(ks) / ip_f * 9) if ks is not None and ip_f > 0 else None
    except (TypeError, ValueError):
        return None


def _parse_game_date(date_str: Optional[str]) -> Optional[date]:
    if not date_str:
        return None
    try:
        return date.fromisoformat(date_str[:10])
    except ValueError:
        return None


def _days_since_last_start(game_log: list[dict], game_time_utc: str) -> Optional[int]:
    """Days rest from most-recent prior start (game_log most-recent-first)."""
    try:
        gd = datetime.fromisoformat(game_time_utc.rstrip("Z").replace("Z", "+00:00")).date()
    except (ValueError, AttributeError):
        return None
    for entry in game_log:
        ed = _parse_game_date(entry.get("date"))
        if ed is not None and ed < gd:
            return (gd - ed).days
    return None


def _avg_ip_last_starts(game_log: list[dict], n: int = _RECENT_STARTS) -> Optional[float]:
    """Average innings pitched across the last n starts."""
    ip_vals: list[float] = []
    for entry in game_log:
        if len(ip_vals) >= n:
            break
        try:
            ip_vals.append(float(entry["inningsPitched"]))
        except (KeyError, TypeError, ValueError):
            continue
    return _safe_mean(ip_vals)


def _opponent_k_stats(lineup_ids: list[int], db: Any) -> dict[str, Optional[float]]:
    """Mean season and 30-day K/PA across the opposing lineup."""
    season_rates, rolling_rates = [], []
    for pid in lineup_ids:
        season_vals = _stat_values(db.get_player_stats(pid, "k_pa_season", 365))
        rolling_vals = _stat_values(db.get_player_stats(pid, "k_pa", K_LOOKBACK_DAYS))
        if season_vals:
            season_rates.append(season_vals[-1])
        if rolling_vals:
            rolling_rates.append(_safe_mean(rolling_vals))  # type: ignore[arg-type]
    return {"opp_k_rate_season": _safe_mean(season_rates), "opp_k_rate_30d": _safe_mean(rolling_rates)}


def _lineup_lhb_fraction(lineup_ids: list[int], mlb_client: Any) -> Optional[float]:
    """Fraction of opposing lineup batting left-handed."""
    if not lineup_ids:
        return None
    left, resolved = 0, 0
    for pid in lineup_ids:
        side = mlb_client.get_player_info(pid).get("batSide")
        if side is None:
            continue
        resolved += 1
        if side == "L":
            left += 1
    return (left / resolved) if resolved else None


def _american_to_implied_prob(american_odds: Optional[float]) -> Optional[float]:
    if american_odds is None:
        return None
    try:
        o = float(american_odds)
    except (ValueError, TypeError):
        return None
    if o >= 100:
        return 100.0 / (o + 100.0)
    if o < 0:
        return (-o) / (-o + 100.0)
    return None


def _load_k_feature_rows(db: Any) -> list[dict[str, Any]]:
    """Return all raw K feature rows from the features table."""
    try:
        with db._connect() as conn:
            return [
                dict(r) for r in conn.execute(
                    "SELECT player_id, feat_date, feature_json FROM features WHERE prop_type = ?",
                    (_PROP_TYPE,),
                ).fetchall()
            ]
    except Exception as exc:
        logger.error("Failed to query K features: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_k_features(
    pitcher_id: int,
    game_pk: int,
    venue_id: int,
    game_time_utc: str,
    opposing_lineup_ids: list[int],
    umpire_name: str,
    is_home: bool,
    market_line: float,
    market_odds_over: Optional[float],
    db: Any = None,
    mlb_client: Any = None,
    weather_client: Any = None,
    line_movement: float = 0.0,
) -> dict[str, Any]:
    """Compute all K-prop features for a pitcher/game; persists to DB.

    Returns a flat dict of all feature keys.
    """
    db = db or get_db()
    mlb_client = mlb_client or get_client()
    weather_client = weather_client or get_weather_client()
    season = datetime.now(timezone.utc).year

    rolling = _rolling_pitcher_stats(pitcher_id, db)
    swstr = _pitcher_swstr_features(pitcher_id, db)
    opp_xwoba = _opp_lineup_xwoba(opposing_lineup_ids, db)
    k_rate_season = _season_k_rate(pitcher_id, season, mlb_client)

    game_log = mlb_client.get_player_game_log(pitcher_id, season, group="pitching")
    days_rest = _days_since_last_start(game_log, game_time_utc)
    avg_ip = _avg_ip_last_starts(game_log)
    is_opener_risk = int((avg_ip or 0.0) < OPENER_IP_THRESHOLD)
    pitch_count_context = (avg_ip or 0.0) * _PITCHES_PER_INNING

    opp_stats = _opponent_k_stats(opposing_lineup_ids, db)
    lineup_lhb = _lineup_lhb_fraction(opposing_lineup_ids, mlb_client)

    umpire_k_factor = umpires_module.get_k_factor(db, umpire_name)

    venue_info = mlb_client.get_venue_info(venue_id)
    park_row = (
        db.get_park_factor(venue_info.get("name", ""), "neutral")
        if venue_info.get("name") else None
    )
    park_k_factor = float(park_row["hr_factor"]) if park_row and park_row.get("hr_factor") else PARK_K_DEFAULT

    weather_data: dict[str, Any] = {}
    if venue_info.get("lat") and venue_info.get("lon") and venue_info.get("name"):
        try:
            weather_data = weather_client.get_game_weather(
                lat=float(venue_info["lat"]),
                lon=float(venue_info["lon"]),
                game_time_utc=game_time_utc,
                venue_name=venue_info["name"],
            )
        except Exception as exc:
            logger.warning("Weather lookup failed game_pk=%d venue_id=%d: %s", game_pk, venue_id, exc)
    weather_k_factor = float(weather_data.get("k_weather_factor", 1.0))

    features: dict[str, Any] = {
        # Pitcher effectiveness
        "csw_rate_30d": rolling["csw_rate_30d"],
        "k_rate_30d": rolling["k_rate_30d"],
        "k_rate_season": k_rate_season,
        "stuff_plus": rolling["stuff_plus"],
        "whiff_rate_30d": rolling["whiff_rate_30d"],
        # Pitcher stuff metrics (swinging strike rates by pitch type)
        "swstr_rate_30d":    swstr["swstr_rate_30d"],
        "ff_whiff_rate_30d": swstr["ff_whiff_rate_30d"],
        "sl_whiff_rate_30d": swstr["sl_whiff_rate_30d"],
        "ch_whiff_rate_30d": swstr["ch_whiff_rate_30d"],
        "si_whiff_rate_30d": swstr["si_whiff_rate_30d"],
        # Opposing lineup
        "opp_k_rate_season": opp_stats["opp_k_rate_season"],
        "opp_k_rate_30d": opp_stats["opp_k_rate_30d"],
        "lineup_handedness_split": lineup_lhb,
        "opp_lineup_xwoba": opp_xwoba,
        # Game context
        "umpire_k_factor": umpire_k_factor,
        "weather_k_factor": weather_k_factor,
        "park_k_factor": park_k_factor,
        "is_home": int(is_home),
        "days_rest": days_rest,
        # Role context
        "avg_ip_30d": avg_ip,
        "is_opener_risk": is_opener_risk,
        "pitch_count_context": pitch_count_context,
        # Market
        "market_line": market_line,
        "market_implied_over": _american_to_implied_prob(market_odds_over),
        "line_movement": line_movement,
    }

    try:
        db.save_features(pitcher_id, game_time_utc[:10], _PROP_TYPE, features)
    except Exception as exc:
        logger.error("Failed to save K features pitcher_id=%d game_pk=%d: %s", pitcher_id, game_pk, exc)

    logger.debug("K features computed pitcher_id=%d game_pk=%d keys=%d", pitcher_id, game_pk, len(features))
    return features


def get_k_feature_matrix(
    game_pks: list[int],
    db: Any = None,
) -> list[dict[str, Any]]:
    """Load saved K features from DB for a list of game PKs.

    Returns list of feature dicts (game_pk and pitcher_id injected).
    Games with no saved features are silently skipped.
    """
    db = db or get_db()
    pk_set = set(game_pks)
    raw_rows = _load_k_feature_rows(db)

    results: list[dict[str, Any]] = []
    for row in raw_rows:
        try:
            feat = json.loads(row["feature_json"]) if row["feature_json"] else {}
        except (ValueError, TypeError) as exc:
            logger.warning("Skipping corrupt feature row: %s", exc)
            continue
        # game_pk isn't stored in the features table; include rows that match any requested pk
        # When game_pks is provided for filtering, caller is expected to supply game_pk via feat
        stored_gp = feat.get("game_pk")
        if stored_gp is not None and stored_gp not in pk_set:
            continue
        results.append({
            "pitcher_id": row["player_id"],
            "feat_date": row["feat_date"],
            **feat,
        })

    logger.debug("get_k_feature_matrix: %d rows for %d game_pks", len(results), len(game_pks))
    return results


def build_k_training_features(
    season_games: list[dict[str, Any]],
    db: Any = None,
    mlb_client: Any = None,
) -> list[dict[str, Any]]:
    """Build labelled K feature matrix for historical training data.

    Each element of season_games must include:
        game_pk, pitcher_id, venue_id, game_time_utc, opposing_lineup_ids,
        umpire_name, is_home, market_line, market_odds_over, actual_ks

    Returns list of feature dicts with 'actual_ks' label attached.
    """
    db = db or get_db()
    mlb_client = mlb_client or get_client()
    results: list[dict[str, Any]] = []

    for game in season_games:
        try:
            features = compute_k_features(
                pitcher_id=game["pitcher_id"],
                game_pk=game["game_pk"],
                venue_id=game["venue_id"],
                game_time_utc=game["game_time_utc"],
                opposing_lineup_ids=game.get("opposing_lineup_ids", []),
                umpire_name=game.get("umpire_name", ""),
                is_home=bool(game.get("is_home", False)),
                market_line=float(game.get("market_line", 0.0)),
                market_odds_over=game.get("market_odds_over"),
                db=db,
                mlb_client=mlb_client,
            )
        except Exception as exc:
            logger.error(
                "K feature compute failed game_pk=%s pitcher_id=%s: %s",
                game.get("game_pk"), game.get("pitcher_id"), exc,
            )
            continue

        results.append({
            "game_pk": game["game_pk"],
            "pitcher_id": game["pitcher_id"],
            "actual_ks": game.get("actual_ks"),
            **features,
        })

    logger.info("build_k_training_features: %d / %d rows built", len(results), len(season_games))
    return results
