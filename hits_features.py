from __future__ import annotations
"""
Feature engineering for the Hits (H1.5) binary classification model.

Pulls from the SQLite DB (populated by statcast_nightly.py) and the MLB API
to produce a feature vector for each batter-game matchup.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from db import MLBPropsDB, get_db
from mlb_api import MLBClient, get_client
from weather import get_weather_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

LOOKBACK_30 = 30
DEFAULT_RATE = 0.0
DEFAULT_PARK_FACTOR = 1.0
PROP_TYPE = "hits"

# Bayesian shrinkage constants — blend observed rate with league average weighted by sample size.
# Formula: (observed_count + league_avg * prior_n) / (sample_n + prior_n)
_LEAGUE_HIT_RATE = 0.243   # MLB batting average
_LEAGUE_K_RATE   = 0.225   # MLB K per PA
_LEAGUE_BB_RATE  = 0.085   # MLB BB per PA
_LEAGUE_H_PER_9  = 8.67    # MLB hits allowed per 9 IP
_LEAGUE_K_PER_BF = 0.225   # MLB pitcher K per batter faced
_PA_PRIOR        = 400     # prior PA: small-sample batters regress hard toward league avg
_BF_PRIOR        = 350     # prior BF for pitcher K/BF


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_mean(values: list[float]) -> float:
    if not values:
        return DEFAULT_RATE
    return sum(values) / len(values)


def _safe_rate(numerator: float, denominator: float, default: float = DEFAULT_RATE) -> float:
    if not denominator:
        return default
    return numerator / denominator


def _shrink_rate(numerator: float, denominator: float, league_avg: float, prior_n: float) -> float:
    """Empirical Bayes shrinkage toward league average: (x + μ·M) / (n + M)."""
    return (numerator + league_avg * prior_n) / (denominator + prior_n)


def _current_season() -> int:
    return datetime.now(timezone.utc).year


def _rolling_mean(rows: list[dict], value_key: str = "value") -> float:
    values = [float(r[value_key]) for r in rows if r.get(value_key) is not None]
    return _safe_mean(values)


# ---------------------------------------------------------------------------
# Feature sub-builders
# ---------------------------------------------------------------------------

def _batter_contact_features(
    batter_id: int,
    season: int,
    db: MLBPropsDB,
    mlb_client: MLBClient,
) -> dict[str, float]:
    """Batter contact metrics from Statcast DB and MLB API."""
    contact_30 = db.get_player_stats(batter_id, "in_zone_contact", LOOKBACK_30)
    babip_30 = db.get_player_stats(batter_id, "babip", LOOKBACK_30)
    exit_velo_30 = db.get_player_stats(batter_id, "exit_velocity", LOOKBACK_30)
    hard_hit_30 = db.get_player_stats(batter_id, "hard_hit_rate", LOOKBACK_30)
    launch_angle_30 = db.get_player_stats(batter_id, "launch_angle", LOOKBACK_30)
    line_drive_30 = db.get_player_stats(batter_id, "line_drive_rate", LOOKBACK_30)

    if not contact_30:
        logger.warning("No in_zone_contact data (30d) for batter_id=%s", batter_id)

    season_stats = mlb_client.get_player_season_stats(batter_id, season, group="hitting")
    pa = float(season_stats.get("plateAppearances") or 0)
    ba_season = float(season_stats.get("avg") or season_stats.get("battingAverage") or 0.243)

    babip_30d_val = _rolling_mean(babip_30)

    # xBA proxy: use stored xwOBA_30d rolling stat (correlated with xBA; directional signal).
    # At training time train_models.py uses est_ba from statcast_batter_expected_stats directly.
    xba_rows = db.get_player_stats(batter_id, "batter_rolling_xwOBA_30d", days=2)
    if xba_rows:
        # Scale xwOBA to xBA range: league avg xwOBA ~0.320, xBA ~0.250 → ratio ~0.78
        xba_season = float(xba_rows[-1].get("value") or 0.250) * 0.78
    else:
        xba_season = ba_season  # no signal when unavailable; gap becomes 0

    # sweet_spot_pct: batted balls at 8-32° launch angle. Use stored rolling stat if available;
    # fall back to league average (0.340). Added to statcast nightly in a future pipeline update.
    sweet_rows = db.get_player_stats(batter_id, "batter_rolling_sweet_spot_pct", days=2)
    sweet_spot_pct = float(sweet_rows[-1].get("value")) if sweet_rows else 0.340

    return {
        "contact_rate_30d": _rolling_mean(contact_30),
        "babip_30d": babip_30d_val,
        "avg_exit_velo_30d": _rolling_mean(exit_velo_30),
        "hard_hit_rate_30d": _rolling_mean(hard_hit_30),
        "avg_launch_angle_30d": _rolling_mean(launch_angle_30),
        "line_drive_rate_30d": _rolling_mean(line_drive_30),
        # v7 regression signals (replace hit_rate_season)
        "xba_season": xba_season,
        "xba_minus_ba_gap": xba_season - ba_season,
        "babip_deviation": babip_30d_val - 0.300,
        "sweet_spot_pct": sweet_spot_pct,
    }


def _pitcher_opposing_features(
    pitcher_id: int,
    season: int,
    mlb_client: MLBClient,
    db: MLBPropsDB,
) -> dict[str, float]:
    """Pitcher opposition metrics from MLB API and Statcast DB."""
    stats = mlb_client.get_player_season_stats(pitcher_id, season, group="pitching")
    if not stats:
        logger.warning("No pitching stats for pitcher_id=%s season=%s", pitcher_id, season)

    ip = float(stats.get("inningsPitched") or 0)
    h_allowed = float(stats.get("hits") or 0)
    k = float(stats.get("strikeOuts") or 0)
    bf = float(stats.get("battersFaced") or 0)

    babip_30 = db.get_player_stats(pitcher_id, "babip_allowed", LOOKBACK_30)

    # Ground-ball rate: GB pitchers allow more singles, fewer XBH
    go = float(stats.get("groundOuts") or 0)
    ao = float(stats.get("airOuts") or 0)
    pitcher_gb_pct = go / (go + ao) if go + ao > 0 else 0.44  # 0.44 = league avg

    # Prior weight in IP units: _BF_PRIOR / 4.3 ≈ 81 IP (~half season)
    ip_prior = _BF_PRIOR / 4.3
    bf_eff = bf if bf > 0 else ip * 4.3
    return {
        "pitcher_babip_allowed_30d": _rolling_mean(babip_30),
        "pitcher_hit_rate_allowed_season": _shrink_rate(h_allowed, ip, _LEAGUE_H_PER_9 / 9.0, ip_prior) * 9.0,
        "pitcher_k_rate_season": _shrink_rate(k, bf_eff, _LEAGUE_K_PER_BF, _BF_PRIOR),
        "pitcher_gb_pct": pitcher_gb_pct,
    }


def _park_factor_features(
    venue_id: int,
    batter_hand: str,
    db: MLBPropsDB,
    mlb_client: MLBClient,
) -> dict[str, float]:
    """Handedness-split park factor for hits at the venue."""
    venue_info = mlb_client.get_venue_info(venue_id)
    park_name = venue_info.get("name", "")

    if not park_name:
        logger.warning("Could not resolve venue name for venue_id=%s", venue_id)
        return {"park_factor_hits_h": DEFAULT_PARK_FACTOR}

    park_row = db.get_park_factor(park_name, batter_hand)
    if park_row is None:
        logger.warning("No park factor for park='%s' hand='%s'", park_name, batter_hand)
        return {"park_factor_hits_h": DEFAULT_PARK_FACTOR}

    factor = park_row.get("H_factor") or park_row.get("hits_factor")
    if factor is None:
        logger.warning("hits_factor is None for park='%s' hand='%s'", park_name, batter_hand)
        return {"park_factor_hits_h": DEFAULT_PARK_FACTOR}

    return {"park_factor_hits_h": float(factor)}


def _handedness_features(batter_hand: str, pitcher_hand: str) -> dict[str, int]:
    is_same = batter_hand.upper() == pitcher_hand.upper()
    return {"is_platoon_advantage": 0 if is_same else 1}


def _batter_discipline_features(
    batter_id: int,
    season: int,
    db: MLBPropsDB,
    mlb_client: MLBClient,
) -> dict[str, float]:
    """Batter plate discipline from MLB API season stats and stored Statcast rolling stats."""
    stats = mlb_client.get_player_season_stats(batter_id, season, group="hitting")
    pa = float(stats.get("plateAppearances") or 0)
    k = float(stats.get("strikeOuts") or 0)
    bb = float(stats.get("baseOnBalls") or 0)

    k_rate = _shrink_rate(k, pa, _LEAGUE_K_RATE, _PA_PRIOR)
    walk_rate = _shrink_rate(bb, pa, _LEAGUE_BB_RATE, _PA_PRIOR)

    # Chase rate from nightly Statcast computation (stored today or yesterday)
    chase_rows = db.get_player_stats(batter_id, "batter_rolling_chase_rate", days=2)
    chase_rate = float(chase_rows[-1]["value"]) if chase_rows else DEFAULT_RATE

    # Zone contact rate (30d rolling)
    zcon_rows = db.get_player_stats(batter_id, "batter_rolling_zone_contact", days=2)
    zone_contact_rate = float(zcon_rows[-1]["value"]) if zcon_rows else 0.78  # league avg

    # Sprint speed from nightly fetch
    speed_rows = db.get_player_stats(batter_id, "batter_sprint_speed", days=180)
    sprint_speed = float(speed_rows[-1]["value"]) if speed_rows else 27.0  # MLB avg ~27 ft/s

    return {
        "batter_k_rate_season": k_rate,
        "batter_walk_rate_season": walk_rate,
        "chase_rate_30d": chase_rate,
        "sprint_speed": sprint_speed,
        "zone_contact_rate_30d": zone_contact_rate,
    }


def _opp_lineup_xwoba(
    lineup_ids: list[int],
    db: MLBPropsDB,
) -> float:
    """Mean xwOBA_30d across a list of opposing lineup player IDs."""
    vals: list[float] = []
    for pid in lineup_ids:
        rows = db.get_player_stats(pid, "batter_rolling_xwOBA_30d", days=2)
        if rows:
            v = rows[-1].get("value")
            if v is not None:
                vals.append(float(v))
    return _safe_mean(vals) if vals else DEFAULT_RATE


def _pitcher_advanced_features(
    pitcher_id: int,
    season: int,
    db: MLBPropsDB,
    mlb_client: MLBClient,
) -> dict[str, float]:
    """SwStr%, fastball %, and team DER from Statcast and MLB API."""
    swstr_rows = db.get_player_stats(pitcher_id, "pitcher_rolling_swstr_rate", days=2)
    swstr_season = float(swstr_rows[-1]["value"]) if swstr_rows else 0.105  # league avg ~10.5%

    # Fastball % from pitch mix table
    fastball_pct = 0.52  # league average fastball+sinker%
    try:
        from pitch_type_matchup import get_pitcher_pitch_mix
        mix = get_pitcher_pitch_mix(pitcher_id, db)
        fb_pcts = [v for pt, v in mix.items() if pt in ("FF", "SI", "FC")]
        if fb_pcts:
            fastball_pct = sum(fb_pcts)
    except Exception:
        pass

    # Opposing team DER: use pitcher's BABIP allowed as proxy (1 - BABIP_allowed ≈ DER)
    # Higher DER = better defense; pitcher BABIP allowed reflects team defense
    babip_rows = db.get_player_stats(pitcher_id, "babip_allowed", days=30)
    pitcher_babip = float(babip_rows[-1]["value"]) if babip_rows else 0.295
    team_der = round(1.0 - pitcher_babip, 4)  # proxy

    return {
        "pitcher_swstr_season": swstr_season,
        "pitcher_fastball_pct_season": fastball_pct,
        "opposing_team_der_season": max(0.60, min(0.75, team_der)),
    }


def _game_context_features(
    lineup_spot: int,
    pitcher_id: int,
    season: int,
    mlb_client: MLBClient,
    weather_client: Any,
    venue_id: int,
    game_time_utc: str,
) -> dict[str, Any]:
    """TTO exposure and weather for the batter-game."""
    # Expected TTO number: based on lineup spot and pitcher avg_ip
    avg_ip = 5.5  # league average
    try:
        game_log = mlb_client.get_player_game_log(pitcher_id, season, group="pitching")
        recent = [g for g in game_log if g.get("gamesStarted", 0) == 1][:5]
        if recent:
            ip_vals = []
            for g in recent:
                try:
                    ip_vals.append(float(g["inningsPitched"]))
                except (KeyError, TypeError, ValueError):
                    pass
            if ip_vals:
                avg_ip = sum(ip_vals) / len(ip_vals)
    except Exception:
        pass

    # Each TTO = 9 batters; lineup_spot in 1-9
    tto_number = min(3.0, 1.0 + (lineup_spot - 1) / 9.0 + avg_ip / 9.0 * (27.0 / 9.0) / 3.0)
    # Simpler: expected TTO number for this lineup spot given avg_ip
    pa_before_this_batter = lineup_spot - 1
    total_pa = avg_ip * 3.0  # rough PA estimate
    expected_tto = 1.0 + pa_before_this_batter / 9.0
    expected_tto = min(3.0, max(1.0, expected_tto))

    # Weather
    game_temp_f = 75.0
    wind_direction_category = 0.0  # 0=neutral, 1=blowing_out, -1=blowing_in
    if weather_client and venue_id and game_time_utc:
        try:
            from mlb_api import get_client as _get_mlb
            _mlb = _get_mlb()
            venue_info = _mlb.get_venue_info(venue_id)
            if venue_info.get("lat") and venue_info.get("lon"):
                wx = weather_client.get_game_weather(
                    lat=float(venue_info["lat"]),
                    lon=float(venue_info["lon"]),
                    game_time_utc=game_time_utc,
                    venue_name=venue_info.get("name", ""),
                )
                game_temp_f = float(wx.get("temp_f", 75.0))
                # wind_dir: positive k_factor > 1.0 means favorable for offense (blowing out)
                wf = float(wx.get("k_weather_factor", 1.0))
                wind_direction_category = 1.0 if wf > 1.02 else (-1.0 if wf < 0.98 else 0.0)
        except Exception:
            pass

    return {
        "expected_tto_number": expected_tto,
        "game_temp_f": game_temp_f,
        "wind_direction_category": wind_direction_category,
    }


def _market_features(market_line: float, market_odds: int) -> dict[str, float]:
    try:
        odds = int(market_odds)
        implied = 100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        logger.warning("Could not parse market_odds=%s: %s", market_odds, exc)
        implied = DEFAULT_RATE
    return {"market_line": float(market_line), "market_implied_prob": implied}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_hits_features(
    batter_id: int,
    pitcher_id: int,
    game_pk: int,
    venue_id: int,
    batter_hand: str,
    pitcher_hand: str,
    lineup_spot: int,
    market_line: float,
    market_odds: int,
    db: Optional[MLBPropsDB] = None,
    mlb_client: Optional[MLBClient] = None,
    lineup_ids: Optional[list[int]] = None,
    line_movement: float = 0.0,
    weather_client: Any = None,
    game_time_utc: str = "",
    is_home: bool = True,
) -> dict[str, Any]:
    """Compute all Hits features for a single batter-game matchup and persist to DB.

    Returns a flat feature dict. Errors in any sub-builder are caught and logged;
    the corresponding features fall back to module-level defaults.
    """
    _db = db or get_db()
    _mlb = mlb_client or get_client()
    _weather = weather_client or get_weather_client()
    season = _current_season()

    try:
        contact = _batter_contact_features(batter_id, season, _db, _mlb)
    except Exception as exc:
        logger.error("batter contact features failed batter_id=%s: %s", batter_id, exc)
        contact = {
            "contact_rate_30d": DEFAULT_RATE,
            "babip_30d": DEFAULT_RATE,
            "avg_exit_velo_30d": DEFAULT_RATE,
            "hard_hit_rate_30d": DEFAULT_RATE,
            "avg_launch_angle_30d": DEFAULT_RATE,
            "line_drive_rate_30d": DEFAULT_RATE,
            "xba_season": 0.250,
            "xba_minus_ba_gap": 0.0,
            "babip_deviation": 0.0,
            "sweet_spot_pct": 0.340,
        }

    try:
        pitcher = _pitcher_opposing_features(pitcher_id, season, _mlb, _db)
    except Exception as exc:
        logger.error("pitcher features failed pitcher_id=%s: %s", pitcher_id, exc)
        pitcher = {
            "pitcher_babip_allowed_30d": DEFAULT_RATE,
            "pitcher_hit_rate_allowed_season": DEFAULT_RATE,
            "pitcher_k_rate_season": DEFAULT_RATE,
            "pitcher_gb_pct": 0.44,
        }

    try:
        park = _park_factor_features(venue_id, batter_hand, _db, _mlb)
    except Exception as exc:
        logger.error("park factor features failed venue_id=%s: %s", venue_id, exc)
        park = {"park_factor_hits_h": DEFAULT_PARK_FACTOR}

    try:
        batter_hand_str = mlb_client.get_player_info(batter_id).get("batSide", "R") if mlb_client else "R"
        discipline = _batter_discipline_features(batter_id, season, _db, _mlb)
    except Exception as exc:
        logger.error("discipline features failed batter_id=%s: %s", batter_id, exc)
        discipline = {"batter_k_rate_season": DEFAULT_RATE, "batter_walk_rate_season": DEFAULT_RATE,
                      "chase_rate_30d": DEFAULT_RATE, "sprint_speed": 27.0}

    opp_xwoba = _opp_lineup_xwoba(lineup_ids or [], _db)

    try:
        advanced_pitcher = _pitcher_advanced_features(pitcher_id, season, _db, _mlb)
    except Exception as exc:
        logger.error("pitcher advanced features failed: %s", exc)
        advanced_pitcher = {
            "pitcher_swstr_season": 0.105,
            "pitcher_fastball_pct_season": 0.52,
            "opposing_team_der_season": 0.700,
        }

    try:
        game_ctx = _game_context_features(lineup_spot, pitcher_id, season, _mlb, _weather, venue_id, game_time_utc)
    except Exception as exc:
        logger.error("game context features failed: %s", exc)
        game_ctx = {"expected_tto_number": 2.0, "game_temp_f": 75.0, "wind_direction_category": 0.0}

    features: dict[str, Any] = {
        **contact,
        **pitcher,
        **park,
        **discipline,
        **_handedness_features(batter_hand, pitcher_hand),
        "lineup_spot": int(lineup_spot),
        "batting_order_position": int(lineup_spot),
        "opp_lineup_xwoba": opp_xwoba,
        "line_movement": line_movement,
        **_market_features(market_line, market_odds),
        **advanced_pitcher,
        **game_ctx,
    }

    try:
        game_info = _mlb.get_game(game_pk)
        game_date = str(game_info.get("gameDate", ""))[:10]
    except Exception as exc:
        logger.warning("Could not fetch game date for game_pk=%s: %s", game_pk, exc)
        game_date = ""

    if game_date:
        try:
            _db.save_features(batter_id, game_date, PROP_TYPE, features)
        except Exception as exc:
            logger.error("Failed to save features batter_id=%s game_pk=%s: %s", batter_id, game_pk, exc)

    return features


def build_hits_training_features(
    season_games: list[dict[str, Any]],
    db: Optional[MLBPropsDB] = None,
    mlb_client: Optional[MLBClient] = None,
) -> list[dict[str, Any]]:
    """Build the historical training feature set with ``actual_hits`` labels.

    Each element of ``season_games`` must contain: ``game_pk``, ``batter_id``,
    ``pitcher_id``, ``venue_id``, ``batter_hand``, ``pitcher_hand``,
    ``lineup_spot``, ``market_line``, ``market_odds``, ``actual_hits``.
    Rows that raise errors are skipped and logged.
    """
    _db = db or get_db()
    _mlb = mlb_client or get_client()
    training_rows: list[dict[str, Any]] = []

    for idx, game in enumerate(season_games):
        try:
            features = compute_hits_features(
                batter_id=game["batter_id"],
                pitcher_id=game["pitcher_id"],
                game_pk=game["game_pk"],
                venue_id=game["venue_id"],
                batter_hand=game["batter_hand"],
                pitcher_hand=game["pitcher_hand"],
                lineup_spot=game["lineup_spot"],
                market_line=game["market_line"],
                market_odds=game["market_odds"],
                db=_db,
                mlb_client=_mlb,
                game_time_utc=game.get("game_time_utc", ""),
            )
        except KeyError as exc:
            logger.warning("Skipping game %d (game_pk=%s): missing field %s", idx, game.get("game_pk"), exc)
            continue
        except Exception as exc:
            logger.error("Error processing game %d (game_pk=%s): %s", idx, game.get("game_pk"), exc)
            continue

        training_rows.append({
            **features,
            "actual_hits": int(game["actual_hits"]),
            "game_date": game.get("game_date", ""),
            "market_odds": game.get("market_odds"),
        })

    logger.info(
        "build_hits_training_features: %d rows from %d input games",
        len(training_rows), len(season_games),
    )
    return training_rows


def get_hits_feature_matrix(
    game_pks: list[int],
    db: Optional[MLBPropsDB] = None,
    matchups: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Load saved Hits features for a list of game PKs.

    ``matchups`` should be a list of dicts with keys ``game_pk``, ``batter_id``,
    and ``game_date``.  Without matchups an empty list is returned (logged).
    """
    _db = db or get_db()
    game_pk_set = set(game_pks)
    results: list[dict[str, Any]] = []

    if not matchups:
        logger.warning(
            "get_hits_feature_matrix: no matchups provided — cannot resolve features. "
            "Pass matchups=[{game_pk, batter_id, game_date}, ...]."
        )
        return results

    for matchup in matchups:
        if matchup.get("game_pk") not in game_pk_set:
            continue
        batter_id = matchup.get("batter_id")
        feat_date = str(matchup.get("game_date", ""))[:10]
        if not batter_id or not feat_date:
            logger.warning("Skipping matchup with missing batter_id or game_date: %s", matchup)
            continue
        try:
            feat = _db.get_features(batter_id, feat_date, PROP_TYPE)
        except Exception as exc:
            logger.error("get_features failed batter_id=%s date=%s: %s", batter_id, feat_date, exc)
            feat = None
        if feat is not None:
            results.append({**feat, "game_pk": matchup["game_pk"], "batter_id": batter_id})
        else:
            logger.warning(
                "No saved features batter_id=%s date=%s game_pk=%s",
                batter_id, feat_date, matchup.get("game_pk"),
            )

    return results
