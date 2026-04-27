"""
matchup_context.py — Runtime matchup adjustments for prop model probabilities.

Provides adjusted lambdas/probabilities for three markets based on the
specific pitcher vs batter matchup:

  K model:    adjusted Poisson λ  = pitcher K/9 × avg_IP × lineup_whiff_factor × umpire_factor
  HR model:   adjusted Poisson λ  = _MLB_HR_LAMBDA × synthetic BvP factor (pitch-type weighted)
  Hits model: adjusted binomial p = _MLB_BA × pitcher_contact_factor × batter_contact_factor

All functions degrade gracefully to league-average when data is missing.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from lineup_whiff_matchup import (
    LEAGUE_AVG_LINEUP_K_PER_6IP,
    compute_lineup_k_matchup,
)
from pitch_type_matchup import compute_synthetic_bvp_hr

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# League-average constants (used as fallbacks)
# ---------------------------------------------------------------------------

_MLB_BA = 0.255
_MLB_HR_LAMBDA = 0.12
_MLB_K9_LEAGUE = 9.0       # league-average K/9 in 2024
_MLB_AVG_IP = 5.75         # average SP innings pitched per start
_MLB_K_LAMBDA_DEFAULT = (_MLB_K9_LEAGUE / 9.0) * _MLB_AVG_IP  # ≈ 5.75

# Per-player adjustment caps — prevents extreme model swings from small samples
_K_LAMBDA_MIN = 1.5
_K_LAMBDA_MAX = 12.0
_HR_LAMBDA_MIN = 0.02
_HR_LAMBDA_MAX = 0.50
_HIT_PROB_MIN = 0.14
_HIT_PROB_MAX = 0.42

# Minimum innings pitched in season before we trust K/9 (avoids openers with 3 IP)
_MIN_IP_SEASON = 10.0

# Minimum season PA before we trust batter BA
_MIN_PA_SEASON = 30

# Contact factor caps for hits model
_PITCHER_CONTACT_FACTOR_BOUNDS = (0.65, 1.35)
_BATTER_CONTACT_FACTOR_BOUNDS = (0.75, 1.45)


# ---------------------------------------------------------------------------
# Player stat helpers (cached within a process lifetime)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _get_pitcher_season_stats(pitcher_id: int, season: int, mlb_client: Any) -> dict:
    """Fetch and cache pitcher season stats. Returns {} on failure."""
    try:
        stats = mlb_client.get_player_season_stats(pitcher_id, season, group="pitching")
        return stats or {}
    except Exception as exc:
        log.debug("Could not fetch pitcher season stats for %d: %s", pitcher_id, exc)
        return {}


@lru_cache(maxsize=256)
def _get_batter_season_stats(batter_id: int, season: int, mlb_client: Any) -> dict:
    """Fetch and cache batter season stats. Returns {} on failure."""
    try:
        stats = mlb_client.get_player_season_stats(batter_id, season, group="hitting")
        return stats or {}
    except Exception as exc:
        log.debug("Could not fetch batter season stats for %d: %s", batter_id, exc)
        return {}


def _pitcher_k9(pitcher_id: int, season: int, mlb_client: Any) -> float | None:
    """Return pitcher K/9 for the current season; None if insufficient data."""
    stats = _get_pitcher_season_stats(pitcher_id, season, mlb_client)
    if not stats:
        return None

    try:
        ip = float(stats.get("inningsPitched") or 0)
    except (TypeError, ValueError):
        ip = 0.0

    if ip < _MIN_IP_SEASON:
        return None

    # Try pre-computed value first
    try:
        k9 = float(stats.get("strikeoutsPer9Inn") or 0)
        if k9 > 0:
            return k9
    except (TypeError, ValueError):
        pass

    # Compute from raw counts
    try:
        ks = float(stats.get("strikeOuts") or 0)
        return (ks / ip * 9.0) if ip > 0 else None
    except (TypeError, ValueError):
        return None


def _pitcher_avg_ip(pitcher_id: int, season: int, mlb_client: Any) -> float:
    """Return pitcher's average innings per start this season; fallback _MLB_AVG_IP."""
    stats = _get_pitcher_season_stats(pitcher_id, season, mlb_client)
    if not stats:
        return _MLB_AVG_IP
    try:
        gs = float(stats.get("gamesStarted") or 0)
        ip = float(stats.get("inningsPitched") or 0)
        if gs > 0:
            return ip / gs
    except (TypeError, ValueError):
        pass
    return _MLB_AVG_IP


def _batter_season_ba(batter_id: int, season: int, mlb_client: Any) -> float | None:
    """Return batter's season batting average; None if insufficient PA."""
    stats = _get_batter_season_stats(batter_id, season, mlb_client)
    if not stats:
        return None
    try:
        pa = float(stats.get("plateAppearances") or stats.get("atBats") or 0)
        if pa < _MIN_PA_SEASON:
            return None
        return float(stats.get("avg") or 0) or None
    except (TypeError, ValueError):
        return None


def batter_hr_features(
    batter_id: int,
    season: int,
    mlb_client: Any,
    lineup_spot: float = 4.0,
) -> dict:
    """Return HR model feature dict populated with real batter stats where available.

    MLB Stats API provides HR rate (hr/ab). Savant features (barrel_rate, xiso, etc.)
    fall back to league averages since we don't have per-batter Savant data at
    prediction time.

    Parameters
    ----------
    batter_id  : MLBAM player ID.
    season     : Current MLB season year.
    mlb_client : MLBClient instance.
    lineup_spot: Batting order position (1-9), used as a proxy for lineup quality.

    Returns
    -------
    dict with all HR_FEATURE_COLS filled.
    """
    # League-average defaults
    hr_rate = 0.030        # ~3 HR/100 AB
    slg = 0.400            # league SLG proxy for xiso
    hard_hit = 0.385       # league hard-hit rate
    barrel_rate = 0.070    # league barrel rate

    stats = _get_batter_season_stats(batter_id, season, mlb_client)
    if stats:
        try:
            ab = float(stats.get("atBats") or 0)
            if ab >= _MIN_PA_SEASON:
                hr = float(stats.get("homeRuns") or 0)
                hr_rate = hr / ab if ab > 0 else hr_rate

                slg_val = float(stats.get("sluggingPercentage") or 0)
                if slg_val > 0:
                    # Rough xISO proxy: SLG - BA
                    ba_val = float(stats.get("avg") or 0)
                    slg = max(0.0, slg_val - ba_val)
        except (TypeError, ValueError):
            pass
        log.debug(
            "HR features: batter_id=%d hr_rate=%.3f slg=%.3f",
            batter_id, hr_rate, slg,
        )

    return {
        "barrel_rate_30d":        barrel_rate,
        "barrel_rate_60d":        barrel_rate,
        "hard_hit_rate_30d":      hard_hit,
        "xiso_30d":               slg,
        "avg_launch_angle_30d":   10.0,
        "hr_rate_season":         hr_rate,
        "pull_pct_30d":           0.40,
        "park_factor_h":          1.0,
        "weather_hr_multiplier":  1.0,
        "pitcher_hr_rate_season": 0.120,
        "pitcher_gb_pct":         0.44,
        "batter_hand_vs_pitcher": 0.0,
        "is_platoon_advantage":   0.0,
        "lineup_spot":            float(lineup_spot),
        "bvp_factor":             1.0,
        "market_implied_prob":    0.5,  # caller should override with actual market
    }


# ---------------------------------------------------------------------------
# K model adjustment
# ---------------------------------------------------------------------------

def adjusted_k_lambda(
    pitcher_id: int,
    opposing_lineup: list[dict],
    umpire_factor: float,
    db: Any,
    mlb_client: Any,
    season: int,
) -> float:
    """Return an adjusted Poisson λ for pitcher strikeout props.

    Formula:
        λ = pitcher_k9/9 × avg_ip × lineup_whiff_factor × umpire_factor

    Parameters
    ----------
    pitcher_id:
        MLBAM ID of the starting pitcher.
    opposing_lineup:
        List of {batter_id, lineup_spot} dicts for batters facing this pitcher.
    umpire_factor:
        K-factor from todays_umpires (1.0 = neutral).
    db, mlb_client:
        Standard service objects.
    season:
        Current MLB season year.

    Returns
    -------
    float
        Adjusted λ, clamped to [_K_LAMBDA_MIN, _K_LAMBDA_MAX].
    """
    # --- 1. Pitcher baseline ---
    k9 = _pitcher_k9(pitcher_id, season, mlb_client)
    if k9 is None:
        log.debug("No K/9 for pitcher_id=%d — using league average λ", pitcher_id)
        base_lambda = _MLB_K_LAMBDA_DEFAULT
    else:
        avg_ip = _pitcher_avg_ip(pitcher_id, season, mlb_client)
        base_lambda = (k9 / 9.0) * avg_ip
        log.debug("K base λ: pitcher_id=%d k9=%.1f avg_ip=%.1f → %.2f",
                  pitcher_id, k9, avg_ip, base_lambda)

    # --- 2. Lineup whiff matchup ---
    matchup_factor = 1.0
    if opposing_lineup:
        try:
            result = compute_lineup_k_matchup(pitcher_id, opposing_lineup, db)
            matchup_factor = result.get("matchup_factor", 1.0)
            log.debug(
                "K lineup matchup: pitcher_id=%d factor=%.3f quality=%s",
                pitcher_id, matchup_factor, result.get("data_quality"),
            )
        except Exception as exc:
            log.warning("lineup_k_matchup failed pitcher_id=%d: %s", pitcher_id, exc)

    # --- 3. Umpire ---
    ump = umpire_factor if umpire_factor and umpire_factor > 0 else 1.0

    adjusted = base_lambda * matchup_factor * ump
    clamped = max(_K_LAMBDA_MIN, min(_K_LAMBDA_MAX, adjusted))

    if clamped != adjusted:
        log.debug("K λ clamped: raw=%.2f → %.2f", adjusted, clamped)

    log.info(
        "K λ adjustment: pitcher_id=%d base=%.2f × matchup=%.3f × ump=%.3f = %.2f",
        pitcher_id, base_lambda, matchup_factor, ump, clamped,
    )
    return clamped


# ---------------------------------------------------------------------------
# HR model adjustment
# ---------------------------------------------------------------------------

def adjusted_hr_lambda(
    batter_id: int,
    pitcher_id: int,
    db: Any,
) -> float:
    """Return an adjusted Poisson λ for batter home-run props.

    Uses the synthetic BvP factor from pitch_type_matchup:
        λ = _MLB_HR_LAMBDA × bvp_factor

    The BvP factor weights each pitch type by:
        pitcher_mix[pt] × batter_fb_pct_vs[pt] × batter_hr_per_fb_vs[pt]
    normalised to the league-average HR rate per 100 PA.

    Parameters
    ----------
    batter_id, pitcher_id:
        MLBAM player IDs.
    db:
        MLBPropsDB instance.

    Returns
    -------
    float
        Adjusted λ, clamped to [_HR_LAMBDA_MIN, _HR_LAMBDA_MAX].
    """
    try:
        result = compute_synthetic_bvp_hr(batter_id, pitcher_id, db)
        bvp_factor = result.get("bvp_factor", 1.0)
        quality = result.get("data_quality", "fallback")
        log.debug(
            "HR BvP: batter=%d vs pitcher=%d factor=%.3f quality=%s",
            batter_id, pitcher_id, bvp_factor, quality,
        )
    except Exception as exc:
        log.warning(
            "compute_synthetic_bvp_hr failed batter=%d pitcher=%d: %s",
            batter_id, pitcher_id, exc,
        )
        bvp_factor = 1.0

    adjusted = _MLB_HR_LAMBDA * bvp_factor
    clamped = max(_HR_LAMBDA_MIN, min(_HR_LAMBDA_MAX, adjusted))

    log.info(
        "HR λ adjustment: batter_id=%d vs pitcher_id=%d bvp=%.3f → λ=%.4f",
        batter_id, pitcher_id, bvp_factor, clamped,
    )
    return clamped


# ---------------------------------------------------------------------------
# Hits model adjustment
# ---------------------------------------------------------------------------

def adjusted_hit_prob(
    batter_id: int,
    pitcher_id: int,
    db: Any,
    mlb_client: Any,
    season: int,
) -> float:
    """Return an adjusted hit-probability p for the Binomial hits model.

    Formula:
        p = _MLB_BA × pitcher_contact_factor × batter_contact_factor

    Pitcher contact factor:
        league_k9 / pitcher_k9  (lower = pitcher allows more contact)
        Capped to _PITCHER_CONTACT_FACTOR_BOUNDS.

    Batter contact factor:
        batter_ba / _MLB_BA
        Capped to _BATTER_CONTACT_FACTOR_BOUNDS.

    Parameters
    ----------
    batter_id, pitcher_id:
        MLBAM player IDs.
    db:
        MLBPropsDB instance (reserved for future feature lookups).
    mlb_client:
        MLBClient instance for season-stat fetches.
    season:
        Current MLB season year.

    Returns
    -------
    float
        Adjusted p, clamped to [_HIT_PROB_MIN, _HIT_PROB_MAX].
    """
    # --- Pitcher contact factor ---
    pitcher_k9 = _pitcher_k9(pitcher_id, season, mlb_client)
    if pitcher_k9 and pitcher_k9 > 0:
        raw_pitcher_factor = _MLB_K9_LEAGUE / pitcher_k9
        lo, hi = _PITCHER_CONTACT_FACTOR_BOUNDS
        pitcher_factor = max(lo, min(hi, raw_pitcher_factor))
        log.debug(
            "Hits pitcher factor: pitcher_id=%d k9=%.1f → factor=%.3f",
            pitcher_id, pitcher_k9, pitcher_factor,
        )
    else:
        pitcher_factor = 1.0

    # --- Batter contact factor ---
    batter_ba = _batter_season_ba(batter_id, season, mlb_client)
    if batter_ba and batter_ba > 0:
        raw_batter_factor = batter_ba / _MLB_BA
        lo, hi = _BATTER_CONTACT_FACTOR_BOUNDS
        batter_factor = max(lo, min(hi, raw_batter_factor))
        log.debug(
            "Hits batter factor: batter_id=%d ba=%.3f → factor=%.3f",
            batter_id, batter_ba, batter_factor,
        )
    else:
        batter_factor = 1.0

    adjusted = _MLB_BA * pitcher_factor * batter_factor
    clamped = max(_HIT_PROB_MIN, min(_HIT_PROB_MAX, adjusted))

    log.info(
        "Hits p adjustment: batter_id=%d vs pitcher_id=%d pitcher_f=%.3f batter_f=%.3f → p=%.3f",
        batter_id, pitcher_id, pitcher_factor, batter_factor, clamped,
    )
    return clamped
