#!/usr/bin/env python3
"""
Pitch-Type Matchup Module

Computes a Synthetic Batter vs Pitcher (BvP) HR factor by combining a
pitcher's pitch-type mix with a batter's per-pitch-type fly-ball and
HR/flyball tendencies.  Avoids small-sample real BvP data.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Dict, List

import pybaseball

from db import get_db

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEAGUE_AVG_HR_PER_100PA: float = 3.0

LEAGUE_AVG_BY_PITCH: Dict[str, Dict[str, float]] = {
    "FF":    {"fb_pct": 32.0, "hr_per_fb": 11.5},   # Four-seam fastball
    "SI":    {"fb_pct": 18.0, "hr_per_fb":  8.0},   # Sinker
    "SL":    {"fb_pct": 22.0, "hr_per_fb": 10.0},   # Slider
    "CU":    {"fb_pct": 28.0, "hr_per_fb":  9.5},   # Curveball
    "CH":    {"fb_pct": 28.0, "hr_per_fb": 12.0},   # Changeup
    "KC":    {"fb_pct": 25.0, "hr_per_fb":  8.5},   # Knuckle-curve
    "FC":    {"fb_pct": 30.0, "hr_per_fb": 10.5},   # Cutter
    "FS":    {"fb_pct": 24.0, "hr_per_fb":  9.0},   # Splitter
    "OTHER": {"fb_pct": 25.0, "hr_per_fb": 10.0},
}

_MIN_PITCHES_FOR_MIX = 20  # minimum pitch count to trust a DB pitch-mix row
_MIN_PA_FOR_BATTER   = 10  # minimum PA to trust batter per-pitch-type stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _date_window(days: int) -> tuple[str, str]:
    """Return (start_str, end_str) for a lookback window of *days* days."""
    end = date.today()
    start = end - timedelta(days=days)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _normalize_mix(raw: Dict[str, float]) -> Dict[str, float]:
    """Return a new dict with values scaled so they sum to 1.0."""
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in raw.items()}


def _pitch_mix_from_statcast(pitcher_id: int, days: int) -> Dict[str, float]:
    """Fetch pitch mix from pybaseball as a DB fallback."""
    start_str, end_str = _date_window(days)
    try:
        df = pybaseball.statcast_pitcher(start_str, end_str, player_id=pitcher_id)
    except Exception as exc:
        log.warning("pybaseball statcast_pitcher failed for pitcher_id=%s: %s", pitcher_id, exc)
        return {}

    if df is None or df.empty or "pitch_type" not in df.columns:
        return {}

    counts: Dict[str, int] = df["pitch_type"].dropna().value_counts().to_dict()
    total = sum(counts.values())
    if total < _MIN_PITCHES_FOR_MIX:
        log.debug("Insufficient pitch count (%d) for pitcher_id=%s fallback", total, pitcher_id)
        return {}

    return _normalize_mix({str(k): float(v) for k, v in counts.items()})


# ---------------------------------------------------------------------------
# Public data-access functions
# ---------------------------------------------------------------------------

def get_pitcher_pitch_mix(pitcher_id: int, db, days: int = 60) -> Dict[str, float]:
    """Return pitcher's pitch-type distribution as ``{pitch_type: fraction}``.

    Queries ``pitch_type_performance`` (player_type='pitcher') first; falls back
    to a live pybaseball fetch if DB data is absent or below _MIN_PITCHES_FOR_MIX.
    Returns an empty dict on complete failure.
    """
    try:
        with db._connect() as conn:
            rows = conn.execute(
                """
                SELECT pitch_type, pct_thrown_seen, pa_or_pitches
                  FROM pitch_type_performance
                 WHERE player_id = ?
                   AND player_type = 'pitcher'
                """,
                (str(pitcher_id),),
            ).fetchall()
    except Exception as exc:
        log.warning("DB query failed for pitcher pitch mix (pitcher_id=%s): %s", pitcher_id, exc)
        rows = []

    if rows:
        total_pitches = sum(r["pa_or_pitches"] or 0 for r in rows)
        if total_pitches >= _MIN_PITCHES_FOR_MIX:
            raw = {
                r["pitch_type"]: float(r["pct_thrown_seen"] or 0.0)
                for r in rows
                if r["pct_thrown_seen"] is not None
            }
            if raw:
                return _normalize_mix(raw)

    log.debug("DB pitch mix insufficient for pitcher_id=%s; falling back to pybaseball", pitcher_id)
    return _pitch_mix_from_statcast(pitcher_id, days)


def get_batter_pitch_type_stats(
    batter_id: int,
    db,
    days: int = 60,  # noqa: ARG001 — reserved for future time-windowed queries
) -> Dict[str, Dict[str, float]]:
    """Return batter's per-pitch-type stats from ``pitch_type_performance``.

    Returns ``{pitch_type: {'fb_pct': float, 'hr_per_fb': float, 'barrel_rate': float}}``.
    Rows with fewer than _MIN_PA_FOR_BATTER plate appearances are skipped.
    League averages are substituted for any null column values.
    Returns empty dict when no qualifying DB rows exist.
    """
    try:
        with db._connect() as conn:
            rows = conn.execute(
                """
                SELECT pitch_type, fb_pct, hr_per_fb, barrel_pct, pa_or_pitches
                  FROM pitch_type_performance
                 WHERE player_id = ?
                   AND player_type = 'batter'
                """,
                (str(batter_id),),
            ).fetchall()
    except Exception as exc:
        log.warning("DB query failed for batter pitch type stats (batter_id=%s): %s", batter_id, exc)
        rows = []

    result: Dict[str, Dict[str, float]] = {}
    for row in rows:
        if (row["pa_or_pitches"] or 0) < _MIN_PA_FOR_BATTER:
            continue
        league = LEAGUE_AVG_BY_PITCH.get(row["pitch_type"], LEAGUE_AVG_BY_PITCH["OTHER"])
        result = {
            **result,
            row["pitch_type"]: {
                "fb_pct":      float(row["fb_pct"])     if row["fb_pct"]     is not None else league["fb_pct"],
                "hr_per_fb":   float(row["hr_per_fb"])  if row["hr_per_fb"]  is not None else league["hr_per_fb"],
                "barrel_rate": float(row["barrel_pct"]) if row["barrel_pct"] is not None else 0.0,
            },
        }
    return result


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_synthetic_bvp_hr(
    batter_id: int,
    pitcher_id: int,
    db,
    batter_days: int = 60,
    pitcher_days: int = 60,
) -> Dict[str, Any]:
    """Compute a synthetic BvP HR factor for one batter/pitcher matchup.

    Algorithm (per pitch type in pitcher's mix):
        pa_to_flyball        = pitcher_mix[pitch] * (batter_fb_pct_vs[pitch] / 100)
        fb_to_hr             = batter_hr_per_fb_vs[pitch] / 100
        expected_hr_per_pa  += pa_to_flyball * fb_to_hr
        expected_hr_per_100pa = expected_hr_per_pa * 100
        bvp_factor            = expected_hr_per_100pa / LEAGUE_AVG_HR_PER_100PA

    Returns a dict with keys:
        expected_hr_per_100pa, bvp_factor, pitch_types_used, data_quality
    where data_quality is 'full', 'partial', or 'fallback'.
    """
    pitcher_mix  = get_pitcher_pitch_mix(pitcher_id, db, days=pitcher_days)
    batter_stats = get_batter_pitch_type_stats(batter_id, db, days=batter_days)

    has_pitcher_data = bool(pitcher_mix)
    has_batter_data  = bool(batter_stats)

    if not has_pitcher_data:
        pitch_keys  = [k for k in LEAGUE_AVG_BY_PITCH if k != "OTHER"]
        pitcher_mix = _normalize_mix({k: 1.0 for k in pitch_keys})

    expected_hr_per_pa  = 0.0
    pitch_types_used: List[str] = []

    for pitch_type, mix_pct in pitcher_mix.items():
        if mix_pct <= 0.0:
            continue

        if pitch_type in batter_stats:
            stats = batter_stats[pitch_type]
        else:
            league_key = pitch_type if pitch_type in LEAGUE_AVG_BY_PITCH else "OTHER"
            stats = {
                "fb_pct":    LEAGUE_AVG_BY_PITCH[league_key]["fb_pct"],
                "hr_per_fb": LEAGUE_AVG_BY_PITCH[league_key]["hr_per_fb"],
            }

        pa_to_flyball       = mix_pct * (stats["fb_pct"] / 100.0)
        fb_to_hr            = stats["hr_per_fb"] / 100.0
        expected_hr_per_pa += pa_to_flyball * fb_to_hr
        pitch_types_used.append(pitch_type)

    expected_hr_per_100pa = expected_hr_per_pa * 100.0
    bvp_factor = (
        expected_hr_per_100pa / LEAGUE_AVG_HR_PER_100PA
        if LEAGUE_AVG_HR_PER_100PA > 0 else 1.0
    )

    if has_pitcher_data and has_batter_data:
        data_quality = "full"
    elif has_pitcher_data or has_batter_data:
        data_quality = "partial"
    else:
        data_quality = "fallback"

    return {
        "expected_hr_per_100pa": round(expected_hr_per_100pa, 4),
        "bvp_factor":            round(bvp_factor, 4),
        "pitch_types_used":      sorted(pitch_types_used),
        "data_quality":          data_quality,
    }


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------

def batch_compute_bvp(
    matchups: List[Dict[str, Any]],
    db=None,
) -> List[Dict[str, Any]]:
    """Compute synthetic BvP HR factors for a list of matchups.

    Each matchup dict must contain batter_id, pitcher_id, and game_pk.
    Returns the input dicts augmented with keys from compute_synthetic_bvp_hr.
    Logs the count of matchups that fell back to league-average data.
    """
    database = db if db is not None else get_db()
    results: List[Dict[str, Any]] = []
    fallback_count = 0

    for matchup in matchups:
        batter_id  = matchup.get("batter_id")
        pitcher_id = matchup.get("pitcher_id")
        game_pk    = matchup.get("game_pk")

        if batter_id is None or pitcher_id is None:
            log.warning("Skipping matchup with missing batter_id or pitcher_id: %s", matchup)
            continue

        try:
            bvp = compute_synthetic_bvp_hr(batter_id, pitcher_id, database)
        except Exception as exc:
            log.error(
                "compute_synthetic_bvp_hr failed for batter=%s pitcher=%s game=%s: %s",
                batter_id, pitcher_id, game_pk, exc,
            )
            bvp = {
                "expected_hr_per_100pa": LEAGUE_AVG_HR_PER_100PA,
                "bvp_factor":            1.0,
                "pitch_types_used":      [],
                "data_quality":          "fallback",
            }

        if bvp["data_quality"] == "fallback":
            fallback_count += 1

        results.append({**matchup, **bvp})

    log.info(
        "batch_compute_bvp: %d matchups processed, %d used fallback data",
        len(results), fallback_count,
    )
    return results
