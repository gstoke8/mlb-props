#!/usr/bin/env python3
"""
train_models.py — One-time historical model trainer.

Builds labeled training data from 2024 season Statcast + MLB API game logs,
then trains KModel (Poisson GLM), HitsModel (Logistic), and HRModel (Logistic).

Data sources:
  - Baseball Savant via pybaseball: season-level K%, whiff%, barrel%, xwOBA, pitch mix
  - MLB Stats API: per-game outcomes (actual Ks, hits, HRs) and game context

Output: ~/mlb-props/models/{k,hits,hr}_model.pkl

Usage:
    python3 train_models.py [--season 2024] [--min-ip 20] [--min-pa 100]
    python3 train_models.py --dry-run     # build data, skip save
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pybaseball
from sklearn.model_selection import train_test_split

from db import get_db
from mlb_api import get_client
from k_model import KModel, K_FEATURE_COLS
from hits_model import HitsModel, HITS_FEATURE_COLS
from hr_model import HRModel, HR_FEATURE_COLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

pybaseball.cache.enable()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INTER_PLAYER_SLEEP = 0.35   # seconds between MLB API calls
LEAGUE_K_PCT = 0.224        # 2024 MLB average strikeout rate
LEAGUE_BB_PCT = 0.085
LEAGUE_BA = 0.243
MODELS_DIR = Path.home() / "mlb-props" / "models"

# ---------------------------------------------------------------------------
# Savant bulk fetches (one call each)
# ---------------------------------------------------------------------------

def _fetch_savant_data(season: int) -> dict[str, Any]:
    """Fetch all needed Savant season-level stats in bulk."""
    log.info("Fetching Savant pitcher percentile ranks for %d…", season)
    pitcher_pct = pybaseball.statcast_pitcher_percentile_ranks(season)

    log.info("Fetching Savant pitcher expected stats for %d…", season)
    pitcher_exp = pybaseball.statcast_pitcher_expected_stats(season, minPA=30)

    log.info("Fetching Savant pitcher pitch arsenal (mix %%) for %d…", season)
    pitcher_mix = pybaseball.statcast_pitcher_pitch_arsenal(season, minP=50, arsenal_type="n_")

    # Per-pitch-type whiff% not available via pybaseball arsenal endpoint.
    # Training uses overall whiff_pct as proxy for ff/sl/ch; real-time
    # predictions use actual Statcast per-pitch data from the nightly runner.
    pitcher_whiff = None

    log.info("Fetching Savant pitcher exit-velo/barrels for %d…", season)
    pitcher_ev = pybaseball.statcast_pitcher_exitvelo_barrels(season, minBBE=20)

    log.info("Fetching Savant batter percentile ranks for %d…", season)
    batter_pct = pybaseball.statcast_batter_percentile_ranks(season)

    log.info("Fetching Savant batter expected stats for %d…", season)
    batter_exp = pybaseball.statcast_batter_expected_stats(season, minPA=50)

    log.info("Fetching Savant batter exit-velo/barrels for %d…", season)
    batter_ev = pybaseball.statcast_batter_exitvelo_barrels(season, minBBE=20)

    log.info("Fetching Savant sprint speed for %d…", season)
    try:
        sprint_df = pybaseball.statcast_sprint_speed(season)
    except Exception as exc:
        log.warning("sprint speed fetch failed (%s); using empty fallback", exc)
        sprint_df = None

    # Index by player_id for O(1) lookup
    def _idx(df, id_col="player_id"):
        if df is None or df.empty:
            return {}
        return {int(row[id_col]): dict(row) for _, row in df.iterrows()}

    # Sprint speed indexed by player_id; column may be 'player_id' or 'mlbam_id'
    sprint_map: dict[int, float] = {}
    if sprint_df is not None and not sprint_df.empty:
        id_col = "player_id" if "player_id" in sprint_df.columns else "mlbam_id"
        for _, row in sprint_df.iterrows():
            try:
                pid = int(row[id_col])
                speed = float(row.get("sprint_speed", row.get("hp_to_1b", 27.0)))
                if not np.isfinite(speed):
                    speed = 27.0
                sprint_map[pid] = speed
            except (TypeError, ValueError, KeyError):
                continue
    log.info("Sprint speed entries loaded: %d", len(sprint_map))

    return {
        "pitcher_pct":    _idx(pitcher_pct),
        "pitcher_exp":    _idx(pitcher_exp),
        "pitcher_mix":    _idx(pitcher_mix, id_col="pitcher"),
        "pitcher_whiff":  _idx(pitcher_whiff, id_col="pitcher") if pitcher_whiff is not None and not pitcher_whiff.empty else {},
        "pitcher_ev":     _idx(pitcher_ev),
        "batter_pct":     _idx(batter_pct),
        "batter_exp":     _idx(batter_exp),
        "batter_ev":      _idx(batter_ev),
        "sprint_map":     sprint_map,  # int → float (ft/s)
    }


# ---------------------------------------------------------------------------
# Safe numeric helpers
# ---------------------------------------------------------------------------

def _sf(val, default: float = 0.0) -> float:
    """Convert value to float, returning default on None/NaN/inf."""
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Feature builders: K model
# ---------------------------------------------------------------------------

def _build_k_rows(
    pitcher_ids: list[int],
    savant: dict,
    mlb_client,
    season: int,
    game_context_map: dict,
    db,
    min_ip: float = 20.0,
) -> list[dict]:
    """Build one training row per pitcher start."""
    rows: list[dict] = []

    for i, pid in enumerate(pitcher_ids):
        if i > 0 and i % 20 == 0:
            log.info("  K rows: processed %d/%d pitchers, %d rows so far", i, len(pitcher_ids), len(rows))

        pct = savant["pitcher_pct"].get(pid, {})
        exp = savant["pitcher_exp"].get(pid, {})
        whiff_data = savant["pitcher_whiff"].get(pid, {})

        # Season-level stats — _sf guards against pandas NaN
        k_pct_season  = _sf(pct.get("k_percent"), 22.4) / 100.0
        whiff_pct     = _sf(pct.get("whiff_percent"), 24.0) / 100.0
        xwoba_allowed = _sf(exp.get("est_woba"), 0.320)

        # SwStr% ≈ 70% of overall whiff% (swStr counts pitches, whiff counts swings)
        swstr_rate = whiff_pct * 0.70

        # Per-pitch-type whiff rates — try multiple column name formats
        def _pt_whiff(pt: str) -> float:
            for key in (f"{pt}_whiff_pct", f"whiff_pct_{pt}", f"{pt}_avg_pct", pt):
                v = whiff_data.get(key)
                if v is not None:
                    return _sf(v, whiff_pct) / (100.0 if _sf(v, 0) > 1.0 else 1.0)
            return whiff_pct

        ff_whiff = _pt_whiff("ff")
        sl_whiff = _pt_whiff("sl")
        ch_whiff = _pt_whiff("ch")

        try:
            game_log = mlb_client.get_player_game_log(pid, season, group="pitching")
            time.sleep(INTER_PLAYER_SLEEP)
        except Exception as exc:
            log.debug("Game log failed for pitcher %d: %s", pid, exc)
            continue

        if not game_log:
            continue

        # Filter to starts only (inningsPitched >= 3.0, gamesStarted == 1)
        starts = [
            g for g in game_log
            if g.get("gamesStarted", 0) == 1
            and _parse_ip(g.get("inningsPitched", "0")) >= 3.0
        ]

        if len(starts) < 3:
            continue

        # Sort oldest → newest for rolling window computation
        starts_sorted = sorted(starts, key=lambda g: g.get("date", ""))

        for idx, start in enumerate(starts_sorted):
            actual_ks = int(start.get("strikeOuts") or 0)
            ip = _parse_ip(start.get("inningsPitched", "0"))
            is_home = int(bool(start.get("is_home", False)))

            # Rolling avg IP from previous 3 starts (or season proxy)
            prior = starts_sorted[max(0, idx - 3):idx]
            avg_ip = (
                sum(_parse_ip(g.get("inningsPitched", "0")) for g in prior) / len(prior)
                if prior else ip
            )

            # Days rest
            days_rest = _days_rest(starts_sorted, idx)

            # Expected Ks = K% × batters faced (estimate: 3 × ip)
            k_rate_season = k_pct_season * 27 * (ip / 9.0) if ip > 0 else 0.0

            # Game context: umpire K factor + park K factor from game_context_map
            game_pk = start.get("game_pk")
            umpire_k_factor = 1.0
            park_k_factor = 1.0
            if game_pk and game_pk in game_context_map:
                ctx = game_context_map[game_pk]
                ump_name = ctx.get("umpire_name")
                if ump_name:
                    umpire_k_factor = db.get_umpire_k_factor_by_name(ump_name)
                venue_name = ctx.get("venue_name") or ""
                if venue_name:
                    park_row = db.get_park_factor(venue_name, "neutral")
                    if park_row and park_row.get("K_factor"):
                        park_k_factor = float(park_row["K_factor"])

            row = {
                # Core pitcher effectiveness
                "csw_rate_30d":            whiff_pct * 0.85,
                "k_rate_30d":              k_pct_season,
                "k_rate_season":           k_pct_season * 27.0,  # convert K% to K/9 proxy
                "whiff_rate_30d":          whiff_pct,
                # Pitch-type stuff metrics (v2)
                "swstr_rate_30d":          swstr_rate,
                "ff_whiff_rate_30d":       ff_whiff,
                "sl_whiff_rate_30d":       sl_whiff,
                "ch_whiff_rate_30d":       ch_whiff,
                # Opposing lineup (league-avg proxies — no per-game lineup history)
                "opp_k_rate_season":       LEAGUE_K_PCT,
                "opp_k_rate_30d":          LEAGUE_K_PCT,
                "lineup_handedness_split": 0.5,
                "opp_lineup_xwoba":        0.320,
                # Game context — real per-start values
                "umpire_k_factor":         umpire_k_factor,
                "weather_k_factor":        1.0,
                "park_k_factor":           park_k_factor,
                "is_home":                 is_home,
                "days_rest":               float(days_rest if days_rest is not None else 5.0),
                "avg_ip_30d":              avg_ip,
                "is_opener_risk":          int(avg_ip < 4.5),
                "matchup_factor":          1.0,
                "market_implied_over":     0.5,
                "line_movement":           0.0,
                # Label
                "actual_ks":               actual_ks,
            }
            rows.append(row)

    log.info("Built %d K training rows from %d pitchers", len(rows), len(pitcher_ids))
    return rows


# ---------------------------------------------------------------------------
# Feature builders: Hits model
# ---------------------------------------------------------------------------

def _pitcher_bvp_contact_factor(pitcher_id: int, batter_id: int, db) -> float:
    """Compute pitch-type BvP contact factor for a specific pitcher-batter matchup.

    Algorithm:
        weighted_whiff = sum(pitcher_mix[pt] * batter_whiff_rate_vs[pt] for pt in mix)
        contact_factor = 1 - weighted_whiff

    Higher contact_factor = batter makes more contact vs this pitcher's specific arsenal.
    Clamped to [0.50, 0.95]. Falls back to 0.776 (league avg contact rate) if no data.
    """
    try:
        from pitch_type_matchup import get_pitcher_pitch_mix
        from lineup_whiff_matchup import get_batter_whiff_rates

        pitcher_mix = get_pitcher_pitch_mix(pitcher_id, db)
        if not pitcher_mix:
            return 0.776

        whiff_rates = get_batter_whiff_rates(batter_id, db)
        weighted_whiff = sum(
            mix_frac * whiff_rates.get(pt, 0.240)
            for pt, mix_frac in pitcher_mix.items()
            if mix_frac > 0.0
        )
        contact_factor = 1.0 - weighted_whiff
        return min(max(contact_factor, 0.50), 0.95)
    except Exception:
        return 0.776


def _pitcher_gb_pct(stats: dict) -> float:
    """Compute groundball % from MLB API pitching stats dict."""
    try:
        go = float(stats.get("groundOuts") or 0)
        ao = float(stats.get("airOuts") or 0)
        total = go + ao
        return go / total if total > 0 else 0.44
    except (TypeError, ValueError):
        return 0.44


def _pitcher_stats_from_season(mlb_client, pitcher_id: int, season: int) -> dict:
    """Fetch and compute hit-rate, k-rate, BABIP for a pitcher from season stats.

    Returns dict with keys: hit_rate, k_rate, babip (all floats, league-avg fallback).
    """
    defaults = {"hit_rate": 0.243, "k_rate": LEAGUE_K_PCT, "babip": 0.295}
    try:
        stats = mlb_client.get_player_season_stats(pitcher_id, season, group="pitching")
        bf = float(stats.get("battersFaced") or 0)
        if bf < 10:
            return defaults
        hits = float(stats.get("hits") or 0)
        ks   = float(stats.get("strikeOuts") or 0)
        hrs  = float(stats.get("homeRuns") or 0)
        bbs  = float(stats.get("baseOnBalls") or 0)
        hit_rate = hits / bf
        k_rate   = ks / bf
        # BABIP = (H - HR) / (BF - K - HR - BB); denominator is balls in play
        bip = bf - ks - hrs - bbs
        babip = (hits - hrs) / bip if bip > 0 else 0.295
        # Clamp to sane ranges
        babip   = min(max(babip, 0.200), 0.400)
        k_rate  = min(max(k_rate, 0.10), 0.45)
        hit_rate = min(max(hit_rate, 0.150), 0.350)
        return {"hit_rate": hit_rate, "k_rate": k_rate, "babip": babip}
    except Exception:
        return defaults


def _build_hits_rows(
    batter_ids: list[int],
    pitcher_savant: dict,
    batter_savant: dict,
    mlb_client,
    season: int,
    game_pitcher_map: dict[int, dict],
    pitcher_stats_cache: dict[int, dict],
    min_pa: int = 100,
) -> list[dict]:
    """Build one training row per batter-game (games with 1+ AB).

    game_pitcher_map:
        {game_pk: {"home_pitcher_id": int|None, "away_pitcher_id": int|None}}
    pitcher_stats_cache:
        Mutable cache populated lazily: {pitcher_id: {hit_rate, k_rate, babip}}.
        Pre-populate with known pitchers to avoid redundant API calls.
    """
    rows: list[dict] = []

    for i, bid in enumerate(batter_ids):
        if i > 0 and i % 30 == 0:
            log.info("  Hits rows: processed %d/%d batters, %d rows so far", i, len(batter_ids), len(rows))

        bpct = batter_savant["batter_pct"].get(bid, {})
        bexp = batter_savant["batter_exp"].get(bid, {})
        bev = batter_savant["batter_ev"].get(bid, {})

        batter_ba     = _sf(bexp.get("ba"), _sf(bexp.get("est_ba"), LEAGUE_BA))
        hard_hit      = _sf(bpct.get("hard_hit_percent"), 38.0) / 100.0
        exit_velo     = _sf(bpct.get("exit_velocity"), 88.0)
        k_pct_batter  = _sf(bpct.get("k_percent"), 22.4) / 100.0
        bb_pct_batter = _sf(bpct.get("bb_percent"), 8.5) / 100.0
        contact_rate  = 1.0 - k_pct_batter
        avg_hit_angle = _sf(bev.get("avg_hit_angle"), 10.0)
        sprint_speed  = _sf(batter_savant.get("sprint_map", {}).get(bid), 27.0)

        try:
            game_log = mlb_client.get_player_game_log(bid, season, group="hitting")
            time.sleep(INTER_PLAYER_SLEEP)
        except Exception as exc:
            log.debug("Game log failed for batter %d: %s", bid, exc)
            continue

        if not game_log:
            continue

        games_with_ab = [g for g in game_log if int(g.get("atBats") or 0) >= 1]
        if len(games_with_ab) < 10:
            continue

        for g in games_with_ab:
            actual_hits = int(g.get("hits") or 0)
            game_pk = g.get("game_pk")
            is_home = g.get("is_home", True)

            # Look up opposing pitcher and venue for this specific game
            pitcher_stats = {"hit_rate": 0.243, "k_rate": LEAGUE_K_PCT, "babip": 0.295}
            opp_pitcher_id = None
            train_park_hits = 1.0
            if game_pk and game_pk in game_pitcher_map:
                gp = game_pitcher_map[game_pk]
                # Batter is home → opposing pitcher is away, and vice versa
                opp_pitcher_id = gp.get("away_pitcher_id") if is_home else gp.get("home_pitcher_id")
                if opp_pitcher_id:
                    if opp_pitcher_id not in pitcher_stats_cache:
                        pitcher_stats_cache[opp_pitcher_id] = _pitcher_stats_from_season(
                            mlb_client, opp_pitcher_id, season
                        )
                        time.sleep(INTER_PLAYER_SLEEP)
                    pitcher_stats = pitcher_stats_cache[opp_pitcher_id]
                venue_name_hits = gp.get("venue_name") or ""
                if venue_name_hits:
                    pf_row = db.get_park_factor(venue_name_hits, "neutral")
                    if pf_row and pf_row.get("H_factor"):
                        train_park_hits = float(pf_row["H_factor"])

            # Platoon advantage: batter vs. opposing pitcher handedness
            train_platoon = 0.0
            if opp_pitcher_id:
                try:
                    b_hand = mlb_client.get_player_handedness(bid)
                    p_hand = mlb_client.get_player_handedness(opp_pitcher_id)
                    bat_side = b_hand.get("bat_side")
                    pitch_hand = p_hand.get("pitch_hand")
                    if bat_side and pitch_hand and bat_side != "S":
                        train_platoon = 1.0 if bat_side != pitch_hand else 0.0
                except Exception:
                    pass

            row = {
                # Batter contact quality
                "contact_rate_30d":          contact_rate,
                "babip_30d":                 min(0.500, max(0.0, (batter_ba - 0.03) / max(contact_rate, 0.40))),
                "avg_exit_velo_30d":         exit_velo,
                "hard_hit_rate_30d":         hard_hit,
                "hit_rate_season":           batter_ba,
                "avg_launch_angle_30d":      avg_hit_angle,
                "line_drive_rate_30d":       0.20,  # no per-game historical line drive data
                # Plate discipline (v2)
                "batter_k_rate_season":      k_pct_batter,
                "batter_walk_rate_season":   bb_pct_batter,
                "chase_rate_30d":            0.30,  # no per-game historical chase rate
                "sprint_speed":              sprint_speed,
                # Opposing pitcher — real per-game matchup data
                "pitcher_babip_allowed_30d":       pitcher_stats["babip"],
                "pitcher_babip_allowed_season":    pitcher_stats["babip"],
                "pitcher_hit_rate_allowed_season": pitcher_stats["hit_rate"],
                "pitcher_k_rate_season":           pitcher_stats["k_rate"],
                "pitcher_bvp_contact_factor":      _pitcher_bvp_contact_factor(bid, opp_pitcher_id, get_db()) if opp_pitcher_id else 0.776,
                # Context — real per-game values
                "park_factor_hits_h":        train_park_hits,
                "lineup_spot":               4.0,  # historical lineup data not available via game log
                "is_platoon_advantage":      train_platoon,
                "opp_lineup_xwoba":          0.320,  # no historical per-game lineup xwOBA available
                "market_implied_prob":       0.5,    # historical odds not stored
                "line_movement":             0.0,
                # Label
                "actual_hits":               actual_hits,
            }
            rows.append(row)

    log.info("Built %d Hits training rows from %d batters", len(rows), len(batter_ids))
    return rows


# ---------------------------------------------------------------------------
# Feature builders: HR model
# ---------------------------------------------------------------------------

def _build_hr_rows(
    batter_ids: list[int],
    pitcher_savant: dict,
    batter_savant: dict,
    mlb_client,
    season: int,
    game_context_map: dict,
    pitcher_stats_cache: dict,
    db,
    min_pa: int = 100,
) -> list[dict]:
    """Build one training row per batter-game."""
    rows: list[dict] = []

    for i, bid in enumerate(batter_ids):
        if i > 0 and i % 30 == 0:
            log.info("  HR rows: processed %d/%d batters, %d rows so far", i, len(batter_ids), len(rows))

        bpct = batter_savant["batter_pct"].get(bid, {})
        bexp = batter_savant["batter_exp"].get(bid, {})
        bev = batter_savant["batter_ev"].get(bid, {})

        barrel_rate   = _sf(bev.get("brl_percent"), _sf(bpct.get("brl_percent"), 5.0)) / 100.0
        hard_hit      = _sf(bpct.get("hard_hit_percent"), 38.0) / 100.0
        xiso          = _sf(bpct.get("xiso"), 0.150)
        avg_hit_angle = _sf(bev.get("avg_hit_angle"), 10.0)
        exit_velo     = _sf(bpct.get("exit_velocity"), 88.0)

        try:
            game_log = mlb_client.get_player_game_log(bid, season, group="hitting")
            time.sleep(INTER_PLAYER_SLEEP)
        except Exception as exc:
            log.debug("Game log failed for batter %d: %s", bid, exc)
            continue

        if not game_log:
            continue

        games_with_ab = [g for g in game_log if int(g.get("atBats") or 0) >= 1]
        if len(games_with_ab) < 10:
            continue

        total_ab = sum(int(g.get("atBats") or 0) for g in games_with_ab)
        total_hr = sum(int(g.get("homeRuns") or 0) for g in games_with_ab)
        hr_rate_season = total_hr / total_ab if total_ab > 0 else 0.03

        for g in games_with_ab:
            actual_hr = int(g.get("homeRuns") or 0)
            is_home = g.get("is_home", True)
            game_pk = g.get("game_pk")

            # Opposing pitcher from game context map
            opp_pitcher_hr_id = None
            park_factor_h = 1.0
            if game_pk and game_pk in game_context_map:
                ctx = game_context_map[game_pk]
                opp_pitcher_hr_id = ctx.get("away_pitcher_id") if is_home else ctx.get("home_pitcher_id")
                venue_name = ctx.get("venue_name") or ""
                if venue_name:
                    park_row = db.get_park_factor(venue_name, "neutral")
                    if park_row and park_row.get("HR_factor"):
                        park_factor_h = float(park_row["HR_factor"])

            # Pitcher HR rate and GB% from season stats
            pitcher_hr_rate = 0.030
            pitcher_gb = 0.44
            if opp_pitcher_hr_id:
                if opp_pitcher_hr_id not in pitcher_stats_cache:
                    pitcher_stats_cache[opp_pitcher_hr_id] = _pitcher_stats_from_season(
                        mlb_client, opp_pitcher_hr_id, season
                    )
                    time.sleep(INTER_PLAYER_SLEEP)
                cached = pitcher_stats_cache[opp_pitcher_hr_id]
                # hit_rate is hits/bf; we need hr/bf — fetch separately if not cached
                hr_stats_key = f"hr_gb_{opp_pitcher_hr_id}"
                if hr_stats_key not in pitcher_stats_cache:
                    try:
                        pst = mlb_client.get_player_season_stats(opp_pitcher_hr_id, season, group="pitching")
                        bf = float(pst.get("battersFaced") or 0)
                        pitcher_hr_rate = float(pst.get("homeRuns") or 0) / bf if bf > 0 else 0.030
                        pitcher_gb = _pitcher_gb_pct(pst)
                        pitcher_stats_cache[hr_stats_key] = {"hr_rate": pitcher_hr_rate, "gb_pct": pitcher_gb}
                    except Exception:
                        pitcher_stats_cache[hr_stats_key] = {"hr_rate": 0.030, "gb_pct": 0.44}
                hr_gb = pitcher_stats_cache[hr_stats_key]
                pitcher_hr_rate = hr_gb["hr_rate"]
                pitcher_gb = hr_gb["gb_pct"]

            # BvP factor from synthetic pitch-type matchup
            bvp_factor = 1.0
            if opp_pitcher_hr_id:
                try:
                    from pitch_type_matchup import compute_synthetic_bvp_hr
                    bvp_result = compute_synthetic_bvp_hr(bid, opp_pitcher_hr_id, db)
                    bvp_factor = bvp_result.get("bvp_factor", 1.0)
                except Exception:
                    pass

            # pull_pct_30d from DB rolling stats (populated by statcast_nightly)
            hr_pull_pct = 0.40
            try:
                pull_rows = db.get_player_stats(bid, "pull_pct", days=30)
                if pull_rows:
                    hr_pull_pct = float(pull_rows[-1]["value"])
            except Exception:
                pass

            # Platoon advantage for HR training
            hr_platoon = 0.0
            hr_batter_hand = 0.0  # 1=L, 0=R/S
            if opp_pitcher_hr_id:
                try:
                    b_hand = mlb_client.get_player_handedness(bid)
                    p_hand = mlb_client.get_player_handedness(opp_pitcher_hr_id)
                    bat_side = b_hand.get("bat_side")
                    pitch_hand = p_hand.get("pitch_hand")
                    if bat_side and pitch_hand:
                        hr_batter_hand = 1.0 if bat_side == "L" else 0.0
                        if bat_side != "S":
                            hr_platoon = 1.0 if bat_side != pitch_hand else 0.0
                except Exception:
                    pass

            row = {
                # Batter power metrics
                "barrel_rate_30d":        barrel_rate,
                "barrel_rate_60d":        barrel_rate,
                "hard_hit_rate_30d":      hard_hit,
                "xiso_30d":               xiso,
                "avg_launch_angle_30d":   avg_hit_angle,
                "hr_rate_season":         hr_rate_season,
                "pull_pct_30d":           hr_pull_pct,
                # Context — real park factor per game
                "park_factor_h":          park_factor_h,
                "weather_hr_multiplier":  1.0,  # historical weather not fetched during training
                # Opposing pitcher — real per-game matchup data
                "pitcher_hr_rate_season": pitcher_hr_rate,
                "pitcher_gb_pct":         pitcher_gb,
                "batter_hand_vs_pitcher": hr_batter_hand,
                "is_platoon_advantage":   hr_platoon,
                "lineup_spot":            4.0,  # historical lineup data not available via game log
                "bvp_factor":             bvp_factor,
                # Lineup quality signal (v2)
                "opp_lineup_xwoba":       0.320,  # no historical per-game lineup xwOBA available
                # Market
                "market_implied_prob":    0.5,    # historical odds not stored
                "line_movement":          0.0,
                # Label
                "actual_hr":              actual_hr,
            }
            rows.append(row)

    log.info("Built %d HR training rows from %d batters", len(rows), len(batter_ids))
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ip(ip_str) -> float:
    """Convert '5.2' (MLB format: 5 full innings + 2 outs) to decimal IP."""
    try:
        ip = float(ip_str)
        whole = int(ip)
        partial = ip - whole
        return whole + (partial / 3.0 * 10.0 / 10.0)
    except (TypeError, ValueError):
        return 0.0


def _days_rest(starts: list[dict], idx: int) -> int | None:
    """Days between start at idx and prior start."""
    if idx == 0:
        return None
    try:
        d1 = datetime.strptime(starts[idx - 1]["date"][:10], "%Y-%m-%d").date()
        d2 = datetime.strptime(starts[idx]["date"][:10], "%Y-%m-%d").date()
        return (d2 - d1).days
    except (KeyError, ValueError):
        return None


def _is_missing(v) -> bool:
    """True if value is None, NaN, or inf."""
    if v is None:
        return True
    try:
        return not np.isfinite(float(v))
    except (TypeError, ValueError):
        return True


def _impute(rows: list[dict], feature_cols: list[str]) -> list[dict]:
    """Replace None/NaN/inf with column medians, then drop rows still invalid."""
    for col in feature_cols:
        vals = [float(r[col]) for r in rows if not _is_missing(r.get(col))]
        median = float(np.median(vals)) if vals else 0.0
        for r in rows:
            if _is_missing(r.get(col)):
                r[col] = median
    # Final safety: drop any row that still has a non-finite value
    clean = [
        r for r in rows
        if all(not _is_missing(r.get(col)) for col in feature_cols)
    ]
    dropped = len(rows) - len(clean)
    if dropped:
        log.info("  Impute: dropped %d rows with persistent NaN after median fill", dropped)
    return clean


def _report(name: str, meta: dict) -> None:
    log.info("── %s ──────────────────────", name)
    for k, v in meta.items():
        if k == "feature_importances" or k == "coefficients":
            top = sorted(v.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
            log.info("  Top features: %s", top)
        else:
            log.info("  %s: %s", k, v)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    season: int = 2024,
    min_ip: float = 20.0,
    min_pa: int = 100,
    dry_run: bool = False,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mlb_client = get_client()

    # 1. Fetch all Savant data in bulk
    savant = _fetch_savant_data(season)

    pitcher_ids = sorted(savant["pitcher_pct"].keys())
    batter_ids = sorted(savant["batter_pct"].keys())
    log.info(
        "Savant loaded: %d pitchers, %d batters",
        len(pitcher_ids), len(batter_ids),
    )

    # 2. Pre-fetch season game context (pitchers, umpires, venues) — shared by all models
    log.info("Fetching season game context for %d…", season)
    game_context_map = mlb_client.get_season_game_context(season)
    log.info("  Game context map: %d games", len(game_context_map))
    db = get_db()

    # 3. K model
    log.info("Building K training data (%d pitchers)…", len(pitcher_ids))
    k_rows = _build_k_rows(pitcher_ids, savant, mlb_client, season, game_context_map, db, min_ip=min_ip)

    if len(k_rows) < 500:
        log.warning("Only %d K rows — below 500 minimum. Skipping K model training.", len(k_rows))
    else:
        k_rows = _impute(k_rows, K_FEATURE_COLS)
        k_train, k_test = train_test_split(k_rows, test_size=0.15, random_state=42)
        log.info("K split: %d train / %d test", len(k_train), len(k_test))

        km = KModel()
        try:
            meta = km.train(k_train)
            _report("KModel", meta)
            # Validation: MAE on test set
            test_mae = _k_mae(km, k_test)
            log.info("  Test MAE (lambda vs actual_ks): %.3f", test_mae)
        except ValueError as e:
            log.error("KModel training failed: %s", e)
            km = None

        if km and not dry_run:
            km.save()
            log.info("KModel saved to %s", MODELS_DIR / "k_model.pkl")

    # 4. Hits model
    pitcher_stats_cache: dict[int, dict] = {}
    log.info("Building Hits training data (%d batters)…", len(batter_ids))
    hits_rows = _build_hits_rows(
        batter_ids, savant, savant, mlb_client, season,
        game_pitcher_map=game_context_map,
        pitcher_stats_cache=pitcher_stats_cache,
        min_pa=min_pa,
    )

    if len(hits_rows) < 500:
        log.warning("Only %d Hits rows — skipping Hits model training.", len(hits_rows))
    else:
        hits_rows = _impute(hits_rows, HITS_FEATURE_COLS)
        h_train, h_test = train_test_split(hits_rows, test_size=0.15, random_state=42)
        log.info("Hits split: %d train / %d test", len(h_train), len(h_test))

        hm = HitsModel()
        try:
            meta = hm.train(h_train)
            _report("HitsModel", meta)
            acc, brier = _hits_metrics(hm, h_test)
            log.info("  Test accuracy: %.3f | Brier score: %.4f", acc, brier)
        except ValueError as e:
            log.error("HitsModel training failed: %s", e)
            hm = None

        if hm and not dry_run:
            hm.save()
            log.info("HitsModel saved to %s", MODELS_DIR / "hits_model.pkl")

    # 5. HR model
    log.info("Building HR training data (%d batters)…", len(batter_ids))
    hr_rows = _build_hr_rows(
        batter_ids, savant, savant, mlb_client, season,
        game_context_map=game_context_map,
        pitcher_stats_cache=pitcher_stats_cache,
        db=db,
        min_pa=min_pa,
    )

    if len(hr_rows) < 500:
        log.warning("Only %d HR rows — skipping HR model training.", len(hr_rows))
    else:
        hr_rows = _impute(hr_rows, HR_FEATURE_COLS)
        hr_train, hr_test = train_test_split(hr_rows, test_size=0.15, random_state=42)
        log.info("HR split: %d train / %d test", len(hr_train), len(hr_test))

        hrm = HRModel()
        try:
            meta = hrm.train(hr_train)
            _report("HRModel", meta)
            acc, brier = _hr_metrics(hrm, hr_test)
            log.info("  Test accuracy: %.3f | Brier score: %.4f", acc, brier)
        except ValueError as e:
            log.error("HRModel training failed: %s", e)
            hrm = None

        if hrm and not dry_run:
            hrm.save()
            log.info("HRModel saved to %s", MODELS_DIR / "hr_model.pkl")

    log.info("Training complete.")


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def _k_mae(km: KModel, test_rows: list[dict]) -> float:
    errors = []
    for row in test_rows:
        try:
            lam = km.predict_lambda(row)
            errors.append(abs(lam - row["actual_ks"]))
        except Exception:
            continue
    return float(np.mean(errors)) if errors else float("nan")


def _hits_metrics(hm: HitsModel, test_rows: list[dict]) -> tuple[float, float]:
    preds, labels = [], []
    for row in test_rows:
        try:
            p = hm.predict_proba(row)
            preds.append(p)
            labels.append(1 if int(row["actual_hits"]) >= 1 else 0)
        except Exception:
            continue
    if not preds:
        return float("nan"), float("nan")
    preds_arr = np.array(preds)
    labels_arr = np.array(labels)
    acc = float(np.mean((preds_arr >= 0.5) == labels_arr))
    brier = float(np.mean((preds_arr - labels_arr) ** 2))
    return acc, brier


def _hr_metrics(hrm: HRModel, test_rows: list[dict]) -> tuple[float, float]:
    preds, labels = [], []
    for row in test_rows:
        try:
            p = hrm.predict_proba(row)
            preds.append(p)
            labels.append(1 if int(row["actual_hr"]) >= 1 else 0)
        except Exception:
            continue
    if not preds:
        return float("nan"), float("nan")
    preds_arr = np.array(preds)
    labels_arr = np.array(labels)
    acc = float(np.mean((preds_arr >= 0.5) == labels_arr))
    brier = float(np.mean((preds_arr - labels_arr) ** 2))
    return acc, brier


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLB prop models from historical data.")
    parser.add_argument("--season", type=int, default=2024, help="Season year (default 2024)")
    parser.add_argument("--min-ip", type=float, default=20.0, help="Min SP innings pitched to include")
    parser.add_argument("--min-pa", type=int, default=100, help="Min batter PA to include")
    parser.add_argument("--dry-run", action="store_true", help="Build data and train, but don't save models")
    args = parser.parse_args()

    run(
        season=args.season,
        min_ip=args.min_ip,
        min_pa=args.min_pa,
        dry_run=args.dry_run,
    )
