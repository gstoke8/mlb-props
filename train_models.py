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
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from db import get_db
from mlb_api import get_client
from k_model import KModel, K_FEATURE_COLS
from hits_model import HitsModel, HITS_FEATURE_COLS
from hr_model import HRModel, HR_FEATURE_COLS
from outs_model import OutsModel, OUTS_FEATURE_COLS

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

    log.info("Fetching Savant pitcher pitch average speed by type for %d…", season)
    try:
        pitcher_velo_mix = pybaseball.statcast_pitcher_pitch_arsenal(season, minP=50, arsenal_type="avg_speed")
    except Exception as exc:
        log.warning("pitcher velo arsenal fetch failed: %s; using fallback", exc)
        pitcher_velo_mix = None

    # avg_break_z is not a supported arsenal_type; vertical break unavailable from Savant bulk endpoints.
    pitcher_break_z = None

    log.info("Fetching Savant pitcher exit-velo/barrels for %d…", season)
    pitcher_ev = pybaseball.statcast_pitcher_exitvelo_barrels(season, minBBE=20)

    # Baseball Reference pitching stats: real SwStr% (StS), CStr% (StL), K%, BB%.
    # bref mlbID column is the MLBAM player_id, so no crosswalk needed.
    # StS = swinging strikes / pitches (= SwStr% used at inference time).
    # StL = called strikes / pitches (= CStr%); StS+StL = CSW%.
    log.info("Fetching Baseball Reference pitching stats for %d…", season)
    pitcher_swstr_csw: dict[int, dict[str, float | None]] = {}
    try:
        bref = pybaseball.pitching_stats_bref(season)
        for _, row in bref.iterrows():
            try:
                mlb_id = int(str(row.get("mlbID") or "").strip())
                if mlb_id <= 0:
                    continue
            except (TypeError, ValueError):
                continue
            bf  = float(row.get("BF") or 0)
            so  = float(row.get("SO") or 0)
            bb  = float(row.get("BB") or 0)
            sts = float(row.get("StS") or 0)   # swinging strike rate (fraction of pitches)
            stl = float(row.get("StL") or 0)   # called strike rate (fraction of pitches)
            pitcher_swstr_csw[mlb_id] = {
                "k_pct":      so / bf if bf > 0 else None,
                "bb_pct":     bb / bf if bf > 0 else None,
                "swstr_rate": sts if sts > 0 else None,
                "csw_rate":   (sts + stl) if sts > 0 and stl > 0 else None,
                # whiff% = ss/swings ≈ SwStr% / league-avg swing rate (≈ 0.46)
                "whiff_rate": sts / 0.46 if sts > 0 else None,
            }
        log.info("Bref pitching stats loaded for %d pitchers", len(pitcher_swstr_csw))
    except Exception as exc:
        log.warning("Bref pitching fetch failed: %s; using approximations", exc)

    log.info("Fetching Savant batter percentile ranks for %d…", season)
    batter_pct = pybaseball.statcast_batter_percentile_ranks(season)

    log.info("Fetching Savant batter expected stats for %d…", season)
    batter_exp = pybaseball.statcast_batter_expected_stats(season, minPA=50)

    log.info("Fetching Savant batter exit-velo/barrels for %d…", season)
    batter_ev = pybaseball.statcast_batter_exitvelo_barrels(season, minBBE=20)

    # Baseball Reference batting stats: actual K%, BB% per batter.
    # bpct k_percent/bb_percent are percentile ranks (1-100), not actual rates.
    # bref provides true fractions aligned with inference-time MLB API values.
    log.info("Fetching Baseball Reference batting stats for %d…", season)
    batter_bref_map: dict[int, dict[str, float | None]] = {}
    try:
        bref_bat = pybaseball.batting_stats_bref(season)
        for _, row in bref_bat.iterrows():
            try:
                mlb_id = int(str(row.get("mlbID") or "").strip())
                if mlb_id <= 0:
                    continue
            except (TypeError, ValueError):
                continue
            pa = float(row.get("PA") or 0)
            batter_bref_map[mlb_id] = {
                "k_pct":  float(row.get("SO") or 0) / pa if pa > 0 else None,
                "bb_pct": float(row.get("BB") or 0) / pa if pa > 0 else None,
            }
        log.info("Bref batting stats loaded for %d batters", len(batter_bref_map))
    except Exception as exc:
        log.warning("Bref batting fetch failed: %s; using league-avg fallbacks", exc)

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
        "pitcher_pct":       _idx(pitcher_pct),
        "pitcher_exp":       _idx(pitcher_exp),
        "pitcher_mix":       _idx(pitcher_mix, id_col="pitcher"),
        "pitcher_velo_mix":  _idx(pitcher_velo_mix, id_col="pitcher") if pitcher_velo_mix is not None else {},
        "pitcher_ev":        _idx(pitcher_ev),
        "pitcher_swstr_csw": pitcher_swstr_csw,
        "batter_pct":        _idx(batter_pct),
        "batter_exp":        _idx(batter_exp),
        "batter_ev":         _idx(batter_ev),
        "batter_bref":       batter_bref_map,
        "sprint_map":        sprint_map,  # int → float (ft/s)
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
        whiff_data = savant.get("pitcher_whiff", {}).get(pid, {})

        # Real season-level rates from Baseball Reference (actual values, not percentile ranks).
        # Savant percentile_ranks are ordinal only; bref provides true fractions.
        _bref = savant.get("pitcher_swstr_csw", {}).get(pid, {})
        xwoba_allowed = _sf(exp.get("est_woba"), 0.320)

        # Actual K% from bref; fallback to league avg if bref data unavailable
        k_pct_season = _bref.get("k_pct") or LEAGUE_K_PCT

        # SwStr%, CSW%, Whiff% from bref — exact same definition as inference (statcast_nightly)
        swstr_rate = _bref.get("swstr_rate") or 0.110   # league avg SwStr%
        csw_rate   = _bref.get("csw_rate")   or 0.280   # league avg CSW%
        whiff_rate = _bref.get("whiff_rate") or 0.240   # league avg Whiff%

        # Fastball velocity from Savant arsenal (column: ff_avg_speed, pitch ID: pitcher).
        # Vertical break not available from Savant bulk endpoints; use league avg default.
        _vm = savant.get("pitcher_velo_mix", {}).get(pid, {})
        _ff_v = _sf(_vm.get("ff_avg_speed"), 0.0)
        ff_perceived_velo = _ff_v if _ff_v > 50.0 else 93.5
        max_vbreak   = 12.5   # league avg IVB; not available from bulk Savant endpoints
        vbreak_range = 8.0    # league avg IVB range across pitch types

        # Per-pitch-type whiff rates — try multiple column name formats
        def _pt_whiff(pt: str) -> float:
            for key in (f"{pt}_whiff_pct", f"whiff_pct_{pt}", f"{pt}_avg_pct", pt):
                v = whiff_data.get(key)
                if v is not None:
                    return _sf(v, whiff_rate) / (100.0 if _sf(v, 0) > 1.0 else 1.0)
            return whiff_rate

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
                # Core pitcher effectiveness — real values from FanGraphs crosswalk
                "csw_rate_30d":            csw_rate,
                "k_rate_30d":              k_pct_season,
                "k_rate_season":           k_pct_season * 27.0,  # convert K% to K/9 proxy
                "whiff_rate_30d":          whiff_rate,
                # Pitch-type stuff metrics (v2) — real values from FanGraphs/Savant
                "swstr_rate_30d":          swstr_rate,
                "ff_whiff_rate_30d":       ff_whiff,
                "sl_whiff_rate_30d":       sl_whiff,
                "ch_whiff_rate_30d":       ch_whiff,
                # Opposing lineup (v4) — league-average defaults for training.
                # Historical per-game lineup data is not fetched during training;
                # real values are injected at inference time via daily_runner.py.
                "opp_k_rate_season":       LEAGUE_K_PCT,
                "opp_lineup_whiff_factor": 1.0,
                "lineup_lhb_pct":          0.5,
                # Game context — real per-start values
                "umpire_k_factor":         umpire_k_factor,
                "park_k_factor":           park_k_factor,
                "is_home":                 is_home,
                "days_rest":               float(days_rest if days_rest is not None else 5.0),
                "avg_ip_30d":              avg_ip,
                "is_opener_risk":          int(avg_ip < 4.5),
                # New k-v6 features — real values where available, league avg otherwise
                "foul_rate_30d":          0.195,   # not in season-level endpoints; league avg default
                "stuff_plus_30d":         _sf(pct.get("stuff_plus_stuff_plus") or pct.get("stuff_plus"), 100.0),
                "max_vbreak_30d":         max_vbreak,
                "vbreak_range_30d":       vbreak_range,
                "ff_perceived_velo_30d":  ff_perceived_velo,
                "rp_horiz_std_30d":       0.04,    # not in season-level endpoints; league avg default
                "opp_lineup_o_swing_30d": 0.300,   # league avg O-Swing%
                "game_temp_f":            75.0,    # neutral temperature default
                "wind_dir_binary":        0.0,     # neutral wind default
                "expected_tto3_pa_pct":   max(0.0, min(1.0, (avg_ip * 4.3 - 18) / max(avg_ip * 4.3, 1))),
                "k_rate_eb_30d":          k_pct_season,   # at training time, season rate IS the EB estimate
                "k_prev_game":            float(starts_sorted[max(0, idx-1)].get("strikeOuts") or 0) if idx > 0 else 5.0,
                "k_prev3_weighted":       float(starts_sorted[idx-1].get("strikeOuts") or 5.0) if idx > 0 else 5.0,
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
    """Fetch and compute hit-rate, k-rate, BABIP, gb_pct for a pitcher from season stats.

    Returns dict with keys: hit_rate, k_rate, babip, gb_pct (all floats, league-avg fallback).
    """
    defaults = {"hit_rate": 0.243, "k_rate": LEAGUE_K_PCT, "babip": 0.295, "gb_pct": 0.44}
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
        gb_pct = _pitcher_gb_pct(stats)
        return {"hit_rate": hit_rate, "k_rate": k_rate, "babip": babip, "gb_pct": gb_pct}
    except Exception:
        return defaults


_PA_WEIGHT_BY_SPOT: dict[int, float] = {
    1: 3.20, 2: 3.10, 3: 3.00, 4: 2.90, 5: 2.80,
    6: 2.70, 7: 2.65, 8: 2.55, 9: 2.40,
}
_PA_WEIGHT_DEFAULT = 2.85  # midpoint fallback when lineup spot is unknown


def _build_hits_rows(
    batter_ids: list[int],
    pitcher_savant: dict,
    batter_savant: dict,
    mlb_client,
    season: int,
    game_pitcher_map: dict[int, dict],
    pitcher_stats_cache: dict[int, dict],
    db,
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
        # bpct hard_hit_percent is a percentile rank (1-100), not actual rate.
        # Use bev ev95percent (% BBE with EV ≥ 95mph) as actual hard-hit rate.
        hard_hit      = _sf(bev.get("ev95percent"), 38.0) / 100.0
        exit_velo     = _sf(bpct.get("exit_velocity"), 88.0)
        # bpct k_percent/bb_percent are percentile ranks (1-100), not actual rates.
        # Use bref actual K/PA and BB/PA aligned with inference MLB API values.
        _bref_bat     = batter_savant.get("batter_bref", {}).get(bid, {})
        k_pct_batter  = _bref_bat.get("k_pct") if _bref_bat.get("k_pct") is not None else LEAGUE_K_PCT
        bb_pct_batter = _bref_bat.get("bb_pct") if _bref_bat.get("bb_pct") is not None else 0.084
        contact_rate  = 1.0 - k_pct_batter
        avg_hit_angle = _sf(bev.get("avg_hit_angle"), 10.0)
        sprint_speed  = _sf(batter_savant.get("sprint_map", {}).get(bid), 27.0)

        # v7 regression-signal features
        xba_season    = _sf(bexp.get("est_ba"), batter_ba)
        # xBA - BA gap: positive = batter due for positive regression (books lag xBA)
        xba_minus_ba_gap = _sf(bexp.get("est_ba_minus_ba_diff"), xba_season - batter_ba)
        # BABIP proxy from season stats: roughly BA / contact_rate
        _babip_proxy  = min(0.500, batter_ba / max(contact_rate, 0.400))
        babip_deviation = _babip_proxy - 0.300
        # Sweet spot % = % batted balls 8-32° launch angle (anglesweetspotpercent in Savant)
        _ss_raw = _sf(bev.get("anglesweetspotpercent"), _sf(bev.get("sweet_spot_percent"), 34.0))
        sweet_spot_pct = (_ss_raw / 100.0) if _ss_raw > 1.0 else _ss_raw

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

            # expected_pa: derive from lineup spot if available, else use midpoint default.
            # The MLB game log API does not return per-game batting order, so we fall back
            # to _PA_WEIGHT_DEFAULT (2.85, midpoint of spots 1-9) for all training rows.
            _spot = int(g.get("batting_order") or g.get("lineup_spot") or 0)
            _expected_pa = _PA_WEIGHT_BY_SPOT.get(_spot, _PA_WEIGHT_DEFAULT)

            # hits-v8: chase_rate — bpct oz_swing_percent is a percentile rank (1-100),
            # not an actual O-Swing%. Use league average (0.31) as training value since
            # actual per-batter O-Swing% is unavailable from bulk Savant endpoints.
            _chase_rate_30d = 0.31

            # hits-v8: fastball pct from pitcher arsenal mix
            opp_mix = pitcher_savant["pitcher_mix"].get(opp_pitcher_id or -1, {})
            ff_pct = _sf(opp_mix.get("n_ff"), 0.0)
            if ff_pct > 1.0:
                ff_pct /= 100.0
            _pitcher_ff_pct = max(0.0, min(1.0, ff_pct)) if ff_pct > 0 else 0.52

            row = {
                # Batter contact quality
                "contact_rate_30d":          contact_rate,
                "babip_30d":                 min(0.500, max(0.0, (batter_ba - 0.03) / max(contact_rate, 0.40))),
                "avg_exit_velo_30d":         exit_velo,
                "hard_hit_rate_30d":         hard_hit,
                "avg_launch_angle_30d":      avg_hit_angle,
                # v7 regression-signal features (replaces hit_rate_season)
                "xba_season":                xba_season,
                "xba_minus_ba_gap":          xba_minus_ba_gap,
                "babip_deviation":           babip_deviation,
                "sweet_spot_pct":            sweet_spot_pct,
                # Plate discipline (v2)
                "batter_k_rate_season":      k_pct_batter,
                "batter_walk_rate_season":   bb_pct_batter,
                "sprint_speed":              sprint_speed,
                # Opposing pitcher — real per-game matchup data
                "pitcher_babip_allowed_30d":       pitcher_stats["babip"],
                "pitcher_babip_allowed_season":    pitcher_stats["babip"],
                "pitcher_hit_rate_allowed_season": pitcher_stats["hit_rate"],
                "pitcher_k_rate_season":           pitcher_stats["k_rate"],
                "pitcher_gb_pct":                  pitcher_stats.get("gb_pct", 0.44),
                # BvP factor requires real-time pitch mix data from DB (populated by nightly).
                # Training skips the live lookup to avoid per-game API calls (10+ hours).
                # Real values injected at inference time via daily_runner.py.
                "pitcher_bvp_contact_factor":      0.776,
                # Context — real per-game values
                "expected_pa":               _expected_pa,
                "team_implied_runs":         4.5,  # not in historical data; live signal only
                "park_factor_hits_h":        train_park_hits,
                "is_platoon_advantage":      train_platoon,
                # New hits-v8 features
                "chase_rate_30d":            _chase_rate_30d,
                "zone_contact_rate_30d":     0.78,   # league avg Z-Con% default
                "batting_order_position":    float(_spot if _spot > 0 else 5),
                "pitcher_swstr_season":      _sf(pitcher_savant["pitcher_pct"].get(opp_pitcher_id, {}).get("whiff_percent"), 24.0) / 100.0 * 0.70 if opp_pitcher_id else 0.105,
                "pitcher_fastball_pct_season": _pitcher_ff_pct,
                "opposing_team_der_season":  round(1.0 - pitcher_stats["babip"], 4) if pitcher_stats.get("babip") else 0.700,
                "expected_tto_number":       min(3.0, max(1.0, 1.0 + (_spot - 1) / 9.0)) if _spot > 0 else 2.0,
                "game_temp_f":               75.0,
                "wind_direction_category":   0.0,
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

            # Pitcher contact quality allowed — real per-pitcher values from Savant exit-velo data.
            # pitcher_ev is indexed by player_id (MLBAM).
            # brl_percent: barrel rate (8.0 = 8%); ev95percent: hard-hit% (39.0 = 39%).
            _pitcher_ev_data = pitcher_savant.get("pitcher_ev", {}).get(opp_pitcher_hr_id or -1, {})
            def _pct_frac(val: Any, default_frac: float) -> float:
                v = _sf(val, None)
                if v is None:
                    return default_frac
                return v / 100.0 if v > 1.0 else v
            pitcher_brl_allowed = _pct_frac(_pitcher_ev_data.get("brl_percent"), 0.07)
            pitcher_hhpct_allowed = _pct_frac(_pitcher_ev_data.get("ev95percent"), 0.38)

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

            # BvP factor: skip live lookup during training (would make per-game API calls).
            # Real values injected at inference time.
            bvp_factor = 1.0

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

            # Sweet spot pct from exit velo barrels (anglesweetspotpercent)
            _ss_raw_hr = _sf(bev.get("anglesweetspotpercent"), _sf(bpct.get("sweet_spot_percent"), 34.0))
            sweet_spot_pct_hr = (_ss_raw_hr / 100.0) if _ss_raw_hr > 1.0 else _ss_raw_hr

            # Empirical Bayes HR rate — regress season rate toward league mean
            _LEAGUE_HR_RATE_TR = 0.033
            _HR_EB_PRIOR_PA_TR = 300.0
            total_pa = sum(int(g.get("atBats") or 0) + int(g.get("baseOnBalls") or 0) + int(g.get("hitByPitch") or 0) for g in games_with_ab)
            hr_rate_eb_30d = (hr_rate_season * total_pa + _LEAGUE_HR_RATE_TR * _HR_EB_PRIOR_PA_TR) / max(total_pa + _HR_EB_PRIOR_PA_TR, 1.0)

            row = {
                # Batter power metrics
                "barrel_rate_30d":        barrel_rate * 0.95,
                "barrel_rate_60d":        barrel_rate,
                "hard_hit_rate_30d":      hard_hit,
                "xiso_30d":               xiso,
                "avg_launch_angle_30d":   avg_hit_angle,
                "hr_rate_season":         hr_rate_season,
                "hr_rate_eb_30d":         hr_rate_eb_30d,
                "pull_pct_30d":           hr_pull_pct,
                # Batted ball quality — new in hr-v4 (league-average defaults; nightly populates real values)
                "fly_ball_rate_30d":      0.35,
                "sweet_spot_pct_30d":     sweet_spot_pct_hr,
                # Context — real park factor per game
                "park_factor_h":          park_factor_h,
                "game_temp_f":            75.0,   # neutral temperature for historical training
                "wind_hr_factor":         1.0,    # neutral wind for historical training
                # Opposing pitcher — real per-game matchup data
                "pitcher_hr_rate_season": pitcher_hr_rate,
                "pitcher_gb_pct":         pitcher_gb,
                # Pitcher contact quality allowed — real per-pitcher values from Savant ev data
                "pitcher_barrel_rate_allowed":   pitcher_brl_allowed,
                "pitcher_hard_hit_pct_allowed":  pitcher_hhpct_allowed,
                "batter_hand_vs_pitcher": hr_batter_hand,
                "is_platoon_advantage":   hr_platoon,
                "pull_x_platoon":         hr_pull_pct * hr_platoon,
                "bvp_factor":             bvp_factor,
                # Label
                "actual_hr":              actual_hr,
            }
            rows.append(row)

    log.info("Built %d HR training rows from %d batters", len(rows), len(batter_ids))
    return rows


# ---------------------------------------------------------------------------
# Feature builders: Outs model
# ---------------------------------------------------------------------------

def _ip_to_outs(ip_str) -> int:
    """Convert MLB inningsPitched string ('6.2') to integer outs (6*3+2=20)."""
    try:
        ip = float(ip_str)
        whole = int(ip)
        partial = int(round((ip - whole) * 10))
        return whole * 3 + partial
    except (TypeError, ValueError):
        return 0


def _build_outs_rows(
    pitcher_ids: list[int],
    savant: dict,
    mlb_client,
    season: int,
    game_context_map: dict,
    db,
    min_ip: float = 20.0,
) -> list[dict]:
    """Build one training row per pitcher start for the outs model.

    Four features use training-time constants (same limitation as K model):
      - bullpen_usage_3d: 0     — no historical per-game bullpen data available
      - opp_pa_per_k_30d: LEAGUE — per-game opposing lineup data not fetched here
      - manager_hook_factor: 1.0 — team tendency table not populated
      - game_total: 8.5         — historical totals not in game_context_map

    These become zero-variance in training and are dropped by the NB solver.
    Their coefficients default to 0.  Real values injected at inference time via
    outs_features.compute_outs_features() will have no effect on outs-v1 predictions.
    Populate historical data and retrain (outs-v2) to unlock these features.
    """
    rows: list[dict] = []

    # League-average fallback constants
    LEAGUE_P_PER_IP = 15.5
    LEAGUE_BB_RATE = 0.085
    LEAGUE_K_RATE = 0.224
    LEAGUE_PA_PER_K = 1.0 / LEAGUE_K_RATE
    LEAGUE_OUTS_PER_START = 16.5
    LEAGUE_XWOBA = 0.320

    for i, pid in enumerate(pitcher_ids):
        if i > 0 and i % 20 == 0:
            log.info("  Outs rows: processed %d/%d pitchers, %d rows so far", i, len(pitcher_ids), len(rows))

        pct = savant["pitcher_pct"].get(pid, {})
        exp = savant["pitcher_exp"].get(pid, {})

        # Use bref actual K/BF and BB/BF — pct k_percent/bb_percent are percentile ranks
        _bref_o = savant.get("pitcher_swstr_csw", {}).get(pid, {})
        bb_rate_season = _bref_o.get("bb_pct") if _bref_o.get("bb_pct") is not None else LEAGUE_BB_RATE
        k_rate_season  = _bref_o.get("k_pct")  if _bref_o.get("k_pct")  is not None else LEAGUE_K_RATE
        xwoba_allowed  = _sf(exp.get("est_woba"), LEAGUE_XWOBA)
        # P/IP proxy: estimated from walk rate (more walks → more pitches per inning)
        p_per_ip = 14.5 + 10.0 * bb_rate_season

        try:
            game_log = mlb_client.get_player_game_log(pid, season, group="pitching")
            time.sleep(INTER_PLAYER_SLEEP)
        except Exception as exc:
            log.debug("Game log failed for pitcher %d: %s", pid, exc)
            continue

        if not game_log:
            continue

        starts = [
            g for g in game_log
            if g.get("gamesStarted", 0) == 1
            and _parse_ip(g.get("inningsPitched", "0")) >= 3.0
        ]
        if len(starts) < 3:
            continue

        starts_sorted = sorted(starts, key=lambda g: g.get("date", ""))

        # Season totals for season_outs_per_start
        total_ip = sum(_parse_ip(g.get("inningsPitched", "0")) for g in starts_sorted)
        n_starts = len(starts_sorted)
        season_outs_per_start = (total_ip * 3.0 / n_starts) if n_starts > 0 else LEAGUE_OUTS_PER_START

        for idx, start in enumerate(starts_sorted):
            actual_outs = _ip_to_outs(start.get("inningsPitched", "0"))
            if actual_outs == 0:
                continue  # incomplete game log entry

            is_home = int(bool(start.get("is_home", False)))
            days_rest = _days_rest(starts_sorted, idx)

            # Rolling avg IP from prior 3 starts
            prior = starts_sorted[max(0, idx - 3):idx]
            avg_ip_last_3 = (
                sum(_parse_ip(g.get("inningsPitched", "0")) for g in prior) / len(prior)
                if prior else total_ip / n_starts
            )

            # Previous start pitch count
            prev_pc = 85.0
            if idx > 0:
                prev = starts_sorted[idx - 1]
                pc = prev.get("pitchesThrown") or prev.get("numberOfPitches")
                if pc is not None:
                    prev_pc = _sf(pc, 85.0)
                else:
                    prev_pc = _parse_ip(prev.get("inningsPitched", "0")) * LEAGUE_P_PER_IP

            # Third-time-through proximity: > 0 when pitcher goes past 2× through lineup
            third_time = max(0.0, (avg_ip_last_3 / 3.0) - 2.0)

            row = {
                "pitches_per_inning_30d":      min(max(p_per_ip, 12.0), 22.0),
                "bb_rate_30d":                 min(max(bb_rate_season, 0.0), 0.25),
                "avg_ip_last_3_starts":        min(max(avg_ip_last_3, 0.0), 9.0),
                "bullpen_usage_3d":            0.0,   # not available historically
                "opp_pa_per_k_30d":            LEAGUE_PA_PER_K,
                "prev_start_pitch_count":      min(max(prev_pc, 40.0), 130.0),
                "manager_hook_factor":         1.0,   # neutral default for training
                "contact_quality_allowed_30d": min(max(xwoba_allowed, 0.200), 0.450),
                "game_total":                  8.5,   # league-average game total
                "k_rate_30d":                  min(max(k_rate_season, 0.05), 0.45),
                "days_rest":                   float(days_rest if days_rest is not None else 5.0),
                "is_home":                     float(is_home),
                "season_outs_per_start":       min(max(season_outs_per_start, 6.0), 27.0),
                "third_time_through_proximity": min(third_time, 1.0),
                # Label
                "actual_outs":                 actual_outs,
            }
            rows.append(row)

    log.info("Built %d Outs training rows from %d pitchers", len(rows), len(pitcher_ids))
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ip(ip_str) -> float:
    """Convert '5.2' (MLB format: 5 full innings + 2 outs) to decimal IP.

    MLB uses a base-3 fractional convention: the digit after the decimal is
    the number of outs recorded (0, 1, or 2), NOT a decimal fraction.
    So '6.2' = 6 full innings + 2 outs = 6 + 2/3 = 6.667, not 6.2.
    """
    try:
        ip = float(ip_str)
        whole = int(ip)
        outs = int(round((ip - whole) * 10))  # extract the outs digit (0, 1, or 2)
        return whole + outs / 3.0
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
    models: str = "all",
) -> None:
    train_k    = models in ("k", "all")
    train_hits = models in ("hits", "all")
    train_hr   = models in ("hr", "all")
    train_outs = models in ("outs", "all")

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

    if not train_k:
        log.info("Skipping K model (--model=%s)", models)
    # 3. K model
    if train_k:
        log.info("Building K training data (%d pitchers)…", len(pitcher_ids))
        k_rows = _build_k_rows(pitcher_ids, savant, mlb_client, season, game_context_map, db, min_ip=min_ip)

    if train_k and len(k_rows) < 500:
        log.warning("Only %d K rows — below 500 minimum. Skipping K model training.", len(k_rows))
    elif train_k:
        k_rows = _impute(k_rows, K_FEATURE_COLS)

        # 5-fold CV
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_maes: list[float] = []
        for fold_train_idx, fold_val_idx in kf.split(k_rows):
            fold_train = [k_rows[i] for i in fold_train_idx]
            fold_val   = [k_rows[i] for i in fold_val_idx]
            fold_km = KModel()
            try:
                fold_km.train(fold_train)
                cv_maes.append(_k_mae(fold_km, fold_val))
            except Exception as e:
                log.warning("K CV fold failed: %s", e)
        if cv_maes:
            log.info("K 5-fold CV — MAE=%.3f±%.3f", np.mean(cv_maes), np.std(cv_maes))

        k_train, k_test = train_test_split(k_rows, test_size=0.15, random_state=42)
        log.info("K split: %d train / %d test", len(k_train), len(k_test))

        km = KModel()
        try:
            meta = km.train(k_train)
            _report("KModel", meta)
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
    if not train_hits:
        log.info("Skipping Hits model (--model=%s)", models)
    else:
        log.info("Building Hits training data (%d batters)…", len(batter_ids))
        hits_rows = _build_hits_rows(
            batter_ids, savant, savant, mlb_client, season,
            game_pitcher_map=game_context_map,
            pitcher_stats_cache=pitcher_stats_cache,
            db=db,
            min_pa=min_pa,
        )

    if train_hits and len(hits_rows) < 500:
        log.warning("Only %d Hits rows — skipping Hits model training.", len(hits_rows))
    elif train_hits:
        hits_rows = _impute(hits_rows, HITS_FEATURE_COLS)

        # 5-fold stratified CV
        y_hits_all = np.array([1 if r["actual_hits"] >= 1 else 0 for r in hits_rows])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_h_accs, cv_h_briers = [], []
        for fold_train_idx, fold_val_idx in skf.split(hits_rows, y_hits_all):
            fold_train = [hits_rows[i] for i in fold_train_idx]
            fold_val   = [hits_rows[i] for i in fold_val_idx]
            fold_hm = HitsModel()
            try:
                fold_hm.train(fold_train)
                acc, brier = _hits_metrics(fold_hm, fold_val)
                cv_h_accs.append(acc)
                cv_h_briers.append(brier)
            except Exception as e:
                log.warning("Hits CV fold failed: %s", e)
        if cv_h_accs:
            log.info("Hits 5-fold CV — acc=%.3f±%.3f brier=%.4f±%.4f",
                     np.mean(cv_h_accs), np.std(cv_h_accs),
                     np.mean(cv_h_briers), np.std(cv_h_briers))

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
    if not train_hr:
        log.info("Skipping HR model (--model=%s)", models)
    else:
        log.info("Building HR training data (%d batters)…", len(batter_ids))
        hr_rows = _build_hr_rows(
            batter_ids, savant, savant, mlb_client, season,
            game_context_map=game_context_map,
            pitcher_stats_cache=pitcher_stats_cache,
            db=db,
            min_pa=min_pa,
        )

    if train_hr and len(hr_rows) < 500:
        log.warning("Only %d HR rows — skipping HR model training.", len(hr_rows))
    elif train_hr:
        hr_rows = _impute(hr_rows, HR_FEATURE_COLS)

        # 5-fold stratified CV
        y_hr_all = np.array([1 if r["actual_hr"] >= 1 else 0 for r in hr_rows])
        skf_hr = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_hr_accs, cv_hr_briers = [], []
        for fold_train_idx, fold_val_idx in skf_hr.split(hr_rows, y_hr_all):
            fold_train = [hr_rows[i] for i in fold_train_idx]
            fold_val   = [hr_rows[i] for i in fold_val_idx]
            fold_hrm = HRModel()
            try:
                fold_hrm.train(fold_train)
                acc, brier = _hr_metrics(fold_hrm, fold_val)
                cv_hr_accs.append(acc)
                cv_hr_briers.append(brier)
            except Exception as e:
                log.warning("HR CV fold failed: %s", e)
        if cv_hr_accs:
            log.info("HR 5-fold CV — acc=%.3f±%.3f brier=%.4f±%.4f",
                     np.mean(cv_hr_accs), np.std(cv_hr_accs),
                     np.mean(cv_hr_briers), np.std(cv_hr_briers))

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

    # 6. Outs model
    if not train_outs:
        log.info("Skipping Outs model (--model=%s)", models)
    else:
        log.info("Building Outs training data (%d pitchers)…", len(pitcher_ids))
        outs_rows = _build_outs_rows(pitcher_ids, savant, mlb_client, season, game_context_map, db, min_ip=min_ip)

        if len(outs_rows) < 500:
            log.warning("Only %d Outs rows — below 500 minimum. Skipping Outs model training.", len(outs_rows))
        else:
            outs_rows = _impute(outs_rows, OUTS_FEATURE_COLS)

            # 5-fold CV
            kf_outs = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_outs_maes: list[float] = []
            for fold_train_idx, fold_val_idx in kf_outs.split(outs_rows):
                fold_train = [outs_rows[i] for i in fold_train_idx]
                fold_val   = [outs_rows[i] for i in fold_val_idx]
                fold_om = OutsModel()
                try:
                    fold_om.train(fold_train)
                    cv_outs_maes.append(_outs_mae(fold_om, fold_val))
                except Exception as e:
                    log.warning("Outs CV fold failed: %s", e)
            if cv_outs_maes:
                log.info("Outs 5-fold CV — MAE=%.3f±%.3f", np.mean(cv_outs_maes), np.std(cv_outs_maes))

            outs_train, outs_test = train_test_split(outs_rows, test_size=0.15, random_state=42)
            log.info("Outs split: %d train / %d test", len(outs_train), len(outs_test))

            om = OutsModel()
            try:
                meta = om.train(outs_train)
                _report("OutsModel", meta)
                test_mae = _outs_mae(om, outs_test)
                log.info("  Test MAE (lambda vs actual_outs): %.3f", test_mae)
            except ValueError as e:
                log.error("OutsModel training failed: %s", e)
                om = None

            if om and not dry_run:
                om.save()
                log.info("OutsModel saved to %s", MODELS_DIR / "outs_model.pkl")

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


def _outs_mae(om: OutsModel, test_rows: list[dict]) -> float:
    errors = []
    for row in test_rows:
        try:
            lam = om.predict_lambda(row)
            errors.append(abs(lam - row["actual_outs"]))
        except Exception:
            continue
    return float(np.mean(errors)) if errors else float("nan")


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
    parser.add_argument(
        "--model",
        choices=["k", "hits", "hr", "outs", "all"],
        default="all",
        help="Which model(s) to train (default: all)",
    )
    args = parser.parse_args()

    run(
        season=args.season,
        min_ip=args.min_ip,
        min_pa=args.min_pa,
        dry_run=args.dry_run,
        models=args.model,
    )
