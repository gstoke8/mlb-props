#!/usr/bin/env python3
"""
Statcast Nightly Batch Fetcher

Nightly fetch of Statcast data via pybaseball. Stores raw plate-appearance
and pitch-level rows into the SQLite props.db, then computes rolling
batter and pitcher metrics for downstream prop modelling.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import pybaseball

from db import get_db

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "statcast_nightly_v1"

# pybaseball enforces ~25,000 rows per query; stay well under with 60-day chunks
CHUNK_DAYS = 60
MAX_QUERY_DAYS = 90  # threshold above which we split into chunks

INTER_CHUNK_SLEEP = 2.0   # seconds between chunk requests
INTER_PLAYER_SLEEP = 1.5  # seconds between player requests

HARD_HIT_THRESHOLD = 95.0  # launch_speed mph

# hc_x thresholds for pull/oppo/center classification
PULL_HC_X_RHH_MAX = 100.0   # right-handed hitters pull toward 1B side (low hc_x)
PULL_HC_X_LHH_MIN = 170.0   # left-handed hitters pull toward 3B side (high hc_x)
CENTER_HC_X_LOW   = 100.0
CENTER_HC_X_HIGH  = 170.0

# Statcast batter fields to extract per plate-appearance row
BATTER_FIELDS = [
    "game_date",
    "batter",
    "game_pk",
    "pitch_type",
    "events",
    "description",
    "stand",
    "p_throws",
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
    "woba_value",
    "barrel",
    "hc_x",
    "hc_y",
    "zone",          # 1-9 in-zone, 11-14 out-of-zone — required for chase rate
    "type",    # B/S/X — needed for zone contact rate (zone contact = in-zone pitches where type='X')
]

# Statcast pitcher fields to extract per pitch row
PITCHER_FIELDS = [
    "game_date",
    "pitcher",
    "game_pk",
    "pitch_type",
    "events",
    "description",
    "stand",
    "p_throws",
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "estimated_woba_using_speedangle",
    "woba_value",
    "barrel",
    "type",        # B / S / X — needed for CSW
    "release_speed",        # raw velocity (mph)
    "release_extension",    # extension toward home plate (feet)
    "release_pos_x",        # horizontal release point (feet, catcher's perspective)
    "pfx_z",               # induced vertical break (inches, after gravity removal)
    "pfx_x",               # horizontal break (inches)
]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _date_chunks(
    start_date: date, end_date: date, chunk_days: int = CHUNK_DAYS
) -> List[tuple[date, date]]:
    """Return non-overlapping (start, end) date pairs of at most chunk_days each."""
    chunks: List[tuple[date, date]] = []
    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end_date)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


def _needs_chunking(start_date: date, end_date: date) -> bool:
    return (end_date - start_date).days >= MAX_QUERY_DAYS


def _safe_float(value: Any) -> Optional[float]:
    """Convert a value to float, returning None on failure."""
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _keep_columns(df: pd.DataFrame, fields: List[str]) -> pd.DataFrame:
    """Return a copy of df with only the columns in fields that actually exist."""
    available = [c for c in fields if c in df.columns]
    return df[available].copy()


# ---------------------------------------------------------------------------
# Fetch functions
# ---------------------------------------------------------------------------

def fetch_statcast_batter(
    player_id: int,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Fetch Statcast batter data for a single player over a date range.

    Automatically chunks requests longer than MAX_QUERY_DAYS days to avoid
    the pybaseball 25,000-row hard limit. Sleeps INTER_CHUNK_SLEEP seconds
    between chunk requests.

    Parameters
    ----------
    player_id:
        MLB MLBAM player ID.
    start_date:
        Inclusive start of the query window.
    end_date:
        Inclusive end of the query window.

    Returns
    -------
    pd.DataFrame
        Combined results; empty DataFrame if no data or on error.
    """
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    if not _needs_chunking(start_date, end_date):
        return _fetch_batter_chunk(player_id, start_str, end_str)

    chunks = _date_chunks(start_date, end_date)
    log.debug(
        "Splitting batter fetch for player %s into %d chunks", player_id, len(chunks)
    )
    frames: List[pd.DataFrame] = []
    for i, (chunk_start, chunk_end) in enumerate(chunks):
        if i > 0:
            time.sleep(INTER_CHUNK_SLEEP)
        frame = _fetch_batter_chunk(
            player_id,
            chunk_start.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        )
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _fetch_batter_chunk(player_id: int, start_str: str, end_str: str) -> pd.DataFrame:
    try:
        df = pybaseball.statcast_batter(start_str, end_str, player_id=player_id)
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except ConnectionError as exc:
        log.warning(
            "ConnectionError fetching batter %s (%s–%s): %s",
            player_id, start_str, end_str, exc,
        )
        return pd.DataFrame()
    except ValueError as exc:
        log.warning(
            "ValueError fetching batter %s (%s–%s): %s",
            player_id, start_str, end_str, exc,
        )
        return pd.DataFrame()
    except Exception as exc:  # noqa: BLE001
        log.error(
            "Unexpected error fetching batter %s (%s–%s): %s",
            player_id, start_str, end_str, exc,
        )
        return pd.DataFrame()


def fetch_statcast_pitcher(
    player_id: int,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Fetch Statcast pitcher data for a single player over a date range.

    Same chunking and error-handling logic as fetch_statcast_batter.

    Parameters
    ----------
    player_id:
        MLB MLBAM player ID.
    start_date:
        Inclusive start of the query window.
    end_date:
        Inclusive end of the query window.

    Returns
    -------
    pd.DataFrame
        Combined results; empty DataFrame if no data or on error.
    """
    if not _needs_chunking(start_date, end_date):
        return _fetch_pitcher_chunk(
            player_id,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

    chunks = _date_chunks(start_date, end_date)
    log.debug(
        "Splitting pitcher fetch for player %s into %d chunks", player_id, len(chunks)
    )
    frames: List[pd.DataFrame] = []
    for i, (chunk_start, chunk_end) in enumerate(chunks):
        if i > 0:
            time.sleep(INTER_CHUNK_SLEEP)
        frame = _fetch_pitcher_chunk(
            player_id,
            chunk_start.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        )
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _fetch_pitcher_chunk(player_id: int, start_str: str, end_str: str) -> pd.DataFrame:
    try:
        df = pybaseball.statcast_pitcher(start_str, end_str, player_id=player_id)
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except ConnectionError as exc:
        log.warning(
            "ConnectionError fetching pitcher %s (%s–%s): %s",
            player_id, start_str, end_str, exc,
        )
        return pd.DataFrame()
    except ValueError as exc:
        log.warning(
            "ValueError fetching pitcher %s (%s–%s): %s",
            player_id, start_str, end_str, exc,
        )
        return pd.DataFrame()
    except Exception as exc:  # noqa: BLE001
        log.error(
            "Unexpected error fetching pitcher %s (%s–%s): %s",
            player_id, start_str, end_str, exc,
        )
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Nightly upsert runners
# ---------------------------------------------------------------------------

def run_batter_nightly(
    player_ids: Sequence[int],
    lookback_days: int = 180,
    db=None,
) -> int:
    """Fetch and store Statcast batter data for a list of players.

    For each player, fetches the last lookback_days of plate-appearance data
    and upserts each row into the player_stats table with stat_type='statcast_pa'.

    Parameters
    ----------
    player_ids:
        Iterable of MLB MLBAM player IDs.
    lookback_days:
        Number of calendar days to look back from today.
    db:
        An MLBPropsDB instance. If None, one is created via get_db().

    Returns
    -------
    int
        Total number of rows upserted.
    """
    database = db if db is not None else get_db()
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)
    total_upserted = 0

    for i, player_id in enumerate(player_ids):
        if i > 0:
            time.sleep(INTER_PLAYER_SLEEP)

        log.info("Fetching Statcast batter data for player_id=%s", player_id)
        raw_df = fetch_statcast_batter(player_id, start_date, end_date)

        if raw_df.empty:
            log.debug("No batter data returned for player_id=%s", player_id)
            continue

        df = _keep_columns(raw_df, BATTER_FIELDS)
        count = _upsert_batter_rows(player_id, df, database)
        total_upserted += count
        log.info(
            "Upserted %d statcast_pa rows for player_id=%s", count, player_id
        )

        # Derived metrics from the raw DataFrame (chase rate, pitch-type whiff)
        try:
            _compute_batter_chase_rate(player_id, df, database)
        except Exception as exc:
            log.warning("chase_rate compute failed player_id=%s: %s", player_id, exc)

        try:
            _compute_batter_pitch_type_whiff(player_id, df, database)
        except Exception as exc:
            log.warning("batter_pitch_type_whiff compute failed player_id=%s: %s", player_id, exc)

        try:
            _compute_batter_zone_contact(player_id, df, database)
        except Exception as exc:
            log.warning("batter_zone_contact compute failed player_id=%s: %s", player_id, exc)

        try:
            _compute_batter_batted_ball_stats(player_id, df, database)
        except Exception as exc:
            log.warning("batter_batted_ball_stats compute failed player_id=%s: %s", player_id, exc)

    return total_upserted


def _upsert_batter_rows(player_id: int, df: pd.DataFrame, database) -> int:
    """Upsert each row of batter DataFrame into player_stats."""
    count = 0
    for _, row in df.iterrows():
        game_date = _row_game_date(row)
        if game_date is None:
            continue

        # Pack non-scalar fields into the value column as a JSON-encoded float
        # The primary key is (player_id, stat_date, stat_type); we use game_pk
        # in the stat_date prefix to allow multiple PAs per day.
        game_pk = row.get("game_pk")
        stat_date = (
            f"{game_date}_{int(game_pk)}" if pd.notna(game_pk) else game_date
        )

        # Store xwOBA as the primary value; extras as rolling columns
        xwoba = _safe_float(row.get("estimated_woba_using_speedangle"))
        barrel = _safe_float(row.get("barrel"))
        launch_speed = _safe_float(row.get("launch_speed"))
        launch_angle = _safe_float(row.get("launch_angle"))

        database.upsert_player_stat(
            player_id=player_id,
            stat_date=stat_date,
            stat_type="statcast_pa",
            value=xwoba if xwoba is not None else 0.0,
            rolling_3=barrel,
            rolling_7=launch_speed,
            rolling_14=launch_angle,
            rolling_30=_safe_float(row.get("woba_value")),
        )
        count += 1
    return count


def run_pitcher_nightly(
    player_ids: Sequence[int],
    lookback_days: int = 180,
    db=None,
) -> int:
    """Fetch and store Statcast pitcher data for a list of players.

    For each player, fetches the last lookback_days of pitch-level data
    and upserts each row into the player_stats table with stat_type='statcast_pitch'.

    Parameters
    ----------
    player_ids:
        Iterable of MLB MLBAM player IDs.
    lookback_days:
        Number of calendar days to look back from today.
    db:
        An MLBPropsDB instance. If None, one is created via get_db().

    Returns
    -------
    int
        Total number of rows upserted.
    """
    database = db if db is not None else get_db()
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)
    total_upserted = 0

    for i, player_id in enumerate(player_ids):
        if i > 0:
            time.sleep(INTER_PLAYER_SLEEP)

        log.info("Fetching Statcast pitcher data for player_id=%s", player_id)
        raw_df = fetch_statcast_pitcher(player_id, start_date, end_date)

        if raw_df.empty:
            log.debug("No pitcher data returned for player_id=%s", player_id)
            continue

        df = _keep_columns(raw_df, PITCHER_FIELDS)
        count = _upsert_pitcher_rows(player_id, df, database)
        total_upserted += count
        log.info(
            "Upserted %d statcast_pitch rows for player_id=%s", count, player_id
        )

        # Derived metrics from the raw DataFrame (swstr% by pitch type + pitch mix)
        try:
            _compute_pitcher_swstr_stats(player_id, df, database)
        except Exception as exc:
            log.warning("swstr_stats compute failed player_id=%s: %s", player_id, exc)

        try:
            _compute_pitcher_pitch_mix(player_id, df, database)
        except Exception as exc:
            log.warning("pitcher_pitch_mix compute failed player_id=%s: %s", player_id, exc)

        try:
            _compute_pitcher_movement_stats(player_id, df, database)
        except Exception as exc:
            log.warning("pitcher_movement_stats compute failed player_id=%s: %s", player_id, exc)

        try:
            _compute_pitcher_contact_quality_allowed(player_id, df, database)
        except Exception as exc:
            log.warning("pitcher_contact_quality_allowed compute failed player_id=%s: %s", player_id, exc)

    return total_upserted


def _upsert_pitcher_rows(player_id: int, df: pd.DataFrame, database) -> int:
    """Upsert each row of pitcher DataFrame into player_stats."""
    count = 0
    for _, row in df.iterrows():
        game_date = _row_game_date(row)
        if game_date is None:
            continue

        game_pk = row.get("game_pk")
        stat_date = (
            f"{game_date}_{int(game_pk)}" if pd.notna(game_pk) else game_date
        )

        launch_speed = _safe_float(row.get("launch_speed"))
        barrel = _safe_float(row.get("barrel"))

        database.upsert_player_stat(
            player_id=player_id,
            stat_date=stat_date,
            stat_type="statcast_pitch",
            value=1.0,
            rolling_3=barrel,
            rolling_7=launch_speed,
            rolling_14=_safe_float(row.get("launch_angle")),
            rolling_30=_safe_float(row.get("woba_value")),
        )
        count += 1
    return count


# ---------------------------------------------------------------------------
# Derived metrics computed from raw DataFrames at ingest time
# ---------------------------------------------------------------------------

_SWINGING_STRIKE_DESCS = frozenset({
    "swinging_strike", "swinging_strike_blocked", "foul_tip",
})
_SWING_DESCS = frozenset({
    "hit_into_play", "swinging_strike", "swinging_strike_blocked",
    "foul", "foul_tip", "foul_bunt",
})
# Pitch-type codes mapped to storage labels
_PITCH_TYPE_LABELS: Dict[str, str] = {
    "FF": "ff",   # four-seam fastball
    "SL": "sl",   # slider
    "CH": "ch",   # changeup
    "SI": "si",   # sinker
    "CU": "cu",   # curveball
}
_MIN_PITCHES_FOR_TYPE = 20   # require ≥20 pitches of a type to store whiff rate
_MIN_SWINGS_FOR_CHASE = 15   # require ≥15 out-of-zone pitches for chase rate


def _compute_pitcher_swstr_stats(
    player_id: int,
    df: pd.DataFrame,
    database,
) -> None:
    """Compute and store swinging-strike rates from a raw pitcher DataFrame.

    Stores:
      - ``pitcher_rolling_swstr_rate``  — overall swinging-strike % (last ~180d)
      - ``pitcher_rolling_{pt}_whiff``  — per-pitch-type whiff % (FF/SL/CH/SI/CU)

    All stored under today's date so downstream readers use *days=1* to get
    the latest value.
    """
    if df.empty:
        return

    if "description" not in df.columns or "pitch_type" not in df.columns:
        log.debug("pitcher swstr: missing description/pitch_type cols for player %s", player_id)
        return

    today_str = date.today().strftime("%Y-%m-%d")
    total = len(df)
    if total == 0:
        return

    # Overall swinging-strike rate
    n_swstr = int(df["description"].isin(_SWINGING_STRIKE_DESCS).sum())
    swstr_rate = n_swstr / total
    database.upsert_player_stat(
        player_id=player_id,
        stat_date=today_str,
        stat_type="pitcher_rolling_swstr_rate",
        value=swstr_rate,
    )

    # CSW rate: Called Strike + Whiff / total pitches. Typical: 27-32%; swStr: 10-14%.
    n_called = int((df["description"] == "called_strike").sum())
    csw_rate = (n_swstr + n_called) / total
    database.upsert_player_stat(
        player_id=player_id,
        stat_date=today_str,
        stat_type="pitcher_rolling_csw_rate",
        value=csw_rate,
    )

    # Whiff rate: swinging strikes / swings (not / total pitches).
    # Typical: 22-35%. This is what Baseball Savant's whiff_percent reports.
    n_swings = int(df["description"].isin(_SWING_DESCS).sum())
    whiff_rate = n_swstr / n_swings if n_swings > 0 else swstr_rate
    database.upsert_player_stat(
        player_id=player_id,
        stat_date=today_str,
        stat_type="pitcher_rolling_whiff_rate",
        value=whiff_rate,
    )

    # Per-pitch-type whiff rates
    for pt_code, pt_label in _PITCH_TYPE_LABELS.items():
        pt_mask = df["pitch_type"] == pt_code
        pt_df = df[pt_mask]
        if len(pt_df) < _MIN_PITCHES_FOR_TYPE:
            continue
        n_whiff = int(pt_df["description"].isin(_SWINGING_STRIKE_DESCS).sum())
        whiff_rate = n_whiff / len(pt_df)
        database.upsert_player_stat(
            player_id=player_id,
            stat_date=today_str,
            stat_type=f"pitcher_rolling_{pt_label}_whiff",
            value=whiff_rate,
        )

    log.debug(
        "pitcher_swstr player_id=%s: swstr=%.3f over %d pitches",
        player_id, swstr_rate, total,
    )


def _compute_pitcher_pitch_mix(
    player_id: int,
    df: pd.DataFrame,
    database,
) -> None:
    """Compute and store pitcher pitch-type mix in ``pitch_type_performance``.

    Computes the fraction of pitches thrown for each pitch type (FF/SL/CH/SI/CU
    plus CU/KC/FC/FS bucketed as-is) over the full lookback window.  Also stores
    the per-pitch-type whiff rate in the same row so ``get_pitcher_pitch_mix``
    gets real data on every lookup.

    Skips pitch types with fewer than ``_MIN_PITCHES_FOR_TYPE`` pitches.
    """
    if df.empty or "pitch_type" not in df.columns:
        return

    pitch_counts = df["pitch_type"].dropna().value_counts()
    total = int(pitch_counts.sum())
    if total < _MIN_PITCHES_FOR_TYPE:
        return

    for pt_code, n_pitches in pitch_counts.items():
        pt_code = str(pt_code)
        if n_pitches < _MIN_PITCHES_FOR_TYPE:
            continue
        pct = float(n_pitches) / total

        whiff_pct: Optional[float] = None
        if "description" in df.columns:
            pt_df = df[df["pitch_type"] == pt_code]
            n_whiff = int(pt_df["description"].isin(_SWINGING_STRIKE_DESCS).sum())
            whiff_pct = n_whiff / len(pt_df)

        try:
            database.upsert_pitch_type_perf({
                "player_id":       str(player_id),
                "player_type":     "pitcher",
                "pitch_type":      pt_code,
                "pct_thrown_seen": pct,
                "whiff_pct":       whiff_pct,
                "pa_or_pitches":   int(n_pitches),
            })
        except Exception as exc:
            log.warning(
                "upsert_pitch_type_perf (pitcher mix) failed player_id=%s pt=%s: %s",
                player_id, pt_code, exc,
            )

    log.debug(
        "pitcher_pitch_mix player_id=%s: %d pitch types over %d pitches",
        player_id, len(pitch_counts), total,
    )


_MIN_PITCHES_FOR_MOVEMENT = 50  # minimum pitches to compute movement stats

def _compute_pitcher_movement_stats(
    player_id: int,
    df: pd.DataFrame,
    database,
) -> None:
    """Compute and store pitcher movement, release, velocity, and foul-rate stats.

    Stores (all under today's date):
      - pitcher_rolling_foul_rate     — foul balls / total pitches (xK% component)
      - pitcher_rolling_max_vbreak    — max IVB across pitch types (inches)
      - pitcher_rolling_vbreak_range  — range of IVB across pitch types (inches)
      - pitcher_rolling_ff_perceived_velo — extension-adjusted fastball effective velocity
      - pitcher_rolling_rp_horiz_std  — SD of horizontal release point (command proxy)
      - pitcher_rolling_stuff_plus    — simple Stuff+ proxy: (velo - 92) * 0.3 + ivb * 0.5
    """
    if df.empty:
        return

    today_str = date.today().strftime("%Y-%m-%d")
    total = len(df)
    if total < _MIN_PITCHES_FOR_MOVEMENT:
        return

    # Foul rate: foul / total pitches
    if "description" in df.columns:
        _FOUL_DESCS = frozenset({"foul", "foul_tip", "foul_bunt"})
        n_foul = int(df["description"].isin(_FOUL_DESCS).sum())
        foul_rate = n_foul / total
        database.upsert_player_stat(
            player_id=player_id,
            stat_date=today_str,
            stat_type="pitcher_rolling_foul_rate",
            value=foul_rate,
        )

    # IVB (pfx_z) by pitch type — max and range
    if "pfx_z" in df.columns and "pitch_type" in df.columns:
        pt_ivb: dict[str, float] = {}
        for pt_code in df["pitch_type"].dropna().unique():
            pt_df = df[df["pitch_type"] == pt_code]
            if len(pt_df) < 10:
                continue
            valid_ivb = pt_df["pfx_z"].dropna()
            valid_ivb = valid_ivb[np.isfinite(valid_ivb)]
            if len(valid_ivb) >= 10:
                pt_ivb[str(pt_code)] = float(valid_ivb.mean())

        if pt_ivb:
            ivb_vals = list(pt_ivb.values())
            max_vbreak = max(ivb_vals)
            vbreak_range = max(ivb_vals) - min(ivb_vals)
            database.upsert_player_stat(
                player_id=player_id,
                stat_date=today_str,
                stat_type="pitcher_rolling_max_vbreak",
                value=max_vbreak,
            )
            database.upsert_player_stat(
                player_id=player_id,
                stat_date=today_str,
                stat_type="pitcher_rolling_vbreak_range",
                value=vbreak_range,
            )

    # Perceived velocity: release_speed + (release_extension - 6.0) * 1.3 for FF/SI
    if "release_speed" in df.columns and "release_extension" in df.columns and "pitch_type" in df.columns:
        ff_mask = df["pitch_type"].isin(["FF", "SI", "FC"])
        ff_df = df[ff_mask]
        if len(ff_df) >= 20:
            # Use rows where both are available
            valid_mask = ff_df["release_speed"].notna() & ff_df["release_extension"].notna()
            if valid_mask.sum() >= 10:
                v_arr = ff_df.loc[valid_mask, "release_speed"].values.astype(float)
                e_arr = ff_df.loc[valid_mask, "release_extension"].values.astype(float)
                valid = np.isfinite(v_arr) & np.isfinite(e_arr)
                if valid.sum() >= 10:
                    perceived = v_arr[valid] + (e_arr[valid] - 6.0) * 1.3
                    database.upsert_player_stat(
                        player_id=player_id,
                        stat_date=today_str,
                        stat_type="pitcher_rolling_ff_perceived_velo",
                        value=float(np.mean(perceived)),
                    )

    # Release point horizontal consistency (SD of release_pos_x)
    if "release_pos_x" in df.columns:
        rx = df["release_pos_x"].dropna()
        rx = rx[np.isfinite(rx)]
        if len(rx) >= 30:
            rp_std = float(rx.std())
            database.upsert_player_stat(
                player_id=player_id,
                stat_date=today_str,
                stat_type="pitcher_rolling_rp_horiz_std",
                value=rp_std,
            )

    # Stuff+ proxy: simple linear combination of average fastball velo and max IVB
    if "release_speed" in df.columns and "pfx_z" in df.columns and "pitch_type" in df.columns:
        ff_df = df[df["pitch_type"].isin(["FF", "SI"])]
        if len(ff_df) >= 20:
            ff_velo = ff_df["release_speed"].dropna()
            ff_ivb = ff_df["pfx_z"].dropna()
            ff_velo = ff_velo[np.isfinite(ff_velo)]
            ff_ivb = ff_ivb[np.isfinite(ff_ivb)]
            if len(ff_velo) >= 10 and len(ff_ivb) >= 10:
                # Simple Stuff+ proxy centered at 100 (league avg ~92 mph, ~12 inches IVB)
                stuff_proxy = 100.0 + (float(ff_velo.mean()) - 92.0) * 3.0 + (float(ff_ivb.mean()) - 12.0) * 2.0
                database.upsert_player_stat(
                    player_id=player_id,
                    stat_date=today_str,
                    stat_type="pitcher_rolling_stuff_plus",
                    value=round(stuff_proxy, 1),
                )


def _compute_batter_pitch_type_whiff(
    player_id: int,
    df: pd.DataFrame,
    database,
) -> None:
    """Compute and store per-pitch-type whiff rates from a raw batter DataFrame.

    For each pitch type in ``_PITCH_TYPE_LABELS`` (FF/SL/CH/SI/CU), computes:
        whiff_pct = swinging_strikes_on_that_type / total_pitches_of_that_type

    Skips types with fewer than ``_MIN_PITCHES_FOR_TYPE`` pitches seen.
    Stores rows via ``upsert_pitch_type_perf`` with ``player_type='batter'``.
    """
    if df.empty:
        return

    if "description" not in df.columns or "pitch_type" not in df.columns:
        log.debug(
            "batter_whiff: missing description/pitch_type cols for player %s", player_id
        )
        return

    for pt_code in _PITCH_TYPE_LABELS:
        pt_mask = df["pitch_type"] == pt_code
        pt_df = df[pt_mask]
        n_pitches = len(pt_df)
        if n_pitches < _MIN_PITCHES_FOR_TYPE:
            continue

        n_whiff = int(pt_df["description"].isin(_SWINGING_STRIKE_DESCS).sum())
        whiff_pct = n_whiff / n_pitches

        try:
            database.upsert_pitch_type_perf({
                "player_id":       str(player_id),
                "player_type":     "batter",
                "pitch_type":      pt_code,
                "whiff_pct":       whiff_pct,
                "pa_or_pitches":   n_pitches,
            })
        except Exception as exc:
            log.warning(
                "upsert_pitch_type_perf failed player_id=%s pt=%s: %s",
                player_id, pt_code, exc,
            )

    log.debug(
        "batter_pitch_type_whiff player_id=%s: processed %d pitches",
        player_id, len(df),
    )


def _compute_batter_chase_rate(
    player_id: int,
    df: pd.DataFrame,
    database,
) -> None:
    """Compute and store chase rate (O-Swing%) from raw batter Statcast data.

    Zones 11-14 are out-of-zone pitches; swings on those pitches = chases.
    Stores as ``batter_rolling_chase_rate`` under today's date.
    """
    if df.empty:
        return

    if "zone" not in df.columns or "description" not in df.columns:
        log.debug("batter chase_rate: missing zone/description cols for player %s", player_id)
        return

    today_str = date.today().strftime("%Y-%m-%d")

    # Out-of-zone pitches: zone values 11-14
    ooz_mask = df["zone"].notna() & (df["zone"] > 9)
    ooz_df = df[ooz_mask]
    if len(ooz_df) < _MIN_SWINGS_FOR_CHASE:
        return

    n_chases = int(ooz_df["description"].isin(_SWING_DESCS).sum())
    chase_rate = n_chases / len(ooz_df)

    database.upsert_player_stat(
        player_id=player_id,
        stat_date=today_str,
        stat_type="batter_rolling_chase_rate",
        value=chase_rate,
    )
    log.debug(
        "batter_chase player_id=%s: chase=%.3f (%d ooz pitches)",
        player_id, chase_rate, len(ooz_df),
    )


_MIN_IN_ZONE_FOR_CONTACT = 30  # minimum in-zone pitches

def _compute_batter_zone_contact(
    player_id: int,
    df: pd.DataFrame,
    database,
) -> None:
    """Compute and store zone contact rate (Z-Con%) from raw batter Statcast data.

    In-zone pitches: zone values 1-9.
    Zone contact = in-zone swings that result in contact (hit_into_play or foul), not whiff.
    Stores as batter_rolling_zone_contact under today's date.
    """
    if df.empty:
        return

    if "zone" not in df.columns or "description" not in df.columns:
        return

    today_str = date.today().strftime("%Y-%m-%d")

    # In-zone pitches: zone 1-9
    inzone_mask = df["zone"].notna() & (df["zone"] >= 1) & (df["zone"] <= 9)
    inzone_df = df[inzone_mask]

    if len(inzone_df) < _MIN_IN_ZONE_FOR_CONTACT:
        return

    # In-zone swings: description is a swing (hit, whiff, foul)
    _CONTACT_DESCS = frozenset({"hit_into_play", "foul", "foul_tip", "foul_bunt"})
    _INZONE_SWING_DESCS = _SWING_DESCS  # defined at module level

    inzone_swings = inzone_df[inzone_df["description"].isin(_INZONE_SWING_DESCS)]
    if len(inzone_swings) < 10:
        return

    inzone_contact = inzone_df[inzone_df["description"].isin(_CONTACT_DESCS)]
    zone_contact_rate = len(inzone_contact) / len(inzone_swings)

    database.upsert_player_stat(
        player_id=player_id,
        stat_date=today_str,
        stat_type="batter_rolling_zone_contact",
        value=zone_contact_rate,
    )


_BATTED_BALL_EVENTS = frozenset({
    "single", "double", "triple", "home_run", "field_out",
    "grounded_into_double_play", "double_play", "sac_fly",
    "sac_fly_double_play", "fielders_choice", "fielders_choice_out",
    "force_out", "sac_bunt", "catcher_interf", "fan_interference",
})
_MIN_BATTED_BALLS = 15


def _compute_batter_batted_ball_stats(
    player_id: int,
    df: pd.DataFrame,
    database,
) -> None:
    """Compute fly_ball_rate, sweet_spot_pct, and pull_pct from raw batter Statcast data.

    Stores (under today's date):
      - batter_rolling_fly_ball_rate  — FB% (launch angle 25–50°) of all BBE
      - batter_rolling_sweet_spot_pct — sweet spot% (8–32° LA) of all BBE
      - pull_pct                      — fraction of BBE hit to the pull side (via hc_x + stand)
    """
    if df.empty or "launch_angle" not in df.columns:
        return

    today_str = date.today().strftime("%Y-%m-%d")

    if "events" in df.columns:
        bb_df = df[df["events"].isin(_BATTED_BALL_EVENTS) & df["launch_angle"].notna()]
    else:
        bb_df = df[df["launch_angle"].notna()]

    if len(bb_df) < _MIN_BATTED_BALLS:
        return

    la_vals = bb_df["launch_angle"].values.astype(float)
    valid_la = la_vals[np.isfinite(la_vals)]

    if len(valid_la) < _MIN_BATTED_BALLS:
        return

    fly_ball_rate = float(np.sum((valid_la >= 25) & (valid_la <= 50)) / len(valid_la))
    database.upsert_player_stat(
        player_id=player_id,
        stat_date=today_str,
        stat_type="batter_rolling_fly_ball_rate",
        value=fly_ball_rate,
    )

    sweet_spot_rate = float(np.sum((valid_la >= 8) & (valid_la <= 32)) / len(valid_la))
    database.upsert_player_stat(
        player_id=player_id,
        stat_date=today_str,
        stat_type="batter_rolling_sweet_spot_pct",
        value=sweet_spot_rate,
    )

    # pull_pct: fraction of BBE hit to the pull side using hc_x and batter handedness.
    # RHH pull = hc_x <= PULL_HC_X_RHH_MAX (toward 1B/right field side at low hc_x).
    # LHH pull = hc_x >= PULL_HC_X_LHH_MIN (toward 3B/left field side at high hc_x).
    if "hc_x" in bb_df.columns and "stand" in bb_df.columns:
        hcx_df = bb_df[bb_df["hc_x"].notna() & bb_df["stand"].notna()].copy()
        if len(hcx_df) >= _MIN_BATTED_BALLS:
            hcx = hcx_df["hc_x"].values.astype(float)
            stand = hcx_df["stand"].values
            pulled = np.where(
                stand == "R",
                hcx <= PULL_HC_X_RHH_MAX,
                hcx >= PULL_HC_X_LHH_MIN,
            )
            pull_pct = float(np.sum(pulled) / len(pulled))
            database.upsert_player_stat(
                player_id=player_id,
                stat_date=today_str,
                stat_type="pull_pct",
                value=pull_pct,
            )
            log.debug(
                "batter_batted_ball player_id=%s: fb=%.3f ss=%.3f pull=%.3f (%d BBE)",
                player_id, fly_ball_rate, sweet_spot_rate, pull_pct, len(valid_la),
            )
            return

    log.debug(
        "batter_batted_ball player_id=%s: fb=%.3f ss=%.3f (%d BBE)",
        player_id, fly_ball_rate, sweet_spot_rate, len(valid_la),
    )


_MIN_PITCHER_BBE = 20


def _compute_pitcher_contact_quality_allowed(
    player_id: int,
    df: pd.DataFrame,
    database,
) -> None:
    """Compute barrel_rate_allowed and hard_hit_pct_allowed from pitcher Statcast data.

    Stores (under today's date):
      - pitcher_rolling_barrel_rate_allowed  — barrels / total BBE
      - pitcher_rolling_hard_hit_pct_allowed — hard hits (≥95 mph EV) / BBE with speed
    """
    if df.empty or "events" not in df.columns:
        return

    today_str = date.today().strftime("%Y-%m-%d")

    bb_df = df[df["events"].isin(_BATTED_BALL_EVENTS)]
    if len(bb_df) < _MIN_PITCHER_BBE:
        return

    if "barrel" in bb_df.columns:
        barrels = bb_df["barrel"].fillna(0).values.astype(float)
        barrel_rate = float(np.sum(barrels >= 1.0) / len(bb_df))
        database.upsert_player_stat(
            player_id=player_id,
            stat_date=today_str,
            stat_type="pitcher_rolling_barrel_rate_allowed",
            value=barrel_rate,
        )

    if "launch_speed" in bb_df.columns:
        speed_df = bb_df[bb_df["launch_speed"].notna()]
        if len(speed_df) >= _MIN_PITCHER_BBE:
            speeds = speed_df["launch_speed"].values.astype(float)
            valid_speeds = speeds[np.isfinite(speeds)]
            if len(valid_speeds) >= _MIN_PITCHER_BBE:
                hard_hit_pct = float(np.sum(valid_speeds >= HARD_HIT_THRESHOLD) / len(valid_speeds))
                database.upsert_player_stat(
                    player_id=player_id,
                    stat_date=today_str,
                    stat_type="pitcher_rolling_hard_hit_pct_allowed",
                    value=hard_hit_pct,
                )

    log.debug(
        "pitcher_contact_quality player_id=%s: %d BBE processed",
        player_id, len(bb_df),
    )


# ---------------------------------------------------------------------------
# Sprint speed fetch
# ---------------------------------------------------------------------------

def fetch_sprint_speed(season: int) -> pd.DataFrame:
    """Fetch MLB sprint speed (ft/s) for all qualified players in *season*.

    Uses pybaseball.statcast_sprint_speed(). Returns empty DataFrame on error.
    """
    try:
        df = pybaseball.statcast_sprint_speed(season)
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception as exc:
        log.warning("statcast_sprint_speed(%d) failed: %s", season, exc)
        return pd.DataFrame()


def run_sprint_speed_nightly(
    player_ids: Sequence[int],
    season: int,
    db=None,
) -> int:
    """Store sprint speed for *player_ids* from the season-level dataset.

    Returns
    -------
    int
        Number of players with sprint speed stored.
    """
    database = db if db is not None else get_db()
    df = fetch_sprint_speed(season)
    if df.empty:
        log.warning("No sprint speed data returned for season %d", season)
        return 0

    today_str = date.today().strftime("%Y-%m-%d")
    pid_set = {int(p) for p in player_ids}
    count = 0

    # pybaseball uses 'player_id' or 'mlbam_id' depending on version
    id_col = next((c for c in ("player_id", "mlbam_id") if c in df.columns), None)
    speed_col = next((c for c in ("sprint_speed", "ft_sec") if c in df.columns), None)

    if id_col is None or speed_col is None:
        log.warning("sprint speed DataFrame missing expected columns: %s", list(df.columns))
        return 0

    for _, row in df.iterrows():
        raw_pid = row.get(id_col)
        if raw_pid is None:
            continue
        try:
            pid = int(raw_pid)
        except (TypeError, ValueError):
            continue
        if pid not in pid_set:
            continue

        speed = _safe_float(row.get(speed_col))
        if speed is None:
            continue

        database.upsert_player_stat(
            player_id=pid,
            stat_date=today_str,
            stat_type="batter_sprint_speed",
            value=speed,
        )
        count += 1

    log.info("run_sprint_speed_nightly: stored sprint speed for %d / %d players", count, len(pid_set))
    return count


# ---------------------------------------------------------------------------
# Rolling stat computations
# ---------------------------------------------------------------------------

def compute_batter_rolling_stats(
    player_id: int,
    db,
    days: int = 30,
) -> Dict[str, Optional[float]]:
    """Compute rolling batter metrics from stored statcast_pa rows.

    Fetches the last `days` days of statcast_pa rows for the player and
    computes barrel_rate, hard_hit_rate, avg_launch_angle, pull/oppo/center
    percentages, and xwOBA_30d. Each metric is upserted as its own
    stat_type='batter_rolling' row in player_stats.

    Parameters
    ----------
    player_id:
        MLB MLBAM player ID.
    db:
        An MLBPropsDB instance.
    days:
        Rolling window in calendar days.

    Returns
    -------
    dict
        Computed metric name → float value (or None if insufficient data).
    """
    raw_rows = db.get_player_stats(player_id, stat_type="statcast_pa", days=days)
    if not raw_rows:
        log.debug(
            "No statcast_pa rows for player_id=%s in last %d days", player_id, days
        )
        return {}

    pa_count = len(raw_rows)
    # rolling_3=barrel, rolling_7=launch_speed, rolling_14=launch_angle,
    # rolling_30=woba_value, value=xwoba

    barrels = [r["rolling_3"] for r in raw_rows if r["rolling_3"] is not None]
    launch_speeds = [r["rolling_7"] for r in raw_rows if r["rolling_7"] is not None]
    launch_angles = [r["rolling_14"] for r in raw_rows if r["rolling_14"] is not None]
    xwoba_vals = [r["value"] for r in raw_rows if r["value"] is not None]

    barrel_count = sum(1 for b in barrels if b >= 1.0)
    barrel_rate = barrel_count / pa_count if pa_count > 0 else None

    batted_balls = [s for s in launch_speeds if s is not None]
    hard_hit_count = sum(1 for s in batted_balls if s >= HARD_HIT_THRESHOLD)
    hard_hit_rate = hard_hit_count / len(batted_balls) if batted_balls else None

    avg_launch_angle = (
        sum(launch_angles) / len(launch_angles) if launch_angles else None
    )

    avg_exit_velocity = (
        sum(batted_balls) / len(batted_balls) if batted_balls else None
    )

    xwoba_30d = sum(xwoba_vals) / len(xwoba_vals) if xwoba_vals else None

    today_str = date.today().strftime("%Y-%m-%d")

    metrics: Dict[str, Optional[float]] = {
        "barrel_rate": barrel_rate,
        "hard_hit_rate": hard_hit_rate,
        "avg_launch_angle": avg_launch_angle,
        "avg_exit_velocity": avg_exit_velocity,
        "xwOBA_30d": xwoba_30d,
    }

    for metric_name, metric_value in metrics.items():
        if metric_value is not None:
            db.upsert_player_stat(
                player_id=player_id,
                stat_date=today_str,
                stat_type=f"batter_rolling_{metric_name}",
                value=metric_value,
            )

    log.debug(
        "Computed batter rolling stats for player_id=%s: %s", player_id, metrics
    )
    return metrics


def _compute_spray_percentages(
    player_id: int,
    raw_rows: List[Dict],
) -> Dict[str, Optional[float]]:
    """Derive pull/oppo/center split from stored features (hc_x in feature_json)."""
    # hc_x was stored in the features table by save_features when available.
    # For rows we stored via upsert_player_stat the hc_x is not persisted in
    # player_stats. Return None to indicate unavailable unless caller enriches.
    return {"pull_pct": None, "oppo_pct": None, "center_pct": None}


def compute_pitcher_rolling_stats(
    player_id: int,
    db,
    days: int = 30,
) -> Dict[str, Any]:
    """Compute rolling pitcher metrics from stored statcast_pitch rows.

    Fetches the last `days` days of statcast_pitch rows and computes
    k_rate, pitch_mix, and csw_rate. Each scalar metric is upserted as
    stat_type='pitcher_rolling_<metric>'. pitch_mix is saved via
    db.save_features().

    Parameters
    ----------
    player_id:
        MLB MLBAM player ID.
    db:
        An MLBPropsDB instance.
    days:
        Rolling window in calendar days.

    Returns
    -------
    dict
        Computed metric name → value (float or dict).
    """
    raw_rows = db.get_player_stats(player_id, stat_type="statcast_pitch", days=days)
    if not raw_rows:
        log.debug(
            "No statcast_pitch rows for player_id=%s in last %d days", player_id, days
        )
        return {}

    total_pitches = len(raw_rows)
    today_str = date.today().strftime("%Y-%m-%d")

    # k_rate: rows where woba_value (rolling_30) is used as a proxy is not
    # reliable; events are not stored in player_stats. We store a placeholder.
    # csw (called strike + swinging strike): not directly stored either.
    # These will be None until the pipeline stores richer per-pitch data.
    k_rate: Optional[float] = None
    csw_rate: Optional[float] = None

    metrics: Dict[str, Any] = {
        "k_rate": k_rate,
        "csw_rate": csw_rate,
    }

    for metric_name, metric_value in metrics.items():
        if metric_value is not None:
            db.upsert_player_stat(
                player_id=player_id,
                stat_date=today_str,
                stat_type=f"pitcher_rolling_{metric_name}",
                value=metric_value,
            )

    log.debug(
        "Computed pitcher rolling stats for player_id=%s: %s", player_id, metrics
    )
    return metrics


# ---------------------------------------------------------------------------
# Full nightly orchestrator
# ---------------------------------------------------------------------------

def run_full_nightly(
    batter_ids: Sequence[int],
    pitcher_ids: Sequence[int],
    lookback_days: int = 180,
    season: Optional[int] = None,
) -> Dict[str, Any]:
    """Orchestrate the complete nightly Statcast pipeline.

    Runs in order:
    1. run_batter_nightly — raw batter PA fetch & upsert
    2. run_pitcher_nightly — raw pitcher pitch fetch & upsert
    3. compute_batter_rolling_stats — per-batter rolling metrics
    4. compute_pitcher_rolling_stats — per-pitcher rolling metrics
    5. db.log_model_run — records this pipeline execution

    Parameters
    ----------
    batter_ids:
        MLB MLBAM IDs for batters to process.
    pitcher_ids:
        MLB MLBAM IDs for pitchers to process.
    lookback_days:
        Lookback window (days) for the raw fetch.

    Returns
    -------
    dict
        {
          "batters_fetched": int,
          "pitchers_fetched": int,
          "errors": list[str],
        }
    """
    database = get_db()
    errors: List[str] = []
    batters_fetched = 0
    pitchers_fetched = 0
    if season is None:
        season = date.today().year

    # --- Step 1: batter raw fetch ---
    try:
        batters_fetched = run_batter_nightly(
            batter_ids, lookback_days=lookback_days, db=database
        )
        log.info("Batter nightly fetch complete: %d rows", batters_fetched)
    except Exception as exc:  # noqa: BLE001
        msg = f"run_batter_nightly failed: {exc}"
        log.error(msg)
        errors.append(msg)

    # --- Step 2: pitcher raw fetch ---
    try:
        pitchers_fetched = run_pitcher_nightly(
            pitcher_ids, lookback_days=lookback_days, db=database
        )
        log.info("Pitcher nightly fetch complete: %d rows", pitchers_fetched)
    except Exception as exc:  # noqa: BLE001
        msg = f"run_pitcher_nightly failed: {exc}"
        log.error(msg)
        errors.append(msg)

    # --- Step 3: batter rolling stats ---
    for player_id in batter_ids:
        try:
            compute_batter_rolling_stats(player_id, database)
        except Exception as exc:  # noqa: BLE001
            msg = f"compute_batter_rolling_stats failed for {player_id}: {exc}"
            log.error(msg)
            errors.append(msg)

    # --- Step 4: pitcher rolling stats ---
    for player_id in pitcher_ids:
        try:
            compute_pitcher_rolling_stats(player_id, database)
        except Exception as exc:  # noqa: BLE001
            msg = f"compute_pitcher_rolling_stats failed for {player_id}: {exc}"
            log.error(msg)
            errors.append(msg)

    # --- Step 5: sprint speed (season-level, one batch call) ---
    try:
        sprint_count = run_sprint_speed_nightly(batter_ids, season, database)
        log.info("Sprint speed update complete: %d players", sprint_count)
    except Exception as exc:  # noqa: BLE001
        msg = f"run_sprint_speed_nightly failed: {exc}"
        log.error(msg)
        errors.append(msg)

    # --- Step 6: log model run ---
    today_str = date.today().strftime("%Y-%m-%d")
    try:
        database.log_model_run(
            run_date=today_str,
            model_version=MODEL_VERSION,
            prop_type="statcast_nightly",
            n_picks=batters_fetched + pitchers_fetched,
            mean_edge=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"log_model_run failed: {exc}"
        log.error(msg)
        errors.append(msg)

    summary: Dict[str, Any] = {
        "batters_fetched": batters_fetched,
        "pitchers_fetched": pitchers_fetched,
        "errors": errors,
    }
    log.info("Nightly pipeline complete: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Row helper
# ---------------------------------------------------------------------------

def _row_game_date(row: Any) -> Optional[str]:
    """Extract a YYYY-MM-DD string from a DataFrame row's game_date field."""
    val = row.get("game_date")
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, str):
        return val[:10]
    if hasattr(val, "strftime"):
        return val.strftime("%Y-%m-%d")
    return str(val)[:10]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="Statcast nightly pipeline")
    parser.add_argument("--lookback-days", type=int, default=35)
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--batter-ids", type=str, default="",
                        help="Comma-separated MLBAM batter IDs (optional override)")
    parser.add_argument("--pitcher-ids", type=str, default="",
                        help="Comma-separated MLBAM pitcher IDs (optional override)")
    args = parser.parse_args()

    _db = get_db()
    season_yr = args.season or date.today().year

    # Build player list from unresolved bets + today's scheduled players
    if args.batter_ids or args.pitcher_ids:
        batter_ids_main = [int(x) for x in args.batter_ids.split(",") if x.strip()]
        pitcher_ids_main = [int(x) for x in args.pitcher_ids.split(",") if x.strip()]
    else:
        batter_ids_main: list[int] = []
        pitcher_ids_main: list[int] = []

        # Seed from open bets first
        try:
            for bet in _db.get_unresolved_bets():
                pid = bet.get("player_id")
                if not pid:
                    continue
                pid = int(pid)
                prop = bet.get("prop_type", "")
                if "strikeout" in prop.lower() or "outs" in prop.lower():
                    pitcher_ids_main.append(pid)
                else:
                    batter_ids_main.append(pid)
        except Exception as _exc:
            log.exception("Failed to build player list from bets: %s", _exc)

        # Augment with today's probable pitchers and confirmed/probable batters
        # This ensures new starters always have current Statcast data before daily_runner.
        try:
            from mlb_api import get_client as _get_mlb_client
            _mlb = _get_mlb_client()
            today_str = date.today().isoformat()
            _games = _mlb.get_schedule(today_str)
            for _game in _games:
                _game_pk = _game.get("game_pk")
                if not _game_pk:
                    continue
                # Probable pitchers (available pre-game)
                _probables = _mlb.get_probable_pitchers(today_str)
                _pp = _probables.get(_game_pk, {})
                for _key in ("home_pitcher_id", "away_pitcher_id"):
                    _pid = _pp.get(_key)
                    if _pid:
                        pitcher_ids_main.append(int(_pid))
                # Confirmed lineups (may be empty if posted late)
                try:
                    _lineup = _mlb.get_confirmed_lineup(_game_pk)
                    for _side in ("home", "away"):
                        for _player in _lineup.get(_side, []):
                            _bid = _player.get("player_id")
                            if _bid:
                                batter_ids_main.append(int(_bid))
                except Exception:
                    pass  # lineups not yet posted — skip
            log.info(
                "Schedule seeding: added %d batters / %d pitchers from today's games",
                len(batter_ids_main), len(pitcher_ids_main),
            )
        except Exception as _exc:
            log.warning("Failed to augment player list from schedule: %s", _exc)

        batter_ids_main = list(dict.fromkeys(batter_ids_main))
        pitcher_ids_main = list(dict.fromkeys(pitcher_ids_main))

    log.info(
        "=== Statcast nightly: %d batters, %d pitchers (lookback=%dd) ===",
        len(batter_ids_main), len(pitcher_ids_main), args.lookback_days,
    )

    result = run_full_nightly(
        batter_ids=batter_ids_main,
        pitcher_ids=pitcher_ids_main,
        lookback_days=args.lookback_days,
        season=season_yr,
    )
    log.info("Done: %s", result)
    sys.exit(0 if not result.get("errors") else 1)
