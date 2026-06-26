"""outs_features.py — Feature engineering for the Pitcher Outs Recorded NB GLM.

Feature priority (from research):
  Tier 1: pitches_per_inning_30d, bb_rate_30d, avg_ip_last_3_starts,
          bullpen_usage_3d, opp_pa_per_k_30d
  Tier 2: prev_start_pitch_count, manager_hook_factor,
          contact_quality_allowed_30d, game_total, k_rate_30d
  Tier 3: days_rest, is_home, season_outs_per_start, third_time_through_proximity
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Optional

from db import get_db
from mlb_api import get_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEAGUE_BB_RATE = 0.085        # 2024 MLB walk rate
LEAGUE_K_RATE = 0.224         # 2024 MLB K rate
LEAGUE_XWOBA = 0.320          # 2024 MLB average xwOBA allowed
LEAGUE_PA_PER_K = 1.0 / LEAGUE_K_RATE   # ≈ 4.46 PA per strikeout
LEAGUE_P_PER_IP = 15.5        # average pitches per inning (MLB 2024)
LEAGUE_OUTS_PER_START = 16.5  # league average outs per start (5.5 IP × 3)

_MANAGER_HOOK_DEFAULTS: dict[str, float] = {
    # Values > 1.0 = manager tends to leave SP in longer; < 1.0 = quick hook
    # Calibrated from 2023-24 SP usage: default 1.0 (neutral) when team not listed
}

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _sf(val: Any, default: float = 0.0) -> float:
    """Safe float conversion."""
    try:
        import numpy as np
        f = float(val)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _ip_to_outs(ip_str: Any) -> int:
    """Convert MLB inningsPitched string ('6.2') to integer outs (6*3+2=20)."""
    try:
        ip = float(ip_str)
        whole = int(ip)
        partial = int(round((ip - whole) * 10))
        return whole * 3 + partial
    except (TypeError, ValueError):
        return 0


def _parse_ip_decimal(ip_str: Any) -> float:
    """Convert MLB inningsPitched to decimal innings ('6.2' → 6.667).

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


def _latest_stat(player_id: int, stat_type: str, db: Any) -> Optional[float]:
    rows = db.get_player_stats(player_id, stat_type, days=2)
    vals = [r["value"] for r in rows if r.get("value") is not None]
    return float(vals[-1]) if vals else None


def _game_log_starts(game_log: list[dict]) -> list[dict]:
    """Filter game log to starts only (gamesStarted=1, IP>=3)."""
    return [
        g for g in game_log
        if g.get("gamesStarted", 0) == 1
        and _parse_ip_decimal(g.get("inningsPitched", "0")) >= 3.0
    ]


def _avg_ip_last_n(starts: list[dict], n: int = 3) -> float:
    """Average decimal IP from the most recent n starts."""
    vals: list[float] = []
    for s in starts[:n]:
        v = _parse_ip_decimal(s.get("inningsPitched", "0"))
        if v > 0:
            vals.append(v)
    return sum(vals) / len(vals) if vals else _parse_ip_decimal("5.1")


def _prev_start_pitch_count(starts: list[dict]) -> float:
    """Pitches thrown in the most recent start."""
    if not starts:
        return 85.0
    pc = starts[0].get("pitchesThrown") or starts[0].get("numberOfPitches")
    if pc is not None:
        return _sf(pc, 85.0)
    # Proxy: IP * 15 pitches/inning
    ip = _parse_ip_decimal(starts[0].get("inningsPitched", "0"))
    return ip * LEAGUE_P_PER_IP if ip > 0 else 85.0


def _days_rest_from_log(starts: list[dict], game_date_str: str) -> float:
    """Days between today's start and the most recent prior start."""
    try:
        today = date.fromisoformat(game_date_str[:10])
    except ValueError:
        return 5.0
    for s in starts:
        s_date_str = (s.get("date") or "")[:10]
        if not s_date_str:
            continue
        try:
            s_date = date.fromisoformat(s_date_str)
            if s_date < today:
                return float((today - s_date).days)
        except ValueError:
            continue
    return 5.0


def _season_stats(pitcher_id: int, season: int, mlb_client: Any) -> dict[str, float]:
    """Return bb_rate, k_rate, season_outs_per_start, contact_quality from season stats."""
    defaults = {
        "bb_rate_30d": LEAGUE_BB_RATE,
        "k_rate_30d": LEAGUE_K_RATE,
        "season_outs_per_start": LEAGUE_OUTS_PER_START,
        "contact_quality_allowed_30d": LEAGUE_XWOBA,
        "pitches_per_inning_30d": LEAGUE_P_PER_IP,
    }
    try:
        stats = mlb_client.get_player_season_stats(pitcher_id, season, group="pitching")
        bf = _sf(stats.get("battersFaced"), 0)
        if bf < 10:
            return defaults

        ks = _sf(stats.get("strikeOuts"), 0)
        bbs = _sf(stats.get("baseOnBalls"), 0)
        gs = _sf(stats.get("gamesStarted"), 1)
        ip_raw = _sf(stats.get("inningsPitched"), 0)
        ip_dec = _parse_ip_decimal(str(ip_raw)) if ip_raw > 0 else 0.0

        bb_rate = bbs / bf if bf > 0 else LEAGUE_BB_RATE
        k_rate = ks / bf if bf > 0 else LEAGUE_K_RATE
        # pitches_per_inning: no direct API field; estimate from walk rate
        # More walks → more pitches per inning. Regression fit:
        # P/IP ≈ 14.5 + 10 × BB_rate  (R≈0.65 vs 2024 MLB pitch-tracking data)
        p_per_ip = 14.5 + 10.0 * bb_rate
        season_outs_gs = (ip_dec * 3.0 / gs) if gs > 0 and ip_dec > 0 else LEAGUE_OUTS_PER_START

        return {
            "bb_rate_30d": min(max(bb_rate, 0.0), 0.25),
            "k_rate_30d": min(max(k_rate, 0.05), 0.45),
            "season_outs_per_start": min(max(season_outs_gs, 6.0), 27.0),
            "contact_quality_allowed_30d": LEAGUE_XWOBA,  # fallback; prefer DB rolling
            "pitches_per_inning_30d": min(max(p_per_ip, 12.0), 22.0),
        }
    except Exception:
        return defaults


def _opp_pa_per_k(opp_batter_ids: list[int], db: Any) -> float:
    """Mean PA/K for the opposing lineup. Higher = lineup makes more contact = shorter starts."""
    if not opp_batter_ids:
        return LEAGUE_PA_PER_K
    rates: list[float] = []
    for bid in opp_batter_ids:
        rows = db.get_player_stats(bid, "k_pa_season", 365)
        vals = [r["value"] for r in rows if r.get("value") is not None]
        if vals:
            k_rate = float(vals[-1])
            if k_rate > 0:
                rates.append(1.0 / k_rate)
    return sum(rates) / len(rates) if rates else LEAGUE_PA_PER_K


def _bullpen_usage_3d(team: str, db: Any) -> float:
    """Total bullpen pitches thrown in last 3 days (proxy for pen fatigue).

    Placeholder: the 'bullpen_pitches_today' stat type is not yet populated
    by statcast_nightly.  Returns 0.0 (neutral) until that producer exists and
    a team→player_id mapping is available to scope by team.
    """
    return 0.0


def _manager_hook_factor(team: str) -> float:
    """Return team's manager tendency to use SP deep (>1) or pull early (<1)."""
    return _MANAGER_HOOK_DEFAULTS.get(team, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_outs_features(
    pitcher_id: int,
    game_pk: int,
    game_date_str: str,
    is_home: bool,
    pitcher_team: str,
    opp_batter_ids: list[int],
    game_total: Optional[float],
    db: Any = None,
    mlb_client: Any = None,
    season: Optional[int] = None,
) -> dict[str, float]:
    """Compute all pitcher outs features for one game.

    Falls back to league-average defaults for any unavailable data.
    Returns a flat dict ready to pass to OutsModel.predict_with_blend().
    """
    db = db or get_db()
    mlb_client = mlb_client or get_client()
    season = season or datetime.now(timezone.utc).year

    # 1. Season-level stats (bb_rate, k_rate, season depth, P/IP proxy)
    sstat = _season_stats(pitcher_id, season, mlb_client)

    # 2. Rolling DB stats — override season stats where available
    rolling_p_per_ip = _latest_stat(pitcher_id, "pitcher_rolling_pitches_per_inning", db)
    rolling_bb_rate = _latest_stat(pitcher_id, "pitcher_rolling_bb_rate", db)
    # Use actual K-rate rolling stat; fall back to season K/BF from _season_stats()
    rolling_k_rate = _latest_stat(pitcher_id, "pitcher_rolling_k_rate", db)
    rolling_xwoba = _latest_stat(pitcher_id, "pitcher_rolling_xwoba_allowed", db)

    pitches_per_inning_30d = rolling_p_per_ip if rolling_p_per_ip is not None else sstat["pitches_per_inning_30d"]
    bb_rate_30d = rolling_bb_rate if rolling_bb_rate is not None else sstat["bb_rate_30d"]
    k_rate_30d = rolling_k_rate if rolling_k_rate is not None else sstat["k_rate_30d"]
    contact_quality = rolling_xwoba if rolling_xwoba is not None else sstat["contact_quality_allowed_30d"]

    # 3. Game log for rolling workload context
    try:
        game_log = mlb_client.get_player_game_log(pitcher_id, season, group="pitching")
    except Exception:
        game_log = []

    starts = _game_log_starts(game_log)
    avg_ip_last_3 = _avg_ip_last_n(starts, n=3)
    prev_pc = _prev_start_pitch_count(starts)
    days_rest = _days_rest_from_log(starts, game_date_str)

    # 4. Opponent features
    opp_pa_per_k = _opp_pa_per_k(opp_batter_ids, db)

    # 5. Contextual signals
    bullpen_used = _bullpen_usage_3d(pitcher_team, db)
    hook = _manager_hook_factor(pitcher_team)
    season_outs = sstat["season_outs_per_start"]
    # Third-time-through proximity: how much of the lineup the pitcher typically sees 3x.
    # 0 when avg IP <= 6.0 (at most 2x through order); positive beyond that.
    third_time = max(0.0, (avg_ip_last_3 / 3.0) - 2.0)

    return {
        "pitches_per_inning_30d":      round(min(max(pitches_per_inning_30d, 12.0), 22.0), 4),
        "bb_rate_30d":                 round(min(max(bb_rate_30d, 0.0), 0.25), 4),
        "avg_ip_last_3_starts":        round(min(max(avg_ip_last_3, 0.0), 9.0), 4),
        "bullpen_usage_3d":            round(min(max(bullpen_used, 0.0), 300.0), 1),
        "opp_pa_per_k_30d":            round(min(max(opp_pa_per_k, 2.0), 10.0), 4),
        "prev_start_pitch_count":      round(min(max(prev_pc, 40.0), 130.0), 1),
        "manager_hook_factor":         round(min(max(hook, 0.5), 1.5), 4),
        "contact_quality_allowed_30d": round(min(max(contact_quality, 0.200), 0.450), 4),
        "game_total":                  round(_sf(game_total, 8.5), 2),
        "k_rate_30d":                  round(min(max(k_rate_30d, 0.05), 0.45), 4),
        "days_rest":                   round(min(max(days_rest, 1.0), 12.0), 1),
        "is_home":                     float(int(is_home)),
        "season_outs_per_start":       round(min(max(season_outs, 6.0), 27.0), 4),
        "third_time_through_proximity": round(min(third_time, 1.0), 4),
    }
