#!/usr/bin/env python3
"""
Bet Analysis Agent

Audits individual bets for alignment between model probability and market consensus.
Flags bets as CONFIRMED, QUESTIONABLE, or VOID before or after they are placed.

Triggers automatically when edge > HIGH_EDGE_THRESHOLD (default 15%).

Usage:
    python3 bet_analysis.py --date 2026-04-17          # audit all bets for a date
    python3 bet_analysis.py --bet-id 860               # audit a single bet by DB id
    python3 bet_analysis.py --date today --threshold 0.10  # lower threshold
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import poisson, binom

from db import get_db
from mlb_api import get_client
from odds import get_odds_client, implied_prob, ALL_MARKETS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIGH_EDGE_THRESHOLD = 0.15   # auto-flag when |edge| exceeds this
SANITY_EDGE_THRESHOLD = 0.25  # VOID if model vs sanity check diverges this much

# Minimum number of books that must show a line for it to be considered reliable
MIN_BOOKS_FOR_CONSENSUS = 2

# How many recent games to use for the sanity check
RECENT_GAMES_WINDOW = 15

# Expected PA by lineup spot (for hits sanity check)
_PA_BY_SPOT: dict[int, float] = {
    1: 4.65, 2: 4.55, 3: 4.35, 4: 4.20, 5: 4.05,
    6: 3.90, 7: 3.70, 8: 3.55, 9: 3.35,
}

PROP_MARKET_MAP = {
    "strikeouts": "pitcher_strikeouts",
    "hits":        "batter_hits",
    "home_runs":   "batter_home_runs",
    "total_bases": "batter_total_bases",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _no_vig(over_implied: float, under_implied: float) -> float:
    """De-vig the over implied probability."""
    total = over_implied + under_implied
    return over_implied / total if total > 0 else 0.5


def _poisson_over(lam: float, line: float) -> float:
    return float(1.0 - poisson.cdf(math.floor(line), lam))


def _binom_over(line: float, n_pa: float, p: float) -> float:
    lo = int(n_pa)
    hi = lo + 1
    frac = n_pa - lo
    p_lo = float(1.0 - binom.cdf(int(line), n=lo, p=p))
    p_hi = float(1.0 - binom.cdf(int(line), n=hi, p=p))
    return p_lo + frac * (p_hi - p_lo)


# ---------------------------------------------------------------------------
# Multi-book consensus
# ---------------------------------------------------------------------------

def get_consensus(
    odds_client: Any,
    game_pk: int,
    player_name: str,
    prop_type: str,
    line: float,
    pick_side: str,
    date_str: str,
) -> dict:
    """
    Fetch the prop line from all available books and compute consensus no-vig.

    Returns
    -------
    dict with:
        books_seen        – list of (book, over_odds, under_odds) tuples
        book_count        – number of books with this line
        consensus_no_vig  – average no-vig probability for the pick side
        min_no_vig        – lowest book no-vig (most pessimistic for pick)
        max_no_vig        – highest book no-vig (most optimistic for pick)
        is_stale_line     – True if fewer than MIN_BOOKS_FOR_CONSENSUS have the line
        notes             – list of strings describing anomalies
    """
    market_key = PROP_MARKET_MAP.get(prop_type)
    if not market_key:
        return {"book_count": 0, "notes": [f"Unknown prop_type: {prop_type}"], "is_stale_line": True}

    notes: list[str] = []
    books_seen: list[dict] = []

    try:
        events = odds_client.get_mlb_events(date_str=date_str)
        target_event = None
        for ev in events:
            if ev.get("game_pk") == game_pk:
                target_event = ev
                break

        if not target_event:
            notes.append(f"Game pk={game_pk} not found in Odds API events for {date_str}.")
            return {"book_count": 0, "notes": notes, "is_stale_line": True}

        event_data = odds_client.get_event_props(
            target_event["id"],
            markets=[market_key],
        )
    except Exception as exc:
        notes.append(f"Odds API fetch failed: {exc}")
        return {"book_count": 0, "notes": notes, "is_stale_line": True}

    player_lower = player_name.lower()
    no_vig_vals: list[float] = []

    for book_key, book_markets in event_data.get("props", {}).items():
        outcomes = book_markets.get(market_key, [])
        # Find matching player + line
        over_row = next(
            (o for o in outcomes
             if o.get("player_name", "").lower() == player_lower
             and o.get("pick", "").lower() == "over"
             and o.get("line") == line),
            None,
        )
        under_row = next(
            (o for o in outcomes
             if o.get("player_name", "").lower() == player_lower
             and o.get("pick", "").lower() == "under"
             and o.get("line") == line),
            None,
        )

        if over_row and under_row:
            nv = _no_vig(over_row["implied_prob"], under_row["implied_prob"])
            pick_nv = nv if pick_side.lower() == "over" else (1.0 - nv)
            no_vig_vals.append(pick_nv)
            books_seen.append({
                "book":       book_key,
                "over_odds":  over_row["odds"],
                "under_odds": under_row["odds"],
                "no_vig":     round(pick_nv, 4),
            })
        elif over_row or under_row:
            notes.append(f"{book_key}: only one side available (line={line})")

    book_count = len(books_seen)
    consensus_no_vig = sum(no_vig_vals) / book_count if book_count else None
    min_no_vig = min(no_vig_vals) if no_vig_vals else None
    max_no_vig = max(no_vig_vals) if no_vig_vals else None

    if book_count < MIN_BOOKS_FOR_CONSENSUS:
        notes.append(
            f"Only {book_count} book(s) offering this line — may be stale or low-liquidity."
        )

    return {
        "books_seen":       books_seen,
        "book_count":       book_count,
        "consensus_no_vig": consensus_no_vig,
        "min_no_vig":       min_no_vig,
        "max_no_vig":       max_no_vig,
        "is_stale_line":    book_count < MIN_BOOKS_FOR_CONSENSUS,
        "notes":            notes,
    }


# ---------------------------------------------------------------------------
# Sanity checks by prop type
# ---------------------------------------------------------------------------

def _sanity_check_strikeouts(
    player_id: int,
    line: float,
    pick_side: str,
    season: int,
    mlb_client: Any,
) -> dict:
    """
    Compute a naïve Poisson-based probability from the pitcher's recent K/IP
    and compare to the model. Returns sanity_prob and supporting stats.
    """
    try:
        game_log = mlb_client.get_player_game_log(player_id, season, group="pitching")
        starts = [g for g in game_log if int(g.get("gamesStarted") or 0) == 1][:RECENT_GAMES_WINDOW]
    except Exception as exc:
        return {"error": str(exc)}

    if not starts:
        return {"error": "No recent starts found."}

    total_k  = sum(int(g.get("strikeOuts") or 0) for g in starts)
    total_ip = sum(float(g.get("inningsPitched") or 0) for g in starts)
    per_start_k = [int(g.get("strikeOuts") or 0) for g in starts]
    avg_k = total_k / len(starts)
    k_per_9 = (total_k / total_ip * 9) if total_ip > 0 else None

    sanity_prob_over = _poisson_over(avg_k, line)
    sanity_prob = sanity_prob_over if pick_side.lower() == "over" else (1.0 - sanity_prob_over)

    recent_display = per_start_k[:8]

    return {
        "avg_k_per_start": round(avg_k, 2),
        "k_per_9":         round(k_per_9, 2) if k_per_9 else None,
        "total_ip":        total_ip,
        "n_starts":        len(starts),
        "recent_k":        recent_display,
        "sanity_prob":     round(sanity_prob, 4),
        "sanity_prob_over": round(sanity_prob_over, 4),
    }


def _sanity_check_hits(
    player_id: int,
    line: float,
    pick_side: str,
    season: int,
    mlb_client: Any,
) -> dict:
    """
    Compute empirical hit rate from recent game log, then apply Binomial
    for the given line.
    """
    try:
        game_log = mlb_client.get_player_game_log(player_id, season, group="hitting")
        games = [g for g in game_log if int(g.get("atBats") or 0) >= 1][:RECENT_GAMES_WINDOW]
    except Exception as exc:
        return {"error": str(exc)}

    if not games:
        return {"error": "No recent games with AB found."}

    total_h  = sum(int(g.get("hits") or 0) for g in games)
    total_ab = sum(int(g.get("atBats") or 0) for g in games)
    hit_rate = total_h / total_ab if total_ab > 0 else 0.255
    per_game_h = [int(g.get("hits") or 0) for g in games]

    # Empirical probability from game outcomes directly
    n_over = sum(1 for h in per_game_h if h > line)
    empirical_over = n_over / len(per_game_h)

    # Binomial-based probability using hit rate and avg PA=4.2
    binom_prob_over = _binom_over(line, 4.2, hit_rate)
    sanity_prob = binom_prob_over if pick_side.lower() == "over" else (1.0 - binom_prob_over)

    return {
        "hit_rate":          round(hit_rate, 3),
        "total_hits":        total_h,
        "total_ab":          total_ab,
        "n_games":           len(games),
        "recent_hits":       per_game_h[:8],
        "empirical_over":    round(empirical_over, 4),
        "sanity_prob":       round(sanity_prob, 4),
        "sanity_prob_over":  round(binom_prob_over, 4),
    }


def _sanity_check_hr(
    player_id: int,
    line: float,
    pick_side: str,
    season: int,
    mlb_client: Any,
) -> dict:
    """
    Compute empirical HR rate and Poisson-based probability.
    """
    try:
        game_log = mlb_client.get_player_game_log(player_id, season, group="hitting")
        games = [g for g in game_log if int(g.get("atBats") or 0) >= 1][:RECENT_GAMES_WINDOW]
    except Exception as exc:
        return {"error": str(exc)}

    if not games:
        return {"error": "No recent games found."}

    total_hr = sum(int(g.get("homeRuns") or 0) for g in games)
    total_ab = sum(int(g.get("atBats") or 0) for g in games)
    hr_per_ab = total_hr / total_ab if total_ab > 0 else 0.030
    hr_per_game = total_hr / len(games)

    sanity_prob_over = _poisson_over(hr_per_game, line)
    sanity_prob = sanity_prob_over if pick_side.lower() == "over" else (1.0 - sanity_prob_over)

    return {
        "hr_per_game":      round(hr_per_game, 4),
        "hr_per_ab":        round(hr_per_ab, 4),
        "n_games":          len(games),
        "recent_hr":        [int(g.get("homeRuns") or 0) for g in games[:8]],
        "sanity_prob":      round(sanity_prob, 4),
        "sanity_prob_over": round(sanity_prob_over, 4),
    }


def run_sanity_check(
    bet: dict,
    mlb_client: Any,
    season: int,
) -> dict:
    """Dispatch to the right sanity checker based on prop_type."""
    prop  = bet["prop_type"]
    pid   = bet["player_id"]
    line  = bet["line"]
    pick  = bet["pick"]

    if prop == "strikeouts":
        return _sanity_check_strikeouts(pid, line, pick, season, mlb_client)
    elif prop == "hits":
        return _sanity_check_hits(pid, line, pick, season, mlb_client)
    elif prop == "home_runs":
        return _sanity_check_hr(pid, line, pick, season, mlb_client)
    else:
        return {"error": f"No sanity check implemented for prop_type={prop}"}


# ---------------------------------------------------------------------------
# Model inspection — per-feature contribution breakdown
# ---------------------------------------------------------------------------

_MODELS_DIR = Path.home() / "mlb-props" / "models"

_MODEL_PATHS = {
    "strikeouts": _MODELS_DIR / "k_model.pkl",
    "hits":       _MODELS_DIR / "hits_model.pkl",
    "home_runs":  _MODELS_DIR / "hr_model.pkl",
}

_FLAG_CONSTANT_FEATURES = {
    # Known-constant features that were not varied during training.
    # These coefficients encode no real signal.
    "strikeouts": {"umpire_k_factor", "weather_k_factor", "park_k_factor", "matchup_factor"},
}


def _load_model(prop_type: str) -> Any | None:
    """Load and return the pickled model object, or None on failure."""
    path = _MODEL_PATHS.get(prop_type)
    if path is None or not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        log.warning("Could not load model for %s: %s", prop_type, exc)
        return None


def _get_coefficients(model: Any) -> tuple[dict[str, float] | None, str]:
    """
    Extract feature coefficients from a loaded model object.

    Returns (coef_dict, model_type) where model_type is "logistic", "poisson_glm",
    or "poisson_sklearn".  coef_dict keys are feature names; intercept is stored
    under "__intercept__".
    """
    feature_cols = getattr(model, "feature_cols", [])

    # KModel: has _use_sklearn attribute — check this first to distinguish from
    # HitsModel/HRModel which also wrap sklearn but via LogisticRegression.
    if hasattr(model, "_use_sklearn"):
        use_sklearn = getattr(model, "_use_sklearn", False)
        if use_sklearn and hasattr(model, "model") and hasattr(model.model, "coef_"):
            coefs = dict(zip(feature_cols, model.model.coef_.tolist()))
            # sklearn PoissonRegressor exposes intercept_ only when fit_intercept=True
            intercept_val = float(model.model.intercept_) if hasattr(model.model, "intercept_") else 0.0
            coefs["__intercept__"] = intercept_val
            return coefs, "poisson_sklearn"
        elif not use_sklearn and hasattr(model, "model") and hasattr(model.model, "params"):
            # statsmodels GLM result
            params = model.model.params.tolist()
            col_names = ["const"] + list(feature_cols)
            coefs = dict(zip(col_names, params))
            if "const" in coefs:
                coefs["__intercept__"] = coefs.pop("const")
            return coefs, "poisson_glm"
        # Fallback to train_meta for KModel
        if hasattr(model, "train_meta") and "coefficients" in model.train_meta:
            raw = dict(model.train_meta["coefficients"])
            if "const" in raw:
                raw["__intercept__"] = raw.pop("const")
            return {k: v for k, v in raw.items() if k in feature_cols or k == "__intercept__"}, "poisson_glm"

    # HitsModel / HRModel: wrap sklearn LogisticRegression
    if hasattr(model, "model") and hasattr(model.model, "coef_"):
        lr = model.model
        coefs = dict(zip(feature_cols, lr.coef_[0].tolist()))
        coefs["__intercept__"] = float(lr.intercept_[0])
        return coefs, "logistic"

    return None, "unknown"


def _feature_contributions(
    features: dict[str, float],
    coefs: dict[str, float],
    feature_cols: list[str],
    model_type: str,
) -> list[dict]:
    """
    Compute per-feature contribution to log-odds (logistic) or log-lambda (Poisson).

    Returns a list of dicts sorted by |contribution| descending:
        feature, value, coefficient, contribution, flagged
    """
    intercept = coefs.get("__intercept__", 0.0)
    rows = []
    for feat in feature_cols:
        val = features.get(feat, 0.0)
        coef = coefs.get(feat, 0.0)
        contrib = coef * val
        rows.append({
            "feature":      feat,
            "value":        round(float(val), 5),
            "coefficient":  round(float(coef), 5),
            "contribution": round(float(contrib), 5),
            "flagged":      False,
        })
    rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    return rows


def run_model_inspection(
    bet: dict,
    mlb_client: Any,
    db: Any,
    season: int,
) -> dict:
    """
    Load the trained model for this prop type, re-compute the prediction using
    either stored feature_snapshot or live-fetched fallback features, and return
    a per-feature contribution breakdown.

    Returns a dict with:
        model_type        – logistic / poisson_glm / poisson_sklearn / unknown
        feature_source    – "stored_snapshot" | "live_fetch" | "unavailable"
        features_used     – dict of feature_name → value
        contributions     – sorted list of {feature, value, coefficient, contribution, flagged}
        log_score         – intercept + sum(contributions)  (log-odds or log-lambda)
        raw_model_prob    – model probability before market blend
        recomputed_prob   – our recomputed probability (should match raw_model_prob)
        stored_prob       – model_prob stored with the bet
        delta_vs_stored   – abs diff between recomputed and stored
        constant_features – features known to be training constants (zero-signal)
        dominant_features – top 3 features by |contribution|
        notes             – list of warning strings
    """
    prop_type = bet["prop_type"]
    notes: list[str] = []

    model = _load_model(prop_type)
    if model is None:
        return {"feature_source": "unavailable", "notes": [f"No trained model found for {prop_type}"]}

    coefs, model_type = _get_coefficients(model)
    if coefs is None:
        return {"feature_source": "unavailable", "notes": ["Could not extract model coefficients."]}

    feature_cols = getattr(model, "feature_cols", [])

    # ── 1. Use stored feature_snapshot if present ────────────────────────────
    stored_snapshot = bet.get("feature_snapshot")
    if stored_snapshot:
        if isinstance(stored_snapshot, str):
            try:
                stored_snapshot = json.loads(stored_snapshot)
            except Exception:
                stored_snapshot = None

    if stored_snapshot and isinstance(stored_snapshot, dict):
        features_used = stored_snapshot
        feature_source = "stored_snapshot"
    else:
        # ── 2. Live-fetch features (best-effort) ─────────────────────────────
        features_used = _fetch_live_features(bet, model, mlb_client, db, season, notes)
        feature_source = "live_fetch" if features_used else "unavailable"

    if not features_used:
        return {"feature_source": "unavailable", "notes": notes or ["Could not fetch features."]}

    # ── 3. Per-feature contribution breakdown ────────────────────────────────
    contributions = _feature_contributions(features_used, coefs, feature_cols, model_type)

    intercept = coefs.get("__intercept__", 0.0)
    log_score = intercept + sum(r["contribution"] for r in contributions)

    # Recompute probability
    if model_type == "logistic":
        recomputed_prob = float(1.0 / (1.0 + math.exp(-log_score)))
    else:
        # Poisson: log_score = log(lambda)
        lam = math.exp(log_score)
        line = float(bet.get("line", 0))
        pick = (bet.get("pick") or "over").lower()
        p_over = float(1.0 - poisson.cdf(math.floor(line), lam))
        recomputed_prob = p_over if pick == "over" else 1.0 - p_over

    stored_prob = float(bet.get("model_prob") or 0.0)
    delta = abs(recomputed_prob - stored_prob)

    # Flag constant training features (known zero-signal)
    constant_feats = _FLAG_CONSTANT_FEATURES.get(prop_type, set())
    for r in contributions:
        if r["feature"] in constant_feats:
            r["flagged"] = True
            notes.append(
                f"Feature '{r['feature']}' (value={r['value']}, contrib={r['contribution']:+.4f}) "
                f"was constant during training — coefficient encodes no real signal."
            )

    if delta > 0.05:
        notes.append(
            f"Recomputed prob ({recomputed_prob:.1%}) differs from stored ({stored_prob:.1%}) "
            f"by {delta:.1%} — market blend or rounding may explain small gaps."
        )

    dominant = [r["feature"] for r in contributions[:3]]

    return {
        "model_type":       model_type,
        "feature_source":   feature_source,
        "features_used":    features_used,
        "contributions":    contributions,
        "log_score":        round(log_score, 5),
        "recomputed_prob":  round(recomputed_prob, 4),
        "stored_prob":      round(stored_prob, 4),
        "delta_vs_stored":  round(delta, 4),
        "constant_features": list(constant_feats),
        "dominant_features": dominant,
        "notes":            notes,
    }


def _fetch_live_features(
    bet: dict,
    model: Any,
    mlb_client: Any,
    db: Any,
    season: int,
    notes: list[str],
) -> dict | None:
    """
    Best-effort live feature fetch for a single bet.  Returns a partial feature
    dict (missing features will be zero-filled before scoring, with a warning).
    """
    prop_type = bet["prop_type"]
    player_id = bet.get("player_id")
    feature_cols = getattr(model, "feature_cols", [])
    features: dict[str, float] = {}

    if not player_id:
        notes.append("No player_id stored with bet — cannot fetch live features.")
        return None

    try:
        if prop_type == "strikeouts":
            _fill_k_features(features, player_id, season, mlb_client, db, notes)
        elif prop_type == "hits":
            _fill_hits_features(features, player_id, season, mlb_client, db, notes)
        elif prop_type == "home_runs":
            _fill_hr_features(features, player_id, season, mlb_client, db, notes)
        else:
            notes.append(f"No live feature fetch implemented for {prop_type}.")
            return None
    except Exception as exc:
        notes.append(f"Live feature fetch error: {exc}")

    # Zero-fill any missing columns and warn
    for col in feature_cols:
        if col not in features:
            features[col] = 0.0
            notes.append(f"Feature '{col}' not fetched — defaulted to 0.0 (may skew score).")

    return features


def _fill_k_features(
    out: dict, player_id: int, season: int, mlb_client: Any, db: Any, notes: list[str]
) -> None:
    """Fill K-model features from live sources."""
    # Rolling pitch metrics from DB
    swstr_rows = db.get_player_stats(player_id, "pitcher_rolling_swstr_rate", days=2)
    swstr = float(swstr_rows[-1]["value"]) if swstr_rows else 0.0
    ff_rows = db.get_player_stats(player_id, "pitcher_rolling_ff_whiff", days=2)
    ff_whiff = float(ff_rows[-1]["value"]) if ff_rows else 0.0
    sl_rows = db.get_player_stats(player_id, "pitcher_rolling_sl_whiff", days=2)
    sl_whiff = float(sl_rows[-1]["value"]) if sl_rows else 0.0
    ch_rows = db.get_player_stats(player_id, "pitcher_rolling_ch_whiff", days=2)
    ch_whiff = float(ch_rows[-1]["value"]) if ch_rows else 0.0

    k_rate = 0.224
    avg_ip = 5.5
    try:
        pstats = mlb_client.get_player_season_stats(player_id, season, group="pitching")
        bf = float(pstats.get("battersFaced") or 0)
        if bf > 0:
            k_rate = float(pstats.get("strikeOuts") or 0) / bf
        gs = float(pstats.get("gamesStarted") or 0)
        ip = float(pstats.get("inningsPitched") or 0)
        if gs > 0:
            avg_ip = ip / gs
    except Exception as exc:
        notes.append(f"K pitcher season stats fetch failed: {exc}")

    out.update({
        "csw_rate_30d":            swstr,
        "k_rate_30d":              k_rate,
        "k_rate_season":           k_rate,
        "whiff_rate_30d":          swstr,
        "swstr_rate_30d":          swstr,
        "ff_whiff_rate_30d":       ff_whiff,
        "sl_whiff_rate_30d":       sl_whiff,
        "ch_whiff_rate_30d":       ch_whiff,
        "opp_k_rate_season":       0.224,
        "opp_k_rate_30d":          0.224,
        "lineup_handedness_split": 0.5,
        "opp_lineup_xwoba":        0.320,
        "umpire_k_factor":         1.0,
        "weather_k_factor":        1.0,
        "park_k_factor":           1.0,
        "is_home":                 0.0,
        "days_rest":               5.0,
        "avg_ip_30d":              avg_ip,
        "is_opener_risk":          0.0,
        "matchup_factor":          1.0,
        "market_implied_over":     0.5,
        "line_movement":           0.0,
    })


def _fill_hits_features(
    out: dict, player_id: int, season: int, mlb_client: Any, db: Any, notes: list[str]
) -> None:
    """Fill Hits-model features from live sources."""
    batter_k_rate = 0.224
    batter_walk_rate = 0.084
    hit_rate_season = 0.248
    batter_hr_rate = 0.030
    try:
        bstats = mlb_client.get_player_season_stats(player_id, season, group="hitting")
        bpa = float(bstats.get("plateAppearances") or 0)
        if bpa > 0:
            batter_k_rate = float(bstats.get("strikeOuts") or 0) / bpa
            batter_walk_rate = float(bstats.get("baseOnBalls") or 0) / bpa
            batter_hr_rate = float(bstats.get("homeRuns") or 0) / bpa
        raw_avg = bstats.get("avg")
        if raw_avg:
            hit_rate_season = float(raw_avg)
    except Exception as exc:
        notes.append(f"Hits batter season stats fetch failed: {exc}")

    hh_rows = db.get_player_stats(player_id, "batter_rolling_hard_hit_rate", days=2)
    hard_hit_30d = float(hh_rows[-1]["value"]) if hh_rows else 0.38
    ev_rows = db.get_player_stats(player_id, "batter_rolling_avg_exit_velocity", days=2)
    avg_ev = float(ev_rows[-1]["value"]) if ev_rows else 88.0
    la_rows = db.get_player_stats(player_id, "batter_rolling_avg_launch_angle", days=2)
    avg_la = float(la_rows[-1]["value"]) if la_rows else 10.0
    speed_rows = db.get_player_stats(player_id, "batter_sprint_speed", days=180)
    sprint_speed = float(speed_rows[-1]["value"]) if speed_rows else 27.0
    chase_rows = db.get_player_stats(player_id, "batter_rolling_chase_rate", days=2)
    chase_rate = float(chase_rows[-1]["value"]) if chase_rows else 0.30

    contact_rate = min(0.95, max(0.50, 1.0 - batter_k_rate))
    _bip = 1.0 - batter_k_rate - batter_hr_rate - batter_walk_rate
    _hit_bip = hit_rate_season - batter_hr_rate
    babip = min(0.500, max(0.150, _hit_bip / max(_bip, 0.30))) if _bip > 0 else 0.295

    out.update({
        "contact_rate_30d":                contact_rate,
        "babip_30d":                       babip,
        "avg_exit_velo_30d":               avg_ev,
        "hard_hit_rate_30d":               hard_hit_30d,
        "hit_rate_season":                 hit_rate_season,
        "avg_launch_angle_30d":            avg_la,
        "line_drive_rate_30d":             0.20,
        "batter_k_rate_season":            batter_k_rate,
        "batter_walk_rate_season":         batter_walk_rate,
        "chase_rate_30d":                  chase_rate,
        "sprint_speed":                    sprint_speed,
        "pitcher_babip_allowed_30d":       0.295,
        "pitcher_babip_allowed_season":    0.295,
        "pitcher_hit_rate_allowed_season": 0.248,
        "pitcher_k_rate_season":           0.224,
        "pitcher_bvp_contact_factor":      0.776,
        "park_factor_hits_h":              1.0,
        "lineup_spot":                     4.0,
        "is_platoon_advantage":            0.0,
        "opp_lineup_xwoba":                0.320,
        "market_implied_prob":             0.5,
        "line_movement":                   0.0,
    })


def _fill_hr_features(
    out: dict, player_id: int, season: int, mlb_client: Any, db: Any, notes: list[str]
) -> None:
    """Fill HR-model features from live sources."""
    hr_rate = 0.030
    try:
        bstats = mlb_client.get_player_season_stats(player_id, season, group="hitting")
        bpa = float(bstats.get("plateAppearances") or 0)
        if bpa > 0:
            hr_rate = float(bstats.get("homeRuns") or 0) / bpa
    except Exception as exc:
        notes.append(f"HR batter season stats fetch failed: {exc}")

    barrel_rows = db.get_player_stats(player_id, "batter_rolling_barrel_rate", days=2)
    barrel = float(barrel_rows[-1]["value"]) if barrel_rows else 0.06
    hh_rows = db.get_player_stats(player_id, "batter_rolling_hard_hit_rate", days=2)
    hard_hit = float(hh_rows[-1]["value"]) if hh_rows else 0.38
    la_rows = db.get_player_stats(player_id, "batter_rolling_avg_launch_angle", days=2)
    avg_la = float(la_rows[-1]["value"]) if la_rows else 10.0

    out.update({
        "barrel_rate_30d":         barrel,
        "barrel_rate_60d":         barrel,
        "hard_hit_rate_30d":       hard_hit,
        "xiso_30d":                0.150,
        "avg_launch_angle_30d":    avg_la,
        "hr_rate_season":          hr_rate,
        "pull_pct_30d":            0.40,
        "park_factor_h":           1.0,
        "weather_hr_multiplier":   1.0,
        "pitcher_hr_rate_season":  0.030,
        "pitcher_gb_pct":          0.45,
        "batter_hand_vs_pitcher":  0.0,
        "is_platoon_advantage":    0.0,
        "lineup_spot":             4.0,
        "bvp_factor":              1.0,
        "opp_lineup_xwoba":        0.320,
        "market_implied_prob":     0.5,
        "line_movement":           0.0,
    })


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def _verdict(
    model_prob: float,
    consensus_no_vig: float | None,
    sanity_prob: float | None,
    is_stale_line: bool,
    notes: list[str],
) -> tuple[str, list[str]]:
    """
    Return (verdict, reasons).

    VOID        – model probability diverges >25 pp from both market and sanity check
    QUESTIONABLE – model diverges from market OR sanity check by 15-25 pp, or stale line
    CONFIRMED   – all checks within tolerance
    """
    reasons: list[str] = []

    market_divergence = abs(model_prob - consensus_no_vig) if consensus_no_vig is not None else None
    sanity_divergence = abs(model_prob - sanity_prob) if sanity_prob is not None else None

    if is_stale_line:
        reasons.append("Stale/low-liquidity line: fewer than 2 books offering this prop.")

    if market_divergence is not None:
        if market_divergence > SANITY_EDGE_THRESHOLD:
            reasons.append(
                f"Model ({model_prob:.1%}) diverges {market_divergence:.1%} from market consensus "
                f"({consensus_no_vig:.1%}) — exceeds {SANITY_EDGE_THRESHOLD:.0%} void threshold."
            )
        elif market_divergence > HIGH_EDGE_THRESHOLD:
            reasons.append(
                f"Model ({model_prob:.1%}) diverges {market_divergence:.1%} from market "
                f"({consensus_no_vig:.1%}) — above {HIGH_EDGE_THRESHOLD:.0%} flag threshold."
            )

    if sanity_divergence is not None:
        if sanity_divergence > SANITY_EDGE_THRESHOLD:
            reasons.append(
                f"Model ({model_prob:.1%}) diverges {sanity_divergence:.1%} from recent-stats "
                f"sanity check ({sanity_prob:.1%}) — exceeds void threshold."
            )
        elif sanity_divergence > HIGH_EDGE_THRESHOLD:
            reasons.append(
                f"Model ({model_prob:.1%}) diverges {sanity_divergence:.1%} from recent-stats "
                f"sanity check ({sanity_prob:.1%}) — above flag threshold."
            )

    # Determine verdict
    void_conditions = [
        market_divergence is not None and market_divergence > SANITY_EDGE_THRESHOLD,
        sanity_divergence is not None and sanity_divergence > SANITY_EDGE_THRESHOLD,
    ]
    questionable_conditions = [
        is_stale_line,
        market_divergence is not None and market_divergence > HIGH_EDGE_THRESHOLD,
        sanity_divergence is not None and sanity_divergence > HIGH_EDGE_THRESHOLD,
    ]

    if any(void_conditions):
        return "VOID", reasons
    elif any(questionable_conditions):
        return "QUESTIONABLE", reasons
    else:
        return "CONFIRMED", reasons + ["All checks passed."]


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_bet(
    bet: dict,
    odds_client: Any,
    mlb_client: Any,
    db: Any,
    season: int,
) -> dict:
    """
    Full analysis of a single bet. Returns a structured audit dict.
    """
    player  = bet["player_name"]
    prop    = bet["prop_type"]
    line    = bet["line"]
    pick    = bet["pick"]
    book    = bet["book"]
    odds    = bet["odds"]
    model_prob = bet["model_prob"]
    stored_edge = bet["edge"]
    game_date = (bet.get("game_date") or "")[:10]
    game_pk = bet.get("game_pk")

    log.info("Analyzing: %s | %s %s %s @ %s (model=%.1f%% edge=%.1f%%)",
             player, prop, pick, line, book,
             model_prob * 100, stored_edge * 100)

    # 1. Multi-book consensus
    consensus = get_consensus(
        odds_client, game_pk, player, prop, line, pick, game_date,
    )

    # 2. Recent-stats sanity check
    sanity = run_sanity_check(bet, mlb_client, season)

    sanity_prob = sanity.get("sanity_prob")
    consensus_nv = consensus.get("consensus_no_vig")

    # 3. Model inspection — how the model is projecting this outcome
    inspection = run_model_inspection(bet, mlb_client, db, season)

    # 4. Verdict
    verdict, reasons = _verdict(
        model_prob=model_prob,
        consensus_no_vig=consensus_nv,
        sanity_prob=sanity_prob,
        is_stale_line=consensus.get("is_stale_line", False),
        notes=consensus.get("notes", []),
    )

    return {
        "bet_id":       bet.get("id"),
        "player":       player,
        "prop":         prop,
        "line":         line,
        "pick":         pick,
        "book":         book,
        "book_odds":    odds,
        "model_prob":   model_prob,
        "stored_edge":  stored_edge,
        # Multi-book
        "book_count":   consensus["book_count"],
        "books_seen":   consensus.get("books_seen", []),
        "consensus_no_vig": consensus_nv,
        "is_stale_line": consensus.get("is_stale_line"),
        "consensus_notes": consensus.get("notes", []),
        # Sanity check
        "sanity":       sanity,
        # Model projection breakdown
        "inspection":   inspection,
        # Verdict
        "verdict":      verdict,
        "reasons":      reasons,
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def _bar(val: float, width: int = 30) -> str:
    filled = int(round(val * width))
    return "[" + "█" * filled + "░" * (width - filled) + f"] {val:.1%}"


def print_report(result: dict) -> None:
    verdict = result["verdict"]
    icon = {"CONFIRMED": "✅", "QUESTIONABLE": "⚠️ ", "VOID": "🚫"}.get(verdict, "?")

    print()
    print("─" * 60)
    print(f"{icon} {verdict}  |  {result['player']}  —  {result['prop'].upper()}")
    print(f"   Pick: {result['pick']} {result['line']}   Book: {result['book']} ({result['book_odds']:+d})")
    print()
    print(f"   Model prob      {_bar(result['model_prob'])}")
    if result['consensus_no_vig'] is not None:
        print(f"   Market consensus{_bar(result['consensus_no_vig'])}  ({result['book_count']} books)")
    sanity = result.get("sanity", {})
    if sanity.get("sanity_prob") is not None:
        print(f"   Sanity check    {_bar(sanity['sanity_prob'])}")
    print()

    # Prop-specific context
    if result["prop"] == "strikeouts":
        if "avg_k_per_start" in sanity:
            print(f"   K/start avg ({sanity['n_starts']}g): {sanity['avg_k_per_start']}  "
                  f"K/9: {sanity.get('k_per_9', '?')}  "
                  f"Recent: {sanity.get('recent_k', [])}")
    elif result["prop"] == "hits":
        if "hit_rate" in sanity:
            print(f"   Hit rate ({sanity['n_games']}g): {sanity['hit_rate']:.3f}  "
                  f"Empirical over {result['line']}: {sanity.get('empirical_over', '?'):.1%}  "
                  f"Recent hits: {sanity.get('recent_hits', [])}")
    elif result["prop"] == "home_runs":
        if "hr_per_game" in sanity:
            print(f"   HR/game ({sanity['n_games']}g): {sanity['hr_per_game']}  "
                  f"Recent: {sanity.get('recent_hr', [])}")
    print()

    if result["books_seen"]:
        print("   Books:")
        for b in result["books_seen"]:
            print(f"     {b['book']:<15} over {b['over_odds']:+d}  under {b['under_odds']:+d}  "
                  f"no-vig: {b['no_vig']:.1%}")
    print()

    # ── Model projection breakdown ─────────────────────────────────────────
    insp = result.get("inspection", {})
    feature_source = insp.get("feature_source", "unavailable")
    if feature_source != "unavailable" and insp.get("contributions"):
        model_type = insp.get("model_type", "?")
        log_score = insp.get("log_score", 0.0)
        recomp = insp.get("recomputed_prob", 0.0)
        delta = insp.get("delta_vs_stored", 0.0)

        if model_type == "logistic":
            score_label = f"log-odds = {log_score:+.4f}  →  sigmoid = {recomp:.1%}"
        else:
            lam = math.exp(log_score) if log_score > -30 else 0.0
            score_label = f"log(λ) = {log_score:+.4f}  →  λ = {lam:.2f}  →  P(over) = {recomp:.1%}"

        src_tag = "(stored snapshot)" if feature_source == "stored_snapshot" else "(live fetch)"
        print(f"   Model internals  [{model_type}]  {src_tag}")
        print(f"   {score_label}")
        if delta > 0.02:
            print(f"   (recomputed vs stored delta: {delta:.1%} — blend/rounding)")
        print()
        print(f"   {'Feature':<38} {'Value':>9}  {'Coef':>8}  {'Contrib':>8}")
        print(f"   {'-'*38}  {'-'*9}  {'-'*8}  {'-'*8}")
        intercept = log_score - sum(r["contribution"] for r in insp["contributions"])
        print(f"   {'__intercept__':<38} {'':>9}  {'':>8}  {intercept:>+8.4f}")
        for r in insp["contributions"]:
            flag = " !" if r.get("flagged") else ""
            print(f"   {r['feature']:<38} {r['value']:>9.4f}  {r['coefficient']:>+8.4f}  {r['contribution']:>+8.4f}{flag}")
        print()
        if insp.get("dominant_features"):
            print(f"   Top drivers: {', '.join(insp['dominant_features'])}")
        if insp.get("constant_features"):
            print(f"   Zero-signal features (training constants): {', '.join(insp['constant_features'])}")
        print()
        insp_notes = insp.get("notes", [])
        if insp_notes:
            print("   Model notes:")
            for n in insp_notes:
                print(f"     ℹ  {n}")
        print()
    elif feature_source == "unavailable":
        if insp.get("notes"):
            print(f"   Model inspection unavailable: {insp['notes'][0]}")
            print()

    print("   Reasons:")
    for r in result["reasons"]:
        print(f"     • {r}")

    if result["consensus_notes"]:
        print("   Market notes:")
        for n in result["consensus_notes"]:
            print(f"     ℹ  {n}")

    print("─" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Bet Analysis Agent")
    parser.add_argument("--date", default="today", help="Game date YYYY-MM-DD or 'today'")
    parser.add_argument("--bet-id", type=int, default=None, help="Analyze a single bet by DB id")
    parser.add_argument("--threshold", type=float, default=HIGH_EDGE_THRESHOLD,
                        help=f"Edge threshold for auto-flagging (default {HIGH_EDGE_THRESHOLD})")
    parser.add_argument("--all", action="store_true", help="Analyze all bets, not just high-edge ones")
    args = parser.parse_args()

    game_date = date.today().isoformat() if args.date == "today" else args.date
    season = int(game_date[:4])

    db = get_db()
    mlb_client = get_client()
    odds_client = get_odds_client()

    # Fetch bets
    if args.bet_id:
        bets_raw = db.get_bets(date=game_date)
        bets = [b for b in bets_raw if b["id"] == args.bet_id]
        if not bets:
            log.error("Bet id=%d not found for date %s", args.bet_id, game_date)
            sys.exit(1)
    else:
        bets_raw = db.get_bets(date=game_date)
        if args.all:
            bets = bets_raw
        else:
            bets = [b for b in bets_raw if abs(b.get("edge") or 0) >= args.threshold]

    if not bets:
        print(f"No bets found for {game_date} with edge >= {args.threshold:.0%}.")
        return

    print(f"\nBet Analysis — {game_date}  ({len(bets)} bet(s) to analyze)\n")

    results = []
    for bet in bets:
        result = analyze_bet(bet, odds_client, mlb_client, db, season)
        print_report(result)
        results.append(result)

    # Summary
    verdicts = [r["verdict"] for r in results]
    confirmed    = verdicts.count("CONFIRMED")
    questionable = verdicts.count("QUESTIONABLE")
    voided       = verdicts.count("VOID")

    print(f"\n{'═'*60}")
    print(f"  SUMMARY  {game_date}")
    print(f"  ✅ CONFIRMED:    {confirmed}")
    print(f"  ⚠️  QUESTIONABLE: {questionable}")
    print(f"  🚫 VOID:         {voided}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
