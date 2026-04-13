#!/usr/bin/env python3
"""
MLB Props Flask Dashboard

Serves the main HTML dashboard and JSON API endpoints for bet tracking,
P/L analysis, and closing line value summaries.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from datetime import date, timedelta

from flask import Flask, jsonify, render_template

from db import DB_PATH, calculate_implied_prob  # noqa: F401 — re-exported for callers

# ---------------------------------------------------------------------------
# Database backend selection
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PORT = int(os.getenv("DASHBOARD_PORT", "5050"))
PLT_DAYS = 30
WEEK_DAYS = 7

PROP_LABELS: dict[str, str] = {
    "batter_home_runs": "HR",
    "pitcher_strikeouts": "K",
    "batter_hits": "H",
    "batter_rbis": "RBI",
    "batter_runs_scored": "R",
    "batter_total_bases": "TB",
    "batter_walks": "BB",
    "batter_strikeouts": "SO",
    "pitcher_hits_allowed": "HA",
    "pitcher_walks": "BB",
    "pitcher_earned_runs": "ER",
    "pitcher_outs": "OUT",
}

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
def _query_sqlite(sql: str, params: tuple) -> list[dict]:
    """Execute a read-only query against the local SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        cur = conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def _query_pg(sql: str, params: tuple) -> list[dict]:
    """Execute a read-only query against the Postgres database on Railway."""
    import psycopg2  # imported lazily — not required when running locally

    # Adapt SQLite-style placeholders and functions to Postgres equivalents
    pg_sql = sql.replace("?", "%s").replace("datetime('now')", "NOW()")

    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(pg_sql, params)
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def _query(sql: str, params: tuple = ()) -> list[dict]:
    """Execute a read-only SQL query and return rows as plain dicts.

    Routes to Postgres when DATABASE_URL is set (Railway deployment),
    otherwise falls back to the local SQLite database.
    """
    try:
        if DATABASE_URL:
            return _query_pg(sql, params)
        return _query_sqlite(sql, params)
    except Exception as exc:
        logger.error("DB query failed: %s | sql=%s params=%s", exc, sql, params)
        raise


# ---------------------------------------------------------------------------
# Routes — HTML
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("dashboard.html")


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------
@app.route("/api/today")
def api_today():
    today = date.today().isoformat()
    rows = _query(
        """
        SELECT id, player_name, team, opponent, prop_type, line, pick,
               book, odds, edge, confidence, units, is_live,
               outcome, pl_units, actual_stat
        FROM   bets
        WHERE  game_date = ?
        ORDER  BY confidence DESC, edge DESC
        """,
        (today,),
    )
    total_units = sum(r["units"] or 0 for r in rows)
    live_count = sum(1 for r in rows if r["is_live"])
    paper_count = len(rows) - live_count
    return jsonify(
        {
            "date": today,
            "bets": rows,
            "total_units": round(total_units, 4),
            "paper_count": paper_count,
            "live_count": live_count,
        }
    )


@app.route("/api/week")
def api_week():
    cutoff = (date.today() - timedelta(days=WEEK_DAYS - 1)).isoformat()
    rows = _query(
        """
        SELECT id, game_date, player_name, team, opponent, prop_type,
               line, pick, book, odds, edge, confidence, units, is_live,
               outcome, pl_units, actual_stat
        FROM   bets
        WHERE  game_date >= ?
        ORDER  BY game_date DESC, confidence DESC
        """,
        (cutoff,),
    )
    # Group by date for the accordion view
    grouped: dict[str, dict] = {}
    for row in rows:
        d = row["game_date"]
        if d not in grouped:
            grouped[d] = {"date": d, "bets": [], "daily_pl": 0.0, "bet_count": 0}
        grouped[d]["bets"].append(row)
        grouped[d]["daily_pl"] = round(
            grouped[d]["daily_pl"] + (row["pl_units"] or 0), 4
        )
        grouped[d]["bet_count"] += 1
    days = sorted(grouped.values(), key=lambda x: x["date"], reverse=True)
    return jsonify({"days": days, "total_bets": len(rows)})


@app.route("/api/pl")
def api_pl():
    cutoff = (date.today() - timedelta(days=PLT_DAYS - 1)).isoformat()
    rows = _query(
        """
        SELECT game_date,
               SUM(pl_units)  AS daily_pl,
               COUNT(*)       AS bets
        FROM   bets
        WHERE  game_date >= ?
          AND  outcome IS NOT NULL
        GROUP  BY game_date
        ORDER  BY game_date ASC
        """,
        (cutoff,),
    )
    series = []
    cumulative = 0.0
    for row in rows:
        cumulative = round(cumulative + (row["daily_pl"] or 0), 4)
        series.append(
            {
                "date": row["game_date"],
                "daily_pl": round(row["daily_pl"] or 0, 4),
                "cumulative_pl": cumulative,
                "bets": row["bets"],
            }
        )
    daily_pls = [s["daily_pl"] for s in series]
    return jsonify(
        {
            "series": series,
            "total_pl": cumulative,
            "best_day": max(daily_pls, default=0.0),
            "worst_day": min(daily_pls, default=0.0),
        }
    )


@app.route("/api/clv")
def api_clv():
    rows = _query(
        """
        SELECT b.confidence, bc.clv
        FROM   bet_clv bc
        JOIN   bets b ON b.id = bc.bet_id
        WHERE  bc.clv IS NOT NULL
        """
    )
    if not rows:
        return jsonify(
            {
                "avg_clv": 0.0,
                "clv_positive_pct": 0.0,
                "total_bets_with_clv": 0,
                "by_confidence": {},
            }
        )
    total = len(rows)
    all_clv = [r["clv"] for r in rows]
    positive = sum(1 for c in all_clv if c > 0)
    avg_clv = round(sum(all_clv) / total, 4)
    clv_positive_pct = round(positive / total * 100, 2)
    by_conf: dict[str, dict] = {}
    for row in rows:
        conf = row["confidence"] or "UNKNOWN"
        bucket = by_conf.setdefault(conf, {"clv_sum": 0.0, "count": 0})
        bucket["clv_sum"] += row["clv"]
        bucket["count"] += 1
    by_confidence = {
        conf: {
            "avg_clv": round(v["clv_sum"] / v["count"], 4),
            "count": v["count"],
        }
        for conf, v in by_conf.items()
    }
    return jsonify(
        {
            "avg_clv": avg_clv,
            "clv_positive_pct": clv_positive_pct,
            "total_bets_with_clv": total,
            "by_confidence": by_confidence,
        }
    )


@app.route("/api/summary")
def api_summary():
    rows = _query(
        """
        SELECT outcome, pl_units, units
        FROM   bets
        WHERE  outcome IS NOT NULL
        """
    )
    total_bets = len(rows)
    wins = sum(1 for r in rows if r["outcome"] == "WIN")
    losses = sum(1 for r in rows if r["outcome"] == "LOSS")
    pushes = sum(1 for r in rows if r["outcome"] == "PUSH")
    total_pl = round(sum(r["pl_units"] or 0 for r in rows), 4)
    total_units_wagered = sum(r["units"] or 0 for r in rows)
    roi_pct = (
        round(total_pl / total_units_wagered * 100, 2) if total_units_wagered else 0.0
    )
    hit_rate = round(wins / (wins + losses), 4) if (wins + losses) else 0.0
    # Last 7 days P/L
    cutoff_7 = (date.today() - timedelta(days=6)).isoformat()
    recent = _query(
        """
        SELECT SUM(pl_units) AS pl FROM bets
        WHERE game_date >= ? AND outcome IS NOT NULL
        """,
        (cutoff_7,),
    )
    last_7_pl = round(recent[0]["pl"] or 0, 4) if recent else 0.0
    # Streak: consecutive same-outcome from most recent resolved bet
    streak_rows = _query(
        """
        SELECT outcome FROM bets
        WHERE outcome IN ('WIN','LOSS')
        ORDER BY game_date DESC, id DESC
        LIMIT 20
        """
    )
    streak = _calc_streak(streak_rows)
    # Average CLV
    clv_row = _query("SELECT AVG(clv) AS avg_clv FROM bet_clv WHERE clv IS NOT NULL")
    avg_clv = round(clv_row[0]["avg_clv"] or 0, 4) if clv_row else 0.0
    return jsonify(
        {
            "total_bets": total_bets,
            "record": f"{wins}-{losses}-{pushes}",
            "hit_rate": hit_rate,
            "total_pl": total_pl,
            "roi_pct": roi_pct,
            "avg_clv": avg_clv,
            "last_7_pl": last_7_pl,
            "streak": streak,
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _calc_streak(rows: list[dict]) -> str:
    """Return a streak string like '+3' or '-2' from ordered outcome rows."""
    if not rows:
        return "0"
    first = rows[0]["outcome"]
    count = 0
    for row in rows:
        if row["outcome"] == first:
            count += 1
        else:
            break
    prefix = "+" if first == "WIN" else "-"
    return f"{prefix}{count}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting MLB Props Dashboard on port %d", PORT)
    app.run(host="0.0.0.0", port=PORT, debug=False)
