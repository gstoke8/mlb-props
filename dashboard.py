from __future__ import annotations
#!/usr/bin/env python3
"""
MLB Props Flask Dashboard

Serves the main HTML dashboard and JSON API endpoints for bet tracking,
P/L analysis, and closing line value summaries.
"""

import logging
import os
import plistlib
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

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
    # API market keys
    "batter_home_runs": "HR",
    "pitcher_strikeouts": "K",
    "batter_hits": "H",
    "batter_rbis": "RBI",
    "batter_runs_scored": "R",
    "batter_total_bases": "TB",
    "batter_singles": "1B",
    "batter_doubles": "2B",
    "batter_hits_runs_rbis": "H+R+RBI",
    "batter_walks": "BB",
    "batter_strikeouts": "SO",
    "pitcher_hits_allowed": "HA",
    "pitcher_walks": "BB",
    "pitcher_earned_runs": "ER",
    "pitcher_outs": "OUT",
    # Internal prop_type names (after mapping)
    "home_runs": "HR",
    "strikeouts": "K",
    "hits": "H",
    "rbis": "RBI",
    "runs_scored": "R",
    "total_bases": "TB",
    "singles": "1B",
    "doubles": "2B",
    "hits_runs_rbis": "H+R+RBI",
}

# ---------------------------------------------------------------------------
# Schedule enrichment cache (populated lazily, refreshed hourly)
# ---------------------------------------------------------------------------
_schedule_cache: dict[str, dict] = {}   # game_pk → {away_team, home_team, game_time_et}
_schedule_cache_date: str = ""

_ET_OFFSET = timedelta(hours=4)  # EDT (Mar–Nov)


def _get_schedule_info(game_date: str) -> dict[int, dict]:
    """Return game_pk → {away_team, home_team, game_time_et} for game_date."""
    global _schedule_cache, _schedule_cache_date
    if _schedule_cache_date == game_date and _schedule_cache:
        return _schedule_cache
    try:
        from mlb_api import get_client
        schedule = get_client().get_schedule(game_date)
        result: dict[int, dict] = {}
        for g in schedule:
            gk = g.get("game_pk")
            if not gk:
                continue
            gt_utc = g.get("game_time_utc", "")
            game_time_et = ""
            if gt_utc:
                try:
                    dt_utc = datetime.strptime(gt_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                    dt_et = dt_utc - _ET_OFFSET
                    game_time_et = dt_et.strftime("%-I:%M %p ET")
                except ValueError:
                    pass
            result[gk] = {
                "away_team": g.get("away_team", ""),
                "home_team": g.get("home_team", ""),
                "game_time_et": game_time_et,
            }
        _schedule_cache = result
        _schedule_cache_date = game_date
        return result
    except Exception as exc:
        logger.warning("Could not fetch schedule for enrichment: %s", exc)
        return {}


def _enrich_bet(bet: dict, schedule_info: dict[int, dict]) -> dict:
    """Fill missing game_time/matchup on a bet row from schedule data."""
    if bet.get("game_time") and (bet.get("team") or bet.get("opponent")):
        return bet
    gk = bet.get("game_pk")
    if not gk or gk not in schedule_info:
        return bet
    ginfo = schedule_info[gk]
    result = {**bet}
    if not result.get("game_time"):
        result["game_time"] = ginfo["game_time_et"]
    # When team is unknown, show the full matchup in opponent so the game is identifiable
    if not result.get("team") and not result.get("opponent"):
        away = ginfo.get("away_team", "")
        home = ginfo.get("home_team", "")
        if away and home:
            result["opponent"] = f"{away} @ {home}"
    return result


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
# Schedule helpers
# ---------------------------------------------------------------------------
_LAUNCH_AGENTS = Path.home() / "Library" / "LaunchAgents"
_ET_TZ = timezone(timedelta(hours=-4))   # EDT (UTC-4, Mar–Nov)

SLOT_LABELS: dict[str, str] = {
    "morning":  "Morning Snapshot",
    "slot":     "Model Run",
    "results":  "Results Pull",
    "statcast": "Statcast Nightly",
}


def _read_schedule() -> list[dict]:
    """
    Parse all com.mlb-props.*.plist files from LaunchAgents and return a
    sorted list of scheduled runs for today with status annotations.
    """
    now_et = datetime.now(_ET_TZ)
    today_date = now_et.date()
    entries: list[dict] = []

    for plist_path in sorted(_LAUNCH_AGENTS.glob("com.mlb-props.*.plist")):
        label = plist_path.stem  # e.g. com.mlb-props.slot-1735
        # Skip dashboard / tunnel / unrelated plists
        slug = label.removeprefix("com.mlb-props.")  # morning | slot-1735 | statcast | …

        try:
            with plist_path.open("rb") as fh:
                data = plistlib.load(fh)
        except Exception:
            continue

        cal = data.get("StartCalendarInterval")
        if not cal:
            continue

        hour   = cal.get("Hour")
        minute = cal.get("Minute", 0)
        if hour is None:
            continue

        run_dt = datetime(
            today_date.year, today_date.month, today_date.day,
            hour, minute, 0, tzinfo=_ET_TZ,
        )
        elapsed_min = (now_et - run_dt).total_seconds() / 60

        if elapsed_min > 10:
            status = "done"
        elif elapsed_min > -5:
            status = "running"
        else:
            status = "upcoming"

        # Skip results plists — resolved silently in the background
        if slug == "results" or slug.startswith("results-"):
            continue

        # Human-readable label
        if slug == "morning":
            display = "Morning Snapshot"
            role    = "morning"
        elif slug.startswith("slot"):
            display = "Model Run"
            role    = "slot"
        elif slug == "statcast":
            display = "Statcast Nightly"
            role    = "statcast"
        elif slug == "scheduler":
            display = "Daily Scheduler"
            role    = "scheduler"
        else:
            display = slug
            role    = "other"

        entries.append({
            "label":      label,
            "slug":       slug,
            "role":       role,
            "display":    display,
            "time_et":    run_dt.strftime("%-I:%M %p"),
            "hour":       hour,
            "minute":     minute,
            "status":     status,
            "sort_key":   hour * 60 + minute,
        })

    entries.sort(key=lambda e: e["sort_key"])
    # Mark first upcoming as "next"
    for e in entries:
        if e["status"] == "upcoming":
            e["status"] = "next"
            break

    return entries


# ---------------------------------------------------------------------------
# Routes — HTML
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("dashboard.html")


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------
@app.route("/api/schedule")
def api_schedule():
    entries = _read_schedule()
    now_et = datetime.now(_ET_TZ)
    return jsonify({
        "entries":     entries,
        "current_time": now_et.strftime("%-I:%M %p ET"),
        "run_count":   len(entries),
        "done_count":  sum(1 for e in entries if e["status"] == "done"),
        "upcoming_count": sum(1 for e in entries if e["status"] in ("next", "upcoming")),
    })


@app.route("/api/today")
def api_today():
    today = date.today().isoformat()
    rows = _query(
        """
        SELECT id, game_pk, player_name, team, opponent, game_time, prop_type, line, pick,
               book, odds, edge, confidence, units, is_live,
               outcome, pl_units, actual_stat, model_projection, model_prob, no_vig_prob
        FROM   bets
        WHERE  game_date = ?
        ORDER  BY game_time, prop_type, player_name
        """,
        (today,),
    )
    # Enrich rows with game_time / matchup from today's schedule when missing
    schedule_info = _get_schedule_info(today)
    rows = [_enrich_bet(r, schedule_info) for r in rows]

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



@app.route("/api/breakdown")
def api_breakdown():
    """Return betting record and ROI broken down by prop type, odds band, and edge bucket."""
    _PROP_LABEL_SQL = """
        CASE prop_type
            WHEN 'pitcher_strikeouts'   THEN 'K'
            WHEN 'strikeouts'           THEN 'K'
            WHEN 'batter_hits'          THEN 'H'
            WHEN 'hits'                 THEN 'H'
            WHEN 'batter_home_runs'     THEN 'HR'
            WHEN 'home_runs'            THEN 'HR'
            WHEN 'batter_rbis'          THEN 'RBI'
            WHEN 'rbis'                 THEN 'RBI'
            WHEN 'batter_runs_scored'   THEN 'R'
            WHEN 'runs_scored'          THEN 'R'
            WHEN 'batter_total_bases'   THEN 'TB'
            WHEN 'total_bases'          THEN 'TB'
            WHEN 'batter_hits_runs_rbis' THEN 'H+R+RBI'
            WHEN 'hits_runs_rbis'       THEN 'H+R+RBI'
            WHEN 'batter_singles'       THEN '1B'
            WHEN 'singles'              THEN '1B'
            WHEN 'batter_doubles'       THEN '2B'
            WHEN 'doubles'              THEN '2B'
            WHEN 'batter_walks'         THEN 'BB'
            WHEN 'walks'                THEN 'BB'
            ELSE prop_type
        END
    """

    # By prop type
    prop_rows = _query(
        f"""
        SELECT
            ({_PROP_LABEL_SQL})                                                    AS prop_label,
            COUNT(*)                                                               AS bets,
            SUM(CASE WHEN outcome='WIN'  THEN 1 ELSE 0 END)                       AS wins,
            SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END)                       AS losses,
            SUM(CASE WHEN outcome='PUSH' THEN 1 ELSE 0 END)                       AS pushes,
            ROUND(100.0 * SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END)
                / NULLIF(SUM(CASE WHEN outcome IN ('WIN','LOSS') THEN 1 ELSE 0 END), 0), 1) AS win_pct,
            ROUND(SUM(pl_units), 2)                                                AS pl_units,
            ROUND(100.0 * SUM(pl_units) / NULLIF(SUM(units), 0), 1)               AS roi_pct,
            ROUND(AVG(edge) * 100, 1)                                              AS avg_edge_pct
        FROM bets
        WHERE outcome IS NOT NULL
        GROUP BY prop_label
        ORDER BY bets DESC
        """,
    )
    by_prop_type = list(prop_rows)

    # By odds band
    odds_rows = _query(
        """
        SELECT
            CASE
                WHEN odds <= -151             THEN '≤ -151'
                WHEN odds BETWEEN -150 AND -111 THEN '-150 to -111'
                WHEN odds BETWEEN -110 AND -101 THEN '-110 to -101'
                WHEN odds BETWEEN -100 AND  100 THEN '-100 to +100'
                WHEN odds BETWEEN  101 AND  150 THEN '+101 to +150'
                ELSE                              '+151 and over'
            END AS odds_band,
            CASE
                WHEN odds <= -151             THEN 1
                WHEN odds BETWEEN -150 AND -111 THEN 2
                WHEN odds BETWEEN -110 AND -101 THEN 3
                WHEN odds BETWEEN -100 AND  100 THEN 4
                WHEN odds BETWEEN  101 AND  150 THEN 5
                ELSE                              6
            END AS sort_order,
            COUNT(*)                                                               AS bets,
            SUM(CASE WHEN outcome='WIN'  THEN 1 ELSE 0 END)                       AS wins,
            SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END)                       AS losses,
            ROUND(100.0 * SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END)
                / NULLIF(SUM(CASE WHEN outcome IN ('WIN','LOSS') THEN 1 ELSE 0 END), 0), 1) AS win_pct,
            ROUND(SUM(pl_units), 2)                                                AS pl_units,
            ROUND(100.0 * SUM(pl_units) / NULLIF(SUM(units), 0), 1)               AS roi_pct,
            ROUND(AVG(odds), 0)                                                    AS avg_odds
        FROM bets
        WHERE outcome IS NOT NULL
        GROUP BY odds_band, sort_order
        ORDER BY sort_order
        """,
    )
    by_odds = [{k: v for k, v in r.items() if k != "sort_order"} for r in odds_rows]

    # By edge bucket
    edge_rows = _query(
        """
        SELECT
            CASE
                WHEN edge * 100 <  5  THEN '1.5–5%'
                WHEN edge * 100 < 10  THEN '5–10%'
                WHEN edge * 100 < 15  THEN '10–15%'
                WHEN edge * 100 < 20  THEN '15–20%'
                WHEN edge * 100 < 30  THEN '20–30%'
                ELSE                       '30%+'
            END AS edge_bucket,
            CASE
                WHEN edge * 100 <  5  THEN 1
                WHEN edge * 100 < 10  THEN 2
                WHEN edge * 100 < 15  THEN 3
                WHEN edge * 100 < 20  THEN 4
                WHEN edge * 100 < 30  THEN 5
                ELSE                       6
            END AS sort_order,
            COUNT(*)                                                               AS bets,
            SUM(CASE WHEN outcome='WIN'  THEN 1 ELSE 0 END)                       AS wins,
            SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END)                       AS losses,
            ROUND(100.0 * SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END)
                / NULLIF(SUM(CASE WHEN outcome IN ('WIN','LOSS') THEN 1 ELSE 0 END), 0), 1) AS win_pct,
            ROUND(SUM(pl_units), 2)                                                AS pl_units,
            ROUND(100.0 * SUM(pl_units) / NULLIF(SUM(units), 0), 1)               AS roi_pct,
            ROUND(AVG(edge) * 100, 1)                                              AS avg_edge_pct,
            ROUND(AVG(COALESCE(no_vig_prob, implied_prob)) * 100, 1)               AS avg_mkt_prob_pct,
            ROUND(AVG(model_prob) * 100, 1)                                        AS avg_model_prob_pct
        FROM bets
        WHERE outcome IS NOT NULL
        GROUP BY edge_bucket, sort_order
        ORDER BY sort_order
        """,
    )
    by_edge = [{k: v for k, v in r.items() if k != "sort_order"} for r in edge_rows]

    # Totals
    totals_rows = _query(
        """
        SELECT
            COUNT(*)                                                               AS bets,
            SUM(units)                                                             AS units_staked,
            SUM(CASE WHEN outcome='WIN'  THEN 1 ELSE 0 END)                       AS wins,
            SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END)                       AS losses,
            ROUND(100.0 * SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END)
                / NULLIF(SUM(CASE WHEN outcome IN ('WIN','LOSS') THEN 1 ELSE 0 END), 0), 1) AS win_pct,
            ROUND(SUM(pl_units), 2)                                                AS pl_units,
            ROUND(100.0 * SUM(pl_units) / NULLIF(SUM(units), 0), 1)               AS roi_pct,
            ROUND(AVG(edge) * 100, 1)                                              AS avg_edge_pct
        FROM bets
        WHERE outcome IS NOT NULL
        """,
    )
    totals = totals_rows[0] if totals_rows else {}

    return jsonify({
        "by_prop_type": by_prop_type,
        "by_odds": by_odds,
        "by_edge": by_edge,
        "totals": totals,
    })


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
    # Last 3 days ROI (for mood face)
    cutoff_3 = (date.today() - timedelta(days=2)).isoformat()
    recent_3d = _query(
        """
        SELECT SUM(pl_units) AS pl, SUM(units) AS staked FROM bets
        WHERE game_date >= ? AND outcome IS NOT NULL
        """,
        (cutoff_3,),
    )
    r3d = recent_3d[0] if recent_3d else {}
    pl_3d   = r3d.get("pl")     or 0.0
    staked_3d = r3d.get("staked") or 0.0
    roi_3d = round(pl_3d / staked_3d * 100, 1) if staked_3d else 0.0
    return jsonify(
        {
            "total_bets": total_bets,
            "record": f"{wins}-{losses}-{pushes}",
            "hit_rate": hit_rate,
            "total_pl": total_pl,
            "roi_pct": roi_pct,
            "last_7_pl": last_7_pl,
            "roi_3d": roi_3d,
            "pl_3d": round(pl_3d, 2),
        }
    )


@app.route("/api/models")
def api_models():
    """Return current model versions loaded from each pkl's train_meta."""
    import joblib
    models_dir = Path.home() / "mlb-props" / "models"
    result = {}
    pairs = [
        ("hits_model.pkl", "hits"),
        ("k_model.pkl",    "strikeouts"),
        ("hr_model.pkl",   "home_runs"),
    ]
    for fname, key in pairs:
        pkl = models_dir / fname
        if not pkl.exists():
            result[key] = {"version": "not found", "n_train": None, "trained_at": None}
            continue
        try:
            obj = joblib.load(pkl)
            meta = getattr(obj, "train_meta", {}) or {}
            result[key] = {
                "version":  meta.get("model_version", "unknown"),
                "n_train":  meta.get("n_train"),
                "trained_at": datetime.fromtimestamp(
                    pkl.stat().st_mtime, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M UTC"),
            }
        except Exception as exc:
            result[key] = {"version": "error", "error": str(exc)}
    return jsonify(result)


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
