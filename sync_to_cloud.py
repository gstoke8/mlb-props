#!/usr/bin/env python3
"""sync_to_cloud.py — One-way SQLite → Railway Postgres sync.

Pushes bets, closing_lines, bet_clv, and daily_summary. Safe to re-run;
all writes use ON CONFLICT upsert. Never reads from Postgres.
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from datetime import date, timedelta
from typing import Optional

import psycopg2
import psycopg2.extras

from db import DB_PATH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------

def get_pg_conn():
    """Connect to Postgres via DATABASE_URL. Raises RuntimeError if unset."""
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Export it before running sync_to_cloud:\n"
            "  export DATABASE_URL='postgresql://user:pass@host:5432/dbname'"
        )
    return psycopg2.connect(url)


def _sqlite_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------

_PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS bets (
    id                SERIAL PRIMARY KEY,
    bet_date          TEXT NOT NULL,
    game_date         TEXT NOT NULL,
    game_time         TEXT,
    game_pk           INTEGER,
    player_id         INTEGER NOT NULL,
    player_name       TEXT NOT NULL,
    team              TEXT NOT NULL,
    opponent          TEXT NOT NULL,
    prop_type         TEXT NOT NULL,
    line              REAL NOT NULL,
    pick              TEXT NOT NULL,
    book              TEXT NOT NULL,
    odds              INTEGER NOT NULL,
    implied_prob      REAL NOT NULL,
    model_projection  REAL NOT NULL,
    model_prob        REAL NOT NULL,
    edge              REAL NOT NULL,
    confidence        TEXT NOT NULL,
    model_version     TEXT NOT NULL,
    feature_snapshot  TEXT,
    units             REAL NOT NULL,
    stake_usd         REAL,
    open_line         INTEGER,
    line_at_open      REAL,
    is_live           BOOLEAN NOT NULL DEFAULT FALSE,
    notes             TEXT,
    created_at        TEXT NOT NULL DEFAULT (NOW()::TEXT),
    actual_stat       REAL,
    outcome           TEXT,
    pl_units          REAL,
    pl_usd            REAL,
    result_updated_at TEXT
);
CREATE TABLE IF NOT EXISTS closing_lines (
    id                 SERIAL PRIMARY KEY,
    game_date          TEXT NOT NULL,
    game_pk            INTEGER,
    player_id          INTEGER NOT NULL,
    player_name        TEXT NOT NULL,
    prop_type          TEXT NOT NULL,
    line               REAL NOT NULL,
    close_odds_over    INTEGER,
    close_odds_under   INTEGER,
    close_implied_over REAL,
    source             TEXT,
    captured_at        TEXT NOT NULL DEFAULT (NOW()::TEXT)
);
CREATE TABLE IF NOT EXISTS bet_clv (
    bet_id        INTEGER PRIMARY KEY,
    close_odds    INTEGER,
    close_implied REAL,
    clv           REAL,
    clv_pct       REAL
);
CREATE TABLE IF NOT EXISTS daily_summary (
    summary_date  TEXT PRIMARY KEY,
    n_bets        INTEGER,
    n_live        INTEGER,
    n_paper       INTEGER,
    units_wagered REAL,
    pl_units      REAL,
    pl_usd        REAL,
    roi_pct       REAL,
    avg_clv       REAL,
    win_rate      REAL,
    updated_at    TEXT
);
"""


def ensure_pg_schema(pg_conn) -> None:
    """Create Postgres tables if they do not already exist."""
    with pg_conn.cursor() as cur:
        cur.execute(_PG_SCHEMA)
    pg_conn.commit()
    logger.info("Postgres schema verified.")

# ---------------------------------------------------------------------------
# Upsert helper
# ---------------------------------------------------------------------------

def _upsert(pg_conn, table: str, pk: str, rows: list[dict]) -> None:
    """Batch-upsert rows into a Postgres table using ON CONFLICT DO UPDATE."""
    if not rows:
        return
    columns = list(rows[0].keys())
    upsert_sql = (
        f"INSERT INTO {table} ({', '.join(columns)}) "
        f"VALUES ({', '.join(['%s'] * len(columns))}) "
        f"ON CONFLICT ({pk}) DO UPDATE SET "
        + ", ".join(f"{c} = EXCLUDED.{c}" for c in columns if c != pk)
    )
    with pg_conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, upsert_sql, [[r[c] for c in columns] for r in rows])
    pg_conn.commit()

# ---------------------------------------------------------------------------
# Per-table sync functions
# ---------------------------------------------------------------------------

def _fetch_sqlite(db: sqlite3.Connection, sql: str, params: tuple) -> list[dict]:
    cur = db.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def sync_bets(sqlite_db: sqlite3.Connection, pg_conn, since_date: Optional[str] = None) -> int:
    """Upsert bets rows; optionally filtered to game_date >= since_date."""
    sql, params = "SELECT * FROM bets", ()
    if since_date:
        sql += " WHERE game_date >= ?"
        params = (since_date,)
    rows = _fetch_sqlite(sqlite_db, sql, params)
    _upsert(pg_conn, "bets", "id", rows)
    logger.info("sync_bets: %d rows (since=%s)", len(rows), since_date)
    return len(rows)


def sync_closing_lines(
    sqlite_db: sqlite3.Connection, pg_conn, since_date: Optional[str] = None
) -> int:
    """Upsert closing_lines rows; optionally filtered to game_date >= since_date."""
    sql, params = "SELECT * FROM closing_lines", ()
    if since_date:
        sql += " WHERE game_date >= ?"
        params = (since_date,)
    rows = _fetch_sqlite(sqlite_db, sql, params)
    _upsert(pg_conn, "closing_lines", "id", rows)
    logger.info("sync_closing_lines: %d rows (since=%s)", len(rows), since_date)
    return len(rows)


def sync_bet_clv(sqlite_db: sqlite3.Connection, pg_conn) -> int:
    """Upsert all bet_clv rows."""
    rows = _fetch_sqlite(sqlite_db, "SELECT * FROM bet_clv", ())
    _upsert(pg_conn, "bet_clv", "bet_id", rows)
    logger.info("sync_bet_clv: %d rows", len(rows))
    return len(rows)


def sync_daily_summary(
    sqlite_db: sqlite3.Connection, pg_conn, since_date: Optional[str] = None
) -> int:
    """Upsert daily_summary rows; optionally filtered to summary_date >= since_date."""
    sql, params = "SELECT * FROM daily_summary", ()
    if since_date:
        sql += " WHERE summary_date >= ?"
        params = (since_date,)
    rows = _fetch_sqlite(sqlite_db, sql, params)
    _upsert(pg_conn, "daily_summary", "summary_date", rows)
    logger.info("sync_daily_summary: %d rows (since=%s)", len(rows), since_date)
    return len(rows)

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_full_sync(since_days: int = 7) -> dict:
    """Sync all four tables from SQLite to Postgres.

    Parameters
    ----------
    since_days:
        Only rows dated >= today - since_days are synced. Pass 0 for full history.

    Returns
    -------
    dict
        {bets, closing_lines, bet_clv, daily_summary: int, errors: list[str]}
    """
    since_date: Optional[str] = None
    if since_days > 0:
        since_date = (date.today() - timedelta(days=since_days)).isoformat()

    summary: dict = {"bets": 0, "closing_lines": 0, "bet_clv": 0, "daily_summary": 0, "errors": []}

    try:
        pg_conn = get_pg_conn()
    except RuntimeError as exc:
        summary["errors"].append(str(exc))
        logger.error("Cannot connect to Postgres: %s", exc)
        return summary

    sqlite_db = _sqlite_conn()
    try:
        ensure_pg_schema(pg_conn)
        tasks = [
            ("bets", lambda: sync_bets(sqlite_db, pg_conn, since_date)),
            ("closing_lines", lambda: sync_closing_lines(sqlite_db, pg_conn, since_date)),
            ("bet_clv", lambda: sync_bet_clv(sqlite_db, pg_conn)),
            ("daily_summary", lambda: sync_daily_summary(sqlite_db, pg_conn, since_date)),
        ]
        for label, fn in tasks:
            try:
                summary[label] = fn()
            except Exception as exc:
                msg = f"{label}: {exc}"
                summary["errors"].append(msg)
                logger.error("Sync error — %s", msg)
    finally:
        sqlite_db.close()
        pg_conn.close()

    return summary

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description="Sync local props.db to Railway Postgres.")
    parser.add_argument("--since-days", type=int, default=7, metavar="N",
                        help="Sync rows from the last N days (default: 7).")
    parser.add_argument("--full", action="store_true", default=False,
                        help="Sync all historical rows (overrides --since-days).")
    args = parser.parse_args()

    since_days = 0 if args.full else args.since_days
    result = run_full_sync(since_days=since_days)
    logger.info("Sync complete: %s", result)
    sys.exit(1 if result["errors"] else 0)


if __name__ == "__main__":
    main()
