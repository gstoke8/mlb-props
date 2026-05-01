#!/usr/bin/env python3
"""
MLB Props Database Module

SQLite schema and interface for tracking MLB player prop bets, results,
closing line value, park/umpire factors, and model performance.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
DB_PATH = Path.home() / "mlb-props" / "props.db"

_REQUIRED_BET_FIELDS = (
    "bet_date",
    "game_date",
    "player_id",
    "player_name",
    "team",
    "opponent",
    "prop_type",
    "line",
    "pick",
    "book",
    "odds",
    "implied_prob",
    "model_projection",
    "model_prob",
    "edge",
    "confidence",
    "model_version",
    "units",
)

_RESULT_FIELDS = frozenset(
    {"actual_stat", "outcome", "pl_units", "pl_usd", "result_updated_at"}
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS bets (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    bet_date         TEXT NOT NULL,
    game_date        TEXT NOT NULL,
    game_time        TEXT,
    game_pk          INTEGER,
    player_id        INTEGER NOT NULL,
    player_name      TEXT NOT NULL,
    team             TEXT NOT NULL,
    opponent         TEXT NOT NULL,
    prop_type        TEXT NOT NULL,
    line             REAL NOT NULL,
    pick             TEXT NOT NULL,
    book             TEXT NOT NULL,
    odds             INTEGER NOT NULL,
    implied_prob     REAL NOT NULL,
    model_projection REAL NOT NULL,
    model_prob       REAL NOT NULL,
    edge             REAL NOT NULL,
    confidence       TEXT NOT NULL,
    model_version    TEXT NOT NULL,
    feature_snapshot TEXT,
    units            REAL NOT NULL,
    stake_usd        REAL,
    open_line        INTEGER,
    line_at_open     REAL,
    is_live          INTEGER NOT NULL DEFAULT 0,
    notes            TEXT,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    actual_stat      REAL,
    outcome          TEXT,
    pl_units         REAL,
    pl_usd           REAL,
    result_updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_bets_game_date   ON bets(game_date);
CREATE INDEX IF NOT EXISTS idx_bets_bet_date    ON bets(bet_date);
CREATE INDEX IF NOT EXISTS idx_bets_player_id   ON bets(player_id);
CREATE INDEX IF NOT EXISTS idx_bets_prop_type   ON bets(prop_type);
CREATE INDEX IF NOT EXISTS idx_bets_outcome     ON bets(outcome);
CREATE INDEX IF NOT EXISTS idx_bets_confidence  ON bets(confidence);
CREATE INDEX IF NOT EXISTS idx_bets_is_live     ON bets(is_live);

CREATE TABLE IF NOT EXISTS closing_lines (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    game_date         TEXT NOT NULL,
    game_pk           INTEGER,
    player_id         INTEGER NOT NULL,
    player_name       TEXT NOT NULL,
    prop_type         TEXT NOT NULL,
    line              REAL NOT NULL,
    close_odds_over   INTEGER,
    close_odds_under  INTEGER,
    close_implied_over REAL,
    source            TEXT,
    captured_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_cl_game_date  ON closing_lines(game_date);
CREATE INDEX IF NOT EXISTS idx_cl_player_id  ON closing_lines(player_id);
CREATE INDEX IF NOT EXISTS idx_cl_prop_type  ON closing_lines(prop_type);

CREATE TABLE IF NOT EXISTS bet_clv (
    bet_id        INTEGER PRIMARY KEY REFERENCES bets(id),
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

CREATE TABLE IF NOT EXISTS player_stats (
    player_id  INTEGER NOT NULL,
    stat_date  TEXT NOT NULL,
    stat_type  TEXT NOT NULL,
    value      REAL,
    rolling_3  REAL,
    rolling_7  REAL,
    rolling_14 REAL,
    rolling_30 REAL,
    PRIMARY KEY (player_id, stat_date, stat_type)
);

CREATE INDEX IF NOT EXISTS idx_ps_player_id ON player_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_ps_stat_date ON player_stats(stat_date);
CREATE INDEX IF NOT EXISTS idx_ps_stat_type ON player_stats(stat_type);

CREATE TABLE IF NOT EXISTS features (
    player_id    INTEGER NOT NULL,
    feat_date    TEXT NOT NULL,
    prop_type    TEXT NOT NULL,
    feature_json TEXT,
    created_at   TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (player_id, feat_date, prop_type)
);

CREATE TABLE IF NOT EXISTS park_factors (
    park_name    TEXT NOT NULL,
    handedness   TEXT NOT NULL,
    K_factor     REAL,
    HR_factor    REAL,
    runs_factor  REAL,
    updated_at   TEXT,
    PRIMARY KEY (park_name, handedness)
);

CREATE TABLE IF NOT EXISTS umpire_factors (
    umpire_id     INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    K_factor      REAL,
    BB_factor     REAL,
    games_counted INTEGER,
    last_updated  TEXT
);

CREATE TABLE IF NOT EXISTS pitch_type_performance (
    player_id       TEXT NOT NULL,
    player_type     TEXT NOT NULL,
    pitch_type      TEXT NOT NULL,
    pct_thrown_seen REAL,
    barrel_pct      REAL,
    whiff_pct       REAL,
    gb_pct          REAL,
    fb_pct          REAL,
    ld_pct          REAL,
    woba            REAL,
    iso             REAL,
    hr_per_fb       REAL,
    pa_or_pitches   INTEGER,
    last_updated    TIMESTAMP,
    PRIMARY KEY (player_id, pitch_type)
);

CREATE TABLE IF NOT EXISTS model_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date      TEXT NOT NULL,
    model_version TEXT NOT NULL,
    prop_type     TEXT NOT NULL,
    n_picks       INTEGER,
    mean_edge     REAL,
    created_at    TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_mr_run_date ON model_runs(run_date);

CREATE TABLE IF NOT EXISTS prop_opening_snapshots (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date  TEXT NOT NULL,
    player_name    TEXT NOT NULL,
    prop_type      TEXT NOT NULL,
    line           REAL,
    no_vig_implied REAL NOT NULL,
    captured_at    TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(snapshot_date, player_name, prop_type, line)
);

CREATE INDEX IF NOT EXISTS idx_pos_date ON prop_opening_snapshots(snapshot_date);
"""


# ---------------------------------------------------------------------------
# Module-level P/L helper functions
# ---------------------------------------------------------------------------

def calculate_pl(outcome: str, odds: int, units: float) -> float:
    """Return P/L in units for a resolved bet.

    Parameters
    ----------
    outcome:
        One of ``'WIN'``, ``'LOSS'``, ``'PUSH'``, ``'NO_ACTION'``.
    odds:
        American-style odds (e.g. ``-110``, ``+150``).
    units:
        Size of the wager in units.

    Returns
    -------
    float
        Positive value for a win, negative for a loss, 0 for push/void.
    """
    if outcome in ("PUSH", "NO_ACTION"):
        return 0.0
    if outcome == "LOSS":
        return -abs(units)
    if outcome == "WIN":
        if odds > 0:
            return units * (odds / 100.0)
        return units * (100.0 / abs(odds))
    raise ValueError("outcome must be WIN, LOSS, PUSH, or NO_ACTION")


def calculate_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability (no vig removal).

    Parameters
    ----------
    odds:
        American-style odds integer.

    Returns
    -------
    float
        Probability in the range ``(0, 1)``.
    """
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def calculate_clv(our_implied: float, close_implied: float) -> float:
    """Compute closing line value in percentage points.

    A positive result means we bet at better odds than the closing line
    (beat the close).

    Parameters
    ----------
    our_implied:
        Implied probability at the time we placed the bet.
    close_implied:
        Implied probability at closing.

    Returns
    -------
    float
        CLV in percentage points; e.g. ``2.5`` means we were 2.5pp better.
    """
    return (close_implied - our_implied) * 100.0


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class MLBPropsDB:
    """SQLite database interface for MLB player-props bet tracking."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Infrastructure
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self):
        """Context manager that yields a committed (or rolled-back) connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create all tables and indexes if they do not already exist."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    # ------------------------------------------------------------------
    # Core bet operations
    # ------------------------------------------------------------------

    def save_bet(self, bet: Dict[str, Any]) -> int:
        """Insert a new bet row and return its auto-assigned ``id``.

        Parameters
        ----------
        bet:
            Dictionary containing bet fields.  All fields listed in
            ``_REQUIRED_BET_FIELDS`` must be present and non-``None``.

        Returns
        -------
        int
            The ``id`` of the newly inserted row.

        Raises
        ------
        ValueError
            If any required field is missing or ``None``.
        """
        for field in _REQUIRED_BET_FIELDS:
            if bet.get(field) is None:
                raise ValueError("Missing required bet field: {}".format(field))

        feature_snapshot = bet.get("feature_snapshot")
        if isinstance(feature_snapshot, dict):
            feature_snapshot = json.dumps(feature_snapshot)

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO bets (
                    bet_date, game_date, game_time, game_pk,
                    player_id, player_name, team, opponent,
                    prop_type, line, pick, book, odds,
                    implied_prob, no_vig_prob, model_projection, model_prob,
                    edge, confidence, model_version,
                    feature_snapshot, units, stake_usd,
                    open_line, line_at_open, is_live, notes
                ) VALUES (
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?
                )
                """,
                (
                    bet["bet_date"],
                    bet["game_date"],
                    bet.get("game_time"),
                    bet.get("game_pk"),
                    bet["player_id"],
                    bet["player_name"],
                    bet["team"],
                    bet["opponent"],
                    bet["prop_type"],
                    bet["line"],
                    bet["pick"],
                    bet["book"],
                    bet["odds"],
                    bet["implied_prob"],
                    bet.get("no_vig_prob"),
                    bet["model_projection"],
                    bet["model_prob"],
                    bet["edge"],
                    bet["confidence"],
                    bet["model_version"],
                    feature_snapshot,
                    bet["units"],
                    bet.get("stake_usd"),
                    bet.get("open_line"),
                    bet.get("line_at_open"),
                    int(bet.get("is_live", 0)),
                    bet.get("notes"),
                ),
            )
            return cursor.lastrowid

    def update_result(
        self,
        bet_id: int,
        actual_stat: float,
        outcome: str,
        pl_units: float,
        pl_usd: Optional[float] = None,
    ) -> None:
        """Record the final result for a bet.

        Only the result columns (``actual_stat``, ``outcome``, ``pl_units``,
        ``pl_usd``, ``result_updated_at``) are ever written by this method —
        core betting fields are immutable after ``save_bet``.

        Parameters
        ----------
        bet_id:
            Primary key of the bet to resolve.
        actual_stat:
            The real-world stat value the player posted.
        outcome:
            One of ``'WIN'``, ``'LOSS'``, ``'PUSH'``, ``'NO_ACTION'``.
        pl_units:
            Profit/loss in units.
        pl_usd:
            Profit/loss in dollars (optional).

        Raises
        ------
        ValueError
            If ``outcome`` is not a recognised value.
        """
        valid_outcomes = {"WIN", "LOSS", "PUSH", "NO_ACTION"}
        if outcome not in valid_outcomes:
            raise ValueError(
                "outcome must be one of {}; got {!r}".format(valid_outcomes, outcome)
            )

        now = datetime.now(EASTERN).strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE bets
                SET actual_stat        = ?,
                    outcome            = ?,
                    pl_units           = ?,
                    pl_usd             = ?,
                    result_updated_at  = ?
                WHERE id = ?
                """,
                (actual_stat, outcome, pl_units, pl_usd, now, bet_id),
            )

    def save_closing_line(self, data: Dict[str, Any]) -> int:
        """Insert a closing-line snapshot and return its ``id``.

        Parameters
        ----------
        data:
            Dictionary with keys matching the ``closing_lines`` columns.
            ``game_date``, ``player_id``, ``player_name``, ``prop_type``,
            and ``line`` are required.

        Returns
        -------
        int
            The ``id`` of the newly inserted row.
        """
        required = ("game_date", "player_id", "player_name", "prop_type", "line")
        for field in required:
            if data.get(field) is None:
                raise ValueError(
                    "Missing required closing_line field: {}".format(field)
                )

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO closing_lines (
                    game_date, game_pk, player_id, player_name,
                    prop_type, line, close_odds_over, close_odds_under,
                    close_implied_over, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["game_date"],
                    data.get("game_pk"),
                    data["player_id"],
                    data["player_name"],
                    data["prop_type"],
                    data["line"],
                    data.get("close_odds_over"),
                    data.get("close_odds_under"),
                    data.get("close_implied_over"),
                    data.get("source"),
                ),
            )
            return cursor.lastrowid

    def save_clv(
        self,
        bet_id: int,
        close_odds: int,
        close_implied: float,
        clv: float,
        clv_pct: float,
    ) -> None:
        """Insert or replace a CLV record for a resolved bet.

        Parameters
        ----------
        bet_id:
            Foreign key referencing ``bets.id``.
        close_odds:
            American odds at closing.
        close_implied:
            Implied probability at closing.
        clv:
            Raw CLV value (percentage points).
        clv_pct:
            CLV expressed as a percentage of the original implied probability.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO bet_clv (bet_id, close_odds, close_implied, clv, clv_pct)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(bet_id) DO UPDATE SET
                    close_odds    = excluded.close_odds,
                    close_implied = excluded.close_implied,
                    clv           = excluded.clv,
                    clv_pct       = excluded.clv_pct
                """,
                (bet_id, close_odds, close_implied, clv, clv_pct),
            )

    def get_bets(
        self,
        date: Optional[str] = None,
        is_live: Optional[int] = None,
        prop_type: Optional[str] = None,
    ) -> List[Dict]:
        """Retrieve bets with optional filters.

        Parameters
        ----------
        date:
            Filter by ``game_date`` (``YYYY-MM-DD``).
        is_live:
            ``1`` for live bets, ``0`` for paper, ``None`` for all.
        prop_type:
            One of ``'strikeouts'``, ``'home_runs'``, ``'hits'``,
            ``'total_bases'``, or ``None`` for all.

        Returns
        -------
        list[dict]
            List of bet rows as plain dictionaries.
        """
        clauses: List[str] = []
        params: List[Any] = []

        if date is not None:
            clauses.append("game_date = ?")
            params.append(date)
        if is_live is not None:
            clauses.append("is_live = ?")
            params.append(int(is_live))
        if prop_type is not None:
            clauses.append("prop_type = ?")
            params.append(prop_type)

        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        sql = "SELECT * FROM bets {} ORDER BY game_date, game_time, player_name".format(
            where
        )

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]

    def get_unresolved_bets(self) -> List[Dict]:
        """Return bets from past dates that have no recorded outcome.

        Returns
        -------
        list[dict]
            Unresolved bet rows sorted by ``game_date``, ``player_name``.
        """
        today = datetime.now(EASTERN).strftime("%Y-%m-%d")
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM bets
                WHERE game_date <= ? AND outcome IS NULL
                ORDER BY game_date, player_name
                """,
                (today,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_bet_player_names(self, game_date: str) -> set:
        """Return the set of player_name values already bet on for *game_date*.

        Used by the analysis pipeline to enforce the one-bet-per-player-per-day
        rule across multiple slot runs.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT player_name FROM bets WHERE game_date = ?",
                (game_date,),
            ).fetchall()
            return {row["player_name"] for row in rows}

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def get_running_pl(self, is_live: Optional[int] = None) -> float:
        """Return cumulative P/L in units across all resolved bets.

        Parameters
        ----------
        is_live:
            ``1`` to sum only live bets, ``0`` for paper only, ``None`` for all.

        Returns
        -------
        float
            Total units won/lost.
        """
        if is_live is not None:
            sql = (
                "SELECT COALESCE(SUM(pl_units), 0) AS total "
                "FROM bets WHERE pl_units IS NOT NULL AND is_live = ?"
            )
            params: tuple = (int(is_live),)
        else:
            sql = (
                "SELECT COALESCE(SUM(pl_units), 0) AS total "
                "FROM bets WHERE pl_units IS NOT NULL"
            )
            params = ()

        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
            return float(row["total"] or 0.0)

    def get_recent_record(
        self,
        days: int = 7,
        prop_type: Optional[str] = None,
    ) -> Dict:
        """Return a performance summary for the last *N* days.

        Parameters
        ----------
        days:
            How many calendar days to look back.
        prop_type:
            Optionally restrict to a single prop type.

        Returns
        -------
        dict
            Keys: ``total``, ``wins``, ``losses``, ``hit_rate``,
            ``pl_units``, ``avg_clv``.
        """
        clauses = ["b.game_date >= date('now', '-{} days')".format(days),
                   "b.outcome IS NOT NULL"]
        params: List[Any] = []

        if prop_type is not None:
            clauses.append("b.prop_type = ?")
            params.append(prop_type)

        where = " AND ".join(clauses)
        sql = """
            SELECT
                COUNT(*)                                              AS total,
                SUM(CASE WHEN b.outcome = 'WIN'  THEN 1 ELSE 0 END)  AS wins,
                SUM(CASE WHEN b.outcome = 'LOSS' THEN 1 ELSE 0 END)  AS losses,
                COALESCE(SUM(b.pl_units), 0)                          AS pl_units,
                AVG(c.clv)                                            AS avg_clv
            FROM bets b
            LEFT JOIN bet_clv c ON c.bet_id = b.id
            WHERE {}
        """.format(where)

        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()

        total = row["total"] or 0
        wins = row["wins"] or 0
        losses = row["losses"] or 0
        decided = wins + losses

        return {
            "total": total,
            "wins": wins,
            "losses": losses,
            "hit_rate": wins / decided if decided > 0 else None,
            "pl_units": row["pl_units"] or 0.0,
            "avg_clv": row["avg_clv"],
        }

    def get_record_by_confidence(
        self, prop_type: Optional[str] = None
    ) -> Dict[str, Dict]:
        """Return W/L breakdown grouped by confidence tier.

        Parameters
        ----------
        prop_type:
            Optionally restrict to a single prop type.

        Returns
        -------
        dict
            Mapping of confidence label → ``{wins, losses, hit_rate, pl_units}``.
        """
        clauses = ["outcome IS NOT NULL", "confidence IN ('HIGH', 'MEDIUM', 'LOW')"]
        params: List[Any] = []

        if prop_type is not None:
            clauses.append("prop_type = ?")
            params.append(prop_type)

        where = " AND ".join(clauses)
        sql = """
            SELECT
                confidence,
                SUM(CASE WHEN outcome = 'WIN'  THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) AS losses,
                COALESCE(SUM(pl_units), 0)                         AS pl_units
            FROM bets
            WHERE {}
            GROUP BY confidence
        """.format(where)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        result: Dict[str, Dict] = {}
        for row in rows:
            wins = row["wins"] or 0
            losses = row["losses"] or 0
            decided = wins + losses
            result[row["confidence"]] = {
                "wins": wins,
                "losses": losses,
                "hit_rate": wins / decided if decided > 0 else None,
                "pl_units": row["pl_units"] or 0.0,
            }
        return result

    def get_avg_clv(
        self, days: int = 30, prop_type: Optional[str] = None
    ) -> float:
        """Return average CLV (percentage points) over the last *N* days.

        Parameters
        ----------
        days:
            Look-back window in calendar days.
        prop_type:
            Optionally restrict to a single prop type.

        Returns
        -------
        float
            Average CLV, or ``0.0`` if no data is available.
        """
        clauses = ["b.game_date >= date('now', '-{} days')".format(days),
                   "c.clv IS NOT NULL"]
        params: List[Any] = []

        if prop_type is not None:
            clauses.append("b.prop_type = ?")
            params.append(prop_type)

        where = " AND ".join(clauses)
        sql = """
            SELECT AVG(c.clv) AS avg_clv
            FROM bets b
            JOIN bet_clv c ON c.bet_id = b.id
            WHERE {}
        """.format(where)

        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return float(row["avg_clv"] or 0.0)

    # ------------------------------------------------------------------
    # Reference / lookup
    # ------------------------------------------------------------------

    def upsert_park_factor(
        self,
        park_name: str,
        handedness: str,
        k_factor: float,
        hr_factor: float,
        runs_factor: float,
    ) -> None:
        """Insert or update a park factor row.

        Parameters
        ----------
        park_name:
            Name of the ballpark (e.g. ``'Fenway Park'``).
        handedness:
            ``'L'``, ``'R'``, or ``'ALL'``.
        k_factor:
            Strikeout park factor relative to league average (1.0 = neutral).
        hr_factor:
            Home-run park factor.
        runs_factor:
            Runs park factor.
        """
        now = datetime.now(EASTERN).strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO park_factors (park_name, handedness, K_factor, HR_factor, runs_factor, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(park_name, handedness) DO UPDATE SET
                    K_factor    = excluded.K_factor,
                    HR_factor   = excluded.HR_factor,
                    runs_factor = excluded.runs_factor,
                    updated_at  = excluded.updated_at
                """,
                (park_name, handedness, k_factor, hr_factor, runs_factor, now),
            )

    def upsert_umpire_factor(
        self,
        umpire_id: int,
        name: str,
        k_factor: float,
        bb_factor: float,
        games_counted: int,
    ) -> None:
        """Insert or update an umpire factor row.

        Parameters
        ----------
        umpire_id:
            Unique umpire identifier.
        name:
            Umpire's full name.
        k_factor:
            Strikeout rate factor relative to league average.
        bb_factor:
            Walk rate factor relative to league average.
        games_counted:
            Number of games used to compute these factors.
        """
        now = datetime.now(EASTERN).strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO umpire_factors (umpire_id, name, K_factor, BB_factor, games_counted, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(umpire_id) DO UPDATE SET
                    name          = excluded.name,
                    K_factor      = excluded.K_factor,
                    BB_factor     = excluded.BB_factor,
                    games_counted = excluded.games_counted,
                    last_updated  = excluded.last_updated
                """,
                (umpire_id, name, k_factor, bb_factor, games_counted, now),
            )

    def upsert_pitch_type_perf(self, data: Dict[str, Any]) -> None:
        """Insert or update a row in ``pitch_type_performance``.

        Parameters
        ----------
        data:
            Dictionary with keys matching the ``pitch_type_performance``
            columns.  ``player_id``, ``player_type``, and ``pitch_type``
            are required.
        """
        required = ("player_id", "player_type", "pitch_type")
        for field in required:
            if data.get(field) is None:
                raise ValueError(
                    "Missing required pitch_type_perf field: {}".format(field)
                )

        now = datetime.now(EASTERN).strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO pitch_type_performance (
                    player_id, player_type, pitch_type,
                    pct_thrown_seen, barrel_pct, whiff_pct,
                    gb_pct, fb_pct, ld_pct,
                    woba, iso, hr_per_fb,
                    pa_or_pitches, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(player_id, pitch_type) DO UPDATE SET
                    player_type     = excluded.player_type,
                    pct_thrown_seen = excluded.pct_thrown_seen,
                    barrel_pct      = excluded.barrel_pct,
                    whiff_pct       = excluded.whiff_pct,
                    gb_pct          = excluded.gb_pct,
                    fb_pct          = excluded.fb_pct,
                    ld_pct          = excluded.ld_pct,
                    woba            = excluded.woba,
                    iso             = excluded.iso,
                    hr_per_fb       = excluded.hr_per_fb,
                    pa_or_pitches   = excluded.pa_or_pitches,
                    last_updated    = excluded.last_updated
                """,
                (
                    str(data["player_id"]),
                    data["player_type"],
                    data["pitch_type"],
                    data.get("pct_thrown_seen"),
                    data.get("barrel_pct"),
                    data.get("whiff_pct"),
                    data.get("gb_pct"),
                    data.get("fb_pct"),
                    data.get("ld_pct"),
                    data.get("woba"),
                    data.get("iso"),
                    data.get("hr_per_fb"),
                    data.get("pa_or_pitches"),
                    data.get("last_updated", now),
                ),
            )

    def get_park_factor(
        self, park_name: str, handedness: str = "ALL"
    ) -> Optional[Dict]:
        """Retrieve a park factor row.

        Parameters
        ----------
        park_name:
            Name of the ballpark.
        handedness:
            ``'L'``, ``'R'``, or ``'ALL'`` (default).

        Returns
        -------
        dict or None
            Row as a dictionary, or ``None`` if not found.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM park_factors WHERE park_name = ? AND handedness = ?",
                (park_name, handedness),
            ).fetchone()
            return dict(row) if row else None

    def get_umpire_factor(self, umpire_id: int) -> Optional[Dict]:
        """Retrieve an umpire factor row by ID.

        Parameters
        ----------
        umpire_id:
            Unique umpire identifier.

        Returns
        -------
        dict or None
            Row as a dictionary, or ``None`` if not found.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM umpire_factors WHERE umpire_id = ?",
                (umpire_id,),
            ).fetchone()
            return dict(row) if row else None

    def get_umpire_k_factor_by_name(self, umpire_name: str) -> float:
        """Return K_factor for an umpire by name, defaulting to 1.0.

        Tries an exact match first, then a partial match on last name.
        """
        if not umpire_name:
            return 1.0
        with self._connect() as conn:
            row = conn.execute(
                "SELECT k_factor FROM umpire_factors WHERE umpire_name = ?",
                (umpire_name,),
            ).fetchone()
            if row and row["k_factor"] is not None:
                return float(row["k_factor"])
            # Partial match on last name (e.g. "Angel Hernandez" → "Hernandez")
            last_name = umpire_name.strip().split()[-1]
            row = conn.execute(
                "SELECT k_factor FROM umpire_factors WHERE umpire_name LIKE ?",
                (f"%{last_name}%",),
            ).fetchone()
            if row and row["k_factor"] is not None:
                return float(row["k_factor"])
        return 1.0

    def log_model_run(
        self,
        run_date: str,
        model_version: str,
        prop_type: str,
        n_picks: int,
        mean_edge: float,
    ) -> int:
        """Record metadata about a model execution.

        Parameters
        ----------
        run_date:
            Date the model ran (``YYYY-MM-DD``).
        model_version:
            Version string identifying the model.
        prop_type:
            Prop category modelled on this run.
        n_picks:
            Number of picks generated.
        mean_edge:
            Average edge across picks.

        Returns
        -------
        int
            The ``id`` of the new ``model_runs`` row.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO model_runs (run_date, model_version, prop_type, n_picks, mean_edge)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_date, model_version, prop_type, n_picks, mean_edge),
            )
            return cursor.lastrowid

    def refresh_daily_summary(self, date: str) -> None:
        """Recompute and upsert the ``daily_summary`` row for a given date.

        Parameters
        ----------
        date:
            The ``game_date`` to summarise (``YYYY-MM-DD``).
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*)                                                AS n_bets,
                    SUM(is_live)                                            AS n_live,
                    SUM(1 - is_live)                                        AS n_paper,
                    COALESCE(SUM(units), 0)                                 AS units_wagered,
                    COALESCE(SUM(pl_units), 0)                              AS pl_units,
                    COALESCE(SUM(pl_usd), 0)                                AS pl_usd,
                    CASE WHEN SUM(units) > 0
                         THEN COALESCE(SUM(pl_units), 0) / SUM(units) * 100
                         ELSE 0 END                                         AS roi_pct,
                    SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END)        AS wins,
                    SUM(CASE WHEN outcome IN ('WIN','LOSS') THEN 1 ELSE 0 END) AS decided
                FROM bets
                WHERE game_date = ?
                """,
                (date,),
            ).fetchone()

            avg_clv_row = conn.execute(
                """
                SELECT AVG(c.clv) AS avg_clv
                FROM bets b
                JOIN bet_clv c ON c.bet_id = b.id
                WHERE b.game_date = ?
                """,
                (date,),
            ).fetchone()

            n_bets = row["n_bets"] or 0
            wins = row["wins"] or 0
            decided = row["decided"] or 0
            win_rate = wins / decided if decided > 0 else None
            now = datetime.now(EASTERN).strftime("%Y-%m-%d %H:%M:%S")

            conn.execute(
                """
                INSERT INTO daily_summary (
                    summary_date, n_bets, n_live, n_paper,
                    units_wagered, pl_units, pl_usd,
                    roi_pct, avg_clv, win_rate, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(summary_date) DO UPDATE SET
                    n_bets        = excluded.n_bets,
                    n_live        = excluded.n_live,
                    n_paper       = excluded.n_paper,
                    units_wagered = excluded.units_wagered,
                    pl_units      = excluded.pl_units,
                    pl_usd        = excluded.pl_usd,
                    roi_pct       = excluded.roi_pct,
                    avg_clv       = excluded.avg_clv,
                    win_rate      = excluded.win_rate,
                    updated_at    = excluded.updated_at
                """,
                (
                    date,
                    n_bets,
                    row["n_live"] or 0,
                    row["n_paper"] or 0,
                    row["units_wagered"] or 0.0,
                    row["pl_units"] or 0.0,
                    row["pl_usd"] or 0.0,
                    row["roi_pct"] or 0.0,
                    avg_clv_row["avg_clv"],
                    win_rate,
                    now,
                ),
            )

    # ------------------------------------------------------------------
    # Player stats
    # ------------------------------------------------------------------

    def upsert_player_stat(
        self,
        player_id: int,
        stat_date: str,
        stat_type: str,
        value: float,
        rolling_3: Optional[float] = None,
        rolling_7: Optional[float] = None,
        rolling_14: Optional[float] = None,
        rolling_30: Optional[float] = None,
    ) -> None:
        """Insert or update a single player stat observation.

        Parameters
        ----------
        player_id:
            MLB player ID.
        stat_date:
            Date of the observation (``YYYY-MM-DD``).
        stat_type:
            Stat label (e.g. ``'strikeouts'``, ``'hits'``).
        value:
            Raw stat value for this game/date.
        rolling_3, rolling_7, rolling_14, rolling_30:
            Optional rolling averages over the respective windows.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO player_stats (
                    player_id, stat_date, stat_type, value,
                    rolling_3, rolling_7, rolling_14, rolling_30
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(player_id, stat_date, stat_type) DO UPDATE SET
                    value      = excluded.value,
                    rolling_3  = excluded.rolling_3,
                    rolling_7  = excluded.rolling_7,
                    rolling_14 = excluded.rolling_14,
                    rolling_30 = excluded.rolling_30
                """,
                (
                    player_id,
                    stat_date,
                    stat_type,
                    value,
                    rolling_3,
                    rolling_7,
                    rolling_14,
                    rolling_30,
                ),
            )

    def get_player_stats(
        self, player_id: int, stat_type: str, days: int = 30
    ) -> List[Dict]:
        """Return recent stat observations for a player.

        Parameters
        ----------
        player_id:
            MLB player ID.
        stat_type:
            Stat label to retrieve.
        days:
            Look-back window in calendar days (default 30).

        Returns
        -------
        list[dict]
            Rows ordered by ``stat_date`` ascending.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM player_stats
                WHERE player_id = ?
                  AND stat_type = ?
                  AND stat_date >= date('now', '-{} days')
                ORDER BY stat_date ASC
                """.format(days),
                (player_id, stat_type),
            ).fetchall()
            return [dict(row) for row in rows]

    def save_features(
        self,
        player_id: int,
        feat_date: str,
        prop_type: str,
        features: Dict[str, Any],
    ) -> None:
        """Persist a feature snapshot for a player/date/prop combination.

        Parameters
        ----------
        player_id:
            MLB player ID.
        feat_date:
            Date the features were computed (``YYYY-MM-DD``).
        prop_type:
            The prop category these features relate to.
        features:
            Arbitrary dictionary; will be serialised as JSON.
        """
        feature_json = json.dumps(features)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO features (player_id, feat_date, prop_type, feature_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(player_id, feat_date, prop_type) DO UPDATE SET
                    feature_json = excluded.feature_json,
                    created_at   = datetime('now')
                """,
                (player_id, feat_date, prop_type, feature_json),
            )

    def get_features(
        self, player_id: int, feat_date: str, prop_type: str
    ) -> Optional[Dict]:
        """Retrieve a stored feature snapshot.

        Parameters
        ----------
        player_id:
            MLB player ID.
        feat_date:
            Date the features were computed (``YYYY-MM-DD``).
        prop_type:
            The prop category.

        Returns
        -------
        dict or None
            Deserialised feature dictionary, or ``None`` if not found.
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT feature_json FROM features
                WHERE player_id = ? AND feat_date = ? AND prop_type = ?
                """,
                (player_id, feat_date, prop_type),
            ).fetchone()

        if row is None or row["feature_json"] is None:
            return None
        return json.loads(row["feature_json"])

    # ------------------------------------------------------------------
    # Opening line snapshots
    # ------------------------------------------------------------------

    def save_opening_snapshot(
        self,
        snapshot_date: str,
        player_name: str,
        prop_type: str,
        line: float,
        no_vig_implied: float,
    ) -> None:
        """Persist an opening-line no-vig implied probability.

        Uses INSERT OR IGNORE so the first snapshot of the day wins.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO prop_opening_snapshots
                    (snapshot_date, player_name, prop_type, line, no_vig_implied)
                VALUES (?, ?, ?, ?, ?)
                """,
                (snapshot_date, player_name, prop_type, line, no_vig_implied),
            )

    def get_opening_snapshot(
        self,
        snapshot_date: str,
        player_name: str,
        prop_type: str,
        line: float,
    ) -> Optional[float]:
        """Return the stored no-vig implied for a player/prop/line on *snapshot_date*.

        Returns None if no snapshot was captured for that combination.
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT no_vig_implied FROM prop_opening_snapshots
                WHERE snapshot_date = ? AND player_name = ? AND prop_type = ? AND line = ?
                ORDER BY captured_at ASC
                LIMIT 1
                """,
                (snapshot_date, player_name, prop_type, line),
            ).fetchone()
        return float(row["no_vig_implied"]) if row else None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_db_instance: Optional[MLBPropsDB] = None


def get_db() -> MLBPropsDB:
    """Return the module-level ``MLBPropsDB`` singleton, creating it on first call."""
    global _db_instance
    if _db_instance is None:
        _db_instance = MLBPropsDB()
    return _db_instance
