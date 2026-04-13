"""
Closing line capture and CLV computation for MLB player props.

Captures odds ~CLOSING_WINDOW_MINUTES before first pitch for every active bet,
then computes and persists closing line value (CLV) post-game.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any

from db import calculate_clv, get_db
from mlb_api import get_client
from odds import get_odds_client, implied_prob

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLOSING_WINDOW_MINUTES = 5
MARKETS = ["batter_home_runs", "pitcher_strikeouts"]
BOOKS = ["draftkings", "fanduel", "betmgm"]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def capture_closing_lines_for_date(
    game_date: date,
    db: Any,
    odds_client: Any,
    mlb_client: Any,
) -> dict[int, int]:
    """Fetch and persist closing lines for all games with active bets on game_date.

    Returns {game_pk: rows_saved}.
    """
    date_str = game_date.isoformat()
    schedule = mlb_client.get_schedule(date_str)

    active_game_pks = _get_active_bet_game_pks(game_date, db)
    if not active_game_pks:
        logger.info("No active bets found for %s — skipping closing line capture", date_str)
        return {}

    events = odds_client.get_mlb_events(date_str)
    results: dict[int, int] = {}

    for event in events:
        game_pk = odds_client.match_event_to_game_pk(event, schedule)
        if game_pk is None or game_pk not in active_game_pks:
            continue

        event_id = event["event_id"]
        logger.info(
            "Capturing closing lines for game_pk=%d event_id=%s", game_pk, event_id
        )

        try:
            rows_saved = odds_client.capture_closing_lines(
                event_id=event_id,
                game_pk=game_pk,
                markets=list(MARKETS),
                db=db,
            )
            results = {**results, game_pk: rows_saved}
        except Exception as exc:
            logger.error(
                "Failed to capture closing lines for game_pk=%d: %s", game_pk, exc
            )

    logger.info(
        "Closing line capture complete for %s: %d games processed",
        date_str,
        len(results),
    )
    return results


def compute_clv_for_bet(bet_row: dict[str, Any], db: Any) -> float | None:
    """Compute and save CLV for a single resolved bet.

    bet_row must have: bet_id, player_id, prop_type, pick, odds_american, book.
    Returns clv_pct, or None if no matching closing line is found.
    """
    bet_id = bet_row["bet_id"]
    player_id = bet_row["player_id"]
    prop_type = bet_row["prop_type"]
    pick = bet_row["pick"]
    our_odds = bet_row["odds_american"]
    game_pk = bet_row.get("game_pk")

    closing_line = _lookup_closing_line(
        player_id=player_id,
        prop_type=prop_type,
        pick=pick,
        game_pk=game_pk,
        db=db,
    )

    if closing_line is None:
        logger.warning(
            "No closing line found for bet_id=%d player_id=%d prop_type=%s pick=%s",
            bet_id,
            player_id,
            prop_type,
            pick,
        )
        return None

    close_odds = _pick_close_odds(closing_line, pick)
    if close_odds is None:
        logger.warning(
            "Closing line row for bet_id=%d has no odds for pick=%s", bet_id, pick
        )
        return None

    our_implied = implied_prob(our_odds)
    close_implied = implied_prob(close_odds)

    clv_pct = calculate_clv(our_implied, close_implied)
    db.save_clv(
        bet_id=bet_id,
        close_odds=close_odds,
        close_implied=close_implied,
        clv=clv_pct,
        clv_pct=clv_pct,
    )

    logger.info(
        "CLV computed for bet_id=%d: close_odds=%d our_implied=%.4f "
        "close_implied=%.4f clv_pct=%.2f",
        bet_id,
        close_odds,
        our_implied,
        close_implied,
        clv_pct,
    )
    return clv_pct


def run_clv_computation(game_date: date, db: Any) -> dict[str, int]:
    """Compute CLV for all resolved bets on game_date that lack a CLV entry.

    Returns {computed: N, missing_lines: N}.
    """
    date_str = game_date.isoformat()
    bets = _get_resolved_bets_without_clv(date_str, db)

    if not bets:
        logger.info("No resolved bets without CLV found for %s", date_str)
        return {"computed": 0, "missing_lines": 0}

    computed = 0
    missing_lines = 0

    for bet in bets:
        # Normalise key: bets table stores odds as 'odds'; bet_row contract uses 'odds_american'
        bet_row = {**bet, "odds_american": bet.get("odds_american") or bet.get("odds")}
        clv_pct = compute_clv_for_bet(bet_row, db)
        if clv_pct is not None:
            computed += 1
        else:
            missing_lines += 1

    logger.info(
        "CLV computation finished for %s: computed=%d missing_lines=%d",
        date_str,
        computed,
        missing_lines,
    )
    return {"computed": computed, "missing_lines": missing_lines}


def run_pregame_capture(db: Any, odds_client: Any, mlb_client: Any) -> None:
    """Capture closing lines for today's games; called by scheduler ~T-10min before first pitch."""
    today = date.today()
    logger.info("Starting pregame closing line capture for %s", today.isoformat())

    results = capture_closing_lines_for_date(
        game_date=today,
        db=db,
        odds_client=odds_client,
        mlb_client=mlb_client,
    )

    total_rows = sum(results.values())
    credits_remaining = odds_client.get_credits_remaining()

    logger.info(
        "Pregame capture complete: %d games, %d rows saved, credits_remaining=%s",
        len(results),
        total_rows,
        credits_remaining,
    )


def get_daily_clv_summary(game_date: date, db: Any) -> dict[str, Any]:
    """Return CLV summary for game_date (used in daily email reports).

    Keys: avg_clv, bets_with_clv, bets_total, clv_positive_count, clv_negative_count.
    """
    date_str = game_date.isoformat()

    with db._connect() as conn:
        summary_row = conn.execute(
            """
            SELECT
                COUNT(b.id)                                             AS bets_total,
                COUNT(c.bet_id)                                         AS bets_with_clv,
                AVG(c.clv_pct)                                          AS avg_clv,
                SUM(CASE WHEN c.clv_pct > 0 THEN 1 ELSE 0 END)         AS clv_positive_count,
                SUM(CASE WHEN c.clv_pct <= 0 THEN 1 ELSE 0 END)        AS clv_negative_count
            FROM bets b
            LEFT JOIN bet_clv c ON c.bet_id = b.id
            WHERE b.game_date = ?
            """,
            (date_str,),
        ).fetchone()

    bets_with_clv = summary_row["bets_with_clv"] or 0

    return {
        "avg_clv": round(float(summary_row["avg_clv"] or 0.0), 4),
        "bets_with_clv": bets_with_clv,
        "bets_total": summary_row["bets_total"] or 0,
        "clv_positive_count": summary_row["clv_positive_count"] or 0
        if bets_with_clv
        else 0,
        "clv_negative_count": summary_row["clv_negative_count"] or 0
        if bets_with_clv
        else 0,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_active_bet_game_pks(game_date: date, db: Any) -> set[int]:
    """Return the set of game_pks that have at least one unresolved bet."""
    date_str = game_date.isoformat()
    with db._connect() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT game_pk
            FROM bets
            WHERE game_date = ? AND outcome IS NULL AND game_pk IS NOT NULL
            """,
            (date_str,),
        ).fetchall()
    return {row["game_pk"] for row in rows}


def _get_resolved_bets_without_clv(date_str: str, db: Any) -> list[dict[str, Any]]:
    """Return resolved bets on date_str that have no entry in bet_clv."""
    with db._connect() as conn:
        rows = conn.execute(
            """
            SELECT b.*
            FROM bets b
            LEFT JOIN bet_clv c ON c.bet_id = b.id
            WHERE b.game_date = ?
              AND b.outcome IS NOT NULL
              AND c.bet_id IS NULL
            ORDER BY b.id
            """,
            (date_str,),
        ).fetchall()
    return [dict(row) for row in rows]


def _lookup_closing_line(
    player_id: int,
    prop_type: str,
    pick: str,
    game_pk: int | None,
    db: Any,
) -> dict[str, Any] | None:
    """Find the best-matching closing line row for a bet.

    Prefers a game_pk match when available; falls back to player_id + prop_type.
    """
    with db._connect() as conn:
        if game_pk is not None:
            row = conn.execute(
                """
                SELECT * FROM closing_lines
                WHERE player_id = ? AND prop_type = ? AND game_pk = ?
                ORDER BY captured_at DESC
                LIMIT 1
                """,
                (player_id, prop_type, game_pk),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT * FROM closing_lines
                WHERE player_id = ? AND prop_type = ?
                ORDER BY captured_at DESC
                LIMIT 1
                """,
                (player_id, prop_type),
            ).fetchone()

    return dict(row) if row else None


def _pick_close_odds(closing_line: dict[str, Any], pick: str) -> int | None:
    """Extract the correct side's odds from a closing_lines row."""
    pick_lower = pick.lower()
    if pick_lower == "over":
        return closing_line.get("close_odds_over")
    if pick_lower == "under":
        return closing_line.get("close_odds_under")
    # Fallback: try over odds
    return closing_line.get("close_odds_over")
