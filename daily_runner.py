"""daily_runner.py — Main entry point for the MLB props betting pipeline.

Runs in two modes triggered by launchd:
  analysis  (~90 min before first game): fetch data, run models, send picks email
  results   (~30 min after last game):   resolve bets, compute CLV, send results email
  statcast  (nightly): refresh Statcast rolling metrics for yesterday
"""

import argparse
import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from db import get_db
from mlb_api import get_client
from weather import get_weather_client
from odds import get_odds_client
import umpires
import notifier
import statcast_nightly
import closing_lines
import sizing
import risk

# ---------------------------------------------------------------------------
# Config — all values from environment
# ---------------------------------------------------------------------------

DRY_RUN = os.getenv("DRY_RUN", "0") == "1"
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
VISUAL_CROSSING_KEY = os.getenv("VISUAL_CROSSING_API_KEY", "")
BANKROLL = float(os.getenv("MLB_BANKROLL", "1000"))
BOOKS = ["draftkings", "fanduel", "betmgm"]
MARKETS = ["batter_home_runs", "pitcher_strikeouts"]

# Phase 1 stub model version identifier
_MODEL_VERSION = "stub-v0"

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prop_type_from_market(market_key: str) -> str:
    """Map an Odds API market key to an internal prop_type string."""
    mapping = {
        "batter_home_runs": "home_runs",
        "pitcher_strikeouts": "strikeouts",
    }
    return mapping.get(market_key, market_key)


def _resolve_outcome(pick: str, line: float, actual: float) -> str:
    """Determine WIN/LOSS/PUSH from pick direction and actual stat."""
    if pick.lower() == "over":
        if actual > line:
            return "WIN"
        if actual < line:
            return "LOSS"
        return "PUSH"
    if pick.lower() == "under":
        if actual < line:
            return "WIN"
        if actual > line:
            return "LOSS"
        return "PUSH"
    return "NO_ACTION"


def _compute_pl_units(outcome: str, odds: int, units: float) -> float:
    """Compute profit/loss in units given outcome, American odds, and stake."""
    if outcome == "WIN":
        if odds > 0:
            return units * (odds / 100.0)
        return units * (100.0 / abs(odds))
    if outcome == "LOSS":
        return -units
    return 0.0


def _group_stat_key(prop_type: str) -> str:
    """Return the MLB Stats API stat group for a prop type."""
    if prop_type in ("strikeouts", "pitcher_strikeouts"):
        return "pitching"
    return "hitting"


def _extract_stat_value(game_log_entry: dict, prop_type: str) -> float | None:
    """Pull the relevant stat from a game log entry dict."""
    stat_map = {
        "home_runs": "homeRuns",
        "strikeouts": "strikeOuts",
    }
    key = stat_map.get(prop_type)
    if key is None:
        return None
    val = game_log_entry.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# run_analysis
# ---------------------------------------------------------------------------

def run_analysis(
    game_date: date,
    db,
    mlb_client,
    weather_client,
    odds_client,
) -> list[dict]:
    """Fetch schedule + props, run preflight, return qualifying picks list.

    Phase 1 (stub): edge is always 0.0; should_bet() will return False for all
    props until a real model is plugged in.  The full pipeline still executes
    end-to-end so downstream components can be validated.

    Returns
    -------
    list[dict]
        Each entry is a qualifying prop dict (may be empty when no games today
        or all props fail preflight).
    """
    date_str = game_date.isoformat()
    log.info("=== Analysis run for %s (dry_run=%s) ===", date_str, DRY_RUN)

    # 1. Schedule
    schedule = mlb_client.get_schedule(date_str)
    if not schedule:
        log.info("No games found for %s — skipping analysis.", date_str)
        return []

    game_pks = [g["game_pk"] for g in schedule if g.get("game_pk")]
    log.info("Found %d games: %s", len(schedule), game_pks)

    # 2. Umpires
    try:
        umpires.refresh_umpire_factors(db)
        todays_umpires = umpires.get_todays_umpires(mlb_client, game_pks)
        log.info("Umpire data loaded for %d games.", len(todays_umpires))
    except Exception:
        log.exception("Umpire refresh failed — continuing with neutral factors.")
        todays_umpires = {}

    # 3. Probable pitchers (keyed by game_pk)
    try:
        probable_pitchers = mlb_client.get_probable_pitchers(date_str)
    except Exception:
        log.exception("Failed to fetch probable pitchers.")
        probable_pitchers = {}

    # 4. Odds events + props
    try:
        events = odds_client.get_mlb_events(date_str)
        log.info("Fetched %d Odds API events.", len(events))
        all_props = odds_client.get_all_games_props(events, markets=MARKETS)
    except Exception:
        log.exception("Failed to fetch odds/props — aborting analysis.")
        raise

    # Map event_id -> game_pk using schedule
    event_to_game_pk: dict[str, int | None] = {}
    for event in events:
        gk = odds_client.match_event_to_game_pk(event, schedule)
        event_to_game_pk[event["event_id"]] = gk

    # 5. Per-game weather (cache by game_pk)
    venue_weather: dict[int, dict] = {}
    for game in schedule:
        gk = game.get("game_pk")
        if gk is None:
            continue
        try:
            venue_id = game.get("venue_id")
            venue_info = mlb_client.get_venue_info(venue_id) if venue_id else {}
            lat = venue_info.get("lat") or 0.0
            lon = venue_info.get("lon") or 0.0
            game_time_utc = game.get("game_time_utc", "")
            venue_name = game.get("venue_name", "")
            wx = weather_client.get_game_weather(
                lat=float(lat),
                lon=float(lon),
                game_time_utc=str(game_time_utc),
                venue_name=str(venue_name),
            )
            venue_weather[gk] = wx
        except Exception:
            log.exception("Weather fetch failed for game_pk=%s; using neutral.", gk)
            venue_weather[gk] = {}

    # 6. Iterate props — preflight → sizing → build picks list
    picks: list[dict] = []
    today_str = date_str

    for event_id, props_data in all_props.items():
        game_pk = event_to_game_pk.get(event_id)
        weather = venue_weather.get(game_pk, {}) if game_pk else {}
        pitchers = probable_pitchers.get(game_pk, {}) if game_pk else {}

        for book_key, book_markets in props_data.get("props", {}).items():
            if book_key not in BOOKS:
                continue

            for market_key, outcomes in book_markets.items():
                prop_type = _prop_type_from_market(market_key)

                for outcome in outcomes:
                    player_name: str = outcome.get("player_name", "")
                    pick: str = outcome.get("pick", "")
                    line = outcome.get("line")
                    odds_american: int = outcome.get("odds", -110)

                    if not player_name or line is None:
                        continue

                    # For Phase 1 we don't have player IDs from odds; stub with 0
                    player_id = 0
                    pitcher_id = (
                        pitchers.get("home_pitcher_id")
                        or pitchers.get("away_pitcher_id")
                        or 0
                    )

                    # Preflight — skip on critical failures (lineup/injury/weather)
                    try:
                        preflight = risk.run_preflight(
                            game_pk=game_pk or 0,
                            player_id=player_id,
                            pitcher_id=pitcher_id or 0,
                            prop_type=prop_type,
                            opening_line=float(odds_american),
                            current_line=float(odds_american),
                            weather=weather,
                            mlb_client=mlb_client,
                            db=db,
                        )
                    except Exception:
                        log.exception(
                            "Preflight error for %s %s — skipping.", player_name, prop_type
                        )
                        continue

                    critical_checks = {"lineup_confirmed", "injury_status", "weather_playable"}
                    failed_critical = [
                        c for c in preflight.get("checks", [])
                        if c["name"] in critical_checks and not c["passed"]
                    ]
                    if failed_critical:
                        reasons = [c["reason"] for c in failed_critical]
                        log.debug(
                            "Skip %s %s — critical preflight failed: %s",
                            player_name, prop_type, reasons,
                        )
                        continue

                    # Phase 1 stub: edge=0.0, model provides no real signal yet
                    edge = 0.0
                    model_prob = outcome.get("implied_prob", 0.5)
                    confidence = sizing.classify_confidence(edge)

                    bet_ok, reason = sizing.should_bet(
                        edge=edge,
                        odds_american=odds_american,
                        game_pk=game_pk or 0,
                        db=db,
                    )
                    units = 0.0
                    if bet_ok:
                        units = sizing.compute_units(
                            edge=edge,
                            odds_american=odds_american,
                            bankroll=BANKROLL,
                        )

                    pick_dict = {
                        "game_date": today_str,
                        "game_pk": game_pk,
                        "event_id": event_id,
                        "player_name": player_name,
                        "player_id": player_id,
                        "prop_type": prop_type,
                        "line": line,
                        "pick": pick,
                        "book": book_key,
                        "odds": odds_american,
                        "implied_prob": outcome.get("implied_prob"),
                        "model_prob": model_prob,
                        "model_projection": line,  # stub: projection = line
                        "edge": edge,
                        "confidence": confidence,
                        "units": units,
                        "bet_ok": bet_ok,
                        "skip_reason": None if bet_ok else reason,
                        "preflight": preflight,
                        "model_version": _MODEL_VERSION,
                        "is_live": 0,
                        "stake_usd": round(units * BANKROLL, 2) if bet_ok else 0.0,
                    }
                    picks.append(pick_dict)
                    log.debug(
                        "Qualified: %s %s %s %.1f @ %s (edge=%.4f bet_ok=%s)",
                        player_name, prop_type, pick, line, book_key, edge, bet_ok,
                    )

    bettable = [p for p in picks if p["bet_ok"]]
    log.info(
        "Analysis complete — %d qualifying props, %d bettable.", len(picks), len(bettable)
    )

    # 7. Persist bets (skip in dry-run)
    if not DRY_RUN:
        today_full = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        for p in bettable:
            try:
                bet_row = {
                    "bet_date": today_full,
                    "game_date": p["game_date"],
                    "game_time": None,
                    "game_pk": p["game_pk"],
                    "player_id": p["player_id"],
                    "player_name": p["player_name"],
                    "team": "",
                    "opponent": "",
                    "prop_type": p["prop_type"],
                    "line": p["line"],
                    "pick": p["pick"],
                    "book": p["book"],
                    "odds": p["odds"],
                    "implied_prob": p["implied_prob"],
                    "model_projection": p["model_projection"],
                    "model_prob": p["model_prob"],
                    "edge": p["edge"],
                    "confidence": p["confidence"],
                    "model_version": p["model_version"],
                    "feature_snapshot": None,
                    "units": p["units"],
                    "stake_usd": p["stake_usd"],
                    "open_line": p["odds"],
                    "line_at_open": p["line"],
                    "is_live": p["is_live"],
                    "notes": None,
                }
                db.save_bet(bet_row)
            except Exception:
                log.exception("Failed to save bet for %s %s.", p["player_name"], p["prop_type"])

    return picks


# ---------------------------------------------------------------------------
# run_results
# ---------------------------------------------------------------------------

def run_results(
    game_date: date,
    db,
    mlb_client,
) -> dict:
    """Resolve open bets and compute CLV for the given game date.

    Returns
    -------
    dict
        {
          'resolved': int,
          'wins': int,
          'losses': int,
          'pushes': int,
          'daily_pl': float,
          'clv_summary': dict,
          'results': list[dict],
        }
    """
    date_str = game_date.isoformat()
    log.info("=== Results run for %s (dry_run=%s) ===", date_str, DRY_RUN)

    unresolved = db.get_unresolved_bets()
    target_bets = [b for b in unresolved if b.get("game_date", "")[:10] == date_str]
    log.info("Found %d unresolved bets for %s.", len(target_bets), date_str)

    season = game_date.year
    results: list[dict] = []
    wins = losses = pushes = 0
    daily_pl = 0.0

    for bet in target_bets:
        bet_id = bet["id"]
        player_id = bet.get("player_id")
        prop_type = bet.get("prop_type", "")
        line = float(bet.get("line") or 0.0)
        pick = bet.get("pick", "over")
        odds_american = int(bet.get("odds") or -110)
        units = float(bet.get("units") or 0.0)

        if not player_id:
            log.warning("Bet id=%d has no player_id — skipping resolution.", bet_id)
            continue

        stat_group = _group_stat_key(prop_type)
        try:
            game_log = mlb_client.get_player_game_log(
                player_id=int(player_id),
                season=season,
                group=stat_group,
            )
        except Exception:
            log.exception("Failed to fetch game log for player_id=%s.", player_id)
            continue

        # Find entry matching the game date
        actual_stat: float | None = None
        for entry in game_log:
            if str(entry.get("date", ""))[:10] == date_str:
                actual_stat = _extract_stat_value(entry, prop_type)
                break

        if actual_stat is None:
            log.warning(
                "No game log entry found for player_id=%s on %s — skipping.",
                player_id, date_str,
            )
            continue

        outcome = _resolve_outcome(pick, line, actual_stat)
        pl_units = _compute_pl_units(outcome, odds_american, units)
        pl_usd = round(pl_units * BANKROLL, 2)

        if not DRY_RUN:
            try:
                db.update_result(
                    bet_id=bet_id,
                    actual_stat=actual_stat,
                    outcome=outcome,
                    pl_units=pl_units,
                    pl_usd=pl_usd,
                )
            except Exception:
                log.exception("Failed to update result for bet id=%d.", bet_id)
                continue

        if outcome == "WIN":
            wins += 1
        elif outcome == "LOSS":
            losses += 1
        elif outcome == "PUSH":
            pushes += 1

        daily_pl += pl_units
        results.append({**bet, "actual_stat": actual_stat, "outcome": outcome,
                        "pl_units": pl_units, "pl_usd": pl_usd})

    log.info(
        "Resolution complete — %d resolved (%d W / %d L / %d P), daily P/L=%.2f u",
        len(results), wins, losses, pushes, daily_pl,
    )

    # CLV computation
    clv_summary: dict = {}
    try:
        closing_lines.run_clv_computation(game_date, db)
        clv_summary = closing_lines.get_daily_clv_summary(game_date, db)
        log.info("CLV summary: %s", clv_summary)
    except Exception:
        log.exception("CLV computation failed.")

    # Sync to cloud if DATABASE_URL is configured
    if os.getenv("DATABASE_URL"):
        try:
            from sync_to_cloud import run_full_sync
            sync_result = run_full_sync(since_days=1)
            log.info("Cloud sync: %s", sync_result)
        except Exception as e:
            log.warning("Cloud sync failed (non-fatal): %s", e)

    running_pl = db.get_running_pl()

    return {
        "resolved": len(results),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "daily_pl": daily_pl,
        "running_pl": running_pl,
        "clv_summary": clv_summary,
        "results": results,
    }


# ---------------------------------------------------------------------------
# run_statcast_update
# ---------------------------------------------------------------------------

def run_statcast_update(db) -> dict:
    """Refresh Statcast rolling metrics for yesterday's slate.

    Pulls the active player roster from recent bets so only relevant players
    are fetched.

    Returns
    -------
    dict
        Result dict from statcast_nightly.run_full_nightly().
    """
    log.info("=== Statcast nightly update ===")

    # Collect player IDs from recent bets as a lightweight roster proxy
    try:
        recent = db.get_unresolved_bets()
        batter_ids: list[int] = []
        pitcher_ids: list[int] = []
        for bet in recent:
            pid = bet.get("player_id")
            if not pid:
                continue
            pid = int(pid)
            prop = bet.get("prop_type", "")
            if "strikeout" in prop.lower():
                pitcher_ids.append(pid)
            else:
                batter_ids.append(pid)
        # De-duplicate
        batter_ids = list(dict.fromkeys(batter_ids))
        pitcher_ids = list(dict.fromkeys(pitcher_ids))
    except Exception:
        log.exception("Failed to build player ID list for Statcast — using empty lists.")
        batter_ids = []
        pitcher_ids = []

    log.info(
        "Running Statcast for %d batters, %d pitchers.",
        len(batter_ids), len(pitcher_ids),
    )

    try:
        result = statcast_nightly.run_full_nightly(
            batter_ids=batter_ids,
            pitcher_ids=pitcher_ids,
        )
        log.info("Statcast nightly complete: %s", result)
        return result
    except Exception:
        log.exception("statcast_nightly.run_full_nightly() failed.")
        raise


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLB player props daily pipeline runner."
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date to process (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--mode",
        choices=["analysis", "results", "statcast"],
        default="analysis",
        help="Pipeline mode to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Skip all writes (bets, DB updates, emails) but run everything else.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    # Honour CLI flag in addition to env var
    global DRY_RUN
    if args.dry_run:
        DRY_RUN = True
    if DRY_RUN:
        log.info("DRY RUN mode — no writes or emails will be sent.")

    # Resolve target date
    if args.date:
        try:
            game_date = date.fromisoformat(args.date)
        except ValueError:
            log.error("Invalid --date value %r. Expected YYYY-MM-DD.", args.date)
            sys.exit(1)
    else:
        game_date = date.today()

    log.info("Mode=%s  Date=%s  DryRun=%s", args.mode, game_date, DRY_RUN)

    try:
        db = get_db()
    except Exception:
        log.exception("Failed to initialise database.")
        sys.exit(1)

    try:
        if args.mode == "analysis":
            mlb_client = get_client()
            weather_client = get_weather_client()
            odds_client = get_odds_client()

            picks = run_analysis(
                game_date=game_date,
                db=db,
                mlb_client=mlb_client,
                weather_client=weather_client,
                odds_client=odds_client,
            )

            running_pl = db.get_running_pl()

            if not DRY_RUN:
                if picks:
                    notifier.send_picks_email(
                        picks=picks,
                        running_pl=running_pl,
                        date=game_date.isoformat(),
                    )
                    log.info("Picks email sent (%d picks).", len(picks))
                else:
                    schedule = get_client().get_schedule(game_date.isoformat())
                    notifier.send_no_picks_email(
                        games_evaluated=len(schedule),
                        date=game_date.isoformat(),
                    )
                    log.info("No-picks email sent.")
            else:
                log.info("DRY RUN — suppressed picks email. Picks count: %d", len(picks))

        elif args.mode == "results":
            mlb_client = get_client()

            summary = run_results(
                game_date=game_date,
                db=db,
                mlb_client=mlb_client,
            )

            if not DRY_RUN:
                notifier.send_results_email(
                    date=game_date.isoformat(),
                    results=summary["results"],
                    daily_pl=summary["daily_pl"],
                    running_pl=summary["running_pl"],
                )
                log.info("Results email sent.")
            else:
                log.info(
                    "DRY RUN — suppressed results email. Summary: %s/%s/%s pl=%.2fu",
                    summary["wins"], summary["losses"], summary["pushes"],
                    summary["daily_pl"],
                )

        elif args.mode == "statcast":
            run_statcast_update(db)

    except Exception as exc:
        log.exception("Pipeline failed with unhandled exception: %s", exc)
        if not DRY_RUN:
            try:
                notifier.send_alert_email(
                    subject=f"Pipeline failure: {args.mode} {game_date}",
                    message=str(exc),
                    level="CRITICAL",
                )
            except Exception:
                log.exception("Failed to send alert email after pipeline failure.")
        sys.exit(1)


if __name__ == "__main__":
    main()
