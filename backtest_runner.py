#!/usr/bin/env python3
"""
Standalone Backtest Runner — Sept 2024

Fetches real MLB game results from the Stats API (no API key needed),
then replays our current scipy baseline models (Binomial/Poisson) against
actual player outcomes to measure hit rate, calibration, and simulated ROI.

No ML training required — this tests the statistical priors we're using today.

Usage:
    python backtest_runner.py                  # Sept 1–29 2024, all markets
    python backtest_runner.py --start 2024-08-01 --end 2024-09-29
    python backtest_runner.py --market hits    # hits only
    python backtest_runner.py --market k       # strikeouts only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests
from scipy.stats import binom, poisson

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Baseline model constants (must match daily_runner.py)
# ---------------------------------------------------------------------------
_MLB_BA = 0.255
_MLB_PA_PER_GAME = 4
_MLB_K_LAMBDA = 5.75

# Market line assumptions (used when no historical odds data is available)
# Typical DraftKings/FanDuel juice on player props
_HITS_LINE = 1.5       # Over 1.5 hits (most common line)
_HITS_ODDS = -115      # standard juice
_K_LINES = [4.5, 5.5, 6.5]  # we evaluate at all common K lines
_K_ODDS = -115

# Sizing thresholds (match sizing.py)
MIN_EDGE = 0.015
EDGE_TIERS = [(0.06, 1.50), (0.03, 0.75), (0.015, 0.25)]

# API
_MLB_BASE = "https://statsapi.mlb.com/api/v1"
_REQUEST_TIMEOUT = 15
_RATE_DELAY = 0.3   # seconds between API calls


# ---------------------------------------------------------------------------
# Model probability functions
# ---------------------------------------------------------------------------

def hits_prob_over(line: float) -> float:
    """P(hits > line) using Binomial(n=4, p=0.255)."""
    return float(1.0 - binom.cdf(int(line), n=_MLB_PA_PER_GAME, p=_MLB_BA))


def hits_prob_under(line: float) -> float:
    return 1.0 - hits_prob_over(line)


def k_prob_over(line: float) -> float:
    """P(Ks > line) using Poisson(λ=5.75)."""
    return float(1.0 - poisson.cdf(int(line), mu=_MLB_K_LAMBDA))


def k_prob_under(line: float) -> float:
    return 1.0 - k_prob_over(line)


def implied_prob_american(odds: int) -> float:
    """Vig-inclusive implied probability from American odds."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def calc_edge(model_prob: float, book_odds: int) -> float:
    return model_prob - implied_prob_american(book_odds)


def classify_confidence(edge: float) -> str | None:
    for threshold, _ in EDGE_TIERS:
        if edge >= threshold:
            idx = EDGE_TIERS.index((threshold, _))
            return ["HIGH", "MEDIUM", "LOW"][idx]
    return None


def tier_units(edge: float) -> float:
    for threshold, units in EDGE_TIERS:
        if edge >= threshold:
            return units
    return 0.0


def calc_pl(outcome: str, odds: int, units: float) -> float:
    """Calculate P/L in units for a standard American odds bet."""
    if outcome == "WIN":
        if odds >= 0:
            return units * (odds / 100.0)
        return units * (100.0 / abs(odds))
    if outcome == "LOSS":
        return -units
    return 0.0  # PUSH


# ---------------------------------------------------------------------------
# MLB Stats API helpers
# ---------------------------------------------------------------------------

_session = requests.Session()


def _get(path: str, params: dict | None = None) -> dict:
    url = f"{_MLB_BASE}{path}"
    time.sleep(_RATE_DELAY)
    try:
        r = _session.get(url, params=params, timeout=_REQUEST_TIMEOUT)
        if r.status_code == 404:
            return {}
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        log.warning("MLB API error %s: %s", url, exc)
        return {}


def get_schedule_range(start: str, end: str) -> list[dict]:
    """Fetch all Final games between start and end (YYYY-MM-DD inclusive)."""
    data = _get("/schedule", {
        "sportId": 1,
        "startDate": start,
        "endDate": end,
        "gameType": "R",
        "hydrate": "team",
    })
    games = []
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            status = g.get("status", {}).get("detailedState", "")
            if status != "Final":
                continue
            teams = g.get("teams", {})
            games.append({
                "game_pk": g["gamePk"],
                "game_date": date_block["date"],
                "away_team": teams.get("away", {}).get("team", {}).get("name", ""),
                "home_team": teams.get("home", {}).get("team", {}).get("name", ""),
            })
    return games


def get_boxscore(game_pk: int) -> dict:
    """Return raw boxscore data for a completed game."""
    data = _get(f"/game/{game_pk}/boxscore")
    return data.get("teams", {})


def parse_hitting_lines(teams: dict, game_date: str, game_pk: int) -> list[dict]:
    """
    Extract per-batter actual hit counts from a boxscore teams dict.
    Returns list of {player_id, name, team, hits, at_bats, game_date, game_pk}.
    Only includes players with at least 2 at-bats (excludes pitchers etc.).
    """
    rows = []
    for side in ("home", "away"):
        team_data = teams.get(side, {})
        team_name = team_data.get("team", {}).get("name", side)
        players = team_data.get("players", {})
        batting_order = team_data.get("battingOrder", [])
        for pid in batting_order:
            key = f"ID{pid}"
            p = players.get(key, {})
            stats = p.get("stats", {}).get("batting", {})
            at_bats = stats.get("atBats", 0)
            if at_bats < 2:
                continue
            rows.append({
                "player_id": pid,
                "name": p.get("person", {}).get("fullName", "Unknown"),
                "team": team_name,
                "hits": stats.get("hits", 0),
                "at_bats": at_bats,
                "game_date": game_date,
                "game_pk": game_pk,
            })
    return rows


def parse_pitching_lines(teams: dict, game_date: str, game_pk: int) -> list[dict]:
    """
    Extract per-pitcher actual strikeout counts from a boxscore.
    Only includes starting pitchers (those with 3+ innings pitched, or highest IP per team).
    """
    rows = []
    for side in ("home", "away"):
        team_data = teams.get(side, {})
        team_name = team_data.get("team", {}).get("name", side)
        players = team_data.get("players", {})
        pitchers = team_data.get("pitchers", [])

        if not pitchers:
            continue

        # Starter = first pitcher listed (MLB orders by appearance)
        starter_id = pitchers[0]
        key = f"ID{starter_id}"
        p = players.get(key, {})
        stats = p.get("stats", {}).get("pitching", {})

        # Parse innings pitched — stored as float where 0.1 = 1/3 inning
        ip_str = str(stats.get("inningsPitched", "0.0"))
        try:
            ip = float(ip_str)
        except ValueError:
            ip = 0.0

        # Only count starters who pitched 3+ innings (1-inning openers skew results)
        if ip < 3.0:
            continue

        rows.append({
            "player_id": starter_id,
            "name": p.get("person", {}).get("fullName", "Unknown"),
            "team": team_name,
            "strikeouts": stats.get("strikeOuts", 0),
            "innings_pitched": ip,
            "game_date": game_date,
            "game_pk": game_pk,
        })
    return rows


# ---------------------------------------------------------------------------
# Bet simulation
# ---------------------------------------------------------------------------

@dataclass
class BetResult:
    player_name: str
    team: str
    game_date: str
    game_pk: int
    market: str         # "hits" or "k"
    line: float
    pick: str           # "Over" or "Under"
    model_prob: float
    book_odds: int
    edge: float
    confidence: str
    units: float
    actual: float       # actual stat value
    outcome: str        # WIN / LOSS
    pl: float


def simulate_hits_bets(row: dict) -> list[BetResult]:
    """Evaluate Over and Under at _HITS_LINE for a batter row."""
    results = []
    actual = float(row["hits"])
    line = _HITS_LINE

    for pick, model_prob, book_odds in [
        ("Over", hits_prob_over(line), _HITS_ODDS),
        ("Under", hits_prob_under(line), _HITS_ODDS),
    ]:
        edge = calc_edge(model_prob, book_odds)
        if edge < MIN_EDGE:
            continue
        conf = classify_confidence(edge)
        if conf is None:
            continue
        units = tier_units(edge)
        if pick == "Over":
            outcome = "WIN" if actual > line else "LOSS"
        else:
            outcome = "WIN" if actual <= line else "LOSS"

        results.append(BetResult(
            player_name=row["name"],
            team=row["team"],
            game_date=row["game_date"],
            game_pk=row["game_pk"],
            market="hits",
            line=line,
            pick=pick,
            model_prob=model_prob,
            book_odds=book_odds,
            edge=edge,
            confidence=conf,
            units=units,
            actual=actual,
            outcome=outcome,
            pl=calc_pl(outcome, book_odds, units),
        ))
    return results


def simulate_k_bets(row: dict) -> list[BetResult]:
    """Evaluate Over and Under at each K line for a starter row."""
    results = []
    actual = float(row["strikeouts"])

    for line in _K_LINES:
        for pick, model_prob, book_odds in [
            ("Over", k_prob_over(line), _K_ODDS),
            ("Under", k_prob_under(line), _K_ODDS),
        ]:
            edge = calc_edge(model_prob, book_odds)
            if edge < MIN_EDGE:
                continue
            conf = classify_confidence(edge)
            if conf is None:
                continue
            units = tier_units(edge)
            if pick == "Over":
                outcome = "WIN" if actual > line else "LOSS"
            else:
                outcome = "WIN" if actual <= line else "LOSS"

            results.append(BetResult(
                player_name=row["name"],
                team=row["team"],
                game_date=row["game_date"],
                game_pk=row["game_pk"],
                market=f"k_{line}",
                line=line,
                pick=pick,
                model_prob=model_prob,
                book_odds=book_odds,
                edge=edge,
                confidence=conf,
                units=units,
                actual=actual,
                outcome=outcome,
                pl=calc_pl(outcome, book_odds, units),
            ))
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class MarketSummary:
    market: str
    n_bets: int = 0
    n_wins: int = 0
    total_units_wagered: float = 0.0
    total_pl: float = 0.0
    avg_edge: float = 0.0
    by_confidence: dict = field(default_factory=dict)
    by_pick: dict = field(default_factory=dict)


def aggregate(bets: list[BetResult]) -> dict[str, MarketSummary]:
    summaries: dict[str, MarketSummary] = {}

    for b in bets:
        if b.market not in summaries:
            summaries[b.market] = MarketSummary(market=b.market)
        s = summaries[b.market]
        s.n_bets += 1
        s.total_units_wagered += b.units
        s.total_pl += b.pl
        if b.outcome == "WIN":
            s.n_wins += 1

        # By confidence
        c = s.by_confidence.setdefault(b.confidence, {"n": 0, "wins": 0, "pl": 0.0, "wagered": 0.0})
        c["n"] += 1
        c["wagered"] += b.units
        c["pl"] += b.pl
        if b.outcome == "WIN":
            c["wins"] += 1

        # By pick direction
        p = s.by_pick.setdefault(b.pick, {"n": 0, "wins": 0, "pl": 0.0, "wagered": 0.0})
        p["n"] += 1
        p["wagered"] += b.units
        p["pl"] += b.pl
        if b.outcome == "WIN":
            p["wins"] += 1

    # Compute derived metrics
    for s in summaries.values():
        s.avg_edge = sum(b.edge for b in bets if b.market == s.market) / s.n_bets if s.n_bets else 0.0
        for bucket in list(s.by_confidence.values()) + list(s.by_pick.values()):
            bucket["hit_rate"] = round(bucket["wins"] / bucket["n"], 3) if bucket["n"] else 0.0
            bucket["roi_pct"] = round(bucket["pl"] / bucket["wagered"] * 100, 1) if bucket["wagered"] else 0.0

    return summaries


def print_report(summaries: dict[str, MarketSummary], bets: list[BetResult]) -> None:
    total_bets = len(bets)
    total_wins = sum(1 for b in bets if b.outcome == "WIN")
    total_pl = sum(b.pl for b in bets)
    total_wagered = sum(b.units for b in bets)

    print("\n" + "=" * 60)
    print(f"  BACKTEST RESULTS  ({len(set(b.game_date for b in bets))} days)")
    print("=" * 60)
    print(f"  Total bets   : {total_bets}")
    print(f"  Overall W-L  : {total_wins}-{total_bets - total_wins}")
    print(f"  Overall WR   : {total_wins / total_bets:.1%}" if total_bets else "  No bets")
    print(f"  Total P/L    : {total_pl:+.2f} units")
    print(f"  Overall ROI  : {total_pl / total_wagered * 100:+.1f}%" if total_wagered else "")
    print()

    for market, s in sorted(summaries.items()):
        wr = s.n_wins / s.n_bets if s.n_bets else 0.0
        roi = s.total_pl / s.total_units_wagered * 100 if s.total_units_wagered else 0.0
        print(f"  [{market}]  {s.n_bets} bets  WR={wr:.1%}  P/L={s.total_pl:+.2f}u  ROI={roi:+.1f}%  avg_edge={s.avg_edge:.3f}")
        for conf in ["HIGH", "MEDIUM", "LOW"]:
            if conf in s.by_confidence:
                c = s.by_confidence[conf]
                print(f"    {conf:6s}: {c['n']:3d} bets  WR={c['hit_rate']:.1%}  ROI={c['roi_pct']:+.1f}%")
        for pick in ["Over", "Under"]:
            if pick in s.by_pick:
                p = s.by_pick[pick]
                print(f"    {pick:5s}: {p['n']:3d} bets  WR={p['hit_rate']:.1%}  ROI={p['roi_pct']:+.1f}%")
        print()

    print("=" * 60)

    # Model calibration note
    hits_over_actual = sum(1 for b in bets if b.market == "hits" and b.pick == "Over" and b.outcome == "WIN")
    hits_over_total = sum(1 for b in bets if b.market == "hits" and b.pick == "Over")
    if hits_over_total:
        actual_rate = hits_over_total and hits_over_actual / hits_over_total
        model_rate = hits_prob_over(_HITS_LINE)
        print(f"\n  Hits calibration:")
        print(f"    Model P(Over {_HITS_LINE}) = {model_rate:.3f}")
        print(f"    Actual Over rate          = {actual_rate:.3f}  (n={hits_over_total})")
        print(f"    Bias = {actual_rate - model_rate:+.3f}")

    k_overs = [(b, line) for b in bets for line in _K_LINES if b.market == f"k_{line}" and b.pick == "Over"]
    for line in _K_LINES:
        k_line_bets = [b for b in bets if b.market == f"k_{line}" and b.pick == "Over"]
        if k_line_bets:
            actual_over_rate = sum(1 for b in k_line_bets if b.outcome == "WIN") / len(k_line_bets)
            model_rate = k_prob_over(line)
            print(f"\n  K calibration (line={line}):")
            print(f"    Model P(Over {line}) = {model_rate:.3f}")
            print(f"    Actual Over rate   = {actual_over_rate:.3f}  (n={len(k_line_bets)})")
            print(f"    Bias = {actual_over_rate - model_rate:+.3f}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(start: str, end: str, market_filter: str | None) -> None:
    log.info("Fetching schedule %s → %s", start, end)
    games = get_schedule_range(start, end)
    log.info("Found %d completed games", len(games))

    all_bets: list[BetResult] = []
    hitting_rows_total = 0
    pitching_rows_total = 0

    for i, game in enumerate(games):
        game_pk = game["game_pk"]
        log.info("[%d/%d] game_pk=%d  %s @ %s  (%s)",
                 i + 1, len(games), game_pk,
                 game["away_team"], game["home_team"], game["game_date"])

        teams = get_boxscore(game_pk)
        if not teams:
            log.warning("No boxscore for game_pk=%d, skipping", game_pk)
            continue

        if not market_filter or market_filter == "hits":
            hit_rows = parse_hitting_lines(teams, game["game_date"], game_pk)
            hitting_rows_total += len(hit_rows)
            for row in hit_rows:
                all_bets.extend(simulate_hits_bets(row))

        if not market_filter or market_filter == "k":
            k_rows = parse_pitching_lines(teams, game["game_date"], game_pk)
            pitching_rows_total += len(k_rows)
            for row in k_rows:
                all_bets.extend(simulate_k_bets(row))

    log.info("Parsed %d batter lines, %d starter lines", hitting_rows_total, pitching_rows_total)
    log.info("Total qualifying bets (edge >= %.3f): %d", MIN_EDGE, len(all_bets))

    if not all_bets:
        print("\nNo qualifying bets found. The models produce no edge at standard -115 juice.")
        print("This means the model probability is below the book's implied probability for all lines.")
        print(f"\n  Hits Over {_HITS_LINE}:  model={hits_prob_over(_HITS_LINE):.3f}  book_implied={implied_prob_american(_HITS_ODDS):.3f}  edge={hits_prob_over(_HITS_LINE) - implied_prob_american(_HITS_ODDS):+.4f}")
        for line in _K_LINES:
            print(f"  K Over {line}:       model={k_prob_over(line):.3f}  book_implied={implied_prob_american(_K_ODDS):.3f}  edge={k_prob_over(line) - implied_prob_american(_K_ODDS):+.4f}")
            print(f"  K Under {line}:      model={k_prob_under(line):.3f}  book_implied={implied_prob_american(_K_ODDS):.3f}  edge={k_prob_under(line) - implied_prob_american(_K_ODDS):+.4f}")
        return

    summaries = aggregate(all_bets)
    print_report(summaries, all_bets)

    # Save results
    out_dir = Path.home() / "mlb-props" / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"backtest_{start}_{end}.json"
    out_path.write_text(json.dumps({
        "start": start,
        "end": end,
        "total_bets": len(all_bets),
        "summaries": {k: asdict(v) for k, v in summaries.items()},
        "bets": [asdict(b) for b in all_bets],
    }, indent=2, default=str))
    log.info("Results saved to %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="MLB props baseline model backtest")
    parser.add_argument("--start", default="2024-09-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-09-29", help="End date YYYY-MM-DD")
    parser.add_argument("--market", choices=["hits", "k"], default=None,
                        help="Only run one market (default: both)")
    args = parser.parse_args()
    run(args.start, args.end, args.market)


if __name__ == "__main__":
    main()
