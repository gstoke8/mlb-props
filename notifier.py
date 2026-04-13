#!/usr/bin/env python3
from __future__ import annotations
"""
MLB Player Props — Email Notifier

Sends picks, results, no-picks, and alert emails via the Resend API.

Env vars:
    RESEND_API_KEY — Resend.com API key
    RESEND_FROM    — Sender address (default: MLB Props <onboarding@resend.dev>)
"""

import json
import logging
import os
import traceback
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

# ── Config ─────────────────────────────────────────────────────────────────────
RESEND_KEY  = os.getenv('RESEND_API_KEY', '')
RESEND_FROM = os.getenv('RESEND_FROM', 'MLB Props <onboarding@resend.dev>')
RECIPIENTS  = ['gstocker24@gmail.com']
RESEND_URL  = 'https://api.resend.com/emails'

EASTERN = ZoneInfo('America/New_York')

logger = logging.getLogger(__name__)

# ── Colours ────────────────────────────────────────────────────────────────────
BG_PAGE  = '#0f172a'
BG_CARD  = '#1e293b'
BG_HEAD  = '#0f172a'
TEXT     = '#f1f5f9'
TEXT_DIM = '#94a3b8'
BORDER   = '#334155'
ACCENT   = '#3b82f6'

CONF_COLORS = {
    'HIGH':   '#22c55e',
    'MEDIUM': '#eab308',
    'LOW':    '#94a3b8',
}

WIN_BG   = '#14532d'
LOSS_BG  = '#7f1d1d'
PUSH_BG  = '#1e293b'

MODEL_VERSION = os.getenv('MLB_MODEL_VERSION', 'v1.0')


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _send(subject: str, html: str) -> bool:
    """POST to Resend API. Returns True on success, False on failure (logs error)."""
    if not RESEND_KEY:
        logger.warning('RESEND_API_KEY not set — skipping email: "%s"', subject)
        return False
    try:
        body = json.dumps(
            {'from': RESEND_FROM, 'to': RECIPIENTS, 'subject': subject, 'html': html},
            ensure_ascii=False,
        ).encode('utf-8')
        resp = requests.post(
            RESEND_URL,
            headers={
                'Authorization': f'Bearer {RESEND_KEY.strip()}',
                'Content-Type': 'application/json; charset=utf-8',
            },
            data=body,
            timeout=15,
        )
        if resp.status_code in (200, 201):
            logger.info('Email sent: "%s"', subject)
            return True
        logger.error('Resend error %s: %s', resp.status_code, resp.text)
        return False
    except Exception as exc:
        logger.error('Email failed: %s\n%s', exc, traceback.format_exc())
        return False


def _pl_color(pl: float) -> str:
    return '#22c55e' if pl >= 0 else '#ef4444'


def _pl_str(pl: float) -> str:
    return f'{pl:+.1f}u'


def _odds_display(odds: int | float | None) -> str:
    if odds is None:
        return '—'
    odds_i = int(odds)
    return f'+{odds_i}' if odds_i > 0 else str(odds_i)


def _conf_badge(conf: str) -> str:
    color = CONF_COLORS.get(conf, TEXT_DIM)
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;'
        f'background:{color}22;color:{color};font-size:11px;font-weight:bold;'
        f'letter-spacing:0.5px;">{conf}</span>'
    )


def _base_html(title: str, body_content: str) -> str:
    """Wrap content in the shared dark-mode shell (max-width 600 px, mobile-safe)."""
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title}</title>
</head>
<body style="margin:0;padding:0;background:{BG_PAGE};font-family:Arial,Helvetica,sans-serif;color:{TEXT};">
  <table width="100%" cellspacing="0" cellpadding="0" style="background:{BG_PAGE};">
    <tr>
      <td align="center" style="padding:24px 12px;">
        <table width="100%" cellspacing="0" cellpadding="0"
               style="max-width:600px;background:{BG_CARD};border-radius:8px;
                      border:1px solid {BORDER};overflow:hidden;">
          {body_content}
        </table>
      </td>
    </tr>
  </table>
</body>
</html>'''


def _header_row(date_str: str, running_pl: float, sport: str = 'MLB') -> str:
    pl_color = _pl_color(running_pl)
    return f'''
<tr>
  <td style="background:{BG_HEAD};padding:16px 20px;border-bottom:1px solid {BORDER};">
    <table width="100%" cellspacing="0" cellpadding="0">
      <tr>
        <td style="color:{TEXT_DIM};font-size:13px;">{date_str}</td>
        <td style="text-align:center;color:{TEXT_DIM};font-size:13px;">Sport: {sport}</td>
        <td style="text-align:right;font-size:13px;">
          Running P/L:&nbsp;
          <span style="color:{pl_color};font-weight:bold;">{_pl_str(running_pl)}</span>
        </td>
      </tr>
    </table>
  </td>
</tr>'''


def _section_heading(text: str) -> str:
    return (
        f'<tr><td style="padding:16px 20px 6px;">'
        f'<p style="margin:0;font-size:13px;font-weight:bold;color:{TEXT_DIM};'
        f'text-transform:uppercase;letter-spacing:0.8px;">{text}</p>'
        f'</td></tr>'
    )


def _footer(extra_note: str = '') -> str:
    note = extra_note or 'Paper trading — for informational purposes'
    return f'''
<tr>
  <td style="padding:16px 20px;border-top:1px solid {BORDER};">
    <p style="margin:0;font-size:11px;color:{TEXT_DIM};">
      Model {MODEL_VERSION} &nbsp;·&nbsp; {note}
    </p>
  </td>
</tr>'''


# ── Date helpers ───────────────────────────────────────────────────────────────

def _resolve_date(date: str | None) -> datetime:
    if date:
        try:
            return datetime.strptime(date, '%Y-%m-%d').replace(tzinfo=EASTERN)
        except ValueError:
            pass
    return datetime.now(EASTERN)


def _format_subject_date(dt: datetime) -> str:
    """Return e.g. 'Sunday, April 13' for use in email subjects."""
    return dt.strftime('%A, %B %-d')


def _format_display_date(dt: datetime) -> str:
    return dt.strftime('%A, %B %-d, %Y')


# ── Picks email ────────────────────────────────────────────────────────────────

_CONF_ORDER = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}


def _sort_picks(picks: list[dict]) -> list[dict]:
    return sorted(
        picks,
        key=lambda p: (
            _CONF_ORDER.get(p.get('confidence', 'LOW'), 9),
            p.get('prop_type', ''),
        ),
    )


def _picks_table(picks: list[dict]) -> str:
    if not picks:
        return (
            f'<tr><td style="padding:20px;color:{TEXT_DIM};text-align:center;">'
            f'No picks above threshold today.</td></tr>'
        )

    header = f'''
<tr style="background:{BG_HEAD};">
  <th style="padding:8px 10px;text-align:left;font-size:12px;color:{TEXT_DIM};font-weight:normal;">Player</th>
  <th style="padding:8px 6px;text-align:center;font-size:12px;color:{TEXT_DIM};font-weight:normal;">Prop</th>
  <th style="padding:8px 6px;text-align:center;font-size:12px;color:{TEXT_DIM};font-weight:normal;">Line</th>
  <th style="padding:8px 6px;text-align:center;font-size:12px;color:{TEXT_DIM};font-weight:normal;">Pick</th>
  <th style="padding:8px 6px;text-align:center;font-size:12px;color:{TEXT_DIM};font-weight:normal;">Book</th>
  <th style="padding:8px 6px;text-align:center;font-size:12px;color:{TEXT_DIM};font-weight:normal;">Odds</th>
  <th style="padding:8px 6px;text-align:center;font-size:12px;color:{TEXT_DIM};font-weight:normal;">Edge</th>
  <th style="padding:8px 6px;text-align:center;font-size:12px;color:{TEXT_DIM};font-weight:normal;">Conf</th>
  <th style="padding:8px 10px;text-align:center;font-size:12px;color:{TEXT_DIM};font-weight:normal;">Units</th>
</tr>'''

    rows = ''
    for i, p in enumerate(_sort_picks(picks)):
        row_bg = BG_CARD if i % 2 == 0 else '#253047'
        conf = p.get('confidence', 'LOW')
        conf_color = CONF_COLORS.get(conf, TEXT_DIM)
        edge_val = p.get('edge', 0.0)
        edge_sign = '+' if edge_val >= 0 else ''
        pick_dir = p.get('pick', '')
        pick_color = '#22c55e' if pick_dir == 'OVER' else '#ef4444'
        units = p.get('units', 1.0)

        rows += f'''
<tr style="background:{row_bg};border-top:1px solid {BORDER};">
  <td style="padding:9px 10px;font-size:13px;font-weight:bold;">{p.get('player_name', '—')}</td>
  <td style="padding:9px 6px;text-align:center;font-size:12px;color:{TEXT_DIM};">{p.get('prop_type', '—')}</td>
  <td style="padding:9px 6px;text-align:center;font-size:13px;">{p.get('line', '—')}</td>
  <td style="padding:9px 6px;text-align:center;font-size:13px;font-weight:bold;color:{pick_color};">{pick_dir}</td>
  <td style="padding:9px 6px;text-align:center;font-size:12px;color:{TEXT_DIM};">{p.get('book', '—')}</td>
  <td style="padding:9px 6px;text-align:center;font-size:13px;">{_odds_display(p.get('odds'))}</td>
  <td style="padding:9px 6px;text-align:center;font-size:13px;color:#60a5fa;">{edge_sign}{edge_val:.1f}%</td>
  <td style="padding:9px 6px;text-align:center;">
    <span style="display:inline-block;padding:2px 7px;border-radius:4px;
                 background:{conf_color}22;color:{conf_color};
                 font-size:11px;font-weight:bold;">{conf}</span>
  </td>
  <td style="padding:9px 10px;text-align:center;font-size:13px;">{units:.1f}u</td>
</tr>'''

    return f'''
<tr>
  <td style="padding:0 20px 4px;">
    <table width="100%" cellspacing="0" cellpadding="0"
           style="border-collapse:collapse;border:1px solid {BORDER};border-radius:6px;overflow:hidden;">
      <thead>{header}</thead>
      <tbody>{rows}</tbody>
    </table>
  </td>
</tr>'''


def _model_breakdown(picks: list[dict]) -> str:
    if not picks:
        return ''
    counts: dict[str, int] = {}
    for p in picks:
        pt = p.get('prop_type', 'unknown')
        counts[pt] = counts.get(pt, 0) + 1
    if len(counts) <= 1:
        return ''
    items = ''.join(
        f'<span style="margin-right:16px;font-size:13px;">'
        f'<span style="color:{TEXT_DIM};">{pt}:</span> <strong>{n}</strong></span>'
        for pt, n in sorted(counts.items(), key=lambda x: -x[1])
    )
    return f'''
{_section_heading('Model Breakdown')}
<tr>
  <td style="padding:4px 20px 12px;">
    <div style="background:{BG_HEAD};border-radius:6px;padding:10px 14px;border:1px solid {BORDER};">
      {items}
    </div>
  </td>
</tr>'''


def _key_factors(picks: list[dict]) -> str:
    """Scan picks for weather / umpire notes; render block if any found."""
    weather_notes: list[str] = []
    umpire_notes: list[str] = []

    for p in picks:
        notes = p.get('notes', '') or ''
        # Weather: wind >10 mph or temp >85°F flags surfaced in notes
        if 'wind' in notes.lower() or 'temp' in notes.lower():
            weather_notes.append(f"{p.get('player_name', '?')}: {notes}")
        # Umpire: k_factor outside 0.85–1.20
        if 'umpire' in notes.lower() or 'k_factor' in notes.lower():
            umpire_notes.append(f"{p.get('player_name', '?')}: {notes}")

    if not weather_notes and not umpire_notes:
        return ''

    items = ''
    for note in weather_notes:
        items += (
            f'<li style="margin-bottom:4px;font-size:13px;">'
            f'<span style="color:#fbbf24;">Weather</span> — {note}</li>'
        )
    for note in umpire_notes:
        items += (
            f'<li style="margin-bottom:4px;font-size:13px;">'
            f'<span style="color:#a78bfa;">Umpire</span> — {note}</li>'
        )

    return f'''
{_section_heading('Key Factors')}
<tr>
  <td style="padding:4px 20px 12px;">
    <ul style="margin:0;padding-left:18px;color:{TEXT};">{items}</ul>
  </td>
</tr>'''


def send_picks_email(
    picks: list[dict],
    running_pl: float,
    date: str | None = None,
) -> bool:
    """
    Send the daily picks email.

    Args:
        picks:      List of pick dicts. All picks are included — is_live is not gated.
        running_pl: Season running P/L in units.
        date:       ISO date string YYYY-MM-DD; defaults to today ET.

    Returns:
        True on successful delivery, False otherwise.
    """
    dt = _resolve_date(date)
    display_date = _format_subject_date(dt)
    n = len(picks)
    subject = f'\u26be MLB Props \u2014 {display_date} [{n} play{"s" if n != 1 else ""}]'

    pl_sign = '+' if running_pl >= 0 else ''
    pl_color = _pl_color(running_pl)

    # Title card
    title_block = f'''
<tr>
  <td style="padding:20px 20px 12px;">
    <h1 style="margin:0 0 4px;font-size:22px;color:{TEXT};">\u26be MLB Props</h1>
    <p style="margin:0;font-size:13px;color:{TEXT_DIM};">{_format_display_date(dt)}</p>
  </td>
</tr>'''

    # Summary bar
    summary_block = f'''
<tr>
  <td style="padding:0 20px 16px;">
    <table width="100%" cellspacing="0" cellpadding="0"
           style="background:{BG_HEAD};border:1px solid {BORDER};border-radius:6px;">
      <tr>
        <td style="padding:12px 16px;border-right:1px solid {BORDER};">
          <p style="margin:0;font-size:24px;font-weight:bold;">{n}</p>
          <p style="margin:0;font-size:11px;color:{TEXT_DIM};">PLAYS</p>
        </td>
        <td style="padding:12px 16px;border-right:1px solid {BORDER};">
          <p style="margin:0;font-size:14px;font-weight:bold;color:{CONF_COLORS['HIGH']};">
            {sum(1 for p in picks if p.get('confidence') == 'HIGH')} HIGH
          </p>
          <p style="margin:0;font-size:14px;font-weight:bold;color:{CONF_COLORS['MEDIUM']};">
            {sum(1 for p in picks if p.get('confidence') == 'MEDIUM')} MED
          </p>
          <p style="margin:0;font-size:14px;font-weight:bold;color:{CONF_COLORS['LOW']};">
            {sum(1 for p in picks if p.get('confidence') == 'LOW')} LOW
          </p>
        </td>
        <td style="padding:12px 16px;text-align:right;">
          <p style="margin:0;font-size:11px;color:{TEXT_DIM};">Running P/L</p>
          <p style="margin:0;font-size:20px;font-weight:bold;color:{pl_color};">
            {pl_sign}{running_pl:.1f}u
          </p>
        </td>
      </tr>
    </table>
  </td>
</tr>'''

    picks_section = f'''
{_section_heading(f'Picks ({n})')}
{_picks_table(picks)}'''

    breakdown = _model_breakdown(picks)
    factors   = _key_factors(picks)
    footer    = _footer('Paper trading \u2014 for informational purposes')

    body = title_block + summary_block + picks_section + breakdown + factors + footer
    html = _base_html('MLB Props Picks', body)
    return _send(subject, html)


# ── Results email ──────────────────────────────────────────────────────────────

def _results_table(results: list[dict]) -> str:
    if not results:
        return (
            f'<tr><td style="padding:20px;color:{TEXT_DIM};text-align:center;">'
            f'No results to display.</td></tr>'
        )

    header = f'''
<tr style="background:{BG_HEAD};">
  <th style="padding:8px 10px;text-align:left;font-size:12px;color:{TEXT_DIM};font-weight:normal;">Player</th>
  <th style="padding:8px 6px;text-align:center;font-size:12px;color:{TEXT_DIM};font-weight:normal;">Prop</th>
  <th style="padding:8px 6px;text-align:center;font-size:12px;color:{TEXT_DIM};font-weight:normal;">Pick</th>
  <th style="padding:8px 6px;text-align:center;font-size:12px;color:{TEXT_DIM};font-weight:normal;">Line</th>
  <th style="padding:8px 6px;text-align:center;font-size:12px;color:{TEXT_DIM};font-weight:normal;">Actual</th>
  <th style="padding:8px 6px;text-align:center;font-size:12px;color:{TEXT_DIM};font-weight:normal;">Outcome</th>
  <th style="padding:8px 10px;text-align:center;font-size:12px;color:{TEXT_DIM};font-weight:normal;">P/L</th>
</tr>'''

    rows = ''
    for r in results:
        outcome = (r.get('outcome') or '').upper()
        if outcome == 'WIN':
            row_bg = WIN_BG
            outcome_color = '#4ade80'
        elif outcome == 'LOSS':
            row_bg = LOSS_BG
            outcome_color = '#f87171'
        else:
            row_bg = PUSH_BG
            outcome_color = TEXT_DIM

        pl_val = r.get('pl', 0.0) or 0.0
        pl_color = _pl_color(pl_val)
        pick_dir = r.get('pick', '')
        pick_color = '#22c55e' if pick_dir == 'OVER' else '#ef4444'

        rows += f'''
<tr style="background:{row_bg};border-top:1px solid {BORDER};">
  <td style="padding:9px 10px;font-size:13px;font-weight:bold;">{r.get('player_name', '—')}</td>
  <td style="padding:9px 6px;text-align:center;font-size:12px;color:{TEXT_DIM};">{r.get('prop_type', '—')}</td>
  <td style="padding:9px 6px;text-align:center;font-size:13px;font-weight:bold;color:{pick_color};">{pick_dir}</td>
  <td style="padding:9px 6px;text-align:center;font-size:13px;">{r.get('line', '—')}</td>
  <td style="padding:9px 6px;text-align:center;font-size:13px;">{r.get('actual', '—')}</td>
  <td style="padding:9px 6px;text-align:center;font-size:13px;font-weight:bold;color:{outcome_color};">{outcome or '—'}</td>
  <td style="padding:9px 10px;text-align:center;font-size:13px;font-weight:bold;color:{pl_color};">{_pl_str(pl_val)}</td>
</tr>'''

    return f'''
<tr>
  <td style="padding:0 20px 4px;">
    <table width="100%" cellspacing="0" cellpadding="0"
           style="border-collapse:collapse;border:1px solid {BORDER};border-radius:6px;overflow:hidden;">
      <thead>{header}</thead>
      <tbody>{rows}</tbody>
    </table>
  </td>
</tr>'''


def _conf_record_block(results: list[dict]) -> str:
    """W-L breakdown by confidence tier."""
    tiers = ['HIGH', 'MEDIUM', 'LOW']
    data: dict[str, dict[str, int]] = {t: {'w': 0, 'l': 0} for t in tiers}
    for r in results:
        conf = (r.get('confidence') or 'LOW').upper()
        outcome = (r.get('outcome') or '').upper()
        if conf in data:
            if outcome == 'WIN':
                data[conf]['w'] += 1
            elif outcome == 'LOSS':
                data[conf]['l'] += 1

    cells = ''
    for t in tiers:
        w, l = data[t]['w'], data[t]['l']
        if w + l == 0:
            continue
        color = CONF_COLORS.get(t, TEXT_DIM)
        total = w + l
        hr = w / total * 100 if total else 0
        cells += (
            f'<td style="padding:10px 14px;border-right:1px solid {BORDER};">'
            f'<p style="margin:0;font-size:11px;color:{TEXT_DIM};">{t}</p>'
            f'<p style="margin:0;font-size:15px;font-weight:bold;color:{color};">'
            f'{w}-{l}</p>'
            f'<p style="margin:0;font-size:11px;color:{TEXT_DIM};">{hr:.0f}%</p>'
            f'</td>'
        )

    if not cells:
        return ''

    return f'''
{_section_heading('By Confidence')}
<tr>
  <td style="padding:4px 20px 12px;">
    <table cellspacing="0" cellpadding="0"
           style="background:{BG_HEAD};border:1px solid {BORDER};border-radius:6px;">
      <tr>{cells}</tr>
    </table>
  </td>
</tr>'''


def send_results_email(
    date: str,
    results: list[dict],
    daily_pl: float,
    running_pl: float,
) -> bool:
    """
    Send the nightly results / grading email.

    Args:
        date:       ISO date string YYYY-MM-DD for the slate being graded.
        results:    List of graded pick dicts.
        daily_pl:   P/L for this slate in units.
        running_pl: Season running P/L in units.

    Returns:
        True on successful delivery, False otherwise.
    """
    wins   = sum(1 for r in results if (r.get('outcome') or '').upper() == 'WIN')
    losses = sum(1 for r in results if (r.get('outcome') or '').upper() == 'LOSS')
    pushes = sum(1 for r in results if (r.get('outcome') or '').upper() == 'PUSH')
    total  = wins + losses
    hit_rate = wins / total * 100 if total else 0.0

    d_sign = '+' if daily_pl >= 0 else ''
    subject = (
        f'\u26be MLB Results \u2014 {date} | {wins}-{losses} | {d_sign}{daily_pl:.1f}u'
    )

    d_color  = _pl_color(daily_pl)
    r_color  = _pl_color(running_pl)

    # Summary bar
    push_cell = (
        f'<td style="padding:12px 16px;border-right:1px solid {BORDER};">'
        f'<p style="margin:0;font-size:20px;font-weight:bold;">{pushes}</p>'
        f'<p style="margin:0;font-size:11px;color:{TEXT_DIM};">PUSH</p>'
        f'</td>'
        if pushes else ''
    )

    summary = f'''
<tr>
  <td style="padding:0 20px 16px;">
    <table width="100%" cellspacing="0" cellpadding="0"
           style="background:{BG_HEAD};border:1px solid {BORDER};border-radius:6px;">
      <tr>
        <td style="padding:12px 16px;border-right:1px solid {BORDER};">
          <p style="margin:0;font-size:22px;font-weight:bold;color:{CONF_COLORS['HIGH']};">{wins}</p>
          <p style="margin:0;font-size:11px;color:{TEXT_DIM};">WINS</p>
        </td>
        <td style="padding:12px 16px;border-right:1px solid {BORDER};">
          <p style="margin:0;font-size:22px;font-weight:bold;color:#ef4444;">{losses}</p>
          <p style="margin:0;font-size:11px;color:{TEXT_DIM};">LOSSES</p>
        </td>
        {push_cell}
        <td style="padding:12px 16px;border-right:1px solid {BORDER};">
          <p style="margin:0;font-size:20px;font-weight:bold;">{hit_rate:.0f}%</p>
          <p style="margin:0;font-size:11px;color:{TEXT_DIM};">HIT RATE</p>
        </td>
        <td style="padding:12px 16px;border-right:1px solid {BORDER};">
          <p style="margin:0;font-size:11px;color:{TEXT_DIM};">Daily P/L</p>
          <p style="margin:0;font-size:18px;font-weight:bold;color:{d_color};">{d_sign}{daily_pl:.1f}u</p>
        </td>
        <td style="padding:12px 16px;">
          <p style="margin:0;font-size:11px;color:{TEXT_DIM};">Running P/L</p>
          <p style="margin:0;font-size:18px;font-weight:bold;color:{r_color};">{_pl_str(running_pl)}</p>
        </td>
      </tr>
    </table>
  </td>
</tr>'''

    # CLV block — only shown when at least one result carries clv data
    clv_values = [r['clv'] for r in results if r.get('clv') is not None]
    clv_block = ''
    if clv_values:
        avg_clv = sum(clv_values) / len(clv_values)
        clv_sign = '+' if avg_clv >= 0 else ''
        clv_color = _pl_color(avg_clv)
        clv_block = f'''
{_section_heading('Closing Line Value')}
<tr>
  <td style="padding:4px 20px 12px;">
    <div style="background:{BG_HEAD};border:1px solid {BORDER};border-radius:6px;padding:10px 16px;">
      <span style="font-size:13px;color:{TEXT_DIM};">Avg CLV today:&nbsp;</span>
      <span style="font-size:15px;font-weight:bold;color:{clv_color};">{clv_sign}{avg_clv:.1f}%</span>
    </div>
  </td>
</tr>'''

    conf_block = _conf_record_block(results)

    title_block = f'''
<tr>
  <td style="padding:20px 20px 12px;">
    <h1 style="margin:0 0 4px;font-size:22px;color:{TEXT};">\u26be MLB Results</h1>
    <p style="margin:0;font-size:13px;color:{TEXT_DIM};">{date}</p>
  </td>
</tr>'''

    results_section = f'''
{_section_heading('Results')}
{_results_table(results)}'''

    footer = _footer()

    body = title_block + summary + results_section + conf_block + clv_block + footer
    html = _base_html('MLB Results', body)
    return _send(subject, html)


# ── No-picks email ─────────────────────────────────────────────────────────────

def send_no_picks_email(
    games_evaluated: int,
    date: str | None = None,
) -> bool:
    """
    Send a brief notification when no picks passed the edge threshold.

    Args:
        games_evaluated: Number of games scanned.
        date:            ISO date YYYY-MM-DD; defaults to today ET.

    Returns:
        True on successful delivery, False otherwise.
    """
    dt = _resolve_date(date)
    display_date = _format_subject_date(dt)
    subject = f'\u26be MLB Props \u2014 No picks today ({games_evaluated} games evaluated)'

    now_str = datetime.now(EASTERN).strftime('%I:%M %p ET').lstrip('0')
    game_word = 'game' if games_evaluated == 1 else 'games'

    body = f'''
<tr>
  <td style="padding:24px 20px 8px;">
    <h1 style="margin:0 0 4px;font-size:22px;color:{TEXT};">\u26be MLB Props</h1>
    <p style="margin:0;font-size:13px;color:{TEXT_DIM};">{_format_display_date(dt)} &nbsp;&middot;&nbsp; {now_str}</p>
  </td>
</tr>
<tr>
  <td style="padding:8px 20px 24px;">
    <div style="background:{BG_HEAD};border-left:4px solid {BORDER};
                border-radius:4px;padding:14px 16px;">
      <p style="margin:0 0 6px;font-weight:bold;color:{TEXT};">No qualifying picks found</p>
      <p style="margin:0;font-size:13px;color:{TEXT_DIM};">
        Evaluated {games_evaluated} {game_word} — no edges passed the confidence threshold.
      </p>
    </div>
  </td>
</tr>
{_footer()}'''

    html = _base_html('MLB Props — No Picks', body)
    return _send(subject, html)


# ── Alert email ────────────────────────────────────────────────────────────────

def send_alert_email(
    subject: str,
    message: str,
    level: str = 'WARNING',
) -> bool:
    """
    Send a plain alert email for data quality issues and abort triggers.

    Args:
        subject: Short description of the alert.
        message: Full alert body (plain text or HTML snippet).
        level:   Severity label — 'WARNING', 'ERROR', 'CRITICAL', etc.

    Returns:
        True on successful delivery, False otherwise.
    """
    full_subject = f'\U0001f6a8 MLB Alert [{level}]: {subject}'

    level_upper = level.upper()
    if level_upper == 'CRITICAL':
        level_color = '#ef4444'
        level_bg    = '#450a0a'
    elif level_upper == 'ERROR':
        level_color = '#f97316'
        level_bg    = '#431407'
    else:
        level_color = '#eab308'
        level_bg    = '#422006'

    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

    body = f'''
<tr>
  <td style="padding:20px 20px 12px;">
    <div style="display:inline-block;padding:4px 12px;border-radius:4px;
                background:{level_bg};border:1px solid {level_color}33;margin-bottom:12px;">
      <span style="font-size:13px;font-weight:bold;color:{level_color};">{level_upper}</span>
    </div>
    <h1 style="margin:0 0 4px;font-size:20px;color:{TEXT};">{subject}</h1>
    <p style="margin:0;font-size:12px;color:{TEXT_DIM};">{ts}</p>
  </td>
</tr>
<tr>
  <td style="padding:0 20px 20px;">
    <div style="background:{BG_HEAD};border:1px solid {BORDER};border-radius:6px;
                padding:16px;font-size:14px;color:{TEXT};line-height:1.6;
                white-space:pre-wrap;">{message}</div>
  </td>
</tr>
{_footer('MLB Props model alert')}'''

    html = _base_html(f'MLB Alert: {subject}', body)
    return _send(full_subject, html)
